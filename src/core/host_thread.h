// std::thread with glibc's 8 MiB stack: guest callbacks run on the host
// thread's own stack, and macOS's 512 KiB default is below a guest thread's.
#pragma once

#include <thread>

#if defined(__APPLE__)
#include <pthread.h>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>

class HostThread {
public:
    static constexpr std::size_t kStack = std::size_t{8} << 20;
    // Partial pages are refused (EINVAL).
    static_assert(kStack % 16384 == 0, "the stack size must be whole pages");

    HostThread() noexcept = default;
    template <class F, class... A>
    explicit HostThread(F&& f, A&&... a) {
        auto st = std::make_shared<State>();
        st->fn = [fn = std::decay_t<F>(std::forward<F>(f)), args = std::make_tuple(std::decay_t<A>(std::forward<A>(a))...)]() mutable {
            std::apply(std::move(fn), std::move(args));
        };
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, kStack);
        auto* arg = new std::shared_ptr<State>(st);
        const int e = pthread_create(&th_, &attr, entry, arg);
        pthread_attr_destroy(&attr);
        if (e) {
            delete arg;
            throw std::system_error(e, std::generic_category(), "HostThread");
        }
        st_ = std::move(st);
    }
    HostThread(HostThread&& o) noexcept : th_(o.th_), st_(std::move(o.st_)) {}
    HostThread& operator=(HostThread&& o) noexcept {
        if (joinable()) std::terminate();  // as std::thread
        th_ = o.th_;
        st_ = std::move(o.st_);
        return *this;
    }
    HostThread(const HostThread&) = delete;
    HostThread& operator=(const HostThread&) = delete;
    ~HostThread() {
        if (joinable()) std::terminate();  // as std::thread
    }

    bool joinable() const noexcept { return st_ != nullptr; }
    // The default id until the thread has started running.
    std::thread::id get_id() const noexcept { return st_ ? st_->id.load() : std::thread::id(); }
    void join() {
        if (!joinable()) throw std::system_error(std::make_error_code(std::errc::invalid_argument), "HostThread::join");
        pthread_join(th_, nullptr);
        st_.reset();
    }
    void detach() {
        if (!joinable()) throw std::system_error(std::make_error_code(std::errc::invalid_argument), "HostThread::detach");
        pthread_detach(th_);
        st_.reset();
    }

private:
    struct State {
        std::function<void()> fn;
        std::atomic<std::thread::id> id{};
    };
    static void* entry(void* p) {
        const std::unique_ptr<std::shared_ptr<State>> st(static_cast<std::shared_ptr<State>*>(p));
        (*st)->id.store(std::this_thread::get_id());
        (*st)->fn();
        return nullptr;
    }

    pthread_t th_{};
    std::shared_ptr<State> st_;
};
#else
using HostThread = std::thread;
#endif
