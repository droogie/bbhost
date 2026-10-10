// Exercise the real HLE thread lifecycle with a delayed native create return.
// Guest context setup and arena allocation are replaced; pthread scheduling is real.
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <thread>

static std::mutex test_mu;
static std::condition_variable test_cv;
static bool native_finished;
static bool delay_create;
static bool fail_create;
static bool fail_detach;
static int free_count;
static void* allocation;
struct Start {
    void* (*fn)(void*);
    void* arg;
};
static void* start_shim(void* p) {
    auto s = *static_cast<Start*>(p);
    delete static_cast<Start*>(p);
    void* result = s.fn(s.arg);
    {
        std::lock_guard<std::mutex> lock(test_mu);
        native_finished = true;
    }
    test_cv.notify_all();
    return result;
}
static int delayed_create(pthread_t* out, const pthread_attr_t* attr, void* (*fn)(void*), void* arg) {
    if (fail_create)
        return EAGAIN;
    auto* start = new Start{fn, arg};
    int rc = pthread_create(out, attr, start_shim, start);
    if (rc)
        delete start;
    if (!rc && delay_create) {
        // Give the old child time to exit and release its handle before this returns.
        // A correct start gate keeps it waiting until the creator publishes pt.
        std::unique_lock<std::mutex> lock(test_mu);
        test_cv.wait_for(lock, std::chrono::milliseconds(50), [] { return native_finished; });
    }
    return rc;
}
static int controlled_detach(pthread_t pt) { return fail_detach ? EBUSY : pthread_detach(pt); }
#define pthread_create delayed_create
#define pthread_detach controlled_detach
#include "hle/pthread.cpp"
#undef pthread_create
#undef pthread_detach

namespace gsync {
Thread* thread_new() {
    allocation = std::calloc(1, 256);
    return static_cast<Thread*>(allocation);
}
void thread_free(Thread* t) {
    std::lock_guard<std::mutex> lock(test_mu);
    ++free_count;
    // Quarantine freed storage so a late creator write is observable.
    std::memset(t, 0xa5, 256);
}
bool arena_owns(const void* p) { return p != nullptr; }
void set_current_thread(Thread*) {}
} // namespace gsync
void* guest_thread_enter() { return nullptr; }
void guest_thread_leave(void*) {}
std::uint32_t host_thread_id() { return 1; }
void host_thread_set_name(const char*) {}
extern "C" GUEST_ABI int hle_setjmp_raw(void*) { return 0; }

static int failures;
#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                                                  \
            ++failures;                                                                                                \
        }                                                                                                              \
    } while (0)
static void reset() {
    native_finished = false;
    free_count = 0;
    allocation = nullptr;
    delay_create = false;
    fail_create = false;
    fail_detach = false;
}
static void finished() {
    std::unique_lock<std::mutex> lock(test_mu);
    if (!test_cv.wait_for(lock, std::chrono::seconds(3), [] { return native_finished; })) {
        std::fprintf(stderr, "thread timed out\n");
        std::exit(2);
    }
}
static void intact() {
    std::lock_guard<std::mutex> lock(test_mu);
    CHECK(free_count == 1);
    auto* bytes = static_cast<unsigned char*>(allocation);
    bool okay = true;
    for (int i = 0; i < 256; ++i)
        okay &= bytes[i] == 0xa5;
    CHECK(okay);
}
static void* quick(void*) { return reinterpret_cast<void*>(0x1234); }
static void* blocked(void* p) {
    auto* go = static_cast<std::atomic<bool>*>(p);
    while (!go->load(std::memory_order_acquire))
        std::this_thread::yield();
    return quick(nullptr);
}
static void* detach_self(void* p) {
    CHECK(pthread_detach_impl(*static_cast<void**>(p)) == 0);
    return quick(nullptr);
}
int main() {
    unsetenv("BBHOST_TEST_THREAD_START_MS");
    // Already detached, finishes before native create returns.
    reset();
    delay_create = true;
    PthreadAttr attr{kAttrMagic, 1, 0, 0, 700};
    void* a = &attr;
    void* th = nullptr;
    CHECK(pthread_create_impl(&th, &a, reinterpret_cast<void*>(&quick), nullptr, nullptr) == 0);
    finished();
    intact();
    std::free(allocation);
    // Detach after the native thread has already returned.
    reset();
    th = nullptr;
    CHECK(pthread_create_impl(&th, nullptr, reinterpret_cast<void*>(&quick), nullptr, nullptr) == 0);
    finished();
    CHECK(pthread_detach_impl(th) == 0);
    intact();
    std::free(allocation);
    // Detach while running: the child releases exactly once on completion.
    reset();
    th = nullptr;
    std::atomic<bool> go{false};
    CHECK(pthread_create_impl(&th, nullptr, reinterpret_cast<void*>(&blocked), &go, nullptr) == 0);
    CHECK(pthread_detach_impl(th) == 0);
    go.store(true, std::memory_order_release);
    finished();
    intact();
    std::free(allocation);
    // Join works even if a thread has not started when create returns.
    reset();
    th = nullptr;
    CHECK(pthread_create_impl(&th, nullptr, reinterpret_cast<void*>(&quick), nullptr, nullptr) == 0);
    void* result = nullptr;
    CHECK(pthread_join_impl(th, &result) == 0);
    CHECK(result == reinterpret_cast<void*>(0x1234));
    intact();
    std::free(allocation);
    // The guest can detach its own published handle.
    reset();
    th = nullptr;
    CHECK(pthread_create_impl(&th, nullptr, reinterpret_cast<void*>(&detach_self), &th, nullptr) == 0);
    finished();
    intact();
    std::free(allocation);
    // A failed native detach must leave the handle joinable.
    reset();
    th = nullptr;
    go.store(false);
    CHECK(pthread_create_impl(&th, nullptr, reinterpret_cast<void*>(&blocked), &go, nullptr) == 0);
    fail_detach = true;
    CHECK(pthread_detach_impl(th) == gsync::kEBUSY);
    go.store(true, std::memory_order_release);
    finished();
    CHECK(pthread_join_impl(th, nullptr) == 0);
    intact();
    std::free(allocation);
    // Creation failure returns a null handle and releases it once.
    reset();
    fail_create = true;
    th = reinterpret_cast<void*>(1);
    CHECK(pthread_create_impl(&th, nullptr, reinterpret_cast<void*>(&quick), nullptr, nullptr) == gsync::kEAGAIN);
    CHECK(th == nullptr);
    intact();
    std::free(allocation);
    CHECK(g_threads_starting.load() == 0);
    std::printf("thread_lifetime_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
