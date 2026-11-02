#include "core/futex.h"
#include "core/portable.h"
#include "log.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#else
#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

namespace {

// Functions' statics, so a lock taken during another file's static
// initialization still sees the switches.
bool native_on() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_FUTEX");
        return !(e && e[0] == '0');
    }();
    return on;
}
unsigned spin_budget() {
    static const unsigned n = [] {
        const char* e = std::getenv("BBHOST_FUTEX_SPIN");
        return e ? static_cast<unsigned>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    return n;
}

std::atomic<std::uint64_t> g_sleeps{0}, g_timeouts{0}, g_wakes{0};

// BBHOST_FUTEX=0: the words sleep on parked condition variables, one of 64
// picked by the word's address. A waiter checks the word under the bucket's
// mutex and a waker takes that mutex before it notifies, so a change between
// the check and the sleep cannot be missed; every waiter of the bucket wakes
// and re-checks (they are few, and this is the comparison arm).
struct Bucket {
    std::mutex m;
    std::condition_variable cv;
};
Bucket g_buckets[64];
Bucket& bucket_for(const void* p) {
    const auto a = reinterpret_cast<std::uintptr_t>(p);
    return g_buckets[((a >> 2) * 0x9E3779B97F4A7C15ull) >> 58];
}

bool parked_wait(std::atomic<std::uint32_t>* word, std::uint32_t expected, const HostDeadline* deadline) {
    Bucket& b = bucket_for(word);
    std::unique_lock<std::mutex> lk(b.m);
    if (word->load(std::memory_order_acquire) != expected) return true;
    if (!deadline) {
        b.cv.wait(lk);
        return true;
    }
    return b.cv.wait_until(lk, *deadline) == std::cv_status::no_timeout || std::chrono::steady_clock::now() < *deadline;
}

void parked_wake(std::atomic<std::uint32_t>* word) {
    Bucket& b = bucket_for(word);
    std::lock_guard<std::mutex> lk(b.m);
    b.cv.notify_all();
}

#if defined(_WIN32)
// A timed wait in three stages. While more than kCoarseUs are left, whole
// milliseconds of WaitOnAddress, one short of the time left: its timeout runs
// on the system timer (1 ms resolution, asked for at start - core/host_clock.h)
// and can end up to a tick late. Then slices of at most kSliceUs on the
// thread's high-resolution waitable timer (host_sleep_us), the word checked
// after each - a wake in a slice is seen at its end, at most kSliceUs late,
// since a waiter is not in WaitOnAddress's list then and the wake goes to
// another (who re-checks too). The last kSpinUs are spun with `pause`, the
// word checked each time round.
constexpr long long kCoarseUs = 2500;
constexpr long long kSliceUs = 200;
constexpr long long kSpinUs = 40;

inline void cpu_pause() {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#endif
}
#else
long futex_call(std::atomic<std::uint32_t>* word, int op, std::uint32_t val, const timespec* ts, std::uint32_t bitset) {
    return syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(word), op, val, ts, nullptr, bitset);
}
#endif

}  // namespace

unsigned host_futex_spin() { return spin_budget(); }
bool host_futex_native() { return native_on(); }

HostFutexCounts host_futex_counts() {
    return {g_sleeps.load(std::memory_order_relaxed), g_timeouts.load(std::memory_order_relaxed),
            g_wakes.load(std::memory_order_relaxed)};
}

bool host_futex_wait(std::atomic<std::uint32_t>* word, std::uint32_t expected, const HostDeadline* deadline) {
    if (word->load(std::memory_order_acquire) != expected) return true;
    g_sleeps.fetch_add(1, std::memory_order_relaxed);
    if (!native_on()) {
        const bool in_time = parked_wait(word, expected, deadline);
        if (!in_time) g_timeouts.fetch_add(1, std::memory_order_relaxed);
        return in_time;
    }
#if defined(_WIN32)
    if (!deadline) {
        WaitOnAddress(reinterpret_cast<volatile VOID*>(word), &expected, sizeof(expected), INFINITE);
        return true;
    }
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= *deadline) {
            g_timeouts.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const long long left_us = std::chrono::duration_cast<std::chrono::microseconds>(*deadline - now).count();
        if (left_us > kCoarseUs) {
            const auto ms = static_cast<DWORD>(std::min<long long>(left_us / 1000 - 1, 0x7fffffffLL));
            if (WaitOnAddress(reinterpret_cast<volatile VOID*>(word), &expected, sizeof(expected), ms)) return true;
            if (GetLastError() != ERROR_TIMEOUT) return true;  // the caller re-checks
            if (word->load(std::memory_order_acquire) != expected) return true;
            continue;
        }
        if (left_us > kSpinUs) {
            host_sleep_us(static_cast<std::uint64_t>(std::min(left_us - kSpinUs, kSliceUs)));
            if (word->load(std::memory_order_acquire) != expected) return true;
            continue;
        }
        while (std::chrono::steady_clock::now() < *deadline) {
            if (word->load(std::memory_order_acquire) != expected) return true;
            cpu_pause();
        }
    }
#else
    if (!deadline) {
        futex_call(word, FUTEX_WAIT_PRIVATE, expected, nullptr, 0);
        return true;
    }
    // steady_clock is CLOCK_MONOTONIC here; the bitset form takes an
    // absolute time on it, with the kernel's own timer precision.
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline->time_since_epoch()).count();
    if (ns <= 0) return std::chrono::steady_clock::now() < *deadline;
    timespec ts{static_cast<time_t>(ns / 1000000000), static_cast<long>(ns % 1000000000)};
    const long r = futex_call(word, FUTEX_WAIT_BITSET_PRIVATE, expected, &ts, FUTEX_BITSET_MATCH_ANY);
    if (r != 0 && errno == ETIMEDOUT) {
        g_timeouts.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
#endif
}

void host_futex_wake_one(std::atomic<std::uint32_t>* word) {
    g_wakes.fetch_add(1, std::memory_order_relaxed);
    if (!native_on()) {
        parked_wake(word);
        return;
    }
#if defined(_WIN32)
    WakeByAddressSingle(reinterpret_cast<PVOID>(word));
#else
    futex_call(word, FUTEX_WAKE_PRIVATE, 1, nullptr, 0);
#endif
}

void host_futex_wait64(std::atomic<std::uint64_t>* word, std::uint64_t expected) {
#if defined(_WIN32)
    WaitOnAddress(reinterpret_cast<volatile VOID*>(word), &expected, sizeof(expected), INFINITE);
#else
    word->wait(expected, std::memory_order_seq_cst);
#endif
}

void host_futex_wake_one64(std::atomic<std::uint64_t>* word) {
#if defined(_WIN32)
    WakeByAddressSingle(reinterpret_cast<PVOID>(word));
#else
    word->notify_one();
#endif
}

void host_futex_wake_all64(std::atomic<std::uint64_t>* word) {
#if defined(_WIN32)
    WakeByAddressAll(reinterpret_cast<PVOID>(word));
#else
    word->notify_all();
#endif
}

void host_futex_wake_all(std::atomic<std::uint32_t>* word) {
    g_wakes.fetch_add(1, std::memory_order_relaxed);
    if (!native_on()) {
        parked_wake(word);
        return;
    }
#if defined(_WIN32)
    WakeByAddressAll(reinterpret_cast<PVOID>(word));
#else
    futex_call(word, FUTEX_WAKE_PRIVATE, 0x7fffffff, nullptr, 0);
#endif
}

void HostLock::lock_slow(std::uint32_t c) {
    // Drepper's mutex: mark the word "sleepers possible" and sleep while it
    // says so; whoever takes it this way leaves the mark, so the unlock wakes.
    // No spin first (Kyo's measurement, futex.h): host locks here guard short
    // bookkeeping, and a spinning core on the APU takes the GPU's power.
    if (c != 2) c = w_.exchange(2, std::memory_order_acquire);
    while (c != 0) {
        host_futex_wait(&w_, 2, nullptr);
        c = w_.exchange(2, std::memory_order_acquire);
    }
}
