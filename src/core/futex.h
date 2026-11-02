// Sleeping on a word: the one primitive the guest's mutexes, condition
// variables, read/write locks, semaphores and event queues are built on.
//
// On Windows the HLE sync slow paths slept on std::mutex and
// std::condition_variable, which in this build are winpthreads': a condition
// variable's wait and signal are a few system calls on semaphores and
// critical sections each, every contended object grows kernel objects, and
// timeouts are whole milliseconds measured against the 15.6/1 ms system
// clock - a timed wait under a millisecond returned at once, and the caller's
// predicate loop spun out the rest. Kyo's KyoPS4x measured the
// same cost on the guest's mutexes there (a kernel mutex under every
// pthread_mutex: 15-25% of the GX workers' and Havok threads' CPU, "#215"),
// and took it out the same way: an atomic word in user space, a sleep only
// when a thread must wait, no spinning before it (a spin before sleeping cost
// 2-5x throughput in his stress test, "#228").
//
// Here the sleep is WaitOnAddress / WakeByAddress* on Windows (the waiter
// list lives in user space; a wake with nobody asleep costs no system call)
// and futex(2) on Linux. A timed wait sleeps whole milliseconds on the system
// timer while more than a couple are left, then the rest on the thread's
// high-resolution waitable timer in short slices, and spins the last few
// microseconds with `pause` - so a guest's 200 us timed wait neither returns
// at once and spins nor sleeps a whole tick.
//
// BBHOST_FUTEX=0 puts the same words on parked std::condition_variables
// (winpthreads on Windows), the primitives the sync layer used before, for
// an A/B. BBHOST_FUTEX_SPIN=<n> spins up to n `pause`s before a contended
// guest lock sleeps (0, the default, per Kyo's measurement).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

using HostDeadline = std::chrono::steady_clock::time_point;

// Sleeps while *word == expected, until woken or `deadline` (nullptr: none).
// False only when it returns because the deadline passed; a wake, a changed
// word or a spurious return give true - callers re-check their condition.
bool host_futex_wait(std::atomic<std::uint32_t>* word, std::uint32_t expected, const HostDeadline* deadline);
void host_futex_wake_one(std::atomic<std::uint32_t>* word);
void host_futex_wake_all(std::atomic<std::uint32_t>* word);
// A 64-bit word, untimed, always the native sleep (WaitOnAddress takes 8
// bytes; on Linux, whose futex takes 4, std::atomic's own wait): sleeps while
// *word == expected, may return spuriously. Not counted in host_futex_counts.
void host_futex_wait64(std::atomic<std::uint64_t>* word, std::uint64_t expected);
void host_futex_wake_one64(std::atomic<std::uint64_t>* word);
void host_futex_wake_all64(std::atomic<std::uint64_t>* word);
// The spin budget (BBHOST_FUTEX_SPIN), and whether the native sleep is in use.
unsigned host_futex_spin();
bool host_futex_native();

// Sleeps and wakes since the start, for the 300-flip report.
struct HostFutexCounts {
    std::uint64_t sleeps, timeouts, wakes;
};
HostFutexCounts host_futex_counts();

// A lock word (0 free, 1 held, 2 held with sleepers possible): one
// compare-and-swap to take or give back when nobody waits. Meets
// BasicLockable/Lockable, so std::lock_guard and std::unique_lock take it.
class HostLock {
public:
    void lock() {
        std::uint32_t c = 0;
        if (!w_.compare_exchange_strong(c, 1, std::memory_order_acquire, std::memory_order_relaxed)) lock_slow(c);
    }
    bool try_lock() {
        std::uint32_t c = 0;
        return w_.compare_exchange_strong(c, 1, std::memory_order_acquire, std::memory_order_relaxed);
    }
    void unlock() {
        if (w_.exchange(0, std::memory_order_release) == 2) host_futex_wake_one(&w_);
    }

private:
    void lock_slow(std::uint32_t c);
    std::atomic<std::uint32_t> w_{0};
};

// A condition variable over a sequence word: a waiter notes the word under
// its lock, counts itself, lets the lock go and sleeps while the word is
// unchanged; a notify bumps the word and wakes. The state the predicate reads
// must change under the same lock (or before the notifier takes and drops
// it), as with std::condition_variable. Same interface for the uses here
// (wait, wait_until, wait_for, with and without a predicate).
//
// The count and the word are sequentially consistent: a waiter counts itself
// and then compares the word, a notifier bumps the word and then reads the
// count, so at least one of the two sees the other - a waiter that compared
// the old word is counted, and a notifier that found no count bumped the word
// before the waiter compared it. Spurious wakes are allowed (a notify_one
// can wake two), lost ones are not.
class HostCondVar {
public:
    template <class Lock>
    void wait(Lock& lk) {
        const std::uint32_t s = seq_.load(std::memory_order_seq_cst);
        waiters_.fetch_add(1, std::memory_order_seq_cst);
        lk.unlock();
        host_futex_wait(&seq_, s, nullptr);
        lk.lock();
        waiters_.fetch_sub(1, std::memory_order_relaxed);
    }
    template <class Lock, class Pred>
    void wait(Lock& lk, Pred pred) {
        while (!pred()) wait(lk);
    }
    template <class Lock>
    std::cv_status wait_until(Lock& lk, const HostDeadline& t) {
        const std::uint32_t s = seq_.load(std::memory_order_seq_cst);
        waiters_.fetch_add(1, std::memory_order_seq_cst);
        lk.unlock();
        const bool in_time = host_futex_wait(&seq_, s, &t);
        lk.lock();
        waiters_.fetch_sub(1, std::memory_order_relaxed);
        return in_time ? std::cv_status::no_timeout : std::cv_status::timeout;
    }
    template <class Lock, class Pred>
    bool wait_until(Lock& lk, const HostDeadline& t, Pred pred) {
        while (!pred()) {
            if (wait_until(lk, t) == std::cv_status::timeout) return pred();
        }
        return true;
    }
    template <class Lock, class Rep, class Period>
    std::cv_status wait_for(Lock& lk, const std::chrono::duration<Rep, Period>& d) {
        return wait_until(lk, std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(d));
    }
    template <class Lock, class Rep, class Period, class Pred>
    bool wait_for(Lock& lk, const std::chrono::duration<Rep, Period>& d, Pred pred) {
        return wait_until(lk, std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(d), pred);
    }
    void notify_one() {
        seq_.fetch_add(1, std::memory_order_seq_cst);
        if (waiters_.load(std::memory_order_seq_cst)) host_futex_wake_one(&seq_);
    }
    void notify_all() {
        seq_.fetch_add(1, std::memory_order_seq_cst);
        if (waiters_.load(std::memory_order_seq_cst)) host_futex_wake_all(&seq_);
    }

private:
    std::atomic<std::uint32_t> seq_{0};
    std::atomic<std::uint32_t> waiters_{0};
};
