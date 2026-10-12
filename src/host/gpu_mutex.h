#pragma once

// The renderer's one mutex (g.mu), with an optional profile of who holds it.
//
// BBHOST_LOCK_PROFILE=1: every lock records the site that took it (its
// caller's return address) and its thread; every unlock adds the hold time
// to that site, and a lock that had to wait adds the wait to the waiting
// site. Every few seconds the log gets the sites that held it longest and
// the ones that waited longest, symbolized. Off, it is a HostLock (core/futex.h:
// one compare-and-swap to take and give back, a sleep on the word only when a
// thread must wait) and one predictable branch. It was a std::mutex, which in
// this build is winpthreads': every contended take a kernel wait and every
// release to a sleeper a kernel wake. An ETW CPU trace in Kyo's area at ~112
// fps uncapped (Radeon 8060S) had 25% of the command processor in that
// mutex's slow path and 23,400 context switches a second on it - the vblank's
// flip waiter took it some 2,000 times a second to ask the fences.

#include "core/futex.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

class GpuMutex {
public:
    void lock() {
        if (!profiling()) {
            m_.lock();
            held(__builtin_return_address(0));
            return;
        }
        lock_profiled(__builtin_return_address(0));
        held(__builtin_return_address(0));
    }
    bool try_lock() {
        if (!m_.try_lock()) return false;
        if (profiling()) acquired(__builtin_return_address(0), 0);
        held(__builtin_return_address(0));
        return true;
    }
    void unlock() {
        holder_.store(nullptr, std::memory_order_relaxed);
        if (!profiling()) {
            m_.unlock();
            return;
        }
        released();
        m_.unlock();
        report();
    }

    static bool profiling();

    // For the hang watchdog: the site that holds the lock now (null when it is
    // free) and how many times it has been taken, both racy on purpose.
    void* holder() const { return holder_.load(std::memory_order_relaxed); }
    std::uint64_t acquires() const { return acquires_.load(std::memory_order_relaxed); }
    // Names an address from this binary ("function at file:line").
    static std::string name_of(void* site);

private:
    __attribute__((noinline)) void lock_profiled(void* site);
    void acquired(void* site, std::uint64_t waited_ns);
    void released();
    static void report();

    void held(void* site) {
        holder_.store(site, std::memory_order_relaxed);
        acquires_.fetch_add(1, std::memory_order_relaxed);
    }

    std::atomic<void*> holder_{nullptr};
    std::atomic<std::uint64_t> acquires_{0};
    HostLock m_;
    void* site_ = nullptr;          // under m_
    std::uint64_t since_ns_ = 0;    // under m_
};
