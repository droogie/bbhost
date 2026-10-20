// The host's clocks, its timer settings and the priorities of the threads
// that pace frames.
//
// Clocks. The guest reads time through gettimeofday, sceKernelGetProcessTime,
// sceKernelReadTsc, sceKernelClockGettime, time and clock - the main loop,
// its job threads and the frame limiter, many times a frame. On Windows
// those went through std::chrono (libstdc++) into winpthreads' clock_gettime,
// which asks QueryPerformanceFrequency and divides on every call, behind the
// FS- and stack-switching thunk. These read the counter once and scale it
// with a multiply set up at start, and the HLE entries that use them are
// bound raw (no thunk): they touch no thread-local state.
//
// Timer settings (Windows). The process asks for 1 ms timer resolution at
// start (timeBeginPeriod(1)) - it used to happen at the first host_sleep_us,
// after the window pump had been sleeping 15.6 ms steps - and opts out of
// power throttling: no EcoQoS clock drops for its threads, and Windows 11
// keeps the resolution when the window is hidden or minimized (it otherwise
// lets a hidden window's process fall back to 15.6 ms, which every timed
// wait and Sleep then rounds to). BBHOST_TIMER_PERIOD=0, BBHOST_WIN_QOS=0.
//
// Priorities (Windows). The vblank clock above the rest, the threads that
// feed the GPU (command processors, recorder, submission, presenter) one
// step above normal: on a busy machine a guest worker no longer takes a
// pacing thread's core. BBHOST_THREAD_PRIO=0 leaves every thread normal.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

// Monotonic nanoseconds (QueryPerformanceCounter / CLOCK_MONOTONIC): the
// same count std::chrono::steady_clock gives on both hosts, read cheaper.
std::uint64_t host_clock_monotonic_ns();
// Wall-clock time since 1970 (GetSystemTimePreciseAsFileTime /
// CLOCK_REALTIME), in nanoseconds and in microseconds.
std::int64_t host_clock_realtime_ns();
inline std::int64_t host_clock_realtime_us() { return host_clock_realtime_ns() / 1000; }
// The monotonic count scaled to `hz` ticks a second (the guest's 1.6 GHz
// time-stamp counter), from the raw counter with one multiply.
std::uint64_t host_clock_ticks_at(std::uint64_t hz);

// Once, at the start of main(): timer resolution and power throttling
// (Windows; logs what it set and what the timer resolution now is).
void host_timing_init();

// The calling thread's scheduling class (Windows: SetThreadPriority; macOS:
// its QoS class; a no-op on Linux). Each of these threads starts once, and
// each says so in the log.
enum class HostThreadClass {
    Pacing,   // the vblank clock: little work, on time
    GpuFeed,  // command processors, recorder, submission, presenter
};
void host_thread_set_class(HostThreadClass c, const char* who);

// A call counter for entries many guest threads hit at once, raw (no thunk,
// no host thread-locals): sixteen cache lines, one picked by the caller's
// stack address (each thread has its own stack), so two threads rarely bump
// the same line.
struct HostShardedCount {
    struct alignas(64) Line {
        std::atomic<std::uint64_t> n{0};
    };
    Line lines[16];
    void add() {
        int probe = 0;
        const auto sp = reinterpret_cast<std::uintptr_t>(&probe) >> 20;
        lines[(sp * 0x9E3779B97F4A7C15ull) >> 60].n.fetch_add(1, std::memory_order_relaxed);
    }
    std::uint64_t total() const {
        std::uint64_t t = 0;
        for (const Line& l : lines) t += l.n.load(std::memory_order_relaxed);
        return t;
    }
};
