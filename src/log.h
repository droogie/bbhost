#pragma once

#include <atomic>
#include <cstdio>
#include <cstring>
#include <utility>
#if !defined(_WIN32)
#include <unistd.h>
#endif

// The setup window's "Tooling and logging" (bbhost.toml [debug] tooling,
// BBHOST_TOOLING=0|1; on by default): the periodic reports - the 300-flip
// block, the renderer's, the GX threads', the presenter's - the stall
// detector, the GPU busy meter and the recent-draw ring, and the log itself.
// Off, none of them runs and nothing is logged (host_log_decide below), and
// on Windows no console window opens for the log. Set once, by main() after
// the configuration is final, before the game's threads start.
inline bool g_host_tooling = true;
inline bool host_tooling() { return g_host_tooling; }

// Whether the log is written: undecided until main() has the configuration
// (the lines meanwhile held, up to kHoldBytes, and written or dropped then),
// then on or off (host_log_decide; a crash's report turns it back on,
// host_log_force_on). Off, host_log returns before it formats anything.
enum : int { kHostLogUndecided = 0, kHostLogOn = 1, kHostLogOff = 2 };
inline std::atomic<int> g_host_log_state{kHostLogUndecided};
inline bool host_logging() { return g_host_log_state.load(std::memory_order_relaxed) != kHostLogOff; }

namespace host_log_detail {
constexpr std::size_t kHoldBytes = 32768;
inline char g_hold[kHoldBytes];
inline std::size_t g_hold_n = 0;
inline std::atomic_flag g_hold_lock = ATOMIC_FLAG_INIT;

inline void write_raw(const char* p, std::size_t left) {
#if defined(_WIN32)
    std::fwrite(p, 1, left, stderr);
#else
    // write(2), not stdio: async-signal-safe (the fault handler logs) and no
    // FILE lock to deadlock on when a fault lands inside fputs.
    while (left) {
        const ssize_t n = ::write(2, p, left);
        if (n <= 0) {
            break;
        }
        p += n;
        left -= static_cast<std::size_t>(n);
    }
#endif
}
}  // namespace host_log_detail

// One write per line so lines from different threads do not interleave.
// (Out of line: host_log is instantiated at thousands of call sites.)
__attribute__((noinline)) inline void host_log_line(const char* line) {
    using namespace host_log_detail;
    const int state = g_host_log_state.load(std::memory_order_acquire);
    if (state == kHostLogOff) return;
    const std::size_t len = std::strlen(line);
    if (state == kHostLogUndecided) {
        // Held for the decision; written at once when the buffer is full or
        // taken (a fault's report must not wait on a lock its thread holds).
        if (!g_hold_lock.test_and_set(std::memory_order_acquire)) {
            const int now = g_host_log_state.load(std::memory_order_relaxed);
            if (now == kHostLogUndecided && g_hold_n + len <= kHoldBytes) {
                std::memcpy(g_hold + g_hold_n, line, len);
                g_hold_n += len;
                g_hold_lock.clear(std::memory_order_release);
                return;
            }
            g_hold_lock.clear(std::memory_order_release);
            if (now == kHostLogOff) return;
        }
    }
    write_raw(line, len);
}

// The log on or off, from the configuration (main(): Tooling and logging,
// BBHOST_LOG=1 to keep it whatever that says). The first decision writes the
// lines held until then, or drops them; later ones (the setup window may
// change the setting) only switch.
__attribute__((noinline)) inline void host_log_decide(bool on) {
    using namespace host_log_detail;
    while (g_hold_lock.test_and_set(std::memory_order_acquire)) {
    }
    const int was = g_host_log_state.exchange(on ? kHostLogOn : kHostLogOff, std::memory_order_acq_rel);
    if (was == kHostLogUndecided) {
        if (on && g_hold_n) write_raw(g_hold, g_hold_n);
        g_hold_n = 0;
    }
    g_hold_lock.clear(std::memory_order_release);
}

// A crash's report is written whatever the setting says (to wherever stderr
// goes); what was held is left held.
inline void host_log_force_on() { g_host_log_state.store(kHostLogOn, std::memory_order_release); }

template <typename... Args>
inline void host_log(const char* fmt, Args&&... args) {
    if (g_host_log_state.load(std::memory_order_relaxed) == kHostLogOff) {
        return;
    }
    char buf[1024];
    int n = std::snprintf(buf, sizeof(buf) - 1, "[bbhost] ");
    if (n < 0) {
        return;
    }
    int m;
    if constexpr (sizeof...(Args) == 0) {
        m = std::snprintf(buf + n, sizeof(buf) - 1 - static_cast<std::size_t>(n), "%s", fmt);
    } else {
        m = std::snprintf(buf + n, sizeof(buf) - 1 - static_cast<std::size_t>(n), fmt,
                          std::forward<Args>(args)...);
    }
    if (m < 0) {
        return;
    }
    std::size_t len = static_cast<std::size_t>(n) + static_cast<std::size_t>(m);
    if (len > sizeof(buf) - 2) {
        len = sizeof(buf) - 2;
    }
    buf[len] = '\n';
    buf[len + 1] = 0;
    host_log_line(buf);
}
