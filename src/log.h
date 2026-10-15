#pragma once

#include <cstdio>
#include <cstring>
#include <utility>
#if !defined(_WIN32)
#include <unistd.h>
#endif

// The setup window's "Tooling and logging" (bbhost.toml [debug] tooling,
// BBHOST_TOOLING=0|1; on by default): the periodic reports - the 300-flip
// block, the renderer's, the GX threads', the presenter's - the stall
// detector, the GPU busy meter and the recent-draw ring. Off, none of them
// runs and the log has the start, warnings and errors. Set once, by main()
// after the configuration is final, before the game's threads start.
inline bool g_host_tooling = true;
inline bool host_tooling() { return g_host_tooling; }

// One write per line so lines from different threads do not interleave.
inline void host_log_line(const char* line) {
#if defined(_WIN32)
    std::fputs(line, stderr);
#else
    // write(2), not stdio: async-signal-safe (the fault handler logs) and no
    // FILE lock to deadlock on when a fault lands inside fputs.
    const char* p = line;
    std::size_t left = std::strlen(line);
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

template <typename... Args>
inline void host_log(const char* fmt, Args&&... args) {
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
