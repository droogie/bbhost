#include "core/host_clock.h"
#include "log.h"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#include <timeapi.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace {

bool env_off(const char* name) {
    const char* e = std::getenv(name);
    return e && e[0] == '0';
}

#if defined(_WIN32)
// QueryPerformanceCounter ticks to nanoseconds and to other rates. The
// frequency is fixed at boot (10 MHz on Windows 10 and later, where QPC runs
// off the invariant TSC through the shared user page), so the scale is set
// up once: a whole multiplier when the rate divides a second evenly, else a
// 32.32 fixed-point one through a 128-bit product.
struct QpcScale {
    std::uint64_t freq = 1;
    std::uint64_t ns_whole = 0;  // ns per tick, when exact
    std::uint64_t ns_fixed = 0;  // (1e9 << 32) / freq otherwise
};
QpcScale make_qpc_scale() {
    QpcScale s;
    LARGE_INTEGER f;
    s.freq = QueryPerformanceFrequency(&f) && f.QuadPart > 0 ? static_cast<std::uint64_t>(f.QuadPart) : 10000000ull;
    if (1000000000ull % s.freq == 0) {
        s.ns_whole = 1000000000ull / s.freq;
    } else {
        s.ns_fixed = static_cast<std::uint64_t>((static_cast<unsigned __int128>(1000000000ull) << 32) / s.freq);
    }
    return s;
}
// A function's static, not a namespace one: the first reader sets it up
// whatever the order of static initialization (host_timing_init reads it
// first, at the start of main).
const QpcScale& qpc() {
    static const QpcScale s = make_qpc_scale();
    return s;
}

inline std::uint64_t qpc_now() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return static_cast<std::uint64_t>(c.QuadPart);
}

// Ticks to `hz`, exact for any whole number of seconds and the remainder in
// one 128-bit multiply-divide; no 64-bit overflow for centuries of uptime.
inline std::uint64_t qpc_scale_to(std::uint64_t ticks, std::uint64_t hz) {
    const std::uint64_t f = qpc().freq;
    const std::uint64_t sec = ticks / f, rem = ticks % f;
    return sec * hz + static_cast<std::uint64_t>(static_cast<unsigned __int128>(rem) * hz / f);
}
#endif

}  // namespace

std::uint64_t host_clock_monotonic_ns() {
#if defined(_WIN32)
    const QpcScale& s = qpc();
    const std::uint64_t t = qpc_now();
    if (s.ns_whole) return t * s.ns_whole;
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(t) * s.ns_fixed) >> 32);
#else
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<std::uint64_t>(ts.tv_nsec);
#endif
}

std::int64_t host_clock_realtime_ns() {
#if defined(_WIN32)
    // 100 ns units since 1601; the precise form reads the same counter as
    // QPC plus the system's offset (no system call on Windows 8 and later).
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    const std::uint64_t t = (static_cast<std::uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
    constexpr std::uint64_t kUnixEpoch = 116444736000000000ull;
    return static_cast<std::int64_t>(t - kUnixEpoch) * 100;
#else
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<std::int64_t>(ts.tv_sec) * 1000000000ll + static_cast<std::int64_t>(ts.tv_nsec);
#endif
}

std::uint64_t host_clock_ticks_at(std::uint64_t hz) {
#if defined(_WIN32)
    const std::uint64_t f = qpc().freq;
    if (hz == f) return qpc_now();
    if (hz % f == 0) return qpc_now() * (hz / f);  // 1.6 GHz from 10 MHz: x160
    return qpc_scale_to(qpc_now(), hz);
#else
    const std::uint64_t ns = host_clock_monotonic_ns();
    return ns / 1000000000ull * hz + static_cast<std::uint64_t>(static_cast<unsigned __int128>(ns % 1000000000ull) * hz / 1000000000ull);
#endif
}

void host_timing_init() {
#if defined(_WIN32)
    // 1 ms timer resolution for the whole run. host_sleep_us's waitable
    // timers are high-resolution and do not need it; Sleep, the timed waits
    // (WaitOnAddress, winpthreads' condition variables) and the driver's own
    // waits do.
    const bool period = !env_off("BBHOST_TIMER_PERIOD");
    if (period) timeBeginPeriod(1);
    // Power throttling off for the process (HighQoS): EXECUTION_SPEED keeps
    // Windows from running its threads at efficiency clocks (EcoQoS) when it
    // judges the window unimportant, IGNORE_TIMER_RESOLUTION keeps the 1 ms
    // above honoured while the window is hidden or minimized (Windows 11
    // drops it otherwise). Looked up by name: SetProcessInformation is
    // Windows 8 and later, and older SDK headers lack the flags.
    bool qos = false;
    if (!env_off("BBHOST_WIN_QOS")) {
        struct PowerThrottling {
            ULONG version, control_mask, state_mask;
        };
        constexpr ULONG kExecutionSpeed = 0x1, kIgnoreTimerResolution = 0x4;
        constexpr int kProcessPowerThrottling = 4;  // PROCESS_INFORMATION_CLASS
        using SetInfo = BOOL(WINAPI*)(HANDLE, int, LPVOID, DWORD);
        HMODULE k = GetModuleHandleW(L"kernel32.dll");
        const auto set_info = k ? reinterpret_cast<SetInfo>(reinterpret_cast<void*>(GetProcAddress(k, "SetProcessInformation"))) : nullptr;
        if (set_info) {
            PowerThrottling s{1, kExecutionSpeed | kIgnoreTimerResolution, 0};
            qos = set_info(GetCurrentProcess(), kProcessPowerThrottling, &s, sizeof(s)) != 0;
            if (!qos) {
                // Before Windows 11 the timer-resolution flag is unknown and
                // the call fails: the execution-speed half alone.
                s.control_mask = kExecutionSpeed;
                qos = set_info(GetCurrentProcess(), kProcessPowerThrottling, &s, sizeof(s)) != 0;
            }
        }
    }
    // What the system timer runs at now (100 ns units), from ntdll.
    using QueryRes = LONG(NTAPI*)(PULONG, PULONG, PULONG);
    HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    const auto query = nt ? reinterpret_cast<QueryRes>(reinterpret_cast<void*>(GetProcAddress(nt, "NtQueryTimerResolution"))) : nullptr;
    ULONG coarsest = 0, finest = 0, now = 0;
    const bool have_res = query && query(&coarsest, &finest, &now) == 0;
    host_log("timing: timer resolution %s (system now %.3f ms, finest %.3f ms); power throttling %s; QPC %llu Hz",
             period ? "1 ms asked at start" : "left to the system (BBHOST_TIMER_PERIOD=0)",
             have_res ? now / 10000.0 : 0.0, have_res ? finest / 10000.0 : 0.0,
             env_off("BBHOST_WIN_QOS") ? "left on (BBHOST_WIN_QOS=0)" : (qos ? "off for this process (HighQoS, resolution kept when hidden)" : "could not be switched off"),
             static_cast<unsigned long long>(qpc().freq));
#endif
}

void host_thread_set_class(HostThreadClass c, const char* who) {
    static const bool on = !env_off("BBHOST_THREAD_PRIO");
#if defined(_WIN32)
    const char* const level = c == HostThreadClass::Pacing ? "highest" : "above-normal";
    if (!on) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) host_log("timing: the pacing threads run at normal priority (BBHOST_THREAD_PRIO=0)");
        return;
    }
    const int prio = c == HostThreadClass::Pacing ? THREAD_PRIORITY_HIGHEST : THREAD_PRIORITY_ABOVE_NORMAL;
    // Each of these threads starts once (a command processor per queue), so
    // one line each.
    if (SetThreadPriority(GetCurrentThread(), prio)) {
        host_log("timing: %s runs at %s priority (BBHOST_THREAD_PRIO=0: normal)", who, level);
    } else {
        host_log("timing: %s could not be set to %s priority (error %lu)", who, level, static_cast<unsigned long>(GetLastError()));
    }
#elif defined(__APPLE__)
    // macOS schedules by QoS class, not priority. The game's threads keep the
    // default class (still P-cores); these go above it.
    (void)c;
    if (!on) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) host_log("timing: the pacing threads run at the default QoS class (BBHOST_THREAD_PRIO=0)");
        return;
    }
    if (const int e = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); e == 0) {
        host_log("timing: %s runs at the user-interactive QoS class (BBHOST_THREAD_PRIO=0: default)", who);
    } else {
        host_log("timing: %s could not be set to the user-interactive QoS class (%s)", who, std::strerror(e));
    }
#else
    (void)on;
    (void)c;
    (void)who;
#endif
}
