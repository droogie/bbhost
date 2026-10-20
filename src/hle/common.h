#pragma once

#include "core/host_clock.h"
#include "core/imports.h"
#include "guest_abi.h"
#include "log.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <time.h>  // clock_gettime: winpthreads' on Windows

inline constexpr int kSceOk = 0;

// Host errno -> FreeBSD (Orbis) errno. Microsoft CRT values also differ:
// EAGAIN is 11 there (guest 35), EDEADLK is 36 (guest 11).
inline int to_freebsd(int e) {
    if (e == 0) {
        return 0;
    }
#if defined(__linux__) || defined(_WIN32)
    switch (e) {
        case EAGAIN: return 35;
        case EDEADLK: return 11;
        case EINPROGRESS: return 36;
        case EALREADY: return 37;
        case ENOTSOCK: return 38;
        case EDESTADDRREQ: return 39;
        case EMSGSIZE: return 40;
        case EPROTOTYPE: return 41;
        case ENOPROTOOPT: return 42;
        case EPROTONOSUPPORT: return 43;
#ifdef ESOCKTNOSUPPORT
        case ESOCKTNOSUPPORT: return 44;
#endif
        case EOPNOTSUPP: return 45;
#ifdef EPFNOSUPPORT
        case EPFNOSUPPORT: return 46;
#endif
        case EAFNOSUPPORT: return 47;
        case EADDRINUSE: return 48;
        case EADDRNOTAVAIL: return 49;
        case ENETDOWN: return 50;
        case ENETUNREACH: return 51;
        case ENETRESET: return 52;
        case ECONNABORTED: return 53;
        case ECONNRESET: return 54;
        case ENOBUFS: return 55;
        case EISCONN: return 56;
        case ENOTCONN: return 57;
#ifdef ESHUTDOWN
        case ESHUTDOWN: return 58;
#endif
#ifdef ETOOMANYREFS
        case ETOOMANYREFS: return 59;
#endif
        case ETIMEDOUT: return 60;
        case ECONNREFUSED: return 61;
        case ELOOP: return 62;
        case ENAMETOOLONG: return 63;
#ifdef EHOSTDOWN
        case EHOSTDOWN: return 64;
#endif
        case EHOSTUNREACH: return 65;
        case ENOTEMPTY: return 66;
#ifdef EUSERS
        case EUSERS: return 68;
#endif
#ifdef EDQUOT
        case EDQUOT: return 69;
#endif
#ifdef ESTALE
        case ESTALE: return 70;
#endif
#ifdef EREMOTE
        case EREMOTE: return 71;
#endif
        case ENOLCK: return 77;
        case ENOSYS: return 78;
        case EIDRM: return 82;
        case ENOMSG: return 83;
        case EOVERFLOW: return 84;
        case ECANCELED: return 85;
        case EILSEQ: return 86;
        case EBADMSG: return 89;
#ifdef EMULTIHOP
        case EMULTIHOP: return 90;
#endif
        case ENOLINK: return 91;
        case EPROTO: return 92;
        case ENOTRECOVERABLE: return 95;
        case EOWNERDEAD: return 96;
        default: break;
    }
#elif defined(__APPLE__)
    // macOS matches FreeBSD up to ENEEDAUTH (81), plus EOVERFLOW and ENOTSUP.
    switch (e) {
        case EIDRM: return 82;
        case ENOMSG: return 83;
        case ECANCELED: return 85;
        case EILSEQ: return 86;
        case ENOATTR: return 87;
        case EBADMSG: return 89;
        case EMULTIHOP: return 90;
        case ENOLINK: return 91;
        case EPROTO: return 92;
        case EOPNOTSUPP: return 45;
        case ENOTRECOVERABLE: return 95;
        case EOWNERDEAD: return 96;
        // macOS-only: the nearest FreeBSD code, not the one with their number.
        case ETIME: return 60;   // ETIMEDOUT
        case EQFULL: return 55;  // ENOBUFS
        case EPWROFF:
        case EDEVERR:
        case EBADEXEC:
        case EBADARCH:
        case ESHLIBVERS:
        case EBADMACHO:
        case ENODATA:
        case ENOSR:
        case ENOSTR:
        case ENOPOLICY: return 5;  // EIO
        default: break;
    }
#endif
    return e;
}

// Guest-visible errno (what __error() points at), one per thread. Every HLE
// call that fails the libc way (-1 / NULL) must set it.
extern thread_local int t_guest_errno;
inline void set_guest_errno_host(int host_errno) { t_guest_errno = to_freebsd(host_errno); }
inline void set_guest_errno_bsd(int bsd_errno) { t_guest_errno = bsd_errno; }
// Propagate a host libc result: on -1 copy errno across. Returns r.
inline int libc_result(int r) {
    if (r < 0) {
        set_guest_errno_host(errno);
    }
    return r;
}

inline int sce_err(int e) {
    if (e == 0) {
        return kSceOk;
    }
    return static_cast<int>(0x80020000u | (static_cast<unsigned>(to_freebsd(e)) & 0xffffu));
}

// The guest's microsecond clock and TSC tick with the monotonic clock on both
// hosts (winpthreads' is QueryPerformanceCounter). On Windows they were
// GetTickCount64, which moves in 15.6 ms steps whatever timeBeginPeriod says:
// at 60 fps about one step a frame. BBHOST_CLOCK_QUANTUM_US=N (checks) puts
// them back on steps of N us, 15625 for that clock.
inline std::uint64_t guest_clock_quantum_us() {
    static const std::uint64_t q = [] {
        const char* e = std::getenv("BBHOST_CLOCK_QUANTUM_US");
        return e ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    return q;
}

inline std::uint64_t monotonic_ns() {
    const std::uint64_t ns = host_clock_monotonic_ns();  // QPC / CLOCK_MONOTONIC, core/host_clock.h
    if (const std::uint64_t q = guest_clock_quantum_us()) return ns / (q * 1000ull) * (q * 1000ull);
    return ns;
}

inline std::uint64_t now_us() { return monotonic_ns() / 1000ull; }

// The PS4's time-stamp counter runs at a fixed 1.6 GHz and the game
// converts sceKernelReadTsc deltas with that constant (it never asks for
// the frequency). A synthesized 1.6 GHz counter from the monotonic clock
// keeps its timers honest whatever the host TSC runs at (a 2.1 GHz host
// made menus repeat keys and pace animations 1.3x too fast). Scaled from the
// performance counter directly (x160 from Windows' 10 MHz).
constexpr std::uint64_t kGuestTscHz = 1600000000ull;
inline std::uint64_t rdtsc_now() {
    if (guest_clock_quantum_us()) {
        const std::uint64_t ns = monotonic_ns();
        return ns / 1000000000ull * kGuestTscHz + ns % 1000000000ull * 16ull / 10ull;
    }
    return host_clock_ticks_at(kGuestTscHz);
}

// The guest's clock reads by entry, since the start. They run raw (no
// thunk), so BBHOST_HLE_COUNT does not see them; the 300-flip report's
// timing line gives these as rates (hle_timing_window, kernel.cpp).
struct HleClockReads {
    HostShardedCount gettimeofday, clock_gettime, process_time, tsc, time, clock;
};
extern HleClockReads g_hle_clock_reads;
// Whether the clock entries are bound raw (GS mode and BBHOST_RAW_CLOCK not 0).
bool hle_raw_clock();
// Clock reads per second over `secs`, since the last call, and (into `sync`)
// the sync layer's slow paths: waits that slept, wakes (kernel.cpp).
std::string hle_timing_window(double secs, std::string* sync);

GUEST_ABI int hle_ok();

#define HLE_REG(name, fn) register_hle_fn((name), reinterpret_cast<void*>(fn))
