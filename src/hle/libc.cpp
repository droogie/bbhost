#include "core/portable.h"
#include "host/gpu.h"
#include "host/frame_stats.h"
#include <locale>
#include "core/write_watch.h"
#include "hle/common.h"
#include "hle/libc_ctype.h"
#include "hle/fs.h"
#include "core/tls_rewrite.h"
#include "hle/hle.h"
#include "core/thunk.h"
#include "hle/modules.h"
#include "hle/platform.h"
#include "hle/sysv_va.h"

#include <atomic>
#include <condition_variable>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <limits>
#include <csetjmp>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/time.h>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {


alignas(8) std::uint64_t g_stack_chk_guard = 0x2a2a2a2a2a2a2a2aull;
alignas(64) static unsigned char g_dummy_object[512];
FILE* g_stdin_slot = stdin;
FILE* g_stdout_slot = stdout;
FILE* g_stderr_slot = stderr;
int g_need_scelibc = 1;
std::mutex g_atexit_mu;
struct AtexitEntry {
    void* fn;
    void* arg;
    bool cxa;
};
std::vector<AtexitEntry> g_atexit_fns;

// std::qsort has no context argument; the guest comparator for the current
// sort lives here and is entered through hle_call_guest (guest FS).
thread_local void* t_qsort_cmp = nullptr;
int qsort_host_cmp(const void* a, const void* b) {
    return hle_call_guest<int>(t_qsort_cmp, a, b);
}

short g_ctype[257];
short g_tolower[257];
short g_toupper[257];
lconv g_lconv{};
std::once_flag g_ctype_once;

void init_ctype() {
    for (int i = 0; i < 257; ++i) {
        const int c = i - 1;
        g_ctype[i] = bb::libc::ctype_flags(c);
        g_tolower[i] = static_cast<short>(c >= 'A' && c <= 'Z' ? c + 32 : c);
        g_toupper[i] = static_cast<short>(c >= 'a' && c <= 'z' ? c - 32 : c);
    }
    if (struct lconv* lc = localeconv()) {
        g_lconv = *lc;
    }
}

GUEST_ABI void hle_init_env() {}
GUEST_ABI void hle_catch_return_from_main(int64_t) {}

GUEST_ABI void hle_stack_chk_fail() {
    host_log("__stack_chk_fail");
    std::abort();
}

GUEST_ABI int hle_atexit(void* fn) {
    std::lock_guard<std::mutex> lock(g_atexit_mu);
    g_atexit_fns.push_back(AtexitEntry{fn, nullptr, false});
    return 0;
}
GUEST_ABI int hle_cxa_atexit(void* fn, void* arg, void*) {
    std::lock_guard<std::mutex> lock(g_atexit_mu);
    g_atexit_fns.push_back(AtexitEntry{fn, arg, true});
    return 0;
}
void run_guest_atexit() {
    std::vector<AtexitEntry> fns;
    {
        std::lock_guard<std::mutex> lock(g_atexit_mu);
        fns.swap(g_atexit_fns);
    }
    for (auto it = fns.rbegin(); it != fns.rend(); ++it) {
        if (it->cxa) {
            hle_call_guest<std::int64_t>(it->fn, it->arg);
        } else {
            hle_call_guest<std::int64_t>(it->fn);
        }
    }
}
GUEST_ABI void hle_exit(int code) {
    host_log("guest exit(%d): running %zu atexit handlers", code, g_atexit_fns.size());
    run_guest_atexit();
    host_gpu_save_pipeline_cache_at_exit();
    hle_kernel_memory_report();
    std::exit(code);
}
GUEST_ABI void hle_abort() { std::abort(); }

// Guest allocations come from the host heap, where a block can hold what host
// code freed there. They are zeroed: the game reads fields it never sets. The
// blend-state constructor (0x2e51640) ORs each render target's write mask into
// +0x164 without clearing it, so leftover bits reached CB_TARGET_MASK and a
// YEBIS pass that writes alpha only wrote colour too (a red, green or cream
// veil over the world, depending on the run).
GUEST_ABI void* hle_malloc(std::uint64_t n) { return host_guest_calloc(1, n ? n : 1); }
GUEST_ABI void hle_free(void* p) { host_guest_free(p); }
GUEST_ABI void* hle_calloc(std::uint64_t n, std::uint64_t sz) { return host_guest_calloc(n, sz); }
GUEST_ABI void* hle_realloc(void* p, std::uint64_t n) { return host_guest_realloc(p, n); }
// memalign's alignment is the caller's contract; calloc only promises 16.
// Zeroed, like the others.
GUEST_ABI void* hle_memalign(std::uint64_t align, std::uint64_t n) {
    if (align <= 16) return host_guest_calloc(1, n ? n : 1);
    if (align & (align - 1)) return nullptr;  // not a power of two, as on the console
    return host_guest_memalign(static_cast<std::size_t>(align), static_cast<std::size_t>(n));
}
GUEST_ABI void* hle_new_nothrow(std::uint64_t n, const void*) { return host_guest_calloc(1, n ? n : 1); }

GUEST_ABI void* hle_memcpy(void* d, const void* s, std::uint64_t n) { return std::memcpy(d, s, n); }
GUEST_ABI void* hle_memmove(void* d, const void* s, std::uint64_t n) { return std::memmove(d, s, n); }
GUEST_ABI void* hle_memset(void* d, int c, std::uint64_t n) { return std::memset(d, c, n); }
// Sony's libc (Dinkum) returns exactly -1/0/1 from every compare, and the
// eboot depends on it: the behavior-name map lookup tests wcscmp() == 1.
inline int sgn(int v) { return v < 0 ? -1 : (v > 0 ? 1 : 0); }
GUEST_ABI int hle_memcmp(const void* a, const void* b, std::uint64_t n) { return sgn(std::memcmp(a, b, n)); }
GUEST_ABI const void* hle_memchr(const void* s, int c, std::uint64_t n) { return std::memchr(s, c, n); }

GUEST_ABI std::uint64_t hle_strlen(const char* s) { return s ? std::strlen(s) : 0; }
GUEST_ABI char* hle_strcpy(char* d, const char* s) { return std::strcpy(d, s); }
GUEST_ABI char* hle_strncpy(char* d, const char* s, std::uint64_t n) { return std::strncpy(d, s, n); }
GUEST_ABI int hle_strcmp(const char* a, const char* b) { return sgn(std::strcmp(a, b)); }
GUEST_ABI int hle_strncmp(const char* a, const char* b, std::uint64_t n) { return sgn(std::strncmp(a, b, n)); }
GUEST_ABI char* hle_strcat(char* d, const char* s) { return std::strcat(d, s); }
GUEST_ABI char* hle_strncat(char* d, const char* s, std::uint64_t n) { return std::strncat(d, s, n); }
GUEST_ABI const char* hle_strchr(const char* s, int c) { return std::strchr(s, c); }
GUEST_ABI const char* hle_strrchr(const char* s, int c) { return std::strrchr(s, c); }
GUEST_ABI const char* hle_strstr(const char* h, const char* n) { return std::strstr(h, n); }
GUEST_ABI char* hle_strtok(char* s, const char* d) { return std::strtok(s, d); }
GUEST_ABI std::uint64_t hle_strcspn(const char* s, const char* r) { return std::strcspn(s, r); }
GUEST_ABI std::uint64_t hle_strspn(const char* s, const char* r) { return std::strspn(s, r); }
GUEST_ABI const char* hle_strpbrk(const char* s, const char* r) { return std::strpbrk(s, r); }
GUEST_ABI int hle_strcasecmp(const char* a, const char* b) {
#if defined(_WIN32)
    return sgn(_stricmp(a, b));
#else
    return sgn(strcasecmp(a, b));
#endif
}
GUEST_ABI int hle_strncasecmp(const char* a, const char* b, std::uint64_t n) {
#if defined(_WIN32)
    return sgn(_strnicmp(a, b, static_cast<std::size_t>(n)));
#else
    return sgn(strncasecmp(a, b, static_cast<std::size_t>(n)));
#endif
}
GUEST_ABI int hle_strcoll(const char* a, const char* b) { return sgn(std::strcmp(a, b)); }
GUEST_ABI const char* hle_strerror(int e) { return std::strerror(e); }
GUEST_ABI long hle_strtol(const char* s, char** end, int base) { return std::strtol(s, end, base); }
GUEST_ABI std::uint64_t hle_strtoumax(const char* s, char** end, int base) {
    return static_cast<std::uint64_t>(std::strtoull(s, end, base));
}

// The printf family walks the guest's own va_list (hle/sysv_va.h): the
// v-entries get the guest's pointer, the variadic ones are asm shims
// (SYSV_VA_ENTRY, after this namespace) that build one from the registers.
GUEST_ABI int hle_vprintf(const char* fmt, SysvVaList* ap) { return sysv_vfprintf(stdout, fmt, ap); }
GUEST_ABI int hle_vsnprintf(char* b, std::uint64_t n, const char* fmt, SysvVaList* ap) {
    return sysv_vsnprintf(b, n, fmt, ap);
}
GUEST_ABI int hle_vsprintf(char* b, const char* fmt, SysvVaList* ap) {
    return sysv_vsnprintf(b, ~std::size_t(0), fmt, ap);
}
GUEST_ABI int hle_vfprintf(FILE* f, const char* fmt, SysvVaList* ap) { return sysv_vfprintf(f ? f : stdout, fmt, ap); }
GUEST_ABI int hle_puts(const char* s) { return std::puts(s); }
GUEST_ABI int hle_putchar(int c) { return std::putchar(c); }
GUEST_ABI void hle_perror(const char* s) { std::perror(s); }

GUEST_ABI FILE* hle_fopen(const char* path, const char* mode) {
    std::string host = hle_fs_map_path(path);
    if (host.empty()) {
        return nullptr;
    }
    FILE* f = std::fopen(host.c_str(), mode ? mode : "r");
    static int logs;
    if (logs < 8) {
        host_log("fopen %s -> %s %s %s", path ? path : "", host.c_str(), mode ? mode : "",
                 f ? "ok" : "fail");
        ++logs;
    }
    // Diagnostic: at the first menu .gfx open, log the SprjScaleform
    // resident-menu-table selector counters (guest globals). >0 on any of
    // them makes sub_235d0e0 pick a reduced menu table (6/9/22 entries)
    // instead of the full 79, so title.gfx and most menus never load.
    static const bool trace_menuload = [] {
        const char* e = std::getenv("BBHOST_TRACE_MENULOAD");
        return e && e[0] == '1';
    }();
    if (trace_menuload && path && std::strstr(path, "/menu/") && std::strstr(path, ".gfx")) {
        static std::atomic<int> probed{0};
        if (probed.fetch_add(1) < 200) {
            // These live in the eboot's own data segment (guest VA == host
            // address), which is always mapped; hle_kernel_va_mapped only
            // tracks direct-memory maps, so read directly.
            auto rd = [](std::uint64_t va) -> long {
                return static_cast<long>(*reinterpret_cast<const volatile std::int32_t*>(static_cast<std::uintptr_t>(va)));
            };
            const std::uint64_t sp = *reinterpret_cast<const volatile std::uint64_t*>(static_cast<std::uintptr_t>(0x5940368));
            const std::uint64_t loader = sp ? *reinterpret_cast<const volatile std::uint64_t*>(static_cast<std::uintptr_t>(sp + 8)) : 0;
            host_log("menu-load %s: selectors d08=%ld fd8=%ld a38=%ld; scaleform=0x%llx loader(+8)=0x%llx",
                     std::strrchr(path, '/') ? std::strrchr(path, '/') + 1 : path,
                     rd(0x5978d08), rd(0x5978fd8), rd(0x5978a38),
                     static_cast<unsigned long long>(sp), static_cast<unsigned long long>(loader));
        }
    }
    return f;
}
GUEST_ABI FILE* hle_freopen(const char* path, const char* mode, FILE* f) {
    std::string host = hle_fs_map_path(path);
    if (host.empty()) {
        return nullptr;
    }
    return std::freopen(host.c_str(), mode ? mode : "r", f);
}
GUEST_ABI int hle_fclose(FILE* f) { return f ? std::fclose(f) : 0; }
GUEST_ABI int hle_fflush(FILE* f) { return std::fflush(f); }
GUEST_ABI std::uint64_t hle_fread(void* p, std::uint64_t sz, std::uint64_t n, FILE* f) {
    return write_watch_fread(p, static_cast<std::size_t>(sz), static_cast<std::size_t>(n), f);
}
GUEST_ABI std::uint64_t hle_fwrite(const void* p, std::uint64_t sz, std::uint64_t n, FILE* f) {
    return std::fwrite(p, static_cast<std::size_t>(sz), static_cast<std::size_t>(n), f);
}
GUEST_ABI int hle_fseek(FILE* f, long off, int whence) { return std::fseek(f, off, whence); }
GUEST_ABI long hle_ftell(FILE* f) { return std::ftell(f); }
GUEST_ABI void hle_rewind(FILE* f) { std::rewind(f); }
GUEST_ABI int hle_feof(FILE* f) { return std::feof(f); }
GUEST_ABI int hle_ferror(FILE* f) { return std::ferror(f); }
GUEST_ABI void hle_clearerr(FILE* f) { std::clearerr(f); }
GUEST_ABI int hle_fgetc(FILE* f) { return std::fgetc(f); }
GUEST_ABI int hle_getc(FILE* f) { return std::getc(f); }
GUEST_ABI int hle_ungetc(int c, FILE* f) { return std::ungetc(c, f); }
GUEST_ABI char* hle_fgets(char* s, int n, FILE* f) { return std::fgets(s, n, f); }
GUEST_ABI int hle_fputc(int c, FILE* f) { return std::fputc(c, f); }
GUEST_ABI int hle_fputs(const char* s, FILE* f) { return std::fputs(s, f); }
GUEST_ABI int hle_fgetpos(FILE* f, std::fpos_t* p) { return std::fgetpos(f, p); }
GUEST_ABI int hle_setvbuf(FILE* f, char* b, int m, std::uint64_t n) {
    return std::setvbuf(f, b, m, static_cast<std::size_t>(n));
}

GUEST_ABI int hle_usleep(unsigned int usec) {
    MainThreadWait timed(1);
#if defined(_WIN32)
    host_sleep_us(usec);
    return 0;
#else
    return ::usleep(usec);
#endif
}
GUEST_ABI int hle_sce_usleep(unsigned int usec) { return hle_usleep(usec); }
GUEST_ABI unsigned int hle_sleep(unsigned int s) {
#if defined(_WIN32)
    Sleep(s * 1000);
    return 0;
#else
    return ::sleep(s);
#endif
}

// The guest's timeval is FreeBSD's: two 64-bit fields. mingw's is two
// 32-bit ones (8 bytes), so the host's struct cannot be written straight
// into the guest's - the game's save-load manager read garbage seconds from
// the high half, decided its last task was minutes old, tore the task's
// config down under it and mounted "0005" instead of "SPRJ0005" (6.3).
//
// Bound raw (hle_raw_clock): no errno, no thread-locals. The time is
// GetSystemTimePreciseAsFileTime read directly (core/host_clock.h), where
// it went through std::chrono and winpthreads' clock_gettime behind the
// thunk. A null tv is not an error: FreeBSD's gettimeofday fills what it is
// given and returns 0 (the timezone is left alone, as before).
GUEST_ABI int hle_gettimeofday(void* tv, void*) {
    g_hle_clock_reads.gettimeofday.add();
    if (!tv) return 0;
    const std::int64_t us = host_clock_realtime_us();
    const std::int64_t out[2] = {us / 1000000, us % 1000000};
    std::memcpy(tv, out, sizeof(out));
    return 0;
}
GUEST_ABI std::int64_t hle_time(std::int64_t* t) {
    g_hle_clock_reads.time.add();
    std::time_t now = std::time(nullptr);
    if (t) {
        *t = static_cast<std::int64_t>(now);
    }
    return static_cast<std::int64_t>(now);
}
GUEST_ABI std::int64_t hle_clock() {
    g_hle_clock_reads.clock.add();
    return static_cast<std::int64_t>(std::clock());
}
GUEST_ABI double hle_difftime(std::int64_t a, std::int64_t b) {
    return std::difftime(static_cast<std::time_t>(a), static_cast<std::time_t>(b));
}
// The guest's struct tm is FreeBSD's: the nine ints, then tm_gmtoff (long)
// and tm_zone (char*), 56 bytes. glibc's is the same; mingw's stops after
// the ints (36 bytes), so on Windows the guest layout is written by hand
// from the CRT's fields, the offset and the zone name from the CRT's
// timezone state, and never a CRT struct handed to the guest as-is (the
// gettimeofday lesson).
struct GuestTm {
    int sec, min, hour, mday, mon, year, wday, yday, isdst;
    int pad;
    std::int64_t gmtoff;
    const char* zone;
};
static_assert(sizeof(GuestTm) == 56);
#if defined(_WIN32)
void guest_tm_from(const std::tm& src, GuestTm* dst, bool utc) {
    dst->sec = src.tm_sec;
    dst->min = src.tm_min;
    dst->hour = src.tm_hour;
    dst->mday = src.tm_mday;
    dst->mon = src.tm_mon;
    dst->year = src.tm_year;
    dst->wday = src.tm_wday;
    dst->yday = src.tm_yday;
    dst->isdst = src.tm_isdst;
    dst->pad = 0;
    if (utc) {
        dst->gmtoff = 0;
        dst->zone = "UTC";
        return;
    }
    long tz = 0, dst_bias = 0;
    _get_timezone(&tz);  // seconds west of UTC, standard time
    _get_dstbias(&dst_bias);
    dst->gmtoff = -(tz + (src.tm_isdst > 0 ? dst_bias : 0));
    static thread_local char zone[64];
    std::size_t n = 0;
    if (_get_tzname(&n, zone, sizeof(zone), src.tm_isdst > 0 ? 1 : 0) != 0 || n == 0) std::strcpy(zone, "");
    dst->zone = zone;
}
thread_local GuestTm t_guest_tm;
#endif
GUEST_ABI void* hle_gmtime(const std::int64_t* t) {
    std::time_t tt = t ? static_cast<std::time_t>(*t) : 0;
#if defined(_WIN32)
    std::tm tmp{};
    if (gmtime_s(&tmp, &tt) != 0) return nullptr;
    guest_tm_from(tmp, &t_guest_tm, true);
    return &t_guest_tm;
#else
    return std::gmtime(&tt);
#endif
}
GUEST_ABI void* hle_localtime(const std::int64_t* t) {
    std::time_t tt = t ? static_cast<std::time_t>(*t) : 0;
#if defined(_WIN32)
    std::tm tmp{};
    if (localtime_s(&tmp, &tt) != 0) return nullptr;
    guest_tm_from(tmp, &t_guest_tm, false);
    return &t_guest_tm;
#else
    return std::localtime(&tt);
#endif
}
GUEST_ABI void* hle_localtime_s(const std::int64_t* t, GuestTm* buf) {
    std::time_t tt = t ? static_cast<std::time_t>(*t) : 0;
    if (!buf) {
        return nullptr;
    }
#if defined(_WIN32)
    std::tm tmp{};
    if (localtime_s(&tmp, &tt) != 0) return nullptr;
    guest_tm_from(tmp, buf, false);
    return buf;
#else
    return localtime_r(&tt, reinterpret_cast<std::tm*>(buf));
#endif
}
#if defined(_WIN32)
// The CRT's tm from the guest's: the nine ints lead both layouts.
std::tm crt_tm_from(const GuestTm* g) {
    std::tm t{};
    t.tm_sec = g->sec;
    t.tm_min = g->min;
    t.tm_hour = g->hour;
    t.tm_mday = g->mday;
    t.tm_mon = g->mon;
    t.tm_year = g->year;
    t.tm_wday = g->wday;
    t.tm_yday = g->yday;
    t.tm_isdst = g->isdst;
    return t;
}
#endif
GUEST_ABI std::int64_t hle_mktime(GuestTm* t) {
#if defined(_WIN32)
    std::tm c = crt_tm_from(t);
    const std::time_t r = std::mktime(&c);
    guest_tm_from(c, t, false);  // mktime normalizes its argument
    return static_cast<std::int64_t>(r);
#else
    return static_cast<std::int64_t>(std::mktime(reinterpret_cast<std::tm*>(t)));
#endif
}
GUEST_ABI std::uint64_t hle_strftime(char* s, std::uint64_t n, const char* fmt, const GuestTm* t) {
#if defined(_WIN32)
    const std::tm c = crt_tm_from(t);
    return std::strftime(s, static_cast<std::size_t>(n), fmt, &c);
#else
    return std::strftime(s, static_cast<std::size_t>(n), fmt, reinterpret_cast<const std::tm*>(t));
#endif
}


GUEST_ABI int* hle_error() { return &t_guest_errno; }
GUEST_ABI int hle_getpagesize() {
#if defined(_WIN32)
    return 4096;
#else
    return static_cast<int>(sysconf(_SC_PAGESIZE));
#endif
}

// Itanium C++ ABI function-local statics. The guard is 8 bytes: byte 0 is
// "initialised", byte 1 is "initialisation in progress". acquire() must claim
// the guard WITHOUT marking it initialised - only release() may do that.
// Marking it initialised in acquire() lets another thread's inlined
// `if (guard == 0)` fast path fall straight through and read the static before
// the initialiser has written it, yielding zeroes. Bloodborne hits this in the
// Scaleform matrix helpers (sub_4a7d60 / sub_4a7ff0), whose SIMD constant masks
// then come back zero and collapse every menu shape mesh.
std::mutex g_guard_mu;
std::condition_variable g_guard_cv;

GUEST_ABI int hle_cxa_guard_acquire(std::uint8_t* g) {
    if (!g) {
        return 0;
    }
    if (__atomic_load_n(&g[0], __ATOMIC_ACQUIRE)) {
        return 0;  // already initialised
    }
    std::unique_lock<std::mutex> lk(g_guard_mu);
    for (;;) {
        if (__atomic_load_n(&g[0], __ATOMIC_ACQUIRE)) {
            return 0;  // another thread finished it while we waited
        }
        if (!g[1]) {
            g[1] = 1;  // claim: in progress, NOT yet initialised
            return 1;
        }
        g_guard_cv.wait(lk);  // someone else is initialising; wait for release
    }
}
GUEST_ABI void hle_cxa_guard_release(std::uint8_t* g) {
    if (!g) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_guard_mu);
        g[1] = 0;
        __atomic_store_n(&g[0], 1, __ATOMIC_RELEASE);  // now visible as initialised
    }
    g_guard_cv.notify_all();
}
GUEST_ABI void hle_cxa_guard_abort(std::uint8_t* g) {
    if (!g) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_guard_mu);
        g[1] = 0;
        __atomic_store_n(&g[0], 0, __ATOMIC_RELEASE);
    }
    g_guard_cv.notify_all();
}
GUEST_ABI void* hle_cxa_allocate_exception(std::uint64_t n) { return std::malloc(n ? n + 64 : 64); }
GUEST_ABI void hle_cxa_throw(void*, void*, void*) {
    host_log("__cxa_throw");
    std::abort();
}
GUEST_ABI void hle_cxa_rethrow() {
    host_log("__cxa_rethrow");
    std::abort();
}
GUEST_ABI void* hle_cxa_begin_catch(void* p) { return p; }
GUEST_ABI void hle_cxa_end_catch() {}
GUEST_ABI void hle_cxa_pure_virtual() {
    host_log("__cxa_pure_virtual");
    std::abort();
}
GUEST_ABI void hle_cxa_call_unexpected() {
    host_log("__cxa_call_unexpected");
    std::abort();
}
GUEST_ABI void hle_unwind_resume() {
    host_log("_Unwind_Resume");
    std::abort();
}
GUEST_ABI void hle_terminate() {
    host_log("std::terminate");
    std::abort();
}
GUEST_ABI void hle_xbad_alloc() {
    host_log("std::bad_alloc");
    std::abort();
}
GUEST_ABI void hle_xlength(const char* s) {
    host_log("length_error %s", s ? s : "");
    std::abort();
}
GUEST_ABI void hle_xrange(const char* s) {
    host_log("out_of_range %s", s ? s : "");
    std::abort();
}
GUEST_ABI void hle_xbad_function() {
    host_log("bad_function_call");
    std::abort();
}
GUEST_ABI int hle_syserror_map(int e) { return e; }
GUEST_ABI void hle_throw_c_error(int e) {
    host_log("_Throw_C_error %d", e);
    std::abort();
}
// _Assert(message, function) does not abort. The game's own libc
// (sce_module/libc.prx, NID -QgqOT5u2Vk) writes "<message> in function
// <function>\n" to stderr, flushes, and returns, and the code that calls it
// carries on. YEBIS's motion blur asserts nBaseSamples >= 3 with an area's 2
// ("GPUMath.cpp(3199) ... in function GPUTexUtil_GetRecursiveSampleParameters")
// and then uses 2. Aborting here killed the game in the intro area
// (2026-09-29). The same message can come every frame: each is logged a few
// times, then counted.
GUEST_ABI void hle_assert(const char* message, const char* function) {
    static std::mutex mu;
    static std::unordered_map<std::uintptr_t, std::uint64_t> seen;  // by message address
    std::uint64_t n;
    {
        std::lock_guard<std::mutex> lock(mu);
        n = ++seen[reinterpret_cast<std::uintptr_t>(message)];
    }
    if (n <= 3 || (n & (n - 1)) == 0) {
        host_log("_Assert (the game carries on, as its libc does): %s%s%s%s", message ? message : "(null)",
                 function ? " in function " : "", function ? function : "",
                 n > 3 ? (" - " + std::to_string(n) + " times").c_str() : "");
    }
}



GUEST_ABI void hle_qsort(void* base, std::uint64_t n, std::uint64_t sz, int (*cmp)(const void*, const void*)) {
    if (base && cmp && n && sz) {
        void* saved = t_qsort_cmp;
        t_qsort_cmp = reinterpret_cast<void*>(cmp);
        std::qsort(base, static_cast<std::size_t>(n), static_cast<std::size_t>(sz), qsort_host_cmp);
        t_qsort_cmp = saved;
    }
}
GUEST_ABI void hle_srand(unsigned s) { std::srand(s); }
GUEST_ABI int hle_rand() { return std::rand(); }

GUEST_ABI int hle_mkdir(const char* p, int mode) {
    std::string host = hle_fs_map_path(p);
    if (host.empty()) {
        set_guest_errno_host(ENOENT);
        return -1;
    }
#if defined(_WIN32)
    (void)mode;
    return libc_result(_mkdir(host.c_str()));
#else
    return libc_result(::mkdir(host.c_str(), static_cast<mode_t>(mode ? mode : 0777)));
#endif
}
GUEST_ABI int hle_rmdir(const char* p) {
    std::string host = hle_fs_map_path(p);
    if (host.empty()) {
        set_guest_errno_host(ENOENT);
        return -1;
    }
#if defined(_WIN32)
    return libc_result(_rmdir(host.c_str()));
#else
    return libc_result(::rmdir(host.c_str()));
#endif
}
GUEST_ABI int hle_remove(const char* p) {
    std::string host = hle_fs_map_path(p);
    if (host.empty()) {
        set_guest_errno_host(ENOENT);
        return -1;
    }
    return libc_result(std::remove(host.c_str()));
}
GUEST_ABI int hle_rename(const char* a, const char* b) {
    std::string ha = hle_fs_map_path(a);
    std::string hb = hle_fs_map_path(b);
    if (ha.empty() || hb.empty()) {
        set_guest_errno_host(ENOENT);
        return -1;
    }
    return libc_result(std::rename(ha.c_str(), hb.c_str()));
}
GUEST_ABI int hle_stat(const char* p, void* sb) {
    int r = hle_fs_stat_path(p, sb);
    if (r == 0) {
        return 0;
    }
    // hle_fs_stat_path returns an SCE error whose low 16 bits are already BSD.
    set_guest_errno_bsd(static_cast<int>(static_cast<unsigned>(r) & 0xffffu));
    return -1;
}

GUEST_ABI char* hle_setlocale(int cat, const char* loc) { return std::setlocale(cat, loc); }
GUEST_ABI lconv* hle_localeconv() {
    std::call_once(g_ctype_once, init_ctype);
    return &g_lconv;
}
// The tables are built when libc registers (hle_register_libc), so these
// touch no thread-local state and run raw: the main loop calls them ~850
// times a frame.
GUEST_ABI short* hle_getpctype() { return g_ctype + 1; }
GUEST_ABI short* hle_getptolower() { return g_tolower + 1; }
GUEST_ABI short* hle_getptoupper() { return g_toupper + 1; }

GUEST_ABI double hle_stod(const char* s, char** end, long) { return std::strtod(s, end); }
GUEST_ABI long long hle_stoll(const char* s, char** end, long) { return std::strtoll(s, end, 10); }
GUEST_ABI unsigned long hle_stoul(const char* s, char** end, long) { return std::strtoul(s, end, 10); }
GUEST_ABI unsigned long long hle_stoull(const char* s, char** end, long) {
    return std::strtoull(s, end, 10);
}

GUEST_ABI int hle_dtest(double* p) {
    if (!p) {
        return 0;
    }
    if (std::isnan(*p)) {
        return 2;
    }
    if (std::isinf(*p)) {
        return 1;
    }
    if (*p == 0.0) {
        return -1;
    }
    return 0;
}

GUEST_ABI double hle_sin(double x) { return std::sin(x); }
// Dinkumware _Sin/_FSin(x, qoff) compute sin(x + qoff*pi/2), i.e. the quarter
// -cycle offset selects the function: qoff 0 = sin, 1 = cos, 2 = -sin, 3 = -cos.
// The C library's cos() is literally _Sin(x, 1), so ignoring qoff silently turns
// every cos() into sin(). Bloodborne's Scaleform matrix decomposition
// (sub_4a7ff0) builds a rotation from sin/cos of an angle that is 0 for an
// axis-aligned matrix; with qoff ignored cos(0) returned 0 instead of 1, making
// the mesh packing matrix all-zero and collapsing every menu shape.
// Registered raw (REG_RAW): sin and cos set errno only for an infinite
// argument, and errno is thread-local - on a guest thread, the guest's. The
// result for one is NaN either way.
GUEST_ABI double hle_sin_q(double x, unsigned q) {
    if (std::isinf(x)) return std::numeric_limits<double>::quiet_NaN();
    switch (q & 3) {
        case 0: return std::sin(x);
        case 1: return std::cos(x);
        case 2: return -std::sin(x);
        default: return -std::cos(x);
    }
}
GUEST_ABI float hle_sinf_q(float x, unsigned q) {
    if (std::isinf(x)) return std::numeric_limits<float>::quiet_NaN();
    switch (q & 3) {
        case 0: return std::sin(x);
        case 1: return std::cos(x);
        case 2: return -std::sin(x);
        default: return -std::cos(x);
    }
}
GUEST_ABI double hle_tan(double x) { return std::tan(x); }
GUEST_ABI float hle_tanf(float x) { return std::tanf(x); }
GUEST_ABI float hle_tanhf(float x) { return std::tanhf(x); }
GUEST_ABI double hle_acos(double x) { return std::acos(x); }
GUEST_ABI float hle_acosf(float x) { return std::acosf(x); }
GUEST_ABI double hle_asin(double x) { return std::asin(x); }
GUEST_ABI float hle_asinf(float x) { return std::asinf(x); }
GUEST_ABI double hle_atan(double x) { return std::atan(x); }
GUEST_ABI float hle_atanf(float x) { return std::atanf(x); }
GUEST_ABI double hle_atan2(double y, double x) { return std::atan2(y, x); }
GUEST_ABI float hle_atan2f(float y, float x) { return std::atan2f(y, x); }
GUEST_ABI double hle_exp(double x) { return std::exp(x); }
GUEST_ABI float hle_expf(float x) { return std::expf(x); }
GUEST_ABI double hle_exp2(double x) { return std::exp2(x); }
GUEST_ABI float hle_exp2f(float x) { return std::exp2f(x); }
GUEST_ABI double hle_log(double x) { return std::log(x); }
GUEST_ABI float hle_logf(float x) { return std::logf(x); }
GUEST_ABI double hle_pow(double x, double y) { return std::pow(x, y); }
GUEST_ABI float hle_powf(float x, float y) { return std::powf(x, y); }
GUEST_ABI double hle_sqrt(double x) { return std::sqrt(x); }
GUEST_ABI float hle_sqrtf(float x) { return std::sqrtf(x); }
GUEST_ABI double hle_ceil(double x) { return std::ceil(x); }
GUEST_ABI float hle_ceilf(float x) { return std::ceilf(x); }
GUEST_ABI double hle_floor(double x) { return std::floor(x); }
GUEST_ABI float hle_floorf(float x) { return std::floorf(x); }
GUEST_ABI double hle_fabs(double x) { return std::fabs(x); }
GUEST_ABI float hle_fabsf(float x) { return std::fabsf(x); }
GUEST_ABI double hle_fmod(double x, double y) { return std::fmod(x, y); }
GUEST_ABI float hle_fmodf(float x, float y) { return std::fmodf(x, y); }
GUEST_ABI double hle_frexp(double x, int* e) { return std::frexp(x, e); }
GUEST_ABI float hle_frexpf(float x, int* e) { return std::frexpf(x, e); }
GUEST_ABI double hle_ldexp(double x, int e) { return std::ldexp(x, e); }
GUEST_ABI float hle_ldexpf(float x, int e) { return std::ldexpf(x, e); }
GUEST_ABI double hle_modf(double x, double* i) { return std::modf(x, i); }
GUEST_ABI float hle_modff(float x, float* i) { return std::modff(x, i); }
GUEST_ABI float hle_coshf(float x) { return std::coshf(x); }
GUEST_ABI float hle_sinhf(float x) { return std::sinhf(x); }

GUEST_ABI ldiv_t hle_ldiv(long n, long d) { return std::ldiv(n, d); }

// Orbis wchar_t is UTF-16. Linux wchar_t is UTF-32 — never wrap host wcs*.
using Gwchar = char16_t;

GUEST_ABI std::uint64_t hle_wcslen(const Gwchar* s) {
    std::uint64_t n = 0;
    if (s) {
        while (s[n]) {
            ++n;
        }
    }
    return n;
}
GUEST_ABI Gwchar* hle_wcscpy(Gwchar* d, const Gwchar* s) {
    Gwchar* out = d;
    if (d && s) {
        while ((*d++ = *s++) != 0) {
        }
    }
    return out;
}
GUEST_ABI Gwchar* hle_wcsncpy(Gwchar* d, const Gwchar* s, std::uint64_t n) {
    if (!d) {
        return d;
    }
    std::uint64_t i = 0;
    if (s) {
        for (; i < n && s[i]; ++i) {
            d[i] = s[i];
        }
    }
    for (; i < n; ++i) {
        d[i] = 0;
    }
    return d;
}
GUEST_ABI int hle_wcscmp(const Gwchar* a, const Gwchar* b) {
    if (!a || !b) {
        return a == b ? 0 : (a ? 1 : -1);
    }
    while (*a && *a == *b) {
        ++a;
        ++b;
    }
    return sgn(static_cast<int>(*a) - static_cast<int>(*b));
}
GUEST_ABI int hle_wcsncmp(const Gwchar* a, const Gwchar* b, std::uint64_t n) {
    for (std::uint64_t i = 0; i < n; ++i) {
        Gwchar ca = a ? a[i] : 0;
        Gwchar cb = b ? b[i] : 0;
        if (ca != cb || ca == 0) {
            return sgn(static_cast<int>(ca) - static_cast<int>(cb));
        }
    }
    return 0;
}
GUEST_ABI const Gwchar* hle_wcschr(const Gwchar* s, Gwchar c) {
    if (!s) {
        return nullptr;
    }
    for (; *s; ++s) {
        if (*s == c) {
            return s;
        }
    }
    return c == 0 ? s : nullptr;
}
GUEST_ABI const Gwchar* hle_wcsrchr(const Gwchar* s, Gwchar c) {
    const Gwchar* last = nullptr;
    if (!s) {
        return nullptr;
    }
    for (; *s; ++s) {
        if (*s == c) {
            last = s;
        }
    }
    return c == 0 ? s : last;
}
GUEST_ABI const Gwchar* hle_wcsstr(const Gwchar* h, const Gwchar* n) {
    if (!h || !n) {
        return h;
    }
    if (!*n) {
        return h;
    }
    for (; *h; ++h) {
        const Gwchar* a = h;
        const Gwchar* b = n;
        while (*a && *b && *a == *b) {
            ++a;
            ++b;
        }
        if (!*b) {
            return h;
        }
    }
    return nullptr;
}
GUEST_ABI const Gwchar* hle_wcspbrk(const Gwchar* s, const Gwchar* r) {
    if (!s || !r) {
        return nullptr;
    }
    for (; *s; ++s) {
        for (const Gwchar* p = r; *p; ++p) {
            if (*s == *p) {
                return s;
            }
        }
    }
    return nullptr;
}
GUEST_ABI std::uint64_t hle_wcsspn(const Gwchar* s, const Gwchar* r) {
    std::uint64_t n = 0;
    if (!s) {
        return 0;
    }
    for (; s[n]; ++n) {
        bool ok = false;
        if (r) {
            for (const Gwchar* p = r; *p; ++p) {
                if (s[n] == *p) {
                    ok = true;
                    break;
                }
            }
        }
        if (!ok) {
            break;
        }
    }
    return n;
}
GUEST_ABI long hle_wcstol(const Gwchar* s, Gwchar** e, int b) {
    std::string tmp;
    if (s) {
        for (const Gwchar* p = s; *p && tmp.size() < 64; ++p) {
            tmp.push_back(static_cast<char>(*p));
        }
    }
    char* end = nullptr;
    long v = std::strtol(tmp.c_str(), &end, b);
    if (e && s) {
        *e = const_cast<Gwchar*>(s) + (end ? (end - tmp.c_str()) : tmp.size());
    }
    return v;
}
GUEST_ABI long long hle_wcstoll(const Gwchar* s, Gwchar** e, int b) {
    return hle_wcstol(s, e, b);
}
GUEST_ABI Gwchar* hle_wmemcpy(Gwchar* d, const Gwchar* s, std::uint64_t n) {
    if (d && s && n) {
        std::memcpy(d, s, static_cast<std::size_t>(n) * sizeof(Gwchar));
    }
    return d;
}
GUEST_ABI Gwchar* hle_wmemmove(Gwchar* d, const Gwchar* s, std::uint64_t n) {
    if (d && s && n) {
        std::memmove(d, s, static_cast<std::size_t>(n) * sizeof(Gwchar));
    }
    return d;
}
GUEST_ABI Gwchar* hle_wmemset(Gwchar* d, Gwchar c, std::uint64_t n) {
    if (d) {
        for (std::uint64_t i = 0; i < n; ++i) {
            d[i] = c;
        }
    }
    return d;
}
GUEST_ABI int hle_wmemcmp(const Gwchar* a, const Gwchar* b, std::uint64_t n) {
    for (std::uint64_t i = 0; i < n; ++i) {
        Gwchar ca = a ? a[i] : 0;
        Gwchar cb = b ? b[i] : 0;
        if (ca != cb) {
            return sgn(static_cast<int>(ca) - static_cast<int>(cb));
        }
    }
    return 0;
}
GUEST_ABI const Gwchar* hle_wmemchr(const Gwchar* s, Gwchar c, std::uint64_t n) {
    if (!s) {
        return nullptr;
    }
    for (std::uint64_t i = 0; i < n; ++i) {
        if (s[i] == c) {
            return s + i;
        }
    }
    return nullptr;
}
GUEST_ABI int hle_swprintf(Gwchar* b, std::uint64_t n, const Gwchar* fmt, ...) {
    if (!b || n == 0) {
        return 0;
    }
    b[0] = 0;
    (void)fmt;
    return 0;
}
GUEST_ABI int hle_wprintf(const Gwchar*, ...) { return 0; }
// swscanf: narrow both strings (ASCII data in practice) and defer to vsscanf.
// %s/%c conversions would need wide targets; they are rejected (return 0).
GUEST_ABI int hle_swscanf_body(const Gwchar* s, const Gwchar* fmt, SysvVaList* ap) {
    if (!s || !fmt) {
        return -1;
    }
    std::string ns, nf;
    for (const Gwchar* p = s; *p; ++p) {
        ns.push_back(*p < 0x80 ? static_cast<char>(*p) : '?');
    }
    for (const Gwchar* p = fmt; *p; ++p) {
        nf.push_back(*p < 0x80 ? static_cast<char>(*p) : '?');
    }
    for (std::size_t i = 0; i + 1 < nf.size(); ++i) {
        if (nf[i] == '%' && (nf[i + 1] == 's' || nf[i + 1] == 'c' || nf[i + 1] == '[')) {
            host_log("HLE fake: swscanf with %%%c rejected", nf[i + 1]);
            return 0;
        }
    }
    return sysv_vsscanf(ns.c_str(), nf.c_str(), ap);
}
GUEST_ABI double hle_wstod(const Gwchar* s, Gwchar** end, long) {
    std::string tmp;
    if (s) {
        for (const Gwchar* p = s; *p && tmp.size() < 64; ++p) {
            tmp.push_back(static_cast<char>(*p));
        }
    }
    char* e = nullptr;
    double v = std::strtod(tmp.c_str(), &e);
    if (end && s) {
        *end = const_cast<Gwchar*>(s) + (e ? (e - tmp.c_str()) : tmp.size());
    }
    return v;
}
GUEST_ABI float hle_wstof(const Gwchar* s, Gwchar** end, long) {
    return static_cast<float>(hle_wstod(s, end, 0));
}

GUEST_ABI void hle_ios_init() {}
GUEST_ABI void hle_ios_dtor(void*) {}
GUEST_ABI void hle_winit() {}
GUEST_ABI void hle_winit_dtor(void*) {}
GUEST_ABI void hle_locinfo_ctor(void*, const char*) {}
GUEST_ABI void hle_locinfo_dtor(void*) {}

}  // namespace

// The variadic entries: asm shims over bodies that take the guest va_list.
extern "C" GUEST_ABI int hle_va_printf_body(const char* fmt, SysvVaList* ap) { return sysv_vfprintf(stdout, fmt, ap); }
extern "C" GUEST_ABI int hle_va_snprintf_body(char* b, std::uint64_t n, const char* fmt, SysvVaList* ap) {
    return sysv_vsnprintf(b, n, fmt, ap);
}
extern "C" GUEST_ABI int hle_va_sprintf_body(char* b, const char* fmt, SysvVaList* ap) {
    return sysv_vsnprintf(b, ~std::size_t(0), fmt, ap);
}
extern "C" GUEST_ABI int hle_va_fprintf_body(FILE* f, const char* fmt, SysvVaList* ap) {
    return sysv_vfprintf(f ? f : stdout, fmt, ap);
}
extern "C" GUEST_ABI int hle_va_fscanf_body(FILE* f, const char* fmt, SysvVaList* ap) { return sysv_vfscanf(f, fmt, ap); }
extern "C" GUEST_ABI int hle_va_sscanf_body(const char* s, const char* fmt, SysvVaList* ap) {
    return sysv_vsscanf(s, fmt, ap);
}
extern "C" GUEST_ABI int hle_va_swscanf_body(const Gwchar* s, const Gwchar* fmt, SysvVaList* ap) {
    return hle_swscanf_body(s, fmt, ap);
}
SYSV_VA_ENTRY(hle_printf, hle_va_printf_body, 1, rsi)
SYSV_VA_ENTRY(hle_snprintf, hle_va_snprintf_body, 3, rcx)
SYSV_VA_ENTRY(hle_sprintf, hle_va_sprintf_body, 2, rdx)
SYSV_VA_ENTRY(hle_sprintf_s, hle_va_snprintf_body, 3, rcx)
SYSV_VA_ENTRY(hle_fprintf, hle_va_fprintf_body, 2, rdx)
SYSV_VA_ENTRY(hle_fscanf, hle_va_fscanf_body, 2, rdx)
SYSV_VA_ENTRY(hle_sscanf, hle_va_sscanf_body, 2, rdx)
SYSV_VA_ENTRY(hle_swscanf, hle_va_swscanf_body, 2, rdx)

void hle_register_libc() {
#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
// Called straight from the guest, without the FS- and stack-switching thunk:
// only functions that touch no thread-local state - no errno, no TLS - and
// need little stack. The game's main loop and job workers call these hundreds
// of thousands of times a second, and the thunk cost more than the work.
// BBHOST_RAW_LIBC=0: the raw (unthunked) entries below go through the thunk
// like everything else, to tell a raw-entry problem from a libc one.
    static const bool raw_libc = [] {
        const char* e = std::getenv("BBHOST_RAW_LIBC");
        return !e || e[0] != '0';
    }();
#define REG_RAW(name, fn) \
    (raw_libc ? register_hle_fn_raw(name, reinterpret_cast<void*>(fn)) : register_hle_fn(name, reinterpret_cast<void*>(fn)))
// Raw only in GS mode (core/tls_rewrite.h; Windows always): functions that
// may set errno - libm's on a domain or range error - need the host's FS,
// which GS mode leaves in place while the guest runs.
#define REG_RAW_GS(name, fn) \
    (raw_libc && tls_gs_mode() ? register_hle_fn_raw(name, reinterpret_cast<void*>(fn)) \
                               : register_hle_fn(name, reinterpret_cast<void*>(fn)))
    std::call_once(g_ctype_once, init_ctype);  // the ctype getters below run raw
    REG("_init_env", hle_init_env);
    REG("catchReturnFromMain", hle_catch_return_from_main);
    REG("__stack_chk_fail", hle_stack_chk_fail);
    REG("atexit", hle_atexit);
    REG("__cxa_atexit", hle_cxa_atexit);
    REG("malloc", hle_malloc);
    REG("free", hle_free);
    REG("calloc", hle_calloc);
    REG("realloc", hle_realloc);
    REG("memalign", hle_memalign);
    REG("_Znwm", hle_malloc);
    REG("_Znam", hle_malloc);
    REG("_ZnwmRKSt9nothrow_t", hle_new_nothrow);
    REG("_ZdlPv", hle_free);
    REG("_ZdaPv", hle_free);
    REG_RAW("memcpy", hle_memcpy);
    REG_RAW("memmove", hle_memmove);
    REG_RAW("memset", hle_memset);
    REG_RAW("memcmp", hle_memcmp);
    REG_RAW("memchr", hle_memchr);
    REG_RAW("strlen", hle_strlen);
    REG_RAW("strcpy", hle_strcpy);
    REG_RAW("strncpy", hle_strncpy);
    REG_RAW("strcmp", hle_strcmp);
    REG_RAW("strncmp", hle_strncmp);
    REG_RAW("strcat", hle_strcat);
    REG_RAW("strncat", hle_strncat);
    REG_RAW("strchr", hle_strchr);
    REG_RAW("strstr", hle_strstr);
    REG_RAW("strrchr", hle_strrchr);
    REG("strtok", hle_strtok);
    REG_RAW("strcspn", hle_strcspn);
    REG_RAW("strspn", hle_strspn);
    REG_RAW("strpbrk", hle_strpbrk);
    REG("strcasecmp", hle_strcasecmp);
    REG("strncasecmp", hle_strncasecmp);
    REG("strcoll", hle_strcoll);
    REG("strerror", hle_strerror);
    REG("strtol", hle_strtol);
    REG("strtoumax", hle_strtoumax);
    REG("printf", hle_printf);
    REG("vprintf", hle_vprintf);
    REG("vsnprintf", hle_vsnprintf);
    REG("snprintf", hle_snprintf);
    REG("sprintf", hle_sprintf);
    REG("sprintf_s", hle_sprintf_s);
    REG("vsprintf", hle_vsprintf);
    REG("fprintf", hle_fprintf);
    REG("vfprintf", hle_vfprintf);
    REG("puts", hle_puts);
    REG("putchar", hle_putchar);
    REG("perror", hle_perror);
    REG("fopen", hle_fopen);
    REG("freopen", hle_freopen);
    REG("fclose", hle_fclose);
    REG("fflush", hle_fflush);
    REG("fread", hle_fread);
    REG("fwrite", hle_fwrite);
    REG("fseek", hle_fseek);
    REG("ftell", hle_ftell);
    REG("rewind", hle_rewind);
    REG("feof", hle_feof);
    REG("ferror", hle_ferror);
    REG("clearerr", hle_clearerr);
    REG("fgetc", hle_fgetc);
    REG("getc", hle_getc);
    REG("ungetc", hle_ungetc);
    REG("fgets", hle_fgets);
    REG("fputc", hle_fputc);
    REG("fputs", hle_fputs);
    REG("fgetpos", hle_fgetpos);
    REG("setvbuf", hle_setvbuf);
    REG("fscanf", hle_fscanf);
    REG("sscanf", hle_sscanf);
    REG("usleep", hle_usleep);
    REG("sceKernelUsleep", hle_sce_usleep);
    REG("sleep", hle_sleep);
#define REG_CLOCK(name, fn) \
    (hle_raw_clock() ? register_hle_fn_raw(name, reinterpret_cast<void*>(fn)) : register_hle_fn(name, reinterpret_cast<void*>(fn)))
    // The clock entries' statics set up here, on a host thread, rather than
    // in a raw entry's first call (the CRT's clock() start time with them).
    (void)host_clock_realtime_ns();
    (void)std::clock();
    REG_CLOCK("gettimeofday", hle_gettimeofday);
    REG_CLOCK("time", hle_time);
    REG_CLOCK("clock", hle_clock);
#undef REG_CLOCK
    REG("difftime", hle_difftime);
    REG("gmtime", hle_gmtime);
    REG("localtime", hle_localtime);
    REG("localtime_s", hle_localtime_s);
    REG("mktime", hle_mktime);
    REG("strftime", hle_strftime);
    REG("exit", hle_exit);
    REG("abort", hle_abort);
    REG("_Assert", hle_assert);
    REG("__error", hle_error);
    REG("getpagesize", hle_getpagesize);
    REG("__cxa_guard_acquire", hle_cxa_guard_acquire);
    REG("__cxa_guard_release", hle_cxa_guard_release);
    REG("__cxa_guard_abort", hle_cxa_guard_abort);
    REG("__cxa_allocate_exception", hle_cxa_allocate_exception);
    REG("__cxa_throw", hle_cxa_throw);
    REG("__cxa_rethrow", hle_cxa_rethrow);
    REG("__cxa_begin_catch", hle_cxa_begin_catch);
    REG("__cxa_end_catch", hle_cxa_end_catch);
    REG("__cxa_pure_virtual", hle_cxa_pure_virtual);
    REG("__cxa_call_unexpected", hle_cxa_call_unexpected);
    REG("_Unwind_Resume", hle_unwind_resume);
    REG("_ZSt9terminatev", hle_terminate);
    REG("_ZSt11_Xbad_allocv", hle_xbad_alloc);
    REG("_ZSt14_Xlength_errorPKc", hle_xlength);
    REG("_ZSt14_Xout_of_rangePKc", hle_xrange);
    REG("_ZSt19_Xbad_function_callv", hle_xbad_function);
    REG("_ZSt13_Syserror_mapi", hle_syserror_map);
    REG("_ZSt14_Throw_C_errori", hle_throw_c_error);
    // Raw: must capture / restore the guest frame, so no FS thunk.
    register_hle_fn_raw("setjmp", reinterpret_cast<void*>(&hle_setjmp_raw));
    register_hle_fn_raw("longjmp", reinterpret_cast<void*>(&hle_longjmp_raw));
    REG("qsort", hle_qsort);
    REG("srand", hle_srand);
    REG("rand", hle_rand);
    REG("mkdir", hle_mkdir);
    REG("rmdir", hle_rmdir);
    REG("remove", hle_remove);
    REG("rename", hle_rename);
    REG("stat", hle_stat);
    REG("setlocale", hle_setlocale);
    REG("localeconv", hle_localeconv);
    REG_RAW("_Getpctype", hle_getpctype);
    REG_RAW("_Getptolower", hle_getptolower);
    REG_RAW("_Getptoupper", hle_getptoupper);
    REG("_Stod", hle_stod);
    REG("_Stoll", hle_stoll);
    REG("_Stoul", hle_stoul);
    REG("_Stoull", hle_stoull);
    REG("_WStod", hle_wstod);
    REG("_WStof", hle_wstof);
    REG("_Dtest", hle_dtest);
    REG_RAW_GS("sin", hle_sin);
    REG_RAW("_Sin", hle_sin_q);
    REG_RAW("_FSin", hle_sinf_q);
    REG_RAW_GS("tan", hle_tan);
    REG_RAW_GS("tanf", hle_tanf);
    REG_RAW_GS("tanhf", hle_tanhf);
    REG_RAW_GS("acos", hle_acos);
    REG_RAW_GS("acosf", hle_acosf);
    REG_RAW_GS("asin", hle_asin);
    REG_RAW_GS("asinf", hle_asinf);
    REG_RAW_GS("atan", hle_atan);
    REG_RAW_GS("atanf", hle_atanf);
    REG_RAW_GS("atan2", hle_atan2);
    REG_RAW_GS("atan2f", hle_atan2f);
    REG_RAW_GS("exp", hle_exp);
    REG_RAW_GS("expf", hle_expf);
    REG_RAW_GS("exp2", hle_exp2);
    REG_RAW_GS("exp2f", hle_exp2f);
    REG_RAW_GS("pow", hle_pow);
    REG_RAW_GS("powf", hle_powf);
    REG_RAW_GS("sqrt", hle_sqrt);
    REG_RAW_GS("sqrtf", hle_sqrtf);
    REG_RAW("ceil", hle_ceil);
    REG_RAW("ceilf", hle_ceilf);
    REG_RAW("floor", hle_floor);
    REG_RAW("floorf", hle_floorf);
    REG_RAW("fabs", hle_fabs);
    REG_RAW("fabsf", hle_fabsf);
    REG_RAW_GS("fmod", hle_fmod);
    REG_RAW_GS("fmodf", hle_fmodf);
    REG_RAW_GS("frexp", hle_frexp);
    REG_RAW_GS("frexpf", hle_frexpf);
    REG_RAW_GS("ldexp", hle_ldexp);
    REG_RAW_GS("ldexpf", hle_ldexpf);
    REG_RAW("modf", hle_modf);
    REG_RAW("modff", hle_modff);
    REG_RAW_GS("_Log", hle_log);
    REG_RAW_GS("_FLog", hle_logf);
    REG_RAW_GS("_FCosh", hle_coshf);
    REG_RAW_GS("_FSinh", hle_sinhf);
    REG_RAW("ldiv", hle_ldiv);
    REG_RAW("wcslen", hle_wcslen);
    REG_RAW("wcscpy", hle_wcscpy);
    REG_RAW("wcsncpy", hle_wcsncpy);
    REG_RAW("wcscmp", hle_wcscmp);
    REG_RAW("wcsncmp", hle_wcsncmp);
    REG_RAW("wcschr", hle_wcschr);
    REG_RAW("wcsrchr", hle_wcsrchr);
    REG_RAW("wcsstr", hle_wcsstr);
    REG_RAW("wcspbrk", hle_wcspbrk);
    REG_RAW("wcsspn", hle_wcsspn);
    REG("wcstol", hle_wcstol);
    REG("wcstoll", hle_wcstoll);
    REG_RAW("wmemcpy", hle_wmemcpy);
    REG_RAW("wmemmove", hle_wmemmove);
    REG_RAW("wmemset", hle_wmemset);
    REG_RAW("wmemcmp", hle_wmemcmp);
    REG_RAW("wmemchr", hle_wmemchr);
    REG("swprintf", hle_swprintf);
    REG("swscanf", hle_swscanf);
    REG("swprintf_s", hle_swprintf);
    REG("wprintf", hle_wprintf);
    REG("_ZNSt8ios_base4InitC1Ev", hle_ios_init);
    REG("_ZNSt8ios_base4InitD1Ev", hle_ios_dtor);
    REG("_ZNSt6_WinitC1Ev", hle_winit);
    REG("_ZNSt6_WinitD1Ev", hle_winit_dtor);
    REG("_ZNSt8_LocinfoC1EPKc", hle_locinfo_ctor);
    REG("_ZNSt8_LocinfoD1Ev", hle_locinfo_dtor);
    REG("_ZNSt8ios_base7failureD1Ev", hle_ios_dtor);
    REG("_ZNKSt9exception6_RaiseEv", hle_terminate);
    REG("_Stdout", &g_stdout_slot);
    REG("_Stdin", &g_stdin_slot);
    REG("_Stderr", &g_stderr_slot);
    REG("Need_sceLibc", &g_need_scelibc);
    REG("_ZSt7nothrow", g_dummy_object);
    REG("_ZSt3cin", g_dummy_object);
    REG("_ZSt4cout", g_dummy_object);
    REG("_ZSt4cerr", g_dummy_object);
    REG("_ZSt4wcin", g_dummy_object);
    REG("_ZSt5wcout", g_dummy_object);
    REG("_ZSt5wcerr", g_dummy_object);
    REG("_ZSt7_BADOFF", g_dummy_object);
    REG("_ZTIi", g_dummy_object);
    REG("_ZTVN10__cxxabiv117__class_type_infoE", g_dummy_object);
    REG("_ZTVN10__cxxabiv120__si_class_type_infoE", g_dummy_object);
    REG("_ZTVN10__cxxabiv121__vmi_class_type_infoE", g_dummy_object);
    REG("_ZTVNSt8ios_base7failureE", g_dummy_object);
    REG("_ZTVSt12system_error", g_dummy_object);
    REG("_ZTVSt13runtime_error", g_dummy_object);
    REG("__stack_chk_guard", &g_stack_chk_guard);
    REG("__gxx_personality_v0", hle_ok);
#undef REG
}
