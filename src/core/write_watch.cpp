#include "core/write_watch.h"

#include "log.h"

#include <algorithm>
#include <atomic>
#include <vector>
#include <cerrno>
#include <cstdlib>
#include <mutex>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include "core/mac_vm.h"
#endif

namespace {

constexpr unsigned kPageShift = 12;   // host pages: what mprotect works in
constexpr unsigned kChunkShift = 16;  // 64 Ki pages, 256 MiB of address space, per chunk
constexpr std::uint64_t kChunkPages = 1ull << kChunkShift;
constexpr std::uint64_t kChunks = 1ull << (40 - kPageShift - kChunkShift);  // guest addresses are 40-bit

using Entry = std::atomic<std::uint32_t>;
// Chunks are allocated by arming, under g_alloc_mu, and never freed, so the
// fault handler can read the pointers without a lock.
std::atomic<Entry*> g_chunks[kChunks];
// A second level: the latest sequence stamped on any page of each group of
// 256 pages (1 MiB). A check skips a group whose latest is not past its
// watch's arming - a 4 MiB texture is four reads instead of 1024 whenever
// something anywhere was written.
constexpr unsigned kGroupShift = 8;
std::atomic<Entry*> g_groups[kChunks];
std::mutex g_alloc_mu;

// Sequence numbers: 1 marks a page armed and never written, and every write
// or mapping change takes the next one.
std::atomic<std::uint32_t> g_seq{1};
// Bumped after every stamp: a watch whose last check saw the same count has
// nothing to scan.
std::atomic<std::uint64_t> g_changes{0};
std::atomic<std::uint64_t> g_arms{0}, g_faults{0}, g_releases{0}, g_forgets{0}, g_refused{0}, g_ahead_pages{0};

const bool g_enabled = [] {
    const char* e = std::getenv("BBHOST_WRITE_WATCH");
    return !e || e[0] != '0';
}();
// BBHOST_WATCH_AHEAD=0: a write fault releases its own page only.
const bool g_ahead_on = [] {
    const char* e = std::getenv("BBHOST_WATCH_AHEAD");
    return !e || e[0] != '0';
}();
constexpr std::uint64_t kAheadPages = 16;  // 64 KiB: what a fault may release past its own page
constexpr std::uint32_t kRecentSeqs = 256;  // "a moment ago", in sequence numbers (one a fault or release)

// Each protected range splits a mapping, and the kernel caps mappings per
// process (vm.max_map_count, 65530 on many systems). Arming stops well short of
// the cap, so the single-page unprotects faults make always have room.
std::atomic<bool> g_arming_off{false};
std::uint64_t g_ops_since_count = 0;  // under the kernel's map lock, like arming

// Windows itself has no cap on a process's mappings, but wine's
// VirtualProtect is mprotect underneath and the Linux cap applies to the wine
// process; wine's Z: drive is the host root, so the same two files are read
// there. On real Windows the open fails and no check runs.
#if defined(_WIN32)
#define BB_PROC_PREFIX "Z:"
#else
#define BB_PROC_PREFIX ""
#endif

long read_small_file_number(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return -1;
    char buf[32] = {};
    const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    return n > 0 ? std::strtol(buf, nullptr, 10) : -1;
}

long count_mappings() {
    std::FILE* f = std::fopen(BB_PROC_PREFIX "/proc/self/maps", "rb");
    if (!f) return -1;
    long lines = 0;
    static char buf[1 << 16];  // one caller at a time: check_mapping_headroom's flag
    for (std::size_t n; (n = std::fread(buf, 1, sizeof(buf), f)) > 0;) {
        for (std::size_t i = 0; i < n; ++i) lines += buf[i] == '\n';
    }
    std::fclose(f);
    return lines;
}

// Counting reads all of /proc/self/maps - tens of thousands of lines once
// protection has split the mappings - and ran on the arming thread, the
// command processor: ~5% of the frame a fight's new buffers arrive in. It
// runs on a thread of its own; arming reads only its verdict.
void check_mapping_headroom() {
    static const long max_maps = read_small_file_number(BB_PROC_PREFIX "/proc/sys/vm/max_map_count");
    if (++g_ops_since_count < 1024 || max_maps <= 0) return;
    g_ops_since_count = 0;
    static std::atomic<bool> counting{false};  // count_mappings' buffer is one caller's
    if (counting.exchange(true)) return;
    std::thread([] {
        const long n = count_mappings();
        if (n >= 0 && n > max_maps - 16384) {
            g_arming_off.store(true);
            host_log("write-watch: %ld mappings of the kernel's %ld; no more surfaces are watched (the hashes carry on)", n,
                     max_maps);
        }
        counting.store(false);
    }).detach();
}

#if defined(_WIN32)
// The page protections, through VirtualProtect: a view of the direct-memory
// section takes any protection within the section's access, and the fault
// handler may call it (no errno, no lock).
// Returns 0, or ENOMEM when the system is out of mappings or resources, else
// another errno value.
__attribute__((no_stack_protector, always_inline)) inline int protect_pages(std::uint64_t addr, std::uint64_t len,
                                                                            bool writable) {
    DWORD old = 0;
    if (VirtualProtect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(addr)), static_cast<SIZE_T>(len),
                       writable ? PAGE_READWRITE : PAGE_READONLY, &old)) {
        return 0;
    }
    const DWORD e = GetLastError();
    return (e == ERROR_NOT_ENOUGH_MEMORY || e == ERROR_NO_SYSTEM_RESOURCES || e == ERROR_COMMITMENT_LIMIT) ? ENOMEM
                                                                                                          : EACCES;
}
#elif defined(__APPLE__)
// libc's mprotect: the raw call below is Linux's system call convention.
// macOS's libc reaches errno through GS, which stays the system's on a guest
// thread too - the guest's TCB is one pthread key's value there
// (core/tls_rewrite.h) - so the wrapper is safe wherever this runs.
// Returns 0 or the errno value.
__attribute__((no_stack_protector, always_inline)) inline int protect_pages(std::uint64_t addr, std::uint64_t len,
                                                                            bool writable) {
    if (mprotect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(addr)), static_cast<std::size_t>(len),
                 writable ? (PROT_READ | PROT_WRITE) : PROT_READ) == 0) {
        return 0;
    }
    return errno;
}
#else
// mprotect without libc: its wrapper sets errno on failure, and errno is
// thread-local - through FS, which on a guest thread is the guest's.
__attribute__((no_stack_protector, always_inline)) inline long raw_mprotect(std::uint64_t addr, std::uint64_t len,
                                                                             long prot) {
    long ret;
    asm volatile("syscall"
                 : "=a"(ret)
                 : "0"(static_cast<long>(SYS_mprotect)), "D"(addr), "S"(len), "d"(prot)
                 : "rcx", "r11", "memory");
    return ret;
}
// Returns 0 or the errno value (the raw syscall's negated return).
__attribute__((no_stack_protector, always_inline)) inline int protect_pages(std::uint64_t addr, std::uint64_t len,
                                                                            bool writable) {
    const long r = raw_mprotect(addr, len, writable ? (PROT_READ | PROT_WRITE) : PROT_READ);
    return r == 0 ? 0 : static_cast<int>(-r);
}
#endif

__attribute__((no_stack_protector, always_inline)) inline Entry* entry_of(std::uint64_t page) {
    const std::uint64_t c = page >> kChunkShift;
    if (c >= kChunks) return nullptr;
    Entry* chunk = g_chunks[c].load(std::memory_order_acquire);
    return chunk ? &chunk[page & (kChunkPages - 1)] : nullptr;
}

// Stores `s` unless the entry already holds a later one: two stamps racing
// must not leave the earlier.
__attribute__((no_stack_protector, always_inline)) inline void stamp(Entry& e, std::uint32_t s) {
    std::uint32_t cur = e.load(std::memory_order_relaxed);
    while (cur < s && !e.compare_exchange_weak(cur, s, std::memory_order_release, std::memory_order_relaxed)) {
    }
}

// Stamps page `page` (whose entry is `e`) and its group.
__attribute__((no_stack_protector, always_inline)) inline void stamp_page(std::uint64_t page, Entry& e, std::uint32_t s) {
    stamp(e, s);
    Entry* groups = g_groups[page >> kChunkShift].load(std::memory_order_acquire);
    if (groups) stamp(groups[(page & (kChunkPages - 1)) >> kGroupShift], s);
}

__attribute__((no_stack_protector, always_inline)) inline std::uint32_t next_seq() {
    return g_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
}

}  // namespace

bool write_watch_enabled() { return g_enabled && !g_arming_off.load(std::memory_order_relaxed); }

bool write_watch_arm_locked(std::uint64_t va, std::size_t len, WriteWatch& w) {
    w.armed = 0;
    if (!write_watch_enabled() || !len) return false;
    const std::uint64_t lo = va >> kPageShift, hi = (va + len - 1) >> kPageShift;  // inclusive
    if ((hi >> kChunkShift) >= kChunks) {
        g_refused.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(g_alloc_mu);
        for (std::uint64_t c = lo >> kChunkShift; c <= hi >> kChunkShift; ++c) {
            if (!g_chunks[c].load(std::memory_order_relaxed)) {
                g_groups[c].store(new Entry[kChunkPages >> kGroupShift](), std::memory_order_release);
                g_chunks[c].store(new Entry[kChunkPages](), std::memory_order_release);
            }
        }
    }
    // The order that makes the watch sound (see the header): count, then
    // sequence, then protection; the caller reads the memory after.
    const std::uint64_t seen = g_changes.load(std::memory_order_acquire);
    const std::uint32_t armed = g_seq.load(std::memory_order_acquire);
    for (std::uint64_t p = lo; p <= hi; ++p) {
        std::uint32_t zero = 0;
        entry_of(p)->compare_exchange_strong(zero, 1, std::memory_order_acq_rel);
    }
    if (const int err = protect_pages(lo << kPageShift, (hi - lo + 1) << kPageShift, false)) {
        g_refused.fetch_add(1, std::memory_order_relaxed);
        if (err == ENOMEM) {
            g_arming_off.store(true);
            host_log("write-watch: protecting 0x%llx failed (out of mappings); no more surfaces are watched",
                     static_cast<unsigned long long>(va));
        }
        return false;
    }
    if (g_arms.fetch_add(1, std::memory_order_relaxed) == 0) {
        host_log("write-watch: on - texture memory is write-protected, and a write to it re-uploads the texture "
                 "on its next use");
    }
    check_mapping_headroom();
    w.armed = armed;
    w.seen = seen;
    return true;
}

std::uint32_t write_watch_dirty(std::uint64_t va, std::size_t len, WriteWatch& w) {
    if (!w.armed || !len) return 0;
    const std::uint64_t c = g_changes.load(std::memory_order_acquire);
    if (c == w.seen) return 0;
    std::uint32_t written = 0;
    const std::uint64_t lo = va >> kPageShift, hi = (va + len - 1) >> kPageShift;
    for (std::uint64_t g0 = lo >> kGroupShift; g0 <= hi >> kGroupShift; ++g0) {
        const std::uint64_t first = std::max(lo, g0 << kGroupShift), last = std::min(hi, ((g0 + 1) << kGroupShift) - 1);
        const std::uint64_t chunk = first >> kChunkShift;
        const Entry* groups = chunk < kChunks ? g_groups[chunk].load(std::memory_order_acquire) : nullptr;
        if (groups && groups[g0 & ((kChunkPages >> kGroupShift) - 1)].load(std::memory_order_acquire) <= w.armed) continue;
        for (std::uint64_t p = first; p <= last; ++p) {
            const Entry* e = entry_of(p);
            if (!e || e->load(std::memory_order_acquire) > w.armed) ++written;
        }
    }
    // Only a clean answer moves the fast path on: a written watch stays
    // written, however often it is asked, until it is armed again.
    if (!written) w.seen = c;
    return written;
}

// The last faults' address and frame, for the census of who writes a surface.
constexpr unsigned kFaultRing = 16384;
std::atomic<std::uint64_t> g_fault_addr[kFaultRing];
std::atomic<std::uint64_t> g_fault_frame[kFaultRing][kWriteWatchFrame];
std::atomic<std::uint64_t> g_fault_when[kFaultRing];  // the fault's sequence number
std::atomic<std::uint32_t> g_fault_next{0};

__attribute__((no_stack_protector)) bool write_watch_on_fault(std::uint64_t addr, bool is_write, const std::uint64_t* frame) {
    if (!is_write || !g_enabled) return false;
    const std::uint64_t page = addr >> kPageShift;
    Entry* e = entry_of(page);
    if (!e || e->load(std::memory_order_acquire) == 0) return false;
    // A writer streaming through protected pages - GX copying a new
    // resource's data into memory dead resources' watches still cover, the
    // engine's parallel copies of a streamed file - faulted once per 4 KiB
    // page: 359,397 faults in a Central Yharnam tour, and the main loop's
    // waits for those copies halved with the watch off. When the pages just
    // below this one were written a moment ago, the armed pages above it go
    // with it, as many as the run below has (at most kAheadPages): a 1 MiB
    // copy faults ~20 times instead of 256. A page released ahead that the
    // writer then does not reach reads as written - an extra upload of
    // whatever watches it, never a missed write.
    std::uint64_t ahead = 0;
    if (g_ahead_on) {
        const std::uint32_t now = g_seq.load(std::memory_order_acquire);
        std::uint64_t run = 0;
        while (run < kAheadPages && page > run) {
            const Entry* b = entry_of(page - 1 - run);
            const std::uint32_t v = b ? b->load(std::memory_order_acquire) : 0;
            if (v <= 1 || now - v > kRecentSeqs) break;  // unarmed, armed and unwritten, or written a while ago
            ++run;
        }
        while (ahead < run) {
            const Entry* a = entry_of(page + 1 + ahead);
            if (!a || a->load(std::memory_order_acquire) == 0) break;
            ++ahead;
        }
    }
    // A page that was never protected, or was unmapped since, fails here or
    // faults again; neither is ours to hide, so the handler goes on as before.
    // Writable first, then the sequence number (the header's order) - for
    // the pages ahead too.
    if (protect_pages(page << kPageShift, (1 + ahead) << kPageShift, true) != 0) {
        if (!ahead || protect_pages(page << kPageShift, 1ull << kPageShift, true) != 0) return false;
        ahead = 0;  // across two mappings (or views): this page alone
    }
    const std::uint32_t s = next_seq();
    stamp_page(page, *e, s);
    for (std::uint64_t k = 1; k <= ahead; ++k) stamp_page(page + k, *entry_of(page + k), s);
    g_changes.fetch_add(1, std::memory_order_acq_rel);
    g_faults.fetch_add(1, std::memory_order_relaxed);
    if (ahead) g_ahead_pages.fetch_add(ahead, std::memory_order_relaxed);
    const std::uint32_t n = g_fault_next.fetch_add(1, std::memory_order_relaxed);
    const unsigned slot = n % kFaultRing;
    for (int k = 0; k < kWriteWatchFrame; ++k) g_fault_frame[slot][k].store(frame ? frame[k] : 0, std::memory_order_relaxed);
    g_fault_when[slot].store(n + 1, std::memory_order_relaxed);
    g_fault_addr[slot].store(addr, std::memory_order_release);
    return true;
}

int write_watch_recent_fault_span(std::uint64_t va, std::size_t len, std::uint64_t* first_seq, std::uint64_t* last_seq) {
    int n = 0;
    std::uint64_t lo = ~0ull, hi = 0;
    for (unsigned i = 0; i < kFaultRing; ++i) {
        const std::uint64_t a = g_fault_addr[i].load(std::memory_order_acquire);
        if (!a || a < va || a - va >= len) continue;
        const std::uint64_t w = g_fault_when[i].load(std::memory_order_relaxed);
        lo = std::min(lo, w);
        hi = std::max(hi, w);
        ++n;
    }
    *first_seq = n ? lo : 0;
    *last_seq = hi;
    return n;
}

std::uint64_t write_watch_fault_seq() { return g_fault_next.load(std::memory_order_relaxed); }

int write_watch_fault_writers(std::uint64_t* pcs, std::uint64_t* callers, std::uint32_t* counts, int max) {
    struct Writer {
        std::uint64_t pc, caller;
        std::uint32_t n;
    };
    std::vector<Writer> seen;
#if defined(__APPLE__)
    // Host code lies among guest addresses on macOS; the guest's is in the
    // eboot's window.
    const auto guest = [](std::uint64_t a) { return a >= kMacVmImageBase && a < kMacVmImageBase + kMacVmImageSize; };
#else
    const auto guest = [](std::uint64_t a) { return a > 0x10000 && a < 0x100000000000ull; };
#endif
    for (unsigned i = 0; i < kFaultRing; ++i) {
        if (!g_fault_addr[i].load(std::memory_order_acquire)) continue;
        std::uint64_t pc = 0, caller = 0;
        for (int k = 0; k < kWriteWatchFrame; ++k) {
            const std::uint64_t a = g_fault_frame[i][k].load(std::memory_order_relaxed);
            if (!guest(a)) continue;
            if (!pc) {
                pc = a;
            } else if (a != pc) {
                caller = a;
                break;
            }
        }
        bool found = false;
        for (Writer& w : seen) {
            if (w.pc == pc && w.caller == caller) {
                ++w.n;
                found = true;
                break;
            }
        }
        if (!found && seen.size() < 4096) seen.push_back({pc, caller, 1});
    }
    std::sort(seen.begin(), seen.end(), [](const Writer& a, const Writer& b) { return a.n > b.n; });
    int n = 0;
    for (const Writer& w : seen) {
        if (n >= max) break;
        pcs[n] = w.pc;
        callers[n] = w.caller;
        counts[n] = w.n;
        ++n;
    }
    return n;
}

int write_watch_recent_faults(std::uint64_t va, std::size_t len, std::uint64_t* pcs, int max) {
    int n = 0;
    for (unsigned i = 0; i < kFaultRing && n < max; ++i) {
        const std::uint64_t a = g_fault_addr[i].load(std::memory_order_acquire);
        if (!a || a < va || a - va >= len) continue;
        for (int k = 0; k < kWriteWatchFrame && n < max; ++k) {
            const std::uint64_t pc = g_fault_frame[i][k].load(std::memory_order_relaxed);
            if (!pc) continue;
            bool dup = false;
            for (int j = 0; j < n; ++j) dup |= pcs[j] == pc;
            if (!dup) pcs[n++] = pc;
        }
    }
    return n;
}

#if defined(_WIN32)
namespace {
// bbhost.exe's own addresses, which are not the game's: the census takes the
// first guest address of a frame for the writer, and on Linux the host's
// addresses lie above the guest's range anyway.
std::uint64_t g_host_lo = 0, g_host_hi = 0;

std::uint64_t not_host(std::uint64_t a) { return a >= g_host_lo && a < g_host_hi ? 0 : a; }

// The vectored handler, first in line: a write to a watched page is made
// writable, recorded, and the instruction retried; anything else goes on to
// the next handler (win_watch's, then the crash reporter). The writer goes
// to the census as the Linux handler reads it: the pc, the top of the stack
// (return addresses when the write is in a leaf) and [rbp+8], from the
// stack's own page. The words below rsp are already Windows' exception
// frame by now; the ones above it are the thread's.
LONG CALLBACK write_watch_veh(EXCEPTION_POINTERS* ep) {
    const EXCEPTION_RECORD* r = ep->ExceptionRecord;
    if (r->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || r->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    const bool is_write = r->ExceptionInformation[0] == 1;
    std::uint64_t frame[kWriteWatchFrame] = {};
    if (is_write) {
        const CONTEXT* c = ep->ContextRecord;
        frame[0] = not_host(c->Rip);
        const std::uint64_t rsp = c->Rsp, rbp = c->Rbp;
        const auto page_ok = [](std::uint64_t a) { return a > 0x10000 && (a & 0xfff) <= 0xff8; };
        for (int k = 0; k < 6; ++k) {
            const std::uint64_t at = rsp + 8ull * static_cast<std::uint64_t>(k);
            if (!page_ok(at) || (at >> 12) != (rsp >> 12)) break;
            frame[1 + k] = not_host(*reinterpret_cast<const std::uint64_t*>(at));
        }
        if (page_ok(rbp + 8) && ((rbp + 8) >> 12) == (rsp >> 12)) frame[7] = not_host(*reinterpret_cast<const std::uint64_t*>(rbp + 8));
    }
    if (write_watch_on_fault(static_cast<std::uint64_t>(r->ExceptionInformation[1]), is_write, frame)) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
}  // namespace

bool write_watch_install() {
    if (!g_enabled) return false;
    if (const auto* base = reinterpret_cast<const std::uint8_t*>(GetModuleHandleW(nullptr))) {
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
        g_host_lo = reinterpret_cast<std::uintptr_t>(base);
        g_host_hi = g_host_lo + nt->OptionalHeader.SizeOfImage;
    }
    static PVOID h = AddVectoredExceptionHandler(1, write_watch_veh);
    return h != nullptr;
}
#endif

// No lock, no thread-local, no stack protector, no lambda (whose body the
// compiler may outline with the protector in it): a decomp leaf calls this on
// the game's thread as the game left it (decomp/memory/game_memcpy.cpp).
__attribute__((no_stack_protector)) bool write_watch_release(void* p, std::size_t n) {
    if (!g_enabled || !p || !n) return false;
    const std::uint64_t va = reinterpret_cast<std::uintptr_t>(p);
    const std::uint64_t lo = va >> kPageShift, hi = (va + n - 1) >> kPageShift;
    // Writable first, then the sequence number and the stamps - the fault
    // path's order (see the header). Stamped first, an arm that loaded its
    // sequence after the stamp and protected before the unprotect was left
    // with a clean watch over a writable page, and every later write to it
    // went unseen.
    bool any = false;
    std::uint64_t run = 0, run_pages = 0;
    for (std::uint64_t pg = lo; pg <= hi; ++pg) {
        const Entry* e = entry_of(pg);
        if (e && e->load(std::memory_order_acquire) != 0) {
            if (!run_pages) run = pg;
            ++run_pages;
            any = true;
            continue;
        }
        if (run_pages) protect_pages(run << kPageShift, run_pages << kPageShift, true);
        run_pages = 0;
        if (!e) pg = ((pg >> kChunkShift) + 1) * kChunkPages - 1;  // a whole chunk nobody armed: to its end
    }
    if (run_pages) protect_pages(run << kPageShift, run_pages << kPageShift, true);
    if (!any) return false;
    const std::uint32_t s = next_seq();
    for (std::uint64_t pg = lo; pg <= hi; ++pg) {
        Entry* e = entry_of(pg);
        if (!e) {
            pg = ((pg >> kChunkShift) + 1) * kChunkPages - 1;
            continue;
        }
        if (e->load(std::memory_order_acquire) != 0) stamp_page(pg, *e, s);
    }
    g_changes.fetch_add(1, std::memory_order_acq_rel);
    g_releases.fetch_add(1, std::memory_order_relaxed);
    return true;
}

std::size_t write_watch_fread(void* p, std::size_t size, std::size_t n, std::FILE* f) {
    write_watch_release(p, size * n);
    return std::fread(p, size, n, f);
}

void write_watch_forget(std::uint64_t va, std::size_t len) {
    if (!g_enabled || !len) return;
    const std::uint64_t lo = va >> kPageShift;
    const std::uint64_t hi = std::min<std::uint64_t>((va + len - 1) >> kPageShift, kChunks * kChunkPages - 1);
    std::uint32_t s = 0;
    for (std::uint64_t pg = lo; pg <= hi; ++pg) {
        Entry* e = entry_of(pg);
        if (!e) {
            pg = ((pg >> kChunkShift) + 1) * kChunkPages - 1;
            continue;
        }
        if (e->load(std::memory_order_acquire) == 0) continue;
        if (!s) s = next_seq();
        stamp_page(pg, *e, s);
    }
    if (!s) return;
    g_changes.fetch_add(1, std::memory_order_acq_rel);
    g_forgets.fetch_add(1, std::memory_order_relaxed);
}

void write_watch_describe(std::uint64_t va, std::size_t len, const WriteWatch& w, char* out, std::size_t cap) {
    int n = std::snprintf(out, cap, "armed %u seen %llu (changes %llu, seq %u); entries", w.armed,
                          static_cast<unsigned long long>(w.seen), static_cast<unsigned long long>(g_changes.load()), g_seq.load());
    const std::uint64_t lo = va >> kPageShift, hi = (va + len - 1) >> kPageShift;
    for (std::uint64_t p = lo; p <= hi && n > 0 && static_cast<std::size_t>(n) < cap; ++p) {
        const Entry* e = entry_of(p);
        n += std::snprintf(out + n, cap - static_cast<std::size_t>(n), " %u", e ? e->load() : 0u);
    }
}

WriteWatchStats write_watch_stats() {
    WriteWatchStats s;
    s.arms = g_arms.load();
    s.faults = g_faults.load();
    s.releases = g_releases.load();
    s.forgets = g_forgets.load();
    s.refused = g_refused.load();
    s.ahead_pages = g_ahead_pages.load();
    return s;
}
