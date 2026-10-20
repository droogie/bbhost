#include "host/sampler.h"

#include "core/portable.h"
#include "core/thunk.h"
#include "hle/modules.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <linux/perf_event.h>
#include <dirent.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <unwind.h>
#include <fcntl.h>
#include <setjmp.h>
#include <signal.h>
#include <asm/prctl.h>
#include <sys/auxv.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(__linux__)
namespace {

constexpr int kDepth = 40;
constexpr std::size_t kSlots = 1 << 16;  // a ring; the writer keeps well ahead of wrapping

struct Slot {
    std::atomic<std::uint32_t> ready{0};
    std::uint64_t seq = 0;
    std::uint32_t tid = 0;
    std::uint64_t flip = 0;
    int n = 0;
    int guest = 0;  // host frames before this index, the guest's after
    void* pc[kDepth];
};

Slot* g_slots = nullptr;
std::atomic<std::uint64_t> g_next{0};
std::atomic<std::uint64_t> g_dropped{0};
std::uint64_t g_guest_lo = 0, g_guest_hi = 0;
const bool g_fsgsbase = (getauxval(AT_HWCAP2) & (1ul << 1)) != 0;  // HWCAP2_FSGSBASE

void* read_fs() {
    if (g_fsgsbase) {
        void* v;
        asm volatile("rdfsbase %0" : "=r"(v));
        return v;
    }
    unsigned long v = 0;
    syscall(SYS_arch_prctl, ARCH_GET_FS, &v);
    return reinterpret_cast<void*>(v);
}

void write_fs(void* base) {
    if (g_fsgsbase) {
        asm volatile("wrfsbase %0" : : "r"(base) : "memory");
    } else {
        syscall(SYS_arch_prctl, ARCH_SET_FS, reinterpret_cast<unsigned long>(base));
    }
}

bool in_guest(std::uint64_t pc) { return pc >= g_guest_lo && pc < g_guest_hi; }

// Walks guest frames from rbp: [rbp] the caller's rbp, [rbp+8] the return
// address, while each frame sits above the last and returns into the eboot.
void walk_guest(Slot& s, std::uint64_t rbp, std::uint64_t floor) {
    while (s.n < kDepth && rbp > floor && rbp - floor < (64u << 20) && (rbp & 7) == 0) {
        const auto* f = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(rbp));
        if (!in_guest(f[1])) break;
        s.pc[s.n++] = reinterpret_cast<void*>(f[1]);
        if (f[0] <= rbp) break;
        rbp = f[0];
    }
}

// Set while this thread's handler unwinds: a stack the unwinder misreads
// (code with wrong or no unwind info) faults inside it, and the SIGSEGV
// handler comes back here instead of taking the process down.
thread_local sigjmp_buf* t_unwinding = nullptr;
std::atomic<std::uint64_t> g_unwind_faults{0};

struct Unwind {
    Slot* s;
    int skip;                   // this handler and the signal trampoline
    std::uint64_t guest_rbp = 0;  // where the guest's frames continue, once reached
};

_Unwind_Reason_Code unwind_step(_Unwind_Context* c, void* arg) {
    auto* u = static_cast<Unwind*>(arg);
    const std::uint64_t ip = _Unwind_GetIP(c);
    if (u->skip > 0) {
        --u->skip;
        return _URC_NO_REASON;
    }
    if (!ip || u->s->n >= kDepth) return _URC_END_OF_STACK;
    u->s->pc[u->s->n++] = reinterpret_cast<void*>(ip);
    if (in_guest(ip)) {
        // Stepping out of the HLE trampoline restored rbp from where it saved
        // the guest's (its CFI says so): the guest frames go on from there.
        u->guest_rbp = _Unwind_GetGR(c, 6);  // DWARF register 6 is rbp
        return _URC_END_OF_STACK;
    }
    return _URC_NO_REASON;
}

// The sample's frames: the guest's alone when it landed in guest code, else
// the host's down to the HLE trampoline if a guest thread called in, then
// the guest's. Both walks read frames that may not be what they seem (guest
// code that uses rbp for data, code with wrong or no unwind info), so both
// run under the sigsetjmp. Kept out of line so the sigsetjmp owns this frame
// alone.
__attribute__((noinline)) void walk_stack(Slot& s, const ucontext_t* uc) {
    const std::uint64_t rip = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RIP]);
    sigjmp_buf jb;
    if (sigsetjmp(jb, 1) == 0) {
        t_unwinding = &jb;
        if (in_guest(rip)) {
            // The eboot keeps frame pointers.
            s.pc[s.n++] = reinterpret_cast<void*>(rip);
            walk_guest(s, static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RBP]),
                       static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RSP]));
            s.guest = 0;  // every frame is the guest's
        } else {
            Unwind u{&s, 3};  // this function, the handler and the signal trampoline
            _Unwind_Backtrace(unwind_step, &u);
            s.guest = s.n;  // host frames end here
            // The guest's stack is not the host's: its first frame is only
            // checked for alignment, the rest for climbing.
            if (u.guest_rbp) walk_guest(s, u.guest_rbp, u.guest_rbp - 16);
        }
    } else {
        g_unwind_faults.fetch_add(1, std::memory_order_relaxed);
        s.pc[0] = reinterpret_cast<void*>(rip);  // the leaf is still worth having
        s.n = 1;
        s.guest = in_guest(rip) ? 0 : 1;
    }
    t_unwinding = nullptr;
}

// The watchdog's dump (log_all_threads): while set, a signal's stack goes
// here, one slot per answer, instead of into the sampler's ring.
constexpr int kDumpSlots = 512;
std::atomic<Slot*> g_dump_slots{nullptr};
std::atomic<int> g_dump_next{0};

void on_sigprof(int, siginfo_t*, void* ctx) {
    // A guest thread arrives with the guest's FS: nothing below (errno, the
    // unwinder) may run on it.
    void* const fs = read_fs();
    hle_fs_host();
    const int saved = errno;
    std::uint64_t i = 0;
    Slot* sp = g_dump_slots.load(std::memory_order_acquire);
    if (sp) {
        const int k = g_dump_next.fetch_add(1, std::memory_order_relaxed);
        if (k >= kDumpSlots) {
            errno = saved;
            write_fs(fs);
            return;
        }
        sp += k;
    } else {
        if (!g_slots) {  // the watchdog's signal, answered after its dump gave up waiting
            errno = saved;
            write_fs(fs);
            return;
        }
        i = g_next.fetch_add(1, std::memory_order_relaxed);
        sp = &g_slots[i % kSlots];
    }
    Slot& s = *sp;
    if (s.ready.load(std::memory_order_acquire) != 0) {  // the writer has not caught up
        g_dropped.fetch_add(1, std::memory_order_relaxed);
        errno = saved;
        write_fs(fs);
        return;
    }
    s.seq = i;
    s.tid = static_cast<std::uint32_t>(syscall(SYS_gettid));
    s.flip = hle_video_flip_count();
    const auto* uc = static_cast<const ucontext_t*>(ctx);
    s.n = 0;
    // A sample can land in the SIGSEGV handler - a write-watch fault, hundreds
    // a second in play - which runs with SIGSEGV blocked. A fault in the walk
    // is then not delivered to the handler that would bring it back here: the
    // kernel kills the process (two sampled soaks dumped core in walk_guest
    // this way, under 0x2cdd6b9's fault). Unblocked for the walk; returning
    // from this handler restores the mask.
    if (sigismember(&uc->uc_sigmask, SIGSEGV) || sigismember(&uc->uc_sigmask, SIGBUS)) {
        sigset_t faults;
        sigemptyset(&faults);
        sigaddset(&faults, SIGSEGV);
        sigaddset(&faults, SIGBUS);
        pthread_sigmask(SIG_UNBLOCK, &faults, nullptr);
    }
    walk_stack(s, uc);
    s.ready.store(1, std::memory_order_release);
    errno = saved;
    write_fs(fs);
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::size_t a = 0;
    while (a <= s.size()) {
        const std::size_t b = s.find(',', a);
        const std::string part = s.substr(a, b == std::string::npos ? std::string::npos : b - a);
        if (!part.empty()) out.push_back(part);
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return out;
}

struct Tid {
    int tid;
    std::string name;
};

std::atomic<int> g_main_tid{0};

std::vector<Tid> matching_tids(const std::vector<std::string>& prefixes) {
    std::vector<Tid> out;
    DIR* d = opendir("/proc/self/task");
    if (!d) return out;
    while (dirent* e = readdir(d)) {
        const int tid = std::atoi(e->d_name);
        if (tid <= 0) continue;
        char path[64], name[32] = {};
        std::snprintf(path, sizeof(path), "/proc/self/task/%d/comm", tid);
        const int fd = open(path, O_RDONLY);
        if (fd < 0) continue;
        const ssize_t n = read(fd, name, sizeof(name) - 1);
        close(fd);
        if (n <= 0) continue;
        name[n] = 0;
        if (char* nl = std::strchr(name, '\n')) *nl = 0;
        for (const std::string& p : prefixes) {
            // "main": the thread the game's main loop runs on (sampler_note_main_thread).
            const bool hit = p == "main" ? tid == g_main_tid.load() : std::strncmp(name, p.c_str(), p.size()) == 0;
            if (hit) {
                out.push_back({tid, name});
                break;
            }
        }
    }
    closedir(d);
    return out;
}

void copy_file(const char* from, const std::string& to) {
    FILE* in = std::fopen(from, "r");
    FILE* out = std::fopen(to.c_str(), "w");
    if (in && out) {
        char buf[65536];
        std::size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), in)) > 0) std::fwrite(buf, 1, n, out);
    }
    if (in) std::fclose(in);
    if (out) std::fclose(out);
}

}  // namespace

void sampler_recover() {
    if (sigjmp_buf* jb = t_unwinding) {
        t_unwinding = nullptr;
        siglongjmp(*jb, 1);
    }
}

void sampler_note_main_thread() { g_main_tid.store(static_cast<int>(syscall(SYS_gettid))); }

namespace {
// The SIGPROF handler, installed once for the sampler and the watchdog, with
// the unwinder loaded here rather than inside a handler.
void install_sigprof() {
    static std::once_flag once;
    std::call_once(once, [] {
        void* warm[4];
        backtrace(warm, 4);
        struct sigaction sa{};
        sa.sa_sigaction = on_sigprof;
        sa.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGPROF, &sa, nullptr);
    });
}

struct Module {
    std::uint64_t lo, hi, base;  // an executable mapping; base: the file's ELF address 0
    std::string name;
};

std::vector<Module> read_modules() {
    std::vector<Module> out;
    std::vector<std::pair<std::string, std::uint64_t>> bases;
    FILE* f = std::fopen("/proc/self/maps", "r");
    if (!f) return out;
    char line[640];
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long long lo = 0, hi = 0, off = 0;
        char perm[8] = {}, path[512] = {};
        if (std::sscanf(line, "%llx-%llx %7s %llx %*s %*s %511[^\n]", &lo, &hi, perm, &off, path) < 5 || path[0] != '/') continue;
        // The first mapping of a file sits at its ELF address 0 (as tools/wallprof.py takes it).
        auto it = std::find_if(bases.begin(), bases.end(), [&](const auto& b) { return b.first == path; });
        const std::uint64_t base = it == bases.end() ? lo - off : it->second;
        if (it == bases.end()) bases.emplace_back(path, base);
        if (perm[2] != 'x') continue;
        const char* slash = std::strrchr(path, '/');
        out.push_back({lo, hi, base, slash ? slash + 1 : path});
    }
    std::fclose(f);
    return out;
}

// A frame as the log shows it: guest code by its Binary Ninja address, ours
// as bbhost+offset (addr2line -e bbhost takes it), a library's by the export
// before it when there is one close by.
std::string frame_name(std::uint64_t pc, const std::vector<Module>& mods, const Module* self) {
    char buf[256];
    if (in_guest(pc)) {
        std::snprintf(buf, sizeof(buf), " g%llx", static_cast<unsigned long long>(pc - g_guest_lo + 0x400000ull));
        return buf;
    }
    for (const Module& m : mods) {
        if (pc < m.lo || pc >= m.hi) continue;
        Dl_info di{};
        if (&m != self && dladdr(reinterpret_cast<void*>(static_cast<std::uintptr_t>(pc)), &di) && di.dli_sname && di.dli_saddr &&
            pc - reinterpret_cast<std::uintptr_t>(di.dli_saddr) < 0x4000) {
            std::snprintf(buf, sizeof(buf), " %s!%s+0x%llx", m.name.c_str(), di.dli_sname,
                          static_cast<unsigned long long>(pc - reinterpret_cast<std::uintptr_t>(di.dli_saddr)));
        } else {
            std::snprintf(buf, sizeof(buf), " %s+0x%llx", m.name.c_str(), static_cast<unsigned long long>(pc - m.base));
        }
        return buf;
    }
    std::snprintf(buf, sizeof(buf), " %llx", static_cast<unsigned long long>(pc));
    return buf;
}

// Every thread of this process but the caller, its stack unwound by its own
// SIGPROF handler as a sample is. A signal can cut a blocking call short, so
// this only runs once the game has stopped flipping.
void log_all_threads(const char* why) {
    install_sigprof();
    static Slot* const slots = new Slot[kDumpSlots];
    for (int k = 0; k < kDumpSlots; ++k) slots[k].ready.store(0, std::memory_order_relaxed);
    g_dump_next.store(0, std::memory_order_relaxed);
    g_dump_slots.store(slots, std::memory_order_release);
    const pid_t pid = getpid();
    const int self = static_cast<int>(syscall(SYS_gettid));
    std::vector<Tid> tids = matching_tids({""});
    tids.erase(std::remove_if(tids.begin(), tids.end(), [self](const Tid& t) { return t.tid == self; }), tids.end());
    for (const Tid& t : tids) syscall(SYS_tgkill, pid, t.tid, SIGPROF);
    int answered = 0;
    for (const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2); std::chrono::steady_clock::now() < until;) {
        answered = 0;
        const int n = std::min(g_dump_next.load(std::memory_order_acquire), kDumpSlots);
        for (int k = 0; k < n; ++k) answered += slots[k].ready.load(std::memory_order_acquire) != 0;
        if (answered >= static_cast<int>(tids.size())) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    g_dump_slots.store(nullptr, std::memory_order_release);
    const std::vector<Module> mods = read_modules();
    const Module* me = nullptr;
    const auto here = reinterpret_cast<std::uintptr_t>(&read_modules);
    for (const Module& m : mods) me = here >= m.lo && here < m.hi ? &m : me;
    host_log("watchdog: %s - %zu threads, %d answered:", why, tids.size(), answered);
    const int n = std::min(g_dump_next.load(std::memory_order_acquire), kDumpSlots);
    for (const Tid& t : tids) {
        const Slot* s = nullptr;
        for (int k = 0; k < n && !s; ++k) {
            if (slots[k].ready.load(std::memory_order_acquire) && static_cast<int>(slots[k].tid) == t.tid) s = &slots[k];
        }
        std::string line;
        if (!s) line = " (no answer)";
        for (int k = 0; s && k < s->n && k < 18; ++k) {
            line += frame_name(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(s->pc[k])), mods, me);
        }
        host_log("  %d %s:%s", t.tid, t.tid == g_main_tid.load() ? "main" : t.name.c_str(), line.c_str());
    }
}
}  // namespace

// The Windows watchdog's stall reports and stack dumps, once the game has
// flipped a while: the start is left alone, where a signal's cut-short call
// is the one most likely to matter (BBHOST_SAMPLE_FROM_FLIP).
void sampler_watchdog_start(std::uint64_t guest_lo, std::uint64_t guest_hi) {
    if (!g_guest_lo) {
        g_guest_lo = guest_lo;
        g_guest_hi = guest_hi;
    }
    // BBHOST_WATCHDOG_S (default 15; 0 off): how long without a flip is a stall.
    static const int stall_s = [] {
        const char* e = std::getenv("BBHOST_WATCHDOG_S");
        return e ? std::max(0, std::atoi(e)) : 15;
    }();
    if (!stall_s) return;
    install_sigprof();
    std::thread([] {
        host_thread_set_name("bb-watchdog");
        std::uint64_t last_flip = hle_video_flip_count();
        int still_s = 0, dumps = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            const std::uint64_t flip = hle_video_flip_count();
            if (flip != last_flip || flip < 60) {
                if (still_s >= stall_s) host_log("watchdog: flips again after %d s (flip %llu)", still_s, static_cast<unsigned long long>(flip));
                last_flip = flip;
                still_s = 0;
                continue;
            }
            ++still_s;
            if ((still_s == stall_s || still_s == stall_s + 30) && dumps < 6) {
                ++dumps;
                char why[64];
                std::snprintf(why, sizeof(why), "no flip for %d s at flip %llu", still_s, static_cast<unsigned long long>(flip));
                log_all_threads(why);
            }
        }
    }).detach();
}

namespace {
// BBHOST_SAMPLE_PERF=1: the same samples through perf_events instead of
// signals - a task-clock event per thread (on-CPU time only) with the user
// call chain the kernel walks by frame pointer, read from each event's ring.
// Nothing is delivered to the sampled thread, so a blocking call is never cut
// short; signals to the game's main loop hung it.
struct PerfRing {
    int fd = -1;
    int tid = 0;
    void* map = nullptr;
    std::size_t data_size = 0;
};

bool perf_open(PerfRing& r, int tid, int hz) {
    perf_event_attr a{};
    a.size = sizeof(a);
    a.type = PERF_TYPE_SOFTWARE;
    a.config = PERF_COUNT_SW_TASK_CLOCK;
    a.freq = 1;
    a.sample_freq = static_cast<std::uint64_t>(hz);
    a.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_CALLCHAIN;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.exclude_callchain_kernel = 1;
    a.sample_max_stack = 64;
    const long fd = syscall(SYS_perf_event_open, &a, tid, -1, -1, 0);
    if (fd < 0) return false;
    const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    r.data_size = page * 256;
    r.map = mmap(nullptr, page + r.data_size, PROT_READ | PROT_WRITE, MAP_SHARED, static_cast<int>(fd), 0);
    if (r.map == MAP_FAILED) {
        close(static_cast<int>(fd));
        return false;
    }
    r.fd = static_cast<int>(fd);
    r.tid = tid;
    return true;
}

// Writes the ring's new samples as sampler lines.
void perf_drain(PerfRing& r, FILE* f) {
    auto* meta = static_cast<perf_event_mmap_page*>(r.map);
    const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    const auto* data = static_cast<const std::uint8_t*>(r.map) + page;
    std::uint64_t head = __atomic_load_n(&meta->data_head, __ATOMIC_ACQUIRE);
    std::uint64_t tail = meta->data_tail;
    const std::uint64_t flip = hle_video_flip_count();
    std::vector<std::uint8_t> rec;
    while (tail < head) {
        perf_event_header h;
        for (std::size_t k = 0; k < sizeof(h); ++k) reinterpret_cast<std::uint8_t*>(&h)[k] = data[(tail + k) % r.data_size];
        if (!h.size) break;
        rec.resize(h.size);
        for (std::size_t k = 0; k < h.size; ++k) rec[k] = data[(tail + k) % r.data_size];
        tail += h.size;
        if (h.type != PERF_RECORD_SAMPLE) continue;
        const std::uint8_t* p = rec.data() + sizeof(h);
        std::uint64_t ip;
        std::uint32_t pid, tid;
        std::uint64_t nr;
        std::memcpy(&ip, p, 8);
        std::memcpy(&pid, p + 8, 4);
        std::memcpy(&tid, p + 12, 4);
        std::memcpy(&nr, p + 16, 8);
        const auto* chain = reinterpret_cast<const std::uint64_t*>(p + 24);
        std::fprintf(f, "%llu %u", static_cast<unsigned long long>(flip), tid);
        for (std::uint64_t k = 0; k < nr && k < 64; ++k) {
            const std::uint64_t pc = chain[k];
            if (pc >= static_cast<std::uint64_t>(PERF_CONTEXT_MAX)) continue;  // context markers
            if (in_guest(pc)) {
                std::fprintf(f, " g%llx", static_cast<unsigned long long>(pc - g_guest_lo + 0x400000ull));
            } else {
                std::fprintf(f, " %llx", static_cast<unsigned long long>(pc));
            }
        }
        std::fputc('\n', f);
    }
    __atomic_store_n(&meta->data_tail, tail, __ATOMIC_RELEASE);
}
}  // namespace

void sampler_start(std::uint64_t guest_lo, std::uint64_t guest_hi) {
    const char* names = std::getenv("BBHOST_SAMPLE");
    if (!names || !*names) return;
    const std::vector<std::string> prefixes = split(names);
    const char* hz_env = std::getenv("BBHOST_SAMPLE_HZ");
    const int hz = hz_env ? std::max(10, std::atoi(hz_env)) : 997;
    const char* out_env = std::getenv("BBHOST_SAMPLE_OUT");
    const std::string out = out_env ? out_env : "build/samples";

    g_guest_lo = guest_lo;
    g_guest_hi = guest_hi;
    if (const char* pe = std::getenv("BBHOST_SAMPLE_PERF"); pe && pe[0] == '1') {
        FILE* f = std::fopen((out + ".txt").c_str(), "w");
        if (!f) return;
        host_log("sampler: %s at %d Hz of on-CPU time through perf_events into %s.txt", names, hz, out.c_str());
        std::thread([prefixes, hz, out, f] {
            host_thread_set_name("bb-sampler");
            std::vector<PerfRing> rings;
            std::vector<int> named;
            static const std::uint64_t from_flip = [] {
                const char* e = std::getenv("BBHOST_SAMPLE_FROM_FLIP");
                return e ? std::strtoull(e, nullptr, 10) : 0ull;
            }();
            auto last_scan = std::chrono::steady_clock::now() - std::chrono::seconds(10), last_maps = last_scan;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (from_flip && hle_video_flip_count() < from_flip) continue;
                const auto now = std::chrono::steady_clock::now();
                if (now - last_scan > std::chrono::seconds(1)) {
                    last_scan = now;
                    for (const Tid& t : matching_tids(prefixes)) {
                        if (std::find(named.begin(), named.end(), t.tid) != named.end()) continue;
                        named.push_back(t.tid);
                        PerfRing r;
                        if (!perf_open(r, t.tid, hz)) {
                            host_log("sampler: perf_event_open for thread %d failed (errno %d)", t.tid, errno);
                            continue;
                        }
                        rings.push_back(r);
                        std::fprintf(f, "#thread %d %s\n", t.tid, t.name.c_str());
                    }
                }
                for (PerfRing& r : rings) perf_drain(r, f);
                if (now - last_maps > std::chrono::seconds(5)) {
                    std::fflush(f);
                    copy_file("/proc/self/maps", out + ".maps");
                    last_maps = now;
                }
            }
        }).detach();
        return;
    }
    g_slots = new Slot[kSlots];
    install_sigprof();

    FILE* f = std::fopen((out + ".txt").c_str(), "w");
    if (!f) {
        host_log("sampler: cannot write %s.txt", out.c_str());
        return;
    }
    host_log("sampler: %s at %d Hz into %s.txt", names, hz, out.c_str());
    std::thread([prefixes, hz, out, f] {
        host_thread_set_name("bb-sampler");
        const pid_t pid = getpid();
        const auto period = std::chrono::nanoseconds(1000000000 / hz);
        auto next = std::chrono::steady_clock::now();
        auto last_scan = next - std::chrono::seconds(10), last_maps = next;
        std::vector<Tid> tids;
        std::vector<int> named;
        std::uint64_t written = 0;
        std::uint64_t said_faults = 0, said_dropped = 0;
        std::uint64_t skipped = 0, said_skipped = 0, worst_late_flip = 0;
        double worst_late_ms = 0;
        // BBHOST_SAMPLE_FROM_FLIP=N: no samples before flip N. A signal cuts a
        // blocking call short, and the guest's loading code does not always
        // take that well; sampling the main loop from boot kept the world
        // from loading.
        static const std::uint64_t from_flip = [] {
            const char* e = std::getenv("BBHOST_SAMPLE_FROM_FLIP");
            return e ? std::strtoull(e, nullptr, 10) : 0ull;
        }();
        for (;;) {
            next += period;
            std::this_thread::sleep_until(next);
            if (from_flip && hle_video_flip_count() < from_flip) continue;
            const auto now = std::chrono::steady_clock::now();
            // Held up (a slow write below, or no CPU for this thread): the
            // ticks it missed are dropped, not fired back to back. Catching
            // up put 150-350 samples a thread on one flip every ~30 s of a
            // soak - frames of that length that never happened.
            if (now - next > 4 * period) {
                skipped += static_cast<std::uint64_t>((now - next) / period);
                const double late_ms = std::chrono::duration<double, std::milli>(now - next).count();
                if (late_ms > worst_late_ms) {
                    worst_late_ms = late_ms;
                    worst_late_flip = hle_video_flip_count();
                }
                next = now;
            }
            if (now - last_scan > std::chrono::seconds(1)) {
                tids = matching_tids(prefixes);
                last_scan = now;
                for (const Tid& t : tids) {
                    if (std::find(named.begin(), named.end(), t.tid) != named.end()) continue;
                    named.push_back(t.tid);
                    std::fprintf(f, "#thread %d %s\n", t.tid, t.name.c_str());
                }
                const double scan_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - now).count();
                if (scan_ms > 20) host_log("sampler: finding the threads took %.0f ms", scan_ms);
            }
            for (const Tid& t : tids) syscall(SYS_tgkill, pid, t.tid, SIGPROF);
            // Write what the handlers finished, in order.
            const std::uint64_t upto = g_next.load(std::memory_order_acquire);
            while (written < upto) {
                Slot& s = g_slots[written % kSlots];
                if (s.ready.load(std::memory_order_acquire) == 0 || s.seq != written) {
                    if (upto - written < 1024) break;  // still being filled
                    ++written;                          // dropped: its slot was busy
                    continue;
                }
                std::fprintf(f, "%llu %u", static_cast<unsigned long long>(s.flip), s.tid);
                for (int k = 0; k < s.n; ++k) {
                    const auto pc = static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(s.pc[k]));
                    if (in_guest(pc)) {
                        std::fprintf(f, " g%llx", pc - g_guest_lo + 0x400000ull);  // Binary Ninja's address
                    } else {
                        std::fprintf(f, " %llx", pc);
                    }
                }
                std::fputc('\n', f);
                s.ready.store(0, std::memory_order_release);
                ++written;
            }
            if (now - last_maps > std::chrono::seconds(5)) {
                const auto t0 = std::chrono::steady_clock::now();
                std::fflush(f);
                const auto t1 = std::chrono::steady_clock::now();
                copy_file("/proc/self/maps", out + ".maps");
                const double flush_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                const double maps_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
                if (flush_ms + maps_ms > 20) host_log("sampler: writing the samples took %.0f ms, copying the maps %.0f ms", flush_ms, maps_ms);
                if (skipped != said_skipped) {
                    host_log("sampler: %llu ticks skipped while this thread was held up (the longest %.0f ms, at flip %llu)",
                             static_cast<unsigned long long>(skipped), worst_late_ms, static_cast<unsigned long long>(worst_late_flip));
                    said_skipped = skipped;
                }
                // What the handlers could not do, when it changes.
                const std::uint64_t faults = g_unwind_faults.load(std::memory_order_relaxed);
                const std::uint64_t dropped = g_dropped.load(std::memory_order_relaxed);
                if (faults != said_faults || dropped != said_dropped) {
                    host_log("sampler: %llu stack walks faulted (those samples keep their leaf), %llu samples dropped",
                             static_cast<unsigned long long>(faults), static_cast<unsigned long long>(dropped));
                    said_faults = faults;
                    said_dropped = dropped;
                }
                last_maps = now;
            }
        }
    }).detach();
}
#elif defined(_WIN32)

// The same sampler on Windows, without signals: the sampler thread suspends
// each thread it watches, reads its context, unwinds it (host frames by the
// PE unwind tables through RtlVirtualUnwind, guest frames by the frame
// pointers the eboot keeps) and resumes it, all before it touches the CRT.
// A thread found inside an HLE call unwinds to the HLE function and stops
// there: the trampoline is generated code with no unwind entry, so the
// guest's frames below it are not reached. The output is the Linux one:
// <out>.txt lines and <out>.maps, the modules in /proc/self/maps shape with
// the module's link base as the offset, so tools/wallprof.py symbolizes
// bbhost.exe with addr2line at its link addresses.

#include "core/portable.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>

#include "host/win_crash.h"

namespace {

constexpr int kDepth = 40;
std::uint64_t g_guest_lo = 0, g_guest_hi = 0;
std::atomic<DWORD> g_main_tid{0};

bool in_guest(std::uint64_t pc) { return pc >= g_guest_lo && pc < g_guest_hi; }

bool rd8(std::uint64_t a, std::uint64_t* v) {
    return host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(a)), v, 8);
}

struct Sample {
    std::uint64_t flip = 0;
    DWORD tid = 0;
    int n = 0;
    std::uint64_t pc[kDepth];
};

void walk_guest(Sample& s, std::uint64_t rbp, std::uint64_t floor) {
    while (s.n < kDepth && rbp > floor && rbp - floor < (64u << 20) && (rbp & 7) == 0) {
        std::uint64_t next = 0, ret = 0;
        if (!rd8(rbp, &next) || !rd8(rbp + 8, &ret) || !in_guest(ret)) break;
        s.pc[s.n++] = ret;
        if (next <= rbp) break;
        rbp = next;
    }
}

void unwind(Sample& s, CONTEXT ctx) {
    int leaf_steps = 0;
    while (s.n < kDepth) {
        const std::uint64_t rip = ctx.Rip;
        if (!rip) break;
        s.pc[s.n++] = rip;
        if (in_guest(rip)) {
            walk_guest(s, ctx.Rbp, ctx.Rsp);
            break;
        }
        DWORD64 base = 0;
        RUNTIME_FUNCTION* fn = RtlLookupFunctionEntry(rip, &base, nullptr);
        if (!fn) {
            // No entry: a leaf whose return address is at rsp - when that
            // word is code. Otherwise this is generated code (the HLE
            // trampoline, whose frame holds the guest's registers), where the
            // host's part of the stack ends.
            if (++leaf_steps > 1) break;
            std::uint64_t ret = 0;
            DWORD64 ret_base = 0;
            if (!rd8(ctx.Rsp, &ret) || !(in_guest(ret) || RtlLookupFunctionEntry(ret, &ret_base, nullptr))) break;
            ctx.Rip = ret;
            ctx.Rsp += 8;
            continue;
        }
        leaf_steps = 0;
        PVOID handler_data = nullptr;
        DWORD64 establisher = 0;
        const std::uint64_t rsp_before = ctx.Rsp;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, rip, fn, &ctx, &handler_data, &establisher, nullptr);
        if (ctx.Rsp <= rsp_before && ctx.Rip == rip) break;
    }
}

using GetDesc = HRESULT(WINAPI*)(HANDLE, PWSTR*);
GetDesc get_desc() {
    static const GetDesc f = [] {
        HMODULE k = GetModuleHandleW(L"kernelbase.dll");
        if (!k) k = GetModuleHandleW(L"kernel32.dll");
        return k ? reinterpret_cast<GetDesc>(GetProcAddress(k, "GetThreadDescription")) : nullptr;
    }();
    return f;
}

std::string thread_name(HANDLE h) {
    std::string out;
    PWSTR w = nullptr;
    if (get_desc() && SUCCEEDED(get_desc()(h, &w)) && w) {
        for (int i = 0; w[i] && i < 63; ++i) out.push_back(static_cast<char>(w[i] < 128 ? w[i] : '?'));
        LocalFree(w);
    }
    return out;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::size_t a = 0;
    while (a <= s.size()) {
        const std::size_t b = s.find(',', a);
        const std::string part = s.substr(a, b == std::string::npos ? std::string::npos : b - a);
        if (!part.empty()) out.push_back(part);
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return out;
}

struct Watched {
    DWORD tid;
    HANDLE h;
    std::string name;
};

// Adds every thread of this process whose description starts with one of the
// prefixes ("main": the game's main loop) and is not yet watched.
void scan_threads(const std::vector<std::string>& prefixes, std::vector<Watched>& watched, FILE* f) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te{};
    te.dwSize = sizeof(te);
    const DWORD pid = GetCurrentProcessId(), self = GetCurrentThreadId();
    static std::vector<DWORD> seen;
    std::string seen_names;
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) continue;
        bool known = false;
        for (const Watched& w : watched) known = known || w.tid == te.th32ThreadID;
        if (known) continue;
        const bool first_time = std::find(seen.begin(), seen.end(), te.th32ThreadID) == seen.end();
        if (first_time) seen.push_back(te.th32ThreadID);
        HANDLE h = OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION | SYNCHRONIZE, FALSE,
                              te.th32ThreadID);
        if (!h) {
            if (first_time) seen_names += " " + std::to_string(te.th32ThreadID) + ":(no handle)";
            continue;
        }
        std::string name = thread_name(h);
        if (first_time) seen_names += " " + std::to_string(te.th32ThreadID) + ":" + (name.empty() ? "-" : name);
        bool hit = false;
        for (const std::string& p : prefixes) {
            hit = p == "main" ? te.th32ThreadID == g_main_tid.load() : name.compare(0, p.size(), p) == 0;
            if (hit) break;
        }
        if (!hit) {
            CloseHandle(h);
            continue;
        }
        if (name.empty()) name = te.th32ThreadID == g_main_tid.load() ? "main" : "bbhost";
        watched.push_back({te.th32ThreadID, h, name});
        std::fprintf(f, "#thread %lu %s\n", static_cast<unsigned long>(te.th32ThreadID), name.c_str());
    }
    CloseHandle(snap);
    // Each scan lists the threads it saw for the first time, named or not:
    // the names come from SetThreadDescription (host_thread_set_name); a
    // thread never named shows as "-".
    if (!seen_names.empty()) host_log("sampler: threads seen:%s", seen_names.c_str());
}

void write_maps(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, 0);
    if (snap != INVALID_HANDLE_VALUE) {
        MODULEENTRY32 me{};
        me.dwSize = sizeof(me);
        for (BOOL ok = Module32First(snap, &me); ok; ok = Module32Next(snap, &me)) {
            const auto base = reinterpret_cast<std::uint64_t>(me.modBaseAddr);
            std::uint64_t link = base;
            const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(me.modBaseAddr);
            if (dos->e_magic == IMAGE_DOS_SIGNATURE) {
                const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(me.modBaseAddr + dos->e_lfanew);
                if (nt->Signature == IMAGE_NT_SIGNATURE) link = nt->OptionalHeader.ImageBase;
            }
            std::fprintf(f, "%llx-%llx r-xp %llx 00:00 0 %s\n", static_cast<unsigned long long>(base),
                         static_cast<unsigned long long>(base + me.modBaseSize), static_cast<unsigned long long>(link),
                         me.szExePath);
        }
        CloseHandle(snap);
    }
    std::fclose(f);
}

// Every thread of this process but the caller: named, its stack unwound as
// a sample is, while it is suspended; formatted and logged after it runs
// again (nothing that takes a lock - the log's, the heap's - is touched while
// a thread that may hold it is stopped).
void log_all_threads(const char* why) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    struct Taken {
        DWORD tid;
        std::string name;
        Sample s;
    };
    std::vector<Taken> taken;
    THREADENTRY32 te{};
    te.dwSize = sizeof(te);
    const DWORD pid = GetCurrentProcessId(), self = GetCurrentThreadId();
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) continue;
        HANDLE h = OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
        if (!h) continue;
        Taken t;
        t.tid = te.th32ThreadID;
        t.name = thread_name(h);
        t.s.n = 0;
        if (SuspendThread(h) != static_cast<DWORD>(-1)) {
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            if (GetThreadContext(h, &ctx)) unwind(t.s, ctx);
            ResumeThread(h);
        }
        CloseHandle(h);
        taken.push_back(std::move(t));
    }
    CloseHandle(snap);
    host_log("watchdog: %s - %zu threads:", why, taken.size());
    for (const Taken& t : taken) {
        std::string line;
        for (int k = 0; k < t.s.n && k < 12; ++k) {
            const std::uint64_t pc = t.s.pc[k];
            char buf[160];
            if (in_guest(pc)) {
                std::snprintf(buf, sizeof(buf), " g%llx", static_cast<unsigned long long>(pc - g_guest_lo + 0x400000ull));
            } else if (win_crash_where(pc, buf + 1, sizeof(buf) - 1)) {
                buf[0] = ' ';
                if (char* sp = std::strstr(buf, " (link")) *sp = 0;  // module+offset is enough here
            } else {
                std::snprintf(buf, sizeof(buf), " %llx", static_cast<unsigned long long>(pc));
            }
            line += buf;
        }
        host_log("  %lu %s:%s", static_cast<unsigned long>(t.tid), t.name.empty() ? "-" : t.name.c_str(), line.c_str());
    }
}

}  // namespace

void sampler_recover() {}
void sampler_note_main_thread() { g_main_tid.store(GetCurrentThreadId()); }

void sampler_watchdog_start(std::uint64_t guest_lo, std::uint64_t guest_hi) {
    if (!g_guest_lo) {
        g_guest_lo = guest_lo;
        g_guest_hi = guest_hi;
    }
    // BBHOST_WATCHDOG_S (default 15; 0 off): how long without a flip is a stall.
    static const int stall_s = [] {
        const char* e = std::getenv("BBHOST_WATCHDOG_S");
        return e ? std::max(0, std::atoi(e)) : 15;
    }();
    if (!stall_s) return;
    std::thread([] {
        host_thread_set_name("bb-watchdog");
        std::uint64_t last_flip = hle_video_flip_count();
        int still_s = 0, dumps = 0;
        for (;;) {
            Sleep(1000);
            const std::uint64_t flip = hle_video_flip_count();
            if (flip != last_flip) {
                if (still_s >= stall_s) host_log("watchdog: flips again after %d s (flip %llu)", still_s, static_cast<unsigned long long>(flip));
                last_flip = flip;
                still_s = 0;
                continue;
            }
            ++still_s;
            // A stall: said every 5 s, and every thread's stack at its start
            // and 30 s on (at most six dumps a run).
            if (still_s >= stall_s && (still_s - stall_s) % 5 == 0) {
                host_log("watchdog: no flip for %d s (flip %llu)", still_s, static_cast<unsigned long long>(flip));
            }
            if ((still_s == stall_s || still_s == stall_s + 30) && dumps < 6) {
                ++dumps;
                char why[64];
                std::snprintf(why, sizeof(why), "no flip for %d s", still_s);
                log_all_threads(why);
            }
        }
    }).detach();
}

void sampler_start(std::uint64_t guest_lo, std::uint64_t guest_hi) {
    const char* names = std::getenv("BBHOST_SAMPLE");
    if (!names || !*names) return;
    const std::vector<std::string> prefixes = split(names);
    const char* hz_env = std::getenv("BBHOST_SAMPLE_HZ");
    const int hz = hz_env ? std::max(10, std::atoi(hz_env)) : 997;
    const char* out_env = std::getenv("BBHOST_SAMPLE_OUT");
    const std::string out = out_env ? out_env : "build/samples";
    g_guest_lo = guest_lo;
    g_guest_hi = guest_hi;
    if (const char* pe = std::getenv("BBHOST_SAMPLE_PERF"); pe && pe[0] == '1') {
        host_log("sampler: BBHOST_SAMPLE_PERF is Linux only; sampling by thread contexts instead");
    }
    FILE* f = std::fopen((out + ".txt").c_str(), "w");
    if (!f) {
        host_log("sampler: cannot write %s.txt", out.c_str());
        return;
    }
    host_log("sampler: %s at %d Hz into %s.txt (thread contexts)", names, hz, out.c_str());
    std::thread([prefixes, hz, out, f] {
        host_thread_set_name("bb-sampler");
        const auto period = std::chrono::nanoseconds(1000000000 / hz);
        auto next = std::chrono::steady_clock::now();
        auto last_scan = next - std::chrono::seconds(10), last_maps = next - std::chrono::seconds(10);
        std::vector<Watched> watched;
        static const std::uint64_t from_flip = [] {
            const char* e = std::getenv("BBHOST_SAMPLE_FROM_FLIP");
            return e ? std::strtoull(e, nullptr, 10) : 0ull;
        }();
        Sample s;
        for (;;) {
            next += period;
            std::this_thread::sleep_until(next);
            if (from_flip && hle_video_flip_count() < from_flip) continue;
            const auto now = std::chrono::steady_clock::now();
            if (now - last_scan > std::chrono::seconds(1)) {
                last_scan = now;
                for (std::size_t i = 0; i < watched.size();) {
                    if (WaitForSingleObject(watched[i].h, 0) == WAIT_OBJECT_0) {  // gone
                        CloseHandle(watched[i].h);
                        watched.erase(watched.begin() + static_cast<std::ptrdiff_t>(i));
                    } else {
                        ++i;
                    }
                }
                scan_threads(prefixes, watched, f);
            }
            const std::uint64_t flip = hle_video_flip_count();
            for (const Watched& w : watched) {
                s.flip = flip;
                s.tid = w.tid;
                s.n = 0;
                if (SuspendThread(w.h) == static_cast<DWORD>(-1)) continue;
                CONTEXT ctx{};
                ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
                const bool got = GetThreadContext(w.h, &ctx) != 0;
                if (got) unwind(s, ctx);
                ResumeThread(w.h);
                if (!got || !s.n) continue;
                std::fprintf(f, "%llu %lu", static_cast<unsigned long long>(s.flip), static_cast<unsigned long>(s.tid));
                for (int k = 0; k < s.n; ++k) {
                    const std::uint64_t pc = s.pc[k];
                    if (in_guest(pc)) {
                        std::fprintf(f, " g%llx", static_cast<unsigned long long>(pc - g_guest_lo + 0x400000ull));
                    } else {
                        std::fprintf(f, " %llx", static_cast<unsigned long long>(pc));
                    }
                }
                std::fputc('\n', f);
            }
            if (now - last_maps > std::chrono::seconds(5)) {
                std::fflush(f);
                write_maps(out + ".maps");
                last_maps = now;
            }
        }
    }).detach();
}
#elif defined(__APPLE__)

// macOS samples the Windows way: a SIGPROF handler would run on a guest
// thread under Rosetta's signal emulation and cut blocking calls short. The
// HLE trampoline's frame links host and guest stacks (core/thunk.cpp), so one
// frame-pointer walk covers both.
//
// While a thread is suspended nothing that takes a lock is touched (malloc,
// the log, dyld, stdio): it may hold one. Rosetta's own thread answers
// thread_get_state for the others, so it is never suspended.
//
// <out>.maps gives each image's link address as its offset, so
// tools/wallprof.py takes link addresses (bbhost is not PIE).

#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <mach/mach.h>
#include <pthread.h>

namespace {

constexpr int kDepth = 40;
std::uint64_t g_guest_lo = 0, g_guest_hi = 0;
std::atomic<std::uint32_t> g_main_tid{0};

bool in_guest(std::uint64_t pc) { return pc >= g_guest_lo && pc < g_guest_hi; }

struct Image {
    std::uint64_t lo, hi;    // __TEXT in memory
    std::uint64_t header;    // the Mach-O header's address: module+offset is from here (atos -l)
    std::uint64_t link;      // __TEXT's link address
    std::string path;
};

// Every loaded image's __TEXT, from dyld - which takes its lock, so never
// while a thread is suspended.
std::vector<Image> read_images() {
    std::vector<Image> out;
    const std::uint32_t n = _dyld_image_count();
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto* h = reinterpret_cast<const mach_header_64*>(_dyld_get_image_header(i));
        const char* name = _dyld_get_image_name(i);
        if (!h || !name || h->magic != MH_MAGIC_64) continue;
        const std::uint64_t slide = static_cast<std::uint64_t>(_dyld_get_image_vmaddr_slide(i));
        const auto* lc = reinterpret_cast<const load_command*>(h + 1);
        for (std::uint32_t k = 0; k < h->ncmds; ++k) {
            if (lc->cmd == LC_SEGMENT_64) {
                const auto* seg = reinterpret_cast<const segment_command_64*>(lc);
                if (std::strncmp(seg->segname, "__TEXT", 16) == 0) {
                    out.push_back({seg->vmaddr + slide, seg->vmaddr + slide + seg->vmsize,
                                   static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(h)), seg->vmaddr, name});
                    break;
                }
            }
            lc = reinterpret_cast<const load_command*>(reinterpret_cast<const std::uint8_t*>(lc) + lc->cmdsize);
        }
    }
    std::sort(out.begin(), out.end(), [](const Image& a, const Image& b) { return a.lo < b.lo; });
    return out;
}

// Filled while no thread is suspended, read (no allocation) while one is.
struct Code {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;  // sorted
    std::uint64_t kernel_lo = 0, kernel_hi = 0;                   // libsystem_kernel: the system call stubs

    void set(const std::vector<Image>& imgs) {
        ranges.clear();
        for (const Image& m : imgs) {
            ranges.emplace_back(m.lo, m.hi);
            const char* slash = std::strrchr(m.path.c_str(), '/');
            if (std::strcmp(slash ? slash + 1 : m.path.c_str(), "libsystem_kernel.dylib") == 0) {
                kernel_lo = m.lo;
                kernel_hi = m.hi;
            }
        }
    }
    bool has(std::uint64_t pc) const {
        if (in_guest(pc)) return true;
        auto it = std::upper_bound(ranges.begin(), ranges.end(), std::make_pair(pc, ~0ull));
        return it != ranges.begin() && pc < std::prev(it)->second;
    }
    bool in_kernel_stubs(std::uint64_t pc) const { return pc >= kernel_lo && pc < kernel_hi; }
};

// vm_read_overwrite fails instead of faulting on a frame pointer that leads
// to an unmapped page.
struct StackReader {
    static constexpr int kPages = 4;
    static constexpr std::uint64_t kPage = 4096;  // Rosetta's pages
    std::uint64_t page[kPages];
    bool ok[kPages];
    int next = 0;
    alignas(16) std::uint8_t data[kPages][kPage];

    void reset() {
        for (int k = 0; k < kPages; ++k) page[k] = ~0ull;
    }
    bool rd8(std::uint64_t a, std::uint64_t* v) {
        if (a < 0x10000 || (a & 7)) return false;
        const std::uint64_t p = a & ~(kPage - 1);
        int k = 0;
        while (k < kPages && page[k] != p) ++k;
        if (k == kPages) {
            k = next;
            next = (next + 1) % kPages;
            page[k] = p;
            vm_size_t got = 0;
            ok[k] = vm_read_overwrite(mach_task_self(), static_cast<vm_address_t>(p), kPage,
                                      reinterpret_cast<vm_address_t>(data[k]), &got) == KERN_SUCCESS &&
                    got == kPage;
        }
        if (!ok[k]) return false;
        std::memcpy(v, data[k] + (a - p), 8);
        return true;
    }
};

struct Sample {
    std::uint64_t flip = 0;
    std::uint32_t tid = 0;
    int n = 0;
    std::uint64_t pc[kDepth];
};

// The chain may jump between the host and guest stacks at HLE calls and
// guest callbacks.
void unwind(Sample& s, const x86_thread_state64_t& st, StackReader& r, const Code& code) {
    r.reset();
    s.pc[s.n++] = st.__rip;
    // A system call stub keeps no frame: the caller's return address is at
    // rsp, rbp still the caller's frame.
    if (code.in_kernel_stubs(st.__rip)) {
        std::uint64_t ret = 0;
        if (r.rd8(st.__rsp, &ret) && code.has(ret)) s.pc[s.n++] = ret;
    }
    std::uint64_t rbp = st.__rbp;
    int jumps = 0;
    while (s.n < kDepth) {
        std::uint64_t up = 0, ret = 0;
        if (!r.rd8(rbp, &up) || !r.rd8(rbp + 8, &ret) || !code.has(ret)) break;
        s.pc[s.n++] = ret;
        if (up == rbp) break;
        if ((up < rbp || up - rbp > (64ull << 20)) && ++jumps > 4) break;
        rbp = up;
    }
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::size_t a = 0;
    while (a <= s.size()) {
        const std::size_t b = s.find(',', a);
        const std::string part = s.substr(a, b == std::string::npos ? std::string::npos : b - a);
        if (!part.empty()) out.push_back(part);
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return out;
}

struct Thread {
    thread_act_t port = MACH_PORT_NULL;  // holds a send right until release_threads
    std::uint32_t tid = 0;
    std::string name;  // pthread_setname_np's; unnamed: "main" for the game's main loop, else "bbhost" (Linux's comm)
};

bool rosetta_thread(const char* name) { return std::strncmp(name, "com.apple.rosetta", 17) == 0; }

// Asked once, without suspending: a thread it fails for is never suspended.
bool readable(thread_act_t port, std::uint32_t tid) {
    static std::mutex mu;
    static std::vector<std::pair<std::uint32_t, bool>> known;
    std::lock_guard<std::mutex> lock(mu);
    for (const auto& k : known) {
        if (k.first == tid) return k.second;
    }
    x86_thread_state64_t st;
    mach_msg_type_number_t c = x86_THREAD_STATE64_COUNT;
    const bool ok = thread_get_state(port, x86_THREAD_STATE64, reinterpret_cast<thread_state_t>(&st), &c) == KERN_SUCCESS;
    known.emplace_back(tid, ok);
    return ok;
}

// The ports are released by release_threads.
std::vector<Thread> list_threads(int* skipped = nullptr) {
    std::vector<Thread> out;
    thread_act_array_t th = nullptr;
    mach_msg_type_number_t n = 0;
    if (task_threads(mach_task_self(), &th, &n) != KERN_SUCCESS) return out;
    const mach_port_t self = mach_thread_self();
    out.reserve(n);
    int skip = 0;
    for (mach_msg_type_number_t i = 0; i < n; ++i) {
        thread_identifier_info_data_t id{};
        thread_extended_info_data_t ei{};
        mach_msg_type_number_t c = THREAD_IDENTIFIER_INFO_COUNT;
        const bool have_id = thread_info(th[i], THREAD_IDENTIFIER_INFO, reinterpret_cast<thread_info_t>(&id), &c) == KERN_SUCCESS;
        c = THREAD_EXTENDED_INFO_COUNT;
        if (thread_info(th[i], THREAD_EXTENDED_INFO, reinterpret_cast<thread_info_t>(&ei), &c) != KERN_SUCCESS) ei.pth_name[0] = 0;
        ei.pth_name[sizeof(ei.pth_name) - 1] = 0;
        const auto tid = static_cast<std::uint32_t>(id.thread_id);
        if (th[i] == self || !have_id || rosetta_thread(ei.pth_name) || !readable(th[i], tid)) {
            if (th[i] != self) ++skip;
            mach_port_deallocate(mach_task_self(), th[i]);
            continue;
        }
        Thread t;
        t.port = th[i];
        t.tid = tid;
        t.name = ei.pth_name[0] ? ei.pth_name : tid == g_main_tid.load() ? "main" : "bbhost";
        out.push_back(std::move(t));
    }
    mach_port_deallocate(mach_task_self(), self);
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(th), n * sizeof(th[0]));
    if (skipped) *skipped = skip;
    return out;
}

void release_threads(std::vector<Thread>& ts) {
    for (Thread& t : ts) mach_port_deallocate(mach_task_self(), t.port);
    ts.clear();
}

// Nothing between suspend and resume may take a lock. False when the thread
// is gone or would not answer.
bool take(const Thread& t, Sample& s, StackReader& r, const Code& code) {
    s.tid = t.tid;
    s.n = 0;
    if (thread_suspend(t.port) != KERN_SUCCESS) return false;
    x86_thread_state64_t st;
    mach_msg_type_number_t c = x86_THREAD_STATE64_COUNT;
    const bool got = thread_get_state(t.port, x86_THREAD_STATE64, reinterpret_cast<thread_state_t>(&st), &c) == KERN_SUCCESS;
    if (got) unwind(s, st, r, code);
    thread_resume(t.port);
    return got && s.n;
}

void write_maps(const std::string& path, const std::vector<Image>& imgs) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    for (const Image& m : imgs) {
        std::fprintf(f, "%llx-%llx r-xp %llx 00:00 0 %s\n", static_cast<unsigned long long>(m.lo), static_cast<unsigned long long>(m.hi),
                     static_cast<unsigned long long>(m.link), m.path.c_str());
    }
    std::fclose(f);
}

// Ours as bbhost+offset from the Mach-O header, which atos -l takes.
std::string frame_name(std::uint64_t pc, const std::vector<Image>& imgs) {
    char buf[256];
    if (in_guest(pc)) {
        std::snprintf(buf, sizeof(buf), " g%llx", static_cast<unsigned long long>(pc - g_guest_lo + 0x400000ull));
        return buf;
    }
    for (const Image& m : imgs) {
        if (pc < m.lo || pc >= m.hi) continue;
        const char* slash = std::strrchr(m.path.c_str(), '/');
        const char* name = slash ? slash + 1 : m.path.c_str();
        Dl_info di{};
        if (m.header != host_image_base() && dladdr(reinterpret_cast<void*>(static_cast<std::uintptr_t>(pc)), &di) && di.dli_sname &&
            di.dli_saddr && pc - reinterpret_cast<std::uintptr_t>(di.dli_saddr) < 0x4000) {
            std::snprintf(buf, sizeof(buf), " %s!%s+0x%llx", name, di.dli_sname,
                          static_cast<unsigned long long>(pc - reinterpret_cast<std::uintptr_t>(di.dli_saddr)));
        } else {
            std::snprintf(buf, sizeof(buf), " %s+0x%llx", name, static_cast<unsigned long long>(pc - m.header));
        }
        return buf;
    }
    std::snprintf(buf, sizeof(buf), " %llx", static_cast<unsigned long long>(pc));
    return buf;
}

// Formatted and logged only once every thread runs again.
void log_all_threads(const char* why) {
    const std::vector<Image> imgs = read_images();
    static Code code;
    code.set(imgs);
    static StackReader* const reader = new StackReader;
    int skipped = 0;
    std::vector<Thread> ts = list_threads(&skipped);
    std::vector<Sample> samples(ts.size());
    int answered = 0;
    for (std::size_t i = 0; i < ts.size(); ++i) answered += take(ts[i], samples[i], *reader, code);
    host_log("watchdog: %s - %zu threads, %d answered (%d of Rosetta's or unreadable left out):", why, ts.size(), answered, skipped);
    for (std::size_t i = 0; i < ts.size(); ++i) {
        std::string line;
        if (!samples[i].n) line = " (no answer)";
        for (int k = 0; k < samples[i].n && k < 18; ++k) line += frame_name(samples[i].pc[k], imgs);
        host_log("  %u %s:%s", ts[i].tid, ts[i].name.c_str(), line.c_str());
    }
    release_threads(ts);
}

}  // namespace

void sampler_recover() {}  // the walks read through vm_read_overwrite: nothing faults
void sampler_note_main_thread() { g_main_tid.store(host_thread_id()); }

// As on Windows: suspending a thread does not cut a blocking call short, so
// the watchdog runs from the start.
void sampler_watchdog_start(std::uint64_t guest_lo, std::uint64_t guest_hi) {
    if (!g_guest_lo) {
        g_guest_lo = guest_lo;
        g_guest_hi = guest_hi;
    }
    // BBHOST_WATCHDOG_S (default 15; 0 off): how long without a flip is a stall.
    static const int stall_s = [] {
        const char* e = std::getenv("BBHOST_WATCHDOG_S");
        return e ? std::max(0, std::atoi(e)) : 15;
    }();
    if (!stall_s) return;
    std::thread([] {
        host_thread_set_name("bb-watchdog");
        std::uint64_t last_flip = hle_video_flip_count();
        int still_s = 0, dumps = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            const std::uint64_t flip = hle_video_flip_count();
            if (flip != last_flip) {
                if (still_s >= stall_s) host_log("watchdog: flips again after %d s (flip %llu)", still_s, static_cast<unsigned long long>(flip));
                last_flip = flip;
                still_s = 0;
                continue;
            }
            ++still_s;
            // A stall: said every 5 s, and every thread's stack at its start
            // and 30 s on (at most six dumps a run).
            if (still_s >= stall_s && (still_s - stall_s) % 5 == 0) {
                host_log("watchdog: no flip for %d s (flip %llu)", still_s, static_cast<unsigned long long>(flip));
            }
            if ((still_s == stall_s || still_s == stall_s + 30) && dumps < 6) {
                ++dumps;
                char why[64];
                std::snprintf(why, sizeof(why), "no flip for %d s", still_s);
                log_all_threads(why);
            }
        }
    }).detach();
}

void sampler_start(std::uint64_t guest_lo, std::uint64_t guest_hi) {
    const char* names = std::getenv("BBHOST_SAMPLE");
    if (!names || !*names) return;
    const std::vector<std::string> prefixes = split(names);
    const char* hz_env = std::getenv("BBHOST_SAMPLE_HZ");
    const int hz = hz_env ? std::max(10, std::atoi(hz_env)) : 997;
    const char* out_env = std::getenv("BBHOST_SAMPLE_OUT");
    const std::string out = out_env ? out_env : "build/samples";
    g_guest_lo = guest_lo;
    g_guest_hi = guest_hi;
    if (const char* pe = std::getenv("BBHOST_SAMPLE_PERF"); pe && pe[0] == '1') {
        host_log("sampler: BBHOST_SAMPLE_PERF is Linux only; sampling by thread states instead");
    }
    FILE* f = std::fopen((out + ".txt").c_str(), "w");
    if (!f) {
        host_log("sampler: cannot write %s.txt", out.c_str());
        return;
    }
    host_log("sampler: %s at %d Hz into %s.txt (thread states)", names, hz, out.c_str());
    std::thread([prefixes, hz, out, f] {
        host_thread_set_name("bb-sampler");
        const auto period = std::chrono::nanoseconds(1000000000 / hz);
        auto next = std::chrono::steady_clock::now();
        auto last_scan = next - std::chrono::seconds(10), last_maps = next - std::chrono::seconds(10);
        std::vector<Thread> watched;
        std::vector<std::uint32_t> named;
        std::vector<Image> imgs;
        Code code;
        auto* reader = new StackReader;
        std::uint64_t skipped = 0, said_skipped = 0, worst_late_flip = 0;
        double worst_late_ms = 0;
        static const std::uint64_t from_flip = [] {
            const char* e = std::getenv("BBHOST_SAMPLE_FROM_FLIP");
            return e ? std::strtoull(e, nullptr, 10) : 0ull;
        }();
        Sample s;
        for (;;) {
            next += period;
            std::this_thread::sleep_until(next);
            if (from_flip && hle_video_flip_count() < from_flip) continue;
            const auto now = std::chrono::steady_clock::now();
            // Held up: the ticks missed are dropped, not fired back to back (as on Linux).
            if (now - next > 4 * period) {
                skipped += static_cast<std::uint64_t>((now - next) / period);
                const double late_ms = std::chrono::duration<double, std::milli>(now - next).count();
                if (late_ms > worst_late_ms) {
                    worst_late_ms = late_ms;
                    worst_late_flip = hle_video_flip_count();
                }
                next = now;
            }
            if (now - last_scan > std::chrono::seconds(1)) {
                last_scan = now;
                imgs = read_images();
                code.set(imgs);
                release_threads(watched);
                for (Thread& t : list_threads()) {
                    bool hit = false;
                    for (const std::string& p : prefixes) {
                        hit = p == "main" ? t.tid == g_main_tid.load() : t.name.compare(0, p.size(), p) == 0;
                        if (hit) break;
                    }
                    if (!hit) {
                        mach_port_deallocate(mach_task_self(), t.port);
                        continue;
                    }
                    if (std::find(named.begin(), named.end(), t.tid) == named.end()) {
                        named.push_back(t.tid);
                        std::fprintf(f, "#thread %u %s\n", t.tid, t.name.c_str());
                    }
                    watched.push_back(std::move(t));
                }
                const double scan_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - now).count();
                if (scan_ms > 20) host_log("sampler: finding the threads took %.0f ms", scan_ms);
            }
            s.flip = hle_video_flip_count();
            for (const Thread& t : watched) {
                if (!take(t, s, *reader, code)) continue;
                std::fprintf(f, "%llu %u", static_cast<unsigned long long>(s.flip), s.tid);
                for (int k = 0; k < s.n; ++k) {
                    const std::uint64_t pc = s.pc[k];
                    if (in_guest(pc)) {
                        std::fprintf(f, " g%llx", static_cast<unsigned long long>(pc - g_guest_lo + 0x400000ull));  // Binary Ninja's address
                    } else {
                        std::fprintf(f, " %llx", static_cast<unsigned long long>(pc));
                    }
                }
                std::fputc('\n', f);
            }
            if (now - last_maps > std::chrono::seconds(5)) {
                std::fflush(f);
                write_maps(out + ".maps", imgs);
                if (skipped != said_skipped) {
                    host_log("sampler: %llu ticks skipped while this thread was held up (the longest %.0f ms, at flip %llu)",
                             static_cast<unsigned long long>(skipped), worst_late_ms, static_cast<unsigned long long>(worst_late_flip));
                    said_skipped = skipped;
                }
                last_maps = now;
            }
        }
    }).detach();
}
#endif
