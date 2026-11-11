// Stage 4b.1a (host/gxvk_lists.h): the pool that looks ahead of the command
// processor at each GX-native draw a GX thread has stored, through the caches'
// reader paths (host/cache_rw.h), and the command processor's compare of its
// own answers with the pool's (BBHOST_GXVK_LISTS=2).
//
// The feed: the recording thread, once it has stored a draw's slot, puts the
// token's id on a ring (draw_stored). A pool thread takes the next id, holds
// the slot for reading (hle_gx_native_pin: the command processor's take waits
// for it), runs the draw's lookups under the shared lock (render.cpp
// lists_look_draw), writes its answers into the record of the same index as
// the slot, and gives the slot back. The record of token `id` is written only
// while its slot is held and read by the command processor only after it took
// the slot, and the next token of that index cannot be stored before the
// command processor released this one: the slot's state orders every access.
#include "host/gxvk_lists.h"
#include "host/gxvk_lists_internal.h"
#include "host/gxvk_rest.h"

#include "hle/modules.h"
#include "host/cache_rw.h"
#include "host/gpu.h"
#include "host/gpu_internal.h"
#include "core/portable.h"
#include "core/host_clock.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if !defined(_WIN32)
#include <fstream>
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace gxvk_lists {

thread_local constinit bool t_cp_look = false;
const unsigned g_look_every = [] {
    const char* e = std::getenv("BBHOST_GXVK_LISTS_SAMPLE");
    const long n = e ? std::strtol(e, nullptr, 10) : 1;
    return static_cast<unsigned>(n < 1 ? 1 : n > 4096 ? 4096 : n);
}();

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using ull = unsigned long long;

static_assert(static_cast<int>(kLookProgram) == cache_rw::kPrograms && static_cast<int>(kLookFetch) == cache_rw::kFetch &&
                  static_cast<int>(kLookPaths) == cache_rw::kPaths && static_cast<int>(kLookPlan) == cache_rw::kPlans &&
                  static_cast<int>(kLookView) == cache_rw::kViews && static_cast<int>(kLookSampler) == cache_rw::kSamplers &&
                  static_cast<int>(kLookKinds) == cache_rw::kCaches,
              "a lookup's kind is its cache");
const char* const kKindName[kLookKinds] = {"programs", "fetch shaders", "translations", "stage plans", "views", "samplers"};

void pause() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

// A counter one thread writes and the report reads.
struct Count {
    std::atomic<u64> n{0};
    void add(u64 k = 1) { n.store(n.load(std::memory_order_relaxed) + k, std::memory_order_relaxed); }
    u64 get() const { return n.load(std::memory_order_relaxed); }
};
// A counter several threads write.
struct Shared {
    std::atomic<u64> n{0};
    void add(u64 k = 1) { n.fetch_add(k, std::memory_order_relaxed); }
    u64 get() const { return n.load(std::memory_order_relaxed); }
};

// ---- the feed (host/gxvk_lists_internal.h) --------------------------------
using detail::Feed;
using detail::g_feed;
using detail::g_feed_next;
using detail::g_feed_pos;
using detail::kFeed;
Shared g_fed, g_feed_lapped, g_feed_skipped, g_lists_fed;
Shared g_parked_ticks, g_parks;  // BBHOST_GXVK_LISTS_TIMES: the threads' time parked (rdtsc ticks)

// BBHOST_GXVK_LISTS_TIMES on Linux: each pool thread's own hardware counters (user space only) - cycles, instructions,
// last-level cache misses and its task clock - read for the lines, so a commit's cost can be told clock-, cache- or
// contention-bound. None where perf_event_open refuses (counted).
#if !defined(_WIN32)
constexpr int kHwCounters = 4;  // cycles, instructions, LLC misses, task clock (ns)
constexpr int kHwThreads = 8;
int g_hw_fd[kHwThreads][kHwCounters];
std::atomic<int> g_hw_threads{0};
std::atomic<int> g_hw_refused{0};
void hw_open() {
    const int t = g_hw_threads.fetch_add(1, std::memory_order_relaxed);
    if (t >= kHwThreads) return;
    const std::pair<std::uint32_t, std::uint64_t> what[kHwCounters] = {{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
                                                                      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
                                                                      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
                                                                      {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK}};
    for (int k = 0; k < kHwCounters; ++k) {
        perf_event_attr a{};
        a.size = sizeof(a);
        a.type = what[k].first;
        a.config = what[k].second;
        a.exclude_kernel = 1;
        a.exclude_hv = 1;
        g_hw_fd[t][k] = static_cast<int>(syscall(SYS_perf_event_open, &a, 0, -1, -1, 0));
        if (g_hw_fd[t][k] < 0) g_hw_refused.fetch_add(1, std::memory_order_relaxed);
    }
}
std::string hw_line() {
    const int n = std::min(g_hw_threads.load(std::memory_order_relaxed), kHwThreads);
    u64 sum[kHwCounters] = {};
    for (int t = 0; t < n; ++t) {
        for (int k = 0; k < kHwCounters; ++k) {
            u64 v = 0;
            if (g_hw_fd[t][k] >= 0 && read(g_hw_fd[t][k], &v, sizeof(v)) == sizeof(v)) sum[k] += v;
        }
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), " | hardware (%d threads, %d refused): cycles %llu, instructions %llu, llc misses %llu, task clock ns %llu", n,
                  g_hw_refused.load(std::memory_order_relaxed), static_cast<ull>(sum[0]), static_cast<ull>(sum[1]), static_cast<ull>(sum[2]),
                  static_cast<ull>(sum[3]));
    return buf;
}
#else
void hw_open() {}
std::string hw_line() { return std::string(); }
#endif

// ---- the rest at a frame cap (host/gxvk_lists.h resting) ----
const bool g_rest_on = [] {
    const char* e = std::getenv("BBHOST_GXVK_LISTS_REST");
    return !(e && e[0] == '0');
}();
// BBHOST_GXVK_LISTS_REST=cycle:N (a check, as lavapipe never meets a cap): rest N frames, work N frames, whatever the
// limiter waited - every transition met mid-list, as play meets them. =always: it rests from the first frame on (what a
// resting pool costs against the pool off, BBHOST_GXVK_LISTS=0).
const unsigned g_rest_cycle = [] {
    const char* e = std::getenv("BBHOST_GXVK_LISTS_REST");
    return e && std::strncmp(e, "cycle:", 6) == 0 ? static_cast<unsigned>(std::max(1L, std::strtol(e + 6, nullptr, 10))) : 0u;
}();
const bool g_rest_always = [] {
    const char* e = std::getenv("BBHOST_GXVK_LISTS_REST");
    return e && std::strcmp(e, "always") == 0;
}();
std::atomic<u64> g_cp_busy_ns{0};  // bb-cp0's jobs' time (one writer)
std::atomic<u64> g_cp_held_ns{0};  // the game's waits for the command processor's backlog (cp_held)
std::atomic<u64> g_rests{0}, g_rested_ns{0};  // the rests begun, and their time (ended ones)
// The rests by why they began (gxvk_rest.h Why: a cap, GPU-bound, an experiment's, an experiment's decision), the
// experiments (probes from working, asks from a decided rest - those a heavier place began) and what they decided, and
// the rests a loading screen ended.
std::atomic<u64> g_rests_by[gxvk_rest::kWhys] = {}, g_probes{0}, g_asks{0}, g_held_asks{0}, g_decided_rest{0}, g_decided_work{0},
    g_load_wakes{0}, g_quiet_asks{0}, g_trials_rested{0};
// BBHOST_GXVK_LISTS_REST_LOG=1: every window of the duty's rule logged, with the state it left the pool in.
const bool g_rest_log = [] {
    const char* e = std::getenv("BBHOST_GXVK_LISTS_REST_LOG");
    return e && e[0] && e[0] != '0';
}();

// ---- the answers -----------------------------------------------------------
// One record per token slot (the slot ring's size and index).
constexpr u64 kLookSlots = 16384;
struct LookRecord {
    u64 id = 0;
    u64 gen[cache_rw::kCaches] = {};
    LookSink sink;
};
std::atomic<LookRecord*> g_look{nullptr};

// The pool's counts (its threads').
Shared g_looked, g_late, g_lost, g_look_ns;
Shared g_kind_looked[kLookKinds], g_kind_found[kLookKinds];

// The command processor's (compare mode).
Count g_cmp_draws, g_cmp_no_record, g_cmp_empty, g_cmp_lost;
// equal; not made yet (the pool found none, and the cache was written since); missed (the pool found none, and
// nothing was written between: a reader path that does not see an entry the command processor's does); changed
// since; different (both found one, nothing written between: must stay 0); the pool's only.
enum { kEqual, kNotMade, kMissed, kChanged, kDifferent, kPoolOnly, kClasses };
Count g_cmp[kLookKinds][kClasses];
std::atomic<int> g_diff_logs{0}, g_miss_logs{0};

// ---- the pool's threads ------------------------------------------------------
std::once_flag g_started;
// The pool's parking: its threads sleep on g_park_cv while the feed is empty, counted in g_sleepers, which a
// recording thread reads after each store (sequentially consistent on both sides, so a store that a thread about
// to sleep has not seen wakes it).
std::mutex g_park_mu;
std::condition_variable g_park_cv;
std::atomic<int> g_sleepers{0};
std::atomic<bool> g_ready{false};
std::atomic<bool> g_stop{false};
std::vector<std::thread> g_threads;
std::atomic<u64> g_next_report{0};

void report_line(bool at_exit);

void look(u64 id) {
    GxDrawObjects o;
    const GpuDrawInputs* in = hle_gx_native_pin(id, &o);
    if (!in) {
        g_late.add();  // the command processor took it first, or its slot moved on
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    LookRecord& r = g_look.load(std::memory_order_relaxed)[id & (kLookSlots - 1)];  // made before the pool's threads
    r.sink.n = 0;
    r.sink.lost = 0;
    {
        cache_rw::Read rd;
        for (int c = 0; c < cache_rw::kCaches; ++c) r.gen[c] = cache_rw::gen(c);
        gpu::lists_look_draw(*in, o, r.sink);
    }
    r.id = id;
    hle_gx_native_unpin(id);
    g_looked.add();
    if (r.sink.lost) g_lost.add(r.sink.lost);
    for (u32 i = 0; i < r.sink.n; ++i) {
        const int kind = static_cast<int>(r.sink.e[i].key >> 56);
        if (kind >= kLookKinds) continue;
        g_kind_looked[kind].add();
        if (r.sink.e[i].answer) g_kind_found[kind].add();
    }
    g_look_ns.add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
}

void pool_main_loop(int index);
void pool_main(int index) {
    char name[16];
    std::snprintf(name, sizeof(name), "bb-list%d", index);
    host_thread_set_name(name);
    host_thread_set_class(HostThreadClass::GpuFeed, name);
    if (g_times) hw_open();
    pool_main_loop(index);
    if (splice()) gpu::lists_thread_end();  // its statistics merged (the exit's lines come after the join)
}
void pool_main_loop(int index) {
    unsigned idle = 0;
    while (!g_stop.load(std::memory_order_acquire)) {
        u64 pos = g_feed_next.load(std::memory_order_relaxed);
        const u64 end = g_feed_pos.load(std::memory_order_acquire);
        if (pos >= end) {
            // Nothing stored since: a few pauses (the next draw of a burst is microseconds off), then parked
            // until a recording thread stores one (draw_stored wakes it only while a thread sleeps) or 2 ms
            // pass. Spinning longer took the hyperthread siblings of the command processor and bb-record
            // (on an RTX 4070, bb-cp0 +0.13-0.48 ms a flip with the pool on).
            if (++idle < 64) {
                pause();
            } else {
                // (Stage 4b.2: a list's close is fed from FinishCommandList's leaf, which wakes nobody - a parked
                // thread looks again within half a millisecond.)
                const std::uint64_t p0 = g_times ? __builtin_ia32_rdtsc() : 0;
                std::unique_lock<std::mutex> lk(g_park_mu);
                g_sleepers.fetch_add(1, std::memory_order_seq_cst);
                if (g_feed_next.load(std::memory_order_seq_cst) >= g_feed_pos.load(std::memory_order_seq_cst) &&
                    !g_stop.load(std::memory_order_acquire))
                    g_park_cv.wait_for(lk, resting()   ? std::chrono::microseconds(100000)  // (nothing is fed meanwhile)
                                           : segments() ? std::chrono::microseconds(500)
                                                        : std::chrono::microseconds(2000));
                g_sleepers.fetch_sub(1, std::memory_order_seq_cst);
                idle = 0;
                if (p0) {
                    g_parked_ticks.add(__builtin_ia32_rdtsc() - p0);
                    g_parks.add();
                }
            }
            if (index == 0 && host_tooling()) {
                const u64 flip = hle_video_flip_count();
                u64 next = g_next_report.load(std::memory_order_relaxed);
                if (!next) {
                    g_next_report.store(flip + 1000, std::memory_order_relaxed);
                } else if (flip >= next) {
                    g_next_report.store(flip + 1000, std::memory_order_relaxed);
                    report_line(false);
                }
            }
            continue;
        }
        idle = 0;
        if (end - pos > kFeed) {
            // Lapped: the recording threads stored more than the ring holds since.
            const u64 to = end - kFeed / 2;
            if (g_feed_next.compare_exchange_strong(pos, to, std::memory_order_relaxed)) g_feed_skipped.add(to - pos);
            continue;
        }
        if (!g_feed_next.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) continue;
        Feed& f = g_feed[pos & (kFeed - 1)];
        u64 seq = 0;
        for (unsigned spins = 0; (seq = f.seq.load(std::memory_order_acquire)) < pos + 1; ++spins) {
            if (g_stop.load(std::memory_order_relaxed)) return;
            if (spins < 1024) {
                pause();
            } else {
                std::this_thread::yield();
            }
        }
        const u64 id = f.id.load(std::memory_order_relaxed);
        const std::uint32_t kind = f.kind.load(std::memory_order_relaxed);
        if (seq != pos + 1 || f.seq.load(std::memory_order_acquire) != pos + 1) {
            g_feed_lapped.add();
            continue;
        }
        if (kind == detail::kFeedList) {
            g_lists_fed.add();
            gpu::lists_segment(static_cast<std::uint32_t>(id));  // stage 4b.2: the list's segments
            continue;
        }
        if (splice()) {
            // The slot two entries on - the other pool thread most likely takes the next - asked for while this draw
            // is committed: a GX thread's core wrote its lines, and the copy that starts a commit waited for them.
            const Feed& ahead = g_feed[(pos + 2) & (kFeed - 1)];
            if (ahead.seq.load(std::memory_order_relaxed) == pos + 3 && ahead.kind.load(std::memory_order_relaxed) == detail::kFeedDraw)
                hle_gx_native_prefetch(ahead.id.load(std::memory_order_relaxed), false);
            gpu::lists_commit(id);  // stage 4b.1b
        } else {
            look(id);
        }
    }
}

// ---- the pool's cores (keep_off_pool_cpus) ----
#if !defined(_WIN32)
cpu_set_t g_pool_cpu[4];          // pool thread i's own CPU
cpu_set_t g_off_pool;             // the process's CPUs less the pool's cores (both hardware threads)
std::atomic<bool> g_off_ready{false};
// The pool's cores from the process's CPUs, its last `n` physical cores; false
// when there are too few to spare them (four cores at least must remain).
bool choose_pool_cpus(int n) {
    const char* e = std::getenv("BBHOST_GXVK_LIST_CPUS");
    if (e && std::strcmp(e, "off") == 0) return false;
    if (!splice() && !(e && std::strcmp(e, "auto") == 0)) return false;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return false;
    // Each allowed CPU's core: the lowest CPU among its siblings.
    std::vector<int> core_of(CPU_SETSIZE, -1);
    std::vector<int> cores;  // a core's lowest allowed CPU, in order
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        std::string list;
        std::getline(f, list);
        const int first = list.empty() ? cpu : std::atoi(list.c_str());  // "a,b" or "a-b": the lowest first
        core_of[cpu] = first;
        bool known = false;
        for (const int c : cores) known |= c == first;
        if (!known) cores.push_back(first);
    }
    if (static_cast<int>(cores.size()) < n + 4) return false;
    CPU_ZERO(&g_off_pool);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) CPU_SET(cpu, &g_off_pool);
    }
    std::string said;
    for (int i = 0; i < n; ++i) {
        const int core = cores[cores.size() - 1 - static_cast<std::size_t>(i)];
        CPU_ZERO(&g_pool_cpu[i]);
        CPU_SET(core, &g_pool_cpu[i]);
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (core_of[cpu] == core) CPU_CLR(cpu, &g_off_pool);
        }
        said += (said.empty() ? "" : ", ") + std::to_string(core);
    }
    host_log("gxvk-lists: the pool's threads on CPUs %s, their cores' other hardware threads left idle; the command processor and the "
             "recorder kept off those cores (BBHOST_GXVK_LIST_CPUS=off: anywhere)",
             said.c_str());
    return true;
}
#endif

void start() {
    g_look.store(new LookRecord[kLookSlots], std::memory_order_release);
    const int n = pool_threads();
    if (segments()) {
        host_log("gxvk-lists: stage 4b.2's pool: %d thread%s commit each stored GX-native draw ahead of the command processor and build "
                 "each list's segments, which it splices with one op each (BBHOST_GXVK_LISTS=%s)",
                 n, n == 1 ? "" : "s", segments_compare() ? "segments-compare: drawn in order too, and compared" : "segments");
    } else if (splice()) {
        host_log("gxvk-lists: stage 4b.1b's pool: %d thread%s commit each stored GX-native draw ahead of the command processor, which "
                 "splices them (BBHOST_GXVK_LISTS=%s)",
                 n, n == 1 ? "" : "s", splice_compare() ? "splice-compare: drawn in order too, and compared" : "splice");
    } else {
        host_log("gxvk-lists: stage 4b.1a's pool: %d thread%s look ahead of the command processor at each stored GX-native draw "
                 "(BBHOST_GXVK_LISTS=%d%s, 1 draw in %u)",
                 n, n == 1 ? "" : "s", pool_mode(), pool_mode() == 2 ? ": the command processor compares its answers with the pool's" : "",
                 g_look_every);
    }
#if !defined(_WIN32)
    const bool pinned = choose_pool_cpus(n < 4 ? n : 4);
#endif
    for (int i = 0; i < n; ++i) {
        g_threads.emplace_back(pool_main, i);
#if !defined(_WIN32)
        if (pinned && i < 4) pthread_setaffinity_np(g_threads.back().native_handle(), sizeof(cpu_set_t), &g_pool_cpu[i]);
#endif
    }
#if !defined(_WIN32)
    if (pinned) g_off_ready.store(true, std::memory_order_release);
#endif
    g_ready.store(true, std::memory_order_release);
}

std::string pct(u64 a, u64 b) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f%%", b ? 100.0 * static_cast<double>(a) / static_cast<double>(b) : 0.0);
    return buf;
}

// The counters a line reports, taken at once; the periodic lines report the last window's (the difference from
// the previous line's), the exit line the whole run's.
struct Snap {
    u64 looked = 0, look_ns = 0, fed = 0, late = 0, lapped = 0, skipped = 0, lost = 0;
    u64 kind_looked[kLookKinds] = {}, kind_found[kLookKinds] = {};
    u64 writes = 0, writes_waited = 0, reads = 0, reads_waited = 0;
    u64 cmp_draws = 0, cmp_no_record = 0, cmp_empty = 0, cmp_lost = 0;
    u64 cmp[kLookKinds][kClasses] = {};
};
Snap snap_now() {
    Snap n;
    n.looked = g_looked.get();
    n.look_ns = g_look_ns.get();
    n.fed = g_fed.get();
    n.late = g_late.get();
    n.lapped = g_feed_lapped.get();
    n.skipped = g_feed_skipped.get();
    n.lost = g_lost.get();
    for (int k = 0; k < kLookKinds; ++k) {
        n.kind_looked[k] = g_kind_looked[k].get();
        n.kind_found[k] = g_kind_found[k].get();
        for (int c = 0; c < kClasses; ++c) n.cmp[k][c] = g_cmp[k][c].get();
    }
    const cache_rw::Counts c = cache_rw::counts();
    n.writes = c.writes;
    n.writes_waited = c.writes_waited;
    n.reads = c.reads;
    n.reads_waited = c.reads_waited;
    n.cmp_draws = g_cmp_draws.get();
    n.cmp_no_record = g_cmp_no_record.get();
    n.cmp_empty = g_cmp_empty.get();
    n.cmp_lost = g_cmp_lost.get();
    return n;
}
Snap minus(const Snap& a, const Snap& b) {
    Snap d;
    const auto* pa = reinterpret_cast<const u64*>(&a);
    const auto* pb = reinterpret_cast<const u64*>(&b);
    auto* pd = reinterpret_cast<u64*>(&d);
    for (std::size_t i = 0; i < sizeof(Snap) / sizeof(u64); ++i) pd[i] = pa[i] - pb[i];
    return d;
}
static_assert(sizeof(Snap) % sizeof(u64) == 0, "a snapshot of counters");
Snap g_last_snap;  // the periodic lines' (bb-list0's)
u64 g_last_flip = 0;

void report_line(bool at_exit) {
    const Snap now = snap_now();
    const u64 flip = hle_video_flip_count();
    const Snap d = at_exit ? now : minus(now, g_last_snap);
    char what[64];
    if (at_exit) {
        std::snprintf(what, sizeof(what), "the whole run");
    } else {
        std::snprintf(what, sizeof(what), "flips %llu-%llu", static_cast<ull>(g_last_flip), static_cast<ull>(flip));
        g_last_snap = now;
        g_last_flip = flip;
    }
    std::string kinds;
    for (int k = 0; k < kLookKinds; ++k) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s %llu (%s found)%s", kKindName[k], static_cast<ull>(d.kind_looked[k]),
                      pct(d.kind_found[k], d.kind_looked[k]).c_str(), k + 1 < kLookKinds ? ", " : "");
        kinds += buf;
    }
    if (splice()) {
        host_log("gxvk-lists (%s): stage 4b.1b: %s", what, host_gpu_splice_report(!at_exit).c_str());
        if (segments()) {
            host_log("gxvk-lists (%s): stage 4b.2: %s", what, host_gpu_segment_report(false).c_str());
            host_log("gxvk-lists (%s): stage 4b.2: %s", what, host_gpu_segment_report(true).c_str());
        }
        if (g_times) {
            for (int part = 0; part < 4; ++part)
                host_log("gxvk-lists (%s): %s%s", what, host_gpu_lists_times_report(part, g_parked_ticks.get(), g_parks.get()).c_str(),
                         part == 1 ? hw_line().c_str() : "");
        }
        if (g_rest_on && (g_rests.load(std::memory_order_relaxed) || at_exit)) {
            const u64 rests = g_rests.load(std::memory_order_relaxed);
            const auto by = [](int k) { return static_cast<ull>(g_rests_by[k].load(std::memory_order_relaxed)); };
            const auto n = [](const std::atomic<u64>& a) { return static_cast<ull>(a.load(std::memory_order_relaxed)); };
            host_log("gxvk-lists (%s): the pool rested %llu times (at a frame cap %llu, GPU-bound %llu, an experiment's %llu, decided %llu, "
                     "a trial near the cap %llu of which %llu rested on), "
                     "%.1f s in the ended ones%s; experiments: %llu probes, %llu asks (%llu at a heavier place; %llu not made, the rest "
                     "holding the game back under 10%%), %llu decided it rests, %llu that it works; %llu rests ended at a loading screen",
                     what, static_cast<ull>(rests), by(gxvk_rest::kWhyCap), by(gxvk_rest::kWhyGpu), by(gxvk_rest::kWhyProbe),
                     by(gxvk_rest::kWhyJudged), by(gxvk_rest::kWhyNear), n(g_trials_rested),
                     static_cast<double>(g_rested_ns.load(std::memory_order_relaxed)) / 1e9,
                     resting() ? "; resting now" : "", n(g_probes), n(g_asks), n(g_held_asks), n(g_quiet_asks), n(g_decided_rest),
                     n(g_decided_work), n(g_load_wakes));
        }
        return;
    }
    host_log("gxvk-lists (%s): the pool looked at %llu draws, %.2f us a draw (fed %llu; the command processor first %llu; feed lapped "
             "%llu, skipped %llu; answers past a record %llu); its lookups: %s; cache_rw: writes %llu (%llu waited for a reader), "
             "reads %llu (%llu waited for the writer)",
             what, static_cast<ull>(d.looked), d.looked ? static_cast<double>(d.look_ns) / 1000.0 / static_cast<double>(d.looked) : 0.0,
             static_cast<ull>(d.fed), static_cast<ull>(d.late), static_cast<ull>(d.lapped), static_cast<ull>(d.skipped),
             static_cast<ull>(d.lost), kinds.c_str(), static_cast<ull>(d.writes), static_cast<ull>(d.writes_waited),
             static_cast<ull>(d.reads), static_cast<ull>(d.reads_waited));
    if (pool_mode() != 2) return;
    std::string cmp;
    u64 different = 0, missed = 0;
    for (int k = 0; k < kLookKinds; ++k) {
        char buf[200];
        std::snprintf(buf, sizeof(buf), "%s %llu equal / %llu not made yet / %llu missed / %llu changed since / %llu different / %llu the "
                      "pool's only%s",
                      kKindName[k], static_cast<ull>(d.cmp[k][kEqual]), static_cast<ull>(d.cmp[k][kNotMade]), static_cast<ull>(d.cmp[k][kMissed]),
                      static_cast<ull>(d.cmp[k][kChanged]), static_cast<ull>(d.cmp[k][kDifferent]), static_cast<ull>(d.cmp[k][kPoolOnly]),
                      k + 1 < kLookKinds ? "; " : "");
        cmp += buf;
        different += d.cmp[k][kDifferent];
        missed += d.cmp[k][kMissed];
    }
    host_log("gxvk-lists (%s): the command processor compared %llu draws with the pool's answers (no record %llu, none to compare %llu, "
             "its notes past %d %llu): %llu different, %llu missed; %s",
             what, static_cast<ull>(d.cmp_draws), static_cast<ull>(d.cmp_no_record), static_cast<ull>(d.cmp_empty), 128,
             static_cast<ull>(d.cmp_lost), static_cast<ull>(different), static_cast<ull>(missed), cmp.c_str());
}

// The command processor's notes for the draw under way (compare mode).
constexpr u32 kCpMost = 128;
thread_local u64 t_cp_id = 0;
thread_local u32 t_cp_n = 0;
thread_local LookEntry t_cp_e[kCpMost];
// Its notes by key, for the draw's compare: an open-addressed table whose
// slots are told current by the draw's stamp.
constexpr u32 kCpTable = 256;
struct CpSlot {
    u64 stamp = 0, key = 0, answer = 0;
};
thread_local CpSlot t_cp_table[kCpTable];
thread_local u64 t_cp_stamp = 0;

u32 slot_of(u64 key) { return static_cast<u32>((key * 0x9e3779b97f4a7c15ull) >> 56) & (kCpTable - 1); }

}  // namespace

void settle(bool homes_bind, const char* homes_why) {
    static std::once_flag once;
    std::call_once(once, [&] {
        const int env = pool_mode_from_env();
        const int n = pool_threads();
        const unsigned hw = host_cpu_threads();
        int mode = env < 0 ? 0 : env;
        if (env < 0) {
            // Unset: stage 4b.2's segments where the homes bind by handle (their spliced share needs them), off elsewhere.
            if (!n) {
                host_log("gxvk-lists: segments off - %u hardware threads, the pool needs 8 or more", hw);
            } else if (!homes_bind) {
                // (No why: the homes are off by their default, whose line said why and names both switches.)
                if (homes_why) host_log("gxvk-lists: segments off - %s (BBHOST_GXVK_LISTS=segments: on all the same)", homes_why);
            } else {
                mode = 5;
                host_log("gxvk-lists: segments on with the constant buffers' homes (BBHOST_GXVK_LISTS=0: off)");
            }
        } else if (env > 0 && !n) {
            host_log("gxvk-lists: BBHOST_GXVK_LISTS asks for the pool, which has no threads here (%u hardware threads): off", hw);
            mode = 0;
        } else if (env >= 3 && !homes_bind) {
            host_log("gxvk-lists: the homes are off (%s): most draws read a copy version and are not committed",
                     homes_why ? homes_why : "by their default here, as their line says");
        }
        if (mode && n) cache_rw::g_readers.store(true, std::memory_order_release);
        g_pool_mode.store(mode && n ? mode : 0, std::memory_order_release);
    });
}

void keep_off_pool_cpus() {
#if !defined(_WIN32)
    static thread_local bool applied = false;
    if (applied || !g_off_ready.load(std::memory_order_acquire)) return;
    applied = true;
    sched_setaffinity(0, sizeof(g_off_pool), &g_off_pool);
#endif
}

std::atomic<bool> g_resting{false};

void cp_busy(std::uint64_t ns) { g_cp_busy_ns.store(g_cp_busy_ns.load(std::memory_order_relaxed) + ns, std::memory_order_relaxed); }
void cp_held(std::uint64_t ns) { g_cp_held_ns.fetch_add(ns, std::memory_order_relaxed); }

namespace {
const char* duty_state(const gxvk_rest::State& st) {
    using namespace gxvk_rest;
    switch (st.phase) {
    case kProbeRest: return "resting (a probe)";
    case kProbeWork: return "working (a probe's comparison)";
    case kAskWork: return "working (an ask)";
    case kAskRest: return "resting (an ask's comparison)";
    default: break;
    }
    if (!st.resting) return "working";
    switch (st.why) {
    case kWhyCap: return "resting (a cap)";
    case kWhyGpu: return "resting (GPU-bound)";
    case kWhyJudged: return "resting (decided)";
    case kWhyNear: return "resting (a trial near the cap)";
    default: return "resting";
    }
}
}  // namespace

void frame_paced(std::uint64_t wait_ns, float cap_fps, bool loading) {
    if (!g_rest_on || !pool_mode()) return;
    if (g_rest_always) {
        if (!resting()) {
            g_resting.store(true, std::memory_order_relaxed);
            g_rests.fetch_add(1, std::memory_order_relaxed);
            host_log("gxvk-lists: the pool rests from here on (BBHOST_GXVK_LISTS_REST=always)");
        }
        return;
    }
    // The main thread's alone: windows of half a second (or 120 frames), each decided by host/gxvk_rest.h.
    static u64 last = 0, win_ns = 0, win_wait = 0, frames = 0, cp_at = 0, held_at = 0, gpu_at = 0, cycle_n = 0, since = 0;
    static bool win_loading = false, gpu_known = false;
    static gxvk_rest::State state;
    static int logs = 0;
    if (g_rest_cycle) {
        // BBHOST_GXVK_LISTS_REST=cycle:N: rest N frames, work N frames, whatever the limiter waited (its windows logged
        // with BBHOST_GXVK_LISTS_REST_LOG=1: what each switch costs, and for how long).
        const bool rest = (cycle_n++ / g_rest_cycle) & 1;
        if (rest != resting()) {
            g_resting.store(rest, std::memory_order_relaxed);
            if (rest) g_rests.fetch_add(1, std::memory_order_relaxed);
            if (g_rest_log) host_log("gxvk-lists: duty cycle: the pool %s at frame %llu", rest ? "rests" : "works", static_cast<ull>(cycle_n));
            since = 0;
        }
        ++since;
        if (!g_rest_log) return;
    }
    const u64 now = host_clock_monotonic_ns();
    if (!last) {
        last = now;
        cp_at = g_cp_busy_ns.load(std::memory_order_relaxed);
        held_at = g_cp_held_ns.load(std::memory_order_relaxed);
        const GpuBusy b = host_gpu_busy();
        gpu_known = b.on;
        gpu_at = b.busy_ns;
        return;
    }
    win_ns += now - last;
    last = now;
    win_wait += wait_ns;
    win_loading = win_loading || loading;
    ++frames;
    if (win_ns < 500'000'000ull && frames < 120) return;
    gxvk_rest::Window w;
    const double span = static_cast<double>(win_ns);
    w.slack = static_cast<double>(win_wait) / span;
    const u64 cp_now = g_cp_busy_ns.load(std::memory_order_relaxed);
    w.cp = static_cast<double>(cp_now - cp_at) / span;
    const u64 held_now = g_cp_held_ns.load(std::memory_order_relaxed);
    w.held = static_cast<double>(held_now - held_at) / span;
    const GpuBusy b = host_gpu_busy();
    if (b.on && gpu_known) w.gpu = static_cast<double>(b.busy_ns - gpu_at) / span;
    gpu_known = b.on;
    gpu_at = b.busy_ns;
    w.cap_fps = cap_fps;
    w.loading = win_loading;
    const double fps = static_cast<double>(frames) * 1e9 / span;
    w.fps = fps;
    cp_at = cp_now;
    held_at = held_now;
    win_ns = win_wait = frames = 0;
    win_loading = false;
    if (g_rest_cycle) {
        host_log("gxvk-lists: duty window: %.1f fps, the limiter %.0f%%, cp %.0f%%, held %.1f%%, gpu %.0f%%%s - %s, %llu frames since the switch",
                 fps, 100.0 * w.slack, 100.0 * w.cp, 100.0 * w.held, 100.0 * w.gpu, w.loading ? ", loading" : "",
                 resting() ? "resting" : "working", static_cast<ull>(since));
        return;
    }
    const u64 rest_at = state.rest_at;
    const bool was_probe = state.phase == gxvk_rest::kProbeRest || state.phase == gxvk_rest::kProbeWork;
    const gxvk_rest::Step step = gxvk_rest::step(state, w, now);
    if (g_rest_log)
        host_log("gxvk-lists: duty window: %.1f fps, the limiter %.0f%% (cap %.0f), cp %.0f%%, held %.1f%%, gpu %.0f%%%s - %s",
                 fps, 100.0 * w.slack, static_cast<double>(cap_fps), 100.0 * w.cp, 100.0 * w.held, 100.0 * w.gpu,
                 w.loading ? ", loading" : "", duty_state(state));
    if (step == gxvk_rest::kRest) {
        g_resting.store(true, std::memory_order_relaxed);
        g_rests.fetch_add(1, std::memory_order_relaxed);
        g_rests_by[state.why].fetch_add(1, std::memory_order_relaxed);
    } else if (step == gxvk_rest::kWake) {
        g_resting.store(false, std::memory_order_relaxed);
        g_rested_ns.fetch_add(now - rest_at, std::memory_order_relaxed);
    }
    const double rested_s = static_cast<double>(now - rest_at) / 1e9;
    const char* what = was_probe ? "probe" : "ask";
    const bool log = state.event != gxvk_rest::kEvNone && logs++ < 96;
    switch (state.event) {
    case gxvk_rest::kEvCapRest:
        if (log)
            host_log("gxvk-lists: the pool rests - the frame waits %.0f%% of its time for its cap (%.0f fps at %.0f), the command "
                     "processor busy %.0f%%: its draws are drawn in order (BBHOST_GXVK_LISTS_REST=0: never)",
                     100.0 * w.slack, fps, static_cast<double>(cap_fps), 100.0 * w.cp);
        break;
    case gxvk_rest::kEvGpuRest:
        if (log)
            host_log("gxvk-lists: the pool rests - the frame is the GPU's (busy %.0f%% of the window, %.0f fps): its draws are drawn in "
                     "order (BBHOST_GXVK_LISTS_REST=0: never)",
                     100.0 * w.gpu, fps);
        break;
    case gxvk_rest::kEvProbe:
        g_probes.fetch_add(1, std::memory_order_relaxed);
        if (log)
            host_log("gxvk-lists: the pool rests for a probe (%.1f fps working, the command processor busy %.0f%%): whether resting "
                     "costs the frame anything here (BBHOST_GXVK_LISTS_REST=0: never rests)",
                     state.before, 100.0 * w.cp);
        break;
    case gxvk_rest::kEvProbeBack:
        if (log)
            host_log("gxvk-lists: the probe's rest ran %.1f fps against %.1f working before it (the game held back %.0f%% of it, the "
                     "median): the pool works again to compare",
                     state.middle, state.before, 100.0 * state.rest_held);
        break;
    case gxvk_rest::kEvAsk:
        g_asks.fetch_add(1, std::memory_order_relaxed);
        if (state.held_ask) g_held_asks.fetch_add(1, std::memory_order_relaxed);
        if (log)
            host_log("gxvk-lists: the pool works again to ask anew after %.1f s at rest (%.1f fps resting%s)", rested_s, state.before,
                     state.held_ask ? ", the game held back 20% or more and the frame 5% slower than the rest began: a heavier place" : "");
        break;
    case gxvk_rest::kEvAskBack:
        if (log)
            host_log("gxvk-lists: the ask's work ran %.1f fps against %.1f resting before it: the pool rests again to compare", state.middle,
                     state.before);
        break;
    case gxvk_rest::kEvRests: {
        g_decided_rest.fetch_add(1, std::memory_order_relaxed);
        const double r = was_probe ? state.middle : (state.before + state.after) / 2, k = was_probe ? (state.before + state.after) / 2 : state.middle;
        if (log)
            host_log("gxvk-lists: the %s rests the pool - %.1f fps resting against %.1f working (%.1f, %.1f, %.1f), the game held back "
                     "%.0f%% resting: %s; asked anew in %.0f s",
                     what, r, k, state.before, state.middle, state.after, 100.0 * state.rest_held,
                     r >= k * (1.0 - gxvk_rest::kCost) ? "resting costs the frame nothing here"
                                                        : "the command processor holds nothing back, the pool cannot shorten the frame",
                     static_cast<double>(state.interval_ns) / 1e9);
        break;
    }
    case gxvk_rest::kEvWorks: {
        g_decided_work.fetch_add(1, std::memory_order_relaxed);
        const bool early = state.after <= 0;
        const double r = was_probe ? state.middle : (early ? state.before : (state.before + state.after) / 2);
        const double k = was_probe ? (early ? state.before : (state.before + state.after) / 2) : state.middle;
        char third[32] = ", decided at once";
        if (!early) std::snprintf(third, sizeof(third), ", %.1f", state.after);
        if (log)
            host_log("gxvk-lists: the %s keeps the pool working - %.1f fps resting against %.1f working (%.1f, %.1f%s): its work "
                     "shortens the frame %.1f%%; the next probe in %.0f s",
                     what, r, k, state.before, state.middle, third, r > 0 ? 100.0 * (k / r - 1.0) : 0.0,
                     static_cast<double>(state.interval_ns) / 1e9);
        break;
    }
    case gxvk_rest::kEvCapWake:
        if (log)
            host_log("gxvk-lists: the pool works again after %.1f s - the frame waits %.0f%% of its time (%.0f fps, cap %.0f), the "
                     "command processor busy %.0f%%, the game held back by it %.0f%%, the GPU busy %.0f%%; the next probe in %.0f s%s",
                     rested_s, 100.0 * w.slack, fps, static_cast<double>(cap_fps), 100.0 * w.cp, 100.0 * w.held, 100.0 * w.gpu,
                     static_cast<double>(state.interval_ns) / 1e9,
                     state.allowed_at > now ? (" (a rest at a cap: the next no sooner than " +
                                               std::to_string(static_cast<int>(state.backoff_ns / 1'000'000'000ull)) + " s on)").c_str()
                                            : "");
        break;
    case gxvk_rest::kEvQuiet:
        g_quiet_asks.fetch_add(1, std::memory_order_relaxed);
        if (log)
            host_log("gxvk-lists: the rest's ask is not made - the game held back %.1f%% resting (under 10%%: nothing for the pool to "
                     "shorten); the next in %.0f s",
                     100.0 * gxvk_rest::detail::median(state.held, state.held_n), static_cast<double>(state.interval_ns) / 1e9);
        break;
    case gxvk_rest::kEvNearTrial:
        if (log)
            host_log("gxvk-lists: the pool rests for a trial - the frame waits %.0f%% of its time for its cap (%.0f fps at %.0f), near it, the "
                     "command processor busy %.0f%%: whether the cap holds the frame without the pool (BBHOST_GXVK_LISTS_REST=0: never rests)",
                     100.0 * w.slack, fps, static_cast<double>(cap_fps), 100.0 * w.cp);
        break;
    case gxvk_rest::kEvNearRests:
        g_trials_rested.fetch_add(1, std::memory_order_relaxed);
        if (log)
            host_log("gxvk-lists: the trial rests on - without the pool the frame waits %.0f%% of its time for its cap (the median; %.0f fps "
                     "at %.0f), the command processor busy %.0f%%: its draws are drawn in order as at any cap",
                     100.0 * gxvk_rest::detail::median(state.trial_slack, state.trial_n), fps, static_cast<double>(cap_fps),
                     100.0 * gxvk_rest::detail::median(state.trial_cp, state.trial_n));
        break;
    case gxvk_rest::kEvNearWorks:
        if (log)
            host_log("gxvk-lists: the trial wakes the pool - without it the frame waits %.0f%% of its time for its cap (the median; %.0f fps "
                     "at %.0f), the command processor busy %.0f%%; the next trial no sooner than %d s on",
                     100.0 * gxvk_rest::detail::median(state.trial_slack, state.trial_n), fps, static_cast<double>(cap_fps),
                     100.0 * gxvk_rest::detail::median(state.trial_cp, state.trial_n),
                     static_cast<int>(state.backoff_ns / 1'000'000'000ull));
        break;
    case gxvk_rest::kEvLoadWake:
        g_load_wakes.fetch_add(1, std::memory_order_relaxed);
        if (log)
            host_log("gxvk-lists: the pool works again at a loading screen after %.1f s at rest: the next place's own experiments decide",
                     rested_s);
        break;
    default:
        break;
    }
}

void draw_stored(std::uint64_t id, std::uint64_t ctx, std::uint64_t token_va, bool feed) {
    if (!g_ready.load(std::memory_order_acquire)) std::call_once(g_started, start);
    // Not fed (the pool rested when it was stored - the caller's decision, written in its slot): no list record, and
    // the command processor draws it in order (counted there).
    if (!feed) return;
    if (g_times) {
        detail::DrawTime& t = detail::g_draw_times[id & (detail::kTimeSlots - 1)];
        t.id.store(id, std::memory_order_relaxed);
        t.stored.store(__builtin_ia32_rdtsc(), std::memory_order_relaxed);
        t.feed.store(detail::feed_push(id, detail::kFeedDraw), std::memory_order_relaxed);
    } else {
        detail::feed_push(id, detail::kFeedDraw);
    }
    if (segments() && ctx && ctx != detail::g_immediate.load(std::memory_order_relaxed)) {
        // Stage 4b.2: into its context's open list record (this thread's while it records the context).
        if (detail::CtxSlot* s = detail::ctx_slot(ctx)) {
            std::uint32_t open = s->open.load(std::memory_order_relaxed);
            if (!open) {
                for (int tries = 0; tries < 16 && !open; ++tries) {
                    const u64 at = detail::g_rec_pos.fetch_add(1, std::memory_order_relaxed) % detail::kRecs;
                    std::uint32_t free_state = detail::kRecFree;
                    if (detail::g_recs[at].state.compare_exchange_strong(free_state, detail::kRecOpen, std::memory_order_acq_rel)) {
                        detail::ListRec& r = detail::g_recs[at];
                        r.n = 0;
                        r.ctx = ctx;
                        r.list = 0;
                        open = static_cast<std::uint32_t>(at) + 1;
                    }
                }
                if (!open) {
                    detail::g_rec_none.fetch_add(1, std::memory_order_relaxed);
                    open = detail::kRecs + 1;  // none for this list (FinishCommandList clears it)
                }
                s->open.store(open, std::memory_order_relaxed);
            }
            if (open <= detail::kRecs) {
                detail::ListRec& r = detail::g_recs[open - 1];
                r.id[r.n] = id;
                r.va[r.n] = token_va;
                if (++r.n == detail::kRecMost) detail::list_closed(*s, 0);  // full: closed as it is
            } else {
                detail::g_rec_none.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    if (g_sleepers.load(std::memory_order_seq_cst)) {  // a parked pool thread: once a burst, not once a draw
        std::lock_guard<std::mutex> lk(g_park_mu);
        g_park_cv.notify_one();
    }
    g_fed.add();
}

void cp_look_begin(std::uint64_t id) {
    t_cp_look = true;
    t_cp_id = id;
    t_cp_n = 0;
}

void cp_note_slow(LookKind kind, std::uint64_t key, std::uint64_t answer) {
    if (t_cp_n < kCpMost) {
        t_cp_e[t_cp_n++] = {look_key(kind, key), answer};
    } else {
        g_cmp_lost.add();
    }
}

void cp_look_end() {
    t_cp_look = false;
    const LookRecord* const look = g_look.load(std::memory_order_acquire);
    if (!look) {
        g_cmp_no_record.add();
        return;
    }
    const LookRecord& r = look[t_cp_id & (kLookSlots - 1)];
    if (r.id != t_cp_id) {
        g_cmp_no_record.add();
        return;
    }
    if (!r.sink.n || !t_cp_n) {
        g_cmp_empty.add();
        return;
    }
    g_cmp_draws.add();
    const u64 stamp = ++t_cp_stamp;
    for (u32 i = 0; i < t_cp_n; ++i) {
        const LookEntry& e = t_cp_e[i];
        for (u32 s = slot_of(e.key), probe = 0; probe < kCpTable; ++probe, s = (s + 1) & (kCpTable - 1)) {
            CpSlot& c = t_cp_table[s];
            if (c.stamp != stamp || c.key == e.key) {  // the draw's last answer for a key stands
                c = {stamp, e.key, e.answer};
                break;
            }
        }
    }
    for (u32 i = 0; i < r.sink.n; ++i) {
        const LookEntry& e = r.sink.e[i];
        const int kind = static_cast<int>(e.key >> 56);
        if (kind >= kLookKinds) continue;
        const CpSlot* found = nullptr;
        for (u32 s = slot_of(e.key), probe = 0; probe < kCpTable; ++probe, s = (s + 1) & (kCpTable - 1)) {
            const CpSlot& c = t_cp_table[s];
            if (c.stamp != stamp) break;
            if (c.key == e.key) {
                found = &c;
                break;
            }
        }
        if (!found) {
            g_cmp[kind][kPoolOnly].add();
            continue;
        }
        const bool written = cache_rw::gen(kind) != r.gen[kind];
        if (found->answer == e.answer) {
            g_cmp[kind][kEqual].add();
        } else if (!e.answer) {
            g_cmp[kind][written ? kNotMade : kMissed].add();
            if (!written && g_miss_logs.fetch_add(1, std::memory_order_relaxed) < 8) {
                host_log("gxvk-lists: compare: token %llu's %s key 0x%014llx: the pool found none, the command processor 0x%llx, with "
                         "no write between (flip %llu)",
                         static_cast<ull>(t_cp_id), kKindName[kind], static_cast<ull>(e.key & 0x00ffffffffffffffull),
                         static_cast<ull>(found->answer), static_cast<ull>(hle_video_flip_count()));
            }
        } else if (written) {
            g_cmp[kind][kChanged].add();
        } else {
            g_cmp[kind][kDifferent].add();
            if (g_diff_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
                host_log("gxvk-lists: compare: token %llu's %s key 0x%014llx: the pool answered 0x%llx, the command processor 0x%llx, "
                         "with no write between (flip %llu)",
                         static_cast<ull>(t_cp_id), kKindName[kind], static_cast<ull>(e.key & 0x00ffffffffffffffull),
                         static_cast<ull>(e.answer), static_cast<ull>(found->answer), static_cast<ull>(hle_video_flip_count()));
            }
        }
    }
}

void pool_report(bool at_exit) {
    if (!pool_mode()) return;
    if (at_exit) {
        // Stopped and joined first: an exit through std::exit must not race them.
        g_stop.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(g_park_mu);
            g_park_cv.notify_all();
        }
        for (std::thread& t : g_threads) {
            if (t.joinable()) t.join();
        }
        g_threads.clear();
    }
    report_line(at_exit);
}

}  // namespace gxvk_lists
