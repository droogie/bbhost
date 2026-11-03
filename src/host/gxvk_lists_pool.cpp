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

#include "hle/modules.h"
#include "host/cache_rw.h"
#include "host/gpu.h"
#include "host/gpu_internal.h"
#include "core/portable.h"
#include "core/host_clock.h"
#include "log.h"

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
#include <sched.h>
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

// ---- the feed ----------------------------------------------------------
struct Feed {
    std::atomic<u64> seq{0};  // its place plus one once `id` is written
    std::atomic<u64> id{0};
};
constexpr u64 kFeed = 1u << 15;
Feed* g_feed = nullptr;  // made before the first draw is fed (start)
std::atomic<u64> g_feed_pos{0};   // the recording threads' next place
std::atomic<u64> g_feed_next{0};  // the pool's next place
Shared g_fed, g_feed_lapped, g_feed_skipped;

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

void pool_main(int index) {
    char name[16];
    std::snprintf(name, sizeof(name), "bb-list%d", index);
    host_thread_set_name(name);
    host_thread_set_class(HostThreadClass::GpuFeed, name);
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
                std::unique_lock<std::mutex> lk(g_park_mu);
                g_sleepers.fetch_add(1, std::memory_order_seq_cst);
                if (g_feed_next.load(std::memory_order_seq_cst) >= g_feed_pos.load(std::memory_order_seq_cst) &&
                    !g_stop.load(std::memory_order_acquire))
                    g_park_cv.wait_for(lk, std::chrono::milliseconds(2));
                g_sleepers.fetch_sub(1, std::memory_order_seq_cst);
                idle = 0;
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
        if (seq != pos + 1 || f.seq.load(std::memory_order_acquire) != pos + 1) {
            g_feed_lapped.add();
            continue;
        }
        if (splice()) {
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
    g_feed = new Feed[kFeed];
    g_look.store(new LookRecord[kLookSlots], std::memory_order_release);
    const int n = pool_threads();
    if (splice()) {
        host_log("gxvk-lists: stage 4b.1b's pool: %d thread%s commit each stored GX-native draw ahead of the command processor, which "
                 "splices them (BBHOST_GXVK_LISTS=%s)",
                 n, n == 1 ? "" : "s", splice_compare() ? "splice-compare: drawn in order too, and compared" : "splice");
    } else {
        host_log("gxvk-lists: stage 4b.1a's pool: %d thread%s look ahead of the command processor at each stored GX-native draw "
                 "(BBHOST_GXVK_LISTS=%d%s, 1 draw in %u)",
                 n, n == 1 ? "" : "s", g_pool_mode, g_pool_mode == 2 ? ": the command processor compares its answers with the pool's" : "",
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
        return;
    }
    host_log("gxvk-lists (%s): the pool looked at %llu draws, %.2f us a draw (fed %llu; the command processor first %llu; feed lapped "
             "%llu, skipped %llu; answers past a record %llu); its lookups: %s; cache_rw: writes %llu (%llu waited for a reader), "
             "reads %llu (%llu waited for the writer)",
             what, static_cast<ull>(d.looked), d.looked ? static_cast<double>(d.look_ns) / 1000.0 / static_cast<double>(d.looked) : 0.0,
             static_cast<ull>(d.fed), static_cast<ull>(d.late), static_cast<ull>(d.lapped), static_cast<ull>(d.skipped),
             static_cast<ull>(d.lost), kinds.c_str(), static_cast<ull>(d.writes), static_cast<ull>(d.writes_waited),
             static_cast<ull>(d.reads), static_cast<ull>(d.reads_waited));
    if (g_pool_mode != 2) return;
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

void keep_off_pool_cpus() {
#if !defined(_WIN32)
    static thread_local bool applied = false;
    if (applied || !g_off_ready.load(std::memory_order_acquire)) return;
    applied = true;
    sched_setaffinity(0, sizeof(g_off_pool), &g_off_pool);
#endif
}

void draw_stored(std::uint64_t id) {
    if (!g_ready.load(std::memory_order_acquire)) std::call_once(g_started, start);
    const u64 pos = g_feed_pos.fetch_add(1, std::memory_order_seq_cst);
    Feed& f = g_feed[pos & (kFeed - 1)];
    f.id.store(id, std::memory_order_relaxed);
    f.seq.store(pos + 1, std::memory_order_seq_cst);
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
    if (!g_pool_mode) return;
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
