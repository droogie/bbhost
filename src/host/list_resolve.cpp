// Resolving draws where their command lists are finished: the run census
// (host/list_resolve.h).
#include "host/list_resolve.h"

#include "core/futex.h"
#include "core/host_clock.h"
#include "core/portable.h"
#include "hle/modules.h"
#include "host/gpu.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace list_resolve {

const int g_mode = [] {
    const char* e = std::getenv("BBHOST_LIST_RESOLVE");
    if (!e || !*e) return 0;
    if (std::strcmp(e, "runs") == 0) return 1;
    return std::atoi(e) ? 1 : 0;
}();

namespace {
// BBHOST_RUN_COMMIT's mode: on (2) unless it says otherwise - 0 or off, fork
// (1), compare (3), or the number.
int run_commit_from_env() {
    const char* e = std::getenv("BBHOST_RUN_COMMIT");
    if (!e || !*e || std::strcmp(e, "on") == 0) return 2;
    if (std::strcmp(e, "off") == 0) return 0;
    if (std::strcmp(e, "fork") == 0) return 1;
    if (std::strcmp(e, "compare") == 0) return 3;
    return std::clamp(std::atoi(e), 0, 3);
}
}  // namespace

const int g_batch = [] {
    const char* e = std::getenv("BBHOST_RUN_BATCH");
    // A run's parts take its targets as batching shares them: on with the commit.
    if (!e || !*e) return run_commit_from_env() >= 2 ? 1 : 0;
    if (std::strcmp(e, "compare") == 0) return 2;
    return std::atoi(e) ? 1 : 0;
}();

const int g_draw_ctx = [] {
    const char* e = std::getenv("BBHOST_DRAW_CONTEXT");
    if (!e || !*e) return 0;
    if (std::strcmp(e, "compare") == 0) return 2;
    return std::strcmp(e, "chunks") == 0 || e[0] == '1' ? 1 : 0;
}();

const int g_commit = run_commit_from_env();

namespace {

thread_local std::uint64_t t_serial = 1;
thread_local std::uint32_t t_run_left = 0;  // BBHOST_RUN_COMMIT: draws of the forked run still to come
thread_local std::uint32_t t_run_hold = 0;  // BBHOST_RUN_COMMIT=on: draws of a run not forked, still to come (no look ahead)
void log_commit();  // BBHOST_RUN_COMMIT's line (below)
std::atomic<std::uint64_t> g_batch_n[kCounts] = {};

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using ull = unsigned long long;

// Draws a run holds, by powers of two: 1, 2-3, 4-7, ..., 128 and more.
constexpr int kSizes = 8;
// The colour targets a run's draws write: none (depth only), 1, 2-3, 4 or more (the G-buffer's).
constexpr int kShapes = 4;
// Flips from a draw's GX call to the command processor reaching it: 0, 1, 2, 3 and more.
constexpr int kLags = 4;

const char* const kSplitName[kSplits] = {"targets",  "wait",        "label",       "copy",  "dispatch",    "query",
                                         "YEBIS/Scaleform", "CP write", "predication", "constant engine", "call", "draw packet",
                                         "submission end"};

struct Counts {
    u64 runs = 0, draws = 0, segments = 0, contexts = 0, lists = 0, barriers = 0;
    u64 sizes[kSizes] = {}, size_draws[kSizes] = {};
    u64 why[kSplits] = {}, why_draws[kSplits] = {};
    u64 shape_draws[kShapes] = {};
    u64 lag[kLags] = {};
};
// Added to by the command-processor threads, read by the reports.
struct Shared {
    std::atomic<u64> runs{0}, draws{0}, segments{0}, contexts{0}, lists{0}, barriers{0};
    std::atomic<u64> sizes[kSizes] = {}, size_draws[kSizes] = {};
    std::atomic<u64> why[kSplits] = {}, why_draws[kSplits] = {};
    std::atomic<u64> shape_draws[kShapes] = {};
    std::atomic<u64> lag[kLags] = {};
    Counts load() const {
        Counts c;
        c.runs = runs.load(std::memory_order_relaxed);
        c.draws = draws.load(std::memory_order_relaxed);
        c.segments = segments.load(std::memory_order_relaxed);
        c.contexts = contexts.load(std::memory_order_relaxed);
        c.lists = lists.load(std::memory_order_relaxed);
        c.barriers = barriers.load(std::memory_order_relaxed);
        for (int k = 0; k < kSizes; ++k) {
            c.sizes[k] = sizes[k].load(std::memory_order_relaxed);
            c.size_draws[k] = size_draws[k].load(std::memory_order_relaxed);
        }
        for (int k = 0; k < kSplits; ++k) {
            c.why[k] = why[k].load(std::memory_order_relaxed);
            c.why_draws[k] = why_draws[k].load(std::memory_order_relaxed);
        }
        for (int k = 0; k < kShapes; ++k) c.shape_draws[k] = shape_draws[k].load(std::memory_order_relaxed);
        for (int k = 0; k < kLags; ++k) c.lag[k] = lag[k].load(std::memory_order_relaxed);
        return c;
    }
};
Shared g_n;

// The run under way on this command-processor thread.
struct Run {
    u64 targets = 0;  // its render pass's identity (target_key)
    u64 ctx = 0;      // the GX context its last draw was recorded on
    u64 list = ~0ull;  // the GX context of the last draw, run or not: a change is another command list
    u32 draws = 0;
    int shape = 0;     // its colour targets (kShapes)
    u32 segment = 0;  // draws since the last split other than a change of targets
    u64 next_report = 0;
};
thread_local Run t_run;

u64 mix(u64 h, u64 v) { return (h ^ v) * 1099511628211ull; }

// What the render pass a run draws in is made of: each written colour
// target's base, view, format and resource generation; the depth target's,
// when the draw tests or writes depth; and a clear the draw itself asks for
// (DB_RENDER_CONTROL), which the command processor makes in its order.
u64 target_key(const GpuDrawInputs& in) {
    u64 h = mix(1469598103934665603ull, in.target_mask);
    for (int t = 0; t < 8; ++t) {
        if (!((in.target_mask >> (4 * t)) & 0xf) || !in.color[t][0]) continue;
        h = mix(h, (static_cast<u64>(t) << 32) | in.color[t][0]);
        h = mix(h, (static_cast<u64>(in.color[t][3]) << 32) | in.color[t][4]);
        h = mix(h, in.color_gen[t]);
    }
    if (((in.depth_control & 3) || (in.render_control & 3)) && in.z_read_base) {
        h = mix(h, (static_cast<u64>(in.z_read_base) << 32) | in.depth_view);
        h = mix(h, (static_cast<u64>(in.z_info) << 32) | in.depth_gen);
    }
    return mix(h, in.render_control & 3);
}

int size_of(u32 draws) {
    int k = 0;
    while (k + 1 < kSizes && draws >= (2u << k)) ++k;
    return k;
}

void end_run(Split why) {
    Run& r = t_run;
    if (!r.draws) return;
    const int k = size_of(r.draws);
    g_n.runs.fetch_add(1, std::memory_order_relaxed);
    g_n.draws.fetch_add(r.draws, std::memory_order_relaxed);
    g_n.sizes[k].fetch_add(1, std::memory_order_relaxed);
    g_n.size_draws[k].fetch_add(r.draws, std::memory_order_relaxed);
    g_n.why[why].fetch_add(1, std::memory_order_relaxed);
    g_n.why_draws[why].fetch_add(r.draws, std::memory_order_relaxed);
    g_n.shape_draws[r.shape].fetch_add(r.draws, std::memory_order_relaxed);
    r.draws = 0;
}

int shape_of(const GpuDrawInputs& in) {
    int colours = 0;
    for (int t = 0; t < 8; ++t) colours += ((in.target_mask >> (4 * t)) & 0xf) && in.color[t][0];
    return colours == 0 ? 0 : colours == 1 ? 1 : colours < 4 ? 2 : 3;
}

double pct(u64 part, u64 whole) { return whole ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0; }

void log_counts(const Counts& c, const char* what) {
    std::string sizes, why, lag;
    static const char* const kSizeName[kSizes] = {"1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128+"};
    for (int k = 0; k < kSizes; ++k) {
        char b[96];
        std::snprintf(b, sizeof(b), "%s%s %.1f%% (%.1f%% of draws)", k ? ", " : "", kSizeName[k], pct(c.sizes[k], c.runs),
                      pct(c.size_draws[k], c.draws));
        sizes += b;
    }
    for (int k = 0; k < kSplits; ++k) {
        if (!c.why[k]) continue;
        char b[96];
        std::snprintf(b, sizeof(b), "%s%s %.1f%% (%.1f draws)", why.empty() ? "" : ", ", kSplitName[k], pct(c.why[k], c.runs),
                      static_cast<double>(c.why_draws[k]) / static_cast<double>(c.why[k]));
        why += b;
    }
    u64 lags = 0;
    for (u64 v : c.lag) lags += v;
    for (int k = 0; k < kLags; ++k) {
        char b[48];
        std::snprintf(b, sizeof(b), "%s%d%s %.1f%%", k ? ", " : "", k, k + 1 == kLags ? "+" : "", pct(c.lag[k], lags));
        lag += b;
    }
    const auto per = [](u64 a, u64 b) { return b ? static_cast<double>(a) / static_cast<double>(b) : 0.0; };
    host_log("list-resolve: runs (%s): %llu draws in %llu runs, %.1f draws a run; %llu stretches between in-order work, %.1f draws "
             "each; %llu GX command lists, %.1f draws each; draws a run: %s; runs ended by: %s; draws in runs writing no colour "
             "target %.1f%%, one %.1f%%, two or three %.1f%%, four or more %.1f%%; draws recorded on another GX context than the run's "
             "draw before %llu (%.1f%%); barrier tokens met %llu (no end of a run); flips from the GX call to the command processor: %s",
             what, static_cast<ull>(c.draws), static_cast<ull>(c.runs), per(c.draws, c.runs), static_cast<ull>(c.segments),
             per(c.draws, c.segments), static_cast<ull>(c.lists), per(c.draws, c.lists), sizes.c_str(), why.c_str(),
             pct(c.shape_draws[0], c.draws), pct(c.shape_draws[1], c.draws), pct(c.shape_draws[2], c.draws), pct(c.shape_draws[3], c.draws),
             static_cast<ull>(c.contexts), pct(c.contexts, c.draws), static_cast<ull>(c.barriers), lag.c_str());
}

// Compare mode: each ring's chunks taken for the recording under way.
struct RecordingChunks {
    u64 recording = ~0ull;
    std::vector<std::pair<u64, u64>> chunks;
};
RecordingChunks g_chunks[3];

void log_ctx() {
    const auto n = [](int k) { return g_batch_n[k].load(std::memory_order_relaxed); };
    const u64 draws = n(kCtxDraws);
    const auto per = [draws](u64 v) { return draws ? static_cast<double>(v) / static_cast<double>(draws) : 0.0; };
    host_log("list-resolve: draw context (chunks): %llu draws through it; chunks taken of the GX ring %llu (%llu KiB placed), the "
             "constant-buffer copies' ring %llu (%llu KiB copied), the vertex copies' ring %llu (%llu KiB copied); constant-buffer "
             "snapshots bound from its memo %llu; still taken outside it, a draw: params slots %.2f, descriptor sets %.3f, draw packets %.2f",
             static_cast<ull>(draws), static_cast<ull>(n(kCtxGxChunks)), static_cast<ull>(n(kCtxGxBytes) >> 10),
             static_cast<ull>(n(kCtxCbChunks)), static_cast<ull>(n(kCtxCbBytes) >> 10), static_cast<ull>(n(kCtxSnapChunks)),
             static_cast<ull>(n(kCtxSnapBytes) >> 10), static_cast<ull>(n(kCtxCbShared)), per(n(kCtxOutParams)), per(n(kCtxOutSets)),
             per(n(kCtxOutPackets)));
    if (g_draw_ctx == 2) {
        host_log("list-resolve: draw context compare: chunks checked %llu, overlapping another of their recording or leaving their ring "
                 "%llu; copies bound from a context's memo checked %llu, changed since their copy %llu",
                 static_cast<ull>(n(kCtxChunksChecked)), static_cast<ull>(n(kCtxChunksBad)), static_cast<ull>(n(kCtxMemoChecked)),
                 static_cast<ull>(n(kCtxMemoChanged)));
    }
}

void log_batch() {
    const auto n = [](int k) { return g_batch_n[k].load(std::memory_order_relaxed); };
    const u64 asked = n(kBatchTargetsAsked);
    host_log("list-resolve: run batching (%s): targets shared with the draw before for %llu of %llu draws (%.1f%%); not "
             "shared: something in order since %llu, the GPU mutex taken by another %llu, a target made or resized %llu, an HTILE fill "
             "noted or taken %llu, other targets or state %llu, a draw that does not share %llu; compared %llu, differed %llu; lite drains at a "
             "draw's first vertex record %llu",
             g_batch == 2 ? "compare" : "on", static_cast<ull>(n(kBatchTargetsShared)), static_cast<ull>(asked),
             pct(n(kBatchTargetsShared), asked), static_cast<ull>(n(kBatchMissSplit)), static_cast<ull>(n(kBatchMissLock)),
             static_cast<ull>(n(kBatchMissRemade)), static_cast<ull>(n(kBatchMissFill)), static_cast<ull>(n(kBatchMissInputs)),
             static_cast<ull>(n(kBatchMissKind)),
             static_cast<ull>(n(kBatchCompared)), static_cast<ull>(n(kBatchDiffered)), static_cast<ull>(n(kBatchDrains)));
}

}  // namespace

std::uint64_t run_serial() { return t_serial; }

void check_chunk(int ring, std::uint64_t recording, std::uint64_t lo, std::uint64_t hi, std::uint64_t ring_lo, std::uint64_t ring_hi) {
    RecordingChunks& r = g_chunks[ring];
    if (r.recording != recording) {
        r.recording = recording;
        r.chunks.clear();
    }
    count(kCtxChunksChecked);
    bool bad = lo < ring_lo || hi > ring_hi || hi <= lo;
    for (const auto& [a, b] : r.chunks) bad = bad || (lo < b && a < hi);
    if (bad) {
        count(kCtxChunksBad);
        static int logs = 0;  // under the GPU mutex
        if (logs++ < 8) {
            host_log("list-resolve: draw context compare: ring %d's chunk [0x%llx, 0x%llx) for recording %llu overlaps another of its "
                     "recording or leaves the ring [0x%llx, 0x%llx)",
                     ring, static_cast<ull>(lo), static_cast<ull>(hi), static_cast<ull>(recording), static_cast<ull>(ring_lo),
                     static_cast<ull>(ring_hi));
        }
    }
    r.chunks.emplace_back(lo, hi);
}

// Every count is made under the GPU mutex (draw_impl, the lite shadow): a
// plain add, not a locked one, at every draw.
void count(Count k, std::uint64_t n) {
    // The draw context's counts are its census's (BBHOST_DRAW_CONTEXT): a run
    // part's draws, which take a context of their own without it, count none.
    if (k >= kCtxDraws && k <= kCtxMemoChanged && !g_draw_ctx) return;
    g_batch_n[k].store(g_batch_n[k].load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

void split(Split why) {
    ++t_serial;
    t_run_left = 0;
    t_run_hold = 0;
    if (!g_mode) return;
    end_run(why);
    if (why != kSplitTargets && t_run.segment) {
        g_n.segments.fetch_add(1, std::memory_order_relaxed);
        t_run.segment = 0;
    }
}

void note_barrier() {
    if (g_mode) g_n.barriers.fetch_add(1, std::memory_order_relaxed);
}

void note_draw(const GpuDrawInputs& in, const GxDrawObjects& o) {
    Run& r = t_run;
    const u64 key = target_key(in);
    if (r.draws && key != r.targets) end_run(kSplitTargets);
    if (r.draws && o.ctx != r.ctx) g_n.contexts.fetch_add(1, std::memory_order_relaxed);
    if (o.ctx != r.list) {
        g_n.lists.fetch_add(1, std::memory_order_relaxed);
        r.list = o.ctx;
    }
    if (!r.draws) r.shape = shape_of(in);
    r.targets = key;
    r.ctx = o.ctx;
    ++r.draws;
    ++r.segment;
    const u64 flip = hle_video_flip_count();
    const u64 lag = flip > o.call_flip ? flip - o.call_flip : 0;
    g_n.lag[lag < kLags ? lag : kLags - 1].fetch_add(1, std::memory_order_relaxed);
    if (flip >= r.next_report) {
        if (r.next_report) report(false);
        r.next_report = flip + 1000;
    }
}

void report(bool at_exit) {
    if (at_exit && g_commit) log_commit();
    if (at_exit && g_batch) log_batch();
    if (at_exit && g_draw_ctx) log_ctx();
    if (!g_mode || !host_tooling()) return;
    if (g_batch && !at_exit) log_batch();
    static std::mutex mu;
    static Counts last;
    static u64 last_flip = 0;
    std::lock_guard<std::mutex> lk(mu);
    const Counts now = g_n.load();
    const u64 flip = hle_video_flip_count();
    if (at_exit) {
        log_counts(now, "the whole run");
        return;
    }
    Counts d;
    d.runs = now.runs - last.runs;
    d.draws = now.draws - last.draws;
    d.segments = now.segments - last.segments;
    d.contexts = now.contexts - last.contexts;
    d.lists = now.lists - last.lists;
    d.barriers = now.barriers - last.barriers;
    for (int k = 0; k < kSizes; ++k) {
        d.sizes[k] = now.sizes[k] - last.sizes[k];
        d.size_draws[k] = now.size_draws[k] - last.size_draws[k];
    }
    for (int k = 0; k < kSplits; ++k) {
        d.why[k] = now.why[k] - last.why[k];
        d.why_draws[k] = now.why_draws[k] - last.why_draws[k];
    }
    for (int k = 0; k < kShapes; ++k) d.shape_draws[k] = now.shape_draws[k] - last.shape_draws[k];
    for (int k = 0; k < kLags; ++k) d.lag[k] = now.lag[k] - last.lag[k];
    char what[96];
    const u64 flips = flip > last_flip ? flip - last_flip : 0;
    std::snprintf(what, sizeof(what), "flips %llu-%llu, %.1f runs and %.1f draws a flip", static_cast<ull>(last_flip), static_cast<ull>(flip),
                  flips ? static_cast<double>(d.runs) / static_cast<double>(flips) : 0.0,
                  flips ? static_cast<double>(d.draws) / static_cast<double>(flips) : 0.0);
    log_counts(d, what);
    last = now;
    last_flip = flip;
}

std::uint64_t targets_of(const GpuDrawInputs& in) { return target_key(in); }

// ---- BBHOST_RUN_COMMIT: the run workers ----

namespace {

int env_int(const char* name, int fallback, int lo, int hi) {
    const char* e = std::getenv(name);
    return e && *e ? std::clamp(std::atoi(e), lo, hi) : fallback;
}
const int g_commit_threads = env_int("BBHOST_RUN_COMMIT_THREADS", 3, 1, 16);
const std::uint32_t g_commit_min = static_cast<std::uint32_t>(env_int("BBHOST_RUN_COMMIT_MIN", 8, 2, 1 << 20));
const std::int64_t g_commit_spin_ns = std::int64_t{env_int("BBHOST_RUN_COMMIT_SPIN_US", 50, 0, 100000)} * 1000;

// A fork bumps `fork`; each worker does its part and counts itself in
// `joined`, then spins on `fork` for a while and sleeps on it after that,
// counted in `sleeping` so the fork wakes only when someone sleeps. The fork
// bumps the word and then reads the count, a worker counts itself and then
// compares the word (both sequentially consistent): one of the two sees the
// other.
struct Pool {
    alignas(64) std::atomic<u64> fork{0};
    void (*fn)(void*, int) = nullptr;  // the fork's parts (run_parts), set before it
    void* job = nullptr;
    int parts = 0;
    alignas(64) std::atomic<u32> joined{0};
    alignas(64) std::atomic<u32> sleeping{0};
};
Pool g_pool;

enum CommitCount : int {
    kForks,       // runs forked
    kForkDraws,   // their draws
    kForkNs,      // from the fork to the join, on the command processor
    kForkWakes,   // forks that woke a sleeping worker
    kLookNs,      // the look ahead's time
    kLooks,       // look aheads
    kShortRuns,   // runs shorter than BBHOST_RUN_COMMIT_MIN (not forked)
    kShortDraws,  // their draws
    kCommitCounts
};
std::atomic<u64> g_commit_n[kCommitCounts] = {};
std::atomic<u64> g_look_stop[kLookStops] = {};
void commit_count(CommitCount k, u64 n = 1) { g_commit_n[k].fetch_add(n, std::memory_order_relaxed); }

inline void cpu_pause() {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#endif
}

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// `seen`: the fork count before the fork that started the pool - a worker that
// read it once running could find that fork already counted and never join it.
void worker(int i, u64 seen) {
    char name[16];
    std::snprintf(name, sizeof(name), "bb-run%d", i);
    host_thread_set_name(name);
    host_thread_set_class(HostThreadClass::GpuFeed, name);  // with the command processor it serves
    for (;;) {
        u64 f = g_pool.fork.load(std::memory_order_acquire);
        if (f == seen) {
            const std::int64_t until = now_ns() + g_commit_spin_ns;
            for (int k = 0; f == seen; ++k) {
                cpu_pause();
                f = g_pool.fork.load(std::memory_order_acquire);
                if ((k & 63) == 63 && now_ns() > until) break;
            }
            if (f == seen) {
                g_pool.sleeping.fetch_add(1, std::memory_order_seq_cst);
                if (g_pool.fork.load(std::memory_order_seq_cst) == seen) host_futex_wait64(&g_pool.fork, seen);
                g_pool.sleeping.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }
        }
        seen = f;
        // Its part of the run (part i + 1: the command processor takes part 0);
        // none in the pricing step.
        if (g_pool.fn && i + 1 < g_pool.parts) g_pool.fn(g_pool.job, i + 1);
        g_pool.joined.fetch_add(1, std::memory_order_release);
    }
}

void start_pool() {
    static std::once_flag once;
    std::call_once(once, [] {
        const u64 at = g_pool.fork.load(std::memory_order_relaxed);  // the command processor's, before its first fork
        for (int i = 0; i < g_commit_threads; ++i) std::thread(worker, i, at).detach();
        host_log("list-resolve: run commit (fork): %d workers, runs of %u draws or more forked, %lld us of spin before a worker sleeps",
                 g_commit_threads, g_commit_min, static_cast<long long>(g_commit_spin_ns / 1000));
    });
}

void log_commit() {
    const auto n = [](int k) { return g_commit_n[k].load(std::memory_order_relaxed); };
    const u64 forks = n(kForks), looks = n(kLooks);
    const u64 draws = n(kForkDraws) + n(kShortDraws);
    host_log("list-resolve: run commit (fork): %llu runs forked holding %llu draws (%.1f%% of the draws looked at), %.2f us from fork to "
             "join a run, %.1f%% of forks woke a sleeping worker; %llu look aheads, %.2f us each; %llu runs not forked, too short or refused (%llu "
             "draws); looks stopped at an in-order packet %llu, other targets %llu, a slot not stored yet %llu, the end of the buffer %llu, "
             "a draw a part does not take %llu",
             static_cast<ull>(forks), static_cast<ull>(n(kForkDraws)), pct(n(kForkDraws), draws),
             forks ? static_cast<double>(n(kForkNs)) / 1000.0 / static_cast<double>(forks) : 0.0, pct(n(kForkWakes), forks),
             static_cast<ull>(looks), looks ? static_cast<double>(n(kLookNs)) / 1000.0 / static_cast<double>(looks) : 0.0,
             static_cast<ull>(n(kShortRuns)), static_cast<ull>(n(kShortDraws)),
             static_cast<ull>(g_look_stop[kLookSplit].load()), static_cast<ull>(g_look_stop[kLookTargets].load()),
             static_cast<ull>(g_look_stop[kLookNotReady].load()), static_cast<ull>(g_look_stop[kLookEnd].load()),
             static_cast<ull>(g_look_stop[kLookKind].load()));
}

}  // namespace

std::uint32_t run_left() { return t_run_left; }
void run_take() { --t_run_left; }

void run_fork(std::uint32_t following, LookStop stop, std::uint64_t look_ns) {
    commit_count(kLooks);
    commit_count(kLookNs, look_ns);
    g_look_stop[stop].fetch_add(1, std::memory_order_relaxed);
    t_run_left = following;
    if (following + 1 < g_commit_min) {
        commit_count(kShortRuns);
        commit_count(kShortDraws, following + 1);
        return;
    }
    const std::int64_t t0 = now_ns();
    run_parts(nullptr, nullptr, 0);
    commit_count(kForks);
    commit_count(kForkDraws, following + 1);
    commit_count(kForkNs, static_cast<u64>(now_ns() - t0));
    report_commit();
}

int run_workers() { return g_commit_threads; }
std::uint32_t run_min() { return g_commit_min; }

void run_parts(void (*fn)(void*, int), void* job, int parts) {
    start_pool();
    g_pool.fn = fn;
    g_pool.job = job;
    g_pool.parts = parts;
    g_pool.joined.store(0, std::memory_order_relaxed);
    g_pool.fork.fetch_add(1, std::memory_order_seq_cst);
    if (g_pool.sleeping.load(std::memory_order_seq_cst)) {
        host_futex_wake_all64(&g_pool.fork);
        commit_count(kForkWakes);
    }
    if (fn && parts > 0) fn(job, 0);  // the command processor's own part
    while (g_pool.joined.load(std::memory_order_acquire) < static_cast<u32>(g_commit_threads)) cpu_pause();
}

void run_set(std::uint32_t following) { t_run_left = following; }
void run_hold(std::uint32_t following) { t_run_hold = following; }
bool run_held() {
    if (!t_run_hold) return false;
    --t_run_hold;
    return true;
}

void report_commit() {
    static u64 next_report = 0;  // the command processor's
    const u64 flip = hle_video_flip_count();
    if (flip >= next_report) {
        if (next_report && host_tooling()) log_commit();
        next_report = flip + 1000;
    }
}

void note_look(LookStop stop, std::uint64_t look_ns) {
    commit_count(kLooks);
    commit_count(kLookNs, look_ns);
    g_look_stop[stop].fetch_add(1, std::memory_order_relaxed);
}

void note_commit(std::uint32_t draws, std::uint64_t ns, bool forked) {
    if (!forked) {
        commit_count(kShortRuns);
        commit_count(kShortDraws, draws);
        return;
    }
    commit_count(kForks);
    commit_count(kForkDraws, draws);
    commit_count(kForkNs, ns);
}

}  // namespace list_resolve
