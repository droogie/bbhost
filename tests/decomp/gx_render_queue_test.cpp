// The render queue's functions (decomp/gx/render_queue.cpp) against the
// game's own, in the loaded eboot:
//   - the partition (sub_26d11d0) on random queues - runs of keys over the
//     table's categories and sub-categories and random ones, sorted or not,
//     a range anywhere in the queue or past it, the parallel-for's limit
//     0-8, the task-stealing byte, a table already holding up to 70 chunks:
//     the same chunks written, every byte of the queue object, and the same
//     count returned - with the census's record and without;
//   - sub_12c7470's loop (ours inside the chunk job) with flags 1, 2 and 3
//     over random items and command streams (links, ends, every type): the
//     same calls of the queue's prepare and command slots in the same order,
//     with the same arguments and the prepare's state as the game's left it,
//     and the context's key fields the same;
//   - the chunk job (sub_2d4a400) whole, its GX calls stubbed: the same calls
//     in the same order and the same value returned, the census on and off;
//   - the cut: on the same random queues, the partition with the cut table
//     against the game's - every game chunk the cut's pieces back to back,
//     each with the chunk's entry fields - and, where the cut's chunks would
//     not fit the table, the game's chunks exactly; the cut table the game's
//     but its two entries;
//   - ranks by cost: only the ranks change, the dearest six 0, the next six 1;
//   - the parallel-for (sub_12c7810) against the game's, its job calls
//     stubbed: the same jobs pushed to the same participants in the same
//     order, the scene's task sort on and off; longest first, the dearest
//     chunk pushed first and the rest by cost, each to the participant its
//     rank and place give it;
// and each of the source's mutants caught.
#define DECOMP_GX_RENDER_QUEUE_MUTANTS
int g_rq_mutant = 0;
#include "decomp/gx/render_queue.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x26d11d0);
int g_failed = 0;
bool g_quiet = false;
long g_cut_seen = 0;

void fail(const char* what, int trial, const char* why) {
    if (++g_failed <= 30 && !g_quiet) std::printf("FAIL: %s case %d: %s\n", what, trial, why);
}

// ---- The partition ----

constexpr std::size_t kQueueBytes = 0x800;  // the queue object, room past a full table
constexpr int kMaxItems = 700;
alignas(64) u8 g_queue_a[kQueueBytes], g_queue_b[kQueueBytes], g_queue_start[kQueueBytes];
alignas(64) u8 g_items_mem[kMaxItems * 0x18];

u64 qa() { return reinterpret_cast<u64>(g_queue_a); }
u64 qb() { return reinterpret_cast<u64>(g_queue_b); }

// The table's own (category, sub-category) starts: where the lookup's edges are.
std::vector<std::pair<u32, u32>> g_starts;

u32 pick_cat_sub() {
    u32 cat, sub;
    const int r = g_rng.range(0, 9);
    if (r < 6 && !g_starts.empty()) {
        const auto& s = g_starts[static_cast<std::size_t>(g_rng.range(0, static_cast<int>(g_starts.size()) - 1))];
        cat = s.first;
        sub = s.second == 0xffffffffu ? static_cast<u32>(g_rng.range(0, 63)) : s.second + static_cast<u32>(g_rng.range(-1, 1));
        if (g_rng.chance(0.3) && cat == 0x3e) sub = static_cast<u32>(g_rng.range(0, 63));
    } else {
        cat = static_cast<u32>(g_rng.range(0, 63));
        sub = static_cast<u32>(g_rng.range(0, 63));
    }
    return ((cat & 0x3f) << 22) | ((sub & 0x3f) << 16);
}

void build_queue(int& count) {
    for (u8& b : g_queue_start) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_items_mem) b = static_cast<u8>(g_rng.u32());
    count = g_rng.chance(0.05) ? 0 : g_rng.range(1, kMaxItems);
    std::vector<u32> keys;
    while (static_cast<int>(keys.size()) < count) {
        const u32 cs = pick_cat_sub();
        const u32 top = static_cast<u32>(g_rng.range(0, 15)) << 28;
        const int len = g_rng.chance(0.3) ? g_rng.range(30, 140) : g_rng.range(1, 30);
        const bool same_top = g_rng.chance(0.8);
        for (int k = 0; k < len && static_cast<int>(keys.size()) < count; ++k)
            keys.push_back((same_top ? top : static_cast<u32>(g_rng.range(0, 15)) << 28) | cs | (g_rng.u32() & 0xffff));
    }
    if (g_rng.chance(0.7)) std::sort(keys.begin(), keys.end());
    for (int i = 0; i < count; ++i) store<u32>(reinterpret_cast<u64>(g_items_mem) + static_cast<u64>(i) * 0x18, keys[static_cast<std::size_t>(i)]);
    const u64 q = reinterpret_cast<u64>(g_queue_start);
    store<u64>(q + kItems, reinterpret_cast<u64>(g_items_mem));
    store<u32>(q + kCount, static_cast<u32>(count));
    const int r = g_rng.range(0, 9);
    store<u32>(q + kChunkCount, r < 6 ? 0u : r < 9 ? static_cast<u32>(g_rng.range(1, 62)) : static_cast<u32>(g_rng.range(63, 70)));
    store<u8>(q + kStealing, g_rng.chance(0.15) ? static_cast<u8>(g_rng.range(1, 255)) : 0);
}

using GamePartition = u32 (*)(u64, u32, u32, u32);

// One case: false when ours differs from the game's.
bool partition_case(int t, bool census, bool report) {
    int count = 0;
    build_queue(count);
    u32 first = 0, last = static_cast<u32>(count);
    if (g_rng.chance(0.25)) first = static_cast<u32>(g_rng.range(0, count + 3));
    if (g_rng.chance(0.25)) last = static_cast<u32>(g_rng.range(0, count + 20));
    if (g_rng.chance(0.03)) last = 0xffffffffu;
    const int mr = g_rng.range(0, 9);
    const u32 max_chunks = mr < 6 ? 6u : static_cast<u32>(g_rng.range(0, 8));
    std::memcpy(g_queue_a, g_queue_start, kQueueBytes);
    std::memcpy(g_queue_b, g_queue_start, kQueueBytes);
    const u32 theirs = eboot_kit::fn<GamePartition>(kPartition)(qa(), first, last, max_chunks);
    static Part part;
    part = Part{};
    const u32 ours = partition(qb(), first, last, max_chunks, census ? &part : nullptr);
    const bool same = ours == theirs && std::memcmp(g_queue_a, g_queue_b, kQueueBytes) == 0;
    if (!same && report) {
        char why[160];
        std::snprintf(why, sizeof why, "items %u-%u of %d, limit %u: ours %u chunks, the game's %u%s", first, last, count, max_chunks,
                      ours, theirs, ours == theirs ? " (the table differs)" : "");
        fail("partition", t, why);
    }
    (void)t;
    return same;
}

// The cut: ours with the cut table refines the game's; where it would not fit,
// it is the game's.
bool cut_case(int t, bool report) {
    int count = 0;
    build_queue(count);
    u32 first = 0, last = static_cast<u32>(count);
    if (g_rng.chance(0.2)) first = static_cast<u32>(g_rng.range(0, count + 3));
    if (g_rng.chance(0.2)) last = static_cast<u32>(g_rng.range(0, count + 20));
    const u32 max_chunks = g_rng.chance(0.8) ? 6u : static_cast<u32>(g_rng.range(0, 8));
    std::memcpy(g_queue_a, g_queue_start, kQueueBytes);
    std::memcpy(g_queue_b, g_queue_start, kQueueBytes);
    const u32 at = load<u32>(qa() + kChunkCount);
    const u32 theirs = eboot_kit::fn<GamePartition>(kPartition)(qa(), first, last, max_chunks);
    const u32 ours = cut_partition(qb(), first, last, max_chunks, nullptr, nullptr);
    // the game's chunks, from the queue object the game wrote
    // the game's own chunks byte for byte, or a refinement of them (and the
    // queue object past its table untouched)
    const bool same = std::memcmp(g_queue_a, g_queue_b, kQueueBytes) == 0;
    bool ok = same;
    if (!same) {
        ok = load<u32>(qb() + kChunkCount) == at + ours && load<u32>(qa() + kChunkCount) == at + theirs &&
             refines(qb(), at, ours, g_queue_a + kChunks, theirs) &&
             std::memcmp(g_queue_a + kChunkCount + 4, g_queue_b + kChunkCount + 4, kQueueBytes - kChunkCount - 4) == 0;
        if (ours == theirs && !load<u8>(qa() + kStealing)) ok = false;  // the same count is the game's chunks exactly
    }
    if (ours > theirs) ++g_cut_seen;
    if (!ok && report) {
        std::size_t off = 0;
        while (off < kQueueBytes && g_queue_a[off] == g_queue_b[off]) ++off;
        char why[200];
        std::snprintf(why, sizeof why, "items %u-%u of %d, limit %u, %u in the table: ours %u chunks, the game's %u; first byte apart +0x%zx "
                      "(chunk %zd field +0x%zx: ours %u, the game's %u)", first, last, count, max_chunks, at, ours, theirs, off,
                      off >= kChunks ? static_cast<std::ptrdiff_t>((off - kChunks) / kChunkSize) : -1,
                      off >= kChunks ? (off - kChunks) % kChunkSize : off, off < kQueueBytes ? g_queue_b[off] : 0,
                      off < kQueueBytes ? g_queue_a[off] : 0);
        fail("cut", t, why);
    }
    return ok;
}

// Ranks by cost: random costs by entry, one queue's chunks ranked.
bool ranks_case(int t, bool report) {
    int count = 0;
    build_queue(count);
    std::memcpy(g_queue_b, g_queue_start, kQueueBytes);
    store<u32>(qb() + kChunkCount, 0);
    s8 entries[64];
    const u32 made = partition(qb(), 0, static_cast<u32>(count), 6, nullptr, 0, nullptr, entries);
    for (u32 e = 0; e < kTableEntries; ++e) g_cost_per_item[e] = g_rng.chance(0.1) ? 0 : (g_rng.u32() & 0xfffff) + 1;
    std::memcpy(g_queue_a, g_queue_b, kQueueBytes);
    g_ranks = true;
    rank_by_cost(qb(), 0, made, entries);
    g_ranks = false;
    bool ok = true;
    // only the ranks moved
    for (u32 k = 0; k < made && ok; ++k) {
        const u64 a = qa() + kChunks + k * kChunkSize, b = qb() + kChunks + k * kChunkSize;
        ok = std::memcmp(reinterpret_cast<void*>(a), reinterpret_cast<void*>(b), 0x10) == 0 &&
             load<u8>(a + 0x14) == load<u8>(b + 0x14);
    }
    ok = ok && std::memcmp(g_queue_a + kChunks + made * kChunkSize, g_queue_b + kChunks + made * kChunkSize,
                           kQueueBytes - kChunks - made * kChunkSize) == 0;
    // the dearest six 0, the next six 1 (a chunk at the game's rank past 5 and under a twentieth of the dearest kept)
    if (ok && made >= 2 && made <= 64) {
        std::vector<std::pair<u64, u32>> by;
        u64 top = 0;
        for (u32 k = 0; k < made; ++k) {
            const u64 c = qb() + kChunks + k * kChunkSize;
            const u64 cost = chunk_cost(load<u32>(c + 4) - load<u32>(c), entries[k]);
            by.emplace_back(cost, k);
            top = std::max(top, cost);
        }
        std::stable_sort(by.begin(), by.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        for (u32 pos = 0; pos < made && ok && top; ++pos) {
            const u32 k = by[pos].second;
            const u32 was = load<u32>(qa() + kChunks + k * kChunkSize + 0x10), now = load<u32>(qb() + kChunks + k * kChunkSize + 0x10);
            if (was > 5 && by[pos].first * 20 < top)
                ok = now == was;
            else
                ok = now == std::min<u32>(5, pos / 6);
        }
    }
    if (!ok && report) fail("ranks", t, "a chunk's rank not the one its cost gives it, or more than the ranks changed");
    return ok;
}

// ---- The item loop and the chunk job ----

struct Call {
    int who;
    u64 a[5];
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_rec;

enum Who { kPrepare, kCommand, kMakeCtx, kCtxClose, kBegin, kEnd, kQueue, kOwner, kRelease };

alignas(64) u8 g_mem[0x20000];
u64 base() { return reinterpret_cast<u64>(g_mem); }
constexpr u64 kQ = 0x0, kQVt = 0x800, kCtx = 0x900, kCtxVt = 0xe00, kParent = 0xe80, kParentVt = 0xec0, kOwnerObj = 0xf80,
              kOwnerVt = 0xfc0, kMgr = 0x1000, kPerWorker = 0x1100, kFunctor = 0x1200, kRange = 0x1240, kItemsAt = 0x1400,
              kStreams = 0x4000;
constexpr u64 kCc = 0x7777000011112222ull;
int g_items_n = 0;

GUEST_ABI u64 rec_prepare(u64 q, u64 ctx, u64 state, u32 key, u32 pre) {
    // What the game's prepare keeps in the state: the key's upper half and the pass.
    const u32 k = load<u32>(state);
    const u64 pass = load<u64>(state + 0x10);
    g_rec.push_back({kPrepare, {q, ctx, (static_cast<u64>(k) << 32) | key, pass, pre}});
    if (pre == 0 && k != (key & 0xffff0000u)) {
        store<u32>(state, key & 0xffff0000u);
        store<u64>(state + 0x10, 0x5000ull + (key >> 16));
    }
    return 0;
}
GUEST_ABI u64 rec_command(u64 q, u64 ctx, u32 type, u64 payload, u64 size) {
    g_rec.push_back({kCommand, {q, ctx, type, payload, size}});
    // the key fields as the command sees them
    g_rec.push_back({kCommand, {load<u32>(ctx + kCtxKeyTop), load<u32>(ctx + kCtxCategory), load<u32>(ctx + kCtxSub),
                                load<u32>(ctx + kCtxLow), 0}});
    return 1;
}
GUEST_ABI u64 rec_make_ctx(u64 parent, u64 per_worker, u64 zero) {
    g_rec.push_back({kMakeCtx, {parent, per_worker, zero & 0xffffffffull, 0, 0}});
    return base() + kCtx;
}
GUEST_ABI u64 rec_ctx_close(u64 ctx) {
    g_rec.push_back({kCtxClose, {ctx, 0, 0, 0, 0}});
    return 0x31;
}
GUEST_ABI u64 rec_begin(u64 cc) {
    g_rec.push_back({kBegin, {cc, 0, 0, 0, 0}});
    return 0x41;
}
GUEST_ABI u64 rec_end(u64 cc) {
    g_rec.push_back({kEnd, {cc, 0, 0, 0, 0}});
    return 0x42;
}
GUEST_ABI u64 rec_queue(u64 gx, u64 cc, u32 key) {
    g_rec.push_back({kQueue, {gx, cc, key, 0, 0}});
    return 0x51;
}
GUEST_ABI u64 rec_owner(u64 ctx) {
    g_rec.push_back({kOwner, {ctx, 0, 0, 0, 0}});
    return base() + kOwnerObj;
}
GUEST_ABI u64 rec_release(u64 owner, u64 ctx) {
    g_rec.push_back({kRelease, {owner, ctx, 0, 0, 0}});
    return 0x61;
}

// A command stream: commands of every type, links to further blocks, an end.
u64 g_stream_at = 0;
u64 stream() {
    const u64 head = base() + kStreams + g_stream_at;
    u64 c = head;
    const int n = g_rng.range(0, 8);
    static const u8 kTypes[] = {1, 2, 3, 4, 6, 7, 8, 0xf8, 5, 9, 0x20, 0xf7, 0xf9};
    for (int k = 0; k < n; ++k) {
        if (g_rng.chance(0.12)) {  // a link to a block further on
            const u8 h = static_cast<u8>(g_rng.range(0, 6));
            store<u8>(c, 0xfe);
            store<u8>(c + 1, h);
            const u64 to = c + 64 + static_cast<u64>(g_rng.range(0, 3)) * 8;
            store<u64>(c + h + 4, to);
            c = to;
            continue;
        }
        const u8 h = static_cast<u8>(g_rng.range(0, 8));
        const u16 payload = static_cast<u16>(g_rng.chance(0.2) ? 0 : g_rng.range(1, 24));
        const u16 len = static_cast<u16>(4 + h + payload);
        store<u8>(c, kTypes[g_rng.range(0, static_cast<int>(sizeof kTypes) - 1)]);
        store<u8>(c + 1, h);
        store<u16>(c + 2, len);
        c += len;
    }
    store<u8>(c, 0xff);
    g_stream_at = (c - (base() + kStreams)) + 16;
    return head;
}

void build_job() {
    for (std::size_t i = 0; i < kStreams; ++i) g_mem[i] = static_cast<u8>(g_rng.u32());
    g_stream_at = 0;
    const u64 b = base();
    store<u64>(b + kQVt + 0x28, reinterpret_cast<u64>(&rec_command));
    store<u64>(b + kQVt + 0x30, reinterpret_cast<u64>(&rec_prepare));
    store<u64>(b + kQ, b + kQVt);
    store<u64>(b + kCtxVt, reinterpret_cast<u64>(&rec_ctx_close));
    store<u64>(b + kCtx, b + kCtxVt);
    store<u64>(b + kCtx + kCtxCommand, kCc);
    store<u64>(b + kParentVt + 0x80, reinterpret_cast<u64>(&rec_make_ctx));
    store<u64>(b + kParent, b + kParentVt);
    store<u64>(b + kOwnerVt + 0x70, reinterpret_cast<u64>(&rec_release));
    store<u64>(b + kOwnerObj, b + kOwnerVt);
    store<u64>(b + kMgr + 0x58, b + kPerWorker);
    for (int w = 0; w < 8; ++w) store<u64>(b + kPerWorker + 8ull * static_cast<u64>(w), 0xabc000ull + static_cast<u64>(w));
    store<u64>(b + kFunctor + 8, b + kParent);
    store<u64>(b + kFunctor + 0x10, b + kQ);
    g_items_n = g_rng.range(0, 40);
    u32 key = g_rng.u32();
    for (int i = 0; i < g_items_n; ++i) {
        const u64 it = b + kItemsAt + static_cast<u64>(i) * 0x18;
        if (g_rng.chance(0.4)) key = g_rng.u32();
        else key = (key & 0xffff0000u) | (g_rng.u32() & 0xffff);
        store<u32>(it, key);
        store<u64>(it + 8, stream());
        store<u8>(it + 0x11, g_rng.chance(0.3) ? static_cast<u8>(g_rng.range(1, 255)) : 0);
    }
    store<u64>(b + kQ + kItems, b + kItemsAt);
    store<u32>(b + kQ + kCount, static_cast<u32>(g_rng.chance(0.1) ? g_rng.range(0, g_items_n) : g_items_n));
    store<u64>(game_address(kGx), b + kMgr);
}

using GameLoop = u64 (*)(u64, u64, u32, u32, u32);
using GameJob = u64 (*)(u64, u64, u32);

bool loop_case(int t, bool report) {
    build_job();
    const u32 first = static_cast<u32>(g_rng.range(0, g_items_n + 2));
    const u32 last = g_rng.chance(0.2) ? static_cast<u32>(g_rng.range(0, g_items_n + 4)) : static_cast<u32>(g_items_n);
    const u32 flags = static_cast<u32>(g_rng.range(1, 3));
    static u8 start[0x20000];
    std::memcpy(start, g_mem, sizeof g_mem);
    g_rec.clear();
    eboot_kit::fn<GameLoop>(0x12c7470)(base() + kQ, base() + kCtx, first, last, flags);
    const std::vector<Call> game = g_rec;
    static u8 after[0x20000];
    std::memcpy(after, g_mem, sizeof g_mem);
    std::memcpy(g_mem, start, sizeof g_mem);
    g_rec.clear();
    static Item out[64];
    const bool census = g_rng.chance(0.5);
    run_items(base() + kQ, base() + kCtx, first, last, flags, census ? out : nullptr, 64);
    const bool same = g_rec == game && std::memcmp(after, g_mem, sizeof g_mem) == 0;
    if (!same && report) {
        char why[160];
        std::snprintf(why, sizeof why, "items %u-%u of %d, flags %u: %zu calls, the game's %zu%s", first, last, g_items_n, flags,
                      g_rec.size(), game.size(), g_rec == game ? " (memory differs)" : "");
        fail("item loop", t, why);
    }
    return same;
}

bool job_case(int t, bool report) {
    build_job();
    const u32 first = static_cast<u32>(g_rng.range(0, g_items_n));
    const u32 last = g_rng.chance(0.2) ? static_cast<u32>(g_rng.range(0, g_items_n + 4)) : static_cast<u32>(g_items_n);
    store<u32>(base() + kRange, first);
    store<u32>(base() + kRange + 4, last);
    store<u32>(base() + kRange + 8, last - first);
    const u32 worker = static_cast<u32>(g_rng.range(0, 5));
    static u8 start[0x20000];
    std::memcpy(start, g_mem, sizeof g_mem);
    g_rec.clear();
    const u64 game_ret = eboot_kit::fn<GameJob>(kChunkJob)(base() + kFunctor, base() + kRange, worker);
    const std::vector<Call> game = g_rec;
    static u8 after[0x20000];
    std::memcpy(after, g_mem, sizeof g_mem);
    std::memcpy(g_mem, start, sizeof g_mem);
    g_rec.clear();
    g_census = g_rng.chance(0.5);
    const u64 ret = render_queue_chunk_job(base() + kFunctor, base() + kRange, worker);
    const bool census = g_census;
    g_census = false;
    const bool same = ret == game_ret && g_rec == game && std::memcmp(after, g_mem, sizeof g_mem) == 0;
    if (!same && report) {
        char why[160];
        std::snprintf(why, sizeof why, "items %u-%u of %d, participant %u, census %d: %zu calls, the game's %zu%s", first, last,
                      g_items_n, worker, census, g_rec.size(), game.size(), ret != game_ret ? " (returns differ)" : "");
        fail("chunk job", t, why);
    }
    return same;
}

// ---- The parallel-for ----

struct Pushed {
    u32 first, last, grain, worker;
    u64 functor_vt, functor_ctx, functor_q;
    bool operator==(const Pushed&) const = default;
};
std::vector<Pushed> g_pushed_jobs;
std::vector<int> g_pf_calls;  // the order of the other calls: 1 wait first, 2 wake, 3 barrier
u64 g_job_ids = 0;

GUEST_ABI u64 rec_job_for(u64 functor, u64 range, u32 worker) {
    g_pushed_jobs.push_back(Pushed{load<u32>(range), load<u32>(range + 4), load<u32>(range + 8), worker, load<u64>(functor),
                                   load<u64>(functor + 8), load<u64>(functor + 0x10)});
    return 0x9000000ull + (++g_job_ids);
}
GUEST_ABI u64 rec_push(u64 jobs, u64 job, u32 worker) {
    const Pushed& last = g_pushed_jobs.back();
    if (job != 0x9000000ull + g_job_ids || worker != last.worker || jobs != base() + kMgr + 0x278) g_pf_calls.push_back(-1);
    return 0;
}
GUEST_ABI u64 rec_wait_first() {
    g_pf_calls.push_back(1);
    return 0;
}
GUEST_ABI u64 rec_wake(u64 jobs, u32 zero, u32 limit) {
    g_pf_calls.push_back(jobs == base() + kMgr + 0x278 && zero == 0 ? 2 : -2);
    g_pf_calls.push_back(static_cast<int>(limit));
    return 0;
}
GUEST_ABI u64 rec_barrier(u64 jobs) {
    g_pf_calls.push_back(jobs == base() + kMgr + 0x278 ? 3 : -3);
    return 0;
}

using GameParallelFor = u64 (*)(u64, u64, u32, u32, u32);

// One queue through the game's parallel-for and ours (the game's partition
// behind the queue's slot 3 for both). `push`: ours longest first.
bool parallel_for_case(int t, bool push, bool report) {
    int count = 0;
    build_queue(count);
    const u64 b = base();
    std::memcpy(g_queue_a, g_queue_start, kQueueBytes);
    store<u64>(b + kQVt + 0x18, game_address(kPartition));
    store<u64>(qa(), b + kQVt);
    store<u32>(qa() + kChunkCount, g_rng.chance(0.8) ? 0u : static_cast<u32>(g_rng.range(1, 20)));
    std::memcpy(g_queue_b, g_queue_a, kQueueBytes);
    store<u64>(game_address(kGx), b + kMgr);
    store<u32>(b + kMgr + 0x280, static_cast<u32>(g_rng.chance(0.8) ? 6 : g_rng.range(1, 8)));
    const bool sorted = g_rng.chance(0.85);
    store<u8>(game_address(kTaskSort), sorted ? 1 : 0);
    const u32 first = g_rng.chance(0.8) ? 0u : static_cast<u32>(g_rng.range(0, count));
    const u32 last = static_cast<u32>(count);
    const u32 limit = g_rng.chance(0.8) ? 6u : static_cast<u32>(g_rng.range(0, 8));
    const u64 ctx = 0x5555000000ull + g_rng.u32();
    g_pushed_jobs.clear();
    g_pf_calls.clear();
    const u32 theirs = static_cast<u32>(eboot_kit::fn<GameParallelFor>(kParallelFor)(qa(), ctx, first, last, limit));
    const std::vector<Pushed> game = g_pushed_jobs;
    const std::vector<int> game_calls = g_pf_calls;
    g_pushed_jobs.clear();
    g_pf_calls.clear();
    for (u32 e = 0; e < kTableEntries; ++e) g_cost_per_item[e] = push ? (g_rng.u32() & 0xfffff) + 1 : 0;
    // the push as the cut has it this frame (BBHOST_RQ_PUSH unset)
    g_push_mode = Push::WithCut;
    g_cut_on.store(push);
    const u32 ours = render_queue_parallel_for(qb(), ctx, first, last, limit);
    g_cut_on.store(true);
    bool ok = ours == theirs && g_pf_calls == game_calls && g_pushed_jobs.size() == game.size();
    for (const Pushed& p : g_pushed_jobs) ok = ok && p.functor_vt == game_address(kFunctorVtable) && p.functor_ctx == ctx && p.functor_q == qb();
    if (ok && !push) {
        for (std::size_t k = 0; k < game.size() && ok; ++k) {
            Pushed a = game[k], c = g_pushed_jobs[k];
            a.functor_q = c.functor_q = 0;  // the queues are two copies
            ok = a == c;
        }
        ok = ok && std::memcmp(g_queue_a + 8, g_queue_b + 8, kQueueBytes - 8) == 0;
    } else if (ok && push) {
        // the same chunks; the sorted range dearest first, the table's last chunk last; each to its participant
        const u32 total = load<u32>(qb() + kChunkCount), at = total - ours;
        const u32 n = load<u32>(qb() + kChunkCount) - at;
        std::vector<std::pair<u32, u32>> a, c;
        for (const Pushed& p : game) a.emplace_back(p.first, p.last);
        for (const Pushed& p : g_pushed_jobs) c.emplace_back(p.first, p.last);
        std::sort(a.begin(), a.end());
        std::sort(c.begin(), c.end());
        ok = a == c;
        const u32 parts = load<u32>(b + kMgr + 0x280);
        u64 prev = ~0ull;
        for (u32 k = 0; k < n && ok; ++k) {
            const Pushed& p = g_pushed_jobs[k];
            const u64 ch = qb() + kChunks + static_cast<u64>(at + k) * kChunkSize;
            ok = load<u32>(ch) == p.first && load<u32>(ch + 4) == p.last;
            const u32 rank = load<u32>(ch + 0x10);
            ok = ok && p.worker == ((sorted && rank <= 5) ? (at + k) % parts : 0u);
            if (sorted && k + 1 < n) {
                const u64 cost = p.last > p.first ? chunk_cost(p.last - p.first, entry_index(entry_of(load<u32>(reinterpret_cast<u64>(g_items_mem) + p.first * 0x18ull), game_address(kTable)), game_address(kTable))) : 0;
                ok = ok && cost <= prev;
                prev = cost;
            }
        }
        if (ok && n >= 1) {  // the last chunk: the partition's last, pushed last
            const Pushed& p = g_pushed_jobs.back();
            ok = p.first == game.back().first && p.last == game.back().last;
        }
    }
    if (!ok && report) {
        char why[160];
        std::snprintf(why, sizeof why, "items %u-%u of %d, sort %d, push %d: %zu jobs (the game's %zu), %u chunks (%u)", first, last, count,
                      sorted, push, g_pushed_jobs.size(), game.size(), ours, theirs);
        fail("parallel-for", t, why);
    }
    return ok;
}

// ---- The cut's choice ----

// BBHOST_RQ_SPLIT unset: the cut, and the push and the costs' learning with
// it, on while the frame rate's limit is above 60, decided each flip; forced
// on or off, the limit changes nothing.
bool choice_case() {
    const struct {
        float fps;
        bool on;
    } kRates[] = {{30, false}, {60, false}, {120, true}, {59.94f, false}, {60.01f, false}, {75, true},
                  {90, true},  {144, true}, {240, true}, {60, false},     {120, true}};
    bool ok = true;
    u32 flip = 1;
    g_split_mode = Split::Auto;
    g_push_mode = Push::WithCut;
    for (const auto& r : kRates) {
        render_queue_flip(flip++, r.fps);
        ok = ok && g_cut_on.load() == r.on && pushing() == r.on && learning() == r.on && cut_this_flip() == r.on &&
             g_said_cut.load() == (r.on ? 1 : 0);
    }
    g_push_mode = Push::Off;
    ok = ok && !pushing() && !learning();
    g_push_mode = Push::On;
    render_queue_flip(flip++, 60);
    ok = ok && !g_cut_on.load() && pushing() && learning();
    g_split_mode = Split::Off;
    g_push_mode = Push::WithCut;
    g_cut_on.store(false);
    render_queue_flip(flip++, 120);
    ok = ok && !g_cut_on.load() && !pushing() && !cut_this_flip();
    g_split_mode = Split::On;
    g_cut_on.store(true);
    render_queue_flip(flip++, 60);
    ok = ok && g_cut_on.load() && pushing() && cut_this_flip();
    return ok;
}

// ---- The segment list's note ----

GUEST_ABI u64 rec_append(u64 list, u64 record, u32 key) {
    g_rec.push_back({kQueue, {list, record, key, 0, 0}});
    return 0x71;
}

bool append_case() {
    alignas(16) static u8 list[0x100];
    store<u32>(reinterpret_cast<u64>(list) + 0x40, 0x0fc12345u);
    g_game_append = reinterpret_cast<void*>(&rec_append);
    g_rec.clear();
    g_census = true;
    const u64 before = g_append_pos.load();
    const u64 r = render_queue_list_append(reinterpret_cast<u64>(list), 0x1234, 0xffffffffu);
    g_census = false;
    const Append& a = g_appends[before & (kAppendRing - 1)];
    // 0x0fc12345: its category 0x3f stepped wraps to 0 (as the game's 0x25b4a22 computes it)
    return r == 0x71 && g_rec.size() == 1 && g_rec[0].a[0] == reinterpret_cast<u64>(list) && g_rec[0].a[1] == 0x1234 &&
           g_rec[0].a[2] == 0xffffffffu && a.key == 0xffffffffu && a.next == 0x00012345u && seen(const_cast<u32&>(a.done)) == 1;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gx_render_queue_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 40000;
    // the entries' instructions
    const struct {
        u64 bn;
        const u8* bytes;
        std::size_t n;
    } kEntries[] = {{kPartition, kPartitionEntry, sizeof kPartitionEntry},
                    {kChunkJob, kChunkJobEntry, sizeof kChunkJobEntry},
                    {kListAppend, kListAppendEntry, sizeof kListAppendEntry}};
    for (const auto& e : kEntries) {
        const u8* at = eboot_kit::file_at("gx_render_queue_test", e.bn);
        if (!at || std::memcmp(at, e.bytes, e.n) != 0) fail("entry", static_cast<int>(e.bn), "the entry's instructions");
    }
    for (u32 j = 0; j < kTableEntries; ++j) {
        const u64 e = game_address(kTable) + j * kTableEntry;
        g_starts.emplace_back(load<u32>(e), load<u32>(e + 4));
    }
    eboot_kit::stub(kBeginRecord, reinterpret_cast<const void*>(&rec_begin));
    eboot_kit::stub(kEndRecord, reinterpret_cast<const void*>(&rec_end));
    eboot_kit::stub(kQueueContext, reinterpret_cast<const void*>(&rec_queue));
    eboot_kit::stub(kOwnerOf, reinterpret_cast<const void*>(&rec_owner));
    eboot_kit::stub(kJobFor, reinterpret_cast<const void*>(&rec_job_for));
    eboot_kit::stub(kPushJob, reinterpret_cast<const void*>(&rec_push));
    eboot_kit::stub(kWaitFirst, reinterpret_cast<const void*>(&rec_wait_first));
    eboot_kit::stub(kWake, reinterpret_cast<const void*>(&rec_wake));
    eboot_kit::stub(kBarrier, reinterpret_cast<const void*>(&rec_barrier));
    // the cut table, from the game's; its two entries the PC build's
    make_cut_table(nullptr);
    if (!g_cut_ready) fail("cut table", 0, "the game's table not the one the limits were written for");
    for (u32 k = 0; k < kTableWords; ++k) {
        const u32 game = load<u32>(game_address(kTable) + 4ull * k);
        const bool changed = k == 17 * 9 + 3 || k == 17 * 9 + 4 || k == 19 * 9 + 3 || k == 19 * 9 + 4;
        if ((g_cut_table[k] != game) != changed) fail("cut table", static_cast<int>(k), "not the game's but for entries 17 and 19");
    }
    if (g_cut_table[17 * 9 + 3] != 5 || g_cut_table[17 * 9 + 4] != 10 || g_cut_table[19 * 9 + 3] != 3 || g_cut_table[19 * 9 + 4] != 10)
        fail("cut table", 0, "entries 17 and 19 not at the PC build's limits");
    g_split_mode = Split::On;
    g_cut_on.store(true);
    // the census's rings, for the job's and the note's census
    g_items = static_cast<Item*>(std::calloc(kItemRing, sizeof(Item)));
    g_chunk_ring = static_cast<Chunk*>(std::calloc(kChunkRing, sizeof(Chunk)));
    g_appends = static_cast<Append*>(std::calloc(kAppendRing, sizeof(Append)));

    long n_part = 0, n_loop = 0, n_job = 0, split = 0;
    for (int t = 0; t < cases; ++t) {
        partition_case(t, (t & 1) != 0, true);
        ++n_part;
        if (load<u32>(qa() + kChunkCount) > load<u32>(reinterpret_cast<u64>(g_queue_start) + kChunkCount) + 1) ++split;
    }
    for (int t = 0; t < cases / 4; ++t, ++n_loop) loop_case(t, true);
    for (int t = 0; t < cases / 4; ++t, ++n_job) job_case(t, true);
    long n_cut = 0, n_ranks = 0, n_pf = 0;
    for (int t = 0; t < cases; ++t, ++n_cut) cut_case(t, true);
    for (int t = 0; t < cases / 4; ++t, ++n_ranks) ranks_case(t, true);
    for (int t = 0; t < cases / 4; ++t, ++n_pf) parallel_for_case(t, (t & 1) != 0, true);
    for (u32 e = 0; e < kTableEntries; ++e) g_cost_per_item[e] = 0;
    if (!append_case()) fail("segment note", 0, "the list's append not passed on as given, or its key not noted");
    if (!choice_case()) fail("the cut's choice", 0, "not on above 60 fps alone, or the push not with it");
    g_split_mode = Split::On;
    g_cut_on.store(true);

    // Every mutant caught.
    const int kPartitionMutants[] = {1, 2, 3, 4, 10, 11, 12};
    const int kLoopMutants[] = {5, 6, 7};
    const int kJobMutants[] = {8, 9};
    const int kCutMutants[] = {13};
    const int kRankMutants[] = {14};
    const int kPushMutants[] = {15};
    int caught = 0, mutants = 0;
    g_quiet = true;
    const auto catches = [&](int m, auto&& one, int tries) {
        g_rq_mutant = m;
        bool hit = false;
        for (int t = 0; t < tries && !hit; ++t) hit = !one(t);
        g_rq_mutant = 0;
        ++mutants;
        if (hit) ++caught;
        else std::printf("FAIL: mutant %d not caught\n", m), ++g_failed;
    };
    for (int m : kPartitionMutants) catches(m, [](int t) { return partition_case(t, false, false); }, 200000);
    for (int m : kLoopMutants) catches(m, [](int t) { return loop_case(t, false); }, 20000);
    for (int m : kJobMutants) catches(m, [](int t) { return job_case(t, false); }, 20000);
    for (int m : kCutMutants) catches(m, [](int t) { return cut_case(t, false); }, 200000);
    for (int m : kRankMutants) catches(m, [](int t) { return ranks_case(t, false); }, 20000);
    for (int m : kPushMutants) catches(m, [](int t) { return parallel_for_case(t, true, false); }, 20000);
    g_quiet = false;

    if (g_failed) {
        std::printf("gx_render_queue_test: %d failures\n", g_failed);
        return 1;
    }
    std::printf("gx_render_queue_test: ok - partition %ld cases (%ld cut a run), item loop %ld, chunk job %ld; the cut %ld (%ld cut further "
                "than the game), ranks %ld, parallel-for %ld; %d of %d mutants caught\n",
                n_part, split, n_loop, n_job, n_cut, g_cut_seen, n_ranks, n_pf, caught, mutants);
    return 0;
}
