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
//   - the scene categories' counts (BBHOST_SCENE_CATEGORIES=1) against a map
//     of the same items, a row a queue and the rest dropped, from one thread
//     and four;
//   - the task-stealing jobs cut finer late (BBHOST_RQ_GUIDED): the game's own
//     splits inside ours on random jobs - a pop's piece and a steal's half by
//     the rule, the job's grain kept, the half's and the run's vtables - and
//     random pops and steals running every item of a few jobs once; and the
//     parallel-for giving our vtable to the task-stealing jobs alone;
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
    // the tail rule's inputs: the costs known or not (then the items decide)
    const bool costs = g_rng.chance(0.7);
    g_even_share.store(costs ? (g_rng.u32() & 0x3fffff) + 1 : 0);
    for (u32 e = 0; e < kTableEntries; ++e) g_cost_per_item[e] = costs && g_rng.chance(0.9) ? (g_rng.u32() & 0x3ffffff) + 1 : 0;
    const u32 theirs = eboot_kit::fn<GamePartition>(kPartition)(qa(), first, last, max_chunks);
    const u64 fallbacks = g_cut_fallbacks.load();
    u32 runs = 0xdead;
    const u32 ours = cut_partition(qb(), first, last, max_chunks, nullptr, nullptr, &runs);
    const bool fell_back = g_cut_fallbacks.load() != fallbacks;
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
    // The runs of the entries the cut changes: each is one of the game's chunks
    // (it gives them one); ours cuts exactly those its rule calls a tail, as far
    // as the cut table's limits allow, and counts them.
    u32 cut_seen = 0;
    bool rule = true;
    if (ok && !same) {
        const u64 table = game_address(kTable);
        u32 o = 0;
        for (u32 g = 0; g < theirs && rule; ++g) {
            const u64 gc = qa() + kChunks + static_cast<u64>(at + g) * kChunkSize;
            const u32 gf = load<u32>(gc), gl = load<u32>(gc + 4);
            u32 pieces = 0, pos = gf;
            do {
                const u64 c = qb() + kChunks + static_cast<u64>(at + o) * kChunkSize;
                pos = load<u32>(c + 4);
                ++pieces;
                ++o;
            } while (pos < gl && o < ours);
            const s8 e = gl > gf ? entry_index(entry_of(load<u32>(reinterpret_cast<u64>(g_items_mem) + gf * 0x18ull), table), table) : -1;
            if (!cut_entry(e)) {
                rule = pieces == 1;
                continue;
            }
            const u32 n = gl - gf;
            const bool tail = g_even_share.load() && chunk_cost(n, e) ? chunk_cost(n, e) >= g_even_share.load() : n >= kTailItems;
            const u32* ce = &g_cut_table[static_cast<u32>(e) * 9];
            u32 want = (n + ce[4] - 1) / ce[4];
            if (want > ce[3]) want = ce[3];
            if (want == 0) want = 1;
            if (want > n / ce[5]) want = n / ce[5];
            const bool can = tail && want >= 2 && max_chunks >= 2 && !load<u8>(qa() + kStealing) && g + 1 < theirs && !fell_back;
            if (pieces > 1) ++cut_seen;
            rule = pieces > 1 ? tail && pieces == want : !can;
        }
    }
    if (ok && (same ? runs != 0 : !rule || runs != cut_seen)) ok = false;
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

// The queue's slot 3 for both parallel-fors: the game's partition, saying
// (as ours does when it cuts a run) that it cut when g_stub_cut is set.
bool g_stub_cut = false;
GUEST_ABI u32 stub_partition(u64 q, u32 first, u32 last, u32 limit) {
    g_pf_cut = g_stub_cut;
    return eboot_kit::fn<u32 (*)(u64, u32, u32, u32)>(kPartition)(q, first, last, limit);
}

// One queue through the game's parallel-for and ours. Mostly the cut on and
// the push with it (BBHOST_RQ_PUSH unset); `cut`: the partition cut a run,
// after which ours pushes longest first and otherwise as the game does.
bool parallel_for_case(int t, bool cut, bool report) {
    int count = 0;
    build_queue(count);
    const u64 b = base();
    std::memcpy(g_queue_a, g_queue_start, kQueueBytes);
    store<u64>(b + kQVt + 0x18, reinterpret_cast<u64>(&stub_partition));
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
    const bool known = g_rng.chance(0.9);
    for (u32 e = 0; e < kTableEntries; ++e) g_cost_per_item[e] = known ? (g_rng.u32() & 0xfffff) + 1 : 0;
    const int m = g_rng.range(0, 9);
    g_push_mode = m < 7 ? Push::WithCut : m < 8 ? Push::On : Push::Off;
    g_cut_on.store(m < 6 || g_rng.chance(0.5));
    g_stub_cut = cut;
    // longest first when forced, or with the cut on after a cut
    const bool push = known && (g_push_mode == Push::On || (g_push_mode == Push::WithCut && g_cut_on.load() && cut));
    const u32 ours = render_queue_parallel_for(qb(), ctx, first, last, limit);
    g_push_mode = Push::WithCut;
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
        std::snprintf(why, sizeof why, "items %u-%u of %d, sort %d, cut %d, mode %d, push %d: %zu jobs (the game's %zu), %u chunks (%u)",
                      first, last, count, sorted, cut, m, push, g_pushed_jobs.size(), game.size(), ours, theirs);
        fail("parallel-for", t, why);
    }
    return ok;
}

// ---- The guided split (BBHOST_RQ_GUIDED) ----
//
// GX's range job (vtable 0x575fbf0): the game's own pop and steal splits run
// inside ours, their job pool (sub_25b7f60) stubbed by fresh objects carrying
// the pool's vtable, the participant they read from the guest's TLS a test
// object. Ours must cut as the rule says - a pop max(M, min(G, ceil(n / K)))
// items off the front of a job of n >= twice that (G its grain, the game's),
// a steal the upper half of a job of n >= 2 M - keep the job's grain G, give
// a stolen half our vtable and G, and give the pool's vtable back before the
// game's run; and over random pops and steals every item of a job is run
// once, in pieces of at least M but for a job shorter than M.
constexpr u64 kPoolAlloc = 0x25b7f60;  // (the participant's pool): a job object
constexpr u64 kPools = 0x5aa6c28;      // the job pools' holder: [0] the pools, 0x60 bytes a participant
constexpr u64 kParticipantTls = 0x57e4658;  // the current participant's TLS offset (-0x698 in the file)
constexpr int kJobObjs = 4096;
alignas(64) u8 g_job_objs[kJobObjs * 0x28];
alignas(64) u8 g_participant[0x530];
alignas(64) u8 g_pools[8 * 0x60];
u64 g_pools_holder[2];
int g_job_next = 0;
u64 g_pool_seen = 0;

GUEST_ABI u64 stub_pool_alloc(u64 pool) {
    g_pool_seen = pool;
    const u64 j = reinterpret_cast<u64>(g_job_objs) + static_cast<u64>(g_job_next++ % kJobObjs) * 0x28;
    std::memset(reinterpret_cast<void*>(j), 0, 0x28);
    store<u64>(j, game_address(kJobVtable));
    return j;
}
u64 g_run_vt = 0, g_run_job = 0;
u32 g_run_participant = 0;
GUEST_ABI u64 rec_job_run(u64 job, u32 participant) {
    g_run_job = job;
    g_run_vt = load<u64>(job);
    g_run_participant = participant;
    return 0x1234;
}

u64 guided_job(u32 f, u32 l, u32 m) {
    const u64 j = stub_pool_alloc(0);
    store<u64>(j, host_address(g_rq_job_vt));
    store<u64>(j + 8, 0x4444000000ull);
    store<u32>(j + 0x10, f);
    store<u32>(j + 0x14, l);
    store<u32>(j + 0x18, m);
    return j;
}

void guided_setup() {
    g_rq_job_vt[3] = load<u64>(game_address(kJobVtable) + 0x18);
    g_rq_job_vt[2] = host_address(reinterpret_cast<void*>(&rq_job_split_steal));
    g_rq_job_vt[1] = host_address(reinterpret_cast<void*>(&rq_job_split_pop));
    g_rq_job_vt[0] = host_address(reinterpret_cast<void*>(&rq_job_run));
    store<u32>(reinterpret_cast<u64>(g_participant) + 0xa0, 3);
    store<u64>(reinterpret_cast<u64>(eboot_kit::tcb()) + load<u64>(game_address(kParticipantTls)), reinterpret_cast<u64>(g_participant));
    g_pools_holder[0] = reinterpret_cast<u64>(g_pools);
    g_pools_holder[1] = 6;
    store<u64>(game_address(kPools), reinterpret_cast<u64>(g_pools_holder));
}

// One pop's split and one steal's on a random job, against the rule.
bool guided_split_case(int t, bool report) {
    const u32 k = static_cast<u32>(g_rng.range(2, 8)), m = static_cast<u32>(g_rng.range(1, 12));
    g_guided_k = k;
    g_guided_least = m;
    const u32 g = static_cast<u32>(g_rng.chance(0.7) ? g_rng.range(10, 40) : g_rng.range(1, 12));
    const u32 f = static_cast<u32>(g_rng.range(0, 1000));
    const u32 n = static_cast<u32>(g_rng.chance(0.1) ? g_rng.range(0, 3) : g_rng.range(0, 300));
    const u32 l = f + n;
    bool ok = true;
    char why[256] = "";
    {
        const u64 j = guided_job(f, l, g);
        u64 out = 0xdead;
        const u64 r = rq_job_split_pop(j, reinterpret_cast<u64>(&out));
        u32 take = (n + k - 1) / k;
        if (take > g) take = g;
        if (take < m) take = m;
        const bool split = n >= 2 * take;
        if ((r & 0xff) != (split ? 1u : 0u) || load<u32>(j + 0x18) != g || load<u64>(j) != host_address(g_rq_job_vt)) ok = false;
        if (ok && split)
            ok = out && load<u32>(out + 0x10) == f && load<u32>(out + 0x14) == f + take && load<u32>(out + 0x18) == take &&
                 load<u64>(out + 8) == load<u64>(j + 8) && load<u64>(out) == game_address(kJobVtable) && load<u32>(j + 0x10) == f + take &&
                 load<u32>(j + 0x14) == l && g_pool_seen == reinterpret_cast<u64>(g_pools) + 3 * 0x60;
        if (ok && !split) ok = out == 0 && load<u32>(j + 0x10) == f && load<u32>(j + 0x14) == l;
        if (!ok)
            std::snprintf(why, sizeof why, "pop: %u items from %u, K %u M %u G %u: split %llu (piece %u-%u grain %u, vt %d, job %u-%u, pool %d)", n, f,
                          k, m, g, static_cast<ull>(r & 0xff), out && out != 0xdead ? load<u32>(out + 0x10) : 0, out && out != 0xdead ? load<u32>(out + 0x14) : 0,
                          out && out != 0xdead ? load<u32>(out + 0x18) : 0, out && out != 0xdead ? load<u64>(out) == game_address(kJobVtable) : -1,
                          load<u32>(j + 0x10), load<u32>(j + 0x14), g_pool_seen == reinterpret_cast<u64>(g_pools) + 3 * 0x60);
    }
    if (ok) {
        const u64 j = guided_job(f, l, g);
        u64 out = 0;  // the steal zeroes it before the split (sub_1247d20); the pop's split does itself
        const u64 r = rq_job_split_steal(j, reinterpret_cast<u64>(&out));
        const bool split = n >= 2 * m;
        const s32 sum = static_cast<s32>(f + l);
        const u32 mid = static_cast<u32>((sum + static_cast<s32>(static_cast<u32>(sum) >> 31)) >> 1);
        if ((r & 0xff) != (split ? 1u : 0u) || load<u32>(j + 0x18) != g) ok = false;
        if (ok && split)
            ok = out && load<u32>(out + 0x10) == mid && load<u32>(out + 0x14) == l && load<u32>(out + 0x18) == g &&
                 load<u64>(out) == host_address(g_rq_job_vt) && load<u32>(j + 0x10) == f && load<u32>(j + 0x14) == mid;
        if (ok && !split) ok = out == 0 && load<u32>(j + 0x10) == f && load<u32>(j + 0x14) == l;
        if (!ok) std::snprintf(why, sizeof why, "steal: %u items from %u, M %u: split %llu", n, f, m, static_cast<ull>(r & 0xff));
    }
    if (ok) {
        const u64 j = guided_job(f, l, g);
        const u32 who = static_cast<u32>(g_rng.range(0, 5));
        const u64 r = rq_job_run(j, who);
        ok = r == 0x1234 && g_run_job == j && g_run_vt == game_address(kJobVtable) && g_run_participant == who;
        if (!ok) std::snprintf(why, sizeof why, "run: the vtable %llx at the game's run", static_cast<ull>(g_run_vt));
    }
    if (!ok && report) fail("guided split", t, why);
    return ok;
}

// Random pops and steals over a few jobs, as the deques make them (a pop
// splits the front job or takes it whole; a steal from a deque of one job
// splits it or takes it whole), until every piece has run: each item once,
// every piece but a job's last at least M items, every job run with the
// pool's vtable.
bool guided_cover_case(int t, bool report) {
    const u32 k = static_cast<u32>(g_rng.range(2, 8)), m = static_cast<u32>(g_rng.range(1, 10));
    g_guided_k = k;
    g_guided_least = m;
    struct Q {
        std::vector<u64> jobs;
    };
    std::vector<Q> deques(static_cast<std::size_t>(g_rng.range(1, 6)));
    std::vector<int> seen;
    u32 at = 0;
    const int njobs = g_rng.range(1, 6);
    for (int i = 0; i < njobs; ++i) {
        const u32 n = static_cast<u32>(g_rng.range(0, 200));
        const u32 g = std::max<u32>(10, n / static_cast<u32>(g_rng.range(1, 12)));  // the partition's grain
        deques[static_cast<std::size_t>(g_rng.range(0, static_cast<int>(deques.size()) - 1))].jobs.push_back(guided_job(at, at + n, g));
        at += n;
    }
    seen.assign(at, 0);
    bool ok = true;
    int small = 0;
    for (int step = 0; step < 100000 && ok; ++step) {
        std::vector<std::size_t> busy;
        for (std::size_t d = 0; d < deques.size(); ++d)
            if (!deques[d].jobs.empty()) busy.push_back(d);
        if (busy.empty()) break;
        Q& q = deques[busy[static_cast<std::size_t>(g_rng.range(0, static_cast<int>(busy.size()) - 1))]];
        u64 piece = 0;
        if (g_rng.chance(0.6) || deques.size() == 1) {
            // a pop: the front job's split (its vtable +8), else the job whole
            const u64 j = q.jobs.front();
            u64 out = 0;
            const u64 r = reinterpret_cast<SplitFn>(load<u64>(load<u64>(j) + 8))(j, reinterpret_cast<u64>(&out));
            if (r & 0xff) piece = out;
            else {
                piece = j;
                q.jobs.erase(q.jobs.begin());
            }
        } else {
            // a steal into an empty deque: a lone job's split (+0x10), else jobs whole from the front
            std::size_t to = 0;
            while (to < deques.size() && !deques[to].jobs.empty()) ++to;
            if (to == deques.size()) continue;
            if (q.jobs.size() == 1) {
                const u64 j = q.jobs.front();
                u64 out = 0;
                const u64 r = reinterpret_cast<SplitFn>(load<u64>(load<u64>(j) + 0x10))(j, reinterpret_cast<u64>(&out));
                if (r & 0xff) deques[to].jobs.push_back(out);
                else {
                    deques[to].jobs.push_back(j);
                    q.jobs.clear();
                }
            } else {
                const std::size_t take = (q.jobs.size() + 1) / 2;
                for (std::size_t i = 0; i < take; ++i) deques[to].jobs.push_back(q.jobs[i]);
                q.jobs.erase(q.jobs.begin(), q.jobs.begin() + static_cast<long>(take));
            }
            continue;
        }
        // the piece runs: its vtable's slot 0 (ours or the game's, which the test stands in for)
        const u32 pf = load<u32>(piece + 0x10), pl = load<u32>(piece + 0x14);
        const u64 vt = load<u64>(piece);
        if (vt == host_address(g_rq_job_vt)) rq_job_run(piece, 0);
        else if (vt == game_address(kJobVtable)) rec_job_run(piece, 0);
        else ok = false;
        if (g_run_vt != game_address(kJobVtable)) ok = false;
        if (pl - pf < m) ++small;
        for (u32 i = pf; i < pl && ok; ++i) ok = i < at && ++seen[i] == 1;
    }
    for (u32 i = 0; i < at && ok; ++i) ok = seen[i] == 1;
    // a piece under M items is only ever a whole job shorter than 2 M - at most one a job
    ok = ok && small <= njobs;
    if (!ok && report) {
        char why[96];
        std::snprintf(why, sizeof why, "%d jobs, %u items, K %u M %u: an item not run once, or %d short pieces", njobs, at, k, m, small);
        fail("guided cover", t, why);
    }
    return ok;
}

// The parallel-for with guided on: the jobs of task-stealing chunks (their
// size field under their span) get grain M and our vtable, the rest as the
// game's; the same jobs to the same participants otherwise.
std::vector<std::pair<u64, u32>> g_guided_pushed;  // job, its grain when made
GUEST_ABI u64 rec_job_for_mem(u64 functor, u64 range, u32 worker) {
    const u64 j = stub_pool_alloc(0);
    store<u64>(j + 8, functor);
    store<u64>(j + 0x10, load<u64>(range));
    store<u32>(j + 0x18, load<u32>(range + 8));
    g_pushed_jobs.push_back(Pushed{load<u32>(range), load<u32>(range + 4), load<u32>(range + 8), worker, load<u64>(functor),
                                   load<u64>(functor + 8), load<u64>(functor + 0x10)});
    return j;
}
GUEST_ABI u64 rec_push_mem(u64, u64 job, u32) {
    g_guided_pushed.emplace_back(job, load<u32>(job + 0x18));
    return 0;
}

bool guided_pf_case(int t, bool report) {
    int count = 0;
    build_queue(count);
    store<u8>(reinterpret_cast<u64>(g_queue_start) + kStealing, 1);
    const u64 b = base();
    std::memcpy(g_queue_a, g_queue_start, kQueueBytes);
    store<u64>(b + kQVt + 0x18, reinterpret_cast<u64>(&stub_partition));
    store<u64>(qa(), b + kQVt);
    store<u32>(qa() + kChunkCount, 0);
    store<u64>(game_address(kGx), b + kMgr);
    store<u32>(b + kMgr + 0x280, 6);
    store<u8>(game_address(kTaskSort), 1);
    g_push_mode = Push::WithCut;
    g_cut_on.store(true);
    g_stub_cut = false;
    for (u32 e = 0; e < kTableEntries; ++e) g_cost_per_item[e] = 0;
    const u32 k = static_cast<u32>(g_rng.range(2, 8)), m = static_cast<u32>(g_rng.range(1, 12));
    g_guided_k = k;
    g_guided_least = m;
    g_pushed_jobs.clear();
    g_guided_pushed.clear();
    g_pf_calls.clear();
    eboot_kit::stub(kJobFor, reinterpret_cast<const void*>(&rec_job_for_mem));
    eboot_kit::stub(kPushJob, reinterpret_cast<const void*>(&rec_push_mem));
    const u32 made = render_queue_parallel_for(qa(), 0x5555000000ull, 0, static_cast<u32>(count), 6);
    eboot_kit::stub(kJobFor, reinterpret_cast<const void*>(&rec_job_for));
    eboot_kit::stub(kPushJob, reinterpret_cast<const void*>(&rec_push));
    g_guided_k = 0;
    bool ok = g_guided_pushed.size() == g_pushed_jobs.size() && g_guided_pushed.size() == made;
    int guided = 0;
    const u32 total = load<u32>(qa() + kChunkCount);
    for (std::size_t i = 0; i < g_guided_pushed.size() && ok; ++i) {
        const u64 c = qa() + kChunks + static_cast<u64>(total - made + i) * kChunkSize;
        const u32 span = load<u32>(c + 4) - load<u32>(c), size = load<u32>(c + 8);
        const bool want = static_cast<s32>(size) < static_cast<s32>(span);
        const u64 j = g_guided_pushed[i].first;
        const u64 vt = load<u64>(j);
        ok = (want ? vt == host_address(g_rq_job_vt) : vt == game_address(kJobVtable)) && g_guided_pushed[i].second == std::min(span, size);
        guided += want;
    }
    if (!ok && report) {
        char why[96];
        std::snprintf(why, sizeof why, "%d items, %u chunks (%d task-stealing): a job's vtable or grain", count, made, guided);
        fail("guided parallel-for", t, why);
    }
    return ok;
}

// ---- The cut's choice ----

// BBHOST_RQ_SPLIT unset: the cut, and the costs' learning with it, on while
// the frame rate's limit is above 60, decided each flip; the push longest
// first after a parallel-for's partition cut a run (BBHOST_RQ_PUSH unset), or
// always or never when set; forced on or off, the limit changes nothing.
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
        ok = ok && g_cut_on.load() == r.on && learning() == r.on && cut_this_flip() == r.on && g_said_cut.load() == (r.on ? 1 : 0) &&
             push_after(true) && !push_after(false);
    }
    g_push_mode = Push::Off;
    ok = ok && !push_after(true) && !push_after(false) && learning();  // the cut's rule still wants the costs
    g_push_mode = Push::On;
    render_queue_flip(flip++, 60);
    ok = ok && !g_cut_on.load() && push_after(false) && learning();
    g_split_mode = Split::Off;
    g_push_mode = Push::WithCut;
    g_cut_on.store(false);
    render_queue_flip(flip++, 120);
    ok = ok && !g_cut_on.load() && !learning() && !cut_this_flip();
    g_split_mode = Split::On;
    g_cut_on.store(true);
    render_queue_flip(flip++, 60);
    ok = ok && g_cut_on.load() && learning() && cut_this_flip();
    return ok;
}

// ---- The game's own behind ours ----

u64 g_orig[5];
int g_orig_calls = 0;
GUEST_ABI u32 orig_partition(u64 q, u32 first, u32 last, u32 limit) {
    ++g_orig_calls;
    g_orig[0] = q, g_orig[1] = first, g_orig[2] = last, g_orig[3] = limit;
    return 0x1234;
}
GUEST_ABI u64 orig_job(u64 functor, u64 range, u32 worker) {
    ++g_orig_calls;
    g_orig[0] = functor, g_orig[1] = range, g_orig[2] = worker;
    return 0x5678;
}
GUEST_ABI u64 orig_parallel_for(u64 q, u64 ctx, u32 first, u32 last, u32 limit) {
    ++g_orig_calls;
    g_orig[0] = q, g_orig[1] = ctx, g_orig[2] = first, g_orig[3] = last, g_orig[4] = limit;
    return 0x9a;
}

// What ours has nothing to add to goes to the game's own as given, and what
// it returns comes back: a partition not the parallel-for's (or with the cut
// off), every chunk job but the census's (timed into its entry's costs while
// the cut is on), and a parallel-for that cannot push.
bool originals_case() {
    void* const keep[3] = {g_game_partition, g_game_chunk_job, g_game_parallel_for};
    g_game_partition = reinterpret_cast<void*>(&orig_partition);
    g_game_chunk_job = reinterpret_cast<void*>(&orig_job);
    g_game_parallel_for = reinterpret_cast<void*>(&orig_parallel_for);
    bool ok = true;
    g_split_mode = Split::Auto;
    g_push_mode = Push::WithCut;
    g_cut_on.store(false);
    g_orig_calls = 0;
    ok = ok && render_queue_partition(0x1000, 3, 9, 6) == 0x1234 && g_orig_calls == 1 && g_orig[0] == 0x1000 && g_orig[1] == 3 &&
         g_orig[2] == 9 && g_orig[3] == 6;
    ok = ok && render_queue_parallel_for(0x2000, 0x3000, 4, 8, 6) == 0x9a && g_orig_calls == 2 && g_orig[0] == 0x2000 &&
         g_orig[1] == 0x3000 && g_orig[2] == 4 && g_orig[3] == 8 && g_orig[4] == 6;
    g_push_mode = Push::Off;
    g_cut_on.store(true);
    ok = ok && render_queue_parallel_for(0x2000, 0x3000, 4, 8, 6) == 0x9a && g_orig_calls == 3;
    g_push_mode = Push::WithCut;
    // a chunk job: the game's, and with the cut on its ticks and items to its entry
    int count = 0;
    while (count < 4) build_queue(count);
    std::memcpy(g_queue_b, g_queue_start, kQueueBytes);
    alignas(8) static u64 functor[3];
    alignas(8) static u32 range[3];
    functor[0] = 0, functor[1] = 0, functor[2] = qb();
    range[0] = 1, range[1] = 3, range[2] = 2;
    const u64 table = game_address(kTable);
    const s8 e = entry_index(entry_of(load<u32>(reinterpret_cast<u64>(g_items_mem) + 0x18), table), table);
    for (u32 k = 0; k < kTableEntries; ++k) g_cost_items[k].store(0), g_cost_ticks[k].store(0);
    g_cut_on.store(false);
    ok = ok && render_queue_chunk_job(reinterpret_cast<u64>(functor), reinterpret_cast<u64>(range), 2) == 0x5678 && g_orig_calls == 4 &&
         g_orig[0] == reinterpret_cast<u64>(functor) && g_orig[1] == reinterpret_cast<u64>(range) && g_orig[2] == 2;
    for (u32 k = 0; k < kTableEntries; ++k) ok = ok && g_cost_items[k].load() == 0;
    g_cut_on.store(true);
    ok = ok && render_queue_chunk_job(reinterpret_cast<u64>(functor), reinterpret_cast<u64>(range), 2) == 0x5678 && g_orig_calls == 5;
    for (u32 k = 0; k < kTableEntries; ++k) ok = ok && g_cost_items[k].load() == (static_cast<s8>(k) == e ? 2u : 0u);
    for (u32 k = 0; k < kTableEntries; ++k) g_cost_items[k].store(0), g_cost_ticks[k].store(0);
    g_game_partition = keep[0], g_game_chunk_job = keep[1], g_game_parallel_for = keep[2];
    return ok;
}

// ---- The segment list's note ----

GUEST_ABI u64 rec_append(u64 list, u64 record, u32 key) {
    g_rec.push_back({kQueue, {list, record, key, 0, 0}});
    return 0x71;
}

// ---- The scene categories (BBHOST_SCENE_CATEGORIES=1) ----

// count_categories against a map of the same items: random queues (sorted or
// not) behind 20 queue objects, random ranges, the first 16 queues counted a
// row each and the rest dropped; then four threads counting one range at
// once; and log_categories leaving every counter at 0.
bool categories_case() {
    constexpr int kObjects = 20;
    alignas(64) static u8 objects[kObjects][0x20];
    std::map<u64, std::map<u32, u64>> want;
    std::vector<u64> rows;
    u64 want_dropped = 0;
    alignas(8) static u32 range[2];
    const auto count_one = [&](u64 q, u32 first, u32 last) {
        range[0] = first, range[1] = last;
        count_categories(q, reinterpret_cast<u64>(range));
        const u32 count = load<u32>(q + kCount), end = count > last ? last : count;
        if (end <= first) return;
        if (std::find(rows.begin(), rows.end(), q) == rows.end()) {
            if (rows.size() == kCategoryQueues) {
                want_dropped += end - first;
                return;
            }
            rows.push_back(q);
        }
        const u64 items = load<u64>(q + kItems);
        for (u32 i = first; i != end; ++i) ++want[q][(load<u32>(items + static_cast<u64>(i) * kItemSize) >> 16) & 0xfff];
    };
    for (int t = 0; t < 4000; ++t) {
        int count = 0;
        build_queue(count);
        const u64 q = reinterpret_cast<u64>(objects[g_rng.range(0, t < 400 ? 9 : kObjects - 1)]);
        store<u64>(q + kItems, load<u64>(reinterpret_cast<u64>(g_queue_start) + kItems));
        store<u32>(q + kCount, static_cast<u32>(count));
        const u32 first = static_cast<u32>(g_rng.range(0, count + 2));
        const u32 last = g_rng.chance(0.3) ? static_cast<u32>(g_rng.range(0, count + 4)) : static_cast<u32>(count);
        count_one(q, first, last);
    }
    if (rows.size() != kCategoryQueues || want_dropped == 0) return false;
    bool ok = true;
    // four threads on one queue's range
    {
        int count = 0;
        while (count < 100) build_queue(count);
        const u64 q = rows[0];
        store<u64>(q + kItems, load<u64>(reinterpret_cast<u64>(g_queue_start) + kItems));
        store<u32>(q + kCount, static_cast<u32>(count));
        alignas(8) static u32 shared_range[2];
        shared_range[0] = 0, shared_range[1] = static_cast<u32>(count);
        std::vector<std::thread> threads;
        for (int k = 0; k < 4; ++k)
            threads.emplace_back([q] {
                for (int n = 0; n < 500; ++n) count_categories(q, reinterpret_cast<u64>(shared_range));
            });
        for (auto& th : threads) th.join();
        const u64 items = load<u64>(q + kItems);
        for (int i = 0; i < count; ++i) want[q][(load<u32>(items + static_cast<u64>(i) * kItemSize) >> 16) & 0xfff] += 4 * 500;
    }
    for (u32 r = 0; r < kCategoryQueues; ++r) {
        const u64 q = g_category_queue[r].load();
        ok = ok && q == rows[r];
        for (u32 pass = 0; pass < kPasses; ++pass) {
            const auto w = want[q].find(pass);
            ok = ok && g_category_items[r][pass].load() == (w == want[q].end() ? 0 : w->second);
        }
    }
    ok = ok && g_category_dropped.load() == want_dropped;
    log_categories(300);
    for (u32 r = 0; r < kCategoryQueues; ++r)
        for (u32 pass = 0; pass < kPasses; ++pass) ok = ok && g_category_items[r][pass].load() == 0;
    return ok && g_category_dropped.load() == 0;
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
    g_guided_k = 0;  // the cases before the guided split's own: the game's splits (their jobs are stand-ins)
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
    // ours leaves to the game's own through these (their trampolines in the game)
    g_game_partition = reinterpret_cast<void*>(eboot_kit::fn<GamePartition>(kPartition));
    g_game_chunk_job = reinterpret_cast<void*>(eboot_kit::fn<GameJob>(kChunkJob));
    g_game_parallel_for = reinterpret_cast<void*>(eboot_kit::fn<GameParallelFor>(kParallelFor));
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
    if (!categories_case()) fail("scene categories", 0, "the counts by queue and pass not the items', or not cleared by the log");
    if (!choice_case()) fail("the cut's choice", 0, "not on above 60 fps alone, or the push not after a cut");
    if (!originals_case()) fail("the game's own", 0, "what ours leaves to the game's functions not passed on as given");
    g_split_mode = Split::On;
    g_cut_on.store(true);
    // the guided split
    eboot_kit::stub(kPoolAlloc, reinterpret_cast<const void*>(&stub_pool_alloc));
    eboot_kit::stub(kJobRun, reinterpret_cast<const void*>(&rec_job_run));
    guided_setup();
    long n_guided = 0;
    for (int t = 0; t < cases / 4; ++t, ++n_guided) guided_split_case(t, true);
    for (int t = 0; t < cases / 40; ++t) guided_cover_case(t, true);
    for (int t = 0; t < cases / 40; ++t) guided_pf_case(t, true);
    g_guided_k = 0;

    // Every mutant caught.
    const int kPartitionMutants[] = {1, 2, 3, 4, 10, 11, 12};
    const int kLoopMutants[] = {5, 6, 7};
    const int kJobMutants[] = {8, 9};
    const int kCutMutants[] = {13, 16};
    const int kRankMutants[] = {14};
    const int kPushMutants[] = {15, 17};
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
    for (int m : kPushMutants) catches(m, [m](int t) { return parallel_for_case(t, m == 15, false); }, 20000);
    for (int m : {18, 19, 20, 21, 23}) catches(m, [](int t) { return guided_split_case(t, false); }, 20000);
    catches(22, [](int t) { return guided_pf_case(t, false); }, 2000);
    g_guided_k = 0;
    g_quiet = false;

    if (g_failed) {
        std::printf("gx_render_queue_test: %d failures\n", g_failed);
        return 1;
    }
    std::printf("gx_render_queue_test: ok - partition %ld cases (%ld cut a run), item loop %ld, chunk job %ld; the cut %ld (%ld cut further "
                "than the game), ranks %ld, parallel-for %ld; the guided split %ld (+ %d covers, %d parallel-fors); %d of %d mutants caught\n",
                n_part, split, n_loop, n_job, n_cut, g_cut_seen, n_ranks, n_pf, n_guided, cases / 40, cases / 40, caught, mutants);
    return 0;
}
