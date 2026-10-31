// The scene's render queue as source (docs/decomp.md), and a census of it:
// what each GX worker records of a frame's draws, and for how long.
//
// The scene draw (sub_26b0c70) builds a render queue of the frame's items -
// 0x18 bytes each: a u32 sort key at +0, a command stream at +8, a byte at
// +0x11 that marks an item drawn before the rest - and executes it
// (sub_12c6f80). With the scene's MultiThreadRendering on and its Enable Task
// Sort (data_5522754) on, the whole queue goes to GX's parallel-for
// (sub_12c7810), which:
//   - cuts it into chunks: the queue's vtable slot 3, the partition
//     (sub_26d11d0, below);
//   - sorts the chunks by a rank from the partition's table (sub_2d49590);
//   - pushes one job a chunk, the ranks up to 5 to participant i mod n, the
//     rest to participant 0 (the main thread), where the other participants
//     (the five GXWorker threads) steal them;
//   - and waits in GX's job barrier (sub_25b8710), running jobs itself.
// A job is the chunk job (sub_2d4a400, below): a context of its own from the
// parent context, the chunk's items recorded into it (sub_12c7470's loop),
// and its command list queued in the frame's segment list
// (gx_queue_frame_command_context, 0x2594340, which appends it with
// sub_25b4a00) under an order key: the first item's key's upper half and the
// first item's index. The frame's end (gx_device_frame_end) sorts the list
// by that key, stably (sub_25b4f70), and plays it in that order.
//
// The partition. A run of items whose keys fall in one entry of a static
// table of 31 (0x553ae90, 9 dwords each: the key's category (bits 22-27) and
// sub-category (16-21) the entry starts at, then how it may be cut) becomes:
//   - no chunk, when no entry covers its keys (its items are not drawn);
//   - one chunk, when the parallel-for allows one, or the entry allows one,
//     or the run has fewer than twice the entry's least items (5 in all);
//   - else min(the entry's chunks, items / 5) chunks of equal size (the first
//     ones an item longer), the entry's chunks either its own count (+8) or
//     ceil(items / its divisor (+0x10)) up to its cap (+0xc);
//   - with the queue's task-stealing byte (+0x634, the scene's Enable Task
//     Stealing) one chunk, its grain (+8) items / chunks, at least 10 - a
//     grain no job reads in this build: the job runner (sub_25b7be0) runs
//     the whole chunk.
// The range's last run is always one chunk, whatever its entry allows, and
// so is everything left once the table (63 chunks at +0x30, 0x18 bytes each:
// first, last, size, the entry's +0x18, its rank (+0x1c), its byte at +0x20)
// would pass 62. Ours does the same; the census says which runs met which.
//
// The cut (BBHOST_RQ_SPLIT; unset, on while the frame rate's limit is above
// 60, at 90, 120 or uncapped but for V-Sync on a 60 Hz display, and off at 60
// and below; 1 always, 0 never). The heaviest of a frame's chunks can be one
// run the table allows one chunk: at the Cainhurst courtyard 88 items of
// category 0x3e from sub-category 5 (2.7-3.7 ms), while the other five
// participants are done in 2 ms and wait. The engine's PC build cuts the same
// entries further: Dark Souls III's table (1.15.2's 0x14473d5a0, 36 entries)
// has, against this one's, the entry after the four-way one (0x45/6 there,
// 0x3e/5 here) at ceil(items / 10) chunks up to 3, and the first of its last
// category (0x45/0, 0x3e/0) at ceil(items / 10) up to 5; its other entries
// that have one here cut as these do or less (one more, 0x45/0x33, up to 6 by
// ceil(items/15), has none). On the parallel-for's calls ours takes those two
// limits (the cut table: the game's with entries 17 and 19 changed) for a run
// long enough to be the parallel-for's tail alone: its cost - its entry's
// measured ticks an item, times its items - at least the participants' even
// share of the parallel-for's measured work (both moving means), or, while
// either is not known, 40 items or more. In the census only Cainhurst's run of
// 0x3e/5 is one (88 items at the courtyard, 25 to 83 on its roof: 25-27% of
// the work, against a sixth); every other run of the two entries measured is
// 10 to 24 items, under 0.7 ms and 8% of the work, and cut into two or three
// pieces on contexts of their own they cost more than they save (1-4% at
// Cathedral Ward and Old Yharnam, uncapped). A call whose chunks would then
// pass 62 is made with the game's table instead, so the rest of a queue is
// never put in one chunk for it. A cut chunk's pieces keep its entry's fields,
// and the chunk job keys each piece's segment by its own first item as the
// game's does, so the frame's end plays them one after another where the uncut
// chunk was: the census found no other segment's key between them at the four
// places measured, and no item drawing with state an earlier item had set.
// Each piece's first item sets up its pass on its own context, as in the dozen
// categories the game cuts itself. BBHOST_RQ_SPLIT_EVERY=N cuts on N flips,
// then not for N (the draw digest's pair in one run). At 60 and below the
// limiter waits out the time the cut saves, so it buys headroom alone, for CPU
// time an APU's GPU shares its power with: there the partitions stay the
// game's. The choice is made each flip (render_queue_flip, from the frame
// rate's frame_rate_limit_fps), so a V-Sync change in F10 or the options takes
// effect from the next frame's partitions, and each change is logged once.
//
// Two orders for the jobs, each its own switch, neither changing what is drawn
// (the frame's end orders the segments by their keys, not by when a job ran).
// BBHOST_RQ_PUSH (unless set, after a partition that cut a run; the game's
// order otherwise) places the parallel-for (sub_12c7810) as ours, which pushes
// the chunks longest first - by their cost, the jobs' own times as a moving
// mean by entry and item - and pins them by their ranks as the game's does.
// The game's pinning is by place in the rank order (participant i mod 6), so a
// cut that adds even two small pieces moves every later chunk to another
// participant: the cut alone lost 3% at the Nightmare of Mensis that way;
// longest first, the Cainhurst courtyard gains 12-19% uncapped on an RTX 4070.
// Without a cut the longest-first order puts more of the work on the main
// thread at Cathedral Ward (its main loop 0.2 ms longer), so a frame that cuts
// nothing keeps the game's. BBHOST_RQ_RANKS=1 (off) writes each chunk's rank
// from its cost instead - the dearest six 0, the next six 1, and only the
// cheapest keep the ranks that put them on the main thread. Both learn the
// costs in the chunk job.
//
// Where ours has nothing to add, the game's own functions run, through their
// trampolines: every partition but a parallel-for's while the costs are wanted
// (the cut on, BBHOST_RQ_PUSH=1 or BBHOST_RQ_RANKS=1), every chunk job but the
// census's - the game's job, timed while the costs are wanted - and the
// parallel-for while it cannot push. With the cut off, at 60 and below, the
// three are the game's own.
//
// The census (BBHOST_RQ_CENSUS=1; the segment note is placed only with it):
// per parallel-for, its runs and the chunks they became; per chunk job,
// the participant, its start and end (rdtsc), its segment's order key, and
// per item its time, its key and its commands - those that set the context's
// state (1, 2, 6, 7), those that draw (3, 4, 0xf8), and the state types it
// drew with that an earlier item of its chunk had set and it had not; per
// segment appended to the frame's list, its key. The leaves write rings (no
// host thread-locals, locks or allocation); a thread of ours folds them
// every 25 ms into `rq census:` lines every BBHOST_RQ_CENSUS_FLIPS flips
// (300) and at exit and, with BBHOST_RQ_CENSUS_OUT=path, writes the records
// to that file (tools/rq_census.py reads it).
#include "decomp/gx/render_queue.h"

#include "core/thunk.h"
#include "decomp/decomp.h"
#include "decomp/guest.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <x86intrin.h>

using namespace decomp;

#ifdef DECOMP_GX_RENDER_QUEUE_MUTANTS
extern int g_rq_mutant;
#define MUTANT(n) (g_rq_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kPartition = 0x26d11d0;   // the queue's vtable slot 3 (vtable 0x57611d0)
constexpr u64 kChunkJob = 0x2d4a400;    // the parallel-for's job functor's run (vtable 0x56ea060 slot 2)
constexpr u64 kListAppend = 0x25b4a00;  // a segment appended to the frame's list
constexpr u64 kGx = 0x59406c8;          // GX's manager
constexpr u64 kTable = 0x553ae90;       // the partition's 31 entries (and the 32nd's first two dwords)
constexpr u64 kBeginRecord = 0x259fb70;   // (command context): before the chunk's items
constexpr u64 kEndRecord = 0x259fcc0;     // (command context): after them
constexpr u64 kQueueContext = 0x2594340;  // gx_queue_frame_command_context(GX, command context, order key)
constexpr u64 kOwnerOf = 0x247b720;       // the allocator an object came from
constexpr u32 kFromParallelFor = 0x12c7846;  // the parallel-for's call of the partition returns here
constexpr u64 kParallelFor = 0x12c7810;   // GX's parallel-for over a queue's items: (queue, ctx, first, last, limit)
constexpr u64 kSortByRank = 0x2d49590;    // an introsort of the chunks by rank: (first, the last chunk, depth)
constexpr u64 kWaitFirst = 0x25b6fe0;     // a barrier first, when GX's +0x2e0 asks for one
constexpr u64 kJobFor = 0x25b7010;        // (functor, range, participant): a job from the participant's pool
constexpr u64 kPushJob = 0x25b8af0;       // (GX's jobs, job, participant)
constexpr u64 kAfterPush = 0x25b7000;     // returns at once
constexpr u64 kWake = 0x25b8670;          // (GX's jobs, 0, limit): wakes the participants
constexpr u64 kBarrier = 0x25b8710;       // GX's job barrier
constexpr u64 kFunctorVtable = 0x56ea060; // the parallel-for's job functor (relocated; slot 2 the chunk job)
constexpr u64 kTaskSort = 0x5522754;      // the scene's Enable Task Sort (1)

// The queue.
constexpr u64 kItems = 0x8, kCount = 0x10, kChunks = 0x30, kChunkCount = 0x630, kStealing = 0x634;
constexpr u64 kItemSize = 0x18, kChunkSize = 0x18;
constexpr u32 kChunkMax = 0x3f;  // the table's chunks: 63
constexpr u64 kTableEntry = 0x24;
constexpr u32 kTableEntries = 31;
// The context: the item's key as its commands see it, and its command context.
constexpr u64 kCtxKeyTop = 0x110, kCtxCategory = 0x114, kCtxSub = 0x118, kCtxLow = 0x11c, kCtxCommand = 0x430;

// push rbp; push r15; push r14 (5 bytes, whole instructions).
constexpr std::uint8_t kPartitionEntry[] = {0x55, 0x41, 0x57, 0x41, 0x56};
// push rbp; mov rbp, rsp; push r15; push r14 (8 bytes), at both.
constexpr std::uint8_t kChunkJobEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56};
constexpr std::uint8_t kListAppendEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56};
constexpr std::uint8_t kParallelForEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56};

using OneFn = GUEST_ABI u64 (*)(u64 a);
using TwoFn = GUEST_ABI u64 (*)(u64 a, u64 b);
using ContextFn = GUEST_ABI u64 (*)(u64 parent, u64 per_worker, u64 zero);
using QueueFn = GUEST_ABI u64 (*)(u64 gx, u64 command_context, u32 key);
using PrepareFn = GUEST_ABI u64 (*)(u64 queue, u64 ctx, u64 state, u32 key, u32 pre);
using CommandFn = GUEST_ABI u64 (*)(u64 queue, u64 ctx, u32 type, u64 payload, u64 size);
using AppendFn = GUEST_ABI u64 (*)(u64 list, u64 record, u32 key);
using PartitionFn = GUEST_ABI u32 (*)(u64 queue, u32 first, u32 last, u32 limit);
using SortFn = GUEST_ABI u64 (*)(u64 first, u64 last, u64 depth);
using VoidFn = GUEST_ABI u64 (*)();
using JobForFn = GUEST_ABI u64 (*)(u64 functor, u64 range, u32 participant);
using PushFn = GUEST_ABI u64 (*)(u64 jobs, u64 job, u32 participant);
using WakeFn = GUEST_ABI u64 (*)(u64 jobs, u32 zero, u32 limit);
using ChunkJobFn = GUEST_ABI u64 (*)(u64 functor, u64 range, u32 participant);
using ParallelForFn = GUEST_ABI u64 (*)(u64 queue, u64 ctx, u32 first, u32 last, u32 limit);

template <class F>
inline F method(u64 object, u64 offset) {
    return reinterpret_cast<F>(static_cast<std::uintptr_t>(load<u64>(load<u64>(object) + offset)));
}

inline u64 ticks() { return __rdtsc(); }
inline void publish(u32& done, u32 v) { std::atomic_ref<u32>(done).store(v, std::memory_order_release); }
inline u32 seen(u32& done) { return std::atomic_ref<u32>(done).load(std::memory_order_acquire); }

// ---- The census's rings (written by the leaves, folded by our thread) ----

struct Item {  // 16 bytes
    u32 key;
    u32 ticks;     // its prepare and commands; 0 when skipped
    u16 states;    // commands that set the context's state: 1, 2, 6, 7
    u16 draws;     // commands that draw: 3, 4, 0xf8
    u16 other;     // the rest (8 and unknown types)
    u8 flags;      // kSkipped, kInherits
    u8 inherited;  // the state types (kState*) it drew with from an earlier item of its chunk
};
enum : u8 { kSkipped = 1, kInherits = 2 };
enum : u8 { kState1 = 1, kState2 = 2, kState6 = 4, kState7 = 8 };
static_assert(sizeof(Item) == 16);

struct Chunk {  // 64 bytes
    u64 t0, t1;   // the job's start and end
    u64 queue;
    u32 first, last;
    u32 item_at;  // its first item's slot in the item ring
    u32 n_items;  // slots written (at most kChunkItems)
    u32 gen;      // the parallel-for it ran in
    u32 worker;   // GX's participant: 0 the main thread
    u32 key;      // its segment's order key
    u32 work;     // its items' ticks summed
    u32 lead;     // ticks from its start to its first item
    u32 done;     // gen + 1, once written
};
static_assert(sizeof(Chunk) == 64);

struct Run {
    s8 entry;    // the table's entry, -1 none
    u8 how;      // kHow*
    u16 chunks;  // chunks it became
    u32 start, n;
};
enum : u8 { kHowNone, kHowOne, kHowSplit, kHowStealing, kHowOverflow, kHowLast };
constexpr u32 kRunsKept = 96;

struct Part {
    u64 t;  // its entry
    u64 queue;
    u32 ret;  // the caller's return address, as Binary Ninja's
    u32 first, last, end, max_chunks;
    u32 before, made;
    u32 gen, flip;
    u32 n_runs;  // runs (more than kRunsKept are counted, not kept)
    u32 stealing;
    Run runs[kRunsKept];
    u32 done;
};

struct Append {
    u64 t;
    u64 list;
    u32 key;   // as given (0xffffffff: the list's next category)
    u32 next;  // the key 0xffffffff becomes, from the list's +0x40 then
    u32 ret;   // the caller's return address, as Binary Ninja's
    u32 gen;
    u32 done;
    u32 pad;
};
static_assert(sizeof(Append) == 40);

constexpr u32 kItemRing = 1u << 21, kChunkRing = 1u << 16, kPartRing = 1u << 12, kAppendRing = 1u << 16;
constexpr u32 kChunkItems = 16384;

Item* g_items = nullptr;
Chunk* g_chunk_ring = nullptr;
Part* g_parts = nullptr;
Append* g_appends = nullptr;
std::atomic<u64> g_item_pos{0}, g_chunk_pos{0}, g_part_pos{0}, g_append_pos{0};
std::atomic<u32> g_gen{0};
std::atomic<u32> g_flip{0};
bool g_census = false;

// The switches (decomp_gx_render_queue_add).
enum class Split : u8 { Auto, On, Off };   // BBHOST_RQ_SPLIT: unset, 1, 0
enum class Push : u8 { WithCut, On, Off };  // BBHOST_RQ_PUSH: unset, 1, 0
Split g_split_mode = Split::Auto;  // the cut table on the parallel-for's partitions: Auto above 60 fps
Push g_push_mode = Push::WithCut;  // the parallel-for as ours, longest first
u32 g_split_every = 0;    // BBHOST_RQ_SPLIT_EVERY=N: the cut on N flips, off N
bool g_ranks = false;     // BBHOST_RQ_RANKS=1: ranks by cost
bool g_push_placed = false;  // the parallel-for placed: the push could be on
// The cut now (render_queue_flip: Auto by the frame rate's limit, each flip;
// On and Off once, at placement); the costs' learning with it.
std::atomic<bool> g_cut_on{false};
std::atomic<int> g_said_cut{-1};  // the choice the log last said (Auto)

// The chunk job times its chunks for the costs: the cut's rule needs them,
// and either order by cost.
inline bool learning() { return g_ranks || g_push_mode == Push::On || g_cut_on.load(std::memory_order_relaxed); }
// The parallel-for's order: longest first when BBHOST_RQ_PUSH=1, or (unset)
// when its partition cut a run; the game's rank order otherwise.
inline bool push_after(bool cut) { return g_push_mode == Push::On || (g_push_mode == Push::WithCut && (cut || MUTANT(17))); }
// Our parallel-for's partition cut a run (render_queue_partition). The
// parallel-for runs on one thread, as g_in_parallel_for has it.
bool g_pf_cut = false;
// Our parallel-for's own call of the partition (BBHOST_RQ_PUSH): its return
// address is then ours, not the game's 0x12c7846. The parallel-for runs on the
// main thread alone (the execute's parallel path, sub_26b0c70's).
bool g_in_parallel_for = false;

// The cut table: the game's 31 entries and the 32nd's first two dwords, as
// u32s, with entries 17 (0x3e from sub-category 0) and 19 (0x3e from 5) at the
// PC build's limits (the file's head). Made when the partition is placed.
constexpr u32 kTableWords = (kTableEntries + 1) * 9;
u32 g_cut_table[kTableWords];
bool g_cut_ready = false;
struct CutLimit {
    u32 entry, cap, divisor;
};
constexpr CutLimit kCutLimits[] = {{17, 5, 10}, {19, 3, 10}};

// The costs: per entry, the ticks and items the chunk jobs took since the
// last partition (added by the jobs), and a moving mean of ticks an item
// (x256), which the partition folds them into before each parallel-for.
std::atomic<u64> g_cost_ticks[kTableEntries], g_cost_items[kTableEntries];
u64 g_cost_per_item[kTableEntries];  // ticks x256 an item; 0: not known
std::atomic<u64> g_cut_calls{0}, g_cut_fallbacks{0}, g_ranked{0}, g_pushed{0}, g_cut_runs{0}, g_cut_parts{0};
// The cut's rule (the file's head): a run of the two entries the cut table
// changes is cut only when it is long enough to be its parallel-for's tail -
// its cost (its entry's ticks an item, times its items) at least the
// participants' even share of the parallel-for's work (the chunk jobs' ticks
// over the participants, a moving mean like the costs'); while either is not
// known, at least kTailItems items.
constexpr u32 kTailItems = 40;
std::atomic<u64> g_even_share{0};  // ticks; 0: not known
// The game's own functions behind ours (their trampolines), for what ours
// leaves to them: the partitions the cut has nothing to do with, every chunk
// job but the census's, and the parallel-fors that cannot push.
void* g_game_partition = nullptr;
void* g_game_chunk_job = nullptr;
void* g_game_parallel_for = nullptr;

inline u32 bn_of(u64 guest) { return static_cast<u32>(guest - g_decomp_slide + 0x400000); }
inline u64 host_address(const void* p) { return reinterpret_cast<u64>(p); }

// ---- The partition (sub_26d11d0) ----

// The entry covering a key's category and sub-category, as the game looks it
// up: an entry covers the categories up to the next entry's - all of its own
// when its sub-category is -1, else its own from that sub-category on - and
// two entries of one category split it at the second's sub-category.
inline u64 entry_of(u32 key, u64 table) {
    const u32 cat = (key >> 22) & 0x3f, sub = (key >> 16) & 0x3f;
    u64 e = table;
    for (u32 j = 0; j < kTableEntries; ++j, e += kTableEntry) {
        const u32 lo = load<u32>(e), next = load<u32>(e + kTableEntry);
        if (lo <= cat && next > cat) {
            const u32 s = load<u32>(e + 4);
            if (s == 0xffffffffu || lo != cat || (MUTANT(10) ? s < sub : s <= sub)) return e;
        } else if (lo == cat && next == cat && load<u32>(e + 4) <= sub && load<u32>(e + kTableEntry + 4) > sub) {
            return e;
        }
    }
    return 0;
}

inline void put_chunk(u64 q, u32 at, u32 first, u32 last, u32 size, u64 e) {
    const u64 c = q + kChunks + static_cast<u64>(at) * kChunkSize;
    store<u32>(c, first);
    store<u32>(c + 4, last);
    store<u32>(c + 8, size);
    store<u32>(c + 0xc, load<u32>(e + 0x18));
    store<u32>(c + 0x10, load<u32>(e + 0x1c));
    store<u8>(c + 0x14, load<u8>(e + 0x20));
}

inline s8 entry_index(u64 e, u64 table) { return e ? static_cast<s8>((e - table) / kTableEntry) : s8{-1}; }

inline void note(Part* part, u64 table, u64 e, u8 how, u32 chunks, u32 start, u32 n) {
    if (!part) return;
    if (part->n_runs < kRunsKept) part->runs[part->n_runs] = Run{entry_index(e, table), how, static_cast<u16>(chunks), start, n};
    ++part->n_runs;
}

// A chunk's cost, from its entry's ticks an item: 0 when not known.
inline u64 chunk_cost(u32 items, s8 entry) {
    return entry < 0 ? 0 : (g_cost_per_item[static_cast<u32>(entry)] * items) >> 8;
}

// One of the entries the cut table changes.
inline bool cut_entry(s8 e) {
    for (const CutLimit& c : kCutLimits)
        if (e >= 0 && static_cast<u32>(e) == c.entry) return true;
    return false;
}

// A run long enough to be its parallel-for's tail (the cut's rule).
inline bool tail_run(u32 n, s8 e) {
    if (MUTANT(16)) return true;
    const u64 share = g_even_share.load(std::memory_order_relaxed), cost = chunk_cost(n, e);
    return share && cost ? cost >= share : n >= kTailItems;
}

// The game's partition with the entries of `table` (the game's, or the cut
// table): the chunks appended at the queue's table, how many returned.
// `part`: the census's record, or null. `full`, when given: set when the rest
// of the range went into one chunk for want of room (the table past 62, or at
// 63). `entries`, when given: each new chunk's entry index, -1 none (at most
// 64 kept). `cut_runs`, when given (with the cut table): a run of the entries
// it changes takes its limits only when tail_run says so, the game's
// otherwise, and the runs cut past the game's are counted there.
DECOMP_LEAF u32 partition(u64 q, u32 first, u32 last, u32 max_chunks, Part* part, u64 table = 0, bool* full = nullptr,
                          s8* entries = nullptr, u32* cut_runs = nullptr) {
    const u64 game_table = game_address(kTable);
    if (!table) table = game_table;
    const u32 before = load<u32>(q + kChunkCount);
    const auto entry_note = [&](u32 at, u64 e) {
        if (entries && at >= before && at - before < 64) entries[at - before] = entry_index(e, table);
    };
    const u32 count = load<u32>(q + kCount);
    const u32 end = count > last ? last : count;
    const u64 items = load<u64>(q + kItems);
    u32 n_chunks = load<u32>(q + kChunkCount);
    u64 cur = 0;   // the run's entry
    u64 prev = 0;  // the item before
    u32 start = first;
    bool overflow = false;
    for (u32 i = first; i < end;) {
        const u64 item = items + static_cast<u64>(i) * kItemSize;
        u64 next;
        if (!prev) {
            next = entry_of(load<u32>(item), table);
        } else {
            const u32 key = load<u32>(item);
            const u32 x = load<u32>(prev) ^ key;
            if (x <= 0x0fffffffu && (x & 0x0fc00000u) == 0 && (x & 0x3f0000u) == 0) {
                next = cur;
            } else {
                next = entry_of(key, table);
                if (!next || next != cur) {
                    // The run [start, i) ends.
                    const u32 n = i - start;
                    if (n == 0) {
                        start = i;
                    } else {
                        u32 cut, least;
                        // the entry whose limits cut the run: the cut table's for a tail run
                        u64 lim = cur;
                        bool tail = false;
                        if (cut_runs && cur && cut_entry(entry_index(cur, table))) {
                            tail = tail_run(n, entry_index(cur, table));
                            if (!tail) lim = game_table + (cur - table);
                        }
                        if (!cur) {
                            least = 5;
                            cut = 1;
                        } else {
                            const u32 per = load<u32>(lim + 0x10);
                            least = load<u32>(lim + 0x14);
                            if (per == 0) {
                                cut = load<u32>(lim + 8);
                            } else {
                                const u32 cap = load<u32>(lim + 0xc);
                                cut = n / per + (n % per != 0 ? 1u : 0u);
                                if (cap < cut) cut = cap;
                                if (cut == 0) cut = 1;
                            }
                        }
                        const u32 by_least = n / least;
                        const u32 chunks = cut < by_least ? cut : by_least;
                        if (max_chunks < 2 || chunks < 2 || MUTANT(2)) {
                            if (cur) {
                                put_chunk(q, n_chunks, start, i, MUTANT(1) ? n + 1 : n, cur);
                                entry_note(n_chunks, cur);
                                ++n_chunks;
                                note(part, table, cur, kHowOne, 1, start, n);
                            } else {
                                note(part, table, cur, kHowNone, 0, start, n);
                            }
                            start = i;
                        } else if (load<u8>(q + kStealing) != 0) {
                            u32 grain = n / chunks;
                            if (grain <= 10) grain = MUTANT(11) ? 11 : 10;
                            if (n < 10) grain = n;
                            put_chunk(q, n_chunks, start, i, grain, cur);
                            entry_note(n_chunks, cur);
                            ++n_chunks;
                            note(part, table, cur, kHowStealing, 1, start, n);
                            start = i;
                        } else if (chunks + n_chunks > (MUTANT(3) ? kChunkMax : kChunkMax - 1)) {
                            // Everything left becomes one chunk, with this run's entry.
                            note(part, table, cur, kHowOverflow, 1, start, end - start);
                            overflow = true;
                            break;
                        } else {
                            const u32 size = n / chunks, longer = n % chunks;
                            u32 at = start;
                            for (u32 k = 0; k != chunks; ++k) {
                                const u32 s = size + (k < longer && !MUTANT(4) ? 1u : 0u);
                                put_chunk(q, n_chunks + k, at, at + s, s, cur);
                                entry_note(n_chunks + k, cur);
                                at += s;
                            }
                            note(part, table, cur, kHowSplit, chunks, start, n);
                            if (tail) ++*cut_runs;  // the game's table gives these entries one chunk
                            start = at;
                            n_chunks += chunks;
                        }
                    }
                }
            }
        }
        prev = item;
        ++i;
        cur = next;
        if (MUTANT(12) ? n_chunks > kChunkMax : n_chunks >= kChunkMax) {
            if (i < end) overflow = true;
            break;
        }
    }
    if (full) *full = overflow;
    entry_note(n_chunks, cur);
    // The last chunk: the rest of the range, whatever its entry allows.
    const u64 c = q + kChunks + static_cast<u64>(n_chunks) * kChunkSize;
    store<u32>(c, start);
    store<u32>(c + 4, end);
    store<u32>(c + 8, end - start);
    if (cur) {
        store<u32>(c + 0xc, load<u32>(cur + 0x18));
        store<u32>(c + 0x10, load<u32>(cur + 0x1c));
        store<u8>(c + 0x14, load<u8>(cur + 0x20));
    } else {
        store<u64>(c + 0xc, 0xffffffff00000012ull);
        store<u8>(c + 0x14, 0);
    }
    if (!overflow || n_chunks >= kChunkMax) note(part, table, cur, kHowLast, 1, start, end - start);
    const u32 made = n_chunks + 1 - load<u32>(q + kChunkCount);
    store<u32>(q + kChunkCount, n_chunks + 1);
    return made;
}

// Whether this partition cuts by the cut table: the parallel-for's, with
// the cut on (and, with BBHOST_RQ_SPLIT_EVERY, on this flip).
inline bool from_parallel_for(u32 ret) { return ret == kFromParallelFor || g_in_parallel_for; }

inline bool cut_this_flip() {
    if (!g_cut_ready || !g_cut_on.load(std::memory_order_relaxed)) return false;
    return !g_split_every || ((g_flip.load(std::memory_order_relaxed) / g_split_every) & 1) != 0;
}

inline bool cutting(u32 ret) { return from_parallel_for(ret) && cut_this_flip(); }

// The partition with the cut table, the game's table where the cut's chunks
// would not fit: either way the chunks appended and how many, and in
// `cut_runs` the runs cut past the game's chunks (0 with the game's table).
DECOMP_LEAF u32 cut_partition(u64 q, u32 first, u32 last, u32 max_chunks, Part* part, s8* entries, u32* cut_runs) {
    const u32 before = load<u32>(q + kChunkCount);
    // the slots a partition may write from here (the table's 63 and the one a
    // full table's last chunk takes), kept for the game's chunks below
    constexpr u32 kSlots = kChunkMax + 1;
    u64 kept[kSlots * 3];
    const u32 n_kept = before < kSlots ? kSlots - before : 0;
    for (u32 k = 0; k < n_kept * 3; ++k) kept[k] = load<u64>(q + kChunks + static_cast<u64>(before) * kChunkSize + 8ull * k);
    bool full = false;
    *cut_runs = 0;
    const u32 made = partition(q, first, last, max_chunks, part, host_address(g_cut_table), &full, entries, cut_runs);
    if (!full || MUTANT(13)) {
        g_cut_calls.fetch_add(1, std::memory_order_relaxed);
        return made;
    }
    // The cut's chunks would pass the table: the game's chunks instead, made
    // again over the slots as they were.
    g_cut_fallbacks.fetch_add(1, std::memory_order_relaxed);
    *cut_runs = 0;
    for (u32 k = 0; k < n_kept * 3; ++k) store<u64>(q + kChunks + static_cast<u64>(before) * kChunkSize + 8ull * k, kept[k]);
    store<u32>(q + kChunkCount, before);
    if (part) part->n_runs = 0;
    return partition(q, first, last, max_chunks, part, 0, nullptr, entries);
}

// The costs the chunk jobs measured since the last parallel-for, into the
// moving means (an eighth of the way a frame): ticks an item by entry, and
// the even share of the whole over GX's participants.
DECOMP_LEAF void fold_costs() {
    u64 total = 0;
    for (u32 e = 0; e < kTableEntries; ++e) {
        const u64 items = g_cost_items[e].exchange(0, std::memory_order_relaxed);
        const u64 t = g_cost_ticks[e].exchange(0, std::memory_order_relaxed);
        if (!items) continue;
        total += t;
        const u64 sample = (t << 8) / items;
        const u64 was = g_cost_per_item[e];
        g_cost_per_item[e] = was ? was + (static_cast<s64>(sample - was) / 8) : sample;
    }
    if (!total) return;
    const u64 gx = load<u64>(game_address(kGx));
    const u32 n = gx ? load<u32>(gx + 0x280) : 0;
    const u64 sample = total / (n ? n : 6);
    const u64 was = g_even_share.load(std::memory_order_relaxed);
    g_even_share.store(was ? was + (static_cast<s64>(sample - was) / 8) : sample, std::memory_order_relaxed);
}

// BBHOST_RQ_RANKS: the chunks [at, at + n) ranked by cost - the six dearest 0,
// the next six 1, ... up to 5 - so the parallel-for pushes the dearest first,
// one to each participant; a chunk dearer than nothing known keeps its rank
// past 5 only if it was cheaper than a twentieth of the dearest (the main
// thread's share, as the game gives it the cheap entries).
DECOMP_LEAF void rank_by_cost(u64 q, u32 at, u32 n, const s8* entries) {
    if (n < 2 || n > 64) return;
    u64 cost[64];
    u8 order[64];
    u64 top = 0;
    for (u32 k = 0; k < n; ++k) {
        const u64 c = q + kChunks + static_cast<u64>(at + k) * kChunkSize;
        cost[k] = chunk_cost(load<u32>(c + 4) - load<u32>(c), entries[k]);
        order[k] = static_cast<u8>(k);
        if (cost[k] > top) top = cost[k];
    }
    if (!top) return;  // nothing learned yet
    // dearest first; ties by the table's order
    for (u32 k = 1; k < n; ++k) {
        const u8 o = order[k];
        u32 j = k;
        while (j > 0 && cost[order[j - 1]] < cost[o]) {
            order[j] = order[j - 1];
            --j;
        }
        order[j] = o;
    }
    for (u32 k = 0; k < n; ++k) {
        const u32 i = order[k];
        const u64 c = q + kChunks + static_cast<u64>(at + i) * kChunkSize;
        if (load<u32>(c + 0x10) > 5 && cost[i] * 20 < top) continue;
        store<u32>(c + 0x10, MUTANT(14) ? 0u : std::min<u32>(5, k / 6));
    }
    g_ranked.fetch_add(1, std::memory_order_relaxed);
}

DECOMP_LEAF u32 render_queue_partition(u64 q, u32 first, u32 last, u32 max_chunks) {
    const u32 ret = bn_of(reinterpret_cast<u64>(__builtin_return_address(0)));
    const bool parallel = from_parallel_for(ret);
    // The game's own where ours has nothing to add: no census, and not a
    // parallel-for's call while the cut or an order by cost wants the costs.
    if (!g_census && !(parallel && learning()))
        return reinterpret_cast<PartitionFn>(g_game_partition)(q, first, last, max_chunks);
    if (parallel && learning()) fold_costs();
    const bool cut = cutting(ret);
    Part* p = nullptr;
    if (g_census) {
        const u64 t = ticks();
        p = &g_parts[g_part_pos.fetch_add(1, std::memory_order_relaxed) & (kPartRing - 1)];
        publish(p->done, 0);
        p->t = t;
        p->queue = q;
        p->ret = ret;
        p->first = first;
        p->last = last;
        const u32 count = load<u32>(q + kCount);
        p->end = count > last ? last : count;
        p->max_chunks = max_chunks;
        p->before = load<u32>(q + kChunkCount);
        p->stealing = load<u8>(q + kStealing);  // and 0x100 below when a run was cut
        p->n_runs = 0;
        // A parallel-for's jobs all run before the next partition is called
        // (its barrier waits for them), so they take its generation.
        p->gen = parallel ? g_gen.fetch_add(1, std::memory_order_relaxed) + 1 : 0;
        p->flip = g_flip.load(std::memory_order_relaxed);
    }
    s8 entries[64];
    const u32 at = load<u32>(q + kChunkCount);
    u32 cut_runs = 0;
    const u32 made = cut ? cut_partition(q, first, last, max_chunks, p, entries, &cut_runs)
                         : partition(q, first, last, max_chunks, p, 0, nullptr, entries);
    if (parallel) g_pf_cut = cut_runs != 0;
    if (cut_runs) {
        g_cut_parts.fetch_add(1, std::memory_order_relaxed);
        g_cut_runs.fetch_add(cut_runs, std::memory_order_relaxed);
    }
    if (g_ranks && parallel) rank_by_cost(q, at, made, entries);
    if (p) {
        if (cut_runs) p->stealing |= 0x100u;
        p->made = made;
        publish(p->done, 1);
    }
    return made;
}

// ---- The chunk job (sub_2d4a400) and its items (sub_12c7470's loop) ----

// sub_12c7470: the queue's items [first, last) - with flags & 1 those marked
// to be drawn first (+0x11), and with flags & 2 the rest too; with flags 2
// only the rest. Each: the queue's prepare (slot 6, which sets up the item's
// pass when its key's upper half changes), the key's fields into the
// context, then each command through the queue's slot 5 (0xff ends the
// stream, 0xfe links to another). `out`: the census's slots, or null; the
// items' ticks summed back.
DECOMP_LEAF u64 run_items(u64 q, u64 ctx, u32 first, u32 last, u32 flags, Item* out, u32 out_n) {
    struct {
        u32 key, pad;
        u64 a, pass;
    } state = {0, 0, 0, 0};
    const u32 count = load<u32>(q + kCount);
    const u32 end = count > last ? last : count;
    if (end <= first) return 0;
    const bool rest = (flags & 2) != 0;
    u64 work = 0;
    u8 chunk_states = 0;
    for (u32 i = first; i != end; ++i) {
        const u64 item = load<u64>(q + kItems) + static_cast<u64>(i) * kItemSize;
        const u8 pre = load<u8>(item + 0x11);
        Item* rec = out && i - first < out_n ? &out[i - first] : nullptr;
        u32 arg;
        if (flags & 1) {
            if (pre == 0 && !rest) {
                if (rec) *rec = Item{load<u32>(item), 0, 0, 0, 0, kSkipped, 0};
                continue;
            }
            arg = pre;
        } else {
            if ((pre != 0 && !MUTANT(5)) || !rest) {
                if (rec) *rec = Item{load<u32>(item), 0, 0, 0, 0, kSkipped, 0};
                continue;
            }
            arg = 0;
        }
        u64 c = load<u64>(item + 8);
        const u64 t0 = out ? ticks() : 0;
        method<PrepareFn>(q, 0x30)(q, ctx, reinterpret_cast<u64>(&state), load<u32>(item), arg);
        const u32 key = load<u32>(item);
        store<u32>(ctx + kCtxKeyTop, key >> 28);
        store<u32>(ctx + kCtxCategory, (key >> 22) & 0x3f);
        store<u32>(ctx + kCtxSub, (key >> 16) & 0x3f);
        store<u32>(ctx + kCtxLow, MUTANT(6) ? key : key & 0xffff);
        u16 states = 0, draws = 0, other = 0;
        u8 item_states = 0, inherited = 0;
        for (;;) {
            const u8 type = load<u8>(c);
            if (type == 0xff) break;
            if (type == 0xfe) {
                c = load<u64>(c + load<u8>(c + 1) + 4);
                continue;
            }
            const u64 h = load<u8>(c + 1);
            const u64 size = static_cast<u64>(load<u16>(c + 2)) - 4 - h;
            const u64 payload = size != 0 || MUTANT(7) ? c + h + 4 : 0;
            if (out) {
                switch (type) {
                case 1: item_states |= kState1; ++states; break;
                case 2: item_states |= kState2; ++states; break;
                case 6: item_states |= kState6; ++states; break;
                case 7: item_states |= kState7; ++states; break;
                case 3:
                case 4:
                case 0xf8:
                    inherited |= static_cast<u8>(chunk_states & ~item_states);
                    ++draws;
                    break;
                default: ++other; break;
                }
            }
            method<CommandFn>(q, 0x28)(q, ctx, type, payload, size);
            c += load<u16>(c + 2);
        }
        if (out) {
            const u64 dt = ticks() - t0;
            work += dt;
            chunk_states |= item_states;
            if (rec)
                *rec = Item{key, static_cast<u32>(dt > 0xffffffffull ? 0xffffffffull : dt), states, draws, other,
                            static_cast<u8>(inherited ? kInherits : 0), inherited};
        }
    }
    return work;
}

// A chunk job's ticks by its entry (its first item's), for the cut's rule
// and the orders by cost.
DECOMP_LEAF void learn_chunk(u64 q, u64 range, u64 dt) {
    const u32 first = load<u32>(range), last = load<u32>(range + 4), count = load<u32>(q + kCount);
    const u32 end = count > last ? last : count;
    if (end <= first) return;
    const u64 table = game_address(kTable);
    const s8 e = entry_index(entry_of(load<u32>(load<u64>(q + kItems) + static_cast<u64>(first) * kItemSize), table), table);
    if (e < 0) return;
    g_cost_ticks[static_cast<u32>(e)].fetch_add(dt, std::memory_order_relaxed);
    g_cost_items[static_cast<u32>(e)].fetch_add(end - first, std::memory_order_relaxed);
}

// ---- A plugin's segment ahead of a scene category (plugin API 15) ----
//
// The game's debug draws (RendMan's draw lists: the collision, the hit
// capsules) are played after the scene into the display buffers, over the
// effects and the post-processing. A plugin that wants one drawn inside the
// scene - under the effects, against the scene's depth - is given a command
// context of its own on the worker that records the first chunk at or after
// the pass it names - a key's category and sub-category, (bits 22-27 << 6) |
// bits 16-21: the chunk's first item's pass set up on it (the queue's
// prepare: its targets bound), then the plugin's recording, queued under the
// chunk's own key. The frame's end sorts the segments by key, stably, and ours
// is appended first, so it plays just before the chunk. A chunk whose item
// before it is already at or after the pass is not the first and gets none;
// a frame without that pass gets the segment ahead of the next one.
std::atomic<int> g_insert_pass{-1};
std::atomic<void (*)(u64, void*)> g_insert_fn{nullptr};
std::atomic<void*> g_insert_user{nullptr};

// BBHOST_SCENE_CATEGORIES=1: each queue's items by pass, logged every 300
// flips - which passes a frame's queues hold (what to name above).
bool g_log_categories = false;
std::mutex g_categories_mu;
std::map<u64, std::map<u32, u32>> g_categories;  // queue -> items by pass, under g_categories_mu

DECOMP_LEAF void count_categories(u64 q, u64 range) {
    const u32 first = load<u32>(range), last = load<u32>(range + 4), count = load<u32>(q + kCount);
    const u32 end = count > last ? last : count;
    if (end <= first) return;
    const u64 items = load<u64>(q + kItems);
    std::map<u32, u32> n;
    for (u32 i = first; i != end; ++i) ++n[(load<u32>(items + static_cast<u64>(i) * kItemSize) >> 16) & 0xfff];
    std::lock_guard<std::mutex> lk(g_categories_mu);
    auto& total = g_categories[q];
    for (const auto& [pass, k] : n) total[pass] += k;
}

void log_categories(u32 flips) {
    std::map<u64, std::map<u32, u32>> now;
    {
        std::lock_guard<std::mutex> lk(g_categories_mu);
        now.swap(g_categories);
    }
    for (const auto& [q, n] : now) {
        std::string line;
        for (const auto& [pass, k] : n) {
            char part[32];
            std::snprintf(part, sizeof(part), " %02x/%02x:%.1f", pass >> 6, pass & 0x3f, k / 300.0);
            line += part;
        }
        host_log("scene categories: queue %llx at flip %u, items a flip by category/sub-category:%s", static_cast<unsigned long long>(q), flips,
                 line.c_str());
    }
}

DECOMP_LEAF void scene_insert(u64 functor, u64 range, u32 worker) {
    const u64 q = load<u64>(functor + 0x10);
    if (g_log_categories) count_categories(q, range);
    const int want = g_insert_pass.load(std::memory_order_acquire);
    const auto fn = g_insert_fn.load(std::memory_order_acquire);
    if (want < 0 || !fn) return;
    const u32 first = load<u32>(range), last = load<u32>(range + 4), count = load<u32>(q + kCount);
    const u32 end = count > last ? last : count;
    if (first >= end) return;
    const u64 items = load<u64>(q + kItems);
    const auto pass_of = [items](u32 i) { return (load<u32>(items + static_cast<u64>(i) * kItemSize) >> 16) & 0xfff; };
    // The chunk holding the queue's first item at or after the pass: a table
    // entry can run across passes (3e/21-3e/25 is one), so that item need not
    // start its chunk.
    u32 at = first;
    while (at != end && pass_of(at) < static_cast<u32>(want)) ++at;
    if (at == end || (at == first && first > 0 && pass_of(first - 1) >= static_cast<u32>(want))) return;
    const u32 key = load<u32>(items + static_cast<u64>(first) * kItemSize);
    const u64 gx = load<u64>(game_address(kGx));
    const u64 per_worker = load<u64>(load<u64>(gx + 0x58) + static_cast<u64>(worker) * 8);
    const u64 parent = load<u64>(functor + 8);
    const u64 ctx = method<ContextFn>(parent, 0x80)(parent, per_worker, 0);
    if (!ctx) return;
    const u64 cc = load<u64>(ctx + kCtxCommand);
    game_function<OneFn>(kBeginRecord)(cc);
    struct {
        u32 key, pad;
        u64 a, pass;
    } state = {0, 0, 0, 0};
    method<PrepareFn>(q, 0x30)(q, ctx, reinterpret_cast<u64>(&state), key, 0);
    store<u32>(ctx + kCtxKeyTop, key >> 28);
    store<u32>(ctx + kCtxCategory, (key >> 22) & 0x3f);
    store<u32>(ctx + kCtxSub, (key >> 16) & 0x3f);
    store<u32>(ctx + kCtxLow, key & 0xffff);
    fn(cc, g_insert_user.load(std::memory_order_acquire));
    game_function<OneFn>(kEndRecord)(cc);
    game_function<QueueFn>(kQueueContext)(gx, cc, (key & 0xffff0000u) | (first & 0xffffu));
    const u64 owner = game_function<OneFn>(kOwnerOf)(ctx);
    method<OneFn>(ctx, 0)(ctx);
    method<TwoFn>(owner, 0x70)(owner, ctx);
}

DECOMP_LEAF u64 render_queue_chunk_job(u64 functor, u64 range, u32 worker) {
    scene_insert(functor, range, worker);
    const bool learn = learning();
    if (!g_census) {
        // The game's job, timed when the costs are wanted.
        const auto game = reinterpret_cast<ChunkJobFn>(g_game_chunk_job);
        if (!learn) return game(functor, range, worker);
        const u64 t0 = ticks();
        const u64 ret = game(functor, range, worker);
        learn_chunk(load<u64>(functor + 0x10), range, ticks() - t0);
        return ret;
    }
    // The census's: ours, every item recorded.
    const u64 t0 = ticks();
    Chunk* rec = &g_chunk_ring[g_chunk_pos.fetch_add(1, std::memory_order_relaxed) & (kChunkRing - 1)];
    publish(rec->done, 0);
    Item* out = nullptr;
    u32 out_n = 0;
    {
        const u64 q = load<u64>(functor + 0x10);
        const u32 first = load<u32>(range), last = load<u32>(range + 4), count = load<u32>(q + kCount);
        const u32 end = count > last ? last : count;
        out_n = end > first ? std::min(end - first, kChunkItems) : 0;
        // Its item slots, contiguous: slots that would pass the ring's end are not kept.
        const u32 slot = static_cast<u32>(g_item_pos.fetch_add(out_n, std::memory_order_relaxed) & (kItemRing - 1));
        if (slot + out_n <= kItemRing) out = &g_items[slot];
        else out_n = 0;
        rec->t0 = t0;
        rec->queue = q;
        rec->first = first;
        rec->last = last;
        rec->item_at = slot;
        rec->n_items = out_n;
        rec->gen = g_gen.load(std::memory_order_relaxed);
        rec->worker = worker;
    }
    const u64 gx = load<u64>(game_address(kGx));
    const u64 per_worker = load<u64>(load<u64>(gx + 0x58) + static_cast<u64>(worker) * 8);
    const u64 parent = load<u64>(functor + 8);
    const u64 ctx = method<ContextFn>(parent, 0x80)(parent, per_worker, 0);
    const u64 q = load<u64>(functor + 0x10);
    const u64 cc = load<u64>(ctx + kCtxCommand);
    game_function<OneFn>(kBeginRecord)(cc);
    const u64 lead = ticks() - t0;
    const u64 work = run_items(q, ctx, load<u32>(range), load<u32>(range + 4), MUTANT(8) ? 3 : 2, out, out_n);
    game_function<OneFn>(kEndRecord)(cc);
    const u32 first = load<u32>(range);
    const u32 key = (load<u32>(load<u64>(q + kItems) + static_cast<u64>(first) * kItemSize) & 0xffff0000u) | (first & 0xffffu);
    u64 ret = game_function<QueueFn>(kQueueContext)(load<u64>(game_address(kGx)), cc, MUTANT(9) ? key + 1 : key);
    if (ctx) {
        const u64 owner = game_function<OneFn>(kOwnerOf)(ctx);
        method<OneFn>(ctx, 0)(ctx);
        ret = method<TwoFn>(owner, 0x70)(owner, ctx);
    }
    rec->key = key;
    rec->work = static_cast<u32>(work > 0xffffffffull ? 0xffffffffull : work);
    rec->lead = static_cast<u32>(lead);
    rec->t1 = ticks();
    publish(rec->done, rec->gen + 1);
    if (learn) learn_chunk(q, range, ticks() - t0);
    return ret;
}

// ---- The parallel-for (sub_12c7810), for BBHOST_RQ_PUSH ----

// The chunks [b, e) of a queue's table in place, the dearest first (their
// costs by entry and item), ties in the table's order: an insertion sort over
// at most 63 records of 0x18 bytes.
DECOMP_LEAF void sort_by_cost(u64 q, u64 b, u64 e) {
    const u32 n = static_cast<u32>((e - b) / kChunkSize);
    if (n < 2 || n > 64) return;
    const u64 table = game_address(kTable);
    const u64 items = load<u64>(q + kItems);
    u64 cost[64];
    u8 rec[64][kChunkSize];
    for (u32 k = 0; k < n; ++k) {
        const u64 c = b + static_cast<u64>(k) * kChunkSize;
        const u32 f = load<u32>(c), l = load<u32>(c + 4);
        cost[k] = l > f ? chunk_cost(l - f, entry_index(entry_of(load<u32>(items + static_cast<u64>(f) * kItemSize), table), table)) : 0;
        for (u32 j = 0; j < kChunkSize; ++j) rec[k][j] = load<u8>(c + j);
    }
    u8 order[64];
    for (u32 k = 0; k < n; ++k) order[k] = static_cast<u8>(k);
    for (u32 k = 1; k < n; ++k) {
        const u8 o = order[k];
        u32 j = k;
        while (j > 0 && cost[order[j - 1]] < cost[o]) {
            order[j] = order[j - 1];
            --j;
        }
        order[j] = o;
    }
    for (u32 k = 0; k < n; ++k)
        for (u32 j = 0; j < kChunkSize; ++j) store<u8>(b + static_cast<u64>(k) * kChunkSize + j, rec[order[k]][j]);
    g_pushed.fetch_add(1, std::memory_order_relaxed);
}

// GX's parallel-for over a queue's items [first, last): the partition (the
// queue's slot 3), the new chunks sorted by rank when the scene's Enable Task
// Sort is on - all but the table's last chunk, which the game's sort is not
// given - then one job a chunk (the chunk job, sub_2d4a400, through the
// functor on this frame) pushed in that order, a rank up to 5 to participant
// i mod n and the rest to participant 0; the participants woken, and GX's job
// barrier waited in. BBHOST_RQ_PUSH: the same, sorted dearest first instead
// once the costs are known. Returns the chunks made.
DECOMP_LEAF u32 render_queue_parallel_for(u64 q, u64 ctx, u32 first, u32 last, u32 limit) {
    // The game's own while no order but its own can follow: the push not
    // forced, and no cut to follow.
    if (g_push_mode != Push::On && !(g_push_mode == Push::WithCut && g_cut_on.load(std::memory_order_relaxed)))
        return static_cast<u32>(reinterpret_cast<ParallelForFn>(g_game_parallel_for)(q, ctx, first, last, limit));
    g_in_parallel_for = true;
    g_pf_cut = false;
    const u32 n = method<PartitionFn>(q, 0x18)(q, first, last, limit);
    g_in_parallel_for = false;
    const bool cut = g_pf_cut;
    struct {
        u64 vtable, ctx, queue;
    } functor = {game_address(kFunctorVtable), ctx, q};
    const u64 gx = load<u64>(game_address(kGx));
    const u64 jobs = gx + 0x278;
    const u32 participants = load<u32>(gx + 0x280);
    const bool sorted = load<u8>(game_address(kTaskSort)) != 0;
    const u32 total = load<u32>(q + kChunkCount);
    u32 i = total - n;
    if (sorted) {
        const u64 b = q + kChunks + static_cast<u64>(i) * kChunkSize;
        const u64 e = q + kChunks + static_cast<u64>(total - 1) * kChunkSize;
        bool known = false;
        for (u32 k = 0; k < kTableEntries && !known; ++k) known = g_cost_per_item[k] != 0;
        if (known && push_after(cut) && !MUTANT(15))
            sort_by_cost(q, b, e);
        else
            game_function<SortFn>(kSortByRank)(b, e, (e - b) / kChunkSize);
    }
    game_function<VoidFn>(kWaitFirst)();
    for (; i < load<u32>(q + kChunkCount); ++i) {
        const u64 c = q + kChunks + static_cast<u64>(i) * kChunkSize;
        u32 worker = 0;
        if (load<u8>(game_address(kTaskSort)) != 0 && load<u32>(c + 0x10) <= 5) worker = i % participants;
        struct {
            u32 first, last, grain;
        } range;
        range.first = load<u32>(c);
        range.last = load<u32>(c + 4);
        const s32 span = static_cast<s32>(range.last - range.first), size = load<s32>(c + 8);
        range.grain = static_cast<u32>(span > size ? size : span);
        const u64 job = game_function<JobForFn>(kJobFor)(host_address(&functor), host_address(&range), worker);
        game_function<PushFn>(kPushJob)(jobs, job, worker);
    }
    game_function<VoidFn>(kAfterPush)();
    game_function<WakeFn>(kWake)(jobs, 0, limit);
    game_function<OneFn>(kBarrier)(jobs);
    return n;
}

// ---- The frame's segment list (sub_25b4a00): the key noted, the game's append ----

void* g_game_append = nullptr;

DECOMP_LEAF u64 render_queue_list_append(u64 list, u64 record, u32 key) {
    if (g_census) {
        Append* a = &g_appends[g_append_pos.fetch_add(1, std::memory_order_relaxed) & (kAppendRing - 1)];
        publish(a->done, 0);
        a->t = ticks();
        a->list = list;
        a->key = key;
        const u32 last = load<u32>(list + 0x40);
        a->next = (last & 0xf03fffffu) | ((last + 0x400000u) & 0x0fc00000u);
        a->ret = bn_of(reinterpret_cast<u64>(__builtin_return_address(0)));
        a->gen = g_gen.load(std::memory_order_relaxed);
        publish(a->done, 1);
    }
    return reinterpret_cast<AppendFn>(g_game_append)(list, record, key);
}

// ---- Compare runs: the game's partition does the work, ours must agree ----

DecompCompare g_partition_counts;
std::atomic<int> g_compare_logs{0};
std::atomic<u64> g_cut_refined{0};  // compare runs: pieces past the game's chunks, all refining them

constexpr u64 kTableBytes = static_cast<u64>(kChunkMax) * kChunkSize;

inline void* at_(u64 a) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(a)); }

// Whether the chunks ours made, [at, at + ours), are the game's [at, at +
// game) each cut into pieces: every game chunk is a run of ours, back to back
// over the same items, each piece with the chunk's entry fields (its rank
// too, unless ranks are by cost), its size its own count (the grain, under
// task stealing, is not compared); an empty chunk stays one empty chunk.
bool refines(u64 q, u32 at, u32 ours, const u8* game, u32 theirs) {
    const bool stealing = load<u8>(q + kStealing) != 0;
    u32 o = 0;
    for (u32 g = 0; g < theirs; ++g) {
        const u8* gc = game + static_cast<u64>(at + g) * kChunkSize;
        u32 gf, gl, gd6, grank;
        std::memcpy(&gf, gc, 4);
        std::memcpy(&gl, gc + 4, 4);
        std::memcpy(&gd6, gc + 0xc, 4);
        std::memcpy(&grank, gc + 0x10, 4);
        const u8 gb8 = gc[0x14];
        u32 pos = gf;
        do {
            if (o >= ours) return false;
            const u64 c = q + kChunks + static_cast<u64>(at + o) * kChunkSize;
            const u32 f = load<u32>(c), l = load<u32>(c + 4);
            if (f != pos || l < f || l > gl || (l == f && gl != gf)) return false;
            if (load<u32>(c + 0xc) != gd6 || load<u8>(c + 0x14) != gb8 || (!g_ranks && load<u32>(c + 0x10) != grank)) return false;
            if (!stealing && load<u32>(c + 8) != l - f) return false;
            pos = l;
            ++o;
        } while (pos < gl);
    }
    return o == ours;
}

GUEST_ABI u32 compare_partition(u64 q, u32 first, u32 last, u32 max_chunks) {
    // the queue's table as far as a partition can write it: 63 chunks and the
    // one past them, which a full table's last chunk takes
    constexpr u64 kBytes = kTableBytes + kChunkSize;
    u8 before[kBytes], game[kBytes];
    std::memcpy(before, at_(q + kChunks), kBytes);
    const u32 count_before = load<u32>(q + kChunkCount);
    const u32 theirs = static_cast<u32>(hle_call_guest<u64>(g_game_partition, q, first, last, max_chunks));
    std::memcpy(game, at_(q + kChunks), kBytes);
    const u32 game_count = load<u32>(q + kChunkCount);
    std::memcpy(at_(q + kChunks), before, kBytes);
    store<u32>(q + kChunkCount, count_before);
    // The execute's own calls pass a limit of 1, where no table cuts a run;
    // the parallel-for's fold the costs first, as ours does.
    if (max_chunks >= 2 && learning()) fold_costs();
    const bool cut = max_chunks >= 2 && cut_this_flip();
    u32 runs = 0;
    const u32 ours = cut ? cut_partition(q, first, last, max_chunks, nullptr, nullptr, &runs) : partition(q, first, last, max_chunks, nullptr);
    if (runs) {
        g_cut_parts.fetch_add(1, std::memory_order_relaxed);
        g_cut_runs.fetch_add(runs, std::memory_order_relaxed);
    }
    bool same;
    if (!cut) {
        same = ours == theirs && load<u32>(q + kChunkCount) == game_count && std::memcmp(at_(q + kChunks), game, kBytes) == 0;
    } else {
        same = load<u32>(q + kChunkCount) == count_before + ours && game_count == count_before + theirs &&
               refines(q, count_before, ours, game, theirs);
        if (same && ours > theirs) g_cut_refined.fetch_add(ours - theirs, std::memory_order_relaxed);
    }
    g_partition_counts.calls.fetch_add(1, std::memory_order_relaxed);
    if (!same) {
        g_partition_counts.differ.fetch_add(1, std::memory_order_relaxed);
        if (g_compare_logs.fetch_add(1) < 16)
            host_log("decomp: render_queue_partition %s the game's: items %u-%u, %u chunks made (game %u), count %u (game %u)",
                     cut ? "does not refine" : "differs from", first, last, ours, theirs, load<u32>(q + kChunkCount), game_count);
    }
    std::memcpy(at_(q + kChunks), game, kBytes);
    store<u32>(q + kChunkCount, game_count);
    return theirs;
}

// ---- The fold: our thread, every 25 ms ----

struct Gen {  // one parallel-for
    Part part;
    bool have_part = false;
    std::vector<Chunk> chunks;
    std::vector<std::vector<Item>> items;
};

struct Window {
    u64 flips = 0, gens = 0, chunks = 0, items = 0, drawn = 0, draws = 0, appends = 0;
    double phase = 0, crit = 0, work = 0, crit_items = 0, crit_largest = 0, bound = 0;
    double inherits = 0;            // items that drew with an earlier item's state
    std::map<int, double> crit_by;  // the critical chunk's entry: its ticks
    std::map<int, double> work_by;  // the chunks' work by entry
    u64 overflow = 0, last_cut = 0;  // runs past 62 chunks; last runs whose entry allows more than one chunk
};

std::string entry_name(int e) {
    if (e < 0) return "none";
    const u64 t = game_address(kTable) + static_cast<u64>(e) * kTableEntry;
    char b[32];
    const u32 sub = load<u32>(t + 4);
    if (sub == 0xffffffffu)
        std::snprintf(b, sizeof b, "%02x", load<u32>(t));
    else
        std::snprintf(b, sizeof b, "%02x/%02x", load<u32>(t), sub);
    return b;
}

class Fold {
public:
    void start();
    void flip(u32 n) { flips_.store(n, std::memory_order_relaxed); }
    void report();

private:
    void run();
    void step(bool all);
    void close(Gen& g);
    void line(const Window& w);
    double ms(double t) const { return tick_per_ms_ > 0 ? t / tick_per_ms_ : 0; }

    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<u32> flips_{0};
    std::mutex mu_;
    u64 part_read_ = 0, chunk_read_ = 0, append_read_ = 0;
    std::map<u32, Gen> gens_;
    Window win_;
    u32 win_from_ = 0;
    double tick_per_ms_ = 0;
    u64 t_start_ = 0;
    std::chrono::steady_clock::time_point c_start_;
    FILE* out_ = nullptr;
    u32 every_ = 300;
};

Fold g_fold;

void Fold::start() {
    if (const char* e = std::getenv("BBHOST_RQ_CENSUS_FLIPS"); e && std::atoi(e) > 0) every_ = static_cast<u32>(std::atoi(e));
    if (const char* p = std::getenv("BBHOST_RQ_CENSUS_OUT"); p && *p) {
        out_ = std::fopen(p, "wb");
        if (!out_) host_log("rq census: cannot write %s; lines only", p);
    }
    if (out_) {
        const char magic[8] = {'R', 'Q', 'C', 'E', 'N', 'S', '1', '\n'};
        std::fwrite(magic, 1, sizeof magic, out_);
    }
    t_start_ = ticks();
    c_start_ = std::chrono::steady_clock::now();
    thread_ = std::thread([this] { run(); });
}

void Fold::run() {
    while (!stop_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        std::lock_guard<std::mutex> lk(mu_);
        step(false);
    }
}

// The next record taken, once written (`all`: written or not, at the end).
template <class R>
bool take(R* ring, u32 size, u64& read, std::atomic<u64>& pos, R& got, bool all) {
    const u64 end = pos.load(std::memory_order_acquire);
    if (read >= end) return false;
    if (end - read > size) read = end - size;  // fallen behind: the oldest were written over
    R& r = ring[read & (size - 1)];
    if (seen(r.done) == 0 && !all) return false;
    std::memcpy(static_cast<void*>(&got), static_cast<const void*>(&r), sizeof(R));
    ++read;
    return true;
}

void Fold::step(bool all) {
    const double since = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c_start_).count();
    if (since > 50) tick_per_ms_ = static_cast<double>(ticks() - t_start_) / since;
    Part p;
    while (take(g_parts, kPartRing, part_read_, g_part_pos, p, all)) {
        if (out_) {
            std::fputc('P', out_);
            std::fwrite(&p, sizeof p, 1, out_);
        }
        if (p.gen == 0) continue;  // the execute's own partitions, run on its thread
        Gen& g = gens_[p.gen];
        g.part = p;
        g.have_part = true;
    }
    Chunk c;
    while (take(g_chunk_ring, kChunkRing, chunk_read_, g_chunk_pos, c, all)) {
        std::vector<Item> items(c.n_items);
        if (c.n_items) std::memcpy(static_cast<void*>(items.data()), &g_items[c.item_at], sizeof(Item) * c.n_items);
        if (out_) {
            std::fputc('C', out_);
            std::fwrite(&c, sizeof c, 1, out_);
            if (c.n_items) std::fwrite(items.data(), sizeof(Item), c.n_items, out_);
        }
        // A job of a later parallel-for closes the earlier ones: their jobs
        // were all taken before it.
        for (auto it = gens_.begin(); it != gens_.end() && it->first < c.gen;) {
            close(it->second);
            it = gens_.erase(it);
        }
        Gen& g = gens_[c.gen];
        g.chunks.push_back(c);
        g.items.push_back(std::move(items));
    }
    Append a;
    while (take(g_appends, kAppendRing, append_read_, g_append_pos, a, all)) {
        if (out_) {
            std::fputc('A', out_);
            std::fwrite(&a, sizeof a, 1, out_);
        }
        ++win_.appends;
    }
    if (out_) std::fflush(out_);
    const u32 f = flips_.load(std::memory_order_relaxed);
    if (win_from_ == 0) win_from_ = f;
    if (f >= win_from_ + every_) {
        win_.flips = f - win_from_;
        line(win_);
        win_ = Window{};
        win_from_ = f;
    }
}

void Fold::close(Gen& g) {
    if (!g.have_part || g.chunks.empty()) return;
    Window& w = win_;
    ++w.gens;
    u64 t_end = 0;
    std::size_t crit = 0;
    double work = 0, biggest = 0;
    for (std::size_t k = 0; k < g.chunks.size(); ++k) {
        if (g.chunks[k].t1 > t_end) {
            t_end = g.chunks[k].t1;
            crit = k;
        }
        work += g.chunks[k].work;
    }
    w.phase += t_end > g.part.t ? static_cast<double>(t_end - g.part.t) : 0;
    w.work += work;
    w.chunks += g.chunks.size();
    // Each chunk's entry: the partition's runs, in order.
    std::vector<std::pair<u32, int>> starts;
    for (u32 r = 0; r < std::min(g.part.n_runs, kRunsKept); ++r) {
        const Run& run = g.part.runs[r];
        if (run.how == kHowOverflow) ++w.overflow;
        if (run.how == kHowLast && run.entry >= 0 && run.n >= 10) {
            const u64 e = game_address(kTable) + static_cast<u64>(run.entry) * kTableEntry;
            if ((load<u32>(e + 0x10) ? load<u32>(e + 0xc) : load<u32>(e + 8)) > 1) ++w.last_cut;
        }
        starts.emplace_back(run.start, run.entry);
    }
    const auto entry_at = [&](u32 first) {
        int e = -1;
        for (const auto& [s, en] : starts)
            if (s <= first) e = en;
        return e;
    };
    double largest_crit = 0;
    for (std::size_t k = 0; k < g.chunks.size(); ++k) {
        w.work_by[entry_at(g.chunks[k].first)] += g.chunks[k].work;
        double largest = 0;
        for (const Item& it : g.items[k]) {
            ++w.items;
            if (!(it.flags & kSkipped)) ++w.drawn;
            w.draws += it.draws;
            if (it.flags & kInherits) w.inherits += 1;
            largest = std::max(largest, static_cast<double>(it.ticks));
        }
        biggest = std::max(biggest, largest);
        if (k == crit) largest_crit = largest;
    }
    const Chunk& cc = g.chunks[crit];
    const double crit_t = static_cast<double>(cc.t1 - cc.t0);
    w.crit += crit_t;
    w.crit_by[entry_at(cc.first)] += crit_t;
    w.crit_items += static_cast<double>(cc.last - cc.first);
    w.crit_largest += cc.work ? largest_crit / cc.work : 0;
    // The balanced bound: the work spread evenly over the six participants,
    // and no less than the largest item.
    w.bound += std::max(work / 6.0, biggest);
}

void Fold::line(const Window& w) {
    if (!w.flips || !w.gens) return;
    const double f = static_cast<double>(w.flips);
    std::string crit_by, work_by;
    std::vector<std::pair<double, int>> v;
    for (const auto& [e, t] : w.crit_by) v.emplace_back(t, e);
    std::sort(v.rbegin(), v.rend());
    for (std::size_t i = 0; i < v.size() && i < 4; ++i) {
        char b[48];
        std::snprintf(b, sizeof b, "%s%s %.0f%%", i ? ", " : "", entry_name(v[i].second).c_str(), 100.0 * v[i].first / w.crit);
        crit_by += b;
    }
    v.clear();
    for (const auto& [e, t] : w.work_by) v.emplace_back(t, e);
    std::sort(v.rbegin(), v.rend());
    for (std::size_t i = 0; i < v.size() && i < 5; ++i) {
        char b[48];
        std::snprintf(b, sizeof b, "%s%s %.2f", i ? ", " : "", entry_name(v[i].second).c_str(), ms(v[i].first) / f);
        work_by += b;
    }
    host_log("rq census: %llu flips: %.2f parallel-fors, %.0f chunks, %.0f items (%.0f drawn, %.0f draw commands) a flip; phase "
             "%.2f ms a flip, chunk work %.2f ms (%.2f over six); the critical chunk %.2f ms, %.0f items, its largest item %.0f%% "
             "of it, by entry: %s; work by entry (ms a flip): %s; balanced bound %.2f ms (%.2f under the phase); items drawing "
             "with an earlier item's state %.1f a flip; runs past 62 chunks %llu, last runs held to one chunk %llu; %.0f segments "
             "a flip",
             static_cast<unsigned long long>(w.flips), w.gens / f, w.chunks / f, w.items / f, w.drawn / f, w.draws / f,
             ms(w.phase) / f, ms(w.work) / f, ms(w.work) / f / 6.0, ms(w.crit) / f, w.crit_items / w.gens,
             100.0 * w.crit_largest / w.gens, crit_by.c_str(), work_by.c_str(), ms(w.bound) / f, ms(w.phase - w.bound) / f,
             w.inherits / f, static_cast<unsigned long long>(w.overflow), static_cast<unsigned long long>(w.last_cut),
             w.appends / f);
}

void Fold::report() {
    if (thread_.joinable()) {
        stop_.store(true);
        thread_.join();
    }
    std::lock_guard<std::mutex> lk(mu_);
    step(true);
    for (auto& [gen, g] : gens_) close(g);
    gens_.clear();
    win_.flips = flips_.load() - win_from_;
    line(win_);
    if (out_) {
        std::fclose(out_);
        out_ = nullptr;
    }
}

void render_queue_report() {
    if (g_census) {
        g_fold.report();
        host_log("rq census: %llu partitions, %llu chunk jobs, %llu items, %llu segments recorded",
                 static_cast<unsigned long long>(g_part_pos.load()), static_cast<unsigned long long>(g_chunk_pos.load()),
                 static_cast<unsigned long long>(g_item_pos.load()), static_cast<unsigned long long>(g_append_pos.load()));
    }
    if (g_split_mode != Split::Off)
        host_log("decomp: the render queue's cut (%s): %llu parallel-fors with the PC build's limits%s, %llu of them cutting %llu tail "
                 "runs (%llu items or the even share of the work), %llu made with the game's table for want of room%s",
                 g_split_mode == Split::Auto ? "above 60 fps" : "BBHOST_RQ_SPLIT=1",
                 static_cast<unsigned long long>(g_cut_calls.load()), g_cut_ready ? "" : " (the table not the one written against: off)",
                 static_cast<unsigned long long>(g_cut_parts.load()), static_cast<unsigned long long>(g_cut_runs.load()),
                 static_cast<unsigned long long>(kTailItems), static_cast<unsigned long long>(g_cut_fallbacks.load()),
                 decomp_comparing() ? (std::string("; compared: ") + std::to_string(g_cut_refined.load()) +
                                       " pieces past the game's chunks, each refining them").c_str()
                                    : "");
    if (g_ranks) host_log("decomp: the render queue's ranks by cost: %llu parallel-fors", static_cast<unsigned long long>(g_ranked.load()));
    if (g_push_placed)
        host_log("decomp: the render queue pushed longest first: %llu parallel-fors", static_cast<unsigned long long>(g_pushed.load()));
}

// The cut table, from the game's (placement, once the image's slide is known):
// the two entries it changes must be the ones the limits were written for.
bool make_cut_table(const std::uint8_t*) {
    const u64 t = game_address(kTable);
    for (u32 k = 0; k < kTableWords; ++k) g_cut_table[k] = load<u32>(t + 4ull * k);
    static constexpr u32 kGame[][9] = {{0x3e, 0, 1, 0, 0, 5, 0x08, 5, 0}, {0x3e, 5, 1, 0, 0, 5, 0x09, 3, 0}};
    bool as_written = true;
    for (u32 k = 0; k < 2; ++k)
        for (u32 j = 0; j < 9; ++j) as_written &= g_cut_table[kCutLimits[k].entry * 9 + j] == kGame[k][j];
    if (as_written) {
        for (const CutLimit& c : kCutLimits) {
            g_cut_table[c.entry * 9 + 3] = c.cap;
            g_cut_table[c.entry * 9 + 4] = c.divisor;
        }
    }
    g_cut_ready = as_written;
    return true;  // ours is the game's partition either way
}

}  // namespace

void render_queue_flip(std::uint32_t flips, float limit_fps) {
    g_flip.store(flips, std::memory_order_relaxed);
    if (g_census) g_fold.flip(flips);
    if (g_log_categories && flips % 300 == 0) log_categories(flips);
    if (g_split_mode != Split::Auto || !g_cut_ready) return;
    // Above 60: 90, 120, or uncapped's limit (a display's 59.94 or 60.01 Hz is 60).
    const bool on = std::lround(limit_fps) > 60;
    g_cut_on.store(on, std::memory_order_relaxed);
    if (g_said_cut.exchange(on ? 1 : 0, std::memory_order_relaxed) != (on ? 1 : 0))
        host_log("decomp: the render queue's cut %s at %.0f fps (%s)", on ? "on" : "off", static_cast<double>(limit_fps),
                 on ? (g_push_mode == Push::WithCut ? "above 60: its tail runs, and those frames' jobs pushed longest first"
                                                    : "above 60: its tail runs")
                    : "60 or less: the game's partitions");
}

void render_queue_scene_insert(int pass, void (*fn)(std::uint64_t cc, void* user), void* user) {
    const bool on = pass >= 0 && pass < 0x1000 && fn;
    g_insert_pass.store(-1, std::memory_order_release);
    g_insert_fn.store(on ? fn : nullptr, std::memory_order_release);
    g_insert_user.store(on ? user : nullptr, std::memory_order_release);
    if (on) g_insert_pass.store(pass, std::memory_order_release);
    if (on)
        host_log("decomp: a plugin's segment in the scene ahead of its first chunk at or after %02x/%02x", pass >> 6, pass & 0x3f);
    else
        host_log("decomp: the plugin's segment in the scene taken away");
}

void decomp_gx_render_queue_add() {
    const auto on = [](const char* name, bool otherwise) {
        const char* e = std::getenv(name);
        return e && *e ? e[0] != '0' : otherwise;
    };
    const auto mode = [](const char* name) {
        const char* e = std::getenv(name);
        return !e || !*e ? 0 : e[0] != '0' ? 1 : 2;  // unset, on, off
    };
    g_census = on("BBHOST_RQ_CENSUS", false);
    const int split = mode("BBHOST_RQ_SPLIT"), push = mode("BBHOST_RQ_PUSH");
    g_split_mode = split == 1 ? Split::On : split == 2 ? Split::Off : Split::Auto;
    g_push_mode = push == 1 ? Push::On : push == 2 ? Push::Off : Push::WithCut;
    g_cut_on.store(g_split_mode == Split::On, std::memory_order_relaxed);
    if (const char* e = std::getenv("BBHOST_RQ_SPLIT_EVERY"); e && std::atoi(e) > 0) g_split_every = static_cast<u32>(std::atoi(e));
    g_ranks = on("BBHOST_RQ_RANKS", false);
    g_log_categories = on("BBHOST_SCENE_CATEGORIES", false);
    // What could be on in this run: Auto follows the frame rate's limit, which V-Sync moves while uncapped.
    const bool may_cut = g_split_mode != Split::Off;
    g_push_placed = g_push_mode == Push::On || (g_push_mode == Push::WithCut && may_cut);
    const bool may_learn = g_ranks || g_push_mode == Push::On || may_cut;
    if (g_census) {
        g_items = static_cast<Item*>(std::calloc(kItemRing, sizeof(Item)));
        g_chunk_ring = static_cast<Chunk*>(std::calloc(kChunkRing, sizeof(Chunk)));
        g_parts = static_cast<Part*>(std::calloc(kPartRing, sizeof(Part)));
        g_appends = static_cast<Append*>(std::calloc(kAppendRing, sizeof(Append)));
        if (!g_items || !g_chunk_ring || !g_parts || !g_appends) {
            host_log("rq census: no memory for its rings; off");
            g_census = false;
        } else {
            g_fold.start();
            host_log("rq census: on (the render queue's partition, chunk job and segment list as ours)");
        }
    }
    // The partition stays the game's unless the census, the cut or the costs
    // want ours; the chunk job is always placed, for on_scene_category.
    DecompFunction part{"render_queue_partition", "gx", kPartition, kPartitionEntry, sizeof(kPartitionEntry),
                        reinterpret_cast<void*>(&render_queue_partition), DecompKind::Leaf};
    part.original = &g_game_partition;
    part.compare = reinterpret_cast<void*>(&compare_partition);
    part.counts = &g_partition_counts;
    part.report = &render_queue_report;
    part.body_ok = &make_cut_table;
    if (g_census || may_cut || may_learn) decomp_add(part);
    {
        DecompFunction job{"render_queue_chunk_job", "gx", kChunkJob, kChunkJobEntry, sizeof(kChunkJobEntry),
                           reinterpret_cast<void*>(&render_queue_chunk_job), DecompKind::Leaf};
        job.original = &g_game_chunk_job;
        decomp_add(job);
    }
    if (g_census) {
        DecompFunction append{"render_queue_list_append", "gx", kListAppend, kListAppendEntry, sizeof(kListAppendEntry),
                              reinterpret_cast<void*>(&render_queue_list_append), DecompKind::Leaf};
        append.original = &g_game_append;
        decomp_add(append);
    }
    if (g_push_placed) {
        DecompFunction pf{"render_queue_parallel_for", "gx", kParallelFor, kParallelForEntry, sizeof(kParallelForEntry),
                          reinterpret_cast<void*>(&render_queue_parallel_for), DecompKind::Leaf};
        pf.original = &g_game_parallel_for;
        decomp_add(pf);
    }
}
