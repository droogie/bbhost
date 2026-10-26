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
// The census (BBHOST_RQ_CENSUS=1; the three functions are placed only with
// it): per parallel-for, its runs and the chunks they became; per chunk job,
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

using OneFn = GUEST_ABI u64 (*)(u64 a);
using TwoFn = GUEST_ABI u64 (*)(u64 a, u64 b);
using ContextFn = GUEST_ABI u64 (*)(u64 parent, u64 per_worker, u64 zero);
using QueueFn = GUEST_ABI u64 (*)(u64 gx, u64 command_context, u32 key);
using PrepareFn = GUEST_ABI u64 (*)(u64 queue, u64 ctx, u64 state, u32 key, u32 pre);
using CommandFn = GUEST_ABI u64 (*)(u64 queue, u64 ctx, u32 type, u64 payload, u64 size);
using AppendFn = GUEST_ABI u64 (*)(u64 list, u64 record, u32 key);

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

inline u32 bn_of(u64 guest) { return static_cast<u32>(guest - g_decomp_slide + 0x400000); }

// ---- The partition (sub_26d11d0) ----

// The entry covering a key's category and sub-category, as the game looks it
// up: an entry covers the categories up to the next entry's - all of its own
// when its sub-category is -1, else its own from that sub-category on - and
// two entries of one category split it at the second's sub-category.
inline u64 entry_of(u32 key) {
    const u32 cat = (key >> 22) & 0x3f, sub = (key >> 16) & 0x3f;
    u64 e = game_address(kTable);
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

inline s8 entry_index(u64 e) { return e ? static_cast<s8>((e - game_address(kTable)) / kTableEntry) : s8{-1}; }

inline void note(Part* part, u64 e, u8 how, u32 chunks, u32 start, u32 n) {
    if (!part) return;
    if (part->n_runs < kRunsKept) part->runs[part->n_runs] = Run{entry_index(e), how, static_cast<u16>(chunks), start, n};
    ++part->n_runs;
}

// The game's partition: the chunks appended at the queue's table, how many
// returned. `part`: the census's record, or null.
DECOMP_LEAF u32 partition(u64 q, u32 first, u32 last, u32 max_chunks, Part* part) {
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
            next = entry_of(load<u32>(item));
        } else {
            const u32 key = load<u32>(item);
            const u32 x = load<u32>(prev) ^ key;
            if (x <= 0x0fffffffu && (x & 0x0fc00000u) == 0 && (x & 0x3f0000u) == 0) {
                next = cur;
            } else {
                next = entry_of(key);
                if (!next || next != cur) {
                    // The run [start, i) ends.
                    const u32 n = i - start;
                    if (n == 0) {
                        start = i;
                    } else {
                        u32 cut, least;
                        if (!cur) {
                            least = 5;
                            cut = 1;
                        } else {
                            const u32 per = load<u32>(cur + 0x10);
                            least = load<u32>(cur + 0x14);
                            if (per == 0) {
                                cut = load<u32>(cur + 8);
                            } else {
                                const u32 cap = load<u32>(cur + 0xc);
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
                                ++n_chunks;
                                note(part, cur, kHowOne, 1, start, n);
                            } else {
                                note(part, cur, kHowNone, 0, start, n);
                            }
                            start = i;
                        } else if (load<u8>(q + kStealing) != 0) {
                            u32 grain = n / chunks;
                            if (grain <= 10) grain = MUTANT(11) ? 11 : 10;
                            if (n < 10) grain = n;
                            put_chunk(q, n_chunks, start, i, grain, cur);
                            ++n_chunks;
                            note(part, cur, kHowStealing, 1, start, n);
                            start = i;
                        } else if (chunks + n_chunks > (MUTANT(3) ? kChunkMax : kChunkMax - 1)) {
                            // Everything left becomes one chunk, with this run's entry.
                            note(part, cur, kHowOverflow, 1, start, end - start);
                            overflow = true;
                            break;
                        } else {
                            const u32 size = n / chunks, longer = n % chunks;
                            u32 at = start;
                            for (u32 k = 0; k != chunks; ++k) {
                                const u32 s = size + (k < longer && !MUTANT(4) ? 1u : 0u);
                                put_chunk(q, n_chunks + k, at, at + s, s, cur);
                                at += s;
                            }
                            note(part, cur, kHowSplit, chunks, start, n);
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
        if (MUTANT(12) ? n_chunks > kChunkMax : n_chunks >= kChunkMax) break;
    }
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
    if (!overflow) note(part, cur, kHowLast, 1, start, end - start);
    const u32 made = n_chunks + 1 - load<u32>(q + kChunkCount);
    store<u32>(q + kChunkCount, n_chunks + 1);
    return made;
}

DECOMP_LEAF u32 render_queue_partition(u64 q, u32 first, u32 last, u32 max_chunks) {
    if (!g_census) return partition(q, first, last, max_chunks, nullptr);
    const u64 t = ticks();
    Part* p = &g_parts[g_part_pos.fetch_add(1, std::memory_order_relaxed) & (kPartRing - 1)];
    publish(p->done, 0);
    p->t = t;
    p->queue = q;
    p->ret = bn_of(reinterpret_cast<u64>(__builtin_return_address(0)));
    p->first = first;
    p->last = last;
    const u32 count = load<u32>(q + kCount);
    p->end = count > last ? last : count;
    p->max_chunks = max_chunks;
    p->before = load<u32>(q + kChunkCount);
    p->stealing = load<u8>(q + kStealing);
    p->n_runs = 0;
    // A parallel-for's jobs all run before the next partition is called (its
    // barrier waits for them), so they take its generation.
    const bool parallel = p->ret == kFromParallelFor;
    p->gen = parallel ? g_gen.fetch_add(1, std::memory_order_relaxed) + 1 : 0;
    p->flip = g_flip.load(std::memory_order_relaxed);
    p->made = partition(q, first, last, max_chunks, p);
    publish(p->done, 1);
    return p->made;
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

DECOMP_LEAF u64 render_queue_chunk_job(u64 functor, u64 range, u32 worker) {
    Chunk* rec = nullptr;
    Item* out = nullptr;
    u32 out_n = 0;
    u64 t0 = 0;
    if (g_census) {
        t0 = ticks();
        rec = &g_chunk_ring[g_chunk_pos.fetch_add(1, std::memory_order_relaxed) & (kChunkRing - 1)];
        publish(rec->done, 0);
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
    const u64 lead = rec ? ticks() - t0 : 0;
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
    if (rec) {
        rec->key = key;
        rec->work = static_cast<u32>(work > 0xffffffffull ? 0xffffffffull : work);
        rec->lead = static_cast<u32>(lead);
        rec->t1 = ticks();
        publish(rec->done, rec->gen + 1);
    }
    return ret;
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

void* g_game_partition = nullptr;
DecompCompare g_partition_counts;
std::atomic<int> g_compare_logs{0};

constexpr u64 kTableBytes = static_cast<u64>(kChunkMax) * kChunkSize;

inline void* at_(u64 a) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(a)); }

GUEST_ABI u32 compare_partition(u64 q, u32 first, u32 last, u32 max_chunks) {
    u8 before[kTableBytes], game[kTableBytes];
    std::memcpy(before, at_(q + kChunks), kTableBytes);
    const u32 count_before = load<u32>(q + kChunkCount);
    const u32 theirs = static_cast<u32>(hle_call_guest<u64>(g_game_partition, q, first, last, max_chunks));
    std::memcpy(game, at_(q + kChunks), kTableBytes);
    const u32 game_count = load<u32>(q + kChunkCount);
    std::memcpy(at_(q + kChunks), before, kTableBytes);
    store<u32>(q + kChunkCount, count_before);
    const u32 ours = partition(q, first, last, max_chunks, nullptr);
    const bool same = ours == theirs && load<u32>(q + kChunkCount) == game_count && std::memcmp(at_(q + kChunks), game, kTableBytes) == 0;
    g_partition_counts.calls.fetch_add(1, std::memory_order_relaxed);
    if (!same) {
        g_partition_counts.differ.fetch_add(1, std::memory_order_relaxed);
        if (g_compare_logs.fetch_add(1) < 16)
            host_log("decomp: render_queue_partition differs from the game's: items %u-%u, %u chunks made (game %u), count %u (game %u)",
                     first, last, ours, theirs, load<u32>(q + kChunkCount), game_count);
    }
    std::memcpy(at_(q + kChunks), game, kTableBytes);
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

void census_report() {
    if (!g_census) return;
    g_fold.report();
    host_log("rq census: %llu partitions, %llu chunk jobs, %llu items, %llu segments recorded",
             static_cast<unsigned long long>(g_part_pos.load()), static_cast<unsigned long long>(g_chunk_pos.load()),
             static_cast<unsigned long long>(g_item_pos.load()), static_cast<unsigned long long>(g_append_pos.load()));
}

}  // namespace

void render_queue_census_flip(std::uint32_t flips) {
    if (!g_census) return;
    g_flip.store(flips, std::memory_order_relaxed);
    g_fold.flip(flips);
}

void decomp_gx_render_queue_add() {
    const char* e = std::getenv("BBHOST_RQ_CENSUS");
    if (!e || e[0] != '1') return;
    g_items = static_cast<Item*>(std::calloc(kItemRing, sizeof(Item)));
    g_chunk_ring = static_cast<Chunk*>(std::calloc(kChunkRing, sizeof(Chunk)));
    g_parts = static_cast<Part*>(std::calloc(kPartRing, sizeof(Part)));
    g_appends = static_cast<Append*>(std::calloc(kAppendRing, sizeof(Append)));
    if (!g_items || !g_chunk_ring || !g_parts || !g_appends) {
        host_log("rq census: no memory for its rings; off");
        return;
    }
    g_census = true;
    g_fold.start();
    host_log("rq census: on (the render queue's partition, chunk job and segment list as ours)");
    DecompFunction part{"render_queue_partition", "gx", kPartition, kPartitionEntry, sizeof(kPartitionEntry),
                        reinterpret_cast<void*>(&render_queue_partition), DecompKind::Leaf};
    part.original = &g_game_partition;
    part.compare = reinterpret_cast<void*>(&compare_partition);
    part.counts = &g_partition_counts;
    part.report = &census_report;
    decomp_add(part);
    decomp_add(DecompFunction{"render_queue_chunk_job", "gx", kChunkJob, kChunkJobEntry, sizeof(kChunkJobEntry),
                              reinterpret_cast<void*>(&render_queue_chunk_job), DecompKind::Leaf});
    DecompFunction append{"render_queue_list_append", "gx", kListAppend, kListAppendEntry, sizeof(kListAppendEntry),
                          reinterpret_cast<void*>(&render_queue_list_append), DecompKind::Leaf};
    append.original = &g_game_append;
    decomp_add(append);
}
