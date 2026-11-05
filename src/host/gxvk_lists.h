// GX on Vulkan, stage 4: GX's command lists recorded where they are built,
// the command processor stitching them (docs/rendering.md's design note,
// "Resolving draws where their command lists are finished"). Its first step
// is a census that changes nothing (BBHOST_GXVK_LISTS=census):
//
// - GX's side: each command list a deferred context finishes
//   (FinishCommandList, decomp/gx/command_list.cpp) - its context, its place
//   among that context's lists, the GX-native draw tokens recorded into it and
//   when - and each list the immediate context executes (ExecuteCommandList(s)):
//   its place in the frame's order and when;
// - the command processor's side: each list as its walk meets it - the draws
//   it holds, the render passes they fall into, what the walk does in order
//   inside it and between lists (copies, dispatches, queries, labels, waits,
//   other submitters' draws), what each draw needed that only the command
//   processor's order gives (what a run part would stop at in it: render.cpp
//   RunStop), its vertex slots without a GX handle, and the command
//   processor's time in it and between lists.
//
// Lines every 1,000 flips with tooling and at the exit; with
// BBHOST_GXVK_LISTS_OUT=path every record goes to that file too (their layout:
// gxvk_lists.cpp). The records are made with BBHOST_RUN_COMMIT off (a run
// commit's parts draw without the walk's notes).
#pragma once

#include <atomic>
#include <cstdint>

struct GpuDrawInputs;
struct GxDrawObjects;

namespace gxvk_lists {

// BBHOST_GXVK_LISTS=census, read before any of the game's code runs.
extern const bool g_census;
inline bool census() { return g_census; }

// Stage 4b.1a (BBHOST_GXVK_LISTS=1 or 2): a pool of
// host threads (bb-list*) looks ahead of the command processor at each
// GX-native draw a GX thread has stored - its programs, fetch shader,
// translations, stage plans, views and samplers - through the caches' reader
// paths (host/cache_rw.h), committing nothing. =2 also has the command
// processor compare its own answers at each draw with the pool's: equal, the
// pool's "not made yet" (made since), changed since (a write between), or
// different (a reader's wrong answer: must stay 0). The pool's threads:
// BBHOST_GXVK_LIST_THREADS, else 2 on 16 or more hardware threads, 1 on 8-15,
// none below (the pool off). Read before any of the game's code runs.
// 0 off, 1 the pool's lookups, 2 and the compare, 3 splice, 4 splice-compare,
// 5 segments, 6 segments-compare. Settled once the device is chosen (settle):
// 0 until then.
extern std::atomic<int> g_pool_mode;
inline int pool_mode() { return g_pool_mode.load(std::memory_order_relaxed); }
int pool_threads();             // 0 when the pool is off (or may not run: no switch, fewer than 8 hardware threads)
inline bool pool() { return pool_mode() != 0; }
inline bool compare() { return pool_mode() == 2; }

// Stage 4b.1b (BBHOST_GXVK_LISTS=splice): the pool commits each stored
// GX-native draw ahead of the command processor - through draw_impl in its
// list mode (render.cpp lists_commit: a run part's commit with its own
// packet, its params kept as contents, and what only the walk's place
// answers noted for the splice instead of read) - and the command processor,
// at a token whose draw falls in the run it has open (the run's first draw
// drawn in order: its targets looked up, its pass begun), checks the commit
// and splices its packet (host_gpu_splice). A draw not committed, not yet
// committed, or whose checks fail is drawn in order, counted by why.
// splice-compare: every draw drawn in order (its packet submitted), each
// committed one compared with it field by field (RunDigest) and dropped.
inline bool splice() { return pool_mode() >= 3; }
inline bool splice_compare() { return pool_mode() == 4 || pool_mode() == 6; }

// Stage 4b.2 (BBHOST_GXVK_LISTS=segments): the pool also cuts each list a GX
// context finished into segments - the draws after a run's first, of one
// list, into the same targets, with nothing between their tokens the command
// processor does in order - and builds each segment of committed draws on its
// own thread (render.cpp lists_segment): replayed inline by the recorder, or
// with BBHOST_GXVK_SEGMENTS_REPLAY=secondary recorded into a secondary command
// buffer. The command processor checks a segment once at its first token and
// splices it with one op (host_gpu_splice_segment); a run's first draw is
// spliced or drawn as splice does. segments-compare: every draw drawn in order,
// each draw of a segment that would have been spliced compared with its
// commit. Unset, the switch follows the constant buffers' homes
// (BBHOST_GXVK_HOMES, and their default: host/gxvk_homes.h default_off):
// segments where they bind by handle, off elsewhere (the reason logged).
inline bool segments() { return pool_mode() >= 5; }
inline bool segments_compare() { return pool_mode() == 6; }
// The device chosen and the homes decided (gpu.cpp): the mode settled - the
// switch as given, or with it unset the homes' answer (`homes_bind`: they bind
// by handle; else `homes_why`, why not, or null when the homes' default line
// said so already) - and the readers' lock turned on when the pool will run.
void settle(bool homes_bind, const char* homes_why);


// BBHOST_GXVK_LISTS_TIMES=1 (a diagnostic, in the pool's lines): when each
// stored draw was stored, when its list closed and when the pool's commit of
// it began and ended, against the command processor's arrival at its token -
// by where the pool was then (its feed's entry not taken yet, and how far
// behind; committing; done) - and the commit's time by part, the pool's time
// parked and waiting. Read before any of the game's code runs.
extern const bool g_times;
inline bool times() { return g_times; }

// The pool's cores (stage 4b.1b, BBHOST_GXVK_LIST_CPUS: auto, the default
// with splice - the last physical cores of the process's CPUs, one a pool
// thread, both of a core's hardware threads kept for it; off): the command
// processor and the recorder call this in their loops, and keep off them from
// the first call after the pool has started (one sched_setaffinity each).
void keep_off_pool_cpus();

// The recording thread stored GX-native draw token `id`'s slot
// (gx_trace.cpp native_draw_call; a host function, not a leaf): recorded on
// GX context `ctx`, its token at `token_va` in the context's command buffer
// (segments: the draw goes into the context's open list record too).
void draw_stored(std::uint64_t id, std::uint64_t ctx, std::uint64_t token_va);

// The pool's answers, and the command processor's in compare mode: a lookup's
// kind, its key (the cache's own key, folded to 56 bits) and what it answered
// (0: not found / not made yet).
enum LookKind : std::uint8_t { kLookProgram, kLookFetch, kLookPaths, kLookPlan, kLookView, kLookSampler, kLookKinds };
struct LookEntry {
    std::uint64_t key;  // kind << 56 | the key's low 56 bits
    std::uint64_t answer;
};
inline std::uint64_t look_key(LookKind kind, std::uint64_t key) {
    return static_cast<std::uint64_t>(kind) << 56 | (key & 0x00ffffffffffffffull);
}
constexpr int kLookMost = 56;
struct LookSink {
    std::uint32_t n = 0;
    std::uint32_t lost = 0;  // entries past kLookMost
    LookEntry e[kLookMost];
    void add(LookKind kind, std::uint64_t key, std::uint64_t answer) {
        if (n < kLookMost) {
            e[n++] = {look_key(kind, key), answer};
        } else {
            ++lost;
        }
    }
};

// The command processor (bb-cp0), compare mode: around a GX-native draw
// token's draw (gnm_exec.cpp), and each lookup's answer inside it. With
// BBHOST_GXVK_LISTS_SAMPLE=N only every N-th draw is compared (1, every draw,
// by default), so a timing run can keep the compare on at a small cost.
extern const unsigned g_look_every;
inline bool look_due() {
    static thread_local unsigned n = 0;
    if (++n < g_look_every) return false;
    n = 0;
    return true;
}
void cp_look_begin(std::uint64_t id);
void cp_look_end();
extern thread_local constinit bool t_cp_look;
void cp_note_slow(LookKind kind, std::uint64_t key, std::uint64_t answer);
inline void cp_note(LookKind kind, std::uint64_t key, std::uint64_t answer) {
    if (pool_mode() == 2 && t_cp_look) cp_note_slow(kind, key, answer);
}
// The pool's lines (report): every 1,000 flips with tooling from its first
// thread, and at the exit - its threads stopped and joined first.
void pool_report(bool at_exit);

// Leaf-safe - the decomp leaves call them on the game's threads with the
// game's FS: no locks, no allocation, no host thread-locals. A deferred
// context `ctx` finished `list` (FinishCommandList's body, after its spans
// were taken); the immediate context executes `list`, the `index`-th of the
// `count` it was given (ExecuteCommandList(s)).
void finished(std::uint64_t ctx, std::uint64_t list);
void executed(std::uint64_t list, std::uint32_t index, std::uint32_t count);
// Stage 4b.2: `ctx` executes command lists - the immediate context, whose own
// draws go into no list record (no FinishCommandList ever closes one).
void immediate_context(std::uint64_t ctx);

// A recording thread wrote a GX-native draw's token on `ctx`
// (gx_trace.cpp native_draw_call): the list it goes into, 1 for the
// context's first, as the context's finishes so far tell it.
std::uint32_t draw_recorded(std::uint64_t ctx);

// The command processor (bb-cp0): around a GX-native draw token's draw
// (gnm_exec.cpp), any other packet it acts on (`split`: list_split_of's
// kind, >= 0), and the end of a submission's walk.
void cp_draw_begin(const GpuDrawInputs& in, const GxDrawObjects& o);
void cp_draw_end();
void cp_packet(int split);
void cp_submission_end();

// Every 1,000 flips with host tooling (from the command processor), and at
// the exit.
void report(bool at_exit);

// What the draw under way on the command processor would have stopped a run
// part at (bit RunStop): run_part_stop notes it while the census runs.
extern thread_local constinit std::uint32_t t_effects;

// What the draw under way read that only the command processor's order
// answers, though no run part stops at it (the part reads the answer as it
// stands at the walk's place): a binding answered by a copy version (a
// constant buffer's, a vertex record's, the indices' or the indirect
// arguments'), a constant buffer read in place for a GPU write still to land
// on it, a GX buffer the GPU writes bound from its twin.
enum Order : std::uint32_t {
    kOrderCbVersion = 1u << 0,
    kOrderCbGpuWrite = 1u << 1,
    kOrderVertexVersion = 1u << 2,
    kOrderIndexVersion = 1u << 3,
    kOrderTwin = 1u << 4,
    kOrders = 5
};
extern thread_local constinit std::uint32_t t_order;
inline void note_order(std::uint32_t bit) {
    if (g_census) t_order |= bit;
}

}  // namespace gxvk_lists
