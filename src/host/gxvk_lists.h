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
extern const int g_pool_mode;  // 0 off, 1 the pool's lookups, 2 and the compare
int pool_threads();             // 0 when the pool is off
inline bool pool() { return g_pool_mode != 0; }
inline bool compare() { return g_pool_mode == 2; }

// The recording thread stored GX-native draw token `id`'s slot
// (gx_trace.cpp native_draw_call; a host function, not a leaf).
void draw_stored(std::uint64_t id);

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
    if (g_pool_mode == 2 && t_cp_look) cp_note_slow(kind, key, answer);
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
