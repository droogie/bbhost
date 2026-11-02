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

// BBHOST_GXVK_LISTS=census (or 1), read before any of the game's code runs.
extern const bool g_census;
inline bool census() { return g_census; }

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
