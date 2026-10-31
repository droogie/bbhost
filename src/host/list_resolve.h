// Resolving draws where their command lists are finished (docs/rendering.md,
// "Resolving draws where their command lists are finished"; the plan and its
// measurements are docs/list-resolve.md): the first, measuring step. A list's
// draws can be resolved ahead of the command processor only between the
// points where something must run in its order - a wait, a label, a copy, a
// dispatch, a query, a change of render targets. BBHOST_LIST_RESOLVE=runs
// counts how the command processor's token stream falls into such runs of
// GX-native draws: how many draws a run holds, what ends each run, how many
// draws a GX command list (a GX context's stretch of draws) holds, and how
// many flips lie between a draw's GX call and the command processor reaching
// it. GX's pipeline drains (barrier tokens) are counted but end no run: they
// order the command processor's own writes after the work before them, and
// no draw's resolution reads them. Off by default (0): nothing is counted and
// nothing changes.
#pragma once

#include <cstddef>
#include <cstdint>

struct GpuDrawInputs;
struct GxDrawObjects;

namespace list_resolve {

// BBHOST_LIST_RESOLVE: 0 off, 1 (`runs`) the run census.
extern const int g_mode;
// BBHOST_RUN_BATCH: 0 off, 1 on, unset on with the run commit (the default:
// with BBHOST_RUN_COMMIT on or compare), 2 (`compare`) today's work
// with every answer a run would have shared checked against it. What a draw
// of a run takes from the draw before it: its targets' lookups (render.cpp
// draw_targets); and, on, the lite shadow drains GX's notices once a draw
// rather than once a vertex record (no answer to check: compare drains as
// before).
extern const int g_batch;
inline bool batching() { return g_batch != 0; }
inline bool batch_compare() { return g_batch == 2; }
inline bool batch_shares() { return g_batch == 1; }
// BBHOST_DRAW_CONTEXT: 0 off (the default), 1 (`chunks`) the command
// processor's draws take their rings' space through a context of chunks
// (gpu_internal.h DrawCtx), and the census below counts what they still take
// outside it; 2 (`compare`) as 1, and every chunk checked against the others
// its ring gave the same recording, and every constant-buffer copy a context's
// memo binds against the game's bytes.
extern const int g_draw_ctx;
inline bool draw_ctx_compare() { return g_draw_ctx == 2; }
// Compare mode: a chunk [lo, hi) of ring `ring` (0 the GX ring, 1 the
// constant-buffer copies', 2 the vertex copies') taken for recording
// `recording`, its ring [ring_lo, ring_hi); counted when it overlaps another
// of that recording's or leaves the ring. Under the GPU mutex.
void check_chunk(int ring, std::uint64_t recording, std::uint64_t lo, std::uint64_t hi, std::uint64_t ring_lo, std::uint64_t ring_hi);
// BBHOST_RUN_COMMIT (docs/list-resolve.md, "Workers commit runs"): 0 (`off`)
// every draw committed in order; 1 (`fork`) the machinery priced with nothing moved - at a run's
// first draw the command processor looks ahead for the run's other draws, and
// a run of BBHOST_RUN_COMMIT_MIN draws or more (8) is forked to a pool of
// BBHOST_RUN_COMMIT_THREADS workers (3) and joined, the workers given nothing
// to do; 2 (`on`) the run's draws after its first committed in parts by the
// workers and the command processor (render.cpp run_commit), spliced in their
// order at the join - the default since 2026-10-29 (unset, or `on`); 3
// (`compare`) as 2, and each forked run also committed in
// order as before, the parts' packets compared with those and dropped. The
// workers spin BBHOST_RUN_COMMIT_SPIN_US (50) after a part before they sleep
// on the fork's word. On (2, 3) turns BBHOST_RUN_BATCH on unless it is set.
extern const int g_commit;
inline bool committing() { return g_commit != 0; }
inline bool commit_draws() { return g_commit >= 2; }
inline bool commit_compare() { return g_commit == 3; }
// Whether the command processor tells this module what it walks.
inline bool on() { return g_mode != 0 || g_batch != 0 || g_commit != 0; }
// Whether the run census counts.
inline bool census() { return g_mode != 0; }
// This command-processor thread's run: moves at every split, so two draws
// that see the same value had nothing done in order between them.
std::uint64_t run_serial();

// What the batching and the draw context count, always under the GPU mutex
// (exit report, and every 1,000 flips with tooling).
enum Count : int {
    kBatchTargetsAsked,     // draws whose targets it looked at
    kBatchTargetsShared,    // ... that took the draw before's
    kBatchMissSplit,        // not shared: something done in order since that draw
    kBatchMissLock,         // not shared: the GPU mutex taken by another since
    kBatchMissRemade,       // not shared: a target made or resized since (g_rt_gen)
    kBatchMissFill,         // not shared: an HTILE fill noted or taken since
    kBatchMissInputs,       // not shared: other targets, scissors or depth state
    kBatchMissKind,         // not shared: a draw that does not share (tessellation, a geometry shader, YEBIS, a packet draw)
    kBatchCompared,         // compare mode: shared answers checked against the lookup
    kBatchDiffered,         // ... that differed
    kBatchDrains,           // lite drains at a draw's first vertex record
    kCtxDraws,              // draws committed through a draw context
    kCtxGxChunks,           // chunks it took of the GX ring
    kCtxCbChunks,           // ... of the constant-buffer copies' ring
    kCtxSnapChunks,         // ... of the vertex copies' ring
    kCtxGxBytes,            // bytes it placed in the GX ring
    kCtxCbBytes,            // ... copied into the constant-buffer copies' ring
    kCtxSnapBytes,          // ... copied into the vertex copies' ring
    kCtxCbShared,           // constant-buffer snapshots bound from its memo
    kCtxOutParams,          // still taken outside it: params slots
    kCtxOutSets,            // ... descriptor sets allocated
    kCtxOutPackets,         // ... draw packets
    kCtxChunksChecked,      // compare: chunks checked against their recording's others
    kCtxChunksBad,          // ... that overlapped one or left their ring
    kCtxMemoChecked,        // compare: copies bound from a context's memo checked
    kCtxMemoChanged,        // ... whose bytes the game had changed since
    kCounts
};
void count(Count k, std::uint64_t n = 1);

// What ends a run: the draw after it renders into other targets, or the
// command processor meets something it must do in order first.
enum Split : int {
    kSplitTargets,         // the next draw's colour or depth targets (or a clearing draw)
    kSplitWait,            // WAIT_REG_MEM
    kSplitLabel,           // EVENT_WRITE_EOP / EOS, RELEASE_MEM
    kSplitCopy,            // a GX copy, fill, upload, image copy, clear or CopyStructureCount token
    kSplitDispatch,        // a dispatch token or packet
    kSplitQuery,           // an occlusion query's token, a pixel-pipe dump
    kSplitOtherDraw,       // a YEBIS or Scaleform draw token
    kSplitCpWrite,         // WRITE_DATA, DMA_DATA
    kSplitPredication,     // SET_PREDICATION
    kSplitConstantEngine,  // the constant engine's RAM and counters
    kSplitCall,            // INDIRECT_BUFFER(_CONST)
    kSplitPacketDraw,      // a draw packet the hooks did not take
    kSplitSubmission,      // the end of a submission's walk
    kSplits
};

// The command processor met `why` (any packet it acts on that is not a
// GX-native draw token).
void split(Split why);
// A GX pipeline drain's barrier token (no split).
void note_barrier();
// A GX-native draw token's draw, with its inputs and objects.
void note_draw(const GpuDrawInputs& in, const GxDrawObjects& o);
// Every few seconds while the game runs (host tooling), and at the exit.
void report(bool at_exit);

// The render pass a draw's inputs draw in, as the census tells runs apart: two
// draws with the same value draw into the same targets.
std::uint64_t targets_of(const GpuDrawInputs& in);
// Why the run look ahead stopped (gnm_exec.cpp run_following).
enum LookStop : int {
    kLookSplit,     // a packet the command processor does in order (split)
    kLookTargets,   // a draw token into other targets
    kLookNotReady,  // a draw token whose slot is not stored yet
    kLookEnd,       // the end of the command buffer, or a packet it cannot read
    kLookKind,      // BBHOST_RUN_COMMIT=on: a draw a part does not take (tessellated, a geometry shader's, indirect)
    kLookStops
};
// BBHOST_RUN_COMMIT, at a GX-native draw token on the command processor,
// before it is drawn. run_left(): draws of the run a fork covered still to
// come on this thread - a draw token with some left calls run_take(); with
// none it looks ahead and calls run_fork with the run's draws after this
// one, why the look stopped, and the look's time. A split ends the run.
std::uint32_t run_left();
void run_take();
void run_fork(std::uint32_t following, LookStop stop, std::uint64_t look_ns);
// BBHOST_RUN_COMMIT=on: the look's census (note_look); a run committed in
// parts or, too short or refused, not (note_commit); the draws a commit
// covered after the run's first, which the walk then passes over (run_set);
// and the line every 1,000 flips with tooling (report_commit).
void note_look(LookStop stop, std::uint64_t look_ns);
void note_commit(std::uint32_t draws, std::uint64_t ns, bool forked);
void run_set(std::uint32_t following);
// A run that was not forked: its `following` draws are drawn as the walk
// reaches them, and none of them looks ahead again (run_held takes one).
void run_hold(std::uint32_t following);
bool run_held();
void report_commit();
// The run workers (BBHOST_RUN_COMMIT_THREADS) and the shortest run forked.
int run_workers();
std::uint32_t run_min();
// Runs fn(job, k) for k = 0 .. parts - 1 - part 0 on the calling thread, the
// others on the run workers (parts <= run_workers() + 1) - and returns once
// every worker is done.
void run_parts(void (*fn)(void* job, int part), void* job, int parts);

}  // namespace list_resolve
