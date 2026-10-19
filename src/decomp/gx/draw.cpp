// GX's immediate context as source (docs/decomp.md): its six draw methods,
// its two dispatch methods and the state commit without a draw - Binary Ninja
// 0x25696c0..0x2569ba0, the D3D11 context's Draw* and Dispatch* as the
// engine's GX layer has them (Dark Souls III's GXImmediateContext, D3D11
// under the same layer) - and three of what each of them calls: the
// output-merger flush (gx_flush_output_merger, 0x2ad2230), which writes what
// changed in the blend, depth-stencil, stencil reference and depth bounds
// state and runs the deferred clears of targets no longer bound; the
// post-draw (gx_post_draw, 0x2acbd50), which folds the flushes' dirty bits
// into the flags the next commit reads; and the command buffer's flush
// (gx_command_buffer_flush, 0x2aaf700), which every GX operation ends with and
// which closes a segment when the context records into an arena chunk.
//
// Each method is a call sequence: the output-merger, input-assembly and
// rasterizer flushes (0x2ad2230, 0x2abba30, 0x2ad3370: what changed in the
// context's state since the last draw, as register packets), the flush, the
// commit (0x2abc2e0: the shaders' user data and resource tables), the draw's
// packets (Gnm's writers: the start vertex, the instance count, the draw
// itself), the post-draw, the flush again; a dispatch has the compute stage's
// commit and its own post (0x2acbe50, 0x2accf20) around its packet. Ours makes
// the same calls with the same arguments in the same order - each to the
// game's address, so a callee on the decomp list is ours and one that is not
// stays the game's.
//
// Before the sequence, ours hands the call to the GX trace (decomp/gx/draw.h,
// hle/gx_trace.cpp gx_hook) as the prologue stubs did before these methods
// were ours: a draw it can take whole becomes a host-draw token there, and
// the sequence does not run; a dispatch and the commit it only notes, and the
// sequence runs (the dispatch's token is written at Gnm's dispatch writer).
// That is the place a draw is seen as a D3D11-shaped command - the context,
// the pending state, the arguments - which is where a plugin's per-draw hook
// would sit (docs/modding.md).
//
// The trace knows a GX method's packets by the return address of the Gnm
// writer that writes them (a draw's records go with the packet; a dispatch
// through GX's compute commit is one); ours call those writers from the
// host's code, so the methods are a section of their own and the trace takes
// a return address inside it as the game's from the same method
// (decomp_gx::game_return).
//
// Leaves: they read the game's memory and call the game's functions, and of
// the host's only the trace's entry, which switches to the host's FS itself.
#include "decomp/gx/draw.h"

#include "decomp/decomp.h"
#include "decomp/guest.h"

#include "core/thunk.h"
#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstring>

using namespace decomp;

// tests/decomp/gx_draw_test.cpp builds this file with mutants: small wrong
// turns it switches on one at a time and must catch.
#ifdef DECOMP_GX_DRAW_MUTANTS
extern int g_draw_mutant;
#define MUTANT(n) (g_draw_mutant == (n))
#else
#define MUTANT(n) false
#endif

// The writer callers' section (decomp/gx/draw.h) and its bounds.
#define GX_METHOD DECOMP_GX_WRITER_CALLER
#ifdef _WIN32
// COFF: input sections of one name before the $ make one section, in the
// order of what follows the $. The two ends are functions of their own (not
// alike, so that no linker folds them into one).
extern "C" __attribute__((section("bbgxm$a"), used, noinline)) int bbhost_gx_methods_lo() { return 1; }
extern "C" __attribute__((section("bbgxm$z"), used, noinline)) int bbhost_gx_methods_hi() { return 2; }
#define GX_METHODS_LO reinterpret_cast<std::uint64_t>(&bbhost_gx_methods_lo)
#define GX_METHODS_HI reinterpret_cast<std::uint64_t>(&bbhost_gx_methods_hi)
#else
// ELF: the linker's bounds of a section named as a C identifier.
extern "C" __attribute__((visibility("hidden"))) const char __start_bbhost_gx_methods[];
extern "C" __attribute__((visibility("hidden"))) const char __stop_bbhost_gx_methods[];
#define GX_METHODS_LO reinterpret_cast<std::uint64_t>(__start_bbhost_gx_methods)
#define GX_METHODS_HI reinterpret_cast<std::uint64_t>(__stop_bbhost_gx_methods)
#endif

void* decomp_gx::g_gx_hook_entry = nullptr;
std::uint64_t decomp_gx::g_method_code_lo = 0, decomp_gx::g_method_code_hi = 0;
std::uint64_t decomp_gx::g_yebis_draw_kind = ~0ull, decomp_gx::g_yebis_dispatch_kind = ~0ull;
bool decomp_gx::g_yebis_dispatch_hooked = false;
const char* const decomp_gx::kMethodNames[9] = {
    "gx_immediate_draw",
    "gx_immediate_draw_indexed",
    "gx_immediate_draw_instanced",
    "gx_immediate_draw_indexed_instanced",
    "gx_immediate_draw_instanced_indirect",
    "gx_immediate_draw_indexed_instanced_indirect",
    "gx_context_commit",
    "gx_immediate_dispatch",
    "gx_immediate_dispatch_indirect",
};

namespace {

using decomp_gx::MethodKind;

// The game's return address from each method's draw or dispatch writer, by
// kind (decomp_gx::game_return); the commit writes no packet of its own and
// has its entry.
constexpr u64 kGameReturns[9] = {0x2569753, 0x256981c, 0x25698d4, 0x25699a1, 0x2569a8e, 0x2569b71, 0x2569ba0, 0x256960d, 0x256968f};
// Every writer caller told (note_writer_caller): ours' entry, and the game's
// return address it stands for. Filled as the decomp list is made, before
// any of ours runs; read only after.
struct WriterCaller {
    u64 entry, game_return_bn;
};
WriterCaller g_writer_callers[32];
int g_writer_callers_n = 0;

// What the methods call, by Binary Ninja address (symbols/1.09-names.tsv).
constexpr u64 kFlushOutputMerger = 0x2ad2230;  // gx_flush_output_merger(ctx + 0xb558, ctx, state)
constexpr u64 kFlushInputAssembly = 0x2abba30;  // gx_flush_input_assembly(ctx + 0xb518, ctx, state, indexed)
constexpr u64 kFlushRasterizer = 0x2ad3370;     // gx_flush_rasterizer(ctx + 0xb528, ctx, state)
constexpr u64 kCommandBufferFlush = 0x2aaf700;  // gx_command_buffer_flush(ctx + 0x10)
constexpr u64 kCommitState = 0x2abc2e0;         // gx_commit_state(ctx + 0x50, ctx, state)
constexpr u64 kPostDraw = 0x2acbd50;            // gx_post_draw(ctx + 0x50, ctx)
constexpr u64 kDispatchCommit = 0x2acbe50;      // gx_dispatch_commit(ctx + 0x50, ctx, state)
constexpr u64 kDispatchPost = 0x2accf20;        // gx_dispatch_post(ctx + 0x50, ctx, state)
constexpr u64 kSetIndexOffset = 0x1474040;      // gnm_set_index_offset(cb, offset)
constexpr u64 kSetNumInstances = 0x1475370;     // gnm_set_num_instances(cb, count)
constexpr u64 kDrawIndexAuto = 0x14741b0;       // gnm_draw_index_auto(cb, count)
constexpr u64 kDrawIndexOffset = 0x1474420;     // gnm_draw_index_offset(cb, start index, count)
constexpr u64 kSetBaseIndirectArgs = 0x14742f0;  // gnm_set_base_indirect_args(cb, address)
constexpr u64 kDrawIndirect = 0x14744d0;         // gnm_draw_indirect(cb, offset, stage, vertex sgpr, instance sgpr)
constexpr u64 kDrawIndexIndirect = 0x14746f0;    // gnm_draw_index_indirect(cb, offset, stage, vertex sgpr, instance sgpr)
constexpr u64 kDispatchDirect = 0x1474c50;       // gnm_dispatch_direct(cb, x, y, z)
constexpr u64 kDispatchIndirect = 0x1474de0;     // gnm_dispatch_indirect(cb, offset)
// The post-draw's two writers.
constexpr u64 kSetIaMultiVgtParam = 0x1475960;  // gnm_set_ia_multi_vgt_param(cb, value, flag)
constexpr u64 kSetGsMode = 0x14756b0;           // gnm_set_gs_mode(cb, mode, flag)
// The flush's.
constexpr u64 kReleaseNodePush = 0x2aaa5a0;       // gx_block_pool_release_node_push(pool, &cb, node)
constexpr u64 kFlushUnknown2acd3f0 = 0x2acd3f0;   // (cb + 0x40, cb)
constexpr u64 kRetireChunk = 0x2ad3c80;           // gx_command_arena_retire_chunk(arena, cb, chunk)
constexpr u64 kCloseSegment = 0x2ab0b00;          // gx_command_buffer_close_segment(cb, cb or 0, &header, kind)
constexpr u64 kDefaultRefill = 0x57e43c8;         // the GX arena's refill, the one a context's buffer has
// The output-merger flush's.
constexpr u64 kWriteClearColors = 0x2ad1f80;       // gx_om_write_clear_colors(om, ctx)
constexpr u64 kFastClearEliminate = 0x2ab5a70;     // gx_fast_clear_eliminate_pass(ctx, view, resource, colour, colour, 0)
constexpr u64 kDepthDecompress = 0x2ab6810;        // gx_depth_decompress_pass(ctx, view, resource, ?, ?, 0)
constexpr u64 kSetAlphaToMask = 0x1473030;         // gnm_set_alpha_to_mask(cb, DB_ALPHA_TO_MASK)
constexpr u64 kSetRenderTargetMask = 0x1472fd0;    // gnm_set_render_target_mask(cb, CB_TARGET_MASK)
constexpr u64 kSetBlendControl = 0x1473090;        // gnm_set_blend_control(cb, slot, CB_BLENDn_CONTROL)
constexpr u64 kSetBlendColor = 0x1473100;          // gnm_set_blend_color(cb, red, green, blue, alpha)
constexpr u64 kSetDepthStencilControl = 0x14733a0;  // gnm_set_depth_stencil_control(cb, DB_DEPTH_CONTROL)
constexpr u64 kSetStencilControl = 0x1473400;      // gnm_set_stencil_control(cb, DB_STENCIL_CONTROL)
constexpr u64 kSetStencilRef = 0x1473190;          // gnm_set_stencil_ref(cb, DB_STENCILREFMASK and _BF)
constexpr u64 kSetDepthBoundsRange = 0x1473520;    // gnm_set_depth_bounds_range(cb, min, max)
constexpr u64 kDefaultBlendState = 0x586c8f4;      // the blend state an unset slot stands for
constexpr u64 kDefaultDepthStencilState = 0x586ca1c;  // the depth-stencil state an unset slot stands for

using Ctx3 = GUEST_ABI void (*)(u64, u64, u64);
using Ctx3Indexed = GUEST_ABI void (*)(u64, u64, u64, u32);
using Cb = GUEST_ABI void (*)(u64);
using Ctx2 = GUEST_ABI void (*)(u64, u64);
using CbU32 = GUEST_ABI void (*)(u64, u32);
using CbU32U32 = GUEST_ABI void (*)(u64, u32, u32);
using CbU64 = GUEST_ABI void (*)(u64, u64);
using CbIndirect = GUEST_ABI void (*)(u64, u32, u32, u32, u32);
using CbU32x3 = GUEST_ABI void (*)(u64, u32, u32, u32);
using Writer = GUEST_ABI u64 (*)(u64, u32, u32);

void flush_output_merger(u64 ctx, u64 state) { game_function<Ctx3>(kFlushOutputMerger)(ctx + 0xb558, ctx, state); }
void flush_input_assembly(u64 ctx, u64 state, u32 indexed) {
    game_function<Ctx3Indexed>(kFlushInputAssembly)(ctx + 0xb518, ctx, state, indexed);
}
void flush_rasterizer(u64 ctx, u64 state) { game_function<Ctx3>(kFlushRasterizer)(ctx + 0xb528, ctx, state); }
void command_buffer_flush(u64 ctx) { game_function<Cb>(kCommandBufferFlush)(ctx + 0x10); }
void commit_state(u64 ctx, u64 state) { game_function<Ctx3>(kCommitState)(ctx + 0x50, ctx, state); }
void post_draw(u64 ctx) { game_function<Ctx2>(kPostDraw)(ctx + 0x50, ctx); }

// Written from every GX thread, which a leaf can tell apart only by where its
// stack is (no lock, no thread-local): a shard of counters by the stack's
// 64 KiB, so the GX workers do not take one cache line from each other at
// every draw.
struct alignas(64) CallShard {
    std::atomic<std::uint64_t> calls[9], taken[9];
};
constexpr int kCallShards = 32;
CallShard g_shards[kCallShards];
CallShard& call_shard() {
    const u64 sp = reinterpret_cast<u64>(__builtin_frame_address(0));
    return g_shards[((sp >> 16) * 0x9e3779b97f4a7c15ull) >> 59];
}
std::uint64_t total_calls(int kind) {
    std::uint64_t n = 0;
    for (const CallShard& s : g_shards) n += s.calls[kind].load(std::memory_order_relaxed);
    return n;
}
std::uint64_t total_taken(int kind) {
    std::uint64_t n = 0;
    for (const CallShard& s : g_shards) n += s.taken[kind].load(std::memory_order_relaxed);
    return n;
}

// A GX method's look from the trace, counted: true when it took the draw
// whole, and the method's sequence is not run.
bool trace_took(MethodKind kind, u64 rdi, u64 rsi, u64 rdx, u64 rcx, u64 r8, u64 r9, u64 ret) {
    CallShard& shard = call_shard();
    shard.calls[kind].fetch_add(1, std::memory_order_relaxed);
    if (!decomp_gx::hand_to_trace(kind, rdi, rsi, rdx, rcx, r8, r9, ret)) return false;
    shard.taken[kind].fetch_add(1, std::memory_order_relaxed);
    return true;
}

#define RET reinterpret_cast<u64>(__builtin_return_address(0))

// An indirect draw's shader stage for the vertex and instance offsets, by
// the context's stage configuration (ctx + 0x50): the VS for a plain one,
// the LS under tessellation, the ES under a geometry shader - Gnm's
// ShaderStage 2, 6 and 4.
u32 indirect_stage(u64 ctx) {
    const s32 v = load<s32>(ctx + 0x50);
    if (v == 0 || v == 0x200) return 2;
    if (v == 0x45 || (v == 0xad && !MUTANT(2))) return 6;
    return 4;
}

}  // namespace

// gx_immediate_draw(ctx, state, vertex count, start vertex).
DECOMP_LEAF GX_METHOD void gx_immediate_draw(u64 ctx, u64 state, u64 count, u64 start) {
    if (trace_took(decomp_gx::kGxDraw, ctx, state, count, start, 0, 0, RET)) return;
    flush_output_merger(ctx, state);
    flush_input_assembly(ctx, state, 0);
    flush_rasterizer(ctx, state);
    command_buffer_flush(ctx);
    commit_state(ctx, state);
    game_function<CbU32>(kSetIndexOffset)(ctx + 0x10, static_cast<u32>(start));
    game_function<CbU32>(kSetNumInstances)(ctx + 0x10, 1);
    game_function<CbU32>(kDrawIndexAuto)(ctx + 0x10, static_cast<u32>(count));
    post_draw(ctx);
    command_buffer_flush(ctx);
}

// gx_immediate_draw_indexed(ctx, state, index count, start index, base vertex).
DECOMP_LEAF GX_METHOD void gx_immediate_draw_indexed(u64 ctx, u64 state, u64 count, u64 start, u64 base) {
    if (trace_took(decomp_gx::kGxDrawIndexed, ctx, state, count, start, base, 0, RET)) return;
    flush_output_merger(ctx, state);
    flush_input_assembly(ctx, state, MUTANT(1) ? 0 : 1);
    flush_rasterizer(ctx, state);
    command_buffer_flush(ctx);
    commit_state(ctx, state);
    game_function<CbU32>(kSetIndexOffset)(ctx + 0x10, static_cast<u32>(base));
    game_function<CbU32>(kSetNumInstances)(ctx + 0x10, 1);
    game_function<CbU32U32>(kDrawIndexOffset)(ctx + 0x10, static_cast<u32>(start), static_cast<u32>(count));
    post_draw(ctx);
    command_buffer_flush(ctx);
}

// gx_immediate_draw_instanced(ctx, state, vertices an instance, instances,
// start vertex, start instance) - the start instance is not used.
DECOMP_LEAF GX_METHOD void gx_immediate_draw_instanced(u64 ctx, u64 state, u64 count, u64 instances, u64 start, u64 start_instance) {
    if (trace_took(decomp_gx::kGxDrawInstanced, ctx, state, count, instances, start, start_instance, RET)) return;
    flush_output_merger(ctx, state);
    flush_input_assembly(ctx, state, 0);
    flush_rasterizer(ctx, state);
    command_buffer_flush(ctx);
    commit_state(ctx, state);
    game_function<CbU32>(kSetIndexOffset)(ctx + 0x10, static_cast<u32>(start));
    game_function<CbU32>(kSetNumInstances)(ctx + 0x10, static_cast<u32>(instances));
    game_function<CbU32>(kDrawIndexAuto)(ctx + 0x10, static_cast<u32>(count));
    post_draw(ctx);
    command_buffer_flush(ctx);
}

// gx_immediate_draw_indexed_instanced(ctx, state, indices an instance,
// instances, start index, base vertex).
DECOMP_LEAF GX_METHOD void gx_immediate_draw_indexed_instanced(u64 ctx, u64 state, u64 count, u64 instances, u64 start, u64 base) {
    if (trace_took(decomp_gx::kGxDrawIndexedInstanced, ctx, state, count, instances, start, base, RET)) return;
    flush_output_merger(ctx, state);
    flush_input_assembly(ctx, state, 1);
    flush_rasterizer(ctx, state);
    command_buffer_flush(ctx);
    commit_state(ctx, state);
    game_function<CbU32>(kSetIndexOffset)(ctx + 0x10, static_cast<u32>(base));
    game_function<CbU32>(kSetNumInstances)(ctx + 0x10, static_cast<u32>(instances));
    game_function<CbU32U32>(kDrawIndexOffset)(ctx + 0x10, static_cast<u32>(start), static_cast<u32>(count));
    post_draw(ctx);
    command_buffer_flush(ctx);
}

// gx_immediate_draw_instanced_indirect(ctx, state, argument buffer, offset):
// the buffer's memory (+0x20) as the indirect arguments' base.
DECOMP_LEAF GX_METHOD void gx_immediate_draw_instanced_indirect(u64 ctx, u64 state, u64 buffer, u64 offset) {
    if (trace_took(decomp_gx::kGxDrawInstancedIndirect, ctx, state, buffer, offset, 0, 0, RET)) return;
    flush_output_merger(ctx, state);
    flush_input_assembly(ctx, state, 0);
    flush_rasterizer(ctx, state);
    command_buffer_flush(ctx);
    commit_state(ctx, state);
    game_function<CbU64>(kSetBaseIndirectArgs)(ctx + 0x10, load<u64>(buffer + 0x20));
    game_function<CbIndirect>(kDrawIndirect)(ctx + 0x10, static_cast<u32>(offset), indirect_stage(ctx), 0, 0);
    post_draw(ctx);
    command_buffer_flush(ctx);
}

// gx_immediate_draw_indexed_instanced_indirect(ctx, state, argument buffer, offset).
DECOMP_LEAF GX_METHOD void gx_immediate_draw_indexed_instanced_indirect(u64 ctx, u64 state, u64 buffer, u64 offset) {
    if (trace_took(decomp_gx::kGxDrawIndexedInstancedIndirect, ctx, state, buffer, offset, 0, 0, RET)) return;
    flush_output_merger(ctx, state);
    flush_input_assembly(ctx, state, 1);
    flush_rasterizer(ctx, state);
    command_buffer_flush(ctx);
    commit_state(ctx, state);
    game_function<CbU64>(kSetBaseIndirectArgs)(ctx + 0x10, load<u64>(buffer + 0x20));
    game_function<CbIndirect>(kDrawIndexIndirect)(ctx + 0x10, static_cast<u32>(offset), indirect_stage(ctx), 0, 0);
    post_draw(ctx);
    command_buffer_flush(ctx);
}

// gx_context_commit(ctx, state): a draw's state work without the draw.
DECOMP_LEAF GX_METHOD void gx_context_commit(u64 ctx, u64 state) {
    if (trace_took(decomp_gx::kGxCommit, ctx, state, 0, 0, 0, 0, RET)) return;
    flush_output_merger(ctx, state);
    flush_input_assembly(ctx, state, 1);
    flush_rasterizer(ctx, state);
    command_buffer_flush(ctx);
    commit_state(ctx, state);
    post_draw(ctx);
    command_buffer_flush(ctx);
}

// gx_immediate_dispatch(ctx, state, x, y, z).
DECOMP_LEAF GX_METHOD void gx_immediate_dispatch(u64 ctx, u64 state, u64 x, u64 y, u64 z) {
    if (trace_took(decomp_gx::kGxDispatch, ctx, state, x, y, z, 0, RET)) return;
    command_buffer_flush(ctx);
    game_function<Ctx3>(kDispatchCommit)(ctx + 0x50, ctx, state);
    game_function<CbU32x3>(kDispatchDirect)(ctx + 0x10, static_cast<u32>(MUTANT(6) ? y : x), static_cast<u32>(MUTANT(6) ? x : y),
                                            static_cast<u32>(z));
    game_function<Ctx3>(kDispatchPost)(ctx + 0x50, ctx, state);
    command_buffer_flush(ctx);
}

// gx_immediate_dispatch_indirect(ctx, state, argument buffer, offset).
DECOMP_LEAF GX_METHOD void gx_immediate_dispatch_indirect(u64 ctx, u64 state, u64 buffer, u64 offset) {
    if (trace_took(decomp_gx::kGxDispatchIndirect, ctx, state, buffer, offset, 0, 0, RET)) return;
    command_buffer_flush(ctx);
    game_function<Ctx3>(kDispatchCommit)(ctx + 0x50, ctx, state);
    game_function<CbU64>(kSetBaseIndirectArgs)(ctx + 0x10, load<u64>(buffer + 0x20));
    game_function<CbU32>(kDispatchIndirect)(ctx + 0x10, static_cast<u32>(offset));
    game_function<Ctx3>(kDispatchPost)(ctx + 0x50, ctx, state);
    command_buffer_flush(ctx);
}

#undef RET

bool decomp_gx::hand_to_trace(std::uint64_t kind, std::uint64_t rdi, std::uint64_t rsi, std::uint64_t rdx, std::uint64_t rcx, std::uint64_t r8,
                              std::uint64_t r9, std::uint64_t ret) {
    const auto entry = reinterpret_cast<GUEST_ABI u64 (*)(u64, const u64*)>(g_gx_hook_entry);
    if (!entry) return false;
    // A return slot, then what the stubs push: r9 r8 rcx rdx rsi rdi, the
    // return address and the first stack argument (none of these have one).
    const u64 frame[9] = {0, r9, r8, rcx, rdx, rsi, rdi, ret, 0};
    return entry(kind, frame + 1) != 0;
}

void decomp_gx::note_writer_caller(const void* ours, std::uint64_t game_return_bn) {
    if (g_writer_callers_n < static_cast<int>(sizeof g_writer_callers / sizeof g_writer_callers[0]))
        g_writer_callers[g_writer_callers_n++] = {reinterpret_cast<u64>(ours), game_return_bn};
    g_method_code_lo = GX_METHODS_LO;
    g_method_code_hi = GX_METHODS_HI;
}

std::uint64_t decomp_gx::game_return_in_methods(std::uint64_t ret) {
    // The function whose code holds it: the one entered last below it (the
    // section holds the writer callers told and nothing else).
    int best = -1;
    for (int k = 0; k < g_writer_callers_n; ++k) {
        if (g_writer_callers[k].entry <= ret && (best < 0 || g_writer_callers[k].entry > g_writer_callers[best].entry)) best = k;
    }
    return best < 0 ? ret : game_address(g_writer_callers[best].game_return_bn);
}

// gx_flush_output_merger(om = ctx + 0xb558, ctx, state): what changed in the
// output merger's state since the context last wrote it, against what it
// wrote (the block keeps it).
//
// First the deferred clears: of the colour targets the block recorded as
// bound (+0x146 of them, in the record set bit 0 of +0x188 selects: +0x60, a
// set of 0x48 bytes) each that the state no longer binds (+0x4e0) and whose
// texture has a clear pending (+0x3b & 0x8b) and a clear kind (+0x40, 1 a
// clear to zero, 2..8 to the colour at +0x48) gets its fast-clear eliminate
// pass, after the clear colours are written; the recorded depth target (+0xa0
// in the set bit 1 selects) one the state no longer binds (+0x520), with a
// clear pending (& 8), not one of a kind +0x5c sets and of a kind +0x48 of
// 1..6, its decompress pass.
//
// Then the blend state (+0x4d0, the default one for none) when it is not the
// one the block last wrote whole (+0x190): DB_ALPHA_TO_MASK (its dithered
// offsets, and alpha to coverage from +0x24), the target mask (+0), the eight
// targets' blend controls (+4) - each written only when it differs from what
// the block holds, and the block marked as holding no whole state while it
// does; the blend constants (+0x568: compared as floats, so -0 is +0 and a
// NaN always differs); the depth-stencil state (+0x4d8) the same way against
// +0x198 - the depth and stencil controls - and the stencil reference and
// masks with the state's reference byte (+0x584) as its test and op values;
// the depth bounds (+0x578, compared as bits).
//
// Gnm's register helpers the game calls on a local for DB_ALPHA_TO_MASK
// (0x1485600, 0x1485610; and 0x1484ea0 on one it never reads) are folded:
// they write nothing but the local.
DECOMP_LEAF void gx_flush_output_merger(u64 om, u64 ctx, u64 state) {
    using ClearPass = GUEST_ABI void (*)(u64 ctx, u64 view, u64 resource, u64 a, u64 b, u64 zero);
    for (u32 i = 0; i < load<u8>(om + 0x146); ++i) {
        const u64 rt = load<u64>(om + (load<u32>(om + 0x188) & 1) * 0x48 + 0x60 + 8ull * i);
        if (rt == load<u64>(state + 0x4e0 + 8ull * i) || rt + 1 < 2) continue;
        const u64 tex = load<u64>(rt + 8);
        if (!tex || !(load<u8>(tex + 0x3b) & 0x8b)) continue;
        const u32 kind = load<u32>(rt + 0x40);
        if (kind - 1 > 7) continue;
        const bool zero = kind == 1 && !MUTANT(11);
        const u64 c0 = zero ? 0 : load<u32>(rt + 0x48), c1 = zero ? 0 : load<u32>(rt + 0x4c);
        game_function<Ctx2>(kWriteClearColors)(om, ctx);
        game_function<ClearPass>(kFastClearEliminate)(ctx, rt + 0x10, tex + 0x18, c0, c1, 0);
    }
    const u32 depth_set = MUTANT(7) ? load<u32>(om + 0x188) & 1 : (load<u32>(om + 0x188) >> 1) & 1;
    const u64 depth = load<u64>(om + depth_set * 0x48 + 0xa0);
    if (depth != load<u64>(state + 0x520) && depth + 1 >= 2) {
        const u64 tex = load<u64>(depth + 8);
        if (tex && (load<u8>(tex + 0x3b) & 8) && load<u32>(depth + 0x5c) == 0 && load<u32>(depth + 0x48) - 1 <= 5) {
            game_function<ClearPass>(kDepthDecompress)(ctx, depth + 0x10, tex + 0x18, load<u32>(depth + 0x54), load<u32>(depth + 0x58), 0);
        }
    }

    const u64 cb = ctx + 0x10;
    using Set = GUEST_ABI void (*)(u64 cb, u64 value);
    const u64 blend = load<u64>(state + 0x4d0);
    if (blend != load<u64>(om + 0x190)) {
        const u64 bs = blend ? blend : game_address(kDefaultBlendState);
        const u32 coverage = load<u8>(bs + 0x24);
        if (coverage != load<u32>(om)) {
            // AlphaToMaskControl: the four pixels' offsets 2 (dithered), enabled by coverage's bit.
            game_function<Set>(kSetAlphaToMask)(cb, 0xaa00u | (coverage & 1));
            store<u32>(om, coverage);
        }
        const u64 mask = load<u32>(bs);
        if (mask != load<u64>(om + 8)) {
            game_function<Set>(kSetRenderTargetMask)(cb, mask);
            store<u64>(om + 0x190, ~0ull);
            store<u64>(om + 8, mask);
        }
        using SetBlend = GUEST_ABI void (*)(u64 cb, u64 slot, u64 value);
        for (u32 i = 0; i < 8; ++i) {
            const u32 control = load<u32>(bs + 4 + 4ull * i);
            if (load<u32>(om + 0x10 + 4ull * i) == control) continue;
            game_function<SetBlend>(kSetBlendControl)(cb, i, control);
            store<u32>(om + 0x10 + 4ull * i, load<u32>(bs + 4 + 4ull * i));
            store<u64>(om + 0x190, ~0ull);
        }
        store<u64>(om + 0x190, blend);
    }

    u32 colour_bits[4];
    float colour[4], held[4];
    for (int k = 0; k < 4; ++k) {
        colour_bits[k] = load<u32>(state + 0x568 + 4ull * k);
        std::memcpy(&colour[k], &colour_bits[k], 4);
        held[k] = load<float>(om + 0x30 + 4ull * k);
    }
    const bool same_colour = MUTANT(8) ? std::memcmp(colour, held, sizeof colour) == 0
                                       : colour[0] == held[0] && colour[1] == held[1] && colour[2] == held[2] && colour[3] == held[3];
    if (!same_colour) {
        using SetColour = GUEST_ABI void (*)(u64 cb, float r, float g, float b, float a);
        game_function<SetColour>(kSetBlendColor)(cb, colour[0], colour[1], colour[2], colour[3]);
        for (int k = 0; k < 4; ++k) store<u32>(om + 0x30 + 4ull * k, colour_bits[k]);
    }

    const u64 ds = load<u64>(state + 0x4d8);
    u64 stencil = om + 0x50;  // the reference and masks the block holds, unless a state gives them
    if (ds != load<u64>(om + 0x198)) {
        const u64 dss = ds ? ds : game_address(kDefaultDepthStencilState);
        const u32 depth_control = load<u32>(dss);
        if (depth_control != load<u32>(om + 0x48)) {
            game_function<Set>(kSetDepthStencilControl)(cb, depth_control);
            store<u32>(om + 0x48, depth_control);
            store<u64>(om + 0x198, ~0ull);
        }
        const u32 stencil_control = load<u32>(dss + 4);
        if (stencil_control != load<u32>(om + 0x4c)) {
            game_function<Set>(kSetStencilControl)(cb, stencil_control);
            store<u32>(om + 0x4c, stencil_control);
            store<u64>(om + 0x198, ~0ull);
        }
        if (MUTANT(9)) store<u64>(om + 0x198, ds);
        stencil = dss + 8;
    }
    // StencilControl: test value, mask, write mask, op value - the reference
    // byte as the test and op values, the masks from the state.
    const u32 reference = load<u8>(state + 0x584);
    const u32 ref = (load<u32>(stencil) & 0xffff00u) | reference | (MUTANT(10) ? 0 : reference << 24);
    if (ref != load<u32>(om + 0x50)) {
        game_function<Set>(kSetStencilRef)(cb, ref);
        store<u32>(om + 0x50, ref);
        store<u64>(om + 0x198, ~0ull);
    }

    const u64 bounds = load<u32>(state + 0x578) | static_cast<u64>(load<u32>(state + 0x57c)) << 32;
    if (bounds != load<u64>(om + 0x58)) {
        using SetBounds = GUEST_ABI void (*)(u64 cb, float min, float max);
        game_function<SetBounds>(kSetDepthBoundsRange)(cb, load<float>(state + 0x578), load<float>(state + 0x57c));
        store<u64>(om + 0x58, bounds);
    }
}

// gx_post_draw(state block = ctx + 0x50, ctx): the flushes' dirty bits
// (+0xb368 of the block: 0x80 and 0x10 a target or depth change, 0x102 and
// 0x60 how) folded into the context's flags for the next commit (+0xb498,
// +0xb478, and +0xb480 cleared when a resolve is owed), the bits cleared; then
// the input assembly's multi-VGT parameter written back to its default when
// the draw changed it (+0x8 not 0xff), and the GS mode turned off when a
// geometry shader was bound (+0x10). Returns what the game's returns: the
// last writer's result, else the 0x60 bits.
DECOMP_LEAF u64 gx_post_draw(u64 block, u64 ctx) {
    const u32 dirty = load<u32>(block + 0xb368);
    const u32 how = dirty & 0x102;
    u64 result = dirty & 0x60;
    for (const u32 bit : {0x80u, 0x10u}) {
        if (!(dirty & bit)) continue;
        const u32 flags = load<u32>(ctx + 0xb498) | 1;
        store<u32>(ctx + 0xb498, flags);
        const u32 commit = load<u32>(ctx + 0xb478);
        if (how) {
            store<u32>(ctx + 0xb478, commit | 5);
            store<u32>(ctx + 0xb480, 0);
        } else {
            store<u32>(ctx + 0xb478, commit | (result ? (MUTANT(3) ? 0x7u : 0xfu) : 1u));
        }
    }
    store<u32>(block + 0xb368, 0);
    if (load<u32>(block + 0x8) != 0xff) result = game_function<Writer>(kSetIaMultiVgtParam)(ctx + 0x10, 0xff, 0);
    if (load<u64>(block + 0x10) != 0) return game_function<Writer>(kSetGsMode)(ctx + 0x10, 0, 0);
    return result;
}

// gx_command_buffer_flush(cb = ctx + 0x10): the write pointer noted as
// flushed (+0xb4b8); and for a context recording into an arena chunk - with
// chunks queued (+0xb4d8) and the arena's refill as its own - the segment
// closed: the release nodes owed by the slots in +0xb460's bits (+0xb3d8
// onwards) and the pending list (+0xb458) handed to the block pool
// (+0xb3d0), 0x2acd3f0 on the state at +0x40, every chunk of the queue
// retired into the arena (+0xb4c8) - a queue page's last slot links the next
// page, the last page holds +0xb4ec of them - the queue handed to the object
// at +0xb4d0 (its +0x70), the queue cleared; then the segment closed - with a
// fresh start at the write pointer, unless fewer than seven dwords are left
// (the refill asked for a new chunk instead) or the compute ring's counter
// says this is its last one (closed without one).
DECOMP_LEAF void gx_command_buffer_flush(u64 cb) {
    store<u64>(cb + 0xb4b8, load<u64>(cb + 0x10));
    if (!load<u64>(cb + 0xb4d8) || load<u64>(cb + 0x18) != load<u64>(game_address(kDefaultRefill))) return;
    using Push = GUEST_ABI void (*)(u64 pool, u64* cb_ref, u64 node);
    const auto push = game_function<Push>(kReleaseNodePush);
    u64 cb_ref = cb;  // what the pool's push is handed: where the command buffer is
    for (u32 bits = load<u32>(cb + 0xb460); bits;) {
        const u32 slot = static_cast<u32>(__builtin_ctz(bits));
        for (u64 node = load<u64>(cb + 0xb3d8 + 8ull * slot); node;) {
            const u64 next = load<u64>(node + 0x28);
            push(load<u64>(cb + 0xb3d0), &cb_ref, node);
            node = next;
        }
        store<u64>(cb + 0xb3d8 + 8ull * slot, 0);
        bits -= 1u << slot;
        if (!bits) store<u32>(cb + 0xb460, 0);
    }
    for (u64 node = load<u64>(cb + 0xb458); node;) {
        const u64 next = load<u64>(node + 0x28);
        push(load<u64>(cb + 0xb3d0), &cb_ref, node);
        node = next;
    }
    store<u64>(cb + 0xb458, 0);
    game_function<Ctx2>(kFlushUnknown2acd3f0)(cb + 0x40, cb);
    const u64 last = load<u32>(cb + 0xb4e8);  // a queue page's slot that links the next
    for (u64 page = load<u64>(cb + 0xb4d8); page; page = load<u64>(page + 8 * last)) {
        for (u64 k = 0;; ++k) {
            const u64 count = load<u64>(page + 8 * last) ? last : load<u32>(cb + 0xb4ec);
            if (k >= count) break;
            game_function<Ctx3>(kRetireChunk)(load<u64>(cb + 0xb4c8), cb, load<u64>(page + 8 * k));
            if (!MUTANT(4)) store<u64>(cb + 0xb4b8, load<u64>(cb + 0x10));
        }
    }
    if (const u64 owner = load<u64>(cb + 0xb4d0)) {
        const u64 link = load<u32>(cb + 0xb4e8);
        if (const u64 queue = load<u64>(cb + 0xb4d8)) {
            using Take = GUEST_ABI void (*)(u64 self, u64 page);
            u64 page = load<u64>(queue + 8 * link);
            reinterpret_cast<Take>(static_cast<std::uintptr_t>(load<u64>(load<u64>(owner) + 0x70)))(owner, queue);
            while (page) {
                const u64 self = load<u64>(cb + 0xb4d0);
                const u64 next = load<u64>(page + 8 * link);
                reinterpret_cast<Take>(static_cast<std::uintptr_t>(load<u64>(load<u64>(self) + 0x70)))(self, page);
                page = next;
            }
        }
    }
    store<u32>(cb + 0xb4ec, 0);
    store<u64>(cb + 0xb4e0, 0);
    store<u64>(cb + 0xb4d8, 0);
    const u8 kind = load<u8>(cb + 0xb4f0);
    const u64 ring = load<u64>(load<u64>(cb + 0xb490) + 0xb728);
    u64 header[8];
    u64 fresh = 0;
    if (load<u32>(load<u64>(ring + 0x20)) == load<u32>(load<u64>(ring + 0x18)) - 1) {
        for (int k = 0; k < 8; ++k) header[k] = load<u64>(cb + 8ull * k);
    } else {
        const u64 wp = load<u64>(cb + 0x10);
        if (static_cast<u32>((load<u64>(cb + 0x8) - wp) >> 2) < 7) {
            // The refill for no dwords, the write pointer left in its fourth
            // argument's register as the game's leaves it.
            using Refill = GUEST_ABI void (*)(u64 cb, u32 dwords, u64 arg, u64 left);
            reinterpret_cast<Refill>(static_cast<std::uintptr_t>(load<u64>(cb + 0x18)))(cb, 0, load<u64>(cb + 0x20), wp);
            return;
        }
        for (int k = 0; k < 8; ++k) header[k] = load<u64>(cb + 8ull * (MUTANT(5) && k == 7 ? 6 : k));
        store<u64>(cb, load<u64>(cb + 0x10));
        fresh = cb;
    }
    using Close = GUEST_ABI void (*)(u64 cb, u64 fresh, u64* header, u32 kind);
    game_function<Close>(kCloseSegment)(cb, fresh, header, kind);
    const u64 wp = load<u64>(cb + 0x10);
    store<u64>(cb, wp);
    store<u64>(cb + 0xb4b8, wp);
}

namespace {

// ---- BBHOST_DECOMP_COMPARE=1 ----
// The post-draw and the flush's common case run the game's version in ours'
// place, ours beside it on the same memory, compared, the game's result kept.
// A post-draw whose writers might need the refill, and a flush that closes a
// segment (it hands nodes, chunks and the segment on - nothing to run twice),
// run the game's alone. The output-merger flush the same way: the game's
// alone when a deferred clear would run (the clear passes draw) or its
// writers might need the refill.
DecompCompare g_counts_post, g_counts_flush, g_counts_om;
void* g_original_post = nullptr;
void* g_original_flush = nullptr;
void* g_original_om = nullptr;
std::atomic<std::uint64_t> g_post_alone{0}, g_flush_alone{0}, g_om_alone{0};
std::atomic<int> g_compare_logs{0};

struct PostState {
    u32 dirty, commit, resolve, flags;
    u8 cb[0x30];
    u8 packets[64];
};
PostState post_state(u64 block, u64 ctx, u64 wp) {
    PostState s;
    s.dirty = load<u32>(block + 0xb368);
    s.commit = load<u32>(ctx + 0xb478);
    s.resolve = load<u32>(ctx + 0xb480);
    s.flags = load<u32>(ctx + 0xb498);
    std::memcpy(s.cb, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(ctx + 0x10)), sizeof s.cb);
    std::memcpy(s.packets, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), sizeof s.packets);
    return s;
}
void put_post_state(u64 block, u64 ctx, u64 wp, const PostState& s) {
    store<u32>(block + 0xb368, s.dirty);
    store<u32>(ctx + 0xb478, s.commit);
    store<u32>(ctx + 0xb480, s.resolve);
    store<u32>(ctx + 0xb498, s.flags);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(ctx + 0x10)), s.cb, sizeof s.cb);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(wp)), s.packets, sizeof s.packets);
}

u64 run_guest(void* fn, u64 a0, u64 a1, u64 a2 = 0) {
    GuestCallRegs g{};
    g.fn = fn;
    g.ints[0] = a0;
    g.ints[1] = a1;
    g.ints[2] = a2;
    hle_call_guest_regs(&g);
    return g.ret;
}

GUEST_ABI u64 compare_post_draw(u64 block, u64 ctx) {
    const u64 cb = ctx + 0x10, wp = load<u64>(cb + 0x10);
    // Room for both writers' packets without the refill (16 dwords).
    if (!load<u64>(cb) || load<u64>(cb + 0x8) < wp + 64) {
        g_post_alone.fetch_add(1, std::memory_order_relaxed);
        return run_guest(g_original_post, block, ctx);
    }
    const PostState before = post_state(block, ctx, wp);
    const u64 game_ret = run_guest(g_original_post, block, ctx);
    const PostState game = post_state(block, ctx, wp);
    put_post_state(block, ctx, wp, before);
    const u64 our_ret = run_guest(reinterpret_cast<void*>(&gx_post_draw), block, ctx);
    const PostState ours = post_state(block, ctx, wp);
    g_counts_post.calls.fetch_add(1, std::memory_order_relaxed);
    if (our_ret != game_ret || std::memcmp(&ours, &game, sizeof ours) != 0) {
        g_counts_post.differ.fetch_add(1, std::memory_order_relaxed);
        if (g_compare_logs.fetch_add(1) < 16) {
            host_log("decomp: gx_post_draw differs from the game's: dirty 0x%x, flags 0x%x/0x%x, commit 0x%x/0x%x, returned 0x%llx/0x%llx",
                     before.dirty, ours.flags, game.flags, ours.commit, game.commit, static_cast<ull>(our_ret), static_cast<ull>(game_ret));
        }
    }
    put_post_state(block, ctx, wp, game);
    return game_ret;
}

GUEST_ABI void compare_flush(u64 cb) {
    if (load<u64>(cb + 0xb4d8) && load<u64>(cb + 0x18) == load<u64>(decomp_guest(kDefaultRefill))) {
        g_flush_alone.fetch_add(1, std::memory_order_relaxed);
        run_guest(g_original_flush, cb, 0);
        return;
    }
    const u64 before = load<u64>(cb + 0xb4b8);
    run_guest(g_original_flush, cb, 0);
    const u64 game = load<u64>(cb + 0xb4b8);
    store<u64>(cb + 0xb4b8, before);
    run_guest(reinterpret_cast<void*>(&gx_command_buffer_flush), cb, 0);
    g_counts_flush.calls.fetch_add(1, std::memory_order_relaxed);
    if (load<u64>(cb + 0xb4b8) != game) {
        g_counts_flush.differ.fetch_add(1, std::memory_order_relaxed);
        if (g_compare_logs.fetch_add(1) < 16) host_log("decomp: gx_command_buffer_flush differs from the game's at cb 0x%llx", static_cast<ull>(cb));
    }
    store<u64>(cb + 0xb4b8, game);
}

// Whether the output-merger flush would run a deferred clear: its own tests,
// as gx_flush_output_merger makes them.
bool om_clears(u64 om, u64 state) {
    const u32 sets = load<u32>(om + 0x188);
    for (u32 i = 0; i < load<u8>(om + 0x146); ++i) {
        const u64 rt = load<u64>(om + (sets & 1) * 0x48 + 0x60 + 8ull * i);
        if (rt == load<u64>(state + 0x4e0 + 8ull * i) || rt + 1 < 2) continue;
        const u64 tex = load<u64>(rt + 8);
        if (tex && (load<u8>(tex + 0x3b) & 0x8b) && load<u32>(rt + 0x40) - 1 <= 7) return true;
    }
    const u64 depth = load<u64>(om + ((sets >> 1) & 1) * 0x48 + 0xa0);
    if (depth == load<u64>(state + 0x520) || depth + 1 < 2) return false;
    const u64 tex = load<u64>(depth + 8);
    return tex && (load<u8>(tex + 0x3b) & 8) && load<u32>(depth + 0x5c) == 0 && load<u32>(depth + 0x48) - 1 <= 5;
}

// What the flush writes: the block (to +0x198's last byte), the command
// buffer's header and its packets (fewer than 64 dwords, all of them written).
constexpr std::size_t kOmBlock = 0x1a0, kOmPackets = 0x100;
struct OmState {
    u8 block[kOmBlock];
    u8 cb[0x30];
    u8 packets[kOmPackets];
};
void om_state(u64 om, u64 ctx, u64 wp, OmState& s) {
    std::memcpy(s.block, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(om)), sizeof s.block);
    std::memcpy(s.cb, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(ctx + 0x10)), sizeof s.cb);
    std::memcpy(s.packets, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), sizeof s.packets);
}
void put_om_state(u64 om, u64 ctx, u64 wp, const OmState& s) {
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(om)), s.block, sizeof s.block);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(ctx + 0x10)), s.cb, sizeof s.cb);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(wp)), s.packets, sizeof s.packets);
}

GUEST_ABI void compare_om_flush(u64 om, u64 ctx, u64 state) {
    const u64 cb = ctx + 0x10, wp = load<u64>(cb + 0x10);
    if (om_clears(om, state) || !load<u64>(cb) || load<u64>(cb + 0x8) < wp + kOmPackets) {
        g_om_alone.fetch_add(1, std::memory_order_relaxed);
        run_guest(g_original_om, om, ctx, state);
        return;
    }
    OmState before, game, ours;
    om_state(om, ctx, wp, before);
    run_guest(g_original_om, om, ctx, state);
    om_state(om, ctx, wp, game);
    put_om_state(om, ctx, wp, before);
    run_guest(reinterpret_cast<void*>(&gx_flush_output_merger), om, ctx, state);
    om_state(om, ctx, wp, ours);
    g_counts_om.calls.fetch_add(1, std::memory_order_relaxed);
    if (std::memcmp(&ours, &game, sizeof ours) != 0) {
        g_counts_om.differ.fetch_add(1, std::memory_order_relaxed);
        if (g_compare_logs.fetch_add(1) < 16) {
            std::size_t at = 0;
            while (at < sizeof ours && reinterpret_cast<const u8*>(&ours)[at] == reinterpret_cast<const u8*>(&game)[at]) ++at;
            host_log("decomp: gx_flush_output_merger differs from the game's at byte 0x%zx of the %s (block 0x%llx)", at,
                     at < kOmBlock ? "block" : at < kOmBlock + 0x30 ? "command buffer's header" : "packets", static_cast<ull>(om));
        }
    }
    put_om_state(om, ctx, wp, game);
}

void report() {
    static bool once = false;
    if (once) return;
    once = true;
    using decomp_gx::kMethodNames;
    std::uint64_t calls = 0, taken = 0;
    char by[512];
    int at = 0;
    for (int k = 0; k < 9; ++k) {
        const std::uint64_t n = total_calls(k), t = total_taken(k);
        calls += n;
        taken += t;
        if (n && at < static_cast<int>(sizeof by) - 64)
            at += std::snprintf(by + at, sizeof by - static_cast<std::size_t>(at), " %s %llu/%llu;", kMethodNames[k] + 3,
                                static_cast<ull>(t), static_cast<ull>(n));
    }
    by[at] = 0;
    host_log("decomp: GX's immediate context: %llu calls of its draw, dispatch and commit methods, %llu taken whole by the trace (host-draw "
             "tokens), the rest run as the game's sequence - taken/called:%s",
             static_cast<ull>(calls), static_cast<ull>(taken), by);
    if (g_counts_post.calls.load() || g_counts_flush.calls.load() || g_counts_om.calls.load()) {
        host_log("decomp: compared (BBHOST_DECOMP_COMPARE=1): gx_flush_output_merger %llu (%llu differing, %llu the game's alone: a deferred "
                 "clear, or the writers might need the refill), gx_post_draw %llu (%llu differing, %llu the game's alone: the writers might need "
                 "the refill), gx_command_buffer_flush %llu (%llu differing, %llu the game's alone: a segment closed)",
                 static_cast<ull>(g_counts_om.calls.load()), static_cast<ull>(g_counts_om.differ.load()), static_cast<ull>(g_om_alone.load()),
                 static_cast<ull>(g_counts_post.calls.load()), static_cast<ull>(g_counts_post.differ.load()), static_cast<ull>(g_post_alone.load()),
                 static_cast<ull>(g_counts_flush.calls.load()), static_cast<ull>(g_counts_flush.differ.load()),
                 static_cast<ull>(g_flush_alone.load()));
    }
}

// The entries: the methods' prologues as the GX trace's stubs knew them
// (push rbp; mov rbp, rsp; push r15..rbx; then push rax, sub rsp, or a mov).
#define PUSH6 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53
constexpr u8 kEntryPushRax[] = {PUSH6, 0x50};
constexpr u8 kEntrySub18[] = {PUSH6, 0x48, 0x83, 0xec, 0x18};
// push rbp; mov rbp, rsp; push r15, r14, r12, rbx; mov r14, rsi.
constexpr u8 kEntryCommit[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x49, 0x89, 0xf6};
// push rbp; mov rbp, rsp; push r14; push rbx; mov rbx, rsi; mov r14, rdi.
constexpr u8 kEntryPostDraw[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89, 0xf3, 0x49, 0x89, 0xfe};
// PUSH6; sub rsp, 0x58 (the stack guard's rip-relative load follows).
constexpr u8 kEntryFlush[] = {PUSH6, 0x48, 0x83, 0xec, 0x58};
// PUSH6; sub rsp, 0x38; mov r15, rdx.
constexpr u8 kEntryOutputMerger[] = {PUSH6, 0x48, 0x83, 0xec, 0x38, 0x49, 0x89, 0xd7};
#undef PUSH6

}  // namespace

void decomp_gx_draw_add() {
    using decomp_gx::kMethodNames;
    struct Method {
        u64 bn;
        const u8* entry;
        std::size_t n;
        void* ours;
    };
    static const Method kMethods[9] = {
        {0x25696c0, kEntryPushRax, sizeof kEntryPushRax, reinterpret_cast<void*>(&gx_immediate_draw)},
        {0x2569780, kEntryPushRax, sizeof kEntryPushRax, reinterpret_cast<void*>(&gx_immediate_draw_indexed)},
        {0x2569840, kEntryPushRax, sizeof kEntryPushRax, reinterpret_cast<void*>(&gx_immediate_draw_instanced)},
        {0x2569900, kEntrySub18, sizeof kEntrySub18, reinterpret_cast<void*>(&gx_immediate_draw_indexed_instanced)},
        {0x25699d0, kEntryPushRax, sizeof kEntryPushRax, reinterpret_cast<void*>(&gx_immediate_draw_instanced_indirect)},
        {0x2569ab0, kEntryPushRax, sizeof kEntryPushRax, reinterpret_cast<void*>(&gx_immediate_draw_indexed_instanced_indirect)},
        {0x2569ba0, kEntryCommit, sizeof kEntryCommit, reinterpret_cast<void*>(&gx_context_commit)},
        {0x25695c0, kEntryPushRax, sizeof kEntryPushRax, reinterpret_cast<void*>(&gx_immediate_dispatch)},
        {0x2569640, kEntryPushRax, sizeof kEntryPushRax, reinterpret_cast<void*>(&gx_immediate_dispatch_indirect)},
    };
    for (int k = 0; k < 9; ++k) {
        DecompFunction fn{kMethodNames[k], "gx", kMethods[k].bn, kMethods[k].entry, kMethods[k].n, kMethods[k].ours, DecompKind::Leaf};
        if (k == 0) fn.report = &report;
        decomp_add(fn);
        decomp_gx::note_writer_caller(kMethods[k].ours, kGameReturns[k]);
    }
    DecompFunction om{"gx_flush_output_merger", "gx", kFlushOutputMerger, kEntryOutputMerger, sizeof kEntryOutputMerger,
                      reinterpret_cast<void*>(&gx_flush_output_merger), DecompKind::Leaf};
    om.original = &g_original_om;
    om.compare = reinterpret_cast<void*>(&compare_om_flush);
    om.counts = &g_counts_om;
    decomp_add(om);
    DecompFunction post{"gx_post_draw", "gx", kPostDraw, kEntryPostDraw, sizeof kEntryPostDraw, reinterpret_cast<void*>(&gx_post_draw),
                        DecompKind::Leaf};
    post.original = &g_original_post;
    post.compare = reinterpret_cast<void*>(&compare_post_draw);
    post.counts = &g_counts_post;
    decomp_add(post);
    DecompFunction flush{"gx_command_buffer_flush", "gx", kCommandBufferFlush, kEntryFlush, sizeof kEntryFlush,
                         reinterpret_cast<void*>(&gx_command_buffer_flush), DecompKind::Leaf};
    flush.original = &g_original_flush;
    flush.compare = reinterpret_cast<void*>(&compare_flush);
    flush.counts = &g_counts_flush;
    decomp_add(flush);
}
