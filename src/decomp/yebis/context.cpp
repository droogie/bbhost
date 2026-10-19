// YEBIS's device context as source (docs/decomp.md). YEBIS - the
// post-process library under the engine's PPFX layer - keeps its own
// D3D11-shaped device and context on the console (engine/gx_resources.cpp
// names the device's creators): the context's vtable, 0x570a380, holds
// ID3D11DeviceContext's 115 methods in ID3D11DeviceContext's order and 18 of
// YEBIS's own after them. PPFX calls it as it would call D3D11 - its
// SetRenderTarget (0x10713e0) calls slot 33, OMSetRenderTargets, with a
// render target count, the views and the depth view - and the context turns
// each call into Gnm work through YEBIS's Gnm wrappers
// (decomp/yebis/gnm.cpp).
//
// Here: OMSetRenderTargets (slot 33, 0x15d74c0) with the hazard drain it ends
// with, the draws' and the dispatches' prologue (0x15d7400, 0x2e433a0), the
// four draws and two dispatches PPFX can call (slots 12, 13, 20, 21, 41,
// 42), Flush (slot 111, 0x15d99e0), YEBIS's two flushes of its own (slots
// 131 and 132) and the three waits its compute work brings (0x15e8e20,
// 0x15ebc20, 0x15ebc90). Leaves: each reads and writes the context and calls
// the game's functions in the game's order with the game's arguments.
//
// YEBIS tracks its own hazards, as a D3D11 driver does: the context counts
// draws (+0xc20) and remembers the count at its last drain (+0xc28); a
// texture remembers the count it was last bound at (+0xa0 of the object at
// +0x80 of its view). A target bound again after a draw since the last drain
// costs a drain - an end-of-pipe label and a wait for it, then the caches
// written back and invalidated - before anything draws into it; a dispatch
// after a draw waits for the pixel shaders, one after a dispatch for the
// compute shaders, a draw after a dispatch for the compute shaders. That is
// the place a mod sees YEBIS's passes as D3D11 calls - which targets each
// draws into, in which order - and where a post-processing chain could be
// reordered or extended at D3D11 granularity (docs/modding.md).
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include "core/thunk.h"
#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstring>

using namespace decomp;

// tests/decomp/yebis_context_test.cpp builds this file with mutants: small
// wrong turns it switches on one at a time and must catch.
#ifdef DECOMP_YEBIS_CONTEXT_MUTANTS
extern int g_context_mutant;
#define MUTANT(n) (g_context_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

// What these read and write of the context.
constexpr u64 kGnm = 0xa98;           // YEBIS's Gnm context (decomp/yebis/gnm.cpp's)
constexpr u64 kCb = 0xaa0;            //   and its +0x8: the command buffer
constexpr u64 kDraws = 0xc20;         // draws and dispatches so far
constexpr u64 kDrained = 0xc28;       // that count at the last drain
constexpr u64 kFlushLabel = 0xc30;    // the flushes' label
constexpr u64 kTargets = 0x2e80;      // the colour targets' views, 8
constexpr u64 kTargetMask = 0x2ec0;   // their channels, 4 bits a slot
constexpr u64 kDepthView = 0x2ec8;
constexpr u64 kSize = 0x2ed0;         // the targets' size (two u32), and a byte: changed
constexpr u64 kSizeChanged = 0x2ed8;
constexpr u64 kDirty = 0x33c4;        // a byte of state to write at the next draw (0x10 the pixel shader)
constexpr u64 kStageSet = 0x33c5;     // a byte, cleared by the flushes
constexpr u64 kFlushes = 0x3400;      // flushes so far; a view keeps the count it was bound at
constexpr u64 kIndexStride = 0x34c0;  // u32: an index's size in bytes
constexpr u64 kIndexBase = 0x34c8;    // the index buffer's address
constexpr u64 kIndexSize = 0x34d0;    // u32: Gnm's index size
constexpr u64 kDispatched = 0x34d4;   // u32: 1 when the last draw or dispatch was a dispatch
constexpr u64 kComputeMode = 0x34d8;  // u32: 1 in a run of dispatches (the command buffer's flag 2)
constexpr u64 kPrimType = 0x34dc;     // u32
constexpr u64 kComputeShader = 0x3188;
constexpr u64 kExport = 0x3518;       // u32: which of the pixel shader's variants the targets need
constexpr u64 kAaMode = 0x3528;       // (fragments, samples), as log2, a row of the table below
constexpr u64 kAlphaToMask = 0x3530;  // Gnm's AlphaToMaskControl
// The (fragments, samples) table, rows of 0x28 bytes by fragments, a u64 a
// sample count (data at 0x570a7b0, through a relocated pointer), and the mode
// a call starts from (u32 fragments at +0, samples at +8).
constexpr u64 kAaModes = 0x57e4530, kDefaultAaMode = 0x5709420;
// The pixel shader variants' marks (the draw's prologue picks the variant by them).
constexpr u32 kExportAny = 0x43d11c, kExportWide = 0x98967f, kExportMixed = 0;

constexpr u64 kAcquireMem = 0x1477f40;     // gnm_acquire_mem(cb, cache action, extended action, engine)
constexpr u64 kEmbedData = 0x14760c0;      // gnm_embed_data(cb, bytes, alignment as log2): room in a NOP
constexpr u64 kEndOfPipe = 0x14778f0;      // gnm_write_end_of_pipe_event(cb, event, dst, label, data, value, cache action, policy)
constexpr u64 kEndOfShader = 0x1478320;    // gnm_write_end_of_shader_event(cb, event, label, value)
constexpr u64 kWaitRegMem = 0x1477b30;     // gnm_wait_reg_mem(cb, address, mask, function, reference)
constexpr u64 kSetDepthTarget = 0x14739b0; // gnm_set_depth_render_target(cb, depth target or null)
constexpr u64 kSetPrimType = 0x1473fe0;    // gnm_set_primitive_type(cb, type)
constexpr u64 kSetIndexSize = 0x1475270;   // gnm_set_index_size(cb, size)
constexpr u64 kSetInstances = 0x1475370;   // gnm_set_num_instances(cb, count)
constexpr u64 kSetIndirectBase = 0x14742f0;  // gnm_set_base_indirect_args(cb, address)
constexpr u64 kDefaultState = 0x1472870;   // gnm_init_default_hardware_state(cb)
constexpr u64 kRegister = 0x2e42eb0;       // a view's first bind: its owner's list (owner, view), under a mutex
constexpr u64 kSetTarget = 0x15fbdb0;      // yebis_gnm_set_render_target(gnm, slot, Gnm render target or null)
constexpr u64 kApplyStages = 0x15d8140;    // the shaders and their resources, at a draw
constexpr u64 kApplyOutput = 0x15d8830;    // depth-stencil, blend, rasterizer, viewport and scissor, at a draw
constexpr u64 kSetComputeShader = 0x15fb840;  // yebis_gnm_set_cs_shader(gnm, shader)
constexpr u64 kSubmit = 0x15fb2c0;         // yebis_gnm_submit(gnm): sceGnmSubmitCommandBuffers
constexpr u64 kNextBuffer = 0x15fae90;     // yebis_gnm_next_buffer(gnm): the ring's next command buffers
constexpr u64 kGnmDrawIndex = 0x15fbc40;   // decomp/yebis/gnm.cpp
constexpr u64 kGnmDrawAuto = 0x15fbb50;
constexpr u64 kGnmDispatch = 0x15fbd10;
constexpr u64 kGnmDispatchIndirect = 0x15fbd70;  // yebis_gnm_dispatch_indirect(gnm, offset)
constexpr u64 kPrepDraw = 0x15f9f30;       // yebis_gnm_prep_draw(state, cb, a, b)
// Ours, called by the game's address as the game calls them.
constexpr u64 kPrepareDraw = 0x15d7400, kPrepareDispatch = 0x2e433a0;
constexpr u64 kWaitCompute = 0x15e8e20, kWaitPixels = 0x15ebc20, kWaitDispatch = 0x15ebc90;

// Gnm's cache actions and end-of-pipe/shader events, as these pass them.
constexpr u32 kFlushInvalidateL1L2 = 0x38, kInvalidateL1 = 0x10;
constexpr u32 kEopFlushCbDb = 4, kEosCsDone = 0x2f, kEosPsDone = 0x30;

using One = GUEST_ABI u64 (*)(u64);
using Two = GUEST_ABI u64 (*)(u64, u64);
using Three = GUEST_ABI u64 (*)(u64, u64, u64);
using Four = GUEST_ABI u64 (*)(u64, u64, u64, u64);
using Five = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64);
using Eight = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64, u64, u64, u64);

u64 cb_of(u64 ctx) { return load<u64>(ctx + kCb); }

// A target's texture drawn since the last drain (or bound at a count since):
// it takes the current count, and the call ends with a drain.
bool claim(u64 ctx, u64 view) {
    const u64 tex = load<u64>(view + 0x80);
    const u64 draws = load<u64>(ctx + kDraws);
    if (load<u64>(tex + 0xa0) >= draws) return false;
    if (MUTANT(1) ? load<u64>(ctx + kDrained) > load<u64>(tex + 0xa0) : load<u64>(ctx + kDrained) >= load<u64>(tex + 0xa0)) return false;
    store<u64>(tex + 0xa0, draws);
    return true;
}

// A view's first bind puts it on its owner's list.
void register_once(u64 view) {
    if (load<u8>(view + 0x90)) return;
    store<u8>(view + 0x90, 1);
    game_function<Two>(kRegister)(load<u64>(view + 0x80), view);
}

// A view bound now: it, and its texture, keep the flush count.
void stamp(u64 ctx, u64 view) {
    store<u64>(view + 0x10, load<u64>(ctx + kFlushes));
    store<u64>(load<u64>(view + 0x80) + 0x10, load<u64>(ctx + kFlushes));
}

// The label, the end-of-pipe event that writes it and the wait for it - the
// drain - then the caches.
u64 drain(u64 ctx) {
    store<u64>(ctx + kDrained, load<u64>(ctx + kDraws));
    const u64 label = game_function<Three>(kEmbedData)(cb_of(ctx), 8, 3);
    store<u64>(label, 0);
    game_function<Eight>(kEndOfPipe)(cb_of(ctx), kEopFlushCbDb, 0, label, 2, 1, 0, 0);
    game_function<Five>(kWaitRegMem)(cb_of(ctx), label, 0xffffffffu, 3, 1);
    return game_function<Four>(kAcquireMem)(cb_of(ctx), kFlushInvalidateL1L2, 0, 0);
}

// Gnm's RenderTarget fields these read (its accessors, folded): the size kept
// beside the registers (two u16 at +0x18), and CB_COLOR_ATTRIB's log2 sample
// and fragment counts.
u32 rt_samples(u64 rt) { return (load<u32>(rt + 0x14) >> 12) & 7; }
u32 rt_fragments(u64 rt) { return (load<u32>(rt + 0x14) >> 15) & 3; }
u64 aa_mode(u32 fragments, u32 samples) {
    return load<u64>(load<u64>(game_address(kAaModes)) + static_cast<u64>(fragments) * 0x28 + static_cast<u64>(samples) * 8);
}

}  // namespace

// OMSetRenderTargets(ctx, count, views, depth view): the depth target set
// (or unset), up to eight colour targets set and the rest unset, the views
// kept with the flush count, the targets' size, sample mode and channel mask
// noted for the draw's prologue, and the pixel shader variant their formats
// need. Returns 0, or after a drain what the cache action's writer returns.
DECOMP_LEAF u64 yebis_context_om_set_render_targets(u64 ctx, u32 count, u64 views, u64 depth) {
    const u64 dflt = game_address(kDefaultAaMode);
    store<u64>(ctx + kAaMode, aa_mode(load<u32>(dflt), load<u32>(dflt + 8)));
    store<u32>(ctx + kAlphaToMask, 0xaa00u);  // Gnm's AlphaToMaskControl reset (0x1485600)
    u32 width = 0, height = 0;
    bool drained = false;
    if (!depth) {
        game_function<Two>(kSetDepthTarget)(cb_of(ctx), 0);
    } else {
        register_once(depth);
        game_function<Two>(kSetDepthTarget)(cb_of(ctx), depth + 0xc0);
        const u32 size = load<u32>(depth + 0xf0);
        width = size & 0xffff;
        height = size >> 16;
        stamp(ctx, depth);
        drained = claim(ctx, depth);
    }
    store<u64>(ctx + kDepthView, depth);
    store<u32>(ctx + kTargetMask, 0);
    for (u32 i = 0; i < 8; ++i) {
        const u64 slot = ctx + kTargets + 8ull * i;
        const u64 view = i < count ? load<u64>(views + 8ull * i) : 0;
        if (!view) {
            game_function<Three>(kSetTarget)(ctx + kGnm, i, 0);
            // The view this slot held loses the flush count it took now.
            const u64 old = load<u64>(slot);
            if (old && load<u64>(old + 0x10) == load<u64>(ctx + kFlushes) && load<u64>(ctx + kFlushes)) {
                store<u64>(old + 0x10, load<u64>(old + 0x10) - 1);
                store<u64>(load<u64>(load<u64>(slot) + 0x80) + 0x10, load<u64>(ctx + kFlushes) - 1);
            }
            store<u64>(slot, 0);
            continue;
        }
        register_once(view);
        const u64 rt = load<u64>(load<u64>(view + 0xd8));
        game_function<Three>(kSetTarget)(ctx + kGnm, i, rt);
        if (MUTANT(2) || !width) {
            width = load<u16>(rt + 0x18);
            height = load<u16>(rt + 0x1a);
        }
        if (rt_samples(rt) != load<u32>(ctx + kAaMode + 4) || rt_fragments(rt) != load<u32>(ctx + kAaMode)) {
            store<u64>(ctx + kAaMode, aa_mode(rt_fragments(rt), rt_samples(rt)));
        }
        store<u64>(slot, view);
        store<u32>(ctx + kTargetMask, load<u32>(ctx + kTargetMask) | 0xfu << (MUTANT(3) ? i : 4 * i));
        stamp(ctx, view);
        if (claim(ctx, view)) drained = true;
        store<u32>(view + 0xe0, i);
    }
    if (width != load<u32>(ctx + kSize) || height != load<u32>(ctx + kSize + 4)) {
        store<u32>(ctx + kSize, width);
        store<u32>(ctx + kSize + 4, height);
        store<u8>(ctx + kSizeChanged, 1);
    }
    // The targets' formats (YEBIS's, +0xe4 of a view), up to the first slot
    // unset: none past 0x10, any variant; one past it, the wide one, unless
    // one is 0x10 too.
    u32 variant = kExportAny;
    for (int s = 0; s < 8; ++s) {
        const u64 view = load<u64>(ctx + kTargets + 8ull * s);
        if (!view) break;
        if (load<u32>(view + 0xe4) < 0x11) continue;
        variant = kExportWide;
        for (int t = 0; t < 8; ++t) {
            const u64 other = load<u64>(ctx + kTargets + 8ull * t);
            if (!other) break;
            if (load<u32>(other + 0xe4) == (MUTANT(4) ? 0x11u : 0x10u)) {
                variant = kExportMixed;
                break;
            }
        }
        break;
    }
    if (variant != load<u32>(ctx + kExport)) {
        store<u32>(ctx + kExport, variant);
        store<u8>(ctx + kDirty, load<u8>(ctx + kDirty) | 0x10);
    }
    return drained ? drain(ctx) : 0;
}

// The prologue of every draw (and of the flushes): the caches, out of a run
// of dispatches, the shaders' and the output state written, the primitive
// type, a wait for the compute shaders after a dispatch, the count. Returns
// what its last writer returned, as the game's leaves in rax.
DECOMP_LEAF u64 yebis_context_prepare_draw(u64 ctx) {
    game_function<Four>(kAcquireMem)(cb_of(ctx), kFlushInvalidateL1L2, 0, 0);
    if (load<u32>(ctx + kComputeMode)) {
        store<u32>(ctx + kComputeMode, 0);
        const u64 cb = cb_of(ctx);
        store<u64>(cb + 0x28, load<u64>(cb + 0x28) & ~2ull);
    }
    game_function<One>(kApplyStages)(ctx);
    game_function<One>(kApplyOutput)(ctx);
    u64 r = game_function<Two>(kSetPrimType)(cb_of(ctx), load<u32>(ctx + kPrimType));
    if (load<u32>(ctx + kDispatched) == 1) r = game_function<One>(kWaitCompute)(cb_of(ctx));
    if (!MUTANT(5)) store<u32>(ctx + kDispatched, 0);
    store<u64>(ctx + kDraws, load<u64>(ctx + kDraws) + 1);
    return r;
}

// The prologue of a dispatch: the caches, into a run of dispatches, the
// compute shader, a wait for the work before (the compute shaders after a
// dispatch, the pixel shaders after a draw), the count.
DECOMP_LEAF u64 yebis_context_prepare_dispatch(u64 ctx) {
    game_function<Four>(kAcquireMem)(cb_of(ctx), kFlushInvalidateL1L2, 0, 0);
    if (load<u32>(ctx + kComputeMode) != 1) {
        store<u32>(ctx + kComputeMode, 1);
        const u64 cb = cb_of(ctx);
        store<u64>(cb + 0x28, load<u64>(cb + 0x28) | 2);
    }
    game_function<Two>(kSetComputeShader)(ctx + kGnm, load<u64>(load<u64>(ctx + kComputeShader) + 0x88));
    const u64 r = game_function<One>(load<u32>(ctx + kDispatched) == 1 ? kWaitDispatch : kWaitPixels)(cb_of(ctx));
    store<u32>(ctx + kDispatched, 1);
    store<u64>(ctx + kDraws, load<u64>(ctx + kDraws) + 1);
    return r;
}

// DrawIndexed(ctx, index count, start index, base vertex).
DECOMP_LEAF u64 yebis_context_draw_indexed(u64 ctx, u32 count, u32 start, u32 base_vertex) {
    game_function<One>(kPrepareDraw)(ctx);
    game_function<Two>(kSetIndexSize)(cb_of(ctx), load<u32>(ctx + kIndexSize));
    const u64 indices = static_cast<u64>(MUTANT(6) ? static_cast<u64>(start) * load<u32>(ctx + kIndexStride)
                                                   : static_cast<u32>(start * load<u32>(ctx + kIndexStride))) +
                        load<u64>(ctx + kIndexBase);
    return game_function<Five>(kGnmDrawIndex)(ctx + kGnm, count, indices, base_vertex, 0);
}

// Draw(ctx, vertex count): the start vertex is not passed on (YEBIS's draws start at 0).
DECOMP_LEAF u64 yebis_context_draw(u64 ctx, u32 count) {
    game_function<One>(kPrepareDraw)(ctx);
    return game_function<Two>(kGnmDrawAuto)(ctx + kGnm, count);
}

// The game's assertion trap (int 0x41), where a draw is given what YEBIS's
// context does not do: a start instance, a start vertex.
#define YEBIS_TRAP() __asm__ volatile("int $0x41")

// DrawIndexedInstanced(ctx, indices an instance, instances, start index, base vertex, start instance).
DECOMP_LEAF u64 yebis_context_draw_indexed_instanced(u64 ctx, u32 count, u32 instances, u32 start, u32 base_vertex, u32 start_instance) {
    if (start_instance) YEBIS_TRAP();
    game_function<One>(kPrepareDraw)(ctx);
    game_function<Two>(kSetIndexSize)(cb_of(ctx), load<u32>(ctx + kIndexSize));
    game_function<Two>(kSetInstances)(cb_of(ctx), instances);
    const u64 indices = static_cast<u64>(static_cast<u32>(start * load<u32>(ctx + kIndexStride))) + load<u64>(ctx + kIndexBase);
    game_function<Five>(kGnmDrawIndex)(ctx + kGnm, count, indices, base_vertex, start_instance);
    return game_function<Two>(kSetInstances)(cb_of(ctx), 1);
}

// DrawInstanced(ctx, vertices an instance, instances, start vertex, start
// instance). The prep is given the context's +0x80, not the Gnm context's
// state - as the game's does; both values are 0 by then.
DECOMP_LEAF u64 yebis_context_draw_instanced(u64 ctx, u32 count, u32 instances, u32 start_vertex, u32 start_instance) {
    if (start_vertex) YEBIS_TRAP();
    if (start_instance) YEBIS_TRAP();
    game_function<One>(kPrepareDraw)(ctx);
    game_function<Two>(kSetInstances)(cb_of(ctx), instances);
    game_function<Four>(kPrepDraw)(ctx + 0x80, cb_of(ctx), start_vertex, start_instance);
    game_function<Two>(kGnmDrawAuto)(ctx + kGnm, count);
    return game_function<Two>(kSetInstances)(cb_of(ctx), MUTANT(7) ? instances : 1);
}

#undef YEBIS_TRAP

// Dispatch(ctx, x, y, z).
DECOMP_LEAF u64 yebis_context_dispatch(u64 ctx, u32 x, u32 y, u32 z) {
    game_function<One>(kPrepareDispatch)(ctx);
    return game_function<Four>(kGnmDispatch)(ctx + kGnm, x, y, z);
}

// DispatchIndirect(ctx, argument buffer, offset): the buffer's memory (+0x98) as the indirect base.
DECOMP_LEAF u64 yebis_context_dispatch_indirect(u64 ctx, u64 buffer, u32 offset) {
    game_function<One>(kPrepareDispatch)(ctx);
    game_function<Two>(kSetIndirectBase)(cb_of(ctx), load<u64>(buffer + 0x98));
    return game_function<Two>(kGnmDispatchIndirect)(ctx + kGnm, offset);
}

namespace {
// The end of the flushes: YEBIS's command buffers submitted, the ring's next
// ones started from the default state, the draw prologue run.
u64 submit_and_restart(u64 ctx) {
    store<u64>(ctx + kDrained, load<u64>(ctx + kDraws));
    game_function<One>(kSubmit)(ctx + kGnm);
    game_function<One>(kNextBuffer)(ctx + kGnm);
    game_function<One>(kDefaultState)(cb_of(ctx));
    store<u8>(ctx + kStageSet, 0);
    if (!MUTANT(8)) store<u8>(ctx + kSizeChanged, 1);
    return game_function<One>(kPrepareDraw)(ctx);
}
// A label written at the end of the pipe with a value, and the wait for it
// (the flush count's label read again for the wait, as the game's does).
void end_of_pipe(u64 ctx, u64 label, u64 value) {
    game_function<Eight>(kEndOfPipe)(cb_of(ctx), kEopFlushCbDb, 0, label, 2, value, 0, 0);
}
void wait_label(u64 ctx, u64 label, u64 value) {
    game_function<Five>(kWaitRegMem)(cb_of(ctx), label, 0xffffffffu, 3, static_cast<u32>(value));
}
// The next flush count.
u64 next_flush(u64 ctx) {
    const u64 n = load<u64>(ctx + kFlushes);
    store<u64>(ctx + kFlushes, n + 1);
    return n;
}
}  // namespace

// Flush (slot 111): the flush count's label and the wait for it, then the
// submission and a fresh start. Returns what the draw prologue returns.
DECOMP_LEAF u64 yebis_context_flush(u64 ctx) {
    const u64 n = next_flush(ctx);
    end_of_pipe(ctx, load<u64>(ctx + kFlushLabel), n);
    wait_label(ctx, load<u64>(ctx + kFlushLabel), n);
    return submit_and_restart(ctx);
}

// Slot 131: Flush with the caller's label and value.
DECOMP_LEAF u64 yebis_context_flush_to(u64 ctx, u64 label, u64 value) {
    end_of_pipe(ctx, label, value);
    wait_label(ctx, label, value);
    return submit_and_restart(ctx);
}

// Slot 132: the flush count's label and the wait, without the submission.
// Returns the draw count.
DECOMP_LEAF u64 yebis_context_wait_flush(u64 ctx) {
    const u64 n = next_flush(ctx);
    end_of_pipe(ctx, load<u64>(ctx + kFlushLabel), n);
    wait_label(ctx, load<u64>(ctx + kFlushLabel), n);
    const u64 draws = load<u64>(ctx + kDraws);
    store<u64>(ctx + kDrained, draws);
    return draws;
}

namespace {
// A label in the command buffer, an end-of-shader event writing 1 into it,
// the wait for it, a cache action.
u64 wait_for_shaders(u64 cb, u32 event, u32 cache) {
    const u64 label = game_function<Three>(kEmbedData)(cb, 8, 3);
    store<u64>(label, 0);
    game_function<Four>(kEndOfShader)(cb, event, label, 1);
    game_function<Five>(kWaitRegMem)(cb, label, 0xffffffffu, 3, 1);
    return game_function<Four>(kAcquireMem)(cb, cache, 0, 1);
}
}  // namespace

// The waits (cb): for the compute shaders before a draw, for the pixel
// shaders before a dispatch, for the compute shaders before a dispatch.
DECOMP_LEAF u64 yebis_wait_compute(u64 cb) { return wait_for_shaders(cb, kEosCsDone, kFlushInvalidateL1L2); }
DECOMP_LEAF u64 yebis_wait_pixels(u64 cb) { return wait_for_shaders(cb, MUTANT(9) ? kEosCsDone : kEosPsDone, kFlushInvalidateL1L2); }
DECOMP_LEAF u64 yebis_wait_dispatch(u64 cb) { return wait_for_shaders(cb, kEosCsDone, kInvalidateL1); }

// OMSetRenderTargets's compare stand-in (BBHOST_DECOMP_COMPARE=1): the game's
// version runs, what it touched is kept, put back, ours runs and the two are
// compared, and the game's stays. A call that drains, or binds a view for the
// first time (which puts it on its owner's list), or finds too little room
// for the setters' packets is the game's alone.
namespace {
DecompCompare g_counts_omsrt;
void* g_original_omsrt = nullptr;
std::atomic<std::uint64_t> g_omsrt_alone{0};
std::atomic<int> g_compare_logs{0};

constexpr u64 kViewBytes = 0xe8, kTexBytes = 0xa8, kPacketBytes = 0x400;
// The context's four ranges, the command buffer's header and packets, 17
// views and their textures: under 0x2000 bytes.
constexpr std::size_t kTouchedBytes = 0x2000;
struct Range {
    u64 at, n;
};
// What a call touches: the context's fields, the views it binds or unbinds
// and their textures, the command buffer's header and the packets after its
// write pointer.
struct Touched {
    Range ranges[44];
    int n = 0;
    void add(u64 at, u64 bytes) {
        for (int k = 0; k < n; ++k)
            if (ranges[k].at == at) return;
        ranges[n++] = {at, bytes};
    }
    u64 bytes() const {
        u64 t = 0;
        for (int k = 0; k < n; ++k) t += ranges[k].n;
        return t;
    }
    void save(u8* out) const {
        for (int k = 0; k < n; ++k) {
            std::memcpy(out, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(ranges[k].at)), ranges[k].n);
            out += ranges[k].n;
        }
    }
    void put(const u8* in) const {
        for (int k = 0; k < n; ++k) {
            std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(ranges[k].at)), in, ranges[k].n);
            in += ranges[k].n;
        }
    }
};
bool would_drain(u64 ctx, u64 view) {
    const u64 tex = load<u64>(view + 0x80);
    return load<u64>(tex + 0xa0) < load<u64>(ctx + kDraws) && load<u64>(ctx + kDrained) < load<u64>(tex + 0xa0);
}

u64 run_guest4(void* fn, u64 a0, u64 a1, u64 a2, u64 a3) {
    GuestCallRegs g{};
    g.fn = fn;
    g.ints[0] = a0;
    g.ints[1] = a1;
    g.ints[2] = a2;
    g.ints[3] = a3;
    hle_call_guest_regs(&g);
    return g.ret;
}

GUEST_ABI u64 compare_om_set_render_targets(u64 ctx, u64 count_arg, u64 views, u64 depth) {
    const u32 count = static_cast<u32>(count_arg);
    const u64 cb = cb_of(ctx), wp = load<u64>(cb + 0x10);
    bool alone = !load<u64>(cb) || load<u64>(cb + 0x8) < wp + kPacketBytes;
    Touched t;
    t.add(ctx + kDraws, 0x18);
    t.add(ctx + kTargets, 0x60);
    t.add(ctx + kDirty, 4);
    t.add(ctx + kExport, 0x20);
    t.add(cb, 0x30);
    if (!alone) t.add(wp, kPacketBytes);
    const auto view_at = [&](u64 view) {
        if (!view) return;
        if (!load<u8>(view + 0x90) || would_drain(ctx, view)) alone = true;
        t.add(view, kViewBytes);
        t.add(load<u64>(view + 0x80), kTexBytes);
    };
    view_at(depth);
    for (u32 i = 0; i < 8 && !alone; ++i) {
        if (i < count) view_at(load<u64>(views + 8ull * i));
        if (const u64 old = load<u64>(ctx + kTargets + 8ull * i)) {
            t.add(old, kViewBytes);
            t.add(load<u64>(old + 0x80), kTexBytes);
        }
    }
    if (alone) {
        g_omsrt_alone.fetch_add(1, std::memory_order_relaxed);
        return run_guest4(g_original_omsrt, ctx, count, views, depth);
    }
    const u64 n = t.bytes();
    // Hosted (a compare stand-in runs with the host's FS), so the buffers can be the thread's own.
    static thread_local u8 before[kTouchedBytes], game[kTouchedBytes], ours[kTouchedBytes];
    if (n > kTouchedBytes) {
        g_omsrt_alone.fetch_add(1, std::memory_order_relaxed);
        return run_guest4(g_original_omsrt, ctx, count, views, depth);
    }
    t.save(before);
    const u64 game_ret = run_guest4(g_original_omsrt, ctx, count, views, depth);
    t.save(game);
    t.put(before);
    const u64 our_ret = run_guest4(reinterpret_cast<void*>(&yebis_context_om_set_render_targets), ctx, count, views, depth);
    t.save(ours);
    g_counts_omsrt.calls.fetch_add(1, std::memory_order_relaxed);
    if (our_ret != game_ret || std::memcmp(ours, game, n) != 0) {
        g_counts_omsrt.differ.fetch_add(1, std::memory_order_relaxed);
        if (g_compare_logs.fetch_add(1) < 16) {
            u64 at = 0;
            while (at < n && ours[at] == game[at]) ++at;
            host_log("decomp: yebis_context_om_set_render_targets differs from the game's: %u targets, depth 0x%llx, first byte %llu of %llu, "
                     "returned 0x%llx/0x%llx",
                     count, static_cast<ull>(depth), static_cast<ull>(at), static_cast<ull>(n), static_cast<ull>(our_ret),
                     static_cast<ull>(game_ret));
        }
    }
    t.put(game);
    return game_ret;
}

void report() {
    const u64 alone = g_omsrt_alone.load(std::memory_order_relaxed);
    if (alone) host_log("decomp: yebis_context_om_set_render_targets compare: %llu calls left to the game's alone (a drain, a first bind or no room)",
                        static_cast<ull>(alone));
}

// The entries the decomp list places them at.
#define PUSH6 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53
// sub rsp, 0x58
constexpr u8 kEntryOmSetRenderTargets[] = {PUSH6, 0x48, 0x83, 0xec, 0x58};
// push rbx; push rax; mov rbx, rdi
constexpr u8 kEntryPrepareDraw[] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb};
// push r14; push rbx; mov rbx, rdi; lea r14, [rbx + 0xa98]
constexpr u8 kEntryPrepareDispatch[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89, 0xfb, 0x4c, 0x8d, 0xb3, 0x98, 0x0a, 0x00, 0x00};
// push r15, r14, r12, rbx; mov r14d, ecx
constexpr u8 kEntryDrawIndexed[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xce};
// push r14, rbx; mov r14d, esi; mov rbx, rdi
constexpr u8 kEntryDraw[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb};
// push rax; mov r14d, r9d
constexpr u8 kEntryDrawIndexedInstanced[] = {PUSH6, 0x50, 0x45, 0x89, 0xce};
// push rax; mov r15d, r8d
constexpr u8 kEntryDrawInstanced[] = {PUSH6, 0x50, 0x45, 0x89, 0xc7};
// push r15, r14, r12, rbx; mov r14d, ecx
constexpr u8 kEntryDispatch[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xce};
// push r15, r14, rbx, rax; mov r14d, edx
constexpr u8 kEntryDispatchIndirect[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x41, 0x89, 0xd6};
// push r15, r14, rbx; sub rsp, 0x18
constexpr u8 kEntryFlush[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x48, 0x83, 0xec, 0x18};
// push r15, r14, r12, rbx; sub rsp, 0x10
constexpr u8 kEntryFlushTo[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x10};
// push r14, rbx; sub rsp, 0x10
constexpr u8 kEntryWaitFlush[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x83, 0xec, 0x10};
// push r14, rbx; mov r14, rdi (the three waits alike)
constexpr u8 kEntryWait[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x49, 0x89, 0xfe};
#undef PUSH6

}  // namespace

void decomp_yebis_context_add() {
    DecompFunction omsrt{"yebis_context_om_set_render_targets", "yebis", 0x15d74c0, kEntryOmSetRenderTargets, sizeof kEntryOmSetRenderTargets,
                         reinterpret_cast<void*>(&yebis_context_om_set_render_targets), DecompKind::Leaf};
    omsrt.original = &g_original_omsrt;
    omsrt.compare = reinterpret_cast<void*>(&compare_om_set_render_targets);
    omsrt.counts = &g_counts_omsrt;
    omsrt.report = &report;
    decomp_add(omsrt);
    const DecompFunction fns[] = {
        {"yebis_context_prepare_draw", "yebis", kPrepareDraw, kEntryPrepareDraw, sizeof kEntryPrepareDraw,
         reinterpret_cast<void*>(&yebis_context_prepare_draw), DecompKind::Leaf},
        {"yebis_context_prepare_dispatch", "yebis", kPrepareDispatch, kEntryPrepareDispatch, sizeof kEntryPrepareDispatch,
         reinterpret_cast<void*>(&yebis_context_prepare_dispatch), DecompKind::Leaf},
        {"yebis_context_draw_indexed", "yebis", 0x2e40880, kEntryDrawIndexed, sizeof kEntryDrawIndexed,
         reinterpret_cast<void*>(&yebis_context_draw_indexed), DecompKind::Leaf},
        {"yebis_context_draw", "yebis", 0x2e408e0, kEntryDraw, sizeof kEntryDraw, reinterpret_cast<void*>(&yebis_context_draw), DecompKind::Leaf},
        {"yebis_context_draw_indexed_instanced", "yebis", 0x2e40aa0, kEntryDrawIndexedInstanced, sizeof kEntryDrawIndexedInstanced,
         reinterpret_cast<void*>(&yebis_context_draw_indexed_instanced), DecompKind::Leaf},
        {"yebis_context_draw_instanced", "yebis", 0x2e40b40, kEntryDrawInstanced, sizeof kEntryDrawInstanced,
         reinterpret_cast<void*>(&yebis_context_draw_instanced), DecompKind::Leaf},
        {"yebis_context_dispatch", "yebis", 0x2e410b0, kEntryDispatch, sizeof kEntryDispatch, reinterpret_cast<void*>(&yebis_context_dispatch),
         DecompKind::Leaf},
        {"yebis_context_dispatch_indirect", "yebis", 0x2e410f0, kEntryDispatchIndirect, sizeof kEntryDispatchIndirect,
         reinterpret_cast<void*>(&yebis_context_dispatch_indirect), DecompKind::Leaf},
        {"yebis_context_flush", "yebis", 0x15d99e0, kEntryFlush, sizeof kEntryFlush, reinterpret_cast<void*>(&yebis_context_flush),
         DecompKind::Leaf},
        {"yebis_context_flush_to", "yebis", 0x15d9ab0, kEntryFlushTo, sizeof kEntryFlushTo, reinterpret_cast<void*>(&yebis_context_flush_to),
         DecompKind::Leaf},
        {"yebis_context_wait_flush", "yebis", 0x15d9b70, kEntryWaitFlush, sizeof kEntryWaitFlush,
         reinterpret_cast<void*>(&yebis_context_wait_flush), DecompKind::Leaf},
        {"yebis_wait_compute", "yebis", kWaitCompute, kEntryWait, sizeof kEntryWait, reinterpret_cast<void*>(&yebis_wait_compute),
         DecompKind::Leaf},
        {"yebis_wait_pixels", "yebis", kWaitPixels, kEntryWait, sizeof kEntryWait, reinterpret_cast<void*>(&yebis_wait_pixels), DecompKind::Leaf},
        {"yebis_wait_dispatch", "yebis", kWaitDispatch, kEntryWait, sizeof kEntryWait, reinterpret_cast<void*>(&yebis_wait_dispatch),
         DecompKind::Leaf},
    };
    for (const DecompFunction& fn : fns) decomp_add(fn);
}
