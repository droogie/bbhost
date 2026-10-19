// Gnmx's constant update engine around a draw, as source (docs/decomp.md):
// the pre-draw flush (0x1492e80) and the post-draw (0x1493150) of the Gnmx
// graphics context the flip pass (0x25d5f80) draws its full-screen pass
// with, once a frame, before it submits that context's command buffer
// itself (two contexts at 0x5aa6d28, alternating; the binary's own Gnmx
// messages name lwgfxcontext.cpp). The engine (+0x2c8 of the context) keeps
// each shader stage's resources in constant RAM and dumps the changed ones
// out before a draw; around the draw:
//
//   - the pre-draw: the constant command buffer waits for the draw engine
//     (WAIT_ON_DE_COUNTER_DIFF, the game's packet in every mode: the NOP
//     below goes over it, decomp/gx/constant_engine.cpp); the pixel
//     shader's input table written when the pixel or vertex shader changed
//     (the table the vertex shader's outputs and the pixel shader's inputs
//     make, or one given);
//     for each stage in use - Gnm's stages 1 and 2 (pixel, vertex), 3 and 4
//     (geometry, export) under the engine's flag 0x20, 5 and 6 (hull, local)
//     under flag 1 - the shader's prefetch and its dirty resources
//     applied (0x1491f10); a global table written inline when it changed;
//     the constant engine's counter stepped and the draw engine's wait for
//     it when the stages used it, and otherwise its wait undone (a NOP over
//     it);
//   - the post-draw: the draw engine's counter stepped when the constant
//     engine was used.
//
// The flip pass draws once a frame; this is the Gnmx side of the frame's
// end (decomp/gx/present.cpp has its pacing). Leaves: the game's calls in the
// game's order; its constant- and draw-engine writers are on the list.
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <cstdint>
#include <cstring>

using namespace decomp;

#ifdef DECOMP_GNMX_CUE_MUTANTS
extern int g_cue_mutant;
#define MUTANT(n) (g_cue_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

// The constant update engine.
constexpr u64 kDcb = 0x10, kCcb = 0x18, kFlags = 0x28, kDiff = 0x8;
constexpr u64 kStages = 0x450, kStageBytes = 0x438;  // a stage's prefetch address (+0, in 256-byte blocks) and size (+4), its dirty count (+8, u16)
constexpr u64 kVsShader = 0xe348, kPsShader = 0xe350, kPsInputs = 0xe3c8;
constexpr u64 kCeUsed = 0xe3d8, kCeCounted = 0xe3d9, kStagesDirty = 0xe3de, kShadersDirty = 0xe3df;
constexpr u64 kGlobalAddress = 0xe3e8, kGlobalTable = 0xe3f0, kGlobalDirty = 0xe4b0, kPrefetchOn = 0xe4b1;

constexpr u64 kCeWait = 0x15fd230;        // gnm_ce_wait_on_de_counter_diff(ccb, difference)
constexpr u64 kBuildPsInputs = 0x14873a0; // the pixel shader's input table (out, vs outputs, count, ps inputs, count)
constexpr u64 kSetPsInputs = 0x1475000;   // SPI_PS_INPUT_CNTL_* (dcb, table, count)
constexpr u64 kPrefetch = 0x1473f50;      // gnm_draw_shader_prefetch_dma(dcb, address, bytes)
constexpr u64 kApplyStage = 0x1491f10;    // the stage's dirty resources (cue, stage)
constexpr u64 kWriteData = 0x1473ce0;     // gnm_write_data_inline(dcb, dst, src, dwords, dst select, confirm)
constexpr u64 kAcquireMem = 0x1477f40;
constexpr u64 kCeIncrement = 0x15fd190;   // gnm_ce_increment_ce_counter(ccb)
constexpr u64 kDeWaitOnCe = 0x14781d0;    // gnm_wait_on_ce_counter(dcb)
constexpr u64 kNop = 0x15fd4a0;           // a NOP of n dwords (cb, n)
constexpr u64 kDeIncrement = 0x1478220;   // gnm_increment_de_counter(dcb)

using One = GUEST_ABI u64 (*)(u64);
using Two = GUEST_ABI u64 (*)(u64, u64);
using Three = GUEST_ABI u64 (*)(u64, u64, u64);
using Four = GUEST_ABI u64 (*)(u64, u64, u64, u64);
using Five = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64);
using Six = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64, u64);

}  // namespace

// gnmx_cue_pre_draw(cue).
DECOMP_LEAF void gnmx_cue_pre_draw(u64 cue) {
    const u32 flags = load<u32>(cue + kFlags);
    store<u8>(cue + kCeCounted, 0);
    game_function<Two>(kCeWait)(load<u64>(cue + kCcb), load<u32>(cue + kDiff) >> 2);
    // The constant command buffer's header as the wait left it: its NOP goes there if the constant engine stays unused.
    u64 ccb[8];
    std::memcpy(ccb, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(load<u64>(cue + kCcb))), sizeof ccb);
    const u64 ps = load<u64>(cue + kPsShader);
    if (ps && (load<u8>(cue + kShadersDirty) & 6)) {
        const u32 inputs = load<u8>(ps + 0x38);
        if (inputs) {
            u64 table = load<u64>(cue + kPsInputs);
            u32 count = inputs;
            alignas(16) u8 built[0x80];  // SPI_PS_INPUT_CNTL_0..31
            if (!table) {
                const u64 vs = load<u64>(cue + kVsShader);
                game_function<Five>(kBuildPsInputs)(reinterpret_cast<u64>(built),
                                                    vs + (static_cast<u64>(load<u8>(vs + 0x24)) + load<u8>(vs + 3)) * 4 + 0x28, load<u8>(vs + 0x25),
                                                    ps + static_cast<u64>(load<u8>(ps + 3)) * 4 + 0x3c, inputs);
                table = reinterpret_cast<u64>(built);
                count = load<u8>(load<u64>(cue + kPsShader) + 0x38);
            }
            game_function<Three>(kSetPsInputs)(load<u64>(cue + kDcb), table, count);
        }
    }
    // The stages in use (Gnm's numbering): 1 and 2, 3 and 4 (geometry, export) under flag 0x20, 5 and 6 (hull, local) under flag 1.
    const bool geometry = flags & 0x20, tessellation = flags & 1;
    u32 bit = 2;
    for (u32 stage = 1; stage < 7;) {
        const u64 at = cue + kStages + static_cast<u64>(stage) * kStageBytes;
        if (load<u8>(cue + kPrefetchOn) && (load<u8>(cue + kShadersDirty) & bit))
            game_function<Three>(kPrefetch)(load<u64>(cue + kDcb), static_cast<u64>(load<u32>(at)) << 8, load<u32>(at + 4));
        u8 dirty = load<u8>(cue + kStagesDirty);
        if ((dirty & bit) && load<u16>(at + 8)) {
            game_function<Two>(kApplyStage)(cue, stage);
            dirty = load<u8>(cue + kStagesDirty);
        }
        store<u8>(cue + kShadersDirty, static_cast<u8>(load<u8>(cue + kShadersDirty) & ~bit));
        store<u8>(cue + kStagesDirty, static_cast<u8>(dirty & ~bit));
        ++stage;
        bit <<= 1;
        if (stage == 3 && !(MUTANT(1) || geometry)) {
            stage = 5;
            bit = 0x20;
        }
        if (stage == 5 && !tessellation) break;
    }
    if (load<u8>(cue + kGlobalDirty)) {
        game_function<Six>(kWriteData)(load<u64>(cue + kDcb), load<u64>(cue + kGlobalAddress), cue + kGlobalTable, 0x30, 0, 1);
        game_function<Four>(kAcquireMem)(load<u64>(cue + kDcb), 0, 0x8000000, 1);
        store<u8>(cue + kGlobalDirty, 0);
    }
    if (load<u8>(cue + kCeCounted)) {
        game_function<One>(kCeIncrement)(load<u64>(cue + kCcb));
        game_function<One>(kDeWaitOnCe)(load<u64>(cue + kDcb));
    }
    if (!load<u8>(cue + kCeUsed) && !MUTANT(4)) {
        ccb[2] -= MUTANT(2) ? 4 : 8;
        game_function<Two>(kNop)(reinterpret_cast<u64>(ccb), 2);
    }
    store<u8>(cue + kCeUsed, 0);
}

// gnmx_cue_post_draw(cue).
DECOMP_LEAF void gnmx_cue_post_draw(u64 cue) {
    if (!load<u8>(cue + kCeCounted)) return;
    game_function<One>(kDeIncrement)(load<u64>(cue + kDcb));
    store<u16>(cue + kCeCounted, MUTANT(3) ? 0 : 0x100);
}

namespace {
// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; sub rsp, 0xc8
constexpr u8 kEntryPreDraw[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xc8, 0x00, 0x00, 0x00};
// push rbp; mov rbp, rsp; push rbx; push rax; mov rbx, rdi
constexpr u8 kEntryPostDraw[] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb};
}  // namespace

void decomp_gnmx_cue_add() {
    const DecompFunction fns[] = {
        {"gnmx_cue_pre_draw", "gnmx", 0x1492e80, kEntryPreDraw, sizeof kEntryPreDraw, reinterpret_cast<void*>(&gnmx_cue_pre_draw), DecompKind::Leaf},
        {"gnmx_cue_post_draw", "gnmx", 0x1493150, kEntryPostDraw, sizeof kEntryPostDraw, reinterpret_cast<void*>(&gnmx_cue_post_draw),
         DecompKind::Leaf},
    };
    for (const DecompFunction& fn : fns) decomp_add(fn);
}
