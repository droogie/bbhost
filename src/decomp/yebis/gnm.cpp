// YEBIS's Gnm backend as source (docs/decomp.md): the three wrappers every
// draw and dispatch of YEBIS - the post-process library under the engine's
// PPFX layer (tone mapping, glare, depth of field, motion blur, colour
// grading) - goes through on its way to Gnm. yebis_gnm_draw_index (0x15fbc40)
// and yebis_gnm_draw_auto (0x15fbb50) are the indexed and the auto-indexed
// draw: the prep (0x15f9f30: the draw's last two arguments into two user-data
// slots), the commit (0x15f9720: the constant engine's ring opened, every
// stage's shader, resources and user data), Gnm's draw packet, the post-draw
// (0x15f9d10); yebis_gnm_dispatch (0x15fbd10) the compute pass: its commit
// (0x15f9d20), DISPATCH_DIRECT, its post (0x15f9f20). The YEBIS context they
// take keeps its command buffer at +0x8, the constant command buffer at +0x10
// and its state at +0x28.
//
// The GX trace hooked the indexed draw and the dispatch (YEBIS's draws become
// tokens from YEBIS's own state; the dispatches with BBHOST_GX_YEBIS_DISPATCH):
// ours hand the call to it first, as the hooks did, and a call it takes
// whole runs nothing more. The trace knows these wrappers' packets by the
// writer call's return address, so the three are writer callers
// (decomp/gx/draw.h). Leaves, each a call sequence to the game's addresses.
// That is the place each of YEBIS's passes is seen as a command with its
// context - the per-command plugin surface of docs/modding.md for
// post-processing.
#include "decomp/decomp.h"
#include "decomp/guest.h"
#include "decomp/gx/draw.h"

#include <cstdint>

using namespace decomp;

#ifdef DECOMP_YEBIS_GNM_MUTANTS
extern int g_yebis_mutant;
#define MUTANT(n) (g_yebis_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kPrepDraw = 0x15f9f30;        // yebis_gnm_prep_draw(state, cb, a, b)
constexpr u64 kCommitDraw = 0x15f9720;      // yebis_gnm_commit_draw(state, cb, ccb)
constexpr u64 kPostDraw = 0x15f9d10;        // yebis_gnm_post_draw(state, cb, ccb)
constexpr u64 kCommitDispatch = 0x15f9d20;  // yebis_gnm_commit_dispatch(state, cb, ccb)
constexpr u64 kPostDispatch = 0x15f9f20;    // yebis_gnm_post_dispatch(state, cb, ccb)
constexpr u64 kDrawIndex = 0x1474360;       // gnm_draw_index(cb, index count, index address)
constexpr u64 kDrawIndexAuto = 0x14741b0;   // gnm_draw_index_auto(cb, vertex count)
constexpr u64 kDispatchDirect = 0x1474c50;  // gnm_dispatch_direct(cb, x, y, z)
// The game's return addresses from each wrapper's draw or dispatch writer.
constexpr u64 kDrawIndexReturn = 0x15fbc85, kDrawAutoReturn = 0x15fbb7a, kDispatchReturn = 0x15fbd4a;

using Three = GUEST_ABI u64 (*)(u64, u64, u64);
using Four = GUEST_ABI u64 (*)(u64, u64, u64, u64);

u64 state(u64 y) { return load<u64>(y + 0x28); }
u64 cb(u64 y) { return load<u64>(y + 0x8); }
u64 ccb(u64 y) { return load<u64>(y + 0x10); }

}  // namespace

#define RET reinterpret_cast<u64>(__builtin_return_address(0))

// yebis_gnm_draw_index(yebis ctx, index count, index address, a, b): a and b
// go to the prep (the draw's two user-data values). Returns what the post-draw
// returns, as the game's tail call does.
DECOMP_LEAF DECOMP_GX_WRITER_CALLER u64 yebis_gnm_draw_index(u64 y, u64 count, u64 indices, u64 a, u64 b) {
    if (decomp_gx::hand_to_trace(decomp_gx::g_yebis_draw_kind, y, count, indices, a, b, 0, RET)) return 0;
    game_function<Four>(kPrepDraw)(state(y), cb(y), static_cast<u32>(a), static_cast<u32>(MUTANT(1) ? a : b));
    game_function<Three>(kCommitDraw)(state(y), cb(y), ccb(y));
    game_function<Three>(kDrawIndex)(cb(y), static_cast<u32>(count), indices);
    return game_function<Three>(kPostDraw)(state(y), cb(y), ccb(y));
}

// yebis_gnm_draw_auto(yebis ctx, vertex count).
DECOMP_LEAF DECOMP_GX_WRITER_CALLER u64 yebis_gnm_draw_auto(u64 y, u64 count) {
    game_function<Three>(kCommitDraw)(state(y), cb(y), ccb(y));
    using Two = GUEST_ABI u64 (*)(u64, u64);
    game_function<Two>(kDrawIndexAuto)(cb(y), static_cast<u32>(count));
    return game_function<Three>(MUTANT(2) ? kPostDispatch : kPostDraw)(state(y), cb(y), ccb(y));
}

// yebis_gnm_dispatch(yebis ctx, x, y, z).
DECOMP_LEAF DECOMP_GX_WRITER_CALLER u64 yebis_gnm_dispatch(u64 y, u64 x, u64 groups_y, u64 z) {
    if (decomp_gx::g_yebis_dispatch_hooked && decomp_gx::hand_to_trace(decomp_gx::g_yebis_dispatch_kind, y, x, groups_y, z, 0, 0, RET)) return 0;
    game_function<Three>(kCommitDispatch)(state(y), cb(y), ccb(y));
    game_function<Four>(kDispatchDirect)(cb(y), static_cast<u32>(x), static_cast<u32>(MUTANT(3) ? z : groups_y), static_cast<u32>(z));
    return game_function<Three>(kPostDispatch)(state(y), cb(y), ccb(y));
}

#undef RET

namespace {
// The entries the decomp list places them at.
// push rbp; mov rbp, rsp; push r15; push r14; push rbx; push rax; mov r14, rdx; mov r15d, esi
constexpr u8 kEntryDrawIndex[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xd6, 0x41, 0x89, 0xf7};
// push rbp; mov rbp, rsp; push r14; push rbx; mov r14d, esi; mov rbx, rdi
constexpr u8 kEntryDrawAuto[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb};
// push rbp; mov rbp, rsp; push r15; push r14; push r12; push rbx; mov r14d, ecx
constexpr u8 kEntryDispatch[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xce};
}  // namespace

void decomp_yebis_gnm_add() {
    const DecompFunction fns[] = {
        {"yebis_gnm_draw_index", "yebis", 0x15fbc40, kEntryDrawIndex, sizeof kEntryDrawIndex, reinterpret_cast<void*>(&yebis_gnm_draw_index),
         DecompKind::Leaf},
        {"yebis_gnm_draw_auto", "yebis", 0x15fbb50, kEntryDrawAuto, sizeof kEntryDrawAuto, reinterpret_cast<void*>(&yebis_gnm_draw_auto),
         DecompKind::Leaf},
        {"yebis_gnm_dispatch", "yebis", 0x15fbd10, kEntryDispatch, sizeof kEntryDispatch, reinterpret_cast<void*>(&yebis_gnm_dispatch),
         DecompKind::Leaf},
    };
    for (const DecompFunction& fn : fns) decomp_add(fn);
    decomp_gx::note_writer_caller(reinterpret_cast<const void*>(&yebis_gnm_draw_index), kDrawIndexReturn);
    decomp_gx::note_writer_caller(reinterpret_cast<const void*>(&yebis_gnm_draw_auto), kDrawAutoReturn);
    decomp_gx::note_writer_caller(reinterpret_cast<const void*>(&yebis_gnm_dispatch), kDispatchReturn);
}
