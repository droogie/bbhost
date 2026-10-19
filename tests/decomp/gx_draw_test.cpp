// GX's immediate methods, the output-merger flush, the post-draw and the
// command buffer's flush (decomp/gx/draw.cpp) beside the game's own
// (tests/decomp/eboot_kit.h).
//
// The methods are call sequences: every callee is replaced by a recorder
// (eboot_kit::stub - the game's method and ours both reach it), and on
// generated contexts, states and arguments - garbage above a 32-bit
// argument's bits, every stage configuration an indirect draw reads - ours
// must make the game's calls, with the game's arguments, in the game's order.
// With the GX trace's entry set, a method must hand the trace the frame the
// prologue stubs handed it and run nothing more when it takes the call, and
// the whole sequence when it does not. The writer that writes a method's
// draw or dispatch is called from the game's method at the return address
// the table in draw.cpp gives, and from ours at one decomp_gx::game_return
// takes for it.
//
// The output-merger flush on generated blocks and states: colour targets
// recorded in either set, bound or replaced, with a clear pending or not, of
// every clear kind; the depth target likewise; a blend and a depth-stencil
// state, the default ones or the one last written, with some fields equal to
// what the block holds; blend constants that are equal, -0 against +0, NaNs;
// depth bounds and stencil references equal or not - ours must make the
// game's calls (the floats' bits) and leave the block as the game's does.
//
// The post-draw and the flush on generated memory: dirty bits, the
// multi-VGT parameter's mark, a geometry shader or none; release lists on
// any of sixteen slots and pending, arena queues of up to three pages, an
// owner or none, the compute ring's counter on its last value or not, room
// for a fresh start or not, a context whose refill is the arena's or
// another's - ours must leave the memory as the game's does, return what it
// returns and make its calls with its arguments (what a stack pointer points
// at, for the two that take one).
//
// And each of a handful of deliberate mistakes (MUTANT in draw.cpp) must be
// caught, and the entries the decomp list places them at hold the
// instructions written against. Skips without the 1.09 eboot.
#define DECOMP_GX_DRAW_MUTANTS
int g_draw_mutant = 0;
#include "decomp/gx/draw.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <utility>
#include <vector>

#include <sys/mman.h>

namespace {

eboot_kit::Rng g_rng(0xd4a3);

// ---- Recorders for every callee ----
struct Call {
    int who;
    u64 a[6];
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_rec;
u64 g_writer_ret[2] = {};

struct Callee {
    u64 bn;
    int args;          // how many to compare
    u32 narrow;        // a bit per argument that is 32-bit
    const char* name;
};
// The order is the recorders' ids.
const Callee kCallees[] = {
    {kFlushOutputMerger, 3, 0, "flush_output_merger"},
    {kFlushInputAssembly, 4, 0x8, "flush_input_assembly"},
    {kFlushRasterizer, 3, 0, "flush_rasterizer"},
    {kCommandBufferFlush, 1, 0, "command_buffer_flush"},
    {kCommitState, 3, 0, "commit_state"},
    {kPostDraw, 2, 0, "post_draw"},
    {kDispatchCommit, 3, 0, "dispatch_commit"},
    {kDispatchPost, 3, 0, "dispatch_post"},
    {kSetIndexOffset, 2, 0x2, "set_index_offset"},
    {kSetNumInstances, 2, 0x2, "set_num_instances"},
    {kDrawIndexAuto, 2, 0x2, "draw_index_auto"},
    {kDrawIndexOffset, 3, 0x6, "draw_index_offset"},
    {kSetBaseIndirectArgs, 2, 0, "set_base_indirect_args"},
    {kDrawIndirect, 5, 0x1e, "draw_indirect"},
    {kDrawIndexIndirect, 5, 0x1e, "draw_index_indirect"},
    {kDispatchDirect, 4, 0xe, "dispatch_direct"},
    {kDispatchIndirect, 2, 0x2, "dispatch_indirect"},
    // The post-draw's writers.
    {kSetIaMultiVgtParam, 3, 0x6, "set_ia_multi_vgt_param"},
    {kSetGsMode, 3, 0x6, "set_gs_mode"},
    // The flush's.
    {kReleaseNodePush, 3, 0, "release_node_push"},
    {kFlushUnknown2acd3f0, 2, 0, "sub_2acd3f0"},
    {kRetireChunk, 3, 0, "retire_chunk"},
    {kCloseSegment, 4, 0x8, "close_segment"},
    // The output-merger flush's.
    {kWriteClearColors, 2, 0, "write_clear_colors"},
    {kFastClearEliminate, 6, 0, "fast_clear_eliminate_pass"},
    {kDepthDecompress, 6, 0, "depth_decompress_pass"},
    {kSetAlphaToMask, 2, 0, "set_alpha_to_mask"},
    {kSetRenderTargetMask, 2, 0, "set_render_target_mask"},
    {kSetBlendControl, 3, 0, "set_blend_control"},
    {kSetBlendColor, 5, 0, "set_blend_color"},  // the floats' bits
    {kSetDepthStencilControl, 2, 0, "set_depth_stencil_control"},
    {kSetStencilControl, 2, 0, "set_stencil_control"},
    {kSetStencilRef, 2, 0, "set_stencil_ref"},
    {kSetDepthBoundsRange, 3, 0, "set_depth_bounds_range"},  // the floats' bits
};
constexpr int kNumCallees = static_cast<int>(std::size(kCallees));
constexpr int kPush = 19, kClose = 22, kWriterIa = 17, kWriterGs = 18;
constexpr int kBlendColorId = 29, kDepthBoundsId = 33;
constexpr int kFirstEmitter = 10, kLastEmitter = 16;  // the draw and dispatch writers (not 12, the base)

void record(int who, u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    Call c{who, {a0, a1, a2, a3, a4, a5}};
    const Callee& e = kCallees[who];
    for (int k = 0; k < 6; ++k) {
        if (k >= e.args) c.a[k] = 0;
        else if ((e.narrow >> k) & 1) c.a[k] &= 0xffffffffull;
    }
    // What a stack pointer points at, not where: the pool's push gets where
    // the command buffer is; the segment's close gets the header's copy.
    if (who == kPush) c.a[1] = load<u64>(c.a[1]);
    if (who == kClose) {
        const u64 h = c.a[2];
        u64 sum = 0;
        for (int k = 0; k < 8; ++k) sum = sum * 0x100000001b3ull ^ load<u64>(h + 8ull * k);
        c.a[2] = sum;
    }
    g_rec.push_back(c);
}

constexpr int kRetire = 21;
u64 g_emit_ret = 0;  // where the last draw or dispatch writer was called from
template <int W>
GUEST_ABI u64 recorder(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    record(W, a0, a1, a2, a3, a4, a5);
    if (W >= kFirstEmitter && W <= kLastEmitter && W != 12) g_emit_ret = reinterpret_cast<u64>(__builtin_return_address(0));
    // A chunk's retirement writes its label into the command buffer: the
    // write pointer moves, which the flush notes after each.
    if (W == kRetire) store<u64>(a1 + 0x10, load<u64>(a1 + 0x10) + 8);
    if (W == kWriterIa) return g_writer_ret[0];
    if (W == kWriterGs) return g_writer_ret[1];
    return 0;
}
u64 float_bits(float f) {
    u32 b;
    std::memcpy(&b, &f, 4);
    return b;
}
// The two writers that take floats (xmm0..3).
template <int W>
GUEST_ABI u64 float_recorder(u64 cb, float a, float b, float c, float d) {
    record(W, cb, float_bits(a), float_bits(b), float_bits(c), float_bits(d), 0);
    return 0;
}
template <int W>
const void* recorder_for() {
    if constexpr (W == kBlendColorId || W == kDepthBoundsId) return reinterpret_cast<const void*>(&float_recorder<W>);
    else return reinterpret_cast<const void*>(&recorder<W>);
}
// The callees the methods share with what is tested on its own (the
// output-merger flush, the post-draw, the flush) are stubbed for the methods
// and the game's again for those: their first bytes kept, put back.
constexpr int kOmId = 0, kPostDrawId = 5, kFlushId = 3;
constexpr int kShared[3] = {kOmId, kPostDrawId, kFlushId};
u8 g_original_bytes[3][16];
template <int... W>
void stub_all(std::integer_sequence<int, W...>) {
    for (int k = 0; k < 3; ++k) std::memcpy(g_original_bytes[k], eboot_kit::at(kCallees[kShared[k]].bn), 16);
    ((W == kOmId || W == kPostDrawId || W == kFlushId ? void() : eboot_kit::stub(kCallees[W].bn, recorder_for<W>())), ...);
}
void put_bytes(u64 bn, const u8* bytes) {
    const u64 page = bn & ~0xfffull, end = (bn + 16 + 0xfff) & ~0xfffull;
    mprotect(reinterpret_cast<void*>(page), end - page, PROT_READ | PROT_WRITE | PROT_EXEC);
    std::memcpy(eboot_kit::at(bn), bytes, 16);
    mprotect(reinterpret_cast<void*>(page), end - page, PROT_READ | PROT_EXEC);
}
// true: the methods' recorders over the output-merger flush, the post-draw
// and the flush; false: the game's.
void shared_stubbed(bool on) {
    if (on) {
        eboot_kit::stub(kCallees[kOmId].bn, reinterpret_cast<const void*>(&recorder<kOmId>));
        eboot_kit::stub(kCallees[kPostDrawId].bn, reinterpret_cast<const void*>(&recorder<kPostDrawId>));
        eboot_kit::stub(kCallees[kFlushId].bn, reinterpret_cast<const void*>(&recorder<kFlushId>));
    } else {
        for (int k = 0; k < 3; ++k) put_bytes(kCallees[kShared[k]].bn, g_original_bytes[k]);
    }
}

// The flush's other callees, reached through pointers: the refill, the owner's +0x70.
struct Extra {
    int who;  // 100 refill, 101 owner
    u64 a[4];
    bool operator==(const Extra&) const = default;
};
std::vector<Extra> g_extra;
GUEST_ABI u64 refill_stub(u64 cb, u64 dwords, u64 arg, u64 left) {
    g_extra.push_back({100, {cb, dwords & 0xffffffffull, arg, left}});
    return 0;
}
GUEST_ABI u64 owner_stub(u64 self, u64 page) {
    g_extra.push_back({101, {self, page, 0, 0}});
    return 0;
}

int g_failed = 0;
bool g_quiet = false;  // a mutant's run: its failures are counted, not printed
void fail(const char* what, int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: %s, case %d: %s\n", what, trial, why);
}

u64 garbage_u32() {
    const u32 v = g_rng.chance(0.3) ? static_cast<u32>(g_rng.range(0, 8)) : g_rng.u32();
    return (static_cast<u64>(g_rng.u32()) << 32) | v;  // the bits above a 32-bit argument, as a register may carry them
}

// ---- The methods ----
alignas(64) u8 g_ctx[0xc000];
alignas(64) u8 g_buffer[0x40];

enum Method { kMDraw, kMDrawIndexed, kMDrawInstanced, kMDrawIndexedInstanced, kMIndirect, kMIndexedIndirect, kMCommit, kMDispatch, kMDispatchIndirect };
const u64 kMethodBn[9] = {0x25696c0, 0x2569780, 0x2569840, 0x2569900, 0x25699d0, 0x2569ab0, 0x2569ba0, 0x25695c0, 0x2569640};

void call_method(int m, bool game, const u64* a) {
    const u64 ctx = reinterpret_cast<u64>(g_ctx);
    using F2 = GUEST_ABI void (*)(u64, u64);
    using F4 = GUEST_ABI void (*)(u64, u64, u64, u64);
    using F5 = GUEST_ABI void (*)(u64, u64, u64, u64, u64);
    using F6 = GUEST_ABI void (*)(u64, u64, u64, u64, u64, u64);
    const u64 bn = kMethodBn[m];
    switch (m) {
        case kMDraw: game ? eboot_kit::fn<F4>(bn)(ctx, a[0], a[1], a[2]) : gx_immediate_draw(ctx, a[0], a[1], a[2]); break;
        case kMDrawIndexed:
            game ? eboot_kit::fn<F5>(bn)(ctx, a[0], a[1], a[2], a[3]) : gx_immediate_draw_indexed(ctx, a[0], a[1], a[2], a[3]);
            break;
        case kMDrawInstanced:
            game ? eboot_kit::fn<F6>(bn)(ctx, a[0], a[1], a[2], a[3], a[4]) : gx_immediate_draw_instanced(ctx, a[0], a[1], a[2], a[3], a[4]);
            break;
        case kMDrawIndexedInstanced:
            game ? eboot_kit::fn<F6>(bn)(ctx, a[0], a[1], a[2], a[3], a[4])
                 : gx_immediate_draw_indexed_instanced(ctx, a[0], a[1], a[2], a[3], a[4]);
            break;
        case kMIndirect:
            game ? eboot_kit::fn<F4>(bn)(ctx, a[0], a[1], a[2]) : gx_immediate_draw_instanced_indirect(ctx, a[0], a[1], a[2]);
            break;
        case kMIndexedIndirect:
            game ? eboot_kit::fn<F4>(bn)(ctx, a[0], a[1], a[2]) : gx_immediate_draw_indexed_instanced_indirect(ctx, a[0], a[1], a[2]);
            break;
        case kMCommit: game ? eboot_kit::fn<F2>(bn)(ctx, a[0]) : gx_context_commit(ctx, a[0]); break;
        case kMDispatch:
            game ? eboot_kit::fn<F5>(bn)(ctx, a[0], a[1], a[2], a[3]) : gx_immediate_dispatch(ctx, a[0], a[1], a[2], a[3]);
            break;
        case kMDispatchIndirect:
            game ? eboot_kit::fn<F4>(bn)(ctx, a[0], a[1], a[2]) : gx_immediate_dispatch_indirect(ctx, a[0], a[1], a[2]);
            break;
        default: break;
    }
}

void random_method_case(int m, u64* a) {
    for (u8& b : g_ctx) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_buffer) b = static_cast<u8>(g_rng.u32());
    static const s32 kStages[] = {0, 0x200, 0x45, 0xad, 0x1ff, 0x201, -1, 0x44};
    store<s32>(reinterpret_cast<u64>(g_ctx) + 0x50, g_rng.chance(0.8) ? kStages[g_rng.range(0, 7)] : static_cast<s32>(g_rng.u32()));
    a[0] = (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32();  // the state
    for (int k = 1; k < 6; ++k) a[k] = garbage_u32();
    if (m == kMIndirect || m == kMIndexedIndirect || m == kMDispatchIndirect) a[1] = reinterpret_cast<u64>(g_buffer);
}

// The trace's entry as the test's: what it was handed, and its answer.
u64 g_hook_answer = 0;
std::vector<u64> g_hook_seen;
GUEST_ABI u64 test_hook(u64 kind, const u64* f) {
    g_hook_seen.assign({kind, f[-1], f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]});
    return g_hook_answer;
}

int run_methods(int cases) {
    int compared = 0;
    for (int m = 0; m < 9; ++m) {
        for (int t = 0; t < cases; ++t) {
            u64 a[6];
            random_method_case(m, a);
            decomp_gx::g_gx_hook_entry = nullptr;
            g_rec.clear();
            g_emit_ret = 0;
            call_method(m, true, a);
            const std::vector<Call> game = g_rec;
            // The table's return address is the game's own; ours' is taken for it.
            const u64 want_ret = m == kMCommit ? 0 : game_address(kGameReturns[m]);
            if (g_emit_ret != want_ret) fail(decomp_gx::kMethodNames[m], t, "the game's writer return address is not the table's");
            g_rec.clear();
            g_emit_ret = 0;
            call_method(m, false, a);
            if (m != kMCommit && (!g_emit_ret || decomp_gx::game_return(g_emit_ret) != want_ret))
                fail(decomp_gx::kMethodNames[m], t, "ours' writer return address is not taken for the game's");
            if (g_rec != game) {
                char why[160];
                std::snprintf(why, sizeof why, "the calls (%zu, the game's %zu; first %s)", g_rec.size(), game.size(),
                              game.empty() ? "-" : kCallees[game[0].who].name);
                fail(decomp_gx::kMethodNames[m], t, why);
            }
            // The trace's entry: taken, nothing runs; not taken, the sequence.
            decomp_gx::g_gx_hook_entry = reinterpret_cast<void*>(&test_hook);
            for (const u64 answer : {1ull, 0ull}) {
                g_hook_answer = answer;
                g_rec.clear();
                g_hook_seen.clear();
                call_method(m, false, a);
                const u64 ctx = reinterpret_cast<u64>(g_ctx);
                if (g_hook_seen.size() != 10 || g_hook_seen[0] != static_cast<u64>(m)) {
                    fail(decomp_gx::kMethodNames[m], t, "the trace's entry not called with its kind");
                    continue;
                }
                // kind, slot, r9 r8 rcx rdx rsi rdi, ret, stack arg
                const u64 want_regs[6] = {m == kMDrawInstanced || m == kMDrawIndexedInstanced ? a[4] : 0,
                                          (m == kMDrawIndexed || m == kMDrawInstanced || m == kMDrawIndexedInstanced || m == kMDispatch) ? a[3] : 0,
                                          m == kMCommit ? 0 : a[2], m == kMCommit ? 0 : a[1], a[0], ctx};
                // (r9 r8 rcx rdx rsi rdi)
                bool frame_ok = g_hook_seen[1] == 0 && g_hook_seen[8] != 0 && g_hook_seen[9] == 0;
                for (int k = 0; k < 6; ++k) frame_ok = frame_ok && g_hook_seen[2 + k] == want_regs[k];
                if (!frame_ok) fail(decomp_gx::kMethodNames[m], t, "the frame handed to the trace");
                if (answer == 1 && !g_rec.empty()) fail(decomp_gx::kMethodNames[m], t, "a call made after the trace took it");
                if (answer == 0 && g_rec != game) fail(decomp_gx::kMethodNames[m], t, "the sequence after the trace left it");
            }
            decomp_gx::g_gx_hook_entry = nullptr;
            ++compared;
        }
    }
    if (!g_quiet) std::printf("  methods: %d cases against the game's (each also through the trace's entry, taken and not)\n", compared);
    return compared;
}

// ---- The post-draw ----
void random_post_case() {
    for (u8& b : g_ctx) b = static_cast<u8>(g_rng.u32());
    const u64 ctx = reinterpret_cast<u64>(g_ctx), block = ctx + 0x50;
    u32 dirty = g_rng.u32();
    if (g_rng.chance(0.5)) {
        static const u32 kBits[] = {0x80, 0x10, 0x102, 0x100, 0x2, 0x60, 0x40, 0x20};
        dirty = 0;
        for (int k = 0; k < 8; ++k)
            if (g_rng.chance(0.4)) dirty |= kBits[k];
    }
    store<u32>(block + 0xb368, dirty);
    store<u32>(block + 0x8, g_rng.chance(0.5) ? 0xffu : g_rng.u32());
    store<u64>(block + 0x10, g_rng.chance(0.5) ? 0 : (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32());
    g_writer_ret[0] = (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32();
    g_writer_ret[1] = (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32();
}

int run_post(int cases) {
    static u8 saved[sizeof g_ctx], game_mem[sizeof g_ctx];
    for (int t = 0; t < cases; ++t) {
        random_post_case();
        const u64 ctx = reinterpret_cast<u64>(g_ctx);
        std::memcpy(saved, g_ctx, sizeof g_ctx);
        g_rec.clear();
        const u64 game_ret = eboot_kit::fn<GUEST_ABI u64 (*)(u64, u64)>(kPostDraw)(ctx + 0x50, ctx);
        const std::vector<Call> game = g_rec;
        std::memcpy(game_mem, g_ctx, sizeof g_ctx);
        std::memcpy(g_ctx, saved, sizeof g_ctx);
        g_rec.clear();
        const u64 ours = gx_post_draw(ctx + 0x50, ctx);
        if (ours != game_ret) fail("gx_post_draw", t, "the return value");
        if (std::memcmp(g_ctx, game_mem, sizeof g_ctx) != 0) fail("gx_post_draw", t, "the memory");
        if (g_rec != game) fail("gx_post_draw", t, "the writers' calls");
    }
    if (!g_quiet) std::printf("  gx_post_draw: %d cases against the game's\n", cases);
    return cases;
}

// ---- The flush ----
alignas(64) u8 g_nodes[64][0x30];
alignas(64) u64 g_pages[3][12];
alignas(64) u8 g_ring_holder[0xb740];
alignas(64) u64 g_ring[8];
u32 g_counter_a = 0, g_counter_b = 0;
alignas(64) u64 g_owner[2];
alignas(64) u64 g_owner_vtable[16];

int g_flush_slow = 0, g_flush_fresh = 0, g_flush_refilled = 0, g_flush_last = 0;

void random_flush_case() {
    for (u8& b : g_ctx) b = static_cast<u8>(g_rng.u32());
    const u64 cb = reinterpret_cast<u64>(g_ctx) + 0x10;
    const u64 arena_refill = load<u64>(kDefaultRefill);  // the image's, relocated
    const bool slow = g_rng.chance(0.7);
    // The fast way out: no queue, or another refill.
    store<u64>(cb + 0xb4d8, 0);
    store<u64>(cb + 0x18, g_rng.chance(0.5) ? arena_refill : reinterpret_cast<u64>(&refill_stub));
    if (!slow) {
        if (g_rng.chance(0.5)) store<u64>(cb + 0xb4d8, reinterpret_cast<u64>(g_pages[0]));
        if (load<u64>(cb + 0xb4d8)) store<u64>(cb + 0x18, reinterpret_cast<u64>(&refill_stub));
        return;
    }
    // The chunk: room for a fresh start or not.
    const u64 wp = reinterpret_cast<u64>(g_ctx) + 0x2000;
    store<u64>(cb + 0x10, wp);
    store<u64>(cb + 0x8, wp + 4ull * static_cast<u64>(g_rng.chance(0.3) ? g_rng.range(0, 7) : g_rng.range(7, 4000)));
    store<u64>(cb + 0x18, arena_refill);
    // Release lists on up to sixteen slots, and the pending list.
    int node = 0;
    u32 bits = 0;
    for (int s = 0; s < 16; ++s) {
        store<u64>(cb + 0xb3d8 + 8ull * s, (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32());
        if (!g_rng.chance(0.25) || node >= 56) continue;
        bits |= 1u << s;
        u64 head = 0;
        for (int n = g_rng.range(0, 3); n > 0; --n) {
            u8* p = g_nodes[node++];
            for (int k = 0; k < 0x30; ++k) p[k] = static_cast<u8>(g_rng.u32());
            store<u64>(reinterpret_cast<u64>(p) + 0x28, head);
            head = reinterpret_cast<u64>(p);
        }
        store<u64>(cb + 0xb3d8 + 8ull * s, head);
    }
    store<u32>(cb + 0xb460, bits);
    u64 pending = 0;
    for (int n = g_rng.range(0, 3); n > 0 && node < 64; --n) {
        u8* p = g_nodes[node++];
        for (int k = 0; k < 0x30; ++k) p[k] = static_cast<u8>(g_rng.u32());
        store<u64>(reinterpret_cast<u64>(p) + 0x28, pending);
        pending = reinterpret_cast<u64>(p);
    }
    store<u64>(cb + 0xb458, pending);
    // The arena queue: up to three pages of `last` chunks and a link.
    const u32 last = static_cast<u32>(g_rng.range(1, 10));
    store<u32>(cb + 0xb4e8, last);
    const int pages = g_rng.range(1, 3);
    for (int p = 0; p < pages; ++p) {
        for (u32 k = 0; k < last; ++k) g_pages[p][k] = (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32();
        g_pages[p][last] = p + 1 < pages ? reinterpret_cast<u64>(g_pages[p + 1]) : 0;
    }
    store<u64>(cb + 0xb4d8, reinterpret_cast<u64>(g_pages[0]));
    store<u32>(cb + 0xb4ec, static_cast<u32>(g_rng.range(0, static_cast<int>(last))));
    // The owner and its +0x70, or none.
    for (u64& v : g_owner_vtable) v = (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32();
    g_owner_vtable[0x70 / 8] = reinterpret_cast<u64>(&owner_stub);
    g_owner[0] = reinterpret_cast<u64>(g_owner_vtable);
    store<u64>(cb + 0xb4d0, g_rng.chance(0.7) ? reinterpret_cast<u64>(g_owner) : 0);
    // The compute ring's counters: on the last value or not.
    store<u64>(cb + 0xb490, reinterpret_cast<u64>(g_ring_holder));
    store<u64>(reinterpret_cast<u64>(g_ring_holder) + 0xb728, reinterpret_cast<u64>(g_ring));
    g_ring[3] = reinterpret_cast<u64>(&g_counter_a);
    g_ring[4] = reinterpret_cast<u64>(&g_counter_b);
    g_counter_a = g_rng.u32();
    g_counter_b = g_rng.chance(0.4) ? g_counter_a - 1 : g_rng.u32();
}

int run_flush(int cases) {
    static u8 saved[sizeof g_ctx], game_mem[sizeof g_ctx];
    for (int t = 0; t < cases; ++t) {
        random_flush_case();
        const u64 cb = reinterpret_cast<u64>(g_ctx) + 0x10;
        const bool slow = load<u64>(cb + 0xb4d8) && load<u64>(cb + 0x18) == load<u64>(kDefaultRefill);
        std::memcpy(saved, g_ctx, sizeof g_ctx);
        g_rec.clear();
        g_extra.clear();
        eboot_kit::fn<GUEST_ABI void (*)(u64)>(kCommandBufferFlush)(cb);
        const std::vector<Call> game = g_rec;
        const std::vector<Extra> game_extra = g_extra;
        std::memcpy(game_mem, g_ctx, sizeof g_ctx);
        std::memcpy(g_ctx, saved, sizeof g_ctx);
        g_rec.clear();
        g_extra.clear();
        gx_command_buffer_flush(cb);
        if (std::memcmp(g_ctx, game_mem, sizeof g_ctx) != 0) fail("gx_command_buffer_flush", t, "the memory");
        if (g_rec != game) fail("gx_command_buffer_flush", t, "the calls");
        if (g_extra != game_extra) fail("gx_command_buffer_flush", t, "the refill's or the owner's calls");
        if (slow) {
            ++g_flush_slow;
            bool closed = false, refilled = false;
            for (const Call& c : game) closed |= c.who == kClose;
            for (const Extra& e : game_extra) refilled |= e.who == 100;
            if (refilled) ++g_flush_refilled;
            else if (closed && g_counter_b == g_counter_a - 1) ++g_flush_last;
            else if (closed) ++g_flush_fresh;
        }
    }
    if (!g_quiet) std::printf("  gx_command_buffer_flush: %d cases against the game's (%d closing a segment: %d with a fresh start, %d the ring's last, %d asking "
                "the refill)\n",
                cases, g_flush_slow, g_flush_fresh, g_flush_last, g_flush_refilled);
    return cases;
}

// ---- The output-merger flush ----
alignas(64) u8 g_om_mem[0x1200];
constexpr u64 kOmState = 0x200, kOmTargets = 0x800, kOmTextures = 0xd00, kOmBlends = 0x1000, kOmDepthStencils = 0x1100;
u64 om_base() { return reinterpret_cast<u64>(g_om_mem); }
u64 om_target(int k) { return om_base() + kOmTargets + 0x80ull * static_cast<u64>(k); }

// A record of a bound target: none, the end mark, or one of ten targets.
u64 random_record() {
    const int pick = g_rng.range(0, 19);
    if (pick == 0) return 0;
    if (pick == 1) return ~0ull;
    return om_target(pick % 10);
}
// A float's bits as the generator wants them against another's: equal, -0
// against +0, the same NaN, a denormal, any.
void float_pair(u32* a, u32* b) {
    switch (g_rng.range(0, 5)) {
        case 0: *a = *b = g_rng.u32(); break;
        case 1: *a = 0x80000000u; *b = 0; if (g_rng.chance(0.5)) std::swap(*a, *b); break;
        case 2: *a = *b = 0x7fc00000u | (g_rng.u32() & 0xffff); break;
        case 3: *a = g_rng.u32() & 0x007fffffu; *b = g_rng.chance(0.5) ? *a : 0; break;
        default: *a = g_rng.u32(); *b = g_rng.u32(); break;
    }
}

void random_om_case() {
    for (u8& b : g_om_mem) b = static_cast<u8>(g_rng.u32());
    const u64 om = om_base(), state = om + kOmState;
    for (int k = 0; k < 10; ++k) {
        const u64 t = om_target(k), tex = om + kOmTextures + 0x40ull * static_cast<u64>(k);
        store<u64>(t + 8, g_rng.chance(0.8) ? tex : 0);
        store<u32>(t + 0x40, g_rng.chance(0.85) ? static_cast<u32>(g_rng.range(0, 9)) : g_rng.u32());
        if (g_rng.chance(0.5)) store<u32>(t + 0x48, static_cast<u32>(g_rng.range(0, 7)));
        if (g_rng.chance(0.6)) store<u32>(t + 0x5c, 0);
        store<u8>(tex + 0x3b, g_rng.chance(0.3) ? 0 : static_cast<u8>(g_rng.u32()));
    }
    store<u8>(om + 0x146, static_cast<u8>(g_rng.range(0, 8)));
    for (int set = 0; set < 2; ++set) {
        for (int i = 0; i < 9; ++i) store<u64>(om + 0x60 + 0x48ull * set + 8ull * i, random_record());
    }
    const u32 sets = load<u32>(om + 0x188);
    for (int i = 0; i < 8; ++i) {
        const u64 recorded = load<u64>(om + (sets & 1) * 0x48 + 0x60 + 8ull * i);
        store<u64>(state + 0x4e0 + 8ull * i, g_rng.chance(0.5) ? recorded : random_record());
    }
    const u64 depth = load<u64>(om + ((sets >> 1) & 1) * 0x48 + 0xa0);
    store<u64>(state + 0x520, g_rng.chance(0.5) ? depth : random_record());

    // The blend states: some fields what the block holds.
    for (int k = 0; k < 3; ++k) {
        const u64 b = om + kOmBlends + 0x40ull * static_cast<u64>(k);
        if (g_rng.chance(0.5)) store<u8>(b + 0x24, static_cast<u8>(g_rng.range(0, 1)));
        if (g_rng.chance(0.4)) store<u32>(om, load<u8>(b + 0x24));
        if (g_rng.chance(0.4)) store<u64>(om + 8, load<u32>(b));
        for (int i = 0; i < 8; ++i)
            if (g_rng.chance(0.6)) store<u32>(om + 0x10 + 4ull * i, load<u32>(b + 4 + 4ull * i));
    }
    // A state's object: none (the default one), the one the block last wrote whole, or one of three.
    const auto pick_state = [&](u64 first, u64 stride, u64 held) -> u64 {
        const int pick = g_rng.range(0, 4);
        if (pick == 0) return 0;
        if (pick == 1) return held;
        return first + stride * static_cast<u64>(pick - 2);
    };
    store<u64>(om + 0x190, g_rng.chance(0.3) ? om + kOmBlends : g_rng.chance(0.2) ? ~0ull : load<u64>(om + 0x190));
    store<u64>(state + 0x4d0, pick_state(om + kOmBlends, 0x40, load<u64>(om + 0x190)));
    // The depth-stencil states.
    for (int k = 0; k < 3; ++k) {
        const u64 d = om + kOmDepthStencils + 0x20ull * static_cast<u64>(k);
        if (g_rng.chance(0.4)) store<u32>(om + 0x48, load<u32>(d));
        if (g_rng.chance(0.4)) store<u32>(om + 0x4c, load<u32>(d + 4));
    }
    store<u64>(om + 0x198, g_rng.chance(0.3) ? om + kOmDepthStencils : g_rng.chance(0.2) ? ~0ull : load<u64>(om + 0x198));
    store<u64>(state + 0x4d8, pick_state(om + kOmDepthStencils, 0x20, load<u64>(om + 0x198)));
    // The blend constants, the depth bounds, the stencil reference.
    for (int k = 0; k < 4; ++k) {
        u32 a, b;
        float_pair(&a, &b);
        store<u32>(state + 0x568 + 4ull * k, a);
        store<u32>(om + 0x30 + 4ull * k, b);
    }
    if (g_rng.chance(0.4)) {
        for (int k = 0; k < 4; ++k) store<u32>(om + 0x30 + 4ull * k, load<u32>(state + 0x568 + 4ull * k));
    }
    if (g_rng.chance(0.5)) store<u64>(om + 0x58, load<u64>(state + 0x578));
    if (g_rng.chance(0.3)) store<u8>(state + 0x584, 0);
    if (g_rng.chance(0.4)) {
        // The reference the flush makes, held already: from the state's
        // depth-stencil object when it is not the one written whole, else
        // from what the block holds.
        const u64 ds = load<u64>(state + 0x4d8);
        const u64 from = ds != load<u64>(om + 0x198) ? (ds ? ds : game_address(kDefaultDepthStencilState)) + 8 : om + 0x50;
        const u32 r = load<u8>(state + 0x584);
        store<u32>(om + 0x50, (load<u32>(from) & 0xffff00u) | r | r << 24);
    }
}

int run_om(int cases) {
    int g_om_calls[kNumCallees] = {};
    static u8 saved[sizeof g_om_mem], game_mem[sizeof g_om_mem];
    const u64 om = om_base(), state = om + kOmState, ctx = reinterpret_cast<u64>(g_ctx);
    for (int t = 0; t < cases; ++t) {
        random_om_case();
        std::memcpy(saved, g_om_mem, sizeof g_om_mem);
        g_rec.clear();
        eboot_kit::fn<GUEST_ABI void (*)(u64, u64, u64)>(kFlushOutputMerger)(om, ctx, state);
        const std::vector<Call> game = g_rec;
        std::memcpy(game_mem, g_om_mem, sizeof g_om_mem);
        std::memcpy(g_om_mem, saved, sizeof g_om_mem);
        g_rec.clear();
        gx_flush_output_merger(om, ctx, state);
        if (std::memcmp(g_om_mem, game_mem, sizeof g_om_mem) != 0) fail("gx_flush_output_merger", t, "the memory");
        if (g_rec != game) {
            char why[320];
            std::size_t at = 0;
            while (at < g_rec.size() && at < game.size() && g_rec[at] == game[at]) ++at;
            const Call none{-1, {}};
            const Call& o = at < g_rec.size() ? g_rec[at] : none;
            const Call& g = at < game.size() ? game[at] : none;
            std::snprintf(why, sizeof why, "the calls (%zu, the game's %zu): #%zu ours %s %llx %llx %llx %llx, the game's %s %llx %llx %llx %llx",
                          g_rec.size(), game.size(), at, o.who < 0 ? "-" : kCallees[o.who].name, static_cast<ull>(o.a[1]),
                          static_cast<ull>(o.a[2]), static_cast<ull>(o.a[3]), static_cast<ull>(o.a[4]), g.who < 0 ? "-" : kCallees[g.who].name,
                          static_cast<ull>(g.a[1]), static_cast<ull>(g.a[2]), static_cast<ull>(g.a[3]), static_cast<ull>(g.a[4]));
            fail("gx_flush_output_merger", t, why);
        }
        for (const Call& c : game) ++g_om_calls[c.who];
    }
    if (!g_quiet) {
        std::printf("  gx_flush_output_merger: %d cases against the game's (calls:", cases);
        for (int w = 23; w < kNumCallees; ++w) std::printf(" %s %d", kCallees[w].name, g_om_calls[w]);
        std::printf(")\n");
    }
    return cases;
}

// ---- The mutants ----
// Each must be caught by its own part of the compare.
bool caught(int mutant) {
    const int before = g_failed;
    g_draw_mutant = mutant;
    g_quiet = true;
    const int quiet = g_failed;
    switch (mutant) {
        case 1: case 2: case 6: shared_stubbed(true); run_methods(40); break;
        case 3: shared_stubbed(false); run_post(400); break;
        case 4: case 5: shared_stubbed(false); run_flush(400); break;
        case 7: case 8: case 9: case 10: case 11: shared_stubbed(false); run_om(400); break;
        default: break;
    }
    g_draw_mutant = 0;
    g_quiet = false;
    const bool found = g_failed > quiet;
    g_failed = before;  // a mutant's failures are the point, not failures
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gx_draw_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 2000;
    stub_all(std::make_integer_sequence<int, kNumCallees>{});
    // The arena's refill itself (what +0x18 holds on a context's buffer), which
    // the flush asks for a chunk when the segment has no room left.
    eboot_kit::stub(load<u64>(kDefaultRefill), reinterpret_cast<const void*>(&refill_stub));

    // The decomp list places each at its entry only when these bytes are there.
    struct Entry {
        u64 bn;
        const u8* bytes;
        std::size_t n;
        const char* name;
    };
    const Entry kEntries[] = {
        {0x25696c0, kEntryPushRax, sizeof kEntryPushRax, "gx_immediate_draw"},
        {0x2569780, kEntryPushRax, sizeof kEntryPushRax, "gx_immediate_draw_indexed"},
        {0x2569840, kEntryPushRax, sizeof kEntryPushRax, "gx_immediate_draw_instanced"},
        {0x2569900, kEntrySub18, sizeof kEntrySub18, "gx_immediate_draw_indexed_instanced"},
        {0x25699d0, kEntryPushRax, sizeof kEntryPushRax, "gx_immediate_draw_instanced_indirect"},
        {0x2569ab0, kEntryPushRax, sizeof kEntryPushRax, "gx_immediate_draw_indexed_instanced_indirect"},
        {0x2569ba0, kEntryCommit, sizeof kEntryCommit, "gx_context_commit"},
        {0x25695c0, kEntryPushRax, sizeof kEntryPushRax, "gx_immediate_dispatch"},
        {0x2569640, kEntryPushRax, sizeof kEntryPushRax, "gx_immediate_dispatch_indirect"},
        {kPostDraw, kEntryPostDraw, sizeof kEntryPostDraw, "gx_post_draw"},
        {kCommandBufferFlush, kEntryFlush, sizeof kEntryFlush, "gx_command_buffer_flush"},
        {kFlushOutputMerger, kEntryOutputMerger, sizeof kEntryOutputMerger, "gx_flush_output_merger"},
    };
    for (const Entry& e : kEntries) {
        const u8* at = eboot_kit::file_at("gx_draw_test", e.bn);
        if (!at || std::memcmp(at, e.bytes, e.n) != 0) fail(e.name, -1, "the entry's instructions");
    }

    // The methods' code range and entries, as the decomp list's add notes them.
    decomp_gx_draw_add();
    if (decomp_gx::game_return(reinterpret_cast<u64>(&gx_post_draw)) != reinterpret_cast<u64>(&gx_post_draw))
        fail("game_return", -1, "an address outside the methods is not itself");

    shared_stubbed(false);
    long total = run_om(cases * 5);
    total += run_post(cases * 5);
    total += run_flush(cases * 5);
    shared_stubbed(true);
    total += run_methods(cases);

    // The mutants, each caught.
    for (const int mutant : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }

    if (g_failed) {
        std::printf("gx_draw_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("gx_draw_test: ok - %ld cases, 11 mutants caught\n", total);
    return 0;
}
