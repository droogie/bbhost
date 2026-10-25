// YEBIS's device context (decomp/yebis/context.cpp) beside the game's own
// (tests/decomp/eboot_kit.h). Every callee off the list is a recorder; the
// draw and dispatch prologues and the three waits, which are on it, run as
// the game's inside both versions of what calls them (and are compared
// themselves). On generated contexts - colour and depth views, their
// textures and Gnm render targets, old targets among the new ones, draw and
// flush counts that make the hazard test go both ways, garbage above each
// 32-bit argument's bits - ours must make the game's calls with the game's
// arguments in the game's order, leave every byte of the context, the views,
// the textures and the command buffer's header as the game's leaves them, and
// return what the game's returns. Nine deliberate mistakes must be caught, and
// the entries hold the instructions written against. Skips without the 1.09
// eboot.
//
// One difference is ours on purpose: a drain (at the end of a target bind) and
// the three shader waits write a barrier token where the game's embed a label
// in the command buffer, have an end-of-pipe or end-of-shader event write it
// and wait for it. So for those, the compare leaves out the game's three calls
// (embed_data, end_of_pipe or end_of_shader, wait_reg_mem) and its zeroing of
// the embedded label, and checks instead that ours wrote one barrier token for
// each - {0xc0001000, kGxBarrierMagic} at the command buffer's write pointer,
// the pointer moved on by two dwords, the refill asked for two dwords with the
// room left when there was none, nothing written when it refused - and made
// every other call of the game's, the cache action's included, in order.
#define DECOMP_YEBIS_CONTEXT_MUTANTS
int g_context_mutant = 0;
#include "decomp/yebis/context.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <utility>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x7eb15);

struct Call {
    int who;
    u64 a[8];
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_rec;
struct Callee {
    u64 bn;
    int args;
    u32 narrow;  // a bit per argument that is 32-bit
    const char* name;
};
const Callee kCallees[] = {
    {kAcquireMem, 4, 0xe, "acquire_mem"},
    {kEmbedData, 3, 0x6, "embed_data"},
    {kEndOfPipe, 8, 0xd6, "end_of_pipe"},
    {kEndOfShader, 4, 0xa, "end_of_shader"},
    {kWaitRegMem, 5, 0x1c, "wait_reg_mem"},
    {kSetDepthTarget, 2, 0, "set_depth_target"},
    {kSetPrimType, 2, 0x2, "set_prim_type"},
    {kSetIndexSize, 2, 0x2, "set_index_size"},
    {kSetInstances, 2, 0x2, "set_instances"},
    {kSetIndirectBase, 2, 0, "set_indirect_base"},
    {kDefaultState, 1, 0, "default_state"},
    {kRegister, 2, 0, "register"},
    {kSetTarget, 3, 0x2, "set_target"},
    {kApplyStages, 1, 0, "apply_stages"},
    {kApplyOutput, 1, 0, "apply_output"},
    {kSetComputeShader, 2, 0, "set_compute_shader"},
    {kSubmit, 1, 0, "submit"},
    {kNextBuffer, 1, 0, "next_buffer"},
    {kGnmDrawIndex, 5, 0x1a, "gnm_draw_index"},
    {kGnmDrawAuto, 2, 0x2, "gnm_draw_auto"},
    {kGnmDispatch, 4, 0xe, "gnm_dispatch"},
    {kGnmDispatchIndirect, 2, 0x2, "gnm_dispatch_indirect"},
    {kPrepDraw, 4, 0xc, "prep_draw"},
};
constexpr int kNumCallees = static_cast<int>(std::size(kCallees));
constexpr int kEmbedId = 1;
u64 g_returns[kNumCallees];  // what each recorder returns this case
u64 g_label = 0;             // where the embedded label goes

template <int W>
GUEST_ABI u64 recorder(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7) {
    Call c{W, {a0, a1, a2, a3, a4, a5, a6, a7}};
    for (int k = 0; k < 8; ++k) {
        if (k >= kCallees[W].args) c.a[k] = 0;
        else if ((kCallees[W].narrow >> k) & 1) c.a[k] &= 0xffffffffull;
    }
    g_rec.push_back(c);
    return W == kEmbedId ? g_label : g_returns[W];
}
template <int... W>
void stub_all(std::integer_sequence<int, W...>) {
    (eboot_kit::stub(kCallees[W].bn, reinterpret_cast<const void*>(&recorder<W>)), ...);
}

int g_failed = 0;
bool g_quiet = false;
void fail(const char* what, int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: %s, case %d: %s\n", what, trial, why);
}

// The arena: the context, the command buffer's header, 20 views, their
// textures, Gnm render targets and the cells views point at them through,
// the views array, the label, a compute shader and an argument buffer.
constexpr u64 kCtxAt = 0x0, kCbAt = 0x3600, kViewsAt = 0x3700, kTexAt = 0x5000, kRtAt = 0x6000, kCellAt = 0x6600, kArrayAt = 0x6800,
              kLabelAt = 0x6900, kShaderAt = 0x6a00, kBufferAt = 0x6b00, kPktAt = 0x7000, kRefillAt = 0x7200, kArenaBytes = 0x7400;
constexpr u64 kPktBytes = 0x200;
constexpr int kPool = 20;
alignas(64) u8 g_arena[kArenaBytes];
alignas(64) u8 g_saved[kArenaBytes];
alignas(64) u8 g_game[kArenaBytes];
u64 A(u64 off) { return reinterpret_cast<u64>(g_arena) + off; }
u64 view_at(int k) { return A(kViewsAt + 0x100ull * k); }

u64 garbage32() { return (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32(); }
u64 small(int hi) { return static_cast<u64>(g_rng.range(0, hi)); }

// The command buffer's refill, as its owner's callback: asked for dwords with
// room left; gives a new chunk (kRefillAt) or refuses (g_refill_gives).
struct RefillCall {
    u64 cb, arg;
    u32 dwords, room;
    bool operator==(const RefillCall&) const = default;
};
std::vector<RefillCall> g_refills;
bool g_refill_gives = true;
GUEST_ABI u8 refill_stub(u64 cb, u32 dwords, u64 arg, u32 room) {
    g_refills.push_back({cb, arg, dwords, room});
    if (!g_refill_gives) return 0;
    store<u64>(cb + 0x0, A(kRefillAt));
    store<u64>(cb + 0x8, A(kRefillAt) + kPktBytes);
    store<u64>(cb + 0x10, A(kRefillAt));
    return 1;
}

// The context and its objects, generated.
void generate() {
    for (u8& b : g_arena) b = static_cast<u8>(g_rng.u32());
    const u64 ctx = A(kCtxAt);
    store<u64>(ctx + kCb, A(kCbAt));
    // The command buffer: its chunk, its end (now and then too near for a
    // token's two dwords, or before the write pointer), its write pointer, its
    // refill and the refill's argument.
    {
        const u64 cb = A(kCbAt), wp = A(kPktAt) + 8 * small(8);
        const u64 room = g_rng.chance(0.15) ? small(1) * 4 : 8 * (1 + small(20));
        store<u64>(cb + 0x0, g_rng.chance(0.03) ? 0 : A(kPktAt));
        store<u64>(cb + 0x8, wp + room);
        store<u64>(cb + 0x10, wp);
        store<u64>(cb + 0x18, reinterpret_cast<u64>(&refill_stub));
        g_refill_gives = !g_rng.chance(0.2);
    }
    store<u64>(ctx + kDraws, small(12));
    store<u64>(ctx + kDrained, small(12));
    store<u64>(ctx + kFlushes, g_rng.chance(0.2) ? 0 : small(4));
    for (int i = 0; i < 8; ++i) store<u64>(ctx + kTargets + 8ull * i, g_rng.chance(0.3) ? 0 : view_at(g_rng.range(0, kPool - 1)));
    store<u32>(ctx + kAaMode, static_cast<u32>(small(3)));
    store<u32>(ctx + kAaMode + 4, static_cast<u32>(small(7)));
    store<u32>(ctx + kSize, static_cast<u32>(small(3)));
    store<u32>(ctx + kSize + 4, static_cast<u32>(small(3)));
    const u32 marks[] = {kExportAny, kExportWide, kExportMixed, g_rng.u32()};
    store<u32>(ctx + kExport, marks[g_rng.range(0, 3)]);
    store<u32>(ctx + kComputeMode, static_cast<u32>(g_rng.chance(0.4) ? small(2) : g_rng.u32()));
    store<u32>(ctx + kDispatched, static_cast<u32>(small(2)));
    if (g_rng.chance(0.3)) store<u32>(ctx + kIndexStride, g_rng.u32());  // past 32 bits when multiplied
    store<u64>(ctx + kComputeShader, A(kShaderAt));
    for (int k = 0; k < kPool; ++k) {
        const u64 v = view_at(k), tex = A(kTexAt + 0xc0ull * k), rt = A(kRtAt + 0x40ull * k), cell = A(kCellAt + 8ull * k);
        store<u64>(v + 0x10, small(4));
        store<u64>(v + 0x80, g_rng.chance(0.1) ? A(kTexAt) : tex);  // now and then two views of one texture
        store<u8>(v + 0x90, static_cast<u8>(g_rng.chance(0.5) ? 0 : 1 + small(3)));
        store<u64>(v + 0xd8, cell);
        store<u64>(cell, rt);
        const u32 formats[] = {0x0f, 0x10, 0x11, 0x12, g_rng.u32()};
        store<u32>(v + 0xe4, formats[g_rng.range(0, 4)]);
        if (g_rng.chance(0.5)) store<u32>(v + 0xf0, static_cast<u32>(small(3)) << 16 | (g_rng.chance(0.3) ? 0 : static_cast<u32>(small(3))));
        store<u64>(tex + 0x10, small(4));
        store<u64>(tex + 0xa0, small(12));
        store<u16>(rt + 0x18, static_cast<u16>(g_rng.chance(0.3) ? 0 : small(3)));
        store<u16>(rt + 0x1a, static_cast<u16>(small(3)));
    }
    for (int i = 0; i < 10; ++i) store<u64>(A(kArrayAt) + 8ull * i, g_rng.chance(0.25) ? 0 : view_at(g_rng.range(0, kPool - 1)));
    for (u64& r : g_returns) r = garbage32();
    g_label = A(kLabelAt);
}

enum Fn {
    kFOmSetRenderTargets, kFPrepareDraw, kFPrepareDispatch, kFDrawIndexed, kFDraw, kFDrawIndexedInstanced, kFDrawInstanced, kFDispatch,
    kFDispatchIndirect, kFFlush, kFFlushTo, kFWaitFlush, kFWaitCompute, kFWaitPixels, kFWaitDispatch, kFns
};
const char* const kNames[kFns] = {"yebis_context_om_set_render_targets", "yebis_context_prepare_draw", "yebis_context_prepare_dispatch",
                                  "yebis_context_draw_indexed",          "yebis_context_draw",         "yebis_context_draw_indexed_instanced",
                                  "yebis_context_draw_instanced",        "yebis_context_dispatch",     "yebis_context_dispatch_indirect",
                                  "yebis_context_flush",                 "yebis_context_flush_to",     "yebis_context_wait_flush",
                                  "yebis_wait_compute",                  "yebis_wait_pixels",          "yebis_wait_dispatch"};
const u64 kBn[kFns] = {0x15d74c0, kPrepareDraw, kPrepareDispatch, 0x2e40880, 0x2e408e0, 0x2e40aa0, 0x2e40b40, 0x2e410b0,
                       0x2e410f0, 0x15d99e0,    0x15d9ab0,        0x15d9b70, kWaitCompute, kWaitPixels, kWaitDispatch};
void* const kOurs[kFns] = {
    reinterpret_cast<void*>(&yebis_context_om_set_render_targets), reinterpret_cast<void*>(&yebis_context_prepare_draw),
    reinterpret_cast<void*>(&yebis_context_prepare_dispatch),      reinterpret_cast<void*>(&yebis_context_draw_indexed),
    reinterpret_cast<void*>(&yebis_context_draw),                  reinterpret_cast<void*>(&yebis_context_draw_indexed_instanced),
    reinterpret_cast<void*>(&yebis_context_draw_instanced),        reinterpret_cast<void*>(&yebis_context_dispatch),
    reinterpret_cast<void*>(&yebis_context_dispatch_indirect),     reinterpret_cast<void*>(&yebis_context_flush),
    reinterpret_cast<void*>(&yebis_context_flush_to),              reinterpret_cast<void*>(&yebis_context_wait_flush),
    reinterpret_cast<void*>(&yebis_wait_compute),                  reinterpret_cast<void*>(&yebis_wait_pixels),
    reinterpret_cast<void*>(&yebis_wait_dispatch)};

// Each function's arguments: the context (the command buffer for the
// waits), then what it takes - 32-bit ones with garbage above.
void arguments(int f, u64* a) {
    const u64 ctx = A(kCtxAt);
    for (int k = 0; k < 6; ++k) a[k] = garbage32();
    a[0] = f >= kFWaitCompute ? A(kCbAt) : ctx;
    switch (f) {
        case kFOmSetRenderTargets:
            a[1] = (a[1] & ~0xffffffffull) | small(9);
            a[2] = A(kArrayAt);
            a[3] = g_rng.chance(0.3) ? 0 : view_at(g_rng.range(0, kPool - 1));
            break;
        case kFDrawIndexedInstanced: a[5] &= ~0xffffffffull; break;  // no start instance (it traps)
        case kFDrawInstanced:
            a[3] &= ~0xffffffffull;  // no start vertex, no start instance
            a[4] &= ~0xffffffffull;
            break;
        case kFDispatchIndirect: a[1] = A(kBufferAt); break;
        default: break;
    }
}

u64 call(void* fn, const u64* a) {
    using F6 = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64, u64);
    return reinterpret_cast<F6>(fn)(a[0], a[1], a[2], a[3], a[4], a[5]);
}

// What the target binds covered: drains, first binds, slots unbound; and the
// cases that wrote barrier tokens.
long g_drained = 0, g_registered = 0, g_unbound = 0, g_binds = 0, g_tokens = 0;

long run(int cases) {
    long n = 0;
    for (int f = 0; f < kFns; ++f) {
        for (int t = 0; t < cases; ++t) {
            generate();
            u64 a[6];
            arguments(f, a);
            std::memcpy(g_saved, g_arena, kArenaBytes);
            g_rec.clear();
            const u64 game_ret = call(reinterpret_cast<void*>(eboot_kit::at(kBn[f])), a);
            const std::vector<Call> game = g_rec;
            std::memcpy(g_game, g_arena, kArenaBytes);
            if (f == kFOmSetRenderTargets && !g_quiet) {
                ++g_binds;
                bool drained = false, registered = false, unbound = false;
                for (const Call& c : game) {
                    drained |= c.who == 2;
                    registered |= c.who == 11;
                    unbound |= c.who == 12 && c.a[2] == 0;
                }
                g_drained += drained;
                g_registered += registered;
                g_unbound += unbound;
            }
            std::memcpy(g_arena, g_saved, kArenaBytes);
            g_rec.clear();
            g_refills.clear();
            const u64 our_ret = call(kOurs[f], a);
            if (our_ret != game_ret) fail(kNames[f], t, "the return value");
            // The drain's and the shader waits' label calls, the deliberate
            // difference: left out of the game's, one barrier token each
            // expected of ours instead (the header comment).
            std::vector<Call> want;
            int barriers = 0;
            const bool tokens = f == kFOmSetRenderTargets || f >= kFWaitCompute;
            for (std::size_t k = 0; k < game.size(); ++k) {
                if (tokens && k + 2 < game.size() && game[k].who == kEmbedId && (game[k + 1].who == 2 || game[k + 1].who == 3) &&
                    game[k + 2].who == 4) {
                    ++barriers;
                    k += 2;
                    continue;
                }
                want.push_back(game[k]);
            }
            if (g_rec != want) fail(kNames[f], t, "the calls");
            // The tokens: what the command buffer should hold and say after
            // `barriers` of them from the header as generated.
            u8 expect[kArenaBytes];
            std::memcpy(expect, g_game, kArenaBytes);
            if (barriers) {
                std::memcpy(expect + kLabelAt, g_saved + kLabelAt, 8);  // the game zeroed its label; ours has none
                const u64 cb = A(kCbAt);
                u64 chunk, end, wp;
                std::memcpy(&chunk, g_saved + kCbAt, 8);
                std::memcpy(&end, g_saved + kCbAt + 8, 8);
                std::memcpy(&wp, g_saved + kCbAt + 0x10, 8);
                std::vector<RefillCall> refills;
                bool refused = false;
                for (int b = 0; b < barriers; ++b) {
                    if (!chunk || end < wp + 8) {
                        u64 arg;
                        std::memcpy(&arg, g_saved + kCbAt + 0x20, 8);
                        refills.push_back({cb, arg, 2, end > wp ? static_cast<u32>((end - wp) >> 2) : 0});
                        if (!g_refill_gives) {
                            refused = true;
                            continue;
                        }
                        chunk = A(kRefillAt);
                        end = A(kRefillAt) + kPktBytes;
                        wp = A(kRefillAt);
                    }
                    const u32 token[2] = {0xc0001000u, kGxBarrierMagic};
                    std::memcpy(expect + (wp - A(0)), token, 8);
                    wp += 8;
                }
                std::memcpy(expect + kCbAt, &chunk, 8);
                std::memcpy(expect + kCbAt + 8, &end, 8);
                std::memcpy(expect + kCbAt + 0x10, &wp, 8);
                if (g_refills != refills) fail(kNames[f], t, refused ? "the refill (refused)" : "the refill");
            } else if (!g_refills.empty()) {
                fail(kNames[f], t, "a refill with no token to write");
            }
            if (std::memcmp(g_arena, expect, kArenaBytes) != 0) fail(kNames[f], t, barriers ? "the memory (with the tokens)" : "the memory");
            if (barriers && !g_quiet) ++g_tokens;
            ++n;
        }
    }
    return n;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_context_mutant = mutant;
    g_quiet = true;
    run(300);
    g_quiet = false;
    g_context_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("yebis_context_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 4000;
    stub_all(std::make_integer_sequence<int, kNumCallees>{});
    struct Entry {
        u64 bn;
        const u8* bytes;
        std::size_t n;
    };
    const Entry kEntries[kFns] = {
        {0x15d74c0, kEntryOmSetRenderTargets, sizeof kEntryOmSetRenderTargets},
        {kPrepareDraw, kEntryPrepareDraw, sizeof kEntryPrepareDraw},
        {kPrepareDispatch, kEntryPrepareDispatch, sizeof kEntryPrepareDispatch},
        {0x2e40880, kEntryDrawIndexed, sizeof kEntryDrawIndexed},
        {0x2e408e0, kEntryDraw, sizeof kEntryDraw},
        {0x2e40aa0, kEntryDrawIndexedInstanced, sizeof kEntryDrawIndexedInstanced},
        {0x2e40b40, kEntryDrawInstanced, sizeof kEntryDrawInstanced},
        {0x2e410b0, kEntryDispatch, sizeof kEntryDispatch},
        {0x2e410f0, kEntryDispatchIndirect, sizeof kEntryDispatchIndirect},
        {0x15d99e0, kEntryFlush, sizeof kEntryFlush},
        {0x15d9ab0, kEntryFlushTo, sizeof kEntryFlushTo},
        {0x15d9b70, kEntryWaitFlush, sizeof kEntryWaitFlush},
        {kWaitCompute, kEntryWait, sizeof kEntryWait},
        {kWaitPixels, kEntryWait, sizeof kEntryWait},
        {kWaitDispatch, kEntryWait, sizeof kEntryWait},
    };
    for (int f = 0; f < kFns; ++f) {
        const u8* at = eboot_kit::file_at("yebis_context_test", kEntries[f].bn);
        if (!at || std::memcmp(at, kEntries[f].bytes, kEntries[f].n) != 0) fail(kNames[f], -1, "the entry's instructions");
    }
    const long total = run(cases);
    for (int mutant = 1; mutant <= 9; ++mutant) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("yebis_context_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("yebis_context_test: ok - %ld cases over %d functions (of %ld target binds, %ld drained, %ld bound a view the first time, %ld unset a "
                "slot; %ld wrote barrier tokens), 9 mutants caught\n",
                total, kFns, g_binds, g_drained, g_registered, g_unbound, g_tokens);
    return 0;
}
