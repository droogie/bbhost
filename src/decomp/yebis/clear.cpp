// YEBIS's ClearRenderTargetView as source (docs/decomp.md): slot 50 of its
// D3D11-shaped context (decomp/yebis/context.cpp) and the compute utility its
// slow path runs.
//
//   - yebis_context_clear_render_target_view (0x2e41320): a target with a
//     fast-clear mask is cleared by YEBIS's fast clear (0x15ebe60) and its
//     slot's clear colour set, the colour as 8-bit unorm (each component
//     floor(c * 256), 255 for exactly 1 - right for the 8-bit targets that
//     have the mask); any other target by a compute pass (below), the wait
//     for it and the caches, the texture keeping the flush count;
//   - yebis_clear_target_compute (0x15e8e90): the target's memory acquired
//     for the compute pass, a texture of the target made, and the clear;
//   - yebis_compute_clear (0x15e8930): the compute shader for the texture's
//     type (1D, 2D and its arrays, 3D; a cube as a 2D array), the colour in a
//     16-byte constant buffer, then every mip level and array slice cleared
//     by a dispatch of 64, 8x8 or 4x4x4 threads; then the wait for the compute
//     shaders and the context's compute shader, texture and constant buffer
//     put back. A texture of a type it does not clear ends it early, as the
//     game's does (the command buffer left in compute mode).
//
// About one a frame (YEBIS's targets without the mask). Leaves: the game's
// calls in the game's order; Gnm's texture accessors are called, as the
// game's code does, on a texture of its own on the stack.
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <immintrin.h>

#include <cstdint>

using namespace decomp;

#ifdef DECOMP_YEBIS_CLEAR_MUTANTS
extern int g_clear_mutant;
#define MUTANT(n) (g_clear_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

// YEBIS's context and its Gnm context (decomp/yebis/context.cpp).
constexpr u64 kGnm = 0xa98, kCb = 0xaa0, kFlushes = 0x3400, kComputeMode = 0x34d8;

constexpr u64 kHasFastClear = 0x1486d80;     // Gnm RenderTarget: the CMASK fast clear on
constexpr u64 kFastClear = 0x15ebe60;        // YEBIS's fast clear (gnm, render target)
constexpr u64 kSetClearColor = 0x1473840;    // gnm_set_cb_clear_color(cb, slot, words*)
constexpr u64 kClearTarget = 0x15e8e90;      // yebis_clear_target_compute (ours)
constexpr u64 kWaitCompute = 0x15e8e20;      // yebis_wait_compute (ours)
constexpr u64 kAcquireMem = 0x1477f40;
constexpr u64 kAcquireRange = 0x1475e10;     // gnm_acquire_mem_range(cb, base 256, size 256, cb/db action, extended, cache action, engine)
constexpr u64 kRtBase = 0x1486ca0, kRtFormat = 0x1486cf0, kRtSliceTiles = 0x1486cc0, kRtFragments = 0x14868f0;
constexpr u64 kBitsPerElement = 0x1471750;  // DataFormat's bits an element
constexpr u64 kTextureFromTarget = 0x1488d90;  // Gnm Texture from a RenderTarget (out, rt, 0)
constexpr u64 kComputeClear = 0x15e8930;     // yebis_compute_clear (ours)
// YEBIS's state and its Gnm context's setters.
constexpr u64 kStateShaders = 0x15efa00;     // (state) -> the bound shaders (the compute shader at +0x38)
constexpr u64 kStateTexture = 0x15efb80;     // (out texture, state, stage, slot)
constexpr u64 kStateBuffer = 0x15efc00;      // (state, stage, slot) -> the bound constant buffer (16 bytes)
constexpr u64 kSetComputeShader = 0x15fb840; // (gnm, shader)
constexpr u64 kSetBuffers = 0x15fb6d0;       // (gnm, stage, slot, count, buffers*)
constexpr u64 kSetTextures = 0x15fb6a0;      // (gnm, stage, slot, count, textures*)
constexpr u64 kDispatch = 0x15fbd10;         // yebis_gnm_dispatch (ours)
constexpr u64 kEmbedData = 0x14760c0, kEndOfShader = 0x1478320, kWaitRegMem = 0x1477b30;
constexpr u64 kClearShaders = 0x593a048;     // the clear's compute shaders: 1D (+0), 2D (+8), 3D (+0x10)
// Gnm's Texture (T#) and Buffer (V#) methods.
constexpr u64 kTexType = 0x1489830, kTexWidth = 0x14898a0, kTexHeight = 0x1489100, kTexDepth = 0x14890e0;
constexpr u64 kTexBaseLevel = 0x1489900, kTexLastLevel = 0x1489910, kTexBaseSlice = 0x14890f0, kTexLastSlice = 0x14890d0;
constexpr u64 kTexA = 0x1489960, kTexB = 0x14898f0, kTexC = 0x1489940, kTexD = 0x14898b0;
constexpr u64 kTexInit2d = 0x1487ee0, kTexSetD = 0x14890c0, kTexSetMemoryType = 0x1488330;
constexpr u64 kTexSetLevels = 0x1488510, kTexSetSlices = 0x14884f0, kTexValid = 0x14899c0;
constexpr u64 kBufInitConstant = 0x1470960, kBufSetStride = 0x14706d0, kBufValid = 0x1471230;

using One = GUEST_ABI u64 (*)(u64);
using Two = GUEST_ABI u64 (*)(u64, u64);
using Three = GUEST_ABI u64 (*)(u64, u64, u64);
using Four = GUEST_ABI u64 (*)(u64, u64, u64, u64);
using Five = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64);
using Seven = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64, u64, u64);
using Nine = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64, u64, u64, u64, u64);
struct Pair {
    u64 lo, hi;
};
using PairOf3 = GUEST_ABI Pair (*)(u64, u64, u64);

u32 call1(u64 fn, u64 a) { return static_cast<u32>(game_function<One>(fn)(a)); }

// A colour component as 8-bit unorm, as the fast path makes it: 255 for
// exactly 1, else floor(c * 256) in double, truncated to an int (the
// conversion's 0x80000000 for a NaN or a value past the int's range), its
// low byte.
__attribute__((target("sse4.1"))) u32 unorm8(float c) {
    double d = 255.0;
    if (c != 1.0f) d = static_cast<double>(c) * (MUTANT(1) ? 255.0 : 256.0);
    const __m128d r = _mm_round_sd(_mm_set_sd(255.0), _mm_set_sd(d), _MM_FROUND_TO_NEG_INF);
    return static_cast<u32>(_mm_cvttsd_si32(r));
}

}  // namespace

// yebis_compute_clear(gnm, texture*, colour*).
DECOMP_LEAF void yebis_compute_clear(u64 y, u64 in, u64 colour) {
    const u64 cb0 = load<u64>(y + 0x8);
    store<u64>(cb0 + 0x28, load<u64>(cb0 + 0x28) | 2);
    const u64 state = load<u64>(y + 0x28);
    const u64 saved_shader = load<u64>(game_function<One>(kStateShaders)(state) + 0x38);
    alignas(16) u8 saved_tex[0x20];
    game_function<Four>(kStateTexture)(reinterpret_cast<u64>(saved_tex), load<u64>(y + 0x28), 0, 0);
    const Pair saved_buf_value = game_function<PairOf3>(kStateBuffer)(load<u64>(y + 0x28), 0, 0);
    alignas(16) u64 saved_buf[2] = {saved_buf_value.lo, saved_buf_value.hi};
    alignas(16) u8 tex[0x20];
    std::memcpy(tex, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(in)), sizeof tex);
    const u64 t = reinterpret_cast<u64>(tex);
    const u32 kind = call1(kTexType, in) - 8;
    if (kind > 5) return;
    u64 shader;
    switch (kind) {
        case 0: case 4: shader = load<u64>(game_address(kClearShaders)); break;
        case 2: shader = load<u64>(game_address(kClearShaders + 0x10)); break;
        case 3: {  // a cube: cleared as a 2D array of its faces
            const u32 w = call1(kTexWidth, t) + 1, h = call1(kTexHeight, t) + 1, d = call1(kTexLastSlice, t) + 1, l = call1(kTexLastLevel, t) + 1;
            const u32 a = call1(kTexA, t), b = call1(kTexB, t), c = call1(kTexC, t);
            game_function<Nine>(kTexInit2d)(t, w, h, d, l, a, b, c, 0);
            game_function<Two>(kTexSetD)(t, call1(kTexD, in));
        }
            [[fallthrough]];
        default: shader = load<u64>(game_address(kClearShaders + 8)); break;
    }
    game_function<Two>(kSetComputeShader)(y, shader);
    // The colour, in the command buffer, as a 16-byte constant buffer.
    const u64 data = game_function<Three>(kEmbedData)(load<u64>(y + 0x8), 0x10, 2);
    const u64 c0 = load<u64>(colour), c1 = load<u64>(colour + 8);
    store<u64>(data + 8, c1);
    store<u64>(data, c0);
    alignas(16) u8 vsharp[0x10];
    const u64 v = reinterpret_cast<u64>(vsharp);
    game_function<Three>(kBufInitConstant)(v, data, 0x10);
    game_function<Two>(kBufSetStride)(v, 0x10);
    game_function<Five>(kSetBuffers)(y, 0, 0, 1, v);
    game_function<Two>(kTexSetMemoryType)(t, 0x6d);
    const u32 base_level = call1(kTexBaseLevel, t), last_level = call1(kTexLastLevel, t);
    const u32 base_slice = call1(kTexBaseSlice, t), last_slice = call1(kTexLastSlice, t);
    for (u32 m = base_level; m <= last_level; ++m) {
        game_function<Three>(kTexSetLevels)(t, m, m);
        u32 w = (call1(kTexWidth, t) + 1) >> (m & 31);
        if (!w) w = 1;
        u32 h = (call1(kTexHeight, t) + 1) >> (m & 31);
        if (!h) h = 1;
        const u32 d = (call1(kTexDepth, t) + 1) >> (m & 31);
        if (base_slice > last_slice) continue;
        const u32 x64 = (w + 63) >> 6, x8 = (w + 7) >> 3, y8 = (h + 7) >> 3, x4 = (w + 3) >> 2, y4 = (h + 3) >> 2;
        const u32 z4 = d ? (d + 3) >> 2 : 1;
        for (u32 s = base_slice; s <= last_slice; ++s) {
            game_function<Three>(kTexSetSlices)(t, s, s);
            game_function<Five>(kSetTextures)(y, 0, 0, 1, t);
            const u32 type = call1(kTexType, t);
            if (type > 13) return;
            if ((0x2a00u >> type) & 1) game_function<Four>(kDispatch)(y, x8, y8, 1);
            else if ((0x1100u >> type) & 1) game_function<Four>(kDispatch)(y, MUTANT(2) ? x8 : x64, 1, 1);
            else if (type == 10) game_function<Four>(kDispatch)(y, x4, y4, z4);
            else return;
        }
    }
    // The wait for the compute shaders (yebis_wait_compute's, inline), then the context's state back.
    const u64 cb = load<u64>(y + 0x8);
    const u64 label = game_function<Three>(kEmbedData)(cb, 8, 3);
    store<u64>(label, 0);
    game_function<Four>(kEndOfShader)(cb, 0x2f, label, 1);
    game_function<Five>(kWaitRegMem)(cb, label, 0xffffffffu, 3, 1);
    game_function<Four>(kAcquireMem)(cb, 0x38, 0, 1);
    game_function<Two>(kSetComputeShader)(y, saved_shader);
    game_function<Five>(kSetTextures)(y, 0, 0, 1, static_cast<u8>(call1(kTexValid, reinterpret_cast<u64>(saved_tex))) ? reinterpret_cast<u64>(saved_tex) : 0);
    game_function<Five>(kSetBuffers)(y, 0, 0, 1, static_cast<u8>(call1(kBufValid, reinterpret_cast<u64>(saved_buf))) ? reinterpret_cast<u64>(saved_buf) : 0);
    const u64 cb1 = load<u64>(y + 0x8);
    store<u64>(cb1 + 0x28, load<u64>(cb1 + 0x28) & ~2ull);
}

// yebis_clear_target_compute(gnm, render target, colour*).
DECOMP_LEAF void yebis_clear_target_compute(u64 y, u64 rt, u64 colour) {
    const u64 cb = load<u64>(y + 0x8);
    const u32 base = call1(kRtBase, rt);
    alignas(4) u32 format = call1(kRtFormat, rt);
    const u32 bits = call1(kBitsPerElement, reinterpret_cast<u64>(&format));
    u32 size = ((call1(kRtSliceTiles, rt) << 6) + 0x40) * bits;
    size = (size << (call1(kRtFragments, rt) & 31)) >> 8;
    game_function<Seven>(kAcquireRange)(cb, base, size, 0x3fc0, 0, 0x2000000, 1);
    alignas(16) u8 tex[0x20];
    game_function<Three>(kTextureFromTarget)(reinterpret_cast<u64>(tex), rt, 0);
    game_function<Three>(kComputeClear)(y, reinterpret_cast<u64>(tex), colour);
}

// ClearRenderTargetView(ctx, view, colour*) (slot 50).
DECOMP_LEAF void yebis_context_clear_render_target_view(u64 ctx, u64 view, u64 colour) {
    const u64 rt = load<u64>(load<u64>(view + 0xd8));
    if (call1(kHasFastClear, rt)) {
        game_function<Two>(kFastClear)(ctx + kGnm, rt);
        const u32 r = unorm8(load<float>(colour)), g = unorm8(load<float>(colour + 4)), b = unorm8(load<float>(colour + 8));
        const u32 a = unorm8(load<float>(colour + 12));
        alignas(8) u32 words[2] = {(r & 0xff) | (g & 0xff) << 8 | (b & 0xff) << 16 | a << 24, 0xffffffffu};
        game_function<Three>(kSetClearColor)(load<u64>(ctx + kCb), load<u32>(view + 0xe0), reinterpret_cast<u64>(words));
        return;
    }
    game_function<Three>(kClearTarget)(ctx + kGnm, load<u64>(load<u64>(view + 0xd8)), colour);
    game_function<One>(kWaitCompute)(load<u64>(ctx + kCb));
    game_function<Four>(kAcquireMem)(load<u64>(ctx + kCb), 0x38, 0, 0);
    store<u64>(load<u64>(view + 0x80) + 0x10, load<u64>(ctx + kFlushes));
    if (!MUTANT(3)) store<u32>(ctx + kComputeMode, 0);
}

namespace {
#define PUSH6 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53
// sub rsp, 0xb8
constexpr u8 kEntryComputeClear[] = {PUSH6, 0x48, 0x81, 0xec, 0xb8, 0x00, 0x00, 0x00};
// sub rsp, 0x48
constexpr u8 kEntryClearTarget[] = {PUSH6, 0x48, 0x83, 0xec, 0x48};
// sub rsp, 0x18
constexpr u8 kEntryClearView[] = {PUSH6, 0x48, 0x83, 0xec, 0x18};
#undef PUSH6
}  // namespace

void decomp_yebis_clear_add() {
    const DecompFunction fns[] = {
        {"yebis_context_clear_render_target_view", "yebis", 0x2e41320, kEntryClearView, sizeof kEntryClearView,
         reinterpret_cast<void*>(&yebis_context_clear_render_target_view), DecompKind::Leaf},
        {"yebis_clear_target_compute", "yebis", 0x15e8e90, kEntryClearTarget, sizeof kEntryClearTarget,
         reinterpret_cast<void*>(&yebis_clear_target_compute), DecompKind::Leaf},
        {"yebis_compute_clear", "yebis", 0x15e8930, kEntryComputeClear, sizeof kEntryComputeClear, reinterpret_cast<void*>(&yebis_compute_clear),
         DecompKind::Leaf},
    };
    for (const DecompFunction& fn : fns) decomp_add(fn);
}
