// The Gnm packet writers whose packets nothing on the host reads, as our
// source (docs/decomp.md). Twenty-one small leaf methods of Gnm's command
// buffers, each of which makes room for one packet, writes it and moves the
// write pointer - nothing else:
//
// - the ACQUIRE_MEM cache actions, four writers (the draw and the compute
//   command buffers', each with and without a range): the renderer orders its
//   own work with Vulkan barriers, and the command processor has no case for
//   them at all;
// - fifteen context-register writers: the multi-sample sample locations,
//   centroid priority and configuration, the depth-stencil disable, the blend
//   constants, alpha to mask, the scratch ring size, the screen offset, the
//   guard band's clip and discard adjustments, the instance step rates, the
//   pixel shader's input table, the vertex rounding and the line width. Every
//   draw is a host token carrying its own state, so nothing reads the context
//   registers any more - while the command processor keeps no register file
//   (a shadow or packet-path mode reads them: then ours writes them);
// - Gnm's shader prefetch, a DMA_DATA into a register;
// - prepareFlip's room for the console driver's flip packets, a 64-dword NOP
//   the system fills at the flip - the flip is ours. Nothing reads it either,
//   but it writes the game's NOP by default: without it the frames' flips
//   settled into the jittery pacing regime far more often and Fishing
//   Hamlet's flips straddled the vblank (a refresh missed every 3-4 s in six
//   runs of six), since it moves where a frame's command buffer ends in its
//   chunk (render/cporder 6e70f0e). Kept until the pacing itself is robust.
//
// Over the 12-place tour that was 13.8M ACQUIRE_MEM (96.9M dwords), 21.4M
// SET_CONTEXT_REG (95.3M dwords, 31% of everything the command processor
// walked), 248k prefetches and 45k flip reservations (render/cporder's
// census, which found them; it suppressed them with byte patches at their
// entries, which these replace).
//
// Each is the game's function as source, instruction for instruction: the
// same room check (the low 32 bits of the dwords left, against the packet's
// size), the same refill call when there is no room (cb, dwords, its
// argument, the room left, as a 64-bit count), the same packet, the same
// write-pointer move and the same return value (the packet's address, or the
// write pointer after it, as each one returns; the refill's result when it
// gives nothing). tests/decomp/gx_dead_packets_test.cpp checks all of it
// against the game's code on generated inputs. What the twenty dead ones write
// by default is nothing (GnmDeadMode, set by hle_gx_trace_arm): no room asked
// for, no packet, the write pointer returned as the address - which no caller
// reads (the poison check: every one returning an address no caller can
// follow, 12 places without a fault). Nor does a caller move the write
// pointer back after one, write below a write pointer it reads again, keep
// the write pointer across the call or write through a copy of the command
// buffer (their 159 call sites read in the binary): a writer whose packet
// its caller rewinds over cannot write nothing - the constant engine's wait
// (decomp/gx/constant_engine.cpp). BBHOST_CP_DEAD_KEEP=<places> keeps
// chosen ones writing. As source, each is also where a host command can be
// recorded in the packet's place.
//
// The command buffer they write through: +0x8 the chunk's end, +0x10 the write
// pointer, +0x18 the refill, +0x20 its argument.
//
// Leaves: entered straight from the game's code, they touch only the game's
// memory and their mode; BBHOST_DECOMP_COMPARE=1 runs the game's version of
// each in their place and checks ours against every call (hosted).
#include "decomp/gx/dead_packets.h"

#include "core/thunk.h"
#include "decomp/decomp.h"
#include "decomp/guest.h"
#include "log.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

using namespace decomp;

namespace {

enum Writer : std::size_t {
    kAcquireMemRange,
    kAcquireMem,
    kComputeAcquireMemRange,
    kComputeAcquireMem,
    kSampleLocations,
    kSampleLocationsAndCentroid,
    kCentroidPriority,
    kAaConfig,
    kDepthStencilDisable,
    kBlendColor,
    kAlphaToMask,
    kTmpringSize,
    kScreenOffset,
    kGuardBandClip,
    kGuardBandDiscard,
    kInstanceStepRate,
    kPsInputCntl,
    kVtxCntl,
    kLineCntl,
    kShaderPrefetch,
    kFlipRoom,
    kWriters
};

// Same order as render/cporder's table, so BBHOST_CP_DEAD_KEEP's places are
// its: 0-19 the twenty dead writers, 20 prepareFlip's reservation (not on its
// list since 6e70f0e; here it writes the game's NOP by default).
const GnmDeadWriter kTable[kWriters] = {
    {"gnm_acquire_mem_range", "ACQUIRE_MEM", 0x1475e10, 14,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50}, 7, false, true},
    {"dcb_acquire_mem", "ACQUIRE_MEM (b)", 0x1477f40, 14,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xce}, 7, false, true},
    {"gnm_compute_acquire_mem_range", "ACQUIRE_MEM (queue)", 0x148b550, 14,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50}, 7, false, true},
    {"gnm_compute_acquire_mem", "ACQUIRE_MEM (queue b)", 0x148b610, 15,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x89, 0xd3, 0x41, 0x89, 0xf7}, 7, false, true},
    {"gnm_set_aa_sample_locations", "sample locations", 0x14772c0, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89, 0xf3, 0x49, 0x89, 0xfe, 0x49, 0x8b, 0x4e, 0x08}, 18, true, true},
    {"gnm_set_aa_sample_locations_and_centroid", "sample locations (b)", 0x14773b0, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89, 0xf3, 0x49, 0x89, 0xfe, 0x49, 0x8b, 0x4e, 0x08}, 22, true, true},
    {"gnm_set_centroid_priority", "PA_SC_CENTROID_PRIORITY", 0x1477350, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}, 4, true, true},
    {"gnm_set_aa_config", "PA_SC_AA_CONFIG", 0x14771c0, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}, 3, true, true},
    {"gnm_set_depth_stencil_disable", "DB_DEPTH_CONTROL 0", 0x14734c0, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08, 0x48, 0x8b, 0x43, 0x10}, 3, true, true},
    {"gnm_set_blend_color", "the blend constants", 0x1473100, 16,
     {0x55, 0x48, 0x89, 0xe5, 0x53, 0x48, 0x83, 0xec, 0x18, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}, 6, true, true},
    {"gnm_set_alpha_to_mask", "DB_ALPHA_TO_MASK", 0x1473030, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}, 3, true, true},
    {"gnm_set_spi_tmpring_size", "SPI_TMPRING_SIZE", 0x1472ed0, 15,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x89, 0xd3, 0x41, 0x89, 0xf7}, 3, true, true},
    {"gnm_set_hardware_screen_offset", "PA_SU_HARDWARE_SCREEN_OFFSET", 0x1473650, 15,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x89, 0xd3, 0x41, 0x89, 0xf7}, 3, true, true},
    {"gnm_set_guard_band_clip_adjust", "the guard band", 0x14736d0, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08, 0x48, 0x8b, 0x43, 0x10}, 6, true, true},
    {"gnm_set_guard_band_discard_adjust", "the guard band (b)", 0x1473750, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08, 0x48, 0x8b, 0x43, 0x10}, 6, true, true},
    {"gnm_set_instance_step_rate", "the instance step rate", 0x1474f90, 16,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x41, 0x89, 0xd6, 0x41, 0x89, 0xf7}, 4, true, true},
    {"gnm_set_ps_input_cntl", "SPI_PS_INPUT_CNTL", 0x1475000, 16,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x89, 0xd3, 0x49, 0x89, 0xf4}, 10, true, true},
    {"gnm_set_su_vtx_cntl", "PA_SU_VTX_CNTL", 0x1476560, 14,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xce}, 3, true, true},
    {"gnm_set_su_line_cntl", "PA_SU_LINE_CNTL", 0x1476e60, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}, 3, true, true},
    {"gnm_draw_shader_prefetch_dma", "the shader prefetch (DMA_DATA into a register)", 0x1473f50, 16,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x41, 0x89, 0xd6, 0x49, 0x89, 0xf7}, 7, false, true},
    {"gnm_reserve_flip_packets", "the flip's room for the system's packets (a 64-dword NOP)", 0x14784a0, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08, 0x48, 0x8b, 0x43, 0x10}, 64, false, false},
};

// GnmDeadMode per writer, read by the leaves. Until hle_gx_trace_arm sets them
// every writer writes the game's packet (none of the game's code runs before).
std::uint8_t g_mode[kWriters] = {};

constexpr u64 kPoison = 0xdead000000000000ull;

// The refill every writer calls when there is no room: (cb, dwords, its
// argument, the room left in dwords); its result is a bool in the low byte,
// and a writer that gets false returns the whole register as it came back.
using Refill = GUEST_ABI u64 (*)(u64 cb, u32 dwords, u64 arg, u64 room);

// The room for `need` dwords: true with the write pointer to write at; false
// with the refill's result when it gave nothing. The game's check is the low
// 32 bits of the dwords left against the packet's size (`cmp ecx, n - 1; ja`,
// or `cmp ecx, n; jae` for the pixel-shader input table, the same for n > 0),
// and after a refill it reads the write pointer again without a second check.
__attribute__((always_inline)) inline bool make_room(u64 cb, u32 need, u64* wp, u64* failed) {
    u64 p = load<u64>(cb + 0x10);
    const u64 room = (load<u64>(cb + 0x8) - p) >> 2;
    if (static_cast<u32>(room) < need) {
        const auto refill = reinterpret_cast<Refill>(static_cast<std::uintptr_t>(load<u64>(cb + 0x18)));
        const u64 r = refill(cb, need, load<u64>(cb + 0x20), room);
        if (static_cast<u8>(r) == 0) {
            *failed = r;
            return false;
        }
        p = load<u64>(cb + 0x10);
    }
    *wp = p;
    return true;
}

__attribute__((always_inline)) inline u64 nothing(u64 cb, u8 mode) {
    return mode == static_cast<u8>(GnmDeadMode::kPoison) ? kPoison : load<u64>(cb + 0x10);
}

__attribute__((always_inline)) inline void put(u64 at, u32 v) { store<u32>(at, v); }
__attribute__((always_inline)) inline u32 bits(float f) { return std::bit_cast<u32>(f); }

// The game's functions, written out. Each returns what the game's returns.

// ACQUIRE_MEM over a range (0x1475e10): CP_COHER_CNTL from five arguments,
// CP_COHER_SIZE, CP_COHER_BASE, the poll interval.
inline u64 w_acquire_mem_range(u64 cb, u32 base, u32 size, u32 cntl_a, u32 targets, u32 cntl_b, u32 engine) {
    u64 wp, failed;
    if (!make_room(cb, 7, &wp, &failed)) return failed;
    const u32 cntl = ((targets & 0xb) << 15) | ((targets & 0x30) << 18) | ((cntl_b | cntl_a) & 0x7fffffff) | (engine << 31);
    put(wp, 0xc0055800u);
    put(wp + 4, cntl);
    put(wp + 8, size);
    put(wp + 12, 0);
    put(wp + 16, base);
    put(wp + 20, 0);
    put(wp + 24, 0xa);
    store<u64>(cb + 0x10, wp + 28);
    return wp;
}

// ACQUIRE_MEM of the whole memory (0x1477f40): a size of 1 at base 0.
inline u64 w_acquire_mem(u64 cb, u32 targets, u32 cntl, u32 engine) {
    u64 wp, failed;
    if (!make_room(cb, 7, &wp, &failed)) return failed;
    put(wp, 0xc0055800u);
    put(wp + 4, ((targets & 0xb) << 15) | ((targets & 0x30) << 18) | (cntl & 0x7fffffff) | (engine << 31));
    put(wp + 8, 1);
    put(wp + 12, 0);
    put(wp + 16, 0);
    put(wp + 20, 0);
    put(wp + 24, 0xa);
    store<u64>(cb + 0x10, wp + 28);
    return wp;
}

// The compute command buffer's ACQUIRE_MEM over a range (0x148b550): bits 25
// and 26 of the last argument dropped.
inline u64 w_compute_acquire_mem_range(u64 cb, u32 base, u32 size, u32 cntl_a, u32 targets, u32 cntl_b) {
    u64 wp, failed;
    if (!make_room(cb, 7, &wp, &failed)) return failed;
    put(wp, 0xc0055800u);
    put(wp + 4, (cntl_a & 0x7fffffff) | (cntl_b & 0x79ffffff) | ((targets & 0x30) << 18) | ((targets & 0xb) << 15));
    put(wp + 8, size);
    put(wp + 12, 0);
    put(wp + 16, base);
    put(wp + 20, 0);
    put(wp + 24, 0xa);
    store<u64>(cb + 0x10, wp + 28);
    return wp;
}

// ... and of no range (0x148b610): size and base 0.
inline u64 w_compute_acquire_mem(u64 cb, u32 targets, u32 cntl) {
    u64 wp, failed;
    if (!make_room(cb, 7, &wp, &failed)) return failed;
    put(wp, 0xc0055800u);
    put(wp + 4, (cntl & 0x79ffffff) | ((targets & 0x30) << 18) | ((targets & 0xb) << 15));
    store<u64>(wp + 8, 0);
    store<u64>(wp + 16, 0);
    put(wp + 24, 0xa);
    store<u64>(cb + 0x10, wp + 28);
    return wp;
}

// PA_SC_AA_SAMPLE_LOCS_PIXEL_* (0x2fe): sixteen dwords from the caller's.
inline void sample_locations(u64 wp, u64 locs) {
    store<u64>(wp, 0x000002fec0106900ull);
    for (u64 k = 0; k < 64; k += 8) store<u64>(wp + 8 + k, load<u64>(locs + k));
}

inline u64 w_sample_locations(u64 cb, u64 locs) {
    u64 wp, failed;
    if (!make_room(cb, 18, &wp, &failed)) return failed;
    sample_locations(wp, locs);
    store<u64>(cb + 0x10, wp + 0x48);
    return wp;
}

// ... and PA_SC_CENTROID_PRIORITY_0/1 (0x2f5) from the qword after them.
inline u64 w_sample_locations_and_centroid(u64 cb, u64 locs) {
    u64 wp, failed;
    if (!make_room(cb, 22, &wp, &failed)) return failed;
    sample_locations(wp, locs);
    const u64 prio = load<u64>(locs + 0x40);
    put(wp + 0x48, 0xc0026900u);
    put(wp + 0x4c, 0x2f5);
    put(wp + 0x50, static_cast<u32>(prio));
    put(wp + 0x54, static_cast<u32>(prio >> 32));
    store<u64>(cb + 0x10, wp + 0x58);
    return wp;
}

inline u64 w_centroid_priority(u64 cb, u64 prio) {
    u64 wp, failed;
    if (!make_room(cb, 4, &wp, &failed)) return failed;
    put(wp, 0xc0026900u);
    put(wp + 4, 0x2f5);
    put(wp + 8, static_cast<u32>(prio));
    put(wp + 12, static_cast<u32>(prio >> 32));
    store<u64>(cb + 0x10, wp + 16);
    return wp;
}

// PA_SC_AA_CONFIG (0x2f8) from the sample count's log2: the maximum sample
// distance from the game's byte table at 0x4b2bf4c, indexed by the whole
// argument as the game indexes it.
inline u64 w_aa_config(u64 cb, u32 log2_samples) {
    u64 wp, failed;
    if (!make_room(cb, 3, &wp, &failed)) return failed;
    const u32 dist = load<u8>(game_address(0x4b2bf4c) + log2_samples) & 0xfu;
    const u32 n = log2_samples & 7;
    put(wp, 0xc0016900u);
    put(wp + 4, 0x2f8);
    put(wp + 8, (n << 20) | n | (dist << 13));
    store<u64>(cb + 0x10, wp + 12);
    return wp + 12;
}

// DB_DEPTH_CONTROL (0x200) 0.
inline u64 w_depth_stencil_disable(u64 cb) {
    u64 wp, failed;
    if (!make_room(cb, 3, &wp, &failed)) return failed;
    put(wp, 0xc0016900u);
    put(wp + 4, 0x200);
    put(wp + 8, 0);
    store<u64>(cb + 0x10, wp + 12);
    return wp + 12;
}

// CB_BLEND_RED..ALPHA (0x105), the floats' bits as they came.
inline u64 w_blend_color(u64 cb, float r, float g, float b, float a) {
    u64 wp, failed;
    if (!make_room(cb, 6, &wp, &failed)) return failed;
    put(wp, 0xc0046900u);
    put(wp + 4, 0x105);
    put(wp + 8, bits(r));
    put(wp + 12, bits(g));
    put(wp + 16, bits(b));
    put(wp + 20, bits(a));
    store<u64>(cb + 0x10, wp + 24);
    return wp;
}

// DB_ALPHA_TO_MASK (0x2dc).
inline u64 w_alpha_to_mask(u64 cb, u32 v) {
    u64 wp, failed;
    if (!make_room(cb, 3, &wp, &failed)) return failed;
    put(wp, 0xc0016900u);
    put(wp + 4, 0x2dc);
    put(wp + 8, v);
    store<u64>(cb + 0x10, wp + 12);
    return wp + 12;
}

// SPI_TMPRING_SIZE (0x1ba): waves in bits 0-11, the wave size in 12-24.
inline u64 w_tmpring_size(u64 cb, u32 waves, u32 wave_size) {
    u64 wp, failed;
    if (!make_room(cb, 3, &wp, &failed)) return failed;
    put(wp, 0xc0016900u);
    put(wp + 4, 0x1ba);
    put(wp + 8, ((wave_size << 12) & 0x1fff000) | (waves & 0xfff));
    store<u64>(cb + 0x10, wp + 12);
    return wp + 12;
}

// PA_SU_HARDWARE_SCREEN_OFFSET (0x8d).
inline u64 w_screen_offset(u64 cb, u32 x, u32 y) {
    u64 wp, failed;
    if (!make_room(cb, 3, &wp, &failed)) return failed;
    put(wp, 0xc0016900u);
    put(wp + 4, 0x8d);
    put(wp + 8, ((y << 16) & 0x1fc0000) | (x & 0x1fc));
    store<u64>(cb + 0x10, wp + 12);
    return wp + 12;
}

// The guard band: the vertical adjustment (the second float) at `vert`, the
// horizontal (the first) at vert + 2, each a packet of its own.
inline u64 guard_band(u64 cb, u32 vert, float horz, float v) {
    u64 wp, failed;
    if (!make_room(cb, 6, &wp, &failed)) return failed;
    put(wp, 0xc0016900u);
    put(wp + 4, vert);
    put(wp + 8, bits(v));
    put(wp + 12, 0xc0016900u);
    put(wp + 16, vert + 2);
    put(wp + 20, bits(horz));
    store<u64>(cb + 0x10, wp + 24);
    return wp + 24;
}

inline u64 w_guard_band_clip(u64 cb, float horz, float vert) { return guard_band(cb, 0x2fa, horz, vert); }
inline u64 w_guard_band_discard(u64 cb, float horz, float vert) { return guard_band(cb, 0x2fb, horz, vert); }

// VGT_INSTANCE_STEP_RATE_0/1 (0x2a8).
inline u64 w_instance_step_rate(u64 cb, u32 rate0, u32 rate1) {
    u64 wp, failed;
    if (!make_room(cb, 4, &wp, &failed)) return failed;
    put(wp, 0xc0026900u);
    put(wp + 4, 0x2a8);
    put(wp + 8, rate0);
    put(wp + 12, rate1);
    store<u64>(cb + 0x10, wp + 16);
    return wp;
}

// SPI_PS_INPUT_CNTL_0.. (0x191): `count` dwords of the caller's table, the
// room and the move count + 2 dwords in 32-bit arithmetic, and the table
// copied only for counts 1 to 32 (the game's jump table) - room is taken for
// any count all the same.
inline u64 w_ps_input_cntl(u64 cb, u64 table, u32 count) {
    const u32 need = count + 2;
    u64 wp, failed;
    if (!make_room(cb, need, &wp, &failed)) return failed;
    put(wp, (count << 16) | 0xc0006900u);
    put(wp + 4, 0x191);
    if (count - 1 <= 31) {
        for (u32 k = count; k-- > 0;) put(wp + 8 + 4 * static_cast<u64>(k), load<u32>(table + 4 * static_cast<u64>(k)));
    }
    const u64 next = wp + 4 * static_cast<u64>(need);
    store<u64>(cb + 0x10, next);
    return next;
}

// PA_SU_VTX_CNTL (0x2f9): the pixel centre in bit 0, rounding in 1-2,
// quantisation in 3-5.
inline u64 w_vtx_cntl(u64 cb, u32 quant, u32 round, u32 center) {
    u64 wp, failed;
    if (!make_room(cb, 3, &wp, &failed)) return failed;
    put(wp, 0xc0016900u);
    put(wp + 4, 0x2f9);
    put(wp + 8, (center & 1) | (((round + round) & 6) | ((quant << 3) & 0x38)));
    store<u64>(cb + 0x10, wp + 12);
    return wp + 12;
}

// PA_SU_LINE_CNTL (0x282): the line width, its low 16 bits.
inline u64 w_line_cntl(u64 cb, u32 width) {
    u64 wp, failed;
    if (!make_room(cb, 3, &wp, &failed)) return failed;
    put(wp, 0xc0016900u);
    put(wp + 4, 0x282);
    put(wp + 8, width & 0xffff);
    store<u64>(cb + 0x10, wp + 12);
    return wp + 12;
}

// The shader prefetch: a DMA_DATA from the program into a register.
inline u64 w_shader_prefetch(u64 cb, u64 addr, u32 size) {
    u64 wp, failed;
    if (!make_room(cb, 7, &wp, &failed)) return failed;
    put(wp, 0xc0055000u);
    put(wp + 4, 0x60000000u);
    put(wp + 8, static_cast<u32>(addr));
    put(wp + 12, static_cast<u32>(addr >> 32));
    put(wp + 16, 0x3022c);
    put(wp + 20, 0);
    put(wp + 24, (size & 0x1fffff) | 0x28200000u);
    store<u64>(cb + 0x10, wp + 28);
    return wp;
}

// prepareFlip's 64 dwords: the NOP's header and its marker, the rest left as
// it was for the system to fill.
inline u64 w_flip_room(u64 cb) {
    u64 wp, failed;
    if (!make_room(cb, 64, &wp, &failed)) return failed;
    put(wp, 0xc03e1000u);
    put(wp + 4, 0x68750777u);
    store<u64>(cb + 0x10, wp + 0x100);
    return wp + 0x100;
}

// The leaves on the decomp list: the mode, then the game's function.
#define DEAD_WRITER(name, index, params, call)                                     \
    DECOMP_LEAF u64 name params {                                                  \
        if (const u8 mode = g_mode[index]) return nothing(cb, mode);               \
        return call;                                                               \
    }

DEAD_WRITER(gnm_acquire_mem_range, kAcquireMemRange, (u64 cb, u32 base, u32 size, u32 a, u32 t, u32 b, u32 e),
            w_acquire_mem_range(cb, base, size, a, t, b, e))
DEAD_WRITER(dcb_acquire_mem, kAcquireMem, (u64 cb, u32 t, u32 c, u32 e), w_acquire_mem(cb, t, c, e))
DEAD_WRITER(gnm_compute_acquire_mem_range, kComputeAcquireMemRange, (u64 cb, u32 base, u32 size, u32 a, u32 t, u32 b),
            w_compute_acquire_mem_range(cb, base, size, a, t, b))
DEAD_WRITER(gnm_compute_acquire_mem, kComputeAcquireMem, (u64 cb, u32 t, u32 c), w_compute_acquire_mem(cb, t, c))
DEAD_WRITER(gnm_set_aa_sample_locations, kSampleLocations, (u64 cb, u64 locs), w_sample_locations(cb, locs))
DEAD_WRITER(gnm_set_aa_sample_locations_and_centroid, kSampleLocationsAndCentroid, (u64 cb, u64 locs),
            w_sample_locations_and_centroid(cb, locs))
DEAD_WRITER(gnm_set_centroid_priority, kCentroidPriority, (u64 cb, u64 prio), w_centroid_priority(cb, prio))
DEAD_WRITER(gnm_set_aa_config, kAaConfig, (u64 cb, u32 n), w_aa_config(cb, n))
DEAD_WRITER(gnm_set_depth_stencil_disable, kDepthStencilDisable, (u64 cb), w_depth_stencil_disable(cb))
DEAD_WRITER(gnm_set_blend_color, kBlendColor, (u64 cb, float r, float g, float b, float a), w_blend_color(cb, r, g, b, a))
DEAD_WRITER(gnm_set_alpha_to_mask, kAlphaToMask, (u64 cb, u32 v), w_alpha_to_mask(cb, v))
DEAD_WRITER(gnm_set_spi_tmpring_size, kTmpringSize, (u64 cb, u32 waves, u32 size), w_tmpring_size(cb, waves, size))
DEAD_WRITER(gnm_set_hardware_screen_offset, kScreenOffset, (u64 cb, u32 x, u32 y), w_screen_offset(cb, x, y))
DEAD_WRITER(gnm_set_guard_band_clip_adjust, kGuardBandClip, (u64 cb, float h, float v), w_guard_band_clip(cb, h, v))
DEAD_WRITER(gnm_set_guard_band_discard_adjust, kGuardBandDiscard, (u64 cb, float h, float v), w_guard_band_discard(cb, h, v))
DEAD_WRITER(gnm_set_instance_step_rate, kInstanceStepRate, (u64 cb, u32 r0, u32 r1), w_instance_step_rate(cb, r0, r1))
DEAD_WRITER(gnm_set_ps_input_cntl, kPsInputCntl, (u64 cb, u64 table, u32 count), w_ps_input_cntl(cb, table, count))
DEAD_WRITER(gnm_set_su_vtx_cntl, kVtxCntl, (u64 cb, u32 q, u32 r, u32 c), w_vtx_cntl(cb, q, r, c))
DEAD_WRITER(gnm_set_su_line_cntl, kLineCntl, (u64 cb, u32 w), w_line_cntl(cb, w))
DEAD_WRITER(gnm_draw_shader_prefetch_dma, kShaderPrefetch, (u64 cb, u64 addr, u32 size), w_shader_prefetch(cb, addr, size))
DEAD_WRITER(gnm_reserve_flip_packets, kFlipRoom, (u64 cb), w_flip_room(cb))

#undef DEAD_WRITER

// Every writer's arguments as the registers carry them: the six integer
// registers, the first stack argument, xmm0-3.
struct Args {
    u64 r[6];
    u64 s0;
    float x[4];
};

// The game's function, written out, on those registers (ours, whatever the mode).
u64 write_as_game(std::size_t i, const Args& a) {
    const u64 cb = a.r[0];
    const auto d = [&](int k) { return static_cast<u32>(a.r[k]); };
    switch (i) {
        case kAcquireMemRange: return w_acquire_mem_range(cb, d(1), d(2), d(3), d(4), d(5), static_cast<u32>(a.s0));
        case kAcquireMem: return w_acquire_mem(cb, d(1), d(2), d(3));
        case kComputeAcquireMemRange: return w_compute_acquire_mem_range(cb, d(1), d(2), d(3), d(4), d(5));
        case kComputeAcquireMem: return w_compute_acquire_mem(cb, d(1), d(2));
        case kSampleLocations: return w_sample_locations(cb, a.r[1]);
        case kSampleLocationsAndCentroid: return w_sample_locations_and_centroid(cb, a.r[1]);
        case kCentroidPriority: return w_centroid_priority(cb, a.r[1]);
        case kAaConfig: return w_aa_config(cb, d(1));
        case kDepthStencilDisable: return w_depth_stencil_disable(cb);
        case kBlendColor: return w_blend_color(cb, a.x[0], a.x[1], a.x[2], a.x[3]);
        case kAlphaToMask: return w_alpha_to_mask(cb, d(1));
        case kTmpringSize: return w_tmpring_size(cb, d(1), d(2));
        case kScreenOffset: return w_screen_offset(cb, d(1), d(2));
        case kGuardBandClip: return w_guard_band_clip(cb, a.x[0], a.x[1]);
        case kGuardBandDiscard: return w_guard_band_discard(cb, a.x[0], a.x[1]);
        case kInstanceStepRate: return w_instance_step_rate(cb, d(1), d(2));
        case kPsInputCntl: return w_ps_input_cntl(cb, a.r[1], d(2));
        case kVtxCntl: return w_vtx_cntl(cb, d(1), d(2), d(3));
        case kLineCntl: return w_line_cntl(cb, d(1));
        case kShaderPrefetch: return w_shader_prefetch(cb, a.r[1], d(2));
        case kFlipRoom: return w_flip_room(cb);
        default: return 0;
    }
}

// The room the game's version asks for, in dwords.
u32 room_for(std::size_t i, const Args& a) {
    return i == kPsInputCntl ? static_cast<u32>(a.r[2]) + 2 : kTable[i].dwords;
}

// BBHOST_DECOMP_COMPARE=1: the game's version runs in ours' place on every
// call, and ours is run on the same command buffer beside it - the packet's
// bytes, the write pointer after it and the return value compared - then the
// game's result is left in place. A call the refill would serve runs the
// game's alone (the refill cannot run twice).
DecompCompare g_counts[kWriters];
void* g_original[kWriters] = {};
std::atomic<std::uint64_t> g_compare_refills{0};
std::atomic<int> g_compare_logs{0};
constexpr u32 kCompareMax = 64;  // dwords; a larger pixel-shader input table runs the game's alone

u64 compare(std::size_t i, const Args& a) {
    const u64 cb = a.r[0];
    GuestCallRegs g{};
    g.fn = g_original[i];
    for (int k = 0; k < 6; ++k) g.ints[k] = a.r[k];
    for (int k = 0; k < 4; ++k) g.xmm[k] = bits(a.x[k]);
    g.stack[0] = a.s0;
    const u64 wp = load<u64>(cb + 0x10);
    const u32 need = room_for(i, a);
    if (static_cast<u32>((load<u64>(cb + 0x8) - wp) >> 2) < need || need > kCompareMax) {
        g_compare_refills.fetch_add(1, std::memory_order_relaxed);
        hle_call_guest_regs(&g);
        return g.ret;
    }
    const std::size_t bytes = static_cast<std::size_t>(need) * 4;
    u8 before[kCompareMax * 4], game[kCompareMax * 4];
    std::memcpy(before, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), bytes);
    hle_call_guest_regs(&g);
    const u64 game_ret = g.ret, game_wp = load<u64>(cb + 0x10);
    std::memcpy(game, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), bytes);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(wp)), before, bytes);
    store<u64>(cb + 0x10, wp);
    const u64 our_ret = write_as_game(i, a);
    const u64 our_wp = load<u64>(cb + 0x10);
    const bool same = our_ret == game_ret && our_wp == game_wp &&
                      std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), game, bytes) == 0;
    g_counts[i].calls.fetch_add(1, std::memory_order_relaxed);
    if (!same) {
        g_counts[i].differ.fetch_add(1, std::memory_order_relaxed);
        if (g_compare_logs.fetch_add(1) < 16) {
            host_log("decomp: %s differs from the game's: returned 0x%llx (game 0x%llx), write pointer +%lld (game +%lld)", kTable[i].name,
                     static_cast<ull>(our_ret), static_cast<ull>(game_ret), static_cast<long long>(our_wp - wp),
                     static_cast<long long>(game_wp - wp));
        }
    }
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(wp)), game, bytes);
    store<u64>(cb + 0x10, game_wp);
    return game_ret;
}

template <std::size_t I>
GUEST_ABI u64 compare_writer(u64 r0, u64 r1, u64 r2, u64 r3, u64 r4, u64 r5, u64 s0, float x0, float x1, float x2, float x3) {
    return compare(I, Args{{r0, r1, r2, r3, r4, r5}, s0, {x0, x1, x2, x3}});
}

void report() {
    static bool once = false;
    if (once) return;
    once = true;
    std::string modes[3];
    std::size_t n[3] = {}, compared = 0;
    for (std::size_t i = 0; i < kWriters; ++i) {
        const int placed = decomp_placed(kTable[i].name);
        if (!placed) continue;
        if (placed == 2) {  // the compare stand-in ran the game's version
            ++compared;
            continue;
        }
        const u8 m = g_mode[i] < 3 ? g_mode[i] : 0;
        modes[m] += modes[m].empty() ? kTable[i].name : std::string(", ") + kTable[i].name;
        ++n[m];
    }
    if (compared) {
        host_log("decomp: Gnm packet writers nothing reads: %zu compared, writing the game's packets (BBHOST_DECOMP_COMPARE=1)", compared);
    } else {
        host_log("decomp: Gnm packet writers nothing reads: %zu write nothing%s%s, %zu nothing and a poisoned address%s%s, %zu the game's "
                 "packet%s%s",
                 n[1], n[1] ? ": " : "", modes[1].c_str(), n[2], n[2] ? ": " : "", modes[2].c_str(), n[0], n[0] ? ": " : "",
                 modes[0].c_str());
    }
    if (g_compare_refills.load()) {
        host_log("decomp: Gnm packet writers compared: %llu calls ran the game's alone (the refill served them)",
                 static_cast<ull>(g_compare_refills.load()));
    }
}

template <std::size_t... I>
std::array<void*, kWriters> compare_table(std::index_sequence<I...>) {
    return {reinterpret_cast<void*>(&compare_writer<I>)...};
}

}  // namespace

static_assert(kWriters == kGnmDeadWriterCount, "decomp/gx/dead_packets.h counts them");

const GnmDeadWriter& gnm_dead_writer(std::size_t i) { return kTable[i < kWriters ? i : 0]; }
bool gnm_dead_writer_placed(std::size_t i) {
    return i < kWriters && decomp_placed(kTable[i].name) != 0;
}

std::uint64_t gnm_dead_keep_mask(const char* list) {
    std::uint64_t mask = 0;
    for (const char* q = list; q && *q;) {
        char* end = nullptr;
        const long v = std::strtol(q, &end, 10);
        if (end == q) break;
        if (v >= 0 && v < 64) mask |= std::uint64_t{1} << v;
        q = *end == ',' ? end + 1 : end;
    }
    return mask;
}

void gnm_dead_writers_configure(GnmDeadMode others, GnmDeadMode context, std::uint64_t keep) {
    for (std::size_t i = 0; i < kWriters; ++i) {
        const bool write = !kTable[i].dead || ((keep >> i) & 1);
        g_mode[i] = static_cast<u8>(write ? GnmDeadMode::kWrite : kTable[i].context ? context : others);
    }
}

GnmDeadMode gnm_dead_writer_mode(std::size_t i) {
    return i < kWriters ? static_cast<GnmDeadMode>(g_mode[i]) : GnmDeadMode::kWrite;
}

// The census of who calls them (BBHOST_CP_WRITERS=1) and the old hook
// measurement (BBHOST_GX_DEAD_WRITERS=1) hook the game's own prologues, so
// with either the game's versions stay.
void decomp_gx_dead_packets_add() {
    for (const char* hooked : {"BBHOST_CP_WRITERS", "BBHOST_GX_DEAD_WRITERS"}) {
        if (const char* e = std::getenv(hooked); e && e[0] == '1') return;
    }
    static void* const kOurs[kWriters] = {
        reinterpret_cast<void*>(&gnm_acquire_mem_range),
        reinterpret_cast<void*>(&dcb_acquire_mem),
        reinterpret_cast<void*>(&gnm_compute_acquire_mem_range),
        reinterpret_cast<void*>(&gnm_compute_acquire_mem),
        reinterpret_cast<void*>(&gnm_set_aa_sample_locations),
        reinterpret_cast<void*>(&gnm_set_aa_sample_locations_and_centroid),
        reinterpret_cast<void*>(&gnm_set_centroid_priority),
        reinterpret_cast<void*>(&gnm_set_aa_config),
        reinterpret_cast<void*>(&gnm_set_depth_stencil_disable),
        reinterpret_cast<void*>(&gnm_set_blend_color),
        reinterpret_cast<void*>(&gnm_set_alpha_to_mask),
        reinterpret_cast<void*>(&gnm_set_spi_tmpring_size),
        reinterpret_cast<void*>(&gnm_set_hardware_screen_offset),
        reinterpret_cast<void*>(&gnm_set_guard_band_clip_adjust),
        reinterpret_cast<void*>(&gnm_set_guard_band_discard_adjust),
        reinterpret_cast<void*>(&gnm_set_instance_step_rate),
        reinterpret_cast<void*>(&gnm_set_ps_input_cntl),
        reinterpret_cast<void*>(&gnm_set_su_vtx_cntl),
        reinterpret_cast<void*>(&gnm_set_su_line_cntl),
        reinterpret_cast<void*>(&gnm_draw_shader_prefetch_dma),
        reinterpret_cast<void*>(&gnm_reserve_flip_packets),
    };
    static const std::array<void*, kWriters> kCompare = compare_table(std::make_index_sequence<kWriters>{});
    for (std::size_t i = 0; i < kWriters; ++i) {
        const GnmDeadWriter& w = kTable[i];
        DecompFunction fn{w.name, "gx", w.bn, w.bytes, w.n, kOurs[i], DecompKind::Leaf};
        fn.original = &g_original[i];
        fn.compare = kCompare[i];
        fn.counts = &g_counts[i];
        fn.report = &report;
        decomp_add(fn);
    }
}
