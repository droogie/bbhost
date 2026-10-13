// BBHOST_GX_TRACE=<first>,<last>: coverage of the engine's GX layer, with
// no behaviour change.
//
// Entry hooks on the GX immediate-context draw and dispatch methods snapshot
// the pending state they are handed. Entry hooks on the Gnm draw and dispatch
// packet emitters record where each packet is written and who called the
// emitter. When the command processor executes a draw or dispatch packet it
// looks the packet up by address, so the summary logged after flip <last> says
// how many executed packets came through GX, through other emitter callers, or
// from code without a hook, and checks the state-block layout against what the
// packets did.
#include "guest_abi.h"
#include "engine/yebis_bind.h"
#include "hle/modules.h"

#include "core/memory.h"
#include "core/portable.h"
#include "core/tally.h"
#include "core/thunk.h"
#include "engine/addr.h"
#include "host/gpu.h"
#include "engine/gx_resources.h"
#include "engine/gx_state.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <immintrin.h>
#include <new>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using ull = unsigned long long;

enum Kind : std::uint8_t {
    // GX immediate context (the hook id is the kind).
    kDraw,
    kDrawIndexed,
    kDrawInstanced,
    kDrawIndexedInstanced,
    kDrawIndirect,
    kDrawIndexedIndirect,
    kCommit,
    kDispatch,
    kDispatchIndirect,
    // Gnm packet emitters.
    kEmitAuto,
    kEmitOffset2,
    kEmitIndirect,
    kEmitIndexIndirect,
    kEmitIndex2,
    kEmitIndex2b,
    kEmitDispatch,
    kEmitDispatchIndirect,
    kEmitDispatchQueue,
    // GX UpdateSubresource (how constant buffers are written).
    kUpdateSubresource,
    // GX buffer-copy passes (copy tokens, BBHOST_GX_NATIVE_COPIES).
    kCopyBuffers,
    kCopyBuffer,
    // YEBIS's own D3D11-shaped context draw (YEBIS tokens).
    kYebisDraw,
    // GX's deferred colour-clear pass, timed to size up the clear family.
    kClearColour,
    // The fill primitive and the per-slice colour clear, as fill tokens.
    kFill,
    kClearRtv,
    kUploadTexture,
    kCopyTexture,
    // The Scaleform HAL's draws (0x73e670 indexed,
    // 0x73e6b0 instanced, 0x73e640 non-indexed) and the Gnm wrapper's
    // index-buffer bind (0x14752c0), which is the index buffer the HAL bound.
    kScaleformDraw,
    kScaleformDrawInstanced,
    kScaleformDrawAuto,
    kSetIndexBuffer,
    // The wrapper's user-data write (0x14764a0: cb, stage, first SGPR, dwords,
    // count), which is how a resource the layout routes to the SGPRs directly
    // reaches them when it is appended.
    kSetUserData,
    // The wrapper's other user-data writers and its
    // state setters (the host keeps the wrapper's context state per command
    // buffer), the depth decompress pass and the depth-stencil clear pass.
    kUd4a,
    kUd8,
    kUd4b,
    kUd1,
    kUdPtr,
    kSetVs,
    kSetPs,
    kSetRt,
    kSetDsv,
    kSetViewport,
    kSetScreenScissor,
    kSetVte,
    kSetScMode,
    kSetBlend,
    kSetTargetMask,
    kSetDsControl,
    kSetStencilControl,
    kSetStencilRef,
    kSetPrimType,
    kSetNumInstances,
    kSetIndexSize,
    kSetClip,
    kSetPrimSetup,
    kSetStages,
    kSetColorControl,
    kSetRenderControl,
    kSetStencilClear,
    kSetVportScissor,
    kSetIndxOffset,
    // The compute program setter and SET_BASE (indirect arguments).
    kSetCs,
    kSetBase,
    // The compute-typed user-data writers (COMPUTE_USER_DATA, SET_SH_REG with
    // the shader-type bit): N dwords, 8, 4, 1 and a pointer.
    kCsUdN,
    kCsUd8,
    kCsUd4,
    kCsUd1,
    kCsUdPtr,
    kDepthDecompress,
    kDepthStencilClearPass,
    // GX shader creators (per-shader compiles, step 4): the Sony shader container.
    kCreateShader156,    // 0x2566d00: containers of type 1 (vertex), 5 and 6
    kCreateShader15,     // 0x2566f20: types 1 and 5
    kCreatePixelShader,  // 0x25672d0: type 2
    kCreateComputeShader,  // 0x25673e0: type 4
    kKinds
};
const char* const kKindName[kKinds] = {
    "Draw", "DrawIndexed", "DrawInstanced", "DrawIndexedInstanced", "DrawIndirect", "DrawIndexedIndirect", "CommitOnly",
    "Dispatch", "DispatchIndirect", "emit DRAW_INDEX_AUTO", "emit DRAW_INDEX_OFFSET_2", "emit DRAW_INDIRECT",
    "emit DRAW_INDEX_INDIRECT", "emit DRAW_INDEX_2", "emit DRAW_INDEX_2 (b)", "emit DISPATCH_DIRECT",
    "emit DISPATCH_INDIRECT", "emit DISPATCH_DIRECT (compute queue)", "UpdateSubresource", "copy buffers", "copy buffer", "YEBIS draw", "deferred colour clear",
    "fill", "clear render target", "texture upload", "texture copy",
    "Scaleform draw", "Scaleform instanced draw", "Scaleform non-indexed draw", "set index buffer", "set user data",
    "set user data (4)", "set user data (8)", "set user data (4b)", "set user data (1)", "set user data pointer", "set VS", "set PS",
    "set render target", "set depth target", "set viewport", "set screen scissor", "set VTE", "set SC mode", "set blend", "set target mask",
    "set depth-stencil control", "set stencil control", "set stencil ref", "set primitive type", "set instances", "set index size",
    "set clip control", "set primitive setup", "set shader stages", "set colour control", "set render control", "set stencil clear",
    "set viewport scissor", "set index offset", "set CS", "set base", "set CS user data (n)", "set CS user data (8)", "set CS user data (4)",
    "set CS user data (1)", "set CS user data pointer", "depth decompress pass", "depth-stencil clear pass",
    "create shader (types 1, 5, 6)", "create shader (types 1, 5)", "create pixel shader", "create compute shader"};
const char* const kStageName[6] = {"VS", "HS", "DS", "GS", "PS", "CS"};
bool is_dispatch(int k) { return k == kDispatch || k == kDispatchIndirect; }

// Binary Ninja addresses (guest VA at the preferred slide) and the prologue
// each hook displaces: whole, position-independent instructions, >= 14 bytes.
struct Hook {
    std::uint64_t va;
    std::uint8_t n;
    std::uint8_t bytes[20];
};
#define BB_PUSH6 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53
#define BB_PUSH4_ECX 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xce
#define BB_PUSH2_ESI 0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08
#define BB_YEBIS_DRAW 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xd6, 0x41, 0x89, 0xf7
const Hook kHooks[kKinds] = {
    {0x25696c0, 14, {BB_PUSH6, 0x50}},                    // push rbp .. push rbx; push rax
    {0x2569780, 14, {BB_PUSH6, 0x50}},
    {0x2569840, 14, {BB_PUSH6, 0x50}},
    {0x2569900, 17, {BB_PUSH6, 0x48, 0x83, 0xec, 0x18}},  // ...; sub rsp, 0x18
    {0x25699d0, 14, {BB_PUSH6, 0x50}},
    {0x2569ab0, 14, {BB_PUSH6, 0x50}},
    {0x2569ba0, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x49, 0x89, 0xf6}},
    {0x25695c0, 14, {BB_PUSH6, 0x50}},
    {0x2569640, 14, {BB_PUSH6, 0x50}},
    {0x14741b0, 17, {BB_PUSH2_ESI}},  // ...; mov r14d, esi; mov rbx, rdi; mov rcx, [rbx+8]
    {0x1474420, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x41, 0x89, 0xd6, 0x41, 0x89, 0xf7}},
    {0x14744d0, 14, {BB_PUSH6, 0x50}},
    {0x14746f0, 14, {BB_PUSH6, 0x50}},
    {0x14740a0, 14, {BB_PUSH6, 0x50}},
    {0x1474360, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xd7, 0x41, 0x89, 0xf6}},
    {0x1474c50, 14, {BB_PUSH4_ECX}},  // ...; mov r14d, ecx
    {0x1474de0, 17, {BB_PUSH2_ESI}},
    {0x148a350, 14, {BB_PUSH4_ECX}},
    {0x2568a20, 17, {BB_PUSH6, 0x48, 0x83, 0xec, 0x28}},  // ...; sub rsp, 0x28
    {0x2ab2810, 17, {BB_PUSH6, 0x48, 0x83, 0xec, 0x78}},  // ...; sub rsp, 0x78
    {0x2ab2670, 17, {BB_PUSH6, 0x48, 0x83, 0xec, 0x48}},  // ...; sub rsp, 0x48
    {0x15fbc40, 16, {BB_YEBIS_DRAW}},                     // YEBIS draw wrapper
    {0x2ab5a70, 17, {BB_PUSH6, 0x48, 0x83, 0xec, 0x78}},  // deferred colour clear
    {0x2ab5380, 17, {BB_PUSH6, 0x48, 0x83, 0xec, 0x58}},  // the fill primitive
    {0x2ab4ef0, 20, {BB_PUSH6, 0x48, 0x81, 0xec, 0xb8, 0x00, 0x00, 0x00}},  // the colour clear (sub rsp, 0xb8)
    {0x2ab35e0, 20, {BB_PUSH6, 0x48, 0x81, 0xec, 0xa8, 0x01, 0x00, 0x00}},  // the texture upload (sub rsp, 0x1a8)
    {0x2ab2f40, 20, {BB_PUSH6, 0x48, 0x81, 0xec, 0xc8, 0x00, 0x00, 0x00}},  // the texture copy (sub rsp, 0xc8)
    // push rbp; mov rbp, rsp; push r15; push r14; push rbx; push rax; mov r15, r8; mov r14d, esi
    {0x73e670, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x4d, 0x89, 0xc7, 0x41, 0x89, 0xf6}},
    {0x73e6b0, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x4d, 0x89, 0xc7, 0x41, 0x89, 0xf6}},
    // push rbp; mov rbp, rsp; push r14; push rbx; mov r14d, esi; mov rbx, [rdi+0x2ca30]
    {0x73e640, 17, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x8b, 0x9f, 0x30, 0xca, 0x02, 0x00}},
    // push rbp; mov rbp, rsp; push r14; push rbx; mov r14, rsi; mov rbx, rdi; mov rcx, [rbx+8]
    {0x14752c0, 17, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}},
    {0x14764a0, 14, {BB_PUSH6, 0x50}},
    // push rbp; mov rbp, rsp; push r15; push r14; push r12; push rbx; mov r14, rcx (r14d, ecx)
    {0x14761a0, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x49, 0x89, 0xce}},
    {0x1476240, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x49, 0x89, 0xce}},
    {0x14762f0, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x49, 0x89, 0xce}},
    {0x1476420, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xce}},
    {0x1476390, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x49, 0x89, 0xce}},
    {0x1475480, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x41, 0x89, 0xd6, 0x49, 0x89, 0xf7}},
    {0x14753c0, 17, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x43, 0x08}},
    {0x14738b0, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x48, 0x89, 0xd3, 0x41, 0x89, 0xf7}},
    {0x14739b0, 18, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89, 0xf3, 0x49, 0x89, 0xfe, 0xbe, 0x06, 0x00, 0x00, 0x00}},
    {0x14765e0, 15, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x10}},
    {0x1476790, 14, {BB_PUSH6, 0x50}},
    {0x1476980, 17, {BB_PUSH2_ESI}},
    {0x1477150, 15, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x89, 0xd3, 0x41, 0x89, 0xf7}},
    {0x1473090, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x41, 0x89, 0xd6, 0x41, 0x89, 0xf7}},
    {0x1472fd0, 17, {BB_PUSH2_ESI}},
    {0x14733a0, 17, {BB_PUSH2_ESI}},
    {0x1473400, 17, {BB_PUSH2_ESI}},
    {0x1473190, 17, {BB_PUSH2_ESI}},
    {0x1473fe0, 17, {BB_PUSH2_ESI}},
    {0x1475370, 17, {BB_PUSH2_ESI}},
    {0x1475270, 17, {BB_PUSH2_ESI}},
    {0x1476b30, 17, {BB_PUSH2_ESI}},
    {0x1476e00, 17, {BB_PUSH2_ESI}},
    {0x1476030, 17, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x43, 0x08}},
    {0x1473320, 15, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x41, 0x89, 0xd7, 0x89, 0xf3}},
    {0x14769e0, 17, {BB_PUSH2_ESI}},
    {0x14735f0, 17, {BB_PUSH2_ESI}},
    {0x14766d0, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50}},
    {0x1474040, 17, {BB_PUSH2_ESI}},
    // push rbp; mov rbp, rsp; push r14; push rbx; mov r14, rsi; mov rbx, rdi; mov rax/rcx, [rbx+8]
    {0x1475840, 17, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x43, 0x08}},
    {0x14742f0, 17, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}},
    {0x148ad00, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x89, 0xcb, 0x49, 0x89, 0xd7}},
    {0x148ac00, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xd6, 0x41, 0x89, 0xf7}},
    {0x148aad0, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xd6, 0x41, 0x89, 0xf7}},
    {0x148ada0, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x41, 0x89, 0xd6, 0x41, 0x89, 0xf7}},
    {0x148ac90, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xd6, 0x41, 0x89, 0xf7}},
    {0x2ab6810, 20, {BB_PUSH6, 0x48, 0x81, 0xec, 0x88, 0x00, 0x00, 0x00}},  // the depth decompress pass (sub rsp, 0x88)
    {0x2ab5f50, 20, {BB_PUSH6, 0x48, 0x81, 0xec, 0xe8, 0x00, 0x00, 0x00}},  // the depth-stencil clear pass (sub rsp, 0xe8)
    // push rbp; mov rbp, rsp; push r15; push r14; push rbx; sub rsp, 0x138 / 0x128
    {0x2566d00, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x48, 0x81, 0xec, 0x38, 0x01, 0x00, 0x00}},
    {0x2566f20, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x48, 0x81, 0xec, 0x28, 0x01, 0x00, 0x00}},
    {0x25672d0, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x48, 0x81, 0xec, 0x28, 0x01, 0x00, 0x00}},
    {0x25673e0, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x48, 0x81, 0xec, 0x28, 0x01, 0x00, 0x00}},
};
#undef BB_PUSH6
#undef BB_PUSH4_ECX
#undef BB_PUSH2_ESI
constexpr std::uint64_t kGxImmediateLo = 0x25695c0, kGxImmediateHi = 0x2569c30;
constexpr std::uint64_t kReplayLo = 0x2aaad80, kReplayHi = 0x2aab800;

std::uint64_t g_slide = kPreferredGuestSlide;
std::uint64_t g_image_size = 0;
std::uint64_t g_first = 1200, g_last = 1500;
std::atomic<bool> g_active{false};
std::atomic<bool> g_printed{false};
std::atomic<std::int64_t> g_draw_scan_budget{2000};
std::atomic<std::int64_t> g_dispatch_scan_budget{1000};
bool g_trace = false;  // BBHOST_GX_TRACE: coverage statistics
int g_backend = 0;     // BBHOST_GX_BACKEND: 1 draws from GX inputs, 2 compares them
// BBHOST_GX_NATIVE=1 (shadow): with BBHOST_GX_BACKEND=1, each
// immediate GX draw without HS/DS/GS writes a host-draw token (a NOP {magic,
// id}) at the start of the method, which then runs as before. The CP
// snapshots, at the token, the register fields render mode still takes from
// the register file, and compares them at the draw packet: what changed is
// what the draw's own packets write, which a draw without packets must take
// from GX.
int g_native = 0;
// Token ids: every GX worker takes one a draw, so the counter has its line to
// itself; the per-draw statistics are Tallies (core/tally.h).
alignas(64) std::atomic<std::uint64_t> g_native_next{0};
// Token ids, handed to each GX thread in blocks: the shared counter is touched
// once a block instead of at every draw, and a thread knows its next id - so
// it can ask for that slot's lines while the game does other work
// (prefetch_next_slot). Nothing orders by id: the ring is indexed by it.
constexpr std::uint64_t kIdBlock = 64;
// A line asked for with the intent to write it (PREFETCHW; a no-op on CPUs
// without it), so the store that follows finds it already this core's.
inline void prefetch_for_write(const void* p) { asm volatile("prefetchw %0" : : "m"(*static_cast<const char*>(p))); }
thread_local std::uint64_t t_id_next = 0, t_id_end = 0;  // [next, end) of this thread's block
std::uint64_t next_native_id() {
    if (t_id_next == t_id_end) {
        t_id_next = g_native_next.fetch_add(kIdBlock, std::memory_order_relaxed) + 1;
        t_id_end = t_id_next + kIdBlock;
    }
    return t_id_next++;
}
alignas(64) std::atomic<std::uint64_t> g_native_refills{0}, g_native_first_chunks{0}, g_native_failed{0}, g_native_seen{0}, g_native_matched{0},
    g_native_unmatched{0};
Tally<> g_native_tokens;
enum NativeField {
    kNfPsInput, kNfRenderControl, kNfDepthClear, kNfStencilClear, kNfGenericScissor, kNfVportScissor, kNfModeCntl,
    kNfUserNamed, kNfUserUnnamed, kNativeFields
};
std::atomic<std::uint64_t> g_native_changed[kNativeFields] = {};
std::atomic<std::uint64_t> g_native_unnamed_slot[2][16] = {};  // VS / PS user-data slots no descriptor writes, changed
// BBHOST_GX_NATIVE=2: draws the host ran from their tokens, draws that did not
// qualify, and draws whose token could not be written after the flushes ran.
std::atomic<std::uint64_t> g_native_drawn{0}, g_native_late{0};
Tally<> g_native_ineligible;
// Dwords each guest call of a native draw writes into its context's Gnm command
// buffer (the write pointer moving within a chunk), and calls that moved to
// another chunk, which are not counted.
enum NativeCall { kNcOmFlush, kNcIaFlush, kNcRsFlush, kNcBookkeeping, kNcOmCommit, kNcPostDraw, kNcBookkeepingAfter, kNativeCalls };
Tally<kNativeCalls> g_native_call_dwords;
Tally<> g_native_call_switches;
// BBHOST_GX_NATIVE_FLUSHES=1 (checks): native draws run the method's
// output-merger, input-assembly and rasterizer flushes as it does. By default
// they skip the input-assembly and rasterizer flushes, and run the
// output-merger flush only when it may run a deferred clear. Apart from those
// clears the three only write register state against their own caches
// (0x2abba30: primitive type, INDEX_BASE, INDEX_TYPE; 0x2ad3370: rasterizer
// state, viewports, guard band, scissors; 0x2ad2230: blend state, blend
// factor, depth-stencil state, stencil reference, depth bounds), and a native
// draw renders from the GX objects instead. With neither the caches nor the CP
// register file changed, the next packet-path draw's flushes still write
// whatever differs.
const bool g_native_all_flushes = [] {
    const char* e = std::getenv("BBHOST_GX_NATIVE_FLUSHES");
    return e && e[0] == '1';
}();
// What a *tessellated* draw's own packets set, which is what a
// native one would have to carry from its GX objects instead. These draws are
// the whole of what is left of the packet path (98.1% of the draws never
// offered a token), and the state they need is not in GpuDrawInputs: the
// stages enabled, the patch and domain configuration, the clamps, and the LS
// and HS programs with their user data. Shadowed the way the rest was before
// it moved: record at the token, compare at the draw packet, count what
// differs.
struct NativeTess {
    std::uint32_t stages, ls_hs_config, tf_param, hos_max, hos_min;
    std::uint32_t ls_pgm[4], hs_pgm[4];
    std::uint32_t ls_user[16], hs_user[16];
    std::uint32_t tess_cb[2];  // the V# of the tessellation constants, in VS user data 8-9
};
enum TessField { kTfStages, kTfConfig, kTfParam, kTfHos, kTfLsPgm, kTfHsPgm, kTfLsUser, kTfHsUser, kTfConstants, kTessFields };
const char* const kTessFieldName[kTessFields] = {"VGT_SHADER_STAGES_EN", "VGT_LS_HS_CONFIG", "VGT_TF_PARAM",
                                                 "VGT_HOS_MAX/MIN_TESS_LEVEL", "the LS program", "the HS program",
                                                 "the LS user data", "the HS user data",
                                                 "the tessellation constants' V#"};
std::atomic<std::uint64_t> g_tess_shadowed{0}, g_tess_matched{0}, g_tess_unmatched{0};
std::atomic<std::uint64_t> g_tess_changed[kTessFields] = {};
enum BuiltField { kBfStages, kBfConfig, kBfTf, kBfHos, kBfLsPgm, kBfHsPgm, kBfVsPgm, kBfCount, kBuiltFields };
const char* const kBuiltFieldName[kBuiltFields] = {"VGT_SHADER_STAGES_EN", "the control-point counts", "VGT_TF_PARAM",
                                                   "the clamps", "the LS program", "the HS program",
                                                   "the domain shader's program", "the patch count"};
std::atomic<std::uint64_t> g_tess_built[kBuiltFields] = {}, g_tess_built_bad[kBuiltFields] = {};
std::atomic<std::uint64_t> g_tess_have_inputs{0}, g_tess_in_ok{0};
thread_local std::unordered_map<std::uint64_t, NativeTess> t_native_tess;
// Native draws that ran the output-merger flush.
Tally<> g_native_om_flushes;
// BBHOST_GX_NATIVE_OM: what a native draw does about the
// output-merger commit (0x2ad2790) and flush (0x2ad2230).
//   2 (default) neither runs. The commit emits target registers against its
//     record of bound targets, which the renderer takes from the GX objects
//     instead; the flush's state packets are likewise against caches the
//     renderer does not read. What the pair was kept for is the
//     deferred pass the flush runs when a recorded target the state no longer
//     binds has +0x3b & 0x8b (colour) or & 8 (depth) set on its texture:
//     0x2ab5a70 draws the target with CB_COLOR_CONTROL mode 2 (eliminate fast
//     clear) and 0x2ab6810 with DB_RENDER_CONTROL bits 5 and 6 (stencil and
//     depth compress disable) - the hardware's CMASK and HTILE decompression
//     before the target is sampled, which has nothing to decompress here (the
//     images are their own storage, and skipping the colour pass outright
//     left frames unchanged). The guest's record then describes what
//     the register file holds, as it did without the commit, so a packet-path
//     draw's commit and flush still emit what differs.
//   1 the guest's: the commit on every native draw (1.04 dwords) and the
//     flush when om_flush_may_clear says it may run a pass (2.1 dwords a draw
//     on average) - the state before draws became tokens, for A/B.
//   0 the flush for passes, no commit (the 7.2 measurement).
const int g_native_om = [] {
    const char* e = std::getenv("BBHOST_GX_NATIVE_OM");
    return e && e[0] ? std::atoi(e) : 2;
}();
// The GX utility passes and the once-a-flip present
// draw as tokens.
//
// What is left of the packet path after Scaleform is GX's own utility code
// and the flip: the two deferred decompress passes (0x2ab6810 depth,
// 0x2ab5a70 colour), the depth-stencil clear pass (0x2ab5f50), the gamma
// pass (0x2ab6cb0) and the present draw in the flip function (0x25d5f80,
// through the flip's own Gnm context data_5ac3cb8). None goes through a GX
// draw method: each sets its state through the Gnm wrapper's setters and
// emits its draw packet itself.
//
// The decompress passes have nothing to do on the host (the images are
// their own storage; skipping the colour pass left frames unchanged)
// and are skipped. The clear pass fills HTILE with the clear word
// and draws a quad that writes depth and stencil: the fill token carries
// the depth and stencil, which is what the host's HTILE clear applies, and
// the draw is skipped. The gamma pass and the present draw are draws whose
// inputs the wrapper's setters were given: the host keeps the wrapper's
// context state per command buffer as the setters are called - the
// programs (0x1475480, 0x14753c0), the user data (0x14761a0, 0x1476240,
// 0x14762f0, 0x1476420, 0x1476390, 0x14764a0), the targets (0x14738b0,
// 0x14739b0), the viewport (0x14765e0), the scissors, the blend,
// depth-stencil, primitive and rasterizer words - and at the draw's
// emitter writes a token built from it instead of the packet.
// BBHOST_GX_NATIVE_UTILITY: 0 packet path, 1 shadow (token and packet, the
// CP compares them), 2 native (the default with native GX draws).
// BBHOST_GX_NATIVE_PASSES (bits): 1 the depth decompress pass is skipped, 2
// the colour eliminate pass is skipped, 4 the clear pass is a fill token;
// default 7, 0 leaves the passes as they are.
const int g_native_utility = [] {
    const char* e = std::getenv("BBHOST_GX_NATIVE_UTILITY");
    return e && e[0] ? std::atoi(e) : 2;
}();
// BBHOST_GX_YEBIS_FULL: a YEBIS token (BBHOST_GX_NATIVE_YEBIS=5) completed
// from the wrapper's context state - targets, blend, depth-stencil, viewport,
// scissors, primitive and rasterizer words - so it reads nothing from the
// register file: 0 off (partial, the register file's), 1 shadow (partial,
// the wrapper's state compared with the register file at the token), 2
// (default) complete.
const int g_yebis_full = [] {
    const char* e = std::getenv("BBHOST_GX_YEBIS_FULL");
    return e && e[0] ? std::atoi(e) : 2;
}();
bool wrapper_complete_inputs(std::uint64_t cb, GpuDrawInputs& in, std::uint32_t* index_type);
// BBHOST_GX_NATIVE_DISPATCH: compute dispatches as tokens
// from the wrapper's context state, at the dispatch emitters (0x1474c50,
// 0x148a350 direct; 0x1474de0 indirect through SET_BASE): 0 packet path
// (the default), 1 shadow (token and packet, compared at the packet), 2
// native. Default 0: a dispatch's compute user data is not all written on the thread
// and buffer that dispatches - the resource builder mirrors some of it to the
// asynchronous compute queue's buffer and the constant engine writes some of
// it from the CCB, which the command processor interleaves by its counters.
// 8% of dispatches a walk carried stale user data in the shadow, so compute
// keeps its packets and the SH register file.
const int g_native_dispatch = [] {
    const char* e = std::getenv("BBHOST_GX_NATIVE_DISPATCH");
    return e && e[0] ? std::atoi(e) : 0;
}();
const int g_native_passes = [] {
    const char* e = std::getenv("BBHOST_GX_NATIVE_PASSES");
    return e && e[0] ? std::atoi(e) : 7;
}();
constexpr std::uint64_t kGammaPassDraw = 0x2ab70ff, kPresentDraw = 0x25d84a2;
constexpr std::uint64_t kYebisDrawSite = 0x15fbc40;  // the YEBIS draw wrapper, for the compare's logs
constexpr std::uint64_t kHtileSizeGetter = 0x1472630;

std::atomic<std::uint64_t> g_wd_taken[3] = {}, g_wd_refused{0}, g_wd_skipped_depth{0}, g_wd_skipped_colour{0}, g_wd_clear_tokens{0},
    g_wd_clear_refused{0}, g_wd_clear_no_htile{0}, g_wd_clear_nothing{0};
std::map<std::uint64_t, std::uint64_t> g_wd_other_callers;  // under g_sf_mu: wrapper draws from callers no token is built for
std::map<std::uint64_t, std::uint64_t> g_wd_taken_callers;  // under g_sf_mu: other callers whose draws took a token


// Why a GX draw never reaches native_eligible() - the gate below it
// returns before the tally there is touched, so these are the draws that are
// not even asked. They are what is left of the packet path, and the register
// file the output-merger commit keeps is kept for them.
enum NativeSkip { kSkipKind, kSkipTess, kSkipGeometry, kSkipInputs, kNativeSkips };
std::atomic<std::uint64_t> g_gs_object_draws{0};
Tally<kNativeSkips> g_native_skipped;
const char* const kNativeSkipName[kNativeSkips] = {"indirect or an unsupported kind", "a hull or domain shader",
                                                   "a geometry shader", "inputs that could not be built"};
// BBHOST_GX_NATIVE_REGS=1 (checks): decode what each guest call of a native
// draw writes, counting SET_CONTEXT_REG writes by register and other type-3
// packets by opcode (0xff for a packet that is not type 3).
const bool g_native_regs = [] {
    const char* e = std::getenv("BBHOST_GX_NATIVE_REGS");
    return e && e[0] == '1';
}();
std::atomic<std::uint64_t> g_native_call_regs[kNativeCalls][0x400] = {};
std::atomic<std::uint64_t> g_native_call_ops[kNativeCalls][0x100] = {};

// BBHOST_GNM_SOURCES=1 (checks): cut each Gnm command buffer at its draw and
// dispatch packets and add each piece's dwords to the source of the packet
// that ends it: the emitter's caller (Binary Ninja address), or a native draw
// for its token. A piece is everything written on that command buffer since
// the previous draw, dispatch or token packet began, the packet itself going
// to the next piece. Pieces that cross chunks are counted, not measured.
// `gnm sources` lines each window.
const bool g_gnm_sources = [] {
    const char* e = std::getenv("BBHOST_GNM_SOURCES");
    return e && e[0] == '1';
}();
constexpr std::uint64_t kSourceNative = 1;
constexpr std::uint64_t kSourceCopy = 2;
constexpr std::uint64_t kSourceScaleform = 3;
constexpr std::uint64_t kSourceWrapper = 4;
struct SourceCount {
    std::uint64_t pieces = 0;
    std::uint64_t dwords = 0;
};
std::mutex g_sources_mu;
std::unordered_map<std::uint64_t, SourceCount> g_sources;
std::uint64_t g_source_jumps = 0;
std::uint64_t g_source_largest = 0;
// Command buffer -> write pointer where its current piece began, and the
// chunk's end then. Gnmx allocates from the back of a chunk, so the end only
// moves down within one; a piece counts when the write pointer has not passed
// the end recorded at its start (chunks lie next to each other). Shared, not
// per thread: a command buffer need not stay on one thread.
struct PieceStart {
    std::uint64_t wp = 0;
    std::uint64_t end = 0;
};
std::unordered_map<std::uint64_t, PieceStart> g_piece_start;

void note_source(std::uint64_t cb, std::uint64_t wp, std::uint64_t end, std::uint64_t source) {
    std::lock_guard<std::mutex> lk(g_sources_mu);
    PieceStart& start = g_piece_start[cb];
    const bool measured = start.wp && wp >= start.wp && wp < start.end && wp - start.wp < (1u << 20);
    const std::uint64_t dwords = measured ? (wp - start.wp) / 4 : 0;
    start.wp = wp;
    start.end = end;
    if (!measured) {
        ++g_source_jumps;
        return;
    }
    SourceCount& c = g_sources[source];
    ++c.pieces;
    c.dwords += dwords;
    g_source_largest = std::max(g_source_largest, dwords);
}

// BBHOST_GX_COST=1: time the GX hooks and the CP
// hand-off per draw (`gx-cost:` lines) and hash sampled constant buffers at
// the call so the renderer can count how many change before it reads them.
const bool g_gx_cost_enabled = [] {
    const char* e = std::getenv("BBHOST_GX_COST");
    return e && e[0] == '1';
}();
// kCostNative: a native draw's work inside the call hook (BBHOST_GX_NATIVE=2);
// kCostToken: the command processor turning its token into draw inputs.
// The rest split kCostRecords, kCostNative and kCostToken into their parts.
enum GxCost { kCostCall, kCostEmit, kCostRecords, kCostHandoff, kCostNative, kCostToken, kCostRecAlloc, kCostRecShader,
              kCostRecTex, kCostRecSmp, kCostRecCb, kCostRecHash, kCostNativeRecords, kCostNativeGuest, kCostNativeStore,
              kCostTokenLookup, kGxCosts };
std::atomic<std::uint64_t> g_gx_cost_ns[kGxCosts] = {}, g_gx_cost_n[kGxCosts] = {};
// dynamic_buffer_address under BBHOST_GX_COST: lookups through the key's table, and page scans with the pages they read.
std::atomic<std::uint64_t> g_dynbuf_table{0}, g_dynbuf_scans{0}, g_dynbuf_pages{0};
// Records sets the free list could not supply (new_records).
std::atomic<std::uint64_t> g_records_new{0};
// GX UpdateSubresource calls by kind (note_update). Only counted, so the hook
// is armed only for the trace, BBHOST_GX_COST and BBHOST_WATCH_UPDATE.
Tally<> g_updates_static, g_updates_dynamic, g_updates_other, g_update_static_bytes;
// The texture updates, which are the ones that stage into a
// scratch buffer and dispatch a compute shader to write the image
// (sub_2ab35e0). Counted here so the hook's population can be checked against
// the dispatch packets attributed to that caller before anything is taken over.
Tally<> g_updates_tex, g_update_tex_bytes;
std::mutex g_update_dst_mu;
std::map<std::uint64_t, std::uint64_t> g_update_dst;  // destination address -> calls
const bool g_upload_census = [] {
    const char* e = std::getenv("BBHOST_GX_UPLOAD");
    return e && (e[0] == '1' || e[0] == '2');
}();
std::atomic<std::uint64_t> g_updates_watched{0};
const std::uint64_t g_watch_update = [] {
    const char* e = std::getenv("BBHOST_WATCH_UPDATE");
    return e ? std::strtoull(e, nullptr, 0) : 0ull;
}();
struct GxCostTimer {
    GxCost what;
    std::chrono::steady_clock::time_point start;
    explicit GxCostTimer(GxCost w) : what(w), start(g_gx_cost_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    ~GxCostTimer() {
        if (!g_gx_cost_enabled) return;
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        g_gx_cost_ns[what].fetch_add(static_cast<std::uint64_t>(ns), std::memory_order_relaxed);
        g_gx_cost_n[what].fetch_add(1, std::memory_order_relaxed);
    }
};
// Consecutive sections of one function: to(k) books the time since the last
// stamp (or construction) to cost k.
struct GxCostStamp {
    std::chrono::steady_clock::time_point t = g_gx_cost_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    void to(GxCost what) {
        if (!g_gx_cost_enabled) return;
        const auto now = std::chrono::steady_clock::now();
        g_gx_cost_ns[what].fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count()),
                                     std::memory_order_relaxed);
        g_gx_cost_n[what].fetch_add(1, std::memory_order_relaxed);
        t = now;
    }
};

// BBHOST_GX_REPEATS=1 (a diagnostic; its hashing costs the GX threads ~2 us
// a draw): consecutive draws on one context that repeat the previous draw's
// records and inputs (native_draw_call, records_signature), and each stage's
// parts. A soak on 2026-09-27: records 3.6%, inputs 31.4%; per stage the
// shader 56%, views 73%, samplers 88%, constant buffers 23%.
const bool g_gx_repeats = [] {
    const char* e = std::getenv("BBHOST_GX_REPEATS");
    return e && e[0] == '1';
}();
std::atomic<std::uint64_t> g_memo_draws{0}, g_memo_same_records{0}, g_memo_same_inputs{0}, g_memo_same_both{0};
thread_local std::uint64_t t_memo_ctx = 0, t_memo_records = 0, t_memo_inputs = 0;
// Per stage (set 0, pixel): the shader object, the views and the samplers the
// shader reads, and the constant buffers' records, each as the previous draw
// on the context had them.
std::atomic<std::uint64_t> g_memo_stages{0}, g_memo_same_shader{0}, g_memo_same_views{0}, g_memo_same_samplers{0}, g_memo_same_cbs{0};
thread_local std::uint64_t t_memo_part[2][4] = {};
std::uint64_t memo_mix(std::uint64_t h, const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

void report_gx_costs() {
    static std::atomic<std::uint64_t> next{200000};
    const std::uint64_t n = g_gx_cost_n[kCostHandoff].load(std::memory_order_relaxed) + g_gx_cost_n[kCostToken].load(std::memory_order_relaxed);
    if (!g_gx_cost_enabled || n < next.load(std::memory_order_relaxed)) return;
    next.store(n + 200000, std::memory_order_relaxed);
    const auto per = [](int k) {
        const std::uint64_t c = g_gx_cost_n[k].load(std::memory_order_relaxed);
        return static_cast<unsigned long long>(c ? g_gx_cost_ns[k].load(std::memory_order_relaxed) / c : 0);
    };
    const auto calls = [](int k) { return static_cast<unsigned long long>(g_gx_cost_n[k].load(std::memory_order_relaxed)); };
    host_log("gx-cost: ns each: draw-call snapshot %llu (%llu calls), of which native draw work %llu (%llu), emitter hook %llu "
             "(%llu), of which records %llu (%llu), CP hand-off %llu (%llu), CP token %llu (%llu)",
             per(kCostCall), calls(kCostCall), per(kCostNative), calls(kCostNative), per(kCostEmit), calls(kCostEmit),
             per(kCostRecords), calls(kCostRecords), per(kCostHandoff), calls(kCostHandoff), per(kCostToken), calls(kCostToken));
    host_log("gx-cost: records split, ns each: allocation %llu (%llu, %llu newly allocated), per stage: shader %llu, textures %llu, samplers %llu, "
             "constant buffers %llu, content hashing %llu (%llu stages); native draw split: records %llu, guest calls %llu per group "
             "(two a draw), token and store %llu; CP token lookup %llu; dynamic buffer lookups: table %llu, scans %llu (%llu pages)",
             per(kCostRecAlloc), calls(kCostRecAlloc), static_cast<unsigned long long>(g_records_new.load()), per(kCostRecShader), per(kCostRecTex), per(kCostRecSmp), per(kCostRecCb),
             per(kCostRecHash), calls(kCostRecHash), per(kCostNativeRecords), per(kCostNativeGuest), per(kCostNativeStore),
             per(kCostTokenLookup), static_cast<unsigned long long>(g_dynbuf_table.load()),
             static_cast<unsigned long long>(g_dynbuf_scans.load()), static_cast<unsigned long long>(g_dynbuf_pages.load()));
    if (g_gx_repeats) {
        const double d = static_cast<double>(std::max<std::uint64_t>(1, g_memo_draws.load(std::memory_order_relaxed)));
        const double st = static_cast<double>(std::max<std::uint64_t>(1, g_memo_stages.load(std::memory_order_relaxed)));
        host_log("gx-cost: consecutive stages on a context: shader the previous draw's %.1f%%, views %.1f%%, samplers %.1f%%, constant "
                 "buffers %.1f%% (of %llu)",
                 100.0 * static_cast<double>(g_memo_same_shader.load(std::memory_order_relaxed)) / st,
                 100.0 * static_cast<double>(g_memo_same_views.load(std::memory_order_relaxed)) / st,
                 100.0 * static_cast<double>(g_memo_same_samplers.load(std::memory_order_relaxed)) / st,
                 100.0 * static_cast<double>(g_memo_same_cbs.load(std::memory_order_relaxed)) / st,
                 static_cast<unsigned long long>(g_memo_stages.load(std::memory_order_relaxed)));
        host_log("gx-cost: consecutive draws on a context: records the previous draw's %.1f%%, inputs %.1f%%, both %.1f%% (of %llu)",
                 100.0 * static_cast<double>(g_memo_same_records.load(std::memory_order_relaxed)) / d,
                 100.0 * static_cast<double>(g_memo_same_inputs.load(std::memory_order_relaxed)) / d,
                 100.0 * static_cast<double>(g_memo_same_both.load(std::memory_order_relaxed)) / d,
                 static_cast<unsigned long long>(g_memo_draws.load(std::memory_order_relaxed)));
    }
    host_log("gx-cost: UpdateSubresource calls: static buffers %llu (%llu KiB), dynamic buffers %llu, other resources %llu; "
             "of the watched buffer %llu",
             static_cast<ull>(g_updates_static.load()), static_cast<ull>(g_update_static_bytes.load() >> 10),
             static_cast<ull>(g_updates_dynamic.load()), static_cast<ull>(g_updates_other.load()),
             static_cast<ull>(g_updates_watched.load()));
    {
        // Printed here rather than in the trace summary: that one runs during
        // the run, so it reported zero while the exit report counted 1,495.
        std::lock_guard<std::mutex> lk(g_update_dst_mu);
        host_log("gx:   of those, texture updates: %llu calls, %llu KiB, %zu distinct destinations (compare the `gnm dispatches=` "
                 "line, printed at exit too)",
                 static_cast<ull>(g_updates_tex.load()), static_cast<ull>(g_update_tex_bytes.load() >> 10), g_update_dst.size());
    }
}
std::uint64_t gx_cb_fnv(std::uint64_t va, std::uint64_t bytes) {
    std::uint64_t h = 1469598103934665603ull;
    const auto* p = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(va));
    for (std::uint64_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

std::uint64_t bn(std::uint64_t va) { return va - g_slide + kPreferredGuestSlide; }
std::uint64_t guest(std::uint64_t bn_va) { return g_slide + (bn_va - kPreferredGuestSlide); }
std::uint64_t rd64(std::uint64_t va) {
    std::uint64_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), 8);
    return v;
}
std::uint32_t rd32(std::uint64_t va) {
    std::uint32_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), 4);
    return v;
}
// Reads of objects we only have a pointer to: a bad pointer fails the call
// instead of faulting the game.
// The audit: how often the fallback below is taken, and what it costs. A guest
// structure the game itself just dereferenced is readable, so a read that lands
// here is one the fast path could not recognise - the YEBIS flip lost about
// 40 us a draw to exactly that.
std::atomic<std::uint64_t> g_safe_reads_slow{0}, g_safe_slow_ns{0}, g_safe_slow_bytes{0};
std::atomic<std::uint64_t> g_safe_reads_stack{0};  // taken off that path by the stack check
// Where those reads point, by megabyte, so the ranges worth teaching
// hle_kernel_va_readable() about can be named rather than guessed. An
// open-addressed table of relaxed counters: a collision just merges two
// megabytes, which is fine for a histogram read once at exit.
constexpr int kSlowBuckets = 512;
std::atomic<std::uint64_t> g_slow_where[kSlowBuckets];
std::atomic<std::uint64_t> g_slow_count[kSlowBuckets];

// And which call site asked, by return address: the fix for a read that
// happens a million times is usually in the caller (read it once a draw)
// rather than in the read.
std::atomic<std::uint64_t> g_slow_from[kSlowBuckets];
std::atomic<std::uint64_t> g_slow_from_n[kSlowBuckets];

void note_slow_site(std::uint64_t ret) {
    std::size_t i = static_cast<std::size_t>((ret * 0x9e3779b97f4a7c15ull) >> 55) % kSlowBuckets;
    for (int probe = 0; probe < 8; ++probe, i = (i + 1) % kSlowBuckets) {
        std::uint64_t held = g_slow_from[i].load(std::memory_order_relaxed);
        if (held == ret) break;
        if (held == 0 && g_slow_from[i].compare_exchange_strong(held, ret, std::memory_order_relaxed)) break;
        if (held == ret) break;
    }
    g_slow_from_n[i].fetch_add(1, std::memory_order_relaxed);
}

void note_slow_read(std::uint64_t va) {
    const std::uint64_t mb = va >> 20;
    std::size_t i = static_cast<std::size_t>((mb * 0x9e3779b97f4a7c15ull) >> 55) % kSlowBuckets;
    for (int probe = 0; probe < 8; ++probe, i = (i + 1) % kSlowBuckets) {
        std::uint64_t held = g_slow_where[i].load(std::memory_order_relaxed);
        if (held == mb + 1) break;
        if (held == 0 && g_slow_where[i].compare_exchange_strong(held, mb + 1, std::memory_order_relaxed)) break;
        if (held == mb + 1) break;
    }
    g_slow_count[i].fetch_add(1, std::memory_order_relaxed);
}

bool safe_read(std::uint64_t va, void* out, std::size_t n) {
    // Nothing to read. hle_kernel_va_readable() rejects an empty range, so
    // without this every zero-length read took the system call below and copied
    // nothing: 194,320 of them a run, 1,355 ns each, 263 ms.
    if (n == 0) return true;
    // Readable guest mappings copy directly; a system call per read made the
    // per-draw snapshots cost about 100 us (gx-cost-01).
    if (hle_kernel_va_readable(va, n)) {
        std::memcpy(out, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), n);
        return true;
    }
    // The reading thread's own stack. The GX layer passes pointers to
    // structures that live there - a shader's descriptor block, a stage's
    // tables - and those are the reads that dominated this fallback: 8.6
    // million a soak, 25 s of process_vm_readv, all of them in two anonymous
    // regions that turned out to be thread stacks (core/portable.h).
    std::uint64_t slo = 0, shi = 0;
    if (host_thread_stack(&slo, &shi) && va >= slo && va + n <= shi && va + n >= va) {
        std::memcpy(out, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), n);
        g_safe_reads_stack.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), out, n);
    g_safe_reads_slow.fetch_add(1, std::memory_order_relaxed);
    g_safe_slow_bytes.fetch_add(n, std::memory_order_relaxed);
    note_slow_read(va);
    note_slow_site(reinterpret_cast<std::uint64_t>(__builtin_return_address(0)));
    g_safe_slow_ns.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - t0).count()),
                             std::memory_order_relaxed);
    return ok;
}
bool in_window(std::uint64_t flip) { return g_trace && flip >= g_first && flip < g_last; }
int tid() {
    static thread_local int t = static_cast<int>(host_thread_id());
    return t;
}

struct StageSnap {
    std::uint64_t shader = 0;
    std::uint8_t samplers = 0, srvs = 0, cbs = 0;
};
struct alignas(16) GxRec {
    Kind kind = kKinds;
    int tid = 0;
    std::uint64_t ret = 0, ctx = 0, state = 0, flip = 0;
    std::uint64_t arg[4] = {};
    std::uint64_t indirect_va = 0;  // an indirect draw's arguments, resolved at the call
    std::uint64_t op1 = 0, ib = 0, blend = 0, ds = 0, rtv0 = 0, dsv = 0, pre = 0;
    std::uint64_t ib_base = 0;  // the index buffer's address, renamed ones included
    std::uint32_t topo = 0, ib_fmt = 0, ib_off = 0, stencil_ref = 0;
    std::uint8_t vbs = 0, rtvs = 0;
    // The fetch shader's vertex table as the commit would build it at this
    // call (build_vertex_table): the VS user-data slot it is bound at (0xff:
    // none) and the 16-byte V# record per table slot.
    std::uint8_t vtx_ud = 0xff;
    std::uint16_t vtx_valid = 0;
    std::uint32_t vtx_rec[16][4] = {};
    std::uint64_t vtx_obj[16] = {};
    StageSnap st[6];
    // For the layout probes: viewport 0 (x y w h minz maxz), scissor 0 (l t r
    // b), the rasterizer object and the first bound PS/VS resource slots.
    float vp[6] = {};
    std::int32_t sc[4] = {};
    std::uint64_t raster = 0;
    std::uint32_t blend_id = 0, ds_id = 0, raster_id = 0;  // engine/gx_state.h, at the call
    // The vertex-stage program's object (the domain shader's in a
    // tessellated draw), the pixel shader's and the input layout's, as
    // engine/gx_resources.h ids.
    std::uint32_t vs_prog_id = 0, ps_prog_id = 0, il_id = 0;
    struct Slot {
        std::uint8_t slot = 0;
        std::uint64_t obj = 0;
    };
    Slot srv[8], smp[4], cb[4], vcb[4];
    std::uint8_t nsrv = 0, nsmp = 0, ncb = 0, nvcb = 0;
    // BBHOST_GX_BACKEND: the draw's inputs as the GX objects give them.
    GpuDrawInputs in;
    bool in_ok = false;
    std::uint64_t native_id = 0;  // BBHOST_GX_NATIVE: this call's host-draw token (0: none)
    bool tess = false;            // shadowed as a tessellated draw (mode 1)
};
// Records sets (about 13 KB each) are made on the recording threads and freed
// on the command processor's, so malloc took each from a recording thread's
// arena and returned it under that arena's lock: about 1 us a draw
// (cost-native-04). A free list recycles them instead. Each thread caches up to
// 2 * kRecordsBatch and trades batches with a shared list, which keeps at most
// kRecordsShared and frees the rest.
struct GxRecordsRelease {
    void operator()(GxDrawRecords* p) const noexcept;
};
using GxRecordsPtr = std::unique_ptr<GxDrawRecords, GxRecordsRelease>;
static_assert(std::is_trivially_destructible_v<GxDrawRecords>);
constexpr unsigned kRecordsBatch = 64;
constexpr std::size_t kRecordsShared = 4096;
std::mutex g_records_mu;
std::vector<GxDrawRecords*>* const g_records_free = new std::vector<GxDrawRecords*>;  // under g_records_mu; never destroyed
// Trivially destructible, so records released during thread exit still find it.
thread_local GxDrawRecords* t_records_cache[2 * kRecordsBatch] = {};
thread_local unsigned t_records_cached = 0;
GxRecordsPtr new_records() {
    if (!t_records_cached) {
        std::lock_guard<std::mutex> lk(g_records_mu);
        const auto n = static_cast<unsigned>(std::min<std::size_t>(g_records_free->size(), kRecordsBatch));
        std::copy(g_records_free->end() - n, g_records_free->end(), t_records_cache);
        g_records_free->resize(g_records_free->size() - n);
        t_records_cached = n;
    }
    void* raw = nullptr;
    if (t_records_cached) {
        raw = t_records_cache[--t_records_cached];
        // The next draw's set: its stage headers, which the constructor and
        // the snapshot write first, asked for now (a pooled set is cold).
        if (t_records_cached) {
            const auto* next = reinterpret_cast<const char*>(t_records_cache[t_records_cached - 1]);
            for (int k = 0; k < kGxRecords; ++k) prefetch_for_write(next + offsetof(GxDrawRecords, stage) + k * sizeof(GxStageRecords));
        }
    } else {
        raw = ::operator new(sizeof(GxDrawRecords), std::align_val_t{alignof(GxDrawRecords)});
        g_records_new.fetch_add(1, std::memory_order_relaxed);
    }
    // Default-initialized, as `new GxDrawRecords` was: counts and slot bits
    // cleared, the large arrays left as they are.
    return GxRecordsPtr(new (raw) GxDrawRecords);
}
void GxRecordsRelease::operator()(GxDrawRecords* p) const noexcept {
    if (t_records_cached == 2 * kRecordsBatch) {
        std::lock_guard<std::mutex> lk(g_records_mu);
        for (unsigned i = kRecordsBatch; i < 2 * kRecordsBatch; ++i) {
            if (g_records_free->size() < kRecordsShared) {
                g_records_free->push_back(t_records_cache[i]);
            } else {
                ::operator delete(t_records_cache[i], std::align_val_t{alignof(GxDrawRecords)});
            }
        }
        t_records_cached = kRecordsBatch;
    }
    t_records_cache[t_records_cached++] = p;
}

struct EmitRec {
    Kind kind = kKinds;
    std::uint64_t ret = 0, flip = 0;
    bool gx = false;
    GxRec rec;
    GxRecordsPtr records;  // sampled GX draws
};

void count_stage(StageSnap& st, const std::uint64_t* q) {
    st.samplers = st.srvs = st.cbs = 0;
    for (int i = 0; i < 16; ++i) st.samplers += q[i] != 0;
    for (int i = 16; i < 16 + 128; ++i) st.srvs += q[i] != 0;
    for (int i = 144; i < 144 + 14; ++i) st.cbs += q[i] != 0;
    st.shader = q[158];
}

// The address the input-assembly flush (0x2abba30) and the vertex builder
// (0x2acfcf0) give a buffer whose byte +0x40 is set (0x2ad0ff0). The context
// keeps a map at ctx + 0xb6f8 of 24-byte entries (key at +8, value at +0x10)
// in chained pages: each holds u32 +0x18 entries followed by the next page's
// address, the last one u32 +0x1c. The key is the object's
// (u32 +0x28 << 40) | u64 +0x20 (0x256e1b0 of its +0x18); the value's +8 is
// the address. No entry: the object's own +0x20. Read-only.
std::uint64_t vertex_buffer_address(std::uint64_t map, std::uint64_t key_obj) {
    const std::uint64_t key = (static_cast<std::uint64_t>(rd32(key_obj + 0x10)) << 40) | rd64(key_obj + 8);
    const std::uint64_t per_page = rd32(map + 0x18);
    std::uint64_t page = rd64(map + 8);
    for (int pages = 0; page && pages < 4096; ++pages) {
        const std::uint64_t next = rd64(page + 0x18 * per_page);
        const std::uint64_t entries = next ? per_page : rd32(map + 0x1c);
        for (std::uint64_t i = 0; i < entries; ++i) {
            if (rd64(page + 0x18 * i + 8) == key) return rd64(rd64(page + 0x18 * i + 0x10) + 8);
        }
        page = next;
    }
    return rd64(key_obj + 8);
}

// The fetch shader's vertex table as the commit's vertex builder (0x2acfcf0)
// makes it for the VS at this call, from the input layout
// (state +0x0) and the vertex streams:
// - layout dwords at +0x30 (count byte +0x44): a stream (byte 2), its first
//   element (byte 0) and element count (byte 1);
// - per element: a 16-byte template (+0x20) and a slot dword (+0x28) whose
//   low byte is its table slot;
// - a record is the template with the address (its 44-bit offset + the
//   buffer's address + the stream offset, 0x1470710), the stride (0x1470730)
//   and the count (buffer size +0x2c / stride, 0x14707b0), then 0x14710c0
//   with (0x10, 0, 0), or (0x6e, 0, 1) for a renamed buffer;
// - streams in the table mask (+0x50) fill slots byte +0x47 up to
//   byte +0x47 + byte +0x48, bound at the VS user-data slot in the low byte
//   of the dword at *(+0x38). Streams in +0x4c bind each element to its own
//   user-data slot instead; those are not built here.
void build_vertex_table(GxRec& r, std::uint64_t s) {
    r.vtx_ud = 0xff;
    r.vtx_valid = 0;
    std::memset(r.vtx_obj, 0, sizeof(r.vtx_obj));
    const std::uint64_t layout = r.op1;
    if (!layout) return;
    const std::uint32_t table_mask = rd32(layout + 0x50);
    const std::uint64_t bind = rd64(layout + 0x38);
    if (!table_mask || !bind) return;
    r.vtx_ud = static_cast<std::uint8_t>(rd32(bind) & 0xff);
    const std::uint32_t first = rd32(layout + 0x47) & 0xff, slots = rd32(layout + 0x48) & 0xff;
    const std::uint32_t ranges = rd32(layout + 0x44) & 0xff;
    const std::uint64_t range_va = rd64(layout + 0x30), template_va = rd64(layout + 0x20), slot_va = rd64(layout + 0x28);
    for (std::uint32_t i = 0; i < ranges; ++i) {
        const std::uint32_t range = rd32(range_va + 4 * static_cast<std::uint64_t>(i));
        const std::uint32_t j = (range >> 16) & 0xff;
        if (j >= 32 || !((table_mask >> j) & 1)) continue;
        const std::uint64_t obj = rd64(s + 0x20 + 8 * static_cast<std::uint64_t>(j));
        const std::uint32_t stride = rd32(s + 0x120 + 4 * static_cast<std::uint64_t>(j));
        if (!obj || !stride) continue;
        const bool renamed = (rd32(obj + 0x40) & 0xff) != 0;
        const std::uint64_t base = (renamed ? vertex_buffer_address(r.ctx + 0xb6f8, obj + 0x18) : rd64(obj + 0x20)) +
                                   rd32(s + 0x1a0 + 4 * static_cast<std::uint64_t>(j));
        const std::uint32_t count = rd32(obj + 0x2c) / stride;
        for (std::uint32_t e = range & 0xff, end = e + ((range >> 8) & 0xff); e < end; ++e) {
            const std::uint32_t slot = rd32(slot_va + 4 * static_cast<std::uint64_t>(e)) & 0xff;
            std::uint32_t rec[4];
            if (slot < first || slot >= first + slots || slot >= 16 ||
                !safe_read(template_va + 16 * static_cast<std::uint64_t>(e), rec, sizeof(rec))) {
                continue;
            }
            const std::uint64_t addr = base + (static_cast<std::uint64_t>(rec[0]) | (static_cast<std::uint64_t>(rec[1] & 0xfff) << 32));
            rec[0] = static_cast<std::uint32_t>(addr);
            rec[1] = (rec[1] & 0xfffff000) | (static_cast<std::uint32_t>(addr >> 32) & 0xfff);
            rec[1] = (rec[1] & 0xc000ffff) | ((stride << 16) & 0x3fff0000);
            rec[2] = count;
            const std::uint32_t f = renamed ? 0x6e : 0x10;
            rec[1] = ((f & 3) << 14) | (rec[1] & 0xffff0fff) | (renamed ? 0x3000u : 0u);
            rec[3] = (((f >> 2) & 4) << 27) | (rec[3] & 0xc7ffffff);
            std::memcpy(r.vtx_rec[slot], rec, sizeof(rec));
            r.vtx_obj[slot] = obj;
            r.vtx_valid = static_cast<std::uint16_t>(r.vtx_valid | (1u << slot));
        }
    }
}

// BBHOST_GX_VIEW_CHECK=1: whether a shader-resource view's T#
// and a sampler's S# are still what they were made with when a draw binds
// them - which decides whether the host can know a view by its id alone.
const bool g_view_check = [] {
    const char* e = std::getenv("BBHOST_GX_VIEW_CHECK");
    return e && e[0] == '1';
}();
std::atomic<std::uint64_t> g_view_words[2][3] = {};  // [view, sampler][same, changed, not registered]
std::atomic<int> g_view_changed_logs{0};
std::uint32_t check_object_words(std::uint64_t object, const std::uint8_t* words, int kind) {
    std::uint64_t at_creation = 0;
    const std::uint32_t id = gx_object_lookup(object, &at_creation);
    if (!id || !at_creation) {
        g_view_words[kind][2].fetch_add(1, std::memory_order_relaxed);
        return id;
    }
    const bool same = gx_words_hash(words) == at_creation;
    g_view_words[kind][same ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
    if (!same && g_view_changed_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
        std::uint32_t w[8];
        std::memcpy(w, words, sizeof(w));
        host_log("gx: %s 0x%llx (id %u) is bound with words other than it was made with: %08x %08x %08x %08x %08x %08x %08x %08x",
                 kind ? "sampler" : "shader-resource view", static_cast<unsigned long long>(object), id, w[0], w[1], w[2], w[3], w[4], w[5],
                 w[6], w[7]);
    }
    return same ? id : 0;  // a view whose words moved on is not its id any more
}

// The draw's program and input-layout objects as ids, taken
// here where the objects are alive (build_inputs reads the same stages).
void snapshot_ids(GxRec& r) {
    const bool tess = r.st[1].shader || r.st[2].shader;
    r.vs_prog_id = gx_object_id(r.st[tess ? 2 : 0].shader);
    r.ps_prog_id = gx_object_id(r.st[4].shader);
    r.il_id = gx_object_id(r.op1);
}

// Pending draw-state layout. Six stages in D3D11 order
// (VS HS DS GS PS CS), 0x4f8 bytes each from +0x598: samplers[16], SRVs[128],
// constant buffers[14], then the shader. The commit (0x2abc320) reads the
// shaders at +0xa88 / +0xf80 / +0x1970 / +0x1e68 and walks the VS SRV mask
// over +0x618.
void snapshot(GxRec& r, std::uint64_t s) {
    r.op1 = rd64(s + 0x0);
    r.topo = rd32(s + 0x8);
    r.ib = rd64(s + 0x10);
    r.ib_fmt = rd32(s + 0x18);
    r.ib_off = rd32(s + 0x1c);
    // The slot counts and lists (vertex buffers, targets, the PS and VS
    // resource slots) are the trace's statistics and probes; without the trace
    // they were ~200 reads a draw nothing used.
    r.vbs = 0;
    if (g_trace) {
        for (int i = 0; i < 32; ++i) r.vbs += rd64(s + 0x20 + 8 * i) != 0;
    }
    // INDEX_BASE without the state offset, as the input-assembly flush takes it.
    r.ib_base = !r.ib ? 0 : (rd32(r.ib + 0x40) & 0xff) ? vertex_buffer_address(r.ctx + 0xb6f8, r.ib + 0x18) : rd64(r.ib + 0x20);
    build_vertex_table(r, s);
    r.blend = rd64(s + 0x4d0);
    r.ds = rd64(s + 0x4d8);
    r.rtvs = 0;
    if (g_trace) {
        for (int i = 0; i < 8; ++i) r.rtvs += rd64(s + 0x4e0 + 8 * i) != 0;
    }
    r.rtv0 = rd64(s + 0x4e0);
    r.dsv = rd64(s + 0x520);
    r.stencil_ref = rd32(s + 0x584);
    r.pre = rd64(s + 0x590);
    std::memcpy(r.vp, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(s + 0x248)), sizeof(r.vp));
    std::memcpy(r.sc, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(s + 0x3c8)), sizeof(r.sc));
    r.raster = rd64(s + 0x240);
    // The state objects as ids, here where they are alive (a
    // null pointer is GX's default state, which the registry knows too).
    r.blend_id = gx_blend_id(r.blend);
    r.ds_id = gx_depth_stencil_id(r.ds);
    r.raster_id = gx_raster_id(r.raster);
    const auto slots = [](std::uint64_t base, int n, GxRec::Slot* out, std::uint8_t& count, int cap) {
        count = 0;
        for (int i = 0; i < n && count < cap; ++i) {
            const std::uint64_t o = rd64(base + 8 * static_cast<std::uint64_t>(i));
            if (o) out[count++] = {static_cast<std::uint8_t>(i), o};
        }
    };
    const std::uint64_t vs_group = s + 0x598, ps_group = s + 0x598 + 0x4f8 * 4;
    if (g_trace) {
        slots(ps_group + 0x80, 128, r.srv, r.nsrv, 8);
        slots(ps_group, 16, r.smp, r.nsmp, 4);
        slots(ps_group + 0x480, 14, r.cb, r.ncb, 4);
        slots(vs_group + 0x480, 14, r.vcb, r.nvcb, 4);
    } else {
        r.nsrv = r.nsmp = r.ncb = r.nvcb = 0;
    }
    for (int k = 0; k < 6; ++k) {
        std::uint64_t q[159];
        const std::uint64_t g = s + 0x598 + 0x4f8 * static_cast<std::uint64_t>(k);
        if (!g_trace) {
            // Only the shader: the slot counts are the trace's statistics, and
            // reading six 1272-byte groups a draw to count them was ~10% of the
            // GX workers' time - which the main loop spends waiting for them.
            std::uint64_t shader = 0;
            if (k < 5) {
                std::memcpy(&shader, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(g + 158 * 8)), 8);
            } else if (!safe_read(g + 158 * 8, &shader, 8)) {
                shader = 0;
            }
            r.st[k] = StageSnap{};
            r.st[k].shader = shader;
            continue;
        }
        if (k < 5) {
            std::memcpy(q, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(g)), sizeof(q));
        } else if (!safe_read(g, q, sizeof(q))) {  // the CS group may run past the block
            r.st[k] = StageSnap{};
            continue;
        }
        count_stage(r.st[k], q);
    }
    snapshot_ids(r);
}

// Dispatches: only the CS group, through safe reads, since the compute state
// may be a smaller structure than the draw state.
void snapshot_compute(GxRec& r, std::uint64_t s) {
    for (auto& st : r.st) st = StageSnap{};
    if (!g_trace) {  // only the shader, as for draws
        std::uint64_t shader = 0;
        if (safe_read(s + 0x598 + 0x4f8 * 5 + 158 * 8, &shader, 8)) r.st[5].shader = shader;
        return;
    }
    std::uint64_t q[159];
    if (safe_read(s + 0x598 + 0x4f8 * 5, q, sizeof(q))) count_stage(r.st[5], q);
}

// A draw's inputs from the GX objects it was recorded with, as the flush
// functions and the Gnm setters turn
// them into registers. The command processor adds what GX does not give yet.
// False for draws the backend leaves on the PM4 path: indirect draws and
// tessellation / geometry stages.
constexpr std::uint64_t kDefaultBlend = 0x586c8f4, kDefaultDepthStencil = 0x586ca1c, kDefaultRasterizer = 0x586ca58;
constexpr std::uint64_t kTopologyTable = 0x4c6fb20;
constexpr std::uint64_t kDefaultSamplerCache = 0x586c8b0;

// `allow_tess` is off for the draw path until the rest of the tessellation work lands: the
// renderer draws from these inputs whenever they are present, and a
// tessellated draw's still lack the user data its stages bind. The shadow
// passes true to check what it can build against what the draw's own packets
// write, and nothing draws from that.
// The inputs only the programs decide, hashed here once rather
// than at every draw on the command processor (render.cpp's pipeline key).
std::uint64_t program_fingerprint(const GpuDrawInputs& in) {
    std::uint32_t k[36];
    k[0] = in.vte_cntl;
    k[1] = in.ps_input_ena;
    k[2] = in.vs_out_cntl;
    k[3] = in.cb_shader_mask;
    std::memcpy(k + 4, in.ps_input_cntl, sizeof(in.ps_input_cntl));
    std::uint64_t h = 1469598103934665603ull;
    for (const std::uint32_t w : k) h = (h ^ w) * 1099511628211ull;
    return h | 1;
}

// What build_inputs takes from a draw's shaders and input layout: the stages'
// program registers and the PS input table, whose linkage (each PS input
// matched against the VS outputs) was a loop over both shader structures at
// every draw. All three objects never change once made, so each GX thread
// keeps it per (VS, PS, input layout) registry ids.
struct LinkInfo {
    std::uint32_t vs_id = 0, ps_id = 0, il_id = 0;  // 0 vs_id: empty
    std::uint32_t vs_pgm[4], ps_pgm[4];
    std::uint32_t vs_out_cntl, ps_input_ena, ps_in_control, cb_shader_mask, ps_col_format, ps_input_count;
    std::uint32_t ps_input_cntl[32];
};
constexpr std::size_t kLinkInfos = 256;
thread_local LinkInfo* t_link_infos = nullptr;
Tally<2> g_link_info;  // [0] from the cache, [1] worked out
// BBHOST_GX_LINK_CHECK=1: every draw the cache serves is worked out as well
// and the two compared.
const bool g_link_check = [] {
    const char* e = std::getenv("BBHOST_GX_LINK_CHECK");
    return e && e[0] == '1';
}();
std::atomic<std::uint64_t> g_link_checked{0}, g_link_differ{0};

bool build_inputs(const GxRec& r, GpuDrawInputs& in, bool allow_tess = false) {
    // A bound geometry-shader object (st[3]) does not refuse the draw: the
    // register file never enabled a geometry stage in any run, so
    // GX draws these as VS/PS and so does the host.
    if (r.kind > kDrawIndexedIndirect || !r.st[0].shader) return false;
    if (!allow_tess && (r.st[1].shader || r.st[2].shader)) return false;
    // A tessellated draw is GX's VS/HS/DS/PS, which is the
    // pipeline's vertex/control/evaluation/fragment. The domain shader is the
    // hardware's vertex stage, so it fills vs_*; GX's VS is the LS.
    const bool tess = r.st[1].shader || r.st[2].shader;
    if (tess && !(r.st[1].shader && r.st[2].shader)) return false;  // one without the other is a shape we have not seen
    const std::uint64_t s = r.state;
    in = GpuDrawInputs{};
    in.ps_input_count = 0;
    // The shaders' part from the per-thread cache, when all three objects have
    // ids (a tessellated draw's four stages are worked out each time).
    LinkInfo* link = nullptr;
    LinkInfo cached;
    bool check_cached = false;
    const std::uint32_t link_il = r.op1 ? r.il_id : 0;
    if (!tess && r.vs_prog_id && (r.ps_prog_id || !r.st[4].shader) && (link_il || !r.op1)) {
        if (!t_link_infos) t_link_infos = new LinkInfo[kLinkInfos];
        const std::uint32_t h = (r.vs_prog_id * 0x9e3779b1u) ^ (r.ps_prog_id * 0x85ebca6bu) ^ (link_il * 0xc2b2ae35u);
        link = &t_link_infos[(h ^ (h >> 16)) & (kLinkInfos - 1)];
        if (link->vs_id == r.vs_prog_id && link->ps_id == r.ps_prog_id && link->il_id == link_il && g_link_check) {
            g_link_info.add(1, 0);
            cached = *link;
            check_cached = true;
            link = nullptr;  // worked out below and compared
        } else if (link->vs_id == r.vs_prog_id && link->ps_id == r.ps_prog_id && link->il_id == link_il) {
            g_link_info.add(1, 0);
            std::memcpy(in.vs_pgm, link->vs_pgm, sizeof(in.vs_pgm));
            std::memcpy(in.ps_pgm, link->ps_pgm, sizeof(in.ps_pgm));
            in.vs_out_cntl = link->vs_out_cntl;
            in.ps_input_ena = link->ps_input_ena;
            in.ps_in_control = link->ps_in_control;
            in.cb_shader_mask = link->cb_shader_mask;
            in.ps_col_format = link->ps_col_format;
            in.ps_input_count = link->ps_input_count;
            std::memcpy(in.ps_input_cntl, link->ps_input_cntl, sizeof(in.ps_input_cntl));
            goto shaders_done;
        }
        if (!check_cached) g_link_info.add(1, 1);
    }
    {
    // Shaders: sceGnmSetVsShader / sceGnmSetPsShader take the Gnm shader
    // structure (+0x100) + 8, whose leading dwords are the stage registers.
    const std::uint64_t vs_gnm = rd64(r.st[tess ? 2 : 0].shader + 0x100);
    if (!vs_gnm) return false;
    for (int i = 0; i < 4; ++i) in.vs_pgm[i] = rd32(vs_gnm + 8 + 4 * static_cast<std::uint64_t>(i));
    if (!tess && r.op1) in.vs_pgm[2] |= rd32(r.op1 + 0x40);  // the input layout's fetch-shader modifier
    in.vs_out_cntl = rd32(vs_gnm + 8 + 0x18);
    if (tess) {
        in.tess = true;
        const std::uint64_t ls_gnm = rd64(r.st[0].shader + 0x100), hs_gnm = rd64(r.st[1].shader + 0x100);
        if (!ls_gnm || !hs_gnm) return false;
        for (int i = 0; i < 4; ++i) {
            in.ls_pgm[i] = rd32(ls_gnm + 8 + 4 * static_cast<std::uint64_t>(i));
            in.hs_pgm[i] = rd32(hs_gnm + 8 + 4 * static_cast<std::uint64_t>(i));
        }
        if (r.op1) in.ls_pgm[2] |= rd32(r.op1 + 0x40);  // the LS is the stage the fetch shader belongs to
        // The patch and domain configuration, from the hull shader's own
        // structure. Read off a run rather than assumed: dword 4 is
        // VGT_TF_PARAM (0x42, a quad domain), 5 and 6 the clamps - the shader
        // that asks for outer factors of 3 has 0x40400000 in 5, which is the
        // 3 the register file shows - and 7 and 8 the input and output control
        // points.
        in.tf_param = rd32(hs_gnm + 8 + 4 * 4);
        in.hos_max = rd32(hs_gnm + 8 + 4 * 5);
        in.hos_min = rd32(hs_gnm + 8 + 4 * 6);
        const std::uint32_t in_cp = rd32(hs_gnm + 8 + 4 * 7), out_cp = rd32(hs_gnm + 8 + 4 * 8);
        // VGT_LS_HS_CONFIG's own bits: patches a group in [7:0], which nothing
        // here reads, then the two control-point counts. A count that is not
        // one is a shape the renderer refuses rather than draws wrongly.
        in.ls_hs_config = ((out_cp & 0x3f) << 14) | ((in_cp & 0x3f) << 8);
        // VGT_SHADER_STAGES_EN: LS and HS on with the domain shader as the
        // vertex stage, which is what a GX draw with a hull and a domain
        // shader and no geometry shader is.
        in.stages = 0x45;
    }
    if (r.st[4].shader) {
        const std::uint64_t ps_gnm = rd64(r.st[4].shader + 0x100);
        if (!ps_gnm) return false;
        for (int i = 0; i < 4; ++i) in.ps_pgm[i] = rd32(ps_gnm + 8 + 4 * static_cast<std::uint64_t>(i));
        in.ps_input_ena = rd32(ps_gnm + 8 + 0x18);
        in.ps_in_control = rd32(ps_gnm + 8 + 0x20);
        in.cb_shader_mask = rd32(ps_gnm + 8 + 0x2c);
        in.ps_col_format = rd32(ps_gnm + 8 + 0x14);
        // SPI_PS_INPUT_CNTL, as the shader bind 0x2ace1e0 computes it
        // (0x14873a0) and writes one entry per PS input (0x1475000). VS
        // outputs are {semantic, register} bytes; PS inputs are words with the
        // semantic in the low byte. Each entry: the register of the VS output
        // with that semantic or 0x20, bit 12 of the word as bit 5, the high
        // byte's low two bits at 8, bit 12 or 10 at 10.
        const auto u8 = [](std::uint64_t va) { return rd32(va) & 0xff; };
        const std::uint64_t vs_out = vs_gnm + ((static_cast<std::uint64_t>(u8(vs_gnm + 0x24)) + u8(vs_gnm + 3)) << 2) + 0x28;
        const std::uint32_t vs_outs = u8(vs_gnm + 0x25);
        const std::uint64_t ps_in = ps_gnm + (static_cast<std::uint64_t>(u8(ps_gnm + 3)) << 2) + 0x3c;
        in.ps_input_count = std::min<std::uint32_t>(u8(ps_gnm + 0x38), 32);
        for (std::uint32_t i = 0; i < in.ps_input_count; ++i) {
            const std::uint32_t word = rd32(ps_in + 2 * static_cast<std::uint64_t>(i)) & 0xffff;
            std::uint32_t reg = 0x20;
            for (std::uint32_t j = 0; j < vs_outs; ++j) {
                if (u8(vs_out + 2 * static_cast<std::uint64_t>(j)) == (word & 0xff)) {
                    reg = u8(vs_out + 2 * static_cast<std::uint64_t>(j) + 1);
                    break;
                }
            }
            in.ps_input_cntl[i] = (((word >> 7) & 0x1e0) | reg) & 0x3f;
            in.ps_input_cntl[i] |= ((word >> 8) & 3) << 8;
            in.ps_input_cntl[i] |= (((word >> 12) | (word >> 10)) & 1) << 10;
        }
    }
    if (link) {
        link->vs_id = r.vs_prog_id;
        link->ps_id = r.ps_prog_id;
        link->il_id = link_il;
        std::memcpy(link->vs_pgm, in.vs_pgm, sizeof(in.vs_pgm));
        std::memcpy(link->ps_pgm, in.ps_pgm, sizeof(in.ps_pgm));
        link->vs_out_cntl = in.vs_out_cntl;
        link->ps_input_ena = in.ps_input_ena;
        link->ps_in_control = in.ps_in_control;
        link->cb_shader_mask = in.cb_shader_mask;
        link->ps_col_format = in.ps_col_format;
        link->ps_input_count = in.ps_input_count;
        std::memcpy(link->ps_input_cntl, in.ps_input_cntl, sizeof(in.ps_input_cntl));
    }
    }
    if (check_cached) {
        g_link_checked.fetch_add(1, std::memory_order_relaxed);
        const bool same = std::memcmp(cached.vs_pgm, in.vs_pgm, sizeof(in.vs_pgm)) == 0 &&
                          std::memcmp(cached.ps_pgm, in.ps_pgm, sizeof(in.ps_pgm)) == 0 && cached.vs_out_cntl == in.vs_out_cntl &&
                          cached.ps_input_ena == in.ps_input_ena && cached.ps_in_control == in.ps_in_control &&
                          cached.cb_shader_mask == in.cb_shader_mask && cached.ps_col_format == in.ps_col_format &&
                          cached.ps_input_count == in.ps_input_count &&
                          std::memcmp(cached.ps_input_cntl, in.ps_input_cntl, sizeof(in.ps_input_cntl)) == 0;
        if (!same && g_link_differ.fetch_add(1, std::memory_order_relaxed) < 8) {
            host_log("gx-native: the shader linkage cached for VS %u PS %u layout %u differs from the draw's", r.vs_prog_id, r.ps_prog_id,
                     link_il);
        }
    }
shaders_done:
    const std::uint32_t topology = rd32(s + 0x8);
    if (topology > 0x29) return false;
    in.prim = rd32(guest(kTopologyTable) + 4 * static_cast<std::uint64_t>(topology));

    // Output merger (0x2ad2230): blend [0] CB_TARGET_MASK, [1..8] CB_BLENDn_CONTROL.
    const std::uint64_t blend = r.blend ? r.blend : guest(kDefaultBlend);
    in.target_mask = rd32(blend);
    for (int t = 0; t < 8; ++t) in.blend[t] = rd32(blend + 4 + 4 * static_cast<std::uint64_t>(t));
    std::memcpy(in.blend_const, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(s + 0x568)), sizeof(in.blend_const));
    for (int t = 0; t < 8; ++t) {
        const std::uint64_t rtv = rd64(s + 0x4e0 + 8 * static_cast<std::uint64_t>(t));
        if (!rtv) continue;
        for (int k = 0; k < 5; ++k) in.color[t][k] = rd32(rtv + 0x10 + 4 * static_cast<std::uint64_t>(k));
        in.color_extent[t] = gx_view_extent(rtv);
    }
    const std::uint64_t ds = r.ds ? r.ds : guest(kDefaultDepthStencil);
    in.depth_control = rd32(ds);
    in.stencil_control = rd32(ds + 4);
    const std::uint32_t ref = rd32(s + 0x584) & 0xff;
    in.stencil_ref = in.stencil_ref_bf = (rd32(ds + 8) & 0xffff00) | ref | ref << 24;  // both, as 0x1473190 writes
    std::memcpy(in.depth_bounds, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(s + 0x578)), sizeof(in.depth_bounds));
    if (r.dsv) {
        in.z_info = rd32(r.dsv + 0x10);
        in.stencil_info = rd32(r.dsv + 0x14);
        in.z_read_base = rd32(r.dsv + 0x18);
        in.depth_size = rd32(r.dsv + 0x28);
        // The descriptor's DEPTH_VIEW word (register 0x02); SetDepthRenderTarget
        // (0x14739b0) packs it after the eight register-0x10 words.
        in.depth_view = rd32(r.dsv + 0x30);
        in.htile_base = rd32(r.dsv + 0x34);
        in.depth_extent = gx_view_extent(r.dsv);
    }

    // Rasterizer (0x2ad3370), with the masks the Gnm setters apply.
    const std::uint64_t raster = r.raster ? r.raster : guest(kDefaultRasterizer);
    in.clip_cntl = rd32(raster) & 0xf7fe03f;
    in.su_sc_mode = rd32(raster + 4) & 0x393fff;
    in.vte_cntl = 0x43f;
    // PA_SC_MODE_CNTL_0 (sub_1477150): bit 0 from the rasterizer's byte +0x14,
    // bit 1 (VPORT_SCISSOR_ENABLE) from +0x15. With that set the flush writes
    // scissor 0 {left, top, right, bottom} (pending state +0x3c8, each clamped
    // at 0) into PA_SC_VPORT_SCISSOR_0 with WINDOW_OFFSET_DISABLE (sub_14766d0).
    const std::uint32_t modes = rd32(raster + 0x14);
    in.sc_mode_cntl_0 = (modes & 1) | ((modes >> 8) & 1) << 1;
    if ((modes >> 8) & 0xff) {
        const auto edge = [](std::uint64_t va) {
            const auto v = static_cast<std::int32_t>(rd32(va));
            return static_cast<std::uint32_t>(v < 0 ? 0 : v);
        };
        in.vport_scissor[0] = 0x80000000u | ((edge(s + 0x3cc) << 16) & 0x7fff0000) | (edge(s + 0x3c8) & 0x7fff);
        in.vport_scissor[1] = ((edge(s + 0x3d4) << 16) & 0x7fff0000) | (edge(s + 0x3d0) & 0x7fff);
    }
    // Viewport 0 {x, y, w, h, min z, max z}: scale (w/2, -h/2, max z - min z),
    // offset (x + w/2, y + h/2, 0); the screen scissor is its rectangle.
    const float* vp = r.vp;
    in.vport[0] = vp[2] * 0.5f;
    in.vport[1] = vp[0] + vp[2] * 0.5f;
    in.vport[2] = vp[3] * -0.5f;
    in.vport[3] = vp[1] + vp[3] * 0.5f;
    in.vport[4] = vp[5] - vp[4];
    in.vport[5] = 0.0f;
    const auto pack = [](float x, float y) {
        return (static_cast<std::uint32_t>(static_cast<int>(y)) << 16) | (static_cast<std::uint32_t>(static_cast<int>(x)) & 0xffff);
    };
    in.screen_scissor[0] = pack(vp[0], vp[1]);
    in.screen_scissor[1] = pack(vp[0] + vp[2], vp[1] + vp[3]);

    // VGT_INDX_OFFSET: the start vertex / base vertex argument (0x1474040).
    switch (r.kind) {
        case kDraw: in.base_vertex = static_cast<std::int32_t>(r.arg[1]); break;
        case kDrawIndexed: in.base_vertex = static_cast<std::int32_t>(r.arg[2]); break;
        case kDrawInstanced: in.base_vertex = static_cast<std::int32_t>(r.arg[2]); break;
        case kDrawIndexedInstanced: in.base_vertex = static_cast<std::int32_t>(r.arg[3]); break;
        default: break;
    }
    return true;
}

struct CpStats {
    std::uint64_t total = 0, gx = 0, other = 0, none = 0, arg_ok = 0, arg_bad = 0, scans = 0;
    std::map<std::uint32_t, std::uint64_t> ops, unmatched, unmatched_prog;
    std::map<std::uint64_t, std::uint64_t> other_callers;
};
struct Stats {
    // At the call.
    std::uint64_t calls[kKinds] = {};
    std::map<std::pair<int, std::uint64_t>, std::uint64_t> callers;  // (kind, return address) -> calls
    std::map<std::pair<int, std::uint64_t>, std::uint64_t> ctxs;     // (thread, context) -> calls
    std::uint64_t emit_gx[kKinds] = {};
    std::uint64_t near_end = 0;
    std::uint64_t gx_state = 0, dsv = 0, blend = 0, ds = 0, pre = 0;
    std::map<std::uint32_t, std::uint64_t> topo, ib_fmt, vbs, rtvs, stencil;
    std::uint64_t stage_set[6] = {}, stage_samplers[6] = {}, stage_srvs[6] = {}, stage_cbs[6] = {};
    std::uint64_t cs_calls = 0, cs_set = 0, cs_samplers = 0, cs_srvs = 0, cs_cbs = 0;
    std::map<std::uint64_t, std::uint64_t> op1;
    // At execution: [0] draws, [1] dispatches.
    CpStats cp[2];
    std::uint64_t cp_gx_kind[kKinds] = {};
    std::uint64_t inst_ok = 0, inst_bad = 0;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint64_t> fmt_pairs, topo_pairs;
    std::map<int, std::uint64_t> ib_slot;
    std::map<std::tuple<int, int, char>, std::uint64_t> stage_hits;
    std::vector<std::string> bad_samples;
};
// One GX draw in BBHOST_GX_BIND_EVERY (default 16, 0 = none)
// on each thread has its resource records copied at the emitter.
// Rendering from GX (BBHOST_GX_BACKEND=1) resolves every draw's resources from
// its records, so it takes them on every draw; comparing samples 1 in 16.
bool sample_bindings() {
    static const int every = [] {
        const char* e = std::getenv("BBHOST_GX_BIND_EVERY");
        return e ? std::atoi(e) : g_backend == 1 ? 1 : 16;
    }();
    thread_local unsigned n = 0;
    return every > 0 && (n++ % static_cast<unsigned>(every)) == 0;
}

// The address the commit gives a buffer (0x2aae840): dynamic buffers are
// renamed through a per-context map at ctx + 0xb3c0, keyed by the buffer's
// +0x18 sub-object. The map's pages hold 128 16-byte {key, address} entries.
// A map id (+0x18) up to 15 indexes the key's own table (+0x38, one 1-based
// u16 per map id); a larger id scans the pages. No entry: the key's +8, the
// buffer's own address. Read-only.
std::uint64_t dynamic_buffer_address(std::uint64_t map, std::uint64_t key) {
    const std::uint32_t id = rd32(map + 0x18);
    const std::uint64_t pages = rd64(map + 8);
    const std::uint32_t used = rd32(map + 0x10);
    if (pages && id <= 0xf) {
        if (g_gx_cost_enabled) g_dynbuf_table.fetch_add(1, std::memory_order_relaxed);
        const std::uint64_t table = rd64(key + 0x38);
        std::uint16_t index1 = 0;
        if (table && safe_read(table + 2 * static_cast<std::uint64_t>(id), &index1, 2) && index1 && used > index1 - 1u) {
            const std::uint32_t index = index1 - 1u;
            const std::uint64_t page = rd64(pages + 8 * static_cast<std::uint64_t>(index >> 7));
            const std::uint64_t entry = page + 16 * static_cast<std::uint64_t>(index & 0x7f);
            if (page && rd64(entry) == key) return rd64(entry + 8);
        }
    } else if (pages) {
        if (g_gx_cost_enabled) g_dynbuf_scans.fetch_add(1, std::memory_order_relaxed);
        const std::uint32_t last = used ? ((used + 0x7f) & 0x7f) + 1 : 0;
        for (std::uint64_t slot = pages;; slot += 8) {
            const std::uint64_t page = rd64(slot);
            if (!page) break;
            if (g_gx_cost_enabled) g_dynbuf_pages.fetch_add(1, std::memory_order_relaxed);
            const std::uint32_t entries = rd64(slot + 8) ? 0x80 : last;
            for (std::uint32_t i = 0; i < entries; ++i) {
                if (rd64(page + 16 * static_cast<std::uint64_t>(i)) == key) return rd64(page + 16 * static_cast<std::uint64_t>(i) + 8);
            }
        }
    }
    return rd64(key + 8);
}

// The records the commit has just written for this draw: VS arrays at ctx
// +0x2930 / +0x8930 / +0x9530 / +0x9b30, PS
// at +0x6930 / +0x9130 / +0x9930 / +0xa030 and the extended blocks at +0xa8f0
// (VS) / +0xa6f0 (PS), with each shader's descriptors (+0xe0, count in the low dword of
// +0xe8).
// What a draw's records take from a shader object, once per object: its
// resource masks, its user-data descriptors with their fingerprint, whether one
// wants the context's 0xc0-byte block, and the slots render mode reads. A GX
// shader object never changes once made (programs are found by its id);
// its registry id tells a new object at an old address apart.
// Per thread, so nothing is shared: the GX threads build draws on the game's
// critical path (its job barrier waits for them), and this was ~20% of a
// draw's records there (gx-cost "shader").
struct ShaderInfo {
    std::uint64_t shader = 0;
    std::uint32_t id = 0;  // 0: empty
    std::uint32_t ndesc = 0;
    std::uint64_t mask[3] = {};
    std::uint64_t desc_fp = 0;
    std::uint64_t tex_needed = 0;
    std::uint32_t smp_needed = 0, cb_needed = 0;
    bool ctx_block = false;  // a descriptor of type 7 or 0x14
    std::uint32_t desc[64];
};
constexpr std::size_t kShaderInfos = 256;
thread_local ShaderInfo* t_shader_infos = nullptr;
Tally<2> g_shader_info;  // [0] records from the cache, [1] read from the object

GxRecordsPtr snapshot_records(const GxRec& r) {
    struct Arrays {
        std::uint64_t tex, smp, cache, cb, ext;
    };
    constexpr Arrays kArrays[2] = {{0x2930, 0x8930, 0x9530, 0x9b30, 0xa8f0}, {0x6930, 0x9130, 0x9930, 0xa030, 0xa6f0}};
    // Not zeroed: readers check the per-slot bits and counts.
    GxRecordsPtr out;
    {
        GxCostTimer timer(kCostRecAlloc);
        out = new_records();
    }
    static thread_local unsigned hash_count = 0;
    const bool hash_cbs = g_gx_cost_enabled && (hash_count++ % 16) == 0;
    // Which GX stage each record slot holds. A tessellated draw's
    // domain shader is the hardware's vertex stage, so it is what binds set 0;
    // GX's own VS is then the LS, whose user data is wanted and whose
    // bindings are none. Slot 2 is only filled for those draws.
    const bool tess = r.st[1].shader || r.st[2].shader;
    const int set0_stage = tess && r.st[2].shader ? 2 : 0;
    const int stage_of[kGxRecords] = {set0_stage, 4, 0, 1};
    const std::uint64_t shaders[kGxRecords] = {r.st[set0_stage].shader, r.st[4].shader, tess ? r.st[0].shader : 0,
                                               tess ? r.st[1].shader : 0};
    const int nrec = tess ? 4 : 2;
    // The pending state's slot objects for each stage (group: samplers [16]
    // at +0, SRVs [128] at +0x80, constant buffers [14] at +0x480).
    std::uint64_t groups[kGxRecords];
    for (int k = 0; k < kGxRecords; ++k) groups[k] = r.state + 0x598 + 0x4f8ull * static_cast<std::uint64_t>(stage_of[k]);
    std::uint64_t tex_needed[kGxRecords] = {};
    std::uint32_t smp_needed[kGxRecords] = {}, cb_needed[kGxRecords] = {};
    // The shaders' registry ids (snapshot_ids): set 0's stage and the pixel
    // stage. The LS of a tessellated draw is read each time.
    const std::uint32_t shader_ids[kGxRecords] = {r.vs_prog_id, r.ps_prog_id, 0, 0};
    if (!t_shader_infos) t_shader_infos = new ShaderInfo[kShaderInfos];
    for (int k = 0; k < nrec; ++k) {
        GxStageRecords& s = out->stage[k];
        GxCostStamp stamp;
        ShaderInfo* info = nullptr;
        if (g_backend == 1 && shaders[k] && shader_ids[k]) {
            info = &t_shader_infos[shader_ids[k] & (kShaderInfos - 1)];
            if (info->id == shader_ids[k] && info->shader == shaders[k]) {
                g_shader_info.add(1, 0);
                std::memcpy(s.mask, info->mask, sizeof(s.mask));
                s.ndesc = info->ndesc;
                std::memcpy(s.desc, info->desc, info->ndesc * 4u);
                s.desc_fp = info->desc_fp;
                s.has_ctx_block = info->ctx_block && safe_read(r.ctx + 0xb2f0, s.ctx_block, sizeof(s.ctx_block));
                tex_needed[k] = info->tex_needed;
                smp_needed[k] = info->smp_needed;
                cb_needed[k] = info->cb_needed;
                stamp.to(kCostRecShader);
                continue;
            }
            g_shader_info.add(1, 1);
        }
        if (shaders[k]) {
            // Only what was read whole goes into the cache.
            bool whole = safe_read(shaders[k], s.mask, sizeof(s.mask));
            std::uint64_t table = 0, count = 0;
            whole = whole && safe_read(shaders[k] + 0xe0, &table, 8) && safe_read(shaders[k] + 0xe8, &count, 8);
            if (whole && table) {
                const std::uint32_t n = std::min<std::uint32_t>(static_cast<std::uint32_t>(count), 64);
                if (safe_read(table, s.desc, n * 4u)) {
                    s.ndesc = n;
                } else {
                    whole = false;
                }
            }
            if (!whole) info = nullptr;
        }
        // The descriptors' fingerprint, which keys the renderer's binding plans.
        std::uint64_t fp = 1469598103934665603ull ^ s.ndesc;
        for (std::uint32_t i = 0; i < s.ndesc; ++i) fp = (fp ^ s.desc[i]) * 1099511628211ull;
        s.desc_fp = s.ndesc ? fp | 1 : 0;
        // The context's 0xc0-byte block for a descriptor of type 7 or 0x14
        // (0x2ace5e0 copies gx+0x50+0xb2a0 into its ring for them).
        s.has_ctx_block = false;
        for (std::uint32_t i = 0; i < s.ndesc; ++i) {
            const std::uint32_t type = s.desc[i] & 0xff;
            if (type == 0x7 || type == 0x14) {
                s.has_ctx_block = safe_read(r.ctx + 0xb2f0, s.ctx_block, sizeof(s.ctx_block));
                break;
            }
        }
        // Render mode reads only the slots the shader can use: its resource
        // masks (textures 0-63; sampler caches in bits 0-15, constant buffers
        // 16-29, samplers 36-51 of the third) and every slot its user-data
        // descriptors name. Compare mode checks every slot against the
        // commit's arrays, so it keeps them all.
        tex_needed[k] = ~0ull;
        smp_needed[k] = 0xffff;
        cb_needed[k] = 0x3fff;
        if (g_backend == 1) {
            tex_needed[k] = s.mask[0];
            smp_needed[k] = static_cast<std::uint32_t>((s.mask[2] & 0xffff) | ((s.mask[2] >> 36) & 0xffff));
            cb_needed[k] = static_cast<std::uint32_t>((s.mask[2] >> 16) & 0x3fff);
            for (std::uint32_t i = 0; i < s.ndesc; ++i) {
                const std::uint32_t type = s.desc[i] & 0xff, slot = (s.desc[i] >> 16) & 0xff;
                switch (type) {
                    case 0x0: case 0x1: case 0xd: case 0xe:
                        if (slot < 64) tex_needed[k] |= 1ull << slot;
                        break;
                    case 0x2: case 0x3: case 0x4: case 0xf: case 0x10: case 0x11:
                        if (slot < 16) smp_needed[k] |= 1u << slot;
                        break;
                    case 0x5: case 0x12:
                        if (slot < 14) cb_needed[k] |= 1u << slot;
                        break;
                    default:
                        break;
                }
            }
        }
        if (info) {
            info->shader = shaders[k];
            info->id = shader_ids[k];
            info->ndesc = s.ndesc;
            std::memcpy(info->mask, s.mask, sizeof(info->mask));
            std::memcpy(info->desc, s.desc, s.ndesc * 4u);
            info->desc_fp = s.desc_fp;
            info->tex_needed = tex_needed[k];
            info->smp_needed = smp_needed[k];
            info->cb_needed = cb_needed[k];
            info->ctx_block = false;
            for (std::uint32_t i = 0; i < s.ndesc; ++i) {
                const std::uint32_t type = s.desc[i] & 0xff;
                if (type == 0x7 || type == 0x14) info->ctx_block = true;
            }
        }
        stamp.to(kCostRecShader);
    }
    // Every object below is a cache miss more often than not (views,
    // samplers and buffers the game wrote on other threads), and read one
    // after another they cost a miss each: ask for all of them first, both
    // stages', so the misses overlap. A prefetch of a bad pointer is harmless.
    const auto prefetch = [](std::uint64_t va) { __builtin_prefetch(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va))); };
    for (int k = 0; k < nrec; ++k) {
        for (std::uint64_t m = tex_needed[k]; m; m &= m - 1) {
            if (const std::uint64_t view = rd64(groups[k] + 0x80 + 8 * static_cast<std::uint64_t>(__builtin_ctzll(m)))) {
                prefetch(view + 0x10);
                prefetch(view + 0x2f);
            }
        }
        for (std::uint32_t m = smp_needed[k]; m; m &= m - 1) {
            if (const std::uint64_t smp = rd64(groups[k] + 8 * static_cast<std::uint64_t>(__builtin_ctz(m)))) {
                prefetch(smp);
                prefetch(smp + 0x1f);
            }
        }
        for (std::uint32_t m = cb_needed[k]; m; m &= m - 1) {
            if (const std::uint64_t cb = rd64(groups[k] + 0x480 + 8 * static_cast<std::uint64_t>(__builtin_ctz(m)))) {
                prefetch(cb + 0x18);
                prefetch(cb + 0x57);
            }
        }
    }
    // The objects sit in a few heaps: the last mapped range one was found in
    // answers the next one without the kernel's map list.
    std::uint64_t mapped_lo = 0, mapped_hi = 0, readable_lo = 0, readable_hi = 0;
    const auto object = [&](std::uint64_t slot_va, std::uint64_t bytes) -> std::uint64_t {
        const std::uint64_t obj = rd64(slot_va);
        if (!obj) return 0;
        if (obj >= mapped_lo && obj + bytes <= mapped_hi) return obj;
        return hle_kernel_va_mapped_in(obj, bytes, &mapped_lo, &mapped_hi) ? obj : 0;
    };
    for (int k = 0; k < nrec; ++k) {
        GxStageRecords& s = out->stage[k];
        const Arrays& a = kArrays[k < 2 ? k : 0];
        const std::uint64_t group = groups[k];
        GxCostStamp stamp;
        if (g_backend != 1 && k < 2) {
            // The commit's records, which only compare mode reads. Only the
            // two stages it was built for have them; the LS's slot is never
            // compared.
            safe_read(r.ctx + a.tex, s.tex, sizeof(s.tex));
            safe_read(r.ctx + a.smp, s.smp, sizeof(s.smp));
            safe_read(r.ctx + a.cache, s.smp_cache, sizeof(s.smp_cache));
            safe_read(r.ctx + a.cb, s.cb, sizeof(s.cb));
            if (a.ext) safe_read(r.ctx + a.ext, s.ext, sizeof(s.ext));
        }
        for (std::uint64_t m = tex_needed[k]; m; m &= m - 1) {
            const int j = __builtin_ctzll(m);
            if (const std::uint64_t view = object(group + 0x80 + 8 * static_cast<std::uint64_t>(j), 0x30)) {
                std::memcpy(s.obj_tex + j * 32, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(view + 0x10)), 32);
                s.obj_tex_set |= 1ull << j;
                s.obj_tex_id[j] = g_view_check ? check_object_words(view, s.obj_tex + j * 32, 0) : gx_object_id(view);
            }
        }
        stamp.to(kCostRecTex);
        for (std::uint32_t m = smp_needed[k]; m; m &= m - 1) {
            const int j = __builtin_ctz(m);
            if (const std::uint64_t smp = object(group + 8 * static_cast<std::uint64_t>(j), 32)) {
                std::memcpy(s.obj_smp + j * 32, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(smp)), 32);
                s.obj_smp_set |= 1u << j;
                if (g_view_check) (void)check_object_words(smp, s.obj_smp + j * 32, 1);
            } else {
                // An empty slot's sampler cache gets the static default record
                // (0x2abc320 loads data_586c8b0).
                std::memcpy(s.obj_smp + j * 32, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(guest(kDefaultSamplerCache))), 16);
                s.obj_smp_default |= 1u << j;
            }
        }
        stamp.to(kCostRecSmp);
        // Constant buffers: 0x1470960's record for the renamed address, then
        // sub_14710c0(record, 0x6e, 0, 0): bits 14-15 of the second dword
        // become 2, bits 27-29 of the fourth are cleared. The record points
        // where the buffer lives: this context's renamed copy of a dynamic
        // buffer, else the buffer itself, which GX updates with GPU copies in
        // command order (static updates, and renamed copies written back when a
        // context's map resolves, 0x2aadf50). A copy taken at the call races
        // them.
        // The renamed address is dynamic_buffer_address's chain of dependent
        // reads (the key's table, its entry, the map page's entry); the
        // buffers' chains run side by side, a level at a time.
        struct Cb {
            int j;
            std::uint64_t key, table, addr;
            std::uint32_t size;
            std::uint16_t index1;
        };
        Cb cbs[14];
        int ncb = 0;
        for (std::uint32_t m = cb_needed[k]; m; m &= m - 1) {
            const int j = __builtin_ctz(m);
            if (const std::uint64_t cb = object(group + 0x480 + 8 * static_cast<std::uint64_t>(j), 0x30)) {
                cbs[ncb++] = Cb{j, cb + 0x18, 0, 0, rd32(cb + 0x2c), 0};
            }
        }
        const std::uint64_t map = r.ctx + 0xb3c0;
        const std::uint32_t id = rd32(map + 0x18);
        const std::uint64_t pages = rd64(map + 8);
        const std::uint32_t used = rd32(map + 0x10);
        if (pages && id <= 0xf) {
            if (g_gx_cost_enabled) g_dynbuf_table.fetch_add(static_cast<std::uint64_t>(ncb), std::memory_order_relaxed);
            for (int i = 0; i < ncb; ++i) {
                cbs[i].table = rd64(cbs[i].key + 0x38);
                cbs[i].addr = rd64(cbs[i].key + 8);  // no entry: the buffer's own address
            }
            for (int i = 0; i < ncb; ++i) {
                // The buffers' per-context index tables sit in a few heaps too:
                // the last readable range, else safe_read's own checks.
                const std::uint64_t at = cbs[i].table + 2 * static_cast<std::uint64_t>(id);
                if (!cbs[i].table) continue;
                if ((at >= readable_lo && at + 2 <= readable_hi) || hle_kernel_va_readable_in(at, 2, &readable_lo, &readable_hi)) {
                    std::memcpy(&cbs[i].index1, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), 2);
                } else {
                    safe_read(at, &cbs[i].index1, 2);
                }
            }
            for (int i = 0; i < ncb; ++i) {
                const std::uint32_t index1 = cbs[i].index1;
                if (!index1 || used <= index1 - 1u) continue;
                const std::uint32_t index = index1 - 1u;
                const std::uint64_t page = rd64(pages + 8 * static_cast<std::uint64_t>(index >> 7));
                const std::uint64_t entry = page + 16 * static_cast<std::uint64_t>(index & 0x7f);
                if (page && rd64(entry) == cbs[i].key) cbs[i].addr = rd64(entry + 8);
            }
        } else {
            for (int i = 0; i < ncb; ++i) cbs[i].addr = dynamic_buffer_address(map, cbs[i].key);
        }
        for (int i = 0; i < ncb; ++i) {
            const std::uint64_t addr = cbs[i].addr;
            const std::uint32_t rec[4] = {static_cast<std::uint32_t>(addr),
                                          ((static_cast<std::uint32_t>(addr >> 32) & 0xfff) | 0x100000 | 0x8000), (cbs[i].size + 15) >> 4,
                                          0x20077fac & 0xc7ffffff};
            std::memcpy(s.obj_cb + cbs[i].j * 16, rec, sizeof(rec));
            s.obj_cb_set |= 1u << cbs[i].j;
        }
        stamp.to(kCostRecCb);
        // BBHOST_GX_COST: the constant buffers' contents at the call.
        if (hash_cbs) {
            for (int j = 0; j < 14; ++j) {
                if (!((s.obj_cb_set >> j) & 1)) continue;
                std::uint32_t rec[4];
                std::memcpy(rec, s.obj_cb + j * 16, sizeof(rec));
                const std::uint64_t addr = rec[0] | (static_cast<std::uint64_t>(rec[1] & 0xfff) << 32);
                const std::uint64_t bytes = static_cast<std::uint64_t>(rec[2]) * 16;
                if (!bytes || bytes > (1u << 20) || !hle_kernel_va_mapped(addr, bytes)) continue;
                s.cb_hash[j] = gx_cb_fnv(addr, bytes);
                s.cb_hashed = static_cast<std::uint16_t>(s.cb_hashed | (1u << j));
            }
        }
        stamp.to(kCostRecHash);
    }
    if (g_gx_repeats) {
        const bool same_ctx = t_memo_ctx == r.ctx;
        for (int k = 0; k < 2; ++k) {
            const GxStageRecords& s = out->stage[k];
            std::uint64_t part[4] = {shaders[k], 1469598103934665603ull, 1469598103934665603ull, 1469598103934665603ull};
            for (std::uint64_t m = tex_needed[k]; m; m &= m - 1) {
                const std::uint64_t view = rd64(groups[k] + 0x80 + 8 * static_cast<std::uint64_t>(__builtin_ctzll(m)));
                part[1] = (part[1] ^ view) * 1099511628211ull;
            }
            for (std::uint32_t m = smp_needed[k]; m; m &= m - 1) {
                const std::uint64_t smp = rd64(groups[k] + 8 * static_cast<std::uint64_t>(__builtin_ctz(m)));
                part[2] = (part[2] ^ smp) * 1099511628211ull;
            }
            for (std::uint32_t m = s.obj_cb_set; m; m &= m - 1) part[3] = memo_mix(part[3], s.obj_cb + __builtin_ctz(m) * 16, 16);
            if (same_ctx) {
                if (part[0] == t_memo_part[k][0]) g_memo_same_shader.fetch_add(1, std::memory_order_relaxed);
                if (part[1] == t_memo_part[k][1]) g_memo_same_views.fetch_add(1, std::memory_order_relaxed);
                if (part[2] == t_memo_part[k][2]) g_memo_same_samplers.fetch_add(1, std::memory_order_relaxed);
                if (part[3] == t_memo_part[k][3]) g_memo_same_cbs.fetch_add(1, std::memory_order_relaxed);
            }
            g_memo_stages.fetch_add(1, std::memory_order_relaxed);
            std::memcpy(t_memo_part[k], part, sizeof(part));
        }
    }
    return out;
}

std::mutex g_mu;
Stats g;
std::unordered_map<std::uint64_t, EmitRec> g_by_packet;  // by emitter write pointer
// Every emit in order, (write pointer, flip), under g_mu: a record the command
// processor has not taken kPacketMaxAge flips after its emit never will be - its
// packet was not a draw or dispatch the CP looks up, or the CP walked past it (the
// CP runs at most BBHOST_CP_BACKLOG submits, 10, behind). Without the age limit
// those records stayed until the map passed 2^20 entries: ~2 KiB each, ~20 MiB a
// minute at 60 FPS, ~2 GiB of the process before the clear.
constexpr std::uint64_t kPacketMaxAge = 600;
std::deque<std::pair<std::uint64_t, std::uint64_t>> g_packet_order;
thread_local GxRec t_last;

// GX UpdateSubresource (0x2568a20: rdi ctx, rsi ?, rdx resource, rcx subresource,
// r8 box, r9 data, ...). A static buffer (type byte +0x38 == 1, flag byte +0x3b
// value 4 clear) is written by a compute copy in command order (0x2ab3560 ->
// 0x2ab1c40); a dynamic one (value 4 set) gets a fresh copy renamed in the
// calling context (0x2aadb90), written back when its map resolves (0x2aadf50).
// Counted, and BBHOST_WATCH_UPDATE=<address> logs updates
// covering that address. The frame is f[0] r9, f[1] r8, f[2] rcx, f[3] rdx,
// f[4] rsi, f[5] rdi.
void note_update(const std::uint64_t* f) {
    const std::uint64_t res = f[3], box = f[1], data = f[0];
    std::uint8_t flags = 0, type = 0;
    if (!res || !safe_read(res + 0x3b, &flags, 1) || !safe_read(res + 0x38, &type, 1)) return;
    const std::uint64_t addr = rd64(res + 0x20);
    const std::uint32_t size = rd32(res + 0x2c);
    std::uint32_t lo = 0, hi = size;
    if (std::uint32_t b[4]; box && safe_read(box, b, sizeof(b))) {
        lo = b[0];
        hi = b[3];
    }
    // BBHOST_GX_UPLOAD=2: who updates which buffer, from flip 3000 on (past
    // the menus), the first six calls of each caller with the first floats
    // of the data - the way to find the code that builds a vertex buffer the
    // GPU then reads. Texture updates are left out.
    static const bool census2 = [] {
        const char* e = std::getenv("BBHOST_GX_UPLOAD");
        return e && e[0] == '2';
    }();
    if (census2 && ((flags & 4) || type == 1) && hle_video_flip_count() >= 3000) {
        static std::mutex mu;
        static std::map<std::uint64_t, int> per_caller;
        const std::uint64_t caller = bn(f[6]);
        int n;
        {
            std::lock_guard<std::mutex> lk(mu);
            n = per_caller[caller]++;
        }
        if (n < 6) {
            float v[12] = {};
            if (data) safe_read(data, v, sizeof(v));
            host_log("gx-update: %s from 0x%llx resource 0x%llx (home 0x%llx, %u bytes) [%u,%u) data 0x%llx: %g %g %g %g | %g %g %g %g | %g %g %g %g",
                     (flags & 4) ? "dynamic" : "static", static_cast<ull>(caller), static_cast<ull>(res), static_cast<ull>(addr), size, lo,
                     hi, static_cast<ull>(data), v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]);
        }
    }
    if (flags & 4) {
        g_updates_dynamic.add();
    } else if (type == 1) {
        g_updates_static.add();
        g_update_static_bytes.add(hi > lo ? hi - lo : 0);
    } else {
        g_updates_other.add();
        // A texture update: the size is the resource's, the box gives the
        // span within it, and the destination is where a native upload would
        // have to put the bytes.
        g_updates_tex.add();
        g_update_tex_bytes.add(hi > lo ? hi - lo : 0);
        if (g_upload_census) {
            std::lock_guard<std::mutex> lk(g_update_dst_mu);
            if (g_update_dst.size() < 4096) ++g_update_dst[addr];
        }
    }
    const std::uint64_t watch = g_watch_update;
    if (!watch || type != 1 || watch < addr + lo || watch >= addr + hi) return;
    g_updates_watched.fetch_add(1, std::memory_order_relaxed);
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) >= 24) return;
    float v[8] = {};
    if (data) safe_read(data + (watch - addr - lo), v, sizeof(v));
    host_log("gx-update: UpdateSubresource of the watched buffer 0x%llx (resource 0x%llx, %s, bytes [%u,%u)) from 0x%llx at "
             "flip %llu, thread %d: %g %g %g %g %g %g %g %g",
             static_cast<ull>(watch), static_cast<ull>(res), (flags & 4) ? "dynamic" : "static", lo, hi, static_cast<ull>(bn(f[6])),
             static_cast<ull>(hle_video_flip_count()), tid(), v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

// BBHOST_GX_NATIVE: writes a host-draw token (or, with kGxCopyMagic, a copy
// token) into the context's Gnm command buffer (ctx + 0x10: +0x8 end, +0x10 write pointer, +0x18 the refill
// callback the emitters call as (buffer, dwords, +0x20, dwords left)).
// Returns its id, 0 when it could not be written.
// `id`: one reserved with next_native_id() for a record put in its map before
// the token is written (0: a new one here). The refill this may call is the
// game's: it can submit the full chunk and wait for room in the command
// processor's backlog, so no lock the command processor takes may be held
// over it.
std::uint64_t write_native_token(std::uint64_t ctx, std::uint32_t magic = kGxNativeMagic, std::uint64_t id = 0) {
    const std::uint64_t cb = ctx + 0x10;
    std::uint64_t cmd = rd64(cb + 0x10), end = rd64(cb + 0x8);
    if (!cmd && !end) {
        // No chunk yet (a context's first write of the frame): the method
        // would take one through the same callback from its bookkeeping.
        const std::uint64_t fn = rd64(cb + 0x18), arg = rd64(cb + 0x20);
        if (fn && (hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(fn)), cb, 4, arg, 0) & 0xff)) {
            g_native_first_chunks.fetch_add(1, std::memory_order_relaxed);
            cmd = rd64(cb + 0x10);
            end = rd64(cb + 0x8);
        }
    }
    if (!cmd || end < cmd) {
        if (g_native_failed.fetch_add(1, std::memory_order_relaxed) < 8) {
            host_log("gx-native: no room for a token: context 0x%llx buffer write 0x%llx end 0x%llx begin 0x%llx refill 0x%llx",
                     static_cast<ull>(ctx), static_cast<ull>(cmd), static_cast<ull>(end), static_cast<ull>(rd64(cb)),
                     static_cast<ull>(rd64(cb + 0x18)));
        }
        return 0;
    }
    if ((end - cmd) / 4 < 4) {
        g_native_refills.fetch_add(1, std::memory_order_relaxed);
        const std::uint64_t fn = rd64(cb + 0x18), arg = rd64(cb + 0x20);
        const bool refilled =
            fn && (hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(fn)), cb, 4, arg, (end - cmd) / 4) & 0xff);
        cmd = rd64(cb + 0x10);
        end = rd64(cb + 0x8);
        if (!refilled || !cmd || end < cmd || (end - cmd) / 4 < 4) {
            g_native_failed.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
    }
    if (!id) id = next_native_id();
    const std::uint32_t token[4] = {0xC0021000u, magic, static_cast<std::uint32_t>(id), static_cast<std::uint32_t>(id >> 32)};
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(cmd)), token, sizeof(token));
    const std::uint64_t next = cmd + sizeof(token);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(cb + 0x10)), &next, sizeof(next));
    if (magic == kGxNativeMagic) g_native_tokens.add();
    return id;
}

// BBHOST_GX_NATIVE=2: the calls whose methods the host ran, by token id,
// until the CP walks the token.
struct NativeDraw {
    // What the command processor draws from, read in place: the inputs
    // build_inputs() made at the call, and the draw's GX objects
    // (build_objects). Not the rest of the call's GxRec: the
    // slot is written on a GX thread, on the main loop's critical path (the
    // game's job barrier waits for those threads), and a third of the record
    // was fields the command processor never read.
    alignas(64) GpuDrawInputs in;
    alignas(16) GxDrawObjects objects;
    GxRecordsPtr records;
    // The call, for the report on tokens that never came back (the
    // overflow map's entries only; a slot leaves them as they are).
    std::uint64_t ret = 0, ctx = 0;
    Kind kind = kKinds;
};
std::mutex g_native_mu;
std::unordered_map<std::uint64_t, NativeDraw> g_native_draws;  // under g_native_mu

// The fast path: a ring of slots indexed by token id. The recording thread
// claims a free slot (0 -> 1), fills it and publishes it (2); the command
// processor builds the draw from it in place and frees it (0). A slot still in
// use sends the call to g_native_draws instead, so no draw is lost.
struct NativeSlot {
    std::atomic<int> state{0};  // 0 free, 1 being written, 2 ready
    std::uint64_t id = 0;
    alignas(64) NativeDraw draw;  // the inputs start a line of their own
};
static_assert(std::is_trivially_copyable_v<GpuDrawInputs> && std::is_trivially_copyable_v<GxDrawObjects>);

// The GX objects behind a draw, from its record: what the command processor's
// draw reads besides the inputs (GxDrawObjects). Built where the record is
// made - the command processor used to build it at every token.
void build_objects(const GxRec& r, GxDrawObjects& o) {
    o = GxDrawObjects{};
    o.ctx = r.ctx;
    o.state = r.state;
    o.caller = bn(r.ret);
    o.call_flip = r.flip;
    for (int k = 0; k < 6; ++k) o.shader[k] = r.st[k].shader;
    o.blend_id = r.blend_id;
    o.depth_stencil_id = r.ds_id;
    o.raster_id = r.raster_id;
    o.vs_prog_id = r.vs_prog_id;
    o.ps_prog_id = r.ps_prog_id;
    o.il_id = r.il_id;
    if (r.kind <= kDrawIndexedInstanced) {
        o.geometry = true;
        o.count = static_cast<std::uint32_t>(r.arg[0]);
        const bool instanced = r.kind == kDrawInstanced || r.kind == kDrawIndexedInstanced;
        o.instances = instanced ? static_cast<std::uint32_t>(r.arg[1]) : 1;
        o.indexed = r.kind == kDrawIndexed || r.kind == kDrawIndexedInstanced;
        if (o.indexed && r.ib) {
            const std::uint64_t start = r.kind == kDrawIndexed ? r.arg[1] : r.arg[2];
            o.index_type = r.ib_fmt == 0x2a ? 1 : 0;
            o.index_va = r.ib_base + r.ib_off + (start & 0xffffffff) * (o.index_type ? 4 : 2);
            o.index_known = true;
        }
        o.fetch_va = r.op1 ? rd64(r.op1 + 0x10) : 0;
        o.vtx_ud = r.vtx_ud;
        o.vtx_valid = r.vtx_valid;
        std::memcpy(o.vtx_rec, r.vtx_rec, sizeof(r.vtx_rec));
        std::memcpy(o.vtx_obj, r.vtx_obj, sizeof(r.vtx_obj));
    } else if (r.kind == kDrawIndirect || r.kind == kDrawIndexedIndirect) {
        // 0x25699d0 / 0x2569ab0 (ctx, state, args buffer holder,
        // byte offset): SET_BASE to the holder's memory, then DRAW_INDIRECT at
        // the offset. The address was resolved at the call (r.indirect_va).
        o.geometry = true;
        o.indirect_va = r.indirect_va;
        o.indexed = r.kind == kDrawIndexedIndirect;
        if (o.indexed && r.ib) {
            o.index_type = r.ib_fmt == 0x2a ? 1 : 0;
            o.index_va = r.ib_base + r.ib_off;
            o.index_known = true;
        }
        o.fetch_va = r.op1 ? rd64(r.op1 + 0x10) : 0;
        o.vtx_ud = r.vtx_ud;
        o.vtx_valid = r.vtx_valid;
        std::memcpy(o.vtx_rec, r.vtx_rec, sizeof(r.vtx_rec));
        std::memcpy(o.vtx_obj, r.vtx_obj, sizeof(r.vtx_obj));
    }
}

// A slot comes round again only after the whole ring (16,384 draws, ~0.2 s),
// so its lines are long out of the recording thread's cache, and a plain copy
// of the 1.5 KB record stalled the recording thread on a read for ownership
// of each of them (~2% of the GX workers' time). The command processor reads
// it much later, so it is streamed past the caches instead; the caller fences
// before publishing the slot.
// The lines of a draw's records its draw reads: descriptors and set slots.
void prefetch_records(const GxDrawRecords& rec) {
    for (const GxStageRecords& s : rec.stage) {
        const auto* desc = reinterpret_cast<const char*>(s.desc);
        for (std::uint32_t off = 0; off < s.ndesc * 4u && off < sizeof(s.desc); off += 64) __builtin_prefetch(desc + off);
        for (std::uint64_t m = s.obj_tex_set; m; m &= m - 1) {
            const int j = __builtin_ctzll(m);
            __builtin_prefetch(s.obj_tex + j * 32);
            __builtin_prefetch(&s.obj_tex_id[j]);
        }
        for (std::uint32_t m = s.obj_smp_set | s.obj_smp_default; m; m &= m - 1) __builtin_prefetch(s.obj_smp + __builtin_ctz(m) * 32);
        for (std::uint32_t m = s.obj_cb_set; m; m &= m - 1) __builtin_prefetch(s.obj_cb + __builtin_ctz(m) * 16);
    }
}

constexpr std::size_t kNativeSlots = 16384;
Tally<> g_native_slot_pending;  // a gauge: +1 a stored draw, -1 a taken one
std::atomic<std::uint64_t> g_native_overflow{0};
NativeSlot& native_slot(std::uint64_t id) {
    static NativeSlot* const slots = [] {
        host_log("gx-native: token ring of %zu slots, %zu KiB", kNativeSlots, kNativeSlots * sizeof(NativeSlot) / 1024);
        return new NativeSlot[kNativeSlots];
    }();
    return slots[id & (kNativeSlots - 1)];
}
// The slot this thread's next draw will write, asked for now: its lines were
// last the command processor's, 16,384 draws ago.
void prefetch_next_slot() {
    if (t_id_next == t_id_end) return;  // the next id starts a new block
    const auto* p = reinterpret_cast<const char*>(&native_slot(t_id_next));
    for (std::size_t off = 0; off < sizeof(NativeSlot); off += 64) prefetch_for_write(p + off);
}

void* guest_fn(std::uint64_t va) {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(g_slide + (va - kPreferredGuestSlide)));
}

// A draw the host can run without its packets: records to build every
// stage's user data from (descriptor types up to 0x1a; 7 and 0x14 need the
// context block, textures in slots 0-63), a nonzero count, and an index
// buffer when indexed.
enum IneligibleWhy { kWhyRecords, kWhyArg, kWhyIndexBuffer, kWhyIndirect, kWhyMask, kWhyDesc, kWhyCount };
const char* const kIneligibleWhyName[kWhyCount] = {"no records", "count or holder 0", "no index buffer", "no indirect address",
                                                   "a stage's second resource mask", "a descriptor type above 0x1a, or 7/0x14 without the context block"};
std::atomic<std::uint64_t> g_native_ineligible_why[kWhyCount] = {};
bool native_eligible(const GxRec& r, const GxDrawRecords* records, IneligibleWhy* why) {
    const auto no = [&](IneligibleWhy w) {
        *why = w;
        return false;
    };
    if (!records) return no(kWhyRecords);
    if (!r.arg[0]) return no(kWhyArg);
    if ((r.kind == kDrawIndexed || r.kind == kDrawIndexedInstanced || r.kind == kDrawIndexedIndirect) && !r.ib) return no(kWhyIndexBuffer);
    if ((r.kind == kDrawIndirect || r.kind == kDrawIndexedIndirect) && !r.indirect_va) return no(kWhyIndirect);
    for (const GxStageRecords& st : records->stage) {
        if (st.mask[1]) return no(kWhyMask);
        for (std::uint32_t i = 0; i < st.ndesc && i < 64; ++i) {
            const std::uint32_t type = st.desc[i] & 0xff;
            if (type > 0x1a) return no(kWhyDesc);
            if ((type == 0x7 || type == 0x14) && !st.has_ctx_block) return no(kWhyDesc);
        }
    }
    return true;
}

// Whether the output-merger flush (0x2ad2230) may run a deferred clear. It
// clears a target it recorded as bound (in the record set that bit 0 of +0x188
// selects for colour, bit 1 for depth) that the state no longer binds, when the
// target's texture has a clear pending: +0x3b & 0x8b for colour (then
// 0x2ad1f80 and 0x2ab5a70), & 8 for depth (0x2ab6810). The flush checks a few
// more of the target's fields before clearing; this errs towards running it.
bool om_flush_may_clear(std::uint64_t om, std::uint64_t state) {
    const std::uint32_t sets = rd32(om + 0x188);
    const auto replaced = [](std::uint64_t recorded, std::uint64_t bound) {
        return recorded != bound && recorded + 1 >= 2;
    };
    const std::uint32_t colour_targets = rd32(om + 0x146) & 0xff;
    for (std::uint32_t i = 0; i < colour_targets; ++i) {
        const std::uint64_t rt = rd64(om + (sets & 1) * 0x48 + 0x60 + i * 8);
        if (!replaced(rt, rd64(state + 0x4e0 + i * 8))) continue;
        const std::uint64_t tex = rd64(rt + 8);
        if (tex && (rd32(tex + 0x3b) & 0x8b)) return true;
    }
    const std::uint64_t ds = rd64(om + ((sets >> 1) & 1) * 0x48 + 0xa0);
    if (replaced(ds, rd64(state + 0x520))) {
        const std::uint64_t tex = rd64(ds + 8);
        if (tex && (rd32(tex + 0x3b) & 8)) return true;
    }
    return false;
}

// BBHOST_GX_NATIVE_REGS=1: count the packets a guest call wrote between two
// write pointers in one chunk.
void count_native_packets(NativeCall call, std::uint64_t from, std::uint64_t to) {
    for (std::uint64_t p = from; p + 4 <= to;) {
        const std::uint32_t h = rd32(p);
        if ((h >> 30) != 3) {
            g_native_call_ops[call][0xff].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const std::uint32_t n = ((h >> 16) & 0x3fff) + 1;
        const std::uint32_t op = (h >> 8) & 0xff;
        if (op == 0x69 && n >= 2) {
            const std::uint32_t reg = rd32(p + 4) & 0x3ff;
            for (std::uint32_t k = 0; k + 1 < n && reg + k < 0x400; ++k) {
                g_native_call_regs[call][reg + k].fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            g_native_call_ops[call][op].fetch_add(1, std::memory_order_relaxed);
        }
        p += 4 + 4ull * n;
    }
}

// BBHOST_GX_NATIVE=2: a draw method's work without its commit and draw
// packets. The three flushes are skipped (g_native_all_flushes runs them as
// the method does) and so are the output-merger commit and flush
// (g_native_om); the post-draw (0x2acbd50, which folds the flushes' dirty
// bits into the flags the next commit reads) and the command-buffer
// bookkeeping around them run as the method runs them. One token takes the
// place of the commit, start vertex, instance count and draw packet. The
// commit's caches then still describe what the register file holds, so the
// next PM4-path draw emits what differs. False when the draw does not qualify
// or no token could be written; the method then runs.
// BBHOST_GX_COST: how often a draw's records and inputs are what the previous
// draw on its context built - what a per-context memo of them would save the
// GX threads, which the game's job barrier waits for.
std::uint64_t records_signature(const GxDrawRecords& rec) {
    std::uint64_t h = 1469598103934665603ull;
    for (const GxStageRecords& s : rec.stage) {
        h = memo_mix(h, &s.ndesc, sizeof(s.ndesc));
        h = memo_mix(h, s.mask, sizeof(s.mask));
        h = memo_mix(h, &s.obj_tex_set, sizeof(s.obj_tex_set));
        h = memo_mix(h, &s.obj_smp_set, sizeof(s.obj_smp_set));
        h = memo_mix(h, &s.obj_cb_set, sizeof(s.obj_cb_set));
        h = memo_mix(h, &s.obj_smp_default, sizeof(s.obj_smp_default));
        h = memo_mix(h, &s.desc_fp, sizeof(s.desc_fp));
        for (std::uint64_t m = s.obj_tex_set; m; m &= m - 1) {
            const int j = __builtin_ctzll(m);
            h = memo_mix(h, s.obj_tex + j * 32, 32);
            h = memo_mix(h, &s.obj_tex_id[j], 4);
        }
        for (std::uint32_t m = s.obj_smp_set; m; m &= m - 1) h = memo_mix(h, s.obj_smp + __builtin_ctz(m) * 32, 32);
        for (std::uint32_t m = s.obj_smp_default; m; m &= m - 1) h = memo_mix(h, s.obj_smp + __builtin_ctz(m) * 32, 16);
        for (std::uint32_t m = s.obj_cb_set; m; m &= m - 1) h = memo_mix(h, s.obj_cb + __builtin_ctz(m) * 16, 16);
        if (s.has_ctx_block) h = memo_mix(h, s.ctx_block, sizeof(s.ctx_block));
    }
    return h;
}

bool native_draw_call(GxRec& r) {
    GxCostTimer timer(kCostNative);
    GxCostStamp stamp;
    GxRecordsPtr records = snapshot_records(r);
    stamp.to(kCostNativeRecords);
    IneligibleWhy why = kWhyRecords;
    if (!native_eligible(r, records.get(), &why)) {
        g_native_ineligible.add();
        g_native_ineligible_why[why].fetch_add(1, std::memory_order_relaxed);
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1, std::memory_order_relaxed) < 6) {
            host_log("gx-native: %s from 0x%llx not qualified for a token: %s (holder 0x%llx, indirect 0x%llx); with the register file not "
                     "kept its packets draw nothing",
                     kKindName[r.kind], static_cast<ull>(bn(r.ret)), kIneligibleWhyName[why], static_cast<ull>(r.arg[0]),
                     static_cast<ull>(r.indirect_va));
        }
        return false;
    }
    if (g_gx_repeats) {
        const std::uint64_t rh = records_signature(*records), ih = memo_mix(1469598103934665603ull, &r.in, sizeof(r.in));
        if (t_memo_ctx == r.ctx) {
            const bool same_r = rh == t_memo_records, same_i = ih == t_memo_inputs;
            if (same_r) g_memo_same_records.fetch_add(1, std::memory_order_relaxed);
            if (same_i) g_memo_same_inputs.fetch_add(1, std::memory_order_relaxed);
            if (same_r && same_i) g_memo_same_both.fetch_add(1, std::memory_order_relaxed);
        }
        g_memo_draws.fetch_add(1, std::memory_order_relaxed);
        t_memo_ctx = r.ctx;
        t_memo_records = rh;
        t_memo_inputs = ih;
    }
    const bool indexed = r.kind == kDrawIndexed || r.kind == kDrawIndexedInstanced || r.kind == kDrawIndexedIndirect;
    // What each guest call still writes into the Gnm stream: the command
    // buffer's write pointer (at ctx + 0x10 + 0x10) before and after it.
    const std::uint64_t write_ptr = r.ctx + 0x20;
    std::uint64_t wp = rd64(write_ptr);
    const auto written = [&](NativeCall call) {
        const std::uint64_t now = rd64(write_ptr);
        if (now >= wp && now - wp < (1u << 20)) {
            g_native_call_dwords.add((now - wp) / 4, call);
            if (g_native_regs) count_native_packets(call, wp, now);
        } else {
            g_native_call_switches.add();
        }
        wp = now;
    };
    if (g_native_all_flushes || (g_native_om < 2 && om_flush_may_clear(r.ctx + 0xb558, r.state))) {
        g_native_om_flushes.add();
        hle_call_guest(guest_fn(0x2ad2230), r.ctx + 0xb558, r.ctx, r.state);
        written(kNcOmFlush);
    }
    if (g_native_all_flushes) {
        hle_call_guest(guest_fn(0x2abba30), r.ctx + 0xb518, r.ctx, r.state, indexed ? 1 : 0);
        written(kNcIaFlush);
        hle_call_guest(guest_fn(0x2ad3370), r.ctx + 0xb528, r.ctx, r.state);
        written(kNcRsFlush);
    }
    hle_call_guest(guest_fn(0x2aaf700), r.ctx + 0x10);
    written(kNcBookkeeping);
    if (g_native_om == 1) {
        hle_call_guest(guest_fn(0x2ad2790), r.ctx + 0xb558, r.ctx, r.state, 0);
        written(kNcOmCommit);
    }
    stamp.to(kCostNativeGuest);
    if (g_gnm_sources) note_source(r.ctx + 0x10, rd64(write_ptr), rd64(r.ctx + 0x18), kSourceNative);
    const std::uint64_t id = write_native_token(r.ctx);
    wp = rd64(write_ptr);
    if (!id) {
        // The method runs its flushes, which write whatever differs from their caches.
        g_native_late.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    NativeSlot& slot = native_slot(id);
    int free_state = 0;
    GxDrawObjects objects;
    build_objects(r, objects);
    if (slot.state.compare_exchange_strong(free_state, 1, std::memory_order_acquire)) {
        // Plain stores: the previous draw on this thread asked for these lines
        // (prefetch_next_slot), and the release below orders them before the
        // slot reads ready. Streamed past the caches, they waited on an sfence
        // for the memory writes (~5% of the GX threads' host time).
        slot.id = id;
        slot.draw.in = r.in;
        slot.draw.objects = objects;
        slot.draw.records = std::move(records);
        slot.state.store(2, std::memory_order_release);
        g_native_slot_pending.add();
        prefetch_next_slot();
    } else {
        g_native_overflow.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(g_native_mu);
        if (g_native_draws.size() > 65536) g_native_draws.clear();
        NativeDraw& d = g_native_draws[id];
        d.in = r.in;
        d.objects = objects;
        d.records = std::move(records);
        d.ret = r.ret;
        d.ctx = r.ctx;
        d.kind = r.kind;
    }
    stamp.to(kCostNativeStore);
    hle_call_guest(guest_fn(0x2acbd50), r.ctx + 0x50, r.ctx);
    written(kNcPostDraw);
    hle_call_guest(guest_fn(0x2aaf700), r.ctx + 0x10);
    written(kNcBookkeepingAfter);
    stamp.to(kCostNativeGuest);
    return true;
}

// The user-data dwords a stage's descriptors write, as build_gx_user_data
// does: records 4 dwords (8 for types 1 and 3), type 6 one, pointers two.
std::uint32_t named_user_dwords(const GxStageRecords& st) {
    std::uint32_t mask = 0;
    for (std::uint32_t i = 0; i < st.ndesc && i < 64; ++i) {
        const std::uint32_t type = st.desc[i] & 0xff, ud = (st.desc[i] >> 8) & 0xff;
        if (ud >= 16) continue;
        const std::uint32_t n = type == 0x1 || type == 0x3 ? 8 : type <= 0x5 ? 4 : type == 0x6 ? 1 : 2;
        mask |= ((1u << n) - 1) << ud;
    }
    return mask & 0xffff;
}

// BBHOST_GX_NATIVE=1: a draw's records taken with its token, at the call,
// for its draw packet's emitter (same thread) to compare with the records
// after the commit: whether a native draw, which never runs the commit, can
// take them at the call.
thread_local GxRecordsPtr t_call_records;
enum RecordKind { kRkDesc, kRkTex, kRkSmp, kRkCb, kRkKinds };
std::atomic<std::uint64_t> g_native_records_compared{0};
std::atomic<std::uint64_t> g_native_records_changed[kRkKinds] = {};

void compare_call_records(const GxDrawRecords& call, const GxDrawRecords& emit, const GxRec& r) {
    g_native_records_compared.fetch_add(1, std::memory_order_relaxed);
    static std::atomic<int> logs{0};
    const auto note = [&](RecordKind kind, int stage, int slot, const std::uint8_t* a, const std::uint8_t* b) {
        g_native_records_changed[kind].fetch_add(1, std::memory_order_relaxed);
        if (logs.fetch_add(1) >= 16) return;
        std::uint32_t x[2] = {}, y[2] = {};
        if (a) std::memcpy(x, a, 8);
        if (b) std::memcpy(y, b, 8);
        static const char* const kNames[kRkKinds] = {"descriptors or masks", "texture", "sampler", "constant buffer"};
        host_log("gx-native: %s %s slot %d differs between %s from 0x%llx and its draw packet: %s%08x %08x -> %s%08x %08x",
                 stage ? "PS" : "VS", kNames[kind], slot, kKindName[r.kind], static_cast<ull>(bn(r.ret)), a ? "" : "(none) ", x[0], x[1],
                 b ? "" : "(none) ", y[0], y[1]);
    };
    for (int k = 0; k < 2; ++k) {
        const GxStageRecords& a = call.stage[k];
        const GxStageRecords& b = emit.stage[k];
        if (a.ndesc != b.ndesc || std::memcmp(a.desc, b.desc, a.ndesc * 4u) != 0 || std::memcmp(a.mask, b.mask, sizeof(a.mask)) != 0) {
            note(kRkDesc, k, -1, nullptr, nullptr);
        }
        for (int j = 0; j < 64; ++j) {
            const bool sa = (a.obj_tex_set >> j) & 1, sb = (b.obj_tex_set >> j) & 1;
            const std::uint8_t* ra = a.obj_tex + j * 32, *rb = b.obj_tex + j * 32;
            if (sa != sb || (sa && std::memcmp(ra, rb, 32) != 0)) note(kRkTex, k, j, sa ? ra : nullptr, sb ? rb : nullptr);
        }
        for (int j = 0; j < 16; ++j) {
            // An object's 32 bytes, or the 16-byte default cache record for an empty slot.
            const bool oa = (a.obj_smp_set >> j) & 1, ob = (b.obj_smp_set >> j) & 1;
            const std::uint8_t* ra = a.obj_smp + j * 32, *rb = b.obj_smp + j * 32;
            if (oa != ob || std::memcmp(ra, rb, oa ? 32 : 16) != 0) note(kRkSmp, k, j, ra, rb);
        }
        for (int j = 0; j < 14; ++j) {
            const bool sa = (a.obj_cb_set >> j) & 1, sb = (b.obj_cb_set >> j) & 1;
            const std::uint8_t* ra = a.obj_cb + j * 16, *rb = b.obj_cb + j * 16;
            if (sa != sb || (sa && std::memcmp(ra, rb, 16) != 0)) note(kRkCb, k, j, sa ? ra : nullptr, sb ? rb : nullptr);
        }
    }
}

void report_native() {
    const auto v = [](const auto& a) { return static_cast<ull>(a.load()); };
    if (g_tess_shadowed.load()) {
        // What a tessellated draw's own packets set after its token,
        // which is what a native one would have to carry from its GX objects.
        const std::uint64_t n = std::max<std::uint64_t>(1, g_tess_matched.load());
        std::string what;
        for (int k = 0; k < kTessFields; ++k) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), " %s %.1f%%;", kTessFieldName[k],
                          100.0 * static_cast<double>(g_tess_changed[k].load()) / static_cast<double>(n));
            what += buf;
        }
        host_log("gx-native: %llu tessellated draws shadowed, %llu matched at their draw, %llu without one; what their own packets "
                 "set after the token:%s",
                 v(g_tess_shadowed), v(g_tess_matched), v(g_tess_unmatched), what.c_str());
        host_log("gx-native: of the shadowed tessellated draws, %llu had inputs built from GX (%llu passed build_inputs)",
                 v(g_tess_have_inputs), v(g_tess_in_ok));
        if (g_tess_built[0].load()) {
            std::string built;
            for (int k = 0; k < kBuiltFields; ++k) {
                char b[96];
                std::snprintf(b, sizeof(b), " %s %llu of %llu;", kBuiltFieldName[k], static_cast<ull>(g_tess_built_bad[k].load()),
                              static_cast<ull>(g_tess_built[k].load()));
                built += b;
            }
            host_log("gx-native: built from the GX objects against the registers the draw wrote, differing:%s", built.c_str());
        }
    }
    host_log("gx-native: tokens %llu (refills %llu, failed %llu), walked %llu, matched at the draw %llu, not matched %llu; "
             "changed between token and draw: PS input %llu, DB_RENDER_CONTROL %llu, depth clear %llu, stencil clear %llu, "
             "generic scissor %llu, viewport scissor %llu, PA_SC_MODE_CNTL_0 %llu, named user data %llu, unnamed user data %llu",
             v(g_native_tokens), v(g_native_refills), v(g_native_failed), v(g_native_seen), v(g_native_matched), v(g_native_unmatched),
             v(g_native_changed[kNfPsInput]), v(g_native_changed[kNfRenderControl]), v(g_native_changed[kNfDepthClear]),
             v(g_native_changed[kNfStencilClear]), v(g_native_changed[kNfGenericScissor]), v(g_native_changed[kNfVportScissor]),
             v(g_native_changed[kNfModeCntl]), v(g_native_changed[kNfUserNamed]), v(g_native_changed[kNfUserUnnamed]));
    host_log("gx-native: records compared at the call and after the commit %llu; slots that differ: descriptors or masks %llu, "
             "textures %llu, samplers %llu, constant buffers %llu",
             v(g_native_records_compared), v(g_native_records_changed[kRkDesc]), v(g_native_records_changed[kRkTex]),
             v(g_native_records_changed[kRkSmp]), v(g_native_records_changed[kRkCb]));
    std::string slots[2];
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 16; ++j) slots[k] += " " + std::to_string(v(g_native_unnamed_slot[k][j]));
    }
    host_log("gx-native: unnamed user data changed by slot, VS:%s; PS:%s", slots[0].c_str(), slots[1].c_str());
}

// BBHOST_GX_NATIVE_COPIES=0 (checks): the GX buffer-copy passes dispatch their
// compute copies as before. =2 (checks) writes the copy token and still runs
// the pass, so the host copy lands just before the compute one. By default, with native draws on, a call of either
// pass writes a copy token where its compute packets would go and records its
// copies; at the token the command processor runs them (host_gpu_copy_guest)
// in the same place in the command stream: a render-target source as an image
// copy, other memory as buffer copies. The pass
// does not run: the compute stage's bindings, their caches and the register
// file keep what they had, and no range table is allocated.
const int g_native_copies = [] {
    const char* e = std::getenv("BBHOST_GX_NATIVE_COPIES");
    return e ? std::atoi(e) : 1;
}();
// BBHOST_GX_COPY_KINDS=batch or single (checks): replace only that pass.
const int g_copy_kinds = [] {
    const char* e = std::getenv("BBHOST_GX_COPY_KINDS");
    if (e && std::strcmp(e, "batch") == 0) return 1;
    if (e && std::strcmp(e, "single") == 0) return 2;
    return 3;
}();
struct GuestCopy {
    std::uint64_t dst;
    std::uint64_t src;
    std::uint64_t bytes;
    bool back = false;  // a renamed dynamic buffer copied back to its own address (host_gpu_copy_back)
};
std::mutex g_copies_mu;
std::unordered_map<std::uint64_t, std::vector<GuestCopy>> g_copies;  // by token id, until the CP walks the token
std::atomic<std::uint64_t> g_copy_calls{0}, g_copy_tokens{0}, g_copy_ranges{0}, g_copy_bytes{0}, g_copy_failed{0},
    g_copy_unmatched{0}, g_copy_unwritten{0};

// The frame holds r9 r8 rcx rdx rsi rdi. 0x2ab2810(buffer, usage, usage,
// entries, count) takes 0x18-byte entries {+0x0 destination, +0x8 source,
// +0x10 size} from the map resolves 0x2aadf50 / 0x2ad1090, which copy renamed
// dynamic buffers back to their own addresses. 0x2ab2670(buffer, destination,
// usage, source, usage, size) is one such copy, and CopyResource's. Both
// shaders copy size >> 2 whole dwords.
std::uint64_t native_copy(Kind kind, const std::uint64_t* f) {
    if (!g_native_copies || g_native != 2 || g_backend != 1) return 0;
    if (!(g_copy_kinds & (kind == kCopyBuffers ? 1 : 2))) return 0;
    const std::uint64_t cb = f[5];
    std::vector<GuestCopy> copies;
    if (kind == kCopyBuffers) {
        const std::uint64_t entries = f[2];
        const std::uint32_t count = static_cast<std::uint32_t>(f[1]);
        if (!entries || !count) return 0;
        copies.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::uint64_t e = entries + 0x18ull * i;
            copies.push_back({rd64(e), rd64(e + 8), (rd32(e + 0x10) >> 2) * 4ull, true});
        }
    } else {
        copies.push_back({f[4], f[2], ((f[0] >> 2) & 0xffffffffull) * 4});
    }
    if (g_copy_calls.fetch_add(1, std::memory_order_relaxed) < 8) {
        host_log("gx-copy: %s from 0x%llx: %zu copies, first 0x%llx -> 0x%llx, %llu bytes", kKindName[kind], static_cast<ull>(bn(f[6])),
                 copies.size(), static_cast<ull>(copies[0].src), static_cast<ull>(copies[0].dst), static_cast<ull>(copies[0].bytes));
    }
    if (g_gnm_sources) note_source(cb, rd64(cb + 0x10), rd64(cb + 0x8), kSourceCopy);
    const std::uint64_t id = write_native_token(cb - 0x10, kGxCopyMagic);
    if (!id) {
        g_copy_unwritten.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    {
        std::lock_guard<std::mutex> lk(g_copies_mu);
        if (g_copies.size() > 65536) g_copies.clear();
        g_copies[id] = std::move(copies);
    }
    g_copy_tokens.fetch_add(1, std::memory_order_relaxed);
    return g_native_copies == 2 ? 0 : 1;
}

// GX's fills as fill tokens. 0x2ab5380(ctx, resource, offset,
// bytes, value*) is the fill primitive (a 16-byte value over
// resource memory + offset, through the fill shader and one dispatch): the
// depth-stencil clear pass fills HTILE with it, the colour clear fills a
// whole resource with it. 0x2ab4ef0(ctx, view, resource, value*) clears the
// view's slices otherwise, one dispatch a slice over the subresource's bytes
// (0x2abaf60 gives their offset and size from the resource's layout, and is
// called here for the same numbers). Each call writes one token and records
// its fills; the command processor runs host_gpu_fill at the token - the
// same work the fill recogniser did after walking the dispatch, without the
// dispatch, its scratch memcpy, the shader bind or the compute pipeline.
// BBHOST_GX_NATIVE_FILLS=0 runs the methods as before.
struct GuestFill {
    std::uint64_t va;
    std::uint64_t bytes;
    float rgba[4];
    // The clear pass's depth and stencil, where the
    // fill primitive's tokens take DB_DEPTH_CLEAR / DB_STENCIL_CLEAR at the CP.
    bool clear_known = false;
    std::uint32_t depth_clear = 0, stencil_clear = 0;
    // A depth target without HTILE (the clear pass's quad wrote it): `va` is
    // the depth surface, rgba[0] the HTILE word for the depth.
    bool depth_target = false;
};
const bool g_native_fills = [] {
    const char* e = std::getenv("BBHOST_GX_NATIVE_FILLS");
    return !e || e[0] != '0';
}();
std::mutex g_fills_mu;
std::unordered_map<std::uint64_t, std::vector<GuestFill>> g_fills;  // by token id, until the CP walks the token
std::atomic<std::uint64_t> g_fill_calls{0}, g_fill_tokens{0}, g_fill_ranges{0}, g_fill_bytes{0}, g_fill_unmatched{0}, g_fill_unwritten{0},
    g_fill_rtv_calls{0}, g_fill_rtv_whole{0}, g_fill_rtv_layout_failed{0};

std::uint64_t fill_token(std::uint64_t ctx, std::vector<GuestFill>&& fills) {
    if (g_gnm_sources) note_source(ctx + 0x10, rd64(ctx + 0x20), rd64(ctx + 0x18), kSourceCopy);
    const std::uint64_t id = write_native_token(ctx, kGxFillMagic);
    if (!id) {
        g_fill_unwritten.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    {
        std::lock_guard<std::mutex> lk(g_fills_mu);
        if (g_fills.size() > 65536) g_fills.clear();
        g_fills[id] = std::move(fills);
    }
    g_fill_tokens.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

// f: rdi ctx, rsi resource, rdx offset, rcx bytes, r8 value*.
std::uint64_t native_fill(const std::uint64_t* f) {
    if (!g_native_fills || g_native != 2 || g_backend != 1) return 0;
    const std::uint64_t ctx = f[5], res = f[4], value = f[1];
    const std::uint64_t memory = res ? rd64(res + 8) : 0;
    if (!memory || !value) return 0;
    GuestFill fill{};
    fill.va = memory + static_cast<std::uint32_t>(f[3]);
    fill.bytes = (f[2] + 15) & ~15ull;  // the dispatch covers whole 16-byte records
    if (!safe_read(value, fill.rgba, sizeof(fill.rgba))) return 0;
    g_fill_calls.fetch_add(1, std::memory_order_relaxed);
    std::vector<GuestFill> fills{fill};
    return fill_token(ctx, std::move(fills));
}

// f: rdi ctx, rsi view, rdx resource, rcx value*. A view over the whole
// resource (0x1486d60 nonzero) fills it through 0x2ab5380, which the hook
// above takes; the per-slice path is taken here.
std::uint64_t native_clear_rtv(const std::uint64_t* f) {
    if (!g_native_fills || g_native != 2 || g_backend != 1) return 0;
    const std::uint64_t ctx = f[5], view = f[4], res = f[3], value = f[2];
    if (!view || !res || !value) return 0;
    if (hle_call_guest<std::int64_t>(guest_fn(0x1486d60), view) & 0xff) {
        g_fill_rtv_whole.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const std::uint32_t dim = rd32(view + 0x30);
    if (dim - 1 > 7) return 0;  // the method does nothing either
    const std::uint32_t first_mip = rd32(view + 0x34), first_slice = rd32(view + 0x38), slices = rd32(view + 0x3c);
    const std::uint32_t mips = rd32(res + 0x32) & 0xff;
    const std::uint64_t memory = rd64(res + 8);
    if (!memory || !mips || !slices || slices > 4096) return 0;
    float rgba[4];
    if (!safe_read(value, rgba, sizeof(rgba))) return 0;
    std::vector<GuestFill> fills;
    fills.reserve(slices);
    for (std::uint32_t s = first_slice; s < first_slice + slices; ++s) {
        const std::uint32_t subresource = first_mip + s * mips;
        alignas(16) std::uint64_t off = 0, bytes = 0;
        const std::int64_t rc = hle_call_guest<std::int64_t>(guest_fn(0x2abaf60), &off, &bytes, res, subresource, 0, 0);
        if (rc & 0xffffffff) {
            g_fill_rtv_layout_failed.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        GuestFill fill{};
        fill.va = memory + off;
        fill.bytes = bytes;
        std::memcpy(fill.rgba, rgba, sizeof(rgba));
        fills.push_back(fill);
    }
    g_fill_rtv_calls.fetch_add(1, std::memory_order_relaxed);
    return fill_token(ctx, std::move(fills));
}

// The texture upload 0x2ab35e0(ctx, resource, data{ptr, row
// pitch at +8, slice pitch at +0xc}, subresource, box*, flag) as an upload
// token. The guest chunks the rows through scratch and tiles them into its
// memory with a dispatch a chunk; here the rows are copied out (the source
// is the caller's, gone after the call) and the command processor copies
// them into the image at the token (host_gpu_upload_region). 2D textures the
// registry has a T# for (engine/gx_resources.h) only, so the surface can be
// made at the token when the cache has none. BBHOST_GX_NATIVE_UPLOADS=0 runs
// the method.
struct GuestUpload {
    std::uint64_t base;
    std::uint32_t tsharp[8];
    std::uint32_t mip, layer, x, y, w, h, row_bytes;
    std::vector<std::uint8_t> data;
};
const bool g_native_uploads = [] {
    const char* e = std::getenv("BBHOST_GX_NATIVE_UPLOADS");
    return !e || e[0] != '0';
}();
std::mutex g_uploads_mu;
std::unordered_map<std::uint64_t, GuestUpload> g_uploads;  // by token id, until the CP walks the token
std::atomic<std::uint64_t> g_upload_calls{0}, g_upload_tokens{0}, g_upload_bytes{0}, g_upload_not_2d{0}, g_upload_no_tsharp{0},
    g_upload_bad_args{0}, g_upload_unmatched{0}, g_upload_unwritten{0}, g_upload_by_type[8] = {};

// The DXGI block-compressed formats (0x46..0x63 as GX numbers them) and their
// block size: BC1/BC4 8 bytes a 4x4 block, the rest 16.
bool dxgi_bc(std::uint32_t fmt, std::uint32_t* bytes_per_block) {
    if (fmt < 0x46 || fmt > 0x63) return false;
    const std::uint32_t k = fmt - 0x46;
    // BC1 (0x46-0x48) and BC4 (0x4f-0x51) are 8-byte blocks: bits 0-2 and 9-11.
    *bytes_per_block = ((0x0e07u >> k) & 1) ? 8 : 16;
    return true;
}

std::uint64_t native_upload(const std::uint64_t* f) {
    if (!g_native_uploads || g_native != 2 || g_backend != 1) return 0;
    // rdi is the command buffer (ctx + 0x10): the method's bookkeeping and
    // dirty flags address it as such (ctx + 0xb478 is its +0xb468).
    const std::uint64_t ctx = f[5] - 0x10, res = f[4], desc = f[3], box = f[1];
    const std::uint32_t subresource = static_cast<std::uint32_t>(f[2]);
    if (!res || !desc) return 0;
    g_upload_calls.fetch_add(1, std::memory_order_relaxed);
    std::uint8_t type = 0, mips = 0, fmt = 0;
    std::uint16_t width = 0, height = 0;
    std::uint64_t memory = 0;
    if (!safe_read(res + 0x20, &type, 1) || !safe_read(res + 0x32, &mips, 1) || !safe_read(res + 0x33, &fmt, 1) ||
        !safe_read(res + 0x2c, &width, 2) || !safe_read(res + 0x2e, &height, 2) || !safe_read(res + 8, &memory, 8)) {
        return 0;
    }
    static const std::uint64_t probe = [] {
        const char* e = std::getenv("BBHOST_TEX_PROBE");
        return e ? std::strtoull(e, nullptr, 0) : 0ull;
    }();
    if (probe && memory == probe) {
        std::uint64_t data = 0;
        std::uint32_t pitch = 0;
        safe_read(desc, &data, 8);
        safe_read(desc + 8, &pitch, 4);
        std::uint64_t head[2] = {};
        if (data) safe_read(data, head, sizeof(head));
        host_log("texture probe: 0x%llx GX upload call: type %u %ux%u mips %u format %u subresource %u box %s, rows at 0x%llx pitch %u: "
                 "%016llx %016llx (flip %llu)",
                 static_cast<unsigned long long>(memory), type, width, height, mips, fmt, subresource, box ? "given" : "none",
                 static_cast<unsigned long long>(data), pitch, static_cast<unsigned long long>(head[0]), static_cast<unsigned long long>(head[1]),
                 static_cast<unsigned long long>(hle_video_flip_count()));
    }
    if ((type != 2 && type != 3) || !mips || !memory) {
        g_upload_not_2d.fetch_add(1, std::memory_order_relaxed);
        if (type < 8) g_upload_by_type[type].fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    if (type == 2) height = 1;  // a 1D texture: one row
    GuestUpload up{};
    up.base = memory;
    if (!gx_resource_tsharp(memory, up.tsharp)) {
        g_upload_no_tsharp.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    up.mip = subresource % mips;
    up.layer = subresource / mips;
    const std::uint32_t mw = std::max(1u, static_cast<std::uint32_t>(width) >> up.mip), mh = std::max(1u, static_cast<std::uint32_t>(height) >> up.mip);
    up.x = 0;
    up.y = 0;
    up.w = mw;
    up.h = mh;
    if (box) {
        std::uint32_t b[6];
        if (!safe_read(box, b, sizeof(b)) || b[3] < b[0] || b[4] < b[1] || b[3] > mw || b[4] > mh || b[5] - b[2] > 1) {
            g_upload_bad_args.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        up.x = b[0];
        up.y = b[1];
        up.w = b[3] - b[0];
        up.h = b[4] - b[1];
    }
    if (!up.w || !up.h) return 0;
    std::uint32_t block = 1, bpe = 0;
    if (dxgi_bc(fmt, &bpe)) {
        block = 4;
    } else {
        bpe = static_cast<std::uint32_t>(hle_call_guest<std::int64_t>(guest_fn(0x2564ed0), fmt));  // bytes per element
    }
    if (!bpe || bpe > 32) {
        g_upload_bad_args.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const std::uint32_t wb = (up.w + block - 1) / block, hb = (up.h + block - 1) / block;
    up.row_bytes = wb * bpe;
    std::uint64_t data = 0;
    std::uint32_t row_pitch = 0;
    if (!safe_read(desc, &data, 8) || !safe_read(desc + 8, &row_pitch, 4) || !data) return 0;
    if (!row_pitch) row_pitch = up.row_bytes;
    if (row_pitch < up.row_bytes || !hle_kernel_va_mapped(data, static_cast<std::size_t>(row_pitch) * (hb - 1) + up.row_bytes)) {
        g_upload_bad_args.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    up.data.resize(static_cast<std::size_t>(up.row_bytes) * hb);
    for (std::uint32_t r = 0; r < hb; ++r) {
        std::memcpy(up.data.data() + static_cast<std::size_t>(r) * up.row_bytes,
                    reinterpret_cast<const void*>(static_cast<std::uintptr_t>(data + static_cast<std::uint64_t>(r) * row_pitch)), up.row_bytes);
    }
    if (g_gnm_sources) note_source(ctx + 0x10, rd64(ctx + 0x20), rd64(ctx + 0x18), kSourceCopy);
    const std::uint64_t id = write_native_token(ctx, kGxUploadMagic);
    if (!id) {
        g_upload_unwritten.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    g_upload_bytes.fetch_add(up.data.size(), std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(g_uploads_mu);
        if (g_uploads.size() > 65536) g_uploads.clear();
        g_uploads[id] = std::move(up);
    }
    g_upload_tokens.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

// The texture copy 0x2ab2f40(ctx, dst resource, dst subresource,
// dst x, y, z, src resource, src subresource, src box*) as a copy-image
// token. The method built a T# a side and dispatched the copy shader for the
// destination's type (8x8 groups for 2D); the box is clamped to the source
// mip and the destination's room as the method clamps it. Taken for 1D and
// 2D textures the registry has T#s for, with the same format (T# dfmt and
// nfmt); anything else runs the method. BBHOST_GX_NATIVE_IMAGE_COPIES=0 runs
// it always.
const bool g_native_image_copies = [] {
    const char* e = std::getenv("BBHOST_GX_NATIVE_IMAGE_COPIES");
    return !e || e[0] != '0';
}();
std::mutex g_image_copies_mu;
std::unordered_map<std::uint64_t, GpuImageCopy> g_image_copies;  // by token id, until the CP walks the token
std::atomic<std::uint64_t> g_icopy_calls{0}, g_icopy_tokens{0}, g_icopy_not_2d{0}, g_icopy_no_tsharp{0}, g_icopy_format{0}, g_icopy_bad{0},
    g_icopy_unmatched{0}, g_icopy_unwritten{0};

std::uint64_t native_copy_texture(const std::uint64_t* f) {
    if (!g_native_image_copies || g_native != 2 || g_backend != 1) return 0;
    const std::uint64_t ctx = f[5], dst = f[4], src = f[7], box = f[9];
    const std::uint32_t dst_sub = static_cast<std::uint32_t>(f[3]), src_sub = static_cast<std::uint32_t>(f[8]);
    const std::uint32_t dst_x = static_cast<std::uint32_t>(f[2]), dst_y = static_cast<std::uint32_t>(f[1]);
    if (!dst || !src) return 0;
    g_icopy_calls.fetch_add(1, std::memory_order_relaxed);
    struct Res {
        std::uint8_t type, mips;
        std::uint16_t w, h;
        std::uint64_t memory;
    };
    const auto read_res = [](std::uint64_t r, Res* o) {
        return safe_read(r + 0x20, &o->type, 1) && safe_read(r + 0x32, &o->mips, 1) && safe_read(r + 0x2c, &o->w, 2) &&
               safe_read(r + 0x2e, &o->h, 2) && safe_read(r + 8, &o->memory, 8);
    };
    Res d{}, s{};
    if (!read_res(dst, &d) || !read_res(src, &s)) return 0;
    if ((d.type != 2 && d.type != 3) || (s.type != 2 && s.type != 3) || !d.mips || !s.mips || !d.memory || !s.memory) {
        g_icopy_not_2d.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    if (d.type == 2) d.h = 1;
    if (s.type == 2) s.h = 1;
    GpuImageCopy c{};
    if (!gx_resource_tsharp(d.memory, c.dst_tsharp) || !gx_resource_tsharp(s.memory, c.src_tsharp)) {
        g_icopy_no_tsharp.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    if (((c.src_tsharp[1] >> 20) & 0x3ff) != ((c.dst_tsharp[1] >> 20) & 0x3ff)) {
        g_icopy_format.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    c.dst_mip = dst_sub % d.mips;
    c.dst_layer = dst_sub / d.mips;
    c.src_mip = src_sub % s.mips;
    c.src_layer = src_sub / s.mips;
    const std::uint32_t sw = std::max(1u, static_cast<std::uint32_t>(s.w) >> c.src_mip), sh = std::max(1u, static_cast<std::uint32_t>(s.h) >> c.src_mip);
    const std::uint32_t dw = std::max(1u, static_cast<std::uint32_t>(d.w) >> c.dst_mip), dh = std::max(1u, static_cast<std::uint32_t>(d.h) >> c.dst_mip);
    std::uint32_t x = 0, y = 0, right = sw, bottom = sh;
    if (box) {
        std::uint32_t b[6];
        if (!safe_read(box, b, sizeof(b)) || b[5] - b[2] > 1) {
            g_icopy_bad.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        x = b[0];
        y = b[1];
        right = std::min(b[3], sw);
        bottom = std::min(b[4], sh);
    }
    if (x >= right || y >= bottom || dst_x >= dw || dst_y >= dh) {
        g_icopy_bad.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    c.src_x = x;
    c.src_y = y;
    c.dst_x = dst_x;
    c.dst_y = dst_y;
    c.w = std::min(right - x, dw - dst_x);
    c.h = std::min(bottom - y, dh - dst_y);
    if (g_gnm_sources) note_source(ctx + 0x10, rd64(ctx + 0x20), rd64(ctx + 0x18), kSourceCopy);
    const std::uint64_t id = write_native_token(ctx, kGxImageCopyMagic);
    if (!id) {
        g_icopy_unwritten.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    {
        std::lock_guard<std::mutex> lk(g_image_copies_mu);
        if (g_image_copies.size() > 65536) g_image_copies.clear();
        g_image_copies[id] = c;
    }
    g_icopy_tokens.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

// BBHOST_GX_NATIVE_YEBIS: YEBIS draws, by token id, until the CP walks the
// token. Shadow only for now - the method still runs - so this says whether the
// token lands where the draw's packets begin and names the draw the call asked
// for. 0x15fbc40(ctx, index count, index VA, ...) runs a prep, the commit
// 0x15f9720, the DRAW_INDEX_2 emitter 0x1474360 and a post-draw; its Gnm
// command buffer is *(ctx + 8), the same struct GX contexts hold at +0x10.
// Set with the other modes once the backend is known: 5 (draws from their
// tokens) by default when the GX backend is on, as BBHOST_GX_NATIVE is.
int g_native_yebis = 0;
std::mutex g_yebis_mu;
std::unordered_map<std::uint64_t, GxYebisDraw> g_yebis_draws;             // under g_yebis_mu
std::unordered_map<std::uint64_t, GpuDrawInputs> g_yebis_inputs;          // built at the call, since YEBIS reuses its state
std::unordered_map<std::uint64_t, yebis_bind::Work> g_yebis_work;         // the windows its ring slots are owed, written at the token
std::atomic<std::uint64_t> g_yebis_calls{0}, g_yebis_tokens{0}, g_yebis_unwritten{0}, g_yebis_walked{0}, g_yebis_matched{0},
    g_yebis_mismatched{0}, g_yebis_unmatched{0};

// What YEBIS binds, so the native path knows what it has to supply. The commit
// walks a descriptor table per stage - the same one-dword {type, resource slot,
// user-data slot, flags} format GX uses - held just past each stage's shader object, with the entry count in
// the object's byte 3. Types 0-7 name resources in YEBIS's own per-stage array
// at *(state + 8) (stride 0x641 dwords a stage); 0x13 and up take their value
// from the block the builder fills. Read-only: this only counts them.
struct YebisStageTable {
    int stage;
    std::uint32_t off;  // the table's offset past the shader object
    std::uint32_t state_off;
};
const YebisStageTable kYebisStages[] = {
    {2, 0x28, 0x148},  // VS
    {1, 0x3c, 0x150},  // PS
    {6, 0x20, 0x160},  // LS
    {5, 0x40, 0x168},  // HS
    {4, 0x20, 0x170},  // ES
    {3, 0x38, 0x178},  // GS
};
std::atomic<std::uint64_t> g_yebis_desc[2][32] = {};  // [stage is PS][type], types past 31 folded
std::atomic<std::uint64_t> g_yebis_slots[2][16] = {};
std::atomic<std::uint64_t> g_yebis_stage_seen[8] = {};
// Where a descriptor's record lives in YEBIS's per-stage resource array: the
// dword index is stage * 0x641 + base + slot * stride, from the switch in
// sub_15f9190. Types 0x13 and up take their value from the builder's block
// instead, so they have no entry here.
struct YebisRecord {
    std::uint32_t base, stride, dwords;
};
bool yebis_record_of(std::uint32_t type, YebisRecord* out) {
    switch (type) {
    case 0: *out = {0x000, 8, 8}; return true;      // 32-byte texture record
    case 4: *out = {0x400, 8, 8}; return true;
    case 1: *out = {0x480, 4, 4}; return true;      // 16-byte record
    case 3: *out = {0x4c0, 4, 4}; return true;
    case 2: *out = {0x540, 4, 4}; return true;      // sampler
    default: return false;
    }
}
std::atomic<int> g_yebis_dumped{0};
void yebis_dump_resources(std::uint64_t state, int stage, std::uint64_t obj, std::uint32_t off, std::uint32_t count) {
    const std::uint64_t res = rd64(state + 8);
    if (!res) return;
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint32_t d = 0;
        if (!safe_read(obj + off + 4ull * i, &d, 4)) return;
        const std::uint32_t type = d & 0xff, slot = (d >> 8) & 0xff, ud = (d >> 16) & 0xff;
        YebisRecord r{};
        if (!yebis_record_of(type, &r)) {
            host_log("yebis:   stage %d type 0x%x slot %u -> user data s[%u] (from the builder's block)", stage, type, slot, ud);
            continue;
        }
        const std::uint64_t at = res + 4ull * (static_cast<std::uint64_t>(stage) * 0x641 + r.base + static_cast<std::uint64_t>(slot) * r.stride);
        std::uint32_t w[8] = {};
        if (!safe_read(at, w, 4ull * r.dwords)) continue;
        char words[80] = {};
        int n = 0;
        for (std::uint32_t k = 0; k < r.dwords && n < static_cast<int>(sizeof(words)) - 10; ++k) {
            n += std::snprintf(words + n, sizeof(words) - n, " %08x", w[k]);
        }
        host_log("yebis:   stage %d type 0x%x slot %u -> user data s[%u], record at 0x%llx:%s", stage, type, slot, ud,
                 static_cast<ull>(at), words);
    }
}

// BBHOST_GX_YEBIS_INVENTORY=1: count what kind of descriptor each YEBIS stage
// binds and in which user-data slot. It is reconnaissance, printed at exit and
// read by nobody in a normal run, and it was costing every YEBIS draw a walk
// of every stage's descriptor table: 4.6 million of the 8.6 million reads that
// needed process_vm_readv in a three-minute soak, and about 14 seconds of
// system call in them. Off by default; the draw does not use any of it.
const bool g_yebis_inventory = [] {
    const char* e = std::getenv("BBHOST_GX_YEBIS_INVENTORY");
    return e && e[0] == '1';
}();

void yebis_inventory(std::uint64_t state) {
    for (const YebisStageTable& t : kYebisStages) {
        const std::uint64_t obj = rd64(state + t.state_off);
        if (!obj) continue;
        g_yebis_stage_seen[t.stage].fetch_add(1, std::memory_order_relaxed);
        std::uint32_t head = 0;
        if (!safe_read(obj, &head, 4)) continue;
        const std::uint32_t count = (head >> 24) & 0xff;  // byte 3 of the object
        if (!count || count > 64) continue;
        const int which = t.stage == 1 ? 1 : 0;
        if (g_native_yebis >= 3 && g_yebis_dumped.fetch_add(1, std::memory_order_relaxed) < 12) {
            host_log("yebis: draw's stage %d table at 0x%llx, %u descriptors", t.stage, static_cast<ull>(obj + t.off), count);
            yebis_dump_resources(state, t.stage, obj, t.off, count);
        }
        // The whole table in one read: a read that cannot take the fast path
        // costs a system call, and this used to take one per descriptor.
        std::uint32_t table[64];
        if (!safe_read(obj + t.off, table, 4ull * count)) continue;
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::uint32_t type = table[i] & 0xff, ud = (table[i] >> 16) & 0xff;
            g_yebis_desc[which][type & 31].fetch_add(1, std::memory_order_relaxed);
            if (ud < 16) g_yebis_slots[which][ud].fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// Step 2: the draw's programs and user data built from YEBIS's own state, to be
// checked against the register file its commit writes before anything is
// suppressed. YEBIS's shader objects are Gnm shader structures themselves -
// the commit passes *(state + 0x148) + 8 to the same setter GX passes
// rd64(shader + 0x100) + 8 to - so the stage registers, the VS output table and
// the PS input table are at the offsets build_inputs() already uses.
//
// User data: a descriptor whose user-data slot is 0..15 puts its record's
// dwords there (sub_15f9190 writes only those; slots 16 and up are offsets into
// the extended block the builder fills, which this does not reconstruct yet).
void yebis_user_data(std::uint64_t state, int stage, std::uint64_t obj, std::uint32_t off, std::uint32_t* user,
                     const std::uint64_t* block = nullptr) {
    const std::uint64_t res = rd64(state + 8);
    if (!res) return;
    const std::uint32_t count = (rd32(obj) >> 24) & 0xff;
    for (std::uint32_t i = 0; i < count && i < 64; ++i) {
        const std::uint32_t d = rd32(obj + off + 4ull * i);
        const std::uint32_t type = d & 0xff, slot = (d >> 8) & 0xff, ud = (d >> 16) & 0xff;
        if (ud > 15) continue;
        YebisRecord r{};
        if (yebis_record_of(type, &r)) {
            const std::uint64_t at =
                res + 4ull * (static_cast<std::uint64_t>(stage) * 0x641 + r.base + static_cast<std::uint64_t>(slot) * r.stride);
            const std::uint32_t n = std::min<std::uint32_t>(r.dwords, 16 - ud);
            std::memcpy(user + ud, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), 4ull * n);
            continue;
        }
        // The types the builder sources from its own GPU rings: it copies the
        // records in and hands the shader the copy's address, so the host does
        // not have to rebuild the table - only to name where it landed.
        // sub_15f7840 computes ((stride * index) << 2) + base from the ring
        // state at state + stage * 0x118 + (slot + K) * 40.
        std::uint64_t addr = 0;
        if (block) {
            // The builder just ran and left each descriptor's address here, so
            // there is nothing to predict.
            addr = block[i];
            if (!addr) continue;
        } else if (type == 0x1a) {
            addr = rd64(state + 0x198);
        } else if (type == 0x1b) {
            // Everything the loop copied goes in one more ring, whose address
            // sub_15f7840 computes after the loop from its own state.
            const std::uint64_t e = state + static_cast<std::uint64_t>(stage) * 0x118;
            const std::uint64_t base = rd64(e + 0x360);
            const std::uint32_t stride = rd32(e + 0x368), count_of = rd32(e + 0x36c);
            if (!base || !count_of) continue;
            addr = base + ((static_cast<std::uint64_t>(stride) * ((rd64(e + 0x358) + 1) % count_of)) << 2);
        } else if (!block) {
            int k = -1;
            std::uint32_t dirty_at = 0;
            switch (type) {
            case 0x13: k = 0; dirty_at = 0x10; break;
            case 0x19: k = 1; dirty_at = 0x17; break;
            case 0x15: k = 2; dirty_at = 0x1e; break;
            case 0x17: k = 3; dirty_at = 0x25; break;
            case 0x16: k = 4; dirty_at = 0x2c; break;
            case 0x18: k = 5; dirty_at = 0x33; break;
            default: break;
            }
            if (k < 0) continue;
            // The builder only copies, and so only advances the ring, when the
            // descriptor's slot is dirty; otherwise the shader keeps reading the
            // copy already there. Predicting the advance unconditionally put
            // most draws one slot ahead of the address the commit wrote.
            (void)dirty_at;
            const std::uint64_t e = state + static_cast<std::uint64_t>(stage) * 0x118 +
                                    (static_cast<std::uint64_t>(slot) + static_cast<std::uint64_t>(k)) * 40;
            const std::uint64_t base = rd64(e + 0x270);
            const std::uint32_t stride = rd32(e + 0x278), count_of = rd32(e + 0x27c);
            if (!base || !count_of) continue;
            // The builder advances the ring before it copies, so this is the
            // slot it is about to use: it has not run yet when the hook reads.
            const std::uint64_t index = (rd64(e + 0x268) + 1) % count_of;
            addr = base + ((static_cast<std::uint64_t>(stride) * index) << 2);
        }
        if (ud + 1 < 16 && addr) {
            user[ud] = static_cast<std::uint32_t>(addr);
            user[ud + 1] = static_cast<std::uint32_t>(addr >> 32);
        }
    }
}

bool build_yebis_inputs(std::uint64_t state, GpuDrawInputs& in, std::uint32_t prep_a, std::uint32_t prep_b,
                        bool with_user_data = true) {
    const std::uint64_t vs_gnm = rd64(state + 0x148), ps_gnm = rd64(state + 0x150);
    if (!vs_gnm || !ps_gnm) return false;
    in = GpuDrawInputs{};
    for (int i = 0; i < 4; ++i) in.vs_pgm[i] = rd32(vs_gnm + 8 + 4 * static_cast<std::uint64_t>(i));
    in.vs_pgm[2] |= rd32(state + 0x138);  // the modifier the commit passes to the VS setter
    in.vs_out_cntl = rd32(vs_gnm + 8 + 0x18);
    for (int i = 0; i < 4; ++i) in.ps_pgm[i] = rd32(ps_gnm + 8 + 4 * static_cast<std::uint64_t>(i));
    in.ps_input_ena = rd32(ps_gnm + 8 + 0x18);
    in.ps_in_control = rd32(ps_gnm + 8 + 0x20);
    in.cb_shader_mask = rd32(ps_gnm + 8 + 0x2c);
    in.ps_col_format = rd32(ps_gnm + 8 + 0x14);
    // SPI_PS_INPUT_CNTL, matched to the VS outputs exactly as build_inputs()
    // does it: the Gnm shader structure is the same one.
    {
        const auto u8 = [](std::uint64_t va) { return rd32(va) & 0xff; };
        const std::uint64_t vs_out = vs_gnm + ((static_cast<std::uint64_t>(u8(vs_gnm + 0x24)) + u8(vs_gnm + 3)) << 2) + 0x28;
        const std::uint32_t vs_outs = u8(vs_gnm + 0x25);
        const std::uint64_t ps_in = ps_gnm + (static_cast<std::uint64_t>(u8(ps_gnm + 3)) << 2) + 0x3c;
        in.ps_input_count = std::min<std::uint32_t>(u8(ps_gnm + 0x38), 32);
        for (std::uint32_t i = 0; i < in.ps_input_count; ++i) {
            const std::uint32_t word = rd32(ps_in + 2 * static_cast<std::uint64_t>(i)) & 0xffff;
            std::uint32_t reg = 0x20;
            for (std::uint32_t j = 0; j < vs_outs; ++j) {
                if (u8(vs_out + 2 * static_cast<std::uint64_t>(j)) == (word & 0xff)) {
                    reg = u8(vs_out + 2 * static_cast<std::uint64_t>(j) + 1);
                    break;
                }
            }
            in.ps_input_cntl[i] = (((word >> 7) & 0x1e0) | reg) & 0x3f;
            in.ps_input_cntl[i] |= ((word >> 8) & 3) << 8;
            in.ps_input_cntl[i] |= (((word >> 12) | (word >> 10)) & 1) << 10;
        }
    }
    if (with_user_data) {
        yebis_user_data(state, 2, vs_gnm, 0x28, in.vs_user);
        yebis_user_data(state, 1, ps_gnm, 0x3c, in.ps_user);
    }
    // Slot 0 of the vertex-side stages is not a table descriptor at all: the
    // commit writes it straight from the state (sub_1476390(cb, stage, 0, ...)).
    if (const std::uint64_t v = rd64(state + 0x120)) {
        in.vs_user[0] = static_cast<std::uint32_t>(v);
        in.vs_user[1] = static_cast<std::uint32_t>(v >> 32);
    }
    // The prep (0x15f9f30) writes the draw's last two arguments into the two
    // user-data slots named by the nibbles of a byte in the first bound
    // vertex-side stage's object - LS and ES at +0x19, the VS at +0x27.
    std::uint32_t slots = 0;
    if (const std::uint64_t ls = rd64(state + 0x160)) {
        slots = rd32(ls + 0x19) & 0xff;
    } else if (const std::uint64_t es = rd64(state + 0x170)) {
        slots = rd32(es + 0x19) & 0xff;
    } else {
        slots = rd32(vs_gnm + 0x27) & 0xff;
    }
    if ((slots & 0xf) != 0 && (slots & 0xf) < 16) in.vs_user[slots & 0xf] = prep_a;
    if (((slots >> 4) & 0xf) != 0 && ((slots >> 4) & 0xf) < 16) in.vs_user[(slots >> 4) & 0xf] = prep_b;
    return true;
}

// What the built inputs got right, by field, against the registers the commit
// wrote for the same draw.
std::atomic<std::uint64_t> g_yebis_checked{0}, g_yebis_field_bad[6] = {}, g_yebis_slot_bad[2][16] = {};
const char* const kYebisField[6] = {"vs_pgm", "ps_pgm", "vs_user", "ps_user", "ps_input_ena", "cb_shader_mask"};

// The flip (BBHOST_GX_NATIVE_YEBIS=5): the parts of the commit that are not
// packets, run as guest calls, and nothing else. 0x15f9720 opens the ring it
// builds into (0x15fd230), builds each stage's resources (0x15f7840), writes
// them and the shaders as packets - which is what this drops - uploads the
// extended block with a WRITE_DATA (0x1473b80) and closes the ring (0x15fd190).
constexpr std::uint64_t kYebisPrep = 0x15f9f30, kYebisRingOpen = 0x15fd230, kYebisBuild = 0x15f7840;
constexpr std::uint64_t kYebisRingClose = 0x15fd190, kYebisWriteData = 0x1473b80, kYebisTailA = 0x1477f40;
constexpr std::uint64_t kYebisTailB = 0x14781d0, kYebisPost = 0x15f9d10;
template <typename... A>
std::int64_t yebis_call(std::uint64_t bn_va, A... a) {
    return hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest(bn_va))),
                                        static_cast<std::int64_t>(a)...);
}

// (ns_now is declared above)
std::atomic<std::uint64_t> g_yebis_flipped{0}, g_yebis_flip_failed{0};
// Where the flip's own time goes: the guest calls it still makes (the ring, the
// builders, the WRITE_DATA tail, the post-draw) against the host work (reading
// YEBIS's state into inputs, the token, the map).
std::atomic<std::uint64_t> g_yebis_ns_builders{0}, g_yebis_ns_host{0}, g_yebis_ns_n{0}, g_yebis_ns_build_only{0}, g_yebis_ns_native{0};
// Per guest call the flip keeps: ring open, WRITE_DATA, tail A, tail B, ring close, post-draw.
std::atomic<std::uint64_t> g_yebis_ns_call[6] = {};
const char* const kYebisCallName[6] = {"ring open 0x15fd230", "WRITE_DATA 0x1473b80", "tail 0x1477f40", "tail 0x14781d0",
                                       "ring close 0x15fd190", "post-draw 0x15f9d10"};
std::uint64_t ns_now() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
// Runs the commit's non-packet work and fills `block` with the address each
// extended descriptor resolved to. Returns false if a stage could not be built.
bool yebis_run_builders(std::uint64_t state, std::uint64_t cb, std::uint64_t arg3, GpuDrawInputs& in, bool* built_here,
                        yebis_bind::Work& work) {
    // BBHOST_YEBIS_BIND: whether the builder is ours. It has to be all of the
    // draw's stages or none of them - a stage that gives up half way has
    // already moved its rings - so the tables are checked before any is built.
    const int bind = yebis_bind::mode();
    bool ours = bind != 0;
    if (ours) {
        for (const YebisStageTable& t : kYebisStages) {
            const std::uint64_t obj = rd64(state + t.state_off);
            if (!obj) continue;
            const std::uint32_t count = (rd32(obj) >> 24) & 0xff;
            if (!count || count > 64 || !yebis_bind::table_known(obj + t.off, count)) {
                ours = false;
                break;
            }
        }
        if (!ours) yebis_bind::note_fallback();
    }
    // Compare mode lets the guest build, so anything that would replace one of
    // its packets has to stay with it - including the payload writes below and
    // the post-draw's counter.
    const bool native_here = ours && bind != 2;
    if (built_here) *built_here = native_here;
    // The guest builder emits into the constant-engine stream, which its ring
    // bracket (WAIT_ON_DE_COUNTER_DIFF / INCREMENT_CE_COUNTER) and the draw
    // side's WAIT_ON_CE_COUNTER exist to synchronise. Ours does not, so when
    // the build is ours none of the four is needed.
    const bool guest_builds = !ours || bind == 2;
    if (guest_builds) {
        const std::uint64_t tc = ns_now();
        yebis_call(kYebisRingOpen, arg3, rd32(state + 0xa10) >> 2);
        g_yebis_ns_call[0].fetch_add(ns_now() - tc, std::memory_order_relaxed);
    }
    for (const YebisStageTable& t : kYebisStages) {
        const std::uint64_t obj = rd64(state + t.state_off);
        if (!obj) continue;
        const std::uint32_t count = (rd32(obj) >> 24) & 0xff;
        if (!count || count > 64) return false;
        // One block a stage: the writer walks it alongside that stage's table,
        // so a shared one would leave every stage but the last with another's.
        std::uint64_t block[64] = {};
        std::uint64_t mine[64] = {};
        if (ours) {
            const std::uint64_t tn = ns_now();
            // In compare mode ours only predicts: the guest's builder is what
            // moves the state, so both see the same marks and ring indices.
            yebis_bind::build_stage(state, t.stage, obj + t.off, count, bind == 2 ? mine : block, work, bind == 2);
            g_yebis_ns_native.fetch_add(ns_now() - tn, std::memory_order_relaxed);
        }
        if (guest_builds) {
            const std::uint64_t tb = ns_now();
            yebis_call(kYebisBuild, state, arg3, t.stage, obj + t.off, count, reinterpret_cast<std::uint64_t>(block));
            g_yebis_ns_build_only.fetch_add(ns_now() - tb, std::memory_order_relaxed);
            if (bind == 2) yebis_bind::compare(t.stage, mine, block, count);
        }
        if (t.stage == 2) yebis_user_data(state, 2, obj, t.off, in.vs_user, block);
        if (t.stage == 1) yebis_user_data(state, 1, obj, t.off, in.ps_user, block);
    }
    if (rd32(state + 0x260)) {
        // A WRITE_DATA of the draw's own 0x30 dwords to the address the type
        // 0x1a descriptor names. The packet carries a copy of them, so the
        // copy is what this does.
        if (native_here) {
            // Deferred with the windows: the packet would have been walked at
            // the draw's place in the stream, which is where the token is.
            const std::uint64_t tn = ns_now();
            yebis_bind::add_copy(work, rd64(state + 0x198),
                                 reinterpret_cast<const void*>(static_cast<std::uintptr_t>(state + 0x1a0)), 0x30 * 4);
            g_yebis_ns_native.fetch_add(ns_now() - tn, std::memory_order_relaxed);
        } else {
            { const std::uint64_t tc = ns_now(); yebis_call(kYebisWriteData, cb, rd64(state + 0x198), state + 0x1a0, 0x30, 1); g_yebis_ns_call[1].fetch_add(ns_now() - tc, std::memory_order_relaxed); }
            // The tail is a DMA_DATA of ten bytes from address 0 to address 0,
            // which the command processor already ignores.
            { const std::uint64_t tc = ns_now(); yebis_call(kYebisTailA, cb, 0, 0x8000000, 0); g_yebis_ns_call[2].fetch_add(ns_now() - tc, std::memory_order_relaxed); }
        }
        const std::uint32_t zero = 0;
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(state + 0x260)), &zero, 4);
    }
    if (guest_builds) {
        { const std::uint64_t tc = ns_now(); yebis_call(kYebisTailB, cb); g_yebis_ns_call[3].fetch_add(ns_now() - tc, std::memory_order_relaxed); }
        { const std::uint64_t tc = ns_now(); yebis_call(kYebisRingClose, arg3); g_yebis_ns_call[4].fetch_add(ns_now() - tc, std::memory_order_relaxed); }
    }
    return true;
}


// What one of GX's clear passes costs, before deciding whether the family is
// worth migrating: it draws a full-screen triangle a slice with a full state
// save and restore, and the ring accounting puts the clear and copy passes at
// 31% of the dwords - but dwords have not predicted CPU once this session.
// An entry hook cannot time the method - it returns before it runs - so this
// counts what a native clear would have to cover: calls and slices, one
// full-screen triangle each. f: rdi ctx, rsi target, rdx resource, ecx first
// slice, r8d slices.
std::atomic<std::uint64_t> g_clear_calls{0}, g_clear_slices{0};
// BBHOST_GX_SKIP_CLEARS=1: do not run the pass at all. The flush runs it to
// prepare a colour target, which on the hardware is partly CMASK/FMASK
// housekeeping that bbhost has no equivalent for - so the first question is
// whether anything it writes is visible here. If frames are unchanged the pass
// is simply not needed; if they break it does real work and has to be migrated
// rather than dropped.
const bool g_skip_clears = [] {
    const char* e = std::getenv("BBHOST_GX_SKIP_CLEARS");
    return e && e[0] == '1';
}();
std::uint64_t clear_colour_cost(const std::uint64_t* f) {
    g_clear_calls.fetch_add(1, std::memory_order_relaxed);
    g_clear_slices.fetch_add(static_cast<std::uint32_t>(f[1]), std::memory_order_relaxed);
    return g_skip_clears ? 1 : 0;
}

std::atomic<std::uint64_t> g_yebis_state_built{0}, g_yebis_state_missing{0};
std::uint64_t native_yebis(const std::uint64_t* f) {
    if (!g_native_yebis) return 0;
    const std::uint64_t ctx = f[5];
    const std::uint32_t index_count = static_cast<std::uint32_t>(f[4]);
    const std::uint64_t index_va = f[3];
    const std::uint64_t cb = ctx ? rd64(ctx + 8) : 0;
    if (!cb) return 0;
    if (g_yebis_calls.fetch_add(1, std::memory_order_relaxed) < 8) {
        host_log("yebis: draw from 0x%llx: %u indices at 0x%llx, command buffer 0x%llx", static_cast<ull>(bn(f[6])), index_count,
                 static_cast<ull>(index_va), static_cast<ull>(cb));
    }
    const std::uint64_t state = rd64(ctx + 0x28);
    if (g_yebis_inventory && g_native_yebis >= 2 && state) yebis_inventory(state);
    if (g_native_yebis == 5) {
        // The method's own work, minus its packets, then the token in their place.
        if (!state) return 0;
        const std::uint64_t t0 = ns_now();
        GpuDrawInputs flip{};
        const bool ok_a = build_yebis_inputs(state, flip, static_cast<std::uint32_t>(f[2]), static_cast<std::uint32_t>(f[1]), false);
        const std::uint64_t t1 = ns_now();
        bool built_here = false;
        yebis_bind::Work work;
        const bool ok_b = ok_a && yebis_run_builders(state, cb, rd64(ctx + 0x10), flip, &built_here, work);
        const std::uint64_t t2 = ns_now();
        g_yebis_ns_host.fetch_add(t1 - t0, std::memory_order_relaxed);
        g_yebis_ns_builders.fetch_add(t2 - t1, std::memory_order_relaxed);
        g_yebis_ns_n.fetch_add(1, std::memory_order_relaxed);
        if (!ok_b) {
            g_yebis_flip_failed.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        // The rest of the draw's state from the wrapper's context
        // state on this command buffer (targets, blend, depth-stencil,
        // viewport, scissors, primitive, rasterizer); complete tokens read
        // nothing from the register file.
        GxYebisDraw record{index_count, index_va};
        if (g_yebis_full && wrapper_complete_inputs(cb, flip, &record.index_type)) {
            record.state_built = true;
            flip.complete = g_yebis_full == 2;
            g_yebis_state_built.fetch_add(1, std::memory_order_relaxed);
        } else if (g_yebis_full) {
            g_yebis_state_missing.fetch_add(1, std::memory_order_relaxed);
        }
        const std::uint64_t fid = write_native_token(cb - 0x10, kGxYebisMagic);
        if (!fid) {
            g_yebis_unwritten.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        {
            std::lock_guard<std::mutex> lk(g_yebis_mu);
            if (g_yebis_draws.size() > 65536) {
                g_yebis_draws.clear();
                g_yebis_inputs.clear();
                g_yebis_work.clear();
            }
            g_yebis_draws[fid] = record;
            g_yebis_inputs[fid] = flip;
            if (work.n && built_here) g_yebis_work[fid] = work;  // in compare mode the guest still writes them
        }
        g_yebis_tokens.fetch_add(1, std::memory_order_relaxed);
        g_yebis_flipped.fetch_add(1, std::memory_order_relaxed);
        // The post-draw clears one byte of state and increments the draw
        // engine's counter - the other half of the ring bracket, so it goes
        // when that does and the two counters stay level.
        if (built_here) {
            *reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(state + 0xa14)) = 0;
        } else {
            const std::uint64_t tc = ns_now();
            yebis_call(kYebisPost, state, cb);
            g_yebis_ns_call[5].fetch_add(ns_now() - tc, std::memory_order_relaxed);
        }
        return 1;  // the method does not run
    }
    const std::uint64_t id = write_native_token(cb - 0x10, kGxYebisMagic);
    if (!id) {
        g_yebis_unwritten.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    GpuDrawInputs built{};
    const bool have = g_native_yebis >= 4 && state &&
                      build_yebis_inputs(state, built, static_cast<std::uint32_t>(f[2]), static_cast<std::uint32_t>(f[1]));
    {
        std::lock_guard<std::mutex> lk(g_yebis_mu);
        if (g_yebis_draws.size() > 65536) {
            g_yebis_draws.clear();
            g_yebis_inputs.clear();
        }
        g_yebis_draws[id] = {index_count, index_va};
        if (have) g_yebis_inputs[id] = built;
    }
    g_yebis_tokens.fetch_add(1, std::memory_order_relaxed);
    return 0;  // shadow: the method still runs
}

// The Scaleform HAL's draws as draw tokens built from HAL
// objects. The PS4 HAL (vtable at BN
// 0x55ba060) draws through three methods - 0x73e670 (indexed), 0x73e6b0
// (instanced) and 0x73e640 (non-indexed) - each of which flushes the Gnm
// wrapper's resource tables (0x1498ee0) and emits the draw packet. Everything
// those two write is in HAL objects when the method is entered:
//   - the shader pair in the ShaderInterface (HAL+0x2b828): +0x1158 the VS
//     wrapper {+8 Gnm shader, +0x118 fetch shader, +0x128 modifier, +0x2a
//     layout}, +0x1168 the PS wrapper {+8 Gnm shader, +0x2a layout}; the
//     wrapper's builder (HAL+0x2ca30, +0x2c8) holds the same at +0xc088/+0xc080
//     with the layouts at +0xc168/+0xc160;
//   - the stage resource tables: the uniform block (0x73f930 copies the dirty
//     range into the command buffer and appends its V# at constant-buffer slot
//     0), the textures (each texture object appends its T# and S#), the mesh's
//     vertex V#s (0x73e2f0 through 0x149a920, vertex-buffer slot 0..) are all
//     records in the builder's per-stage arrays (+0x70 + stage * 0x1800: the
//     pixel stage at +0x1870, the vertex stage at +0x3070), placed at the
//     table index the shader's layout gives each slot. The flush copies the
//     layout's first byte of dwords from there and hands the copy's address
//     (and sub-table addresses at the layout's u16 offsets) to the user-data
//     SGPRs the layout names; the token copies the same dwords into the host
//     ring and names the same SGPRs;
//   - blend: HAL+0x224 the mode, +0x228/+0x229 sourceAc/forceAc (0x495d30),
//     which 0x73e0d0 turns into CB_BLEND0_CONTROL from the mode table at
//     0x54dca20 (24 bytes a mode: func, src factor, dst factor, alpha func,
//     alpha src, alpha dst as indices into 0x4b562d0 / 0x2fc6a90);
//   - depth-stencil: HAL+0x108 the mode, +0x214 the stencil reference, which
//     0x73de50 turns into DB_DEPTH_CONTROL, DB_STENCIL_CONTROL, DB_STENCILREFMASK
//     and CB_TARGET_MASK from the mode table at 0x54dcc00 (9 dwords a mode:
//     depth test, depth write off, stencil, colour write, depth func, stencil
//     func, three stencil ops as indices into 0x2fc6a60 / 0x4b562c0);
//   - the render target: the top of the HAL's target stack (HAL+0x1c0, 0x320
//     bytes an entry, count at +0x1c8) is an object whose +0x18 holds the PS4
//     buffer {+0x20 Gnm::RenderTarget, +0x28 Gnm::DepthRenderTarget}, which
//     0x73da50 binds at slot 0;
//   - the viewport: the view rectangle (HAL+0x1194 {l, t, r, b}, or the
//     viewport {x, y, w, h} at +0x1170 when HAL flag 0x10 is set), which
//     0x73d930 passes to setupScreenViewport (0x1496f60: viewport 0, the
//     screen scissor, PA_CL_VTE_CNTL 0x43f);
//   - the raster mode (HAL+0x21c: 0 solid, 1 wireframe, 2 points), which
//     0x73e210 writes as PA_SU_SC_MODE_CNTL;
//   - the geometry: BeginScene (0x73d780) sets triangle lists, 16-bit indices
//     and VGT_INDX_OFFSET 0; the mesh bind sets INDEX_BASE (0x14752c0, hooked
//     here for the buffer the HAL bound) and the draw gives the index offset
//     and count; an instanced draw gives its instance count (0x73e610 wrote it
//     as NUM_INSTANCES).
// BBHOST_GX_NATIVE_SCALEFORM=1 (shadow) writes the token and lets the method
// run; the command processor compares the token's inputs with the register
// file at the draw packet that follows, by field. =2 (native, the default with
// native GX draws) writes the token and returns: no flush, no draw packet, the
// CP draws from the token. =0 leaves the HAL on the packet path.
const int g_native_scaleform = [] {
    const char* e = std::getenv("BBHOST_GX_NATIVE_SCALEFORM");
    return e && e[0] ? std::atoi(e) : 2;
}();
// BBHOST_SF_UNIFORMS=0: a Scaleform draw's uniform block is read where the
// game left it, when the command processor and the GPU get to the draw. By
// default the token takes it at the call - the game writes the next frame's
// uniforms at the same addresses, and a draw recorded a frame late read them
// (the inventory menu collapsing for a frame).
const bool g_sf_uniforms = [] {
    const char* e = std::getenv("BBHOST_SF_UNIFORMS");
    return !(e && e[0] == '0');
}();
std::atomic<std::uint64_t> g_sf_uniforms_placed{0};
constexpr std::uint64_t kSfDsModes = 0x54dcc00, kSfBlendModes = 0x54dca20, kSfCmpFuncs = 0x2fc6a60, kSfBlendFactors = 0x2fc6a90;
constexpr std::uint64_t kSfStencilOps = 0x4b562c0, kSfBlendFuncs = 0x4b562d0;
constexpr std::uint32_t kSfDsModeCount = 8, kSfBlendModeCount = 0x14;
constexpr std::uint64_t kSfHalDraw = 0x73e670, kSfHalDrawInstanced = 0x73e6b0, kSfHalDrawAuto = 0x73e640;

struct SfTablePtr {
    std::uint8_t ud;
    std::uint32_t off;  // dwords into the stage's table
};
struct SfStage {
    std::vector<std::uint32_t> table;  // the layout's dwords of the builder's stage array
    SfTablePtr ptrs[7];
    int nptrs = 0;
    std::uint32_t named = 0;  // user-data slots the token sets (tables and values)
    std::uint32_t direct = 0;  // slots the layout routes to SGPRs directly (bit 15), which the arrays do not hold
    // The shader's own data (the uniform block: per-quad transforms, colours)
    // as it was at the call, and where its V# went: user-data slot `ud` when
    // direct, else table dword `ud`. Placed beside the tables when the CP
    // walks the token, and the V# pointed at it (BBHOST_SF_UNIFORMS).
    std::vector<std::uint32_t> uniforms;
    int uniform_ud = -1;
    bool uniform_direct = false;
};
struct ScaleformDraw {
    GpuDrawInputs in;
    GxDrawObjects obj;
    SfStage stage[2];  // [0] the vertex stage, [1] the pixel stage
    std::uint64_t hal = 0, builder = 0;
    std::uint32_t blend_mode = 0, ds_mode = 0, raster_mode = 0;
    std::uint32_t hal_flags = 0, vp_flags = 0;
    std::int32_t rect[4] = {};
    bool shadow = false;  // the method's packet follows: compare, do not draw
};
// The wrapper's context state on a command buffer this thread writes, as its
// setters were called.
struct WrapperState {
    std::uint64_t cb = 0;
    std::uint32_t user[8][16] = {};
    std::uint16_t set[8] = {};
    std::uint32_t set_flip[8][16] = {};  // the flip each slot was last set at (diagnostics; 0: never)
    std::uint32_t created_flip = 0;
    std::uint64_t vs_gnm = 0, ps_gnm = 0;
    std::uint32_t vs_mod = 0;
    std::uint32_t rt[8][11] = {};
    std::uint8_t rt_set = 0;
    std::uint32_t dsv[12] = {};  // words 0..7, depth view, HTILE base, HTILE surface, depth info
    bool dsv_set = false;
    float vport[6] = {};
    std::uint32_t screen_scissor[2] = {}, vport_scissor[2] = {}, vte = 0, sc_mode = 0;
    std::uint32_t blend[8] = {}, target_mask = 0, ds_control = 0, stencil_control = 0, stencil_ref = 0;
    std::uint32_t prim = 0, instances = 1, index_size = 0, indx_offset = 0;
    std::uint32_t clip = 0, su_sc_mode = 0, stages = 0, color_control = 0, render_control = 0, stencil_clear = 0;
    std::uint64_t index_base = 0;
    std::uint64_t cs_gnm = 0, indirect_base = 0;
};
// Per command buffer, not per thread: a buffer's setters and its emitter can
// run on different threads (YEBIS's compute passes wrote user data on one and
// dispatched on another, and the token missed those slots). One writer at a
// time per buffer is the wrapper's own rule; the lock covers the lookup and
// the few dwords a setter writes.
std::mutex g_ws_mu;
std::unordered_map<std::uint64_t, WrapperState> g_ws;  // every buffer seen (a few dozen contexts)
WrapperState* wrapper_state_of_locked(std::uint64_t cb) {
    const auto it = g_ws.find(cb);
    return it == g_ws.end() ? nullptr : &it->second;
}
WrapperState& wrapper_state_locked(std::uint64_t cb) {
    if (WrapperState* w = wrapper_state_of_locked(cb)) return *w;
    if (g_ws.size() > 4096) g_ws.clear();
    WrapperState& w = g_ws[cb];
    w.cb = cb;
    w.created_flip = static_cast<std::uint32_t>(hle_video_flip_count());
    return w;
}
// A copy of the buffer's state for a reader (a token being built).
bool wrapper_state_copy(std::uint64_t cb, WrapperState& out) {
    std::lock_guard<std::mutex> lk(g_ws_mu);
    const WrapperState* w = wrapper_state_of_locked(cb);
    if (!w) return false;
    out = *w;
    return true;
}
std::uint64_t sf_index_base(std::uint64_t cb) {
    WrapperState w;
    return wrapper_state_copy(cb, w) ? w.index_base : 0;
}
// BBHOST_GX_UD_TRACE=1 (checks): where each compute user-data write lands.
const bool g_ud_trace = [] {
    const char* e = std::getenv("BBHOST_GX_UD_TRACE");
    return e && e[0] == '1';
}();
void ud_trace(const char* what, std::uint64_t cb, std::uint32_t stage, std::uint32_t sgpr, std::uint32_t count) {
    if (!g_ud_trace || stage != 0) return;
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1, std::memory_order_relaxed) < 40) {
        host_log("ud-trace: %s cb 0x%llx stage %u s[%u] x%u", what, static_cast<ull>(cb), stage, sgpr, count);
    }
}
// The compute stage's user data is not a property of one command buffer: the
// resource builder mirrors some of it to the asynchronous compute queue's
// buffer (0x149a920 writes `builder + 0x68` when the slot asks for it), and
// the command processor keeps one register image per thread over both. So a
// dispatch takes the compute stage from a per-thread image in call order, the
// way the register file has it, and the other stages from the buffer.
struct ComputeUserData {
    std::uint32_t user[16] = {};
    std::uint16_t set = 0;
};
thread_local ComputeUserData t_cs_user;
void note_user_data(std::uint64_t cb, std::uint32_t stage, std::uint32_t sgpr, std::uint64_t data, std::uint32_t count) {
    ud_trace("data", cb, stage, sgpr, count);
    if (stage >= 8 || sgpr >= 16 || !count || count > 16 || sgpr + count > 16 || !data) return;
    std::uint32_t words[16];
    if (!safe_read(data, words, static_cast<std::size_t>(count) * 4)) return;
    if (stage == 0) {
        std::memcpy(t_cs_user.user + sgpr, words, static_cast<std::size_t>(count) * 4);
        t_cs_user.set |= static_cast<std::uint16_t>(((1u << count) - 1) << sgpr);
    }
    std::lock_guard<std::mutex> lk(g_ws_mu);
    WrapperState& w = wrapper_state_locked(cb);
    std::memcpy(w.user[stage] + sgpr, words, static_cast<std::size_t>(count) * 4);
    w.set[stage] |= static_cast<std::uint16_t>(((1u << count) - 1) << sgpr);
    for (std::uint32_t k = 0; k < count; ++k) w.set_flip[stage][sgpr + k] = static_cast<std::uint32_t>(hle_video_flip_count()) | 1u;
}
void note_user_value(std::uint64_t cb, std::uint32_t stage, std::uint32_t sgpr, std::uint64_t value, std::uint32_t count) {
    ud_trace("value", cb, stage, sgpr, count);
    if (stage >= 8 || sgpr >= 16 || sgpr + count > 16) return;
    if (stage == 0) {
        t_cs_user.user[sgpr] = static_cast<std::uint32_t>(value);
        if (count == 2) t_cs_user.user[sgpr + 1] = static_cast<std::uint32_t>(value >> 32);
        t_cs_user.set |= static_cast<std::uint16_t>(((1u << count) - 1) << sgpr);
    }
    std::lock_guard<std::mutex> lk(g_ws_mu);
    WrapperState& w = wrapper_state_locked(cb);
    w.user[stage][sgpr] = static_cast<std::uint32_t>(value);
    if (count == 2) w.user[stage][sgpr + 1] = static_cast<std::uint32_t>(value >> 32);
    w.set[stage] |= static_cast<std::uint16_t>(((1u << count) - 1) << sgpr);
    for (std::uint32_t k = 0; k < count; ++k) w.set_flip[stage][sgpr + k] = static_cast<std::uint32_t>(hle_video_flip_count()) | 1u;
}

// With no draw reading the context register file, the packets
// the wrapper's state setters write are dead: nothing on the host reads
// SET_CONTEXT_REG or SET_UCONFIG_REG, the graphics stages' SET_SH_REG, or the
// index and instance packets any more (every draw is a token carrying its own
// state). The hooks record the state and then return without writing the
// packet. What is *not* suppressed: the compute stage's user data (stage 0,
// whichever writer sets it), the compute program and SET_BASE, which the
// dispatches still take from the register file.
// BBHOST_GX_SKIP_STATE_PACKETS=0 keeps them; 1 forces them off even when the
// register file is kept (which would then be stale - checks only).
const int g_skip_state_packets = [] {
    const char* e = std::getenv("BBHOST_GX_SKIP_STATE_PACKETS");
    return e && e[0] ? std::atoi(e) : -1;  // -1: whenever the register file is not kept
}();
std::atomic<std::uint64_t> g_state_packets_skipped[kKinds] = {};
// Leaf packet writers whose packet nothing reads any more - the
// command processor's census found them: multi-sample configuration, the
// guard band, clip rectangles, blend constants, the alpha-to-mask and SPI
// words, the instance step rate, a DB_DEPTH_CONTROL clear, and the
// ACQUIRE_MEM cache flushes the host's own barriers replace. Hooking them to
// suppress the packet is **off by default and not worth turning on**: they are
// called 7.9M times a soak for 49.5M dwords (189 MiB), which is about 0.1% of
// a frame's work, while a prologue hook on each costs several times that (two
// soaks with them on were no better than two without, and their p95 was worse:
// 16.4 and 16.7 ms against 14.0 and 15.8). The table stays as the record of
// which writers are dead, for whoever wants to patch them out instead.
// BBHOST_GX_DEAD_WRITERS=1 installs them.
const bool g_dead_writers = [] {
    const char* e = std::getenv("BBHOST_GX_DEAD_WRITERS");
    return e && e[0] == '1';
}();
struct DeadWriter {
    std::uint64_t va;
    std::uint8_t n;
    std::uint8_t bytes[20];
    std::uint8_t dwords;  // what the packet would have cost, for the report
    const char* what;
};
const DeadWriter kDeadWriters[] = {
    {0x1475e10, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50}, 7, "ACQUIRE_MEM"},
    {0x1477f40, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xce}, 7, "ACQUIRE_MEM (b)"},
    {0x148b550, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50}, 7, "ACQUIRE_MEM (queue)"},
    {0x148b610, 15, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x89, 0xd3, 0x41, 0x89, 0xf7}, 7, "ACQUIRE_MEM (queue b)"},
    {0x14772c0, 17, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89, 0xf3, 0x49, 0x89, 0xfe, 0x49, 0x8b, 0x4e, 0x08}, 18, "sample locations"},
    {0x14773b0, 17, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89, 0xf3, 0x49, 0x89, 0xfe, 0x49, 0x8b, 0x4e, 0x08}, 18, "sample locations (b)"},
    {0x1477350, 17, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}, 4, "PA_SC_AA_MASK"},
    {0x14771c0, 17, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}, 3, "PA_SC_AA_CONFIG"},
    {0x14734c0, 17, {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08, 0x48, 0x8b, 0x43, 0x10}, 3, "DB_DEPTH_CONTROL 0"},
    {0x1473100, 16, {0x55, 0x48, 0x89, 0xe5, 0x53, 0x48, 0x83, 0xec, 0x18, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}, 6, "the blend constants"},
    {0x1473030, 17, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}, 3, "DB_ALPHA_TO_MASK"},
    {0x1472ed0, 15, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x89, 0xd3, 0x41, 0x89, 0xf7}, 3, "SPI_BARYC_CNTL"},
    {0x1473650, 15, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x89, 0xd3, 0x41, 0x89, 0xf7}, 3, "the clip rectangle"},
    {0x14736d0, 17, {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08, 0x48, 0x8b, 0x43, 0x10}, 6, "the guard band"},
    {0x1473750, 17, {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08, 0x48, 0x8b, 0x43, 0x10}, 6, "the guard band (b)"},
    {0x1474f90, 16, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x41, 0x89, 0xd6, 0x41, 0x89, 0xf7}, 4, "the instance step rate"},
};
constexpr std::size_t kDeadWriterCount = sizeof(kDeadWriters) / sizeof(kDeadWriters[0]);
std::atomic<std::uint64_t> g_dead_calls[kDeadWriterCount] = {};
// The dwords each setter writes, for the report (the packet plus its marker).
std::uint8_t state_packet_dwords(Kind id) {
    switch (id) {
        case kSetVs: return 29;
        case kSetPs: return 40;
        case kSetRt: return 15;
        case kSetDsv: return 24;
        case kSetViewport: return 12;
        case kSetScreenScissor: case kSetVportScissor: return 4;
        case kUd4a: case kUd4b: return 8;
        case kUd8: return 12;
        case kUd1: return 3;
        case kUdPtr: return 4;
        case kSetUserData: return 6;
        case kSetIndexBuffer: return 3;
        case kSetNumInstances: case kSetIndexSize: return 2;
        default: return 3;
    }
}
// The state packets go when nothing reads the register file they write: no
// shadow or packet-path mode, and the command processor not asked to keep it.
bool skip_state_packets() {
    static const bool v = [] {
        if (g_skip_state_packets >= 0) return g_skip_state_packets == 1;
        const char* keep = std::getenv("BBHOST_CP_REGISTER_FILE");
        if (keep && keep[0] == '1') return false;
        return !hle_gx_shadow_wanted();
    }();
    return v;
}

// True when this setter's packet is dead: the host reads none of it.
bool suppressible(Kind id, std::uint32_t stage) {
    switch (id) {
        // The compute stage's user data and program stay: the dispatches read them.
        case kUd4a: case kUd8: case kUd4b: case kUd1: case kUdPtr: case kSetUserData:
            return stage != 0;
        case kSetVs: case kSetPs: case kSetRt: case kSetDsv: case kSetViewport: case kSetScreenScissor:
        case kSetVte: case kSetScMode: case kSetBlend: case kSetTargetMask: case kSetDsControl:
        case kSetStencilControl: case kSetStencilRef: case kSetPrimType: case kSetNumInstances:
        case kSetIndexSize: case kSetClip: case kSetPrimSetup: case kSetStages: case kSetColorControl:
        case kSetRenderControl: case kSetStencilClear: case kSetVportScissor: case kSetIndxOffset:
        case kSetIndexBuffer:
            return true;
        default:
            return false;
    }
}

// A setter's call: f[5] rdi (the command buffer), f[4] rsi, f[3] rdx, f[2] rcx, f[1] r8, f[0] r9, f[7..] stack.
// Returns 1 when the packet is suppressed (the guest's setter does not run).
std::uint64_t wrapper_note(Kind id, const std::uint64_t* f) {
    const std::uint64_t cb = f[5];
    if (!cb) return 0;
    const auto u32 = [&](int i) { return static_cast<std::uint32_t>(f[i]); };
    const auto suppress = [&](std::uint32_t stage) -> std::uint64_t {
        if (!skip_state_packets() || !suppressible(id, stage)) return 0;
        g_state_packets_skipped[id].fetch_add(1, std::memory_order_relaxed);
        return 1;
    };
    switch (id) {
        case kUd4a: case kUd4b: note_user_data(cb, u32(4), u32(3), f[2], 4); return suppress(u32(4));
        case kUd8: note_user_data(cb, u32(4), u32(3), f[2], 8); return suppress(u32(4));
        case kUd1: note_user_value(cb, u32(4), u32(3), f[2], 1); return suppress(u32(4));
        case kUdPtr: note_user_value(cb, u32(4), u32(3), f[2], 2); return suppress(u32(4));
        case kSetUserData: note_user_data(cb, u32(4), u32(3), f[2], u32(1)); return suppress(u32(4));
        // The compute-typed writers: (cb, sgpr, ...) for stage 0. Never suppressed.
        case kCsUdN: note_user_data(cb, 0, u32(4), f[3], u32(2)); return 0;
        case kCsUd8: note_user_data(cb, 0, u32(4), f[3], 4); return 0;
        case kCsUd4: note_user_data(cb, 0, u32(4), f[3], 4); return 0;
        case kCsUd1: note_user_value(cb, 0, u32(4), f[3], 1); return 0;
        case kCsUdPtr: note_user_value(cb, 0, u32(4), f[3], 2); return 0;
        default: break;
    }
    // The state setters: the guest reads below happen before the lock.
    std::uint32_t rt[11] = {};
    std::uint32_t dsv[12] = {};
    float scale[3] = {}, offset[3] = {};
    bool read_ok = true;
    if (id == kSetRt) read_ok = f[3] && safe_read(f[3], rt, sizeof(rt));
    if (id == kSetDsv && f[4]) {
        read_ok = safe_read(f[4], dsv, 32);
        if (read_ok) {
            dsv[8] = rd32(f[4] + 0x20);
            dsv[9] = rd32(f[4] + 0x24);
            dsv[10] = rd32(f[4] + 0x28);
            dsv[11] = rd32(f[4] + 0x2c);
        }
    }
    if (id == kSetViewport) {
        // Only viewport 0 is recorded; the others still write their packet.
        if (u32(4) != 0 || !f[3] || !f[2]) return 0;
        if (!safe_read(f[3], scale, sizeof(scale)) || !safe_read(f[2], offset, sizeof(offset))) return 0;
    }
    std::lock_guard<std::mutex> lk(g_ws_mu);
    const auto wrapper_state = [&](std::uint64_t c) -> WrapperState& { return wrapper_state_locked(c); };
    switch (id) {
        case kSetVs: {
            WrapperState& w = wrapper_state(cb);
            w.vs_gnm = f[4] ? f[4] - 8 : 0;
            w.vs_mod = u32(3);
            break;
        }
        case kSetPs: wrapper_state(cb).ps_gnm = f[4] ? f[4] - 8 : 0; break;
        case kSetRt: {
            const std::uint32_t slot = u32(4);
            if (slot >= 8) return 0;
            WrapperState& w = wrapper_state(cb);
            if (read_ok) {
                std::memcpy(w.rt[slot], rt, sizeof(rt));
                w.rt_set |= static_cast<std::uint8_t>(1u << slot);
            } else {
                std::memset(w.rt[slot], 0, sizeof(w.rt[slot]));
                w.rt_set &= static_cast<std::uint8_t>(~(1u << slot));
            }
            break;
        }
        case kSetDsv: {
            WrapperState& w = wrapper_state(cb);
            std::memset(w.dsv, 0, sizeof(w.dsv));
            w.dsv_set = f[4] && read_ok;
            if (w.dsv_set) std::memcpy(w.dsv, dsv, sizeof(dsv));
            break;
        }
        case kSetViewport: {
            WrapperState& w = wrapper_state(cb);
            for (int i = 0; i < 3; ++i) {
                w.vport[2 * i] = scale[i];
                w.vport[2 * i + 1] = offset[i];
            }
            break;
        }
        case kSetScreenScissor: {
            WrapperState& w = wrapper_state(cb);
            w.screen_scissor[0] = (u32(3) << 16) | (u32(4) & 0xffff);
            w.screen_scissor[1] = (u32(1) << 16) | (u32(2) & 0xffff);
            break;
        }
        case kSetVte: wrapper_state(cb).vte = u32(4) & 0xf3f; break;
        case kSetScMode: wrapper_state(cb).sc_mode = (u32(4) & 1) | ((u32(3) & 1) << 1); break;
        case kSetBlend: if (u32(4) < 8) wrapper_state(cb).blend[u32(4)] = u32(3) & 0xffff1fff; break;
        case kSetTargetMask: wrapper_state(cb).target_mask = u32(4); break;
        case kSetDsControl: wrapper_state(cb).ds_control = u32(4) & 0xc07007ff; break;
        case kSetStencilControl: wrapper_state(cb).stencil_control = u32(4) & 0xffffff; break;
        case kSetStencilRef: wrapper_state(cb).stencil_ref = u32(4); break;
        case kSetPrimType: wrapper_state(cb).prim = u32(4) & 0x3f; break;
        case kSetNumInstances: wrapper_state(cb).instances = u32(4); break;
        case kSetIndexSize: wrapper_state(cb).index_size = u32(4); break;
        case kSetClip: wrapper_state(cb).clip = u32(4) & 0xf7fe03f; break;
        case kSetPrimSetup: wrapper_state(cb).su_sc_mode = u32(4) & 0x393fff; break;
        case kSetStages: wrapper_state(cb).stages = u32(4); break;
        case kSetColorControl: wrapper_state(cb).color_control = ((u32(4) << 4) & 0x70) | ((u32(3) << 16) & 0xff0000); break;
        case kSetRenderControl: wrapper_state(cb).render_control = u32(4) & 0xfff; break;
        case kSetStencilClear: wrapper_state(cb).stencil_clear = u32(4); break;
        case kSetVportScissor: {
            if (u32(4) != 0) break;
            WrapperState& w = wrapper_state(cb);
            w.vport_scissor[0] = (u32(7) == 0 ? 0x80000000u : 0u) | ((u32(2) << 16) & 0x7fff0000) | (u32(3) & 0x7fff);
            w.vport_scissor[1] = ((u32(0) << 16) & 0x7fff0000) | (u32(1) & 0x7fff);
            break;
        }
        case kSetIndxOffset: wrapper_state(cb).indx_offset = u32(4); break;
        case kSetCs: wrapper_state(cb).cs_gnm = f[4] ? f[4] - 8 : 0; break;
        case kSetBase: wrapper_state(cb).indirect_base = f[4] & ~7ull; break;
        case kSetIndexBuffer: wrapper_state(cb).index_base = f[4]; break;
        default: break;
    }
    return suppress(1);  // not a user-data write: the stage does not matter
}

std::mutex g_sf_mu;
std::unordered_map<std::uint64_t, ScaleformDraw> g_sf_draws;  // by token id, until the CP walks the token
enum SfRefusal { kSfRefNoBuilder, kSfRefNoPair, kSfRefPairDiffers, kSfRefLayout, kSfRefTable, kSfRefMode, kSfRefNoTarget, kSfRefNoIndexBuffer, kSfRefusals };
const char* const kSfRefusalName[kSfRefusals] = {"no builder", "no shader pair", "pair differs from the builder's", "layout slot out of range",
                                                  "table unreadable", "mode out of range", "no render target", "no index buffer"};
std::atomic<std::uint64_t> g_sf_calls[3] = {}, g_sf_tokens{0}, g_sf_unwritten{0}, g_sf_refused[kSfRefusals] = {}, g_sf_unmatched{0}, g_sf_drawn{0},
    g_sf_placed_failed{0}, g_sf_direct_layouts{0}, g_sf_no_viewport{0};
std::set<std::uint64_t> g_sf_layouts_seen;  // under g_sf_mu

// A Scaleform or wrapper draw: its record goes in the map, then its token in
// the command buffer - so the command processor finds the record whenever it
// walks the token - without g_sf_mu held over the write (write_native_token).
bool sf_record_then_token(std::uint64_t ctx, ScaleformDraw&& d) {
    const std::uint64_t id = next_native_id();
    {
        std::lock_guard<std::mutex> lk(g_sf_mu);
        if (g_sf_draws.size() > 16384) g_sf_draws.clear();
        g_sf_draws[id] = std::move(d);
    }
    if (write_native_token(ctx, kGxScaleformMagic, id)) return true;
    g_sf_unwritten.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_sf_mu);
    g_sf_draws.erase(id);
    return false;
}

// SPI_PS_INPUT_CNTL from the two Gnm shader structures, as build_inputs()
// does it for GX (the shader bind 0x2ace1e0 and the wrapper's flush both
// compute it through 0x14873a0 from the same structures).
void sf_ps_inputs(std::uint64_t vs_gnm, std::uint64_t ps_gnm, GpuDrawInputs& in) {
    const auto u8 = [](std::uint64_t va) { return rd32(va) & 0xff; };
    const std::uint64_t vs_out = vs_gnm + ((static_cast<std::uint64_t>(u8(vs_gnm + 0x24)) + u8(vs_gnm + 3)) << 2) + 0x28;
    const std::uint32_t vs_outs = u8(vs_gnm + 0x25);
    const std::uint64_t ps_in = ps_gnm + (static_cast<std::uint64_t>(u8(ps_gnm + 3)) << 2) + 0x3c;
    in.ps_input_count = std::min<std::uint32_t>(u8(ps_gnm + 0x38), 32);
    for (std::uint32_t i = 0; i < in.ps_input_count; ++i) {
        const std::uint32_t word = rd32(ps_in + 2 * static_cast<std::uint64_t>(i)) & 0xffff;
        std::uint32_t reg = 0x20;
        for (std::uint32_t j = 0; j < vs_outs; ++j) {
            if (u8(vs_out + 2 * static_cast<std::uint64_t>(j)) == (word & 0xff)) {
                reg = u8(vs_out + 2 * static_cast<std::uint64_t>(j) + 1);
                break;
            }
        }
        in.ps_input_cntl[i] = (((word >> 7) & 0x1e0) | reg) & 0x3f;
        in.ps_input_cntl[i] |= ((word >> 8) & 3) << 8;
        in.ps_input_cntl[i] |= (((word >> 12) | (word >> 10)) & 1) << 10;
    }
}

// One stage's user data as the wrapper's flush (0x1498ee0) builds it from
// the layout (the shader's resource layout, at the wrapper +0x2a): byte 0 the
// table's dwords; bytes 4..0xf the user-data SGPR of the fetch shader (4), the
// sub-tables (5, 6, 8..0xb, at the u16 dword offsets +0x18, +0x20, +0x16,
// +0x1a, +0x1c, +0x1e), the table itself (7), a global pointer (0xc, the
// builder's +0x58), two per-stage dwords (0xd, 0xe) and for the vertex stage
// a thread count (0xf); u16 +0xa0 the slot of the shader's own data as a V#,
// bit 15 of a slot meaning the SGPRs directly. The table dwords come from the
// builder's stage array; the copy is placed in the host ring when the CP walks
// the token, which is when the pointers are filled in.
bool sf_user_data(std::uint64_t rt, int stage, std::uint64_t layout, std::uint64_t gnm, std::uint64_t fetch, SfStage& s,
                  std::uint32_t user[16], SfRefusal* why) {
    const auto u8 = [&](int k) { return rd32(layout + static_cast<std::uint64_t>(k)) & 0xff; };
    const auto u16 = [&](int k) { return rd32(layout + static_cast<std::uint64_t>(k)) & 0xffff; };
    const std::uint32_t n = u8(0);
    const std::uint64_t arr = rt + 0x70 + static_cast<std::uint64_t>(stage) * 0x1800;
    s.table.assign(n, 0);
    if (n && !safe_read(arr, s.table.data(), static_cast<std::size_t>(n) * 4)) {
        *why = kSfRefTable;
        return false;
    }
    bool ok = true;
    const auto set_ptr = [&](int k, std::uint32_t off) {
        const std::uint32_t ud = u8(k);
        if (ud == 0xff) return;
        if (ud + 1 >= 16 || s.nptrs >= 7 || off >= n) {
            ok = false;
            return;
        }
        s.ptrs[s.nptrs++] = {static_cast<std::uint8_t>(ud), off};
        s.named |= 3u << ud;
    };
    const auto set_value = [&](int k, std::uint64_t v) {
        const std::uint32_t ud = u8(k);
        if (ud == 0xff) return;
        if (ud + 1 >= 16) {
            ok = false;
            return;
        }
        user[ud] = static_cast<std::uint32_t>(v);
        user[ud + 1] = static_cast<std::uint32_t>(v >> 32);
        s.named |= 3u << ud;
    };
    const auto set_dword = [&](int k, std::uint32_t v) {
        const std::uint32_t ud = u8(k);
        if (ud == 0xff) return;
        if (ud >= 16) {
            ok = false;
            return;
        }
        user[ud] = v;
        s.named |= 1u << ud;
    };
    // The shader's own data as a V# (the flush writes it into the array before
    // the copy, so it goes into the copy here).
    const std::uint32_t slot = u16(0xa0), size16 = rd32(gnm + 4) & 0xffff;
    if (slot != 0xffff && size16) {
        const std::uint64_t addr = ((static_cast<std::uint64_t>(rd32(gnm + 0xc)) << 40) | (static_cast<std::uint64_t>(rd32(gnm + 8)) << 8)) +
                                   (rd32(gnm) & 0x7fffff);
        const std::uint32_t v[4] = {static_cast<std::uint32_t>(addr), (static_cast<std::uint32_t>(addr >> 32) & 0xfff) | 0x100000, size16,
                                    0x20077fac};
        const std::uint32_t idx = slot & 0x3fff;
        bool placed = false;
        if (slot & 0x8000) {
            if (idx + 3 < 16) {
                std::memcpy(user + idx, v, sizeof(v));
                s.named |= 0xfu << idx;
                placed = true;
            }
        } else if (idx + 3 < n) {
            std::memcpy(s.table.data() + idx, v, sizeof(v));
            placed = true;
        }
        // The block's bytes now: the game writes the next frame's at the same
        // addresses, and the command processor, then the GPU, read it a frame
        // or more after this call.
        if (placed && g_sf_uniforms && size16 <= 256) {
            s.uniforms.resize(static_cast<std::size_t>(size16) * 4);
            if (safe_read(addr, s.uniforms.data(), s.uniforms.size() * 4)) {
                s.uniform_ud = static_cast<int>(idx);
                s.uniform_direct = (slot & 0x8000) != 0;
            } else {
                s.uniforms.clear();
            }
        }
    }
    set_ptr(7, 0);
    set_ptr(8, u16(0x16));
    set_ptr(9, u16(0x1a));
    set_ptr(0xa, u16(0x1c));
    set_ptr(0xb, u16(0x1e));
    if (stage == 2) {
        set_ptr(5, u16(0x18));
        set_ptr(6, u16(0x20));
        set_value(4, fetch);
    }
    set_value(0xc, rd64(rt + 0x58));
    set_dword(0xd, rd32(rt + 0xc0b8 + static_cast<std::uint64_t>(stage) * 4));
    set_dword(0xe, rd32(rt + 0xc0d8 + static_cast<std::uint64_t>(stage) * 4));
    if (stage == 2) set_dword(0xf, (rd32(rt + 0xc558) & 0xffff) * ((rd32(rt + 0xc558) >> 16) & 0xffff));
    // Slots the layout routes to the SGPRs directly (textures at +0x22,
    // samplers +0x62, constant buffers +0x82, vertex buffers +0xaa): their
    // records were written as packets when appended, not into the arrays;
    // the wrapper's user-data write is hooked and the token takes them from
    // what it wrote on this command buffer.
    std::uint32_t cb_direct = 0;  // the direct slots that start a constant buffer's V#
    for (int k = 0x22; k < 0xca; k += 2) {
        if (k == 0xa0) continue;
        const std::uint32_t e = u16(k);
        if (e != 0xffff && (e & 0x8000)) {
            s.direct |= 1u << std::min<std::uint32_t>(e & 0x3fff, 31);
            if (k >= 0x82 && k < 0xa0) cb_direct |= 1u << std::min<std::uint32_t>(e & 0x3fff, 31);
        }
    }
    if (s.direct) {
        WrapperState ud_copy;
        const WrapperState* ud = wrapper_state_copy(rt - 0x2c8, ud_copy) ? &ud_copy : nullptr;
        std::uint32_t pointer_slots = 0;
        for (int i = 0; i < s.nptrs; ++i) pointer_slots |= 3u << s.ptrs[i].ud;
        for (int k = 0; k < 16; ++k) {
            if (((s.named | pointer_slots) >> k) & 1) continue;
            if (!ud || !((ud->set[stage] >> k) & 1)) continue;
            user[k] = ud->user[stage][k];
            s.named |= 1u << k;
        }
        // A constant buffer routed straight to the SGPRs (the uniform block
        // here: VS s[8..11]): its bytes as they are now, when no uniform block
        // was taken through the layout's own slot above.
        for (int k = 0; g_sf_uniforms && s.uniforms.empty() && k + 3 < 16; ++k) {
            if (!((cb_direct >> k) & 1) || (s.named & (0xfu << k)) != (0xfu << k)) continue;
            const std::uint64_t base = user[k] | (static_cast<std::uint64_t>(user[k + 1] & 0xfff) << 32);
            const std::uint32_t stride = (user[k + 1] >> 16) & 0x3fff, records = user[k + 2];
            const std::uint64_t bytes = static_cast<std::uint64_t>(records) * (stride ? stride : 1);
            if (!base || !bytes || bytes > 4096 || (bytes & 3)) continue;
            s.uniforms.resize(static_cast<std::size_t>(bytes / 4));
            if (safe_read(base, s.uniforms.data(), static_cast<std::size_t>(bytes))) {
                s.uniform_ud = k;
                s.uniform_direct = true;
            } else {
                s.uniforms.clear();
            }
        }
    }
    if (!ok) *why = kSfRefLayout;
    return ok;
}

bool build_scaleform_draw(Kind kind, const std::uint64_t* f, ScaleformDraw& d, SfRefusal* why) {
    const std::uint64_t hal = f[5];
    const std::uint32_t count = static_cast<std::uint32_t>(f[4]);
    const bool indexed = kind != kScaleformDrawAuto;
    const std::uint32_t index_offset = indexed ? static_cast<std::uint32_t>(f[1]) : 0;
    const std::uint32_t instances = kind == kScaleformDrawInstanced ? static_cast<std::uint32_t>(f[2]) : 1;
    d.hal = hal;
    d.builder = hal ? rd64(hal + 0x2ca30) : 0;
    if (!d.builder) {
        *why = kSfRefNoBuilder;
        return false;
    }
    const std::uint64_t rt = d.builder + 0x2c8, si = hal + 0x2b828;
    const std::uint64_t vs_pair = rd64(si + 0x1158), ps_pair = rd64(si + 0x1168);
    const std::uint64_t vs_gnm = vs_pair ? rd64(vs_pair + 8) : 0, ps_gnm = ps_pair ? rd64(ps_pair + 8) : 0;
    if (!vs_gnm || !ps_gnm) {
        *why = kSfRefNoPair;
        return false;
    }
    const std::uint64_t vs_layout = vs_pair + 0x2a, ps_layout = ps_pair + 0x2a;
    if (vs_gnm != rd64(rt + 0xc088) || ps_gnm != rd64(rt + 0xc080) || vs_layout != rd64(rt + 0xc168) || ps_layout != rd64(rt + 0xc160)) {
        *why = kSfRefPairDiffers;
        return false;
    }
    const std::uint64_t fetch = rd64(vs_pair + 0x118);
    const std::uint32_t modifier = rd32(vs_pair + 0x128);
    GpuDrawInputs& in = d.in;
    in = GpuDrawInputs{};
    for (int i = 0; i < 4; ++i) in.vs_pgm[i] = rd32(vs_gnm + 8 + 4 * static_cast<std::uint64_t>(i));
    in.vs_pgm[2] |= modifier;
    in.vs_out_cntl = rd32(vs_gnm + 8 + 0x18);
    for (int i = 0; i < 4; ++i) in.ps_pgm[i] = rd32(ps_gnm + 8 + 4 * static_cast<std::uint64_t>(i));
    in.ps_input_ena = rd32(ps_gnm + 8 + 0x18);
    in.ps_in_control = rd32(ps_gnm + 8 + 0x20);
    in.cb_shader_mask = rd32(ps_gnm + 8 + 0x2c);
    in.ps_col_format = rd32(ps_gnm + 8 + 0x14);
    sf_ps_inputs(vs_gnm, ps_gnm, in);
    if (!sf_user_data(rt, 2, vs_layout, vs_gnm, fetch, d.stage[0], in.vs_user, why)) return false;
    if (!sf_user_data(rt, 1, ps_layout, ps_gnm, 0, d.stage[1], in.ps_user, why)) return false;
    in.prim = 4;  // BeginScene: triangle lists
    in.vte_cntl = 0x43f;

    // Depth-stencil mode (0x73de50).
    d.ds_mode = rd32(hal + 0x210);
    if (d.ds_mode >= kSfDsModeCount) {
        *why = kSfRefMode;
        return false;
    }
    {
        const std::uint64_t m = guest(kSfDsModes) + static_cast<std::uint64_t>(d.ds_mode) * 36;
        std::uint32_t w[9];
        for (int i = 0; i < 9; ++i) w[i] = rd32(m + 4 * static_cast<std::uint64_t>(i));
        const auto cmp = [&](std::uint32_t k) { return rd32(guest(kSfCmpFuncs) + 4 * static_cast<std::uint64_t>(k & 0xf)); };
        const auto op = [&](std::uint32_t k) { return rd32(guest(kSfStencilOps) + 4 * static_cast<std::uint64_t>(k & 0x3)); };
        in.target_mask = w[3] ? 0xf : 0;
        std::uint32_t dc = w[2] != 0 ? 1u : 0u;
        if (w[2]) dc |= (cmp(w[5]) & 7) << 8;
        dc |= (cmp(w[4]) << 4) & 0x70;
        dc |= (w[1] == 0 ? 1u : 0u) << 2;
        dc |= (w[0] != 0 ? 1u : 0u) << 1;
        in.depth_control = dc & 0xc07007ff;
        in.stencil_control = w[2] ? (((op(w[6]) & 0xf) << 4) | (op(w[7]) & 0xf) | ((op(w[8]) & 0xf) << 8)) : 0;
        const std::uint32_t ref = rd32(hal + 0x214) & 0xff;
        in.stencil_ref = in.stencil_ref_bf = ref | 0x1ffff00;
    }
    // Blend mode (0x495d30 keeps it, 0x73e0d0 applies it).
    {
        const std::uint32_t mode = rd32(hal + 0x224);
        d.blend_mode = mode <= 0x13 ? mode : 0;
        const bool source_ac = (rd32(hal + 0x228) & 0xff) != 0, force_ac = ((rd32(hal + 0x228) >> 8) & 0xff) != 0;
        const std::uint64_t m = guest(kSfBlendModes) + static_cast<std::uint64_t>(d.blend_mode) * 24;
        std::uint32_t t[6];
        for (int i = 0; i < 6; ++i) t[i] = rd32(m + 4 * static_cast<std::uint64_t>(i));
        const auto factor = [&](std::uint32_t k) { return rd32(guest(kSfBlendFactors) + 4 * static_cast<std::uint64_t>(k & 0x1f)); };
        const auto func = [&](std::uint32_t k) { return rd32(guest(kSfBlendFuncs) + 4 * static_cast<std::uint64_t>(k & 0x7)); };
        std::uint32_t src = factor(t[1]);
        if (source_ac && t[1] == 2) src = 1;
        std::uint32_t v = 1u << 30;
        v |= (src & 0x1f) | ((func(t[0]) & 7) << 5) | ((factor(t[2]) & 0x1f) << 8);
        // The alpha equation only with the viewport's alpha-composite flag
        // (VP.Flags, HAL+0x1190, bit 1) or forceAc.
        d.vp_flags = rd32(hal + 0x1190);
        if ((d.vp_flags & 2) || force_ac) {
            v |= 1u << 29;
            v |= ((factor(t[4]) & 0x1f) << 16) | ((func(t[3]) & 7) << 21) | ((factor(t[5]) & 0x1f) << 24);
        }
        in.blend[0] = v & 0xffff1fff;
    }
    // The render target: the top of the HAL's stack.
    {
        const std::uint32_t depth = rd32(hal + 0x1c8);
        const std::uint64_t stack = rd64(hal + 0x1c0);
        const std::uint64_t entry = depth && stack ? stack + static_cast<std::uint64_t>(depth - 1) * 0x320 : 0;
        const std::uint64_t obj = entry ? rd64(entry) : 0;
        const std::uint64_t buffer = obj ? rd64(obj + 0x18) : 0;
        const std::uint64_t target = buffer ? rd64(buffer + 0x20) : 0;
        if (!target) {
            *why = kSfRefNoTarget;
            return false;
        }
        for (int k = 0; k < 5; ++k) in.color[0][k] = rd32(target + 4 * static_cast<std::uint64_t>(k));
        if (const std::uint64_t ds = rd64(buffer + 0x28)) {
            in.z_info = rd32(ds);
            in.stencil_info = rd32(ds + 4);
            in.z_read_base = rd32(ds + 8);
            in.depth_size = rd32(ds + 0x18);
            in.depth_view = rd32(ds + 0x20);
            in.htile_base = rd32(ds + 0x24);
        }
    }
    // Raster mode (0x73e210): the primitive setup's polygon mode.
    {
        d.raster_mode = rd32(hal + 0x21c);
        const std::uint32_t p = d.raster_mode == 2 ? 0 : d.raster_mode == 1 ? 1 : 2;
        in.su_sc_mode = (((p & 7) << 8) | ((p & 7) << 5) | ((p != 2 ? 1u : 0u) << 3)) & 0x393fff;
    }
    // The viewport (0x73d930 -> setupScreenViewport 0x1496f60).
    {
        d.hal_flags = rd32(hal + 0x5c);
        if (!(d.hal_flags & 0x20)) g_sf_no_viewport.fetch_add(1, std::memory_order_relaxed);
        std::int32_t l, t, r, b;
        if (d.hal_flags & 0x10) {
            const auto x = static_cast<std::int32_t>(rd32(hal + 0x1170)), y = static_cast<std::int32_t>(rd32(hal + 0x1174));
            l = x;
            t = y;
            r = x + static_cast<std::int32_t>(rd32(hal + 0x1178));
            b = y + static_cast<std::int32_t>(rd32(hal + 0x117c));
        } else {
            l = static_cast<std::int32_t>(rd32(hal + 0x1194));
            t = static_cast<std::int32_t>(rd32(hal + 0x1198));
            r = static_cast<std::int32_t>(rd32(hal + 0x119c));
            b = static_cast<std::int32_t>(rd32(hal + 0x11a0));
        }
        if (r == l || b == t) {
            l = t = 0;
            r = b = 1;
        }
        d.rect[0] = l;
        d.rect[1] = t;
        d.rect[2] = r;
        d.rect[3] = b;
        const float w = static_cast<float>(r - l), h = static_cast<float>(b - t);
        in.vport[0] = w * 0.5f;
        in.vport[1] = static_cast<float>(l) + w * 0.5f;
        in.vport[2] = h * -0.5f;
        in.vport[3] = static_cast<float>(t) + h * 0.5f;
        in.vport[4] = 0.5f;
        in.vport[5] = 0.5f;
        const auto pack = [](std::int32_t x, std::int32_t y) {
            return (static_cast<std::uint32_t>(y) << 16) | (static_cast<std::uint32_t>(x) & 0xffff);
        };
        in.screen_scissor[0] = pack(l, t);
        in.screen_scissor[1] = pack(r, b);
    }
    // The geometry.
    d.obj = GxDrawObjects{};
    d.obj.geometry = true;
    d.obj.count = count;
    d.obj.instances = instances ? instances : 1;
    d.obj.indexed = indexed;
    d.obj.caller = kind == kScaleformDraw ? kSfHalDraw : kind == kScaleformDrawInstanced ? kSfHalDrawInstanced : kSfHalDrawAuto;
    d.obj.call_flip = hle_video_flip_count();
    d.obj.fetch_va = fetch;
    d.obj.vtx_ud = 0xff;
    if (indexed) {
        const std::uint64_t base = sf_index_base(d.builder);
        if (!base) {
            *why = kSfRefNoIndexBuffer;
            return false;
        }
        d.obj.index_type = 0;
        d.obj.index_va = (base & ~1ull) + static_cast<std::uint64_t>(index_offset) * 2;
        d.obj.index_known = true;
    }
    return true;
}

// f: rdi HAL, esi count, (instanced: ecx instances), r8 index offset.
std::uint64_t native_scaleform_draw(Kind kind, const std::uint64_t* f) {
    if (!g_native_scaleform || g_backend != 1) return 0;
    if (g_native_scaleform == 2 && g_native != 2) return 0;
    g_sf_calls[kind - kScaleformDraw].fetch_add(1, std::memory_order_relaxed);
    ScaleformDraw d;
    SfRefusal why = kSfRefNoBuilder;
    if (!build_scaleform_draw(kind, f, d, &why)) {
        g_sf_refused[why].fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    if (d.stage[0].direct || d.stage[1].direct) {
        if (g_sf_direct_layouts.fetch_add(1, std::memory_order_relaxed) < 2) {
            host_log("gx-scaleform: layout routes slots to SGPRs directly (VS slots %08x, PS slots %08x); taken from the wrapper's writes",
                     d.stage[0].direct, d.stage[1].direct);
        }
    }
    const std::uint64_t ctx = d.builder - 0x10;
    if (g_gnm_sources) note_source(d.builder, rd64(d.builder + 0x10), rd64(d.builder + 0x8), kSourceScaleform);
    // The record is in the map before the token can be walked, and the lock
    // is not held over the write: the chunk's refill there can submit and
    // wait for the command processor, which takes the lock to look the id up
    // (a Steam Deck run froze that way at a load, the backlog full).
    d.shadow = g_native_scaleform != 2;
    if (!sf_record_then_token(ctx, std::move(d))) return 0;
    g_sf_tokens.fetch_add(1, std::memory_order_relaxed);
    return g_native_scaleform == 2 ? 1 : 0;
}

void appendf(std::string& s, const char* fmt, ...);
// The draw's state - targets, blend, depth-stencil, viewport, scissors,
// primitive, rasterizer - from the wrapper's context state; not the programs
// or user data.
void wrapper_fill_state(const WrapperState& w, GpuDrawInputs& in) {
    in.prim = w.prim;
    in.target_mask = w.target_mask;
    for (int t = 0; t < 8; ++t) {
        if ((w.rt_set >> t) & 1) std::memcpy(in.color[t], w.rt[t], sizeof(in.color[t]));
        else std::memset(in.color[t], 0, sizeof(in.color[t]));
        in.blend[t] = w.blend[t];
    }
    in.depth_control = w.ds_control;
    in.stencil_control = w.stencil_control;
    in.stencil_ref = in.stencil_ref_bf = w.stencil_ref;
    in.z_info = in.stencil_info = in.z_read_base = in.depth_size = in.depth_view = in.htile_base = 0;
    if (w.dsv_set) {
        in.z_info = w.dsv[0];
        in.stencil_info = w.dsv[1];
        in.z_read_base = w.dsv[2];
        in.depth_size = w.dsv[6];
        in.depth_view = w.dsv[8];
        in.htile_base = w.dsv[9];
    }
    in.render_control = w.render_control;
    in.su_sc_mode = w.su_sc_mode;
    in.clip_cntl = w.clip;
    in.vte_cntl = w.vte;
    in.sc_mode_cntl_0 = w.sc_mode;
    std::memcpy(in.vport, w.vport, sizeof(in.vport));
    in.screen_scissor[0] = w.screen_scissor[0];
    in.screen_scissor[1] = w.screen_scissor[1];
    in.vport_scissor[0] = w.vport_scissor[0];
    in.vport_scissor[1] = w.vport_scissor[1];
    in.base_vertex = static_cast<std::int32_t>(w.indx_offset);
}
bool wrapper_complete_inputs(std::uint64_t cb, GpuDrawInputs& in, std::uint32_t* index_type) {
    WrapperState w;
    if (!wrapper_state_copy(cb, w)) return false;
    wrapper_fill_state(w, in);
    *index_type = w.index_size & 1;
    return true;
}

// An emitter hook for a draw from a caller that is not a GX draw method (the
// gamma pass, the present draw, whatever else emits through the wrapper): 0
// not taken, 1 a token was written and the packet still goes (shadow), 2 the
// token replaces the packet.
int wrapper_draw(Kind kind, const std::uint64_t* f, std::uint64_t caller) {
    if (!g_native_utility || g_backend != 1) return 0;
    const bool gx_caller = caller >= kGxImmediateLo && caller < kGxImmediateHi;
    if (gx_caller) return 0;
    if (kind == kEmitDispatch || kind == kEmitDispatchIndirect || kind == kEmitDispatchQueue) return 0;  // compute keeps its packets
    const bool kind_ok = kind == kEmitAuto || kind == kEmitIndex2 || kind == kEmitIndex2b || kind == kEmitOffset2;
    const std::uint64_t cb = f[5];
    WrapperState w_copy;
    const WrapperState* w = kind_ok && wrapper_state_copy(cb, w_copy) ? &w_copy : nullptr;
    const bool state_ok = w && w->vs_gnm && w->ps_gnm && (kind != kEmitOffset2 || w->index_base);
    if (!kind_ok || !state_ok) {
        if (kind_ok) g_wd_refused.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(g_sf_mu);
        if (g_wd_other_callers.size() < 64 || g_wd_other_callers.count(caller)) ++g_wd_other_callers[caller];
        return 0;
    }
    if (g_native_utility == 2 && g_native != 2) return 0;
    ScaleformDraw d;
    GpuDrawInputs& in = d.in;
    in = GpuDrawInputs{};
    d.builder = cb;
    for (int i = 0; i < 4; ++i) in.vs_pgm[i] = rd32(w->vs_gnm + 8 + 4 * static_cast<std::uint64_t>(i));
    in.vs_pgm[2] |= w->vs_mod;
    in.vs_out_cntl = rd32(w->vs_gnm + 8 + 0x18);
    for (int i = 0; i < 4; ++i) in.ps_pgm[i] = rd32(w->ps_gnm + 8 + 4 * static_cast<std::uint64_t>(i));
    in.ps_input_ena = rd32(w->ps_gnm + 8 + 0x18);
    in.ps_in_control = rd32(w->ps_gnm + 8 + 0x20);
    in.cb_shader_mask = rd32(w->ps_gnm + 8 + 0x2c);
    in.ps_col_format = rd32(w->ps_gnm + 8 + 0x14);
    sf_ps_inputs(w->vs_gnm, w->ps_gnm, in);
    std::memcpy(in.vs_user, w->user[2], sizeof(in.vs_user));
    std::memcpy(in.ps_user, w->user[1], sizeof(in.ps_user));
    {
        // BBHOST_GLITCH=1: the gamma pass with nothing in the pixel stage's
        // s[2:3] (its lookup table's T# is read through it: a black frame).
        static const bool watch = [] {
            const char* e = std::getenv("BBHOST_GLITCH");
            return e && e[0] == '1';
        }();
        static std::atomic<int> logs{0};
        if (watch && caller == kGammaPassDraw && !in.ps_user[2] && !in.ps_user[3] && logs.fetch_add(1, std::memory_order_relaxed) < 60) {
            host_log("glitch: gamma pass at flip %llu on cb 0x%llx (made at flip %u): PS s[2:3] is 0, %s; PS slots set %04x, "
                     "s[0] set at %u, s[4] at %u",
                     static_cast<ull>(hle_video_flip_count()), static_cast<ull>(cb), w->created_flip,
                     !(w->set[1] & 0xc) ? "never set on this context" : ("last set at flip " + std::to_string(w->set_flip[1][2] & ~1u)).c_str(),
                     w->set[1], w->set_flip[1][0] & ~1u, w->set_flip[1][4] & ~1u);
        }
    }
    d.stage[0].named = w->set[2];
    d.stage[1].named = w->set[1];
    wrapper_fill_state(*w, in);
    in.complete = true;
    d.obj = GxDrawObjects{};
    d.obj.geometry = true;
    d.obj.count = static_cast<std::uint32_t>(kind == kEmitOffset2 ? f[3] : f[4]);
    d.obj.instances = w->instances ? w->instances : 1;
    d.obj.caller = caller;
    d.obj.call_flip = hle_video_flip_count();
    d.obj.vtx_ud = 0xff;
    if (kind == kEmitIndex2 || kind == kEmitIndex2b) {
        d.obj.indexed = true;
        d.obj.index_va = f[3];
        d.obj.index_type = w->index_size & 1;
        d.obj.index_known = true;
    } else if (kind == kEmitOffset2) {
        // 0x1474420(cb, index offset, count): the index buffer the wrapper bound.
        d.obj.indexed = true;
        d.obj.index_type = w->index_size & 1;
        d.obj.index_va = (w->index_base & ~1ull) + f[4] * (d.obj.index_type ? 4 : 2);
        d.obj.index_known = true;
    }
    if (!d.obj.count) return 0;
    d.shadow = g_native_utility != 2;
    if (!sf_record_then_token(cb - 0x10, std::move(d))) return 0;
    g_wd_taken[caller == kPresentDraw ? 1 : caller == kGammaPassDraw ? 0 : 2].fetch_add(1, std::memory_order_relaxed);
    if (caller != kPresentDraw && caller != kGammaPassDraw) {
        std::lock_guard<std::mutex> lk(g_sf_mu);
        if (g_wd_taken_callers.size() < 64 || g_wd_taken_callers.count(caller)) ++g_wd_taken_callers[caller];
    }
    return g_native_utility == 2 ? 2 : 1;
}

// A dispatch emitted through the wrapper (GX's Dispatch method,
// YEBIS's compute passes, the flip's two a frame) as a token from the
// wrapper's state: the compute program (0x1475840) and its user data (stage
// 0), the workgroup counts from the call, an indirect dispatch's arguments at
// SET_BASE plus the offset. 0 not taken, 1 token and packet (shadow), 2 the
// token replaces the packet.
struct DispatchRecord {
    GpuDispatch d;
    std::uint64_t caller = 0;
    bool shadow = false;
};
std::mutex g_ds_mu;
std::unordered_map<std::uint64_t, DispatchRecord> g_ds_records;
std::map<std::uint64_t, std::uint64_t> g_ds_callers, g_ds_refused_callers;  // under g_ds_mu
std::atomic<std::uint64_t> g_ds_tokens{0}, g_ds_refused{0}, g_ds_unmatched{0}, g_ds_run{0}, g_ds_compared{0}, g_ds_bad[6] = {};
const char* const kDsFieldName[6] = {"the program", "RSRC1/2", "the thread counts", "the user data", "the workgroup counts", "the indirect address"};
int wrapper_dispatch(Kind kind, const std::uint64_t* f, std::uint64_t caller) {
    if (!g_native_dispatch || g_backend != 1) return 0;
    if (kind != kEmitDispatch && kind != kEmitDispatchQueue && kind != kEmitDispatchIndirect) return 0;
    if (g_native_dispatch == 2 && g_native != 2) return 0;
    const std::uint64_t cb = f[5];
    WrapperState w_copy;
    const WrapperState* w = wrapper_state_copy(cb, w_copy) ? &w_copy : nullptr;
    const bool ok = w && w->cs_gnm && (kind != kEmitDispatchIndirect || w->indirect_base);
    if (!ok) {
        g_ds_refused.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(g_ds_mu);
        if (g_ds_refused_callers.size() < 64 || g_ds_refused_callers.count(caller)) ++g_ds_refused_callers[caller];
        return 0;
    }
    if (g_ud_trace) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1, std::memory_order_relaxed) < 20) {
            host_log("ud-trace: dispatch from 0x%llx cb 0x%llx, compute slots set %04x (this buffer's %04x)", static_cast<ull>(caller),
                     static_cast<ull>(cb), t_cs_user.set, w->set[0]);
        }
    }
    DispatchRecord r;
    const std::uint64_t regs = w->cs_gnm + 8;
    r.d.code_va = (static_cast<std::uint64_t>(rd32(regs + 4) & 0xff) << 40) | (static_cast<std::uint64_t>(rd32(regs)) << 8);
    r.d.rsrc1 = rd32(regs + 8);
    r.d.rsrc2 = rd32(regs + 12);
    r.d.threads[0] = rd32(regs + 16);
    r.d.threads[1] = rd32(regs + 20);
    r.d.threads[2] = rd32(regs + 24);
    std::memcpy(r.d.user_data, t_cs_user.user, sizeof(r.d.user_data));
    if (kind == kEmitDispatchIndirect) {
        r.d.indirect_va = w->indirect_base + (f[4] & 0xffffffff);
        r.d.dim[0] = r.d.dim[1] = r.d.dim[2] = 1;
    } else {
        r.d.dim[0] = static_cast<std::uint32_t>(f[4]);
        r.d.dim[1] = static_cast<std::uint32_t>(f[3]);
        r.d.dim[2] = static_cast<std::uint32_t>(f[2]);
    }
    r.caller = caller;
    r.shadow = g_native_dispatch != 2;
    // The record first, then the token, the lock not held over the write
    // (as for Scaleform's: the refill can wait for the command processor).
    const std::uint64_t id = next_native_id();
    {
        std::lock_guard<std::mutex> lk(g_ds_mu);
        if (g_ds_records.size() > 16384) g_ds_records.clear();
        g_ds_records[id] = r;
    }
    if (!write_native_token(cb - 0x10, kGxDispatchMagic, id)) {
        std::lock_guard<std::mutex> lk(g_ds_mu);
        g_ds_records.erase(id);
        return 0;
    }
    {
        std::lock_guard<std::mutex> lk(g_ds_mu);
        if (g_ds_callers.size() < 64 || g_ds_callers.count(caller)) ++g_ds_callers[caller];
    }
    g_ds_tokens.fetch_add(1, std::memory_order_relaxed);
    return g_native_dispatch == 2 ? 2 : 1;
}

thread_local bool t_ds_pending = false;
thread_local DispatchRecord t_ds_rec;
const GpuDispatch* dispatch_token_impl(std::uint64_t id) {
    DispatchRecord r;
    {
        std::lock_guard<std::mutex> lk(g_ds_mu);
        const auto it = g_ds_records.find(id);
        if (it == g_ds_records.end()) {
            g_ds_unmatched.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        r = it->second;
        g_ds_records.erase(it);
    }
    if (r.shadow) {
        t_ds_rec = r;
        t_ds_pending = true;
        return nullptr;
    }
    static thread_local GpuDispatch t_ds;
    t_ds = r.d;
    g_ds_run.fetch_add(1, std::memory_order_relaxed);
    return &t_ds;
}
bool dispatch_pending_impl() { return t_ds_pending; }
void dispatch_check_impl(const GpuDispatch& regs, const std::uint32_t* writers) {
    t_ds_pending = false;
    const GpuDispatch& d = t_ds_rec.d;
    bool bad[6] = {};
    bad[0] = d.code_va != regs.code_va;
    bad[1] = d.rsrc1 != regs.rsrc1 || d.rsrc2 != regs.rsrc2;
    bad[2] = std::memcmp(d.threads, regs.threads, sizeof(d.threads)) != 0;
    bad[3] = std::memcmp(d.user_data, regs.user_data, sizeof(d.user_data)) != 0;
    // An indirect dispatch's counts come from memory at the token, so the
    // placeholder the call carries is not a difference.
    bad[4] = !d.indirect_va && std::memcmp(d.dim, regs.dim, sizeof(d.dim)) != 0;
    bad[5] = d.indirect_va != regs.indirect_va;
    g_ds_compared.fetch_add(1, std::memory_order_relaxed);
    bool any = false;
    for (int k = 0; k < 6; ++k) {
        if (!bad[k]) continue;
        any = true;
        g_ds_bad[k].fetch_add(1, std::memory_order_relaxed);
    }
    static std::atomic<int> logs{0};
    if (any && logs.fetch_add(1, std::memory_order_relaxed) < 12) {
        std::string what;
        for (int k = 0; k < 6; ++k) {
            if (bad[k]) appendf(what, " %s;", kDsFieldName[k]);
        }
        std::string ud;
        for (int k = 0; k < 16; ++k) {
            if (d.user_data[k] != regs.user_data[k]) {
                appendf(ud, " s[%d] %08x/%08x", k, d.user_data[k], regs.user_data[k]);
                if (writers) appendf(ud, " (packet %08x marker %08x)", writers[2 * k], writers[2 * k + 1]);
            }
        }
        host_log("gx-dispatch: token from 0x%llx differs from the packet:%s program %llx/%llx dim %u,%u,%u/%u,%u,%u%s",
                 static_cast<ull>(t_ds_rec.caller), what.c_str(), static_cast<ull>(d.code_va), static_cast<ull>(regs.code_va), d.dim[0], d.dim[1],
                 d.dim[2], regs.dim[0], regs.dim[1], regs.dim[2], ud.c_str());
    }
}
void report_state_packets() {
    std::uint64_t calls = 0, dwords = 0;
    std::string by_kind;
    for (int k = 0; k < kKinds; ++k) {
        const std::uint64_t n = g_state_packets_skipped[k].load();
        if (!n) continue;
        calls += n;
        dwords += n * state_packet_dwords(static_cast<Kind>(k));
        appendf(by_kind, " %s %llu;", kKindName[k], static_cast<ull>(n));
    }
    if (!calls) {
        host_log("gx-state: the wrapper's state packets are %s", skip_state_packets() ? "skipped (none seen)" : "written");
        return;
    }
    host_log("gx-state: %llu state packets not written (about %llu dwords, %.1f MiB): %s", static_cast<ull>(calls), static_cast<ull>(dwords),
             static_cast<double>(dwords) * 4 / (1 << 20), by_kind.c_str());
    std::uint64_t dead_calls = 0, dead_dwords = 0;
    std::string dead_by;
    for (std::size_t i = 0; i < kDeadWriterCount; ++i) {
        const std::uint64_t n = g_dead_calls[i].load();
        if (!n) continue;
        dead_calls += n;
        dead_dwords += n * kDeadWriters[i].dwords;
        appendf(dead_by, " %s %llu;", kDeadWriters[i].what, static_cast<ull>(n));
    }
    if (dead_calls) {
        host_log("gx-state: %llu dead packets not written (about %llu dwords, %.1f MiB):%s", static_cast<ull>(dead_calls),
                 static_cast<ull>(dead_dwords), static_cast<double>(dead_dwords) * 4 / (1 << 20), dead_by.c_str());
    }
}

void report_dispatch_tokens() {
    if (!g_ds_tokens.load() && !g_ds_refused.load()) return;
    std::string callers, refused, fields;
    {
        std::lock_guard<std::mutex> lk(g_ds_mu);
        for (const auto& [c, n] : g_ds_callers) appendf(callers, " 0x%llx %llu;", static_cast<ull>(c), static_cast<ull>(n));
        for (const auto& [c, n] : g_ds_refused_callers) appendf(refused, " 0x%llx %llu;", static_cast<ull>(c), static_cast<ull>(n));
    }
    for (int k = 0; k < 6; ++k) {
        if (const std::uint64_t n = g_ds_bad[k].load()) appendf(fields, " %s %llu;", kDsFieldName[k], static_cast<ull>(n));
    }
    host_log("gx-dispatch: tokens %llu (run %llu, unmatched %llu) by caller:%s refused %llu by caller:%s",
             static_cast<ull>(g_ds_tokens.load()), static_cast<ull>(g_ds_run.load()), static_cast<ull>(g_ds_unmatched.load()),
             callers.empty() ? " none;" : callers.c_str(), static_cast<ull>(g_ds_refused.load()), refused.empty() ? " none" : refused.c_str());
    if (const std::uint64_t n = g_ds_compared.load()) {
        host_log("gx-dispatch: shadow: %llu compared; fields differing:%s", static_cast<ull>(n), fields.empty() ? " none" : fields.c_str());
    }
}

// The depth-stencil clear pass (0x2ab5f50: ctx, depth view, resource,
// stencil, flags; the depth in xmm0, which the pass reads back from
// ctx+0xb650): HTILE is filled with the clear word over the view's slices
// and a quad writes the depth and stencil. The fill token carries the depth
// and the stencil; the host's HTILE clear applies both at the target's next
// draw, so the draw is not needed.
std::uint64_t native_clear_pass(const std::uint64_t* f) {
    if (!(g_native_passes & 4) || g_native != 2 || g_backend != 1 || !g_native_fills) return 0;
    const std::uint64_t ctx = f[5], view = f[4], res = f[3];
    if (!ctx || !view || !res) return 0;
    const std::uint32_t stencil = static_cast<std::uint32_t>(f[2]) & 0xff;
    float depth;
    std::memcpy(&depth, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(ctx + 0xb650)), 4);
    const auto scaled = static_cast<std::uint32_t>(std::floor(depth * 16383.0f));
    const bool stencil_plane = (rd32(view + 4) >> 29) & 1;  // 0x1472750: byte 7 bit 5
    const bool htile = (rd32(view) >> 29) & 1;              // 0x1472730: byte 3 bit 5
    const std::uint32_t word = stencil_plane ? (((scaled << 4) & 0x3fff0) | (scaled << 18)) : (scaled << 18);
    GuestFill fill{};
    for (float& c : fill.rgba) std::memcpy(&c, &word, 4);
    fill.clear_known = true;
    std::memcpy(&fill.depth_clear, &depth, 4);
    fill.stencil_clear = stencil;
    if (htile) {
        const std::uint64_t size = static_cast<std::uint64_t>(hle_call_guest<std::int64_t>(guest_fn(kHtileSizeGetter), view)) & 0xffffffff;
        const std::uint32_t slice_start = rd32(view + 0x20) & 0x7ff, slice_max = (rd32(view + 0x20) >> 13) & 0x7ff;
        const std::uint64_t base = static_cast<std::uint64_t>(rd32(view + 0x24)) << 8;
        if (!size || !base) {
            g_wd_clear_refused.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        std::uint64_t va = base;
        if (slice_start) {
            const std::uint32_t slices = slice_max + 1 - slice_start;
            if (slices) va += (size / slices) * slice_start;
        }
        fill.va = va;
        fill.bytes = (size + 15) & ~15ull;
    } else {
        // No HTILE: the pass only draws its quad, and only when asked to clear
        // a plane the target has (0x2ab5f50: flags bit 0 depth, bit 1 stencil).
        const std::uint32_t flags = static_cast<std::uint32_t>(f[1]);
        const bool wanted = (flags & 1) || ((flags & 2) && stencil_plane);
        const std::uint64_t base = static_cast<std::uint64_t>(rd32(view + 8)) << 8;  // DB_Z_READ_BASE
        if (!base) {
            g_wd_clear_refused.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        if (!wanted) {
            g_wd_clear_nothing.fetch_add(1, std::memory_order_relaxed);
            return 1;
        }
        fill.va = base;
        fill.bytes = 0;
        fill.depth_target = true;
        g_wd_clear_no_htile.fetch_add(1, std::memory_order_relaxed);
    }
    std::vector<GuestFill> fills{fill};
    if (!fill_token(ctx, std::move(fills))) return 0;
    g_wd_clear_tokens.fetch_add(1, std::memory_order_relaxed);
    return (g_native_passes & 8) ? 0 : 1;  // bit 8 (checks): the token and the pass both
}

void report_wrapper_draws() {
    if (!g_wd_taken[0].load() && !g_wd_taken[1].load() && !g_wd_taken[2].load() && !g_wd_skipped_depth.load() && !g_wd_clear_tokens.load() &&
        !g_wd_refused.load() && !g_yebis_state_built.load()) return;
    std::string others, taken;
    {
        std::lock_guard<std::mutex> lk(g_sf_mu);
        for (const auto& [caller, n] : g_wd_other_callers) appendf(others, " 0x%llx %llu;", static_cast<ull>(caller), static_cast<ull>(n));
        for (const auto& [caller, n] : g_wd_taken_callers) appendf(taken, " 0x%llx %llu;", static_cast<ull>(caller), static_cast<ull>(n));
    }
    host_log("gx-wrapper: other callers whose draws took a token:%s; YEBIS tokens completed from the wrapper's state %llu, state missing %llu",
             taken.empty() ? " none" : taken.c_str(), static_cast<ull>(g_yebis_state_built.load()), static_cast<ull>(g_yebis_state_missing.load()));
    host_log("gx-wrapper: draw tokens from the wrapper's state: gamma pass %llu, present %llu, refused (no programs) %llu, not written %llu; "
             "passes skipped: depth decompress %llu, colour eliminate %llu; clear-pass tokens %llu (of them depth targets without HTILE %llu; "
             "passes asked to clear nothing %llu, refused %llu); wrapper draws left on the packet path by caller:%s",
             static_cast<ull>(g_wd_taken[0].load()), static_cast<ull>(g_wd_taken[1].load()), static_cast<ull>(g_wd_refused.load()),
             static_cast<ull>(g_sf_unwritten.load()), static_cast<ull>(g_wd_skipped_depth.load()), static_cast<ull>(g_wd_skipped_colour.load()),
             static_cast<ull>(g_wd_clear_tokens.load()), static_cast<ull>(g_wd_clear_no_htile.load()), static_cast<ull>(g_wd_clear_nothing.load()),
             static_cast<ull>(g_wd_clear_refused.load()), others.empty() ? " none" : others.c_str());
}

// Per-shader compiles: the GX shader creators
// (0x2566d00 and 0x2566f20 build container types 1, 5 and 6; 0x25672d0 builds
// pixel shaders, type 2, through 0x2575280; 0x25673e0 compute shaders, type 4,
// through 0x2572db0) take the Sony shader container in rdx and its size in ecx;
// the renderer learns each shader when the game creates it.
void note_shader_created(int creator, std::uint64_t va, std::uint32_t size) {
    if (!va || size < 0x40 || size > (16u << 20)) return;
    std::vector<std::uint8_t> bytes(size);
    if (!safe_read(va, bytes.data(), size)) return;
    host_gpu_note_shader_created(creator, bytes.data(), bytes.size(), hle_video_flip_count());
}

// frame: r9 r8 rcx rdx rsi rdi, then the return address (see emit_stub).
// Returns nonzero when the host did the method's work; the stub then returns
// to the method's caller without running it.
GUEST_ABI std::uint64_t gx_hook(std::uint64_t id, const std::uint64_t* f) {
    if (id >= kKinds) {
        // A dead packet writer.
        const std::size_t i = static_cast<std::size_t>(id - kKinds);
        if (i >= kDeadWriterCount) return 0;
        g_dead_calls[i].fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
    if (id >= kCreateShader156 && id <= kCreateComputeShader) {
        note_shader_created(static_cast<int>(id - kCreateShader156), f[3], static_cast<std::uint32_t>(f[2]));
        gx_objects_watch_creator(const_cast<std::uint64_t*>(f));  // the object gets an id at the return
        return 0;
    }
    if (id == kUpdateSubresource) {
        note_update(f);
        return 0;
    }
    if (id == kCopyBuffers || id == kCopyBuffer) return native_copy(static_cast<Kind>(id), f);
    if (id == kYebisDraw) return native_yebis(f);
    if (id == kClearColour) {
        if ((g_native_passes & 2) && g_native == 2 && g_backend == 1) {
            g_wd_skipped_colour.fetch_add(1, std::memory_order_relaxed);
            clear_colour_cost(f);
            return 1;
        }
        return clear_colour_cost(f);
    }
    if (id == kFill) return native_fill(f);
    if (id == kClearRtv) return native_clear_rtv(f);
    if (id == kUploadTexture) return native_upload(f);
    if (id == kCopyTexture) return native_copy_texture(f);
    if (id >= kScaleformDraw && id <= kScaleformDrawAuto) return native_scaleform_draw(static_cast<Kind>(id), f);
    if (id >= kSetIndexBuffer && id <= kCsUdPtr) return wrapper_note(static_cast<Kind>(id), f);
    if (id == kDepthDecompress) {
        if (!(g_native_passes & 1) || g_native != 2 || g_backend != 1) return 0;
        g_wd_skipped_depth.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
    if (id == kDepthStencilClearPass) return native_clear_pass(f);
    const std::uint64_t ret = f[6];
    const std::uint64_t flip = hle_video_flip_count();
    if (id < kEmitAuto) {
        GxRec& r = t_last;
        r.kind = static_cast<Kind>(id);
        r.native_id = 0;
        r.tess = false;  // the record is reused; a stale flag would count every draw as one
        r.tid = tid();
        r.ret = ret;
        r.ctx = f[5];
        r.state = f[4];
        r.flip = flip;
        r.arg[0] = f[3];
        r.arg[1] = f[2];
        r.arg[2] = f[1];
        r.arg[3] = f[0];
        r.indirect_va = 0;
        if (id == kDrawIndirect || id == kDrawIndexedIndirect) {
            const std::uint64_t memory = f[3] ? rd64(f[3] + 0x20) : 0;
            r.indirect_va = memory ? memory + (f[2] & 0xffffffff) : 0;
        }
        if (is_dispatch(static_cast<int>(id))) {
            snapshot_compute(r, r.state);
        } else {
            GxCostTimer timer(kCostCall);
            snapshot(r, r.state);
            // A tessellated draw builds its inputs from GX and takes
            // a token, so it writes no packets - which leaves nothing but
            // indirect draws building any. On by default: the state and the
            // user data are checked field by field against the registers the
            // draw would have written (0 differing over 379,602 draws,
            // including the patch count), three soaks each way put main-loop
            // work inside the run-to-run spread with fewer stalls and hitches,
            // and no frame goes over the brightness mark.
            // BBHOST_GX_TESS_NATIVE=0 puts them back on the packet path.
            static const bool tess_native = [] {
                const char* e = std::getenv("BBHOST_GX_TESS_NATIVE");
                return !(e && e[0] == '0');
            }();
            r.in_ok = g_backend && build_inputs(r, r.in, tess_native);
            if (r.in_ok) r.in.program_fp = program_fingerprint(r.in);
            if (g_native && g_backend == 1) {
                if (r.kind > kDrawIndexedIndirect) g_native_skipped.add(1, kSkipKind);
                else if (!tess_native && (r.st[1].shader || r.st[2].shader)) g_native_skipped.add(1, kSkipTess);
                else if (!r.in_ok) g_native_skipped.add(1, kSkipInputs);
                if (r.st[3].shader && g_gs_object_draws.fetch_add(1, std::memory_order_relaxed) < 6) {
                    host_log("gx-native: %s from 0x%llx binds a geometry-shader object 0x%llx (VS 0x%llx, PS 0x%llx); drawn as VS/PS%s",
                             kKindName[r.kind], static_cast<ull>(bn(r.ret)), static_cast<ull>(r.st[3].shader), static_cast<ull>(r.st[0].shader),
                             static_cast<ull>(r.st[4].shader), r.indirect_va ? ", indirect" : "");
                }
            }
            // The tessellated draws are the packet path. With
            // BBHOST_GX_NATIVE=1 they get a token too - nothing is suppressed,
            // it only records what the draw's own packets then set.
            if (g_native == 1 && g_backend == 1 && !tess_native && r.kind <= kDrawIndexedInstanced && !r.st[3].shader &&
                (r.st[1].shader || r.st[2].shader)) {
                r.native_id = write_native_token(r.ctx);
                if (r.native_id) {
                    r.tess = true;
                    g_tess_shadowed.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (g_native && g_backend == 1 && r.in_ok && r.kind <= kDrawIndexedIndirect &&
                (tess_native || (!r.st[1].shader && !r.st[2].shader))) {
                if (g_native == 2) {
                    if (native_draw_call(r)) return 1;
                } else {
                    r.native_id = write_native_token(r.ctx);
                    t_call_records = r.native_id ? snapshot_records(r) : nullptr;
                }
            }
        }
        if (!in_window(flip)) return 0;
        std::lock_guard<std::mutex> lk(g_mu);
        ++g.calls[id];
        ++g.callers[{static_cast<int>(id), bn(ret)}];
        ++g.ctxs[{r.tid, r.ctx}];
        if (is_dispatch(static_cast<int>(id))) {
            ++g.cs_calls;
            if (r.st[5].shader) {
                ++g.cs_set;
                g.cs_samplers += r.st[5].samplers;
                g.cs_srvs += r.st[5].srvs;
                g.cs_cbs += r.st[5].cbs;
            }
            return 0;
        }
        if (id == kCommit) return 0;
        ++g.gx_state;
        ++g.topo[r.topo];
        ++g.ib_fmt[r.ib_fmt];
        ++g.vbs[r.vbs];
        ++g.rtvs[r.rtvs];
        ++g.stencil[r.stencil_ref];
        g.dsv += r.dsv != 0;
        g.blend += r.blend != 0;
        g.ds += r.ds != 0;
        g.pre += r.pre != 0;
        for (int k = 0; k < 6; ++k) {
            if (!r.st[k].shader) continue;
            ++g.stage_set[k];
            g.stage_samplers[k] += r.st[k].samplers;
            g.stage_srvs[k] += r.st[k].srvs;
            g.stage_cbs[k] += r.st[k].cbs;
        }
        if (g.op1.size() < 4096 || g.op1.count(r.op1)) ++g.op1[r.op1];
        return 0;
    }
    // Emitter: the first argument is the Gnm command buffer {+0x8 end, +0x10 write pointer}.
    GxCostTimer timer(kCostEmit);
    const std::uint64_t cb = f[5];
    const std::uint64_t cmd = rd64(cb + 0x10), end = rd64(cb + 0x8);
    const std::uint64_t caller = bn(ret);
    // A utility draw's token from the wrapper's state.
    int wd = wrapper_draw(static_cast<Kind>(id), f, caller);
    if (!wd) wd = wrapper_dispatch(static_cast<Kind>(id), f, caller);
    if (g_gnm_sources) note_source(cb, cmd, end, wd == 2 ? kSourceWrapper : caller);
    if (wd == 2) return 1;
    const bool from_gx = caller >= kGxImmediateLo && caller < kGxImmediateHi && t_last.kind < kEmitAuto;
    GxRecordsPtr records;
    if (from_gx && g_backend && sample_bindings()) {
        GxCostTimer records_timer(kCostRecords);
        records = snapshot_records(t_last);  // after the commit, before the lock
    }
    if (from_gx && t_call_records) {
        if (records && t_last.native_id) compare_call_records(*t_call_records, *records, t_last);
        t_call_records.reset();
    }
    std::lock_guard<std::mutex> lk(g_mu);
    if (in_window(flip)) {
        ++g.calls[id];
        if (from_gx) {
            ++g.emit_gx[id];
        } else {
            ++g.callers[{static_cast<int>(id), caller}];
        }
        if (end < cmd + 64) ++g.near_end;
    }
    if (g_by_packet.size() > (1u << 20)) g_by_packet.clear();
    EmitRec& e = g_by_packet[cmd];
    e.kind = static_cast<Kind>(id);
    e.ret = ret;
    e.flip = flip;
    e.gx = from_gx;
    if (from_gx) e.rec = t_last;
    e.records = std::move(records);
    g_packet_order.emplace_back(cmd, flip);
    while (g_packet_order.front().second + kPacketMaxAge < flip) {
        // Unless the same write pointer was emitted again since (a newer flip).
        const auto it = g_by_packet.find(g_packet_order.front().first);
        if (it != g_by_packet.end() && it->second.flip == g_packet_order.front().second) g_by_packet.erase(it);
        g_packet_order.pop_front();
    }
    return 0;
}

// The GX objects of the draw hle_gx_trace_draw last matched on this (CP)
// thread, for hle_gx_trace_objects.
thread_local GxDrawObjects t_objects;
thread_local bool t_objects_valid = false;
thread_local GxRecordsPtr t_records;  // keeps t_objects.records alive
// The token draw being drawn, in its slot (or taken from the
// overflow map), until hle_gx_native_release.
thread_local NativeSlot* t_native_slot = nullptr;
thread_local NativeDraw t_native_overflow;
thread_local const GxDrawObjects* t_native_objects = nullptr;

// Fills t_objects from a GX call record: context, state block, shaders and,
// for the immediate draws, the geometry the call gives (count, instances; for
// indexed draws the first index at INDEX_BASE, the index buffer's +0x20 plus
// the state's u32 +0x1c, 0x2abba30, + start * index size; a set byte +0x40
// renames the buffer through 0x2ad0ff0), the input layout's fetch shader and
// the vertex table.
void objects_from_call(const GxRec& r, GxRecordsPtr records) {
    build_objects(r, t_objects);
    t_records = std::move(records);
    t_objects.records = t_records.get();
    t_objects_valid = true;
}

// The emitter's write pointer is the packet, unless it wrote something first.
bool take_record(std::uint64_t packet_va, EmitRec& e) {
    std::lock_guard<std::mutex> lk(g_mu);
    for (std::uint64_t d = 0; d <= 64; d += 4) {
        const auto it = g_by_packet.find(packet_va - d);
        if (it == g_by_packet.end()) continue;
        e = std::move(it->second);
        g_by_packet.erase(it);
        return true;
    }
    return false;
}

// Which stage holds the bound program: look for its program-address register
// value (PGM_LO) in the structures each stage's shader object points at.
void scan_program(const GxRec& r, int lo, int hi, std::uint32_t a, char ca, std::uint32_t b, char cb,
                  std::vector<std::tuple<int, int, char>>& hits) {
    for (int k = lo; k <= hi; ++k) {
        std::uint64_t obj[48];
        if (!r.st[k].shader || !safe_read(r.st[k].shader, obj, sizeof obj)) continue;
        for (int i = 0; i < 48; ++i) {
            std::uint32_t dw[64];
            if (obj[i] < 0x10000 || !safe_read(obj[i], dw, sizeof dw)) continue;
            for (int j = 0; j < 64; ++j) {
                if (a >= 0x100 && dw[j] == a) hits.emplace_back(k, i, ca);
                if (b >= 0x100 && dw[j] == b) hits.emplace_back(k, i, cb);
            }
        }
    }
}

void print_summary();

template <typename K>
std::vector<std::pair<K, std::uint64_t>> top(const std::map<K, std::uint64_t>& m, std::size_t n) {
    std::vector<std::pair<K, std::uint64_t>> v(m.begin(), m.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    if (v.size() > n) v.resize(n);
    return v;
}

void appendf(std::string& s, const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    s += buf;
}

std::string hist(const std::map<std::uint32_t, std::uint64_t>& m, std::size_t n) {
    std::string s;
    for (const auto& [k, v] : top(m, n)) appendf(s, " 0x%x:%llu", k, static_cast<ull>(v));
    return s;
}

// Layout probes: where the command processor's register and user-data values
// for a draw sit inside the GX objects the draw was recorded with.
enum ObjKind { kObjRtv, kObjRtvRes, kObjDsv, kObjDsvRes, kObjBlend, kObjDs, kObjRaster, kObjLayout, kObjKinds };
const char* const kObjName[kObjKinds] = {"RTV0",  "RTV0 resource", "DSV",        "DSV resource",
                                         "blend", "depth-stencil", "rasterizer", "input layout"};
struct RegProbe {
    const char* name;
    int reg;  // context register index
};
const RegProbe kRegProbes[] = {
    {"CB_COLOR0_BASE", 0x318},        {"CB_COLOR0_PITCH", 0x319},        {"CB_COLOR0_SLICE", 0x31a},
    {"CB_COLOR0_VIEW", 0x31b},        {"CB_COLOR0_INFO", 0x31c},         {"CB_COLOR0_ATTRIB", 0x31d},
    {"CB_COLOR0_DIM", 0x31e},         {"DB_DEPTH_VIEW", 0x002},          {"DB_HTILE_DATA_BASE", 0x005},
    {"DB_Z_INFO", 0x010},             {"DB_STENCIL_INFO", 0x011},        {"DB_Z_READ_BASE", 0x012},
    {"DB_STENCIL_READ_BASE", 0x013},  {"DB_Z_WRITE_BASE", 0x014},        {"DB_STENCIL_WRITE_BASE", 0x015},
    {"DB_DEPTH_SIZE", 0x016},         {"CB_COLOR_CONTROL", 0x202},       {"CB_BLEND0_CONTROL", 0x1e0},
    {"CB_BLEND1_CONTROL", 0x1e1},     {"CB_TARGET_MASK", 0x08e},         {"CB_SHADER_MASK", 0x08f},
    {"DB_DEPTH_CONTROL", 0x200},      {"DB_STENCILCONTROL", 0x10b},      {"DB_STENCILREFMASK", 0x10c},
    {"DB_STENCILREFMASK_BF", 0x10d},  {"DB_SHADER_CONTROL", 0x203},      {"PA_CL_CLIP_CNTL", 0x204},
    {"PA_SU_SC_MODE_CNTL", 0x205},    {"PA_SC_MODE_CNTL_0", 0x292},      {"PA_SC_MODE_CNTL_1", 0x293},
    {"PA_SU_POLY_OFFSET_DB_FMT_CNTL", 0x2de}, {"PA_SU_POLY_OFFSET_CLAMP", 0x2df},
    {"PA_SU_POLY_OFFSET_FRONT_SCALE", 0x2e0}, {"PA_SU_VTX_CNTL", 0x2f9},
};
constexpr int kRegProbeCount = static_cast<int>(sizeof(kRegProbes) / sizeof(kRegProbes[0]));
std::atomic<std::int64_t> g_probe_budget{400};

struct ProbeStats {
    std::uint64_t draws = 0;
    std::uint64_t samples[64] = {};                                // nonzero values per register probe
    std::map<std::tuple<int, int, int>, std::uint64_t> reg_hits;   // (probe, object, byte offset)
    std::uint64_t vp_ok = 0, vp_bad = 0, sc_generic = 0, sc_vport = 0, sc_neither = 0;
    std::vector<std::string> vp_samples, sc_samples;
    std::uint64_t layout_tested = 0;
    std::map<int, std::uint64_t> layout_user;                      // VS user-data index = input layout +0x10
    std::uint64_t srv_tested = 0, smp_tested = 0, cb_tested = 0, cb_size_ok = 0;
    std::map<std::pair<int, char>, std::uint64_t> srv_user;        // (PS user-data index, table / direct)
    std::map<std::tuple<int, int, char>, std::uint64_t> smp_user;  // (object offset, PS user-data index, t/d)
    std::map<std::pair<int, int>, std::uint64_t> cb_user;          // (object offset, PS user-data index)
    std::vector<std::uint64_t> ps_shaders;                         // a few PS shader objects to decode
};
ProbeStats g_probe;  // under g_mu

float ctx_float(const std::uint32_t* ctx, int i) {
    float v;
    std::memcpy(&v, &ctx[i], 4);
    return v;
}
std::uint64_t user_pair(const std::uint32_t* user, int k) {
    return user[k] | static_cast<std::uint64_t>(user[k + 1]) << 32;
}
bool nonzero(const std::uint8_t* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (b[i]) return true;
    }
    return false;
}

void probe_draw(const GxRec& r, const std::uint32_t* sh, const std::uint32_t* ctx) {
    std::uint64_t ptrs[kObjKinds] = {r.rtv0, 0, r.dsv, 0, r.blend, r.ds, r.raster, r.op1};
    if (r.rtv0) safe_read(r.rtv0 + 8, &ptrs[kObjRtvRes], 8);
    if (r.dsv) safe_read(r.dsv + 8, &ptrs[kObjDsvRes], 8);
    std::uint32_t mem[kObjKinds][96];
    bool have[kObjKinds] = {};
    for (int k = 0; k < kObjKinds; ++k) have[k] = ptrs[k] >= 0x10000 && safe_read(ptrs[k], mem[k], sizeof(mem[k]));
    std::vector<std::tuple<int, int, int>> reg_hits;
    std::vector<int> sampled;
    for (int q = 0; q < kRegProbeCount; ++q) {
        const std::uint32_t v = ctx[kRegProbes[q].reg];
        if (v < 2) continue;
        sampled.push_back(q);
        for (int k = 0; k < kObjKinds; ++k) {
            if (!have[k]) continue;
            for (int j = 0; j < 96; ++j) {
                if (mem[k][j] == v) reg_hits.emplace_back(q, k, j * 4);
            }
        }
    }

    // Viewport 0 (D3D x y w h minz maxz) against PA_CL_VPORT scale / offset.
    const float xs = ctx_float(ctx, 0x10f), xo = ctx_float(ctx, 0x110), ys = ctx_float(ctx, 0x111),
                yo = ctx_float(ctx, 0x112), zs = ctx_float(ctx, 0x113), zo = ctx_float(ctx, 0x114);
    const auto close = [](float a, float b) { return std::fabs(a - b) < 0.01f; };
    const bool vp_ok = close(xs, r.vp[2] * 0.5f) && close(xo, r.vp[0] + r.vp[2] * 0.5f) &&
                       close(std::fabs(ys), r.vp[3] * 0.5f) && close(yo, r.vp[1] + r.vp[3] * 0.5f) &&
                       close(zs, r.vp[5] - r.vp[4]) && close(zo, r.vp[4]);
    // Scissor 0 (l t r b) against the generic and viewport-0 scissor registers.
    const auto rect = [ctx](int tl, int br) {
        return std::array<int, 4>{static_cast<int>(ctx[tl] & 0x7fff), static_cast<int>((ctx[tl] >> 16) & 0x7fff),
                                  static_cast<int>(ctx[br] & 0x7fff), static_cast<int>((ctx[br] >> 16) & 0x7fff)};
    };
    const std::array<int, 4> gx_sc{r.sc[0], r.sc[1], r.sc[2], r.sc[3]};
    const bool sc_generic = rect(0x90, 0x91) == gx_sc, sc_vport = rect(0x94, 0x95) == gx_sc;

    // Input layout +0x10 in the VS user data.
    const std::uint32_t* vs_user = &sh[0x4c];
    const std::uint32_t* ps_user = &sh[0x0c];
    std::uint64_t fetch = 0;
    std::vector<int> layout_k;
    const bool layout_tested = r.op1 && safe_read(r.op1 + 0x10, &fetch, 8) && fetch;
    if (layout_tested) {
        for (int k = 0; k < 15; ++k) {
            if (user_pair(vs_user, k) == fetch) layout_k.push_back(k);
        }
    }

    // PS SRVs: view +0x10 (32 bytes) as a table entry behind PS user data, or directly in it.
    std::vector<std::pair<int, char>> srv_hits;
    int srv_tested = 0;
    for (int i = 0; i < r.nsrv; ++i) {
        std::uint8_t b[32];
        if (!safe_read(r.srv[i].obj + 0x10, b, sizeof(b)) || !nonzero(b, sizeof(b))) continue;
        ++srv_tested;
        for (int k = 0; k < 15; ++k) {
            if (k < 13 && std::memcmp(&ps_user[k], b, 16) == 0) srv_hits.emplace_back(k, 'd');
            const std::uint64_t t = user_pair(ps_user, k);
            std::uint8_t e[32];
            if (t >= 0x10000 && safe_read(t + 32ull * r.srv[i].slot, e, sizeof(e)) && std::memcmp(e, b, 32) == 0) {
                srv_hits.emplace_back(k, 't');
            }
        }
    }
    // PS samplers: 32 bytes at some object offset, as a table entry or directly.
    std::vector<std::tuple<int, int, char>> smp_hits;
    int smp_tested = 0;
    for (int i = 0; i < r.nsmp; ++i) {
        std::uint8_t o[0x48];
        if (!safe_read(r.smp[i].obj, o, sizeof(o))) continue;
        ++smp_tested;
        for (int k = 0; k < 15; ++k) {
            const std::uint64_t t = user_pair(ps_user, k);
            std::uint8_t e[32];
            const bool te = t >= 0x10000 && safe_read(t + 32ull * r.smp[i].slot, e, sizeof(e));
            for (int off = 0; off + 32 <= 0x48; off += 8) {
                if (!nonzero(o + off, 32)) continue;
                if (te && std::memcmp(o + off, e, 32) == 0) smp_hits.emplace_back(off, k, 't');
                if (k < 13 && std::memcmp(o + off, &ps_user[k], 16) == 0) smp_hits.emplace_back(off, k, 'd');
            }
        }
    }
    // PS constant buffers: an object qword equal to a PS user-data address.
    std::vector<std::pair<int, int>> cb_hits;
    int cb_tested = 0, cb_size_ok = 0;
    for (int i = 0; i < r.ncb; ++i) {
        std::uint64_t q[16];
        if (!safe_read(r.cb[i].obj, q, sizeof(q))) continue;
        ++cb_tested;
        const std::uint32_t size = static_cast<std::uint32_t>(q[5] >> 32);  // +0x2c
        bool size_ok = false;
        for (int k = 0; k < 15; ++k) {
            const std::uint64_t u = user_pair(ps_user, k);
            if (u < 0x10000) continue;
            for (int j = 0; j < 16; ++j) {
                if (q[j] != u) continue;
                cb_hits.emplace_back(j * 8, k);
                if (k < 14 && ps_user[k + 2] == size) size_ok = true;
            }
        }
        cb_size_ok += size_ok;
    }

    std::lock_guard<std::mutex> lk(g_mu);
    ProbeStats& p = g_probe;
    ++p.draws;
    for (int q : sampled) ++p.samples[q];
    for (const auto& h : reg_hits) ++p.reg_hits[h];
    (vp_ok ? p.vp_ok : p.vp_bad)++;
    if (!vp_ok && p.vp_samples.size() < 4) {
        std::string s;
        appendf(s, "GX %.1f %.1f %.1f %.1f %.3f %.3f; PA_CL_VPORT scale/offset x %.1f %.1f y %.1f %.1f z %.3f %.3f", r.vp[0],
                r.vp[1], r.vp[2], r.vp[3], r.vp[4], r.vp[5], xs, xo, ys, yo, zs, zo);
        p.vp_samples.push_back(s);
    }
    p.sc_generic += sc_generic;
    p.sc_vport += sc_vport;
    if (!sc_generic && !sc_vport) {
        ++p.sc_neither;
        if (p.sc_samples.size() < 4) {
            std::string s;
            const auto g1 = rect(0x90, 0x91), g2 = rect(0x94, 0x95);
            appendf(s, "GX %d %d %d %d; generic %d %d %d %d; vport0 %d %d %d %d", r.sc[0], r.sc[1], r.sc[2], r.sc[3], g1[0],
                    g1[1], g1[2], g1[3], g2[0], g2[1], g2[2], g2[3]);
            p.sc_samples.push_back(s);
        }
    }
    p.layout_tested += layout_tested;
    for (int k : layout_k) ++p.layout_user[k];
    p.srv_tested += static_cast<std::uint64_t>(srv_tested);
    for (const auto& h : srv_hits) ++p.srv_user[h];
    p.smp_tested += static_cast<std::uint64_t>(smp_tested);
    for (const auto& h : smp_hits) ++p.smp_user[h];
    p.cb_tested += static_cast<std::uint64_t>(cb_tested);
    p.cb_size_ok += static_cast<std::uint64_t>(cb_size_ok);
    for (const auto& h : cb_hits) ++p.cb_user[h];
    if (r.st[4].shader && p.ps_shaders.size() < 4 &&
        std::find(p.ps_shaders.begin(), p.ps_shaders.end(), r.st[4].shader) == p.ps_shaders.end()) {
        p.ps_shaders.push_back(r.st[4].shader);
    }
}

void print_probes(std::string& s) {
    const ProbeStats& p = g_probe;
    host_log("gx: layout probes over %llu GX draws (command-processor values found inside the draw's GX objects)",
             static_cast<ull>(p.draws));
    for (int q = 0; q < kRegProbeCount; ++q) {
        std::map<std::pair<int, int>, std::uint64_t> by;
        for (const auto& [key, v] : p.reg_hits) {
            if (std::get<0>(key) == q) by[{std::get<1>(key), std::get<2>(key)}] += v;
        }
        s.clear();
        for (const auto& [ko, v] : top(by, 3)) appendf(s, " %s+0x%x:%llu", kObjName[ko.first], ko.second, static_cast<ull>(v));
        host_log("gx:   %s (nonzero in %llu):%s", kRegProbes[q].name, static_cast<ull>(p.samples[q]), s.c_str());
    }
    host_log("gx:   viewport 0 -> PA_CL_VPORT: %llu ok / %llu bad", static_cast<ull>(p.vp_ok), static_cast<ull>(p.vp_bad));
    for (const auto& v : p.vp_samples) host_log("gx:     %s", v.c_str());
    host_log("gx:   scissor 0 = generic scissor in %llu, = viewport-0 scissor in %llu, neither in %llu",
             static_cast<ull>(p.sc_generic), static_cast<ull>(p.sc_vport), static_cast<ull>(p.sc_neither));
    for (const auto& v : p.sc_samples) host_log("gx:     %s", v.c_str());
    s.clear();
    for (const auto& [k, v] : top(p.layout_user, 4)) appendf(s, " %d:%llu", k, static_cast<ull>(v));
    host_log("gx:   input layout +0x10 found at VS user data (of %llu):%s", static_cast<ull>(p.layout_tested), s.c_str());
    s.clear();
    for (const auto& [kt, v] : top(p.srv_user, 6)) {
        appendf(s, " %s %d:%llu", kt.second == 't' ? "table at" : "direct at", kt.first, static_cast<ull>(v));
    }
    host_log("gx:   PS texture view +0x10, 32 bytes (%llu tested): PS user data%s", static_cast<ull>(p.srv_tested), s.c_str());
    s.clear();
    for (const auto& [h, v] : top(p.smp_user, 6)) {
        appendf(s, " +0x%x %s %d:%llu", std::get<0>(h), std::get<2>(h) == 't' ? "table at" : "direct at", std::get<1>(h),
                static_cast<ull>(v));
    }
    host_log("gx:   PS sampler object (%llu tested): PS user data%s", static_cast<ull>(p.smp_tested), s.c_str());
    s.clear();
    for (const auto& [h, v] : top(p.cb_user, 6)) appendf(s, " +0x%x = user data %d:%llu", h.first, h.second, static_cast<ull>(v));
    host_log("gx:   PS constant buffer object (%llu tested, size +0x2c matched in %llu):%s", static_cast<ull>(p.cb_tested),
             static_cast<ull>(p.cb_size_ok), s.c_str());
    // The commit walks a shader object's user-data descriptors: pointer +0xe0,
    // count +0xe8, 4-byte entries {type, user-data slot, resource slot, -}.
    for (const std::uint64_t obj : p.ps_shaders) {
        std::uint64_t head[2] = {};
        s.clear();
        if (safe_read(obj + 0xe0, head, sizeof(head))) {
            const std::uint32_t n = std::min<std::uint32_t>(static_cast<std::uint32_t>(head[1]), 16);
            std::uint32_t e[16] = {};
            if (head[0] >= 0x10000 && safe_read(head[0], e, 4u * n)) {
                for (std::uint32_t i = 0; i < n; ++i) {
                    appendf(s, " [%u] type %u ud %u slot %u", i, e[i] & 0xff, (e[i] >> 8) & 0xff, (e[i] >> 16) & 0xff);
                }
            }
            host_log("gx:   PS shader 0x%llx: %u user-data descriptors:%s", static_cast<ull>(obj),
                     static_cast<std::uint32_t>(head[1]), s.c_str());
        }
    }
}

void print_summary() {
    if (g_printed.exchange(true)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    const double flips = static_cast<double>(g_last - g_first);
    const auto per = [flips](std::uint64_t v) { return static_cast<double>(v) / flips; };
    const auto pct = [](std::uint64_t v, std::uint64_t of) {
        return of ? 100.0 * static_cast<double>(v) / static_cast<double>(of) : 0.0;
    };
    const auto avg = [](std::uint64_t v, std::uint64_t of) {
        return of ? static_cast<double>(v) / static_cast<double>(of) : 0.0;
    };
    host_log("gx: coverage for flips %llu-%llu (calls counted by flip at the call, packets by flip at execution)",
             static_cast<ull>(g_first), static_cast<ull>(g_last));
    std::string s;
    for (int k = kDraw; k < kEmitAuto; ++k) appendf(s, " %s=%.1f", kKindName[k], per(g.calls[k]));
    host_log("gx: GX immediate calls per flip:%s", s.c_str());
    for (int k = kDraw; k < kKinds; ++k) {
        std::map<std::uint64_t, std::uint64_t> by;
        for (const auto& [key, v] : g.callers) {
            if (key.first == k) by[key.second] += v;
        }
        if (by.empty()) continue;
        s.clear();
        for (const auto& [ret, v] : top(by, 12)) {
            appendf(s, " 0x%llx%s:%llu", static_cast<ull>(ret), ret >= kReplayLo && ret < kReplayHi ? "(replay)" : "",
                    static_cast<ull>(v));
        }
        host_log("gx:   %s callers%s, %zu distinct return addresses:%s", kKindName[k], k >= kEmitAuto ? " outside GX" : "",
                 by.size(), s.c_str());
    }
    s.clear();
    for (int k = kEmitAuto; k < kKinds; ++k) appendf(s, " %s=%.1f (GX %.1f)", kKindName[k], per(g.calls[k]), per(g.emit_gx[k]));
    host_log("gx: emitter calls per flip:%s; %llu with under 64 bytes left in the chunk", s.c_str(),
             static_cast<ull>(g.near_end));

    for (int c = 0; c < 2; ++c) {
        const CpStats& p = g.cp[c];
        const char* what = c ? "dispatch" : "draw";
        host_log("gx: executed %s packets per flip: %.1f = through GX %.1f + other emitter callers %.1f + no emitter record %.1f",
                 what, per(p.total), per(p.gx), per(p.other), per(p.none));
        s.clear();
        for (const auto& [op, v] : p.ops) {
            const auto u = p.unmatched.find(op);
            appendf(s, " 0x%02x:%llu (no record %llu)", op, static_cast<ull>(v),
                    static_cast<ull>(u == p.unmatched.end() ? 0 : u->second));
        }
        host_log("gx:   %s packets by PM4 opcode:%s", what, s.c_str());
        s.clear();
        for (const auto& [ret, v] : top(p.other_callers, 12)) appendf(s, " 0x%llx:%llu", static_cast<ull>(ret), static_cast<ull>(v));
        host_log("gx:   %s other emitter callers:%s", what, s.c_str());
        s.clear();
        for (const auto& [prog, v] : top(p.unmatched_prog, 12)) appendf(s, " %08x:%llu", prog, static_cast<ull>(v));
        host_log("gx:   %s packets without an emitter record, by program PGM_LO:%s", what, s.c_str());
    }
    s.clear();
    for (int k = kDraw; k < kEmitAuto; ++k) {
        if (k != kCommit) appendf(s, " %s=%llu", kKindName[k], static_cast<ull>(g.cp_gx_kind[k]));
    }
    host_log("gx:   executed through GX, by method:%s", s.c_str());

    host_log("gx: checks: draw count %llu ok / %llu bad, instances %llu ok / %llu bad; dispatch x %llu ok / %llu bad",
             static_cast<ull>(g.cp[0].arg_ok), static_cast<ull>(g.cp[0].arg_bad), static_cast<ull>(g.inst_ok),
             static_cast<ull>(g.inst_bad), static_cast<ull>(g.cp[1].arg_ok), static_cast<ull>(g.cp[1].arg_bad));
    for (const auto& b : g.bad_samples) host_log("gx:   mismatch %s", b.c_str());
    s.clear();
    for (const auto& [p, v] : top(g.fmt_pairs, 8)) appendf(s, " 0x%x->%u:%llu", p.first, p.second, static_cast<ull>(v));
    host_log("gx:   index format, state +0x18 -> INDEX_TYPE:%s", s.c_str());
    s.clear();
    for (const auto& [p, v] : top(g.topo_pairs, 10)) appendf(s, " 0x%x->%u:%llu", p.first, p.second, static_cast<ull>(v));
    host_log("gx:   topology, state +0x8 -> VGT_PRIMITIVE_TYPE:%s", s.c_str());
    s.clear();
    for (const auto& [slot, v] : g.ib_slot) {
        if (slot < 0) {
            appendf(s, " none:%llu", static_cast<ull>(v));
        } else if (slot >= 100) {
            appendf(s, " +0x%x plus offset:%llu", 8 * (slot - 100), static_cast<ull>(v));
        } else {
            appendf(s, " +0x%x:%llu", 8 * slot, static_cast<ull>(v));
        }
    }
    host_log("gx:   index buffer object field equal to INDEX_BASE:%s", s.c_str());

    host_log("gx: state over %llu GX draw calls: vertex buffers%s; render targets%s; DSV %.0f%%, blend %.0f%%, depth-stencil %.0f%%, +0x590 %.0f%%",
             static_cast<ull>(g.gx_state), hist(g.vbs, 8).c_str(), hist(g.rtvs, 9).c_str(), pct(g.dsv, g.gx_state),
             pct(g.blend, g.gx_state), pct(g.ds, g.gx_state), pct(g.pre, g.gx_state));
    host_log("gx:   topology +0x8:%s; index format +0x18:%s; stencil ref +0x584:%s", hist(g.topo, 8).c_str(),
             hist(g.ib_fmt, 6).c_str(), hist(g.stencil, 6).c_str());
    s.clear();
    for (int k = 0; k < 6; ++k) {
        appendf(s, " [%s +0x%x shader %.0f%%, per set: samplers %.1f srvs %.1f cbs %.1f]", kStageName[k], 0x598 + 0x4f8 * k,
                pct(g.stage_set[k], g.gx_state), avg(g.stage_samplers[k], g.stage_set[k]),
                avg(g.stage_srvs[k], g.stage_set[k]), avg(g.stage_cbs[k], g.stage_set[k]));
    }
    host_log("gx:   stage groups:%s", s.c_str());
    host_log("gx:   GX dispatches: CS shader set in %llu of %llu; per set: samplers %.1f srvs %.1f cbs %.1f",
             static_cast<ull>(g.cs_set), static_cast<ull>(g.cs_calls), avg(g.cs_samplers, g.cs_set),
             avg(g.cs_srvs, g.cs_set), avg(g.cs_cbs, g.cs_set));
    s.clear();
    for (const auto& [h, v] : top(g.stage_hits, 12)) {
        const char c = std::get<2>(h);
        appendf(s, " %s register in %s shader via +0x%x:%llu", c == 'V' ? "VS" : c == 'P' ? "PS" : "CS",
                kStageName[std::get<0>(h)], 8 * std::get<1>(h), static_cast<ull>(v));
    }
    host_log("gx:   bound program address found (%llu draws, %llu dispatches scanned):%s", static_cast<ull>(g.cp[0].scans),
             static_cast<ull>(g.cp[1].scans), s.c_str());
    host_log("gx:   state +0x0 pointer: %zu distinct%s", g.op1.size(), g.op1.size() >= 4096 ? " (capped)" : "");
    for (const auto& [ptr, v] : top(g.op1, 4)) {
        std::uint64_t q[6] = {};
        const bool ok = ptr && safe_read(ptr, q, sizeof(q));
        const bool in_image = ok && q[0] >= g_slide && q[0] < g_slide + g_image_size;
        host_log("gx:     0x%llx x%llu:%s %016llx%s %016llx %016llx %016llx %016llx %016llx", static_cast<ull>(ptr),
                 static_cast<ull>(v), ok ? "" : " (unreadable)", static_cast<ull>(q[0]), in_image ? " (in image)" : "",
                 static_cast<ull>(q[1]), static_cast<ull>(q[2]), static_cast<ull>(q[3]), static_cast<ull>(q[4]),
                 static_cast<ull>(q[5]));
    }
    s.clear();
    for (const auto& [key, v] : top(g.ctxs, 12)) {
        appendf(s, " tid %d ctx 0x%llx:%llu", key.first, static_cast<ull>(key.second), static_cast<ull>(v));
    }
    host_log("gx: contexts:%s", s.c_str());
    print_probes(s);
}

// The stub: core/thunk.h's thunk_emit_prologue_stub, over this hook's
// displaced prologue.
std::uint8_t* emit_stub(std::uint8_t* c, std::uint64_t id, void* host, std::uint64_t resume, const Hook& h) {
    return thunk_emit_prologue_stub(c, id, host, resume, h.bytes, h.n);
}

}  // namespace

void hle_gx_trace_arm(GuestMemory* mem) {
    const char* e = std::getenv("BBHOST_GX_TRACE");
    const char* b = std::getenv("BBHOST_GX_BACKEND");
    g_trace = e && *e && !(e[0] == '0' && !e[1]);
    // GX-native draws are the default: the backend renders from the GX
    // objects, and immediate draws in its subset run from host-draw tokens
    // without their Gnm packets. BBHOST_GX_BACKEND=0 or BBHOST_GX_NATIVE=0 goes
    // back to the packet path.
    g_backend = b ? std::atoi(b) : 1;
    const char* n = std::getenv("BBHOST_GX_NATIVE");
    g_native = n ? std::atoi(n) : g_backend == 1 ? 2 : 0;
    const char* y = std::getenv("BBHOST_GX_NATIVE_YEBIS");
    g_native_yebis = y ? std::atoi(y) : g_backend == 1 ? 5 : 0;
    if (!g_trace && !g_backend) return;
    if (const char* comma = g_trace ? std::strchr(e, ',') : nullptr) {
        g_first = std::strtoull(e, nullptr, 10);
        g_last = std::strtoull(comma + 1, nullptr, 10);
    }
    if (g_last <= g_first) g_last = g_first + 300;
    if (const char* c = std::getenv("BBHOST_CP_COPY"); c && c[0] == '1') {
        host_log("gx: BBHOST_CP_COPY=1 walks snapshots, so executed packets will not match their emitters");
    }
    g_slide = mem->slide;
    g_image_size = mem->size;
    void* host = thunk_wrap(reinterpret_cast<void*>(&gx_hook));
    void* page = host_page_alloc(0x8000, true);  // a stub of up to ~230 bytes per hook (with the xmm save)
    if (!page) {
        host_log("gx: stub page allocation failed; trace off");
        return;
    }
    auto* c = static_cast<std::uint8_t*>(page);
    int hooked = 0;
    for (std::uint64_t id = 0; id < kKinds; ++id) {
        if (id == kUpdateSubresource && !g_trace && !g_gx_cost_enabled && !g_watch_update && !g_upload_census) continue;
        const Hook& h = kHooks[id];
        const std::uint64_t va = g_slide + (h.va - kPreferredGuestSlide);
        if (va < mem->slide || va + h.n > mem->slide + mem->size) continue;
        auto* p = static_cast<std::uint8_t*>(guest_ptr(*mem, va));
        if (std::memcmp(p, h.bytes, h.n) != 0) {
            host_log("gx: %s at 0x%llx does not start with the expected prologue; not hooked", kKindName[id],
                     static_cast<ull>(h.va));
            continue;
        }
        const std::uint8_t* stub = c;
        c = emit_stub(c, id, host, va + h.n, h);
        const std::uint64_t lo = va & ~0xfffull, hi = (va + h.n + 0xfff) & ~0xfffull;
        if (!guest_protect_rwx(mem, lo, hi - lo)) {
            host_log("gx: cannot unprotect 0x%llx", static_cast<ull>(h.va));
            continue;
        }
        const std::uint64_t dest = reinterpret_cast<std::uint64_t>(stub);
        p[0] = 0xff;
        p[1] = 0x25;  // jmp [rip+0]
        std::memset(p + 2, 0, 4);
        std::memcpy(p + 6, &dest, 8);
        std::memset(p + 14, 0xcc, h.n - 14u);
        guest_protect_rx(mem, lo, hi - lo);
        ++hooked;
    }
    int dead = 0;
    for (std::size_t i = 0; g_dead_writers && i < kDeadWriterCount; ++i) {
        const DeadWriter& w = kDeadWriters[i];
        Hook h{};
        h.va = w.va;
        h.n = w.n;
        std::memcpy(h.bytes, w.bytes, sizeof(h.bytes));
        const std::uint64_t va = g_slide + (h.va - kPreferredGuestSlide);
        if (va < mem->slide || va + h.n > mem->slide + mem->size) continue;
        auto* p = static_cast<std::uint8_t*>(guest_ptr(*mem, va));
        if (std::memcmp(p, h.bytes, h.n) != 0) {
            host_log("gx: the writer of %s at 0x%llx does not start with the expected prologue; not hooked", w.what,
                     static_cast<ull>(w.va));
            continue;
        }
        const std::uint8_t* stub = c;
        c = emit_stub(c, kKinds + i, host, va + h.n, h);
        const std::uint64_t lo = va & ~0xfffull, hi = (va + h.n + 0xfff) & ~0xfffull;
        if (!guest_protect_rwx(mem, lo, hi - lo)) continue;
        const std::uint64_t dest = reinterpret_cast<std::uint64_t>(stub);
        p[0] = 0xff;
        p[1] = 0x25;
        std::memset(p + 2, 0, 4);
        std::memcpy(p + 6, &dest, 8);
        std::memset(p + 14, 0xcc, h.n - 14u);
        guest_protect_rx(mem, lo, hi - lo);
        ++dead;
    }
    if (g_dead_writers) host_log("gx: %d of %zu dead packet writers hooked; their packets are not written", dead, kDeadWriterCount);
    g_active = hooked > 0;
    host_log("gx: %d of %d hooks armed; trace %s (flips %llu-%llu), backend mode %d", hooked, static_cast<int>(kKinds),
             g_trace ? "on" : "off", static_cast<ull>(g_first), static_cast<ull>(g_last), g_backend);
}

// BBHOST_GX_NATIVE=1: the register fields render mode takes from the register
// file, as the CP had them at each token (CP thread).
struct NativeShadow {
    std::uint32_t user[2][16];
    std::uint32_t ps_input_cntl[32];
    std::uint32_t render_control, depth_clear, stencil_clear, generic_scissor[2], vport_scissor[2], sc_mode_cntl_0;
};
thread_local std::unordered_map<std::uint64_t, NativeShadow> t_native_shadow;



// Register-file fields a token draw takes (below): how often they hold
// something a draw without them would not have. DB_RENDER_CONTROL set, with
// clear bits, and a generic scissor narrower than the screen scissor GX builds.
std::atomic<std::uint64_t> g_token_rc_set{0}, g_token_rc_clear{0}, g_token_generic_narrows{0};
std::atomic<std::uint64_t> g_token_depth_clear_set{0}, g_token_stencil_clear_set{0};  // clear values not 0
// BBHOST_GX_TOKEN_USER_DATA=register (checks): token draws take the register
// file's user data in slots no descriptor names, as they did before, and the
// renderer checks whether any shader variant reads it. By default those slots
// are 0: no variant read a nonzero one over 2.74M token stages (native-regs-05).
const bool g_token_user_data_register = [] {
    const char* e = std::getenv("BBHOST_GX_TOKEN_USER_DATA");
    return e && std::strcmp(e, "register") == 0;
}();

// BBHOST_GNM_SOURCES=1: the command-buffer pieces since the last call by
// source, largest first; empty when the switch is off.
std::string hle_gx_source_window() {
    if (!g_gnm_sources) return {};
    std::vector<std::pair<std::uint64_t, SourceCount>> rows;
    std::uint64_t jumps = 0, largest = 0;
    {
        std::lock_guard<std::mutex> lk(g_sources_mu);
        rows.assign(g_sources.begin(), g_sources.end());
        g_sources.clear();
        jumps = g_source_jumps;
        g_source_jumps = 0;
        largest = g_source_largest;
        g_source_largest = 0;
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.dwords > b.second.dwords; });
    std::uint64_t pieces = 0, dwords = 0;
    for (const auto& [source, c] : rows) {
        pieces += c.pieces;
        dwords += c.dwords;
    }
    std::string out;
    appendf(out, "pieces %llu, dwords %llu, pieces across chunks %llu, largest piece %llu dwords, sources %zu;",
            static_cast<ull>(pieces), static_cast<ull>(dwords), static_cast<ull>(jumps), static_cast<ull>(largest), rows.size());
    for (std::size_t i = 0; i < rows.size() && i < 32; ++i) {
        const auto& [source, c] = rows[i];
        if (source == kSourceNative) {
            appendf(out, " native draw");
        } else if (source == kSourceCopy) {
            appendf(out, " copy token");
        } else if (source == kSourceScaleform) {
            appendf(out, " Scaleform token");
        } else if (source == kSourceWrapper) {
            appendf(out, " wrapper token");
        } else {
            appendf(out, " 0x%llx", static_cast<ull>(source));
        }
        appendf(out, " %llu pieces %llu dwords;", static_cast<ull>(c.pieces), static_cast<ull>(c.dwords));
    }
    return out;
}

// A copy token (BBHOST_GX_NATIVE_COPIES): the call's copies, as buffer copies
// at the token's place in the command stream.
thread_local GpuDrawInputs t_yebis_built{};
thread_local bool t_yebis_have = false;

// With the flip on, the CP draws at the token: the method wrote no draw packet.
const GpuDrawInputs* hle_gx_yebis_token_draw(std::uint64_t id, GxYebisDraw* out) {
    if (g_native_yebis != 5) return nullptr;
    std::lock_guard<std::mutex> lk(g_yebis_mu);
    const auto it = g_yebis_draws.find(id);
    const auto bi = g_yebis_inputs.find(id);
    if (it == g_yebis_draws.end() || bi == g_yebis_inputs.end()) {
        g_yebis_unmatched.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    *out = it->second;
    t_yebis_built = bi->second;
    g_yebis_draws.erase(it);
    g_yebis_inputs.erase(bi);
    // The windows the native binder stashed at the call go to their ring slots
    // here, in the command processor's own order - the place the constant
    // engine's dump held, and the moment before the renderer reads them.
    const auto wi = g_yebis_work.find(id);
    if (wi != g_yebis_work.end()) {
        yebis_bind::run(wi->second);
        g_yebis_work.erase(wi);
    }
    g_yebis_walked.fetch_add(1, std::memory_order_relaxed);
    return &t_yebis_built;
}

bool hle_gx_yebis_token(std::uint64_t id, GxYebisDraw* out) {
    g_yebis_walked.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_yebis_mu);
    const auto it = g_yebis_draws.find(id);
    if (it == g_yebis_draws.end()) {
        g_yebis_unmatched.fetch_add(1, std::memory_order_relaxed);
        t_yebis_have = false;
        return false;
    }
    *out = it->second;
    g_yebis_draws.erase(it);
    const auto bi = g_yebis_inputs.find(id);
    t_yebis_have = bi != g_yebis_inputs.end();
    if (t_yebis_have) {
        t_yebis_built = bi->second;
        g_yebis_inputs.erase(bi);
    }
    return true;
}

// The CP reached the DRAW_INDEX_2 that a token said was coming. By then the
// commit's packets have been walked, so the register file holds what YEBIS
// meant this draw to use and the inputs built from its own state can be checked
// against it, field by field, with nothing suppressed.
void hle_gx_yebis_check(const std::uint32_t* sh) {
    if (!t_yebis_have || !sh) return;
    t_yebis_have = false;
    g_yebis_checked.fetch_add(1, std::memory_order_relaxed);
    const GpuDrawInputs& b = t_yebis_built;
    const bool bad[6] = {
        std::memcmp(b.vs_pgm, &sh[0x48], sizeof(b.vs_pgm)) != 0,
        std::memcmp(b.ps_pgm, &sh[0x08], sizeof(b.ps_pgm)) != 0,
        std::memcmp(b.vs_user, &sh[0x4c], sizeof(b.vs_user)) != 0,
        std::memcmp(b.ps_user, &sh[0x0c], sizeof(b.ps_user)) != 0,
        false,
        false,
    };
    for (int k = 0; k < 6; ++k) {
        if (bad[k]) g_yebis_field_bad[k].fetch_add(1, std::memory_order_relaxed);
    }
    for (int i = 0; i < 16; ++i) {
        if (b.vs_user[i] != sh[0x4c + i]) g_yebis_slot_bad[0][i].fetch_add(1, std::memory_order_relaxed);
        if (b.ps_user[i] != sh[0x0c + i]) g_yebis_slot_bad[1][i].fetch_add(1, std::memory_order_relaxed);
    }
    static std::atomic<int> logs{0};
    if ((bad[0] || bad[1] || bad[2] || bad[3]) && logs.fetch_add(1) < 6) {
        host_log("yebis: built inputs differ: vs_pgm %d ps_pgm %d vs_user %d ps_user %d", bad[0], bad[1], bad[2], bad[3]);
        for (int i = 0; i < 4; ++i) {
            host_log("yebis:   vs_pgm[%d] built %08x registers %08x; ps_pgm[%d] built %08x registers %08x", i, b.vs_pgm[i],
                     sh[0x48 + i], i, b.ps_pgm[i], sh[0x08 + i]);
        }
        for (int i = 0; i < 16; ++i) {
            if (b.vs_user[i] != sh[0x4c + i] || b.ps_user[i] != sh[0x0c + i]) {
                host_log("yebis:   user[%d] vs built %08x registers %08x; ps built %08x registers %08x", i, b.vs_user[i],
                         sh[0x4c + i], b.ps_user[i], sh[0x0c + i]);
            }
        }
    }
}

void hle_gx_yebis_drew(std::uint32_t index_count, std::uint64_t index_va, const GxYebisDraw& want) {
    if (index_count == want.index_count && index_va == want.index_va) {
        g_yebis_matched.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (g_yebis_mismatched.fetch_add(1, std::memory_order_relaxed) < 8) {
        host_log("yebis: token asked for %u indices at 0x%llx, the draw packet has %u at 0x%llx", want.index_count,
                 static_cast<ull>(want.index_va), index_count, static_cast<ull>(index_va));
    }
}

void hle_gx_image_copy_token(std::uint64_t id) {
    GpuImageCopy c;
    {
        std::lock_guard<std::mutex> lk(g_image_copies_mu);
        const auto it = g_image_copies.find(id);
        if (it == g_image_copies.end()) {
            g_icopy_unmatched.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        c = it->second;
        g_image_copies.erase(it);
    }
    host_gpu_copy_image_region(c);
}

void hle_gx_upload_token(std::uint64_t id) {
    GuestUpload up;
    {
        std::lock_guard<std::mutex> lk(g_uploads_mu);
        const auto it = g_uploads.find(id);
        if (it == g_uploads.end()) {
            g_upload_unmatched.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        up = std::move(it->second);
        g_uploads.erase(it);
    }
    host_gpu_upload_region(up.base, up.tsharp, up.mip, up.layer, up.x, up.y, up.w, up.h, up.data.data(), up.data.size(), up.row_bytes);
}

void hle_gx_fill_token(std::uint64_t id, const std::uint32_t* ctx) {
    std::vector<GuestFill> fills;
    {
        std::lock_guard<std::mutex> lk(g_fills_mu);
        const auto it = g_fills.find(id);
        if (it == g_fills.end()) {
            g_fill_unmatched.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        fills = std::move(it->second);
        g_fills.erase(it);
    }
    for (const GuestFill& fl : fills) {
        g_fill_ranges.fetch_add(1, std::memory_order_relaxed);
        g_fill_bytes.fetch_add(fl.bytes, std::memory_order_relaxed);
        if (fl.depth_target) {
            std::uint32_t word;
            std::memcpy(&word, &fl.rgba[0], 4);
            host_gpu_depth_clear(fl.va, word, fl.stencil_clear);
            continue;
        }
        host_gpu_fill(fl.va, fl.bytes, fl.rgba, fl.clear_known ? fl.depth_clear : ctx[0x0b], fl.clear_known ? fl.stencil_clear : ctx[0x0a]);
    }
}

void hle_gx_copy_token(std::uint64_t id) {
    std::vector<GuestCopy> copies;
    {
        std::lock_guard<std::mutex> lk(g_copies_mu);
        const auto it = g_copies.find(id);
        if (it == g_copies.end()) {
            g_copy_unmatched.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        copies = std::move(it->second);
        g_copies.erase(it);
    }
    // BBHOST_COPY_TRACE=<flip>: each token of that flip and the next, with the
    // draw it falls before (BBHOST_DUMP_DRAWS lists the draws), for G9.
    static const long trace_flip = [] {
        const char* e = std::getenv("BBHOST_COPY_TRACE");
        return e ? std::strtol(e, nullptr, 10) : -1L;
    }();
    if (trace_flip >= 0) {
        const long flip = static_cast<long>(hle_video_flip_count());
        if (flip >= trace_flip && flip <= trace_flip + 1) {
            std::uint64_t bytes = 0;
            for (const GuestCopy& c : copies) bytes += c.bytes;
            host_log("copy-trace: flip %ld before draw %llu: %zu copies, %llu bytes; first 0x%llx -> 0x%llx (%llu bytes)", flip,
                     static_cast<ull>(host_gpu_draw_mark()), copies.size(), static_cast<ull>(bytes),
                     copies.empty() ? 0ull : static_cast<ull>(copies[0].src), copies.empty() ? 0ull : static_cast<ull>(copies[0].dst),
                     copies.empty() ? 0ull : static_cast<ull>(copies[0].bytes));
        }
    }
    for (const GuestCopy& c : copies) {
        g_copy_ranges.fetch_add(1, std::memory_order_relaxed);
        g_copy_bytes.fetch_add(c.bytes, std::memory_order_relaxed);
        if (!c.bytes || (c.back ? host_gpu_copy_back(c.dst, c.src, c.bytes) : host_gpu_copy_guest(c.dst, c.src, c.bytes))) continue;
        if (g_copy_failed.fetch_add(1, std::memory_order_relaxed) < 8) {
            host_log("gx-copy: buffer copy 0x%llx -> 0x%llx (%llu bytes) could not be recorded", static_cast<ull>(c.src),
                     static_cast<ull>(c.dst), static_cast<ull>(c.bytes));
        }
    }
    static std::atomic<std::uint64_t> walked{0};
    if (walked.fetch_add(1, std::memory_order_relaxed) % 50000 == 49999) {
        host_log("gx-copy: calls %llu, tokens written %llu, walked %llu, copies %llu (%.1f MiB; render-target images %llu), failed %llu, "
                 "tokens without copies %llu, not written %llu",
                 static_cast<ull>(g_copy_calls.load()), static_cast<ull>(g_copy_tokens.load()), static_cast<ull>(walked.load()),
                 static_cast<ull>(g_copy_ranges.load()), static_cast<double>(g_copy_bytes.load()) / (1 << 20),
                 static_cast<ull>(host_gpu_guest_copy_targets()), static_cast<ull>(g_copy_failed.load()), static_cast<ull>(g_copy_unmatched.load()), static_cast<ull>(g_copy_unwritten.load()));
    }
}

// The CP walked a Scaleform draw token. Native mode
// places the stage tables in the host ring, fills in the user-data pointers
// and hands the inputs to the CP's draw; shadow mode keeps the record for the
// draw packet that follows (hle_gx_scaleform_check).
thread_local bool t_sf_pending = false;
thread_local ScaleformDraw t_sf_rec;
enum SfField {
    kSfVsPgm, kSfPsPgm, kSfVsOutCntl, kSfPsRegs, kSfPsInputCntl, kSfVsTable, kSfPsTable, kSfVsSubTables, kSfPsSubTables, kSfVsValues,
    kSfPsValues, kSfPrim, kSfTargetMask, kSfBlend, kSfColour, kSfDepthControl, kSfStencilControl, kSfStencilRef, kSfDepthTarget,
    kSfSuScMode, kSfVte, kSfViewport, kSfScreenScissor, kSfGeometry, kSfInstances, kSfIndexOffset, kSfFields
};
const char* const kSfFieldName[kSfFields] = {
    "the VS program", "the PS program", "PA_CL_VS_OUT_CNTL", "the PS's SPI/CB registers", "SPI_PS_INPUT_CNTL", "the VS table",
    "the PS table", "the VS sub-table pointers", "the PS sub-table pointers", "the VS value slots", "the PS value slots",
    "VGT_PRIMITIVE_TYPE", "CB_TARGET_MASK", "CB_BLEND0_CONTROL", "CB_COLOR0", "DB_DEPTH_CONTROL", "DB_STENCIL_CONTROL",
    "DB_STENCILREFMASK", "the depth target", "PA_SU_SC_MODE_CNTL", "PA_CL_VTE_CNTL", "the viewport", "the screen scissor",
    "the geometry", "NUM_INSTANCES", "VGT_INDX_OFFSET"};
std::atomic<std::uint64_t> g_sf_compared{0}, g_sf_field_bad[kSfFields] = {}, g_sf_unnamed_set[2][16] = {};
// The register-file fields a token does not carry, at the draws compared:
// their distinct values, to choose what a token draw takes for them.
std::map<std::string, std::map<std::uint32_t, std::uint64_t>> g_sf_inherited;  // under g_sf_mu
std::atomic<std::uint64_t> g_sf_logged{0};

bool sf_place_tables(ScaleformDraw& d) {
    for (int s = 0; s < 2; ++s) {
        SfStage& st = d.stage[s];
        std::uint32_t* user = s == 0 ? d.in.vs_user : d.in.ps_user;
        // The uniform block as it was at the call, and its V# pointed there.
        if (!st.uniforms.empty() && st.uniform_ud >= 0) {
            const std::uint64_t va = host_gpu_ring_place(st.uniforms.data(), st.uniforms.size() * 4, 256);
            if (va) {
                std::uint32_t* v = st.uniform_direct ? user + st.uniform_ud
                                   : static_cast<std::size_t>(st.uniform_ud) + 1 < st.table.size() ? st.table.data() + st.uniform_ud
                                                                                                   : nullptr;
                if (v) {
                    v[0] = static_cast<std::uint32_t>(va);
                    v[1] = (v[1] & ~0xfffu) | (static_cast<std::uint32_t>(va >> 32) & 0xfff);
                    g_sf_uniforms_placed.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        if (st.table.empty()) continue;
        const std::uint64_t va = host_gpu_ring_place(st.table.data(), st.table.size() * 4);
        if (!va) return false;
        for (int i = 0; i < st.nptrs; ++i) {
            const std::uint64_t p = va + static_cast<std::uint64_t>(st.ptrs[i].off) * 4;
            user[st.ptrs[i].ud] = static_cast<std::uint32_t>(p);
            user[st.ptrs[i].ud + 1] = static_cast<std::uint32_t>(p >> 32);
        }
    }
    return true;
}

// The last Scaleform token drawn on this (CP) thread, for a line that asks
// why its user data came out as it did (the glitch hunt).
struct SfLast {
    std::uint32_t n[2] = {}, nptrs[2] = {}, named[2] = {}, direct[2] = {};
    std::uint8_t ptr_ud[2][7] = {};
    std::uint32_t ptr_off[2][7] = {};
    std::uint32_t ps_user[4] = {};
};
thread_local SfLast t_sf_last;

const GpuDrawInputs* hle_gx_scaleform_token(std::uint64_t id) {
    ScaleformDraw d;
    {
        std::lock_guard<std::mutex> lk(g_sf_mu);
        const auto it = g_sf_draws.find(id);
        if (it == g_sf_draws.end()) {
            g_sf_unmatched.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        d = std::move(it->second);
        g_sf_draws.erase(it);
    }
    if (d.shadow) {
        t_sf_rec = std::move(d);
        t_sf_pending = true;
        return nullptr;
    }
    if (!sf_place_tables(d)) {
        g_sf_placed_failed.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    static thread_local GpuDrawInputs t_sf_in;
    t_sf_in = d.in;
    static const bool glitch = [] {
        const char* e = std::getenv("BBHOST_GLITCH");
        return e && e[0] == '1';
    }();
    for (int k = 0; glitch && k < 2; ++k) {
        const SfStage& st = d.stage[k];
        t_sf_last.n[k] = static_cast<std::uint32_t>(st.table.size());
        t_sf_last.nptrs[k] = static_cast<std::uint32_t>(st.nptrs);
        t_sf_last.named[k] = st.named;
        t_sf_last.direct[k] = st.direct;
        for (int i = 0; i < st.nptrs && i < 7; ++i) {
            t_sf_last.ptr_ud[k][i] = st.ptrs[i].ud;
            t_sf_last.ptr_off[k][i] = st.ptrs[i].off;
        }
    }
    if (glitch) std::memcpy(t_sf_last.ps_user, d.in.ps_user, sizeof(t_sf_last.ps_user));
    t_objects = d.obj;
    t_records.reset();
    t_objects.records = nullptr;
    t_objects_valid = true;
    g_sf_drawn.fetch_add(1, std::memory_order_relaxed);
    return &t_sf_in;
}

bool hle_gx_scaleform_pending() { return t_sf_pending; }

bool hle_gx_shadow_wanted() {
    return g_native != 2 || g_backend != 1 || g_native_scaleform == 1 || g_native_utility == 1 || g_yebis_full == 1 ||
           g_native_dispatch == 1 || (g_native_yebis != 5 && g_native_yebis != 0);
}

void sf_compare(const ScaleformDraw& d, bool state_only, std::uint32_t count, std::uint64_t index_va, std::uint32_t index_type,
                std::uint32_t instances, std::uint32_t prim, const std::uint32_t* sh, const std::uint32_t* ctx);

void hle_gx_scaleform_check(std::uint32_t count, std::uint64_t index_va, std::uint32_t index_type, std::uint32_t instances,
                            std::uint32_t prim, const std::uint32_t* sh, const std::uint32_t* ctx) {
    t_sf_pending = false;
    sf_compare(t_sf_rec, false, count, index_va, index_type, instances, prim, sh, ctx);
}

void hle_gx_wrapper_compare(const GpuDrawInputs& in, std::uint32_t count, std::uint64_t index_va, std::uint32_t index_type,
                            std::uint32_t instances, std::uint32_t prim, const std::uint32_t* sh, const std::uint32_t* ctx) {
    ScaleformDraw d;
    d.in = in;
    d.obj.caller = kYebisDrawSite;
    d.obj.geometry = true;
    d.obj.count = count;
    d.obj.instances = instances;
    d.obj.indexed = index_va != 0;
    d.obj.index_va = index_va;
    d.obj.index_type = index_type;
    sf_compare(d, true, count, index_va, index_type, instances, prim, sh, ctx);
}

void sf_compare(const ScaleformDraw& d, bool state_only, std::uint32_t count, std::uint64_t index_va, std::uint32_t index_type,
                std::uint32_t instances, std::uint32_t prim, const std::uint32_t* sh, const std::uint32_t* ctx) {
    const GpuDrawInputs& in = d.in;
    bool bad[kSfFields] = {};
    std::string detail;
    const auto differ = [&](SfField f, const char* what, std::uint64_t built, std::uint64_t regs) {
        bad[f] = true;
        if (detail.size() < 1500) appendf(detail, " %s built %llx registers %llx;", what, static_cast<ull>(built), static_cast<ull>(regs));
    };
    for (int i = 0; i < 4 && !state_only; ++i) {
        if (in.vs_pgm[i] != sh[0x48 + i]) differ(kSfVsPgm, "vs_pgm", in.vs_pgm[i], sh[0x48 + i]);
        if (in.ps_pgm[i] != sh[0x08 + i]) differ(kSfPsPgm, "ps_pgm", in.ps_pgm[i], sh[0x08 + i]);
    }
    if (state_only) goto state;
    if (in.vs_out_cntl != ctx[0x207]) differ(kSfVsOutCntl, "vs_out_cntl", in.vs_out_cntl, ctx[0x207]);
    if (in.ps_input_ena != ctx[0x1b3]) differ(kSfPsRegs, "ps_input_ena", in.ps_input_ena, ctx[0x1b3]);
    if (in.ps_in_control != ctx[0x1b6]) differ(kSfPsRegs, "ps_in_control", in.ps_in_control, ctx[0x1b6]);
    if (in.cb_shader_mask != ctx[0x8f]) differ(kSfPsRegs, "cb_shader_mask", in.cb_shader_mask, ctx[0x8f]);
    if (in.ps_col_format != ctx[0x1c5]) differ(kSfPsRegs, "ps_col_format", in.ps_col_format, ctx[0x1c5]);
    for (std::uint32_t i = 0; i < in.ps_input_count; ++i) {
        if (in.ps_input_cntl[i] != ctx[0x191 + i]) differ(kSfPsInputCntl, "ps_input_cntl", in.ps_input_cntl[i], ctx[0x191 + i]);
    }
    // User data: the table's dwords at the register file's pointer against the
    // copy; the sub-table pointers as offsets from it; the value slots as they
    // are. Slots the token does not name that hold something are counted.
    for (int s = 0; s < 2 && !state_only; ++s) {
        const SfStage& st = d.stage[s];
        const std::uint32_t* regs = sh + (s == 0 ? 0x4c : 0x0c);
        const std::uint32_t* user = s == 0 ? in.vs_user : in.ps_user;
        std::uint64_t base = 0;
        for (int i = 0; i < st.nptrs; ++i) {
            if (st.ptrs[i].off == 0) base = regs[st.ptrs[i].ud] | (static_cast<std::uint64_t>(regs[st.ptrs[i].ud + 1]) << 32);
        }
        if (!st.table.empty()) {
            std::vector<std::uint32_t> theirs(st.table.size());
            if (!base || !safe_read(base, theirs.data(), theirs.size() * 4)) {
                differ(s ? kSfPsTable : kSfVsTable, "table pointer", 0, base);
            } else if (theirs != st.table) {
                std::size_t first = 0;
                while (first < theirs.size() && theirs[first] == st.table[first]) ++first;
                differ(s ? kSfPsTable : kSfVsTable, "table dword", st.table[first], theirs[first]);
                if (detail.size() < 1500) appendf(detail, " (at dword %zu of %zu)", first, theirs.size());
            }
        }
        for (int i = 0; i < st.nptrs; ++i) {
            const std::uint64_t want = base + static_cast<std::uint64_t>(st.ptrs[i].off) * 4;
            const std::uint64_t have = regs[st.ptrs[i].ud] | (static_cast<std::uint64_t>(regs[st.ptrs[i].ud + 1]) << 32);
            if (have != want) differ(s ? kSfPsSubTables : kSfVsSubTables, "sub-table", want, have);
        }
        std::uint32_t pointer_slots = 0;
        for (int i = 0; i < st.nptrs; ++i) pointer_slots |= 3u << st.ptrs[i].ud;
        for (int k = 0; k < 16; ++k) {
            if ((pointer_slots >> k) & 1) continue;
            if ((st.named >> k) & 1) {
                if (user[k] != regs[k]) differ(s ? kSfPsValues : kSfVsValues, "value slot", user[k], regs[k]);
            } else if (regs[k]) {
                g_sf_unnamed_set[s][k].fetch_add(1, std::memory_order_relaxed);
                static std::atomic<int> unnamed_logs{0};
                if (d.obj.caller != kSfHalDraw && unnamed_logs.fetch_add(1, std::memory_order_relaxed) < 12) {
                    host_log("gx-wrapper: token from 0x%llx: %s user slot %d holds %08x in the register file, not in the wrapper's state",
                             static_cast<ull>(d.obj.caller), s ? "PS" : "VS", k, regs[k]);
                    if (d.obj.caller == kGammaPassDraw && unnamed_logs.load() == 1) {
                        // The pass's GX shader objects (VS data_586c530, PS data_586c778): their user-data descriptors.
                        for (std::uint64_t obj : {guest(0x586c530), guest(0x586c778)}) {
                            const std::uint64_t table = rd64(obj + 0xe0), count = rd64(obj + 0xe8) & 0xff;
                            std::string descs;
                            for (std::uint64_t i = 0; i < count && table; ++i) appendf(descs, " %08x", rd32(table + 4 * i));
                            host_log("gx-wrapper: shader object 0x%llx descriptors (%llu):%s", static_cast<ull>(gx_guest_to_bn(obj)),
                                     static_cast<ull>(count), descs.c_str());
                        }
                    }
                }
            }
        }
    }
state:
    if (in.prim != prim) differ(kSfPrim, "prim", in.prim, prim);
    if (in.target_mask != ctx[0x8e]) differ(kSfTargetMask, "target_mask", in.target_mask, ctx[0x8e]);
    if (in.blend[0] != ctx[0x1e0]) differ(kSfBlend, "blend0", in.blend[0], ctx[0x1e0]);
    for (int k = 0; k < 5; ++k) {
        if (in.color[0][k] != ctx[0x318 + k]) differ(kSfColour, "color0", in.color[0][k], ctx[0x318 + k]);
    }
    // DB_DEPTH_CONTROL: with Z_ENABLE (bit 1) off the depth function is not
    // read (YEBIS writes 0 where the flip's setter left ZFUNC always).
    const auto ds_norm = [](std::uint32_t v) { return (v & 2) ? v : v & ~0x70u; };
    if (ds_norm(in.depth_control) != ds_norm(ctx[0x200])) differ(kSfDepthControl, "depth_control", in.depth_control, ctx[0x200]);
    if (in.stencil_control != ctx[0x10b]) differ(kSfStencilControl, "stencil_control", in.stencil_control, ctx[0x10b]);
    // The reference is only written for a mode with stencil on (0x73de50).
    if ((in.depth_control & 1) && in.stencil_ref != ctx[0x10c]) differ(kSfStencilRef, "stencil_ref", in.stencil_ref, ctx[0x10c]);
    if (in.z_info != ctx[0x10]) differ(kSfDepthTarget, "z_info", in.z_info, ctx[0x10]);
    if (in.stencil_info != ctx[0x11]) differ(kSfDepthTarget, "stencil_info", in.stencil_info, ctx[0x11]);
    // No depth target (0x14739b0 with none writes Z_INFO and STENCIL_INFO 0 and
    // leaves the rest): the other words are not read.
    if (in.z_info || ctx[0x10]) {
        if (in.z_read_base != ctx[0x12]) differ(kSfDepthTarget, "z_read_base", in.z_read_base, ctx[0x12]);
        if (in.depth_size != ctx[0x16]) differ(kSfDepthTarget, "depth_size", in.depth_size, ctx[0x16]);
        if (in.depth_view != ctx[0x02]) differ(kSfDepthTarget, "depth_view", in.depth_view, ctx[0x02]);
        if (in.htile_base != ctx[0x05]) differ(kSfDepthTarget, "htile_base", in.htile_base, ctx[0x05]);
    }
    // PA_SU_SC_MODE_CNTL: with POLY_MODE (bits 3-4) off the polygon types
    // (bits 5-10) are not read.
    const auto su_norm = [](std::uint32_t v) { return ((v >> 3) & 3) ? v : v & ~0x7e0u; };
    if (su_norm(in.su_sc_mode) != su_norm(ctx[0x205])) differ(kSfSuScMode, "su_sc_mode", in.su_sc_mode, ctx[0x205]);
    if (in.vte_cntl != ctx[0x206]) differ(kSfVte, "vte_cntl", in.vte_cntl, ctx[0x206]);
    for (int k = 0; k < 6; ++k) {
        std::uint32_t built;
        std::memcpy(&built, &in.vport[k], 4);
        if (built != ctx[0x10f + k]) differ(kSfViewport, "vport", built, ctx[0x10f + k]);
    }
    if (in.screen_scissor[0] != ctx[0x0c] || in.screen_scissor[1] != ctx[0x0d]) {
        differ(kSfScreenScissor, "screen_scissor", (static_cast<std::uint64_t>(in.screen_scissor[0]) << 32) | in.screen_scissor[1],
               (static_cast<std::uint64_t>(ctx[0x0c]) << 32) | ctx[0x0d]);
    }
    if (d.obj.count != count) differ(kSfGeometry, "count", d.obj.count, count);
    if (d.obj.indexed != (index_va != 0)) differ(kSfGeometry, "indexed", d.obj.indexed, index_va != 0);
    if (d.obj.indexed && d.obj.index_va != index_va) differ(kSfGeometry, "index_va", d.obj.index_va, index_va);
    if (d.obj.indexed && d.obj.index_type != index_type) differ(kSfGeometry, "index_type", d.obj.index_type, index_type);
    if (d.obj.instances != (instances ? instances : 1)) differ(kSfInstances, "instances", d.obj.instances, instances);
    if (ctx[0x102] != 0) differ(kSfIndexOffset, "indx_offset", 0, ctx[0x102]);
    g_sf_compared.fetch_add(1, std::memory_order_relaxed);
    bool any = false;
    for (int k = 0; k < kSfFields; ++k) {
        if (!bad[k]) continue;
        any = true;
        g_sf_field_bad[k].fetch_add(1, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lk(g_sf_mu);
        const auto note = [&](const char* name, std::uint32_t v) {
            auto& m = g_sf_inherited[name];
            if (m.size() < 8 || m.count(v)) ++m[v];
        };
        note("PA_CL_CLIP_CNTL", ctx[0x204]);
        note("PA_SC_GENERIC_SCISSOR_TL", ctx[0x90]);
        note("PA_SC_GENERIC_SCISSOR_BR", ctx[0x91]);
        note("PA_SC_VPORT_SCISSOR_0_TL", ctx[0x94]);
        note("PA_SC_VPORT_SCISSOR_0_BR", ctx[0x95]);
        note("PA_SC_MODE_CNTL_0", ctx[0x292]);
        note("DB_RENDER_CONTROL", ctx[0x000]);
        note("CB_COLOR_CONTROL", ctx[0x202]);
        note("DB_DEPTH_BOUNDS_MIN", ctx[0x008]);
        note("DB_DEPTH_BOUNDS_MAX", ctx[0x009]);
        note("CB_BLEND1_CONTROL", ctx[0x1e1]);
    }
    if (any && g_sf_logged.fetch_add(1, std::memory_order_relaxed) < 24) {
        host_log("gx-scaleform: token from 0x%llx (blend mode %u, depth-stencil mode %u, raster mode %u, HAL flags %08x, VP flags %08x, "
                 "rect %d,%d-%d,%d) differs from the draw packet:%s",
                 static_cast<ull>(d.obj.caller), d.blend_mode, d.ds_mode, d.raster_mode, d.hal_flags, d.vp_flags, d.rect[0], d.rect[1], d.rect[2],
                 d.rect[3], detail.c_str());
    }
}

// At exit (main.cpp's exit_reports), not in the periodic report.
void report_scaleform() {
    std::uint64_t calls = 0;
    for (const auto& c : g_sf_calls) calls += c.load();
    if (!calls && !g_sf_tokens.load()) return;
    std::string refused;
    for (int k = 0; k < kSfRefusals; ++k) {
        if (const std::uint64_t n = g_sf_refused[k].load()) appendf(refused, " %s %llu;", kSfRefusalName[k], static_cast<ull>(n));
    }
    std::size_t pending;
    {
        std::lock_guard<std::mutex> lk(g_sf_mu);
        pending = g_sf_draws.size();
    }
    host_log("gx-scaleform: HAL draws hooked %llu (indexed %llu, instanced %llu, non-indexed %llu), tokens %llu, not written %llu, "
             "refused:%s draws with direct-SGPR slots %llu, no viewport flag %llu; walked but unmatched %llu, tables not placed %llu, drawn %llu, pending %zu",
             static_cast<ull>(calls), static_cast<ull>(g_sf_calls[0].load()), static_cast<ull>(g_sf_calls[1].load()),
             static_cast<ull>(g_sf_calls[2].load()), static_cast<ull>(g_sf_tokens.load()), static_cast<ull>(g_sf_unwritten.load()),
             refused.empty() ? " none;" : refused.c_str(), static_cast<ull>(g_sf_direct_layouts.load()), static_cast<ull>(g_sf_no_viewport.load()),
             static_cast<ull>(g_sf_unmatched.load()), static_cast<ull>(g_sf_placed_failed.load()), static_cast<ull>(g_sf_drawn.load()), pending);
    if (const std::uint64_t compared = g_sf_compared.load()) {
        std::string fields;
        for (int k = 0; k < kSfFields; ++k) {
            if (const std::uint64_t n = g_sf_field_bad[k].load()) appendf(fields, " %s %llu;", kSfFieldName[k], static_cast<ull>(n));
        }
        host_log("gx-scaleform: shadow: %llu draws compared; fields differing:%s", static_cast<ull>(compared),
                 fields.empty() ? " none" : fields.c_str());
        std::string unnamed;
        for (int s = 0; s < 2; ++s) {
            for (int k = 0; k < 16; ++k) {
                if (const std::uint64_t n = g_sf_unnamed_set[s][k].load()) appendf(unnamed, " %s[%d] %llu;", s ? "PS" : "VS", k, static_cast<ull>(n));
            }
        }
        host_log("gx-scaleform: shadow: user-data slots the token does not name that hold a value:%s", unnamed.empty() ? " none" : unnamed.c_str());
        std::lock_guard<std::mutex> lk(g_sf_mu);
        for (const auto& [name, values] : g_sf_inherited) {
            std::string vs;
            for (const auto& [v, n] : values) appendf(vs, " %08x x%llu;", v, static_cast<ull>(n));
            host_log("gx-scaleform: shadow: %s at the draws:%s", name.c_str(), vs.c_str());
        }
    }
}
std::string hle_gx_scaleform_last() {
    const SfLast& l = t_sf_last;
    std::string out;
    for (int k = 0; k < 2; ++k) {
        appendf(out, "%s: table %u dwords, named %08x, direct %08x, pointers", k ? "PS" : "VS", l.n[k], l.named[k], l.direct[k]);
        for (std::uint32_t i = 0; i < l.nptrs[k] && i < 7; ++i) appendf(out, " s[%u]->+%u", l.ptr_ud[k][i], l.ptr_off[k][i]);
        out += k ? "; " : "; ";
    }
    appendf(out, "PS user 0-3 %08x %08x %08x %08x", l.ps_user[0], l.ps_user[1], l.ps_user[2], l.ps_user[3]);
    return out;
}

void hle_gx_scaleform_report() {
    if (g_sf_uniforms_placed.load()) {
        host_log("gx-scaleform: %llu uniform blocks taken at the call and read from there (BBHOST_SF_UNIFORMS)",
                 static_cast<unsigned long long>(g_sf_uniforms_placed.load()));
    }
    if (g_view_check) {
        using ull = unsigned long long;
        host_log("gx: bound views' T#s as made %llu, changed since %llu, not registered %llu; samplers' S#s as made %llu, changed %llu, "
                 "not registered %llu",
                 static_cast<ull>(g_view_words[0][0].load()), static_cast<ull>(g_view_words[0][1].load()), static_cast<ull>(g_view_words[0][2].load()),
                 static_cast<ull>(g_view_words[1][0].load()), static_cast<ull>(g_view_words[1][1].load()), static_cast<ull>(g_view_words[1][2].load()));
    }
    report_scaleform();
    report_wrapper_draws();
    report_dispatch_tokens();
    report_state_packets();
}
bool hle_gx_dispatch_native() { return g_native_dispatch == 2; }
const GpuDispatch* hle_gx_dispatch_token(std::uint64_t id) { return dispatch_token_impl(id); }
bool hle_gx_dispatch_pending() { return dispatch_pending_impl(); }
void hle_gx_dispatch_check(const GpuDispatch& regs, const std::uint32_t* writers) { dispatch_check_impl(regs, writers); }


const GpuDrawInputs* hle_gx_native_token(std::uint64_t id, const std::uint32_t* sh, const std::uint32_t* ctx) {
    g_native_seen.fetch_add(1, std::memory_order_relaxed);
    if (g_native == 2) {
        report_gx_costs();
        GxCostTimer timer(kCostToken);
        GxCostStamp stamp;
        // The call's record: in its ring slot, or in the overflow map.
        NativeSlot* slot = &native_slot(id);
        NativeDraw overflow;
        NativeDraw* draw = nullptr;
        if (slot->state.load(std::memory_order_acquire) == 2 && slot->id == id) {
            draw = &slot->draw;
        } else {
            slot = nullptr;
            std::lock_guard<std::mutex> lk(g_native_mu);
            const auto it = g_native_draws.find(id);
            if (it == g_native_draws.end()) {
                g_native_unmatched.fetch_add(1, std::memory_order_relaxed);
                return nullptr;
            }
            overflow = std::move(it->second);
            g_native_draws.erase(it);
            draw = &overflow;
        }
        stamp.to(kCostTokenLookup);
        // The draw's inputs and objects are read where the call
        // left them - its slot, held until the draw is done
        // (hle_gx_native_release) - instead of copied out of it.
        if (!slot) {
            t_native_overflow = std::move(*draw);
            draw = &t_native_overflow;
        }
        // The slot was streamed past the caches, so every line of it is a
        // miss: ask for all of them at once, so the misses overlap instead of
        // coming one field at a time as the draw reads them (a copy used to
        // do that by walking the record in order).
        {
            const auto* in_lines = reinterpret_cast<const char*>(&draw->in);
            for (std::size_t off = 0; off < sizeof(GpuDrawInputs); off += 64) __builtin_prefetch(in_lines + off);
            const auto* obj_lines = reinterpret_cast<const char*>(&draw->objects);
            for (std::size_t off = 0; off < sizeof(GxDrawObjects); off += 64) __builtin_prefetch(obj_lines + off);
            // And the lines of the draw's records it will read: each stage's
            // header, descriptors and the view, sampler and constant-buffer
            // slots that are set. They were written on the recording thread and
            // are just as cold; whichever phase of the draw touched them first
            // paid for them one at a time.
            if (const GxDrawRecords* rec = draw->records.get()) prefetch_records(*rec);
        }
        t_native_slot = slot;
        draw->objects.records = draw->records.get();
        t_native_objects = &draw->objects;
        GpuDrawInputs& t_native_in = draw->in;
        // A token draw takes nothing from the register file.
        // The draw's own packets never change these (the shadow run,
        // gx-native-shadow-01). DB_RENDER_CONTROL was 0, the stencil clear 0
        // and the generic scissor never narrower than the screen scissor at
        // every token (native-regs-01, -02), so a token draw keeps GX's values
        // for them (0, no generic scissor); the counters below keep watching
        // the register file as a check. User data in slots no descriptor names
        // is 0 (the renderer builds the named slots from the records). The
        // depth clear, the last value a token read, is now captured at the
        // HTILE fill that asks for the clear (render.cpp). The PS input table,
        // viewport scissor and PA_SC_MODE_CNTL_0 stay as GX builds them.
        if (ctx[0x000]) g_token_rc_set.fetch_add(1, std::memory_order_relaxed);
        if (ctx[0x000] & 3) g_token_rc_clear.fetch_add(1, std::memory_order_relaxed);
        if (ctx[0x00b]) g_token_depth_clear_set.fetch_add(1, std::memory_order_relaxed);
        if (ctx[0x00a] & 0xff) g_token_stencil_clear_set.fetch_add(1, std::memory_order_relaxed);
        {
            const auto xy = [](std::uint32_t v) { return std::make_pair(v & 0x7fff, (v >> 16) & 0x7fff); };
            const auto [sx0, sy0] = xy(t_native_in.screen_scissor[0]);
            const auto [sx1, sy1] = xy(t_native_in.screen_scissor[1]);
            const auto [gx0, gy0] = xy(ctx[0x090]);
            const auto [gx1, gy1] = xy(ctx[0x091]);
            if (gx0 > sx0 || gy0 > sy0 || (gx1 && sx1 && gx1 < sx1) || (gy1 && sy1 && gy1 < sy1)) {
                g_token_generic_narrows.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (g_token_user_data_register) {
            std::memcpy(t_native_in.vs_user, &sh[0x4c], sizeof(t_native_in.vs_user));
            std::memcpy(t_native_in.ps_user, &sh[0x0c], sizeof(t_native_in.ps_user));
        } else {
            std::memset(t_native_in.vs_user, 0, sizeof(t_native_in.vs_user));
            std::memset(t_native_in.ps_user, 0, sizeof(t_native_in.ps_user));
        }
        if (g_native_drawn.fetch_add(1, std::memory_order_relaxed) % 100000 == 99999) {
            std::size_t pending = 0;
            std::string stale;
            {
                // Calls whose token has not come back within 20,000 later tokens, by caller.
                std::lock_guard<std::mutex> lk(g_native_mu);
                pending = g_native_draws.size() + g_native_slot_pending.load();
                std::map<std::tuple<std::uint64_t, int, std::uint64_t>, std::uint64_t> old;
                for (const auto& [key, draw] : g_native_draws) {
                    if (key + 20000 < id) ++old[{bn(draw.ret), draw.kind, draw.ctx}];
                }
                for (const auto& [k, n] : old) {
                    if (stale.size() > 600) break;
                    appendf(stale, " %s from 0x%llx ctx 0x%llx: %llu;", kKindName[std::get<1>(k)], static_cast<ull>(std::get<0>(k)),
                            static_cast<ull>(std::get<2>(k)), static_cast<ull>(n));
                }
            }
            if (!stale.empty()) host_log("gx-native: calls whose token did not come back:%s", stale.c_str());
            if (g_native_yebis) {
                host_log("yebis: draws %llu, tokens written %llu (not written %llu), walked %llu, the draw packet after the token was "
                         "the one it named %llu, a different draw %llu, token without a recorded draw %llu",
                         static_cast<ull>(g_yebis_calls.load()), static_cast<ull>(g_yebis_tokens.load()),
                         static_cast<ull>(g_yebis_unwritten.load()), static_cast<ull>(g_yebis_walked.load()),
                         static_cast<ull>(g_yebis_matched.load()), static_cast<ull>(g_yebis_mismatched.load()),
                         static_cast<ull>(g_yebis_unmatched.load()));
                if (const std::uint64_t n = g_yebis_checked.load()) {
                    std::string fields;
                    for (int k = 0; k < 6; ++k) {
                        appendf(fields, " %s %llu", kYebisField[k], static_cast<ull>(g_yebis_field_bad[k].load()));
                    }
                    host_log("yebis: inputs built from YEBIS's state checked against the registers its commit wrote: %llu draws, "
                             "differing by field:%s",
                             static_cast<ull>(n), fields.c_str());
                    for (int w = 0; w < 2; ++w) {
                        std::string per;
                        for (int i = 0; i < 16; ++i) {
                            if (const std::uint64_t m = g_yebis_slot_bad[w][i].load()) {
                                appendf(per, " s[%d]:%llu", i, static_cast<ull>(m));
                            }
                        }
                        host_log("yebis:   %s user data differing by slot:%s", w ? "ps" : "vs", per.empty() ? " none" : per.c_str());
                    }
                }
                if (const std::uint64_t n = g_yebis_ns_n.load()) {
                    host_log("yebis: a flipped draw costs %llu ns reading the state and %llu ns building it, of which "
                             "%llu ns is ours and %llu ns the guest's calls (sub_15f7840 itself %llu ns)",
                             static_cast<ull>(g_yebis_ns_host.load() / n), static_cast<ull>(g_yebis_ns_builders.load() / n),
                             static_cast<ull>(g_yebis_ns_native.load() / n), static_cast<ull>((g_yebis_ns_builders.load() - g_yebis_ns_native.load()) / n),
                             static_cast<ull>(g_yebis_ns_build_only.load() / n));
                    std::string per;
                    for (int k = 0; k < 6; ++k) {
                        appendf(per, " %s %llu;", kYebisCallName[k], static_cast<ull>(g_yebis_ns_call[k].load() / n));
                    }
                    host_log("yebis:   ns a draw by call:%s", per.c_str());
                    yebis_bind::report();
                }
                if (g_yebis_flipped.load() || g_yebis_flip_failed.load()) {
                    host_log("yebis: draws run from their token %llu, left to the method %llu",
                             static_cast<ull>(g_yebis_flipped.load()), static_cast<ull>(g_yebis_flip_failed.load()));
                }
                if (g_native_yebis >= 2) {
                    std::string stages;
                    for (int k = 0; k < 8; ++k) {
                        if (const std::uint64_t n = g_yebis_stage_seen[k].load()) {
                            appendf(stages, " stage %d %llu", k, static_cast<ull>(n));
                        }
                    }
                    host_log("yebis: stages bound:%s", stages.empty() ? " none" : stages.c_str());
                    for (int w = 0; w < 2; ++w) {
                        std::string types, slots;
                        for (int k = 0; k < 32; ++k) {
                            if (const std::uint64_t n = g_yebis_desc[w][k].load()) {
                                appendf(types, " 0x%x:%llu", k, static_cast<ull>(n));
                            }
                        }
                        for (int k = 0; k < 16; ++k) {
                            if (const std::uint64_t n = g_yebis_slots[w][k].load()) {
                                appendf(slots, " s[%d]:%llu", k, static_cast<ull>(n));
                            }
                        }
                        if (!types.empty()) {
                            host_log("yebis: %s descriptors by type:%s; by user-data slot:%s", w ? "PS" : "VS and the rest",
                                     types.c_str(), slots.c_str());
                        }
                    }
                }
            }
            if (const std::uint64_t n = g_clear_calls.load()) {
                host_log("gx: deferred colour clears %llu, %llu slices", static_cast<ull>(n),
                         static_cast<ull>(g_clear_slices.load()));
            }
            if (const std::uint64_t on_stack = g_safe_reads_stack.load()) {
                host_log("gx: guest reads served from the reading thread's own stack %llu", static_cast<ull>(on_stack));
            }
            if (const std::uint64_t slow = g_safe_reads_slow.load()) {
                host_log("gx: guest reads that needed host_read_safe %llu, %llu ms in them, %llu ns each, %.1f MiB",
                         static_cast<ull>(slow), static_cast<ull>(g_safe_slow_ns.load() / 1000000),
                         static_cast<ull>(g_safe_slow_ns.load() / slow), static_cast<double>(g_safe_slow_bytes.load()) / (1 << 20));
                std::vector<std::pair<std::uint64_t, std::uint64_t>> where;
                for (int i = 0; i < kSlowBuckets; ++i) {
                    if (const std::uint64_t mb = g_slow_where[i].load()) {
                        where.push_back({g_slow_count[i].load(), mb - 1});
                    }
                }
                std::sort(where.rbegin(), where.rend());
                std::string top;
                for (std::size_t i = 0; i < where.size() && i < 12; ++i) {
                    appendf(top, " 0x%llx:%llu", static_cast<ull>(where[i].second << 20), static_cast<ull>(where[i].first));
                }
                host_log("gx: those reads by megabyte (%zu distinct), most first:%s", where.size(), top.c_str());
                std::vector<std::pair<std::uint64_t, std::uint64_t>> sites;
                for (int i = 0; i < kSlowBuckets; ++i) {
                    if (const std::uint64_t at = g_slow_from[i].load()) sites.push_back({g_slow_from_n[i].load(), at});
                }
                std::sort(sites.rbegin(), sites.rend());
                // Offsets from safe_read itself, so addr2line can name them
                // whatever the image was loaded at.
                const std::uint64_t anchor = reinterpret_cast<std::uint64_t>(&safe_read);
                std::string who;
                for (std::size_t i = 0; i < sites.size() && i < 10; ++i) {
                    appendf(who, " safe_read%+lld:%llu", static_cast<long long>(sites[i].second - anchor),
                            static_cast<ull>(sites[i].first));
                }
                host_log("gx: those reads by caller (%zu sites, offsets from safe_read):%s", sites.size(), who.c_str());
                // What those megabytes are, from the process's own map: the
                // point is to name the region so the read can stop being a
                // system call, not to guess from the address.
#if !defined(_WIN32)
                if (FILE* f = std::fopen("/proc/self/maps", "r")) {
                    char line[512];
                    while (std::fgets(line, sizeof(line), f)) {
                        unsigned long long lo = 0, hi = 0;
                        if (std::sscanf(line, "%llx-%llx", &lo, &hi) != 2) continue;
                        for (std::size_t i = 0; i < where.size() && i < 6; ++i) {
                            const std::uint64_t at = where[i].second << 20;
                            if (at >= lo && at < hi) {
                                std::string t(line);
                                while (!t.empty() && (t.back() == '\n' || t.back() == ' ')) t.pop_back();
                                host_log("gx:   0x%llx (%llu reads) is in %s", static_cast<ull>(at),
                                         static_cast<ull>(where[i].first), t.c_str());
                            }
                        }
                    }
                    std::fclose(f);
                }
#endif
            }
            host_log("gx-native: tokens %llu, drawn from tokens %llu, awaiting their token %llu, not qualified %llu, token not "
                     "written after the flushes %llu, token without a draw %llu, tokens failed %llu, stored past the ring %llu",
                     static_cast<ull>(g_native_tokens.load()), static_cast<ull>(g_native_drawn.load()), static_cast<ull>(pending),
                     static_cast<ull>(g_native_ineligible.load()), static_cast<ull>(g_native_late.load()),
                     static_cast<ull>(g_native_unmatched.load()), static_cast<ull>(g_native_failed.load()),
                     static_cast<ull>(g_native_overflow.load()));
            host_log("gx-native: stage records' shader fields from the per-shader cache %llu, read from the object %llu; inputs' "
                     "shader linkage from the per-thread cache %llu, worked out %llu (checked %llu, differing %llu)",
                     static_cast<ull>(g_shader_info.load(0)), static_cast<ull>(g_shader_info.load(1)), static_cast<ull>(g_link_info.load(0)),
                     static_cast<ull>(g_link_info.load(1)), static_cast<ull>(g_link_checked.load()), static_cast<ull>(g_link_differ.load()));
            host_log("gx-native: register-file fields at tokens: DB_RENDER_CONTROL set %llu (clear bits %llu), depth clear not 0 "
                     "%llu, stencil clear not 0 %llu, generic scissor narrower than the screen scissor %llu",
                     static_cast<ull>(g_token_rc_set.load()),
                     static_cast<ull>(g_token_rc_clear.load()), static_cast<ull>(g_token_depth_clear_set.load()),
                     static_cast<ull>(g_token_stencil_clear_set.load()), static_cast<ull>(g_token_generic_narrows.load()));
            {
                std::string why;
                for (int k = 0; k < kNativeSkips; ++k) {
                    if (const std::uint64_t n = g_native_skipped.load(k)) appendf(why, " %s %llu;", kNativeSkipName[k], static_cast<ull>(n));
                }
                host_log("gx-native: GX draws never offered a token:%s", why.empty() ? " none" : why.c_str());
                std::string not_qualified;
                for (int k = 0; k < kWhyCount; ++k) {
                    if (const std::uint64_t n = g_native_ineligible_why[k].load()) appendf(not_qualified, " %s %llu;", kIneligibleWhyName[k], static_cast<ull>(n));
                }
                host_log("gx-native: GX draws not qualified for a token:%s", not_qualified.empty() ? " none" : not_qualified.c_str());
            }
            const double tokens = static_cast<double>(std::max<std::uint64_t>(1, g_native_tokens.load()));
            const auto per = [&](NativeCall c) { return static_cast<double>(g_native_call_dwords.load(c)) / tokens; };
            host_log("gx-fill: fill calls %llu (colour clears by slice %llu, whole-resource %llu, layout refused %llu), tokens %llu, fills run %llu "
                     "(%.1f MiB), tokens unmatched %llu, not written %llu%s",
                     static_cast<ull>(g_fill_calls.load()), static_cast<ull>(g_fill_rtv_calls.load()), static_cast<ull>(g_fill_rtv_whole.load()),
                     static_cast<ull>(g_fill_rtv_layout_failed.load()), static_cast<ull>(g_fill_tokens.load()), static_cast<ull>(g_fill_ranges.load()),
                     static_cast<double>(g_fill_bytes.load()) / (1 << 20), static_cast<ull>(g_fill_unmatched.load()),
                     static_cast<ull>(g_fill_unwritten.load()), g_native_fills ? "" : " (BBHOST_GX_NATIVE_FILLS=0)");
            host_log("gx-upload: texture upload calls %llu, tokens %llu (%.1f MiB of rows), not 2D %llu, no T# in the registry %llu, refused %llu, "
                     "unmatched %llu, not written %llu%s",
                     static_cast<ull>(g_upload_calls.load()), static_cast<ull>(g_upload_tokens.load()), static_cast<double>(g_upload_bytes.load()) / (1 << 20),
                     static_cast<ull>(g_upload_not_2d.load()), static_cast<ull>(g_upload_no_tsharp.load()), static_cast<ull>(g_upload_bad_args.load()),
                     static_cast<ull>(g_upload_unmatched.load()), static_cast<ull>(g_upload_unwritten.load()), g_native_uploads ? "" : " (BBHOST_GX_NATIVE_UPLOADS=0)");
            host_log("gx-copy-image: texture copy calls %llu, tokens %llu, not 1D/2D %llu, no T# %llu, formats differ %llu, refused %llu, "
                     "unmatched %llu, not written %llu%s",
                     static_cast<ull>(g_icopy_calls.load()), static_cast<ull>(g_icopy_tokens.load()), static_cast<ull>(g_icopy_not_2d.load()),
                     static_cast<ull>(g_icopy_no_tsharp.load()), static_cast<ull>(g_icopy_format.load()), static_cast<ull>(g_icopy_bad.load()),
                     static_cast<ull>(g_icopy_unmatched.load()), static_cast<ull>(g_icopy_unwritten.load()),
                     g_native_image_copies ? "" : " (BBHOST_GX_NATIVE_IMAGE_COPIES=0)");
            host_log("gx-upload: calls not taken by resource type: 1D %llu, 3D %llu, buffers %llu, other %llu", static_cast<ull>(g_upload_by_type[2].load()),
                     static_cast<ull>(g_upload_by_type[4].load()), static_cast<ull>(g_upload_by_type[1].load()),
                     static_cast<ull>(g_upload_by_type[0].load() + g_upload_by_type[5].load() + g_upload_by_type[6].load() + g_upload_by_type[7].load()));
            host_log("gx-native: dwords a native draw writes to its command buffer: output-merger flush %.2f, input-assembly flush %.2f, "
                     "rasterizer flush %.2f, bookkeeping %.2f, output-merger commit %.2f, token 4, post-draw %.2f, bookkeeping after %.2f; "
                     "calls that moved to another chunk %llu; native draws that ran the output-merger flush %llu",
                     per(kNcOmFlush), per(kNcIaFlush), per(kNcRsFlush), per(kNcBookkeeping), per(kNcOmCommit), per(kNcPostDraw),
                     per(kNcBookkeepingAfter), static_cast<ull>(g_native_call_switches.load()),
                     static_cast<ull>(g_native_om_flushes.load()));
            if (g_native_regs) {
                for (const NativeCall c : {kNcOmFlush, kNcOmCommit, kNcPostDraw}) {
                    std::vector<std::pair<std::uint64_t, int>> top;
                    for (int reg = 0; reg < 0x400; ++reg) {
                        if (const std::uint64_t n = g_native_call_regs[c][reg].load()) top.emplace_back(n, reg);
                    }
                    std::sort(top.rbegin(), top.rend());
                    std::string regs, ops;
                    for (std::size_t i = 0; i < top.size() && i < 20; ++i) {
                        appendf(regs, " 0x%03x %llu;", top[i].second, static_cast<ull>(top[i].first));
                    }
                    for (int op = 0; op < 0x100; ++op) {
                        if (const std::uint64_t n = g_native_call_ops[c][op].load()) appendf(ops, " 0x%02x %llu;", op, static_cast<ull>(n));
                    }
                    host_log("gx-native: %s writes over %llu tokens, context registers:%s other packets:%s",
                             c == kNcOmFlush ? "output-merger flush" : c == kNcOmCommit ? "output-merger commit" : "post-draw",
                             static_cast<ull>(g_native_tokens.load()), regs.c_str(), ops.c_str());
                }
            }
        }
        return &t_native_in;
    }
    if (g_native != 1) return nullptr;
    if (t_native_shadow.size() > 4096) t_native_shadow.clear();
    NativeShadow& s = t_native_shadow[id];
    std::memcpy(s.user[0], &sh[0x4c], sizeof(s.user[0]));
    std::memcpy(s.user[1], &sh[0x0c], sizeof(s.user[1]));
    std::memcpy(s.ps_input_cntl, &ctx[0x191], sizeof(s.ps_input_cntl));
    s.render_control = ctx[0x000];
    s.depth_clear = ctx[0x00b];
    s.stencil_clear = ctx[0x00a];
    s.generic_scissor[0] = ctx[0x090];
    s.generic_scissor[1] = ctx[0x091];
    s.vport_scissor[0] = ctx[0x094];
    s.vport_scissor[1] = ctx[0x095];
    s.sc_mode_cntl_0 = ctx[0x292];
    {
        NativeTess& t = t_native_tess[id];
        t.stages = ctx[0x2d5];
        t.ls_hs_config = ctx[0x2d6];
        t.tf_param = ctx[0x2db];
        t.hos_max = ctx[0x286];
        t.hos_min = ctx[0x287];
        std::memcpy(t.ls_pgm, &sh[0x148], sizeof(t.ls_pgm));
        std::memcpy(t.hs_pgm, &sh[0x108], sizeof(t.hs_pgm));
        std::memcpy(t.ls_user, &sh[0x14c], sizeof(t.ls_user));
        std::memcpy(t.hs_user, &sh[0x10c], sizeof(t.hs_user));
        t.tess_cb[0] = sh[0x54];
        t.tess_cb[1] = sh[0x55];
    }
    return nullptr;
}

const GpuDrawInputs* hle_gx_trace_draw(std::uint32_t op, std::uint64_t packet_va, std::uint32_t count,
                                       std::uint32_t instances, std::uint64_t index_base, std::uint32_t index_type,
                                       std::uint32_t prim, const std::uint32_t* sh, const std::uint32_t* ctx,
                                       bool* render) {
    *render = false;
    t_objects_valid = false;
    if (!g_active.load(std::memory_order_relaxed)) return nullptr;
    report_gx_costs();
    GxCostTimer timer(kCostHandoff);
    const std::uint32_t vs_lo = sh[0x48], ps_lo = sh[0x08];
    const std::uint64_t flip = hle_video_flip_count();
    EmitRec e;
    const bool found = take_record(packet_va, e);
    if (g_native == 1 && found && e.gx && e.rec.native_id) {
        // What this draw's own packets changed since its token.
        const auto it = t_native_shadow.find(e.rec.native_id);
        if (it == t_native_shadow.end()) {
            g_native_unmatched.fetch_add(1, std::memory_order_relaxed);
        } else {
            const NativeShadow& s = it->second;
            const auto changed = [](NativeField field, bool c) {
                if (c) g_native_changed[field].fetch_add(1, std::memory_order_relaxed);
            };
            changed(kNfPsInput, std::memcmp(s.ps_input_cntl, &ctx[0x191], sizeof(s.ps_input_cntl)) != 0);
            changed(kNfRenderControl, s.render_control != ctx[0x000]);
            changed(kNfDepthClear, s.depth_clear != ctx[0x00b]);
            changed(kNfStencilClear, s.stencil_clear != ctx[0x00a]);
            changed(kNfGenericScissor, s.generic_scissor[0] != ctx[0x090] || s.generic_scissor[1] != ctx[0x091]);
            changed(kNfVportScissor, s.vport_scissor[0] != ctx[0x094] || s.vport_scissor[1] != ctx[0x095]);
            changed(kNfModeCntl, s.sc_mode_cntl_0 != ctx[0x292]);
            for (int k = 0; k < 2; ++k) {
                const std::uint32_t* now = k == 0 ? &sh[0x4c] : &sh[0x0c];
                // Slots the native draw builds itself: the descriptors' dwords,
                // and for the VS the fetch shader (user data 0) and the vertex table.
                std::uint32_t named = e.records ? named_user_dwords(e.records->stage[k]) : 0;
                if (k == 0) {
                    named |= 3u;
                    if (e.rec.vtx_ud < 15) named |= 3u << e.rec.vtx_ud;
                }
                bool named_changed = false, unnamed_changed = false;
                for (int j = 0; j < 16; ++j) {
                    if (s.user[k][j] == now[j]) continue;
                    if ((named >> j) & 1) {
                        named_changed = true;
                    } else {
                        unnamed_changed = true;
                        g_native_unnamed_slot[k][j].fetch_add(1, std::memory_order_relaxed);
                    }
                }
                changed(kNfUserNamed, named_changed);
                changed(kNfUserUnnamed, unnamed_changed);
            }
            t_native_shadow.erase(it);
            if (g_native_matched.fetch_add(1, std::memory_order_relaxed) % 100000 == 99999) report_native();
        }
    }
    if (g_native == 1 && found && e.gx && e.rec.native_id && e.rec.tess) {
        const auto it = t_native_tess.find(e.rec.native_id);
        if (it == t_native_tess.end()) {
            g_tess_unmatched.fetch_add(1, std::memory_order_relaxed);
        } else {
            const NativeTess& t = it->second;
            const auto changed = [](TessField f, bool c) {
                if (c) g_tess_changed[f].fetch_add(1, std::memory_order_relaxed);
            };
            changed(kTfStages, t.stages != ctx[0x2d5]);
            changed(kTfConfig, t.ls_hs_config != ctx[0x2d6]);
            changed(kTfParam, t.tf_param != ctx[0x2db]);
            changed(kTfHos, t.hos_max != ctx[0x286] || t.hos_min != ctx[0x287]);
            changed(kTfLsPgm, std::memcmp(t.ls_pgm, &sh[0x148], sizeof(t.ls_pgm)) != 0);
            changed(kTfHsPgm, std::memcmp(t.hs_pgm, &sh[0x108], sizeof(t.hs_pgm)) != 0);
            changed(kTfLsUser, std::memcmp(t.ls_user, &sh[0x14c], sizeof(t.ls_user)) != 0);
            changed(kTfHsUser, std::memcmp(t.hs_user, &sh[0x10c], sizeof(t.hs_user)) != 0);
            changed(kTfConstants, t.tess_cb[0] != sh[0x54] || t.tess_cb[1] != sh[0x55]);
            GpuDrawInputs shadow{};
            const bool have = build_inputs(e.rec, shadow, true) && shadow.tess;
            g_tess_have_inputs.fetch_add(have ? 1 : 0, std::memory_order_relaxed);
            g_tess_in_ok.fetch_add(e.rec.in_ok ? 1 : 0, std::memory_order_relaxed);
            if (have) {
                // The differential: what the GX objects gave - carried in the
                // record itself, not a thread-local map the command processor
                // never sees - against the registers the draw's own packets
                // wrote. The control-point counts are the only part of
                // VGT_LS_HS_CONFIG anything reads.
                const GpuDrawInputs& b = shadow;
                const auto bad = [](BuiltField f, bool c) {
                    g_tess_built[f].fetch_add(1, std::memory_order_relaxed);
                    if (c) g_tess_built_bad[f].fetch_add(1, std::memory_order_relaxed);
                };
                bad(kBfStages, b.stages != ctx[0x2d5]);
                bad(kBfConfig, (b.ls_hs_config & 0xfff00) != (ctx[0x2d6] & 0xfff00));
                bad(kBfTf, b.tf_param != ctx[0x2db]);
                bad(kBfHos, b.hos_max != ctx[0x286] || b.hos_min != ctx[0x287]);
                // RSRC2's LDS_SIZE [15:7] aside: the commit works it out from
                // the patches a group and the stride, and the register holds 7
                // where the shader's own structure holds 0. It is the one
                // field of the LS's that the host pipeline has no use for -
                // the control point travels as a stage attribute, so there is
                // no LDS - and the user SGPR count, which is what the
                // translator reads, is 6 in both.
                constexpr std::uint32_t kLdsSize = 0xff80u;
                bad(kBfLsPgm, b.ls_pgm[0] != sh[0x148] || b.ls_pgm[1] != sh[0x149] || b.ls_pgm[2] != sh[0x14a] ||
                                  (b.ls_pgm[3] & ~kLdsSize) != (sh[0x14b] & ~kLdsSize));
                bad(kBfHsPgm, std::memcmp(b.hs_pgm, &sh[0x108], sizeof(b.hs_pgm)) != 0);
                bad(kBfVsPgm, std::memcmp(b.vs_pgm, &sh[0x48], sizeof(b.vs_pgm)) != 0);
                // How many patches the draw is: GX's own first argument
                // against the count in the packet it wrote. A token draw takes
                // GX's, so a disagreement here is patches that would go
                // undrawn.
                bad(kBfCount, static_cast<std::uint32_t>(e.rec.arg[0]) != count);
            }
            t_native_tess.erase(it);
            g_tess_matched.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (in_window(flip)) {
        const bool gx = found && e.gx;
        const GxRec& r = e.rec;
        const bool indexed = r.kind == kDrawIndexed || r.kind == kDrawIndexedInstanced;
        int ib_slot = -2;  // not checked
        if (gx && indexed && r.ib) {
            std::uint64_t q[16];
            ib_slot = -1;
            if (safe_read(r.ib, q, sizeof(q))) {
                for (int i = 0; i < 16; ++i) {
                    if (q[i] == index_base) {
                        ib_slot = i;
                        break;
                    }
                    if (q[i] + r.ib_off == index_base) {
                        ib_slot = 100 + i;
                        break;
                    }
                }
            }
        }
        std::vector<std::tuple<int, int, char>> hits;
        bool scanned = false;
        if (gx && g_draw_scan_budget.fetch_sub(1) > 0) {
            scan_program(r, 0, 5, vs_lo, 'V', ps_lo, 'P', hits);
            scanned = true;
        }
        if (gx && r.kind <= kDrawIndexedInstanced && g_probe_budget.fetch_sub(1) > 0) probe_draw(r, sh, ctx);
        std::lock_guard<std::mutex> lk(g_mu);
        CpStats& p = g.cp[0];
        ++p.total;
        ++p.ops[op];
        if (!found) {
            ++p.none;
            ++p.unmatched[op];
            ++p.unmatched_prog[vs_lo];
        } else if (!gx) {
            ++p.other;
            ++p.other_callers[bn(e.ret)];
        } else {
            ++p.gx;
            ++g.cp_gx_kind[r.kind];
            if (r.kind <= kDrawIndexedInstanced) {
                const auto want_count = static_cast<std::uint32_t>(r.arg[0]);
                const std::uint32_t want_inst =
                    (r.kind == kDrawInstanced || r.kind == kDrawIndexedInstanced) ? static_cast<std::uint32_t>(r.arg[1]) : 1;
                (want_count == count ? p.arg_ok : p.arg_bad)++;
                (want_inst == instances ? g.inst_ok : g.inst_bad)++;
                if ((want_count != count || want_inst != instances) && g.bad_samples.size() < 8) {
                    std::string s;
                    appendf(s, "%s from 0x%llx: args %llu %llu %llu %llu; packet 0x%02x at 0x%llx count %u instances %u",
                            kKindName[r.kind], static_cast<ull>(bn(r.ret)), static_cast<ull>(r.arg[0]),
                            static_cast<ull>(r.arg[1]), static_cast<ull>(r.arg[2]), static_cast<ull>(r.arg[3]), op,
                            static_cast<ull>(packet_va), count, instances);
                    g.bad_samples.push_back(s);
                }
            }
            if (indexed || r.kind == kDrawIndexedIndirect) ++g.fmt_pairs[{r.ib_fmt, index_type & 1}];
            ++g.topo_pairs[{r.topo, prim}];
            if (ib_slot != -2) ++g.ib_slot[ib_slot];
            if (scanned) {
                ++p.scans;
                for (const auto& h : hits) ++g.stage_hits[h];
            }
        }
    }
    if (g_trace && flip >= g_last + 30) print_summary();
    if (g_backend && found && e.gx) objects_from_call(e.rec, std::move(e.records));
    if (!g_backend || !found || !e.gx || !e.rec.in_ok) return nullptr;
    // User data still comes from the register file (resources arrive through
    // the command-buffer tables). Rendering also takes the state compare mode
    // is still checking from it: the PS input table,
    // DB_RENDER_CONTROL with its clear values, the scissors and
    // PA_SC_MODE_CNTL_0.
    static thread_local GpuDrawInputs t_in;
    t_in = e.rec.in;
    std::memcpy(t_in.vs_user, &sh[0x4c], sizeof(t_in.vs_user));
    std::memcpy(t_in.ps_user, &sh[0x0c], sizeof(t_in.ps_user));
    if (g_backend == 1) {
        std::memcpy(t_in.ps_input_cntl, &ctx[0x191], sizeof(t_in.ps_input_cntl));
        t_in.ps_input_count = 32;
        t_in.render_control = ctx[0x000];
        t_in.depth_clear = ctx[0x00b];
        t_in.stencil_clear = ctx[0x00a];
        t_in.generic_scissor[0] = ctx[0x090];
        t_in.generic_scissor[1] = ctx[0x091];
        t_in.vport_scissor[0] = ctx[0x094];
        t_in.vport_scissor[1] = ctx[0x095];
        t_in.sc_mode_cntl_0 = ctx[0x292];
    }
    *render = g_backend == 1;
    return &t_in;
}

const GxDrawObjects* hle_gx_trace_objects() {
    if (t_native_objects) return t_native_objects;
    return t_objects_valid ? &t_objects : nullptr;
}

// A draw token some packets ahead of the one being drawn (the
// command processor's lookahead, gnm_exec.cpp). `next`: the very next draw,
// whose slot the previous lookahead asked for - its records now; otherwise
// the one after it - its slot. The lines then arrive while this draw is drawn.

void hle_gx_native_prefetch(std::uint64_t id, bool next) {
    NativeSlot& slot = native_slot(id);
    if (!next) {
        const auto* lines = reinterpret_cast<const char*>(&slot);
        for (std::size_t off = 0; off < sizeof(NativeSlot); off += 64) __builtin_prefetch(lines + off);
        return;
    }
    if (slot.state.load(std::memory_order_acquire) != 2 || slot.id != id) return;
    if (const GxDrawRecords* rec = slot.draw.records.get()) {
        for (const GxStageRecords& s : rec->stage) __builtin_prefetch(&s);  // the headers the bit masks below are read from
        prefetch_records(*rec);
    }
}

void hle_gx_native_release() {
    t_native_objects = nullptr;
    if (NativeSlot* slot = t_native_slot) {
        t_native_slot = nullptr;
        slot->draw.records.reset();  // back to the pool here, as before, not on a recording thread
        slot->state.store(0, std::memory_order_release);
        g_native_slot_pending.add(~0ull);
    } else {
        t_native_overflow.records.reset();
    }
}
void hle_gx_trace_dispatch(std::uint32_t op, std::uint64_t packet_va, std::uint32_t x, std::uint32_t cs_lo) {
    if (!g_active.load(std::memory_order_relaxed)) return;
    const std::uint64_t flip = hle_video_flip_count();
    EmitRec e;
    const bool found = take_record(packet_va, e);
    if (in_window(flip)) {
        const bool gx = found && e.gx;
        std::vector<std::tuple<int, int, char>> hits;
        bool scanned = false;
        if (gx && g_dispatch_scan_budget.fetch_sub(1) > 0) {
            scan_program(e.rec, 5, 5, cs_lo, 'C', 0, 'C', hits);
            scanned = true;
        }
        std::lock_guard<std::mutex> lk(g_mu);
        CpStats& p = g.cp[1];
        ++p.total;
        ++p.ops[op];
        if (!found) {
            ++p.none;
            ++p.unmatched[op];
            ++p.unmatched_prog[cs_lo];
        } else if (!gx) {
            ++p.other;
            ++p.other_callers[bn(e.ret)];
        } else {
            ++p.gx;
            ++g.cp_gx_kind[e.rec.kind];
            if (e.rec.kind == kDispatch) (static_cast<std::uint32_t>(e.rec.arg[0]) == x ? p.arg_ok : p.arg_bad)++;
            if (scanned) {
                ++p.scans;
                for (const auto& h : hits) ++g.stage_hits[h];
            }
        }
    }
    if (g_trace && flip >= g_last + 30) print_summary();
}
