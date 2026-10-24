// The GX resource registry; see the header.
//
// Hooks (hash-gated through the eboot check the caller makes, prologue hooks
// of engine/graphics_patch.h):
//   creators  0x2565c10 CreateBuffer, 0x2565f90 CreateTexture (1D/2D),
//             0x2566300 CreateTexture3D - (ctx, holder** out, desc*,
//             initial data*, ...). The holder exists only when the method
//             returns, so the hook swaps the return address for gx_res_ret,
//             which reads *out and registers it. CreateBuffer is on the
//             decomp list (decomp/gx/resources.cpp): ours tells the registry
//             itself (buffer_made_entry), and the hook goes on only when
//             ours is not in place (gx_resources_install_late).
//   Map       0x2569c30 (ctx, ?, holder, subresource, type, mapped*): a CPU
//             write is coming. Type 1/2 return the memory itself after a GPU
//             wait; for a buffer, 3 hands out a new renamed copy from the
//             context's allocator (0x2ad0d30) and 4 the copy the context
//             renamed it to last (0x2ad0ff0; error 4 when there is none) -
//             the return hook re-reads +0x20 all the same.
//   shaders   the four shader creators (hooked by gx_trace.cpp) and the
//             input-layout creator 0x25657a0 - (ctx, object** out, ...):
//             the object gets an id at the return.
//   views     0x25667a0 render target, 0x2566610 depth-stencil (ctx, view**
//             out, texture holder, description, ...): an id, and the
//             texture's extent at the view's mip. The view keeps the holder
//             at +0x08 (a reference); the description is D3D11's - a render
//             target's format / dimension / mip / first slice / slices at
//             +0x3c..+0x4c, a depth-stencil view's format / dimension / flags
//             / mip at +0x44..+0x50.
//   destruct  the deleting and the in-place destructors of a texture holder
//             (0x256d1a0 / 0x256d1e0, 3D 0x256d310 / 0x256d350; their
//             prologue starts with a rip-relative lea of the vtable, run in
//             the stub as a movabs), and a buffer's resource release
//             0x256d690 (rdi = holder + 0x18).
#include "engine/gx_resources.h"

#include "core/elf.h"
#include "core/portable.h"
#include "core/write_watch.h"
#include "decomp/decomp.h"
#include "decomp/gx/resources.h"
#include "decomp/gx/upload_heap.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "guest_abi.h"
#include "hle/hle.h"
#include "hle/modules.h"
#include "host/gpu.h"
#include "host/gxvk.h"
#include "log.h"

#if !defined(_WIN32)
#include <dlfcn.h>
#include <link.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

extern "C" void gx_res_ret();
extern "C" GUEST_ABI std::uint64_t gx_res_after(std::uint64_t result);

namespace {

constexpr std::uint64_t kCreateBuffer = 0x2565c10, kCreateTexture = 0x2565f90, kCreateTexture3D = 0x2566300;
constexpr std::uint64_t kMap = 0x2569c30;
constexpr std::uint64_t kCreateInputLayout = 0x25657a0;
constexpr std::uint64_t kCreateRtv = 0x25667a0, kCreateDsv = 0x2566610;
// Shader-resource views (the T# GX built at +0x10) and samplers
// (the S# at +0x0), (ctx, object** out, ...).
constexpr std::uint64_t kCreateSrv = 0x25669a0, kCreateSampler = 0x2565b40;
// BBHOST_GXVK_CENSUS (host/gxvk.h): the unordered-access view creator (ctx,
// view** out, holder, description, ...: the shader-resource view's prologue),
// the block pool's initialisation 0x2aa9b50(pool, allocator, base, bytes) -
// GX's upload heap, the device's one region (0x256b0f0) - and its aligned
// hand-out 0x2aa9fa0(pool, bytes, alignment), beside 0x2aa9d50.
constexpr std::uint64_t kCreateUav = 0x2566b60, kPoolInit = 0x2aa9b50, kPoolAligned = 0x2aa9fa0;
// The pool's other end. Each GX context reaches the pool through a memory
// interface: a deferred context's (vtable 0x575f350) hands out with
// 0x2aa9d50 and 0x2aa9fa0, first fit from the heap's start; the immediate
// context's (0x575f300) with 0x2aaa010(pool, bytes) and 0x2aaa270(pool,
// bytes, alignment), first fit from the free list's tail - the heap's top.
// Its Map discards rename there: the dynamic ring of its user-primitive draws
// takes a 256 KiB copy a frame. One free list serves both ends, so after a
// large load they meet.
constexpr std::uint64_t kPoolTopBlock = 0x2aaa010, kPoolTopAligned = 0x2aaa270;
// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; push rax
constexpr std::uint8_t kPoolInitPrologue[14] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50};
// push rbp; mov rbp, rsp; push r15, r14, r12, rbx; mov rbx, rdx
constexpr std::uint8_t kPoolAlignedPrologue[14] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x89, 0xd3};
// push rbp; mov rbp, rsp; push r15; push r14; push rbx; sub rsp, 0x48; mov rbx, rcx
constexpr std::uint8_t kSamplerPrologue[16] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                               0x53, 0x48, 0x83, 0xec, 0x48, 0x48, 0x89, 0xcb};
// push rbp; mov rbp, rsp; push r15; push r14; push r12; push rbx; sub rsp, 0x70
constexpr std::uint8_t kInputLayoutPrologue[15] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                   0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x70};
constexpr std::uint64_t kRing = 0x2adb1b0;  // the dynamic ring allocator: (ring, out*, ctx, bytes) -> *out
// Where the ring allocator's two Maps of its buffer return (no-overwrite at
// 0x2adb23f, discard at 0x2adb28b when it wraps or no copy is left): each
// allocation maps the ring's whole renamed copy, though the CPU writes only
// [*out, + bytes) of it.
constexpr std::uint64_t kRingMapReturn1 = 0x2adb244, kRingMapReturn2 = 0x2adb290;
// The upload pool's block source: (allocator, bytes = 0x10000) -> a block
// whose memory is at +8. Its one caller is the context's upload pool
// 0x2aaddc0 (pool, ctx, bytes), which carves allocations of up to 64 KiB out
// of these blocks for UpdateSubresource's dynamic copies (0x2aadb90) and some
// 200 sites of the draw-state code.
constexpr std::uint64_t kPoolBlock = 0x2aa9d50;
// GX's UpdateSubresource (0x2568a20) renames only a constant buffer (bind
// flags +0x3b & 4, which D3D11 lets no other flag join): 0x2aadb90 copies the
// bytes into a fresh copy from the upload pool (0x2aaddc0). Every other
// buffer is updated by a compute copy in command order - a GPU write. So the
// pool's memory takes constants from updates and draw tables from the
// commits, neither ever read as vertex data, and the buffer shadow hears only
// of the renamed copies of buffers a draw can read: Maps of buffers that are
// not constant buffers.
// push rbp; mov rbp, rsp; push r15; push r14; push rbx; push rax; mov r14, rsi; mov rbx, rdi
constexpr std::uint8_t kPoolBlockPrologue[16] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                 0x53, 0x50, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb};
constexpr std::uint64_t kBufferRelease = 0x256d690;
constexpr std::uint64_t kTexDelete = 0x256d1a0, kTexDestroy = 0x256d1e0, kTexVtable = 0x575f3e0;
constexpr std::uint64_t kTex3dDelete = 0x256d310, kTex3dDestroy = 0x256d350, kTex3dVtable = 0x575f400;
// The T# builder the 1D/2D creator runs (0x2571d60 -> 0x2fa6af0(size*, tsharp*,
// desc, subresource info, scratch)): its rsi is the T# GX gives the texture,
// before the memory is allocated, so the base is 0 until we fill it in.
constexpr std::uint64_t kTsharpBuild = 0x2fa6af0;
// 1D textures have no holder creator of their own: 0x2ad5e90 builds the
// holder inline around the 1D resource creator 0x2571790(resource* out, ctx,
// desc, initial data, ...), whose T# builder is 0x2fa64a0 (rsi = T# out).
// The resource struct is registered from the creator's return under a key
// made from its memory; the holder's destructors (0x256d030 / 0x256d070,
// vtable 0x575f3c0) find it by the memory at holder+0x20.
constexpr std::uint64_t kCreate1D = 0x2571790, kTsharpBuild1D = 0x2fa64a0;
constexpr std::uint64_t kTex1dDelete = 0x256d030, kTex1dDestroy = 0x256d070, kTex1dVtable = 0x575f3c0;
constexpr std::uint64_t kMemoryKey = 0x8000000000000000ull;
constexpr std::uint8_t kTsharpBuildPrologue[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                                   0x41, 0x54, 0x53, 0x48, 0x81, 0xe4, 0xe0, 0xff, 0xff, 0xff};
constexpr std::uint8_t kCreate1DPrologue[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                                0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xc8, 0x00, 0x00, 0x00};

#define BB_PUSH6 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53
constexpr std::uint8_t kPrologue58[17] = {BB_PUSH6, 0x48, 0x83, 0xec, 0x58};  // the three creators and Map
constexpr std::uint8_t kRingPrologue[17] = {BB_PUSH6, 0x48, 0x83, 0xec, 0x28};
// GX's UpdateSubresource of a buffer that is not a constant buffer starts its
// compute copy here (ctx, resource = holder + 0x18, data, offset, bytes); a
// constant buffer's update never reaches it (0x2568a20 renames those).
constexpr std::uint64_t kStaticBufferUpdate = 0x2ab3560;
constexpr std::uint8_t kStaticUpdatePrologue[17] = {BB_PUSH6, 0x48, 0x83, 0xec, 0x18};
constexpr std::uint8_t kDsvPrologue[17] = {BB_PUSH6, 0x48, 0x83, 0xec, 0x68};  // the render-target view creator's is kPrologue58
// YEBIS's own D3D11-shaped device and context (vtables 0x5708810 / 0x570a380),
// which keep its textures and buffers out of GX: CreateTexture2D (slot 5 ->
// 0x15df570(device, device, initial data, desc, 0, out)), CreateTexture1D /
// 3D (0x15dea90 / 0x15e1750, the same arguments), CreateBuffer (slot 3 ->
// 0x15ecb70(device, initial data, desc, out)), Map (0x15d92c0(ctx, resource,
// subresource, type, flags, mapped*)), UpdateSubresource (0x15d8d70(ctx,
// resource, subresource, box, data, ...)) and the 2D texture's deleting
// destructor (0x2e4b090, vtable 0x570bbf0). An object keeps its memory at
// +0x98, its kind at +0x80 (3 a 2D texture, written by the creator; 2D's
// size at +0x150 and its D3D11 description from +0x188). Census only
// (BBHOST_RESOURCE_CENSUS=1) for now.
constexpr std::uint64_t kYebisCreate2D = 0x15df570, kYebisCreate1D = 0x15dea90, kYebisCreate3D = 0x15e1750, kYebisCreateBuffer = 0x15ecb70;
constexpr std::uint64_t kYebisMap = 0x15d92c0, kYebisUpdate = 0x15d8d70, kYebisTex2DDelete = 0x2e4b090, kYebisTex2DVtable = 0x570bbf0;
constexpr std::uint8_t kYebisCreate2DPrologue[17] = {BB_PUSH6, 0x48, 0x83, 0xec, 0x18};      // sub rsp, 0x18
constexpr std::uint8_t kYebisCreate13Prologue[17] = {BB_PUSH6, 0x50, 0x4d, 0x89, 0xce};      // push rax; mov r14, r9
constexpr std::uint8_t kYebisCreateBufferPrologue[17] = {BB_PUSH6, 0x50, 0x49, 0x89, 0xce};  // push rax; mov r14, rcx
constexpr std::uint8_t kYebisMapPrologue[20] = {BB_PUSH6, 0x48, 0x81, 0xec, 0x98, 0x00, 0x00, 0x00};
constexpr std::uint8_t kYebisUpdatePrologue[20] = {BB_PUSH6, 0x48, 0x81, 0xec, 0x58, 0x01, 0x00, 0x00};
// Unmap (slot 15) is a thunk: mov rdi, [rdi + 0xaa0]; mov esi, 0x38; xor edx, edx; xor ecx, ecx; jmp
// - a cache-flush command; the writes after Map went to the memory itself.
constexpr std::uint64_t kYebisUnmap = 0x15d99c0;
constexpr std::uint8_t kYebisUnmapPrologue[16] = {0x48, 0x8b, 0xbf, 0xa0, 0x0a, 0x00, 0x00, 0xbe, 0x38, 0x00, 0x00, 0x00, 0x31, 0xd2, 0x31, 0xc9};
// Scaleform's PS4 Texture::Initialize (0x7a1d30, slot 2 of vtable 0x55bbf20;
// its prologue is kPrologue58's): true when it made the hardware texture, an
// array (+0x60) of +0x45 planes of 0x58 bytes - size (+0), T# (+4), memory
// (+0x50, from the HAL's memory manager); an image to start from is copied in
// before it returns. Its CPU writes after that all end in the texture
// manager's unmapTexture (0x7a3270, slot +0xd8 of vtable 0x55bc030:
// (manager, texture, ...)): Texture::Update (0x7a24e0, slot +0x90 - the glyph
// cache's) maps through mapTexture (+0xc8, 0x7a3140: pointers into the planes'
// own memory, from the T#) and unmaps there, as Texture::Unmap (0x7a24b0)
// does. ReleaseHWTextures (0x7a2160, slot +0x60) gives the planes' memory to
// the manager's deferred free (+0x108) and zeroes the planes.
constexpr std::uint64_t kScaleformTexInit = 0x7a1d30, kScaleformUnmap = 0x7a3270, kScaleformRelease = 0x7a2160;
// push rbp; mov rbp, rsp; push r14; push rbx; mov r14, rdi; mov rbx, [rsi + 0x50]
constexpr std::uint8_t kScaleformUnmapPrologue[14] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x49, 0x89, 0xfe, 0x48, 0x8b, 0x5e, 0x50};
// push rbp; mov rbp, rsp; push r15; push r14; push r12; push rbx; sub rsp, 0x60
constexpr std::uint8_t kScaleformReleasePrologue[15] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x60};
#undef BB_PUSH6
// push rbp; mov rbp, rsp; push r15; push r14; push rbx; push rax; mov r14, rdi; mov rdi, [r14]
constexpr std::uint8_t kBufferReleasePrologue[16] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                     0x53, 0x50, 0x49, 0x89, 0xfe, 0x49, 0x8b, 0x3e};
// push rbp; mov rbp, rsp; push rbx; push rax; mov rbx, rdi; lea rax, [rip + vtable]
constexpr std::uint8_t kDtorHead[9] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb};

enum Kind : std::uint8_t {
    kBuffer = 1, kTexture = 2, kTexture3D = 3, kMapCall = 4, kTsharp = 5, kResource1D = 6, kObject = 7, kPoolCall = 8,
    kYebisCreate = 9, kYebisMapCall = 10, kScaleformInit = 11, kYebisUpdateCall = 12, kScaleformUnmapCall = 13
};

struct Pending {
    std::uint64_t ret;
    Kind kind;
    std::uint64_t out;      // creators: holder**; Map: mapped* (the address the caller gets)
    std::uint64_t holder;   // Map: the holder; creators: the description
    std::uint64_t type;     // Map: the map type; creators: initial data pointer
    bool ring = false;
    std::uint64_t ring_bytes = 0;
    std::uint64_t extra = 0;  // YEBIS: what was created (1 buffer, 2 1D, 3 2D, 4 3D); its Map's resource
};
// Creators that returned an error (0 is success; 9 an allocation that failed):
// a render target that did not fit is the one live_resolution.cpp watches for.
std::atomic<std::uint64_t> g_create_failures{0};
thread_local std::vector<Pending> t_pending;
// The T# the builder produced on this thread, for the creator's return.
thread_local std::uint32_t t_tsharp[8];
thread_local bool t_tsharp_valid = false;
// BBHOST_GX_TEX_AHEAD=0: do not create textures ahead of their first bind.
const bool g_tex_ahead = [] {
    const char* e = std::getenv("BBHOST_GX_TEX_AHEAD");
    return !e || e[0] != '0';
}();
std::atomic<std::uint64_t> g_ahead_created{0}, g_ahead_bytes{0}, g_ahead_refused{0}, g_ahead_no_tsharp{0}, g_ahead_dropped{0},
    g_ahead_gone{0};
std::atomic<std::uint64_t> g_retire_queued{0};

struct Entry {
    GxResourceInfo info;
    std::uint64_t holder = 0;
    std::uint64_t created_flip = 0;
    std::uint32_t tsharp[8] = {};
    bool has_tsharp = false;
};

std::uint64_t g_slide = 0;
std::mutex g_mu;
std::unordered_map<std::uint64_t, Entry> g_by_holder;      // under g_mu
std::map<std::uint64_t, std::uint64_t> g_by_memory;        // memory base -> holder, live entries only
std::atomic<std::uint64_t> g_write_seq{1};
std::atomic<std::uint64_t> g_gen_next{1};  // GxResourceInfo::gen

// Counters.
std::atomic<std::uint64_t> g_created[5] = {}, g_created_with_data[5] = {}, g_created_bytes[5] = {};
std::atomic<std::uint64_t> g_destroyed{0}, g_destroyed_unknown{0}, g_replaced_holder{0}, g_memory_reused{0};
std::atomic<std::uint64_t> g_released_under_holder{0}, g_release_events{0}, g_protected{0}, g_protect_events{0};
std::atomic<std::uint64_t> g_maps[5][5] = {};  // [resource type][map type]
std::atomic<std::uint64_t> g_map_unknown{0}, g_map_moved{0};
// Where a buffer's Map that succeeded hands the CPU its writes, by map type:
// [0] the buffer's own memory, [1] another live GX buffer's, [2] a live GX
// texture's, [3] no live resource's; and the maps that failed.
std::atomic<std::uint64_t> g_map_where[5][4] = {}, g_map_failed[5] = {};
// The shadow's: Maps whose renamed copy it hears of, constant buffers' it does
// not, and copies freed by a later discard Map of their buffer in the same
// context (0x2ad0d30 frees the old block: never written back, its CPU writes
// long over) - settled then. The last copy each buffer has in each context,
// under g_mu, dropped with the holder.
std::atomic<std::uint64_t> g_maps_told{0}, g_maps_constant{0}, g_copies_freed{0};
// The ring allocator's Maps (BBHOST_SHADOW_RING_EXACT, on unless 0, with the
// shadow acting on hand-outs): not told as the ring's whole copy - each
// allocation is, when the allocator returns (ring_hook): ring Maps passed
// over, allocations told, of a constant-buffer ring not told. The Map's
// constant flag reaches the allocator's return through t_ring_map (the same
// thread, inside the allocator).
bool g_ring_exact = false;
std::atomic<std::uint64_t> g_ring_maps_passed{0}, g_ring_allocs_told{0}, g_ring_allocs_constant{0};
struct RingMap {
    bool valid = false, constant = false;
};
thread_local RingMap t_ring_map;
struct LastCopy {
    std::uint64_t ctx = 0, va = 0;
    std::uint32_t bytes = 0;
};
std::unordered_map<std::uint64_t, std::vector<LastCopy>> g_last_copy;
std::atomic<int> g_map_where_logged{0};
// BBHOST_RESOURCE_CENSUS=1: the ring allocator is hooked too, and what it
// hands out told to the buffer shadow's census (host_gpu_shadow_gx_handed).
bool g_census = false;
std::atomic<std::uint64_t> g_ring_calls{0}, g_ring_bytes{0}, g_pool_calls{0}, g_pool_bytes{0};
// The census of YEBIS's own textures and buffers: what its device made, by
// kind ([1] buffers, [2] 1D, [3] 2D, [4] 3D), and the live ones by memory
// (under g_mu) for the buffer shadow's and the texture cache's censuses.
struct YebisObject {
    std::uint64_t object = 0, bytes = 0;
    std::uint32_t kind = 0;  // YEBIS's kind (1-4); 8 a Scaleform texture plane
};
std::map<std::uint64_t, YebisObject> g_yebis_by_memory;     // under g_mu
std::unordered_map<std::uint64_t, std::uint64_t> g_yebis_by_object;  // object -> memory, under g_mu
std::atomic<std::uint64_t> g_yebis_made[5] = {}, g_yebis_made_bytes[5] = {}, g_yebis_made_data[5] = {}, g_yebis_failed[5] = {};
std::atomic<std::uint64_t> g_yebis_rt{0}, g_yebis_depth{0}, g_yebis_destroyed{0}, g_yebis_destroyed_unknown{0};
std::atomic<std::uint64_t> g_yebis_maps[5][6] = {}, g_yebis_map_own{0}, g_yebis_map_other{0}, g_yebis_updates[5] = {};
std::atomic<std::uint64_t> g_yebis_renamed[5] = {};  // Maps that moved the resource's memory, by kind
std::atomic<int> g_yebis_logged[5] = {}, g_yebis_map_logged{0};
// YEBIS's 2D textures are registered as resources from its creator and
// destructor (key kYebisKey | the object), and its Map, Unmap and
// UpdateSubresource of one tell the texture cache it was written. On unless
// BBHOST_RES_YEBIS=0: a 12-place tour's images on YEBIS's memory then had
// no identity, and the address rule retired ~145 targets and ~140
// shader-written textures a pass as newer objects took their memory; known,
// YEBIS's re-created targets keep their images (taken by the texture made
// over their memory without data, ~400 a pass) and a destroyed one's goes
// once idle.
bool g_res_yebis = true;
constexpr std::uint64_t kYebisKey = 0x4000000000000000ull;
std::atomic<std::uint64_t> g_yebis_registered{0}, g_yebis_unregistered{0}, g_yebis_told{0};
// Scaleform's textures (census): initialized, their planes and bytes, failed.
std::atomic<std::uint64_t> g_sf_inits{0}, g_sf_planes{0}, g_sf_bytes{0}, g_sf_failed{0};
std::atomic<int> g_sf_logged{0};
// Scaleform's texture planes are registered as 2D resources from Initialize
// (key kScaleformKey | texture << 2 | plane), told of the CPU's writes when
// the manager's unmapTexture returns, and dropped at ReleaseHWTextures - but
// for a plane on memory a live GX resource already holds (a texture Scaleform
// wraps), which stays that resource's. Almost every Scaleform texture wraps
// one of the game's (Initialize takes its T#: no memory of its own); the few
// it allocates - the glyph cache (1024x1024, 8-bit) among them - the write
// watch and the hashes followed before. On unless BBHOST_RES_SCALEFORM=0.
bool g_res_scaleform = true;
constexpr std::uint64_t kScaleformKey = 0x2000000000000000ull;
std::uint64_t scaleform_key(std::uint64_t texture, unsigned plane) { return kScaleformKey | (texture << 2) | plane; }
std::atomic<std::uint64_t> g_sf_registered{0}, g_sf_unregistered{0}, g_sf_told{0}, g_sf_wrapped{0};
std::atomic<std::uint64_t> g_lookups{0}, g_lookup_hits{0};
std::atomic<std::uint64_t> g_peak_live{0};
// New resources made over part of a live one's memory: [old type][new type].
std::atomic<std::uint64_t> g_overlapped[5][5] = {};
std::atomic<int> g_overlap_logged{0};

// Object -> id for shader and input-layout objects. Written at
// creation (rare: area loads) under g_obj_mu, read at every draw on the GX
// threads without a lock: a slot's id is stored before its object, and a
// re-created object's new id before the creator returns - before the game
// can hand the object to a draw.
constexpr std::uint32_t kObjSlots = 1u << 17, kObjProbes = 64;
struct ObjSlot {
    std::atomic<std::uint64_t> object{0};
    std::atomic<std::uint32_t> id{0};
    std::atomic<std::uint32_t> extent{0};  // views: gx_view_extent
    std::atomic<std::uint64_t> words{0};   // shader-resource views and samplers: their T#/S# hashed at creation
    std::atomic<std::uint64_t> gen{0};     // views: their resource's generation (gx_view_gen)
};
ObjSlot g_obj[kObjSlots];
std::mutex g_obj_mu;
// A view's resource generation by the view's id (gx_view_gen_by_id): what a
// draw's records name a shader-resource view by. Direct-mapped, written at the
// view's creation, read lock-free by the command processor.
constexpr std::uint32_t kIdGenSlots = 1u << 17;
struct IdGen {
    std::atomic<std::uint32_t> id{0};
    std::atomic<std::uint64_t> gen{0};
};
IdGen g_id_gen[kIdGenSlots];
std::atomic<std::uint32_t> g_obj_next{1};
// No counter on the lookup: six GX threads look up three objects a draw, and
// a shared one would be a contended line; the renderer counts draws without ids.
std::atomic<std::uint64_t> g_obj_created{0}, g_obj_recreated{0}, g_obj_full{0};
std::uint32_t obj_slot(std::uint64_t object) {
    return static_cast<std::uint32_t>(((object >> 3) * 0x9e3779b97f4a7c15ull) >> 47) & (kObjSlots - 1);
}
void register_object(std::uint64_t object, std::uint32_t extent = 0, std::uint64_t words = 0, std::uint64_t gen = 0) {
    if (!object) return;
    const std::uint32_t id = g_obj_next.fetch_add(1, std::memory_order_relaxed);
    g_obj_created.fetch_add(1, std::memory_order_relaxed);
    if (gen) {
        IdGen& ig = g_id_gen[id & (kIdGenSlots - 1)];
        ig.id.store(0, std::memory_order_relaxed);
        ig.gen.store(gen, std::memory_order_relaxed);
        ig.id.store(id, std::memory_order_release);
    }
    std::lock_guard<std::mutex> lk(g_obj_mu);
    const std::uint32_t h = obj_slot(object);
    for (std::uint32_t probe = 0; probe < kObjProbes; ++probe) {
        ObjSlot& s = g_obj[(h + probe) & (kObjSlots - 1)];
        const std::uint64_t o = s.object.load(std::memory_order_relaxed);
        if (o == object) {
            g_obj_recreated.fetch_add(1, std::memory_order_relaxed);
            s.extent.store(extent, std::memory_order_relaxed);
            s.words.store(words, std::memory_order_relaxed);
            s.gen.store(gen, std::memory_order_relaxed);
            s.id.store(id, std::memory_order_release);
            return;
        }
        if (!o) {
            s.id.store(id, std::memory_order_relaxed);
            s.extent.store(extent, std::memory_order_relaxed);
            s.words.store(words, std::memory_order_relaxed);
            s.gen.store(gen, std::memory_order_relaxed);
            s.object.store(object, std::memory_order_release);
            return;
        }
    }
    g_obj_full.fetch_add(1, std::memory_order_relaxed);  // its draws find no id and take the address path
}

bool g_trace = false;  // BBHOST_GX_MAP_TRACE
std::map<std::uint64_t, std::uint64_t> g_trace_callers;  // return address (Binary Ninja) -> calls, under g_mu
std::atomic<int> g_trace_logged{0};

template <typename T>
bool rd(std::uint64_t at, T* out) {
    return host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), out, sizeof(T));
}

std::uint64_t bn(std::uint64_t host_va) { return host_va - g_slide + kPreferredGuestSlide; }

bool old_holder_alive(std::uint64_t holder) { return g_by_holder.count(holder) != 0; }

// `key` names the entry (the holder, or kMemoryKey | memory for a 1D
// resource registered from its struct); `holder` is where the fields are
// read from as a holder lays them out (the struct at +0x18).
// BBHOST_GX_RATE=1: a line for each wall-clock second resources were made
// in, with how many and how many bytes - when a warp's or a walk's
// streaming settles. The second is the epoch's (the map probe's clock).
const bool g_rate_log = std::getenv("BBHOST_GX_RATE") != nullptr;
std::mutex g_rate_mu;
std::int64_t g_rate_sec = 0;
std::uint64_t g_rate_n = 0, g_rate_bytes = 0;
void note_rate(std::uint32_t bytes) {
    const std::int64_t sec =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lk(g_rate_mu);
    if (sec != g_rate_sec) {
        if (g_rate_n) {
            host_log("gx-resources: rate: %llu made (%llu KiB) in second %lld, flip %llu", static_cast<unsigned long long>(g_rate_n),
                     static_cast<unsigned long long>(g_rate_bytes >> 10), static_cast<long long>(g_rate_sec),
                     static_cast<unsigned long long>(hle_video_flip_count()));
        }
        g_rate_sec = sec;
        g_rate_n = 0;
        g_rate_bytes = 0;
    }
    ++g_rate_n;
    g_rate_bytes += bytes;
}

void register_entry(std::uint64_t key, Entry& e);

void register_holder(std::uint64_t key, std::uint64_t holder, Kind kind, bool initial_data) {
    if (!holder) return;
    Entry e;
    e.holder = key;
    e.created_flip = hle_video_flip_count();
    e.info.created_flip = e.created_flip;
    std::uint32_t id = 0;
    if (!rd(holder + 0x20, &e.info.memory) || !rd(holder + 0x28, &id) || !rd(holder + 0x2c, &e.info.bytes) ||
        !rd(holder + 0x38, &e.info.type) || !rd(holder + 0x39, &e.info.usage)) {
        return;
    }
    rd(holder + 0x3b, &e.info.bind);
    e.info.gen = g_gen_next.fetch_add(1, std::memory_order_relaxed);
    e.info.created_submit = hle_gnm_submitted_total();
    e.info.id = id & 0xffffff;
    e.info.initial_data = initial_data;
    e.info.write_seq = initial_data ? g_write_seq.fetch_add(1, std::memory_order_relaxed) + 1 : 0;
    const unsigned t = e.info.type < 5 ? e.info.type : 0;
    g_created[t].fetch_add(1, std::memory_order_relaxed);
    if (initial_data) g_created_with_data[t].fetch_add(1, std::memory_order_relaxed);
    g_created_bytes[t].fetch_add(e.info.bytes, std::memory_order_relaxed);
    if (g_rate_log) note_rate(e.info.bytes);
    (void)kind;
    // With window imports (BBHOST_IMPORT_WINDOWS=1) the memory of a buffer the
    // GPU reads in place soon is imported ahead, off the command processor;
    // gpu.cpp picks which by its usage and D3D11 bind flags (+0x3b). Nothing
    // without window imports.
    if (e.info.type == 1 && e.info.memory && e.info.bytes) {
        std::uint8_t bind = 0;
        rd(holder + 0x3b, &bind);
        host_gpu_import_prefetch(e.info.memory, e.info.bytes, e.info.usage, bind);
        // BBHOST_GX_TWINS=1: one the GPU writes gets a device-local twin at its first use.
        host_gpu_gx_buffer_made(e.info.memory, e.info.bytes, e.info.usage, bind, e.info.gen);
    }
    // GX on Vulkan (host/gxvk.h): a buffer the host keeps gets its entry and
    // its handle at +0x14; BBHOST_GXVK_CENSUS: the class word and canaries in
    // the holder's spare bytes. Not a 1D texture's, whose holder GX builds
    // inline.
    if (!(key & kMemoryKey)) {
        if (e.info.type == 1) {
            std::uint8_t cpu = 0;
            std::uint32_t misc = 0;
            rd(holder + 0x3a, &cpu);
            rd(holder + 0x3c, &misc);
            gxvk::buffer_made(holder, e.info.memory, e.info.bytes, e.info.usage, cpu, e.info.bind, misc, initial_data);
            gxvk::census_buffer_made(holder, initial_data);
        } else if (e.info.type == 3 || e.info.type == 4) {
            gxvk::census_texture_made(holder);
        }
    }
    if (key & kMemoryKey) key = kMemoryKey | e.info.memory;
    register_entry(key, e);
}

// The registry takes a resource under `key`: indexed by its memory, a live
// one there before it dead as far as the memory goes, the mirrors told.
void register_entry(std::uint64_t key, Entry& e) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (auto old = g_by_holder.find(key); old != g_by_holder.end()) {
        // The holder's address reused before its destructor was seen.
        g_replaced_holder.fetch_add(1, std::memory_order_relaxed);
        if (auto m = g_by_memory.find(old->second.info.memory); m != g_by_memory.end() && m->second == key) g_by_memory.erase(m);
        g_by_holder.erase(old);
    }
    if (e.info.memory && e.info.bytes) {
        // The census: live resources the new one's memory overlaps other than
        // at its own base (the same base is the reuse below), by the old one's
        // type - memory the game gave to something new without a destructor
        // or an unmap the registry saw.
        auto o = g_by_memory.lower_bound(e.info.memory);
        if (o != g_by_memory.begin()) --o;
        for (; o != g_by_memory.end() && o->first < e.info.memory + e.info.bytes; ++o) {
            if (o->first == e.info.memory || o->second == key) continue;
            const auto h = g_by_holder.find(o->second);
            if (h == g_by_holder.end()) continue;
            const GxResourceInfo& old = h->second.info;
            if (old.memory + old.bytes <= e.info.memory) continue;
            g_overlapped[old.type < 5 ? old.type : 0][e.info.type < 5 ? e.info.type : 0].fetch_add(1, std::memory_order_relaxed);
            // A buffer the host keeps a copy of ended with this (host/gxvk.h;
            // its lock is never held while the registry's is taken).
            if (old.type == 1) gxvk::buffer_replaced(o->second);
            if (g_overlap_logged.fetch_add(1, std::memory_order_relaxed) < 12) {
                host_log("gx-resources: resource %u (type %u) [0x%llx, +0x%x) made at flip %llu over live resource %u (type %u) [0x%llx, +0x%x) "
                         "made at flip %llu",
                         e.info.id, e.info.type, static_cast<unsigned long long>(e.info.memory), e.info.bytes,
                         static_cast<unsigned long long>(e.created_flip), old.id, old.type, static_cast<unsigned long long>(old.memory), old.bytes,
                         static_cast<unsigned long long>(old.created_flip));
            }
        }
    }
    if (e.info.memory) {
        if (auto m = g_by_memory.find(e.info.memory); m != g_by_memory.end() && m->second != key) {
            // The memory of a resource whose destructor was not seen, now a
            // new resource's: the old one is dead as far as the memory goes.
            if (g_memory_reused.fetch_add(1, std::memory_order_relaxed) < 8) {
                const auto old = g_by_holder.find(m->second);
                host_log("gx-resources: 0x%llx reused by resource %u (type %u) at flip %llu while resource %u (type %u, created at flip %llu, "
                         "holder 0x%llx) still held it",
                         static_cast<unsigned long long>(e.info.memory), e.info.id, e.info.type, static_cast<unsigned long long>(e.created_flip),
                         old != g_by_holder.end() ? old->second.info.id : 0u, old != g_by_holder.end() ? old->second.info.type : 0u,
                         static_cast<unsigned long long>(old != g_by_holder.end() ? old->second.created_flip : 0), static_cast<unsigned long long>(m->second));
            }
            if (const auto old = g_by_holder.find(m->second); old != g_by_holder.end() && old->second.info.type == 1) gxvk::buffer_replaced(m->second);
            if (old_holder_alive(m->second)) g_by_holder.erase(m->second);
        }
        g_by_memory[e.info.memory] = key;
    }
    g_by_holder.emplace(key, e);
    const std::uint64_t live = g_by_memory.size();
    if (live > g_peak_live.load(std::memory_order_relaxed)) g_peak_live.store(live, std::memory_order_relaxed);
    // A resource made: with data, its creator copied the data in before
    // returning, a CPU write the buffer shadow's mirrors are told of rather
    // than finding through the write watch; without, its memory is a
    // resource's from now on - not memory GX hands out any more, if it was -
    // and a mirror page over it decides again whether it needs watching
    // (host_gpu_shadow_gx_created takes a lock of its own, never the
    // registry's).
    if (e.info.memory && e.info.bytes) host_gpu_shadow_gx_created(e.info.memory, e.info.bytes);
}

// Textures made ahead of their first bind are made on a thread of their own,
// not the one that created them: that is the game's main loop while it
// streams, and making the image takes the render lock, which the command
// processor holds through every draw. A lock profile over a soak had the main
// loop wait 449 ms for it (1,952 times) and hold it 127 ms more, in the
// frames that stream. A texture a draw binds before its turn here is made by
// that draw, as it was before textures were made ahead; this one then finds it.
struct AheadJob {
    std::array<std::uint32_t, 8> tsharp;
    std::uint64_t memory = 0, bytes = 0;
    std::uint64_t gen = 0;  // the resource's generation: its image, not another's at the same memory
};
constexpr std::size_t kAheadQueue = 4096;
std::mutex g_ahead_mu;
std::condition_variable g_ahead_cv;
std::deque<AheadJob>* const g_ahead_jobs = new std::deque<AheadJob>;  // under g_ahead_mu; never destroyed

void ahead_thread() {
    host_thread_set_name("bb-tex-ahead");
    for (;;) {
        AheadJob job;
        {
            std::unique_lock<std::mutex> lk(g_ahead_mu);
            g_ahead_cv.wait(lk, [] { return !g_ahead_jobs->empty(); });
            job = g_ahead_jobs->front();
            g_ahead_jobs->pop_front();
        }
        // Destroyed while it waited: its memory may be another resource's by now.
        GxResourceInfo live;
        if (!gx_resource_at(job.memory, std::max<std::uint64_t>(job.bytes, 1), &live) || live.gen != job.gen) {
            g_ahead_gone.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (host_gpu_texture_create_ahead(job.tsharp.data(), job.gen)) {
            g_ahead_created.fetch_add(1, std::memory_order_relaxed);
            g_ahead_bytes.fetch_add(job.bytes, std::memory_order_relaxed);
        } else {
            g_ahead_refused.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// A 2D texture created with initial data: its image is made off this thread,
// from the T# GX built for it (the command processor asks the registry while
// it holds the render lock, so the job takes neither lock here).
void create_ahead(const Entry& e) {
    if (!g_tex_ahead || (e.info.type != 3 && e.info.type != 2) || !e.info.initial_data || !e.info.memory) return;
    if (!t_tsharp_valid) {
        g_ahead_no_tsharp.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    AheadJob job;
    std::memcpy(job.tsharp.data(), t_tsharp, sizeof(t_tsharp));
    job.tsharp[0] = static_cast<std::uint32_t>(e.info.memory >> 8);
    job.tsharp[1] = (job.tsharp[1] & ~0xffu) | static_cast<std::uint32_t>((e.info.memory >> 40) & 0xff);
    job.memory = e.info.memory;
    job.bytes = e.info.bytes;
    job.gen = e.info.gen;
    static std::once_flag started;
    std::call_once(started, [] { std::thread(ahead_thread).detach(); });
    {
        std::lock_guard<std::mutex> lk(g_ahead_mu);
        if (g_ahead_jobs->size() >= kAheadQueue) {
            g_ahead_dropped.fetch_add(1, std::memory_order_relaxed);  // made at its first bind instead
            return;
        }
        g_ahead_jobs->push_back(job);
    }
    g_ahead_cv.notify_one();
}

// A resource's memory is going: its surfaces retire once the command
// processor has retired everything submitted so far.
// The memory goes back to GX's heap and the next resource made there writes
// it at once (the internal creators 0x256d420 and 0x2571d60 copy a buffer's
// or a texture's initial data in): with the dead resource's pages still
// write-protected - by the texture cache until its surface retires behind the
// GPU, by the buffer shadow until its mirror pages are next checked - each of
// them faulted on the game's thread, the main loop's when it streams. The
// resource's whole pages are released now (writable, and counted as written,
// so any watch on them sees the change); the pages it shares with its
// neighbours keep their protection.
void release_pages(std::uint64_t memory, std::uint32_t bytes) {
    const std::uint64_t lo = (memory + 0xfff) & ~0xfffull, hi = (memory + bytes) & ~0xfffull;
    if (hi > lo) write_watch_release(reinterpret_cast<void*>(static_cast<std::uintptr_t>(lo)), hi - lo);
}

// `gen`: the resource that died (its images go, and only its), or 0 for a
// range whose resources the registry no longer tells apart (an unmap).
void retire_memory(std::uint64_t memory, std::uint32_t bytes, std::uint64_t gen) {
    if (!memory || !bytes) return;
    g_retire_queued.fetch_add(1, std::memory_order_relaxed);
    release_pages(memory, bytes);
    host_gpu_texture_retire(memory, bytes, hle_gnm_submitted_total(), gen);
}

void unregister_holder(std::uint64_t holder) {
    std::uint64_t memory = 0, buffer_memory = 0, gen = 0, buffer_gen = 0;
    std::uint32_t bytes = 0, buffer_bytes = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_by_holder.find(holder);
        if (it == g_by_holder.end()) {
            g_destroyed_unknown.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        g_destroyed.fetch_add(1, std::memory_order_relaxed);
        if (it->second.info.type != 1 && it->second.info.alive) {
            memory = it->second.info.memory;
            bytes = it->second.info.bytes;
            gen = it->second.info.gen;
        } else if (it->second.info.type == 1 && it->second.info.alive) {
            buffer_memory = it->second.info.memory;
            buffer_bytes = it->second.info.bytes;
            buffer_gen = it->second.info.gen;
        }
        if (auto m = g_by_memory.find(it->second.info.memory); m != g_by_memory.end() && m->second == holder) g_by_memory.erase(m);
        g_by_holder.erase(it);
        g_last_copy.erase(holder);
    }
    retire_memory(memory, bytes, gen);
    if (buffer_memory && buffer_bytes) {
        release_pages(buffer_memory, buffer_bytes);
        // The buffer's memory is no longer a buffer's: whatever writes it next
        // may not tell the mirrors, so their pages over it are copied again -
        // and decide again whether to be watched - before they are read.
        host_gpu_shadow_gx_write(buffer_memory, buffer_bytes);
        host_gpu_gx_buffer_gone(buffer_memory, buffer_gen);
    }
    if (buffer_memory) gxvk::buffer_gone(holder);
}

// A Map on the holder: the resource is about to be written by the CPU. Its
// memory may have been renamed (type 3/4): re-read +0x20 and re-index.
void note_map(std::uint64_t holder, std::uint64_t type, std::uint64_t mapped_out, std::uint32_t result, std::uint64_t ctx,
              bool ring_map = false) {
    std::uint64_t memory = 0;
    rd(holder + 0x20, &memory);
    std::uint64_t mapped = 0;
    if (result == 0 && mapped_out) rd(mapped_out, &mapped);
    std::uint64_t texture_memory = 0;
    std::uint32_t texture_bytes = 0;
    std::uint32_t buffer_bytes = 0, moved_bytes = 0;
    std::uint64_t moved_from = 0;
    bool constant = false;
    LastCopy freed;
    std::uint64_t own_memory = 0, own_bytes = 0, own_gen = 0;  // a Map handing out the buffer's own memory
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_by_holder.find(holder);
        if (it == g_by_holder.end()) {
            g_map_unknown.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Entry& e = it->second;
        const unsigned rt = e.info.type < 5 ? e.info.type : 0, mt = type < 5 ? static_cast<unsigned>(type) : 0;
        g_maps[rt][mt].fetch_add(1, std::memory_order_relaxed);
        e.info.write_seq = g_write_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        if (memory && memory != e.info.memory) {
            g_map_moved.fetch_add(1, std::memory_order_relaxed);
            if (e.info.type == 1) {
                moved_from = e.info.memory;
                moved_bytes = e.info.bytes;
            }
            if (auto m = g_by_memory.find(e.info.memory); m != g_by_memory.end() && m->second == holder) g_by_memory.erase(m);
            e.info.memory = memory;
            g_by_memory[memory] = holder;
        }
        if (e.info.type >= 2 && e.info.type <= 4) {
            texture_memory = e.info.memory;
            texture_bytes = e.info.bytes;
        }
        if (e.info.type == 1) {
            buffer_bytes = e.info.bytes;
            constant = e.info.bind == 0x4;
            // A discard (type 3) renames the buffer again in this context:
            // the copy it had there is freed, its writes over.
            if (result == 0 && mapped && !constant && type == 3) {
                std::vector<LastCopy>& v = g_last_copy[holder];
                LastCopy* slot = nullptr;
                for (LastCopy& c : v) {
                    if (c.ctx == ctx) slot = &c;
                }
                if (!slot) {
                    v.push_back(LastCopy{ctx, 0, 0});
                    slot = &v.back();
                }
                if (slot->va && slot->va != mapped) freed = *slot;
                *slot = LastCopy{ctx, mapped, e.info.bytes};
            }
            if (result != 0 || !mapped) {
                g_map_failed[mt].fetch_add(1, std::memory_order_relaxed);
            } else {
                unsigned where = 3;
                std::uint32_t other_bytes = 0;
                std::uint8_t other_bind = 0;
                if (mapped >= e.info.memory && mapped < e.info.memory + e.info.bytes) {
                    where = 0;
                    own_memory = e.info.memory;
                    own_bytes = e.info.bytes;
                    own_gen = e.info.gen;
                } else if (auto c = g_by_memory.upper_bound(mapped); c != g_by_memory.begin()) {
                    --c;
                    if (auto h = g_by_holder.find(c->second); h != g_by_holder.end() && h->second.info.alive &&
                                                              mapped < h->second.info.memory + h->second.info.bytes) {
                        where = h->second.info.type == 1 ? 1 : 2;
                        other_bytes = h->second.info.bytes;
                        other_bind = h->second.info.bind;
                    }
                }
                g_map_where[mt][where].fetch_add(1, std::memory_order_relaxed);
                if (where != 3 && g_map_where_logged.fetch_add(1, std::memory_order_relaxed) < 6) {
                    host_log("gx-resources: a map (type %u) of a %u-byte buffer at 0x%llx hands out 0x%llx, in %s (%u bytes, bind 0x%x)",
                             mt, e.info.bytes, static_cast<unsigned long long>(e.info.memory), static_cast<unsigned long long>(mapped),
                             where == 0 ? "its own memory" : where == 1 ? "another live buffer" : "a live texture", other_bytes, other_bind);
                }
            }
        }
    }
    // A buffer mapped: the CPU writes the range the caller got next - a
    // renamed copy, which GX copies back to the buffer's own memory on the
    // GPU when the rename resolves (the buffer shadow sees that copy as a GPU
    // write). The copy's memory is handed out again and again: the buffer
    // shadow is told so (it is written without a call to say when).
    if (buffer_bytes && mapped) {
        // A constant buffer's copy holds constants, never vertex data: the
        // shadow need not hear of it (D3D11 joins no other bind flag to it).
        if (ring_map && g_ring_exact) t_ring_map = RingMap{true, constant};
        if (constant) {
            g_maps_constant.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_maps_told.fetch_add(1, std::memory_order_relaxed);
            if (freed.va && freed.bytes) {
                g_copies_freed.fetch_add(1, std::memory_order_relaxed);
                host_gpu_shadow_gx_settled(freed.va, freed.bytes);
            }
            // The ring allocator's: its allocation is told at its return.
            if (ring_map && g_ring_exact) g_ring_maps_passed.fetch_add(1, std::memory_order_relaxed);
            else host_gpu_shadow_gx_handed(mapped, buffer_bytes, 0);
        }
    }
    // A discard moved the buffer: the memory it leaves is no buffer's now.
    if (moved_from && moved_bytes) host_gpu_shadow_gx_write(moved_from, moved_bytes);
    // The CPU writes the buffer in place: gx twins keep no device-local copy of it.
    if (own_memory) host_gpu_gx_buffer_mapped_own(own_memory, own_bytes, own_gen);
    // A texture mapped (none is, in any run so far): the CPU writes it next,
    // and its image is no longer what its memory holds - the texture cache,
    // which does not watch a GX resource's memory, is told (host textures).
    if (texture_memory && texture_bytes) host_gpu_texture_written(texture_memory, texture_bytes);
    if (buffer_bytes) gxvk::census_buffer_mapped(holder, static_cast<std::uint32_t>(type), mapped, buffer_bytes, result == 0 && mapped, own_memory != 0);
    // A buffer the CPU writes is not one the host keeps a copy of (host/gxvk.h).
    if (buffer_bytes) gxvk::buffer_written(holder, gxvk::kRetireMap);
}

// The creators: rsi = holder**, rcx = initial data. The holder exists at the return.
GUEST_ABI std::int64_t create_hook(std::uint64_t id, const std::uint64_t* saved) {
    t_pending.push_back({saved[6], static_cast<Kind>(id), saved[4], saved[3], saved[2]});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// GX on Vulkan (host/gxvk.h): a buffer's update by a GPU copy - a kept
// default buffer's entry is retired (a lock only for a holder with a handle).
GUEST_ABI std::int64_t static_update_hook(std::uint64_t, const std::uint64_t* saved) {
    if (saved[4]) gxvk::buffer_written(saved[4] - 0x18, gxvk::kRetireUpdate);
    return 0;
}

// GX's CreateBuffer as ours (decomp/gx/resources.cpp) tells the registry
// what it made or why it made nothing: what gx_res_after does at the hooked
// creator's return (the failure counted - the first eight logged with the
// description's first two words, as there - and a new holder registered).
GUEST_ABI void buffer_made_entry(std::uint64_t holder, std::uint64_t desc, std::uint64_t initial_data, std::uint64_t result) {
    if (static_cast<std::uint32_t>(result) != 0) {
        const std::uint64_t n = g_create_failures.fetch_add(1, std::memory_order_relaxed);
        std::uint32_t wh[2] = {0, 0};
        if (n < 8 && desc && rd(desc, &wh))
            host_log("gx-resources: a buffer of %ux%u was not made (error %u)", wh[0], wh[1], static_cast<std::uint32_t>(result));
        return;
    }
    if (holder) register_holder(holder, holder, kBuffer, initial_data != 0);
}

// A render-target or depth-stencil view just made: its 2D texture's extent at
// the view's mip (0 for anything else).
std::atomic<std::uint64_t> g_views[4] = {}, g_views_extent[2] = {};  // render target, depth-stencil, shader resource, sampler
std::atomic<int> g_views_logged{0};
std::uint32_t view_extent(std::uint64_t view, bool depth) {
    std::uint64_t holder = 0;
    std::uint8_t type = 0;
    std::uint16_t w = 0, h = 0;
    std::uint32_t format = 0, dim = 0, mip = 0;
    if (!rd(view + 0x08, &holder) || !holder || !rd(holder + 0x38, &type) || !rd(holder + 0x44, &w) || !rd(holder + 0x46, &h) ||
        !rd(view + (depth ? 0x44 : 0x3c), &format) || !rd(view + (depth ? 0x48 : 0x40), &dim) || !rd(view + (depth ? 0x50 : 0x44), &mip)) {
        return 0;
    }
    // D3D11's dimensions: render target 4 TEXTURE2D, 5 2D array; depth-stencil
    // 3 TEXTURE2D, 4 2D array.
    const bool two_d = type == 3 && (depth ? dim == 3 || dim == 4 : dim == 4 || dim == 5);
    const std::uint32_t mw = mip < 16 ? std::max<std::uint32_t>(1, w >> mip) : 0, mh = mip < 16 ? std::max<std::uint32_t>(1, h >> mip) : 0;
    if (g_views_logged.fetch_add(1, std::memory_order_relaxed) < 24) {
        std::uint32_t regs[6] = {};
        for (int k = 0; k < 6; ++k) rd(view + 0x10 + 4 * static_cast<std::uint64_t>(k), &regs[k]);
        host_log("gx-resources: %s view 0x%llx of texture 0x%llx (type %u, %ux%u): format %u, dimension %u, mip %u -> %ux%u; "
                 "registers %08x %08x %08x %08x %08x %08x",
                 depth ? "depth-stencil" : "render-target", static_cast<unsigned long long>(view), static_cast<unsigned long long>(holder), type,
                 w, h, format, dim, mip, mw, mh, regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
    }
    return two_d && mw && mh ? (mw << 16) | mh : 0;
}

// The input-layout creator: rsi = object**. (The shader creators come through
// gx_objects_watch_creator.)
GUEST_ABI std::int64_t object_create_hook(std::uint64_t, const std::uint64_t* saved) {
    t_pending.push_back({saved[6], kObject, saved[4], 0, 0});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// The view creators: rsi = view**; `id` 1 render target, 2 depth-stencil.
GUEST_ABI std::int64_t view_create_hook(std::uint64_t id, const std::uint64_t* saved) {
    t_pending.push_back({saved[6], kObject, saved[4], 0, id});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// Map: rdx = holder, r8 = type, r9 = mapped*.
GUEST_ABI std::int64_t map_hook(std::uint64_t, const std::uint64_t* saved) {
    Pending p{saved[6], kMapCall, saved[0], saved[3], saved[1]};
    p.extra = saved[5];  // the context
    t_pending.push_back(p);
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// The T# builder: rsi = the T# out.
GUEST_ABI std::int64_t tsharp_hook(std::uint64_t, const std::uint64_t* saved) {
    t_pending.push_back({saved[6], kTsharp, saved[4], 0, 0});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// The upload pool's block source: rdi = the pool, rsi = bytes, the block
// comes back in rax. The pool's aligned hand-out (0x2aa9fa0) has the same
// first two arguments, and so do the top end's two. `top`: 1 for those.
// Once the arena's proof has failed and no census counts, the hand-out runs
// as if not hooked.
GUEST_ABI std::int64_t pool_hook(std::uint64_t top, const std::uint64_t* saved) {
    if (!gxvk::arena_recording() && (top || !(g_census || gxvk::census()))) {
        gxvk::arena_unrecorded();
        return 0;
    }
    Pending p{saved[6], kPoolCall, 0, saved[5], saved[4]};
    p.extra = top;
    t_pending.push_back(p);
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// The block pool's initialisation (BBHOST_GXVK_CENSUS), the game's: rdi = the
// pool, rdx = its region, rcx = the region's bytes.
GUEST_ABI std::int64_t pool_init_hook(std::uint64_t, const std::uint64_t* saved) {
    gxvk::census_heap(saved[5], saved[3], saved[2]);
    return 0;
}

// The same, ours (decomp/gx/upload_heap.cpp): it tells us of the pool and its
// region - the census, and GX on Vulkan's arena (host/gxvk.h), which binds
// the heap in place - and keeps the game's region (0).
GUEST_ABI std::uint64_t upload_region_entry(std::uint64_t pool, std::uint64_t region, std::uint64_t bytes, std::uint64_t bytes_out) {
    (void)bytes_out;
    gxvk::arena_region(pool, region, bytes);
    gxvk::census_heap(pool, region, bytes);
    return 0;
}

// YEBIS's device's creators (census): the object comes back through `out`.
template <std::uint64_t What>
GUEST_ABI std::int64_t yebis_create_hook(std::uint64_t, const std::uint64_t* saved) {
    Pending p{saved[6], kYebisCreate, What == 1 ? saved[2] : saved[0], What == 1 ? saved[3] : saved[2], What == 1 ? saved[4] : saved[3]};
    p.extra = What;
    t_pending.push_back(p);
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}
std::uint32_t yebis_kind(std::uint64_t object) {
    std::uint32_t kind = 0;
    if (!object || !rd(object + 0x80, &kind) || kind > 4) return 0;
    return kind;
}
// Without the census only a 2D texture's Map, Unmap or update matters (the
// registry's): YEBIS maps its buffers ~3,000 times a second, its 2D textures
// never, and those return at once.
bool yebis_skip(std::uint64_t object) { return !g_census && yebis_kind(object) != 3; }
GUEST_ABI std::int64_t yebis_map_hook(std::uint64_t, const std::uint64_t* saved) {
    if (yebis_skip(saved[4])) return 0;
    // The resource's memory before the Map: a discard (type 4) renames it -
    // new memory from YEBIS's ring (ctx + 0x3458 + 0x120, 0x15e5950) at
    // +0x90 and +0x98, the T# (+0xc4) and its views' rebased.
    std::uint64_t before = 0;
    rd(saved[4] + 0x98, &before);
    Pending p{saved[6], kYebisMapCall, saved[0], before, saved[2]};
    p.extra = saved[4];
    t_pending.push_back(p);
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}
GUEST_ABI std::int64_t yebis_update_hook(std::uint64_t, const std::uint64_t* saved) {
    if (yebis_skip(saved[4])) return 0;
    Pending p{saved[6], kYebisUpdateCall, 0, 0, 0};
    p.extra = saved[4];
    t_pending.push_back(p);
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}
// A registered YEBIS texture was written by the CPU (BBHOST_RES_YEBIS): its
// write sequence moves and the texture cache uploads it again at its next use.
// After a Map its memory is read again: a discard renames it (YEBIS's Map
// hands out new memory from its ring and rebases the T#), and the registry
// follows, as it does a GX holder's.
void yebis_written(std::uint64_t object, bool after_map = false) {
    if (!g_res_yebis || !object) return;
    std::uint64_t memory = 0, now = 0;
    std::uint32_t bytes = 0;
    if (after_map) rd(object + 0x98, &now);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const std::uint64_t key = kYebisKey | object;
        auto it = g_by_holder.find(key);
        if (it == g_by_holder.end() || !it->second.info.alive) return;
        Entry& e = it->second;
        if (now && now != e.info.memory) {
            g_map_moved.fetch_add(1, std::memory_order_relaxed);
            if (auto m = g_by_memory.find(e.info.memory); m != g_by_memory.end() && m->second == key) g_by_memory.erase(m);
            e.info.memory = now;
            g_by_memory[now] = key;
        }
        memory = e.info.memory;
        bytes = e.info.bytes;
        e.info.write_seq = g_write_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    g_yebis_told.fetch_add(1, std::memory_order_relaxed);
    if (memory && bytes) host_gpu_texture_written(memory, bytes);
}
GUEST_ABI std::int64_t yebis_unmap_hook(std::uint64_t, const std::uint64_t* saved) {
    if (yebis_skip(saved[4])) return 0;
    yebis_written(saved[4]);
    return 0;
}
GUEST_ABI std::int64_t yebis_tex2d_dtor_hook(std::uint64_t, const std::uint64_t* saved) {
    // The destructor runs for far more 2D objects than the creator makes
    // (37,450 against 2,261 in a 12-place tour): only those it made count.
    bool registered = false;
    if (g_res_yebis) {
        std::lock_guard<std::mutex> lk(g_mu);
        registered = g_by_holder.count(kYebisKey | saved[5]) != 0;
    }
    if (registered) {
        unregister_holder(kYebisKey | saved[5]);
        g_yebis_unregistered.fetch_add(1, std::memory_order_relaxed);
    }
    if (!g_census) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_yebis_by_object.find(saved[5]);
    if (it == g_yebis_by_object.end()) {
        g_yebis_destroyed_unknown.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    g_yebis_destroyed.fetch_add(1, std::memory_order_relaxed);
    if (auto m = g_yebis_by_memory.find(it->second); m != g_yebis_by_memory.end() && m->second.object == saved[5]) g_yebis_by_memory.erase(m);
    g_yebis_by_object.erase(it);
    return 0;
}

GUEST_ABI std::int64_t scaleform_init_hook(std::uint64_t, const std::uint64_t* saved) {
    Pending p{saved[6], kScaleformInit, 0, saved[5], 0};
    t_pending.push_back(p);
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// A Scaleform texture plane as a resource of the registry's
// (BBHOST_RES_SCALEFORM): a 2D texture over [memory, +size) with its data in,
// a new generation - unless a live GX resource holds that memory already.
void register_scaleform_plane(std::uint64_t texture, unsigned plane, std::uint64_t memory, std::uint32_t size) {
    const std::uint64_t key = scaleform_key(texture, plane);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto m = g_by_memory.upper_bound(memory);
        if (m != g_by_memory.begin() && (--m)->second != key) {
            const auto h = g_by_holder.find(m->second);
            if (h != g_by_holder.end() && h->second.info.alive && h->second.info.memory + h->second.info.bytes >= memory + size) {
                g_sf_wrapped.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    }
    Entry e;
    e.holder = key;
    e.created_flip = hle_video_flip_count();
    e.info.created_flip = e.created_flip;
    e.info.memory = memory;
    e.info.bytes = size;
    e.info.type = 3;
    e.info.gen = g_gen_next.fetch_add(1, std::memory_order_relaxed);
    e.info.created_submit = hle_gnm_submitted_total();
    e.info.initial_data = true;
    e.info.write_seq = g_write_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    g_sf_registered.fetch_add(1, std::memory_order_relaxed);
    register_entry(e.holder, e);
}

// A Scaleform texture made its hardware planes: each one's memory and size.
void note_scaleform_texture(std::uint64_t texture) {
    std::uint8_t planes = 0;
    std::uint64_t array = 0;
    if (!rd(texture + 0x45, &planes) || !rd(texture + 0x60, &array) || !array || planes > 4) return;
    g_sf_inits.fetch_add(1, std::memory_order_relaxed);
    for (unsigned i = 0; i < planes; ++i) {
        const std::uint64_t rec = array + i * 0x58ull;
        std::uint32_t size = 0, t[8] = {};
        std::uint64_t memory = 0;
        if (!rd(rec, &size) || !rd(rec + 0x50, &memory)) continue;
        host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(rec + 4)), t, sizeof(t));
        g_sf_planes.fetch_add(1, std::memory_order_relaxed);
        g_sf_bytes.fetch_add(size, std::memory_order_relaxed);
        if (g_sf_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            // T# word 2: width - 1 in bits 0-13, height - 1 in 14-27; word 1: data format in bits 20-25.
            host_log("gx-resources: Scaleform texture 0x%llx plane %u: %ux%u, format %u, memory 0x%llx, %u bytes", static_cast<unsigned long long>(texture),
                     i, (t[2] & 0x3fff) + 1, ((t[2] >> 14) & 0x3fff) + 1, (t[1] >> 20) & 0x3f, static_cast<unsigned long long>(memory), size);
        }
        if (!memory || memory == ~0ull || !size) continue;
        if (g_res_scaleform) register_scaleform_plane(texture, i, memory, size);
        if (!g_census) continue;
        std::lock_guard<std::mutex> lk(g_mu);
        g_yebis_by_memory[memory] = YebisObject{texture, size, 8};
    }
}

// The planes of a texture the registry holds (BBHOST_RES_SCALEFORM), by key.
std::vector<std::uint64_t> scaleform_registered(std::uint64_t texture) {
    std::vector<std::uint64_t> keys;
    std::lock_guard<std::mutex> lk(g_mu);
    for (unsigned i = 0; i < 4; ++i) {
        if (g_by_holder.count(scaleform_key(texture, i))) keys.push_back(scaleform_key(texture, i));
    }
    return keys;
}

// The manager's unmapTexture returned: the CPU's writes into the texture's
// planes since its map are in (Update's rectangles, a Map's), and the texture
// cache uploads them again at their next use.
void scaleform_unmapped(std::uint64_t texture) {
    if (!texture) return;
    for (const std::uint64_t key : scaleform_registered(texture)) {
        std::uint64_t memory = 0;
        std::uint32_t bytes = 0;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            auto it = g_by_holder.find(key);
            if (it == g_by_holder.end() || !it->second.info.alive) continue;
            it->second.info.write_seq = g_write_seq.fetch_add(1, std::memory_order_relaxed) + 1;
            memory = it->second.info.memory;
            bytes = it->second.info.bytes;
        }
        g_sf_told.fetch_add(1, std::memory_order_relaxed);
        if (memory && bytes) host_gpu_texture_written(memory, bytes);
    }
}

GUEST_ABI std::int64_t scaleform_unmap_hook(std::uint64_t, const std::uint64_t* saved) {
    Pending p{saved[6], kScaleformUnmapCall, 0, saved[4], 0};
    t_pending.push_back(p);
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// ReleaseHWTextures: the planes' memory goes back to Scaleform's manager.
GUEST_ABI std::int64_t scaleform_release_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t texture = saved[5];
    if (!texture) return 0;
    if (g_res_scaleform) {
        for (const std::uint64_t key : scaleform_registered(texture)) {
            unregister_holder(key);
            g_sf_unregistered.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (!g_census) return 0;
    std::uint8_t planes = 0;
    std::uint64_t array = 0;
    if (!rd(texture + 0x45, &planes) || !rd(texture + 0x60, &array) || !array || planes > 4) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    for (unsigned i = 0; i < planes; ++i) {
        std::uint64_t memory = 0;
        if (!rd(array + i * 0x58ull + 0x50, &memory) || !memory) continue;
        if (auto m = g_yebis_by_memory.find(memory); m != g_yebis_by_memory.end() && m->second.object == texture && m->second.kind == 8)
            g_yebis_by_memory.erase(m);
    }
    return 0;
}

// A YEBIS 2D texture as a resource of the registry's (BBHOST_RES_YEBIS): its
// memory and size, a new generation, D3D11's usage and bind flags.
void register_yebis_2d(std::uint64_t object, bool initial_data) {
    Entry e;
    e.holder = kYebisKey | object;
    e.created_flip = hle_video_flip_count();
    e.info.created_flip = e.created_flip;
    std::uint32_t bytes = 0, d[11] = {};
    if (!rd(object + 0x98, &e.info.memory) || !rd(object + 0x150, &bytes) ||
        !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(object + 0x188)), d, sizeof(d)) || !e.info.memory || !bytes)
        return;
    e.info.bytes = bytes;
    e.info.type = 3;
    e.info.usage = static_cast<std::uint8_t>(d[7]);
    e.info.bind = static_cast<std::uint8_t>(d[8]);
    e.info.gen = g_gen_next.fetch_add(1, std::memory_order_relaxed);
    e.info.created_submit = hle_gnm_submitted_total();
    e.info.initial_data = initial_data;
    e.info.write_seq = initial_data ? g_write_seq.fetch_add(1, std::memory_order_relaxed) + 1 : 0;
    g_yebis_registered.fetch_add(1, std::memory_order_relaxed);
    register_entry(kYebisKey | object, e);
}

// What YEBIS's device made: its memory, size, kind and (2D) description.
void note_yebis_created(std::uint64_t object, std::uint32_t what, std::uint64_t desc, bool data) {
    std::uint64_t memory = 0;
    rd(object + 0x98, &memory);
    std::uint64_t bytes = 0;
    std::uint32_t d[11] = {};  // 2D: width height mips array format samples quality usage bind cpu misc
    if (what == 3) {
        std::uint32_t b = 0;
        rd(object + 0x150, &b);
        bytes = b;
        host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(object + 0x188)), d, sizeof(d));
        if (d[8] & 0x20) g_yebis_rt.fetch_add(1, std::memory_order_relaxed);
        if (d[8] & 0x40) g_yebis_depth.fetch_add(1, std::memory_order_relaxed);
    } else if (what == 1) {
        std::uint32_t b = 0;
        rd(desc, &b);  // D3D11_BUFFER_DESC.ByteWidth
        bytes = b;
        host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(desc)), d, 6 * sizeof(std::uint32_t));
    }
    g_yebis_made[what].fetch_add(1, std::memory_order_relaxed);
    g_yebis_made_bytes[what].fetch_add(bytes, std::memory_order_relaxed);
    if (data) g_yebis_made_data[what].fetch_add(1, std::memory_order_relaxed);
    if (g_yebis_logged[what].fetch_add(1, std::memory_order_relaxed) < (what == 3 ? 24 : 6)) {
        if (what == 3) {
            host_log("gx-resources: YEBIS made a %ux%u 2D texture (mips %u, array %u, format %u, samples %u, usage %u, bind 0x%x, cpu 0x%x, "
                     "misc 0x%x%s): object 0x%llx, memory 0x%llx, %llu bytes",
                     d[0], d[1], d[2], d[3], d[4], d[5], d[7], d[8], d[9], d[10], data ? ", with data" : "", static_cast<unsigned long long>(object),
                     static_cast<unsigned long long>(memory), static_cast<unsigned long long>(bytes));
        } else {
            std::uint64_t f[12] = {};  // the object's fields from +0x80, to place what the census does not know yet
            host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(object + 0x80)), f, sizeof(f));
            host_log("gx-resources: YEBIS made a %s (desc %08x %08x %08x %08x %08x %08x%s): object 0x%llx, memory 0x%llx; +0x80.. %llx %llx %llx "
                     "%llx %llx %llx %llx %llx %llx %llx %llx %llx",
                     what == 1 ? "buffer" : what == 2 ? "1D texture" : "3D texture", d[0], d[1], d[2], d[3], d[4], d[5], data ? ", with data" : "",
                     static_cast<unsigned long long>(object), static_cast<unsigned long long>(memory), static_cast<unsigned long long>(f[0]),
                     static_cast<unsigned long long>(f[1]), static_cast<unsigned long long>(f[2]), static_cast<unsigned long long>(f[3]),
                     static_cast<unsigned long long>(f[4]), static_cast<unsigned long long>(f[5]), static_cast<unsigned long long>(f[6]),
                     static_cast<unsigned long long>(f[7]), static_cast<unsigned long long>(f[8]), static_cast<unsigned long long>(f[9]),
                     static_cast<unsigned long long>(f[10]), static_cast<unsigned long long>(f[11]));
        }
    }
    if (!memory || !bytes) return;
    std::lock_guard<std::mutex> lk(g_mu);
    if (auto old = g_yebis_by_object.find(object); old != g_yebis_by_object.end()) g_yebis_by_memory.erase(old->second);
    g_yebis_by_memory[memory] = YebisObject{object, bytes, what};
    g_yebis_by_object[object] = memory;
}

// The ring allocator (trace and census only): rsi = out*, rcx = bytes.
GUEST_ABI std::int64_t ring_hook(std::uint64_t, const std::uint64_t* saved) {
    Pending p{saved[6], kMapCall, saved[4], 0, 0};
    p.ring = true;
    p.ring_bytes = saved[2];
    t_pending.push_back(p);
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// A texture holder's destructor: rdi = holder.
GUEST_ABI std::int64_t texture_dtor_hook(std::uint64_t, const std::uint64_t* saved) {
    gxvk::census_texture_gone(saved[5]);
    unregister_holder(saved[5]);
    return 0;
}

// A 1D texture holder's destructor: its entry is keyed by the memory.
GUEST_ABI std::int64_t texture1d_dtor_hook(std::uint64_t, const std::uint64_t* saved) {
    std::uint64_t memory = 0;
    if (rd(saved[5] + 0x20, &memory) && memory) unregister_holder(kMemoryKey | memory);
    return 0;
}

// The 1D resource creator: rdi = the resource struct out, rcx = initial data.
GUEST_ABI std::int64_t create1d_hook(std::uint64_t, const std::uint64_t* saved) {
    t_pending.push_back({saved[6], kResource1D, saved[5], 0, saved[2]});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// A buffer's resource release: rdi = holder + 0x18.
GUEST_ABI std::int64_t buffer_release_hook(std::uint64_t, const std::uint64_t* saved) {
    gxvk::census_buffer_gone(saved[5] - 0x18);
    unregister_holder(saved[5] - 0x18);
    return 0;
}

void trace_map(const Pending& p) {
    if (p.ring) {
        std::uint64_t ptr = 0;
        if (p.out) rd(p.out, &ptr);
        // The 128-byte constant-buffer allocations are thousands a frame, and
        // the menus' quads come before the world: the log starts at
        // BBHOST_GX_MAP_FROM_FLIP (default 3000) and takes 600 of the rest.
        static const std::uint64_t from_flip = [] {
            const char* e = std::getenv("BBHOST_GX_MAP_FROM_FLIP");
            return e && e[0] ? std::strtoull(e, nullptr, 10) : 3000ull;
        }();
        if (p.ring_bytes != 128 && hle_video_flip_count() >= from_flip) {
            static std::mutex mu;
            static std::map<std::uint64_t, int> per_caller;  // the first eight calls of each caller
            const std::uint64_t caller = bn(p.ret);
            int n;
            {
                std::lock_guard<std::mutex> lk(mu);
                n = per_caller[caller]++;
            }
            if (n < 8) {
                host_log("gx-ring: caller 0x%llx takes %llu bytes -> 0x%llx", static_cast<unsigned long long>(caller),
                         static_cast<unsigned long long>(p.ring_bytes), static_cast<unsigned long long>(ptr));
            }
        }
        return;
    }
    std::uint64_t data = 0;
    if (p.out) rd(p.out, &data);
    const std::uint64_t caller = bn(p.ret);
    bool first = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        first = g_trace_callers[caller]++ == 0;
    }
    if (first && g_trace_logged.fetch_add(1) < 64) {
        char who[160] = "";
#if !defined(_WIN32)
        if (p.ret > 0x100000000000ull) {  // a host caller: name it
            Dl_info info{};
            if (dladdr(reinterpret_cast<void*>(static_cast<std::uintptr_t>(p.ret)), &info) && info.dli_sname) {
                std::snprintf(who, sizeof(who), " (host %s+0x%llx)", info.dli_sname,
                              static_cast<unsigned long long>(p.ret - reinterpret_cast<std::uintptr_t>(info.dli_saddr)));
            } else {
                struct Find { std::uint64_t a; std::uint64_t base; const char* name; bool hit; } f{p.ret, 0, nullptr, false};
                dl_iterate_phdr(
                    [](dl_phdr_info* i, std::size_t, void* d) {
                        auto* f = static_cast<Find*>(d);
                        for (int k = 0; k < i->dlpi_phnum; ++k) {
                            const auto& ph = i->dlpi_phdr[k];
                            const std::uint64_t lo = i->dlpi_addr + ph.p_vaddr;
                            if (ph.p_type == PT_LOAD && f->a >= lo && f->a < lo + ph.p_memsz) {
                                *f = {f->a, i->dlpi_addr, i->dlpi_name, true};
                                return 1;
                            }
                        }
                        return 0;
                    },
                    &f);
                if (f.hit) std::snprintf(who, sizeof(who), " (host %s+0x%llx)", f.name && f.name[0] ? f.name : "bbhost",
                                         static_cast<unsigned long long>(p.ret - f.base));
                else std::snprintf(who, sizeof(who), " (host, generated code)");
            }
        }
#endif
        host_log("gx-map: caller 0x%llx%s maps resource 0x%llx type %llu -> 0x%llx", static_cast<unsigned long long>(caller), who,
                 static_cast<unsigned long long>(p.holder), static_cast<unsigned long long>(p.type),
                 static_cast<unsigned long long>(data));
    }
}

}  // namespace

extern "C" GUEST_ABI std::uint64_t gx_res_after(std::uint64_t result) {
    const Pending p = t_pending.back();
    t_pending.pop_back();
    if ((p.kind == kBuffer || p.kind == kTexture || p.kind == kTexture3D) && static_cast<std::uint32_t>(result) != 0) {
        const std::uint64_t n = g_create_failures.fetch_add(1, std::memory_order_relaxed);
        std::uint32_t wh[2] = {0, 0};
        if (n < 8 && p.holder && rd(p.holder, &wh))
            host_log("gx-resources: a %s of %ux%u was not made (error %u)", p.kind == kBuffer ? "buffer" : "texture", wh[0], wh[1],
                     static_cast<std::uint32_t>(result));
    }
    if (p.kind == kMapCall) {
        if (!p.ring) {
            const std::uint64_t caller = bn(p.ret);
            note_map(p.holder, p.type, p.out, static_cast<std::uint32_t>(result), p.extra,
                     caller == kRingMapReturn1 || caller == kRingMapReturn2);
        } else {
            std::uint64_t ptr = 0;
            const bool got = static_cast<std::uint32_t>(result) == 0 && p.out && rd(p.out, &ptr) && ptr && p.ring_bytes;
            if (g_ring_exact) {
                // What this allocation's Map said of its buffer (a constant
                // buffer's copy is never told); none seen: told, to be safe.
                const RingMap m = t_ring_map;
                t_ring_map = RingMap{};
                if (got && m.valid && m.constant) {
                    g_ring_allocs_constant.fetch_add(1, std::memory_order_relaxed);
                } else if (got) {
                    g_ring_allocs_told.fetch_add(1, std::memory_order_relaxed);
                    host_gpu_shadow_gx_handed(ptr, p.ring_bytes, 0);
                }
            }
            if (g_census && got) {
                g_ring_calls.fetch_add(1, std::memory_order_relaxed);
                g_ring_bytes.fetch_add(p.ring_bytes, std::memory_order_relaxed);
                host_gpu_shadow_gx_handed(ptr, p.ring_bytes, 2);
            }
            if (got) gxvk::census_ring(ptr, p.ring_bytes);
        }
        if (g_trace) trace_map(p);
    } else if (p.kind == kYebisCreate) {
        const std::uint32_t what = static_cast<std::uint32_t>(p.extra);
        std::uint64_t object = 0;
        if (static_cast<std::uint32_t>(result) == 0 && p.out && rd(p.out, &object) && object) {
            if (g_census) note_yebis_created(object, what, p.holder, p.type != 0);
            if (g_res_yebis && what == 3) register_yebis_2d(object, p.type != 0);
        } else {
            g_yebis_failed[what].fetch_add(1, std::memory_order_relaxed);
        }
    } else if (p.kind == kScaleformInit) {
        if (result & 0xff) note_scaleform_texture(p.holder);
        else g_sf_failed.fetch_add(1, std::memory_order_relaxed);
    } else if (p.kind == kScaleformUnmapCall) {
        scaleform_unmapped(p.holder);
    } else if (p.kind == kYebisUpdateCall) {
        g_yebis_updates[yebis_kind(p.extra)].fetch_add(1, std::memory_order_relaxed);
        yebis_written(p.extra);
    } else if (p.kind == kYebisMapCall) {
        if (static_cast<std::uint32_t>(result) == 0) yebis_written(p.extra, true);
        const std::uint32_t kind = yebis_kind(p.extra);
        g_yebis_maps[kind][p.type < 6 ? p.type : 0].fetch_add(1, std::memory_order_relaxed);
        std::uint64_t data = 0, memory = 0;
        if (static_cast<std::uint32_t>(result) == 0 && p.out && rd(p.out, &data) && rd(p.extra + 0x98, &memory)) {
            (data == memory ? g_yebis_map_own : g_yebis_map_other).fetch_add(1, std::memory_order_relaxed);
            if (memory != p.holder) {
                g_yebis_renamed[kind].fetch_add(1, std::memory_order_relaxed);
                // The census's: what YEBIS's ring hands out, as the shadow's hand-outs.
                std::uint32_t size = 0, pitch = 0;
                if (kind >= 2) rd(p.extra + 0x150, &size);
                else if (rd(p.out + 8, &pitch)) size = pitch;
                if (g_census) host_gpu_shadow_gx_handed(memory, size ? size : 0x10000, 3);
            }
            if (g_yebis_map_logged.fetch_add(1, std::memory_order_relaxed) < 12)
                host_log("gx-resources: YEBIS maps a kind-%u resource 0x%llx (type %llu): 0x%llx handed out, its memory 0x%llx (0x%llx before)", kind,
                         static_cast<unsigned long long>(p.extra), static_cast<unsigned long long>(p.type),
                         static_cast<unsigned long long>(data), static_cast<unsigned long long>(memory), static_cast<unsigned long long>(p.holder));
        }
    } else if (p.kind == kPoolCall) {
        std::uint64_t block = 0;
        if (g_census && !p.extra && result && p.type && rd(result + 8, &block) && block) {
            g_pool_calls.fetch_add(1, std::memory_order_relaxed);
            g_pool_bytes.fetch_add(p.type, std::memory_order_relaxed);
            host_gpu_shadow_gx_handed(block, p.type, 1);
        }
        if (result && p.type) {
            if (!p.extra) gxvk::census_handout(p.holder, result, p.type);
            gxvk::arena_handout(p.holder, result, p.extra != 0);
        }
    } else if (p.kind == kObject) {
        std::uint64_t object = 0;
        if (p.type == 5) {
            // An unordered-access view (BBHOST_GXVK_CENSUS's hook alone): its canaries.
            if (static_cast<std::uint32_t>(result) == 0 && p.out && rd(p.out, &object) && object) gxvk::census_view_made(object, 5);
        } else if (p.out && rd(p.out, &object) && object) {
            if (p.type == 3 && static_cast<std::uint32_t>(result) == 0) gxvk::census_view_made(object, 3);
            std::uint32_t extent = 0;
            std::uint64_t words = 0;
            if (p.type == 1 || p.type == 2) {  // a view
                const bool depth = p.type == 2;
                extent = view_extent(object, depth);
                g_views[depth].fetch_add(1, std::memory_order_relaxed);
                if (extent) g_views_extent[depth].fetch_add(1, std::memory_order_relaxed);
            } else if (p.type == 3 || p.type == 4) {  // a shader-resource view's T#, a sampler's S#
                std::uint8_t w[32];
                if (host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(object + (p.type == 3 ? 0x10 : 0))), w, 32)) {
                    words = gx_words_hash(w);
                }
                g_views[p.type - 1].fetch_add(1, std::memory_order_relaxed);
            }
            // A view's resource, by its holder: what the view's draws name.
            GxResourceInfo viewed;
            const std::uint64_t gen = p.type >= 1 && p.type <= 3 && gx_view_resource(object, &viewed) ? viewed.gen : 0;
            register_object(object, extent, words, gen);
        }
    } else if (p.kind == kTsharp) {
        t_tsharp_valid = p.out && rd(p.out, &t_tsharp);
    } else {
        // The holder (creators write it through their out pointer), or the 1D
        // resource struct itself, read as if it sat in a holder.
        std::uint64_t holder = 0, key = 0;
        if (p.kind == kResource1D) {
            std::uint8_t type = 0;
            if (p.out && rd(p.out + 0x20, &type) && type == 2) {
                holder = p.out - 0x18;
                key = kMemoryKey;
            }
        } else if (p.out && rd(p.out, &holder)) {
            key = holder;
        }
        if (holder) {
            register_holder(key, holder, p.kind, p.type != 0);
            if (key & kMemoryKey) {
                std::uint64_t memory = 0;
                rd(holder + 0x20, &memory);
                key = kMemoryKey | memory;
            }
            Entry e;
            bool found = false;
            {
                std::lock_guard<std::mutex> lk(g_mu);
                if (auto it = g_by_holder.find(key); it != g_by_holder.end()) {
                    if (t_tsharp_valid && (it->second.info.type == 2 || it->second.info.type == 3)) {
                        std::memcpy(it->second.tsharp, t_tsharp, sizeof(t_tsharp));
                        it->second.has_tsharp = true;
                    }
                    e = it->second;
                    found = true;
                }
            }
            if (found) create_ahead(e);
        }
        t_tsharp_valid = false;
    }
    return p.ret;
}

extern "C" void* g_gx_res_after_thunk = nullptr;
#if defined(_WIN32)
#define GR_TYPE_(n)
#else
#define GR_TYPE_(n) ".type " #n ", @function\n"
#endif
asm(".text\n.globl gx_res_ret\n" GR_TYPE_(gx_res_ret) "gx_res_ret:\n"
    "push %rax\n push %rdx\n mov %rax, %rdi\n call *g_gx_res_after_thunk(%rip)\n"
    "mov %rax, %r11\n pop %rdx\n pop %rax\n jmp *%r11\n");

std::uint64_t gx_resources_create_failures() { return g_create_failures.load(std::memory_order_relaxed); }

void gx_resources_install(ElfImage* image) {
    g_slide = image->mem.slide;
    g_trace = [] {
        const char* e = std::getenv("BBHOST_GX_MAP_TRACE");
        return e && e[0] == '1';
    }();
    if (const char* e = std::getenv("BBHOST_GX_RESOURCES"); e && e[0] == '0') {
        host_log("gx-resources: off (BBHOST_GX_RESOURCES=0)");
        return;
    }
    g_gx_res_after_thunk = hle_wrap_fn(reinterpret_cast<void*>(&gx_res_after));
    const auto at = [&](std::uint64_t bn_va) { return image->mem.slide + (bn_va - kPreferredGuestSlide); };
    int hooked = 0, refused = 0;
    const auto hook = [&](std::uint64_t va, const std::uint8_t* prologue, std::size_t n, void* host, std::uint64_t id) {
        if (engine_prologue_hook(image, at(va), prologue, n, host, id)) ++hooked;
        else {
            ++refused;
            host_log("gx-resources: 0x%llx is not the prologue expected; not hooked", static_cast<unsigned long long>(va));
        }
    };
    hook(kCreateTexture, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&create_hook), kTexture);
    if (gxvk::buffers_on())
        hook(kStaticBufferUpdate, kStaticUpdatePrologue, sizeof(kStaticUpdatePrologue), reinterpret_cast<void*>(&static_update_hook), 0);
    hook(kCreateTexture3D, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&create_hook), kTexture3D);
    hook(kMap, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&map_hook), 0);
    if (g_tex_ahead) hook(kTsharpBuild, kTsharpBuildPrologue, sizeof(kTsharpBuildPrologue), reinterpret_cast<void*>(&tsharp_hook), 0);
    hook(kBufferRelease, kBufferReleasePrologue, sizeof(kBufferReleasePrologue), reinterpret_cast<void*>(&buffer_release_hook), 0);
    // The texture destructors: the prologue's lea of the vtable is
    // rip-relative, so the stub runs a movabs of the slid vtable instead.
    const auto dtor_with = [&](std::uint64_t va, std::uint64_t vtable, void* handler) {
        std::uint8_t prologue[16], exec[19];
        std::memcpy(prologue, kDtorHead, sizeof(kDtorHead));
        prologue[9] = 0x48;
        prologue[10] = 0x8d;
        prologue[11] = 0x05;
        const std::uint32_t disp = static_cast<std::uint32_t>(vtable - (va + 16));
        std::memcpy(prologue + 12, &disp, 4);
        std::memcpy(exec, kDtorHead, sizeof(kDtorHead));
        exec[9] = 0x48;
        exec[10] = 0xb8;  // movabs rax, imm64
        const std::uint64_t slid = at(vtable);
        std::memcpy(exec + 11, &slid, 8);
        if (engine_prologue_hook_exec(image, at(va), prologue, sizeof(prologue), exec, sizeof(exec), handler, 0)) {
            ++hooked;
        } else {
            ++refused;
            host_log("gx-resources: 0x%llx is not the destructor prologue expected; not hooked", static_cast<unsigned long long>(va));
        }
    };
    const auto dtor = [&](std::uint64_t va, std::uint64_t vtable) { dtor_with(va, vtable, reinterpret_cast<void*>(&texture_dtor_hook)); };
    const auto dtor1d = [&](std::uint64_t va) { dtor_with(va, kTex1dVtable, reinterpret_cast<void*>(&texture1d_dtor_hook)); };
    dtor(kTexDelete, kTexVtable);
    dtor(kTexDestroy, kTexVtable);
    dtor(kTex3dDelete, kTex3dVtable);
    dtor(kTex3dDestroy, kTex3dVtable);
    hook(kCreate1D, kCreate1DPrologue, sizeof(kCreate1DPrologue), reinterpret_cast<void*>(&create1d_hook), kResource1D);
    if (g_tex_ahead) hook(kTsharpBuild1D, kTsharpBuildPrologue, sizeof(kTsharpBuildPrologue), reinterpret_cast<void*>(&tsharp_hook), 0);
    dtor1d(kTex1dDelete);
    dtor1d(kTex1dDestroy);
    g_census = [] {
        const char* e = std::getenv("BBHOST_RESOURCE_CENSUS");
        return e && e[0] == '1';
    }();
    g_ring_exact = host_gpu_shadow_hands_out() && [] {
        const char* e = std::getenv("BBHOST_SHADOW_RING_EXACT");
        return !(e && e[0] == '0');
    }();
    if (g_trace || g_census || g_ring_exact || gxvk::census()) hook(kRing, kRingPrologue, sizeof(kRingPrologue), reinterpret_cast<void*>(&ring_hook), 0);
    g_res_yebis = [] {
        const char* e = std::getenv("BBHOST_RES_YEBIS");
        return !(e && e[0] == '0');
    }();
    if (g_census || g_res_yebis) {
        if (g_res_yebis) hook(kYebisUnmap, kYebisUnmapPrologue, sizeof(kYebisUnmapPrologue), reinterpret_cast<void*>(&yebis_unmap_hook), 0);
        hook(kYebisCreate2D, kYebisCreate2DPrologue, sizeof(kYebisCreate2DPrologue), reinterpret_cast<void*>(&yebis_create_hook<3>), 0);
        if (g_census) {  // the registry takes YEBIS's 2D textures only
            hook(kYebisCreate1D, kYebisCreate13Prologue, sizeof(kYebisCreate13Prologue), reinterpret_cast<void*>(&yebis_create_hook<2>), 0);
            hook(kYebisCreate3D, kYebisCreate13Prologue, sizeof(kYebisCreate13Prologue), reinterpret_cast<void*>(&yebis_create_hook<4>), 0);
            hook(kYebisCreateBuffer, kYebisCreateBufferPrologue, sizeof(kYebisCreateBufferPrologue),
                 reinterpret_cast<void*>(&yebis_create_hook<1>), 0);
        }
        hook(kYebisMap, kYebisMapPrologue, sizeof(kYebisMapPrologue), reinterpret_cast<void*>(&yebis_map_hook), 0);
        hook(kYebisUpdate, kYebisUpdatePrologue, sizeof(kYebisUpdatePrologue), reinterpret_cast<void*>(&yebis_update_hook), 0);
        dtor_with(kYebisTex2DDelete, kYebisTex2DVtable, reinterpret_cast<void*>(&yebis_tex2d_dtor_hook));
    }
    g_res_scaleform = [] {
        const char* e = std::getenv("BBHOST_RES_SCALEFORM");
        return !(e && e[0] == '0');
    }();
    if (g_census || g_res_scaleform) {
        hook(kScaleformTexInit, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&scaleform_init_hook), 0);
        hook(kScaleformRelease, kScaleformReleasePrologue, sizeof(kScaleformReleasePrologue), reinterpret_cast<void*>(&scaleform_release_hook), 0);
        if (g_res_scaleform)
            hook(kScaleformUnmap, kScaleformUnmapPrologue, sizeof(kScaleformUnmapPrologue), reinterpret_cast<void*>(&scaleform_unmap_hook), 0);
    }
    // The block pool's hand-outs: for the census, and for GX on Vulkan's arena's
    // release proof (host/gxvk_arena.cpp).
    if (g_census || gxvk::census() || gxvk::arena_wanted())
        hook(kPoolBlock, kPoolBlockPrologue, sizeof(kPoolBlockPrologue), reinterpret_cast<void*>(&pool_hook), 0);
    if (gxvk::census() || gxvk::arena_wanted())
        hook(kPoolAligned, kPoolAlignedPrologue, sizeof(kPoolAlignedPrologue), reinterpret_cast<void*>(&pool_hook), 0);
    // Its top end: the arena's records must hold both ends' hand-outs, or the
    // memory one end reuses reads as the other end's stale record once they meet.
    if (gxvk::arena_wanted()) {
        hook(kPoolTopBlock, kPoolBlockPrologue, sizeof(kPoolBlockPrologue), reinterpret_cast<void*>(&pool_hook), 1);
        hook(kPoolTopAligned, kPoolAlignedPrologue, sizeof(kPoolAlignedPrologue), reinterpret_cast<void*>(&pool_hook), 1);
    }
    if (gxvk::census()) hook(kCreateUav, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&view_create_hook), 5);
    hook(kCreateInputLayout, kInputLayoutPrologue, sizeof(kInputLayoutPrologue), reinterpret_cast<void*>(&object_create_hook), 0);
    hook(kCreateRtv, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&view_create_hook), 1);
    hook(kCreateDsv, kDsvPrologue, sizeof(kDsvPrologue), reinterpret_cast<void*>(&view_create_hook), 2);
    hook(kCreateSrv, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&view_create_hook), 3);
    hook(kCreateSampler, kSamplerPrologue, sizeof(kSamplerPrologue), reinterpret_cast<void*>(&view_create_hook), 4);
    host_log("gx-resources: registry on: %d hooks (creators, Map, destructors)%s%s", hooked, refused ? ", some refused" : "",
             g_trace ? "; Map and ring callers logged" : "");
}

// After the decomp list is placed: GX's CreateBuffer is ours (it calls the
// registry's entry itself), the compare run's stand-in (which calls the
// entry's host function), or the game's - hooked at its prologue as the
// other creators are.
void gx_resources_install_late(ElfImage* image) {
    if (!g_gx_res_after_thunk) return;
    // The upload heap's pool, for the census and GX on Vulkan's arena: ours of
    // its init tells the host of it (the entry above); the game's is hooked at
    // its prologue for the census (the arena then has no heap: today's path).
    if (gxvk::census() || gxvk::arena_wanted() || gxvk::heap_snap_wanted()) {
        if (decomp_placed("gx_upload_pool_init") == 1) {
            decomp_gx::g_upload_region_entry = hle_wrap_fn(reinterpret_cast<void*>(&upload_region_entry));
        } else if (gxvk::census()) {
            const std::uint64_t at = image->mem.slide + (kPoolInit - kPreferredGuestSlide);
            if (!engine_prologue_hook(image, at, kPoolInitPrologue, sizeof(kPoolInitPrologue), reinterpret_cast<void*>(&pool_init_hook), 0))
                host_log("gx-resources: 0x%llx is not the prologue expected; not hooked", static_cast<unsigned long long>(kPoolInit));
        }
    }
    const int placed = decomp_placed("gx_create_buffer");
    if (placed) {
        decomp_gx::g_buffer_made_host = reinterpret_cast<void*>(&buffer_made_entry);
        decomp_gx::g_buffer_made_entry = hle_wrap_fn(reinterpret_cast<void*>(&buffer_made_entry));
        return;
    }
    const std::uint64_t at = image->mem.slide + (kCreateBuffer - kPreferredGuestSlide);
    if (!engine_prologue_hook(image, at, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&create_hook), kBuffer))
        host_log("gx-resources: 0x%llx is not the prologue expected; not hooked", static_cast<unsigned long long>(kCreateBuffer));
}

std::uint64_t gx_guest_to_bn(std::uint64_t va) { return g_slide ? bn(va) : va; }

void gx_objects_watch_creator(std::uint64_t* frame) {
    if (!g_gx_res_after_thunk) return;
    t_pending.push_back({frame[6], kObject, frame[4], 0, 0});
    frame[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
}

std::uint32_t gx_view_extent(std::uint64_t view) {
    if (!view) return 0;
    const std::uint32_t h = obj_slot(view);
    for (std::uint32_t probe = 0; probe < kObjProbes; ++probe) {
        const ObjSlot& s = g_obj[(h + probe) & (kObjSlots - 1)];
        const std::uint64_t o = s.object.load(std::memory_order_acquire);
        if (o == view) return s.extent.load(std::memory_order_relaxed);
        if (!o) break;
    }
    return 0;
}

std::uint32_t gx_view_extent_gen(std::uint64_t view, std::uint64_t* gen) {
    *gen = 0;
    if (!view) return 0;
    const std::uint32_t h = obj_slot(view);
    for (std::uint32_t probe = 0; probe < kObjProbes; ++probe) {
        const ObjSlot& s = g_obj[(h + probe) & (kObjSlots - 1)];
        const std::uint64_t o = s.object.load(std::memory_order_acquire);
        if (o == view) {
            *gen = s.gen.load(std::memory_order_relaxed);
            return s.extent.load(std::memory_order_relaxed);
        }
        if (!o) break;
    }
    return 0;
}

std::uint64_t gx_view_gen_by_id(std::uint32_t id) {
    if (!id) return 0;
    const IdGen& ig = g_id_gen[id & (kIdGenSlots - 1)];
    if (ig.id.load(std::memory_order_acquire) != id) return 0;
    const std::uint64_t gen = ig.gen.load(std::memory_order_relaxed);
    return ig.id.load(std::memory_order_acquire) == id ? gen : 0;
}

std::uint64_t gx_view_gen(std::uint64_t view) {
    if (!view) return 0;
    const std::uint32_t h = obj_slot(view);
    for (std::uint32_t probe = 0; probe < kObjProbes; ++probe) {
        const ObjSlot& s = g_obj[(h + probe) & (kObjSlots - 1)];
        const std::uint64_t o = s.object.load(std::memory_order_acquire);
        if (o == view) return s.gen.load(std::memory_order_relaxed);
        if (!o) break;
    }
    return 0;
}

std::uint64_t gx_words_hash(const std::uint8_t* w) {
    std::uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 32; i += 8) {
        std::uint64_t v;
        std::memcpy(&v, w + i, 8);
        h = (h ^ v) * 1099511628211ull;
    }
    return h | 1;
}

std::uint32_t gx_object_lookup(std::uint64_t object, std::uint64_t* words) {
    if (!object) return 0;
    const std::uint32_t h = obj_slot(object);
    for (std::uint32_t probe = 0; probe < kObjProbes; ++probe) {
        const ObjSlot& s = g_obj[(h + probe) & (kObjSlots - 1)];
        const std::uint64_t o = s.object.load(std::memory_order_acquire);
        if (o == object) {
            const std::uint32_t id = s.id.load(std::memory_order_acquire);
            if (words) *words = s.words.load(std::memory_order_relaxed);
            return id;
        }
        if (!o) break;
    }
    return 0;
}

std::uint32_t gx_object_id(std::uint64_t object) {
    if (!object) return 0;
    const std::uint32_t h = obj_slot(object);
    for (std::uint32_t probe = 0; probe < kObjProbes; ++probe) {
        const ObjSlot& s = g_obj[(h + probe) & (kObjSlots - 1)];
        const std::uint64_t o = s.object.load(std::memory_order_acquire);
        if (o == object) return s.id.load(std::memory_order_acquire);
        if (!o) break;
    }
    return 0;
}

bool gx_resource_tsharp(std::uint64_t memory, std::uint32_t out[8]) {
    std::lock_guard<std::mutex> lk(g_mu);
    const auto m = g_by_memory.find(memory);
    if (m == g_by_memory.end()) return false;
    const auto h = g_by_holder.find(m->second);
    if (h == g_by_holder.end() || !h->second.has_tsharp) return false;
    std::memcpy(out, h->second.tsharp, 32);
    out[0] = static_cast<std::uint32_t>(memory >> 8);
    out[1] = (out[1] & ~0xffu) | static_cast<std::uint32_t>((memory >> 40) & 0xff);
    return true;
}

void gx_resources_memory_protected(std::uint64_t va, std::size_t len, int prot) {
    if (!g_gx_res_after_thunk || !len) return;
    std::lock_guard<std::mutex> lk(g_mu);
    std::uint64_t n = 0;
    std::uint32_t first_id = 0;
    std::uint8_t first_type = 0;
    for (auto it = g_by_memory.lower_bound(va); it != g_by_memory.end() && it->first < va + len; ++it) {
        if (auto h = g_by_holder.find(it->second); h != g_by_holder.end() && !n) {
            first_id = h->second.info.id;
            first_type = h->second.info.type;
        }
        ++n;
    }
    if (!n) return;
    g_protected.fetch_add(n, std::memory_order_relaxed);
    if (g_protect_events.fetch_add(1, std::memory_order_relaxed) < 6) {
        host_log("gx-resources: mprotect of [0x%llx, +0x%zx) to %d at flip %llu covers %llu live resources (first: id %u type %u)",
                 static_cast<unsigned long long>(va), len, prot, static_cast<unsigned long long>(hle_video_flip_count()),
                 static_cast<unsigned long long>(n), first_id, first_type);
    }
}

void gx_resources_memory_released(std::uint64_t va, std::size_t len) {
    gxvk::heap_written(gxvk::kHeapGuardUnmap, va, len);  // GX's upload heap is mapped for the session (stage 1d.1)
    if (!g_gx_res_after_thunk || !len) return;
    std::uint64_t n = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_by_memory.empty()) return;
        for (auto it = g_by_memory.lower_bound(va); it != g_by_memory.end() && it->first < va + len;) {
            auto h = g_by_holder.find(it->second);
            if (h != g_by_holder.end()) h->second.info.alive = false;  // the holder may still be destroyed later
            it = g_by_memory.erase(it);
            ++n;
        }
    }
    retire_memory(va, static_cast<std::uint32_t>(std::min<std::size_t>(len, 0xffffffffu)), 0);
    if (n) gxvk::memory_released(va, len);
    // The mirrors keep pages by their physical memory, which this range no
    // longer names: they hear of it as a write, and a range their map of the
    // guest's mappings no longer has copies every told page again.
    if (n) host_gpu_shadow_gx_write(va, len);
    if (n) {
        g_released_under_holder.fetch_add(n, std::memory_order_relaxed);
        if (g_release_events.fetch_add(1, std::memory_order_relaxed) < 6) {
            host_log("gx-resources: unmap of [0x%llx, +0x%zx) at flip %llu released the memory of %llu live resources", static_cast<unsigned long long>(va),
                     len, static_cast<unsigned long long>(hle_video_flip_count()), static_cast<unsigned long long>(n));
        }
    }
}

bool gx_resource_at(std::uint64_t va, std::size_t bytes, GxResourceInfo* out) {
    g_lookups.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_by_memory.upper_bound(va);
    if (it == g_by_memory.begin()) return false;
    --it;
    auto h = g_by_holder.find(it->second);
    if (h == g_by_holder.end()) return false;
    const GxResourceInfo& r = h->second.info;
    if (va + bytes > r.memory + r.bytes) return false;
    g_lookup_hits.fetch_add(1, std::memory_order_relaxed);
    if (out) *out = r;
    return true;
}

std::uint64_t gx_buffer_end(std::uint64_t va) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_by_memory.upper_bound(va);
    if (it == g_by_memory.begin()) return 0;
    --it;
    const auto h = g_by_holder.find(it->second);
    if (h == g_by_holder.end() || !h->second.info.alive || h->second.info.type != 1) return 0;
    const GxResourceInfo& r = h->second.info;
    return va < r.memory + r.bytes ? r.memory + r.bytes : 0;
}

bool gx_buffers_cover(std::uint64_t va, std::uint64_t bytes) {
    if (!bytes) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_by_memory.upper_bound(va);
    if (it == g_by_memory.begin()) return false;
    --it;
    std::uint64_t covered = va;
    for (; it != g_by_memory.end() && it->first <= covered; ++it) {
        auto h = g_by_holder.find(it->second);
        if (h == g_by_holder.end() || !h->second.info.alive || h->second.info.type != 1) return false;
        const GxResourceInfo& r = h->second.info;
        if (r.memory + r.bytes > covered) covered = r.memory + r.bytes;
        if (covered >= va + bytes) return true;
    }
    return false;
}

bool gx_resource_by_holder(std::uint64_t holder, GxResourceInfo* out) {
    if (!holder) return false;
    std::uint64_t memory = 0;
    const bool have_memory = rd(holder + 0x20, &memory);
    std::lock_guard<std::mutex> lk(g_mu);
    auto h = g_by_holder.find(holder);
    if (h == g_by_holder.end() && have_memory && memory) h = g_by_holder.find(kMemoryKey | memory);
    if (h == g_by_holder.end() || !h->second.info.alive) return false;
    if (out) *out = h->second.info;
    return true;
}

bool gx_view_resource(std::uint64_t view, GxResourceInfo* out) {
    std::uint64_t holder = 0;
    return view && rd(view + 0x08, &holder) && gx_resource_by_holder(holder, out);
}

bool gx_constant_buffer_at(std::uint64_t memory) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto m = g_by_memory.find(memory);
    if (m == g_by_memory.end()) return false;
    auto h = g_by_holder.find(m->second);
    return h != g_by_holder.end() && h->second.info.alive && h->second.info.type == 1 && h->second.info.bind == 0x4;
}

int gx_middleware_at(std::uint64_t va, std::size_t bytes) {
    if (!bytes) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_yebis_by_memory.empty()) return 0;
    auto it = g_yebis_by_memory.upper_bound(va + bytes - 1);
    if (it == g_yebis_by_memory.begin()) return 0;
    --it;
    if (it->first + it->second.bytes <= va) return 0;
    return it->second.kind == 8 ? 2 : 1;
}

bool gx_resource_overlapping(std::uint64_t va, std::size_t bytes, GxResourceInfo* out) {
    if (!bytes) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    // Live resources do not overlap each other: the one before the range if
    // it reaches in, else the first that starts inside it.
    auto it = g_by_memory.lower_bound(va);
    const auto take = [&](std::map<std::uint64_t, std::uint64_t>::const_iterator m) {
        auto h = g_by_holder.find(m->second);
        if (h == g_by_holder.end()) return false;
        const GxResourceInfo& r = h->second.info;
        if (r.memory >= va + bytes || r.memory + r.bytes <= va) return false;
        if (out) *out = r;
        return true;
    };
    if (it != g_by_memory.begin() && take(std::prev(it))) return true;
    for (; it != g_by_memory.end() && it->first < va + bytes; ++it) {
        if (take(it)) return true;
    }
    return false;
}

std::uint64_t gx_write_seq_now() { return g_write_seq.load(std::memory_order_relaxed); }

std::uint64_t gx_write_seq_overlapping(std::uint64_t va, std::size_t bytes, bool* any_resource) {
    if (any_resource) *any_resource = false;
    if (!bytes) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    std::uint64_t seq = 0;
    auto it = g_by_memory.lower_bound(va);
    const auto take = [&](std::map<std::uint64_t, std::uint64_t>::const_iterator m) {
        auto h = g_by_holder.find(m->second);
        if (h == g_by_holder.end()) return;
        const GxResourceInfo& r = h->second.info;
        if (r.memory >= va + bytes || r.memory + r.bytes <= va) return;
        if (any_resource) *any_resource = true;
        seq = std::max(seq, r.write_seq);
    };
    if (it != g_by_memory.begin()) take(std::prev(it));
    for (; it != g_by_memory.end() && it->first < va + bytes; ++it) take(it);
    return seq;
}

bool gx_resource_newer_overlapping(std::uint64_t va, std::size_t bytes, std::uint64_t after_flip) {
    if (!bytes) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    // Live resources do not overlap each other: those that start inside the
    // range, and the one before it if it reaches in.
    auto it = g_by_memory.lower_bound(va);
    const auto newer = [&](std::map<std::uint64_t, std::uint64_t>::const_iterator m) {
        auto h = g_by_holder.find(m->second);
        if (h == g_by_holder.end()) return false;
        const GxResourceInfo& r = h->second.info;
        return r.alive && r.created_flip > after_flip && r.memory < va + bytes && r.memory + r.bytes > va;
    };
    if (it != g_by_memory.begin() && newer(std::prev(it))) return true;
    for (; it != g_by_memory.end() && it->first < va + bytes; ++it) {
        if (newer(it)) return true;
    }
    return false;
}

void gx_resources_report() {
    if (!g_gx_res_after_thunk) return;
    std::size_t live = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        live = g_by_memory.size();
    }
    using ull = unsigned long long;
    static const char* const kType[5] = {"other", "buffers", "1D textures", "2D textures", "3D textures"};
    std::string created;
    for (unsigned t = 0; t < 5; ++t) {
        if (!g_created[t].load()) continue;
        char buf[160];
        std::snprintf(buf, sizeof(buf), " %s %llu (%llu with initial data, %llu MiB)", kType[t], static_cast<ull>(g_created[t].load()),
                      static_cast<ull>(g_created_with_data[t].load()), static_cast<ull>(g_created_bytes[t].load() >> 20));
        created += buf;
    }
    host_log("gx-resources: created%s; destroyed %llu (%llu unknown, %llu holders reused before their destructor, %llu memories reused "
             "before their holder's); memory unmapped under a live holder %llu resources in %llu unmaps, re-protected %llu in %llu mprotects; "
             "live %zu, peak %llu",
             created.c_str(), static_cast<ull>(g_destroyed.load()), static_cast<ull>(g_destroyed_unknown.load()),
             static_cast<ull>(g_replaced_holder.load()), static_cast<ull>(g_memory_reused.load()), static_cast<ull>(g_released_under_holder.load()),
             static_cast<ull>(g_release_events.load()), static_cast<ull>(g_protected.load()), static_cast<ull>(g_protect_events.load()), live,
             static_cast<ull>(g_peak_live.load()));
    std::string maps;
    for (unsigned t = 0; t < 5; ++t) {
        ull n[5];
        ull sum = 0;
        for (unsigned m = 0; m < 5; ++m) sum += n[m] = g_maps[t][m].load();
        if (!sum) continue;
        char buf[200];
        std::snprintf(buf, sizeof(buf), " %s: read %llu, write %llu, no-overwrite %llu, discard %llu;", kType[t], n[1], n[2], n[3], n[4]);
        maps += buf;
    }
    host_log("gx-resources: Map by resource and type%s unknown holders %llu, memory moved by a map %llu; lookups %llu (%llu hits)",
             maps.empty() ? " none;" : maps.c_str(), static_cast<ull>(g_map_unknown.load()), static_cast<ull>(g_map_moved.load()),
             static_cast<ull>(g_lookups.load()), static_cast<ull>(g_lookup_hits.load()));
    {
        std::string where;
        for (unsigned m = 0; m < 5; ++m) {
            ull w[4], sum = g_map_failed[m].load();
            for (unsigned k = 0; k < 4; ++k) sum += w[k] = g_map_where[m][k].load();
            if (!sum) continue;
            char buf[200];
            std::snprintf(buf, sizeof(buf), " type %u: its own memory %llu, another buffer's %llu, a texture's %llu, no resource's %llu, failed %llu;",
                          m, w[0], w[1], w[2], w[3], static_cast<ull>(g_map_failed[m].load()));
            where += buf;
        }
        host_log("gx-resources: buffer Maps told to the shadow %llu, constant buffers' not %llu; a renamed copy freed by its buffer's next "
                 "discard Map in the same context, settled %llu",
                 static_cast<ull>(g_maps_told.load()), static_cast<ull>(g_maps_constant.load()), static_cast<ull>(g_copies_freed.load()));
        host_log("gx-resources: the ring allocator's Maps%s: passed over as the ring's whole copy %llu, its allocations told %llu, of a "
                 "constant-buffer ring not told %llu",
                 g_ring_exact ? "" : " (not told by allocation: BBHOST_SHADOW_RING_EXACT=0 or the shadow off)",
                 static_cast<ull>(g_ring_maps_passed.load()), static_cast<ull>(g_ring_allocs_told.load()),
                 static_cast<ull>(g_ring_allocs_constant.load()));
        host_log("gx-resources: where a buffer's Map hands the CPU its writes:%s ring allocations seen %llu (%llu MiB), upload pool "
                 "blocks %llu (%llu MiB)",
                 where.empty() ? " none;" : where.c_str(), static_cast<ull>(g_ring_calls.load()), static_cast<ull>(g_ring_bytes.load() >> 20),
                 static_cast<ull>(g_pool_calls.load()), static_cast<ull>(g_pool_bytes.load() >> 20));
        static const char* const kYebisKind[5] = {"other", "buffers", "1D textures", "2D textures", "3D textures"};
        std::string made, maps;
        for (unsigned k = 1; k < 5; ++k) {
            char buf[200];
            if (g_yebis_made[k].load() || g_yebis_failed[k].load()) {
                std::snprintf(buf, sizeof(buf), " %s %llu (%llu MiB, %llu with data, %llu failed);", kYebisKind[k], static_cast<ull>(g_yebis_made[k].load()),
                              static_cast<ull>(g_yebis_made_bytes[k].load() >> 20), static_cast<ull>(g_yebis_made_data[k].load()),
                              static_cast<ull>(g_yebis_failed[k].load()));
                made += buf;
            }
        }
        for (unsigned k = 0; k < 5; ++k) {
            ull n[6], sum = 0;
            for (unsigned t = 0; t < 6; ++t) sum += n[t] = g_yebis_maps[k][t].load();
            if (!sum && !g_yebis_updates[k].load()) continue;
            char buf[200];
            std::snprintf(buf, sizeof(buf), " %s: Map types 1-5 %llu/%llu/%llu/%llu/%llu (memory moved %llu), UpdateSubresource %llu;", kYebisKind[k], n[1],
                          n[2], n[3], n[4], n[5], static_cast<ull>(g_yebis_renamed[k].load()), static_cast<ull>(g_yebis_updates[k].load()));
            maps += buf;
        }
        std::size_t live = 0;
        std::uint64_t live_bytes = 0;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            live = g_yebis_by_memory.size();
            for (const auto& [m, o] : g_yebis_by_memory) live_bytes += o.bytes;
        }
        if (g_res_yebis)
            host_log("gx-resources: YEBIS's 2D textures as resources (BBHOST_RES_YEBIS): registered %llu, destroyed %llu, told of CPU writes %llu",
                     static_cast<ull>(g_yebis_registered.load()), static_cast<ull>(g_yebis_unregistered.load()), static_cast<ull>(g_yebis_told.load()));
        host_log("gx-resources: Scaleform's textures initialized %llu (%llu planes, %llu MiB), failed %llu", static_cast<ull>(g_sf_inits.load()),
                 static_cast<ull>(g_sf_planes.load()), static_cast<ull>(g_sf_bytes.load() >> 20), static_cast<ull>(g_sf_failed.load()));
        if (g_res_scaleform)
            host_log("gx-resources: Scaleform's texture planes as resources (BBHOST_RES_SCALEFORM): registered %llu (%llu on a GX resource's memory, "
                     "left to it), released %llu, told of CPU writes at an unmap %llu",
                     static_cast<ull>(g_sf_registered.load()), static_cast<ull>(g_sf_wrapped.load()), static_cast<ull>(g_sf_unregistered.load()),
                     static_cast<ull>(g_sf_told.load()));
        host_log("gx-resources: YEBIS's own device made%s 2D render targets %llu, depth %llu; 2D destroyed %llu (%llu it had not seen made); live "
                 "YEBIS and Scaleform objects %zu (%llu MiB); by resource:%s Map's pointer its memory %llu, elsewhere %llu",
                 made.empty() ? " nothing;" : made.c_str(), static_cast<ull>(g_yebis_rt.load()), static_cast<ull>(g_yebis_depth.load()),
                 static_cast<ull>(g_yebis_destroyed.load()), static_cast<ull>(g_yebis_destroyed_unknown.load()), live,
                 static_cast<ull>(live_bytes >> 20), maps.empty() ? " none;" : maps.c_str(), static_cast<ull>(g_yebis_map_own.load()),
                 static_cast<ull>(g_yebis_map_other.load()));
    }
    {
        std::string ov;
        for (unsigned a = 0; a < 5; ++a) {
            for (unsigned b = 0; b < 5; ++b) {
                if (const ull n = g_overlapped[a][b].load()) {
                    char buf[96];
                    std::snprintf(buf, sizeof(buf), " %s over %s %llu;", kType[b], kType[a], n);
                    ov += buf;
                }
            }
        }
        host_log("gx-resources: made over part of a live resource's memory:%s", ov.empty() ? " none" : ov.c_str());
    }
    host_log("gx-resources: shader and input-layout objects created %llu (%llu at an address an earlier one had), not registered (table "
             "full) %llu",
             static_cast<ull>(g_obj_created.load()), static_cast<ull>(g_obj_recreated.load()), static_cast<ull>(g_obj_full.load()));
    host_log("gx-resources: views made: render target %llu (%llu of a 2D texture with its extent), depth-stencil %llu (%llu), shader "
             "resource %llu; samplers %llu",
             static_cast<ull>(g_views[0].load()), static_cast<ull>(g_views_extent[0].load()), static_cast<ull>(g_views[1].load()),
             static_cast<ull>(g_views_extent[1].load()), static_cast<ull>(g_views[2].load()), static_cast<ull>(g_views[3].load()));
    host_log("gx-resources: textures created ahead of their first bind %llu (%llu MiB), refused by the cache %llu, no T# seen %llu, "
             "destroyed before their turn %llu, queue full %llu; retirements queued %llu%s",
             static_cast<ull>(g_ahead_created.load()), static_cast<ull>(g_ahead_bytes.load() >> 20), static_cast<ull>(g_ahead_refused.load()),
             static_cast<ull>(g_ahead_no_tsharp.load()), static_cast<ull>(g_ahead_gone.load()), static_cast<ull>(g_ahead_dropped.load()),
             static_cast<ull>(g_retire_queued.load()), g_tex_ahead ? "" : " (BBHOST_GX_TEX_AHEAD=0)");
    if (g_trace) {
        std::lock_guard<std::mutex> lk(g_mu);
        host_log("gx-map: %zu distinct callers", g_trace_callers.size());
        for (const auto& [caller, n] : g_trace_callers) {
            host_log("gx-map:   0x%llx x%llu", static_cast<ull>(caller), static_cast<ull>(n));
        }
    }
    gxvk::census_report();
    gxvk::buffers_report();
    gxvk::arena_report();
    gxvk::heap_snap_report();
}
