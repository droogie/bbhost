// The GX resource registry: what GX creates, maps and
// destroys, as the host's record of resource identity.
//
// GX's resource holders are 0x50-byte objects (0x58 for a buffer): a vtable,
// the object allocator (+0x08), a reference count (+0x10, which views take,
// 0x25669a0 etc.), then the resource itself at +0x18: its memory allocator,
// the memory (+0x20), a 24-bit id from one counter (+0x28, 0x2abaef0), the
// byte size (+0x2c), type (+0x38: 1 buffer, 2/3/4 textures), usage (+0x39),
// CPU access (+0x3a), bind flags (+0x3b), misc flags (+0x3c), and for a
// texture its width/height/array/mips/format (+0x44/+0x46/+0x48/+0x4a/+0x4b).
// The creators are 0x2565c10 (buffer), 0x2565f90 (1D/2D texture) and
// 0x2566300 (3D); the reference count falling to 0 runs the deleting
// destructor (0x256d1a0 / 0x256d310, memory freed through the allocator's
// slot +0x70; a buffer's 0x256ce10 through 0x256d690), and a resource made
// over the same memory later gets a new id: the id is the identity, the
// address a lookup key.
//
// The registry is keyed by holder and indexed by memory. The texture cache
// asks it (gx_resource_at) what covers a surface's memory and whether GX
// wrote it since the cache last uploaded (the write sequence: creation with
// initial data and every Map), which is the census that decides whether the
// content hashes can go for covered resources.
//
// BBHOST_GX_MAP_TRACE=1 (reconnaissance) still logs Map's callers
// and the ring allocator's from here.
#pragma once

#include <cstddef>
#include <cstdint>

struct ElfImage;

struct GxResourceInfo {
    std::uint32_t id = 0;
    std::uint8_t type = 0;  // 1 buffer, 2 1D, 3 2D, 4 3D
    std::uint8_t usage = 0;
    std::uint64_t memory = 0;
    std::uint32_t bytes = 0;
    // The registry's write sequence when GX last wrote this resource
    // (creation with initial data, Map), and whether it did at creation.
    std::uint64_t write_seq = 0;
    bool initial_data = false;
    bool alive = true;
    std::uint64_t created_flip = 0;
};
// A guest address as Binary Ninja shows it (the preferred slide).
std::uint64_t gx_guest_to_bn(std::uint64_t va);

void gx_resources_install(ElfImage* image);
// How many buffer and texture creations have returned an error so far (an
// allocation that did not fit: engine/live_resolution.cpp checks it around a
// change). 0 when the registry is off.
std::uint64_t gx_resources_create_failures();
// The live resource whose memory covers [va, va + bytes), if any.
bool gx_resource_at(std::uint64_t va, std::size_t bytes, GxResourceInfo* out);
// Whether a live resource made after flip `after_flip` holds memory in
// [va, va + bytes): the game gave that memory to something newer.
bool gx_resource_newer_overlapping(std::uint64_t va, std::size_t bytes, std::uint64_t after_flip);
// The kernel unmapped [va, va + len): resources whose memory lay there are
// dead for the registry's purposes, holder or no holder (the game releases
// an area's texture pool under holders it keeps).
void gx_resources_memory_released(std::uint64_t va, std::size_t len);
// The guest changed the protection of [va, va + len): counted for the
// resources it covers (a forgotten watch reads as written, which is not a
// write).
void gx_resources_memory_protected(std::uint64_t va, std::size_t len, int prot);
// The T# GX built for the live 2D texture whose memory begins at `memory`,
// base filled in; false when the registry has none.
bool gx_resource_tsharp(std::uint64_t memory, std::uint32_t out[8]);
// The exit report.
void gx_resources_report();

// Shader and input-layout objects. GX's shader creators
// (0x2566d00, 0x2566f20, 0x25672d0, 0x25673e0 - hooked in gx_trace.cpp, which
// hands its frame here) and its input-layout creator (0x25657a0) write the
// new object through rsi; when they return, the object gets an id, never
// reused. The objects do not change after creation, so an id stands for
// what its object holds - a program, a fetch shader - for as long as the
// object lives, and a new object at the same address gets a new id. `frame`
// is the prologue hook's (r9 r8 rcx rdx rsi rdi, then the return address).
void gx_objects_watch_creator(std::uint64_t* frame);
// The id of the object at `object`; 0 for one the registry did not see
// created (made before the hooks, or the table full).
std::uint32_t gx_object_id(std::uint64_t object);
// Render-target and depth-stencil views (0x25667a0 / 0x2566610,
// (ctx, view** out, texture holder, D3D11-shaped description, ...)) are
// registered the same way, with the extent of the 2D texture they view at
// the view's mip - which the register words GX builds give only padded to the
// tiling. (width << 16) | height, 0 when not known (not a 2D texture view, or
// a view the registry did not see made).
std::uint32_t gx_view_extent(std::uint64_t view);
// Shader-resource views (0x25669a0) and samplers (0x2565b40)
// are registered too, with their T# (at +0x10) or S# (at +0x0) hashed at
// creation (gx_words_hash of the 32 bytes). The id of `object` and that hash.
std::uint32_t gx_object_lookup(std::uint64_t object, std::uint64_t* words);
std::uint64_t gx_words_hash(const std::uint8_t* w);
