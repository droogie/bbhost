// GX's upload heap's pool as source (docs/decomp.md): gx_upload_pool_init
// (0x2aa9b50), the block pool's initialisation over the region GX's device
// constructor (0x256b0f0) allocates for it, its one caller. Every renamed
// constant buffer, Map rename, ring allocation and draw table comes from that
// pool.
//
// The pool keeps the region 16-byte aligned: its start at +0xb8, its end at
// +0xc0 and at +0x70, its free bytes at +0xd0 and the ranges out at +0xc8.
// Its node allocator - 0x30-byte nodes from 0xff0-byte blocks of the
// allocator it is given, 8-aligned - sits at its own start (8-aligned), set
// up by 0x2a3baa0 and marked made at +0x67. Its address list (+0x78/+0x80)
// and free list (+0x88/+0x90) hang off the sentinel at +0x68, and +0x98..+0xb0
// are cleared. Then one node, in state 0, takes the whole region and goes on
// both lists. With no node to be had it takes its node allocator down again
// (0x2a3bc40) and returns 0; else the node's address with its low byte 1 (the
// constructor tests that byte).
//
// Ours makes the same calls in the same order and writes the same fields. It
// first tells the host's entry (decomp_gx::g_upload_region_entry), when there
// is one, of the pool and its region, and takes the region the entry gives
// back: GX on Vulkan's arena (host/gxvk.h) puts the heap in memory of its own
// there, and its census learns the heap's bounds. Without the entry, or when
// it keeps the game's region, ours is the game's function.
//
// A leaf: it reads and writes the game's memory and calls the game's
// functions, and of the host's only that entry, which switches to the host's
// FS itself.
#include "decomp/gx/upload_heap.h"

#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <cstdint>

using namespace decomp;

// tests/decomp/gx_upload_heap_test.cpp builds this file with mutants: small
// wrong turns it switches on one at a time and must catch.
#ifdef DECOMP_GX_UPLOAD_HEAP_MUTANTS
extern int g_upload_heap_mutant;
#define MUTANT(n) (g_upload_heap_mutant == (n))
#else
#define MUTANT(n) false
#endif

void* decomp_gx::g_upload_region_entry = nullptr;

namespace {

constexpr u64 kUploadPoolInit = 0x2aa9b50;
// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; push rax
constexpr u8 kEntryUploadPoolInit[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50};

// The node allocator: the engine's fixed-size block allocator.
constexpr u64 kNodesInit = 0x2a3baa0;   // (nodes, allocator, node bytes, block bytes, alignment)
constexpr u64 kNodesTake = 0x2a3b8b0;   // (nodes): a node, or 0
constexpr u64 kNodesClose = 0x2a3bc40;  // (nodes): its blocks given back

using NodesInitFn = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64);
using NodesFn = GUEST_ABI u64 (*)(u64);
using RegionFn = GUEST_ABI u64 (*)(u64, u64, u64, u64);

}  // namespace

// gx_upload_pool_init(pool, allocator, region, bytes).
DECOMP_LEAF u64 gx_upload_pool_init(u64 pool, u64 allocator, u64 region, u64 bytes) {
    // The host's region in place of the game's, when it gives one.
    if (void* e = decomp_gx::g_upload_region_entry) {
        u64 their_bytes = bytes;
        const u64 their = reinterpret_cast<RegionFn>(e)(pool, region, bytes, reinterpret_cast<u64>(&their_bytes));
        if (their) {
            region = their;
            bytes = their_bytes;
        }
    }
    // The region's first 16-byte boundary and its 16-byte length from there.
    const u64 pad = static_cast<u64>(0u - static_cast<u32>(region)) & 0xf;
    const u64 start = region + pad;
    const u64 size = MUTANT(1) ? bytes - pad : (bytes - pad) & ~u64{0xf};
    store<u64>(pool + 0xb8, start);
    store<u64>(pool + 0xc0, start + size);
    // The node allocator at the pool's first 8-byte boundary.
    const u64 nodes = pool + (static_cast<u64>(0u - static_cast<u32>(pool)) & 7);
    if (nodes) game_function<NodesInitFn>(kNodesInit)(nodes, allocator, 0x30, 0xff0, 8);
    store<u8>(pool + 0x67, 1);
    store<u64>(pool + 0xc8, 0);
    store<u64>(pool + 0xd0, size);
    store<u64>(pool + 0x70, region + (pad | size));
    const u64 head = pool + 0x68;
    store<u64>(pool + 0x78, head);
    store<u64>(pool + 0x80, head);
    store<u64>(pool + 0x88, head);
    store<u64>(pool + 0x90, head);
    store<u64>(pool + 0xb0, 0);
    store<u64>(pool + 0xa8, 0);
    store<u64>(pool + 0xa0, 0);
    store<u64>(pool + 0x98, 0);
    const u64 node = game_function<NodesFn>(kNodesTake)(nodes);
    if (!node) {
        if (load<u8>(pool + 0x67) && !MUTANT(2)) {
            game_function<NodesFn>(kNodesClose)(nodes);
            store<u8>(pool + 0x67, 0);
        }
        return 0;
    }
    // The one node over the whole region: first on the address list, last on the free list.
    store<u64>(node, 0);
    store<u64>(node + 0x08, start);
    store<u64>(node + 0x18, head);
    store<u64>(pool + 0x78, node);
    store<u64>(pool + 0x80, node);
    store<u64>(node + 0x10, head);
    const u64 last = load<u64>(pool + 0x90);
    store<u64>(pool + 0x90, node);
    store<u64>(node + 0x28, last);
    store<u64>(node + 0x20, head);
    store<u64>(last + (MUTANT(3) ? 0x28 : 0x20), node);
    return MUTANT(4) ? 1 : (node & ~u64{0xff}) | 1;
}

void decomp_gx_upload_heap_add() {
    decomp_add({"gx_upload_pool_init", "gx", kUploadPoolInit, kEntryUploadPoolInit, sizeof(kEntryUploadPoolInit),
                reinterpret_cast<void*>(&gx_upload_pool_init), DecompKind::Leaf});
}
