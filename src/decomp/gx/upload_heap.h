// GX's upload heap's pool as source (decomp/gx/upload_heap.cpp): what GX on
// Vulkan's arena (host/gxvk.h) and ours share.
#pragma once

#include "guest_abi.h"

#include <cstdint>

namespace decomp_gx {

// The host's entry for the upload heap's region, which ours of
// gx_upload_pool_init calls first: region(pool, game's region, its bytes,
// u64* bytes out) returns the start of the region the pool is to use and
// writes its bytes, or returns 0 to keep the game's. Behind the FS-switching
// thunk (hle_wrap_fn). Null when neither GX on Vulkan's arena nor its census
// asks: ours is then the game's function alone.
extern void* g_upload_region_entry;

}  // namespace decomp_gx

// gx_upload_pool_init(pool, allocator, region, bytes): the game's 0x2aa9b50.
GUEST_ABI std::uint64_t gx_upload_pool_init(std::uint64_t pool, std::uint64_t allocator, std::uint64_t region, std::uint64_t bytes);
