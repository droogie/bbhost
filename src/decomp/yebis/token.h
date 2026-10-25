#pragma once

// The barrier token YEBIS's drains and shader waits are written as
// (decomp/yebis/context.cpp's header): where the game embeds an 8-byte label
// in the command buffer itself, has an end-of-pipe or end-of-shader event
// write 1 into it and the GPU wait until it reads 1, ours writes a two-dword
// NOP {kGxBarrierMagic} at the write pointer - as the game's packet writers
// write, after the command buffer's refill (cb, dwords, its argument, room
// left) when there is no room - and notes it for the host lists. The
// renderer orders its own work, and the command processor takes the token as
// the wait it stood for. Shared by the context's drains and waits and the
// compute clear's wait (decomp/yebis/clear.cpp). For leaves: no host TLS, and
// the counts are relaxed atomics.

#include "decomp/guest.h"
#include "hle/modules.h"
#include "host/host_lists.h"

#include <atomic>
#include <cstdint>

namespace decomp {

// Tokens written (those after a refill among them), tokens not written: the
// refill gave no room. The context's report prints them
// (decomp/yebis/context.cpp).
inline std::atomic<std::uint64_t> g_yebis_tokens{0}, g_yebis_token_refills{0}, g_yebis_token_refused{0};

// Whether it was written. `magic` is kGxBarrierMagic; the decomp tests'
// mutants hand another.
inline bool yebis_barrier_token(u64 cb, std::uint32_t magic = kGxBarrierMagic) {
    using Refill = GUEST_ABI u8 (*)(u64 cb, u32 dwords, u64 arg, u32 room);
    u64 wp = load<u64>(cb + 0x10);
    const u64 end = load<u64>(cb + 0x8);
    if (!load<u64>(cb) || end < wp + 8) {
        const auto refill = reinterpret_cast<Refill>(static_cast<std::uintptr_t>(load<u64>(cb + 0x18)));
        const u32 room = end > wp ? static_cast<u32>((end - wp) >> 2) : 0;
        if (!refill || !refill(cb, 2, load<u64>(cb + 0x20), room) || load<u64>(cb + 0x8) < load<u64>(cb + 0x10) + 8) {
            g_yebis_token_refused.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        wp = load<u64>(cb + 0x10);
        g_yebis_token_refills.fetch_add(1, std::memory_order_relaxed);
    }
    store<u32>(wp, 0xc0001000u);  // NOP, one dword of body
    store<u32>(wp + 4, magic);
    store<u64>(cb + 0x10, wp + 8);
    host_lists::note(cb, wp, 0x10);
    g_yebis_tokens.fetch_add(1, std::memory_order_relaxed);
    return true;
}

}  // namespace decomp
