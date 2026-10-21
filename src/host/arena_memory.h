// Where GX's upload heap goes once it is a Vulkan allocation (GX on Vulkan,
// its stage for the arena: host/gxvk.h). The heap is one region - 160 MiB in
// this build - that GX's own code writes on the CPU (renamed constant
// buffers, the Map renames the game fills, the draws' tables) and the GPU
// reads in place: a binding is the arena's buffer at the address's offset.
//
// The choice, by what reads it on the CPU:
// - Nothing (renamed constant buffers bound in place, no snapshot of them):
//   1. video memory the CPU can write (DEVICE_LOCAL | HOST_VISIBLE |
//      HOST_COHERENT) while its heap's budget has room for the whole arena
//      and the margin the rings keep (ring_budget_fits) - resizable BAR, or
//      an integrated GPU's memory. The GPU reads it at its own speed; the
//      CPU writes it write-combined.
//   2. cached host memory (HOST_VISIBLE | HOST_COHERENT | HOST_CACHED): the
//      GPU reads it across the bus, as it reads guest memory today.
//   3. any coherent host-visible memory outside video memory, then any.
// - The constant-buffer snapshots (copied on the CPU as a draw records,
//   their default): cached memory first - write-combined memory reads at a
//   fraction of the speed - then 1 and 3.
// Never a type without HOST_COHERENT: GX writes the arena with no flush, as
// it writes guest memory. None found: no arena, and GX keeps its own region.
// On a discrete GPU without resizable BAR the CPU-writable video memory is a
// 256 MiB window every process shares, which the rings already take up to
// 160 MiB of: the arena does not fit there and goes to cached memory.
// `forced` (a measurement: BBHOST_GXVK_ARENA) takes that kind whatever the
// budget says, else the order above.
#pragma once

#include "host/ring_budget.h"

#include <cstdint>

namespace arena_memory {

// Vulkan's VkMemoryPropertyFlagBits and VkMemoryHeapFlagBits, as the device reports them.
constexpr std::uint32_t kDeviceLocal = 0x1, kHostVisible = 0x2, kHostCoherent = 0x4, kHostCached = 0x8;
constexpr std::uint32_t kHeapDeviceLocal = 0x1;

struct Type {
    std::uint32_t flags = 0, heap = 0;
};
struct Heap {
    std::uint64_t size = 0;
    std::uint32_t flags = 0;
    // VK_EXT_memory_budget's numbers for the heap, when the driver has them.
    bool have_budget = false;
    std::uint64_t budget = 0, used = 0;
};

enum class Kind { kNone, kBar, kCached, kHost };
inline const char* name(Kind k) {
    return k == Kind::kBar ? "video memory the CPU writes" : k == Kind::kCached ? "cached host memory" : k == Kind::kHost ? "host memory" : "none";
}

struct Choice {
    Kind kind = Kind::kNone;
    std::uint32_t type = UINT32_MAX;
    RingBudget room;  // the budget's answer for the CPU-writable video memory, when asked
};

inline Choice choose(const Type* types, std::uint32_t ntypes, const Heap* heaps, std::uint32_t type_bits, std::uint64_t bytes, bool cpu_reads,
                     Kind forced = Kind::kNone) {
    // The first allowed type with every flag of `want`, none of `avoid`, and
    // (outside_video) a heap that is not video memory.
    const auto find = [&](std::uint32_t want, std::uint32_t avoid, bool outside_video) -> std::uint32_t {
        for (std::uint32_t i = 0; i < ntypes && i < 32; ++i) {
            if (!((type_bits >> i) & 1)) continue;
            if ((types[i].flags & want) != want || (types[i].flags & avoid)) continue;
            if (outside_video && (heaps[types[i].heap].flags & kHeapDeviceLocal)) continue;
            return i;
        }
        return UINT32_MAX;
    };
    Choice c;
    const std::uint32_t bar = find(kDeviceLocal | kHostVisible | kHostCoherent, 0, false);
    const std::uint32_t cached = find(kHostVisible | kHostCoherent | kHostCached, 0, false);
    std::uint32_t host = find(kHostVisible | kHostCoherent, 0, true);
    if (host == UINT32_MAX) host = find(kHostVisible | kHostCoherent, 0, false);
    bool bar_fits = false;
    if (bar != UINT32_MAX) {
        const Heap& h = heaps[types[bar].heap];
        c.room = ring_budget_fits(h.size, h.have_budget, h.budget, h.used, bytes);
        bar_fits = c.room.fits;
    }
    const auto take = [&](Kind k, std::uint32_t t) {
        c.kind = k;
        c.type = t;
        return t != UINT32_MAX;
    };
    if (forced == Kind::kBar && take(Kind::kBar, bar)) return c;
    if (forced == Kind::kCached && take(Kind::kCached, cached)) return c;
    if (forced == Kind::kHost && take(Kind::kHost, host)) return c;
    if (cpu_reads) {
        if (take(Kind::kCached, cached)) return c;
        if (bar_fits && take(Kind::kBar, bar)) return c;
    } else {
        if (bar_fits && take(Kind::kBar, bar)) return c;
        if (take(Kind::kCached, cached)) return c;
    }
    if (take(Kind::kHost, host)) return c;
    c.kind = Kind::kNone;
    c.type = UINT32_MAX;
    return c;
}

}  // namespace arena_memory
