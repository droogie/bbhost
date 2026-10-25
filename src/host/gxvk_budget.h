#pragma once
// GX's buffer copies within a budget (gxvk_vk.cpp): how large the copy pool
// may grow, and which copies go first when it is full or video memory runs
// short. Kept apart from Vulkan so a test can drive it.
//
// On a graphics card with small video memory the copies used to be off
// altogether: placed in a dip, one copy in host memory cost Mensis 18-20 ms
// of GPU a flip. Copies never go to host memory on a discrete GPU since, and
// with them off the buffer shadow's mirrors carry the meshes: at a 4 GB
// card's budget they were refused room at Mensis and 16.8 M draws read their
// vertices across the bus (the GPU 18.45 ms a flip), while the command
// processor paid 1.3-1.7 ms a flip more than with the copies.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace gxvk_budget {

constexpr std::uint64_t kMiB = 1ull << 20;
constexpr std::uint64_t kMinBudget = 256 * kMiB, kMaxBudget = 512 * kMiB;
// A copy not bound for this many flips (two seconds at 60) may be dropped.
constexpr std::uint32_t kIdleFlips = 120;

// The pool's budget in bytes, 0 for none: BBHOST_GXVK_BUFFERS_BUDGET_MB
// (`env`, MiB; 0 none) on any device; unset, with small video memory an
// eighth of the video memory free at the start (`local_budget`, the driver's
// budget for us) from 256 to 512 MiB - a 4 GB card's ~370 - and none
// elsewhere (the pool held at most ~640 MiB over a 12-place tour at 12 GB).
inline std::uint64_t budget_bytes(const char* env, bool small_vram, std::uint64_t local_budget) {
    if (env && *env) return std::strtoull(env, nullptr, 10) * kMiB;
    if (!small_vram) return 0;
    return std::clamp<std::uint64_t>(local_budget / 8, kMinBudget, kMaxBudget);
}

// Whether a copy last bound at flip `last` is idle at flip `now` (the low 32
// bits of the flip count: the difference wraps as they do), after `flips`
// (kIdleFlips; BBHOST_TEST_GXVK_IDLE_FLIPS for tests on slow devices).
inline bool idle(std::uint32_t last, std::uint32_t now, std::uint32_t flips = kIdleFlips) { return now - last >= flips; }

struct Copy {
    std::uint32_t last = 0;   // the flip it was last bound at
    std::uint64_t bytes = 0;  // its range in the pool
};

// The copies to drop to free `need` bytes: idle ones only, the least recently
// bound first (the earlier in `copies` first among equals). It may free less
// than `need` when too few are idle. Positions in `copies`.
inline std::vector<std::size_t> victims(const std::vector<Copy>& copies, std::uint32_t now, std::uint64_t need,
                                        std::uint32_t flips = kIdleFlips) {
    std::vector<std::size_t> idle_ones;
    for (std::size_t i = 0; i < copies.size(); ++i)
        if (idle(copies[i].last, now, flips)) idle_ones.push_back(i);
    std::stable_sort(idle_ones.begin(), idle_ones.end(),
                     [&](std::size_t a, std::size_t b) { return now - copies[a].last > now - copies[b].last; });
    std::vector<std::size_t> out;
    std::uint64_t freed = 0;
    for (const std::size_t i : idle_ones) {
        if (freed >= need) break;
        out.push_back(i);
        freed += copies[i].bytes;
    }
    return out;
}

}  // namespace gxvk_budget
