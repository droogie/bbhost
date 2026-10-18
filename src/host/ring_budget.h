// Whether one of the rings the CPU writes and the GPU reads every draw (the
// params ring, the constant-buffer copies, the rewritten vertex data) may go in
// video memory the CPU can write, by the heap's budget (host/gpu.cpp
// ring_fits_budget, which reads the numbers from VK_EXT_memory_budget).
//
// On a discrete GPU without resizable BAR that heap is a 256 MiB window every
// process on the machine shares: an RTX 4070's (Linux, 2026-10-16) had 218 MiB
// of budget with 27 MiB of it bbhost's own, and the three rings are 160 MiB -
// all three went there and the heap ran over its budget (187 MiB used against
// a budget of 58), while every allocation still succeeded, so the "host memory
// when it is full" fallback never ran. So a ring goes there only while the
// budget leaves room for it and a margin for the rest of the system: 32 MiB or
// an eighth of the budget, whichever is more. Room is the budget less what is
// in use (the extension's meaning); a driver whose budget already leaves out
// what is in use (NVIDIA's for that window: it fell by each allocation) is
// read as having less room than it has, which errs towards host memory.
// Without the extension only a heap of 1 GiB or more takes a ring (resizable
// BAR, or an integrated GPU's memory): a smaller one is a shared window whose
// room nothing says.
#pragma once

#include <algorithm>
#include <cstdint>

struct RingBudget {
    bool fits = false;
    std::uint64_t room = 0;  // bytes the budget leaves (the heap's size without a budget)
    std::uint64_t keep = 0;  // bytes kept free for the rest of the system
};

inline RingBudget ring_budget_fits(std::uint64_t heap_size, bool have_budget, std::uint64_t budget, std::uint64_t used,
                                   std::uint64_t bytes) {
    RingBudget r;
    if (!have_budget) {
        r.room = heap_size;
        r.fits = heap_size >= (1ull << 30) && bytes <= heap_size / 4;
        return r;
    }
    r.room = budget > used ? budget - used : 0;
    r.keep = std::max<std::uint64_t>(32ull << 20, budget / 8);
    r.fits = r.room >= bytes + r.keep;
    return r;
}
