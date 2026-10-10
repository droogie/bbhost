#pragma once

#include <atomic>
#include <cstdint>

namespace bb::libc {

// The dump's libc.prx: rand (cpCOXWMgha0, RVA 0x17000) advances a
// uint64 LCG and returns 30 bits; srand (VPbJwTCgME0, RVA 0x17080)
// zero-extends the seed into that state. Initial state at 0xb88f0 is 1.
// Not the host CRT: Windows rand only has 15 bits, so HavokScript's
// math.random(0, 100), which normalizes by 2^30, always returned zero.
class Random {
public:
    void seed(std::uint32_t value) { state_.store(value, std::memory_order_relaxed); }

    int next() {
        auto previous = state_.load(std::memory_order_relaxed);
        std::uint64_t next;
        do {
            next = previous * 0x5851f42d4c957f2dull + 1;
        } while (!state_.compare_exchange_weak(previous, next, std::memory_order_relaxed));
        return static_cast<int>((next >> 32) & 0x3fffffff);
    }

private:
    // Shared by guest threads, independent of any host library's srand.
    // Atomic updates avoid a C++ data race when callers run concurrently.
    std::atomic<std::uint64_t> state_{1};
};

}  // namespace bb::libc
