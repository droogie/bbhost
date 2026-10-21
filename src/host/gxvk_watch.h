#pragma once
// The copy watch (host/gxvk_buffers.cpp): the ranges of the default buffers
// the host keeps a copy of, which a GPU copy may write. Every buffer copy GX
// records asks it whether its destination meets one - the map resolves copy
// renamed dynamic buffers back to their own addresses many times a frame, on
// GX's threads - and nearly every destination is far from every watched
// range. A scan of the slots answered it for 1% of a core at Central Yharnam;
// the granules answer it first: each watched range counts in the 64 KiB
// granules it covers (a direct-mapped table of 16,384 one-byte counters, 16
// KiB, so granules 1 GiB apart share one), and a destination whose granules
// hold no count is done after a load or two. Small granules keep a copy-back
// into a dynamic buffer next to a watched one out of its granules; the
// watched buffers are few (a dozen at Central Yharnam) and small, so few
// counters are ever set. Slots and counters are set and cleared under the
// caller's lock and read without one; a range's counters are raised before
// its slot is filled and lowered after it is emptied, so whoever sees the
// slot sees the counters. Pure and unit-tested (tests/gxvk_watch_test.cpp).
#include <algorithm>
#include <atomic>
#include <cstdint>

// tests/gxvk_watch_test.cpp builds this with deliberate mistakes, one at a
// time, to show its checks catch them.
#ifdef GXVK_WATCH_MUTANTS
extern int g_watch_mutant;
#define GXVK_WATCH_MUTANT(n) (g_watch_mutant == (n))
#else
#define GXVK_WATCH_MUTANT(n) false
#endif

namespace gxvk {

template <int Slots = 64, unsigned GranuleShift = 16, unsigned Buckets = 16384>
class CopyWatch {
    static_assert(Slots > 0 && Slots <= 255, "a counter holds one count a slot");
    static_assert(Buckets && (Buckets & (Buckets - 1)) == 0, "the table is indexed by a mask");

public:
    // [lo, hi) put in a free slot, whose index comes back; -1 when every slot
    // is taken or the range is empty.
    int add(std::uint64_t lo, std::uint64_t hi) {
        if (!lo || hi <= lo) return -1;
        for (int i = 0; i < Slots; ++i) {
            if (lo_[i].load(std::memory_order_relaxed)) continue;
            count(lo, hi, 1);
            hi_[i].store(hi, std::memory_order_relaxed);
            lo_[i].store(lo, std::memory_order_release);
            used_.fetch_add(1, std::memory_order_relaxed);
            return i;
        }
        return -1;
    }
    void remove(int i) {
        if (i < 0 || i >= Slots) return;
        const std::uint64_t lo = lo_[i].load(std::memory_order_relaxed), hi = hi_[i].load(std::memory_order_relaxed);
        if (!lo) return;
        lo_[i].store(0, std::memory_order_release);
        if (!GXVK_WATCH_MUTANT(3)) count(lo, hi, -1);
        used_.fetch_sub(1, std::memory_order_relaxed);
    }
    bool empty() const { return used_.load(std::memory_order_relaxed) == 0; }
    // Whether a granule of [va, va + bytes) holds a watched range's count:
    // never false where a slot's range meets it.
    bool near(std::uint64_t va, std::uint64_t bytes) const {
        if (!bytes) return false;
        const std::uint64_t g0 = va >> GranuleShift,
                            n = GXVK_WATCH_MUTANT(1) ? 1 : std::min<std::uint64_t>(((va + bytes - 1) >> GranuleShift) - g0 + 1, Buckets);
        for (std::uint64_t g = g0; g < g0 + n; ++g)
            if (counts_[g & (Buckets - 1)].load(std::memory_order_acquire)) return true;
        return false;
    }
    // Whether a slot's range meets [va, va + bytes).
    bool hit(std::uint64_t va, std::uint64_t bytes) const {
        if (!bytes) return false;
        for (int i = 0; i < Slots; ++i) {
            const std::uint64_t lo = lo_[i].load(std::memory_order_acquire);
            if (lo && va < hi_[i].load(std::memory_order_relaxed) && (GXVK_WATCH_MUTANT(4) ? va + bytes >= lo : va + bytes > lo)) return true;
        }
        return false;
    }
    // The question a copy asks: near() first, the slots only after it.
    bool written(std::uint64_t va, std::uint64_t bytes) const { return !empty() && near(va, bytes) && hit(va, bytes); }

private:
    // One count in each counter the range's granules map to (once each, for
    // a range wider than the table).
    void count(std::uint64_t lo, std::uint64_t hi, int delta) {
        const std::uint64_t g0 = lo >> GranuleShift,
                            n = std::min<std::uint64_t>(((hi - 1) >> GranuleShift) - g0 + (GXVK_WATCH_MUTANT(2) ? 0 : 1), Buckets);
        for (std::uint64_t g = g0; g < g0 + n; ++g) {
            std::atomic<std::uint8_t>& c = counts_[g & (Buckets - 1)];
            c.store(static_cast<std::uint8_t>(c.load(std::memory_order_relaxed) + delta), std::memory_order_release);
        }
    }

    std::atomic<std::uint64_t> lo_[Slots] = {}, hi_[Slots] = {};
    std::atomic<std::uint8_t> counts_[Buckets] = {};
    std::atomic<int> used_{0};
};

}  // namespace gxvk
