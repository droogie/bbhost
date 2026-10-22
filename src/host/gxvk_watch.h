#pragma once
// The copy watch (host/gxvk_buffers.cpp): the ranges of the default buffers
// the host keeps a copy of, which a GPU copy may write. Every buffer copy GX
// records asks it whether its destination meets one - the map resolves copy
// renamed dynamic buffers back to their own addresses many times a frame, on
// GX's threads - and no destination has met one in any run so far.
//
// Two answers, cheapest first. The granules: each watched range counts in the
// 64 KiB granules it covers (a direct-mapped table of 16,384 one-byte
// counters, 16 KiB, so granules 1 GiB apart share one), and a destination
// whose granules hold no count is done after a load or two. They let 44% of
// the copies at Old Yharnam through (the copy-backs land next to watched
// buffers), so behind them the ranges themselves answer exactly: sorted by
// their start with the highest end so far beside each, published under a
// sequence count, and searched in halves - six steps for 64 ranges, where a
// scan of the slots read all 64.
//
// Ranges are added and removed under the caller's lock and read without one.
// A range's counters are raised before it is published and lowered after it
// is withdrawn, so whoever finds the range sees its counters. Pure and
// unit-tested (tests/gxvk_watch_test.cpp).
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
    // is taken or the range is empty. Under the caller's lock.
    int add(std::uint64_t lo, std::uint64_t hi) {
        if (!lo || hi <= lo) return -1;
        for (int i = 0; i < Slots; ++i) {
            if (lo_[i]) continue;
            count(lo, hi, 1);
            lo_[i] = lo;
            hi_[i] = hi;
            ++used_n_;
            publish();
            used_.store(used_n_, std::memory_order_relaxed);
            return i;
        }
        return -1;
    }
    // Under the caller's lock.
    void remove(int i) {
        if (i < 0 || i >= Slots || !lo_[i]) return;
        const std::uint64_t lo = lo_[i], hi = hi_[i];
        lo_[i] = hi_[i] = 0;
        --used_n_;
        publish();
        used_.store(used_n_, std::memory_order_relaxed);
        if (!GXVK_WATCH_MUTANT(3)) count(lo, hi, -1);
    }
    bool empty() const { return used_.load(std::memory_order_relaxed) == 0; }
    // Whether a granule of [va, va + bytes) holds a watched range's count:
    // never false where a range meets it.
    bool near(std::uint64_t va, std::uint64_t bytes) const {
        if (!bytes) return false;
        const std::uint64_t g0 = va >> GranuleShift,
                            n = GXVK_WATCH_MUTANT(1) ? 1 : std::min<std::uint64_t>(((va + bytes - 1) >> GranuleShift) - g0 + 1, Buckets);
        for (std::uint64_t g = g0; g < g0 + n; ++g)
            if (counts_[g & (Buckets - 1)].load(std::memory_order_acquire)) return true;
        return false;
    }
    // Whether a watched range meets [va, va + bytes): the last range starting
    // before its end is found in halves, and the highest end among the ranges
    // up to it says whether any reaches past its start.
    bool hit(std::uint64_t va, std::uint64_t bytes) const {
        if (!bytes) return false;
        const std::uint64_t end = va + bytes;
        for (;;) {
            const std::uint32_t s = seq_.load(std::memory_order_acquire);
            if (s & 1u) continue;  // a publish in progress: a few stores long
            const int n = n_.load(std::memory_order_relaxed);
            int lo = 0, hi = n;
            while (lo < hi) {
                const int mid = (lo + hi) / 2;
                if (start_[mid].load(std::memory_order_relaxed) < end) lo = mid + 1;
                else hi = mid;
            }
            bool meets = false;
            if (lo > 0) {
                const std::uint64_t reach = GXVK_WATCH_MUTANT(5) ? own_end_[lo - 1].load(std::memory_order_relaxed)
                                                                 : reach_[lo - 1].load(std::memory_order_relaxed);
                meets = GXVK_WATCH_MUTANT(4) ? reach >= va : reach > va;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq_.load(std::memory_order_relaxed) == s) return meets;
        }
    }
    // The question a copy asks: the granules first, the ranges only behind them.
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
    // The live ranges sorted by start, with the highest end up to each, in
    // place of the published ones: the sequence count is odd while they change.
    void publish() {
        std::uint64_t lo[Slots], hi[Slots];
        int n = 0;
        for (int i = 0; i < Slots; ++i)
            if (lo_[i]) lo[n] = lo_[i], hi[n] = hi_[i], ++n;
        int order[Slots];
        for (int i = 0; i < n; ++i) order[i] = i;
        std::sort(order, order + n, [&](int a, int b) { return lo[a] < lo[b]; });
        const std::uint32_t s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        std::uint64_t reach = 0;
        for (int k = 0; k < n; ++k) {
            reach = std::max(reach, hi[order[k]]);
            start_[k].store(lo[order[k]], std::memory_order_relaxed);
            own_end_[k].store(hi[order[k]], std::memory_order_relaxed);
            reach_[k].store(reach, std::memory_order_relaxed);
        }
        n_.store(n, std::memory_order_relaxed);
        seq_.store(s + 2, std::memory_order_release);
    }

    std::uint64_t lo_[Slots] = {}, hi_[Slots] = {};  // the slots, the caller's lock's
    int used_n_ = 0;
    std::atomic<std::uint32_t> seq_{0};
    std::atomic<int> n_{0};
    std::atomic<std::uint64_t> start_[Slots] = {}, reach_[Slots] = {}, own_end_[Slots] = {};
    std::atomic<std::uint8_t> counts_[Buckets] = {};
    std::atomic<int> used_{0};
};

}  // namespace gxvk
