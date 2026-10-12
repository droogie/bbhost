// SPDX-License-Identifier: GPL-2.0-or-later
// Unit tests for motion_history.h CPU data structures and logic.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "host/motion_history.h"

using namespace gpu::motion;

static void test_indexed_range() {
    // Basic u16
    const std::vector<uint16_t> idx16 = {2, 5, 8, 3, 10};
    const auto r1 = IndexedRange(std::span(idx16), 0, false);
    assert(r1.first == 2);
    assert(r1.count == 9); // 10 - 2 + 1

    // Base vertex offset
    const auto r2 = IndexedRange(std::span(idx16), 100, false);
    assert(r2.first == 102);
    assert(r2.count == 9);

    // Negative base vertex
    const auto r3 = IndexedRange(std::span(idx16), -2, false);
    assert(r3.first == 0);
    assert(r3.count == 9);

    // Negative base vertex causing underflow -> rejected
    const auto r4 = IndexedRange(std::span(idx16), -3, false);
    assert(r4.count == 0);

    // Primitive restart u16
    const std::vector<uint16_t> restart16 = {10, 0xFFFF, 20, 0xFFFF, 15};
    const auto r5 = IndexedRange(std::span(restart16), 0, true);
    assert(r5.first == 10);
    assert(r5.count == 11); // 20 - 10 + 1

    // Primitive restart u32
    const std::vector<uint32_t> restart32 = {50, 0xFFFFFFFFu, 100, 75};
    const auto r6 = IndexedRange(std::span(restart32), 10, true);
    assert(r6.first == 60);
    assert(r6.count == 51); // 100 - 50 + 1

    std::puts("PASS: test_indexed_range");
}

static void test_classify_buffer() {
    assert(ClassifyBuffer(864) == BufferRole::Other);      // Scene constants (camera)
    assert(ClassifyBuffer(416) == BufferRole::Other);      // Model constants
    assert(ClassifyBuffer(20000) == BufferRole::Other);    // Exceeds 16 KiB
    assert(ClassifyBuffer(640) == BufferRole::Skeleton);   // Minimum character skeleton
    assert(ClassifyBuffer(1024) == BufferRole::Skeleton);
    assert(ClassifyBuffer(96) == BufferRole::SmallSkeleton);   // 2 bones (2 * 48)
    assert(ClassifyBuffer(144) == BufferRole::SmallSkeleton);  // 3 bones
    assert(ClassifyBuffer(384) == BufferRole::SmallSkeleton);  // 8 bones
    assert(ClassifyBuffer(100) == BufferRole::Other);          // Not multiple of 48
    assert(ClassifyBuffer(48) == BufferRole::Other);           // Below 96

    std::puts("PASS: test_classify_buffer");
}

static void test_index_range_cache() {
    IndexRangeCache cache;
    IndexRangeCache::Key key{0x1000, 100, 2, 0, false};

    uint64_t scan_calls = 0;
    auto scan_func = [&]() {
        ++scan_calls;
        return IndexRangeCache::Result{{0, 50}, 0xDEADBEEF};
    };

    // Frame 0: cold miss -> scans
    auto res0 = cache.Get(key, 0, scan_func);
    assert(scan_calls == 1);
    assert(res0.range.count == 50);
    assert(cache.stats.scans == 1);
    assert(cache.stats.hits == 0);

    // Frame 10 (< 32 frames): hit
    auto res1 = cache.Get(key, 10, scan_func);
    assert(scan_calls == 1);
    assert(res1.range == res0.range);
    assert(cache.stats.hits == 1);

    // Frame 35 (>= 32 frames): revalidation scan
    auto res2 = cache.Get(key, 35, scan_func);
    assert(scan_calls == 2);
    assert(cache.stats.scans == 2);

    // Trim unused
    assert(cache.Size() == 1);
    cache.Trim(700); // 700 - 35 = 665 > 600
    assert(cache.Size() == 0);

    std::puts("PASS: test_index_range_cache");
}

static void test_history_lifecycle() {
    History hist(1000); // capacity 1000

    Draw d1{
        .shader = 0x1111,
        .geometry = 0x2222,
        .indices = 0x3333,
        .topology = 0x4444,
        .index_count = 60,
        .instances = 1,
        .first_instance = 0,
        .vertices = {0, 100},
    };

    // Frame 0: Cold draw
    auto alloc0 = hist.Prepare(d1);
    assert(alloc0.load == 0);
    assert(alloc0.store > 0);
    assert((alloc0.flags & FlagStore) != 0);
    assert((alloc0.flags & FlagLoad) == 0);
    assert(hist.stats.stored == 1);
    assert(hist.stats.unmatched == 1);
    assert(hist.Used() == 100);

    // Frame advance
    hist.NextFrame();
    assert(hist.CurrentFrame() == 1);
    assert(hist.Used() == 0);

    // Frame 1: Matching draw should load previous store
    auto alloc1 = hist.Prepare(d1);
    assert(alloc1.load == alloc0.store);
    assert(alloc1.store > 0);
    assert(alloc1.store != alloc0.store); // different half
    assert((alloc1.flags & FlagStore) != 0);
    assert((alloc1.flags & FlagLoad) != 0);
    assert(hist.stats.loaded == 1);

    // Reset / Invalidate: advances two frames, clearing both generations
    hist.Invalidate();
    assert(hist.CurrentFrame() == 3);
    assert((hist.CurrentFrame() & 1) == (1 & 1)); // parity preserved!

    // Frame 3 after invalidation: should be cold again
    auto alloc2 = hist.Prepare(d1);
    assert(alloc2.load == 0);
    assert(alloc2.store > 0);
    assert((alloc2.flags & FlagLoad) == 0);

    std::puts("PASS: test_history_lifecycle");
}

static void test_history_gating() {
    History hist(10000);
    History::GateKey gk{
        .shader = 0xAAAA,
        .geometry = 0xBBBB,
        .indices = 0xCCCC,
        .index_count = 30,
        .instances = 1,
        .first_instance = 0,
        .occurrence = 0,
    };

    // Frame 0: First appearance, no previous palette -> still
    bool m0 = hist.Moving(gk, 0x1234);
    assert(!m0);
    assert(hist.stats.still == 1);

    hist.NextFrame();

    // Frame 1: Palette changed -> moving!
    bool m1 = hist.Moving(gk, 0x5678);
    assert(m1);
    assert(hist.stats.still == 1);

    hist.NextFrame();

    // Frame 2: Palette unchanged -> still
    bool m2 = hist.Moving(gk, 0x5678);
    assert(!m2);
    assert(hist.stats.still == 2);

    std::puts("PASS: test_history_gating");
}

static void test_flat_map() {
    struct Hash {
        size_t operator()(uint32_t v) const { return v * 2654435761u; }
    };
    FlatMap<uint32_t, uint32_t, Hash> map;

    map.emplace(10, 100);
    map.emplace(20, 200);
    assert(map.find(10) && *map.find(10) == 100);
    assert(map.find(20) && *map.find(20) == 200);
    assert(map.find(30) == nullptr);

    map.clear();
    assert(map.find(10) == nullptr);
    assert(map.find(20) == nullptr);

    std::puts("PASS: test_flat_map");
}

int main() {
    test_indexed_range();
    test_classify_buffer();
    test_index_range_cache();
    test_history_lifecycle();
    test_history_gating();
    test_flat_map();
    std::puts("ALL MOTION HISTORY TESTS PASSED!");
    return 0;
}
