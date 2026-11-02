// Stage 4b.3c's index of live homes by address (host/gxvk_homes_index.h),
// without the game or a GPU: homes made, ended and made again over the same
// memory - small constant buffers packed in a few pages, as GX's heaps hold
// them, a page shared by several, ranges that run across pages - and every
// answer the command processor asks of it (meets, holding, straddles, and the
// filter's "no") checked against a plain list of the live ranges. The filter
// may say "maybe" where nothing is; it may never say "no" where a home is.
// The handle's packing is checked too (host/gxvk_homes.h).
#include "host/gxvk_homes.h"
#include "host/gxvk_homes_index.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct Live {
    std::uint32_t index;
    std::uint64_t memory;
    std::uint32_t bytes;
};

int g_failures = 0;
void check(bool ok, const char* what, std::uint64_t va, std::uint64_t bytes) {
    if (ok) return;
    if (++g_failures <= 12) std::printf("FAIL: %s at 0x%llx + %llu\n", what, static_cast<unsigned long long>(va), static_cast<unsigned long long>(bytes));
}

// The model: the live homes meeting [va, +bytes), and whether one holds it whole.
void model(const std::vector<Live>& live, std::uint64_t va, std::uint64_t bytes, bool* meets, bool* holding, bool* straddles,
           const Live** first) {
    *meets = *holding = *straddles = false;
    *first = nullptr;
    for (const Live& l : live) {
        if (l.memory < va + bytes && va < l.memory + l.bytes) {
            *meets = true;
            if (va >= l.memory && va + bytes <= l.memory + l.bytes) *holding = true;
            if (!*first || l.memory < (*first)->memory) *first = &l;
        }
    }
    *straddles = *meets && !*holding;
}

}  // namespace

int main() {
    using gxvk_homes::HomeIndex;
    // The handle: tag, generation and index survive the packing; stage 1b's tag is not ours.
    for (std::uint32_t gen = 0; gen < 256; gen += 37) {
        for (std::uint32_t index : {0u, 1u, 4095u, 4096u, gxvk_homes::kIndexMask}) {
            const std::uint32_t h = gxvk_homes::handle_of(index, gen);
            check(gxvk_homes::is_handle(h) && gxvk_homes::handle_index(h) == index && gxvk_homes::handle_gen(h) == gen, "handle packing", index, gen);
        }
    }
    check(!gxvk_homes::is_handle(0xa0000123u) && !gxvk_homes::is_handle(0), "another tag is no handle", 0, 0);

    std::mt19937_64 rng(4321);
    HomeIndex ix;
    std::vector<Live> live;
    std::uint32_t next_index = 0;
    // Two "heaps" of 256 pages each, far apart: buffers of 16 B .. 64 KiB at 16-byte steps, never overlapping a live one.
    const std::uint64_t heaps[2] = {0x200000000ull, 0x7f0000000ull};
    const auto overlaps = [&](std::uint64_t m, std::uint64_t n) {
        for (const Live& l : live) {
            if (l.memory < m + n && m < l.memory + l.bytes) return true;
        }
        return false;
    };
    std::uint64_t queries = 0, maybes = 0;
    for (int step = 0; step < 60000; ++step) {
        const int op = static_cast<int>(rng() % 10);
        if (op < 4 || live.empty()) {
            const std::uint64_t base = heaps[rng() % 2];
            const std::uint32_t bytes = rng() % 8 ? static_cast<std::uint32_t>(16 * (1 + rng() % 64)) : static_cast<std::uint32_t>(16 * (1 + rng() % 4096));
            const std::uint64_t memory = base + 16 * (rng() % (256 * 4096 / 16));
            if (overlaps(memory, bytes)) continue;
            ix.insert(next_index, memory, bytes);
            live.push_back({next_index++, memory, bytes});
        } else if (op < 6) {
            const std::size_t k = rng() % live.size();
            check(ix.erase(live[k].index, live[k].memory), "erase of a live home", live[k].memory, live[k].bytes);
            check(!ix.erase(live[k].index, live[k].memory), "a second erase does nothing", live[k].memory, live[k].bytes);
            live[k] = live.back();
            live.pop_back();
        } else {
            // A query: near a live home (its start, inside it, across its edge, just past it) or anywhere in a heap.
            std::uint64_t va, bytes;
            if (!live.empty() && rng() % 4) {
                const Live& l = live[rng() % live.size()];
                const int where = static_cast<int>(rng() % 5);
                va = where == 0 ? l.memory : where == 1 ? l.memory + 4 * (rng() % (l.bytes / 4)) : where == 2 ? l.memory + l.bytes - 4 : where == 3 ? l.memory + l.bytes : l.memory - 16;
                bytes = 4 * (1 + rng() % 64);
            } else {
                va = heaps[rng() % 2] + 4 * (rng() % (256 * 4096 / 4));
                bytes = rng() % 16 ? 4 * (1 + rng() % 256) : 4096 * (1 + rng() % 200);  // some long ranges: the map's
            }
            bool m = false, h = false, s = false;
            const Live* first = nullptr;
            model(live, va, bytes, &m, &h, &s, &first);
            HomeIndex::Hit hit;
            const bool im = ix.meets(va, bytes, &hit);
            ++queries;
            if (ix.may_meet(va, bytes)) ++maybes;
            check(im == m, "meets", va, bytes);
            check(!m || ix.may_meet(va, bytes), "the filter said no where a home is", va, bytes);
            check(ix.holding(va, bytes, nullptr) == h, "holding", va, bytes);
            check(ix.straddles(va, bytes) == s, "straddles", va, bytes);
            // The hit is a live home that meets the range (the one at or below va, else the lowest above it).
            if (im) {
                bool found = false;
                for (const Live& l : live) found |= l.index == hit.index && l.memory == hit.memory && l.bytes == hit.bytes;
                check(found && hit.memory < va + bytes && va < hit.memory + hit.bytes, "the hit is a live home meeting the range", va, bytes);
                if (h) check(va >= hit.memory && va + bytes <= hit.memory + hit.bytes, "holding names the home that holds it", va, bytes);
            }
        }
        check(ix.size() == live.size(), "size", 0, live.size());
    }
    // Every home erased: nothing meets anything, and the filter is back to "no".
    for (const Live& l : live) check(ix.erase(l.index, l.memory), "final erase", l.memory, l.bytes);
    for (int q = 0; q < 2000; ++q) {
        const std::uint64_t va = heaps[q % 2] + 16 * (rng() % (256 * 4096 / 16));
        check(!ix.meets(va, 64, nullptr) && !ix.may_meet(va, 64), "empty index", va, 64);
    }
    std::printf("gxvk_homes_test: %llu queries, the filter said maybe to %llu; %d failures\n", static_cast<unsigned long long>(queries),
                static_cast<unsigned long long>(maybes), g_failures);
    return g_failures ? 1 : 0;
}
