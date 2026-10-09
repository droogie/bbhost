#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace bbhost::param_detail {

// Owned by a registered table. A new file/row-count registration owns a new cache.
struct MissIndex {
    std::mutex mutex;
    bool ready = false;
    std::unordered_map<std::uint32_t, std::uint64_t> first;
};

// Preserve the original binary-search hit, including which duplicate it chose.
// Modded directories need not be sorted. Index them only on a miss, without
// sorting the file or rescanning it on every missing-row/field lookup.
template <class Reader>
std::uint64_t lookup(std::uint32_t wanted, std::uint32_t count, Reader&& read, MissIndex& misses) {
    std::uint32_t lo = 0, hi = count;
    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;
        std::uint32_t id = 0;
        std::uint64_t data = 0;
        if (!read(mid, &id, &data)) return 0;
        if (id == wanted) return data;
        if (id < wanted) lo = mid + 1;
        else hi = mid;
    }
    std::lock_guard lock(misses.mutex);
    if (!misses.ready) {
        std::unordered_map<std::uint32_t, std::uint64_t> index;
        index.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            std::uint32_t id = 0;
            std::uint64_t data = 0;
            // Do not publish a partial index after an unreadable record.
            if (!read(i, &id, &data)) return 0;
            index.try_emplace(id, data);  // First directory entry when binary search missed.
        }
        misses.first = std::move(index);
        misses.ready = true;
    }
    const auto found = misses.first.find(wanted);
    return found == misses.first.end() ? 0 : found->second;
}

}  // namespace bbhost::param_detail
