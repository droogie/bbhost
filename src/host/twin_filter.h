// The device-local twins' lookups (host/gpu.cpp), the parts that need no GPU:
// the maps of live twins and armed candidates' ranges, the range tests the
// renderer asks of them, and the filter that answers a lookup meeting none
// of them without walking either map.
//
// Every buffer binding of every draw asks the twins first (twin_locate_locked
// and kin) whether a twin serves it. With BBHOST_PARTICLE_TWINS on, a few
// dozen GPU particle buffers are twins or armed candidates, and nearly every
// lookup misses them - but each miss walked two red-black trees, the live
// twins' and the armed ranges', after two counter checks: ~0.5 us a draw on
// the command processor at 1,200 draws a frame. The filter keeps their
// ranges' union as one sorted array, rebuilt when either map's shape moves
// (each counts its inserts and erases: ShapedMap), and a lookup outside it
// gets the answer the maps would give - "no twin, refused by the armed
// filter" - from its bounds and a binary search over a cache line or two.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

namespace twins {

// A map by guest address whose inserts and erases count in shape(): what a
// filter built from it must be rebuilt for. A value may change through an
// iterator, never the range it stands for (the key, and a twin's bytes).
template <class V>
class ShapedMap {
public:
    using Map = std::map<std::uint64_t, V>;
    using iterator = typename Map::iterator;
    using const_iterator = typename Map::const_iterator;

    const_iterator begin() const { return m_.begin(); }
    const_iterator end() const { return m_.end(); }
    iterator begin() { return m_.begin(); }
    iterator end() { return m_.end(); }
    const_iterator find(std::uint64_t k) const { return m_.find(k); }
    iterator find(std::uint64_t k) { return m_.find(k); }
    const_iterator upper_bound(std::uint64_t k) const { return m_.upper_bound(k); }
    std::size_t count(std::uint64_t k) const { return m_.count(k); }
    bool empty() const { return m_.empty(); }
    std::size_t size() const { return m_.size(); }

    template <class... A>
    std::pair<iterator, bool> emplace(A&&... a) {
        ++shape_;
        return m_.emplace(std::forward<A>(a)...);
    }
    iterator erase(const_iterator it) {
        ++shape_;
        return m_.erase(it);
    }
    void assign(std::uint64_t k, const V& v) {
        ++shape_;
        m_[k] = v;
    }
    std::uint64_t shape() const { return shape_; }

private:
    Map m_;
    std::uint64_t shape_ = 0;
};

// [va, va + bytes) inside the live twin at or below va (each value has its
// guest range's .bytes): the twin that serves it, else end().
template <class LiveMap>
typename LiveMap::const_iterator live_holding(const LiveMap& live, std::uint64_t va, std::uint64_t bytes) {
    auto lt = live.upper_bound(va);
    if (lt == live.begin()) return live.end();
    --lt;
    return va >= lt->first && va + bytes <= lt->first + lt->second.bytes ? lt : live.end();
}

// [va, va + bytes) meets a live twin's range.
template <class LiveMap>
bool live_meets(const LiveMap& live, std::uint64_t va, std::uint64_t bytes) {
    auto it = live.upper_bound(va);
    return (it != live.begin() && va < std::prev(it)->first + std::prev(it)->second.bytes) || (it != live.end() && it->first < va + bytes);
}

// [va, va + bytes) meets one of the ranges (start -> end): the armed filter.
template <class RangeMap>
bool ranges_meet(const RangeMap& ranges, std::uint64_t va, std::uint64_t bytes) {
    auto it = ranges.upper_bound(va);
    if (it != ranges.begin() && va < std::prev(it)->second) return true;
    return it != ranges.end() && it->first < va + bytes;
}

// The union of the live twins' ranges and the armed ranges, merged and
// sorted. meets() is false only where live_meets and ranges_meet both are
// (it may be true where they are not: an empty range counts as one byte,
// touching ranges merge), so a "no" from it is theirs.
class MissFilter {
public:
    // Rebuilt when either map's shape has moved since the last build.
    template <class LiveMap, class RangeMap>
    void refresh(const LiveMap& live, const RangeMap& armed) {
        const std::uint64_t shape = live.shape() + armed.shape();  // both only grow: any change moves the sum
        if (built_ && shape == shape_) return;
        spans_.clear();
        for (const auto& [m, t] : live) spans_.push_back({m, m + std::max<std::uint64_t>(t.bytes, 1)});
        for (const auto& [m, e] : armed) spans_.push_back({m, std::max<std::uint64_t>(e, m + 1)});
        std::sort(spans_.begin(), spans_.end());
        starts_.clear();
        ends_.clear();
        for (const auto& [s, e] : spans_) {
            if (!ends_.empty() && s <= ends_.back()) {
                ends_.back() = std::max(ends_.back(), e);
            } else {
                starts_.push_back(s);
                ends_.push_back(e);
            }
        }
        lo_ = starts_.empty() ? ~0ull : starts_.front();
        hi_ = ends_.empty() ? 0 : ends_.back();
        shape_ = shape;
        built_ = true;
        ++builds_;
    }
    // [va, va + bytes) meets the union (bytes > 0).
    bool meets(std::uint64_t va, std::uint64_t bytes) const {
        const std::uint64_t end = va + bytes;
        if (end <= lo_ || va >= hi_) return false;
        // The first span ending past va (there is one: va < hi_), the only one
        // that can hold va or start before end - found without branches, which
        // between a few dozen spans would mispredict at every step.
        const std::uint64_t* e = ends_.data();
        std::size_t n = ends_.size();
        while (n > 1) {
            const std::size_t half = n / 2;
            e = e[half] <= va ? e + half : e;
            n -= half;
        }
        e += *e <= va;
        return starts_[static_cast<std::size_t>(e - ends_.data())] < end;
    }
    std::size_t spans() const { return starts_.size(); }
    std::uint64_t builds() const { return builds_; }

private:
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans_;
    std::vector<std::uint64_t> starts_, ends_;
    std::uint64_t lo_ = ~0ull, hi_ = 0, shape_ = 0, builds_ = 0;
    bool built_ = false;
};

}  // namespace twins
