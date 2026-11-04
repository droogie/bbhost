// Stage 4b.3c (host/gxvk_homes.h): the command processor's index of the live
// homes by their buffers' memory, with a page filter in front of it - the part
// of the homes that needs no GPU (tests/gxvk_homes_test.cpp).
//
// Every lookup by address of a range a home may hold - a dispatch's constant
// buffer, a copy's source or destination, a device address, the CPU copies
// refused over a home's stale guest bytes - asks it, and nearly all meet no
// home: the filter keeps a count per 4 KiB page (2^20 counters, by a hash of
// the page) of the indexed homes touching it, so a miss reads a counter a page
// and never the map. A collision only sends a lookup to the map; a count that
// saturates stays saturated (the map answers that page from then on).
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>

namespace gxvk_homes {

class HomeIndex {
public:
    struct Hit {
        std::uint32_t index = 0;
        std::uint64_t memory = 0;
        std::uint32_t bytes = 0;
    };

    // Home `index` holds [memory, memory + bytes): whatever the index held at
    // that memory before is the caller's to have ended first.
    void insert(std::uint32_t index, std::uint64_t memory, std::uint32_t bytes) {
        if (!filter_) filter_.reset(new std::uint16_t[kSlots]());
        auto [it, fresh] = map_.emplace(memory, Value{index, bytes});
        if (!fresh) {
            count(it->first, it->second.bytes, -1);
            it->second = Value{index, bytes};
        }
        count(memory, bytes, 1);
        lo_ = std::min(lo_, memory);
        hi_ = std::max(hi_, memory + bytes);
        ++version_;
    }
    // Home `index` out of the index, if it is there at `memory`.
    bool erase(std::uint32_t index, std::uint64_t memory) {
        const auto it = map_.find(memory);
        if (it == map_.end() || it->second.index != index) return false;
        count(memory, it->second.bytes, -1);
        map_.erase(it);
        ++version_;
        return true;
    }
    // Whether [va, va + bytes) may meet a home: false only where none does.
    bool may_meet(std::uint64_t va, std::uint64_t bytes) const {
        if (!filter_ || !bytes || va >= hi_ || va + bytes <= lo_) return false;
        const std::uint64_t first = va >> 12, last = (va + bytes - 1) >> 12;
        if (last - first >= 64) return true;  // a long range: the map answers it
        for (std::uint64_t p = first; p <= last; ++p) {
            if (filter_[slot(p)]) return true;
        }
        return false;
    }
    // The home meeting [va, va + bytes): the one at or below va whose range
    // reaches past va, else the first starting inside the range.
    bool meets(std::uint64_t va, std::uint64_t bytes, Hit* out) const {
        if (!may_meet(va, bytes)) return false;
        auto it = map_.upper_bound(va);
        if (it != map_.begin()) {
            const auto p = std::prev(it);
            if (va < p->first + p->second.bytes) {
                if (out) *out = Hit{p->second.index, p->first, p->second.bytes};
                return true;
            }
        }
        if (it != map_.end() && it->first < va + bytes) {
            if (out) *out = Hit{it->second.index, it->first, it->second.bytes};
            return true;
        }
        return false;
    }
    // [va, va + bytes) inside one home.
    bool holding(std::uint64_t va, std::uint64_t bytes, Hit* out) const {
        Hit h;
        if (!meets(va, bytes, &h) || va < h.memory || va + bytes > h.memory + h.bytes) return false;
        if (out) *out = h;
        return true;
    }
    // [va, va + bytes) meets a home but is not inside one.
    bool straddles(std::uint64_t va, std::uint64_t bytes) const {
        Hit h;
        return meets(va, bytes, &h) && (va < h.memory || va + bytes > h.memory + h.bytes);
    }
    std::size_t size() const { return map_.size(); }
    // Moves at every insert and erase: an answer kept beside it is good while it stands.
    std::uint64_t version() const { return version_; }

private:
    static constexpr std::uint32_t kBits = 20, kSlots = 1u << kBits;
    struct Value {
        std::uint32_t index, bytes;
    };
    static std::uint32_t slot(std::uint64_t page) { return static_cast<std::uint32_t>((page * 0x9e3779b97f4a7c15ull) >> (64 - kBits)); }
    void count(std::uint64_t memory, std::uint32_t bytes, int d) {
        for (std::uint64_t p = memory >> 12; p <= (memory + std::max<std::uint32_t>(bytes, 1) - 1) >> 12; ++p) {
            std::uint16_t& c = filter_[slot(p)];
            if (c == 0xffff) continue;  // saturated: stays so
            c = static_cast<std::uint16_t>(d > 0 ? c + 1 : (c ? c - 1 : 0));
        }
    }
    std::map<std::uint64_t, Value> map_;
    std::unique_ptr<std::uint16_t[]> filter_;
    std::uint64_t lo_ = ~0ull, hi_ = 0;  // every indexed home lies inside
    std::uint64_t version_ = 0;
};

}  // namespace gxvk_homes
