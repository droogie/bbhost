// SPDX-License-Identifier: GPL-2.0-or-later
// bbhost: object motion vectors history tracking for temporal upscaling and Frame Generation.
// Adapted from Supermedo's bbport implementation (shadps4/video_core/renderer_vulkan/motion_history.h).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gpu::motion {

struct VertexRange {
    uint32_t first{};
    uint32_t count{};
    bool operator==(const VertexRange&) const = default;
};

// Reserve only the vertices referenced by this draw, not the whole shared vertex buffer.
// Primitive restart markers are not vertex shader invocations. Rejects signed overflow.
template <class Index, size_t Extent = std::dynamic_extent>
VertexRange IndexedRange(std::span<const Index, Extent> indices, int32_t base_vertex, bool restart) {
    uint32_t lo = UINT32_MAX, hi = 0;
    for (const auto index : indices) {
        if (restart && index == std::numeric_limits<Index>::max()) {
            continue;
        }
        lo = std::min(lo, static_cast<uint32_t>(index));
        hi = std::max(hi, static_cast<uint32_t>(index));
    }
    if (lo > hi || static_cast<int64_t>(lo) + base_vertex < 0 ||
        static_cast<int64_t>(hi) + base_vertex > INT32_MAX) {
        return {};
    }
    return {static_cast<uint32_t>(static_cast<int64_t>(lo) + base_vertex), hi - lo + 1};
}

// Vertex shader constant buffers (stride 16) of Bloodborne's G-buffer shaders: the 864-byte
// scene constants (camera), 416-byte model constants and palettes of 3x4 bone matrices
// (48 bytes per bone). Characters have 656+ bytes; weapons and props have 2-8 bones (96-384 bytes).
enum class BufferRole { Other, Skeleton, SmallSkeleton };
inline BufferRole ClassifyBuffer(uint32_t size) {
    if (size == 864 || size == 416 || size > 16384) {
        return BufferRole::Other;
    }
    if (size >= 640) {
        return BufferRole::Skeleton;
    }
    return (size >= 96 && size % 48 == 0) ? BufferRole::SmallSkeleton : BufferRole::Other;
}

// Scanning and hashing the index list of every stored draw each frame is costly on CPU.
// Static meshes keep their index buffers, so a result is cached and re-verified periodically.
class IndexRangeCache {
public:
    static constexpr uint64_t Revalidate = 32, Unused = 600;
    struct Key {
        uint64_t address{};
        uint32_t count{}, index_size{};
        int32_t base_vertex{};
        bool restart{};
        bool operator==(const Key&) const = default;
    };
    struct Result {
        VertexRange range;
        uint64_t topology{};
    };
    struct Stats {
        uint64_t hits{}, scans{}, stale{};
    } stats;

    template <class Scan>
    Result Get(const Key& key, uint64_t frame, Scan&& scan) {
        auto& entry = entries[key];
        entry.last_use = frame;
        if (entry.valid && frame - entry.verified < Revalidate) {
            ++stats.hits;
            return entry.result;
        }
        ++stats.scans;
        const Result fresh = scan();
        if (entry.valid && (fresh.topology != entry.result.topology ||
                            !(fresh.range == entry.result.range))) {
            ++stats.stale;
        }
        entry.result = fresh;
        entry.verified = frame;
        entry.valid = true;
        return entry.result;
    }
    void Trim(uint64_t frame) {
        std::erase_if(entries, [frame](const auto& e) { return frame - e.second.last_use > Unused; });
    }
    size_t Size() const { return entries.size(); }
    void Clear() { entries.clear(); stats = {}; }

private:
    struct Entry {
        Result result;
        uint64_t verified{}, last_use{};
        bool valid{};
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            uint64_t h = k.address * 0x9e3779b97f4a7c15ULL;
            h ^= (uint64_t(k.count) << 32 | uint32_t(k.base_vertex)) + (h << 6) + (h >> 2);
            h ^= uint64_t(k.index_size << 1 | uint32_t(k.restart)) + (h << 6) + (h >> 2);
            return static_cast<size_t>(h);
        }
    };
    std::unordered_map<Key, Entry, KeyHash> entries;
};

// Generational open-addressing map for per-frame draw allocations.
template <class Key, class Value, class Hash>
class FlatMap {
public:
    FlatMap() : slots(64) {}

    Value& operator[](const Key& key) {
        return Insert(key).first->value;
    }
    void emplace(const Key& key, const Value& value) {
        const auto [slot, inserted] = Insert(key);
        if (inserted) {
            slot->value = value;
        }
    }
    const Value* find(const Key& key) const {
        const size_t mask = slots.size() - 1;
        for (size_t i = Start(key) & mask;; i = (i + 1) & mask) {
            const Slot& slot = slots[i];
            if (slot.generation != generation) {
                return nullptr;
            }
            if (slot.key == key) {
                return &slot.value;
            }
        }
    }
    void clear() {
        size = 0;
        if (++generation == 0) {
            for (auto& slot : slots) {
                slot.generation = 0;
            }
            generation = 1;
        }
    }
    void swap(FlatMap& other) noexcept {
        slots.swap(other.slots);
        std::swap(size, other.size);
        std::swap(generation, other.generation);
    }

private:
    struct Slot {
        Key key{};
        Value value{};
        uint32_t generation = 0;
    };
    static size_t Start(const Key& key) {
        const uint64_t h = uint64_t(Hash{}(key)) * 0x9e3779b97f4a7c15ULL;
        return static_cast<size_t>(h ^ (h >> 29));
    }
    std::pair<Slot*, bool> Insert(const Key& key) {
        if ((size + 1) * 2 > slots.size()) {
            Grow();
        }
        const size_t mask = slots.size() - 1;
        for (size_t i = Start(key) & mask;; i = (i + 1) & mask) {
            Slot& slot = slots[i];
            if (slot.generation != generation) {
                slot = {key, Value{}, generation};
                ++size;
                return {&slot, true};
            }
            if (slot.key == key) {
                return {&slot, false};
            }
        }
    }
    void Grow() {
        std::vector<Slot> old(slots.size() * 2);
        old.swap(slots);
        const uint32_t live = generation;
        generation = 1;
        size = 0;
        for (const auto& slot : old) {
            if (slot.generation == live) {
                *Insert(slot.key).first = {slot.key, slot.value, generation};
            }
        }
    }

    std::vector<Slot> slots;
    size_t size = 0;
    uint32_t generation = 1;
};

// Flags for per-draw parameters passed to shaders.
constexpr uint32_t FlagStore = 1;
constexpr uint32_t FlagLoad = 2;

// Standard capacity constants:
// 4M vertices per frame half * 16 bytes = 64 MiB per half (128 MiB total device buffer).
constexpr uint32_t PositionsPerFrame = 4194304;

struct Draw {
    uint64_t shader{}, geometry{}, indices{}, topology{};
    uint32_t index_count{}, instances{}, first_instance{};
    VertexRange vertices;
    bool operator==(const Draw& rhs) const = default;
};

class History {
public:
    explicit History(uint32_t capacity = PositionsPerFrame) : capacity{capacity} {}

    struct Allocation {
        uint32_t store{}, load{}, vertices{}, instances{}, first_vertex{}, first_instance{};
        uint32_t flags{};
    };
    struct Stats {
        uint64_t draws{}, stored{}, loaded{}, unmatched{}, exhausted{}, invalid{}, still{};
    } stats;

    struct GateKey {
        uint64_t shader{}, geometry{}, indices{};
        uint32_t index_count{}, instances{}, first_instance{}, occurrence{};
        bool operator==(const GateKey&) const = default;
    };

    // Small skeletons (weapons, props): only allocate history if bone palette changed.
    bool Moving(GateKey key, uint64_t palette) {
        key.occurrence = 0;
        key.occurrence = gate_occurrences[key]++;
        const auto* prev = gate_previous.find(key);
        const bool moving = prev && *prev != palette;
        gate_current.emplace(key, palette);
        stats.still += moving ? 0 : 1;
        return moving;
    }

    void NextFrame() {
        previous.swap(current);
        current.clear();
        occurrences.clear();
        gate_previous.swap(gate_current);
        gate_current.clear();
        gate_occurrences.clear();
        ++frame;
        used = 0;
    }

    // Two consecutive frame advances clear both current and previous generations,
    // preserving the parity of (frame & 1) relative to GPU ping-pong halves.
    void Invalidate() {
        NextFrame();
        NextFrame();
    }

    Allocation Prepare(const Draw& draw) {
        ++stats.draws;
        Key key{draw, 0};
        key.occurrence = occurrences[key]++;
        const uint64_t count = uint64_t(draw.vertices.count) * draw.instances;
        if (!count || uint64_t(draw.vertices.first) + draw.vertices.count - 1 > INT32_MAX ||
            uint64_t(draw.first_instance) + draw.instances > UINT32_MAX) {
            ++stats.invalid;
            return {};
        }
        const auto* prev = previous.find(key);
        Allocation result{0, 0, draw.vertices.count, draw.instances,
                          draw.vertices.first, draw.first_instance, 0};
        if (prev) {
            result.load = *prev;
            result.flags |= FlagLoad;
            ++stats.loaded;
        } else {
            ++stats.unmatched;
        }
        if (count <= capacity - used) {
            result.store = 1 + uint32_t(frame & 1) * capacity + used;
            result.flags |= FlagStore;
            used += static_cast<uint32_t>(count);
            current.emplace(key, result.store);
            ++stats.stored;
        } else {
            ++stats.exhausted;
        }
        return result;
    }

    uint32_t Used() const { return used; }
    uint64_t CurrentFrame() const { return frame; }
    uint32_t Capacity() const { return capacity; }

private:
    struct Key {
        Draw draw;
        uint32_t occurrence;
        bool operator==(const Key&) const = default;
    };
    struct Hash {
        size_t operator()(const Key& key) const {
            uint64_t h = 0;
            const auto mix = [&](uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
            const auto& d = key.draw;
            mix(d.shader); mix(d.geometry); mix(d.indices); mix(d.topology);
            mix(d.index_count); mix(d.instances); mix(d.first_instance);
            mix(d.vertices.first); mix(d.vertices.count); mix(key.occurrence);
            return static_cast<size_t>(h);
        }
    };
    uint32_t capacity{}, used{};
    uint64_t frame{};
    FlatMap<Key, uint32_t, Hash> previous, current, occurrences;
    struct GateHash {
        size_t operator()(const GateKey& key) const {
            uint64_t h = 0;
            const auto mix = [&](uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
            mix(key.shader); mix(key.geometry); mix(key.indices); mix(key.index_count);
            mix(key.instances); mix(key.first_instance); mix(key.occurrence);
            return static_cast<size_t>(h);
        }
    };
    FlatMap<GateKey, uint64_t, GateHash> gate_previous, gate_current;
    FlatMap<GateKey, uint32_t, GateHash> gate_occurrences;
};

} // namespace gpu::motion
