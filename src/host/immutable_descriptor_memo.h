#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace gpu {
// Fieldwise contents, rather than Vk* struct bytes (which contain padding and
// pointers into scratch storage). Bindings are in descriptor-write order.
struct ImmutableDescriptorBinding {
    std::uint32_t binding = 0, array_element = 0, type = 0;
    std::uint64_t sampler = 0, view = 0;
    std::uint32_t image_layout = 0;
    std::uint64_t buffer = 0, offset = 0, range = 0;
    bool operator==(const ImmutableDescriptorBinding&) const = default;
};
struct ImmutableDescriptorKey {
    std::uint64_t layout = 0;
    std::vector<ImmutableDescriptorBinding> bindings;
    bool operator==(const ImmutableDescriptorKey&) const = default;
};
template <class Handle> std::uint64_t descriptor_handle(Handle handle) {
    if constexpr (std::is_pointer_v<Handle>) return reinterpret_cast<std::uintptr_t>(handle);
    else return static_cast<std::uint64_t>(handle);
}
struct ImmutableDescriptorHash {
    std::size_t operator()(const ImmutableDescriptorKey& key) const {
        std::size_t hash = 0;
        const auto add = [&](std::uint64_t value) {
            hash ^= std::hash<std::uint64_t>{}(value) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        };
        add(key.layout);
        for (const auto& b : key.bindings) {
            add(b.binding); add(b.array_element); add(b.type);
            add(b.sampler); add(b.view); add(b.image_layout);
            add(b.buffer); add(b.offset); add(b.range);
        }
        return hash;
    }
};
// One memo per resettable descriptor pool. Removing entries never frees or
// updates their sets: already recorded/in-flight commands retain them until
// the owning slot's fence permits the pool reset.
template <class Set, std::size_t Limit = 256> class ImmutableDescriptorMemo {
public:
    void observe_epoch(std::uint64_t epoch) {
        if (epoch_ != epoch) { clear(); epoch_ = epoch; }
    }
    Set find(const ImmutableDescriptorKey& key) const {
        const auto it = sets_.find(key);
        return it == sets_.end() ? Set{} : it->second;
    }
    void remember(const ImmutableDescriptorKey& key, Set set) {
        if (sets_.size() < Limit) sets_.try_emplace(key, set);
    }
    void forget_view(std::uint64_t view) {
        for (auto it = sets_.begin(); it != sets_.end();) {
            bool found = false;
            for (const auto& b : it->first.bindings) if (b.view == view) { found = true; break; }
            if (found) it = sets_.erase(it); else ++it;
        }
    }
    void clear() { sets_.clear(); }
    std::size_t size() const { return sets_.size(); }
private:
    std::uint64_t epoch_ = 0;
    std::unordered_map<ImmutableDescriptorKey, Set, ImmutableDescriptorHash> sets_;
};
}  // namespace gpu
