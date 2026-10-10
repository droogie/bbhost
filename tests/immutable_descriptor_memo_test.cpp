#undef NDEBUG
#include "host/immutable_descriptor_memo.h"

#include <cassert>
#include <cstdio>

using namespace gpu;

int main() {
    ImmutableDescriptorKey key{11, {{1, 0, 7, 21, 31, 1, 41, 51, 61}}};
    ImmutableDescriptorMemo<std::uint64_t, 2> slot_a, slot_b;
    slot_a.observe_epoch(1);
    slot_b.observe_epoch(1);
    slot_a.remember(key, 101);
    assert(slot_a.find(key) == 101);
    assert(slot_b.find(key) == 0);  // same contents in another pool need their own set
    slot_a.remember(key, 102);
    assert(slot_a.find(key) == 101);  // recorded immutable entries cannot be replaced

    const auto misses = [&](const ImmutableDescriptorKey& changed) {
        assert(changed != key);
        assert(slot_a.find(changed) == 0);
    };
    auto changed = key;
    ++changed.layout; misses(changed);
    for (int field = 0; field < 9; ++field) {
        changed = key;
        auto& b = changed.bindings[0];
        switch (field) {
            case 0: ++b.binding; break;
            case 1: ++b.array_element; break;
            case 2: ++b.type; break;
            case 3: ++b.sampler; break;
            case 4: ++b.view; break;
            case 5: ++b.image_layout; break;
            case 6: ++b.buffer; break;
            case 7: ++b.offset; break;  // individual non-dynamic params slice
            case 8: ++b.range; break;
        }
        misses(changed);
    }
    changed = key;
    changed.bindings.push_back(key.bindings[0]);
    misses(changed);

    changed = key; ++changed.bindings[0].view;
    slot_a.remember(changed, 102);
    auto overflow = key; ++overflow.layout;
    slot_a.remember(overflow, 103);
    assert(slot_a.size() == 2 && slot_a.find(overflow) == 0);
    slot_a.forget_view(key.bindings[0].view);
    assert(slot_a.find(key) == 0 && slot_a.find(changed) == 102);
    slot_a.remember(key, 104);  // simulated same numeric handle, new view lifetime
    assert(slot_a.find(key) == 104);

    slot_b.remember(key, 201);
    slot_a.clear();  // fence completed; the owning descriptor pool resets
    assert(slot_a.find(key) == 0 && slot_b.find(key) == 201);
    slot_a.remember(key, 105);
    slot_a.observe_epoch(2);  // image/view destruction invalidates older lifetimes
    assert(slot_a.size() == 0);
    slot_a.remember(key, 106);
    slot_a.observe_epoch(2);
    assert(slot_a.find(key) == 106);
    std::puts("immutable descriptor memo tests passed");
}
