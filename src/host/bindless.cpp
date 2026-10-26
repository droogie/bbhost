// Bindless views and samplers (BBHOST_BINDLESS=1).
//
// A graphics stage used to take a descriptor set of its own for every draw
// whose images or samplers differed from a set made before: a material list
// of its views and samplers, hashed and looked up, and on a miss a set
// allocated and written (render.cpp). D3D12-era hosts do not work that way:
// there is one array of image descriptors and one of samplers for the whole
// run, every view and sampler has a slot in it for as long as it lives, and a
// draw names slots. That is what this is:
//
//   set 3, binding kBindlessImages          sampled images, every image type
//                  kBindlessStorageImages   storage images
//                  kBindlessSamplers        samplers
//
// A translated stage (TranslateOptions::bindless) declares one alias of the
// image array per image type it reads and takes each binding's slot from its
// params block (StageParams::image_index / sampler_index); the renderer puts
// a view of the binding's type at that slot (stage_final_views already makes
// the view match what the shader declares, with a dummy of the right shape
// when the texture does not). Stage sets then hold the params block alone.
//
// Slots: a view gets one the first time a draw binds it and keeps it until
// the view is retired (defer_destroy_view); the slot comes free when that
// submission's views are destroyed, after its fence, so no work in flight can
// read a slot being rewritten (UPDATE_UNUSED_WHILE_PENDING). Samplers are
// never destroyed (sampler_for caches them), so their slots are permanent.
// Slot 0 of each array holds a dummy, for a params block a draw left unset.
#include "host/gpu_internal.h"
#include "log.h"

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu {

namespace {

struct SlotArray {
    std::uint32_t capacity = 0;
    std::uint32_t next = 1;  // slot 0: the dummy
    std::vector<std::uint32_t> free;
    std::unordered_map<VkImageView, std::uint32_t> by_view;
    std::uint64_t assigned = 0, released = 0, full = 0;
};
SlotArray g_images, g_storage;
std::unordered_map<VkSampler, std::uint32_t> g_samplers;
std::uint32_t g_sampler_capacity = 0, g_sampler_next = 1;
std::uint64_t g_samplers_full = 0;
// Per submission slot: the image slots its retired views held.
std::vector<std::pair<bool, std::uint32_t>> g_retired[kSlots];
// Direct-mapped fronts of the maps: a draw looks up every view and sampler it
// binds, and an unordered_map find each cost about what the set cache did.
// A retired view's entry is cleared with its map entry.
constexpr std::uint32_t kFront = 4096;
struct FrontEntry {
    std::uint64_t handle = 0;
    std::uint32_t slot = 0;
};
FrontEntry g_front[3][kFront];  // [0] images, [1] storage images, [2] samplers
std::uint32_t front_index(std::uint64_t handle) { return static_cast<std::uint32_t>((handle * 0x9e3779b97f4a7c15ull) >> 52) & (kFront - 1); }
template <typename H>
std::uint64_t handle_bits(H h) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(h));
}

void write_image(bool storage, std::uint32_t slot, VkImageView view) {
    // A depth target in a layout of its own is sampled read-only (render.cpp, depth layouts).
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, view, storage ? VK_IMAGE_LAYOUT_GENERAL : sampled_view_layout(view)};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = g.bindless_set;
    w.dstBinding = storage ? gcn::kBindlessStorageImages : gcn::kBindlessImages;
    w.dstArrayElement = slot;
    w.descriptorCount = 1;
    w.descriptorType = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(g.device, 1, &w, 0, nullptr);
}

void write_sampler(std::uint32_t slot, VkSampler sampler) {
    VkDescriptorImageInfo ii{sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = g.bindless_set;
    w.dstBinding = gcn::kBindlessSamplers;
    w.dstArrayElement = slot;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(g.device, 1, &w, 0, nullptr);
}

}  // namespace

bool bindless_wanted() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_BINDLESS");
        return e && e[0] == '1';
    }();
    return on;
}

bool bindless_create_locked(std::uint32_t max_images, std::uint32_t max_samplers) {
    g_images.capacity = std::min<std::uint32_t>(max_images, 1u << 18);
    g_storage.capacity = std::min<std::uint32_t>(max_images, 1u << 14);
    g_sampler_capacity = std::min<std::uint32_t>(max_samplers, 1u << 12);
    const VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                           VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    const VkDescriptorSetLayoutBinding binds[3] = {
        {gcn::kBindlessImages, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, g_images.capacity, VK_SHADER_STAGE_ALL, nullptr},
        {gcn::kBindlessStorageImages, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, g_storage.capacity, VK_SHADER_STAGE_ALL, nullptr},
        {gcn::kBindlessSamplers, VK_DESCRIPTOR_TYPE_SAMPLER, g_sampler_capacity, VK_SHADER_STAGE_ALL, nullptr},
    };
    const VkDescriptorBindingFlags binding_flags[3] = {flags, flags, flags};
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bfi.bindingCount = 3;
    bfi.pBindingFlags = binding_flags;
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.pNext = &bfi;
    lci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    lci.bindingCount = 3;
    lci.pBindings = binds;
    if (vkCreateDescriptorSetLayout(g.device, &lci, nullptr, &g.bindless_layout) != VK_SUCCESS) {
        host_log("gpu: bindless: the global set's layout failed; off");
        return false;
    }
    const VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, g_images.capacity},
                                           {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, g_storage.capacity},
                                           {VK_DESCRIPTOR_TYPE_SAMPLER, g_sampler_capacity}};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pci.maxSets = 1;
    pci.poolSizeCount = 3;
    pci.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(g.device, &pci, nullptr, &g.bindless_pool) != VK_SUCCESS) {
        host_log("gpu: bindless: the global set's pool failed; off");
        return false;
    }
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = g.bindless_pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &g.bindless_layout;
    if (vkAllocateDescriptorSets(g.device, &ai, &g.bindless_set) != VK_SUCCESS) {
        host_log("gpu: bindless: the global set failed; off");
        return false;
    }
    host_log("gpu: bindless: one global set - %u sampled images, %u storage images, %u samplers (set 3)", g_images.capacity,
             g_storage.capacity, g_sampler_capacity);
    return true;
}

void bindless_dummies_locked(VkImageView image, VkImageView storage, VkSampler sampler) {
    if (!g.bindless) return;
    if (image) write_image(false, 0, image);
    if (storage) write_image(true, 0, storage);
    if (sampler) write_sampler(0, sampler);
}

std::uint32_t bindless_view_slot_locked(VkImageView view, bool storage) {
    if (!view) return 0;
    const std::uint64_t bits = handle_bits(view);
    FrontEntry& f = g_front[storage ? 1 : 0][front_index(bits)];
    if (f.handle == bits) return f.slot;
    SlotArray& a = storage ? g_storage : g_images;
    if (const auto it = a.by_view.find(view); it != a.by_view.end()) {
        f = {bits, it->second};
        return it->second;
    }
    std::uint32_t slot;
    if (!a.free.empty()) {
        slot = a.free.back();
        a.free.pop_back();
    } else if (a.next < a.capacity) {
        slot = a.next++;
    } else {
        if (a.full++ == 0) host_log("gpu: bindless: the %s array is full; views past it read the dummy", storage ? "storage-image" : "image");
        return 0;
    }
    write_image(storage, slot, view);
    a.by_view.emplace(view, slot);
    f = {bits, slot};
    ++a.assigned;
    return slot;
}

std::uint32_t bindless_sampler_slot_locked(VkSampler sampler) {
    if (!sampler) return 0;
    const std::uint64_t bits = handle_bits(sampler);
    FrontEntry& f = g_front[2][front_index(bits)];
    if (f.handle == bits) return f.slot;
    if (const auto it = g_samplers.find(sampler); it != g_samplers.end()) {
        f = {bits, it->second};
        return it->second;
    }
    if (g_sampler_next >= g_sampler_capacity) {
        if (g_samplers_full++ == 0) host_log("gpu: bindless: the sampler array is full; samplers past it read the dummy");
        return 0;
    }
    const std::uint32_t slot = g_sampler_next++;
    write_sampler(slot, sampler);
    g_samplers.emplace(sampler, slot);
    f = {bits, slot};
    return slot;
}

void bindless_view_retired_locked(VkImageView view) {
    if (!g.bindless || !view) return;
    const std::uint64_t bits = handle_bits(view);
    for (const bool storage : {false, true}) {
        FrontEntry& f = g_front[storage ? 1 : 0][front_index(bits)];
        if (f.handle == bits) f = FrontEntry{};
        SlotArray& a = storage ? g_storage : g_images;
        const auto it = a.by_view.find(view);
        if (it == a.by_view.end()) continue;
        g_retired[g.slot].emplace_back(storage, it->second);
        a.by_view.erase(it);
    }
}

void bindless_slot_done_locked(int slot) {
    if (!g.bindless) return;
    for (const auto& [storage, s] : g_retired[slot]) {
        SlotArray& a = storage ? g_storage : g_images;
        a.free.push_back(s);
        ++a.released;
    }
    g_retired[slot].clear();
}

std::string bindless_report() {
    if (!g.bindless) return {};
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "bindless: image slots %llu assigned, %llu released, %zu live (high water %u of %u); storage %zu live; samplers %zu; "
                  "refused (arrays full) %llu, %llu, %llu",
                  static_cast<unsigned long long>(g_images.assigned), static_cast<unsigned long long>(g_images.released),
                  g_images.by_view.size(), g_images.next, g_images.capacity, g_storage.by_view.size(), g_samplers.size(),
                  static_cast<unsigned long long>(g_images.full), static_cast<unsigned long long>(g_storage.full),
                  static_cast<unsigned long long>(g_samplers_full));
    return buf;
}

}  // namespace gpu
