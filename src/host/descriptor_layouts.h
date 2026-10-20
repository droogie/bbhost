#pragma once
// gpu.cpp's descriptor set layouts, apart from the device so
// tests/descriptor_layouts_test.cpp can check them against MoltenVK's limits
// without a GPU.
//
// A per-stage limit counts a binding in every stage its stageFlags name:
// VK_SHADER_STAGE_ALL puts 64 samplers in each stage against MoltenVK's 16.
// Where old_layouts_over_limits says so, bindings name only the stages that
// read them (kGfxSetStages) and a shared set declares no more samplers than
// a stage may bind.

#include "gcn/translate.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace gpu {

// The images, samplers and storage images a shared set declares.
constexpr std::uint32_t kMaxImages = 32;

// Set 0 is the vertex shader's (and the domain and hull shaders'; the rect-list
// geometry shader reads none of it but is pre-rasterization), set 1 the pixel
// shader's.
constexpr VkShaderStageFlags kGfxSetStages[2] = {
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
        VK_SHADER_STAGE_GEOMETRY_BIT,
    VK_SHADER_STAGE_FRAGMENT_BIT};

inline VkShaderStageFlags gfx_set_stages(bool per_stage, int set) { return per_stage ? kGfxSetStages[set] : VK_SHADER_STAGE_ALL; }

// Not the control stage, which reads no constant buffer through a binding:
// MoltenVK puts a control stage's own buffers above every buffer binding its
// flags name, and with these 16 the tessellation levels found no Metal slot.
constexpr VkShaderStageFlags kCbPushStages[2] = {kGfxSetStages[0] & ~VkShaderStageFlags(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT),
                                                 kGfxSetStages[1]};

// MoltenVK binds 16 samplers a stage (Metal's limit).
inline std::uint32_t shared_set_samplers(bool per_stage, std::uint32_t max_per_stage_samplers) {
    return per_stage ? std::min(kMaxImages, max_per_stage_samplers) : kMaxImages;
}

inline std::vector<VkDescriptorSetLayoutBinding> shared_set_bindings(VkDescriptorType params, std::uint32_t samplers, bool buffers,
                                                                     VkShaderStageFlags stages) {
    std::vector<VkDescriptorSetLayoutBinding> binds;
    binds.push_back({gcn::kBindingParams, params, 1, stages, nullptr});
    for (std::uint32_t k = 0; k < kMaxImages; ++k) {
        binds.push_back({gcn::kBindingImage0 + k, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stages, nullptr});
        if (k < samplers) binds.push_back({gcn::kBindingSampler0 + k, VK_DESCRIPTOR_TYPE_SAMPLER, 1, stages, nullptr});
        binds.push_back({gcn::kBindingStorageImage0 + k, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stages, nullptr});
    }
    for (std::uint32_t k = 0; k < gcn::kMaxBuffers && buffers; ++k) {
        binds.push_back({gcn::kBindingStorageBuffer0 + k, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stages, nullptr});
    }
    return binds;
}

// A graphics stage's own set (render.cpp stage_set_layout).
inline std::vector<VkDescriptorSetLayoutBinding> stage_set_bindings(const gcn::TranslateResult& meta, VkDescriptorType params, bool buffers,
                                                                    VkShaderStageFlags stages) {
    std::vector<VkDescriptorSetLayoutBinding> binds;
    binds.push_back({gcn::kBindingParams, params, 1, stages, nullptr});
    for (const gcn::ImageBinding& b : meta.images) {
        if (meta.bindless) break;
        binds.push_back({b.binding, b.storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stages, nullptr});
    }
    for (const gcn::SamplerBinding& b : meta.samplers) {
        if (meta.bindless) break;
        binds.push_back({b.binding, VK_DESCRIPTOR_TYPE_SAMPLER, 1, stages, nullptr});
    }
    for (const gcn::BufferBinding& b : meta.buffers) {
        if (!buffers) break;
        binds.push_back({b.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stages, nullptr});
    }
    std::sort(binds.begin(), binds.end(), [](const VkDescriptorSetLayoutBinding& a, const VkDescriptorSetLayoutBinding& b) {
        return a.binding < b.binding;
    });
    binds.erase(std::unique(binds.begin(), binds.end(),
                            [](const VkDescriptorSetLayoutBinding& a, const VkDescriptorSetLayoutBinding& b) {
                                return a.binding == b.binding && a.descriptorType == b.descriptorType;
                            }),
                binds.end());
    return binds;
}

// Set 2 when the constant buffers are pushed: the vertex stage's 16, then the
// pixel stage's.
inline std::vector<VkDescriptorSetLayoutBinding> cb_push_bindings(bool per_stage) {
    std::vector<VkDescriptorSetLayoutBinding> binds;
    for (std::uint32_t k = 0; k < 2 * gcn::kMaxBuffers; ++k) {
        const int half = k < gcn::kMaxBuffers ? 0 : 1;
        binds.push_back({gcn::kBindingStorageBuffer0 + k, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                         per_stage ? kCbPushStages[half] : VK_SHADER_STAGE_ALL, nullptr});
    }
    return binds;
}

// Metal gives a stage 31 buffer slots. MoltenVK fills them from the bottom
// with an argument buffer per descriptor set (8) and every push-set buffer
// binding naming the stage, from the top with the vertex buffers, and puts
// its dynamic offsets just under those: a pipeline whose offsets find no free
// slot fails to build.
constexpr std::uint32_t kMetalStageBuffers = 31;
constexpr std::uint32_t kMvkArgumentBuffers = 8;

// Without argument buffers set 0's own buffers take their place, fewer than 8,
// so the count holds either way.
inline std::uint32_t mvk_vertex_buffer_room(const std::vector<VkDescriptorSetLayoutBinding>& push_binds) {
    std::uint32_t used = kMvkArgumentBuffers + 1;  // and the dynamic offsets
    for (const VkDescriptorSetLayoutBinding& b : push_binds) {
        const bool buffer = b.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || b.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                            b.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC ||
                            b.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        if (buffer && (b.stageFlags & VK_SHADER_STAGE_VERTEX_BIT)) used += b.descriptorCount;
    }
    return used < kMetalStageBuffers ? kMetalStageBuffers - used : 0;
}

// MoltenVK adds a binding for each attribute at a nonzero offset that ends
// past its binding's stride. `attribute_bytes[i]` is attribute i's size.
inline std::uint32_t mvk_vertex_input_slots(const std::vector<VkVertexInputBindingDescription>& bindings,
                                            const std::vector<VkVertexInputAttributeDescription>& attributes,
                                            const std::vector<std::uint32_t>& attribute_bytes) {
    std::uint32_t slots = 0;
    for (const VkVertexInputBindingDescription& b : bindings) {
        slots = std::max(slots, b.binding + 1);
        for (std::size_t i = 0; i < attributes.size() && i < attribute_bytes.size(); ++i) {
            const VkVertexInputAttributeDescription& a = attributes[i];
            if (a.binding == b.binding && a.offset && a.offset + attribute_bytes[i] > b.stride) ++slots;
        }
    }
    return slots;
}

// An update-after-bind set answers to other limits, not checked here.
struct LayoutSet {
    const std::vector<VkDescriptorSetLayoutBinding>* binds = nullptr;
    bool update_after_bind = false;
};

// What is over the device's limits; empty where all fits.
inline std::string layout_over_limits(const std::vector<LayoutSet>& sets, const VkPhysicalDeviceLimits& l) {
    struct Counts {
        std::uint32_t samplers = 0, uniform = 0, uniform_dynamic = 0, storage = 0, storage_dynamic = 0, sampled = 0, storage_images = 0,
                      resources = 0;
        void add(VkDescriptorType t, std::uint32_t n) {
            switch (t) {
            case VK_DESCRIPTOR_TYPE_SAMPLER: samplers += n; return;  // not a resource
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: samplers += n; sampled += n; break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: sampled += n; break;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: storage_images += n; break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: uniform += n; break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: uniform += n; uniform_dynamic += n; break;
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: storage += n; break;
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: storage += n; storage_dynamic += n; break;
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: break;
            default: return;
            }
            resources += n;
        }
    };
    static const struct {
        VkShaderStageFlagBits bit;
        const char* name;
    } kStages[] = {{VK_SHADER_STAGE_VERTEX_BIT, "vertex"},
                   {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, "tessellation control"},
                   {VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, "tessellation evaluation"},
                   {VK_SHADER_STAGE_GEOMETRY_BIT, "geometry"},
                   {VK_SHADER_STAGE_FRAGMENT_BIT, "fragment"},
                   {VK_SHADER_STAGE_COMPUTE_BIT, "compute"}};
    std::string out;
    const auto over = [&out](const char* where, const char* what, std::uint32_t n, std::uint32_t limit) {
        if (n <= limit) return;
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s%s: %u %s against %u", out.empty() ? "" : "; ", where, n, what, limit);
        out += buf;
    };
    Counts total;
    for (const LayoutSet& s : sets) {
        if (!s.binds || s.update_after_bind) continue;
        for (const VkDescriptorSetLayoutBinding& b : *s.binds) total.add(b.descriptorType, b.descriptorCount);
    }
    for (const auto& stage : kStages) {
        Counts c;
        for (const LayoutSet& s : sets) {
            if (!s.binds || s.update_after_bind) continue;
            for (const VkDescriptorSetLayoutBinding& b : *s.binds) {
                if (b.stageFlags & stage.bit) c.add(b.descriptorType, b.descriptorCount);
            }
        }
        over(stage.name, "samplers", c.samplers, l.maxPerStageDescriptorSamplers);
        over(stage.name, "uniform buffers", c.uniform, l.maxPerStageDescriptorUniformBuffers);
        over(stage.name, "storage buffers", c.storage, l.maxPerStageDescriptorStorageBuffers);
        over(stage.name, "sampled images", c.sampled, l.maxPerStageDescriptorSampledImages);
        over(stage.name, "storage images", c.storage_images, l.maxPerStageDescriptorStorageImages);
        over(stage.name, "resources", c.resources, l.maxPerStageResources);
    }
    over("layout", "samplers", total.samplers, l.maxDescriptorSetSamplers);
    over("layout", "uniform buffers", total.uniform, l.maxDescriptorSetUniformBuffers);
    over("layout", "dynamic uniform buffers", total.uniform_dynamic, l.maxDescriptorSetUniformBuffersDynamic);
    over("layout", "storage buffers", total.storage, l.maxDescriptorSetStorageBuffers);
    over("layout", "dynamic storage buffers", total.storage_dynamic, l.maxDescriptorSetStorageBuffersDynamic);
    over("layout", "sampled images", total.sampled, l.maxDescriptorSetSampledImages);
    over("layout", "storage images", total.storage_images, l.maxDescriptorSetStorageImages);
    return out;
}

// The layouts with every binding in every stage; empty where the device takes
// them. Bindless only takes bindings out, so it is left out.
inline std::string old_layouts_over_limits(VkDescriptorType params, bool push, const VkPhysicalDeviceLimits& l) {
    const std::vector<VkDescriptorSetLayoutBinding> shared = shared_set_bindings(params, kMaxImages, !push, VK_SHADER_STAGE_ALL);
    const std::vector<VkDescriptorSetLayoutBinding> cbs = push ? cb_push_bindings(false) : std::vector<VkDescriptorSetLayoutBinding>{};
    const std::string compute = layout_over_limits({{&shared}}, l);
    const std::string graphics = layout_over_limits({{&shared}, {&shared}, {&cbs}}, l);
    std::string out;
    if (!compute.empty()) out = "compute layout, " + compute;
    if (!graphics.empty()) out += (out.empty() ? "graphics layout, " : "; graphics layout, ") + graphics;
    return out;
}

}  // namespace gpu
