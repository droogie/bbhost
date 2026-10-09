#pragma once
// The descriptor set layouts gpu.cpp makes at device creation, as binding
// lists, and a pipeline layout made of them held to a device's limits - apart
// from the device, so tests/descriptor_layouts_test.cpp can check every one
// against MoltenVK's limits on a machine without a GPU.
//
// A device's per-stage limits count a binding in every stage its stageFlags
// name. Every binding is VK_SHADER_STAGE_ALL, so each graphics stage counts
// both stage sets: 64 samplers a stage against MoltenVK's 16, and the pushed
// constant buffers' 32 storage buffers against its 31. There
// (old_layouts_over_limits, on MoltenVK) the layouts are per stage instead:
// each graphics binding names the stages that read it (kGfxSetStages), and a
// shared set declares no more samplers than a stage may bind. Every other
// driver keeps them as they are; gpu.cpp says when they are over its limits.

#include "gcn/translate.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace gpu {

// The images, samplers and storage images a shared set declares (a stage's
// bindings from gcn::kBindingImage0, kBindingSampler0 and kBindingStorageImage0).
constexpr std::uint32_t kMaxImages = 32;

// The stages that read each graphics set. Set 0 is the vertex shader's: the
// vertex stage, or in a tessellated draw the domain shader's evaluation stage
// and the hull shader's control stage (which reads its params block); the
// rect-list geometry shader reads none of it but is a pre-rasterization stage
// like them. Set 1 is the pixel shader's. The pushed constant buffers (set 2)
// follow the same split, but for the control stage (kCbPushStages): the
// vertex stage's bindings from 112, the pixel stage's from 128.
constexpr VkShaderStageFlags kGfxSetStages[2] = {
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
        VK_SHADER_STAGE_GEOMETRY_BIT,
    VK_SHADER_STAGE_FRAGMENT_BIT};

// The stages that read graphics set `set` (0 or 1): every stage, as the
// layouts are, or with `per_stage` its own (kGfxSetStages).
inline VkShaderStageFlags gfx_set_stages(bool per_stage, int set) { return per_stage ? kGfxSetStages[set] : VK_SHADER_STAGE_ALL; }

// Not the control stage, which reads no constant buffer through a binding:
// MoltenVK puts a control stage's own buffers above every buffer binding its
// flags name, and with these 16 the tessellation levels found no Metal slot.
constexpr VkShaderStageFlags kCbPushStages[2] = {kGfxSetStages[0] & ~VkShaderStageFlags(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT),
                                                 kGfxSetStages[1]};

// The samplers a shared set declares: kMaxImages, or with `per_stage` as many
// as the device binds a stage where that is fewer (MoltenVK: 16, Metal's own
// limit).
inline std::uint32_t shared_set_samplers(bool per_stage, std::uint32_t max_per_stage_samplers) {
    return per_stage ? std::min(kMaxImages, max_per_stage_samplers) : kMaxImages;
}

// The shared set: the params block, kMaxImages sampled and storage images,
// `samplers` samplers, and with `buffers` the constant buffers (when they are
// not pushed), each read by `stages`.
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

// A graphics stage's own set (render.cpp stage_set_layout, BBHOST_SET_LAYOUTS):
// the params block and the bindings its translation declares - not the images
// and samplers under bindless (set 3's), and the constant buffers only with
// `buffers` (when they are not pushed) - in binding order, each read by
// `stages`.
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

// One set of a pipeline layout. An update-after-bind set (bindless.cpp's
// global set) answers to the update-after-bind limits instead, which are not
// checked here.
struct LayoutSet {
    const std::vector<VkDescriptorSetLayoutBinding>* binds = nullptr;
    bool update_after_bind = false;
};

// What of a pipeline layout made of `sets` is over the device's limits - each
// stage's descriptors of each kind against maxPerStageDescriptor* and
// maxPerStageResources, and the layout's totals against maxDescriptorSet* -
// or empty where it is within them.
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

// What of the layouts as they are - the compute layout, and the graphics one
// with both stage sets and (`push`) the pushed constant buffers, every binding
// in every stage - is over the device's limits; empty where they fit, and the
// device keeps them. Bindless only takes bindings out of the stage sets.
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
