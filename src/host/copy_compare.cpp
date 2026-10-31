// BBHOST_COPY_ELIDE=compare: a copy left unmade (render.cpp) is also made at
// once, its destination's sampled bindings still bind the source, and at the
// end of the copy's time pending - made later, forgotten, or the images
// exchanged - the source and the destination are compared texel by texel on
// the GPU (shaders/imagecompare.comp). The source holds what the copy wrote
// for as long as nothing writes it; a difference is a write that reached it
// without the ledger seeing it, so a binding of the destination meanwhile may
// have read the wrong texels. The counts come back once the submission has
// run, by what ended the copy, the first differences logged.
#include "host/gpu_internal.h"
#include "host/shaders/imagecompare.spv.h"
#include "log.h"

#include <cstdio>
#include <deque>
#include <map>
#include <string>

namespace gpu {
namespace {

struct Push {
    std::uint32_t width, height, slot;
};

constexpr std::uint32_t kCompareSlots = 4096;
VkDescriptorSetLayout g_set_layout = VK_NULL_HANDLE;
VkPipelineLayout g_layout = VK_NULL_HANDLE;
VkPipeline g_pipeline = VK_NULL_HANDLE;
DevBuffer g_counts;  // kCompareSlots uints, host-visible
bool g_tried = false, g_ready = false;
std::uint32_t g_next = 0;

struct Pending {
    std::uint32_t slot;
    std::uint64_t serial;  // g.flushes of the recording the compare is in
    std::uint64_t a, b;
    std::uint32_t width, height;
    VkFormat format;
    const char* why;
};
std::deque<Pending> g_pending;  // oldest first
struct Tally {
    std::uint64_t compares = 0, differing = 0, texels = 0;
};
std::map<std::string, Tally> g_by_why;
std::uint64_t g_skipped_format = 0, g_skipped_full = 0, g_skipped_setup = 0, g_logged = 0;

bool make_locked() {
    const VkDescriptorSetLayoutBinding binds[4] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 4;
    sli.pBindings = binds;
    if (vkCreateDescriptorSetLayout(g.device, &sli, nullptr, &g_set_layout) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(g.device, &pli, nullptr, &g_layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(k_imagecompare_spv);
    smi.pCode = k_imagecompare_spv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.device, &smi, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = g_layout;
    const bool ok = vkCreateComputePipelines(g.device, g.cache, 1, &ci, nullptr, &g_pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(g.device, module, nullptr);
    if (!ok) return false;
    MemOwner own_mem("copy elision compares");
    return create_dev_buffer(g_counts, kCompareSlots * sizeof(std::uint32_t), true) && g_counts.map;
}

// A float format whose texels a texelFetch converts the same way for two images.
bool comparable(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32G32_SFLOAT:
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R5G6B5_UNORM_PACK16:
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
            return true;
        default:
            return false;
    }
}

// Compares whose submissions have run: their counts added up.
void harvest_locked() {
    while (!g_pending.empty() && g_pending.front().serial < g.completed_submits) {
        const Pending p = g_pending.front();
        g_pending.pop_front();
        const std::uint32_t n = static_cast<const std::uint32_t*>(g_counts.map)[p.slot];
        Tally& t = g_by_why[p.why];
        ++t.compares;
        if (n) {
            ++t.differing;
            t.texels += n;
            if (g_logged++ < 16) {
                host_log("render: copy elision compare: 0x%llx and 0x%llx (%ux%u format %d) differ in %u texels where the copy ended (%s): a "
                         "write reached the source without the ledger",
                         static_cast<unsigned long long>(p.a), static_cast<unsigned long long>(p.b), p.width, p.height, static_cast<int>(p.format),
                         n, p.why);
            }
        }
    }
}

}  // namespace

void copy_compare_record_locked(RtImage& a, RtImage& b, const char* why) {
    if (!g_tried) {
        g_tried = true;
        g_ready = make_locked();
        host_log("render: copy elision compares %s", g_ready ? "ready" : "unavailable (pipeline or buffer creation failed)");
    }
    if (!g_ready) {
        ++g_skipped_setup;
        return;
    }
    harvest_locked();
    if (!comparable(a.format) || a.format != b.format || a.width != b.width || a.height != b.height || !a.view || !b.view) {
        ++g_skipped_format;
        return;
    }
    if (g_pending.size() >= kCompareSlots) {
        ++g_skipped_full;
        return;
    }
    const VkDescriptorSet set = alloc_set_locked(g_set_layout);
    if (!set) {
        ++g_skipped_setup;
        return;
    }
    const std::uint32_t slot = g_next++ % kCompareSlots;
    // The images' own views (format, colour aspect, layer 0) are sampled ones.
    const VkDescriptorImageInfo ia{VK_NULL_HANDLE, a.view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo ib{VK_NULL_HANDLE, b.view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo is{g.dummy_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    const VkDescriptorBufferInfo ic{g_counts.buffer, 0, kCompareSlots * sizeof(std::uint32_t)};
    VkWriteDescriptorSet w[4] = {};
    for (int k = 0; k < 4; ++k) {
        w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k].dstSet = set;
        w[k].dstBinding = static_cast<std::uint32_t>(k);
        w[k].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[0].pImageInfo = &ia;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[1].pImageInfo = &ib;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w[2].pImageInfo = &is;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[3].pBufferInfo = &ic;
    vkUpdateDescriptorSets(g.device, 4, w, 0, nullptr);
    begin_recording_locked();
    render_end_pass_locked();
    VkCommandBuffer cmd = g_cmd();
    // Everything before it written (attachments, transfers, stores), its slot zeroed.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    vkCmdFillBuffer(cmd, g_counts.buffer, slot * sizeof(std::uint32_t), sizeof(std::uint32_t), 0);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    const Push push{a.width, a.height, slot};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, g_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (a.width + 7) / 8, (a.height + 7) / 8, 1);
    // What comes after (the write that ended the copy) waits for the reads; the host reads the count.
    mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    g_pending.push_back({slot, g.flushes, a.base, b.base, a.width, a.height, a.format, why});
}

std::string copy_compare_report_locked() {
    if (!g_tried) return "compares: none yet";
    harvest_locked();
    std::string out = "compares";
    std::uint64_t all = 0, bad = 0;
    for (const auto& [why, t] : g_by_why) {
        all += t.compares;
        bad += t.differing;
        char b[160];
        std::snprintf(b, sizeof(b), " %s %llu (%llu differing, %llu texels)", why.c_str(), static_cast<unsigned long long>(t.compares),
                      static_cast<unsigned long long>(t.differing), static_cast<unsigned long long>(t.texels));
        out += b;
    }
    char b[200];
    std::snprintf(b, sizeof(b), "; %llu in all, %llu differing; %zu waiting; not compared: format %llu, full %llu, setup %llu",
                  static_cast<unsigned long long>(all), static_cast<unsigned long long>(bad), g_pending.size(),
                  static_cast<unsigned long long>(g_skipped_format), static_cast<unsigned long long>(g_skipped_full),
                  static_cast<unsigned long long>(g_skipped_setup));
    out += b;
    return out;
}

}  // namespace gpu
