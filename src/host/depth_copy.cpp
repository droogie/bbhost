// A depth target copied into an R32_SFLOAT colour snapshot as one compute
// pass (shaders/depthcopy.comp). Depth and colour images cannot be copied
// image to image, so render_copy_target_locked went through a buffer: the
// depth plane out, a barrier, the buffer into the colour image. Once a frame
// in the world (the copy the lighting passes sample), that was the costliest
// single item on the GPU profile - 0.43-0.48 ms on an RTX 4070, above every
// draw pipeline - and GPU time is what the test laptop runs short of. The
// pass reads the depth plane through a sampled view and stores the float as
// it is, so the snapshot holds the same bits. BBHOST_DEPTH_COPY_COMPUTE=0
// keeps the buffer copies.
#include "host/gpu_internal.h"
#include "host/shaders/depthcopy.spv.h"
#include "log.h"

#include <cstdlib>

namespace gpu {
namespace {

struct Push {
    std::uint32_t width, height;
};

VkDescriptorSetLayout g_set_layout = VK_NULL_HANDLE;
VkPipelineLayout g_layout = VK_NULL_HANDLE;
VkPipeline g_pipeline = VK_NULL_HANDLE;
bool g_tried = false;

bool make_pipeline_locked() {
    const VkDescriptorSetLayoutBinding binds[3] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 3;
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
    smi.codeSize = sizeof(k_depthcopy_spv);
    smi.pCode = k_depthcopy_spv;
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
    return ok;
}

}  // namespace

bool depth_copy_available_locked() {
    static const bool wanted = [] {
        const char* e = std::getenv("BBHOST_DEPTH_COPY_COMPUTE");
        return !(e && e[0] == '0');
    }();
    if (!wanted) return false;
    if (!g_tried) {
        g_tried = true;
        const bool ok = make_pipeline_locked();
        if (!ok) g_pipeline = VK_NULL_HANDLE;
        host_log("gpu: depth snapshots by a compute pass %s", ok ? "ready" : "unavailable (pipeline creation failed); copied through a buffer");
    }
    return g_pipeline != VK_NULL_HANDLE;
}

bool depth_copy_record_locked(VkImage depth, VkFormat depth_format, VkImage dst, std::uint32_t width, std::uint32_t height,
                              VkImageLayout depth_layout) {
    // A view of the depth plane alone, made for this copy and gone with its
    // command buffer (one a frame).
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = depth;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = depth_format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    VkImageView depth_view = VK_NULL_HANDLE, dst_view = VK_NULL_HANDLE;
    if (vkCreateImageView(g.device, &vci, nullptr, &depth_view) != VK_SUCCESS) return false;
    vci.image = dst;
    vci.format = VK_FORMAT_R32_SFLOAT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(g.device, &vci, nullptr, &dst_view) != VK_SUCCESS) {
        vkDestroyImageView(g.device, depth_view, nullptr);
        return false;
    }
    const VkDescriptorSet set = alloc_set_locked(g_set_layout);
    if (!set) {
        vkDestroyImageView(g.device, depth_view, nullptr);
        vkDestroyImageView(g.device, dst_view, nullptr);
        return false;
    }
    const VkDescriptorImageInfo depth_info{VK_NULL_HANDLE, depth_view, depth_layout};
    const VkDescriptorImageInfo sampler_info{g.dummy_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    const VkDescriptorImageInfo dst_info{VK_NULL_HANDLE, dst_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[3] = {};
    for (int k = 0; k < 3; ++k) {
        w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k].dstSet = set;
        w[k].dstBinding = static_cast<std::uint32_t>(k);
        w[k].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[0].pImageInfo = &depth_info;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w[1].pImageInfo = &sampler_info;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[2].pImageInfo = &dst_info;
    vkUpdateDescriptorSets(g.device, 3, w, 0, nullptr);
    const Push push{width, height};
    VkCommandBuffer cmd = g_cmd();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, g_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
    defer_destroy_private_view(depth_view);
    defer_destroy_private_view(dst_view);
    return true;
}

}  // namespace gpu
