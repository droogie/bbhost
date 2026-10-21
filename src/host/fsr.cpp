// The presenter's upscale when the game renders smaller than the window: AMD
// FidelityFX Super Resolution 1 (shaders/fsr1/, MIT) - EASU, an
// edge-adaptive upsampling of the display buffer's rendered area, then RCAS,
// a contrast-adaptive sharpening of the result - in place of a bilinear blit.
// A Steam Deck holds 60 fps rendering at 960x600 (the GPU is pixel-bound at
// 1280x800, ~45 fps), and stretched bilinearly that looks soft on its own
// screen. Two compute passes, the second into the swapchain image itself when
// it takes storage writes (window.cpp asks for that where the surface allows),
// else into an image of the output size and a 1:1 blit from it (BBHOST_FSR_DIRECT=0
// always does that). BBHOST_UPSCALE=linear keeps the bilinear blit.
#include "host/gpu_internal.h"
#include "host/shaders/fsr_easu.spv.h"
#include "host/shaders/fsr_easu_h.spv.h"
#include "host/shaders/fsr_rcas.spv.h"
#include "host/shaders/fsr_rcas_h.spv.h"
#include "log.h"

#include <cstdlib>
#include <cstring>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wmissing-braces"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-braces"
#endif
#define A_CPU 1
#include <math.h>  // the CPU half of ffx_a.h calls fabs, floor, sqrt... from the C library
#include <stdint.h>
#include <string.h>
#include "host/shaders/fsr1/ffx_a.h"
#include "host/shaders/fsr1/ffx_fsr1.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace gpu {
namespace {

// RCAS strength in stops below its maximum (AMD's samples use 0.2).
constexpr float kSharpness = 0.2f;

struct Target {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

struct Fsr {
    bool tried = false, ok = false;
    bool half = false;  // the half-precision shaders (fsr_easu_h, fsr_rcas_h)
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline easu = VK_NULL_HANDLE, rcas = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet easu_set = VK_NULL_HANDLE, rcas_set = VK_NULL_HANDLE;
    Target upscaled, sharpened;
    std::uint32_t width = 0, height = 0;
    VkImageView source_view = VK_NULL_HANDLE;  // made each frame: the display images come and go
    // GPU time of the two passes, read back a frame later (after the
    // presenter's fence) and logged every 300 frames.
    VkQueryPool queries = VK_NULL_HANDLE;
    bool queried = false;
    std::uint64_t frames = 0;
    double total_us = 0;
} g_fsr;

const bool g_wanted = [] {
    const char* e = std::getenv("BBHOST_UPSCALE");
    return !(e && std::strcmp(e, "linear") == 0);
}();

VkPipeline make_pipeline(const std::uint32_t* code, std::size_t bytes) {
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = bytes;
    smi.pCode = code;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.device, &smi, nullptr, &module) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = g_fsr.layout;
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(g.device, g.cache, 1, &ci, nullptr, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
    vkDestroyShaderModule(g.device, module, nullptr);
    return p;
}

bool fsr_init_locked() {
    const VkDescriptorSetLayoutBinding binds[3] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 3;
    sli.pBindings = binds;
    if (vkCreateDescriptorSetLayout(g.device, &sli, nullptr, &g_fsr.set_layout) != VK_SUCCESS) return false;
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * 4 * sizeof(std::uint32_t)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_fsr.set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(g.device, &pli, nullptr, &g_fsr.layout) != VK_SUCCESS) return false;
    // In half precision where the device does f16 math (gpu.cpp g.has_f16_math):
    // RDNA packs two f16 operations into one - on a Steam Deck, 960x600 to
    // 1280x800, 0.77 -> 0.60 ms a frame. BBHOST_FSR_HALF=0: the f32 shaders.
    static const bool half_wanted = [] {
        const char* e = std::getenv("BBHOST_FSR_HALF");
        return !(e && e[0] == '0');
    }();
    g_fsr.half = half_wanted && g.has_f16_math;
    g_fsr.easu = g_fsr.half ? make_pipeline(k_fsr_easu_h_spv, sizeof(k_fsr_easu_h_spv)) : make_pipeline(k_fsr_easu_spv, sizeof(k_fsr_easu_spv));
    g_fsr.rcas = g_fsr.half ? make_pipeline(k_fsr_rcas_h_spv, sizeof(k_fsr_rcas_h_spv)) : make_pipeline(k_fsr_rcas_spv, sizeof(k_fsr_rcas_spv));
    if (!g_fsr.easu || !g_fsr.rcas) return false;
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(g.device, &sci, nullptr, &g_fsr.sampler) != VK_SUCCESS) return false;
    const VkDescriptorPoolSize sizes[3] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2}, {VK_DESCRIPTOR_TYPE_SAMPLER, 2}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 2;
    dpi.poolSizeCount = 3;
    dpi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(g.device, &dpi, nullptr, &g_fsr.pool) != VK_SUCCESS) return false;
    const VkDescriptorSetLayout layouts[2] = {g_fsr.set_layout, g_fsr.set_layout};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g_fsr.pool;
    dai.descriptorSetCount = 2;
    dai.pSetLayouts = layouts;
    VkDescriptorSet sets[2] = {};
    if (vkAllocateDescriptorSets(g.device, &dai, sets) != VK_SUCCESS) return false;
    g_fsr.easu_set = sets[0];
    g_fsr.rcas_set = sets[1];
    VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = 2;
    if (vkCreateQueryPool(g.device, &qci, nullptr, &g_fsr.queries) != VK_SUCCESS) g_fsr.queries = VK_NULL_HANDLE;
    return true;
}

// Last frame's pass time, now that its command buffer has finished.
void note_time() {
    if (!g_fsr.queries || !g_fsr.queried) return;
    std::uint64_t ts[2] = {};
    if (vkGetQueryPoolResults(g.device, g_fsr.queries, 0, 2, sizeof(ts), ts, sizeof(ts[0]), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
    g_fsr.total_us += static_cast<double>(ts[1] - ts[0]) * g.timestamp_period_ns / 1000.0;
    if (++g_fsr.frames % 300 == 0 && host_tooling()) {
        host_log("present: FSR 1 took %.2f ms of GPU time a frame (%ux%u)", g_fsr.total_us / 300.0 / 1000.0, g_fsr.width, g_fsr.height);
        g_fsr.total_us = 0;
    }
}

void destroy_target(Target& t) {
    if (t.view) vkDestroyImageView(g.device, t.view, nullptr);
    if (t.image) vkDestroyImage(g.device, t.image, nullptr);
    if (t.memory) vkFreeMemory(g.device, t.memory, nullptr);
    t = Target{};
}

bool make_target(Target& t, std::uint32_t w, std::uint32_t h, VkImageUsageFlags usage) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g.device, &ici, nullptr, &t.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g.device, t.image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    MemOwner own_mem("FSR images");
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &t.memory) != VK_SUCCESS) return false;
    vkBindImageMemory(g.device, t.image, t.memory, 0);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ici.format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return vkCreateImageView(g.device, &vci, nullptr, &t.view) == VK_SUCCESS;
}

void write_set(VkDescriptorSet set, VkImageView src, VkImageView dst) {
    const VkDescriptorImageInfo src_info{VK_NULL_HANDLE, src, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo smp_info{g_fsr.sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    const VkDescriptorImageInfo dst_info{VK_NULL_HANDLE, dst, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[3] = {};
    for (int k = 0; k < 3; ++k) {
        w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k].dstSet = set;
        w[k].dstBinding = static_cast<std::uint32_t>(k);
        w[k].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[0].pImageInfo = &src_info;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w[1].pImageInfo = &smp_info;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[2].pImageInfo = &dst_info;
    vkUpdateDescriptorSets(g.device, 3, w, 0, nullptr);
}

void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src_access,
                   VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

}  // namespace

// Upscales [src_x, src_y, sw x sh] of the display image `src` (its whole size
// src_w x src_h, in GENERAL) into `area` of `dst` (in TRANSFER_DST_OPTIMAL);
// false when it cannot, and the caller blits. On the presenter's command
// buffer, recorded after its fence, so last frame's sets and images are free.
bool fsr_upscale_locked(VkCommandBuffer cmd, VkImage src, VkFormat src_format, std::uint32_t src_w, std::uint32_t src_h,
                        std::uint32_t src_x, std::uint32_t src_y, std::uint32_t sw, std::uint32_t sh, VkImage dst, VkRect2D area,
                        VkImageView dst_view) {
    if (!g_wanted || !sw || !sh || !area.extent.width || !area.extent.height) return false;
    if (!g_fsr.tried) {
        g_fsr.tried = true;
        g_fsr.ok = fsr_init_locked();
        host_log("present: upscaling by FSR 1 (EASU + RCAS%s%s) %s", g_fsr.half ? ", half precision" : "",
                 dst_view ? ", into the swapchain image" : "", g_fsr.ok ? "ready" : "unavailable; bilinear");
    }
    if (!g_fsr.ok) return false;
    note_time();
    const std::uint32_t w = area.extent.width, h = area.extent.height;
    if (w != g_fsr.width || h != g_fsr.height) {
        destroy_target(g_fsr.upscaled);
        destroy_target(g_fsr.sharpened);
        g_fsr.width = g_fsr.height = 0;
        if (!make_target(g_fsr.upscaled, w, h, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT) ||
            !make_target(g_fsr.sharpened, w, h, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            destroy_target(g_fsr.upscaled);
            destroy_target(g_fsr.sharpened);
            return false;
        }
        g_fsr.width = w;
        g_fsr.height = h;
        host_log("present: FSR 1 from %ux%u to %ux%u", sw, sh, w, h);
    }
    if (g_fsr.source_view) vkDestroyImageView(g.device, g_fsr.source_view, nullptr);
    g_fsr.source_view = VK_NULL_HANDLE;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = src;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = src_format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(g.device, &vci, nullptr, &g_fsr.source_view) != VK_SUCCESS) {
        g_fsr.source_view = VK_NULL_HANDLE;
        return false;
    }
    write_set(g_fsr.easu_set, g_fsr.source_view, g_fsr.upscaled.view);
    // RCAS straight into the swapchain image when it can take it: no copy of
    // the frame and no blit (on a Steam Deck ~0.1 ms of its GPU a frame).
    const bool direct = dst_view != VK_NULL_HANDLE;
    write_set(g_fsr.rcas_set, g_fsr.upscaled.view, direct ? dst_view : g_fsr.sharpened.view);

    AU1 easu[16] = {};
    FsrEasuConOffset(easu, easu + 4, easu + 8, easu + 12, static_cast<AF1>(sw), static_cast<AF1>(sh), static_cast<AF1>(src_w),
                     static_cast<AF1>(src_h), static_cast<AF1>(w), static_cast<AF1>(h), static_cast<AF1>(src_x), static_cast<AF1>(src_y));
    AU1 rcas[8] = {};  // the constants, then where the result goes: x, y, width, height
    FsrRcasCon(rcas, kSharpness);
    rcas[4] = direct ? static_cast<AU1>(area.offset.x) : 0;
    rcas[5] = direct ? static_cast<AU1>(area.offset.y) : 0;
    rcas[6] = w;
    rcas[7] = h;
    const std::uint32_t groups_x = (w + 15) / 16, groups_y = (h + 15) / 16;

    if (g_fsr.queries) {
        vkCmdResetQueryPool(cmd, g_fsr.queries, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_fsr.queries, 0);
        g_fsr.queried = true;
    }
    // The display image as the game left it; the two targets from scratch.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    image_barrier(cmd, g_fsr.upscaled.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_fsr.easu);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_fsr.layout, 0, 1, &g_fsr.easu_set, 0, nullptr);
    vkCmdPushConstants(cmd, g_fsr.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(easu), easu);
    vkCmdDispatch(cmd, groups_x, groups_y, 1);
    image_barrier(cmd, g_fsr.upscaled.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (direct) {
        // The caller left it a transfer destination (cleared for the bars, maybe).
        image_barrier(cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    } else {
        image_barrier(cmd, g_fsr.sharpened.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_fsr.rcas);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_fsr.layout, 0, 1, &g_fsr.rcas_set, 0, nullptr);
    vkCmdPushConstants(cmd, g_fsr.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rcas), rcas);
    vkCmdDispatch(cmd, groups_x, groups_y, 1);
    if (direct) {
        // Back to what the caller expects; what follows (the overlay, the
        // present) reads it in any stage.
        image_barrier(cmd, dst, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        if (g_fsr.queries) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_fsr.queries, 1);
        return true;
    }
    image_barrier(cmd, g_fsr.sharpened.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {static_cast<int32_t>(w), static_cast<int32_t>(h), 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.dstOffsets[0] = {area.offset.x, area.offset.y, 0};
    blit.dstOffsets[1] = {area.offset.x + static_cast<int32_t>(w), area.offset.y + static_cast<int32_t>(h), 1};
    vkCmdBlitImage(cmd, g_fsr.sharpened.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   VK_FILTER_NEAREST);
    if (g_fsr.queries) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_fsr.queries, 1);
    return true;
}

}  // namespace gpu
