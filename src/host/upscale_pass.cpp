// The temporal upscaler's call point in the frame (host/upscaler.h, the bridge):
// after the scene's last draw - the lit scene and its transparency - and before
// the first draw that samples it (render.cpp upscale_before_draw_locked finds
// that draw). The upscaler reads the scene, the G-buffer's depth and the
// motion, writes an image of its own, and that goes back into the scene, so
// every pass after it - the reflections' downsample, YEBIS's copy and the post
// chain - reads the upscaled picture without knowing.
//
// The scene is RGBA16F, or B10G11R11 with the Performance patch (its six
// format bytes): an image of the bridge's RGBA16F goes back into it by a copy
// in the first case and a blit, which converts, in the second. A B10G11R11
// scene has no alpha to keep; an RGBA16F one takes the bridge's.
//
// Steps 2 to 4 of Numity1504's DLSS/FSR handoff: the call at the native size
// (DLAA, FSR native AA), with the frame's jitter (render.cpp: the camera's
// draws' viewports) and the camera's motion (shaders/camera_motion.comp, from
// the depth and the two frames' view blocks) - the characters' own motion
// not yet.
#include "host/gpu_internal.h"
#include "host/shaders/camera_motion.spv.h"
#include "host/upscaler.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

void run_forget_recorded_locked();  // render.cpp: the command buffer's cached graphics state forgotten

namespace gpu {
namespace {

struct OwnImage {
    VkImage image = VK_NULL_HANDLE;
    ImageMemory memory;
    VkImageView view = VK_NULL_HANDLE;
    bool initialised = false;
};

OwnImage g_output, g_motion;        // the bridge's picture (RGBA16F) and the motion (R16G16_SFLOAT), the scene's size
std::uint32_t g_w = 0, g_h = 0;     // their size
VkImage g_depth_for = VK_NULL_HANDLE;  // the depth image g_depth_view is of
VkImageView g_depth_view = VK_NULL_HANDLE;
bool g_reset = true;                // the next call has no history
std::atomic<std::uint64_t> g_calls{0}, g_skipped{0}, g_no_camera{0};

void barrier(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to, VkAccessFlags src,
             VkAccessFlags dst) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// The camera as the motion pass wants it (camera_motion.comp's Params), this
// frame's and the last's.
struct Camera {
    float tan_x = 0, tan_y = 0, a = 0, r2w = 0;
    float rows[12] = {};  // view to world: rotation | eye
};
Camera g_cam_now, g_cam_prev;
bool g_cam_noted = false;  // this frame's taken (render.cpp, the first camera draw)
bool g_cam_prev_ok = false;

// The engine's per-view block (864 bytes, the G-buffer's vertex stage), in
// floats: 36-51 the inverse projection (its first row's x the tangent of half
// the horizontal field of view, its second's y the vertical, the third
// (0,0,0,1), the fourth the depth terms), 180-195 the camera, view to world
// (the last row (0,0,0,1), each row's w the eye). Numity1504's appendix A.
bool camera_of(const float* f, Camera* c) {
    for (int i = 0; i < 216; ++i) {
        if (!std::isfinite(f[i])) return false;
    }
    const float tx = f[36], ty = f[41];
    if (!(tx > 0.1f && tx < 10.0f && ty > 0.1f && ty < 10.0f)) return false;
    if (f[44] != 0.0f || f[45] != 0.0f || f[46] != 0.0f || f[47] != 1.0f) return false;
    if (f[192] != 0.0f || f[193] != 0.0f || f[194] != 0.0f || f[195] != 1.0f) return false;
    const float z = f[50], w = f[51];
    if (!(std::fabs(z) > 1e-6f) || std::fabs(f[48]) > 1e-4f || std::fabs(f[49]) > 1e-4f) return false;
    c->tan_x = tx;
    c->tan_y = ty;
    c->a = -w / z;
    c->r2w = 1.0f / z;
    std::memcpy(c->rows, f + 180, sizeof(c->rows));
    return true;
}

VkDescriptorSetLayout g_motion_set_layout = VK_NULL_HANDLE;
VkPipelineLayout g_motion_layout = VK_NULL_HANDLE;
VkPipeline g_motion_pipeline = VK_NULL_HANDLE;
bool g_motion_tried = false;

bool motion_pipeline_locked() {
    if (g_motion_tried) return g_motion_pipeline != VK_NULL_HANDLE;
    g_motion_tried = true;
    const VkDescriptorSetLayoutBinding binds[3] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 3;
    sli.pBindings = binds;
    if (vkCreateDescriptorSetLayout(g.device, &sli, nullptr, &g_motion_set_layout) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_motion_set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(g.device, &pli, nullptr, &g_motion_layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(k_camera_motion_spv);
    smi.pCode = k_camera_motion_spv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.device, &smi, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = g_motion_layout;
    const bool ok = vkCreateComputePipelines(g.device, g.cache, 1, &ci, nullptr, &g_motion_pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(g.device, module, nullptr);
    if (!ok) g_motion_pipeline = VK_NULL_HANDLE;
    host_log("upscaler: the camera's motion pass %s", ok ? "ready" : "failed to build; no motion");
    return ok;
}

// The motion image written from the depth and the two cameras (the last one
// absent: all zeros).
void record_motion(VkCommandBuffer cb, VkImageView depth_view, VkImageLayout depth_layout, VkImage motion, VkImageView motion_view,
                   std::uint32_t w, std::uint32_t h, bool prev_ok) {
    float p[32] = {};
    p[0] = g_cam_now.tan_x;
    p[1] = g_cam_now.tan_y;
    p[2] = g_cam_now.a;
    p[3] = g_cam_now.r2w;
    std::memcpy(p + 4, g_cam_now.rows, sizeof(g_cam_now.rows));
    p[16] = prev_ok ? g_cam_prev.tan_x : 0.0f;
    p[17] = g_cam_prev.tan_y;
    p[18] = static_cast<float>(w);
    p[19] = static_cast<float>(h);
    // World to the last view: the rotation transposed, its eye taken out.
    const float* m = g_cam_prev.rows;
    for (int j = 0; j < 3; ++j) {
        const float rx = m[0 * 4 + j], ry = m[1 * 4 + j], rz = m[2 * 4 + j];
        p[20 + 4 * j + 0] = rx;
        p[20 + 4 * j + 1] = ry;
        p[20 + 4 * j + 2] = rz;
        p[20 + 4 * j + 3] = -(rx * m[3] + ry * m[7] + rz * m[11]);
    }
    const VkDescriptorSet set = alloc_set_locked(g_motion_set_layout);
    if (!set) return;
    const VkDescriptorImageInfo depth_info{VK_NULL_HANDLE, depth_view, depth_layout};
    const VkDescriptorImageInfo sampler_info{g.dummy_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    const VkDescriptorImageInfo motion_info{VK_NULL_HANDLE, motion_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr[3] = {};
    for (int k = 0; k < 3; ++k) {
        wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[k].dstSet = set;
        wr[k].dstBinding = static_cast<std::uint32_t>(k);
        wr[k].descriptorCount = 1;
    }
    wr[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    wr[0].pImageInfo = &depth_info;
    wr[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    wr[1].pImageInfo = &sampler_info;
    wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[2].pImageInfo = &motion_info;
    vkUpdateDescriptorSets(g.device, 3, wr, 0, nullptr);
    barrier(cb, motion, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_motion_pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_motion_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, g_motion_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), p);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    barrier(cb, motion, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT);
}

void drop(OwnImage& o) {
    if (o.view) defer_destroy_private_view(o.view);
    if (o.image) defer_destroy_image(o.image, o.memory);
    o = OwnImage{};
}

bool make(OwnImage& o, VkFormat format, VkImageUsageFlags usage, std::uint32_t w, std::uint32_t h) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g.device, &ici, nullptr, &o.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g.device, o.image, &req);
    if (!image_memory_alloc(req, o.memory, true)) {
        vkDestroyImage(g.device, o.image, nullptr);
        o.image = VK_NULL_HANDLE;
        return false;
    }
    vkBindImageMemory(g.device, o.image, o.memory.memory, o.memory.offset);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = o.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return vkCreateImageView(g.device, &vci, nullptr, &o.view) == VK_SUCCESS;
}

// Whether the bridge's RGBA16F picture can be blitted into a scene of `format`.
bool blit_into(VkFormat format) {
    VkFormatProperties src{}, dst{};
    vkGetPhysicalDeviceFormatProperties(g.phys, VK_FORMAT_R16G16B16A16_SFLOAT, &src);
    vkGetPhysicalDeviceFormatProperties(g.phys, format, &dst);
    return (src.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) && (dst.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);
}


}  // namespace

bool upscale_pass_run_locked(RtImage& scene, RtImage& depth, float jitter_x, float jitter_y) {
    const int method = upscaler_method();
    if (!method) return false;
    const bool same_format = scene.format == VK_FORMAT_R16G16B16A16_SFLOAT;
    static VkFormat checked = VK_FORMAT_UNDEFINED;
    static bool can_blit = false;
    if (!same_format && checked != scene.format) {
        checked = scene.format;
        can_blit = blit_into(scene.format);
        host_log("upscaler: the scene is format %d: %s", static_cast<int>(scene.format),
                 can_blit ? "the upscaled picture goes back by a blit" : "this device cannot blit into it; not upscaled");
    }
    if (!same_format && !can_blit) {
        g_skipped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (scene.width != g_w || scene.height != g_h) {
        drop(g_output);
        drop(g_motion);
        g_w = scene.width;
        g_h = scene.height;
        g_reset = true;
        if (!make(g_output, VK_FORMAT_R16G16B16A16_SFLOAT,
                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, g_w, g_h) ||
            !make(g_motion, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  g_w, g_h)) {
            host_log("upscaler: no images of %ux%u for it; not upscaled", g_w, g_h);
            drop(g_output);
            drop(g_motion);
            g_w = g_h = 0;
            return false;
        }
        host_log("upscaler: %s at %ux%u (native, the scene's size), scene format %d, depth format %d", method == 1 ? "DLAA" : "FSR native AA",
                 g_w, g_h, static_cast<int>(scene.format), static_cast<int>(depth.format));
    }
    if (depth.image != g_depth_for) {
        if (g_depth_view) defer_destroy_private_view(g_depth_view);
        g_depth_view = VK_NULL_HANDLE;
        g_depth_for = depth.image;
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = depth.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = depth.format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(g.device, &vci, nullptr, &g_depth_view) != VK_SUCCESS) {
            g_depth_view = VK_NULL_HANDLE;
            g_depth_for = VK_NULL_HANDLE;
            return false;
        }
    }

    begin_recording_locked();
    if (scene.view_clear) view_clear_apply_locked(scene);
    if (depth.view_clear) view_clear_apply_locked(depth);
    render_end_pass_locked();
    if (depth.managed_layout) rt_set_layout_locked(depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    const VkImageLayout depth_layout = depth.managed_layout ? depth.layout : VK_IMAGE_LAYOUT_GENERAL;
    VkCommandBuffer cb = g_cmd();
    // The scene's and the depth's writes done before the bridge reads them.
    barrier(cb, scene.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    if (!g_motion.initialised) {
        // No motion yet: all zeros (the camera's and the characters' come next).
        barrier(cb, g_motion.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
                VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkClearColorValue zero{};
        const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, g_motion.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
        barrier(cb, g_motion.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT);
        g_motion.initialised = true;
    }
    // The camera's motion, from this frame's view block and the last's; with no
    // block this frame, none (and no history: the motion would be wrong).
    // BBHOST_UPSCALE_MOTION=0: none (a test); =neg: the upscaler told the
    // motion the other way (a test: moving, the picture should smear).
    static const int motion_mode = [] {
        const char* e = std::getenv("BBHOST_UPSCALE_MOTION");
        return !e ? 1 : e[0] == '0' ? 0 : std::strcmp(e, "neg") == 0 ? -1 : 1;
    }();
    if (g_cam_noted && motion_mode != 0 && motion_pipeline_locked()) {
        record_motion(cb, g_depth_view, depth_layout, g_motion.image, g_motion.view, g_w, g_h, g_cam_prev_ok);
        if (!g_cam_prev_ok) g_reset = true;
        g_cam_prev = g_cam_now;
        g_cam_prev_ok = true;
    } else {
        g_no_camera.fetch_add(1, std::memory_order_relaxed);
        barrier(cb, g_motion.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkClearColorValue zero{};
        const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, g_motion.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
        barrier(cb, g_motion.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT);
        g_cam_prev_ok = false;
        g_reset = true;
    }
    g_cam_noted = false;
    UpscaleCall c;
    c.method = method;
    c.quality = 0;
    c.reset = g_reset;
    c.render_w = c.output_w = g_w;
    c.render_h = c.output_h = g_h;
    c.color = {scene.image, scene.view, scene.format, VK_IMAGE_LAYOUT_GENERAL};
    c.depth = {depth.image, g_depth_view, depth.format, depth_layout};
    c.motion = {g_motion.image, g_motion.view, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL};
    c.motion_scale = motion_mode < 0 ? -1.0f : 1.0f;
    c.jitter_x = jitter_x;
    c.jitter_y = jitter_y;
    c.output = {g_output.image, g_output.view, VK_FORMAT_R16G16B16A16_SFLOAT,
                g_output.initialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED};
    const bool ok = upscaler_evaluate(cb, c);
    // The bridge's binds are gone from cb: the next draw binds all of its own.
    ::run_forget_recorded_locked();
    if (!ok) return false;
    g_output.initialised = true;
    g_reset = false;
    // Its picture into the scene.
    barrier(cb, g_output.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
    barrier(cb, scene.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT);
    if (same_format) {
        VkImageCopy r{};
        r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.extent = {g_w, g_h, 1};
        vkCmdCopyImage(cb, g_output.image, VK_IMAGE_LAYOUT_GENERAL, scene.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    } else {
        VkImageBlit r{};
        r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.srcOffsets[1] = r.dstOffsets[1] = {static_cast<std::int32_t>(g_w), static_cast<std::int32_t>(g_h), 1};
        vkCmdBlitImage(cb, g_output.image, VK_IMAGE_LAYOUT_GENERAL, scene.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r, VK_FILTER_NEAREST);
    }
    barrier(cb, scene.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    if (g_calls.fetch_add(1, std::memory_order_relaxed) == 0)
        host_log("upscaler: first upscale of the scene 0x%llx (depth 0x%llx)", static_cast<unsigned long long>(scene.base),
                 static_cast<unsigned long long>(depth.base));
    return true;
}

// A frame it could not run in starts the next one's history over.
void upscale_pass_reset_locked() { g_reset = true; }

bool upscale_camera_noted() { return g_cam_noted; }

bool upscale_camera_note_locked(std::uint64_t base, std::uint64_t bytes) {
    if (g_cam_noted || !base || bytes < 864) return g_cam_noted;
    Camera c;
    if (!camera_of(reinterpret_cast<const float*>(static_cast<std::uintptr_t>(base)), &c)) return false;
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) == 0)
        host_log("upscaler: the camera's view block at 0x%llx (tangents %.3f %.3f, eye %.1f %.1f %.1f)", static_cast<unsigned long long>(base),
                 static_cast<double>(c.tan_x), static_cast<double>(c.tan_y), static_cast<double>(c.rows[3]), static_cast<double>(c.rows[7]),
                 static_cast<double>(c.rows[11]));
    g_cam_now = c;
    g_cam_noted = true;
    return true;
}

std::uint64_t upscale_pass_calls() { return g_calls.load(std::memory_order_relaxed); }

}  // namespace gpu
