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
// Steps 2 and 3 of Numity1504's DLSS/FSR handoff: the call at the native size
// (DLAA, FSR native AA), with the frame's jitter (render.cpp: the camera's
// draws' viewports) and no motion yet - the motion image is all zeros.
#include "host/gpu_internal.h"
#include "host/upscaler.h"
#include "log.h"

#include <algorithm>
#include <atomic>

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
std::atomic<std::uint64_t> g_calls{0}, g_skipped{0};

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
            !make(g_motion, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, g_w, g_h)) {
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
    UpscaleCall c;
    c.method = method;
    c.quality = 0;
    c.reset = g_reset;
    c.render_w = c.output_w = g_w;
    c.render_h = c.output_h = g_h;
    c.color = {scene.image, scene.view, scene.format, VK_IMAGE_LAYOUT_GENERAL};
    c.depth = {depth.image, g_depth_view, depth.format, depth_layout};
    c.motion = {g_motion.image, g_motion.view, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL};
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

std::uint64_t upscale_pass_calls() { return g_calls.load(std::memory_order_relaxed); }

}  // namespace gpu
