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
// the depth and the two frames' view blocks) - and the characters' own
// motion, the meshes' from the game's velocity map (the motion blur's: its
// scales from the draws that write it, render.cpp upscale_velo_note_locked),
// where a mesh moved of itself. Without it the player, standing still on the
// screen as the camera orbits, had the background's motion and smeared.
// BBHOST_UPSCALE_VELO=0: the camera's motion alone (a test). The map is read
// as its passes drew it: with the game's blur on, its dilation reads the map
// and writes it back spread - each moving mesh's motion over what is around
// it, which ghosted the background beside anything moving - so it is copied
// before the dilation's first draw (upscale_velo_copy_locked). With no camera
// pass over it (the camera blur off) it is cleared before the meshes' passes
// (upscale_velo_clear_locked): the game clears it only when it is made.
//
// Frame generation (host/framegen.h) takes the same depth and motion. With no
// upscaler running - Upscaling off, or one this GPU cannot run (DLSS on AMD) -
// the call point makes the motion at the scene's size and prepares frame
// generation alone, the scene left as it is and the draws not moved.
//
// Under render scale (engine/render_scale.cpp: an Upscaling quality above
// Native AA) the scene, its depth and the motion are the render size and the
// picture is the output size, which no longer fits back into the scene: it
// stays in the upscaler's own image, and YEBIS's copy of the scene - the
// post chain's first pass, at the output size - samples it in the scene's
// place (render.cpp, upscale_pass_output_view_locked). Every other reader of
// the scene (the reflections' downsample) reads it at the render size, as
// the passes that draw it do.
//
// Sharpness (Graphics > Sharpness, above 0%): neither the bridge nor DLSS
// sharpens any more, so FSR 1's RCAS runs over the picture into an image of
// its own (shaders/upscale_rcas.comp), which goes on in its place - before the
// post chain and the HUD, which it never touches.
#include "host/framegen.h"
#include "engine/render_scale.h"
#include "host/gpu_internal.h"
#include "host/shaders/camera_motion.spv.h"
#include "host/shaders/motion_check.spv.h"
#include "host/shaders/motion_view.spv.h"
#include "host/shaders/upscale_rcas.spv.h"
#include "host/options.h"
#include "host/settings.h"
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

OwnImage g_output, g_motion;        // the bridge's picture (RGBA16F, the output size) and the motion (R16G16_SFLOAT, the scene's)
OwnImage g_sharp;                   // the picture sharpened (Sharpness), the output size; made at the first use
// Frame generation (host/framegen.h): the frames prepared so far - each
// frame's number - and the one this frame was prepared as (0: not), which its
// HUD-less copy is then of.
std::uint64_t g_fg_frames = 0, g_fg_pending = 0;
// This frame's HUD-less copy, made, waiting for the display buffer it goes to.
VkImage g_fg_copy = VK_NULL_HANDLE;
VkFormat g_fg_copy_format = VK_FORMAT_UNDEFINED;
std::uint32_t g_fg_copy_w = 0, g_fg_copy_h = 0;
std::uint32_t g_w = 0, g_h = 0;     // the render size: the scene's, the motion's
std::uint32_t g_ow = 0, g_oh = 0;   // the output size: the picture's
bool g_scaled = false;              // the picture is larger than the scene: YEBIS's copy reads it (render scale)
bool g_output_ready = false;        // a picture of this size was written
float g_mip_bias = 0.0f;            // the camera's draws' texture LOD bias for this method, quality and size (upscale_pass_mip_bias_locked)
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

// The game's velocity map (the motion blur's): its target, and the scales its
// two kinds of pass write their motion with (render.cpp, the draws' constant
// buffers) - the camera's this frame (0: its pass not drawn, or drawn still:
// the map's background is no motion), the meshes' as last seen. Seen: a pass
// of it drawn this frame.
std::uint64_t g_velo_base = 0;
float g_velo_cam[2] = {}, g_velo_mesh[2] = {};
bool g_velo_cam_seen = false, g_velo_mesh_known = false;
std::uint64_t g_velo_used = 0;  // frames it was read in
// The frame's work on the map before the call point (render.cpp, before the
// draws that need it): the frames so far (counted at the call point), the one
// the map was cleared in, whether a camera pass covered it this frame, and
// its copy taken before the motion blur's dilation and the frame it is of.
std::uint64_t g_velo_frame = 1, g_velo_cleared = 0, g_velo_copied = 0;
bool g_velo_camera_pass = false;
struct VeloCopy {
    VkImage image = VK_NULL_HANDLE;
    ImageMemory memory;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t w = 0, h = 0;
};
VeloCopy g_velo_copy;

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

// The meshes' y against uv's (BBHOST_UPSCALE_VELO_Y=1: the same way, a test;
// the default: NDC's up against uv's down).
float velo_y_sign() {
    static const float s = [] {
        const char* e = std::getenv("BBHOST_UPSCALE_VELO_Y");
        return e && e[0] == '1' ? -1.0f : 1.0f;
    }();
    return s;
}

// How far a texel may be from the camera's pass's value and still be the
// camera's: a step and a half of its 8 bits (BBHOST_UPSCALE_VELO_TOL, in steps).
float velo_tolerance() {
    static const float t = [] {
        const char* e = std::getenv("BBHOST_UPSCALE_VELO_TOL");
        const float steps = e ? static_cast<float>(std::atof(e)) : 1.5f;
        return (steps > 0.0f ? steps : 1.5f) / 255.0f;
    }();
    return t;
}

VkDescriptorSetLayout g_motion_set_layout = VK_NULL_HANDLE;
VkPipelineLayout g_motion_layout = VK_NULL_HANDLE;
VkPipeline g_motion_pipeline = VK_NULL_HANDLE;
bool g_motion_tried = false;

bool motion_pipeline_locked() {
    if (g_motion_tried) return g_motion_pipeline != VK_NULL_HANDLE;
    g_motion_tried = true;
    const VkDescriptorSetLayoutBinding binds[4] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 4;
    sli.pBindings = binds;
    if (vkCreateDescriptorSetLayout(g.device, &sli, nullptr, &g_motion_set_layout) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 160};
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
// absent: all zeros), and the meshes' own from the game's velocity map, a
// view of it `velo` of velo_w x velo_h (null: none).
void record_motion(VkCommandBuffer cb, VkImageView depth_view, VkImageLayout depth_layout, VkImage motion, VkImageView motion_view,
                   std::uint32_t w, std::uint32_t h, bool prev_ok, VkImageView velo, std::uint32_t velo_w, std::uint32_t velo_h,
                   int velo_mode) {
    float p[40] = {};
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
    if (velo && velo_mode == 3) {
        // Rebuilt: NDC motion as it is, at 8 bits x 2 + 0.5.
        const bool b8 = host_opt_motion_rebuilt_bits() == 8;
        p[32] = p[33] = b8 ? 0.5f : 1.0f;
        p[34] = p[35] = b8 ? 0.5f : 0.0f;
        p[36] = static_cast<float>(velo_w);
        p[37] = static_cast<float>(velo_h);
        p[38] = 3.0f;
    } else if (velo) {
        // The camera's pass: (uv now - uv then) x scale x 2.5. A mesh's:
        // (NDC now - NDC then) x scale x 0.5, its y up where uv's is down.
        p[32] = g_velo_cam[0] * 2.5f;
        p[33] = g_velo_cam[1] * 2.5f;
        p[34] = g_velo_mesh[0];
        p[35] = -g_velo_mesh[1] * velo_y_sign();
        p[36] = static_cast<float>(velo_w);
        p[37] = static_cast<float>(velo_h);
        p[38] = static_cast<float>(velo_mode);
        p[39] = velo_tolerance();
    }
    const VkDescriptorSet set = alloc_set_locked(g_motion_set_layout);
    if (!set) return;
    const VkDescriptorImageInfo depth_info{VK_NULL_HANDLE, depth_view, depth_layout};
    const VkDescriptorImageInfo sampler_info{g.dummy_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    const VkDescriptorImageInfo motion_info{VK_NULL_HANDLE, motion_view, VK_IMAGE_LAYOUT_GENERAL};
    // No map: the depth stands in, unread.
    const VkDescriptorImageInfo velo_info{VK_NULL_HANDLE, velo ? velo : depth_view, velo ? VK_IMAGE_LAYOUT_GENERAL : depth_layout};
    VkWriteDescriptorSet wr[4] = {};
    for (int k = 0; k < 4; ++k) {
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
    wr[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    wr[3].pImageInfo = &velo_info;
    vkUpdateDescriptorSets(g.device, 4, wr, 0, nullptr);
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

// Sharpness: RCAS of g_output into g_sharp.
VkDescriptorSetLayout g_rcas_set_layout = VK_NULL_HANDLE;
VkPipelineLayout g_rcas_layout = VK_NULL_HANDLE;
VkPipeline g_rcas_pipeline = VK_NULL_HANDLE;
bool g_rcas_tried = false;

bool rcas_pipeline_locked() {
    if (g_rcas_tried) return g_rcas_pipeline != VK_NULL_HANDLE;
    g_rcas_tried = true;
    const VkDescriptorSetLayoutBinding binds[3] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 3;
    sli.pBindings = binds;
    if (vkCreateDescriptorSetLayout(g.device, &sli, nullptr, &g_rcas_set_layout) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_rcas_set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(g.device, &pli, nullptr, &g_rcas_layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(k_upscale_rcas_spv);
    smi.pCode = k_upscale_rcas_spv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.device, &smi, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = g_rcas_layout;
    const bool ok = vkCreateComputePipelines(g.device, g.cache, 1, &ci, nullptr, &g_rcas_pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(g.device, module, nullptr);
    if (!ok) g_rcas_pipeline = VK_NULL_HANDLE;
    host_log("upscaler: the sharpening pass (RCAS) %s", ok ? "ready" : "failed to build; no sharpening");
    return ok;
}

// The picture sharpened by `sharpness` (0..1, AMD's scale: 1 the most) into
// g_sharp; false when it could not be (the picture stands as it is).
// g_output has been written and made readable.
bool record_rcas(VkCommandBuffer cb, float sharpness) {
    if (!rcas_pipeline_locked()) return false;
    if (!g_sharp.image &&
        !make(g_sharp, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
              g_ow, g_oh)) {
        drop(g_sharp);
        return false;
    }
    const VkDescriptorSet set = alloc_set_locked(g_rcas_set_layout);
    if (!set) return false;
    const VkDescriptorImageInfo src_info{VK_NULL_HANDLE, g_output.view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo sampler_info{g.dummy_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    const VkDescriptorImageInfo dst_info{VK_NULL_HANDLE, g_sharp.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr[3] = {};
    for (int k = 0; k < 3; ++k) {
        wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[k].dstSet = set;
        wr[k].dstBinding = static_cast<std::uint32_t>(k);
        wr[k].descriptorCount = 1;
    }
    wr[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    wr[0].pImageInfo = &src_info;
    wr[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    wr[1].pImageInfo = &sampler_info;
    wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[2].pImageInfo = &dst_info;
    vkUpdateDescriptorSets(g.device, 3, wr, 0, nullptr);
    // FsrRcasCon: the strength in stops below the most - AMD's FSR 2 maps its
    // 0..1 sharpness to 2..0 stops.
    const float stops = 2.0f * (1.0f - std::clamp(sharpness, 0.0f, 1.0f));
    const float scale = std::exp2(-stops);
    std::uint32_t pc[8] = {};
    std::memcpy(&pc[0], &scale, 4);
    pc[4] = g_ow;
    pc[5] = g_oh;
    barrier(cb, g_sharp.image, VK_IMAGE_ASPECT_COLOR_BIT, g_sharp.initialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    g_sharp.initialised = true;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_rcas_pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_rcas_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, g_rcas_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cb, (g_ow + 15) / 16, (g_oh + 15) / 16, 1);
    barrier(cb, g_sharp.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    return true;
}

// Sharpness as the settings have it now, 0..1 (0: none).
float sharpness_now() { return std::clamp(host_settings_ref().sharpness, 0.0f, 1.0f); }

// BBHOST_UPSCALE_SHOW_MOTION=<pixels> (a test): the motion as colours in place
// of the upscaler's picture (shaders/motion_view.comp), a full step of colour
// for that many render pixels of motion (1: 8).
float show_motion() {
    static const float px = [] {
        const char* e = std::getenv("BBHOST_UPSCALE_SHOW_MOTION");
        if (!e || !*e || e[0] == '0') return 0.0f;
        const float v = static_cast<float>(std::atof(e));
        return v > 1.0f ? v : 8.0f;
    }();
    return px;
}

// BBHOST_UPSCALE_CHECK=<gain> (a test): the reprojection check in place of the
// upscaler's picture (shaders/motion_check.comp) - black where the motion is
// right - with this frame's scene kept for the next (1: a gain of 4).
float check_motion() {
    static const float gain = [] {
        const char* e = std::getenv("BBHOST_UPSCALE_CHECK");
        if (!e || !*e || e[0] == '0') return 0.0f;
        const float v = static_cast<float>(std::atof(e));
        return v > 1.0f ? v : 4.0f;
    }();
    return gain;
}

// One of the tests' compute passes over g_output: its pipeline, made at the
// first use from `spv` with the bindings `types` (in order) and 32 bytes of
// push constants.
struct TestPass {
    const char* what;
    const std::uint32_t* spv;
    std::size_t spv_bytes;
    std::vector<VkDescriptorType> types;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool tried = false;
};
TestPass g_view_pass{"motion view (BBHOST_UPSCALE_SHOW_MOTION)", k_motion_view_spv, sizeof(k_motion_view_spv),
                     {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}};
TestPass g_check_pass{"reprojection check (BBHOST_UPSCALE_CHECK)", k_motion_check_spv, sizeof(k_motion_check_spv),
                      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                       VK_DESCRIPTOR_TYPE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}};

bool test_pass_locked(TestPass& t) {
    if (t.tried) return t.pipeline != VK_NULL_HANDLE;
    t.tried = true;
    std::vector<VkDescriptorSetLayoutBinding> binds;
    for (std::size_t i = 0; i < t.types.size(); ++i)
        binds.push_back({static_cast<std::uint32_t>(i), t.types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = static_cast<std::uint32_t>(binds.size());
    sli.pBindings = binds.data();
    if (vkCreateDescriptorSetLayout(g.device, &sli, nullptr, &t.set_layout) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &t.set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(g.device, &pli, nullptr, &t.layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = t.spv_bytes;
    smi.pCode = t.spv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.device, &smi, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = t.layout;
    const bool ok = vkCreateComputePipelines(g.device, g.cache, 1, &ci, nullptr, &t.pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(g.device, module, nullptr);
    if (!ok) t.pipeline = VK_NULL_HANDLE;
    host_log("upscaler: the %s %s", t.what, ok ? "ready" : "failed to build");
    return ok;
}

// The pass `t` over g_output (its last binding), its other bindings the
// images `views` and the sampler; the push constants the output's size, the
// render size and `k`.
void record_test_pass(VkCommandBuffer cb, TestPass& t, std::initializer_list<VkImageView> views, float k) {
    if (!test_pass_locked(t)) return;
    const VkDescriptorSet set = alloc_set_locked(t.set_layout);
    if (!set) return;
    std::vector<VkDescriptorImageInfo> infos(t.types.size());
    std::vector<VkWriteDescriptorSet> wr(t.types.size());
    auto view = views.begin();
    for (std::size_t i = 0; i < t.types.size(); ++i) {
        if (t.types[i] == VK_DESCRIPTOR_TYPE_SAMPLER) infos[i] = {g.dummy_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        else if (t.types[i] == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) infos[i] = {VK_NULL_HANDLE, g_output.view, VK_IMAGE_LAYOUT_GENERAL};
        else infos[i] = {VK_NULL_HANDLE, view != views.end() ? *view++ : VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL};
        wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wr[i].dstSet = set;
        wr[i].dstBinding = static_cast<std::uint32_t>(i);
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = t.types[i];
        wr[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(g.device, static_cast<std::uint32_t>(wr.size()), wr.data(), 0, nullptr);
    const float pc[8] = {static_cast<float>(g_ow), static_cast<float>(g_oh), static_cast<float>(g_w), static_cast<float>(g_h), k, 0, 0, 0};
    barrier(cb, g_output.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, t.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, t.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, t.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cb, (g_ow + 7) / 8, (g_oh + 7) / 8, 1);
}

// The reprojection check's last scene: a copy of each frame's at the call point.
OwnImage g_prev_scene;
VkFormat g_prev_scene_format = VK_FORMAT_UNDEFINED;
bool g_prev_scene_valid = false;

// The check of this frame's motion against the last frame's scene, into
// g_output; then this frame's scene kept for the next.
void record_motion_check(VkCommandBuffer cb, RtImage& scene, float gain) {
    if (g_prev_scene.image && (g_prev_scene_format != scene.format)) {
        drop(g_prev_scene);
        g_prev_scene_valid = false;
    }
    if (!g_prev_scene.image) {
        if (!make(g_prev_scene, scene.format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, g_w, g_h)) {
            drop(g_prev_scene);
            return;
        }
        g_prev_scene_format = scene.format;
        g_prev_scene_valid = false;
    }
    if (g_prev_scene_valid) record_test_pass(cb, g_check_pass, {scene.view, g_prev_scene.view, g_motion.view}, gain);
    barrier(cb, g_prev_scene.image, VK_IMAGE_ASPECT_COLOR_BIT, g_prev_scene_valid ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    barrier(cb, scene.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
    VkImageCopy r{};
    r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.extent = {g_w, g_h, 1};
    vkCmdCopyImage(cb, scene.image, VK_IMAGE_LAYOUT_GENERAL, g_prev_scene.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    barrier(cb, g_prev_scene.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT);
    barrier(cb, scene.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT);
    g_prev_scene_valid = true;
}

// A test's picture in the upscaler's place (no sharpening over it).
bool test_view() { return show_motion() > 0.0f || check_motion() > 0.0f; }

// Whether the bridge's RGBA16F picture can be blitted into a scene of `format`.
bool blit_into(VkFormat format) {
    VkFormatProperties src{}, dst{};
    vkGetPhysicalDeviceFormatProperties(g.phys, VK_FORMAT_R16G16B16A16_SFLOAT, &src);
    vkGetPhysicalDeviceFormatProperties(g.phys, format, &dst);
    return (src.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) && (dst.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);
}

// Frame generation's prepare (host/framegen.h): this frame's depth and motion,
// the jitter the draws had and the camera's vertical field of view. The frame
// it is prepared as is the one its HUD-less copy is then of.
void prepare_frame_generation(VkCommandBuffer cb, const RtImage& depth, VkImageLayout depth_layout, float jitter_x, float jitter_y,
                              float motion_scale) {
    FramegenInputs fi;
    fi.depth = depth.image;
    fi.depth_format = depth.format;
    fi.depth_layout = depth_layout;
    fi.motion = g_motion.image;
    fi.render_w = g_w;
    fi.render_h = g_h;
    fi.jitter_x = jitter_x;
    fi.jitter_y = jitter_y;
    fi.motion_scale = motion_scale;
    if (g_cam_now.tan_y > 0.0f) fi.fov_y = 2.0f * std::atan(g_cam_now.tan_y);
    g_fg_pending = framegen_prepare(cb, g_fg_frames + 1, fi) ? ++g_fg_frames : 0;
    static bool told = false;
    if (!told) host_log("framegen: first prepare %s", g_fg_pending ? "recorded" : "refused");
    told = true;
}

}  // namespace

bool upscale_motion_on() { return upscaler_method() || framegen_active(); }

bool upscale_velo_clear_due(std::uint64_t base, int kind) {
    return kind == 2 && base && g_velo_cleared != g_velo_frame && !g_velo_camera_pass;
}

void upscale_velo_clear_locked(RtImage& velo) {
    g_velo_cleared = g_velo_frame;
    if (!velo.image) return;
    begin_recording_locked();
    if (velo.view_clear) view_clear_apply_locked(velo);
    render_end_pass_locked();
    if (velo.managed_layout) rt_set_layout_locked(velo, VK_IMAGE_LAYOUT_GENERAL);
    VkCommandBuffer cb = g_cmd();
    barrier(cb, velo.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    // No motion: rebuilt's texels 0 with no mark (alpha 0), the game's
    // encoding's zero, 0.5, as its still camera's pass writes it.
    VkClearColorValue none{};
    if (!host_opt_motion_rebuilt_bits()) {
        none.float32[0] = none.float32[1] = 0.5f;
        none.float32[3] = 1.0f;
    }
    const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, velo.image, VK_IMAGE_LAYOUT_GENERAL, &none, 1, &all);
    barrier(cb, velo.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
    static bool told = false;
    if (!told)
        host_log("upscaler: the velocity map 0x%llx cleared before its passes (no camera pass covers it)",
                 static_cast<unsigned long long>(velo.base));
    told = true;
}

bool upscale_velo_copy_due(std::uint64_t base) { return base && base == g_velo_base && g_velo_copied != g_velo_frame; }

void upscale_velo_copy_locked(RtImage& velo) {
    g_velo_copied = g_velo_frame;
    VeloCopy& k = g_velo_copy;
    if (!velo.image) return;
    if (k.image && (k.format != velo.format || k.w != velo.width || k.h != velo.height)) {
        defer_destroy_private_view(k.view);
        defer_destroy_image(k.image, k.memory);
        k = VeloCopy{};
    }
    if (!k.image) {
        OwnImage o;
        if (!make(o, velo.format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, velo.width, velo.height)) {
            drop(o);
            g_velo_copied = 0;
            return;
        }
        k.image = o.image;
        k.memory = o.memory;
        k.view = o.view;
        k.format = velo.format;
        k.w = velo.width;
        k.h = velo.height;
        host_log("upscaler: the velocity map 0x%llx (%ux%u format %d) copied before the motion blur's dilation spreads it",
                 static_cast<unsigned long long>(velo.base), velo.width, velo.height, static_cast<int>(velo.format));
    }
    begin_recording_locked();
    if (velo.view_clear) view_clear_apply_locked(velo);
    render_end_pass_locked();
    if (velo.managed_layout) rt_set_layout_locked(velo, VK_IMAGE_LAYOUT_GENERAL);
    VkCommandBuffer cb = g_cmd();
    barrier(cb, velo.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    barrier(cb, k.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy r{};
    r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.extent = {velo.width, velo.height, 1};
    vkCmdCopyImage(cb, velo.image, VK_IMAGE_LAYOUT_GENERAL, k.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    barrier(cb, velo.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
    barrier(cb, k.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT);
}

bool upscale_pass_run_locked(RtImage& scene, RtImage& depth, float jitter_x, float jitter_y) {
    const int method = upscaler_method();
    // Frame generation with no upscaler running (Upscaling off, or one this
    // GPU cannot run): the motion and its prepare at the scene's size, and the
    // scene left as the game drew it.
    const bool motion_only = !method && framegen_active();
    if (!method && !motion_only) return false;
    // The output size: render scale's when the scene is its render size,
    // else the scene's own (DLAA, FSR native AA).
    RenderScale rs;
    const bool scaled = method && render_scale_on(&rs) && scene.width == rs.render_w && scene.height == rs.render_h &&
                        (rs.output_w != rs.render_w || rs.output_h != rs.render_h);
    // A picture that goes back into the scene: by a copy, or a blit into
    // another format. One that stays here (render scale) needs neither.
    const bool same_format = scene.format == VK_FORMAT_R16G16B16A16_SFLOAT;
    static VkFormat checked = VK_FORMAT_UNDEFINED;
    static bool can_blit = false;
    if (method && !scaled && !same_format && checked != scene.format) {
        checked = scene.format;
        can_blit = blit_into(scene.format);
        host_log("upscaler: the scene is format %d: %s", static_cast<int>(scene.format),
                 can_blit ? "the upscaled picture goes back by a blit" : "this device cannot blit into it; not upscaled");
    }
    if (method && !scaled && !same_format && !can_blit) {
        g_skipped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const std::uint32_t ow = scaled ? rs.output_w : scene.width, oh = scaled ? rs.output_h : scene.height;
    if (scene.width != g_w || scene.height != g_h || ow != g_ow || oh != g_oh || motion_only != !g_output.image) {
        drop(g_output);
        drop(g_motion);
        drop(g_sharp);
        drop(g_prev_scene);
        g_prev_scene_valid = false;
        g_w = scene.width;
        g_h = scene.height;
        g_ow = ow;
        g_oh = oh;
        g_scaled = scaled;
        g_output_ready = false;
        g_reset = true;
        // The upscaler's picture only with an upscaler.
        if ((!motion_only && !make(g_output, VK_FORMAT_R16G16B16A16_SFLOAT,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, g_ow, g_oh)) ||
            !make(g_motion, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  g_w, g_h)) {
            host_log("upscaler: no images of %ux%u and %ux%u for it; not upscaled", g_ow, g_oh, g_w, g_h);
            drop(g_output);
            drop(g_motion);
            g_w = g_h = g_ow = g_oh = 0;
            g_scaled = false;
            return false;
        }
        if (motion_only)
            host_log("framegen: the motion at %ux%u (the scene's size, no upscaler running), scene format %d, depth format %d", g_w, g_h,
                     static_cast<int>(scene.format), static_cast<int>(depth.format));
        // The materials' textures sampled as at the output size: the
        // bridge's mip bias (BBHOST_UPSCALE_MIP_BIAS=0: none, a test).
        static const bool mip_bias_on = [] {
            const char* e = std::getenv("BBHOST_UPSCALE_MIP_BIAS");
            return !(e && e[0] == '0');
        }();
        g_mip_bias = !mip_bias_on || motion_only ? 0.0f
                     : g_scaled               ? upscaler_mip_bias_sized(method, g_w, g_ow)
                                              : upscaler_mip_bias(method, 0, g_ow, g_oh);
        if (!motion_only) {
            host_log("upscaler: the camera's draws sample their textures with a LOD bias of %.3f%s", static_cast<double>(g_mip_bias),
                     mip_bias_on ? "" : " (BBHOST_UPSCALE_MIP_BIAS=0)");
            if (g_scaled)
                host_log("upscaler: %s from %ux%u to %ux%u (Upscaling quality %d), scene format %d, depth format %d; YEBIS's copy reads the picture",
                         method == 1 ? "DLSS" : "FSR", g_w, g_h, g_ow, g_oh, rs.quality, static_cast<int>(scene.format),
                         static_cast<int>(depth.format));
            else
                host_log("upscaler: %s at %ux%u (native, the scene's size), scene format %d, depth format %d",
                         method == 1 ? "DLAA" : "FSR native AA", g_w, g_h, static_cast<int>(scene.format), static_cast<int>(depth.format));
        }
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
    // The meshes' own motion from the game's velocity map, its passes drawn
    // this frame (BBHOST_UPSCALE_VELO=0: none, the camera's alone; =all: the
    // map's motion everywhere, the camera's pass's too - tests). Rebuilt
    // motion vectors (the Motion vectors setting, as it is now): mode 3. The
    // map as its passes drew it: the copy taken before the motion blur's
    // dilation spread it, else the map itself.
    static const int velo_env = [] {
        const char* e = std::getenv("BBHOST_UPSCALE_VELO");
        return !e ? 1 : e[0] == '0' ? 0 : std::strcmp(e, "all") == 0 ? 2 : 1;
    }();
    const int velo_mode = !velo_env ? 0 : host_opt_motion_rebuilt_bits() ? 3 : velo_env;
    VkImageView velo = VK_NULL_HANDLE;
    std::uint32_t velo_w = 0, velo_h = 0;
    if (velo_mode && g_velo_cam_seen && (g_velo_mesh_known || velo_mode == 3) && g_cam_prev_ok) {
        if (g_velo_copied == g_velo_frame && g_velo_copy.view) {
            velo = g_velo_copy.view;
            velo_w = g_velo_copy.w;
            velo_h = g_velo_copy.h;
        } else if (RtImage* map = find_render_target(g_velo_base); map && map->image && map->view && !map->depth) {
            if (map->view_clear) view_clear_apply_locked(*map);
            render_end_pass_locked();
            if (map->managed_layout) rt_set_layout_locked(*map, VK_IMAGE_LAYOUT_GENERAL);
            barrier(cb, map->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
            velo = map->view;
            velo_w = map->width;
            velo_h = map->height;
        }
        if (velo) ++g_velo_used;
    }
    g_velo_cam_seen = false;
    // The next frame's map work (upscale_velo_clear_due, upscale_velo_copy_due).
    ++g_velo_frame;
    g_velo_camera_pass = false;
    if (g_cam_noted && motion_mode != 0 && motion_pipeline_locked()) {
        record_motion(cb, g_depth_view, depth_layout, g_motion.image, g_motion.view, g_w, g_h, g_cam_prev_ok, velo, velo_w, velo_h,
                      velo_mode);
        g_velo_cam[0] = g_velo_cam[1] = 0.0f;
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
    if (motion_only) {
        // No upscaler: frame generation's prepare alone - no jitter (the
        // draws are moved only while an upscaler runs) - and AMD's binds gone
        // from cb, as the bridge's are below.
        prepare_frame_generation(cb, depth, depth_layout, 0.0f, 0.0f, motion_mode < 0 ? -1.0f : 1.0f);
        ::run_forget_recorded_locked();
        return false;
    }
    UpscaleCall c;
    c.method = method;
    c.quality = g_scaled ? static_cast<std::uint32_t>(rs.quality) : 0;
    c.reset = g_reset;
    c.render_w = g_w;
    c.render_h = g_h;
    c.output_w = g_ow;
    c.output_h = g_oh;
    c.color = {scene.image, scene.view, scene.format, VK_IMAGE_LAYOUT_GENERAL};
    c.depth = {depth.image, g_depth_view, depth.format, depth_layout};
    c.motion = {g_motion.image, g_motion.view, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL};
    c.motion_scale = motion_mode < 0 ? -1.0f : 1.0f;
    c.dlss_preset = static_cast<std::uint32_t>(host_settings_ref().dlss_preset);
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
    if (framegen_active()) {
        prepare_frame_generation(cb, depth, depth_layout, jitter_x, jitter_y, c.motion_scale);
        ::run_forget_recorded_locked();
    }
    // (The tests: the motion, or its check, in the picture's place.)
    if (const float px = show_motion(); px > 0.0f) record_test_pass(cb, g_view_pass, {g_motion.view}, px);
    else if (const float gain = check_motion(); gain > 0.0f) record_motion_check(cb, scene, gain);
    barrier(cb, g_output.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    // Sharpness: the sharpened picture goes on in its place.
    const float sharpness = test_view() ? 0.0f : sharpness_now();
    const OwnImage& picture = sharpness > 0.0f && record_rcas(cb, sharpness) ? g_sharp : g_output;
    if (g_scaled) {
        // The picture stays here for YEBIS's copy, which samples it in the
        // scene's place.
        g_output_ready = true;
        if (g_calls.fetch_add(1, std::memory_order_relaxed) == 0)
            host_log("upscaler: first upscale of the scene 0x%llx (depth 0x%llx), %ux%u to %ux%u", static_cast<unsigned long long>(scene.base),
                     static_cast<unsigned long long>(depth.base), g_w, g_h, g_ow, g_oh);
        return true;
    }
    // Its picture into the scene.
    barrier(cb, scene.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT);
    if (same_format) {
        VkImageCopy r{};
        r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.extent = {g_w, g_h, 1};
        vkCmdCopyImage(cb, picture.image, VK_IMAGE_LAYOUT_GENERAL, scene.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    } else {
        VkImageBlit r{};
        r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.srcOffsets[1] = r.dstOffsets[1] = {static_cast<std::int32_t>(g_w), static_cast<std::int32_t>(g_h), 1};
        vkCmdBlitImage(cb, picture.image, VK_IMAGE_LAYOUT_GENERAL, scene.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r, VK_FILTER_NEAREST);
    }
    barrier(cb, scene.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    if (g_calls.fetch_add(1, std::memory_order_relaxed) == 0)
        host_log("upscaler: first upscale of the scene 0x%llx (depth 0x%llx)", static_cast<unsigned long long>(scene.base),
                 static_cast<unsigned long long>(depth.base));
    return true;
}

bool upscale_pass_hudless_due_locked() { return g_fg_pending != 0; }

void upscale_pass_hudless_locked(RtImage& display) {  // `display`: the HUD's target
    if (!g_fg_pending) return;
    // Four of ours in turn: the presenter may still be blitting the last
    // frames' while this one is copied.
    struct Copy {
        OwnImage o;
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::uint32_t w = 0, h = 0;
    };
    static Copy copies[4];
    static unsigned next = 0;
    Copy& k = copies[next++ % 4];
    if (k.o.image && (k.format != display.format || k.w != display.width || k.h != display.height)) drop(k.o);
    if (!k.o.image) {
        if (!make(k.o, display.format, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  display.width, display.height)) {
            g_fg_pending = 0;
            return;
        }
        k.format = display.format;
        k.w = display.width;
        k.h = display.height;
    }
    begin_recording_locked();
    if (display.view_clear) view_clear_apply_locked(display);
    render_end_pass_locked();
    VkCommandBuffer cb = g_cmd();
    barrier(cb, display.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    barrier(cb, k.o.image, VK_IMAGE_ASPECT_COLOR_BIT, k.o.initialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy r{};
    r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.extent = {display.width, display.height, 1};
    vkCmdCopyImage(cb, display.image, VK_IMAGE_LAYOUT_GENERAL, k.o.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    barrier(cb, display.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
    barrier(cb, k.o.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
    k.o.initialised = true;
    g_fg_copy = k.o.image;
    g_fg_copy_format = display.format;
    g_fg_copy_w = display.width;
    g_fg_copy_h = display.height;
}

bool upscale_pass_hudless_copied_locked() { return g_fg_copy != VK_NULL_HANDLE; }

void upscale_pass_hudless_bind_locked(std::uint64_t display_va, std::uint32_t display_w, std::uint32_t display_h) {
    if (g_fg_pending && g_fg_copy)
        framegen_note_hudless(display_va, g_fg_pending, g_fg_copy, g_fg_copy_format, g_fg_copy_w, g_fg_copy_h, display_w, display_h);
    g_fg_pending = 0;  // once a frame
    g_fg_copy = VK_NULL_HANDLE;
}

// A frame it could not run in starts the next one's history over.
void upscale_pass_reset_locked() { g_reset = true; }

float upscale_pass_mip_bias_locked() { return g_w && upscaler_method() ? g_mip_bias : 0.0f; }

VkImageView upscale_pass_output_view_locked(std::uint32_t scene_w, std::uint32_t scene_h) {
    if (!g_scaled || !g_output_ready || !g_output.view || scene_w != g_w || scene_h != g_h) return VK_NULL_HANDLE;
    // The image this frame's upscale leaves its picture in: sharpened or not,
    // as the setting is now.
    return sharpness_now() > 0.0f && !test_view() && g_sharp.initialised && g_sharp.view ? g_sharp.view : g_output.view;
}

bool upscale_camera_noted() { return g_cam_noted; }

void upscale_velo_note_locked(int kind, std::uint64_t base, float kx, float ky) {
    if (kind != 3 && (!std::isfinite(kx) || !std::isfinite(ky) || kx == 0.0f || ky == 0.0f)) return;
    if (kind == 3) kind = 1;  // the camera's pass, still: a scale of 0 (it wrote no motion)
    static int logs = 0;
    const bool changed = kind == 1 ? (kx != g_velo_cam[0] || ky != g_velo_cam[1] || base != g_velo_base)
                                   : (kx != g_velo_mesh[0] || ky != g_velo_mesh[1]);
    if (kind == 1) {
        g_velo_base = base;
        g_velo_cam[0] = kx;
        g_velo_cam[1] = ky;
        g_velo_cam_seen = true;
        g_velo_camera_pass = true;  // it covers the map: no clear before the meshes'
    } else {
        // With the camera's blur off (the Performance patch) there is no
        // camera pass: the map is cleared to no motion before the meshes'
        // passes (upscale_velo_clear_locked) and they alone draw into it.
        g_velo_base = base;
        g_velo_mesh[0] = kx;
        g_velo_mesh[1] = ky;
        g_velo_mesh_known = true;
        g_velo_cam_seen = true;
    }
    if (changed && logs < 8) {
        ++logs;
        const RtImage* rt = find_render_target(base);
        host_log("upscaler: velocity map 0x%llx (%ux%u format %d): the %s pass's scale %g %g (read in %llu frames so far)",
                 static_cast<unsigned long long>(base), rt ? rt->width : 0, rt ? rt->height : 0, rt ? static_cast<int>(rt->format) : 0,
                 kind == 1 ? "camera" : "meshes'", static_cast<double>(kx), static_cast<double>(ky),
                 static_cast<unsigned long long>(g_velo_used));
    }
}

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
