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
// TAA (Graphics > Upscaling: TAA) runs here in the bridge's place:
// Numity1504's TAA (record_taa, shaders/taa_resolve.comp) of the same
// jittered scene, motion and depth, always at the scene's size, with two
// histories of its own and the ember mask inside it. Sharpness follows it as
// it follows the bridge, and frame generation is prepared after it.
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
//
// The embers (Numity1504's handoff step 8, shaders/upscale_embers.comp): after
// the sharpening, the bright one-frame dots the upscaler lost (embers, sparks)
// put back from the game's own scene where its glow target shows them, into
// an image of their own that goes on in the picture's place. The glow is the
// second target, RGBA16F at the scene's size, of a draw that writes the
// scene (render.cpp). BBHOST_EMBERS=0 off.
#include "host/framegen.h"
#include "engine/render_scale.h"
#include "host/gpu_internal.h"
#include "host/shaders/camera_motion.spv.h"
#include "host/shaders/capture_motion_frag.spv.h"
#include "host/shaders/capture_motion_vert.spv.h"
#include "host/shaders/motion_check.spv.h"
#include "host/shaders/motion_view.spv.h"
#include "host/shaders/taa_resolve.spv.h"
#include "host/shaders/upscale_embers.spv.h"
#include "host/shaders/upscale_rcas.spv.h"
#include "host/options.h"
#include "host/settings.h"
#include "host/upscaler.h"
#include "log.h"

#include <algorithm>
#include <unordered_set>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <utility>

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
// The depth the upscaler and frame generation read: an R32_SFLOAT copy of the
// G-buffer's, written by the motion pass. The game's is D32_SFLOAT_S8_UINT,
// and FSR makes its own view of the image it is handed: of a depth-stencil
// image that view did not read clean depth (Numity1504's bridge notes, section
// 5, left it open), so FSR could not tell a moving object's edges from what
// is behind them - no history thrown away where the object had been, its
// motion not taken from the nearest surface: the object smeared with its own
// old frames. BBNextGenSuite hands FSR plain depth. Fresh when the motion
// pass ran this frame (g_depth_r32_fresh), else the game's own is handed on.
OwnImage g_depth_r32;
bool g_depth_r32_fresh = false;
OwnImage g_embers;                  // the picture with the embers put back, the output size; made at the first use
VkImageView g_picture_view = VK_NULL_HANDLE;  // this frame's final picture (render scale: YEBIS's copy samples it)
OwnImage g_sharp;                   // the picture sharpened (Sharpness), the output size; made at the first use
OwnImage g_taa_hist[2];             // TAA's two histories (record_taa): one read, one written, changing places each frame
int g_taa_read = 0;                 // the one this frame's TAA reads
int g_last_method = 0;              // the last call's method (another one's history is not this one's)
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
// BBHOST_GPU_PROFILE: the call point's own work by name (=3 lists them all).
const std::string g_prof_motion = "upscale-motion", g_prof_taa = "upscale-taa", g_prof_bridge = "upscale-bridge",
                  g_prof_sharpen = "upscale-sharpen", g_prof_embers = "upscale-embers";
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
std::atomic<bool> g_cam_noted{false};  // this frame's taken (render.cpp, the first camera draw)
bool g_cam_prev_ok = false;

// What draws note for the upscale - the camera's view block, the G-buffer's
// viewport, the velocity map's scales, the meshes its velocity pass moves -
// is noted under this lock: a run part's draw (BBHOST_RUN_COMMIT) records on
// a worker of its own, beside the other parts. (The characters' capture
// places are taken in order instead - render.cpp stops a part's draw there:
// two parts taking them at once tore the capture's count of each key, and
// the next look-up read through a null - Kyo's friend's crash with an
// upscaler at 4K, ~10 s after a lamp travel, silent when the heap's own
// check caught the damage first.)
std::mutex g_note_mu;

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
// The last view block's tangents and depth decode (view distance =
// 1 / (z raw + w)), for the cutaway (upscale_view_terms).
std::atomic<float> g_view_tan_x{0}, g_view_tan_y{0}, g_view_z{0}, g_view_w{0};

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
    g_view_tan_x.store(tx, std::memory_order_relaxed);
    g_view_tan_y.store(ty, std::memory_order_relaxed);
    g_view_z.store(z, std::memory_order_relaxed);
    g_view_w.store(w, std::memory_order_relaxed);
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
    const VkDescriptorSetLayoutBinding binds[6] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 6;
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
bool velo_seen_test();  // below
VkImageView g_cap_map_view = VK_NULL_HANDLE;  // the characters' capture map (record_capture, below)

// The motion pass's constants (camera_motion.comp's Params; TAA's open with
// the same, taa_resolve.comp).
void motion_params(float (&p)[40], std::uint32_t w, std::uint32_t h, bool prev_ok, VkImageView velo, std::uint32_t velo_w,
                   std::uint32_t velo_h, int velo_mode, bool whole) {
    std::memset(p, 0, sizeof(p));
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
        // Rebuilt: NDC motion as it is, at 8 bits x 2 + 0.5; whole when the
        // pass had last frame's matrices (render.cpp velo_previous_locked).
        const bool b8 = host_opt_motion_rebuilt_bits() == 8;
        p[32] = p[33] = b8 ? 0.5f : 1.0f;
        p[34] = p[35] = b8 ? 0.5f : 0.0f;
        p[36] = static_cast<float>(velo_w);
        p[37] = static_cast<float>(velo_h);
        // (16 and 32 bits: each mesh texel's view distance in blue, tested
        // against the depth - mode 5.)
        p[38] = host_opt_motion_rebuilt_bits() >= 16 && velo_seen_test() ? 5.0f : 3.0f;
        p[39] = whole ? 1.0f : 0.0f;
    } else if (velo) {
        // The camera's pass: (uv now - uv then) x scale x 2.5. A mesh's:
        // (NDC now - NDC then) x scale x 0.5, its y up where uv's is down.
        p[32] = g_velo_cam[0] * 2.5f;
        p[33] = g_velo_cam[1] * 2.5f;
        p[34] = g_velo_mesh[0];
        p[35] = -g_velo_mesh[1] * velo_y_sign();
        p[36] = static_cast<float>(velo_w);
        p[37] = static_cast<float>(velo_h);
        // The game's encoding with the pass given last frame's matrices: the
        // whole motion (mode 4, render.cpp velo_previous_locked).
        p[38] = static_cast<float>(velo_mode == 1 && whole ? 4 : velo_mode);
        p[39] = velo_tolerance();
    }
}

void record_motion(VkCommandBuffer cb, VkImageView depth_view, VkImageLayout depth_layout, VkImage motion, VkImageView motion_view,
                   std::uint32_t w, std::uint32_t h, bool prev_ok, VkImageView velo, std::uint32_t velo_w, std::uint32_t velo_h,
                   int velo_mode, bool whole) {
    float p[40];
    motion_params(p, w, h, prev_ok, velo, velo_w, velo_h, velo_mode, whole);
    const VkDescriptorSet set = alloc_set_locked(g_motion_set_layout);
    if (!set) return;
    const VkDescriptorImageInfo depth_info{VK_NULL_HANDLE, depth_view, depth_layout};
    const VkDescriptorImageInfo sampler_info{g.dummy_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    const VkDescriptorImageInfo motion_info{VK_NULL_HANDLE, motion_view, VK_IMAGE_LAYOUT_GENERAL};
    // No map: the depth stands in, unread.
    const VkDescriptorImageInfo velo_info{VK_NULL_HANDLE, velo ? velo : depth_view, velo ? VK_IMAGE_LAYOUT_GENERAL : depth_layout};
    const VkDescriptorImageInfo depth_copy_info{VK_NULL_HANDLE, g_depth_r32.view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo cap_info{VK_NULL_HANDLE, g_cap_map_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wr[6] = {};
    for (int k = 0; k < 6; ++k) {
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
    wr[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[4].pImageInfo = &depth_copy_info;
    wr[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    wr[5].pImageInfo = &cap_info;
    vkUpdateDescriptorSets(g.device, 6, wr, 0, nullptr);
    barrier(cb, motion, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT);
    barrier(cb, g_depth_r32.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT);  // (every texel written again)
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_motion_pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_motion_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, g_motion_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), p);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    barrier(cb, motion, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT);
    barrier(cb, g_depth_r32.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT);
    g_depth_r32_fresh = true;
}

void drop(OwnImage& o) {
    if (o.view) defer_destroy_private_view(o.view);
    if (o.image) defer_destroy_image(o.image, o.memory);
    o = OwnImage{};
}

bool make(OwnImage& o, VkFormat format, VkImageUsageFlags usage, std::uint32_t w, std::uint32_t h,
          VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
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
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    return vkCreateImageView(g.device, &vci, nullptr, &o.view) == VK_SUCCESS;
}

// The characters' capture: BBNextGenSuite's motion, done natively. Each
// skinned G-buffer draw (render.cpp: a camera draw into the upscaler's depth
// whose vertex shader reads a constant buffer by index - a bone palette) is
// given a place in this frame's slot of the capture buffer, and its translated
// vertex shader stores every vertex's clip position there
// (gcn::TranslateOptions::capture_position). At the upscale the frame's draws
// are matched to the last frame's by key (the vertex shader, the index buffer
// and its range, the base vertex, the vertex buffers; the n-th of a key with
// the n-th), and the matched ones' triangles drawn through the G-buffer's
// jittered viewport with both frames' corners (capture_motion.vert/.frag) into
// g_cap.map: NDC now - NDC then, the view distance, the nearest fragment. The
// motion pass takes it before the velocity map where it is the surface seen
// (camera_motion.comp). So a character's motion is what its own vertex shader
// put on screen this frame and the last, never a velocity draw the game left
// out or drew with other matrices. BBHOST_CHAR_CAPTURE=0: not (the velocity
// map alone).
constexpr std::uint32_t kCapSlots = 3;
constexpr std::uint32_t kCapVertices = 1u << 20;  // a slot's corners (16 bytes each)
constexpr std::uint32_t kCapIndexCap = 1u << 23;  // the meshes' indices kept for the motion pass (4 bytes each)
struct CapIndices {
    std::uint32_t vmin = 0, verts = 0;
    std::vector<std::uint32_t> local;  // each index less vmin
    std::uint32_t first = ~0u;         // its copy's place in the capture's index buffer (~0u: none)
};
struct CapDraw {
    std::uint64_t key = 0;     // the draw's own until record_capture, then with its n-th of the frame
    std::uint64_t order = 0;   // its place in the stream (its packet's number)
    std::uint32_t offset = 0;  // in the slot, in corners
    const CapIndices* idx = nullptr;
};
struct CapState {
    bool tried = false, ok = false;
    VkBuffer buf = VK_NULL_HANDLE;  // kCapSlots x kCapVertices vec4s
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceAddress addr = 0;
    // Each mesh's indices, copied once: the motion pass draws a mesh with them
    // and its two places pushed - a frame's corners were written out one by
    // one, ~250k pairs a frame on the command processor.
    VkBuffer ibuf = VK_NULL_HANDLE;
    VkDeviceMemory ibuf_mem = VK_NULL_HANDLE;
    std::uint8_t* ibuf_map = nullptr;
    std::uint32_t ibuf_used = 0;  // under g_cap_mu
    std::uint32_t slot = 0, used = 0;
    std::vector<CapDraw> now, last;  // now: under g_cap_mu (run parts add to it)
    std::uint32_t last_slot = 0;
    bool last_ok = false;
    std::unordered_map<std::uint64_t, std::uint32_t> keys;  // record_capture's: how many of each key so far
    std::unordered_map<std::uint64_t, std::unique_ptr<CapIndices>> indices;  // under g_cap_mu
    // Replaced while this or the last frame's draws may still name them,
    // with the frame they were replaced in (freed two frames on).
    std::vector<std::pair<std::uint64_t, std::unique_ptr<CapIndices>>> retired;
    VkViewport vp{};
    bool vp_set = false;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    OwnImage map, zbuf;
    std::uint32_t map_w = 0, map_h = 0;
    std::uint64_t frames = 0, draws = 0, matched = 0, corner_total = 0, full = 0, renewed = 0;
    std::atomic<std::uint64_t> part_draws{0};  // draws captured from a run part (render.cpp)
};
CapState g_cap;
// A run part's draw (BBHOST_RUN_COMMIT) takes its place in the capture on a
// worker of its own, beside the other parts: the places and the frame's list
// are taken under this lock, and the frame's draws are put in the stream's
// order (CapDraw::order) before the n-th of a key is counted (record_capture).
// Unlocked, two parts tore the count of each key - a null read at the next
// look-up (the crash with an upscaler at 4K) - and handed two draws one place.
std::mutex g_cap_mu;

bool capture_env() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_CHAR_CAPTURE");
        return !e || e[0] != '0';
    }();
    return on;
}

VkBuffer cap_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags want, VkDeviceMemory& memory, VkDeviceAddress* address,
                    std::uint8_t** map) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage | (address ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(g.device, &bci, nullptr, &buffer) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.device, buffer, &req);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    if (address) mai.pNext = &flags;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, want);
    void* m = nullptr;
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &memory) != VK_SUCCESS ||
        vkBindBufferMemory(g.device, buffer, memory, 0) != VK_SUCCESS || (map && vkMapMemory(g.device, memory, 0, VK_WHOLE_SIZE, 0, &m) != VK_SUCCESS)) {
        if (memory) vkFreeMemory(g.device, memory, nullptr);
        memory = VK_NULL_HANDLE;
        vkDestroyBuffer(g.device, buffer, nullptr);
        return VK_NULL_HANDLE;
    }
    if (map) *map = static_cast<std::uint8_t*>(m);
    if (address) {
        VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        bai.buffer = buffer;
        *address = vkGetBufferDeviceAddress(g.device, &bai);
    }
    return buffer;
}

bool capture_ready_locked() {
    if (g_cap.tried) return g_cap.ok;
    g_cap.tried = true;
    g_cap.buf = cap_buffer(VkDeviceSize{kCapSlots} * kCapVertices * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           g_cap.mem, &g_cap.addr, nullptr);
    bool ok = g_cap.buf != VK_NULL_HANDLE;
    if (ok) {
        g_cap.ibuf = cap_buffer(VkDeviceSize{kCapIndexCap} * 4, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, g_cap.ibuf_mem, nullptr, &g_cap.ibuf_map);
        ok = g_cap.ibuf != VK_NULL_HANDLE;
    }
    if (ok) {
        const VkDescriptorSetLayoutBinding bind{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        sli.bindingCount = 1;
        sli.pBindings = &bind;
        ok = vkCreateDescriptorSetLayout(g.device, &sli, nullptr, &g_cap.set_layout) == VK_SUCCESS;
    }
    if (ok) {
        const VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 8};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &g_cap.set_layout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        ok = vkCreatePipelineLayout(g.device, &pli, nullptr, &g_cap.layout) == VK_SUCCESS;
    }
    VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
    if (ok) {
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = sizeof(k_capture_motion_vert_spv);
        smi.pCode = k_capture_motion_vert_spv;
        ok = vkCreateShaderModule(g.device, &smi, nullptr, &vs) == VK_SUCCESS;
        smi.codeSize = sizeof(k_capture_motion_frag_spv);
        smi.pCode = k_capture_motion_frag_spv;
        ok = ok && vkCreateShaderModule(g.device, &smi, nullptr, &fs) == VK_SUCCESS;
    }
    if (ok) {
        VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        st[0].module = vs;
        st[0].pName = "main";
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module = fs;
        st[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1;
        vps.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;  // (the viewport noted at 0..1: nearer is less)
        VkPipelineColorBlendAttachmentState att{};
        att.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cbs.attachmentCount = 1;
        cbs.pAttachments = &att;
        const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dys{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dys.dynamicStateCount = 2;
        dys.pDynamicStates = dyn;
        const VkFormat color = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkPipelineRenderingCreateInfo pr{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        pr.colorAttachmentCount = 1;
        pr.pColorAttachmentFormats = &color;
        pr.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.pNext = &pr;
        gp.stageCount = 2;
        gp.pStages = st;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vps;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cbs;
        gp.pDynamicState = &dys;
        gp.layout = g_cap.layout;
        ok = vkCreateGraphicsPipelines(g.device, g.cache, 1, &gp, nullptr, &g_cap.pipeline) == VK_SUCCESS;
    }
    if (vs) vkDestroyShaderModule(g.device, vs, nullptr);
    if (fs) vkDestroyShaderModule(g.device, fs, nullptr);
    g_cap.ok = ok;
    host_log("upscaler: the characters' capture %s (%u corners a frame in each of %u slots, %u indices kept; BBHOST_CHAR_CAPTURE=0 turns it off)",
             ok ? "ready" : "failed to set up; the velocity map alone", kCapVertices, kCapSlots, kCapIndexCap);
    return ok;
}

// The capture map at the motion's size, cleared, then this frame's matched
// draws drawn into it; the frame's draws become the last frame's.
void record_capture(VkCommandBuffer cb, std::uint32_t w, std::uint32_t h) {
    if (g_cap.map_w != w || g_cap.map_h != h || !g_cap.map.image) {
        drop(g_cap.map);
        drop(g_cap.zbuf);
        g_cap.map_w = g_cap.map_h = 0;
        if (!make(g_cap.map, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, w, h) ||
            !make(g_cap.zbuf, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, w, h, VK_IMAGE_ASPECT_DEPTH_BIT)) {
            drop(g_cap.map);
            drop(g_cap.zbuf);
            return;
        }
        g_cap.map_w = w;
        g_cap.map_h = h;
    }
    // This frame's draws in the stream's order, each key's n-th counted in it
    // (the n-th draw of a key stands for the last frame's n-th).
    std::lock_guard<std::mutex> cap_lk(g_cap_mu);
    std::stable_sort(g_cap.now.begin(), g_cap.now.end(), [](const CapDraw& a, const CapDraw& b) { return a.order < b.order; });
    for (CapDraw& c : g_cap.now) {
        const std::uint32_t nth = g_cap.keys[c.key]++;
        c.key = fnv1a(&nth, sizeof(nth), c.key);
    }
    // The matched draws: each mesh's indices (its copy in the index buffer)
    // with its places this frame and last.
    struct Matched {
        std::uint32_t first, count, bases[2];
    };
    static std::vector<Matched> matched;
    matched.clear();
    std::uint32_t n = 0;
    if (g_cap.ok && g_cap.last_ok && g_cap.vp_set) {
        std::unordered_map<std::uint64_t, const CapDraw*> last;
        last.reserve(g_cap.last.size());
        for (const CapDraw& l : g_cap.last) last.emplace(l.key, &l);
        const std::uint32_t now_base = g_cap.slot * kCapVertices, last_base = g_cap.last_slot * kCapVertices;
        for (const CapDraw& c : g_cap.now) {
            const auto it = last.find(c.key);
            if (it == last.end() || it->second->idx != c.idx || c.idx->first == ~0u) continue;
            const std::uint32_t k = static_cast<std::uint32_t>(c.idx->local.size());
            matched.push_back({c.idx->first, k, {now_base + c.offset, last_base + it->second->offset}});
            n += k;
            ++g_cap.matched;
        }
    }
    g_cap.corner_total += n;
    g_cap.draws += g_cap.now.size();
    if (++g_cap.frames % 600 == 0) {
        host_log("upscaler: the characters' capture over 600 frames: %llu draws a frame, %llu matched, %llu corners drawn, %llu frames full, "
                 "%llu draws from run parts, %llu meshes' indices read again (changed under their buffer)",
                 static_cast<unsigned long long>(g_cap.draws / 600), static_cast<unsigned long long>(g_cap.matched / 600),
                 static_cast<unsigned long long>(g_cap.corner_total / 600), static_cast<unsigned long long>(g_cap.full),
                 static_cast<unsigned long long>(g_cap.part_draws.exchange(0)), static_cast<unsigned long long>(g_cap.renewed));
        g_cap.draws = g_cap.matched = g_cap.corner_total = g_cap.full = g_cap.renewed = 0;
    }
    render_end_pass_locked();
    // This frame's corners, written by the game's vertex shaders.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkImageMemoryBarrier ib[2] = {{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}};
    ib[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    ib[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    ib[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ib[0].srcQueueFamilyIndex = ib[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib[0].image = g_cap.map.image;
    ib[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    ib[1] = ib[0];
    ib[1].srcAccessMask = 0;
    ib[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    ib[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    ib[1].image = g_cap.zbuf.image;
    ib[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 2, ib);
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = g_cap.map.view;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingAttachmentInfo za{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    za.imageView = g_cap.zbuf.view;
    za.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    za.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    za.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    za.clearValue.depthStencil = {1.0f, 0};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {w, h}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    ri.pDepthAttachment = &za;
    vkCmdBeginRendering(cb, &ri);
    if (n) {
        const VkDescriptorSet set = alloc_set_locked(g_cap.set_layout);
        if (set) {
            const VkDescriptorBufferInfo bi{g_cap.buf, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr.dstSet = set;
            wr.dstBinding = 0;
            wr.descriptorCount = 1;
            wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr.pBufferInfo = &bi;
            vkUpdateDescriptorSets(g.device, 1, &wr, 0, nullptr);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_cap.pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_cap.layout, 0, 1, &set, 0, nullptr);
            vkCmdBindIndexBuffer(cb, g_cap.ibuf, 0, VK_INDEX_TYPE_UINT32);
            vkCmdSetViewport(cb, 0, 1, &g_cap.vp);
            const VkRect2D sc{{0, 0}, {w, h}};
            vkCmdSetScissor(cb, 0, 1, &sc);
            for (const Matched& m : matched) {
                vkCmdPushConstants(cb, g_cap.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(m.bases), m.bases);
                vkCmdDrawIndexed(cb, m.count, 1, m.first, 0, 0);
            }
        }
    }
    vkCmdEndRendering(cb);
    ib[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    ib[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    ib[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ib[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, ib);
    // The next frame.
    if (g_cap.ok) {
        g_cap.last.swap(g_cap.now);
        g_cap.now.clear();
        g_cap.last_slot = g_cap.slot;
        g_cap.last_ok = true;
        g_cap.slot = (g_cap.slot + 1) % kCapSlots;
        g_cap.used = 0;
        g_cap.keys.clear();
        // Entries replaced two frames ago are named by no list now.
        std::erase_if(g_cap.retired, [](const auto& r) { return r.first + 2 <= g_cap.frames; });
        // The index buffer nearly full (areas' meshes since the start): every
        // entry retired - the lists still name them for two frames - and the
        // meshes read and copied again as they are drawn.
        if (g_cap.ibuf_used > kCapIndexCap - kCapIndexCap / 8) {
            for (auto& [k, e] : g_cap.indices) {
                if (e) g_cap.retired.emplace_back(g_cap.frames, std::move(e));
            }
            g_cap.indices.clear();
            g_cap.ibuf_used = 0;
            host_log("upscaler: the characters' capture's index buffer emptied (%u indices kept) - meshes copied again as they are drawn",
                     kCapIndexCap);
        }
    }
    g_cap.vp_set = false;
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

// TAA (Graphics > Upscaling: TAA, host/upscaler.h kUpscaleTaa): Numity1504's
// TAA (his handoff's step 9, with step 8's ember mask in its TAA form) in the
// bridge's place, from the same inputs - the jittered scene, the motion, the
// depth - plus the glow (shaders/taa_resolve.comp). The motion it works out
// itself, from the motion pass's constants, velocity map and capture: the
// motion pass is not run, and its image and the depth's R32 copy are written
// only when something reads them (frame generation, the motion tests). Two histories (RGBA16F,
// colour as light, alpha the motion's length there), one read and one
// written, changing places each frame; the picture is g_output, alpha the
// scene's. No mip bias and no sharpening of its own (Sharpness is the
// player's), and the separate ember pass is not run after it: its mask is
// inside, kept out of the history.
VkDescriptorSetLayout g_taa_set_layout = VK_NULL_HANDLE;
VkPipelineLayout g_taa_layout = VK_NULL_HANDLE;
VkPipeline g_taa_pipeline = VK_NULL_HANDLE;
VkSampler g_taa_sampler = VK_NULL_HANDLE;  // linear, clamped: the history's Catmull-Rom taps
constexpr std::uint32_t kTaaBindings = 11, kTaaPushBytes = 224;

// taa_resolve.comp's bindings: 4 the sampler, 5 6 9 10 written, the rest read.
VkDescriptorType taa_binding_type(std::uint32_t k) {
    return k == 4 ? VK_DESCRIPTOR_TYPE_SAMPLER
         : k == 5 || k == 6 || k == 9 || k == 10 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                 : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
}

bool taa_pipeline_locked() {
    static bool tried = false;
    if (tried) return g_taa_pipeline != VK_NULL_HANDLE;
    tried = true;
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    if (vkCreateSampler(g.device, &si, nullptr, &g_taa_sampler) != VK_SUCCESS) return false;
    // The motion pass's constants and TAA's: more than the 128 bytes every
    // device has (256 on AMD's, NVIDIA's and Intel's).
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g.phys, &props);
    if (props.limits.maxPushConstantsSize < kTaaPushBytes) {
        host_log("upscaler: TAA needs %u bytes of push constants, the device has %u; the game's own anti-aliasing", kTaaPushBytes,
                 props.limits.maxPushConstantsSize);
        return false;
    }
    VkDescriptorSetLayoutBinding binds[kTaaBindings] = {};
    for (std::uint32_t k = 0; k < kTaaBindings; ++k) binds[k] = {k, taa_binding_type(k), 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = kTaaBindings;
    sli.pBindings = binds;
    if (vkCreateDescriptorSetLayout(g.device, &sli, nullptr, &g_taa_set_layout) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, kTaaPushBytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_taa_set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(g.device, &pli, nullptr, &g_taa_layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(k_taa_resolve_spv);
    smi.pCode = k_taa_resolve_spv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.device, &smi, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = g_taa_layout;
    const bool ok = vkCreateComputePipelines(g.device, g.cache, 1, &ci, nullptr, &g_taa_pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(g.device, module, nullptr);
    if (!ok) g_taa_pipeline = VK_NULL_HANDLE;
    host_log("upscaler: TAA's pass %s", ok ? "ready" : "failed to build; the game's own anti-aliasing");
    return ok;
}

// This frame's TAA into g_output and the next frame's history. The scene is
// readable; `depth` the game's; `motion` the motion pass's constants
// (motion_params), `velo` its velocity map (null: none) and the capture map
// g_cap_map_view. `glow` (null: none this frame, no ember mask) the scene's
// size. `write_motion`: g_motion and g_depth_r32 written too (frame
// generation, the motion tests read them).
bool record_taa(VkCommandBuffer cb, const RtImage& scene, VkImageView depth, VkImageLayout depth_layout, float jitter_x, float jitter_y,
                bool reset, const RtImage* glow, const float (&motion)[40], VkImageView velo, bool write_motion) {
    if (!taa_pipeline_locked()) return false;
    for (OwnImage& h : g_taa_hist) {
        if (!h.image && !make(h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, g_w, g_h)) {
            drop(g_taa_hist[0]);
            drop(g_taa_hist[1]);
            return false;
        }
    }
    const VkDescriptorSet set = alloc_set_locked(g_taa_set_layout);
    if (!set) return false;
    OwnImage &read = g_taa_hist[g_taa_read], &write = g_taa_hist[g_taa_read ^ 1];
    const bool no_history = reset || !read.initialised;
    if (glow) {
        barrier(cb, glow->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    // With no history (or no glow) the image is not read, but bound all the
    // same: the scene stands in for it.
    // With no velocity map or capture the scene stands in, unread.
    const VkDescriptorImageInfo infos[kTaaBindings] = {
        {VK_NULL_HANDLE, scene.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, no_history ? scene.view : read.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, velo ? velo : scene.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, depth, depth_layout},
        {g_taa_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED},
        {VK_NULL_HANDLE, g_output.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, write.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, glow ? glow->view : scene.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, g_cap_map_view ? g_cap_map_view : scene.view,
         g_cap_map_view ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, g_motion.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, g_depth_r32.view, VK_IMAGE_LAYOUT_GENERAL},
    };
    VkWriteDescriptorSet wr[kTaaBindings] = {};
    for (std::uint32_t k = 0; k < kTaaBindings; ++k) {
        wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[k].dstSet = set;
        wr[k].dstBinding = k;
        wr[k].descriptorCount = 1;
        wr[k].descriptorType = taa_binding_type(k);
        wr[k].pImageInfo = &infos[k];
    }
    vkUpdateDescriptorSets(g.device, kTaaBindings, wr, 0, nullptr);
    struct {
        float motion[40];
        std::uint32_t size[4];
        float weights[12];
    } pc{{}, {g_w, g_h, no_history ? 1u : 0u, (glow ? 1u : 0u) | (write_motion ? 2u : 0u)}, {}};
    static_assert(sizeof(pc) == kTaaPushBytes);
    std::memcpy(pc.motion, motion, sizeof(pc.motion));
    // The 3x3's weights: each sample lies at its centre less the jitter (the
    // picture moved by +j); exp(-2.29 d^2) of its distance from the pixel's
    // centre, normalised - the same for every pixel, so worked out here.
    float sum = 0.0f;
    for (int k = 0; k < 9; ++k) {
        const float dx = static_cast<float>(k % 3 - 1) - jitter_x, dy = static_cast<float>(k / 3 - 1) - jitter_y;
        pc.weights[k] = std::exp(-2.29f * (dx * dx + dy * dy));
        sum += pc.weights[k];
    }
    for (int k = 0; k < 9; ++k) pc.weights[k] /= std::max(sum, 1e-6f);
    barrier(cb, g_output.image, VK_IMAGE_ASPECT_COLOR_BIT, g_output.initialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    barrier(cb, write.image, VK_IMAGE_ASPECT_COLOR_BIT, write.initialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    if (write_motion) {
        barrier(cb, g_motion.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_SHADER_WRITE_BIT);
        barrier(cb, g_depth_r32.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_SHADER_WRITE_BIT);  // (every texel written again)
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_taa_pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_taa_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, g_taa_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cb, (g_w + 7) / 8, (g_h + 7) / 8, 1);
    barrier(cb, write.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT);
    if (write_motion) {
        barrier(cb, g_motion.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT);
        barrier(cb, g_depth_r32.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT);
        g_depth_r32_fresh = true;
    }
    write.initialised = true;
    g_taa_read ^= 1;
    static bool told = false;
    if (!told)
        host_log("upscaler: TAA at %ux%u (the scene's size, Numity1504's step 9), the ember mask %s", g_w, g_h,
                 glow ? "from the glow" : "off until a glow is found");
    told = true;
    return true;
}

// The embers: `picture` (the upscaled picture, readable) with the scene's
// bright one-frame dots put back where `glow` shows them, into g_embers.
VkDescriptorSetLayout g_ember_set_layout = VK_NULL_HANDLE;
VkPipelineLayout g_ember_layout = VK_NULL_HANDLE;
VkPipeline g_ember_pipeline = VK_NULL_HANDLE;

bool ember_pipeline_locked() {
    static bool tried = false;
    if (tried) return g_ember_pipeline != VK_NULL_HANDLE;
    tried = true;
    const VkDescriptorSetLayoutBinding binds[5] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 5;
    sli.pBindings = binds;
    if (vkCreateDescriptorSetLayout(g.device, &sli, nullptr, &g_ember_set_layout) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_ember_set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(g.device, &pli, nullptr, &g_ember_layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(k_upscale_embers_spv);
    smi.pCode = k_upscale_embers_spv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.device, &smi, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = g_ember_layout;
    const bool ok = vkCreateComputePipelines(g.device, g.cache, 1, &ci, nullptr, &g_ember_pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(g.device, module, nullptr);
    if (!ok) g_ember_pipeline = VK_NULL_HANDLE;
    host_log("upscaler: the embers' pass %s", ok ? "ready" : "failed to build; the embers stay as the upscaler leaves them");
    return ok;
}

bool embers_on() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_EMBERS");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool record_embers(VkCommandBuffer cb, const OwnImage& picture, const RtImage& scene, const RtImage& glow) {
    if (!embers_on() || !ember_pipeline_locked() || !scene.view || !glow.view) return false;
    if (!g_embers.image &&
        !make(g_embers, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
              g_ow, g_oh)) {
        drop(g_embers);
        return false;
    }
    const VkDescriptorSet set = alloc_set_locked(g_ember_set_layout);
    if (!set) return false;
    const VkDescriptorImageInfo infos[5] = {
        {VK_NULL_HANDLE, picture.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, scene.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, glow.view, VK_IMAGE_LAYOUT_GENERAL},
        {g.dummy_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED},
        {VK_NULL_HANDLE, g_embers.view, VK_IMAGE_LAYOUT_GENERAL},
    };
    const VkDescriptorType types[5] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                       VK_DESCRIPTOR_TYPE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    VkWriteDescriptorSet wr[5] = {};
    for (int k = 0; k < 5; ++k) {
        wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[k].dstSet = set;
        wr[k].dstBinding = static_cast<std::uint32_t>(k);
        wr[k].descriptorCount = 1;
        wr[k].descriptorType = types[k];
        wr[k].pImageInfo = &infos[k];
    }
    vkUpdateDescriptorSets(g.device, 5, wr, 0, nullptr);
    const std::uint32_t pc[4] = {g_ow, g_oh, g_w, g_h};
    // The scene and the glow were drawn into: their writes before the reads.
    barrier(cb, scene.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    barrier(cb, glow.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    barrier(cb, g_embers.image, VK_IMAGE_ASPECT_COLOR_BIT, g_embers.initialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    g_embers.initialised = true;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ember_pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ember_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, g_ember_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cb, (g_ow + 15) / 16, (g_oh + 15) / 16, 1);
    barrier(cb, g_embers.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    static std::atomic<bool> told{false};
    if (!told.exchange(true))
        host_log("upscaler: the embers put back from the scene where the glow 0x%llx (%ux%u, format %d) shows them",
                 static_cast<unsigned long long>(glow.base), glow.width, glow.height, static_cast<int>(glow.format));
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

// BBHOST_UPSCALE_CHECK_BOX=<pixels> (with the check): it compares averages over
// the render pixels that many around instead of single pixels (0: single).
float check_box() {
    static const float r = [] {
        const char* e = std::getenv("BBHOST_UPSCALE_CHECK_BOX");
        return e ? std::clamp(static_cast<float>(std::atof(e)), 0.0f, 8.0f) : 0.0f;
    }();
    return r;
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
// render size and `k`, `k2`, `k3`.
void record_test_pass(VkCommandBuffer cb, TestPass& t, std::initializer_list<VkImageView> views, float k, float k2 = 0.0f,
                      float k3 = 0.0f, float k4 = 0.0f) {
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
    const float pc[8] = {static_cast<float>(g_ow), static_cast<float>(g_oh), static_cast<float>(g_w), static_cast<float>(g_h), k, k2, k3, k4};
    barrier(cb, g_output.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, t.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, t.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, t.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cb, (g_ow + 7) / 8, (g_oh + 7) / 8, 1);
}

// Whether the frame the check shows had the velocity pass's last-frame
// matrices (its top-left corner says: white with, grey without).
bool g_check_whole = false;

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
    if (g_prev_scene_valid)
        record_test_pass(cb, g_check_pass, {scene.view, g_prev_scene.view, g_motion.view}, gain, g_check_whole ? 2.0f : 1.0f, check_box(),
                         std::getenv("BBHOST_UPSCALE_CHECK_OVERLAY") ? 1.0f : 0.0f);
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
// What the upscalers judge history by, from the camera: the near and far
// planes the depth stands for (the engine's decode, view distance =
// 1 / (z raw + w): raw 0 and raw 1's distances, whichever is nearer the near
// plane - and with it the depth's direction), the vertical field of view, and
// the time since the last frame. They were left at stand-ins (0.1 m, 10 km,
// 57 degrees, 16.7 ms), which FSR takes without complaint and judges
// disocclusion by wrongly - history kept behind a moving character, a ghost.
struct CameraTerms {
    float near_m = 0.1f, far_m = 10000.0f, fov_y = 1.0f, frame_ms = 16.7f;
    bool inverted = false;
};
CameraTerms camera_terms() {
    static std::chrono::steady_clock::time_point last;
    const auto now = std::chrono::steady_clock::now();
    CameraTerms t;
    if (last.time_since_epoch().count())
        t.frame_ms = std::clamp(std::chrono::duration<float, std::milli>(now - last).count(), 1.0f, 100.0f);
    last = now;
    if (g_cam_now.tan_y > 0.0f) t.fov_y = 2.0f * std::atan(g_cam_now.tan_y);
    // raw r: 1 / (z r + w), z = 1 / r2w, w = -a / r2w.
    const float z = g_cam_now.r2w != 0.0f ? 1.0f / g_cam_now.r2w : 0.0f, w = -g_cam_now.a * z;
    const auto dist = [&](float raw) { const float q = z * raw + w; return q > 1e-7f ? 1.0f / q : 1e6f; };
    const float d0 = dist(0.0f), d1 = dist(1.0f);
    if (z != 0.0f && std::isfinite(d0) && std::isfinite(d1) && d0 > 0.0f && d1 > 0.0f) {
        t.inverted = d1 < d0;
        t.near_m = std::max(std::min(d0, d1), 0.001f);
        t.far_m = std::min(std::max(d0, d1), 1e6f);
    }
    static bool told = false;
    if (!told && g_cam_now.tan_y > 0.0f) {
        told = true;
        host_log("upscaler: camera for the upscalers: near %g m, far %g m, depth %s, vertical field of view %.1f degrees",
                 static_cast<double>(t.near_m), static_cast<double>(t.far_m), t.inverted ? "inverted (near = 1)" : "near = 0",
                 static_cast<double>(t.fov_y * 57.29578f));
    }
    return t;
}
CameraTerms g_terms;  // this frame's (upscale_pass_run_locked)

void prepare_frame_generation(VkCommandBuffer cb, const RtImage& depth, VkImageLayout depth_layout, float jitter_x, float jitter_y,
                              float motion_scale) {
    FramegenInputs fi;
    fi.depth = g_depth_r32_fresh ? g_depth_r32.image : depth.image;
    fi.depth_format = g_depth_r32_fresh ? VK_FORMAT_R32_SFLOAT : depth.format;
    fi.depth_layout = g_depth_r32_fresh ? VK_IMAGE_LAYOUT_GENERAL : depth_layout;
    fi.motion = g_motion.image;
    fi.render_w = g_w;
    fi.render_h = g_h;
    fi.jitter_x = jitter_x;
    fi.jitter_y = jitter_y;
    fi.motion_scale = motion_scale;
    fi.fov_y = g_terms.fov_y;
    fi.near_m = g_terms.near_m;
    fi.far_m = g_terms.far_m;
    fi.frame_ms = g_terms.frame_ms;
    g_fg_pending = framegen_prepare(cb, g_fg_frames + 1, fi) ? ++g_fg_frames : 0;
    static bool told = false;
    if (!told) host_log("framegen: first prepare %s", g_fg_pending ? "recorded" : "refused");
    told = true;
}

// BBHOST_VELO_SEEN: 0 a mesh's motion wherever its pass drew (a test), alt
// the visible-surface test every other frame (the check's corner marks it);
// else on (camera_motion.comp mode 5).
int velo_seen_env() {
    static const int on = [] {
        const char* e = std::getenv("BBHOST_VELO_SEEN");
        return !e ? 1 : e[0] == '0' ? 0 : std::strcmp(e, "alt") == 0 ? 2 : 1;
    }();
    return on;
}
bool velo_seen_test() { return velo_seen_env() == 1 || (velo_seen_env() == 2 && (g_velo_frame & 1) == 0); }

// BBHOST_VELO_PREV: 0 the game's matrices always (a test), alt last frame's
// given every other frame (velo_previous_given), else 1.
int velo_previous_env() {
    static const int on = [] {
        const char* e = std::getenv("BBHOST_VELO_PREV");
        return !e ? 1 : e[0] == '0' ? 0 : std::strcmp(e, "alt") == 0 ? 2 : 1;
    }();
    return on;
}

}  // namespace

bool upscale_motion_on() { return upscaler_method() || framegen_active(); }

std::uint64_t upscale_velo_frame() { return g_velo_frame; }

bool velo_previous_due() {
    // Rebuilt motion vectors, or the game's map while its blur, which would
    // read whole motions, is off (Motion blur, Performance mode).
    const HostSettings& hs = host_settings_ref();
    return velo_previous_env() != 0 && upscale_motion_on() &&
           (host_opt_motion_rebuilt_bits() || !hs.motion_blur || hs.performance_mode);
}

bool velo_previous_given() {
    // BBHOST_VELO_PREV=alt: kept every frame and given every other one, an
    // A/B within one run (the reprojection check marks which).
    return velo_previous_due() && !(velo_previous_env() == 2 && (g_velo_frame & 1));
}

bool velo_cloth_given() {
    return velo_previous_env() != 0 && upscale_motion_on() && !(velo_previous_env() == 2 && (g_velo_frame & 1));
}

bool upscale_velo_clear_due(std::uint64_t base, int kind) {
    if (kind != 2 || !base) return false;
    std::lock_guard<std::mutex> lk(g_note_mu);
    return g_velo_cleared != g_velo_frame && !g_velo_camera_pass;
}

void upscale_velo_clear_locked(RtImage& velo) {
    {
        std::lock_guard<std::mutex> lk(g_note_mu);
        g_velo_cleared = g_velo_frame;
    }
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

bool upscale_pass_run_locked(RtImage& scene, RtImage& depth, float jitter_x, float jitter_y, const RtImage* glow) {
    const int method = upscaler_method();
    // Frame generation with no upscaler running (Upscaling off, or one this
    // GPU cannot run): the motion and its prepare at the scene's size, and the
    // scene left as the game drew it.
    const bool motion_only = !method && framegen_active();
    if (!method && !motion_only) return false;
    // The output size: render scale's when the scene is its render size,
    // else the scene's own (DLAA, FSR native AA).
    RenderScale rs;
    const bool scaled = upscaler_method_scales(method) && render_scale_on(&rs) && scene.width == rs.render_w && scene.height == rs.render_h &&
                        (rs.output_w != rs.render_w || rs.output_h != rs.render_h);
    // A smaller 3D with no quality is a plugin's look the upscaler has just
    // taken over (host/scene_look.h): its frames until the rebuild to the
    // quality's size are left as drawn - the bridge refuses them.
    if (scaled && !rs.quality) {
        g_skipped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
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
        drop(g_depth_r32);
        g_depth_r32_fresh = false;
        drop(g_sharp);
        drop(g_embers);
        drop(g_taa_hist[0]);
        drop(g_taa_hist[1]);
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
                  g_w, g_h) ||
            !make(g_depth_r32, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, g_w, g_h)) {
            host_log("upscaler: no images of %ux%u and %ux%u for it; not upscaled", g_ow, g_oh, g_w, g_h);
            drop(g_output);
            drop(g_motion);
            drop(g_depth_r32);
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
                         method == 1 ? "DLAA" : method == kUpscaleTaa ? "TAA" : "FSR native AA", g_w, g_h, static_cast<int>(scene.format), static_cast<int>(depth.format));
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

    if (method != g_last_method) {
        g_last_method = method;
        g_reset = true;
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
    // Whether this frame's velocity pass had last frame's matrices (as it was
    // asked, before the frame count moves on).
    const bool whole = velo_previous_given();
    // The next frame's map work (upscale_velo_clear_due, upscale_velo_copy_due).
    ++g_velo_frame;
    g_velo_camera_pass = false;
    g_depth_r32_fresh = false;  // (record_motion writes this frame's, or TAA)
    // TAA works the motion out in its own pass (record_taa): the motion pass's
    // constants are kept for it instead, here, before the cameras move on.
    const bool taa = method == kUpscaleTaa && taa_pipeline_locked();
    float taa_motion[40];
    motion_params(taa_motion, g_w, g_h, false, VK_NULL_HANDLE, 0, 0, 0, false);
    VkImageView taa_velo = VK_NULL_HANDLE;
    if (g_cam_noted && motion_mode != 0 && motion_pipeline_locked()) {
        if (capture_env()) capture_ready_locked();
        profile_begin_locked(&g_prof_motion);
        record_capture(cb, g_w, g_h);
        g_cap_map_view = g_cap.map.view;
        if (taa) {
            motion_params(taa_motion, g_w, g_h, g_cam_prev_ok, velo, velo_w, velo_h, velo_mode, whole);
            taa_velo = velo;
        } else {
            record_motion(cb, g_depth_view, depth_layout, g_motion.image, g_motion.view, g_w, g_h, g_cam_prev_ok, velo, velo_w, velo_h,
                          velo_mode, whole);
        }
        profile_end_locked();
        g_check_whole = velo_seen_env() == 2 ? velo_seen_test() : whole;
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
    g_terms = camera_terms();
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
    c.depth = g_depth_r32_fresh ? UpscaleImage{g_depth_r32.image, g_depth_r32.view, VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_GENERAL}
                                : UpscaleImage{depth.image, g_depth_view, depth.format, depth_layout};
    c.motion = {g_motion.image, g_motion.view, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL};
    c.motion_scale = motion_mode < 0 ? -1.0f : 1.0f;
    c.dlss_preset = static_cast<std::uint32_t>(host_settings_ref().dlss_preset);
    c.jitter_x = jitter_x;
    c.jitter_y = jitter_y;
    c.near_m = g_terms.near_m;
    c.far_m = g_terms.far_m;
    c.fov_y = g_terms.fov_y;
    c.frame_ms = g_terms.frame_ms;
    c.depth_inverted = g_terms.inverted;
    c.output = {g_output.image, g_output.view, VK_FORMAT_R16G16B16A16_SFLOAT,
                g_output.initialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED};
    profile_begin_locked(method == kUpscaleTaa ? &g_prof_taa : &g_prof_bridge);
    const bool ok = method == kUpscaleTaa
                        ? taa && record_taa(cb, scene, g_depth_view, depth_layout, jitter_x, jitter_y, c.reset,
                                            glow && glow->width == scene.width && glow->height == scene.height ? glow : nullptr, taa_motion,
                                            taa_velo, framegen_active() || show_motion() > 0.0f || check_motion() > 0.0f)
                        : upscaler_evaluate(cb, c);
    profile_end_locked();
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
    profile_begin_locked(&g_prof_sharpen);
    const OwnImage* picture_p = sharpness > 0.0f && record_rcas(cb, sharpness) ? &g_sharp : &g_output;
    profile_end_locked();
    // The embers put back over it (record_embers), not in a test's view.
    profile_begin_locked(&g_prof_embers);
    // (TAA has its own, in its pass: Numity1504's TAA form, kept out of its history.)
    if (glow && method != kUpscaleTaa && !test_view() && glow->width == scene.width && glow->height == scene.height &&
        record_embers(cb, *picture_p, scene, *glow))
        picture_p = &g_embers;
    profile_end_locked();
    const OwnImage& picture = *picture_p;
    g_picture_view = picture.view;
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
    return g_picture_view ? g_picture_view : g_output.view;
}

bool upscale_camera_noted() { return g_cam_noted; }

void upscale_velo_note_locked(int kind, std::uint64_t base, float kx, float ky) {
    if (kind != 3 && (!std::isfinite(kx) || !std::isfinite(ky) || kx == 0.0f || ky == 0.0f)) return;
    std::lock_guard<std::mutex> lk(g_note_mu);
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
    std::lock_guard<std::mutex> lk(g_note_mu);
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

bool upscale_camera_block_locked(std::uint64_t base, std::uint64_t bytes) {
    if (!base || bytes < 864) return false;
    const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(base));
    if (f[47] != 1.0f || f[195] != 1.0f) return false;  // (cheaply first: most buffers are not one)
    if (!g_cam_noted) return upscale_camera_note_locked(base, bytes);
    Camera c;
    return camera_of(f, &c);
}

bool upscale_view_terms(float* tan_x, float* tan_y, float* depth_z, float* depth_w) {
    *tan_x = g_view_tan_x.load(std::memory_order_relaxed);
    *tan_y = g_view_tan_y.load(std::memory_order_relaxed);
    *depth_z = g_view_z.load(std::memory_order_relaxed);
    *depth_w = g_view_w.load(std::memory_order_relaxed);
    return *tan_x > 0.0f && *tan_y > 0.0f && *depth_z != 0.0f;
}

std::uint64_t upscale_pass_calls() { return g_calls.load(std::memory_order_relaxed); }

// A skinned G-buffer draw's place in the characters' capture (g_cap above):
// the device address its vertex shader stores clip positions at, less its
// first vertex's (an index plus the base vertex is the vertex shader's vertex
// index), or 0 for none. The indices, read once for each buffer and range,
// give the vertices it uses and its triangles.
std::uint64_t upscale_capture_draw_locked(std::uint64_t key, std::uint64_t order, std::uint64_t index_va, const void* indices, std::uint32_t count,
                                          bool i32, std::int32_t base_vertex) {
    if (!capture_env() || !g_cap.ok || !indices || count < 3 || count > (1u << 22)) return 0;
    const std::uint64_t ikey = fnv1a(&count, sizeof(count), index_va ^ (i32 ? 0x32ull : 0x16ull));
    const auto index_at = [&](std::uint32_t k) {
        return i32 ? static_cast<const std::uint32_t*>(indices)[k] : static_cast<std::uint32_t>(static_cast<const std::uint16_t*>(indices)[k]);
    };
    std::lock_guard<std::mutex> lk(g_cap_mu);
    std::unique_ptr<CapIndices>& slot = g_cap.indices[ikey];
    // Read once for its buffer and count - and again when eight of them no
    // longer read as kept (another mesh's indices streamed into the same
    // place): its vertex shader would store past the corners given it.
    if (slot && slot->verts) {
        const std::uint32_t used = static_cast<std::uint32_t>(slot->local.size());
        bool same = used == count - count % 3;
        for (std::uint32_t s = 0; same && s < 8; ++s) {
            const std::uint32_t k = static_cast<std::uint32_t>(static_cast<std::uint64_t>(used - 1) * s / 7);
            same = index_at(k) - slot->vmin == slot->local[k];
        }
        if (!same) {
            g_cap.retired.emplace_back(g_cap.frames, std::move(slot));
            ++g_cap.renewed;
        }
    }
    if (!slot) {
        slot = std::make_unique<CapIndices>();
        std::uint32_t lo = UINT32_MAX, hi = 0;
        std::vector<std::uint32_t> v(count);
        for (std::uint32_t k = 0; k < count; ++k) {
            v[k] = index_at(k);
            lo = std::min(lo, v[k]);
            hi = std::max(hi, v[k]);
        }
        if (hi - lo < kCapVertices) {
            for (std::uint32_t& x : v) x -= lo;
            v.resize(count - count % 3);
            slot->vmin = lo;
            slot->verts = hi - lo + 1;
            slot->local = std::move(v);
            // Its copy for the motion pass, once (the buffer full: the meshes
            // made from here on go uncaptured until it is emptied, below).
            const auto k = static_cast<std::uint32_t>(slot->local.size());
            if (g_cap.ibuf_used + k <= kCapIndexCap) {
                std::memcpy(g_cap.ibuf_map + std::size_t{g_cap.ibuf_used} * 4, slot->local.data(), std::size_t{k} * 4);
                slot->first = g_cap.ibuf_used;
                g_cap.ibuf_used += k;
            }
        }
    }
    const CapIndices* idx = slot.get();
    if (!idx->verts || g_cap.used + idx->verts > kCapVertices) return 0;
    CapDraw c;
    c.key = key;  // its n-th of the frame added in the stream's order (record_capture)
    c.order = order;
    c.offset = g_cap.used;
    c.idx = idx;
    g_cap.now.push_back(c);
    g_cap.used += idx->verts;
    const std::uint64_t first = static_cast<std::uint64_t>(static_cast<std::int64_t>(base_vertex) + idx->vmin);
    return g_cap.addr + (static_cast<std::uint64_t>(g_cap.slot) * kCapVertices + c.offset) * 16 - first * 16;
}

// The G-buffer's viewport as a camera draw had it, the jitter included.
void upscale_capture_viewport_locked(const VkViewport& vp) {
    std::lock_guard<std::mutex> lk(g_cap_mu);
    g_cap.vp = vp;
    g_cap.vp.minDepth = 0.0f;
    g_cap.vp.maxDepth = 1.0f;
    g_cap.vp_set = true;
}

bool upscale_capture_on() { return capture_env() && g_cap.ok; }
void upscale_capture_part_draw() { g_cap.part_draws.fetch_add(1, std::memory_order_relaxed); }

// The meshes the game moves: each index buffer its velocity pass ever drew a
// mesh from (render.cpp), remembered - the G-buffer draws of the same buffer
// are captured every frame, the ones the velocity pass left out included.
std::unordered_set<std::uint64_t> g_cap_moving;  // under g_note_mu
void upscale_capture_note_moving(std::uint64_t index_va) {
    std::lock_guard<std::mutex> lk(g_note_mu);
    g_cap_moving.insert(index_va);
}
bool upscale_capture_moving(std::uint64_t index_va) {
    std::lock_guard<std::mutex> lk(g_note_mu);
    return g_cap_moving.count(index_va) != 0;
}

}  // namespace gpu
