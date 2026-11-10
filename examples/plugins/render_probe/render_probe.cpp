// The render probe: plugin API 27's frame info, render hooks and
// anti-aliasing claim checked on the GPU. An example and a test aid that
// never ships: it is built beside the tests (build/render_probe.so or .dll -
// not build/plugins/, which every run loads), it is opt-in, and no package
// carries it. Copied into a test kit's plugins folder and turned on
// ([plugins] render_probe = true) it claims the anti-aliasing as its `claim`
// option says and registers both hooks (with the motion, as its `motion`
// option says). Each frame:
//   - before the upscale, a 16x16 marker of the frame number's low 16 bits
//     (magenta ones, black zeros) into the scene's top left - its own image,
//     blitted in, the scene being no storage image - and on a 4x4 grid of the
//     3D each point's depth taken back to the world with this frame's
//     inv_view_proj and forward with the last frame's view_proj, against the
//     host's motion there: the matrices, the depth and the motion agreeing;
//   - after the upscale the marker in cyan into the picture's top right,
//     written in place when the picture is a storage image and blitted in
//     when it is not, and that corner copied back, to check that what a hook
//     writes is what the post-processing reads;
//   - on the CPU, the images' sizes against the frame info's, their layouts,
//     write flags and usage, the jitter's range, prev the last frame's now, the
//     frame numbers, the matrices' inverses and the depth range, the claim's
//     effect, each generation change, and from the main thread frame_info and
//     the follow camera's focus point near the screen's middle.
// Every log_every frames a line says what it read back (frames_in_flight
// frames after it was recorded); a check that fails is logged the first
// times it does. Only Vulkan's headers, no loader: every function comes from
// the device the hooks are handed.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "bbhost_plugin.h"
#include "render_probe.spv.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

const BbHostApi* g_api = nullptr;
int g_claim = BB_AA_NONE;
bool g_want_motion = true, g_marker = true, g_on = false;
unsigned g_log_every = 300;

// --- the checks ---------------------------------------------------------------

std::mutex g_check_mu;
std::map<std::string, unsigned> g_failed;  // by check, under g_check_mu
std::uint64_t g_passed = 0;                // under g_check_mu

void check(bool ok, const char* what, const char* fmt = nullptr, double a = 0, double b = 0) {
    std::lock_guard<std::mutex> lk(g_check_mu);
    if (ok) {
        ++g_passed;
        return;
    }
    unsigned& n = g_failed[what];
    if (++n <= 3) {
        char line[256];
        if (fmt) std::snprintf(line, sizeof(line), fmt, a, b);
        g_api->log("render_probe: FAILED %s%s%s", what, fmt ? ": " : "", fmt ? line : "");
    }
}

std::string check_summary() {
    std::lock_guard<std::mutex> lk(g_check_mu);
    std::string s = std::to_string(g_passed) + " ok";
    unsigned failed = 0;
    for (const auto& [name, n] : g_failed) failed += n;
    s += ", " + std::to_string(failed) + " failed";
    for (const auto& [name, n] : g_failed) s += " (" + name + " " + std::to_string(n) + ")";
    return s;
}

// --- matrices -----------------------------------------------------------------

void mul(const float (&a)[16], const float (&b)[16], double (&out)[16]) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) s += static_cast<double>(a[k * 4 + r]) * b[c * 4 + k];
            out[c * 4 + r] = s;
        }
}

// How far m is from the identity, the largest entry's difference.
double off_identity(const double (&m)[16]) {
    double worst = 0.0;
    for (int k = 0; k < 16; ++k) worst = std::max(worst, std::fabs(m[k] - (k % 5 == 0 ? 1.0 : 0.0)));
    return worst;
}

void apply(const float (&m)[16], const double (&v)[4], double (&out)[4]) {
    for (int r = 0; r < 4; ++r) out[r] = m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3];
}

// --- Vulkan --------------------------------------------------------------------

struct Vk {
    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCreateSampler CreateSampler;
    PFN_vkCreateImage CreateImage;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkCreateImageView CreateImageView;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdBlitImage CmdBlitImage;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties;
} vk;

constexpr VkDeviceSize kSlotBytes = 8192, kCornerOffset = 4096;  // a slot's data, then the picture's corner copied back

// The pass's buffer, as render_probe.comp's Slot block lays it out (std430).
struct SlotData {
    float inv_view_proj[16];
    float prev_view_proj[16];
    std::uint32_t size[4];   // render width, height, mode, frame
    std::uint32_t flags[4];  // motion there, glow there, depth reversed, the picture's width
    float point[16][4];
    float motion[16][4];
    float glow[16][4];
};
static_assert(sizeof(SlotData) == 928, "render_probe.comp's Slot");

// What the CPU keeps of each slot until it reads it back.
struct SlotNote {
    bool used = false, corner = false;
    std::uint64_t frame = 0;
    std::uint32_t format = 0;  // the picture's, its corner copied back
};

struct Probe {
    bool tried = false, ok = false;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    std::uint32_t ring = 0;  // frames_in_flight + 1 slots for each point
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkImage marker = VK_NULL_HANDLE;
    VkImageView marker_view = VK_NULL_HANDLE;
    bool marker_ready = false;  // in GENERAL
    VkBuffer buffer = VK_NULL_HANDLE;
    std::uint8_t* mapped = nullptr;
    std::vector<VkDescriptorSet> sets;  // [point][slot]
    std::vector<SlotNote> notes;        // [point][slot]
    std::uint64_t calls[2] = {};
    std::map<std::uint32_t, bool> blit_dst;  // a format's blit support, as asked
    std::uint64_t generation[5] = {};
} g_probe;

std::atomic<std::uint64_t> g_last_frame{0};  // the last frame a hook saw (on_frame compares)

template <typename F>
bool load(F& fn, const BbRenderDevice& d, const char* name, bool instance = false) {
    fn = reinterpret_cast<F>(instance ? d.get_instance_proc_addr(d.instance, name) : d.get_device_proc_addr(d.device, name));
    if (!fn) g_api->log("render_probe: no %s", name);
    return fn != nullptr;
}

std::uint32_t memory_type(std::uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vk.GetPhysicalDeviceMemoryProperties(g_probe.phys, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

// Its pipeline, image, buffer and sets, once, at the first call.
bool setup(const BbRenderDevice& d) {
    Probe& p = g_probe;
    if (p.tried) return p.ok;
    p.tried = true;
    p.device = static_cast<VkDevice>(d.device);
    p.phys = static_cast<VkPhysicalDevice>(d.physical_device);
    bool all = load(vk.CreateShaderModule, d, "vkCreateShaderModule") & load(vk.DestroyShaderModule, d, "vkDestroyShaderModule") &
               load(vk.CreateDescriptorSetLayout, d, "vkCreateDescriptorSetLayout") & load(vk.CreatePipelineLayout, d, "vkCreatePipelineLayout") &
               load(vk.CreateComputePipelines, d, "vkCreateComputePipelines") & load(vk.CreateDescriptorPool, d, "vkCreateDescriptorPool") &
               load(vk.AllocateDescriptorSets, d, "vkAllocateDescriptorSets") & load(vk.UpdateDescriptorSets, d, "vkUpdateDescriptorSets") &
               load(vk.CreateSampler, d, "vkCreateSampler") & load(vk.CreateImage, d, "vkCreateImage") &
               load(vk.GetImageMemoryRequirements, d, "vkGetImageMemoryRequirements") & load(vk.BindImageMemory, d, "vkBindImageMemory") &
               load(vk.CreateImageView, d, "vkCreateImageView") & load(vk.CreateBuffer, d, "vkCreateBuffer") &
               load(vk.GetBufferMemoryRequirements, d, "vkGetBufferMemoryRequirements") & load(vk.BindBufferMemory, d, "vkBindBufferMemory") &
               load(vk.AllocateMemory, d, "vkAllocateMemory") & load(vk.MapMemory, d, "vkMapMemory") &
               load(vk.CmdBindPipeline, d, "vkCmdBindPipeline") & load(vk.CmdBindDescriptorSets, d, "vkCmdBindDescriptorSets") &
               load(vk.CmdDispatch, d, "vkCmdDispatch") & load(vk.CmdPipelineBarrier, d, "vkCmdPipelineBarrier") &
               load(vk.CmdBlitImage, d, "vkCmdBlitImage") & load(vk.CmdCopyImageToBuffer, d, "vkCmdCopyImageToBuffer") &
               load(vk.GetPhysicalDeviceMemoryProperties, d, "vkGetPhysicalDeviceMemoryProperties", true) &
               load(vk.GetPhysicalDeviceFormatProperties, d, "vkGetPhysicalDeviceFormatProperties", true);
    if (!all) return false;
    p.ring = std::max<std::uint32_t>(d.frames_in_flight, 2) + 1;
    const VkDescriptorType types[8] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                       VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER,       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    VkDescriptorSetLayoutBinding binds[8] = {};
    for (std::uint32_t k = 0; k < 8; ++k) binds[k] = {k, types[k], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo sli{};
    sli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sli.bindingCount = 8;
    sli.pBindings = binds;
    if (vk.CreateDescriptorSetLayout(p.device, &sli, nullptr, &p.set_layout) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &p.set_layout;
    if (vk.CreatePipelineLayout(p.device, &pli, nullptr, &p.layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = sizeof(k_render_probe_spv);
    smi.pCode = k_render_probe_spv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vk.CreateShaderModule(p.device, &smi, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = module;
    cpi.stage.pName = "main";
    cpi.layout = p.layout;
    const VkResult pr = vk.CreateComputePipelines(p.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &p.pipeline);
    vk.DestroyShaderModule(p.device, module, nullptr);
    if (pr != VK_SUCCESS) return false;
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vk.CreateSampler(p.device, &si, nullptr, &p.sampler) != VK_SUCCESS) return false;
    // The marker: 16x16 RGBA16F, written by the pass and blitted from.
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ici.extent = {16, 16, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (vk.CreateImage(p.device, &ici, nullptr, &p.marker) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vk.GetImageMemoryRequirements(p.device, p.marker, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory image_memory = VK_NULL_HANDLE;
    if (mai.memoryTypeIndex == UINT32_MAX || vk.AllocateMemory(p.device, &mai, nullptr, &image_memory) != VK_SUCCESS ||
        vk.BindImageMemory(p.device, p.marker, image_memory, 0) != VK_SUCCESS)
        return false;
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = p.marker;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vk.CreateImageView(p.device, &vci, nullptr, &p.marker_view) != VK_SUCCESS) return false;
    // The ring: a slot for each point and frame in flight, host-visible.
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = kSlotBytes * 2 * p.ring;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vk.CreateBuffer(p.device, &bci, nullptr, &p.buffer) != VK_SUCCESS) return false;
    vk.GetBufferMemoryRequirements(p.device, p.buffer, &req);
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
    void* map = nullptr;
    if (mai.memoryTypeIndex == UINT32_MAX || vk.AllocateMemory(p.device, &mai, nullptr, &buffer_memory) != VK_SUCCESS ||
        vk.BindBufferMemory(p.device, p.buffer, buffer_memory, 0) != VK_SUCCESS ||
        vk.MapMemory(p.device, buffer_memory, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS)
        return false;
    p.mapped = static_cast<std::uint8_t*>(map);
    std::memset(p.mapped, 0, static_cast<std::size_t>(bci.size));
    // A set for each point and slot.
    const std::uint32_t n = 2 * p.ring;
    const VkDescriptorPoolSize sizes[4] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4 * n},
                                           {VK_DESCRIPTOR_TYPE_SAMPLER, n},
                                           {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * n},
                                           {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, n}};
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = n;
    dpi.poolSizeCount = 4;
    dpi.pPoolSizes = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vk.CreateDescriptorPool(p.device, &dpi, nullptr, &pool) != VK_SUCCESS) return false;
    p.sets.assign(n, VK_NULL_HANDLE);
    std::vector<VkDescriptorSetLayout> layouts(n, p.set_layout);
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = pool;
    dai.descriptorSetCount = n;
    dai.pSetLayouts = layouts.data();
    if (vk.AllocateDescriptorSets(p.device, &dai, p.sets.data()) != VK_SUCCESS) return false;
    p.notes.assign(n, SlotNote{});
    p.ok = true;
    g_api->log("render_probe: ready - a ring of %u slots for each point, %u frames in flight, Vulkan %u.%u", p.ring, d.frames_in_flight,
               VK_API_VERSION_MAJOR(d.api_version), VK_API_VERSION_MINOR(d.api_version));
    return true;
}

bool can_blit_into(std::uint32_t format) {
    auto it = g_probe.blit_dst.find(format);
    if (it != g_probe.blit_dst.end()) return it->second;
    VkFormatProperties fp{};
    vk.GetPhysicalDeviceFormatProperties(g_probe.phys, static_cast<VkFormat>(format), &fp);
    const bool ok = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
    g_probe.blit_dst[format] = ok;
    if (!ok) g_api->log("render_probe: format %u takes no blit: no marker into it", format);
    return ok;
}

template <typename H>
H handle(std::uint64_t h) {
    return reinterpret_cast<H>(static_cast<std::uintptr_t>(h));
}

void image_barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst,
                   VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    vk.CmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// --- the checks of a call -------------------------------------------------------

BbFrameCamera g_last_now{};
bool g_have_last = false;
std::uint64_t g_last_seen = 0;

void check_image(const BbRenderImage& im, const char* what, std::uint32_t w, std::uint32_t h, bool writable) {
    if (!im.image) return;
    check(im.width == w && im.height == h, what, "%g wide against the frame's %g", im.width, w);
    check(im.view != 0, "a view for every image");
    check(((im.flags & BB_IMAGE_WRITE) != 0) == writable, "the write flags as the point says");
    if (writable) check((im.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0, "a writable image takes copies");
}

void check_call(const BbRenderCall& c) {
    const BbFrameInfo& f = *c.frame;
    const bool after = c.point == BB_RENDER_AFTER_UPSCALE;
    check(c.size >= sizeof(BbRenderCall), "the call's size");
    check(f.size >= sizeof(BbFrameInfo), "the frame info's size");
    check_image(c.scene, "the scene's size", f.render_width, f.render_height, !after);
    check_image(c.depth, "the depth's size", f.render_width, f.render_height, false);
    check_image(c.motion, "the motion's size", f.render_width, f.render_height, false);
    check_image(c.glow, "the glow's size", f.render_width, f.render_height, false);
    if (after) check_image(c.picture, "the picture's size", f.output_width, f.output_height, true);
    check(after ? c.picture.image != 0 : c.picture.image == 0, "a picture after the upscale only");
    check(c.scene.image && c.scene.layout == VK_IMAGE_LAYOUT_GENERAL, "the scene in GENERAL");
    check(!c.depth.image || c.depth.layout == VK_IMAGE_LAYOUT_GENERAL || c.depth.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
          "the depth readable", "layout %g", c.depth.layout);
    check(!c.motion.image || c.motion.format == VK_FORMAT_R16G16_SFLOAT, "the motion R16G16_SFLOAT");
    check(!g_want_motion || c.motion.image != 0, "the motion when asked for");
    if (after) return;  // the rest once a frame
    check(f.output_width >= f.render_width && f.output_height >= f.render_height, "the output at least the render size");
    if (f.method == BB_UPSCALE_NONE) check(f.jitter_x == 0.0f && f.jitter_y == 0.0f, "no jitter without an upscaler", "%g %g", f.jitter_x, f.jitter_y);
    else check(std::fabs(f.jitter_x) <= 0.5f && std::fabs(f.jitter_y) <= 0.5f, "the jitter within half a pixel", "%g %g", f.jitter_x, f.jitter_y);
    if (g_claim == BB_AA_TAKE)
        check(f.method == BB_UPSCALE_NONE && (f.flags & BB_FRAME_AA_PLUGIN), "taken over: no upscaler", "method %g flags %g", f.method, f.flags);
    if (g_last_seen) check(f.frame == g_last_seen + 1 || f.reset, "the frames one after another", "%g after %g", static_cast<double>(f.frame),
                           static_cast<double>(g_last_seen));
    if ((f.flags & BB_FRAME_PREV) && g_have_last) check(std::memcmp(&f.prev, &g_last_now, sizeof(f.prev)) == 0, "prev the last frame's now");
    if (f.flags & BB_FRAME_CAMERA) {
        // view against the basis (the view-to-world matrix), view_proj against its inverse.
        const BbCamera& cam = f.now.camera;
        float to_world[16];
        std::memcpy(to_world + 0, cam.right, 16);
        std::memcpy(to_world + 4, cam.up, 16);
        std::memcpy(to_world + 8, cam.forward, 16);
        std::memcpy(to_world + 12, cam.position, 16);
        double m[16];
        mul(f.now.view, to_world, m);
        const double e1 = off_identity(m);
        check(e1 < 1e-3, "view the inverse of the camera's basis", "%g", e1);
        // view_proj and inv_view_proj: points ahead of the camera there and
        // back (float matrices of a world in hundreds of metres: a millimetre
        // a metre is the check, the depth's own precision far finer).
        double worst = 0.0;
        for (const double dist : {1.0, 10.0, 50.0}) {
            double w[4] = {0.0, 0.0, 0.0, 1.0};
            for (int k = 0; k < 3; ++k) w[k] = cam.position[k] + dist * cam.forward[k] + 0.3 * dist * cam.right[k] - 0.1 * dist * cam.up[k];
            double clip[4], back[4];
            apply(f.now.view_proj, w, clip);
            const double ndc[4] = {clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3], 1.0};
            apply(f.now.inv_view_proj, ndc, back);
            double e = 0.0;
            for (int k = 0; k < 3; ++k) e += (back[k] / back[3] - w[k]) * (back[k] / back[3] - w[k]);
            worst = std::max(worst, std::sqrt(e) / dist);
        }
        check(worst < 1e-3, "inv_view_proj takes view_proj's points back", "%g of their distance", worst);
        // The planes: depth 0 and 1 (the other way round reversed).
        const double nz[4] = {0.0, 0.0, cam.near_z, 1.0}, fz[4] = {0.0, 0.0, cam.far_z, 1.0};
        double cn[4], cf[4];
        apply(f.now.proj, nz, cn);
        apply(f.now.proj, fz, cf);
        const bool rev = (f.flags & BB_FRAME_DEPTH_REVERSED) != 0;
        check(std::fabs(cn[2] / cn[3] - (rev ? 1.0 : 0.0)) < 1e-3, "the near plane's depth", "%g", cn[2] / cn[3]);
        check(std::fabs(cf[2] / cf[3] - (rev ? 0.0 : 1.0)) < 1e-2, "the far plane's depth", "%g", cf[2] / cf[3]);
        check(cam.fov_y > 0.1f && cam.fov_y < 3.0f && cam.aspect > 0.2f && cam.aspect < 8.0f, "a field of view and shape", "%g %g", cam.fov_y,
              cam.aspect);
        g_last_now = f.now;
        g_have_last = true;
    } else {
        g_have_last = false;
    }
    g_last_seen = f.frame;
    // Each image's generation, said when it changes (a Resolution change).
    const BbRenderImage* ims[4] = {&c.scene, &c.depth, &c.motion, &c.glow};
    const char* names[4] = {"scene", "depth", "motion", "glow"};
    for (int k = 0; k < 4; ++k) {
        if (ims[k]->generation == g_probe.generation[k]) continue;
        g_probe.generation[k] = ims[k]->generation;
        g_api->log("render_probe: frame %llu: the %s is generation %llu (%ux%u format %u layout %u usage 0x%x)",
                   static_cast<unsigned long long>(f.frame), names[k], static_cast<unsigned long long>(ims[k]->generation), ims[k]->width,
                   ims[k]->height, ims[k]->format, ims[k]->layout, ims[k]->usage);
    }
}

// --- what a slot held, frames_in_flight frames on -----------------------------------

std::uint32_t half_of(float v) { return v == 1.0f ? 0x3c00u : 0u; }

void read_back(std::uint32_t point, std::uint32_t slot) {
    Probe& p = g_probe;
    SlotNote& note = p.notes[point * p.ring + slot];
    if (!note.used) return;
    note.used = false;
    const std::uint8_t* base = p.mapped + (point * p.ring + slot) * kSlotBytes;
    SlotData d;
    std::memcpy(&d, base, sizeof(d));
    if (point == 0) {
        // The grid: the cameras' motion against the host's, where both are there.
        std::vector<double> errs;
        int surfaces = 0;
        for (const auto& pt : d.point) {
            if (pt[0] > 0.5f) ++surfaces;
            if (pt[2] >= 0.0f) errs.push_back(pt[2]);
        }
        std::sort(errs.begin(), errs.end());
        const double median = errs.empty() ? 0.0 : errs[errs.size() / 2], worst = errs.empty() ? 0.0 : errs.back();
        if (!errs.empty()) check(median < 1.0, "the reprojection against the host's motion", "median %g px, worst %g", median, worst);
        if ((note.frame % g_log_every) == 0)
            g_api->log("render_probe: frame %llu read back: %d of 16 grid points on a surface; reprojection against the host's motion over %zu: "
                       "median %.3f px, worst %.3f px; depth at the middle %.6f",
                       static_cast<unsigned long long>(note.frame), surfaces, errs.size(), median, worst, static_cast<double>(d.point[5][1]));
        return;
    }
    if (!note.corner) return;
    // The picture's corner as it was copied back: the frame's marker in cyan.
    std::uint32_t texel = note.format == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : note.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ? 4 : 0;
    if (!texel) return;
    unsigned wrong = 0;
    std::uint64_t sum = 1469598103934665603ull;  // FNV-1a
    for (std::uint32_t y = 0; y < 16; ++y)
        for (std::uint32_t x = 0; x < 16; ++x) {
            const std::uint8_t* t = base + kCornerOffset + (y * 16 + x) * texel;
            for (std::uint32_t k = 0; k < texel; ++k) sum = (sum ^ t[k]) * 1099511628211ull;
            const bool one = (note.frame >> ((y / 4) * 4 + x / 4)) & 1u;
            if (texel == 8) {
                std::uint16_t h[4];
                std::memcpy(h, t, 8);
                const std::uint16_t want[4] = {0, static_cast<std::uint16_t>(one ? 0x3c00u : 0u), static_cast<std::uint16_t>(one ? 0x3c00u : 0u),
                                               0x3c00u};
                wrong += std::memcmp(h, want, 8) != 0;
            } else {
                std::uint32_t v;
                std::memcpy(&v, t, 4);
                // B10G11R11: 11 bits of red, 11 of green, 10 of blue; 1.0 is 0x3c0 in 11 bits, 0x1e0 in 10.
                const std::uint32_t want = one ? (0x3c0u << 11) | (0x1e0u << 22) : 0u;
                wrong += v != want;
            }
        }
    check(wrong == 0, "the picture's marker read back", "%g of 256 texels wrong (format %g)", wrong, note.format);
    if ((note.frame % g_log_every) == 0)
        g_api->log("render_probe: frame %llu read back: the picture's corner %s (%u of 256 texels wrong), checksum %016llx; checks so far %s",
                   static_cast<unsigned long long>(note.frame), wrong ? "WRONG" : "the marker", wrong, static_cast<unsigned long long>(sum),
                   check_summary().c_str());
}

// --- the hook ----------------------------------------------------------------------

void hook(const BbRenderCall* c, void*) {
    if (!setup(*c->device)) {
        static bool told = false;
        if (!told) g_api->log("render_probe: no pipeline: checks on the CPU alone");
        told = true;
    }
    const BbFrameInfo& f = *c->frame;
    const bool after = c->point == BB_RENDER_AFTER_UPSCALE;
    check_call(*c);
    g_last_frame.store(f.frame, std::memory_order_relaxed);
    Probe& p = g_probe;
    if (!p.ok) return;
    const std::uint32_t point = after ? 1u : 0u;
    const std::uint32_t slot = static_cast<std::uint32_t>(p.calls[point] % p.ring);
    ++p.calls[point];
    read_back(point, slot);  // what it held a ring ago: done on the GPU by now
    SlotNote& note = p.notes[point * p.ring + slot];
    note = SlotNote{};
    note.used = true;
    note.frame = f.frame;
    VkCommandBuffer cb = static_cast<VkCommandBuffer>(c->command_buffer);
    const VkDeviceSize offset = (point * p.ring + slot) * kSlotBytes;
    SlotData* d = reinterpret_cast<SlotData*>(p.mapped + offset);
    std::memcpy(d->inv_view_proj, f.now.inv_view_proj, sizeof(d->inv_view_proj));
    std::memcpy(d->prev_view_proj, f.prev.view_proj, sizeof(d->prev_view_proj));
    const BbRenderImage& pic = c->picture;
    const bool in_place = after && pic.image && (pic.usage & VK_IMAGE_USAGE_STORAGE_BIT) && pic.format == VK_FORMAT_R16G16B16A16_SFLOAT;
    d->size[0] = f.render_width;
    d->size[1] = f.render_height;
    d->size[2] = (after ? 1u : 0u) | (in_place ? 2u : 0u);
    d->size[3] = static_cast<std::uint32_t>(f.frame);
    d->flags[0] = c->motion.image ? 1u : 0u;
    d->flags[1] = c->glow.image ? 1u : 0u;
    d->flags[2] = (f.flags & BB_FRAME_DEPTH_REVERSED) ? 1u : 0u;
    d->flags[3] = after ? pic.width : 0u;
    // The set: the frame's images (stand-ins for those not there, unread).
    const VkImageView scene = handle<VkImageView>(c->scene.view);
    const VkImageView depth = c->depth.image ? handle<VkImageView>(c->depth.view) : scene;
    const VkImageLayout depth_layout = c->depth.image ? static_cast<VkImageLayout>(c->depth.layout) : VK_IMAGE_LAYOUT_GENERAL;
    const VkDescriptorImageInfo infos[8] = {
        {VK_NULL_HANDLE, scene, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, depth, depth_layout},
        {VK_NULL_HANDLE, c->motion.image ? handle<VkImageView>(c->motion.view) : scene, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, c->glow.image ? handle<VkImageView>(c->glow.view) : scene, VK_IMAGE_LAYOUT_GENERAL},
        {p.sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED},
        {VK_NULL_HANDLE, p.marker_view, VK_IMAGE_LAYOUT_GENERAL},
        {},
        {VK_NULL_HANDLE, in_place ? handle<VkImageView>(pic.view) : p.marker_view, VK_IMAGE_LAYOUT_GENERAL},
    };
    const VkDescriptorBufferInfo buffer_info{p.buffer, offset, sizeof(SlotData)};
    const VkDescriptorType types[8] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                       VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER,       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    VkWriteDescriptorSet wr[8] = {};
    const VkDescriptorSet set = p.sets[point * p.ring + slot];
    for (std::uint32_t k = 0; k < 8; ++k) {
        wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[k].dstSet = set;
        wr[k].dstBinding = k;
        wr[k].descriptorCount = 1;
        wr[k].descriptorType = types[k];
        if (k == 6) wr[k].pBufferInfo = &buffer_info;
        else wr[k].pImageInfo = &infos[k];
    }
    vk.UpdateDescriptorSets(p.device, 8, wr, 0, nullptr);
    // The marker: GENERAL once; its last blit's read done before the pass writes it again.
    image_barrier(cb, p.marker, p.marker_ready ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                  VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    p.marker_ready = true;
    vk.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vk.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
    vk.CmdDispatch(cb, 1, 1, 1);
    // Its own writes to its blit and the copy back; the slot's to the host.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    vk.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                          nullptr, 0, nullptr);
    const BbRenderImage& into = after ? pic : c->scene;
    if (g_marker && !in_place && into.image && can_blit_into(into.format)) {
        VkImageBlit r{};
        r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.srcOffsets[1] = {16, 16, 1};
        const std::int32_t x0 = after ? static_cast<std::int32_t>(into.width) - 16 : 0;
        r.dstOffsets[0] = {x0, 0, 0};
        r.dstOffsets[1] = {x0 + 16, 16, 1};
        vk.CmdBlitImage(cb, p.marker, VK_IMAGE_LAYOUT_GENERAL, handle<VkImage>(into.image), static_cast<VkImageLayout>(into.layout), 1, &r,
                        VK_FILTER_NEAREST);
    }
    if (after && pic.image && (pic.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) && (in_place || (g_marker && can_blit_into(pic.format)))) {
        // The corner back, as the post-processing will read it.
        image_barrier(cb, handle<VkImage>(pic.image), static_cast<VkImageLayout>(pic.layout), static_cast<VkImageLayout>(pic.layout),
                      VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy r{};
        r.bufferOffset = offset + kCornerOffset;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageOffset = {static_cast<std::int32_t>(pic.width) - 16, 0, 0};
        r.imageExtent = {16, 16, 1};
        vk.CmdCopyImageToBuffer(cb, handle<VkImage>(pic.image), static_cast<VkImageLayout>(pic.layout), p.buffer, 1, &r);
        VkMemoryBarrier hb{};
        hb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vk.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
        note.corner = true;
        note.format = pic.format;
    }
    static bool told = false;
    if (!told && after) {
        told = true;
        g_api->log("render_probe: frame %llu: first calls made; the picture %ux%u format %u usage 0x%x, %s", static_cast<unsigned long long>(f.frame),
                   pic.width, pic.height, pic.format, pic.usage, in_place ? "written in place" : "blitted into");
    }
}

// --- on and off -----------------------------------------------------------------------

void turn(bool on) {
    if (on == g_on) return;
    g_on = on;
    g_api->aa_claim("render_probe", on ? g_claim : BB_AA_NONE);
    const std::uint32_t points = on ? BB_RENDER_BEFORE_UPSCALE | BB_RENDER_AFTER_UPSCALE | (g_want_motion ? BB_RENDER_MOTION : 0u) : 0u;
    g_api->on_render("render_probe", points, &hook, nullptr);
    g_api->log("render_probe: %s", on ? "on" : "off");
}

void on_option(const char* key, const char* value, void*) {
    if (key && !key[0]) turn(value && std::strcmp(value, "true") == 0);
}

// The main thread's view: frame_info as any thread gets it, and the follow
// camera's focus point (the player raised by the camera's height) near the
// middle of the last frame's screen.
void on_frame(void*) {
    if (!g_on) return;
    static std::uint64_t frames = 0, near_middle = 0, focus_checked = 0;
    if (++frames % 30 != 0) return;
    BbFrameInfo f{};
    f.size = sizeof(f);
    const std::uint64_t last = g_last_frame.load(std::memory_order_relaxed);
    if (g_api->frame_info(&f) != 0) {
        check(last == 0, "frame_info once a hook has run");
        return;
    }
    check(f.size == sizeof(f), "frame_info's size");
    check(f.frame + 3 >= last && f.frame <= last + 3, "frame_info the frame the hooks saw", "%g against %g", static_cast<double>(f.frame),
          static_cast<double>(last));
    float pos[3], focus[3];
    if (!(f.flags & BB_FRAME_CAMERA) || g_api->camera_get(pos, focus) != 0) return;
    const double w[4] = {focus[0], focus[1], focus[2], 1.0};
    double c[4];
    apply(f.now.view_proj, w, c);
    if (c[3] <= 0.0) return;
    ++focus_checked;
    if (std::fabs(c[0] / c[3]) < 0.15 && std::fabs(c[1] / c[3]) < 0.15) ++near_middle;
    if (focus_checked == 1 || focus_checked % 20 == 0)
        g_api->log("render_probe: the follow camera's focus near the screen's middle in %llu of %llu looks (last at %.3f %.3f)",
                   static_cast<unsigned long long>(near_middle), static_cast<unsigned long long>(focus_checked), c[0] / c[3], c[1] / c[3]);
}

}  // namespace

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION, BB_PLUGIN_OPT_IN | BB_PLUGIN_LIVE_SWITCH, "render_probe", "Render probe (a test)", "1.0.0", "bbhost",
    "A test of the plugin API's frame info, render hooks and anti-aliasing claim: markers in the picture's corners and checks in the log."};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_CHOICE, "claim", "Anti-aliasing claim",
     "none: no claim. leave: leaves the anti-aliasing to bbhost. take: takes it over - bbhost's TAA, FSR and DLSS and the game's own "
     "anti-aliasing then stay off. At the next start.",
     "none", 0, 0, "none|leave|take"},
    {BB_OPT_BOOL, "motion", "Ask for the motion", "The hooks are given the frame's motion, made for them without an upscaler too.", "true",
     0, 0, nullptr},
    {BB_OPT_BOOL, "marker", "Markers", "The frame number's marker in the scene's top left and the picture's top right.", "true", 0, 0,
     nullptr},
    {BB_OPT_INT, "log_every", "Log every", "A line of what it read back every this many frames.", "300", 30, 3600, nullptr},
    {BB_OPT_END, nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr}};

extern "C" BB_PLUGIN_EXPORT int bb_plugin_init(const BbHostApi* api) {
    if (api->version < 27 || api->size < offsetof(BbHostApi, aa_claim) + sizeof(void*)) return 1;  // nothing to probe
    g_api = api;
    const std::string claim = api->config("render_probe.claim");
    g_claim = claim == "take" ? BB_AA_TAKE : claim == "leave" ? BB_AA_LEAVE : BB_AA_NONE;
    g_want_motion = std::strcmp(api->config("render_probe.motion"), "false") != 0;
    g_marker = std::strcmp(api->config("render_probe.marker"), "false") != 0;
    const int every = std::atoi(api->config("render_probe.log_every"));
    g_log_every = every >= 30 ? static_cast<unsigned>(every) : 300u;
    api->log("render_probe: claim %s, motion %s, markers %s, a line every %u frames", claim.empty() ? "none" : claim.c_str(),
             g_want_motion ? "asked for" : "not asked for", g_marker ? "on" : "off", g_log_every);
    if (std::strcmp(api->config("plugins.render_probe"), "true") == 0) turn(true);
    return 0;
}

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    api->on_frame(&on_frame, nullptr);
    api->on_option("render_probe", &on_option, nullptr);
    return 0;
}
