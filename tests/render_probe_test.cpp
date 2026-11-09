// tools/render_probe on a host of our own and a pretend Vulkan device: no
// game, no GPU, no Vulkan loader. The probe is loaded as the host loads
// plugins and handed the calls plugin API 27's render hooks make - frame
// info from a known camera (host/render_api.h, as the host makes it), the
// images as the call point hands them - and every function it asks the
// device for is the pretend device's, which records what it is given and
// carries out the one copy the CPU reads back as the probe's pass would leave
// the picture. Checked: on an API 26 host it refuses having called nothing;
// at the start it claims the anti-aliasing as its option says and registers
// both points and the motion; it never asks for anything that submits, waits
// or draws; every barrier it records on a host image leaves its layout as
// given; its sets name the views of the call it records (a new generation's
// at once); a picture that is a storage image is written in place, one that
// is not blitted into, the scene's marker blitted; what it reads back a ring
// later passes its own checks; a frame that breaks the rules (a jitter past
// half a pixel, no motion when asked for) is logged as failed; and its live
// switch takes the hooks and the claim away and back.
//
//   render_probe_test <render_probe.so>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "bbhost_plugin.h"
#include "host/render_api.h"

#include <dlfcn.h>

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

namespace {

// --- the pretend device ----------------------------------------------------------

struct Device {
    std::uint64_t next = 0x10000;
    std::set<std::string> asked;               // every name asked for
    std::vector<std::string> cmds;             // the vkCmd* recorded, in order
    std::vector<VkImageMemoryBarrier> barriers;  // image barriers recorded
    std::map<std::uint64_t, std::vector<std::uint8_t>> memory;  // by VkDeviceMemory
    std::uint64_t buffer_memory = 0;           // the ring's
    std::vector<VkImageView> written;          // the image views the last update named, by binding (0-7)
    std::vector<VkImageBlit> blits;
    std::vector<VkImage> blit_dst;
    int copies = 0, dispatches = 0, forbidden = 0, updates = 0;
} D;

template <typename H>
H fresh() {
    return reinterpret_cast<H>(static_cast<std::uintptr_t>(D.next += 0x10));
}

VKAPI_ATTR VkResult VKAPI_CALL create_shader(VkDevice, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks*, VkShaderModule* m) {
    CHECK(ci->codeSize > 64 && ci->pCode[0] == 0x07230203u);  // SPIR-V
    *m = fresh<VkShaderModule>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_shader(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL create_set_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo* ci, const VkAllocationCallbacks*,
                                                 VkDescriptorSetLayout* l) {
    CHECK(ci->bindingCount == 8);
    *l = fresh<VkDescriptorSetLayout>();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_layout(VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout* l) {
    *l = fresh<VkPipelineLayout>();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_compute(VkDevice, VkPipelineCache, std::uint32_t n, const VkComputePipelineCreateInfo* ci,
                                              const VkAllocationCallbacks*, VkPipeline* p) {
    CHECK(n == 1 && ci->stage.stage == VK_SHADER_STAGE_COMPUTE_BIT);
    *p = fresh<VkPipeline>();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_pool(VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool* p) {
    *p = fresh<VkDescriptorPool>();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL allocate_sets(VkDevice, const VkDescriptorSetAllocateInfo* ai, VkDescriptorSet* sets) {
    for (std::uint32_t k = 0; k < ai->descriptorSetCount; ++k) sets[k] = fresh<VkDescriptorSet>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL update_sets(VkDevice, std::uint32_t n, const VkWriteDescriptorSet* w, std::uint32_t, const VkCopyDescriptorSet*) {
    ++D.updates;
    D.written.assign(8, VK_NULL_HANDLE);
    for (std::uint32_t k = 0; k < n; ++k)
        if (w[k].pImageInfo && w[k].dstBinding < 8) D.written[w[k].dstBinding] = w[k].pImageInfo->imageView;
}
VKAPI_ATTR VkResult VKAPI_CALL create_sampler(VkDevice, const VkSamplerCreateInfo*, const VkAllocationCallbacks*, VkSampler* s) {
    *s = fresh<VkSampler>();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice, const VkImageCreateInfo* ci, const VkAllocationCallbacks*, VkImage* i) {
    CHECK(ci->extent.width == 16 && ci->extent.height == 16);
    *i = fresh<VkImage>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL image_requirements(VkDevice, VkImage, VkMemoryRequirements* r) {
    r->size = 4096;
    r->alignment = 256;
    r->memoryTypeBits = 3;
}
VKAPI_ATTR VkResult VKAPI_CALL bind_image(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL create_view(VkDevice, const VkImageViewCreateInfo*, const VkAllocationCallbacks*, VkImageView* v) {
    *v = fresh<VkImageView>();
    return VK_SUCCESS;
}
VkDeviceSize g_buffer_size = 0;
VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo* ci, const VkAllocationCallbacks*, VkBuffer* b) {
    g_buffer_size = ci->size;
    *b = fresh<VkBuffer>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL buffer_requirements(VkDevice, VkBuffer, VkMemoryRequirements* r) {
    r->size = g_buffer_size;
    r->alignment = 256;
    r->memoryTypeBits = 3;
}
VKAPI_ATTR VkResult VKAPI_CALL bind_buffer(VkDevice, VkBuffer, VkDeviceMemory m, VkDeviceSize) {
    D.buffer_memory = reinterpret_cast<std::uintptr_t>(m);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice, const VkMemoryAllocateInfo* ai, const VkAllocationCallbacks*, VkDeviceMemory* m) {
    *m = fresh<VkDeviceMemory>();
    D.memory[reinterpret_cast<std::uintptr_t>(*m)].assign(static_cast<std::size_t>(ai->allocationSize), 0);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL map_memory(VkDevice, VkDeviceMemory m, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void** out) {
    *out = D.memory[reinterpret_cast<std::uintptr_t>(m)].data();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer, VkPipelineBindPoint bp, VkPipeline) {
    CHECK(bp == VK_PIPELINE_BIND_POINT_COMPUTE);  // compute only
    D.cmds.push_back("bind_pipeline");
}
VKAPI_ATTR void VKAPI_CALL bind_sets(VkCommandBuffer, VkPipelineBindPoint bp, VkPipelineLayout, std::uint32_t, std::uint32_t, const VkDescriptorSet*,
                                     std::uint32_t, const std::uint32_t*) {
    CHECK(bp == VK_PIPELINE_BIND_POINT_COMPUTE);
    D.cmds.push_back("bind_sets");
}
VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    CHECK(x == 1 && y == 1 && z == 1);
    ++D.dispatches;
    D.cmds.push_back("dispatch");
}
VKAPI_ATTR void VKAPI_CALL pipeline_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, std::uint32_t,
                                            const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t n,
                                            const VkImageMemoryBarrier* b) {
    for (std::uint32_t k = 0; k < n; ++k) D.barriers.push_back(b[k]);
    D.cmds.push_back("barrier");
}
VKAPI_ATTR void VKAPI_CALL blit(VkCommandBuffer, VkImage, VkImageLayout, VkImage dst, VkImageLayout, std::uint32_t n, const VkImageBlit* r, VkFilter) {
    for (std::uint32_t k = 0; k < n; ++k) D.blits.push_back(r[k]);
    D.blit_dst.push_back(dst);
    D.cmds.push_back("blit");
}
// The copy back of the picture's corner: the texels the probe's pass leaves
// there - the frame's marker in cyan - as RGBA16F, the frame read from the
// slot the corner goes into (its size[3], 140 bytes into it).
VKAPI_ATTR void VKAPI_CALL copy_back(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t n, const VkBufferImageCopy* r) {
    CHECK(n == 1 && r->imageExtent.width == 16 && r->imageExtent.height == 16);
    ++D.copies;
    D.cmds.push_back("copy_back");
    std::vector<std::uint8_t>& mem = D.memory[D.buffer_memory];
    const VkDeviceSize slot = r->bufferOffset - 4096;
    std::uint32_t frame = 0;
    std::memcpy(&frame, mem.data() + slot + 140, 4);
    for (std::uint32_t y = 0; y < 16; ++y)
        for (std::uint32_t x = 0; x < 16; ++x) {
            const bool one = (frame >> ((y / 4) * 4 + x / 4)) & 1u;
            const std::uint16_t t[4] = {0, static_cast<std::uint16_t>(one ? 0x3c00 : 0), static_cast<std::uint16_t>(one ? 0x3c00 : 0), 0x3c00};
            std::memcpy(mem.data() + r->bufferOffset + (y * 16 + x) * 8, t, 8);
        }
}
VKAPI_ATTR void VKAPI_CALL memory_properties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* p) {
    std::memset(p, 0, sizeof(*p));
    p->memoryTypeCount = 2;
    p->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    p->memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
}
VKAPI_ATTR void VKAPI_CALL format_properties(VkPhysicalDevice, VkFormat, VkFormatProperties* p) {
    std::memset(p, 0, sizeof(*p));
    p->optimalTilingFeatures = VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT;
}
VKAPI_ATTR void VKAPI_CALL forbidden() { ++D.forbidden; }

BbVkVoidFunction device_proc(void*, const char* name) {
    D.asked.insert(name);
    struct Entry {
        const char* name;
        PFN_vkVoidFunction fn;
    };
#define E(n, f) {n, reinterpret_cast<PFN_vkVoidFunction>(&f)}
    static const Entry table[] = {
        E("vkCreateShaderModule", create_shader),     E("vkDestroyShaderModule", destroy_shader),
        E("vkCreateDescriptorSetLayout", create_set_layout), E("vkCreatePipelineLayout", create_layout),
        E("vkCreateComputePipelines", create_compute), E("vkCreateDescriptorPool", create_pool),
        E("vkAllocateDescriptorSets", allocate_sets), E("vkUpdateDescriptorSets", update_sets),
        E("vkCreateSampler", create_sampler),         E("vkCreateImage", create_image),
        E("vkGetImageMemoryRequirements", image_requirements), E("vkBindImageMemory", bind_image),
        E("vkCreateImageView", create_view),          E("vkCreateBuffer", create_buffer),
        E("vkGetBufferMemoryRequirements", buffer_requirements), E("vkBindBufferMemory", bind_buffer),
        E("vkAllocateMemory", allocate_memory),       E("vkMapMemory", map_memory),
        E("vkCmdBindPipeline", bind_pipeline),        E("vkCmdBindDescriptorSets", bind_sets),
        E("vkCmdDispatch", dispatch),                 E("vkCmdPipelineBarrier", pipeline_barrier),
        E("vkCmdBlitImage", blit),                    E("vkCmdCopyImageToBuffer", copy_back),
    };
#undef E
    for (const Entry& e : table)
        if (std::strcmp(e.name, name) == 0) return reinterpret_cast<BbVkVoidFunction>(e.fn);
    return reinterpret_cast<BbVkVoidFunction>(&forbidden);  // anything else: recorded as asked, a failure if called
}
BbVkVoidFunction instance_proc(void*, const char* name) {
    D.asked.insert(name);
    if (std::strcmp(name, "vkGetPhysicalDeviceMemoryProperties") == 0) return reinterpret_cast<BbVkVoidFunction>(&memory_properties);
    if (std::strcmp(name, "vkGetPhysicalDeviceFormatProperties") == 0) return reinterpret_cast<BbVkVoidFunction>(&format_properties);
    return reinterpret_cast<BbVkVoidFunction>(&forbidden);
}

// --- the pretend host ---------------------------------------------------------------

struct Host {
    std::map<std::string, std::string> config = {{"plugins.render_probe", "true"}, {"render_probe.claim", "leave"}, {"render_probe.log_every", "30"}};
    std::vector<std::string> log;
    std::uint32_t points = 0;
    BbRenderFn fn = nullptr;
    void* user = nullptr;
    int claim = -1, claims = 0, renders = 0;
    void (*frame_fn)(void*) = nullptr;
    void (*option_fn)(const char*, const char*, void*) = nullptr;
    BbFrameInfo frame{};
    bool have_frame = false;
} H;

void fake_log(const char* fmt, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    H.log.push_back(b);
}
const char* fake_config(const char* key) {
    static std::string v;
    const auto it = H.config.find(key);
    v = it == H.config.end() ? "" : it->second;
    return v.c_str();
}
int fake_on_render(const char* name, std::uint32_t points, BbRenderFn fn, void* user) {
    CHECK(std::strcmp(name, "render_probe") == 0);
    H.points = points;
    H.fn = fn;
    H.user = user;
    ++H.renders;
    return 0;
}
int fake_aa_claim(const char* name, int claim) {
    CHECK(std::strcmp(name, "render_probe") == 0);
    H.claim = claim;
    ++H.claims;
    return 0;
}
int fake_on_frame(void (*fn)(void*), void*) {
    H.frame_fn = fn;
    return 0;
}
int fake_on_option(const char*, void (*fn)(const char*, const char*, void*), void*) {
    H.option_fn = fn;
    return 0;
}
int fake_frame_info(BbFrameInfo* out) {
    if (!H.have_frame || out->size < 8) return 1;
    const std::uint32_t n = out->size < sizeof(BbFrameInfo) ? out->size : static_cast<std::uint32_t>(sizeof(BbFrameInfo));
    std::memcpy(out, &H.frame, n);
    out->size = n;
    return 0;
}
int fake_camera_get(float pos[3], float focus[3]) {
    // The camera's eye, and ten metres ahead of it: the screen's middle.
    for (int k = 0; k < 3; ++k) {
        pos[k] = H.frame.now.camera.position[k];
        focus[k] = H.frame.now.camera.position[k] + 10.0f * H.frame.now.camera.forward[k];
    }
    return 0;
}

// A frame's camera as the host makes it: the view block's terms through render_api.h.
BbFrameCamera camera_at(float x) {
    render_api::ViewTerms t;
    t.tan_y = 0.394f;
    t.tan_x = t.tan_y * 16.0f / 9.0f;
    t.a = static_cast<float>(3013.0 / (3013.0 - 0.05));
    t.r2w = static_cast<float>(0.05 * 3013.0 / (0.05 - 3013.0));
    const float rows[12] = {1, 0, 0, x, 0, 1, 0, 20, 0, 0, 1, -50};
    std::memcpy(t.rows, rows, sizeof(rows));
    BbFrameCamera c{};
    bool reversed = false;
    render_api::frame_camera(t, &c, &reversed);
    return c;
}

BbRenderImage image(std::uint64_t img, std::uint64_t view, std::uint64_t gen, VkFormat f, VkImageLayout l, VkImageUsageFlags u, std::uint32_t w,
                    std::uint32_t h, bool write) {
    BbRenderImage i{};
    i.image = img;
    i.view = view;
    i.generation = gen;
    i.format = static_cast<std::uint32_t>(f);
    i.layout = static_cast<std::uint32_t>(l);
    i.usage = static_cast<std::uint32_t>(u);
    i.width = w;
    i.height = h;
    i.flags = write ? BB_IMAGE_WRITE : 0u;
    return i;
}

constexpr VkImageUsageFlags kTarget = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
constexpr VkImageUsageFlags kOwn = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT;

BbRenderDevice g_device{};

// One frame's two calls, as the call point makes them: `upscaled` an
// upscaler's own picture (a storage image at the output size) or none (the
// scene itself), `scene_view` the scene's view now, `motion` whether it is
// handed, the jitter as given.
void frame(std::uint64_t n, bool upscaled, std::uint64_t scene_view, std::uint64_t scene_gen, bool motion, float jitter) {
    BbFrameInfo& f = H.frame;
    const BbFrameCamera last = f.now;
    f = BbFrameInfo{};
    f.size = sizeof(f);
    f.frame = n;
    f.flip = n * 2;
    f.render_width = 1920;
    f.render_height = 1080;
    f.output_width = 1920;
    f.output_height = 1080;
    f.method = upscaled ? BB_UPSCALE_TAA : BB_UPSCALE_NONE;
    f.jitter_x = jitter;
    f.now = camera_at(100.0f + 0.01f * static_cast<float>(n));
    f.prev = n > 1 ? last : f.now;
    f.flags = BB_FRAME_CAMERA | (n > 1 ? BB_FRAME_PREV : 0u);
    f.reset = n == 1;
    H.have_frame = true;
    BbRenderCall c{};
    c.size = sizeof(c);
    c.point = BB_RENDER_BEFORE_UPSCALE;
    c.command_buffer = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0xcb00));
    c.frame = &f;
    c.device = &g_device;
    c.scene = image(0xa000, scene_view, scene_gen, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL, kTarget, 1920, 1080, true);
    c.depth = image(0xb000, 0xb100, 2, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    1920, 1080, false);
    if (motion)
        c.motion = image(0xc000, 0xc100, 3, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1920, 1080, false);
    H.fn(&c, H.user);
    CHECK(D.written.size() == 8 && D.written[0] == reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>(scene_view)));
    CHECK(D.written[1] == reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>(0xb100)));
    c.point = BB_RENDER_AFTER_UPSCALE;
    c.scene.flags = 0;
    c.picture = upscaled ? image(0xd000, 0xd100, 4, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL, kOwn, 1920, 1080, true)
                         : image(0xa000, scene_view, 5, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL, kTarget, 1920, 1080, true);
    H.fn(&c, H.user);
    // A storage picture is written in place (its view the pass's binding 7), any other blitted into.
    if (upscaled) CHECK(D.written[7] == reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>(0xd100)));
    else CHECK(D.written[7] != reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>(scene_view)));
    if (H.frame_fn) H.frame_fn(nullptr);
}

int count_log(const char* what) {
    int n = 0;
    for (const std::string& l : H.log) n += l.find(what) != std::string::npos;
    return n;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: render_probe_test <render_probe.so>\n");
        return 2;
    }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::fprintf(stderr, "render_probe_test: %s\n", dlerror());
        return 1;
    }
    auto init = reinterpret_cast<int (*)(const BbHostApi*)>(dlsym(h, "bb_plugin_init"));
    auto image_phase = reinterpret_cast<int (*)(const BbHostApi*)>(dlsym(h, "bb_plugin_image"));
    const auto* info = static_cast<const BbPluginInfo*>(dlsym(h, "bb_plugin_info"));
    CHECK(init && image_phase && info);
    if (!init || !image_phase || !info) return 1;
    CHECK(info->flags & BB_PLUGIN_OPT_IN);  // off unless turned on, wherever it is copied
    CHECK(std::strcmp(info->name, "render_probe") == 0);

    BbHostApi api{};
    api.version = 26;
    api.size = static_cast<std::uint32_t>(offsetof(BbHostApi, frame_info));
    api.log = &fake_log;
    api.config = &fake_config;
    CHECK(init(&api) != 0);  // an API 26 host: nothing to probe
    CHECK(H.renders == 0 && H.claims == 0);

    api.version = BB_PLUGIN_API_VERSION;
    api.size = sizeof(BbHostApi);
    api.on_frame = &fake_on_frame;
    api.on_option = &fake_on_option;
    api.camera_get = &fake_camera_get;
    api.frame_info = &fake_frame_info;
    api.on_render = &fake_on_render;
    api.aa_claim = &fake_aa_claim;
    CHECK(init(&api) == 0);
    CHECK(image_phase(&api) == 0);
    CHECK(H.claim == BB_AA_LEAVE);
    CHECK(H.points == (BB_RENDER_BEFORE_UPSCALE | BB_RENDER_AFTER_UPSCALE | BB_RENDER_MOTION) && H.fn);

    g_device.size = sizeof(g_device);
    g_device.api_version = VK_API_VERSION_1_3;
    g_device.instance = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x1000));
    g_device.physical_device = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x2000));
    g_device.device = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x3000));
    g_device.get_instance_proc_addr = &instance_proc;
    g_device.get_device_proc_addr = &device_proc;
    g_device.queue_family = 0;
    g_device.frames_in_flight = 18;

    // Forty frames with no upscaler, the scene made again at frame 20 (a Resolution change: a new view and
    // generation), then forty with an upscaler's storage picture.
    for (std::uint64_t n = 1; n <= 40; ++n) frame(n, false, n < 20 ? 0xa100 : 0xa200, n < 20 ? 1 : 6, true, 0.0f);
    CHECK(count_log("the scene is generation 6") == 1);
    for (std::uint64_t n = 41; n <= 80; ++n) frame(n, true, 0xa200, 6, true, 0.25f);
    CHECK(D.dispatches == 160);
    CHECK(D.copies == 80);  // each after-call's corner back
    // The scene's marker blitted into its top left each frame, the picture's
    // into its top right for the forty with no storage picture.
    int scene_blits = 0, picture_blits = 0;
    for (const VkImageBlit& b : D.blits) {
        if (b.dstOffsets[0].x == 0) ++scene_blits;
        if (b.dstOffsets[0].x == 1904) ++picture_blits;
    }
    CHECK(scene_blits == 80 && picture_blits == 40);
    // Nothing that submits, waits or draws asked for, nothing else called.
    for (const char* bad : {"vkQueueSubmit", "vkQueueWaitIdle", "vkDeviceWaitIdle", "vkWaitForFences", "vkGetDeviceQueue", "vkCmdDraw",
                            "vkCmdBeginRendering", "vkEndCommandBuffer"})
        CHECK(!D.asked.count(bad));
    CHECK(D.forbidden == 0);
    // Every barrier on a host image leaves its layout as given (its own marker: UNDEFINED once, to GENERAL).
    int own_first = 0;
    for (const VkImageMemoryBarrier& b : D.barriers) {
        if (b.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            ++own_first;
            continue;
        }
        CHECK(b.oldLayout == b.newLayout);
    }
    CHECK(own_first == 1);
    // What came back a ring later passed: no check failed, the marker read back, and the lines said so.
    CHECK(count_log("FAILED") == 0);
    CHECK(count_log("the picture's corner the marker") >= 1);
    CHECK(count_log("reprojection against the host's motion") >= 1);
    CHECK(count_log("focus near the screen's middle") >= 1);

    // A frame that breaks the rules: a jitter past half a pixel, and no motion when asked for.
    frame(81, true, 0xa200, 6, false, 0.75f);
    CHECK(count_log("FAILED the jitter within half a pixel") == 1);
    CHECK(count_log("FAILED the motion when asked for") >= 1);

    // The live switch: off takes the hooks and the claim away; on gives them back.
    CHECK(H.option_fn);
    if (H.option_fn) {
        H.option_fn("", "false", nullptr);
        CHECK(H.points == 0 && H.claim == BB_AA_NONE);
        H.option_fn("", "true", nullptr);
        CHECK(H.points != 0 && H.claim == BB_AA_LEAVE);
    }
    if (g_failures) {
        for (const std::string& l : H.log) std::fprintf(stderr, "  log: %s\n", l.c_str());
        std::fprintf(stderr, "render_probe_test: %d failures\n", g_failures);
        return 1;
    }
    std::printf("render_probe_test: %d dispatches, %d blits and %d copies back on a pretend device, none submitting, waiting or drawing; "
                "the checks pass and catch a bad frame\n",
                D.dispatches, static_cast<int>(D.blits.size()), D.copies);
    return 0;
}
