// The command stream's self-test (recorder.cpp): bbhost --stream-selftest[=N]
// or BBHOST_STREAM_SELFTEST=N, before the guest loads.
//
// Vulkan comes up headless through host_gpu_init and the test runs on the
// production stream - the real command pool and slots, bb-record, bb-submit,
// the fences - not a model of it. Each round draws a seeded random sequence of
// commands and runs it four ways: everything in place (BBHOST_RECORDER=0),
// draws on the recorder and the rest in place (BBHOST_STREAM_SUBMIT=0), the
// stream replayed on the producer (BBHOST_RECORDER_INLINE=1), and the stream
// on bb-record. A CPU model applies the same sequence to arrays; after each
// way the buffers and the image are read back and compared with it, so the
// four agree with the model and with each other.
//
// What a round holds: fills; buffer copies of 1 to 600 regions; a draw
// packet's transfer batch (DrawCmds::copy_buffer, past its 32 copies into
// ops of their own); staged updates the producer writes into upload staging
// and copies (the write-combining hand-off, R2); compute through
// update_sets, bind_sets, push_constants and dispatch; scissored draws
// through a real DrawCmds packet (pipeline, viewport, scissor, a push
// descriptor, the draw) inside a pass begun and ended as ops or by the
// packet, one of them in an occlusion query; image clears and copies both
// ways; timestamps; submissions and flushes at random points; the
// presenter's path (host_gpu_submit_presenter: two submissions on a pool of
// their own joined by a semaphore, the first copying what the game's last
// submission wrote); a temporary buffer and image used and then destroyed
// through the deferred lists. Every array handed to rec() or a packet is
// overwritten with 0xCD right after the call: a record that kept a pointer
// would replay garbage.
//
// Then a cost table: the producer's nanoseconds per op recorded in place and
// appended to the stream, a submission's end and begin, a store fence.

#include "host/gpu_internal.h"
#include "host/shaders/stream_selftest.spv.h"
#include "host/shaders/stream_selftest_frag.spv.h"
#include "host/shaders/stream_selftest_vert.spv.h"
#include "host/stream_ops.h"

#include "core/host_clock.h"
#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace gpu {
namespace {

constexpr std::uint32_t kWords = 65536;  // each buffer: 256 KiB
constexpr std::uint32_t kBuffers = 4;    // A, B, C, and F (what the presenter's path copied)
constexpr std::uint32_t kF = 3;
constexpr std::uint32_t kImage = 64;  // 64 x 64 R32_UINT
constexpr std::uint32_t kTimestamps = 64;
constexpr std::uint32_t kSlotWords = 64;  // image copies and push ranges use 256-byte slots
constexpr std::uint32_t kSlots = kWords / kSlotWords;

// splitmix64: the same sequence on every platform and library.
struct Rng {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    std::uint32_t below(std::uint32_t n) { return n ? static_cast<std::uint32_t>(next() % n) : 0; }
    std::uint32_t u32() { return static_cast<std::uint32_t>(next()); }
};

// Where the arrays handed to rec() and to packets live: overwritten with 0xCD
// after each call (garble), so nothing may keep pointing into it.
struct Scratch {
    std::vector<std::uint64_t> words = std::vector<std::uint64_t>(16384);
    std::size_t used = 0;  // words
    template <class T> T* take(std::size_t n) {
        const std::size_t w = (n * sizeof(T) + 7) / 8;
        if (used + w > words.size()) return nullptr;  // sized for the largest call below
        T* t = reinterpret_cast<T*>(words.data() + used);
        std::memset(static_cast<void*>(t), 0, w * 8);
        used += w;
        return t;
    }
    void garble() {
        std::memset(words.data(), 0xCD, used * 8);
        used = 0;
    }
};

struct Model {
    std::vector<std::uint32_t> buf[kBuffers];
    std::vector<std::uint32_t> img;
    void reset() {
        for (auto& b : buf) b.assign(kWords, 0);
        img.assign(kImage * kImage, 0);
    }
};

struct Resources {
    DevBuffer buf[kBuffers];
    DevBuffer readback;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory image_mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDescriptorSetLayout comp_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout comp_layout = VK_NULL_HANDLE;
    VkPipeline comp = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout empty_layout = VK_NULL_HANDLE, push_layout = VK_NULL_HANDLE;
    VkPipelineLayout gfx_layout = VK_NULL_HANDLE;
    VkPipeline gfx = VK_NULL_HANDLE;
    VkQueryPool ts_pool = VK_NULL_HANDLE, occ_pool = VK_NULL_HANDLE;
    VkCommandPool fpool = VK_NULL_HANDLE;
    VkCommandBuffer fcmd[2] = {};
    VkSemaphore fsem = VK_NULL_HANDLE;
    VkFence ffence = VK_NULL_HANDLE;
    bool draws = false, timestamps = false;
};

struct Way {
    const char* name;
    int mode;  // stream_test_set_mode: 0 none, 1 inline, 2 the recorder thread
    bool submit;
};
constexpr Way kWays[] = {
    {"in place", 0, false},
    {"draws on the recorder", 2, false},
    {"inline", 1, true},
    {"threaded", 2, true},
};

struct RoundStats {
    std::uint64_t steps = 0, submits = 0, flushes = 0, foreign = 0;
    std::uint64_t differences = 0;
    std::uint64_t occ_unresolved = 0, ts_unavailable = 0, ts_backwards = 0;
};

// ---- setup ----

VkShaderModule make_module(const std::uint32_t* words, std::size_t bytes) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = bytes;
    ci.pCode = words;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.device, &ci, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    return m;
}

bool make_image(VkImage& image, VkDeviceMemory& mem, std::uint32_t size, VkImageUsageFlags usage,
                VkFormat format = VK_FORMAT_R32_UINT, VkDeviceSize* allocation_size = nullptr) {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {size, size, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g.device, &ici, nullptr, &image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g.device, image, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyImage(g.device, image, nullptr);
        image = VK_NULL_HANDLE;
        return false;
    }
    if (allocation_size) *allocation_size = req.size;
    return vkBindImageMemory(g.device, image, mem, 0) == VK_SUCCESS;
}

bool setup(Resources& r) {
    for (DevBuffer& b : r.buf) {
        if (!create_dev_buffer(b, kWords * 4ull, false)) return false;
    }
    if (!create_dev_buffer(r.readback, kWords * 4ull * kBuffers + kImage * kImage * 4ull, true, true)) return false;
    if (!make_image(r.image, r.image_mem, kImage,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        return false;
    }
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = r.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R32_UINT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(g.device, &vci, nullptr, &r.view) != VK_SUCCESS) return false;

    // Compute: dst[i] = src[i] * mul + add, both buffers from one set.
    VkDescriptorSetLayoutBinding cb[2] = {};
    for (std::uint32_t k = 0; k < 2; ++k) {
        cb[k].binding = k;
        cb[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        cb[k].descriptorCount = 1;
        cb[k].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 2;
    lci.pBindings = cb;
    if (vkCreateDescriptorSetLayout(g.device, &lci, nullptr, &r.comp_set_layout) != VK_SUCCESS) return false;
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &r.comp_set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(g.device, &plci, nullptr, &r.comp_layout) != VK_SUCCESS) return false;
    const VkShaderModule cm = make_module(k_stream_selftest_spv, sizeof(k_stream_selftest_spv));
    if (!cm) return false;
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = cm;
    cpci.stage.pName = "main";
    cpci.layout = r.comp_layout;
    const VkResult cr = vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &r.comp);
    vkDestroyShaderModule(g.device, cm, nullptr);
    if (cr != VK_SUCCESS) return false;
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 4096;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(g.device, &dpci, nullptr, &r.pool) != VK_SUCCESS) return false;

    // Graphics: a full-viewport triangle whose fragments write the word a
    // push descriptor's range starts at (set 2, as the game's constant
    // buffers are pushed); sets 0 and 1 empty.
    r.draws = g.has_push_descriptor && g.cmd_push_descriptor_set;
    if (r.draws) {
        VkDescriptorSetLayoutCreateInfo eci{};
        eci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        if (vkCreateDescriptorSetLayout(g.device, &eci, nullptr, &r.empty_layout) != VK_SUCCESS) return false;
        VkDescriptorSetLayoutBinding pb{};
        pb.binding = 0;
        pb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pb.descriptorCount = 1;
        pb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        pci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        pci.bindingCount = 1;
        pci.pBindings = &pb;
        if (vkCreateDescriptorSetLayout(g.device, &pci, nullptr, &r.push_layout) != VK_SUCCESS) return false;
        const VkDescriptorSetLayout three[3] = {r.empty_layout, r.empty_layout, r.push_layout};
        VkPipelineLayoutCreateInfo gl{};
        gl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        gl.setLayoutCount = 3;
        gl.pSetLayouts = three;
        if (vkCreatePipelineLayout(g.device, &gl, nullptr, &r.gfx_layout) != VK_SUCCESS) return false;
        const VkShaderModule vm = make_module(k_stream_selftest_vert_spv, sizeof(k_stream_selftest_vert_spv));
        const VkShaderModule fm = make_module(k_stream_selftest_frag_spv, sizeof(k_stream_selftest_frag_spv));
        VkPipelineShaderStageCreateInfo st[2] = {};
        st[0].sType = st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        st[0].module = vm;
        st[0].pName = "main";
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module = fm;
        st[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cbs{};
        cbs.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cbs.attachmentCount = 1;
        cbs.pAttachments = &ba;
        const VkDynamicState dyn[4] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                      VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE};
        VkPipelineDynamicStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        ds.dynamicStateCount = g.has_gpl ? 4 : 2;
        ds.pDynamicStates = dyn;
        const VkFormat fmt = VK_FORMAT_R32_UINT;
        VkPipelineRenderingCreateInfo rci{};
        rci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rci.colorAttachmentCount = 1;
        rci.pColorAttachmentFormats = &fmt;
        VkGraphicsPipelineCreateInfo gp{};
        gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp.pNext = &rci;
        gp.stageCount = 2;
        gp.pStages = st;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vp;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pColorBlendState = &cbs;
        gp.pDynamicState = &ds;
        gp.layout = r.gfx_layout;
        const VkResult gr = vm && fm ? vkCreateGraphicsPipelines(g.device, VK_NULL_HANDLE, 1, &gp, nullptr, &r.gfx) : VK_ERROR_UNKNOWN;
        if (vm) vkDestroyShaderModule(g.device, vm, nullptr);
        if (fm) vkDestroyShaderModule(g.device, fm, nullptr);
        if (gr != VK_SUCCESS) return false;
        VkQueryPoolCreateInfo oci{};
        oci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        oci.queryType = VK_QUERY_TYPE_OCCLUSION;
        oci.queryCount = 1;
        if (vkCreateQueryPool(g.device, &oci, nullptr, &r.occ_pool) != VK_SUCCESS) return false;
    } else {
        host_log("stream selftest: no push descriptors on this device; the draws are left out");
    }

    std::uint32_t nfam = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nfam, nullptr);
    std::vector<VkQueueFamilyProperties> fams(nfam);
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nfam, fams.data());
    r.timestamps = g.family < nfam && fams[g.family].timestampValidBits != 0;
    if (r.timestamps) {
        VkQueryPoolCreateInfo tci{};
        tci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        tci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        tci.queryCount = kTimestamps;
        if (vkCreateQueryPool(g.device, &tci, nullptr, &r.ts_pool) != VK_SUCCESS) r.timestamps = false;
    }

    // The presenter's side: a pool of its own, two command buffers, a
    // semaphore between them and a fence.
    VkCommandPoolCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = g.family;
    if (vkCreateCommandPool(g.device, &cpi, nullptr, &r.fpool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = r.fpool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 2;
    if (vkAllocateCommandBuffers(g.device, &cai, r.fcmd) != VK_SUCCESS) return false;
    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (vkCreateSemaphore(g.device, &sci, nullptr, &r.fsem) != VK_SUCCESS) return false;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    return vkCreateFence(g.device, &fci, nullptr, &r.ffence) == VK_SUCCESS;
}

// ---- the commands, each with its model ----

void full_barrier(Scratch& s) {
    VkMemoryBarrier* mb = s.take<VkMemoryBarrier>(1);
    mb->sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb->srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb->dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, mb, 0, nullptr, 0, nullptr);
    s.garble();
}

void image_to_general(Scratch& s, VkImage image) {
    VkImageMemoryBarrier* ib = s.take<VkImageMemoryBarrier>(1);
    ib->sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ib->srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    ib->dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ib->oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib->newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib->srcQueueFamilyIndex = ib->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib->image = image;
    ib->subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, ib);
    s.garble();
}

// n distinct indices below `range` (a partial shuffle), the first `skip`
// of a shuffle left out: two calls on one Shuffle give disjoint sets.
struct Shuffle {
    std::vector<std::uint32_t> v;
    std::uint32_t at = 0;
    Shuffle(Rng& rng, std::uint32_t range) : v(range), rng_(rng) { std::iota(v.begin(), v.end(), 0u); }
    std::uint32_t take() {
        const std::uint32_t j = at + rng_.below(static_cast<std::uint32_t>(v.size()) - at);
        std::swap(v[at], v[j]);
        return v[at++];
    }

private:
    Rng& rng_;
};

struct Arm {
    Resources& r;
    Model& m;
    Scratch& s;
    Rng rng;
    RoundStats& st;
    std::uint32_t ts_used = 0, foreign_used = 0;
    bool occ_used = false;
    bool failed = false;

    Arm(Resources& r_, Model& m_, Scratch& s_, std::uint64_t seed, RoundStats& st_) : r(r_), m(m_), s(s_), rng{seed}, st(st_) {}

    VkBuffer b(std::uint32_t k) const { return r.buf[k].buffer; }

    void fill() {
        const std::uint32_t x = rng.below(3), off = rng.below(kWords);
        const std::uint32_t len = 1 + rng.below(std::min<std::uint32_t>(4096, kWords - off));
        const std::uint32_t v = rng.u32();
        rec().fill_buffer(b(x), off * 4ull, len * 4ull, v);
        std::fill_n(m.buf[x].begin() + off, len, v);
    }

    // Regions of up to 16 words in 16-word slots: the destinations
    // disjoint, and in one buffer the sources disjoint from them.
    void copy() {
        const std::uint32_t x = rng.below(3), y = rng.below(3);
        const std::uint32_t n = rng.below(4) == 0 ? 1 + rng.below(600) : 1 + rng.below(8);
        Shuffle slots(rng, kWords / 16);
        VkBufferCopy* regions = s.take<VkBufferCopy>(n);
        std::vector<std::uint32_t> src(m.buf[x]);
        for (std::uint32_t k = 0; k < n; ++k) {
            const std::uint32_t len = 1 + rng.below(16);
            const std::uint32_t d = slots.take() * 16, so = (x == y ? slots.take() : rng.below(kWords / 16)) * 16 + rng.below(17 - len);
            regions[k] = VkBufferCopy{so * 4ull, d * 4ull, len * 4ull};
            std::copy_n(src.begin() + so, len, m.buf[y].begin() + d);
        }
        rec().copy_buffer(b(x), b(y), n, regions);
        s.garble();
    }

    // A transfer batch in a draw packet (the copy tokens' and copy
    // versions' way); past the packet's 32 copies the rest are ops.
    void packet_copies() {
        const std::uint32_t x = rng.below(3), y = rng.below(3);
        const std::uint32_t n = 1 + rng.below(40);
        Shuffle slots(rng, kWords / 16);
        DrawCmds c(true);
        if (!c.transfer_begin()) rec().transfer_barrier(true);
        std::vector<std::uint32_t> src(m.buf[x]);
        for (std::uint32_t k = 0; k < n; ++k) {
            const std::uint32_t len = 1 + rng.below(16);
            const std::uint32_t d = slots.take() * 16, so = (x == y ? slots.take() : rng.below(kWords / 16)) * 16;
            if (rng.below(8) == 0 && !c.copy_order()) rec().copy_order_barrier();
            VkBufferCopy* region = s.take<VkBufferCopy>(1);
            *region = VkBufferCopy{so * 4ull, d * 4ull, len * 4ull};
            if (!c.copy_buffer(b(x), b(y), *region)) rec().copy_buffer(b(x), b(y), 1, region);
            s.garble();
            std::copy_n(src.begin() + so, len, m.buf[y].begin() + d);
        }
        if (!c.transfer_end()) rec().transfer_barrier(false);
        c.publish();
    }

    // Words the producer writes into upload staging, then copied: they must
    // leave this core's write-combining buffers before the submission (R2).
    void staged() {
        const std::uint32_t x = rng.below(3), n = 1 + rng.below(2048), d = rng.below(kWords - n + 1);
        DevBuffer span;
        VkDeviceSize off = 0;
        if (!acquire_staging_locked(span, n * 4ull, off)) {
            host_log("stream selftest: no upload staging");
            failed = true;
            return;
        }
        auto* w = static_cast<std::uint32_t*>(span.map);
        for (std::uint32_t k = 0; k < n; ++k) w[k] = m.buf[x][d + k] = rng.u32();
        VkBufferCopy* region = s.take<VkBufferCopy>(1);
        *region = VkBufferCopy{off, d * 4ull, n * 4ull};
        rec().copy_buffer(span.buffer, b(x), 1, region);
        s.garble();
    }

    void compute() {
        const std::uint32_t x = rng.below(3), y = (x + 1 + rng.below(2)) % 3;
        const std::uint32_t n = 1 + rng.below(4096), so = rng.below(kWords - n + 1), d = rng.below(kWords - n + 1);
        const std::uint32_t mul = rng.u32() | 1, add = rng.u32();
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = r.pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &r.comp_set_layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(g.device, &ai, &set) != VK_SUCCESS) {
            host_log("stream selftest: descriptor pool exhausted");
            failed = true;
            return;
        }
        VkDescriptorBufferInfo* infos = s.take<VkDescriptorBufferInfo>(2);
        infos[0] = {b(x), 0, VK_WHOLE_SIZE};
        infos[1] = {b(y), 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet* writes = s.take<VkWriteDescriptorSet>(2);
        for (std::uint32_t k = 0; k < 2; ++k) {
            writes[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[k].dstSet = set;
            writes[k].dstBinding = k;
            writes[k].descriptorCount = 1;
            writes[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[k].pBufferInfo = &infos[k];
        }
        rec().update_sets(2, writes);
        s.garble();
        rec().bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, r.comp);
        VkDescriptorSet* sets = s.take<VkDescriptorSet>(1);
        sets[0] = set;
        rec().bind_sets(VK_PIPELINE_BIND_POINT_COMPUTE, r.comp_layout, 0, 1, sets);
        s.garble();
        std::uint32_t* push = s.take<std::uint32_t>(5);
        push[0] = so;
        push[1] = d;
        push[2] = n;
        push[3] = mul;
        push[4] = add;
        rec().push_constants(r.comp_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, push);
        s.garble();
        rec().dispatch((n + 63) / 64, 1, 1);
        for (std::uint32_t k = 0; k < n; ++k) m.buf[y][d + k] = m.buf[x][so + k] * mul + add;
    }

    void begin_pass() {
        VkRenderingAttachmentInfo* att = s.take<VkRenderingAttachmentInfo>(1);
        att->sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        att->imageView = r.view;
        att->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        att->loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        att->storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo* ri = s.take<VkRenderingInfo>(1);
        ri->sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri->renderArea = {{0, 0}, {kImage, kImage}};
        ri->layerCount = 1;
        ri->colorAttachmentCount = 1;
        ri->pColorAttachments = att;
        rec().begin_rendering(*ri);
        s.garble();
    }

    // One to three scissored draws through a draw packet, in a pass begun
    // as an op or by the first packet, ended as an op or by a packet.
    void draw() {
        const bool query = !occ_used && rng.below(2) == 0;
        if (query) {
            rec().reset_query_pool(r.occ_pool, 0, 1);
            occ_used = true;
        }
        const bool packet_begins = rng.below(2) == 0;
        if (!packet_begins) begin_pass();
        const std::uint32_t draws = 1 + rng.below(3);
        DrawBindingState bindings;
        bindings.use_layout(r.gfx_layout);
        DrawLibraryState previous;
        std::uint32_t last_buffer = 0, last_slot = 0;
        for (std::uint32_t i = 0; i < draws; ++i) {
            const std::uint32_t x0 = rng.below(kImage), y0 = rng.below(kImage);
            const std::uint32_t w = 1 + rng.below(kImage - x0), h = 1 + rng.below(kImage - y0);
            const bool repeat = i && rng.below(2) == 0;
            const std::uint32_t x = repeat ? last_buffer : rng.below(3), slot = repeat ? last_slot : rng.below(kSlots);
            last_buffer = x; last_slot = slot;
            DrawCmds c(true);
            if (i == 0 && packet_begins) {
                VkRenderingAttachmentInfo* att = s.take<VkRenderingAttachmentInfo>(1);
                att->sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                att->imageView = r.view;
                att->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                att->loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                att->storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                VkRenderingInfo* ri = s.take<VkRenderingInfo>(1);
                ri->sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                ri->renderArea = {{0, 0}, {kImage, kImage}};
                ri->layerCount = 1;
                ri->colorAttachmentCount = 1;
                ri->pColorAttachments = att;
                if (!c.begin_rendering(*ri)) rec().begin_rendering(*ri);
                s.garble();
            }
            c.bind_pipeline(r.gfx);
            if (g.has_gpl) {
                DrawLibraryState now;
                now.front = i & 1 ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
                const auto fields = (i ? draw_library_changes(previous, now) : kLibraryAll) & (kLibraryCull | kLibraryFront);
                if (fields) c.library_state(now, false, fields);
                previous = now;
            }
            VkViewport* vp = s.take<VkViewport>(1);
            *vp = VkViewport{0.0f, 0.0f, static_cast<float>(kImage), static_cast<float>(kImage), 0.0f, 1.0f};
            c.viewport(*vp);
            VkRect2D* sc = s.take<VkRect2D>(1);
            *sc = VkRect2D{{static_cast<std::int32_t>(x0), static_cast<std::int32_t>(y0)}, {w, h}};
            c.scissor(*sc);
            std::uint32_t* binding = s.take<std::uint32_t>(1);
            VkDescriptorBufferInfo* info = s.take<VkDescriptorBufferInfo>(1);
            *binding = 0;
            *info = {b(x), slot * kSlotWords * 4ull, kSlotWords * 4ull};
            if (bindings.push(binding, info, 1)) c.push_buffers(r.gfx_layout, binding, info, 1);
            s.garble();
            DrawCall call;
            call.kind = DrawCall::kDirect;
            call.count = 3;
            call.instances = 1;
            if (query && i == 0) {
                call.query_pool = r.occ_pool;
                call.query = 0;
            }
            c.draw(call);
            c.publish();
            const std::uint32_t v = m.buf[x][slot * kSlotWords];
            for (std::uint32_t yy = y0; yy < y0 + h; ++yy) std::fill_n(m.img.begin() + yy * kImage + x0, w, v);
        }
        switch (rng.below(3)) {
        case 0: rec().end_rendering(true); break;
        case 1: rec().end_rendering(false); break;
        default: {
            DrawCmds c(true);
            if (!c.end_rendering(true)) rec().end_rendering(true);
            c.publish();
        }
        }
    }

    void clear() {
        const std::uint32_t v = rng.u32();
        VkClearColorValue color{};
        color.uint32[0] = v;
        VkImageSubresourceRange* range = s.take<VkImageSubresourceRange>(1);
        *range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        rec().clear_color_image(r.image, VK_IMAGE_LAYOUT_GENERAL, color, 1, range);
        s.garble();
        std::fill(m.img.begin(), m.img.end(), v);
    }

    // Rectangles inside distinct 8 x 8 tiles of the image, each against a
    // 64-word slot of a buffer.
    VkBufferImageCopy tile_region(Shuffle& tiles, std::uint32_t slot, std::uint32_t& x0, std::uint32_t& y0, std::uint32_t& w, std::uint32_t& h) {
        const std::uint32_t t = tiles.take();
        w = 1 + rng.below(8);
        h = 1 + rng.below(8);
        x0 = (t % 8) * 8 + rng.below(9 - w);
        y0 = (t / 8) * 8 + rng.below(9 - h);
        VkBufferImageCopy c{};
        c.bufferOffset = slot * kSlotWords * 4ull;
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageOffset = {static_cast<std::int32_t>(x0), static_cast<std::int32_t>(y0), 0};
        c.imageExtent = {w, h, 1};
        return c;
    }

    void to_image() {
        const std::uint32_t x = rng.below(3), n = 1 + rng.below(8);
        Shuffle tiles(rng, 64);
        VkBufferImageCopy* regions = s.take<VkBufferImageCopy>(n);
        for (std::uint32_t k = 0; k < n; ++k) {
            std::uint32_t x0, y0, w, h;
            const std::uint32_t slot = rng.below(kSlots);
            regions[k] = tile_region(tiles, slot, x0, y0, w, h);
            for (std::uint32_t j = 0; j < h; ++j) {
                for (std::uint32_t i = 0; i < w; ++i) m.img[(y0 + j) * kImage + x0 + i] = m.buf[x][slot * kSlotWords + j * w + i];
            }
        }
        rec().copy_buffer_to_image(b(x), r.image, VK_IMAGE_LAYOUT_GENERAL, n, regions);
        s.garble();
    }

    void to_buffer() {
        const std::uint32_t x = rng.below(3), n = 1 + rng.below(8);
        Shuffle tiles(rng, 64), slots(rng, kSlots);
        VkBufferImageCopy* regions = s.take<VkBufferImageCopy>(n);
        for (std::uint32_t k = 0; k < n; ++k) {
            std::uint32_t x0, y0, w, h;
            const std::uint32_t slot = slots.take();
            regions[k] = tile_region(tiles, slot, x0, y0, w, h);
            for (std::uint32_t j = 0; j < h; ++j) {
                for (std::uint32_t i = 0; i < w; ++i) m.buf[x][slot * kSlotWords + j * w + i] = m.img[(y0 + j) * kImage + x0 + i];
            }
        }
        rec().copy_image_to_buffer(r.image, VK_IMAGE_LAYOUT_GENERAL, b(x), n, regions);
        s.garble();
    }

    void timestamp() {
        if (ts_used >= kTimestamps) return;
        rec().write_timestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, r.ts_pool, ts_used++);
    }

    // The presenter's path: the game's submission first, then a command
    // buffer recorded on this thread that copies what it wrote, signalling a
    // semaphore a second one waits for with a fence.
    void foreign() {
        submit_locked();
        ++st.submits;
        if (foreign_used >= kWords / 256) return;
        const std::uint32_t src = rng.below(kWords - 256 + 1), dst = foreign_used++ * 256;
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(r.fcmd[0], &bi);
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(r.fcmd[0], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        const VkBufferCopy region{src * 4ull, dst * 4ull, 256 * 4ull};
        vkCmdCopyBuffer(r.fcmd[0], b(0), b(kF), 1, &region);
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(r.fcmd[0], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        vkEndCommandBuffer(r.fcmd[0]);
        vkBeginCommandBuffer(r.fcmd[1], &bi);
        vkEndCommandBuffer(r.fcmd[1]);
        const std::uint64_t t1 = host_gpu_submit_presenter(r.fcmd[0], nullptr, 0, r.fsem, nullptr);
        const std::uint64_t t2 =
            t1 ? host_gpu_submit_presenter(r.fcmd[1], r.fsem, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, nullptr, r.ffence) : 0;
        if (t2) {
            host_gpu_wait_submitted(t2);
        } else {
            // No submission thread: the presenter submits itself, behind
            // everything the queue's lock drains.
            const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            VkSubmitInfo si[2] = {};
            si[0].sType = si[1].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si[0].commandBufferCount = si[1].commandBufferCount = 1;
            si[0].pCommandBuffers = &r.fcmd[0];
            si[0].signalSemaphoreCount = 1;
            si[0].pSignalSemaphores = &r.fsem;
            si[1].pCommandBuffers = &r.fcmd[1];
            si[1].waitSemaphoreCount = 1;
            si[1].pWaitSemaphores = &r.fsem;
            si[1].pWaitDstStageMask = &stage;
            host_gpu_queue_lock();
            vkQueueSubmit(g.queue, 2, si, r.ffence);
            host_gpu_queue_unlock();
        }
        if (vkWaitForFences(g.device, 1, &r.ffence, VK_TRUE, 10000000000ull) != VK_SUCCESS) {
            host_log("stream selftest: the presenter's submission never finished");
            failed = true;
            return;
        }
        vkResetFences(g.device, 1, &r.ffence);
        vkResetCommandPool(g.device, r.fpool, 0);
        std::copy_n(m.buf[0].begin() + src, 256, m.buf[kF].begin() + dst);
        ++st.foreign;
    }

    // Used, then destroyed through the deferred list while the commands
    // that use it may not have been replayed yet (L1).
    void temp_buffer() {
        DevBuffer t;
        if (!create_dev_buffer(t, 16384, false)) return;
        const std::uint32_t x = rng.below(3), d = rng.below(kWords - 4096 + 1), v = rng.u32();
        rec().fill_buffer(t.buffer, 0, 16384, v);
        full_barrier(s);
        VkBufferCopy* region = s.take<VkBufferCopy>(1);
        *region = VkBufferCopy{0, d * 4ull, 16384};
        rec().copy_buffer(t.buffer, b(x), 1, region);
        s.garble();
        defer_destroy(t);
        std::fill_n(m.buf[x].begin() + d, 4096, v);
    }

    void temp_image() {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        if (!make_image(image, mem, 16, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return;
        const std::uint32_t x = rng.below(3), d = rng.below(kWords - 256 + 1), v = rng.u32();
        image_to_general(s, image);
        VkClearColorValue color{};
        color.uint32[0] = v;
        VkImageSubresourceRange* range = s.take<VkImageSubresourceRange>(1);
        *range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        rec().clear_color_image(image, VK_IMAGE_LAYOUT_GENERAL, color, 1, range);
        s.garble();
        full_barrier(s);
        VkBufferImageCopy* region = s.take<VkBufferImageCopy>(1);
        region->bufferOffset = d * 4ull;
        region->imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region->imageExtent = {16, 16, 1};
        rec().copy_image_to_buffer(image, VK_IMAGE_LAYOUT_GENERAL, b(x), 1, region);
        s.garble();
        defer_destroy_image(image, mem);
        std::fill_n(m.buf[x].begin() + d, 256, v);
    }

    void step() {
        begin_recording_locked();
        // Weights: the stream's common ops most, submissions and the
        // presenter's path now and then.
        const std::uint32_t pick = rng.below(100);
        if (pick < 14) {
            fill();
        } else if (pick < 28) {
            copy();
        } else if (pick < 36) {
            packet_copies();
        } else if (pick < 46) {
            staged();
        } else if (pick < 56) {
            compute();
        } else if (pick < 66) {
            if (r.draws) draw();
        } else if (pick < 70) {
            clear();
        } else if (pick < 76) {
            to_image();
        } else if (pick < 82) {
            to_buffer();
        } else if (pick < 86) {
            if (r.timestamps) timestamp();
        } else if (pick < 91) {
            submit_locked();
            ++st.submits;
            return;
        } else if (pick < 93) {
            flush_locked();
            ++st.flushes;
            return;
        } else if (pick < 95) {
            foreign();
            return;
        } else if (pick < 98) {
            temp_buffer();
        } else {
            temp_image();
        }
        // Each command's writes before the next one's reads and writes, so
        // the result does not depend on how the GPU overlaps them.
        full_barrier(s);
    }
};

std::uint64_t compare(const Resources& r, const Model& m, const char* way, int round, std::uint64_t seed) {
    const auto* rb = static_cast<const std::uint32_t*>(r.readback.map);
    std::uint64_t diffs = 0;
    static const char* const names[kBuffers] = {"A", "B", "C", "F"};
    for (std::uint32_t k = 0; k <= kBuffers; ++k) {
        const std::uint32_t* got = rb + static_cast<std::size_t>(k) * kWords;
        const std::uint32_t* want = k < kBuffers ? m.buf[k].data() : m.img.data();
        const std::uint32_t n = k < kBuffers ? kWords : kImage * kImage;
        std::uint64_t here = 0;
        std::uint32_t first = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            if (got[i] != want[i]) {
                if (!here) first = i;
                ++here;
            }
        }
        if (here) {
            host_log("stream selftest: round %d (seed 0x%llx), %s: %s differs in %llu words, the first at %u: 0x%08x, the model 0x%08x", round,
                     static_cast<unsigned long long>(seed), way, k < kBuffers ? names[k] : "the image", static_cast<unsigned long long>(here),
                     first, got[first], want[first]);
        }
        diffs += here;
    }
    return diffs;
}

// Exercise the production ordinary depth snapshot route in every recorder mode.
// Separate allocations establish the direct image copy's non-overlap guard.
bool ordinary_depth_snapshot(const Way& way, VkFormat format) {
    constexpr std::uint32_t pixels = kImage * kImage;
    constexpr VkDeviceSize bytes = pixels * 4ull;
    RtImage source, destination;
    DevBuffer upload, readback;
    const auto cleanup = [&] {
        flush_locked();
        for (const auto* image : {&source, &destination}) {
            if (image->image) vkDestroyImage(g.device, image->image, nullptr);
            if (image->memory.memory) vkFreeMemory(g.device, image->memory.memory, nullptr);
        }
        for (const auto* buffer : {&upload, &readback}) {
            if (buffer->map) vkUnmapMemory(g.device, buffer->memory);
            if (buffer->buffer) vkDestroyBuffer(g.device, buffer->buffer, nullptr);
            if (buffer->memory) vkFreeMemory(g.device, buffer->memory, nullptr);
        }
    };
    if (!stream_test_set_mode(way.mode, way.submit)) return false;
    begin_recording_locked();
    const auto setup_image = [&](RtImage& image, bool depth, VkFormat image_format) {
        image.format = image_format;
        image.width = image.height = kImage;
        image.depth = depth;
        image.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        if (!make_image(image.image, image.memory.memory, kImage, image.usage, image_format, &image.memory.size)) return false;
        const VkImageAspectFlags aspects = depth ? VK_IMAGE_ASPECT_DEPTH_BIT |
            (image_format == VK_FORMAT_D32_SFLOAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : 0) : VK_IMAGE_ASPECT_COLOR_BIT;
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.image;
        barrier.subresourceRange = {aspects, 0, 1, 0, 1};
        rec().pipeline_barrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &barrier);
        image.initialised = true;
        return true;
    };
    bool ok = setup_image(source, true, format) && setup_image(destination, false, VK_FORMAT_R32_SFLOAT) &&
              create_dev_buffer(upload, bytes, true, true) && create_dev_buffer(readback, bytes, true, true);
    if (!ok) { cleanup(); return false; }
    const char* direct_env = std::getenv("BBHOST_DEPTH_DIRECT_COPY");
    const bool direct_enabled = g.maintenance8_enabled && !(direct_env && direct_env[0] == '0');
    unsigned direct = 0, buffer = 0;
    Scratch scratch;
    for (unsigned iteration = 0; iteration < 4; ++iteration) {
        begin_recording_locked();
        auto* bits = static_cast<std::uint32_t*>(upload.map);
        for (unsigned p = 0; p < pixels; ++p) {
            const float value = float((p * 7919 + iteration * 3571) % 16777216) / 16777216.0f;
            std::memcpy(bits + p, &value, sizeof(value));
        }
        bits[0] = 0; bits[1] = 0x3f800000; bits[2] = 1; bits[3] = 0x3eaaaaab;
        const std::vector<std::uint32_t> expected(bits, bits + pixels);
        write_combine_fence();
        full_barrier(scratch);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        region.imageExtent = {kImage, kImage, 1};
        rec().copy_buffer_to_image(upload.buffer, source.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        full_barrier(scratch);
        // Unknown usage must retain the safe buffer path even with maintenance8.
        const auto usage = source.usage;
        if (iteration == 3) source.usage = 0;
        RenderCopyRoute route{};
        ok = render_depth_snapshot_selftest_locked(source, destination, route) && ok;
        source.usage = usage;
        const auto expected_route = direct_enabled && iteration != 3 ? RenderCopyRoute::DepthDirect : RenderCopyRoute::DepthBuffer;
        ok = route == expected_route && ok;
        direct += route == RenderCopyRoute::DepthDirect;
        buffer += route == RenderCopyRoute::DepthBuffer;
        full_barrier(scratch);
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        rec().copy_image_to_buffer(destination.image, VK_IMAGE_LAYOUT_GENERAL, readback.buffer, 1, &region);
        flush_locked();
        ok = std::memcmp(expected.data(), readback.map, bytes) == 0 && ok;
        begin_recording_locked();
        full_barrier(scratch);
        const VkClearDepthStencilValue clear{0.75f, 29};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        rec().clear_depth_stencil_image(source.image, VK_IMAGE_LAYOUT_GENERAL, clear, 1, &range);
        full_barrier(scratch);
        rec().copy_image_to_buffer(destination.image, VK_IMAGE_LAYOUT_GENERAL, readback.buffer, 1, &region);
        flush_locked();
        ok = std::memcmp(expected.data(), readback.map, bytes) == 0 && ok;
    }
    host_log("stream selftest: ordinary depth snapshot format %u, %s: %u direct, %u buffer, exact bits / overwrite / immutable %s",
             format, way.name, direct, buffer, ok ? "pass" : "FAIL");
    cleanup();
    return ok;
}

// One way of running a round's sequence; its differences from the model.
bool run_way(Resources& r, const Way& way, std::uint64_t seed, int round, RoundStats& st) {
    if (!stream_test_set_mode(way.mode, way.submit)) {
        host_log("stream selftest: could not switch to %s (something open)", way.name);
        return false;
    }
    vkResetDescriptorPool(g.device, r.pool, 0);
    Model m;
    m.reset();
    Scratch s;
    Arm arm(r, m, s, seed, st);
    begin_recording_locked();
    // A known start: zeroes, the image in the general layout, the queries reset.
    image_to_general(s, r.image);
    for (std::uint32_t k = 0; k < kBuffers; ++k) rec().fill_buffer(r.buf[k].buffer, 0, VK_WHOLE_SIZE, 0);
    VkClearColorValue zero{};
    VkImageSubresourceRange* range = s.take<VkImageSubresourceRange>(1);
    *range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    rec().clear_color_image(r.image, VK_IMAGE_LAYOUT_GENERAL, zero, 1, range);
    s.garble();
    if (r.timestamps) rec().reset_query_pool(r.ts_pool, 0, kTimestamps);
    full_barrier(s);
    const std::uint32_t steps = 150 + arm.rng.below(250);
    for (std::uint32_t i = 0; i < steps && !arm.failed; ++i) {
        arm.step();
        ++st.steps;
    }
    // Read everything back through the stream itself.
    begin_recording_locked();
    full_barrier(s);
    for (std::uint32_t k = 0; k < kBuffers; ++k) {
        VkBufferCopy* region = s.take<VkBufferCopy>(1);
        *region = VkBufferCopy{0, k * kWords * 4ull, kWords * 4ull};
        rec().copy_buffer(r.buf[k].buffer, r.readback.buffer, 1, region);
        s.garble();
    }
    VkBufferImageCopy* region = s.take<VkBufferImageCopy>(1);
    region->bufferOffset = kBuffers * kWords * 4ull;
    region->imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region->imageExtent = {kImage, kImage, 1};
    rec().copy_image_to_buffer(r.image, VK_IMAGE_LAYOUT_GENERAL, r.readback.buffer, 1, region);
    s.garble();
    flush_locked();
    ++st.flushes;
    if (arm.failed) {
        ++st.differences;
        return true;
    }
    st.differences += compare(r, m, way.name, round, seed);
    if (arm.occ_used) {
        std::uint64_t q[2] = {};
        const VkResult qr = vkGetQueryPoolResults(g.device, r.occ_pool, 0, 1, sizeof(q), q, sizeof(q),
                                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (qr != VK_SUCCESS || !q[1]) {
            ++st.occ_unresolved;  // AMD resolves none headless (KyoPS4x #220): noted, not a difference
        } else if (!q[0]) {
            host_log("stream selftest: round %d (seed 0x%llx), %s: the occlusion query around a draw counted 0 samples", round,
                     static_cast<unsigned long long>(seed), way.name);
            ++st.differences;
        }
    }
    if (arm.ts_used) {
        std::vector<std::uint64_t> ts(arm.ts_used * 2ull);
        vkGetQueryPoolResults(g.device, r.ts_pool, 0, arm.ts_used, ts.size() * 8, ts.data(), 16,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        for (std::uint32_t k = 0; k < arm.ts_used; ++k) {
            if (!ts[k * 2 + 1]) {
                ++st.ts_unavailable;
            } else if (k && ts[k * 2 + 1 - 2] && ts[k * 2] < ts[k * 2 - 2]) {
                ++st.ts_backwards;
            }
        }
    }
    return true;
}

// ---- the cost table ----

struct Cost {
    double ns[3] = {};  // in place; in place behind the recorder (BBHOST_STREAM_SUBMIT=0); streamed
};

template <class F> double time_ns(std::uint32_t k, F f) {
    const std::uint64_t t0 = host_clock_monotonic_ns();
    for (std::uint32_t i = 0; i < k; ++i) f(i);
    return static_cast<double>(host_clock_monotonic_ns() - t0) / k;
}

std::string benchmark(Resources& r) {
    Cost fill, copy, barrier, ts, pass, dispatch, draw, endbegin;
    double recorder_eb_us = 0.0, sfence = 0.0;
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    VkRenderingAttachmentInfo att{};
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView = r.view;
    att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {kImage, kImage}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;
    for (int way = 0; way < 3; ++way) {
        // In place: no recorder, every call to the driver at once. Behind the
        // recorder: the draws in packets, the rest in place after a drain (the
        // stream before kBegin/kEndSubmit). Streamed: appended, bb-record
        // replays them.
        stream_test_set_mode(way ? 2 : 0, way == 2);
        vkResetDescriptorPool(g.device, r.pool, 0);
        constexpr std::uint32_t K = 2000;
        begin_recording_locked();
        fill.ns[way] = time_ns(K, [&](std::uint32_t i) { rec().fill_buffer(r.buf[0].buffer, (i % 1024) * 64ull, 64, i); });
        const VkBufferCopy region{0, 4096, 256};
        copy.ns[way] = time_ns(K, [&](std::uint32_t) { rec().copy_buffer(r.buf[1].buffer, r.buf[2].buffer, 1, &region); });
        barrier.ns[way] = time_ns(K, [&](std::uint32_t) {
            rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        });
        if (r.timestamps) {
            ts.ns[way] = time_ns(K, [&](std::uint32_t i) {
                if (i % kTimestamps == 0) rec().reset_query_pool(r.ts_pool, 0, kTimestamps);
                rec().write_timestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, r.ts_pool, i % kTimestamps);
            });
        }
        pass.ns[way] = time_ns(K, [&](std::uint32_t) {
            rec().begin_rendering(ri);
            rec().end_rendering(false);
        });
        flush_locked();
        begin_recording_locked();
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = r.pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &r.comp_set_layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        vkAllocateDescriptorSets(g.device, &ai, &set);
        VkDescriptorBufferInfo infos[2] = {{r.buf[1].buffer, 0, VK_WHOLE_SIZE}, {r.buf[2].buffer, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet writes[2] = {};
        for (std::uint32_t k = 0; k < 2; ++k) {
            writes[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[k].dstSet = set;
            writes[k].dstBinding = k;
            writes[k].descriptorCount = 1;
            writes[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[k].pBufferInfo = &infos[k];
        }
        vkUpdateDescriptorSets(g.device, 2, writes, 0, nullptr);
        const std::uint32_t push[5] = {0, 0, 64, 1, 0};
        dispatch.ns[way] = time_ns(K, [&](std::uint32_t) {
            rec().bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, r.comp);
            rec().bind_sets(VK_PIPELINE_BIND_POINT_COMPUTE, r.comp_layout, 0, 1, &set);
            rec().push_constants(r.comp_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, push);
            rec().dispatch(1, 1, 1);
        });
        if (r.draws) {
            rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            rec().begin_rendering(ri);
            const VkViewport vp{0.0f, 0.0f, static_cast<float>(kImage), static_cast<float>(kImage), 0.0f, 1.0f};
            const VkRect2D sc{{0, 0}, {8, 8}};
            const std::uint32_t binding = 0;
            const VkDescriptorBufferInfo info{r.buf[0].buffer, 0, 256};
            DrawCall call;
            call.count = 3;
            call.instances = 1;
            draw.ns[way] = time_ns(1000, [&](std::uint32_t) {
                DrawCmds c(true);
                c.bind_pipeline(r.gfx);
                if (g.has_gpl) c.library_state(DrawLibraryState{}, false, kLibraryCull | kLibraryFront);
                c.viewport(vp);
                c.scissor(sc);
                c.push_buffers(r.gfx_layout, &binding, &info, 1);
                c.draw(call);
                c.publish();
            });
            rec().end_rendering(true);
        }
        flush_locked();
        const std::uint64_t eb0 = g_stream_stats.endbegin_ns.load(), ebn0 = g_stream_stats.endbegin_n.load();
        endbegin.ns[way] = time_ns(200, [&](std::uint32_t) {
            begin_recording_locked();
            submit_locked();
        });
        flush_locked();
        if (way == 2) {
            const std::uint64_t ebn = g_stream_stats.endbegin_n.load() - ebn0;
            recorder_eb_us = ebn ? static_cast<double>(g_stream_stats.endbegin_ns.load() - eb0) / 1000.0 / static_cast<double>(ebn) : 0.0;
        }
    }
    sfence = time_ns(100000, [](std::uint32_t) { write_combine_fence(); });
    const auto t = [](const Cost& c) {
        char s[48];
        std::snprintf(s, sizeof(s), "%.0f/%.0f/%.0f", c.ns[0], c.ns[1], c.ns[2]);
        return std::string(s);
    };
    char buf[900];
    std::snprintf(buf, sizeof(buf),
                  "stream selftest: producer ns per op, in place / in place behind the recorder / streamed: fill %s, copy %s, barrier %s, "
                  "timestamp %s, pass (begin+end) %s, dispatch (bind, set, push, dispatch) %s, draw packet %s, begin+submit %s (the "
                  "recorder's end+begin %.1f us), sfence %.1f",
                  t(fill).c_str(), t(copy).c_str(), t(barrier).c_str(), t(ts).c_str(), t(pass).c_str(), t(dispatch).c_str(), t(draw).c_str(),
                  t(endbegin).c_str(), recorder_eb_us, sfence);
    return buf;
}

}  // namespace
}  // namespace gpu

int host_gpu_stream_selftest(int rounds) {
    using namespace gpu;
    if (rounds <= 0) rounds = 16;
    host_log("stream selftest: %d rounds, each run in place, with draws on the recorder, inline and threaded", rounds);
    if (!host_gpu_init(nullptr, 0, false)) {
        host_log("stream selftest: no Vulkan device");
        return 2;
    }
    std::lock_guard<GpuMutex> lock(g.mu);
    const std::uint64_t validation0 = g_validation_messages.load();
    if (!texture_upload_plan_selftest()) return 1;
    for (const Way& way : kWays)
        for (const auto format : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT})
            if (!ordinary_depth_snapshot(way, format)) return 1;
    Resources r;
    if (!setup(r)) {
        host_log("stream selftest: setting up its buffers, image and pipelines failed");
        return 2;
    }
    bool proof = true;
    const DrawLibraryState original;
    const auto changed = [&](auto alter, std::uint32_t field) {
        auto now = original;
        alter(now);
        proof = draw_library_changes(original, now) == field && proof;
    };
    changed([](auto& s) { s.cull = VK_CULL_MODE_BACK_BIT; }, kLibraryCull);
    changed([](auto& s) { s.front = VK_FRONT_FACE_CLOCKWISE; }, kLibraryFront);
    changed([](auto& s) { s.depth_test = VK_TRUE; }, kLibraryDepthTest);
    changed([](auto& s) { s.depth_write = VK_TRUE; }, kLibraryDepthWrite);
    changed([](auto& s) { s.depth_compare = VK_COMPARE_OP_LESS; }, kLibraryDepthCompare);
    changed([](auto& s) { s.bounds_test = VK_TRUE; }, kLibraryBoundsTest);
    changed([](auto& s) { s.stencil_test = VK_TRUE; }, kLibraryStencilTest);
    changed([](auto& s) { s.front_ops.passOp = VK_STENCIL_OP_REPLACE; }, kLibraryFrontOps);
    changed([](auto& s) { s.back_ops.compareOp = VK_COMPARE_OP_ALWAYS; }, kLibraryBackOps);
    changed([](auto& s) { s.bias_constant = -0.0f; }, kLibraryBias);
    changed([](auto& s) { s.bias_clamp = 1.0f; }, kLibraryBias);
    changed([](auto& s) { s.bias_slope = 1.0f; }, kLibraryBias);
    changed([](auto& s) { s.depth_clamp = VK_TRUE; }, kLibraryDepthClamp);
    changed([](auto& s) { s.front_ops.reference = 57; s.back_ops.writeMask = 255; }, 0);
    proof = !draw_library_changes(original, original) && proof;
    DrawBindingState state;
    const auto check = [&](bool actual, bool expected) { proof = actual == expected && proof; };
    state.use_layout(r.gfx_layout);
    std::uint32_t bindings[2] = {0, 1};
    VkDescriptorBufferInfo infos[2] = {{r.buf[0].buffer, 0, 128}, {r.buf[1].buffer, 128, 256}};
    check(state.push(bindings, infos, 2), true);
    check(state.push(bindings, infos, 2), false);
    ++infos[1].offset; check(state.push(bindings, infos, 2), true);
    ++infos[1].range; check(state.push(bindings, infos, 2), true);
    ++bindings[1]; check(state.push(bindings, infos, 2), true);
    infos[1].buffer = r.buf[2].buffer; check(state.push(bindings, infos, 2), true);
    state.use_layout(r.comp_layout); check(state.push(bindings, infos, 2), true);
    check(state.push(bindings, infos, 1), true);
    check(state.push(bindings, infos, 1), false);
    const auto global = (VkDescriptorSet)(std::uintptr_t)1;
    check(state.bind_global(global), true); check(state.bind_global(global), false);
    state.use_layout(r.gfx_layout); check(state.bind_global(global), true);
    VkBuffer vb[2] = {r.buf[0].buffer, r.buf[1].buffer}; VkDeviceSize offsets[2] = {0, 256};
    check(state.vertex(vb, offsets, 2), true); check(state.vertex(vb, offsets, 2), false);
    ++offsets[1]; check(state.vertex(vb, offsets, 2), true);
    std::swap(vb[0], vb[1]); check(state.vertex(vb, offsets, 2), true);
    check(state.vertex(vb, offsets, 1), true);
    state = {}; state.use_layout(r.gfx_layout);
    check(state.push(bindings, infos, 1), true); check(state.vertex(vb, offsets, 1), true);
    check(state.bind_global(global), true);
    host_log("stream selftest: fieldwise state / exact bindings / layouts / reset proof %s", proof ? "pass" : "FAIL");
    if (!proof) return 1;
    std::uint64_t seed0 = host_clock_monotonic_ns();
    if (const char* e = std::getenv("BBHOST_STREAM_SELFTEST_SEED"); e && *e) seed0 = std::strtoull(e, nullptr, 0);
    RoundStats total;
    std::uint64_t ops = 0;
    int failed_rounds = 0;
    for (int round = 1; round <= rounds; ++round) {
        const std::uint64_t seed = Rng{seed0 + static_cast<std::uint64_t>(round)}.next();
        RoundStats st;
        StreamTestCounts c0{}, c1{};
        bool ok = true;
        for (const Way& way : kWays) {
            RoundStats ws;
            const bool threaded = way.mode == 2 && way.submit;
            if (threaded) c0 = stream_test_counts();
            ok = run_way(r, way, seed, round, ws) && ok;
            if (threaded) {
                c1 = stream_test_counts();
                st.steps = ws.steps;
                st.submits = ws.submits;
                st.flushes = ws.flushes;
                st.foreign = ws.foreign;
            }
            st.differences += ws.differences;
            st.occ_unresolved += ws.occ_unresolved;
            st.ts_unavailable += ws.ts_unavailable;
            st.ts_backwards += ws.ts_backwards;
        }
        ok = ok && st.differences == 0 && st.ts_unavailable == 0;
        failed_rounds += ok ? 0 : 1;
        ops += c1.ops - c0.ops;
        host_log("stream selftest: round %d/%d seed 0x%llx %s (%llu steps; threaded: %llu ops in %llu blocks, %llu draws, %llu submits, "
                 "%llu flushes, %llu presenter submissions, %llu drains)",
                 round, rounds, static_cast<unsigned long long>(seed), ok ? "ok" : "FAILED", static_cast<unsigned long long>(st.steps),
                 static_cast<unsigned long long>(c1.ops - c0.ops), static_cast<unsigned long long>(c1.blocks - c0.blocks),
                 static_cast<unsigned long long>(c1.draws - c0.draws), static_cast<unsigned long long>(c1.submits - c0.submits),
                 static_cast<unsigned long long>(st.flushes), static_cast<unsigned long long>(st.foreign),
                 static_cast<unsigned long long>(c1.drains - c0.drains));
        total.differences += st.differences;
        total.occ_unresolved += st.occ_unresolved;
        total.ts_unavailable += st.ts_unavailable;
        total.ts_backwards += st.ts_backwards;
        if (!g.ok) {
            host_log("stream selftest: the device was lost in round %d; stopping", round);
            failed_rounds += rounds - round;
            break;
        }
    }
    if (g.ok && !failed_rounds) host_log("%s", benchmark(r).c_str());
    const std::uint64_t validation = g_validation_messages.load() - validation0;
    host_log("stream selftest: %d rounds, %llu ops threaded, %llu differences (in place, draws on the recorder, inline, threaded); "
             "validation messages %llu; occlusion queries unresolved %llu; timestamps unavailable %llu, going backwards %llu",
             rounds, static_cast<unsigned long long>(ops), static_cast<unsigned long long>(total.differences),
             static_cast<unsigned long long>(validation), static_cast<unsigned long long>(total.occ_unresolved),
             static_cast<unsigned long long>(total.ts_unavailable), static_cast<unsigned long long>(total.ts_backwards));
    host_log("%s", recorder_report().c_str());
    const bool pass = g.ok && !failed_rounds && !validation;
    host_log("stream selftest: %s", pass ? "pass" : "FAIL");
    return pass ? 0 : 1;
}
