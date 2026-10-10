// The host overlay's Vulkan side. One pipeline, one host-visible vertex
// buffer, triangles with alpha blending, drawn over the presented frame with
// dynamic rendering so there is no render pass or framebuffer to keep in step
// with the swapchain.

#include "host/overlay.h"
#include "host/gpu.h"

#include "host/overlay_font.h"
#include "host/overlay_spv.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

struct Vertex {
    float x, y;
    float u, v;
    float r, g, b, a;
};

// Two lists: the window's pump builds the next one in g_verts from
// host_overlay_reset() on, and host_overlay_commit() hands it to the presenter
// as g_shown. With one list a present that landed between the reset and the
// rebuild found it empty, and the FPS counter blinked out for that frame.
// g_shown_count lets a present with nothing to draw find that out without the
// lock: what a hidden overlay costs the presenter is one atomic load.
std::mutex g_mu;
std::vector<Vertex> g_verts;  // being built by the pump (under g_mu)
std::vector<Vertex> g_shown;  // what presents draw (under g_mu)
float g_build_w = 0.0f, g_build_h = 0.0f;
float g_shown_w = 0.0f, g_shown_h = 0.0f;
std::atomic<std::size_t> g_shown_count{0};

// Presenter-owned, created once for the swapchain's format.
VkDevice g_device = VK_NULL_HANDLE;
VkPipelineLayout g_layout = VK_NULL_HANDLE;
VkPipeline g_pipeline = VK_NULL_HANDLE;
VkBuffer g_vbo = VK_NULL_HANDLE;
VkImage g_atlas = VK_NULL_HANDLE;
VkDeviceMemory g_atlas_mem = VK_NULL_HANDLE;
VkImageView g_atlas_view = VK_NULL_HANDLE;
VkSampler g_sampler = VK_NULL_HANDLE;
VkDescriptorSetLayout g_dsl = VK_NULL_HANDLE;
VkDescriptorPool g_dpool = VK_NULL_HANDLE;
VkDescriptorSet g_dset = VK_NULL_HANDLE;
VkDeviceMemory g_vbo_mem = VK_NULL_HANDLE;
void* g_vbo_map = nullptr;
VkDeviceSize g_vbo_bytes = 0;
VkFormat g_format = VK_FORMAT_UNDEFINED;

constexpr VkDeviceSize kVboBytes = 256u * 1024u;  // ~5.4k triangles a frame

// The atlas texel that is fully lit: a plain rectangle points every corner at
// it, so one pipeline draws rectangles and glyphs alike.
constexpr float kSolidU = (overlay_font::kSolidX + 0.5f) / overlay_font::kAtlasW;
constexpr float kSolidV = (overlay_font::kSolidY + 0.5f) / overlay_font::kAtlasH;

void push(float x, float y, std::uint32_t rgba, float u = kSolidU, float vv = kSolidV) {
    Vertex v;
    v.x = x;
    v.y = y;
    v.u = u;
    v.v = vv;
    v.r = static_cast<float>((rgba >> 24) & 0xff) / 255.0f;
    v.g = static_cast<float>((rgba >> 16) & 0xff) / 255.0f;
    v.b = static_cast<float>((rgba >> 8) & 0xff) / 255.0f;
    v.a = static_cast<float>(rgba & 0xff) / 255.0f;
    g_verts.push_back(v);
}

std::uint32_t memory_type(VkPhysicalDevice phys, std::uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    return UINT32_MAX;
}

VkShaderModule make_module(VkDevice device, const std::uint32_t* words, std::size_t count) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = count * 4;
    ci.pCode = words;
    VkShaderModule m = VK_NULL_HANDLE;
    return vkCreateShaderModule(device, &ci, nullptr, &m) == VK_SUCCESS ? m : VK_NULL_HANDLE;
}

}  // namespace

void host_overlay_reset(float display_w, float display_h) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_verts.clear();
    g_build_w = display_w;
    g_build_h = display_h;
}

void host_overlay_commit() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_shown.swap(g_verts);
    g_shown_w = g_build_w;
    g_shown_h = g_build_h;
    g_shown_count.store(g_shown.size(), std::memory_order_release);
}

bool host_overlay_empty() { return g_shown_count.load(std::memory_order_acquire) == 0; }

void host_overlay_tri(float x0, float y0, float x1, float y1, float x2, float y2, std::uint32_t rgba) {
    std::lock_guard<std::mutex> lk(g_mu);
    push(x0, y0, rgba);
    push(x1, y1, rgba);
    push(x2, y2, rgba);
}

void host_overlay_rect(float x, float y, float w, float h, std::uint32_t rgba) {
    std::lock_guard<std::mutex> lk(g_mu);
    push(x, y, rgba);
    push(x + w, y, rgba);
    push(x, y + h, rgba);
    push(x + w, y, rgba);
    push(x + w, y + h, rgba);
    push(x, y + h, rgba);
}

void host_overlay_cursor(float x, float y) {
    // An arrow, tip at (x, y): the outline first and the fill over it, so the
    // pointer stays visible on a light menu and on a dark one.
    const float s = 22.0f;  // long enough to read at 1080p
    // The tip is (x, y) itself; these are the other two corners, as fractions.
    const struct { float dx, dy; } tail{0.32f, 0.86f}, edge{0.62f, 0.62f};
    const float o = 2.0f;
    host_overlay_tri(x - o, y - o, x + edge.dx * s + o, y + edge.dy * s + o, x + tail.dx * s, y + tail.dy * s + o, 0x000000c0u);
    host_overlay_tri(x, y, x + edge.dx * s, y + edge.dy * s, x + tail.dx * s, y + tail.dy * s, 0xffffffffu);
}

// Uploads the font atlas and builds the descriptor that samples it. One-shot
// command buffer: this runs once, when the swapchain's format is first known.
bool upload_atlas(VkDevice device, VkPhysicalDevice phys, VkQueue queue, std::uint32_t family) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8_UNORM;
    ici.extent = {overlay_font::kAtlasW, overlay_font::kAtlasH, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &ici, nullptr, &g_atlas) != VK_SUCCESS) return false;
    VkMemoryRequirements ireq{};
    vkGetImageMemoryRequirements(device, g_atlas, &ireq);
    VkMemoryAllocateInfo imai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    imai.allocationSize = ireq.size;
    imai.memoryTypeIndex = memory_type(phys, ireq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (imai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(device, &imai, nullptr, &g_atlas_mem) != VK_SUCCESS) return false;
    vkBindImageMemory(device, g_atlas, g_atlas_mem, 0);

    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    VkBufferCreateInfo sbci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    sbci.size = sizeof(overlay_font::kAtlas);
    sbci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(device, &sbci, nullptr, &staging) != VK_SUCCESS) return false;
    VkMemoryRequirements sreq{};
    vkGetBufferMemoryRequirements(device, staging, &sreq);
    VkMemoryAllocateInfo smai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    smai.allocationSize = sreq.size;
    smai.memoryTypeIndex = memory_type(phys, sreq.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (smai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(device, &smai, nullptr, &staging_mem) != VK_SUCCESS) return false;
    vkBindBufferMemory(device, staging, staging_mem, 0);
    void* map = nullptr;
    if (vkMapMemory(device, staging_mem, 0, sreq.size, 0, &map) != VK_SUCCESS) return false;
    std::memcpy(map, overlay_font::kAtlas, sizeof(overlay_font::kAtlas));
    vkUnmapMemory(device, staging_mem);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = family;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device, &cpci, nullptr, &pool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &cbai, &cmd);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = g_atlas;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {overlay_font::kAtlasW, overlay_font::kAtlasH, 1};
    vkCmdCopyBufferToImage(cmd, staging, g_atlas, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    host_gpu_queue_lock();
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    host_gpu_queue_unlock();
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, staging_mem, nullptr);

    VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ivci.image = g_atlas;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R8_UNORM;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(device, &ivci, nullptr, &g_atlas_view) != VK_SUCCESS) return false;
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(device, &sci, nullptr, &g_sampler) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(device, &dpci, nullptr, &g_dpool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = g_dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &g_dsl;
    if (vkAllocateDescriptorSets(device, &dsai, &g_dset) != VK_SUCCESS) return false;
    VkDescriptorImageInfo dii{g_sampler, g_atlas_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = g_dset;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    return true;
}

float host_overlay_text_width(float scale, const char* text) {
    if (!text) return 0.0f;
    std::size_t n = 0;
    for (const char* c = text; *c; ++c) ++n;
    return static_cast<float>(n) * overlay_font::kAdvanceW * scale;
}

float host_overlay_text(float x, float y, float scale, std::uint32_t rgba, const char* text) {
    if (!text) return 0.0f;
    // A character advances by the layout box but is drawn as the whole padded
    // cell, offset so the ink lands where the box says. The padding is why the
    // quad is bigger than the advance: without it linear filtering reaches into
    // the neighbouring cell and puts an underscore over every 'o'.
    const float aw = overlay_font::kAdvanceW * scale;
    const float cw = overlay_font::kCellW * scale, ch = overlay_font::kCellH * scale;
    const float ox = overlay_font::kOriginX * scale, oy = overlay_font::kOriginY * scale;
    float pen = x;
    std::lock_guard<std::mutex> lk(g_mu);
    for (const char* c = text; *c; ++c, pen += aw) {
        const unsigned ch_code = static_cast<unsigned char>(*c);
        if (ch_code < overlay_font::kFirst || ch_code > overlay_font::kLast) continue;  // space included, just blank
        const int idx = static_cast<int>(ch_code) - overlay_font::kFirst;
        const float u0 = static_cast<float>((idx % overlay_font::kCols) * overlay_font::kCellW) / overlay_font::kAtlasW;
        const float v0 = static_cast<float>((idx / overlay_font::kCols) * overlay_font::kCellH) / overlay_font::kAtlasH;
        const float u1 = u0 + static_cast<float>(overlay_font::kCellW) / overlay_font::kAtlasW;
        const float v1 = v0 + static_cast<float>(overlay_font::kCellH) / overlay_font::kAtlasH;
        const float gx = pen - ox, gy = y - oy;
        push(gx, gy, rgba, u0, v0);
        push(gx + cw, gy, rgba, u1, v0);
        push(gx, gy + ch, rgba, u0, v1);
        push(gx + cw, gy, rgba, u1, v0);
        push(gx + cw, gy + ch, rgba, u1, v1);
        push(gx, gy + ch, rgba, u0, v1);
    }
    return pen - x;
}

bool host_overlay_init(VkDevice device, VkPhysicalDevice phys, VkFormat colour_format, VkQueue queue, std::uint32_t family) {
    if (g_pipeline != VK_NULL_HANDLE && g_format == colour_format && g_device == device) return true;
    host_overlay_shutdown(g_device ? g_device : device);
    g_device = device;
    g_format = colour_format;

    VkDescriptorSetLayoutBinding dslb{};
    dslb.binding = 0;
    dslb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    dslb.descriptorCount = 1;
    dslb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 1;
    dslci.pBindings = &dslb;
    if (vkCreateDescriptorSetLayout(device, &dslci, nullptr, &g_dsl) != VK_SUCCESS) return false;

    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 2};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g_dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &g_layout) != VK_SUCCESS) return false;

    VkShaderModule vs = make_module(device, overlay_spv::kVert, sizeof(overlay_spv::kVert) / 4);
    VkShaderModule fs = make_module(device, overlay_spv::kFrag, sizeof(overlay_spv::kFrag) / 4);
    if (!vs || !fs) return false;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";

    VkVertexInputBindingDescription bind{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[3]{{0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
                                               {1, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 2},
                                               {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 4}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &bind;
    vi.vertexAttributeDescriptionCount = 3;
    vi.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 2;
    ds.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo prci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    prci.colorAttachmentCount = 1;
    prci.pColorAttachmentFormats = &g_format;

    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.pNext = &prci;
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &ds;
    gp.layout = g_layout;
    const VkResult r = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp, nullptr, &g_pipeline);
    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
    if (r != VK_SUCCESS) return false;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = kVboBytes;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bci, nullptr, &g_vbo) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, g_vbo, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type(phys, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(device, &mai, nullptr, &g_vbo_mem) != VK_SUCCESS) return false;
    vkBindBufferMemory(device, g_vbo, g_vbo_mem, 0);
    if (vkMapMemory(device, g_vbo_mem, 0, req.size, 0, &g_vbo_map) != VK_SUCCESS) return false;
    g_vbo_bytes = req.size;
    return upload_atlas(device, phys, queue, family);
}

void host_overlay_shutdown(VkDevice device) {
    if (!device) return;
    if (g_vbo_map) vkUnmapMemory(device, g_vbo_mem);
    if (g_vbo) vkDestroyBuffer(device, g_vbo, nullptr);
    if (g_vbo_mem) vkFreeMemory(device, g_vbo_mem, nullptr);
    if (g_dpool) vkDestroyDescriptorPool(device, g_dpool, nullptr);
    if (g_dsl) vkDestroyDescriptorSetLayout(device, g_dsl, nullptr);
    if (g_sampler) vkDestroySampler(device, g_sampler, nullptr);
    if (g_atlas_view) vkDestroyImageView(device, g_atlas_view, nullptr);
    if (g_atlas) vkDestroyImage(device, g_atlas, nullptr);
    if (g_atlas_mem) vkFreeMemory(device, g_atlas_mem, nullptr);
    if (g_pipeline) vkDestroyPipeline(device, g_pipeline, nullptr);
    if (g_layout) vkDestroyPipelineLayout(device, g_layout, nullptr);
    g_vbo_map = nullptr;
    g_vbo = VK_NULL_HANDLE;
    g_vbo_mem = VK_NULL_HANDLE;
    g_pipeline = VK_NULL_HANDLE;
    g_layout = VK_NULL_HANDLE;
    g_dpool = VK_NULL_HANDLE;
    g_dsl = VK_NULL_HANDLE;
    g_sampler = VK_NULL_HANDLE;
    g_atlas_view = VK_NULL_HANDLE;
    g_atlas = VK_NULL_HANDLE;
    g_atlas_mem = VK_NULL_HANDLE;
}

std::uint32_t host_overlay_record(VkCommandBuffer cmd, VkRect2D area, float display_w, float display_h) {
    if (host_overlay_empty() || !g_pipeline || !g_vbo_map || display_w <= 0.0f || display_h <= 0.0f) return 0;
    std::uint32_t count = 0;
    {
        // Straight into the vertex buffer: the presenter's fence says the GPU
        // is done with what the last frame drew from it.
        std::lock_guard<std::mutex> lk(g_mu);
        const VkDeviceSize bytes = g_shown.size() * sizeof(Vertex);
        if (g_shown.empty() || bytes > g_vbo_bytes) return 0;
        std::memcpy(g_vbo_map, g_shown.data(), bytes);
        count = static_cast<std::uint32_t>(g_shown.size());
        // A queued overlay retains its own coordinate space when a loading,
        // fallback or SR transition changes the picture before the next pump.
        if (g_shown_w > 0.0f && g_shown_h > 0.0f) {
            display_w = g_shown_w;
            display_h = g_shown_h;
        }
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_layout, 0, 1, &g_dset, 0, nullptr);
    VkViewport vp{static_cast<float>(area.offset.x), static_cast<float>(area.offset.y), static_cast<float>(area.extent.width),
                  static_cast<float>(area.extent.height), 0.0f, 1.0f};
    VkRect2D sc = area;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    const float inv[2] = {1.0f / display_w, 1.0f / display_h};
    vkCmdPushConstants(cmd, g_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(inv), inv);
    const VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &g_vbo, &off);
    vkCmdDraw(cmd, count, 1, 0, 0);
    return count;
}

bool host_overlay_draw(VkDevice device, VkCommandBuffer cmd, VkImage image, VkImageView view, VkExtent2D extent, VkRect2D area,
                       float display_w, float display_h) {
    (void)image;
    (void)device;
    if (host_overlay_empty() || !g_pipeline || !g_vbo_map || display_w <= 0.0f || display_h <= 0.0f) return false;
    VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    att.imageView = view;
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // The game's frame is already there; the overlay goes over it.
    att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;
    vkCmdBeginRendering(cmd, &ri);
    const bool drew = host_overlay_record(cmd, area, display_w, display_h) != 0;
    vkCmdEndRendering(cmd);
    return drew;
}
