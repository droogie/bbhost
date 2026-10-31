// BBHOST_COPY_ELIDE=compare: a copy left unmade (render.cpp) is also made at
// once, its destination's sampled bindings still bind the source, and at the
// end of the copy's time pending - made later, forgotten, or the images
// exchanged - the source and the destination are compared texel by texel on
// the GPU (shaders/imagecompare.comp). The source holds what the copy wrote
// for as long as nothing writes it; a difference is a write that reached it
// without the ledger seeing it, so a binding of the destination meanwhile may
// have read the wrong texels. The counts come back once the submission has
// run, by what ended the copy, the first differences logged.
//
// Phase 2's coverage canary: a draw proven to write all of a target whose
// pending copy it ends (render.cpp copy_elide_decide_locked) cannot show a
// wrong proof through the images it exchanges - in compare mode both hold the
// source's texels - so the image the draw writes is filled with the canary
// first, and after the draw the texels still equal to it are counted: compared
// with an image of the target's format and size that holds the canary alone,
// the texels that do not differ.
#include "host/gpu_internal.h"
#include "host/shaders/imagecompare.spv.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <deque>
#include <map>
#include <string>
#include <tuple>

namespace gpu {
namespace {

struct Push {
    std::uint32_t width, height, slot;
};

constexpr std::uint32_t kCompareSlots = 4096;
VkDescriptorSetLayout g_set_layout = VK_NULL_HANDLE;
VkPipelineLayout g_layout = VK_NULL_HANDLE;
VkPipeline g_pipeline = VK_NULL_HANDLE;
DevBuffer g_counts;  // kCompareSlots uints, host-visible
bool g_tried = false, g_ready = false;
std::uint32_t g_next = 0;

struct Pending {
    std::uint32_t slot;
    std::uint64_t serial;  // g.flushes of the recording the compare is in
    std::uint64_t a, b;
    std::uint32_t width, height;
    VkFormat format;
    const char* why;
    bool canary = false;  // a coverage canary's count: the texels left are width * height less those differing
};
std::deque<Pending> g_pending;  // oldest first
struct Tally {
    std::uint64_t compares = 0, differing = 0, texels = 0;
};
std::map<std::string, Tally> g_by_why;
std::uint64_t g_skipped_format = 0, g_skipped_full = 0, g_skipped_setup = 0, g_logged = 0;
// The canaries: checks counted, draws that left texels, the texels left; the canary images by format and size.
std::uint64_t g_canary_checks = 0, g_canary_bad = 0, g_canary_left = 0, g_canary_skipped = 0;
std::map<std::string, std::uint64_t> g_canary_by_who;
std::map<std::tuple<int, std::uint32_t, std::uint32_t>, RtImage> g_canary_images;
// 0x5a 0xa5 0x3c 0xc3 as UNORM8; any format stores it the same way in the target and in the canary image.
const VkClearColorValue kCanary = {{0x5a / 255.0f, 0xa5 / 255.0f, 0x3c / 255.0f, 0xc3 / 255.0f}};

bool make_locked() {
    const VkDescriptorSetLayoutBinding binds[4] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 4;
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
    smi.codeSize = sizeof(k_imagecompare_spv);
    smi.pCode = k_imagecompare_spv;
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
    if (!ok) return false;
    MemOwner own_mem("copy elision compares");
    return create_dev_buffer(g_counts, kCompareSlots * sizeof(std::uint32_t), true) && g_counts.map;
}

// A float format whose texels a texelFetch converts the same way for two images.
bool comparable(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32G32_SFLOAT:
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R5G6B5_UNORM_PACK16:
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
            return true;
        default:
            return false;
    }
}

// Compares whose submissions have run: their counts added up.
void harvest_locked() {
    while (!g_pending.empty() && g_pending.front().serial < g.completed_submits) {
        const Pending p = g_pending.front();
        g_pending.pop_front();
        const std::uint32_t n = static_cast<const std::uint32_t*>(g_counts.map)[p.slot];
        if (p.canary) {
            const std::uint64_t left = static_cast<std::uint64_t>(p.width) * p.height - std::min<std::uint64_t>(n, static_cast<std::uint64_t>(p.width) * p.height);
            ++g_canary_checks;
            if (left) {
                ++g_canary_bad;
                g_canary_left += left;
                ++g_canary_by_who[p.why];
                if (g_logged++ < 16) {
                    host_log("render: copy elision canary: draw %s left %llu of %ux%u texels of 0x%llx unwritten (format %d): its proof is wrong", p.why,
                             static_cast<unsigned long long>(left), p.width, p.height, static_cast<unsigned long long>(p.a), static_cast<int>(p.format));
                }
            }
            continue;
        }
        Tally& t = g_by_why[p.why];
        ++t.compares;
        if (n) {
            ++t.differing;
            t.texels += n;
            if (g_logged++ < 16) {
                host_log("render: copy elision compare: 0x%llx and 0x%llx (%ux%u format %d) differ in %u texels where the copy ended (%s): a "
                         "write reached the source without the ledger",
                         static_cast<unsigned long long>(p.a), static_cast<unsigned long long>(p.b), p.width, p.height, static_cast<int>(p.format),
                         n, p.why);
            }
        }
    }
}

}  // namespace

namespace {
bool record_compare_locked(RtImage& a, RtImage& b, const char* why, bool canary);
}  // namespace

void copy_compare_record_locked(RtImage& a, RtImage& b, const char* why) { record_compare_locked(a, b, why, false); }

namespace {
bool record_compare_locked(RtImage& a, RtImage& b, const char* why, bool canary) {
    if (!g_tried) {
        g_tried = true;
        g_ready = make_locked();
        host_log("render: copy elision compares %s", g_ready ? "ready" : "unavailable (pipeline or buffer creation failed)");
    }
    if (!g_ready) {
        ++g_skipped_setup;
        return false;
    }
    harvest_locked();
    if (!comparable(a.format) || a.format != b.format || a.width != b.width || a.height != b.height || !a.view || !b.view) {
        ++g_skipped_format;
        return false;
    }
    if (g_pending.size() >= kCompareSlots) {
        ++g_skipped_full;
        return false;
    }
    const VkDescriptorSet set = alloc_set_locked(g_set_layout);
    if (!set) {
        ++g_skipped_setup;
        return false;
    }
    const std::uint32_t slot = g_next++ % kCompareSlots;
    // The images' own views (format, colour aspect, layer 0) are sampled ones.
    const VkDescriptorImageInfo ia{VK_NULL_HANDLE, a.view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo ib{VK_NULL_HANDLE, b.view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo is{g.dummy_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    const VkDescriptorBufferInfo ic{g_counts.buffer, 0, kCompareSlots * sizeof(std::uint32_t)};
    VkWriteDescriptorSet w[4] = {};
    for (int k = 0; k < 4; ++k) {
        w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k].dstSet = set;
        w[k].dstBinding = static_cast<std::uint32_t>(k);
        w[k].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[0].pImageInfo = &ia;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[1].pImageInfo = &ib;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w[2].pImageInfo = &is;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[3].pBufferInfo = &ic;
    vkUpdateDescriptorSets(g.device, 4, w, 0, nullptr);
    begin_recording_locked();
    render_end_pass_locked();
    VkCommandBuffer cmd = g_cmd();
    // Everything before it written (attachments, transfers, stores), its slot zeroed.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    vkCmdFillBuffer(cmd, g_counts.buffer, slot * sizeof(std::uint32_t), sizeof(std::uint32_t), 0);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    const Push push{a.width, a.height, slot};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, g_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (a.width + 7) / 8, (a.height + 7) / 8, 1);
    // What comes after (the write that ended the copy) waits for the reads; the host reads the count.
    mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    g_pending.push_back({slot, g.flushes, a.base, b.base, a.width, a.height, a.format, why, canary});
    return true;
}

// The canary written into `r`'s image (GENERAL, colour, one layer), with the barriers around the clear.
void clear_canary_locked(RtImage& r, bool initialised) {
    begin_recording_locked();
    render_end_pass_locked();
    VkCommandBuffer cmd = g_cmd();
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ib.oldLayout = initialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = r.image;
    ib.subresourceRange = range;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    vkCmdClearColorImage(cmd, r.image, VK_IMAGE_LAYOUT_GENERAL, &kCanary, 1, &range);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

// An image of `format` and size holding the canary alone, made once.
RtImage* canary_image_locked(VkFormat format, std::uint32_t width, std::uint32_t height) {
    const auto key = std::make_tuple(static_cast<int>(format), width, height);
    if (auto it = g_canary_images.find(key); it != g_canary_images.end()) return it->second.view ? &it->second : nullptr;
    RtImage& r = g_canary_images[key];
    r.base = 0;
    r.format = format;
    r.width = width;
    r.height = height;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {width, height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g.device, &ici, nullptr, &r.image) != VK_SUCCESS) return nullptr;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g.device, r.image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    {
        MemOwner own_mem("copy elision canaries");
        if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &r.memory.memory) != VK_SUCCESS) return nullptr;
    }
    r.memory.size = req.size;
    if (vkBindImageMemory(g.device, r.image, r.memory.memory, 0) != VK_SUCCESS) return nullptr;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = r.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(g.device, &vci, nullptr, &r.view) != VK_SUCCESS) {
        r.view = VK_NULL_HANDLE;
        return nullptr;
    }
    clear_canary_locked(r, false);
    r.initialised = true;
    return &r;
}
}  // namespace

void copy_canary_fill_locked(RtImage& t) {
    if (!comparable(t.format) || t.depth || t.layers != 1 || !t.image || t.managed_layout) {
        ++g_canary_skipped;
        return;
    }
    clear_canary_locked(t, t.initialised);
    t.initialised = true;
}

void copy_canary_check_locked(RtImage& t, const char* who) {
    RtImage* c = comparable(t.format) && !t.depth && t.layers == 1 && !t.managed_layout ? canary_image_locked(t.format, t.width, t.height) : nullptr;
    if (!c || !record_compare_locked(t, *c, who, true)) ++g_canary_skipped;
}

std::string copy_compare_report_locked() {
    if (!g_tried) return "compares: none yet";
    harvest_locked();
    std::string out = "compares";
    std::uint64_t all = 0, bad = 0;
    for (const auto& [why, t] : g_by_why) {
        all += t.compares;
        bad += t.differing;
        char b[160];
        std::snprintf(b, sizeof(b), " %s %llu (%llu differing, %llu texels)", why.c_str(), static_cast<unsigned long long>(t.compares),
                      static_cast<unsigned long long>(t.differing), static_cast<unsigned long long>(t.texels));
        out += b;
    }
    char b[200];
    std::snprintf(b, sizeof(b), "; %llu in all, %llu differing; %zu waiting; not compared: format %llu, full %llu, setup %llu",
                  static_cast<unsigned long long>(all), static_cast<unsigned long long>(bad), g_pending.size(),
                  static_cast<unsigned long long>(g_skipped_format), static_cast<unsigned long long>(g_skipped_full),
                  static_cast<unsigned long long>(g_skipped_setup));
    out += b;
    std::snprintf(b, sizeof(b), "; canaries %llu (%llu left texels, %llu texels in all; not checked %llu)", static_cast<unsigned long long>(g_canary_checks),
                  static_cast<unsigned long long>(g_canary_bad), static_cast<unsigned long long>(g_canary_left),
                  static_cast<unsigned long long>(g_canary_skipped));
    out += b;
    for (const auto& [who, n] : g_canary_by_who) {
        std::snprintf(b, sizeof(b), " %s %llu", who.c_str(), static_cast<unsigned long long>(n));
        out += b;
    }
    return out;
}

}  // namespace gpu
