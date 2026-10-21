// An indirect dispatch past 65,535 thread groups in a dimension runs nothing.
//
// D3D11 caps a dispatch at 65,535 groups a dimension, and DS3 PC - this
// engine's own PC renderer - keeps within it, so no native renderer of the
// game asks for more. A count past it is garbage that reached the arguments:
// at the Forbidden Woods a new particle system's append counter, still
// holding what reused memory held, was copied into a dispatch's x
// (3,176,062,614 groups, float bits) and the GPU ran it until bbhost gave up.
// The arguments are often written on the GPU earlier in the same stream (a
// counter copied into them), so they are checked there, in stream order: each
// indirect dispatch's arguments are copied into a slot of a ring and a
// one-thread pass (shaders/indirectclamp.comp) zeroes a count past the limit
// just before the dispatch reads the slot. The flag it leaves is read when
// the submission is done, and the first clamp of each program is logged with
// what its arguments held. BBHOST_INDIRECT_GUARD=0 reads them in place, as
// before.
#include "host/gpu_internal.h"
#include "host/shaders/indirectclamp.spv.h"
#include "log.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace gpu {
namespace {

constexpr std::uint32_t kClampSlots = 1u << 16;
constexpr std::uint32_t kSlotBytes = 32;

struct Pending {
    std::uint64_t serial, va;
    std::uint32_t slot;
    std::string name;
};

VkBuffer g_ring = VK_NULL_HANDLE;
VkDeviceMemory g_ring_mem = VK_NULL_HANDLE;
std::uint32_t* g_ring_map = nullptr;
VkDeviceAddress g_ring_address = 0;
VkPipelineLayout g_layout = VK_NULL_HANDLE;
VkPipeline g_pipeline = VK_NULL_HANDLE;
bool g_tried = false;
std::uint32_t g_next = 0;
std::vector<Pending> g_pending;   // under g.mu
std::set<std::string> g_said;     // programs whose clamp was logged
std::uint64_t g_clamped = 0, g_checked = 0;

bool make_locked() {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = static_cast<VkDeviceSize>(kClampSlots) * kSlotBytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (vkCreateBuffer(g.device, &bci, nullptr, &g_ring) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.device, g_ring, &req);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &flags;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    MemOwner own_mem("indirect dispatch guard");
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &g_ring_mem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(g.device, g_ring, g_ring_mem, 0) != VK_SUCCESS) return false;
    void* map = nullptr;
    if (vkMapMemory(g.device, g_ring_mem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) return false;
    g_ring_map = static_cast<std::uint32_t*>(map);
    std::memset(map, 0, bci.size);
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = g_ring;
    g_ring_address = vkGetBufferDeviceAddress(g.device, &bai);
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(g.device, &pli, nullptr, &g_layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(k_indirectclamp_spv);
    smi.pCode = k_indirectclamp_spv;
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
    return ok;
}

bool available_locked() {
    static const bool wanted = [] {
        const char* e = std::getenv("BBHOST_INDIRECT_GUARD");
        return !(e && e[0] == '0');
    }();
    if (!wanted) return false;
    if (!g_tried) {
        g_tried = true;
        if (!make_locked()) g_pipeline = VK_NULL_HANDLE;
        host_log("gpu: indirect dispatches checked on the GPU against 65,535 groups a dimension: %s",
                 g_pipeline ? "ready" : "unavailable (setup failed); read in place");
    }
    return g_pipeline != VK_NULL_HANDLE;
}

}  // namespace

bool indirect_clamp_locked(VkBuffer src, VkDeviceSize src_offset, std::uint64_t va, const std::string& name, VkBuffer& buffer,
                           VkDeviceSize& offset) {
    if (!available_locked()) return false;
    const std::uint32_t slot = g_next++ % kClampSlots;
    const VkDeviceSize at = static_cast<VkDeviceSize>(slot) * kSlotBytes;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    const VkBufferCopy region{src_offset, at, 12};
    rec().copy_buffer(src, g_ring, 1, &region);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    rec().bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, g_pipeline);
    const VkDeviceAddress address = g_ring_address + at;
    rec().push_constants(g_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(address), &address);
    rec().dispatch(1, 1, 1);
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    if (g_pending.size() < kClampSlots) g_pending.push_back({g.flushes, va, slot, name});
    buffer = g_ring;
    offset = at;
    return true;
}

void indirect_clamp_retired_locked(std::uint64_t serial) {
    if (g_pending.empty()) return;
    std::size_t keep = 0;
    for (Pending& p : g_pending) {
        if (p.serial > serial) {
            g_pending[keep++] = std::move(p);
            continue;
        }
        ++g_checked;
        const std::uint32_t* v = g_ring_map + static_cast<std::size_t>(p.slot) * (kSlotBytes / 4);
        if (!v[3]) continue;
        ++g_clamped;
        if (g_said.insert(p.name).second) {
            host_log("gpu: an indirect dispatch of %s asked for %ux%ux%u thread groups, past D3D11's 65535 a dimension: it ran nothing, "
                     "as will every such dispatch (arguments at 0x%llx; BBHOST_INDIRECT_GUARD=0 runs them)",
                     p.name.c_str(), v[4], v[5], v[6], static_cast<unsigned long long>(p.va));
        } else if (g_clamped % 1000 == 0) {
            host_log("gpu: %llu indirect dispatches ran nothing for asking past 65535 thread groups a dimension",
                     static_cast<unsigned long long>(g_clamped));
        }
    }
    g_pending.resize(keep);
}

std::string indirect_clamp_report() {
    char b[160];
    std::snprintf(b, sizeof(b), "indirect dispatches checked on the GPU %llu, run as nothing for asking past 65535 groups %llu",
                  static_cast<unsigned long long>(g_checked), static_cast<unsigned long long>(g_clamped));
    return b;
}

}  // namespace gpu
