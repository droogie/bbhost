// PS4 tiled textures untiled on the GPU (shaders/untile.comp). A texture's
// upload used to untile it on the CPU into pinned staging - memory-bound work
// that more threads made slower, ~450 ms of it as an area loads and some on the
// command processor itself when a dispatch needed the texture at once. Here
// the untile is a compute pass that reads the tiled bytes from guest memory by
// device address and writes the linear elements into the same staging span;
// the copy into the image stays as it was (textures.cpp, upload_surface).
#include "host/gpu_internal.h"
#include "host/texture_upload_plan.h"
#include "host/shaders/untile.spv.h"
#include "log.h"

#include <atomic>
#include <cstring>

namespace gpu {
namespace {

// The shader's push constants (std430: the four addresses and sizes first).
struct Push {
    std::uint64_t src, dst, src_slice_bytes, dst_slice_bytes;
    std::uint32_t width_e, height_e, pitch_e, esize, mode, bank_height, aspect, banks;
};
static_assert(sizeof(Push) == 64, "the shader's push-constant block");

VkPipelineLayout g_layout = VK_NULL_HANDLE;
VkPipeline g_pipeline = VK_NULL_HANDLE;
bool g_tried = false;
std::atomic<std::uint64_t> g_checked{0}, g_mismatched{0};

bool make_pipeline_locked() {
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(k_untile_spv);
    smi.pCode = k_untile_spv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.device, &smi, nullptr, &module) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    bool ok = vkCreatePipelineLayout(g.device, &pli, nullptr, &g_layout) == VK_SUCCESS;
    if (ok) {
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = module;
        ci.stage.pName = "main";
        ci.layout = g_layout;
        ok = vkCreateComputePipelines(g.device, g.cache, 1, &ci, nullptr, &g_pipeline) == VK_SUCCESS;
    }
    vkDestroyShaderModule(g.device, module, nullptr);
    host_log("gpu: texture untiling on the GPU %s", ok ? "ready" : "unavailable (pipeline creation failed); the CPU untiles");
    return ok;
}

}  // namespace

bool untile_gpu_available_locked() {
    if (!g_tried) {
        g_tried = true;
        make_pipeline_locked();
    }
    return g_pipeline != VK_NULL_HANDLE;
}

void untile_gpu_record_locked(const UntileGpuPass& pass) {
    const Push push{pass.src,     pass.dst,  pass.src_slice_bytes, pass.dst_slice_bytes, pass.width_e, pass.height_e,
                    pass.pitch_e, pass.esize, pass.mode,           pass.bank_height,     pass.aspect,  pass.banks};
    if (texture_stream_enabled()) {
        rec().bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, g_pipeline);
        rec().push_constants(g_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        rec().dispatch((pass.width_e + 7) / 8, (pass.height_e + 7) / 8, pass.slices);
    } else {
        VkCommandBuffer cmd = g_cmd();
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipeline);
        vkCmdPushConstants(cmd, g_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (pass.width_e + 7) / 8, (pass.height_e + 7) / 8, pass.slices);
    }
}

void untile_checks_run_locked(std::vector<UntileCheck>& checks) {
    for (const UntileCheck& c : checks) {
        g_checked.fetch_add(1, std::memory_order_relaxed);
        if (std::memcmp(c.gpu, c.cpu, c.bytes) == 0) continue;
        std::size_t first = 0;
        while (first < c.bytes && c.gpu[first] == c.cpu[first]) ++first;
        if (g_mismatched.fetch_add(1, std::memory_order_relaxed) < 12) {
            host_log("gpu untile: 0x%llx differs from the CPU's at byte %zu of %zu (GPU %02x, CPU %02x)",
                     static_cast<unsigned long long>(c.base), first, c.bytes, c.gpu[first], c.cpu[first]);
        }
    }
    checks.clear();
}

std::string untile_gpu_report() {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "GPU untile compare: %llu uploads checked, %llu differ",
                  static_cast<unsigned long long>(g_checked.load()), static_cast<unsigned long long>(g_mismatched.load()));
    return buf;
}

}  // namespace gpu
