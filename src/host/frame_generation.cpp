#include "host/frame_generation.h"

#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include "core/config.h"
#include "log.h"
#include "host/gpu.h"
#include "host/settings.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#include "../../tools/ngx_bridge/ngx_bridge.h"

#include <cmath>
#include "host/shaders/fsr_fg_hud.spv.h"

#if defined(BBHOST_HAVE_FSR)
#include <ffx_vk_fsr3_3_1_5_bridge.h>
#endif

namespace host {
namespace {

struct BridgeFg {
    bool loaded = false;
#if defined(_WIN32)
    HMODULE dll = nullptr;
#endif
    PFN_ngxb_fg_available fg_available = nullptr;
    PFN_ngxb_fg_create fg_create = nullptr;
    PFN_ngxb_fg_evaluate fg_evaluate = nullptr;
    PFN_ngxb_fg_max_generated fg_max_generated = nullptr;
    PFN_ngxb_fg_evaluate_index fg_evaluate_index = nullptr;
    PFN_ngxb_fg_release fg_release = nullptr;
} g_bridge_fg;

bool load_bridge_fg() {
    if (g_bridge_fg.loaded) return g_bridge_fg.fg_evaluate != nullptr;
    g_bridge_fg.loaded = true;
#if defined(_WIN32)
    HMODULE dll = GetModuleHandleW(L"ngx_bridge.dll");
    if (!dll) {
        const std::string path = config_exe_dir() + "/ngx_bridge.dll";
        std::wstring wpath(path.size() + 1, L'\0');
        wpath.resize(static_cast<std::size_t>(MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), static_cast<int>(wpath.size()))));
        if (!wpath.empty() && wpath.back() == L'\0') wpath.pop_back();
        dll = LoadLibraryExW(wpath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    if (!dll) {
        host_log("framegen: ngx_bridge.dll not found; FG disabled");
        return false;
    }
    g_bridge_fg.dll = dll;
    auto get = [dll](const char* name) { return reinterpret_cast<void*>(GetProcAddress(dll, name)); };
    g_bridge_fg.fg_available = reinterpret_cast<PFN_ngxb_fg_available>(get("ngxb_fg_available"));
    g_bridge_fg.fg_create = reinterpret_cast<PFN_ngxb_fg_create>(get("ngxb_fg_create"));
    g_bridge_fg.fg_evaluate = reinterpret_cast<PFN_ngxb_fg_evaluate>(get("ngxb_fg_evaluate"));
    g_bridge_fg.fg_max_generated = reinterpret_cast<PFN_ngxb_fg_max_generated>(get("ngxb_fg_max_generated"));
    g_bridge_fg.fg_evaluate_index = reinterpret_cast<PFN_ngxb_fg_evaluate_index>(get("ngxb_fg_evaluate_index"));
    g_bridge_fg.fg_release = reinterpret_cast<PFN_ngxb_fg_release>(get("ngxb_fg_release"));
    if (!g_bridge_fg.fg_available || !g_bridge_fg.fg_create || !g_bridge_fg.fg_evaluate || !g_bridge_fg.fg_release) {
        host_log("framegen: ngx_bridge.dll missing FG entry points; FG disabled");
        g_bridge_fg.fg_evaluate = nullptr;
        return false;
    }
    return true;
#else
    return false;
#endif
}

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    return UINT32_MAX;
}

struct HudPushConstants {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t mode; // 0: extract HUD to ui_hud, 1: composite ui_hud onto generated_target
};

FgStats g_stats;
std::mutex g_stats_mu;

}  // namespace

bool fg_enabled() {
    return host_startup_settings().frame_generation;
}

bool fg_available() {
    const int backend = host_startup_settings().frame_generation_backend;
    if (backend == 1) {
#if defined(BBHOST_HAVE_FSR)
        return true;
#else
        return false;
#endif
    }
    if (!load_bridge_fg()) return false;
    return g_bridge_fg.fg_available ? g_bridge_fg.fg_available() != 0 : false;
}

FgStats fg_get_stats() {
    std::lock_guard<std::mutex> lock(g_stats_mu);
    return g_stats;
}

void fg_note_real_presented() {
    std::lock_guard<std::mutex> lock(g_stats_mu);
    ++g_stats.real_presented;
    if (g_stats.real_presented == 1 || g_stats.real_presented % 300 == 0)
        host_log("framegen: presented %llu real, %llu generated; %llu evaluations failed, %llu suppressed",
                 static_cast<unsigned long long>(g_stats.real_presented),
                 static_cast<unsigned long long>(g_stats.generated_presented),
                 static_cast<unsigned long long>(g_stats.evaluations_failed),
                 static_cast<unsigned long long>(g_stats.interpolation_disabled));
}

void fg_note_generated_presented() {
    std::lock_guard<std::mutex> lock(g_stats_mu);
    ++g_stats.generated_presented;
}

void fg_note_interpolation_disabled() {
    std::lock_guard<std::mutex> lock(g_stats_mu);
    ++g_stats.interpolation_disabled;
    ++g_stats.frames_skipped;
}

FrameGenerator::~FrameGenerator() {
    shutdown();
}

bool FrameGenerator::init(VkInstance instance, VkPhysicalDevice phys, VkDevice device, std::uint32_t queue_family, VkQueue queue) {
    fsr_support_checked_ = false;
    fsr_supported_ = false;
    instance_ = instance;
    phys_ = phys;
    device_ = device;
    queue_family_ = queue_family;
    queue_ = queue;
    if (host_startup_settings().frame_generation_backend == 0) {
        load_bridge_fg();
    }

    if (!device_) return false;
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family_;
    if (vkCreateCommandPool(device_, &pci, nullptr, &eval_pool_) != VK_SUCCESS) return false;
    for (auto& frame : frames_) {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = eval_pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device_, &cai, &frame.cmd) != VK_SUCCESS) return false;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device_, &fci, nullptr, &frame.fence) != VK_SUCCESS) return false;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (vkCreateSemaphore(device_, &sci, nullptr, &frame.ready) != VK_SUCCESS) return false;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys_, &props);
    std::uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, families.data());
    if (queue_family_ < n && families[queue_family_].timestampValidBits == 64) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP; qi.queryCount = kFrameSlots * 2;
        if (vkCreateQueryPool(device_, &qi, nullptr, &times_) == VK_SUCCESS)
            timestamp_period_ns_ = props.limits.timestampPeriod;
    }
    return true;
}

void FrameGenerator::shutdown() {
    if (device_) {
        host_gpu_queue_lock();
        vkDeviceWaitIdle(device_);
        host_gpu_queue_unlock();
    }
    if (feature_created_) {
        if (active_backend_ == 0 && g_bridge_fg.fg_release) {
            g_bridge_fg.fg_release();
        }
#if defined(BBHOST_HAVE_FSR)
        if (active_backend_ == 1 && fsr_fg_context_) {
            ffxVkFsr3_3_1_6FrameGenerationContextDestroy(
                static_cast<FfxVkFsr3_3_1_6FrameGenerationContext*>(fsr_fg_context_));
            fsr_fg_context_ = nullptr;
        }
#endif
    }
    feature_created_ = false;
    fsr_max_render_w_ = fsr_max_render_h_ = 0;
    destroy_hud_pipeline();
    for (auto& frame : frames_) {
        for (auto& image : frame.generated) destroy_image(image);
        destroy_image(frame.real_copy); destroy_image(frame.input);
        destroy_image(frame.hud_mask);
        if (frame.disable_mapped) vkUnmapMemory(device_, frame.disable_mem);
        if (frame.disable_buf) vkDestroyBuffer(device_, frame.disable_buf, nullptr);
        if (frame.disable_mem) vkFreeMemory(device_, frame.disable_mem, nullptr);
        if (frame.fence) vkDestroyFence(device_, frame.fence, nullptr);
        if (frame.ready) vkDestroySemaphore(device_, frame.ready, nullptr);
        frame = {};
    }
    if (times_) vkDestroyQueryPool(device_, times_, nullptr);
    times_ = VK_NULL_HANDLE;
    if (eval_pool_) vkDestroyCommandPool(device_, eval_pool_, nullptr);
    eval_pool_ = VK_NULL_HANDLE;
    width_ = height_ = 0;
    format_ = VK_FORMAT_UNDEFINED;
    device_ = VK_NULL_HANDLE;
    history_.clear();
}

bool FrameGenerator::create_image(OwnedImage& im, std::uint32_t w, std::uint32_t h, VkFormat fmt, VkImageUsageFlags usage) {
    destroy_image(im);
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device_, &ici, nullptr, &im.image) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device_, im.image, &req);
    uint32_t mtype = find_memory_type(phys_, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mtype == UINT32_MAX) return false;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mtype;
    if (vkAllocateMemory(device_, &mai, nullptr, &im.memory) != VK_SUCCESS) return false;
    if (vkBindImageMemory(device_, im.image, im.memory, 0) != VK_SUCCESS) return false;

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = im.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(device_, &vci, nullptr, &im.view) != VK_SUCCESS) return false;

    im.format = fmt;
    im.width = w;
    im.height = h;
    return true;
}

void FrameGenerator::destroy_image(OwnedImage& im) {
    if (im.view) {
        vkDestroyImageView(device_, im.view, nullptr);
        im.view = VK_NULL_HANDLE;
    }
    if (im.image) {
        vkDestroyImage(device_, im.image, nullptr);
        im.image = VK_NULL_HANDLE;
    }
    if (im.memory) {
        vkFreeMemory(device_, im.memory, nullptr);
        im.memory = VK_NULL_HANDLE;
    }
    im.width = im.height = 0;
    im.format = VK_FORMAT_UNDEFINED;
}

bool FrameGenerator::ensure_hud_pipeline() {
    if (hud_pipeline_) return true;

    const VkDescriptorSetLayoutBinding binds[5] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 5;
    sli.pBindings = binds;
    if (vkCreateDescriptorSetLayout(device_, &sli, nullptr, &hud_set_layout_) != VK_SUCCESS) return false;

    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HudPushConstants)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &hud_set_layout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(device_, &pli, nullptr, &hud_pipeline_layout_) != VK_SUCCESS) return false;

    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(k_fsr_fg_hud_spv);
    smi.pCode = k_fsr_fg_hud_spv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device_, &smi, nullptr, &module) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = hud_pipeline_layout_;
    const bool ok = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &hud_pipeline_) == VK_SUCCESS;
    vkDestroyShaderModule(device_, module, nullptr);
    if (!ok) return false;

    VkDescriptorPoolSize pool_sizes[2] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3 * kFrameSlots},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kFrameSlots},
    };
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = kFrameSlots;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &hud_pool_) != VK_SUCCESS) return false;

    for (unsigned s = 0; s < kFrameSlots; ++s) {
        VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc_info.descriptorPool = hud_pool_;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &hud_set_layout_;
        if (vkAllocateDescriptorSets(device_, &alloc_info, &frames_[s].hud_desc_set) != VK_SUCCESS) return false;
    }

    return true;
}

void FrameGenerator::destroy_hud_pipeline() {
    if (hud_pool_) {
        vkDestroyDescriptorPool(device_, hud_pool_, nullptr);
        hud_pool_ = VK_NULL_HANDLE;
    }
    if (hud_pipeline_) {
        vkDestroyPipeline(device_, hud_pipeline_, nullptr);
        hud_pipeline_ = VK_NULL_HANDLE;
    }
    if (hud_pipeline_layout_) {
        vkDestroyPipelineLayout(device_, hud_pipeline_layout_, nullptr);
        hud_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (hud_set_layout_) {
        vkDestroyDescriptorSetLayout(device_, hud_set_layout_, nullptr);
        hud_set_layout_ = VK_NULL_HANDLE;
    }
    for (auto& frame : frames_) {
        frame.hud_desc_set = VK_NULL_HANDLE;
    }
}

bool FrameGenerator::ensure_feature(VkCommandBuffer cmd, std::uint32_t width, std::uint32_t height, VkFormat format) {
    const int backend = host_startup_settings().frame_generation_backend;
    if (backend == 1 && format == VK_FORMAT_B8G8R8A8_UNORM) format = VK_FORMAT_R8G8B8A8_UNORM;
    if (ready() && width_ == width && height_ == height && format_ == format && active_backend_ == backend) return true;

    if (backend == 0) {
        if (!load_bridge_fg() || !g_bridge_fg.fg_available || !g_bridge_fg.fg_available()) return false;
    } else {
#if !defined(BBHOST_HAVE_FSR)
        return false;
#else
        if (!fsr_support_checked_) {
            fsr_support_checked_ = true;
            std::uint32_t count = 0;
            vkEnumerateDeviceExtensionProperties(phys_, nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> extensions(count);
            vkEnumerateDeviceExtensionProperties(phys_, nullptr, &count, extensions.data());
            const bool has_derivatives = std::any_of(extensions.begin(), extensions.end(), [](const auto& ext) {
                return std::strcmp(ext.extensionName, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME) == 0;
            });
            if (has_derivatives) {
                VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
                VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                query.pNext = &derivatives;
                vkGetPhysicalDeviceFeatures2(phys_, &query);
                fsr_supported_ = derivatives.computeDerivativeGroupLinear;
            }
            if (!fsr_supported_)
                host_log("framegen: FSR provider requires KHR linear compute derivatives; retaining ordinary presentation");
        }
        if (!fsr_supported_) return false;
#endif
    }

    // Even after a failed evaluation/released NGX feature, a previous real
    // frame's blit can still read our images. Retire that work before resizing.
    if (device_ && (frames_[0].generated[0].image || frames_[0].real_copy.image || frames_[0].input.image)) {
        host_gpu_queue_lock();
        vkDeviceWaitIdle(device_);
        host_gpu_queue_unlock();
    }
    if (feature_created_) {
        if (active_backend_ == 0 && g_bridge_fg.fg_release) {
            g_bridge_fg.fg_release();
        }
#if defined(BBHOST_HAVE_FSR)
        if (active_backend_ == 1 && fsr_fg_context_) {
            ffxVkFsr3_3_1_6FrameGenerationContextDestroy(
                static_cast<FfxVkFsr3_3_1_6FrameGenerationContext*>(fsr_fg_context_));
            fsr_fg_context_ = nullptr;
        }
#endif
        feature_created_ = false;
        fsr_max_render_w_ = fsr_max_render_h_ = 0;
    }

    const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (backend == 1) {
        // FSR 3.1.6 only supports 2x frame generation
        generated_count_ = 1;
    } else {
        const unsigned requested = std::clamp(host_startup_settings().frame_generation_factor, 2, 4) - 1;
        const unsigned supported = g_bridge_fg.fg_max_generated && g_bridge_fg.fg_evaluate_index ?
            g_bridge_fg.fg_max_generated() : 1;
        generated_count_ = requested <= supported ? requested : 1;
        if (generated_count_ != requested)
            host_log("framegen: requested %ux unsupported (maximum %ux); using 2x", requested + 1, supported + 1);
    }
    for (auto& frame : frames_) {
        for (unsigned i = 0; i < frame.generated.size(); ++i) {
            if (i >= generated_count_) { destroy_image(frame.generated[i]); continue; }
            if (!create_image(frame.generated[i], width, height, format, usage)) {
                host_log("framegen: failed to create generated output image %u", i);
                return false;
            }
        }
        if (!create_image(frame.real_copy, width, height, format, usage)) {
            host_log("framegen: failed to create real copy image");
            return false;
        }
        if (!create_image(frame.input, width, height, format, usage)) return false;
        if (backend == 1) {
            if (!create_image(frame.hud_mask, width, height, format, usage)) {
                host_log("framegen: failed to create hud_mask image");
                return false;
            }
        } else {
            destroy_image(frame.hud_mask);
        }

        if (!frame.disable_buf) {
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = 16;
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            if (vkCreateBuffer(device_, &bci, nullptr, &frame.disable_buf) != VK_SUCCESS) return false;
            VkMemoryRequirements req{};
            vkGetBufferMemoryRequirements(device_, frame.disable_buf, &req);
            uint32_t mtype = find_memory_type(phys_, req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (mtype == UINT32_MAX) {
                vkDestroyBuffer(device_, frame.disable_buf, nullptr); frame.disable_buf = VK_NULL_HANDLE;
                return false;
            }
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            mai.pNext = &flags;
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = mtype;
            if (vkAllocateMemory(device_, &mai, nullptr, &frame.disable_mem) != VK_SUCCESS) {
                vkDestroyBuffer(device_, frame.disable_buf, nullptr); frame.disable_buf = VK_NULL_HANDLE;
                return false;
            }
            if (vkBindBufferMemory(device_, frame.disable_buf, frame.disable_mem, 0) != VK_SUCCESS ||
                vkMapMemory(device_, frame.disable_mem, 0, 16, 0, &frame.disable_mapped) != VK_SUCCESS) {
                vkDestroyBuffer(device_, frame.disable_buf, nullptr); frame.disable_buf = VK_NULL_HANDLE;
                vkFreeMemory(device_, frame.disable_mem, nullptr); frame.disable_mem = VK_NULL_HANDLE;
                return false;
            }
        }

        // Transition owned images to VK_IMAGE_LAYOUT_GENERAL
        VkImageMemoryBarrier barriers[6] = {};
        unsigned barrier_count = 0;
        for (unsigned i = 0; i < generated_count_; ++i) {
            barriers[barrier_count].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[barrier_count].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[barrier_count].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[barrier_count].srcQueueFamilyIndex = barriers[barrier_count].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[barrier_count].image = frame.generated[i].image;
            barriers[barrier_count].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barriers[barrier_count++].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        }
        barriers[barrier_count].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[barrier_count].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[barrier_count].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[barrier_count].srcQueueFamilyIndex = barriers[barrier_count].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[barrier_count].image = frame.real_copy.image;
        barriers[barrier_count].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers[barrier_count++].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

        barriers[barrier_count].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[barrier_count].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[barrier_count].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[barrier_count].srcQueueFamilyIndex = barriers[barrier_count].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[barrier_count].image = frame.input.image;
        barriers[barrier_count].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers[barrier_count++].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

        if (backend == 1 && frame.hud_mask.image) {
            barriers[barrier_count].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[barrier_count].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[barrier_count].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[barrier_count].srcQueueFamilyIndex = barriers[barrier_count].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[barrier_count].image = frame.hud_mask.image;
            barriers[barrier_count].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barriers[barrier_count++].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        }

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, nullptr, 0, nullptr, barrier_count, barriers);
    }

    if (backend == 0) {
        uint32_t r = g_bridge_fg.fg_create(cmd, width, height, format);
        if (ngxb_failed(r)) {
            host_log("framegen: ngxb_fg_create failed (0x%08x) for %ux%u format %d", r, width, height, static_cast<int>(format));
            return false;
        }
        host_log("framegen: DLSS Frame Generation %ux created for %ux%u format %d", generated_count_ + 1, width, height, static_cast<int>(format));
    } else {
#if defined(BBHOST_HAVE_FSR)
        if (!ensure_hud_pipeline()) {
            host_log("framegen: failed to create HUD composite compute pipeline");
            return false;
        }
        // Write descriptors for HUD pipeline for each slot
        for (unsigned s = 0; s < kFrameSlots; ++s) {
            auto& f = frames_[s];
            const VkDescriptorImageInfo img_info[5] = {
                {VK_NULL_HANDLE, f.real_copy.view, VK_IMAGE_LAYOUT_GENERAL},
                {VK_NULL_HANDLE, f.input.view, VK_IMAGE_LAYOUT_GENERAL},
                {VK_NULL_HANDLE, f.hud_mask.view, VK_IMAGE_LAYOUT_GENERAL},
                {VK_NULL_HANDLE, f.generated[0].view, VK_IMAGE_LAYOUT_GENERAL},
                {VK_NULL_HANDLE, f.generated[0].view, VK_IMAGE_LAYOUT_GENERAL},
            };
            VkWriteDescriptorSet writes[5] = {};
            for (int k = 0; k < 5; ++k) {
                writes[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[k].dstSet = f.hud_desc_set;
                writes[k].dstBinding = k;
                writes[k].descriptorCount = 1;
                writes[k].descriptorType = (k == 2 || k == 3) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                writes[k].pImageInfo = &img_info[k];
            }
            vkUpdateDescriptorSets(device_, 5, writes, 0, nullptr);
        }
        // fsr_fg_context_ will be created lazily when guides are available with actual render dimensions
#endif
    }

    width_ = width;
    height_ = height;
    format_ = format;
    active_backend_ = backend;
    feature_created_ = true;
    recreate_ = false;
    history_.clear();
    return true;
}

bool FrameGenerator::record_fsr_fg(VkCommandBuffer cmd, std::uint32_t width, std::uint32_t height,
                                   const gpu::DlssFgGuides& guides, unsigned slot) {
#if defined(BBHOST_HAVE_FSR)
    auto& frame = frames_[slot];

    // Lazy creation / re-creation of FSR FG context with maxRender dimensions from guides
    if (!fsr_fg_context_ || fsr_max_render_w_ < guides.width || fsr_max_render_h_ < guides.height) {
        if (fsr_fg_context_) {
            host_gpu_queue_lock();
            vkDeviceWaitIdle(device_);
            host_gpu_queue_unlock();
            ffxVkFsr3_3_1_6FrameGenerationContextDestroy(
                static_cast<FfxVkFsr3_3_1_6FrameGenerationContext*>(fsr_fg_context_));
            fsr_fg_context_ = nullptr;
        }
        FfxVkFsr3_3_1_6FrameGenerationCreateInfo ci{};
        ci.physicalDevice = phys_;
        ci.device = device_;
        ci.maxRenderWidth = guides.width;
        ci.maxRenderHeight = guides.height;
        ci.displayWidth = width;
        ci.displayHeight = height;
        ci.colorFormat = format_;
        FfxVkFsr3_3_1_6FrameGenerationContext* ctx = nullptr;
        FfxVkFsr3_3_1_6FrameGenerationResult fres = ffxVkFsr3_3_1_6FrameGenerationContextCreate(&ci, &ctx);
        if (fres != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
            host_log("framegen: FSR 3.1.6 ContextCreate failed (%d) for %ux%u format %d", fres, width, height, static_cast<int>(format_));
            return false;
        }
        fsr_fg_context_ = ctx;
        fsr_max_render_w_ = ci.maxRenderWidth;
        fsr_max_render_h_ = ci.maxRenderHeight;
        host_log("framegen: FSR 3.1.6 Frame Generation 2x created (maxRender %ux%u, display %ux%u)",
                 fsr_max_render_w_, fsr_max_render_h_, width, height);
    }

    auto* ctx = static_cast<FfxVkFsr3_3_1_6FrameGenerationContext*>(fsr_fg_context_);
    if (!ctx) return false;

    // Retire any completed frames whose GPU submission fence has signalled
    for (unsigned s = 0; s < kFrameSlots; ++s) {
        if (frames_[s].fsr_frame_id && vkGetFenceStatus(device_, frames_[s].fence) == VK_SUCCESS) {
            ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(ctx, frames_[s].fsr_frame_id);
            frames_[s].fsr_frame_id = 0;
        }
    }

    const bool reset = guides.reset || history_.needs_reset(guides.render_frame_index);
    if (frame.disable_mapped) {
        *static_cast<uint32_t*>(frame.disable_mapped) = reset ? 1u : 0u;
    }

    // Barrier: ensure transfer writes into frame.real_copy and frame.input are visible to compute shader sampling
    VkMemoryBarrier copy_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    copy_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copy_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &copy_barrier, 0, nullptr, 0, nullptr);

    // Step 1: Compute HUD isolation pass
    // Extract RGB difference: abs(real_copy.rgb - input.rgb) > threshold -> ui_hud
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hud_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hud_pipeline_layout_, 0, 1, &frame.hud_desc_set, 0, nullptr);
    HudPushConstants pc{width, height, 0u};
    vkCmdPushConstants(cmd, hud_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

    VkMemoryBarrier hud_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    hud_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hud_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &hud_barrier, 0, nullptr, 0, nullptr);

    // Step 2: Prepare FSR FG
    const std::uint64_t fid = fsr_frame_id_++;
    frame.fsr_frame_id = fid;

    const VkImageUsageFlags color_usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    const VkImageUsageFlags guide_usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

    const float frame_ms = std::isfinite(guides.frame_ms) ? std::clamp(guides.frame_ms, 1.0f, 100.0f) : 16.66f;
    const float near_plane = std::isfinite(guides.camera.near_plane) && guides.camera.near_plane > 0.0f ? guides.camera.near_plane : 0.1f;
    const float far_plane = std::isfinite(guides.camera.far_plane) && guides.camera.far_plane > near_plane ? guides.camera.far_plane : 3000.0f;
    const float vfov = std::isfinite(guides.camera.vertical_fov) && guides.camera.vertical_fov > 0.0f ? guides.camera.vertical_fov : 1.0f;
    const float jx = std::isfinite(guides.camera.jitter[0]) ? guides.camera.jitter[0] : 0.0f;
    const float jy = std::isfinite(guides.camera.jitter[1]) ? guides.camera.jitter[1] : 0.0f;

    FfxVkFsr3_3_1_6FrameGenerationPrepareInfo prep{};
    prep.commandBuffer = cmd;
    prep.color = {frame.input.image, frame.input.format, width, height, VK_IMAGE_LAYOUT_GENERAL, color_usage};
    prep.depth = {guides.depth, guides.depth_format, guides.width, guides.height, VK_IMAGE_LAYOUT_GENERAL, guide_usage};
    prep.motionVectors = {guides.motion, guides.motion_format, guides.width, guides.height, VK_IMAGE_LAYOUT_GENERAL, guide_usage};
    prep.renderWidth = guides.width;
    prep.renderHeight = guides.height;
    prep.jitterOffsetX = jx;
    prep.jitterOffsetY = jy;
    prep.motionVectorScaleX = 1.0f;
    prep.motionVectorScaleY = 1.0f;
    prep.frameTimeMilliseconds = frame_ms;
    prep.minLuminance = 0.0f;
    prep.maxLuminance = 1000.0f;
    prep.transferFunction = FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
    prep.cameraNear = near_plane;
    prep.cameraFar = far_plane;
    prep.viewSpaceToMeters = 1.0f;
    prep.cameraVerticalFovRadians = vfov;
    for (int i = 0; i < 3; ++i) {
        prep.cameraPosition[i] = std::isfinite(guides.camera.position[i]) ? guides.camera.position[i] : 0.0f;
        prep.cameraUp[i] = std::isfinite(guides.camera.up[i]) ? guides.camera.up[i] : (i == 1 ? 1.0f : 0.0f);
        prep.cameraRight[i] = std::isfinite(guides.camera.right[i]) ? guides.camera.right[i] : (i == 0 ? 1.0f : 0.0f);
        prep.cameraForward[i] = std::isfinite(guides.camera.forward[i]) ? guides.camera.forward[i] : (i == 2 ? 1.0f : 0.0f);
    }
    prep.frameId = fid;
    prep.reset = reset ? VK_TRUE : VK_FALSE;

    FfxVkFsr3_3_1_6FrameGenerationResult r_prep = ffxVkFsr3_3_1_6FrameGenerationContextRecordPrepare(ctx, &prep);
    if (r_prep != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
        static int prep_fail = 0;
        if (++prep_fail <= 5) host_log("framegen: FSR RecordPrepare failed (%d)", r_prep);
        ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(ctx, fid);
        frame.fsr_frame_id = 0;
        return false;
    }

    // Step 3: Dispatch FSR FG into generated[0]
    FfxVkFsr3_3_1_6FrameGenerationDispatchInfo disp{};
    disp.commandBuffer = cmd;
    disp.color = prep.color;
    disp.output = {frame.generated[0].image, frame.generated[0].format, width, height, VK_IMAGE_LAYOUT_GENERAL, color_usage};
    disp.displayWidth = width;
    disp.displayHeight = height;
    disp.frameTimeMilliseconds = frame_ms;
    disp.cameraNear = near_plane;
    disp.cameraFar = far_plane;
    disp.viewSpaceToMeters = 1.0f;
    disp.cameraVerticalFovRadians = vfov;
    disp.minLuminance = 0.0f;
    disp.maxLuminance = 1000.0f;
    disp.transferFunction = FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
    disp.frameId = fid;
    disp.reset = reset ? VK_TRUE : VK_FALSE;

    FfxVkFsr3_3_1_6FrameGenerationResult r_disp = ffxVkFsr3_3_1_6FrameGenerationContextRecordDispatch(ctx, &disp);
    {
        std::lock_guard<std::mutex> lock(g_stats_mu);
        ++g_stats.evaluations;
        if (r_disp != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) ++g_stats.evaluations_failed;
    }
    if (r_disp != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
        static int disp_fail = 0;
        if (++disp_fail <= 5) host_log("framegen: FSR RecordDispatch failed (%d)", r_disp);
        ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(ctx, fid);
        frame.fsr_frame_id = 0;
        return false;
    }

    // Barrier after FSR dispatch before HUD compositing
    VkMemoryBarrier fsr_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    fsr_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    fsr_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &fsr_barrier, 0, nullptr, 0, nullptr);

    // Step 4: Composite HUD onto generated[0]
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hud_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hud_pipeline_layout_, 0, 1, &frame.hud_desc_set, 0, nullptr);
    pc.mode = 1u;
    vkCmdPushConstants(cmd, hud_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

    return true;
#else
    return false;
#endif
}

bool FrameGenerator::evaluate(VkCommandBuffer cmd, VkImage color_image, VkImageView color_view, VkFormat color_format,
                              std::uint32_t width, std::uint32_t height, const gpu::DlssFgGuides& guides, unsigned slot) {
    auto& frame = frames_[slot];
    if (!feature_created_) return false;
    if (!guides.valid || !guides.depth || !guides.motion) return false;

    // The game renders a logical picture in the top-left of its maximum-size
    // display allocation. NGX requires the actual picture extent and a matching
    // image, rather than the allocation's unused rows and columns.
    VkMemoryBarrier before_copy{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before_copy.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    before_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &before_copy, 0, nullptr, 0, nullptr);
    VkImageCopy crop{};
    crop.srcSubresource = crop.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    crop.extent = {width, height, 1};

    if (active_backend_ == 1) {
        // FSR 3.1.6 path:
        // Input must be pre-HUD hudless color image. Real copy must always be manual full color image.
        // If guides do not have a valid matching hudless image, suppress generated frame.
        const bool valid_hudless = guides.hudless &&
                                   guides.hudless_width >= width &&
                                   guides.hudless_height >= height &&
                                   (guides.hudless_format == frame.input.format ||
                                    (guides.hudless_format == VK_FORMAT_B8G8R8A8_UNORM &&
                                     frame.input.format == VK_FORMAT_R8G8B8A8_UNORM));
        if (!valid_hudless) {
            return false;
        }

        const auto copy_color = [&](VkImage source, VkFormat source_format, const OwnedImage& target) {
            if (source_format == target.format) {
                vkCmdCopyImage(cmd, source, VK_IMAGE_LAYOUT_GENERAL, target.image,
                               VK_IMAGE_LAYOUT_GENERAL, 1, &crop);
            } else {
                VkImageBlit blit{};
                blit.srcSubresource = blit.dstSubresource = crop.srcSubresource;
                blit.srcOffsets[1] = blit.dstOffsets[1] = {static_cast<int>(width), static_cast<int>(height), 1};
                vkCmdBlitImage(cmd, source, VK_IMAGE_LAYOUT_GENERAL, target.image,
                               VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_NEAREST);
            }
        };
        copy_color(color_image, color_format, frame.real_copy);
        copy_color(guides.hudless, guides.hudless_format, frame.input);
    } else {
        // DLSS path: preserve exact original flow
        // DLSS input is ALWAYS full color image
        vkCmdCopyImage(cmd, color_image, VK_IMAGE_LAYOUT_GENERAL, frame.input.image,
                       VK_IMAGE_LAYOUT_GENERAL, 1, &crop);

        // MFG (> 1 generated) evaluates every temporal position with identical inputs.
        // Retain the real endpoint once manually.
        // For 2x (generated_count_ == 1), NGX OutputReal writes real_copy.
        if (generated_count_ > 1) {
            vkCmdCopyImage(cmd, color_image, VK_IMAGE_LAYOUT_GENERAL, frame.real_copy.image,
                           VK_IMAGE_LAYOUT_GENERAL, 1, &crop);
        }
    }
    (void)color_view;
    (void)color_format;

    if (times_) {
        vkCmdResetQueryPool(cmd, times_, slot * 2, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, times_, slot * 2);
    }

    if (active_backend_ == 1) {
        if (!record_fsr_fg(cmd, width, height, guides, slot)) return false;
    } else {
        if (!g_bridge_fg.fg_evaluate) return false;
        const bool reset = guides.reset || history_.needs_reset(guides.render_frame_index);
    if (frame.disable_mapped) {
        *static_cast<uint32_t*>(frame.disable_mapped) = reset ? 1u : 0u;
    }

    NgxbGenerate gen{};
    gen.color = {frame.input.image, frame.input.view, frame.input.format, width, height, VK_IMAGE_ASPECT_COLOR_BIT};
    if (generated_count_ == 1)
        gen.real = {frame.real_copy.image, frame.real_copy.view, frame.real_copy.format, width, height, VK_IMAGE_ASPECT_COLOR_BIT};
    gen.depth = {guides.depth, guides.depth_view, guides.depth_format, guides.width, guides.height, VK_IMAGE_ASPECT_COLOR_BIT};
    gen.motion = {guides.motion, guides.motion_view, guides.motion_format, guides.width, guides.height, VK_IMAGE_ASPECT_COLOR_BIT};
    if (guides.hudless) {
        const std::uint32_t hw = guides.hudless_width ? guides.hudless_width : guides.width;
        const std::uint32_t hh = guides.hudless_height ? guides.hudless_height : guides.height;
        gen.hudless = {guides.hudless, guides.hudless_view, guides.hudless_format, hw, hh, VK_IMAGE_ASPECT_COLOR_BIT};
    }
    gen.camera = guides.camera;
    gen.disable_interpolation = frame.disable_buf;
    gen.reset = reset;

    VkMemoryBarrier pre_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    pre_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    pre_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

    uint32_t r = 1;
    for (unsigned i = 0; i < generated_count_; ++i) {
        const auto& image = frame.generated[i];
        gen.output = {image.image, image.view, image.format, width, height, VK_IMAGE_ASPECT_COLOR_BIT};
        r = g_bridge_fg.fg_evaluate_index ? g_bridge_fg.fg_evaluate_index(cmd, &gen,
            generated_count_, i + 1, guides.render_frame_index) : g_bridge_fg.fg_evaluate(cmd, &gen);
        if (ngxb_failed(r)) break;
        // Every subframe shares the source inputs and NGX's feature state. Make
        // writes visible before requesting the next temporal position.
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, &pre_barrier, 0, nullptr, 0, nullptr);
    }
    {
        std::lock_guard<std::mutex> lock(g_stats_mu);
        ++g_stats.evaluations;
        if (ngxb_failed(r)) ++g_stats.evaluations_failed;
    }
    if (ngxb_failed(r)) {
        static int fail_count = 0;
        if (++fail_count <= 5) host_log("framegen: evaluate failed (0x%08x)", r);
        return false;
    }
    }

    VkMemoryBarrier post_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    post_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    post_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_SHADER_READ_BIT |
                                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &post_barrier, 0, nullptr, 0, nullptr);

    if (times_) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, times_, slot * 2 + 1);
    return true;
}

bool FrameGenerator::is_interpolation_disabled(unsigned slot) {
    const auto& frame = frames_[slot];
    if (!frame.disable_mapped) return false;
    // Memory is HOST_COHERENT and the evaluation fence has completed.
    return *static_cast<const uint32_t*>(frame.disable_mapped) != 0;
}

bool FrameGenerator::evaluate_submit(VkImage color_image, VkImageView color_view, VkFormat color_format,
                                     std::uint32_t width, std::uint32_t height, const gpu::DlssFgGuides& guides, unsigned slot) {
    return evaluate_submit(color_image, color_view, color_format, width, height, width, height, guides, slot);
}

bool FrameGenerator::evaluate_submit(VkImage color_image, VkImageView color_view, VkFormat color_format,
                                     std::uint32_t width, std::uint32_t height, std::uint32_t full_width, std::uint32_t full_height,
                                     const gpu::DlssFgGuides& guides, unsigned slot) {
    auto& frame = frames_[slot];
    if (!frame.cmd || !frame.fence) return false;
    if (frame.pending) {
        if (vkGetFenceStatus(device_, frame.fence) != VK_SUCCESS) return false;
        frame.pending = false;
    }

    vkResetCommandBuffer(frame.cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(frame.cmd, &bi) != VK_SUCCESS) return false;

    if (!ensure_feature(frame.cmd, width, height, color_format)) {
        vkEndCommandBuffer(frame.cmd);
        return false;
    }

    if (!evaluate(frame.cmd, color_image, color_view, color_format, width, height, guides, slot)) {
        vkEndCommandBuffer(frame.cmd);
        vkResetCommandBuffer(frame.cmd, 0);
#if defined(BBHOST_HAVE_FSR)
        if (active_backend_ == 1 && frame.fsr_frame_id && fsr_fg_context_) {
            ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(
                static_cast<FfxVkFsr3_3_1_6FrameGenerationContext*>(fsr_fg_context_),
                frame.fsr_frame_id);
            frame.fsr_frame_id = 0;
        }
#endif
        recreate_ = true;
        return false;
    }
    if (vkEndCommandBuffer(frame.cmd) != VK_SUCCESS) return false;

    vkResetFences(device_, 1, &frame.fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &frame.cmd;
    // A frame set discarded before presentation still has a signalled binary
    // semaphore. Consume that signal before re-signalling it on slot reuse.
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const VkSemaphore wait = frame.ready_unconsumed ? frame.ready : VK_NULL_HANDLE;
    si.waitSemaphoreCount = wait ? 1 : 0;
    si.pWaitSemaphores = wait ? &wait : nullptr;
    si.pWaitDstStageMask = wait ? &wait_stage : nullptr;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &frame.ready;
    frame.ticket = host_gpu_submit_presenter(frame.cmd, wait, wait_stage, frame.ready, frame.fence);
    VkResult res = VK_SUCCESS;
    if (!frame.ticket) {
        host_gpu_queue_lock();
        res = vkQueueSubmit(queue_, 1, &si, frame.fence);
        host_gpu_queue_unlock();
    }
    frame.pending = res == VK_SUCCESS;
    if (frame.pending) { history_.submitted(guides.render_frame_index); frame.ready_unconsumed = true; }
    else history_.clear();
    return frame.pending;
}

bool FrameGenerator::wait_evaluation(std::uint64_t timeout_ns, unsigned slot) {
    auto& frame = frames_[slot];
    if (!frame.fence || !frame.pending) return false;
    host_gpu_wait_submitted(frame.ticket);
    const bool done = vkWaitForFences(device_, 1, &frame.fence, VK_TRUE, timeout_ns) == VK_SUCCESS;
    // The producer clears pending after observing this fence before slot reuse.
    return done;
}

bool FrameGenerator::evaluation_finished(unsigned slot) const {
    return vkGetFenceStatus(device_, frames_[slot].fence) == VK_SUCCESS;
}

bool FrameGenerator::slot_available(unsigned slot) const {
    return !frames_[slot].pending || evaluation_finished(slot);
}

bool FrameGenerator::evaluation_times(unsigned slot, std::uint64_t* source_ns, std::uint64_t* ready_ns) const {
    if (!times_) return false;
    std::uint64_t values[2]{};
    if (vkGetQueryPoolResults(device_, times_, slot * 2, 2, sizeof(values), values,
            sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return false;
    *source_ns = static_cast<std::uint64_t>(values[0] * timestamp_period_ns_);
    *ready_ns = static_cast<std::uint64_t>(values[1] * timestamp_period_ns_);
    return *ready_ns >= *source_ns;
}

FrameGenerator& fg_get() {
    static FrameGenerator instance;
    return instance;
}

}  // namespace host
