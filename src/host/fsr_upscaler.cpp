// Independent FSR upscaling adapter for Vulkan (AMD FSR 3.1.5 and FSR 4 source-v07).
#include "host/fsr_upscaler.h"
#include "host/settings.h"
#include "core/config.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ffx_vk_fsr3_3_1_5_bridge.h"
#include "ffx_vk_fsr4_v07.h"
#include "ffx_vk_fsr4_v07_assets.h"
#include "ffx_vk_fsr4_v07_schedule.h"
#include "ffx_vk_fsr4_v07_types.h"

namespace gpu {

namespace {

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    return UINT32_MAX;
}

struct OwnedImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
    bool initialized = false;
};

void destroy_image(VkDevice device, OwnedImage& im) {
    if (im.view) {
        vkDestroyImageView(device, im.view, nullptr);
        im.view = VK_NULL_HANDLE;
    }
    if (im.image) {
        vkDestroyImage(device, im.image, nullptr);
        im.image = VK_NULL_HANDLE;
    }
    if (im.memory) {
        vkFreeMemory(device, im.memory, nullptr);
        im.memory = VK_NULL_HANDLE;
    }
    im.format = VK_FORMAT_UNDEFINED;
    im.width = im.height = 0;
    im.initialized = false;
}

bool create_image(VkPhysicalDevice phys, VkDevice device, OwnedImage& im,
                  uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage) {
    destroy_image(device, im);
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
    if (vkCreateImage(device, &ici, nullptr, &im.image) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, im.image, &req);
    uint32_t mtype = find_memory_type(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mtype == UINT32_MAX) {
        destroy_image(device, im);
        return false;
    }

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mtype;
    if (vkAllocateMemory(device, &mai, nullptr, &im.memory) != VK_SUCCESS) {
        destroy_image(device, im);
        return false;
    }
    if (vkBindImageMemory(device, im.image, im.memory, 0) != VK_SUCCESS) {
        destroy_image(device, im);
        return false;
    }

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = im.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(device, &vci, nullptr, &im.view) != VK_SUCCESS) {
        destroy_image(device, im);
        return false;
    }
    im.format = fmt;
    im.width = w;
    im.height = h;
    return true;
}

std::string fsr4_asset_dir() {
    const char* p = std::getenv("BBHOST_FSR4_ASSETS");
    if (p && p[0]) return p;
    p = std::getenv("BB_FSR4_DIR");
    if (p && p[0]) return p;
    return config_exe_dir() + "/fsr4_shaders";
}

bool read_binary_file(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto sz = static_cast<std::size_t>(f.tellg());
    out.resize(sz);
    f.seekg(0);
    return bool(f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(sz)));
}

FfxFsr4ModelPreset select_fsr4_preset(int dlss_mode, uint32_t in_w, uint32_t out_w) {
    if (dlss_mode == 1 || in_w == out_w) return FFX_FSR4_MODEL_PRESET_NATIVE_AA;
    if (dlss_mode >= 2 && dlss_mode <= 5) {
        switch (dlss_mode) {
        case 2: return FFX_FSR4_MODEL_PRESET_QUALITY;
        case 3: return FFX_FSR4_MODEL_PRESET_BALANCED;
        case 4: return FFX_FSR4_MODEL_PRESET_PERFORMANCE;
        case 5: return FFX_FSR4_MODEL_PRESET_ULTRA_PERFORMANCE;
        }
    }
    return ffxFsr4SelectModelPreset(in_w, out_w, false);
}

uint32_t to_ffx_surface_format(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R32G32B32A32_SFLOAT: return FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    case VK_FORMAT_R16G16_SFLOAT:       return FFX_SURFACE_FORMAT_R16G16_FLOAT;
    case VK_FORMAT_R32_SFLOAT:          return FFX_SURFACE_FORMAT_R32_FLOAT;
    case VK_FORMAT_R32_UINT:            return FFX_SURFACE_FORMAT_R32_UINT;
    case VK_FORMAT_R16_SFLOAT:          return FFX_SURFACE_FORMAT_R16_FLOAT;
    case VK_FORMAT_R8G8B8A8_UNORM:      return FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8_UNORM:            return FFX_SURFACE_FORMAT_R8_UNORM;
    default:                            return FFX_SURFACE_FORMAT_UNKNOWN;
    }
}

bool check_fsr4_hardware_support(VkPhysicalDevice phys) {
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f2.pNext = &f12;
    f12.pNext = &f13;
    vkGetPhysicalDeviceFeatures2(phys, &f2);
    return f12.shaderFloat16 && f12.shaderInt8 && f2.features.shaderInt16 &&
           f13.shaderIntegerDotProduct && f2.features.shaderStorageImageExtendedFormats;
}

struct InFlightFrame {
    uint64_t frame_id = 0;
    uint64_t recording_serial = 0;
    int backend = 0; // 1: FSR3, 2: FSR4
    std::array<FfxVkFsr3_3_1_5Resource, 7> fsr3_tokens{};
};

struct FsrContext {
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    int active_backend = 0; // 1: FSR3, 2: FSR4
    uint32_t in_w = 0, in_h = 0;
    uint32_t out_w = 0, out_h = 0;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    VkFormat output_format = VK_FORMAT_UNDEFINED;
    int model_preset = -1;

    // Submission tracking: monotonic serial mapped to in-flight records
    uint64_t next_frame_id = 1;
    std::deque<InFlightFrame> in_flight;

    // FSR 3.1.5 Context & Persistent Shared Resources
    FfxVkFsr3_3_1_5UpscalerContext* fsr3_ctx = nullptr;
    OwnedImage dilated_depth;
    OwnedImage dilated_motion;
    OwnedImage reconstructed_prev_depth;

    // FSR 4 source-v07 Context & State
    std::vector<uint8_t> fsr4_scratch;
    FfxInterface fsr4_backend{};
    bool fsr4_backend_ok = false;
    ffxContext fsr4_ctx{};
    bool fsr4_ctx_ok = false;

    bool fsr4_hardware_checked = false;
    bool fsr4_hardware_supported = false;
    bool fsr4_failed_logged = false;
    bool fsr4_permanently_unavailable = false;

    void shutdown() {
        if (fsr3_ctx) {
            FfxVkFsr3_3_1_5Bridge* bridge = ffxVkFsr3_3_1_5UpscalerContextGetBridge(fsr3_ctx);
            for (auto& inf : in_flight) {
                if (inf.backend == 1 && bridge) {
                    for (auto& tok : inf.fsr3_tokens) {
                        if (tok.resource) {
                            ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, tok);
                            tok.resource = nullptr;
                        }
                    }
                }
            }
            ffxVkFsr3_3_1_5UpscalerContextDestroy(fsr3_ctx);
            fsr3_ctx = nullptr;
        }
        if (device != VK_NULL_HANDLE) {
            destroy_image(device, dilated_depth);
            destroy_image(device, dilated_motion);
            destroy_image(device, reconstructed_prev_depth);
        }
        if (fsr4_ctx_ok) {
            ffxFsr4V07DestroyContext(&fsr4_ctx, nullptr);
            fsr4_ctx_ok = false;
        }
        if (fsr4_backend_ok) {
            ffxFsr4VkDestroyContext(reinterpret_cast<FfxFsr4VkContext*>(fsr4_backend.device));
            fsr4_backend_ok = false;
            fsr4_backend = {};
        }
        fsr4_scratch.clear();
        in_flight.clear();
        active_backend = 0;
        in_w = in_h = out_w = out_h = 0;
        color_format = output_format = VK_FORMAT_UNDEFINED;
        model_preset = -1;
    }

    void retire(uint64_t completed_submits) {
        FfxVkFsr3_3_1_5Bridge* bridge = fsr3_ctx ? ffxVkFsr3_3_1_5UpscalerContextGetBridge(fsr3_ctx) : nullptr;
        while (!in_flight.empty()) {
            const auto& front = in_flight.front();
            // Safe retirement: strictly completed_submits > recording_serial
            if (front.recording_serial < completed_submits) {
                const uint64_t fid = front.frame_id;
                if (front.backend == 1 && fsr3_ctx) {
                    // Call ContextRetireFrame FIRST, then ReleaseImportedImage for retained tokens
                    ffxVkFsr3_3_1_5UpscalerContextRetireFrame(fsr3_ctx, fid);
                    if (bridge) {
                        for (const auto& tok : front.fsr3_tokens) {
                            if (tok.resource) {
                                ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, tok);
                            }
                        }
                    }
                } else if (front.backend == 2 && fsr4_backend_ok) {
                    ffxFsr4VkRetireFrame(&fsr4_backend, fid);
                }
                in_flight.pop_front();
            } else {
                break;
            }
        }
    }

    bool init_fsr3(VkPhysicalDevice in_phys, VkDevice in_device,
                   uint32_t req_in_w, uint32_t req_in_h, uint32_t req_out_w, uint32_t req_out_h,
                   VkFormat req_color_format, VkFormat req_output_format) {
        if (!in_flight.empty()) return false;
        shutdown();
        phys = in_phys;
        device = in_device;
        in_w = req_in_w;
        in_h = req_in_h;
        out_w = req_out_w;
        out_h = req_out_h;
        color_format = req_color_format;
        output_format = req_output_format;

        const bool is_hdr = (color_format == VK_FORMAT_R16G16B16A16_SFLOAT ||
                             color_format == VK_FORMAT_R32G32B32A32_SFLOAT);

        FfxVkFsr3_3_1_5UpscalerCreateInfo ci{};
        ci.physicalDevice = phys;
        ci.device = device;
        ci.maxRenderWidth = in_w;
        ci.maxRenderHeight = in_h;
        ci.maxUpscaleWidth = out_w;
        ci.maxUpscaleHeight = out_h;
        ci.hdrColorInput = is_hdr ? VK_TRUE : VK_FALSE;
        ci.autoExposure = VK_TRUE;

        if (ffxVkFsr3_3_1_5UpscalerContextCreate(&ci, &fsr3_ctx) != FFX_VK_FSR3_3_1_5_OK) {
            host_log("fsr: failed to create FSR 3.1.5 upscaler context");
            shutdown();
            return false;
        }

        FfxVkFsr3_3_1_5SharedResourceDescriptions shared{};
        if (ffxVkFsr3_3_1_5UpscalerContextGetSharedResourceDescriptions(fsr3_ctx, &shared) != FFX_VK_FSR3_3_1_5_OK) {
            host_log("fsr: failed to query shared resource descriptions");
            shutdown();
            return false;
        }

        const VkImageUsageFlags uav_usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (!create_image(phys, device, dilated_depth, shared.dilatedDepth.width, shared.dilatedDepth.height,
                         shared.dilatedDepth.format, uav_usage) ||
            !create_image(phys, device, dilated_motion, shared.dilatedMotionVectors.width, shared.dilatedMotionVectors.height,
                         shared.dilatedMotionVectors.format, uav_usage) ||
            !create_image(phys, device, reconstructed_prev_depth, shared.reconstructedPrevNearestDepth.width,
                         shared.reconstructedPrevNearestDepth.height, shared.reconstructedPrevNearestDepth.format, uav_usage)) {
            host_log("fsr: failed to allocate persistent shared resources");
            shutdown();
            return false;
        }

        active_backend = 1;
        host_log("fsr: initialized FSR 3.1.5 upscaler (%ux%u -> %ux%u, hdr=%d)", in_w, in_h, out_w, out_h, is_hdr);
        return true;
    }

    bool init_fsr4(VkPhysicalDevice in_phys, VkDevice in_device,
                   uint32_t req_in_w, uint32_t req_in_h, uint32_t req_out_w, uint32_t req_out_h,
                   VkFormat req_color_format, VkFormat req_output_format,
                   FfxFsr4ModelPreset preset) {
        if (!in_flight.empty()) return false;
        shutdown();
        phys = in_phys;
        device = in_device;
        in_w = req_in_w;
        in_h = req_in_h;
        out_w = req_out_w;
        out_h = req_out_h;
        color_format = req_color_format;
        output_format = req_output_format;
        model_preset = static_cast<int>(preset);

        if (!fsr4_hardware_checked) {
            fsr4_hardware_checked = true;
            fsr4_hardware_supported = check_fsr4_hardware_support(phys);
        }
        if (!fsr4_hardware_supported) {
            if (!fsr4_failed_logged) {
                host_log("fsr4: hardware lacks required INT8 dot product or storage extended formats; falling back to FSR 3");
                fsr4_failed_logged = true;
            }
            fsr4_permanently_unavailable = true;
            return false;
        }

        FfxFsr4V07AssetSet assets{};
        if (!ffxFsr4V07BuildAssetSet(preset, out_w, out_h, &assets)) {
            if (!fsr4_failed_logged) {
                host_log("fsr4: output resolution %ux%u not supported by v07 model; falling back to FSR 3", out_w, out_h);
                fsr4_failed_logged = true;
            }
            fsr4_permanently_unavailable = true;
            return false;
        }

        const std::string dir = fsr4_asset_dir() + "/";
        std::array<std::vector<uint8_t>, FFX_FSR4_VK_PASS_COUNT> code;
        std::vector<uint8_t> initializer, weights;

        const auto load = [&](const char* name, std::vector<uint8_t>& data) {
            if (read_binary_file(dir + name, data)) return true;
            return false;
        };

        if (!load(assets.pre, code[0])) {
            if (!fsr4_failed_logged) {
                host_log("fsr4: missing asset %s%s; falling back to FSR 3", dir.c_str(), assets.pre);
                fsr4_failed_logged = true;
            }
            fsr4_permanently_unavailable = true;
            return false;
        }
        for (uint32_t pass = 0; pass < FFX_FSR4_MODEL_PASS_COUNT; ++pass) {
            if (!load(assets.model[pass], code[1 + pass])) {
                if (!fsr4_failed_logged) {
                    host_log("fsr4: missing asset %s%s; falling back to FSR 3", dir.c_str(), assets.model[pass]);
                    fsr4_failed_logged = true;
                }
                fsr4_permanently_unavailable = true;
                return false;
            }
        }
        if (!load(assets.post, code[13]) || !load(assets.rcas, code[14]) ||
            !load(assets.spdAutoExposure, code[15]) || !load(assets.initializer, initializer) ||
            !load(assets.prePassWeights, weights)) {
            if (!fsr4_failed_logged) {
                host_log("fsr4: missing post/weights assets in %s; falling back to FSR 3", dir.c_str());
                fsr4_failed_logged = true;
            }
            fsr4_permanently_unavailable = true;
            return false;
        }

        if (initializer.size() != FFX_FSR4_V07_INITIALIZER_BYTES ||
            weights.size() != FFX_FSR4_V07_PRE_PASS_WEIGHTS_BYTES) {
            if (!fsr4_failed_logged) {
                host_log("fsr4: invalid weights size; falling back to FSR 3");
                fsr4_failed_logged = true;
            }
            fsr4_permanently_unavailable = true;
            return false;
        }

        static const std::array<std::string, FFX_FSR4_VK_PASS_COUNT> entries = [] {
            std::array<std::string, FFX_FSR4_VK_PASS_COUNT> names;
            names.fill("main");
            for (uint32_t pass = 1; pass <= FFX_FSR4_MODEL_PASS_COUNT; ++pass) {
                names[pass] = "fsr4_model_v07_i8_pass" + std::to_string(pass);
            }
            return names;
        }();

        FfxFsr4VkCreateInfo ci{};
        ci.device = device;
        ci.physicalDevice = phys;
        for (uint32_t i = 0; i < FFX_FSR4_VK_PASS_COUNT; ++i) {
            if (code[i].size() % 4 != 0) {
                if (!fsr4_failed_logged) {
                    host_log("fsr4: invalid SPIR-V bytecode size; falling back to FSR 3");
                    fsr4_failed_logged = true;
                }
                fsr4_permanently_unavailable = true;
                return false;
            }
            ci.shaders[i] = {reinterpret_cast<const uint32_t*>(code[i].data()), code[i].size(), entries[i].c_str()};
        }
        ci.modelInitializer = initializer.data();
        ci.modelInitializerSize = initializer.size();
        ci.prePassWeights = weights.data();
        ci.prePassWeightsSize = weights.size();

        fsr4_scratch.assign(ffxFsr4VkGetScratchMemorySize(), 0);
        ci.scratchBuffer = fsr4_scratch.data();
        ci.scratchBufferSize = fsr4_scratch.size();

        if (ffxFsr4VkCreateContext(&ci, &fsr4_backend) != VK_SUCCESS) {
            if (!fsr4_failed_logged) {
                host_log("fsr4: failed to create Vulkan backend; falling back to FSR 3");
                fsr4_failed_logged = true;
            }
            fsr4_scratch.clear();
            fsr4_permanently_unavailable = true;
            return false;
        }
        fsr4_backend_ok = true;

        const bool is_hdr = (color_format == VK_FORMAT_R16G16B16A16_SFLOAT ||
                             color_format == VK_FORMAT_R32G32B32A32_SFLOAT);

        ffxCreateContextDescUpscale desc{};
        desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        desc.flags = (is_hdr ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE : 0u) | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
        desc.maxRenderSize = {(out_w + 7) & ~7u, (out_h + 7) & ~7u};
        desc.maxUpscaleSize = {(out_w + 7) & ~7u, (out_h + 7) & ~7u};

        ffxFsr4V07SetBackendInterface(&fsr4_backend);
        const auto created = ffxFsr4V07CreateContext(&fsr4_ctx, &desc.header, nullptr);
        ffxFsr4V07SetBackendInterface(nullptr);
        if (created != FFX_API_RETURN_OK) {
            if (!fsr4_failed_logged) {
                host_log("fsr4: provider context creation failed (%d); falling back to FSR 3", created);
                fsr4_failed_logged = true;
            }
            shutdown();
            fsr4_permanently_unavailable = true;
            return false;
        }
        fsr4_ctx_ok = true;
        active_backend = 2;
        host_log("fsr: initialized FSR 4 source-v07 (%s, tier %s, %ux%u -> %ux%u, hdr=%d)",
                 ffxFsr4ModelPresetName(preset), assets.tier, in_w, in_h, out_w, out_h, is_hdr);
        return true;
    }
};

static FsrContext g_ctx;

} // namespace

bool fsr_upscale_locked(VkPhysicalDevice phys,
                        VkDevice device,
                        VkCommandBuffer cmd,
                        const NgxbImage& color,
                        const NgxbImage& output,
                        const DlssFgGuides& guides,
                        uint64_t recording_serial,
                        uint64_t completed_submits) {
    if (!phys || !device || !cmd || !color.image || !output.image) return false;
    if (!guides.depth || !guides.motion) return false;

    // Retire completed frames first
    g_ctx.retire(completed_submits);

    const auto& settings = host_startup_settings();
    int desired_backend = settings.upscaler_backend; // 1: FSR3, 2: FSR4
    if (desired_backend != 1 && desired_backend != 2) desired_backend = 1;

    // If FSR 4 is permanently unavailable in this session, keep using FSR 3
    if (desired_backend == 2 && g_ctx.fsr4_permanently_unavailable) {
        desired_backend = 1;
    }

    const uint32_t in_w = color.width;
    const uint32_t in_h = color.height;
    const uint32_t out_w = output.width;
    const uint32_t out_h = output.height;
    const VkFormat color_fmt = color.format;
    const VkFormat output_fmt = output.format;
    const FfxFsr4ModelPreset preset = select_fsr4_preset(settings.dlss_mode, in_w, out_w);

    // Ensure / recreate context
    if (desired_backend == 2) {
        if (g_ctx.active_backend != 2 || g_ctx.in_w != in_w || g_ctx.in_h != in_h ||
            g_ctx.out_w != out_w || g_ctx.out_h != out_h || g_ctx.color_format != color_fmt ||
            g_ctx.output_format != output_fmt || g_ctx.model_preset != static_cast<int>(preset)) {
            if (!g_ctx.in_flight.empty()) return false; // wait until safe to recreate
            if (!g_ctx.init_fsr4(phys, device, in_w, in_h, out_w, out_h, color_fmt, output_fmt, preset)) {
                // Fallback to FSR3
                desired_backend = 1;
            }
        }
    }

    if (desired_backend == 1) {
        if (g_ctx.active_backend != 1 || g_ctx.in_w != in_w || g_ctx.in_h != in_h ||
            g_ctx.out_w != out_w || g_ctx.out_h != out_h || g_ctx.color_format != color_fmt ||
            g_ctx.output_format != output_fmt) {
            if (!g_ctx.in_flight.empty()) return false; // wait until safe to recreate
            if (!g_ctx.init_fsr3(phys, device, in_w, in_h, out_w, out_h, color_fmt, output_fmt)) {
                return false;
            }
        }
    }

    const uint64_t frame_id = g_ctx.next_frame_id++;

    if (g_ctx.active_backend == 1) {
        // Dispatch FSR 3.1.5
        // These images belong to the host, so the bridge must see their real
        // initial layout. It restores imported images to GENERAL after dispatch.
        for (auto* image : {&g_ctx.dilated_depth, &g_ctx.dilated_motion, &g_ctx.reconstructed_prev_depth}) {
            if (image->initialized) continue;
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image->image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
            image->initialized = true;
        }
        FfxVkFsr3_3_1_5Bridge* bridge = ffxVkFsr3_3_1_5UpscalerContextGetBridge(g_ctx.fsr3_ctx);
        if (!bridge) return false;

        auto import_img = [&](VkImage im, VkFormat fmt, uint32_t w, uint32_t h,
                              VkImageLayout layout, uint32_t state, VkImageUsageFlags usage) -> FfxVkFsr3_3_1_5Resource {
            FfxVkFsr3_3_1_5ImportedImageDescription desc{};
            desc.image = im;
            desc.format = fmt;
            desc.width = w;
            desc.height = h;
            desc.mipCount = 1;
            desc.arrayLayers = 1;
            desc.layout = layout;
            desc.state = state;
            desc.usage = usage;
            return ffxVkFsr3_3_1_5BridgeImportImage(bridge, &desc);
        };

        const VkImageUsageFlags read_usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        const VkImageUsageFlags uav_usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        FfxVkFsr3_3_1_5Resource r_color = import_img(color.image, color.format, color.width, color.height,
            VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ, read_usage);
        FfxVkFsr3_3_1_5Resource r_depth = import_img(guides.depth, VK_FORMAT_R32_SFLOAT, in_w, in_h,
            VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ, read_usage);
        FfxVkFsr3_3_1_5Resource r_motion = import_img(guides.motion, VK_FORMAT_R16G16_SFLOAT, in_w, in_h,
            VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ, read_usage);
        FfxVkFsr3_3_1_5Resource r_output = import_img(output.image, output.format, output.width, output.height,
            VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS, uav_usage);

        FfxVkFsr3_3_1_5Resource r_dil_depth = import_img(g_ctx.dilated_depth.image, g_ctx.dilated_depth.format,
            g_ctx.dilated_depth.width, g_ctx.dilated_depth.height, VK_IMAGE_LAYOUT_GENERAL,
            FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS, uav_usage);
        FfxVkFsr3_3_1_5Resource r_dil_motion = import_img(g_ctx.dilated_motion.image, g_ctx.dilated_motion.format,
            g_ctx.dilated_motion.width, g_ctx.dilated_motion.height, VK_IMAGE_LAYOUT_GENERAL,
            FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS, uav_usage);
        FfxVkFsr3_3_1_5Resource r_rec_depth = import_img(g_ctx.reconstructed_prev_depth.image, g_ctx.reconstructed_prev_depth.format,
            g_ctx.reconstructed_prev_depth.width, g_ctx.reconstructed_prev_depth.height, VK_IMAGE_LAYOUT_GENERAL,
            FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS, uav_usage);

        if (!r_color.resource || !r_depth.resource || !r_motion.resource || !r_output.resource ||
            !r_dil_depth.resource || !r_dil_motion.resource || !r_rec_depth.resource) {
            // Immediate release on pre-record import failure
            if (r_color.resource) ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, r_color);
            if (r_depth.resource) ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, r_depth);
            if (r_motion.resource) ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, r_motion);
            if (r_output.resource) ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, r_output);
            if (r_dil_depth.resource) ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, r_dil_depth);
            if (r_dil_motion.resource) ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, r_dil_motion);
            if (r_rec_depth.resource) ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, r_rec_depth);
            return false;
        }

        InFlightFrame inf;
        inf.frame_id = frame_id;
        inf.recording_serial = recording_serial;
        inf.backend = 1;
        inf.fsr3_tokens = {r_color, r_depth, r_motion, r_output, r_dil_depth, r_dil_motion, r_rec_depth};

        FfxVkFsr3_3_1_5UpscalerDispatchInfo d{};
        d.commandBuffer = cmd;
        d.color = r_color;
        d.depth = r_depth;
        d.motionVectors = r_motion;
        d.output = r_output;
        d.dilatedDepth = r_dil_depth;
        d.dilatedMotionVectors = r_dil_motion;
        d.reconstructedPrevNearestDepth = r_rec_depth;
        d.jitterOffsetX = guides.jitter_x;
        d.jitterOffsetY = guides.jitter_y;
        d.motionVectorScaleX = 1.0f;
        d.motionVectorScaleY = 1.0f;
        d.renderWidth = in_w;
        d.renderHeight = in_h;
        d.upscaleWidth = out_w;
        d.upscaleHeight = out_h;
        d.frameTimeMilliseconds = guides.frame_ms > 0.0f ? guides.frame_ms : 16.66f;
        d.preExposure = 1.0f;
        d.enableSharpening = VK_FALSE;
        d.sharpness = 0.0f;
        d.reset = guides.reset ? VK_TRUE : VK_FALSE;
        d.cameraNear = guides.camera.near_plane > 0.0f ? guides.camera.near_plane : 0.1f;
        d.cameraFar = guides.camera.far_plane > 0.0f ? guides.camera.far_plane : 1000.0f;
        d.cameraVerticalFovRadians = guides.camera.vertical_fov > 0.0f ? guides.camera.vertical_fov : 0.8f;
        d.viewSpaceToMeters = 1.0f;
        d.frameId = frame_id;

        const auto res = ffxVkFsr3_3_1_5UpscalerContextRecordDispatch(g_ctx.fsr3_ctx, &d);

        // Retain tokens in in_flight until GPU submission fence completes.
        // Even on dispatch failure, retain so the frameId retires safely.
        g_ctx.in_flight.push_back(inf);
        if (res != FFX_VK_FSR3_3_1_5_OK) {
            static unsigned failures = 0;
            if (++failures <= 5) host_log("fsr: upscaler dispatch failed (%d)", static_cast<int>(res));
            return false;
        }
        if (frame_id == 1 || frame_id % 300 == 0)
            host_log("fsr: reconstructed frame %llu", static_cast<unsigned long long>(frame_id));
        return true;
    }

    if (g_ctx.active_backend == 2) {
        // Dispatch FSR 4 source-v07
        VkResult begin = ffxFsr4VkBeginFrame(&g_ctx.fsr4_backend, frame_id);
        if (begin != VK_SUCCESS) {
            return false;
        }

        InFlightFrame inf;
        inf.frame_id = frame_id;
        inf.recording_serial = recording_serial;
        inf.backend = 2;

        auto reg = [&](VkImage im, VkImageView v, VkAccessFlags access) -> bool {
            FfxFsr4VkExternalImageState s{
                .structSize = sizeof(FfxFsr4VkExternalImageState),
                .image = im,
                .view = v,
                .layout = VK_IMAGE_LAYOUT_GENERAL,
                .stageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                .accessMask = access,
                .restoreLayout = VK_IMAGE_LAYOUT_GENERAL,
                .restoreStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                .restoreAccessMask = access,
            };
            return ffxFsr4VkSetExternalImageState(&g_ctx.fsr4_backend, &s) == VK_SUCCESS;
        };

        const VkAccessFlags read = VK_ACCESS_SHADER_READ_BIT;
        if (!reg(color.image, color.view, read) ||
            !reg(guides.depth, guides.depth_view, read) ||
            !reg(guides.motion, guides.motion_view, read) ||
            !reg(output.image, output.view, read | VK_ACCESS_SHADER_WRITE_BIT)) {
            g_ctx.in_flight.push_back(inf);
            return false;
        }

        auto resource = [](VkImageView v, uint32_t w, uint32_t h, uint32_t fmt, uint32_t state) {
            FfxApiResource r{};
            r.resource = reinterpret_cast<void*>(v);
            r.description.type = FFX_RESOURCE_TYPE_TEXTURE2D;
            r.description.format = fmt;
            r.description.width = w;
            r.description.height = h;
            r.description.depth = 1;
            r.description.mipCount = 1;
            r.state = state;
            return r;
        };

        const uint32_t ffx_color_fmt = to_ffx_surface_format(color.format);
        const uint32_t ffx_out_fmt = to_ffx_surface_format(output.format);
        if (ffx_color_fmt == FFX_SURFACE_FORMAT_UNKNOWN || ffx_out_fmt == FFX_SURFACE_FORMAT_UNKNOWN) {
            g_ctx.in_flight.push_back(inf);
            return false;
        }

        ffxDispatchDescUpscale d{};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList = reinterpret_cast<void*>(cmd);
        d.color = resource(color.view, color.width, color.height,
                           ffx_color_fmt, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.depth = resource(guides.depth_view, in_w, in_h,
                           FFX_SURFACE_FORMAT_R32_FLOAT, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.motionVectors = resource(guides.motion_view, in_w, in_h,
                                   FFX_SURFACE_FORMAT_R16G16_FLOAT, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.output = resource(output.view, output.width, output.height,
                            ffx_out_fmt, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        d.jitterOffset = {guides.jitter_x, guides.jitter_y};
        d.motionVectorScale = {1.0f, 1.0f};
        d.renderSize = {in_w, in_h};
        d.upscaleSize = {out_w, out_h};
        d.enableSharpening = false;
        d.sharpness = 0.0f;
        d.enableAutoExposure = true;
        d.frameTimeDelta = guides.frame_ms > 0.0f ? guides.frame_ms : 16.66f;
        d.preExposure = 1.0f;
        d.reset = guides.reset != 0;
        d.cameraNear = guides.camera.near_plane > 0.0f ? guides.camera.near_plane : 0.1f;
        d.cameraFar = guides.camera.far_plane > 0.0f ? guides.camera.far_plane : 1000.0f;
        d.cameraFovAngleVertical = guides.camera.vertical_fov > 0.0f ? guides.camera.vertical_fov : 0.8f;
        d.viewSpaceToMetersFactor = 1.0f;

        const auto res = ffxFsr4V07Dispatch(&g_ctx.fsr4_ctx, &d.header);
        g_ctx.in_flight.push_back(inf);
        if (res != FFX_API_RETURN_OK) {
            static unsigned failures = 0;
            if (++failures <= 5) host_log("fsr4: upscaler dispatch failed (%d)", static_cast<int>(res));
            return false;
        }
        if (frame_id == 1 || frame_id % 300 == 0)
            host_log("fsr4: reconstructed frame %llu", static_cast<unsigned long long>(frame_id));
        return true;
    }

    return false;
}

void fsr_retire_locked(uint64_t completed_submits) {
    g_ctx.retire(completed_submits);
}

void fsr_shutdown_locked() {
    g_ctx.shutdown();
}

} // namespace gpu
