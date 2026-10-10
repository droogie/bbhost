/*
 * Copyright (c) 2026 bbhost contributors
 * SPDX-License-Identifier: MIT
 *
 * Small Vulkan pixel/lifetime regression fixture for the bbhost FSR backend.
 * Exercises FSR 3.1.5 Super Resolution and FSR 3.1.6 Frame Generation using
 * independent provider public C APIs (ffx_vk_fsr3_3_1_5_bridge.h).
 */

#include "ffx_vk_fsr3_3_1_5_bridge.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct AllocatedImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct AllocatedBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & required) == required)
            return i;
    }
    return UINT32_MAX;
}

AllocatedImage create_image(VkPhysicalDevice phys, VkDevice dev, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage) {
    AllocatedImage img{};
    img.format = fmt;
    img.width = w;
    img.height = h;

    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = fmt;
    info.extent = {w, h, 1u};
    info.mipLevels = 1u;
    info.arrayLayers = 1u;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(dev, &info, nullptr, &img.image) != VK_SUCCESS)
        return {};

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(dev, img.image, &req);

    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = find_memory_type(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (alloc.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(dev, &alloc, nullptr, &img.memory) != VK_SUCCESS ||
        vkBindImageMemory(dev, img.image, img.memory, 0u) != VK_SUCCESS) {
        if (img.memory) vkFreeMemory(dev, img.memory, nullptr);
        if (img.image) vkDestroyImage(dev, img.image, nullptr);
        return {};
    }
    return img;
}

void destroy_image(VkDevice dev, AllocatedImage* img) {
    if (!img) return;
    if (img->memory) vkFreeMemory(dev, img->memory, nullptr);
    if (img->image) vkDestroyImage(dev, img->image, nullptr);
    *img = {};
}

AllocatedBuffer create_buffer(VkPhysicalDevice phys, VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memFlags) {
    AllocatedBuffer buf{};
    buf.size = size;

    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(dev, &info, nullptr, &buf.buffer) != VK_SUCCESS)
        return {};

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, buf.buffer, &req);

    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = find_memory_type(phys, req.memoryTypeBits, memFlags);
    if (alloc.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(dev, &alloc, nullptr, &buf.memory) != VK_SUCCESS ||
        vkBindBufferMemory(dev, buf.buffer, buf.memory, 0u) != VK_SUCCESS) {
        if (buf.memory) vkFreeMemory(dev, buf.memory, nullptr);
        if (buf.buffer) vkDestroyBuffer(dev, buf.buffer, nullptr);
        return {};
    }
    return buf;
}

void destroy_buffer(VkDevice dev, AllocatedBuffer* buf) {
    if (!buf) return;
    if (buf->memory) vkFreeMemory(dev, buf->memory, nullptr);
    if (buf->buffer) vkDestroyBuffer(dev, buf->buffer, nullptr);
    *buf = {};
}

uint16_t float_to_half(float val) {
    uint32_t x;
    std::memcpy(&x, &val, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127;
    uint32_t mant = x & 0x7fffff;
    if (exp > 15) return static_cast<uint16_t>(sign | 0x7c00);
    if (exp < -14) return static_cast<uint16_t>(sign);
    return static_cast<uint16_t>(sign | ((exp + 15) << 10) | (mant >> 13));
}

float half_to_float(uint16_t val) {
    uint32_t sign = (val & 0x8000) << 16;
    uint32_t exp = (val & 0x7c00) >> 10;
    uint32_t mant = (val & 0x03ff);
    uint32_t res;
    if (exp == 0) {
        if (mant == 0) {
            res = sign;
        } else {
            exp = 1;
            while (!(mant & 0x0400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ff;
            res = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        res = sign | 0x7f800000 | (mant << 13);
    } else {
        res = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &res, sizeof(float));
    return out;
}

void transition_image_layout(VkCommandBuffer cmd, AllocatedImage& img, VkImageLayout oldLayout, VkImageLayout newLayout,
                             VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.image = img.image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
}


void write_ppm(const char* filename, const uint16_t* halfRgba, uint32_t w, uint32_t h) {
    FILE* f = std::fopen(filename, "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%u %u\n255\n", w, h);
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        float r = half_to_float(halfRgba[i * 4 + 0]);
        float g = half_to_float(halfRgba[i * 4 + 1]);
        float b = half_to_float(halfRgba[i * 4 + 2]);
        rgb[i * 3 + 0] = static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f));
        rgb[i * 3 + 1] = static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f));
        rgb[i * 3 + 2] = static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f));
    }
    std::fwrite(rgb.data(), 1, rgb.size(), f);
    std::fclose(f);
}

bool has_extension(VkPhysicalDevice phys, const char* name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, nullptr) != VK_SUCCESS)
        return false;
    std::vector<VkExtensionProperties> exts(count);
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, exts.data()) != VK_SUCCESS)
        return false;
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0)
            return true;
    }
    return false;
}

} // namespace

int main() {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "fsr_provider_check";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instInfo.pApplicationInfo = &appInfo;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS) {
        std::fprintf(stderr, "Skipping: failed to create Vulkan instance\n");
        return 77;
    }

    uint32_t physCount = 0;
    if (vkEnumeratePhysicalDevices(instance, &physCount, nullptr) != VK_SUCCESS || physCount == 0) {
        std::fprintf(stderr, "Skipping: no Vulkan physical devices found\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    std::vector<VkPhysicalDevice> physicals(physCount);
    vkEnumeratePhysicalDevices(instance, &physCount, physicals.data());

    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t queueFamily = UINT32_MAX;

    for (VkPhysicalDevice p : physicals) {
        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(p, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfProps(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(p, &qfCount, qfProps.data());
        for (uint32_t i = 0; i < qfCount; ++i) {
            if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                physical = p;
                queueFamily = i;
                break;
            }
        }
        if (physical != VK_NULL_HANDLE) break;
    }

    if (physical == VK_NULL_HANDLE) {
        std::fprintf(stderr, "Skipping: no suitable Vulkan compute/graphics queue family\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    VkPhysicalDeviceProperties devProps{};
    vkGetPhysicalDeviceProperties(physical, &devProps);
    std::printf("Testing Vulkan device: %s (driver version 0x%x)\n", devProps.deviceName, devProps.driverVersion);

    // Feature queries matching bbhost gpu.cpp init_locked()
    VkPhysicalDevice16BitStorageFeatures q16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceVulkan12Features q12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features q13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR supportedDerivatives{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};

    VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    query.pNext = &q12;
    q12.pNext = &q13;
    q13.pNext = &q16;
    q16.pNext = &supportedDerivatives;

    vkGetPhysicalDeviceFeatures2(physical, &query);

    if (!query.features.shaderStorageImageReadWithoutFormat ||
        !query.features.shaderStorageImageWriteWithoutFormat) {
        std::fprintf(stderr, "Skipping: device lacks shaderStorageImageRead/WriteWithoutFormat\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    VkPhysicalDevice16BitStorageFeatures enable16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    enable16.storageBuffer16BitAccess = q16.storageBuffer16BitAccess;
    enable16.uniformAndStorageBuffer16BitAccess = q16.uniformAndStorageBuffer16BitAccess;
    enable16.storagePushConstant16 = q16.storagePushConstant16;
    enable16.storageInputOutput16 = q16.storageInputOutput16;

    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enable12.shaderInt8 = q12.shaderInt8;
    enable12.storageBuffer8BitAccess = q12.storageBuffer8BitAccess;
    enable12.uniformAndStorageBuffer8BitAccess = q12.uniformAndStorageBuffer8BitAccess;
    enable12.shaderSubgroupExtendedTypes = q12.shaderSubgroupExtendedTypes;
    enable12.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceVulkan13Features enable13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    enable13.shaderIntegerDotProduct = q13.shaderIntegerDotProduct;

    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR enableDerivatives{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};

    std::vector<const char*> devExts;
    if (has_extension(physical, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME) &&
        supportedDerivatives.computeDerivativeGroupLinear) {
        enableDerivatives.computeDerivativeGroupLinear = VK_TRUE;
        devExts.push_back(VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
    }

    VkPhysicalDeviceFeatures2 enable2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    enable2.features.shaderStorageImageReadWithoutFormat = VK_TRUE;
    enable2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    enable2.pNext = &enable12;
    enable12.pNext = &enable13;
    enable13.pNext = &enable16;
    if (enableDerivatives.computeDerivativeGroupLinear) {
        enable16.pNext = &enableDerivatives;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1u;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devInfo.queueCreateInfoCount = 1u;
    devInfo.pQueueCreateInfos = &queueInfo;
    devInfo.pNext = &enable2;
    devInfo.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    devInfo.ppEnabledExtensionNames = devExts.empty() ? nullptr : devExts.data();

    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &devInfo, nullptr, &device) != VK_SUCCESS) {
        std::fprintf(stderr, "Skipping: failed to create Vulkan device\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queueFamily, 0u, &queue);

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = queueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool) != VK_SUCCESS) {
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    // Extents: Render 640x360, Upscale 960x540
    const uint32_t rw = 640u, rh = 360u;
    const uint32_t ow = 960u, oh = 540u;

    const VkImageUsageFlags usageSampledStorage =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    // Initialise FSR 3.1.5 Upscaler Context
    FfxVkFsr3_3_1_5UpscalerCreateInfo srCreate{};
    srCreate.physicalDevice = physical;
    srCreate.device = device;
    srCreate.maxRenderWidth = rw;
    srCreate.maxRenderHeight = rh;
    srCreate.maxUpscaleWidth = ow;
    srCreate.maxUpscaleHeight = oh;
    srCreate.hdrColorInput = VK_TRUE;
    srCreate.autoExposure = VK_FALSE;

    FfxVkFsr3_3_1_5UpscalerContext* srContext = nullptr;
    if (ffxVkFsr3_3_1_5UpscalerContextCreate(&srCreate, &srContext) != FFX_VK_FSR3_3_1_5_OK || !srContext) {
        std::fprintf(stderr, "Failed to create FSR 3.1.5 upscaler context\n");
        return 1;
    }

    FfxVkFsr3_3_1_5Bridge* srBridge = ffxVkFsr3_3_1_5UpscalerContextGetBridge(srContext);
    FfxVkFsr3_3_1_5SharedResourceDescriptions sharedDesc{};
    ffxVkFsr3_3_1_5UpscalerContextGetSharedResourceDescriptions(srContext, &sharedDesc);

    // Initialise FSR 3.1.6 Frame Generation Context
    FfxVkFsr3_3_1_6FrameGenerationCreateInfo fgCreate{};
    fgCreate.physicalDevice = physical;
    fgCreate.device = device;
    fgCreate.maxRenderWidth = rw;
    fgCreate.maxRenderHeight = rh;
    fgCreate.displayWidth = ow;
    fgCreate.displayHeight = oh;
    fgCreate.colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

    FfxVkFsr3_3_1_6FrameGenerationContext* fgContext = nullptr;
    if (ffxVkFsr3_3_1_6FrameGenerationContextCreate(&fgCreate, &fgContext) != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK || !fgContext) {
        std::fprintf(stderr, "Failed to create FSR 3.1.6 frame generation context\n");
        return 1;
    }

    const VkDeviceSize colorBytes = static_cast<VkDeviceSize>(rw) * rh * 4 * sizeof(uint16_t);
    const VkDeviceSize depthBytes = static_cast<VkDeviceSize>(rw) * rh * sizeof(float);
    const VkDeviceSize motionBytes = static_cast<VkDeviceSize>(rw) * rh * 2 * sizeof(uint16_t);
    const VkDeviceSize maskBytes = static_cast<VkDeviceSize>(rw) * rh * sizeof(uint8_t);
    const VkDeviceSize uploadSliceBytes = colorBytes + depthBytes + motionBytes + maskBytes * 2;
    const VkDeviceSize outBytes = static_cast<VkDeviceSize>(ow) * oh * 4 * sizeof(uint16_t);

    struct Slot {
        AllocatedImage color;
        AllocatedImage depth;
        AllocatedImage motion;
        AllocatedImage reactive;
        AllocatedImage transparency;
        AllocatedImage dilatedDepth;
        AllocatedImage dilatedMotion;
        AllocatedImage prevDepth;
        AllocatedImage output;
        AllocatedImage fgOutput;

        AllocatedBuffer uploadStaging;
        AllocatedBuffer readbackStaging;

        VkFence fence = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        std::vector<FfxVkFsr3_3_1_5Resource> importedTokens;
    };

    Slot slots[2]{};

    for (int s = 0; s < 2; ++s) {
        slots[s].color = create_image(physical, device, rw, rh, VK_FORMAT_R16G16B16A16_SFLOAT, usageSampledStorage);
        slots[s].depth = create_image(physical, device, rw, rh, VK_FORMAT_R32_SFLOAT, usageSampledStorage);
        slots[s].motion = create_image(physical, device, rw, rh, VK_FORMAT_R16G16_SFLOAT, usageSampledStorage);
        slots[s].reactive = create_image(physical, device, rw, rh, VK_FORMAT_R8_UNORM, usageSampledStorage);
        slots[s].transparency = create_image(physical, device, rw, rh, VK_FORMAT_R8_UNORM, usageSampledStorage);
        slots[s].dilatedDepth = create_image(physical, device, sharedDesc.dilatedDepth.width, sharedDesc.dilatedDepth.height,
                                             sharedDesc.dilatedDepth.format, usageSampledStorage);
        slots[s].dilatedMotion = create_image(physical, device, sharedDesc.dilatedMotionVectors.width, sharedDesc.dilatedMotionVectors.height,
                                              sharedDesc.dilatedMotionVectors.format, usageSampledStorage);
        slots[s].prevDepth = create_image(physical, device, sharedDesc.reconstructedPrevNearestDepth.width, sharedDesc.reconstructedPrevNearestDepth.height,
                                          sharedDesc.reconstructedPrevNearestDepth.format, usageSampledStorage);
        slots[s].output = create_image(physical, device, ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT, usageSampledStorage);
        slots[s].fgOutput = create_image(physical, device, ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT, usageSampledStorage);

        slots[s].uploadStaging = create_buffer(physical, device, uploadSliceBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        slots[s].readbackStaging = create_buffer(physical, device, outBytes * 2,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device, &fenceInfo, nullptr, &slots[s].fence);

        VkCommandBufferAllocateInfo cmdAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cmdAlloc.commandPool = cmdPool;
        cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAlloc.commandBufferCount = 1u;
        vkAllocateCommandBuffers(device, &cmdAlloc, &slots[s].cmd);
    }

    auto importImage = [&](FfxVkFsr3_3_1_5Bridge* bridge, const AllocatedImage& img, VkImageLayout layout, uint32_t state) -> FfxVkFsr3_3_1_5Resource {
        FfxVkFsr3_3_1_5ImportedImageDescription desc{};
        desc.image = img.image;
        desc.format = img.format;
        desc.width = img.width;
        desc.height = img.height;
        desc.mipCount = 1u;
        desc.arrayLayers = 1u;
        desc.layout = layout;
        desc.state = state;
        desc.usage = usageSampledStorage;
        return ffxVkFsr3_3_1_5BridgeImportImage(bridge, &desc);
    };

    auto fgImageInfo = [&](const AllocatedImage& img, VkImageLayout layout) -> FfxVkFsr3_3_1_6FrameGenerationImage {
        FfxVkFsr3_3_1_6FrameGenerationImage res{};
        res.image = img.image;
        res.format = img.format;
        res.width = img.width;
        res.height = img.height;
        res.layout = layout;
        res.usage = usageSampledStorage;
        return res;
    };

    // 16 consecutive frames (warming history >= 8 frames)
    const int totalFrames = 16;
    const int rectW = 120, rectH = 80;
    const int rectVy = 0;
    const int rectVx = 16; // 16 pixels displacement per frame

    std::vector<uint16_t> prevRealImage(ow * oh * 4, 0);
    std::vector<uint16_t> currentRealImage(ow * oh * 4, 0);
    std::vector<uint16_t> currentFgImage(ow * oh * 4, 0);

    for (int frame = 0; frame < totalFrames; ++frame) {
        const uint64_t frameId = static_cast<uint64_t>(frame + 1);
        const int slotIdx = frame % 2;
        Slot& slot = slots[slotIdx];

        // Wait on fence for this slot before CPU touches staging buffer or commands
        vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &slot.fence);

        for (const auto& token : slot.importedTokens) {
            ffxVkFsr3_3_1_5BridgeReleaseImportedImage(srBridge, token);
        }
        slot.importedTokens.clear();

        // Retire completed frame on contexts
        if (frame >= 2) {
            const uint64_t retireId = static_cast<uint64_t>(frame - 1);
            ffxVkFsr3_3_1_5UpscalerContextRetireFrame(srContext, retireId);
            ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(fgContext, retireId);
        }

        // Moving rectangle: rectX moves across frames
        const int rectX = 40 + (frame * rectVx) % (static_cast<int>(rw) - rectW - 40);
        const int rectY = 140;

        void* mapped = nullptr;
        vkMapMemory(device, slot.uploadStaging.memory, 0, uploadSliceBytes, 0, &mapped);
        uint16_t* pColor = static_cast<uint16_t*>(mapped);
        float* pDepth = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(mapped) + colorBytes);
        uint16_t* pMotion = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(mapped) + colorBytes + depthBytes);
        uint8_t* pReactive = reinterpret_cast<uint8_t*>(mapped) + colorBytes + depthBytes + motionBytes;
        uint8_t* pTransparency = pReactive + maskBytes;

        // Zero out reactive and transparency masks
        std::memset(pReactive, 0, maskBytes);
        std::memset(pTransparency, 0, maskBytes);

        // FSR standard motion vectors point from current pixel position to previous pixel position: (-rectVx, -rectVy)
        const float mvX = static_cast<float>(-rectVx);
        const float mvY = static_cast<float>(-rectVy);

        for (uint32_t y = 0; y < rh; ++y) {
            for (uint32_t x = 0; x < rw; ++x) {
                const size_t idx = y * rw + x;
                const bool inRect = (static_cast<int>(x) >= rectX && static_cast<int>(x) < rectX + rectW &&
                                     static_cast<int>(y) >= rectY && static_cast<int>(y) < rectY + rectH);
                if (inRect) {
                    // Foreground moving rectangle (bright cyan)
                    pColor[idx * 4 + 0] = float_to_half(0.1f);
                    pColor[idx * 4 + 1] = float_to_half(0.9f);
                    pColor[idx * 4 + 2] = float_to_half(0.9f);
                    pColor[idx * 4 + 3] = float_to_half(1.0f);
                    pDepth[idx] = 0.35f;
                    pMotion[idx * 2 + 0] = float_to_half(mvX);
                    pMotion[idx * 2 + 1] = float_to_half(mvY);
                } else {
                    // Background
                    pColor[idx * 4 + 0] = float_to_half(0.05f);
                    pColor[idx * 4 + 1] = float_to_half(0.05f);
                    pColor[idx * 4 + 2] = float_to_half(0.08f);
                    pColor[idx * 4 + 3] = float_to_half(1.0f);
                    pDepth[idx] = 0.98f;
                    pMotion[idx * 2 + 0] = float_to_half(0.0f);
                    pMotion[idx * 2 + 1] = float_to_half(0.0f);
                }
            }
        }
        vkUnmapMemory(device, slot.uploadStaging.memory);

        VkCommandBuffer cmd = slot.cmd;
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        auto copyBufToImage = [&](VkDeviceSize offset, AllocatedImage& dstImg, VkImageLayout finalLayout) {
            transition_image_layout(cmd, dstImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    0, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkBufferImageCopy region{};
            region.bufferOffset = offset;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {dstImg.width, dstImg.height, 1u};
            vkCmdCopyBufferToImage(cmd, slot.uploadStaging.buffer, dstImg.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            transition_image_layout(cmd, dstImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, finalLayout,
                                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        };

        copyBufToImage(0, slot.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        copyBufToImage(colorBytes, slot.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        copyBufToImage(colorBytes + depthBytes, slot.motion, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        copyBufToImage(colorBytes + depthBytes + motionBytes, slot.reactive, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        copyBufToImage(colorBytes + depthBytes + motionBytes + maskBytes, slot.transparency, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        transition_image_layout(cmd, slot.dilatedDepth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
        transition_image_layout(cmd, slot.dilatedMotion, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
        transition_image_layout(cmd, slot.prevDepth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
        transition_image_layout(cmd, slot.output, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
        transition_image_layout(cmd, slot.fgOutput, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);

        // 1. Dispatch FSR 3.1.5 Upscaler
        FfxVkFsr3_3_1_5Resource rColor = importImage(srBridge, slot.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
        FfxVkFsr3_3_1_5Resource rDepth = importImage(srBridge, slot.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
        FfxVkFsr3_3_1_5Resource rMotion = importImage(srBridge, slot.motion, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
        FfxVkFsr3_3_1_5Resource rReactive = importImage(srBridge, slot.reactive, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
        FfxVkFsr3_3_1_5Resource rTransparency = importImage(srBridge, slot.transparency, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
        FfxVkFsr3_3_1_5Resource rDilatedDepth = importImage(srBridge, slot.dilatedDepth, VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
        FfxVkFsr3_3_1_5Resource rDilatedMotion = importImage(srBridge, slot.dilatedMotion, VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
        FfxVkFsr3_3_1_5Resource rPrevDepth = importImage(srBridge, slot.prevDepth, VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
        FfxVkFsr3_3_1_5Resource rOutput = importImage(srBridge, slot.output, VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);

        slot.importedTokens = {rColor, rDepth, rMotion, rReactive, rTransparency, rDilatedDepth, rDilatedMotion, rPrevDepth, rOutput};

        FfxVkFsr3_3_1_5UpscalerDispatchInfo srDispatch{};
        srDispatch.commandBuffer = cmd;
        srDispatch.color = rColor;
        srDispatch.depth = rDepth;
        srDispatch.motionVectors = rMotion;
        srDispatch.reactive = rReactive;
        srDispatch.transparencyAndComposition = rTransparency;
        srDispatch.dilatedDepth = rDilatedDepth;
        srDispatch.dilatedMotionVectors = rDilatedMotion;
        srDispatch.reconstructedPrevNearestDepth = rPrevDepth;
        srDispatch.output = rOutput;
        srDispatch.jitterOffsetX = 0.0f;
        srDispatch.jitterOffsetY = 0.0f;
        srDispatch.motionVectorScaleX = static_cast<float>(rw);
        srDispatch.motionVectorScaleY = static_cast<float>(rh);
        srDispatch.renderWidth = rw;
        srDispatch.renderHeight = rh;
        srDispatch.upscaleWidth = ow;
        srDispatch.upscaleHeight = oh;
        srDispatch.frameTimeMilliseconds = 16.6667f;
        srDispatch.preExposure = 1.0f;
        srDispatch.enableSharpening = VK_TRUE;
        srDispatch.sharpness = 0.5f;
        srDispatch.reset = (frame == 0) ? VK_TRUE : VK_FALSE;
        srDispatch.cameraNear = 0.1f;
        srDispatch.cameraFar = 1000.0f;
        srDispatch.cameraVerticalFovRadians = 1.0471975512f;
        srDispatch.viewSpaceToMeters = 1.0f;
        srDispatch.frameId = frameId;

        if (ffxVkFsr3_3_1_5UpscalerContextRecordDispatch(srContext, &srDispatch) != FFX_VK_FSR3_3_1_5_OK) {
            std::fprintf(stderr, "Failed to record FSR 3.1.5 upscaler dispatch on frame %d\n", frame);
            return 1;
        }

        // Barrier between SR write to slot.output and FG read
        transition_image_layout(cmd, slot.output, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // 2. Dispatch FSR 3.1.6 Frame Generation (Prepare + Dispatch)
        FfxVkFsr3_3_1_6FrameGenerationPrepareInfo fgPrep{};
        fgPrep.commandBuffer = cmd;
        fgPrep.color = fgImageInfo(slot.output, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        fgPrep.depth = fgImageInfo(slot.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        fgPrep.motionVectors = fgImageInfo(slot.motion, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        fgPrep.renderWidth = rw;
        fgPrep.renderHeight = rh;
        fgPrep.motionVectorScaleX = static_cast<float>(rw);
        fgPrep.motionVectorScaleY = static_cast<float>(rh);
        fgPrep.frameTimeMilliseconds = 16.6667f;
        fgPrep.minLuminance = 0.0f;
        fgPrep.maxLuminance = 1.0f;
        fgPrep.transferFunction = FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
        fgPrep.cameraNear = 0.1f;
        fgPrep.cameraFar = 1000.0f;
        fgPrep.viewSpaceToMeters = 1.0f;
        fgPrep.cameraVerticalFovRadians = 1.0471975512f;
        fgPrep.cameraUp[1] = 1.0f;
        fgPrep.cameraRight[0] = 1.0f;
        fgPrep.cameraForward[2] = -1.0f;
        fgPrep.frameId = frameId;
        fgPrep.reset = (frame == 0) ? VK_TRUE : VK_FALSE;

        if (ffxVkFsr3_3_1_6FrameGenerationContextRecordPrepare(fgContext, &fgPrep) != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
            std::fprintf(stderr, "Failed to record FSR 3.1.6 frame generation prepare on frame %d\n", frame);
            return 1;
        }

        FfxVkFsr3_3_1_6FrameGenerationDispatchInfo fgDisp{};
        fgDisp.commandBuffer = cmd;
        fgDisp.color = fgPrep.color;
        fgDisp.output = fgImageInfo(slot.fgOutput, VK_IMAGE_LAYOUT_GENERAL);
        fgDisp.displayWidth = ow;
        fgDisp.displayHeight = oh;
        fgDisp.frameTimeMilliseconds = 16.6667f;
        fgDisp.cameraNear = 0.1f;
        fgDisp.cameraFar = 1000.0f;
        fgDisp.viewSpaceToMeters = 1.0f;
        fgDisp.cameraVerticalFovRadians = 1.0471975512f;
        fgDisp.maxLuminance = 1.0f;
        fgDisp.transferFunction = FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
        fgDisp.frameId = frameId;
        fgDisp.reset = (frame == 0) ? VK_TRUE : VK_FALSE;

        if (ffxVkFsr3_3_1_6FrameGenerationContextRecordDispatch(fgContext, &fgDisp) != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
            std::fprintf(stderr, "Failed to record FSR 3.1.6 frame generation dispatch on frame %d\n", frame);
            return 1;
        }

        // Read back on the last 2 frames so we have both previous and current real endpoints
        if (frame >= totalFrames - 2) {
            transition_image_layout(cmd, slot.output, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            transition_image_layout(cmd, slot.fgOutput, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

            VkBufferImageCopy regionReal{};
            regionReal.bufferOffset = 0;
            regionReal.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            regionReal.imageExtent = {ow, oh, 1u};
            vkCmdCopyImageToBuffer(cmd, slot.output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.readbackStaging.buffer, 1, &regionReal);

            VkBufferImageCopy regionFg{};
            regionFg.bufferOffset = outBytes;
            regionFg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            regionFg.imageExtent = {ow, oh, 1u};
            vkCmdCopyImageToBuffer(cmd, slot.fgOutput.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.readbackStaging.buffer, 1, &regionFg);
        }

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1u;
        submit.pCommandBuffers = &cmd;
        if (vkQueueSubmit(queue, 1u, &submit, slot.fence) != VK_SUCCESS) {
            std::fprintf(stderr, "Failed to submit frame %d\n", frame);
            return 1;
        }

        if (frame == totalFrames - 2) {
            vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
            void* m = nullptr;
            vkMapMemory(device, slot.readbackStaging.memory, 0, outBytes, 0, &m);
            std::memcpy(prevRealImage.data(), m, outBytes);
            vkUnmapMemory(device, slot.readbackStaging.memory);
        } else if (frame == totalFrames - 1) {
            vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
            void* m = nullptr;
            vkMapMemory(device, slot.readbackStaging.memory, 0, outBytes * 2, 0, &m);
            std::memcpy(currentRealImage.data(), m, outBytes);
            std::memcpy(currentFgImage.data(), reinterpret_cast<uint8_t*>(m) + outBytes, outBytes);
            vkUnmapMemory(device, slot.readbackStaging.memory);
        }
    }

    vkQueueWaitIdle(queue);

    // Pixel content validation
    size_t pixelCount = static_cast<size_t>(ow) * oh;
    double sumReal = 0.0, sumFg = 0.0;
    size_t finiteReal = 0, finiteFg = 0;
    size_t diffCurrentCount = 0, diffPrevCount = 0;
    double maxDiffCurrent = 0.0, maxDiffPrev = 0.0;

    for (size_t i = 0; i < pixelCount * 4; ++i) {
        float curR = half_to_float(currentRealImage[i]);
        float prevR = half_to_float(prevRealImage[i]);
        float fg = half_to_float(currentFgImage[i]);

        if (std::isfinite(curR)) {
            finiteReal++;
            sumReal += std::abs(curR);
        }
        if (std::isfinite(fg)) {
            finiteFg++;
            sumFg += std::abs(fg);
        }

        float diffCur = std::abs(curR - fg);
        if (diffCur > 0.02f) {
            diffCurrentCount++;
            if (diffCur > maxDiffCurrent) maxDiffCurrent = diffCur;
        }

        float diffPrev = std::abs(prevR - fg);
        if (diffPrev > 0.02f) {
            diffPrevCount++;
            if (diffPrev > maxDiffPrev) maxDiffPrev = diffPrev;
        }
    }

    std::printf("Readback Verification:\n");
    std::printf("  Real SR output: %zu / %zu finite floats, sum = %.2f\n", finiteReal, pixelCount * 4, sumReal);
    std::printf("  FG output:      %zu / %zu finite floats, sum = %.2f\n", finiteFg, pixelCount * 4, sumFg);
    std::printf("  Diff vs Current Real: %zu components (max diff: %.4f)\n", diffCurrentCount, maxDiffCurrent);
    std::printf("  Diff vs Previous Real: %zu components (max diff: %.4f)\n", diffPrevCount, maxDiffPrev);

    if (finiteReal != pixelCount * 4 || finiteFg != pixelCount * 4) {
        std::fprintf(stderr, "FAIL: Non-finite (NaN or Inf) pixels detected in SR or FG output!\n");
        return 1;
    }
    if (sumReal < 100.0 || sumFg < 100.0) {
        std::fprintf(stderr, "FAIL: Output images appear empty or black (sum too low)!\n");
        return 1;
    }
    if (diffCurrentCount < 100) {
        std::fprintf(stderr, "FAIL: Generated FG frame is identical to current real endpoint!\n");
        return 1;
    }
    if (diffPrevCount < 100) {
        std::fprintf(stderr, "FAIL: Generated FG frame is identical to previous real endpoint!\n");
        return 1;
    }

    write_ppm("build-dev/fsr_prev_real.ppm", prevRealImage.data(), ow, oh);
    write_ppm("build-dev/fsr_current_real.ppm", currentRealImage.data(), ow, oh);
    write_ppm("build-dev/fsr_gen_interpolated.ppm", currentFgImage.data(), ow, oh);

    // Spatial Moving Rectangle Midpoint Verification:
    // Cyan foreground rectangle has high green & blue channels (> 0.35f); background is ~0.05f-0.08f.
    auto computeCentroidX = [&](const std::vector<uint16_t>& img) -> double {
        double weightedX = 0.0;
        double totalWeight = 0.0;
        for (uint32_t y = 0; y < oh; ++y) {
            for (uint32_t x = 0; x < ow; ++x) {
                size_t p = (y * ow + x) * 4;
                float g = half_to_float(img[p + 1]);
                float b = half_to_float(img[p + 2]);
                float bright = (g + b) * 0.5f;
                if (bright > 0.35f) {
                    weightedX += static_cast<double>(x) * bright;
                    totalWeight += bright;
                }
            }
        }
        return totalWeight > 0.0 ? (weightedX / totalWeight) : 0.0;
    };

    double prevCentroidX = computeCentroidX(prevRealImage);
    double currentCentroidX = computeCentroidX(currentRealImage);
    double fgCentroidX = computeCentroidX(currentFgImage);

    std::printf("Centroid Analysis:\n");
    std::printf("  Previous Real Centroid X: %.2f\n", prevCentroidX);
    std::printf("  Generated FG Centroid X:  %.2f\n", fgCentroidX);
    std::printf("  Current Real Centroid X:  %.2f\n", currentCentroidX);

    if (prevCentroidX <= 0.0 || currentCentroidX <= 0.0 || fgCentroidX <= 0.0) {
        std::fprintf(stderr, "FAIL: Moving rectangle not found in output frames!\n");
        return 1;
    }

    // Centroid of interpolated frame must strictly lie between previous and current real frames
    if (!(fgCentroidX > prevCentroidX && fgCentroidX < currentCentroidX)) {
        std::fprintf(stderr, "FAIL: Generated FG centroid (%.2f) does NOT lie between previous (%.2f) and current (%.2f)!\n",
                     fgCentroidX, prevCentroidX, currentCentroidX);
        return 1;
    }

    // Centroid should be close to halfway between previous and current
    double expectedMidX = (prevCentroidX + currentCentroidX) * 0.5;
    double midDelta = std::abs(fgCentroidX - expectedMidX);
    std::printf("  FG Midpoint Delta from expected (%.2f): %.2f pixels\n", expectedMidX, midDelta);
    if (midDelta > 6.0) {
        std::fprintf(stderr, "FAIL: Generated FG centroid is too far from true midpoint!\n");
        return 1;
    }

    // Bounded Background Error:
    // Sample a large background region away from the moving rectangle (y in [20, 100], x in [20, ow - 20])
    double bgErrorSum = 0.0;
    size_t bgCount = 0;
    for (uint32_t y = 20; y < 100; ++y) {
        for (uint32_t x = 20; x < ow - 20; ++x) {
            size_t p = (y * ow + x) * 4;
            float fgG = half_to_float(currentFgImage[p + 1]);
            float curG = half_to_float(currentRealImage[p + 1]);
            bgErrorSum += std::abs(fgG - curG);
            bgCount++;
        }
    }
    double meanBgError = bgErrorSum / bgCount;
    std::printf("  Mean background error between FG and Real: %.4f\n", meanBgError);
    if (meanBgError > 0.05) {
        std::fprintf(stderr, "FAIL: Background in generated frame deviated too much from real background!\n");
        return 1;
    }

    std::printf("PASS: Output images are non-empty, finite, and FG produces genuine midpoint content distinct from both endpoints with bounded background error.\n");

    // Resize & retirement (< 100 frames bounded lifecycle check)
    std::printf("Testing resize lifecycle (640x360 -> 480x270 -> 720x405)...\n");
    {
        for (int s = 0; s < 2; ++s) {
            vkWaitForFences(device, 1, &slots[s].fence, VK_TRUE, UINT64_MAX);
            for (const auto& token : slots[s].importedTokens) {
                ffxVkFsr3_3_1_5BridgeReleaseImportedImage(srBridge, token);
            }
            slots[s].importedTokens.clear();
        }
        ffxVkFsr3_3_1_5UpscalerContextDestroy(srContext);
        ffxVkFsr3_3_1_6FrameGenerationContextDestroy(fgContext);

        const uint32_t rw2 = 480u, rh2 = 270u;
        const uint32_t ow2 = 720u, oh2 = 405u;

        srCreate.maxRenderWidth = rw2;
        srCreate.maxRenderHeight = rh2;
        srCreate.maxUpscaleWidth = ow2;
        srCreate.maxUpscaleHeight = oh2;
        if (ffxVkFsr3_3_1_5UpscalerContextCreate(&srCreate, &srContext) != FFX_VK_FSR3_3_1_5_OK || !srContext) {
            std::fprintf(stderr, "FAIL: recreate SR context with new dimensions\n");
            return 1;
        }
        srBridge = ffxVkFsr3_3_1_5UpscalerContextGetBridge(srContext);
        ffxVkFsr3_3_1_5UpscalerContextGetSharedResourceDescriptions(srContext, &sharedDesc);

        fgCreate.maxRenderWidth = rw2;
        fgCreate.maxRenderHeight = rh2;
        fgCreate.displayWidth = ow2;
        fgCreate.displayHeight = oh2;
        if (ffxVkFsr3_3_1_6FrameGenerationContextCreate(&fgCreate, &fgContext) != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK || !fgContext) {
            std::fprintf(stderr, "FAIL: recreate FG context with new dimensions\n");
            return 1;
        }

        const VkDeviceSize outBytes2 = static_cast<VkDeviceSize>(ow2) * oh2 * 4 * sizeof(uint16_t);

        // Recreate slot images with resized extents
        for (int s = 0; s < 2; ++s) {
            destroy_image(device, &slots[s].color);
            destroy_image(device, &slots[s].depth);
            destroy_image(device, &slots[s].motion);
            destroy_image(device, &slots[s].reactive);
            destroy_image(device, &slots[s].transparency);
            destroy_image(device, &slots[s].dilatedDepth);
            destroy_image(device, &slots[s].dilatedMotion);
            destroy_image(device, &slots[s].prevDepth);
            destroy_image(device, &slots[s].output);
            destroy_image(device, &slots[s].fgOutput);
            destroy_buffer(device, &slots[s].readbackStaging);

            slots[s].color = create_image(physical, device, rw2, rh2, VK_FORMAT_R16G16B16A16_SFLOAT, usageSampledStorage);
            slots[s].depth = create_image(physical, device, rw2, rh2, VK_FORMAT_R32_SFLOAT, usageSampledStorage);
            slots[s].motion = create_image(physical, device, rw2, rh2, VK_FORMAT_R16G16_SFLOAT, usageSampledStorage);
            slots[s].reactive = create_image(physical, device, rw2, rh2, VK_FORMAT_R8_UNORM, usageSampledStorage);
            slots[s].transparency = create_image(physical, device, rw2, rh2, VK_FORMAT_R8_UNORM, usageSampledStorage);
            slots[s].dilatedDepth = create_image(physical, device, sharedDesc.dilatedDepth.width, sharedDesc.dilatedDepth.height,
                                                 sharedDesc.dilatedDepth.format, usageSampledStorage);
            slots[s].dilatedMotion = create_image(physical, device, sharedDesc.dilatedMotionVectors.width, sharedDesc.dilatedMotionVectors.height,
                                                  sharedDesc.dilatedMotionVectors.format, usageSampledStorage);
            slots[s].prevDepth = create_image(physical, device, sharedDesc.reconstructedPrevNearestDepth.width, sharedDesc.reconstructedPrevNearestDepth.height,
                                              sharedDesc.reconstructedPrevNearestDepth.format, usageSampledStorage);
            slots[s].output = create_image(physical, device, ow2, oh2, VK_FORMAT_R16G16B16A16_SFLOAT, usageSampledStorage);
            slots[s].fgOutput = create_image(physical, device, ow2, oh2, VK_FORMAT_R16G16B16A16_SFLOAT, usageSampledStorage);
            slots[s].readbackStaging = create_buffer(physical, device, outBytes2,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }

        for (int f = 0; f < 4; ++f) {
            const uint64_t fid = static_cast<uint64_t>(100 + f);
            const int s = f % 2;
            vkWaitForFences(device, 1, &slots[s].fence, VK_TRUE, UINT64_MAX);
            vkResetFences(device, 1, &slots[s].fence);

            for (const auto& token : slots[s].importedTokens) {
                ffxVkFsr3_3_1_5BridgeReleaseImportedImage(srBridge, token);
            }
            slots[s].importedTokens.clear();

            if (f >= 2) {
                ffxVkFsr3_3_1_5UpscalerContextRetireFrame(srContext, 100 + f - 2);
                ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(fgContext, 100 + f - 2);
            }

            VkCommandBuffer cmd = slots[s].cmd;
            vkResetCommandBuffer(cmd, 0);
            VkCommandBufferBeginInfo bInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(cmd, &bInfo) != VK_SUCCESS) {
                std::fprintf(stderr, "FAIL: vkBeginCommandBuffer on resized frame %d\n", f);
                return 1;
            }

            transition_image_layout(cmd, slots[s].color, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    0, VK_ACCESS_SHADER_READ_BIT);
            transition_image_layout(cmd, slots[s].depth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    0, VK_ACCESS_SHADER_READ_BIT);
            transition_image_layout(cmd, slots[s].motion, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    0, VK_ACCESS_SHADER_READ_BIT);
            transition_image_layout(cmd, slots[s].reactive, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    0, VK_ACCESS_SHADER_READ_BIT);
            transition_image_layout(cmd, slots[s].transparency, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    0, VK_ACCESS_SHADER_READ_BIT);
            transition_image_layout(cmd, slots[s].dilatedDepth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                    0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
            transition_image_layout(cmd, slots[s].dilatedMotion, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                    0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
            transition_image_layout(cmd, slots[s].prevDepth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                    0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
            transition_image_layout(cmd, slots[s].output, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                    0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
            transition_image_layout(cmd, slots[s].fgOutput, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                    0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);

            FfxVkFsr3_3_1_5Resource rColor = importImage(srBridge, slots[s].color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
            FfxVkFsr3_3_1_5Resource rDepth = importImage(srBridge, slots[s].depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
            FfxVkFsr3_3_1_5Resource rMotion = importImage(srBridge, slots[s].motion, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
            FfxVkFsr3_3_1_5Resource rReactive = importImage(srBridge, slots[s].reactive, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
            FfxVkFsr3_3_1_5Resource rTransparency = importImage(srBridge, slots[s].transparency, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
            FfxVkFsr3_3_1_5Resource rDilatedDepth = importImage(srBridge, slots[s].dilatedDepth, VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
            FfxVkFsr3_3_1_5Resource rDilatedMotion = importImage(srBridge, slots[s].dilatedMotion, VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
            FfxVkFsr3_3_1_5Resource rPrevDepth = importImage(srBridge, slots[s].prevDepth, VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
            FfxVkFsr3_3_1_5Resource rOutput = importImage(srBridge, slots[s].output, VK_IMAGE_LAYOUT_GENERAL, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);

            slots[s].importedTokens = {rColor, rDepth, rMotion, rReactive, rTransparency, rDilatedDepth, rDilatedMotion, rPrevDepth, rOutput};

            FfxVkFsr3_3_1_5UpscalerDispatchInfo srDisp{};
            srDisp.commandBuffer = cmd;
            srDisp.color = rColor;
            srDisp.depth = rDepth;
            srDisp.motionVectors = rMotion;
            srDisp.reactive = rReactive;
            srDisp.transparencyAndComposition = rTransparency;
            srDisp.dilatedDepth = rDilatedDepth;
            srDisp.dilatedMotionVectors = rDilatedMotion;
            srDisp.reconstructedPrevNearestDepth = rPrevDepth;
            srDisp.output = rOutput;
            srDisp.renderWidth = rw2;
            srDisp.renderHeight = rh2;
            srDisp.upscaleWidth = ow2;
            srDisp.upscaleHeight = oh2;
            srDisp.frameTimeMilliseconds = 16.6667f;
            srDisp.preExposure = 1.0f;
            srDisp.reset = (f == 0) ? VK_TRUE : VK_FALSE;
            srDisp.cameraNear = 0.1f;
            srDisp.cameraFar = 1000.0f;
            srDisp.cameraVerticalFovRadians = 1.0471975512f;
            srDisp.viewSpaceToMeters = 1.0f;
            srDisp.frameId = fid;

            if (ffxVkFsr3_3_1_5UpscalerContextRecordDispatch(srContext, &srDisp) != FFX_VK_FSR3_3_1_5_OK) {
                std::fprintf(stderr, "FAIL: SR RecordDispatch on resized frame %d\n", f);
                return 1;
            }

            transition_image_layout(cmd, slots[s].output, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

            FfxVkFsr3_3_1_6FrameGenerationPrepareInfo prep{};
            prep.commandBuffer = cmd;
            prep.color = fgImageInfo(slots[s].output, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            prep.depth = fgImageInfo(slots[s].depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            prep.motionVectors = fgImageInfo(slots[s].motion, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            prep.renderWidth = rw2;
            prep.renderHeight = rh2;
            prep.motionVectorScaleX = static_cast<float>(rw2);
            prep.motionVectorScaleY = static_cast<float>(rh2);
            prep.frameTimeMilliseconds = 16.6667f;
            prep.minLuminance = 0.0f;
            prep.maxLuminance = 1.0f;
            prep.transferFunction = FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
            prep.cameraNear = 0.1f;
            prep.cameraFar = 1000.0f;
            prep.viewSpaceToMeters = 1.0f;
            prep.cameraVerticalFovRadians = 1.0471975512f;
            prep.cameraUp[1] = 1.0f;
            prep.cameraRight[0] = 1.0f;
            prep.cameraForward[2] = -1.0f;
            prep.frameId = fid;
            prep.reset = (f == 0) ? VK_TRUE : VK_FALSE;
            if (ffxVkFsr3_3_1_6FrameGenerationContextRecordPrepare(fgContext, &prep) != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
                std::fprintf(stderr, "FAIL: FG RecordPrepare on resized frame %d\n", f);
                return 1;
            }

            FfxVkFsr3_3_1_6FrameGenerationDispatchInfo disp{};
            disp.commandBuffer = cmd;
            disp.color = prep.color;
            disp.output = fgImageInfo(slots[s].fgOutput, VK_IMAGE_LAYOUT_GENERAL);
            disp.displayWidth = ow2;
            disp.displayHeight = oh2;
            disp.frameTimeMilliseconds = 16.6667f;
            disp.cameraNear = 0.1f;
            disp.cameraFar = 1000.0f;
            disp.viewSpaceToMeters = 1.0f;
            disp.cameraVerticalFovRadians = 1.0471975512f;
            disp.maxLuminance = 1.0f;
            disp.transferFunction = FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
            disp.frameId = fid;
            disp.reset = (f == 0) ? VK_TRUE : VK_FALSE;
            if (ffxVkFsr3_3_1_6FrameGenerationContextRecordDispatch(fgContext, &disp) != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
                std::fprintf(stderr, "FAIL: FG RecordDispatch on resized frame %d\n", f);
                return 1;
            }

            if (f == 3) {
                transition_image_layout(cmd, slots[s].fgOutput, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                VkBufferImageCopy region{};
                region.bufferOffset = 0;
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = {ow2, oh2, 1u};
                vkCmdCopyImageToBuffer(cmd, slots[s].fgOutput.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       slots[s].readbackStaging.buffer, 1, &region);
            }

            if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
                std::fprintf(stderr, "FAIL: vkEndCommandBuffer on resized frame %d\n", f);
                return 1;
            }

            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1u;
            si.pCommandBuffers = &cmd;
            if (vkQueueSubmit(queue, 1u, &si, slots[s].fence) != VK_SUCCESS) {
                std::fprintf(stderr, "FAIL: vkQueueSubmit on resized frame %d\n", f);
                return 1;
            }
        }
        vkQueueWaitIdle(queue);

        // Verify resized readback is non-empty and finite
        void* m2 = nullptr;
        vkMapMemory(device, slots[1].readbackStaging.memory, 0, outBytes2, 0, &m2);
        const uint16_t* readResized = static_cast<const uint16_t*>(m2);
        size_t count2 = static_cast<size_t>(ow2) * oh2 * 4;
        size_t finite2 = 0;
        for (size_t i = 0; i < count2; ++i) {
            float val = half_to_float(readResized[i]);
            if (std::isfinite(val)) finite2++;
        }
        vkUnmapMemory(device, slots[1].readbackStaging.memory);

        if (finite2 != count2) {
            std::fprintf(stderr, "FAIL: Resized FG output contained non-finite floats (%zu / %zu)\n", finite2, count2);
            return 1;
        }
        std::printf("PASS: Resize lifecycle + retirement verified with valid output (%zu finite floats).\n", finite2);
    }

    // Cleanup
    for (int s = 0; s < 2; ++s) {
        for (const auto& token : slots[s].importedTokens) {
            ffxVkFsr3_3_1_5BridgeReleaseImportedImage(srBridge, token);
        }
        slots[s].importedTokens.clear();
        destroy_image(device, &slots[s].color);
        destroy_image(device, &slots[s].depth);
        destroy_image(device, &slots[s].motion);
        destroy_image(device, &slots[s].reactive);
        destroy_image(device, &slots[s].transparency);
        destroy_image(device, &slots[s].dilatedDepth);
        destroy_image(device, &slots[s].dilatedMotion);
        destroy_image(device, &slots[s].prevDepth);
        destroy_image(device, &slots[s].output);
        destroy_image(device, &slots[s].fgOutput);
        destroy_buffer(device, &slots[s].uploadStaging);
        destroy_buffer(device, &slots[s].readbackStaging);
        vkDestroyFence(device, slots[s].fence, nullptr);
    }

    ffxVkFsr3_3_1_5UpscalerContextDestroy(srContext);
    ffxVkFsr3_3_1_6FrameGenerationContextDestroy(fgContext);

    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    std::printf("ALL FSR 3.1.5 + 3.1.6 PIXEL/LIFETIME REGRESSION FIXTURE CHECKS PASSED!\n");
    return 0;
}
