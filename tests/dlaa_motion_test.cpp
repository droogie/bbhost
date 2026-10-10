// Standalone GPU test for DLAA motion vector compute shader (dlaa_mv.spv.h).
// Verifies camera reprojection, animated object motion vector acceptance/rejection,
// depth tolerance thresholding, NaN/Inf filtering, and push constant controls.
// Exits with 77 if no suitable Vulkan 1.2 compute device is available.

#include "host/shaders/dlaa_mv.spv.h"
#include <vulkan/vulkan.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

static void check(VkResult r, const char* msg) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": VkResult " + std::to_string(r));
    }
}

// Convert 16-bit float (IEEE 754 binary16) to 32-bit single precision float.
static float half_to_float(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h & 0x7C00u) >> 10;
    const uint32_t mant = (h & 0x03FFu);

    if (exp == 0x1F) {
        const uint32_t f = sign | 0x7F800000u | (mant << 13);
        return std::bit_cast<float>(f);
    } else if (exp == 0) {
        if (mant == 0) return std::bit_cast<float>(sign);
        const float val = (static_cast<float>(mant) / 1024.0f) * std::ldexp(1.0f, -14);
        return sign ? -val : val;
    } else {
        const uint32_t f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        return std::bit_cast<float>(f);
    }
}

struct PushParams {
    float reproject[16];   // current NDC -> previous clip (column-major)
    float jitter[2];       // pixels, x right, y down
    uint32_t size[2];      // image dimensions (width, height)
    uint32_t zero;         // 1: write zero motion
    uint32_t objects;      // 1: accept valid animated-mesh vectors
    float projection_z;    // projection matrix far z
    uint32_t pad;          // padding to 96 bytes
};
static_assert(sizeof(PushParams) == 96, "PushParams must be exactly 96 bytes");

static void set_identity_mat4(float* m) {
    std::fill(m, m + 16, 0.0f);
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

// Set reprojection matrix to produce a pure camera pixel shift of dx, dy on dims (w, h).
// prev.xy / prev.w - ndc.xy = 2 * (dx, -dy) / dims
static void set_shift_mat4(float* m, float dx, float dy, float w, float h) {
    set_identity_mat4(m);
    m[12] = 2.0f * dx / w;
    m[13] = -2.0f * dy / h;
}

struct Fixture {
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer cmd{};
    VkBuffer buffer{};
    VkDeviceMemory buffer_mem{};
    void* mapped{};

    VkImage depth_image{}, motion_image{}, object_image{};
    VkDeviceMemory depth_mem{}, motion_mem{}, object_mem{};
    VkImageView depth_view{}, motion_view{}, object_view{};

    VkDescriptorSetLayout desc_layout{};
    VkPipelineLayout pipeline_layout{};
    VkDescriptorPool desc_pool{};
    VkDescriptorSet desc_set{};
    VkShaderModule shader_module{};
    VkPipeline pipeline{};

    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
        VkPhysicalDeviceMemoryProperties p{};
        vkGetPhysicalDeviceMemoryProperties(physical, &p);
        for (uint32_t i = 0; i < p.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) && (p.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        throw std::runtime_error("no matching memory type");
    }

    void make_image(VkFormat format, VkImageUsageFlags usage,
                    VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = format;
        ci.extent = {8, 8, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.usage = usage;
        check(vkCreateImage(device, &ci, nullptr, &img), "vkCreateImage failed");

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, img, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(req.memoryTypeBits, 0);
        check(vkAllocateMemory(device, &ai, nullptr, &mem), "vkAllocateMemory image failed");
        check(vkBindImageMemory(device, img, mem, 0), "vkBindImageMemory failed");

        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device, &vi, nullptr, &view), "vkCreateImageView failed");
    }

    void begin() {
        check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer failed");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer failed");
    }

    void submit() {
        check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer failed");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence{};
        check(vkCreateFence(device, &fi, nullptr, &fence), "vkCreateFence failed");
        check(vkQueueSubmit(queue, 1, &si, fence), "vkQueueSubmit failed");
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, 10'000'000'000ull), "vkWaitForFences failed");
        vkDestroyFence(device, fence, nullptr);
    }

    bool init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return false;

        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) return false;
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());

        uint32_t family = 0;
        for (auto candidate : devices) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(candidate, &props);
            if (props.apiVersion < VK_API_VERSION_1_2) continue;

            VkPhysicalDeviceFeatures features{};
            vkGetPhysicalDeviceFeatures(candidate, &features);
            if (!features.shaderStorageImageExtendedFormats) continue;

            uint32_t qn = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &qn, nullptr);
            std::vector<VkQueueFamilyProperties> queues(qn);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &qn, queues.data());
            for (uint32_t i = 0; i < qn; ++i) {
                if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physical = candidate;
                    family = i;
                    break;
                }
            }
            if (physical) break;
        }
        if (!physical) return false;

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;

        VkPhysicalDeviceFeatures enabled_features{};
        enabled_features.shaderStorageImageExtendedFormats = VK_TRUE;

        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        di.pEnabledFeatures = &enabled_features;
        if (vkCreateDevice(physical, &di, nullptr, &device) != VK_SUCCESS) return false;
        vkGetDeviceQueue(device, family, 0, &queue);

        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = family;
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(device, &pi, nullptr, &pool), "vkCreateCommandPool failed");

        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ca.commandPool = pool;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device, &ca, &cmd), "vkAllocateCommandBuffers failed");

        // Staging buffer: 256 B depth + 1024 B object_motion + 256 B motion readback
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = 4096;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(device, &bi, nullptr, &buffer), "vkCreateBuffer failed");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, buffer, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device, &ai, nullptr, &buffer_mem), "vkAllocateMemory buffer failed");
        check(vkBindBufferMemory(device, buffer, buffer_mem, 0), "vkBindBufferMemory failed");
        check(vkMapMemory(device, buffer_mem, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory failed");
        std::memset(mapped, 0, 4096);

        // Images: depth (R32F), motion (RG16F), object_motion (RGBA32F)
        make_image(VK_FORMAT_R32_SFLOAT,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   depth_image, depth_mem, depth_view);
        make_image(VK_FORMAT_R16G16_SFLOAT,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   motion_image, motion_mem, motion_view);
        make_image(VK_FORMAT_R32G32B32A32_SFLOAT,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   object_image, object_mem, object_view);

        // Descriptors: binding 0 (depth), 1 (motion), 2 (object_motion)
        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
        }};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 3;
        li.pBindings = bindings.data();
        check(vkCreateDescriptorSetLayout(device, &li, nullptr, &desc_layout), "vkCreateDescriptorSetLayout failed");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(PushParams);

        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &desc_layout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(device, &pli, nullptr, &pipeline_layout), "vkCreatePipelineLayout failed");

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        check(vkCreateDescriptorPool(device, &dpi, nullptr, &desc_pool), "vkCreateDescriptorPool failed");

        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        da.descriptorPool = desc_pool;
        da.descriptorSetCount = 1;
        da.pSetLayouts = &desc_layout;
        check(vkAllocateDescriptorSets(device, &da, &desc_set), "vkAllocateDescriptorSets failed");

        std::array<VkDescriptorImageInfo, 3> img_infos = {{
            {VK_NULL_HANDLE, depth_view, VK_IMAGE_LAYOUT_GENERAL},
            {VK_NULL_HANDLE, motion_view, VK_IMAGE_LAYOUT_GENERAL},
            {VK_NULL_HANDLE, object_view, VK_IMAGE_LAYOUT_GENERAL}
        }};
        std::array<VkWriteDescriptorSet, 3> writes{};
        for (int i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].descriptorCount = 1;
            writes[i].pImageInfo = &img_infos[i];
        }
        vkUpdateDescriptorSets(device, 3, writes.data(), 0, nullptr);

        // Transition image layouts from UNDEFINED to GENERAL
        begin();
        std::array<VkImageMemoryBarrier, 3> transitions{};
        for (int i = 0; i < 3; ++i) {
            auto& b = transitions[i];
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = (i == 0) ? depth_image : (i == 1) ? motion_image : object_image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, nullptr, 0, nullptr, 3, transitions.data());
        submit();

        // Create compute shader pipeline
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = sizeof(k_dlaa_mv_spv);
        smi.pCode = k_dlaa_mv_spv;
        check(vkCreateShaderModule(device, &smi, nullptr, &shader_module), "vkCreateShaderModule failed");

        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.layout = pipeline_layout;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = shader_module;
        cpi.stage.pName = "main";
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline),
              "vkCreateComputePipelines failed");

        return true;
    }

    void upload_inputs() {
        begin();
        VkBufferImageCopy c_depth{};
        c_depth.bufferOffset = 0;
        c_depth.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c_depth.imageExtent = {8, 8, 1};
        vkCmdCopyBufferToImage(cmd, buffer, depth_image, VK_IMAGE_LAYOUT_GENERAL, 1, &c_depth);

        VkBufferImageCopy c_obj{};
        c_obj.bufferOffset = 256;
        c_obj.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c_obj.imageExtent = {8, 8, 1};
        vkCmdCopyBufferToImage(cmd, buffer, object_image, VK_IMAGE_LAYOUT_GENERAL, 1, &c_obj);

        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
        submit();
    }

    void dispatch(const PushParams& params) {
        begin();
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushParams), &params);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &desc_set, 0, nullptr);
        vkCmdDispatch(cmd, 1, 1, 1);

        VkMemoryBarrier mb1{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb1.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb1.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb1, 0, nullptr, 0, nullptr);

        VkBufferImageCopy c_motion{};
        c_motion.bufferOffset = 2048;
        c_motion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c_motion.imageExtent = {8, 8, 1};
        vkCmdCopyImageToBuffer(cmd, motion_image, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &c_motion);

        VkMemoryBarrier mb2{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &mb2, 0, nullptr, 0, nullptr);
        submit();
    }

    std::pair<float, float> read_pixel(int x, int y) const {
        const auto* ptr = reinterpret_cast<const uint16_t*>(static_cast<const char*>(mapped) + 2048);
        const int idx = (y * 8 + x) * 2;
        return {half_to_float(ptr[idx]), half_to_float(ptr[idx + 1])};
    }

    ~Fixture() {
        if (device) {
            vkDeviceWaitIdle(device);
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (shader_module) vkDestroyShaderModule(device, shader_module, nullptr);
            if (desc_pool) vkDestroyDescriptorPool(device, desc_pool, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (desc_layout) vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
            if (depth_view) vkDestroyImageView(device, depth_view, nullptr);
            if (motion_view) vkDestroyImageView(device, motion_view, nullptr);
            if (object_view) vkDestroyImageView(device, object_view, nullptr);
            if (depth_image) vkDestroyImage(device, depth_image, nullptr);
            if (motion_image) vkDestroyImage(device, motion_image, nullptr);
            if (object_image) vkDestroyImage(device, object_image, nullptr);
            if (depth_mem) vkFreeMemory(device, depth_mem, nullptr);
            if (motion_mem) vkFreeMemory(device, motion_mem, nullptr);
            if (object_mem) vkFreeMemory(device, object_mem, nullptr);
            if (mapped) vkUnmapMemory(device, buffer_mem);
            if (buffer) vkDestroyBuffer(device, buffer, nullptr);
            if (buffer_mem) vkFreeMemory(device, buffer_mem, nullptr);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

static bool verify_pixel(const char* label, std::pair<float, float> got,
                         float exp_x, float exp_y, float eps = 0.05f) {
    const bool ok = std::abs(got.first - exp_x) <= eps && std::abs(got.second - exp_y) <= eps;
    std::printf("  [%s] %s: got (%.4f, %.4f), expected (%.4f, %.4f)\n",
                ok ? "PASS" : "FAIL", label, got.first, got.second, exp_x, exp_y);
    return ok;
}

int main() try {
    Fixture f;
    if (!f.init()) {
        std::puts("SKIP: suitable Vulkan 1.2 compute device unavailable");
        return 77;
    }

    // Populate staging buffer for 8x8 input textures:
    // Offset 0: float depth[64]
    // Offset 256: float object_motion[64 * 4] (RGBA32F)
    auto* depth_map = reinterpret_cast<float*>(static_cast<char*>(f.mapped) + 0);
    auto* obj_map = reinterpret_cast<float*>(static_cast<char*>(f.mapped) + 256);

    for (int i = 0; i < 64; ++i) {
        depth_map[i] = 0.5f;
        obj_map[i * 4 + 0] = 0.0f;
        obj_map[i * 4 + 1] = 0.0f;
        obj_map[i * 4 + 2] = 0.0f;
        obj_map[i * 4 + 3] = 0.0f;
    }

    // Configure distinct test cases across pixel columns at y = 0:
    // Pixel (0, 0): Baseline pixel, no object motion (b = 0)
    depth_map[0] = 0.5f;
    obj_map[0 * 4 + 0] = 0.0f; obj_map[0 * 4 + 1] = 0.0f;
    obj_map[0 * 4 + 2] = 0.0f; obj_map[0 * 4 + 3] = 0.0f;

    // Pixel (1, 0): Synthetic object (rg = 7, -4), b = 1, a = depth -> accepted
    depth_map[1] = 0.5f;
    obj_map[1 * 4 + 0] = 7.0f; obj_map[1 * 4 + 1] = -4.0f;
    obj_map[1 * 4 + 2] = 1.0f; obj_map[1 * 4 + 3] = 0.5f;

    // Pixel (2, 0): b = 0 -> rejected
    depth_map[2] = 0.5f;
    obj_map[2 * 4 + 0] = 7.0f; obj_map[2 * 4 + 1] = -4.0f;
    obj_map[2 * 4 + 2] = 0.0f; obj_map[2 * 4 + 3] = 0.5f;

    // Pixel (3, 0): Wrong depth (a = 0.1 vs z = 0.5, diff = 0.4 > tolerance) -> rejected
    depth_map[3] = 0.5f;
    obj_map[3 * 4 + 0] = 7.0f; obj_map[3 * 4 + 1] = -4.0f;
    obj_map[3 * 4 + 2] = 1.0f; obj_map[3 * 4 + 3] = 0.1f;

    // Pixel (4, 0): NaN in motion vector -> rejected
    depth_map[4] = 0.5f;
    obj_map[4 * 4 + 0] = std::numeric_limits<float>::quiet_NaN();
    obj_map[4 * 4 + 1] = -4.0f;
    obj_map[4 * 4 + 2] = 1.0f; obj_map[4 * 4 + 3] = 0.5f;

    // Pixel (5, 0): Inf in motion vector -> rejected
    depth_map[5] = 0.5f;
    obj_map[5 * 4 + 0] = 7.0f;
    obj_map[5 * 4 + 1] = std::numeric_limits<float>::infinity();
    obj_map[5 * 4 + 2] = 1.0f; obj_map[5 * 4 + 3] = 0.5f;

    // Pixel (6, 0): Near-surface depth z = 1.0, projection_z = 1.0, tolerance = 2e-7.
    // object.a = 1.0001f -> diff = 1e-4 > 2e-7 -> rejected
    depth_map[6] = 1.0f;
    obj_map[6 * 4 + 0] = 7.0f; obj_map[6 * 4 + 1] = -4.0f;
    obj_map[6 * 4 + 2] = 1.0f; obj_map[6 * 4 + 3] = 1.0001f;

    // Pixel (7, 0): Near-surface depth z = 1.0, exact equal depth -> accepted
    depth_map[7] = 1.0f;
    obj_map[7 * 4 + 0] = 7.0f; obj_map[7 * 4 + 1] = -4.0f;
    obj_map[7 * 4 + 2] = 1.0f; obj_map[7 * 4 + 3] = 1.0f;

    f.upload_inputs();

    bool all_ok = true;

    // Test Pass 1: Identity camera reprojection, objects enabled
    std::puts("--- Pass 1: Identity camera, objects enabled ---");
    PushParams p1{};
    set_identity_mat4(p1.reproject);
    p1.jitter[0] = 0.0f; p1.jitter[1] = 0.0f;
    p1.size[0] = 8; p1.size[1] = 8;
    p1.zero = 0;
    p1.objects = 1;
    p1.projection_z = 1.0f;
    f.dispatch(p1);

    all_ok &= verify_pixel("1. identity camera yields 0 without objects", f.read_pixel(0, 0), 0.0f, 0.0f);
    all_ok &= verify_pixel("2. synthetic object rg (7,-4) b 1 equal depth accepted", f.read_pixel(1, 0), 7.0f, -4.0f);
    all_ok &= verify_pixel("3. b 0 rejected -> falls back to camera 0", f.read_pixel(2, 0), 0.0f, 0.0f);
    all_ok &= verify_pixel("4. wrong depth rejected -> falls back to camera 0", f.read_pixel(3, 0), 0.0f, 0.0f);
    all_ok &= verify_pixel("5a. NaN rejected -> falls back to camera 0", f.read_pixel(4, 0), 0.0f, 0.0f);
    all_ok &= verify_pixel("5b. Inf rejected -> falls back to camera 0", f.read_pixel(5, 0), 0.0f, 0.0f);
    all_ok &= verify_pixel("6a. near-surface precision diff > threshold rejects", f.read_pixel(6, 0), 0.0f, 0.0f);
    all_ok &= verify_pixel("6b. near-surface precision diff <= threshold accepts", f.read_pixel(7, 0), 7.0f, -4.0f);

    // Test Pass 2: Camera reprojection with pixel shift of 2 in X, objects enabled
    // When object is invalid, camera reprojection shift of (2, 0) must be observed.
    // When object is valid (pixel 1, 0 and 7, 0), object motion (7, -4) takes precedence.
    std::puts("--- Pass 2: Camera shift 2 in X, verify object fallback and precedence ---");
    PushParams p2{};
    set_shift_mat4(p2.reproject, 2.0f, 0.0f, 8.0f, 8.0f);
    p2.jitter[0] = 0.0f; p2.jitter[1] = 0.0f;
    p2.size[0] = 8; p2.size[1] = 8;
    p2.zero = 0;
    p2.objects = 1;
    p2.projection_z = 1.0f;
    f.dispatch(p2);

    all_ok &= verify_pixel("9a. camera pixel shift 2 verified without objects", f.read_pixel(0, 0), 2.0f, 0.0f);
    all_ok &= verify_pixel("9b. valid object retains (7,-4) under camera shift", f.read_pixel(1, 0), 7.0f, -4.0f);
    all_ok &= verify_pixel("9c. b 0 rejected -> camera pixel shift 2", f.read_pixel(2, 0), 2.0f, 0.0f);
    all_ok &= verify_pixel("9d. wrong depth rejected -> camera pixel shift 2", f.read_pixel(3, 0), 2.0f, 0.0f);
    all_ok &= verify_pixel("9e. NaN rejected -> camera pixel shift 2", f.read_pixel(4, 0), 2.0f, 0.0f);
    all_ok &= verify_pixel("9f. Inf rejected -> camera pixel shift 2", f.read_pixel(5, 0), 2.0f, 0.0f);
    all_ok &= verify_pixel("9g. near-surface diff > threshold rejected -> camera shift 2", f.read_pixel(6, 0), 2.0f, 0.0f);

    // Test Pass 3: objects = 0 branch ignores object motion
    std::puts("--- Pass 3: objects = 0 branch ignores objects ---");
    PushParams p3 = p2;
    p3.objects = 0;
    f.dispatch(p3);

    all_ok &= verify_pixel("8. objects 0 branch ignores valid object -> camera shift 2", f.read_pixel(1, 0), 2.0f, 0.0f);

    // Test Pass 4: zero = 1 override writes all zero
    std::puts("--- Pass 4: zero = 1 override writes all zero ---");
    PushParams p4 = p2;
    p4.zero = 1;
    p4.objects = 1;
    f.dispatch(p4);

    all_ok &= verify_pixel("7a. zero override writes 0 on pixel with camera shift", f.read_pixel(0, 0), 0.0f, 0.0f);
    all_ok &= verify_pixel("7b. zero override writes 0 on pixel with valid object", f.read_pixel(1, 0), 0.0f, 0.0f);

    if (all_ok) {
        std::puts("\nALL DLAA MOTION VECTOR TESTS PASSED");
        return 0;
    } else {
        std::puts("\nSOME DLAA MOTION VECTOR TESTS FAILED");
        return 1;
    }
} catch (const std::exception& e) {
    std::fprintf(stderr, "Fatal error: %s\n", e.what());
    return 1;
}
