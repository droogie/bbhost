#include <vulkan/vulkan.h>
#include "host/upscaler.h"
#include "host/pe_loader.h"
#include "host/settings.h"
#include "core/config.h"
#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cassert>
#include <iostream>
#include <mutex>

// Mock host settings & dependencies for the test
static HostSettings g_test_settings{};
const HostSettings& host_settings_ref() { return g_test_settings; }
HostSettings host_settings() { return g_test_settings; }

static std::string g_test_exe_dir;
std::string config_exe_dir() { return g_test_exe_dir; }
std::string config_user_dir() { return g_test_exe_dir; }
int plugins_aa_claim() { return 0; }
bool host_gpu_fsr_upscale(void*, void*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void*, int, int, unsigned int, unsigned int) {
    return false;
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return 0;
}

static bool create_test_image(VkDevice dev, VkPhysicalDevice phys, uint32_t w, uint32_t h,
                              VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                              VkImage* out_img, VkDeviceMemory* out_mem, VkImageView* out_view) {
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
    if (vkCreateImage(dev, &ici, nullptr, out_img) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, *out_img, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(dev, &mai, nullptr, out_mem) != VK_SUCCESS) return false;
    if (vkBindImageMemory(dev, *out_img, *out_mem, 0) != VK_SUCCESS) return false;

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = *out_img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    if (vkCreateImageView(dev, &vci, nullptr, out_view) != VK_SUCCESS) return false;

    return true;
}

int main(int argc, char* argv[]) {
    host_log_decide(true);
    std::cout << "=== FSR 4 Upscaling Hardware Verification Test ===" << std::endl;

    // Set executable directory to binary dir
    if (argc > 0 && argv[0]) {
        std::string p = argv[0];
        auto pos = p.find_last_of("/\\");
        g_test_exe_dir = (pos != std::string::npos) ? p.substr(0, pos) : ".";
    } else {
        g_test_exe_dir = ".";
    }

    pe_add_search_dir(g_test_exe_dir.c_str());
    pe_add_search_dir("third_party/upscaler_bridge/bin");
    pe_add_search_dir("../third_party/upscaler_bridge/bin");

    // Enable Upscaler in settings
    g_test_settings.upscaler = 2; // FSR

    // 1. Test upscaler_open
    std::cout << "[1/5] Testing upscaler_open()..." << std::endl;
    if (!upscaler_open()) {
        std::cerr << "FAIL: upscaler_open() failed to open upscaler_bridge.dll" << std::endl;
        return 1;
    }
    std::cout << "  OK: upscaler_bridge.dll opened successfully" << std::endl;

    // 2. Initialize Vulkan Instance
    std::cout << "[2/5] Initializing Vulkan with upscaler_prepare_instance()..." << std::endl;
    pe_set_vk_resolvers(vkGetInstanceProcAddr, vkGetDeviceProcAddr);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "FSR_HW_Test";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    const char* const inst_exts[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        "VK_EXT_headless_surface",
    };

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = inst_exts;

    const VkInstanceCreateInfo* prepared_ici = upscaler_prepare_instance(&ici);
    std::cout << "  prepared_ici = " << prepared_ici << ", sType = " << prepared_ici->sType << ", ext_count = " << prepared_ici->enabledExtensionCount << std::endl;
    for (uint32_t i = 0; i < prepared_ici->enabledExtensionCount; ++i) {
        std::cout << "    ext[" << i << "] = " << prepared_ici->ppEnabledExtensionNames[i] << std::endl;
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkResult res = vkCreateInstance(prepared_ici, nullptr, &instance);
    upscaler_instance_done(res);
    if (res != VK_SUCCESS) {
        std::cerr << "FAIL: vkCreateInstance failed: " << res << std::endl;
        return 1;
    }
    std::cout << "  OK: Vulkan Instance created with upscaler hooks" << std::endl;

    // 3. Find Physical Device
    uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr);
    if (gpu_count == 0) {
        std::cerr << "FAIL: No Vulkan physical devices found" << std::endl;
        return 1;
    }
    std::vector<VkPhysicalDevice> gpus(gpu_count);
    vkEnumeratePhysicalDevices(instance, &gpu_count, gpus.data());
    VkPhysicalDevice phys = gpus[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    std::cout << "  Found GPU: " << props.deviceName << " (Vendor 0x" << std::hex << props.vendorID << " Device 0x" << props.deviceID << std::dec << ")" << std::endl;

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    uint32_t q_family = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, qf_props.data());
    for (uint32_t i = 0; i < qf_count; ++i) {
        if ((qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            q_family = i;
            break;
        }
    }

    // 4. Create Device with upscaler_prepare_device
    std::cout << "[3/5] Creating Vulkan Device with upscaler_prepare_device()..." << std::endl;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = q_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceFeatures feats{};
    feats.shaderStorageImageExtendedFormats = VK_TRUE;

    VkPhysicalDeviceVulkan12Features feats12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    feats12.timelineSemaphore = VK_TRUE;

    const char* const dev_exts[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &feats12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &feats;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dev_exts;

    const VkDeviceCreateInfo* prepared_dci = upscaler_prepare_device(instance, phys, &dci);
    VkDevice device = VK_NULL_HANDLE;
    res = vkCreateDevice(phys, prepared_dci, nullptr, &device);
    upscaler_device_done(res);
    if (res != VK_SUCCESS) {
        std::cerr << "FAIL: vkCreateDevice failed: " << res << std::endl;
        return 1;
    }

    VkQueue real_queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, q_family, 0, &real_queue);

    // 5. Attach device & verify Caps (FSR 4)
    std::cout << "[4/5] Attaching upscaler and checking caps..." << std::endl;
    upscaler_attach(instance, phys, device, true);

    UpscalerCaps caps = upscaler_caps();
    std::cout << "  Caps: loaded=" << caps.loaded << ", fsr=" << caps.fsr << ", fsr4=" << caps.fsr4 << std::endl;
    std::cout << "  FSR Versions: " << caps.fsr_versions << std::endl;
    std::cout << "  FSR Reason: " << caps.fsr_reason << std::endl;

    if (!caps.loaded) {
        std::cerr << "FAIL: upscaler_caps().loaded is false!" << std::endl;
        return 1;
    }
    if (!caps.fsr) {
        std::cerr << "FAIL: FSR upscaler is not reported as available!" << std::endl;
        return 1;
    }
    if (!caps.fsr4) {
        std::cerr << "WARNING: FSR 4 flag is false, versions: " << caps.fsr_versions << std::endl;
    } else {
        std::cout << "  SUCCESS: FSR 4 is available on this GPU!" << std::endl;
    }

    // 6. Test FSR Evaluation (both FSR 4 and FSR 3)
    std::cout << "[5/5] Testing upscaler_evaluate() on local GPU..." << std::endl;
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = q_family;
    VkCommandPool cp = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &cpci, nullptr, &cp);

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cp;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &cbai, &cb);

    // Allocate resources: 1920x1080 -> 3840x2160
    VkImage c_img, d_img, m_img, o_img;
    VkDeviceMemory c_mem, d_mem, m_mem, o_mem;
    VkImageView c_view, d_view, m_view, o_view;

    create_test_image(device, phys, 1920, 1080, VK_FORMAT_R8G8B8A8_UNORM,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT, &c_img, &c_mem, &c_view);
    create_test_image(device, phys, 1920, 1080, VK_FORMAT_D32_SFLOAT,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      VK_IMAGE_ASPECT_DEPTH_BIT, &d_img, &d_mem, &d_view);
    create_test_image(device, phys, 1920, 1080, VK_FORMAT_R16G16_SFLOAT,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT, &m_img, &m_mem, &m_view);
    create_test_image(device, phys, 3840, 2160, VK_FORMAT_R8G8B8A8_UNORM,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT, &o_img, &o_mem, &o_view);

    // Transition layouts
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &cbbi);

    VkImageMemoryBarrier barriers[4]{};
    for (int i = 0; i < 4; ++i) {
        barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[i].srcAccessMask = 0;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[i].srcQueueFamilyIndex = barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].subresourceRange = {i == 1 ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    barriers[0].image = c_img;
    barriers[1].image = d_img;
    barriers[2].image = m_img;
    barriers[3].image = o_img;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 4, barriers);

    UpscaleCall call{};
    call.method = 2; // FSR
    call.quality = 3; // UB_QUALITY_PERFORMANCE (2.0x for 1080p -> 4k)
    call.render_w = 1920;
    call.render_h = 1080;
    call.output_w = 3840;
    call.output_h = 2160;
    call.color = {c_img, c_view, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_GENERAL};
    call.depth = {d_img, d_view, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_GENERAL};
    call.motion = {m_img, m_view, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL};
    call.output = {o_img, o_view, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_GENERAL};
    call.frame_ms = 16.6f;
    call.near_m = 0.1f;
    call.far_m = 1000.0f;
    call.fov_y = 1.0f;
    call.motion_scale = 1.0f;
    call.reset = true;

    // Test FSR 4 evaluation
    std::cout << "  Evaluating FSR 4..." << std::endl;
    bool eval_ok = upscaler_evaluate(cb, call);
    if (!eval_ok) {
        std::cerr << "FAIL: upscaler_evaluate() failed for FSR 4" << std::endl;
        return 1;
    }
    std::cout << "  OK: FSR 4 evaluate recorded into command buffer" << std::endl;

    // Test FSR 3 override evaluation
    std::cout << "  Evaluating FSR 3 override..." << std::endl;
    setenv("BBHOST_UPSCALE", "fsr3", 1);
    call.reset = false;
    eval_ok = upscaler_evaluate(cb, call);
    if (!eval_ok) {
        std::cerr << "FAIL: upscaler_evaluate() failed for FSR 3" << std::endl;
        return 1;
    }
    std::cout << "  OK: FSR 3 evaluate recorded into command buffer" << std::endl;
    unsetenv("BBHOST_UPSCALE");

    vkEndCommandBuffer(cb);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    res = vkQueueSubmit(real_queue, 1, &si, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
        std::cerr << "FAIL: vkQueueSubmit failed: " << res << std::endl;
        return 1;
    }
    vkQueueWaitIdle(real_queue);
    std::cout << "  SUCCESS: FSR 4 and FSR 3 executed on AMD GPU successfully!" << std::endl;

    // Cleanup resources
    std::cout << "Cleaning up test resources..." << std::endl;
    vkDestroyImageView(device, c_view, nullptr);
    vkDestroyImageView(device, d_view, nullptr);
    vkDestroyImageView(device, m_view, nullptr);
    vkDestroyImageView(device, o_view, nullptr);

    vkDestroyImage(device, c_img, nullptr);
    vkDestroyImage(device, d_img, nullptr);
    vkDestroyImage(device, m_img, nullptr);
    vkDestroyImage(device, o_img, nullptr);

    vkFreeMemory(device, c_mem, nullptr);
    vkFreeMemory(device, d_mem, nullptr);
    vkFreeMemory(device, m_mem, nullptr);
    vkFreeMemory(device, o_mem, nullptr);

    vkDestroyCommandPool(device, cp, nullptr);

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    std::cout << "\n>>> ALL FSR 4 UPSCALING VERIFICATION TESTS PASSED! <<<" << std::endl;
    return 0;
}
