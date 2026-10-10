// NGX capability smoke test on an actual Vulkan device; no game files needed.
#define VK_NO_PROTOTYPES
#include "ngx_bridge.h"
#include <windows.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

int ngx_gpu_tests(VkInstance,VkPhysicalDevice,VkDevice,uint32_t,PFN_vkGetInstanceProcAddr,PFN_vkGetDeviceProcAddr);
int main(int argc,char**) {
    auto loader = LoadLibraryW(L"vulkan-1.dll");
    if (!loader) return 77;
    auto get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(loader, "vkGetInstanceProcAddr"));
    if (!get) return 1;
#define LOAD(name, instance) auto name = reinterpret_cast<PFN_##name>(get(instance, #name))
    LOAD(vkCreateInstance, nullptr);
    LOAD(vkEnumerateInstanceExtensionProperties, nullptr);
    const std::wstring dir = std::filesystem::current_path().wstring();
    uint32_t ni = 0, nd = 0, count = 0;
    const char *const *ie = nullptr, *const *de = nullptr;
    if (ngxb_failed(ngxb_required_extensions(&ni, &ie, &nd, &de))) return 1;
    std::vector<const char*> instances(ie, ie + ni), devices(de, de + nd);
    const VkExtensionProperties* fg = nullptr;
    auto r = ngxb_fg_extensions(nullptr, nullptr, dir.c_str(), &count, &fg);
    std::printf("FG instance extension query: 0x%08x\n", r);
    auto append = [](auto& into, const char* name) {
        if (std::none_of(into.begin(), into.end(), [&](const char* x) { return !std::strcmp(x,name); })) into.push_back(name);
    };
    if (!ngxb_failed(r)) for (uint32_t i = 0; i < count; ++i) append(instances, fg[i].extensionName);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "bbhost NGX probe";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(instances.size());
    ci.ppEnabledExtensionNames = instances.data();
    VkInstance instance = nullptr;
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) return 1;
    LOAD(vkEnumeratePhysicalDevices, instance);
    LOAD(vkGetPhysicalDeviceProperties, instance);
    LOAD(vkGetPhysicalDeviceQueueFamilyProperties, instance);
    LOAD(vkGetPhysicalDeviceFeatures2, instance);
    LOAD(vkCreateDevice, instance);
    LOAD(vkDestroyInstance, instance);
    LOAD(vkGetDeviceProcAddr, instance);
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> gpus(count);
    vkEnumeratePhysicalDevices(instance, &count, gpus.data());
    VkPhysicalDevice gpu = nullptr;
    for (auto x : gpus) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(x, &properties);
        if (properties.vendorID == 0x10de) { gpu = x; std::printf("GPU: %s\n", properties.deviceName); break; }
    }
    if (!gpu) { vkDestroyInstance(instance, nullptr); return 77; }
    r = ngxb_fg_extensions(instance, gpu, dir.c_str(), &count, &fg);
    std::printf("FG device extension query: 0x%08x\n", r);
    if (!ngxb_failed(r)) for (uint32_t i = 0; i < count; ++i) append(devices, fg[i].extensionName);
    // The probe enables core Vulkan 1.2 buffer addresses below, as the host
    // does. NGX's legacy EXT request conflicts with both that and the KHR API.
    devices.erase(std::remove_if(devices.begin(), devices.end(), [](const char* name) {
        return !std::strcmp(name, "VK_EXT_buffer_device_address");
    }), devices.end());
    for (auto x : devices) std::printf("extension: %s\n", x);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());
    uint32_t family = 0;
    while (family < count && !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++family;
    if (family == count) return 77;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f12.pNext = &f13;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(gpu, &features);
    float priority = 1;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = family; q.queueCount = 1; q.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &features;
    device_info.queueCreateInfoCount = 1; device_info.pQueueCreateInfos = &q;
    device_info.enabledExtensionCount = static_cast<uint32_t>(devices.size());
    device_info.ppEnabledExtensionNames = devices.data();
    VkDevice device = nullptr;
    auto vr = vkCreateDevice(gpu, &device_info, nullptr, &device);
    if (vr != VK_SUCCESS) { std::printf("vkCreateDevice failed: %d\n", vr); return 1; }
    r = ngxb_init(instance, gpu, device, get, vkGetDeviceProcAddr, dir.c_str(), dir.c_str(),
                  [](const char* s) { std::puts(s); });
    std::printf("NGX init: 0x%08x; DLSS FG available: %d\n", r, ngxb_fg_available());
    const bool ok = !ngxb_failed(r);
    if (ok) for (int quality = 0; quality < 4; ++quality) {
        NgxbSettings settings{};
        r = ngxb_optimal_settings(1920,1080,quality,&settings);
        if (ngxb_failed(r) || !settings.width || !settings.height || settings.width > 1920 || settings.height > 1080) {
            std::printf("FAIL optimal render settings, quality %d: 0x%08x\n",quality,r);
            ngxb_shutdown();
            auto destroy = reinterpret_cast<PFN_vkDestroyDevice>(vkGetDeviceProcAddr(device,"vkDestroyDevice"));
            destroy(device,nullptr); vkDestroyInstance(instance,nullptr); return 1;
        }
        std::printf("Optimal settings: quality %d, %ux%u -> 1920x1080\n",quality,settings.width,settings.height);
    }
    const int test = ok && argc>1 ? ngx_gpu_tests(instance,gpu,device,family,get,vkGetDeviceProcAddr) : (ok ? 0 : 1);
    ngxb_shutdown();
    auto destroy = reinterpret_cast<PFN_vkDestroyDevice>(vkGetDeviceProcAddr(device,"vkDestroyDevice"));
    destroy(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return test;
}
