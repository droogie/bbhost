#pragma once
// A pretend Vulkan for the upscaler bridge's tests (ngx_bridge_test,
// upscaler_load_test): vkGetInstanceProcAddr and vkGetDeviceProcAddr of its
// own, one physical device whose vendor, name and driver the test sets, and
// counts of what was called. No Vulkan loader or driver is ever reached, so a
// test can never open a real GPU (on the development machine, the RTX).

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace fake_vk {

inline std::uint32_t g_vendor = 0x1002;  // AMD
inline const char* g_name = "Pretend AMD GPU";
inline VkDriverId g_driver = VK_DRIVER_ID_MESA_RADV;
inline int g_instances = 0, g_live_instances = 0, g_barriers = 0;

inline VkInstance instance() { return reinterpret_cast<VkInstance>(static_cast<std::uintptr_t>(0x1000)); }
inline VkPhysicalDevice physical_device() { return reinterpret_cast<VkPhysicalDevice>(static_cast<std::uintptr_t>(0x2000)); }
inline VkDevice device() { return reinterpret_cast<VkDevice>(static_cast<std::uintptr_t>(0x3000)); }

VKAPI_ATTR inline VkResult VKAPI_CALL create_instance(const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance* out) {
    ++g_instances;
    ++g_live_instances;
    *out = instance();
    return VK_SUCCESS;
}
VKAPI_ATTR inline void VKAPI_CALL destroy_instance(VkInstance, const VkAllocationCallbacks*) { --g_live_instances; }
VKAPI_ATTR inline VkResult VKAPI_CALL enumerate_physical_devices(VkInstance, std::uint32_t* n, VkPhysicalDevice* out) {
    if (out && *n >= 1) out[0] = physical_device();
    *n = 1;
    return VK_SUCCESS;
}
inline void fill(VkPhysicalDeviceProperties* p) {
    std::memset(p, 0, sizeof(*p));
    p->apiVersion = VK_API_VERSION_1_3;
    p->vendorID = g_vendor;
    p->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    std::snprintf(p->deviceName, sizeof(p->deviceName), "%s", g_name);
}
VKAPI_ATTR inline void VKAPI_CALL get_properties(VkPhysicalDevice, VkPhysicalDeviceProperties* p) { fill(p); }
VKAPI_ATTR inline void VKAPI_CALL get_properties2(VkPhysicalDevice, VkPhysicalDeviceProperties2* p) {
    fill(&p->properties);
    for (auto* s = static_cast<VkBaseOutStructure*>(p->pNext); s; s = s->pNext)
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES) {
            auto* d = reinterpret_cast<VkPhysicalDeviceDriverProperties*>(s);
            d->driverID = g_driver;
            std::snprintf(d->driverName, sizeof(d->driverName), "%s", g_driver == VK_DRIVER_ID_MESA_NVK ? "NVK" : "pretend driver");
        }
}
VKAPI_ATTR inline void VKAPI_CALL get_features2(VkPhysicalDevice, VkPhysicalDeviceFeatures2*) {}
VKAPI_ATTR inline VkResult VKAPI_CALL enumerate_instance_extensions(const char*, std::uint32_t* n, VkExtensionProperties*) {
    *n = 0;
    return VK_SUCCESS;
}
VKAPI_ATTR inline VkResult VKAPI_CALL enumerate_device_extensions(VkPhysicalDevice, const char*, std::uint32_t* n, VkExtensionProperties*) {
    *n = 0;
    return VK_SUCCESS;
}
VKAPI_ATTR inline void VKAPI_CALL cmd_pipeline_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, std::uint32_t,
                                                        const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t,
                                                        const VkImageMemoryBarrier*) {
    ++g_barriers;
}

VKAPI_ATTR inline PFN_vkVoidFunction VKAPI_CALL get_device_proc_addr(VkDevice, const char* name) {
    if (std::strcmp(name, "vkCmdPipelineBarrier") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&cmd_pipeline_barrier);
    return nullptr;
}
VKAPI_ATTR inline PFN_vkVoidFunction VKAPI_CALL get_instance_proc_addr(VkInstance, const char* name) {
    struct Entry {
        const char* name;
        PFN_vkVoidFunction fn;
    };
    static const Entry kTable[] = {
        {"vkCreateInstance", reinterpret_cast<PFN_vkVoidFunction>(&create_instance)},
        {"vkDestroyInstance", reinterpret_cast<PFN_vkVoidFunction>(&destroy_instance)},
        {"vkEnumeratePhysicalDevices", reinterpret_cast<PFN_vkVoidFunction>(&enumerate_physical_devices)},
        {"vkGetPhysicalDeviceProperties", reinterpret_cast<PFN_vkVoidFunction>(&get_properties)},
        {"vkGetPhysicalDeviceProperties2", reinterpret_cast<PFN_vkVoidFunction>(&get_properties2)},
        {"vkGetPhysicalDeviceFeatures2", reinterpret_cast<PFN_vkVoidFunction>(&get_features2)},
        {"vkEnumerateInstanceExtensionProperties", reinterpret_cast<PFN_vkVoidFunction>(&enumerate_instance_extensions)},
        {"vkEnumerateDeviceExtensionProperties", reinterpret_cast<PFN_vkVoidFunction>(&enumerate_device_extensions)},
        {"vkGetDeviceProcAddr", reinterpret_cast<PFN_vkVoidFunction>(&get_device_proc_addr)},
        {"vkGetInstanceProcAddr", reinterpret_cast<PFN_vkVoidFunction>(&get_instance_proc_addr)},
    };
    for (const Entry& e : kTable)
        if (std::strcmp(name, e.name) == 0) return e.fn;
    return nullptr;
}

}  // namespace fake_vk
