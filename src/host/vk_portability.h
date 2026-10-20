#pragma once
// The loader shows MoltenVK (a portability driver) only to an instance that
// enables VK_KHR_portability_enumeration. Header-only so tests and tools need
// not link gpu.cpp.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace gpu {

inline void portability_enumeration(VkInstanceCreateInfo& ici, std::vector<const char*>& exts) {
#if defined(__APPLE__)
    std::uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> props(n);
    if (n) vkEnumerateInstanceExtensionProperties(nullptr, &n, props.data());
    for (std::uint32_t i = 0; i < n; ++i) {
        if (std::strcmp(props[i].extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) != 0) continue;
        exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        break;
    }
#else
    (void)ici;
    (void)exts;
#endif
}

}  // namespace gpu
