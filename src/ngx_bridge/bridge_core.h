/*
 * bridge_core.h -- the NGX bridge's parts that need neither NVIDIA's code nor a GPU: the render sizes, jitter and mip
 * bias the upscaler bridge API documents (third_party/upscaler_bridge/upscaler_bridge.h, Numity1504's), the DLSS
 * runtime's file in a folder, and the copies of the host's Vulkan create infos with NGX's extensions added. The tests
 * build it alone (tests/ngx_bridge_test.cpp).
 *
 * Copyright (c) 2026 the bbhost authors. MIT licence: see LICENSE in this folder.
 */
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ngx_bridge {

constexpr std::uint32_t kVendorNvidia = 0x10de;

// The plain ratio of a method and quality (upscaler_bridge.h's UB_METHOD_*, UB_QUALITY_*): the output size over the
// render size each way - native 1, Quality 1.5, Balanced 1/0.58 for DLSS (about 1.72) and 1.7 for FSR, Performance 2.
// 0 for a method or quality it does not know.
double plain_ratio(std::uint32_t method, std::uint32_t quality);
// The render extent of an output extent at a ratio, at least 1: DLSS's rounded to the nearest (NGX's own sizes: 1114x626
// for Balanced at 1920x1080, 1707x960 for Quality at 2560x1440), FSR's cut down (AMD's: 1706x960).
std::uint32_t render_extent(std::uint32_t method, std::uint32_t output, double ratio);
// The jitter sequence's length for a render and an output width: NVIDIA's and AMD's 8 x (output / render) squared,
// rounded up, at least 8 (DLAA 8, Quality 18, Balanced 24, Performance 32).
std::uint32_t jitter_phase_count(std::uint32_t render_w, std::uint32_t output_w);
// The texture LOD bias for the 3D's materials: log2 of the render width over the output's, less 1.
float mip_bias(std::uint32_t render_w, std::uint32_t output_w);
// Halton's sequence: element `index` in `base`.
float halton(std::uint32_t index, std::uint32_t base);
// A frame's jitter: Halton (2, 3) at frame % phases + 1 (never the 0th element, the pixel's corner), centred, so each
// is in -0.5 .. 0.5 render pixels. (0, 0) for 0 phases.
void jitter(std::uint32_t frame, std::uint32_t phases, float* x, float* y);

// The DLSS runtime in a folder: the newest libnvidia-ngx-dlss.so.<version> there, its file name and version.
bool find_runtime(const std::string& dir, std::string* name, std::string* version);
// Dotted versions compared part by part as numbers: <0, 0, >0.
int compare_versions(const std::string& a, const std::string& b);
// UTF-8 to the wide strings NGX takes its paths in (UTF-32 here), and back.
std::wstring widen(const std::string& utf8);
std::string narrow(const wchar_t* wide);
std::string join(const std::vector<std::string>& names, const char* sep = ", ");
bool contains(const std::vector<std::string>& names, const std::string& name);

// vkCreateInstance's info with extensions added: `in`'s own, then each of `need` it lacks. Points into itself: made in
// place, never moved.
struct InstanceCopy {
    VkInstanceCreateInfo info{};
    std::vector<std::string> added;
    std::vector<const char*> names;
};
// False (nothing to add): `in` itself does.
bool copy_instance(const VkInstanceCreateInfo& in, const std::vector<std::string>& need, InstanceCopy* out);

// vkCreateDevice's info with what NGX needs: `need` (its device extensions) checked against `available` (the
// device's), each one `in` lacks added; buffer device address, on a device of Vulkan 1.2 or newer, as the core
// feature (never VK_EXT_buffer_device_address beside it, which Vulkan forbids): turned on in place in the host's own
// VkPhysicalDeviceVulkan12Features or VkPhysicalDeviceBufferDeviceAddressFeatures, else in a structure of the copy's
// put at the head of the chain. Made in place, never moved.
struct DeviceCopy {
    VkDeviceCreateInfo info{};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    std::vector<std::string> added;    // extensions and features added, for the log
    std::vector<std::string> lacking;  // what the device does not have: DLSS cannot run
    std::vector<std::string> storage;  // the added names' text
    std::vector<const char*> names;
    std::vector<std::pair<VkBool32*, VkBool32>> restore;  // the host's fields turned on, and what they held
    bool copied = false;  // info is what to give vkCreateDevice (else `in`, perhaps changed in place)
};
// `in`'s chain may be changed in place (restore_device puts it back). Nothing is added, and nothing is left changed,
// when the device lacks something.
void plan_device(VkDeviceCreateInfo* in, const std::vector<std::string>& need, const std::vector<std::string>& available,
                 std::uint32_t api_version, bool bda_supported, DeviceCopy* out);
void restore_device(DeviceCopy* c);

}  // namespace ngx_bridge
