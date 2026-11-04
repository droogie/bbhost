#pragma once
// The temporal upscalers - NVIDIA DLSS and AMD FSR 3 / 4 - through Numity1504's
// upscaler_bridge.dll (third_party/upscaler_bridge: its C API, MIT), loaded at
// run time when it sits beside bbhost.exe, so bbhost needs no vendor SDK to
// build. The bridge holds the calls into NVIDIA's and AMD's code; the files
// those come in (nvngx_dlss.dll, amd_fidelityfx_loader_vk.dll,
// amd_fidelityfx_upscaler_vk.dll) sit beside it, and a missing one only turns
// its method off. Windows only; elsewhere every call is a no-op and nothing is
// available.
//
// The order (host/gpu.cpp init_locked, for the real instance and device only -
// never the throwaway probe instances): upscaler_open before Vulkan, then
// upscaler_prepare_instance / upscaler_instance_done around vkCreateInstance,
// upscaler_prepare_device / upscaler_device_done around vkCreateDevice, then
// upscaler_attach. A failed create with the bridge's copy is made again with
// the host's own (the bridge's rule): the method that needed it is then off.
//
// BBHOST_UPSCALER=0: the bridge is not loaded at all.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

// Loads the bridge and opens its context; false (and nothing else happens)
// without it.
bool upscaler_open();
// What to give vkCreateInstance: `in` itself, or the bridge's copy with what
// NGX needs. upscaler_instance_done with the result; a failure with a copy is
// made again with `in` by the caller (upscaler_instance_retry says whether).
const VkInstanceCreateInfo* upscaler_prepare_instance(const VkInstanceCreateInfo* in);
void upscaler_instance_done(VkResult r);
bool upscaler_instance_retry(const VkInstanceCreateInfo* in, const VkInstanceCreateInfo* used);
// The same for vkCreateDevice: the host's feature structures in `in`'s chain
// may be changed for the call, and are put back by upscaler_device_done.
const VkDeviceCreateInfo* upscaler_prepare_device(VkInstance instance, VkPhysicalDevice phys, const VkDeviceCreateInfo* in);
void upscaler_device_done(VkResult r);
// After the device: the vendors' runtimes started, and what can run logged.
void upscaler_attach(VkInstance instance, VkPhysicalDevice phys, VkDevice device, std::uint32_t frames_in_flight);

// What can run here, as the bridge says (attach first).
struct UpscalerCaps {
    bool loaded = false;  // the bridge is in and attached
    bool dlss = false, fsr = false, fsr4 = false;
    std::string dlss_version, fsr_versions, dlss_reason, fsr_reason;
};
UpscalerCaps upscaler_caps();

// The method asked for: Graphics > Upscaling (host/settings.h upscaler), or
// BBHOST_UPSCALE over it - 0 none, 1 DLSS ("dlss"), 2 FSR ("fsr"; "fsr3" for
// the newest 3.x where FSR 4 could run) - and only one the bridge says can
// run here.
int upscaler_method();
bool upscaler_fsr3_override();

// One upscale, recorded into `cb` outside a render pass (host/upscale_pass.cpp
// at the call point). Each image's layout is the one it is in when `cb`
// reaches the call, and it is left there. False: nothing was written to
// `output` (the method is turned off after its first error; the plain picture
// stands).
struct UpscaleImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
};
struct UpscaleCall {
    int method = 0;
    std::uint32_t quality = 0;  // 0 native (DLAA / FSR native AA)
    bool reset = false;
    std::uint32_t render_w = 0, render_h = 0, output_w = 0, output_h = 0;
    UpscaleImage color, depth, motion, output;
    float jitter_x = 0, jitter_y = 0;
    float motion_scale = 1.0f;  // the motion texel times this is render pixels (-1: a test of the wrong sign)
    float frame_ms = 16.7f, near_m = 0.1f, far_m = 10000.0f, fov_y = 1.0f;
    bool depth_inverted = false;  // depth 1 at the near plane
    std::uint32_t dlss_preset = 0;  // NGX's preset number (10 J, 11 K, 12 L, 13 M); 0 the bridge's own (K)
};
bool upscaler_evaluate(VkCommandBuffer cb, const UpscaleCall& c);
// The jitter of frame `frame` for a method and quality at an output size: the
// bridge's Halton (2, 3) over its phase count, in render pixels, +x moving the
// picture right and +y down. (0, 0) without the bridge.
void upscaler_jitter(int method, std::uint32_t quality, std::uint32_t output_w, std::uint32_t output_h, std::uint32_t frame, float* x,
                     float* y);
// The texture LOD bias for the 3D's materials at a method and quality (the
// bridge's: log2 of the render size over the output's, less 1), 0 without the
// bridge.
float upscaler_mip_bias(int method, std::uint32_t quality, std::uint32_t output_w, std::uint32_t output_h);
// The same two for a render size the qualities do not name (a Resolution
// under the window's size, engine/render_scale.cpp): the phase count AMD's
// and NVIDIA's guides give, 8 x (output / render) squared, and log2 of the
// render size over the output's, less 1.
void upscaler_jitter_sized(int method, std::uint32_t render_w, std::uint32_t output_w, std::uint32_t frame, float* x, float* y);
float upscaler_mip_bias_sized(int method, std::uint32_t render_w, std::uint32_t output_w);
