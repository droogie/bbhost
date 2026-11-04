#pragma once

// Frame generation: AMD FidelityFX frame generation (SDK 2.4.0) behind the
// presenter. Its swapchain stands in for the presenter's own - acquire,
// present and the images through its replacements - and, between two real
// frames, presents one it generated from the 3D alone: the game's HUD and the
// menus are taken out (the back buffer against the frame before the HUD's
// first draw) and put back over every frame shown, real or generated.
//
// The runtime is AMD's: amd_fidelityfx_loader_vk.dll and
// amd_fidelityfx_framegeneration_vk.dll beside bbhost.exe. Without them, the
// device's three spare queues (host/gpu.h fg_queues) or the setting, the
// presenter keeps a swapchain of its own and nothing here runs.
//
// Built in steps (build/HANDOFF-upscaler.md): this one puts AMD's swapchain
// in place with generation off - every frame presented as it comes - so the
// swapchain and its pacing are proven before a frame is generated.

#include <vulkan/vulkan.h>

#include <cstdint>

// The setting asks for it (read once, as the presenter starts) and the runtime
// loaded. False keeps the presenter's own swapchain.
bool framegen_wanted();
// AMD's swapchain from `sci` (its oldSwapchain ignored: a swapchain there is
// first destroyed with framegen_destroy_swapchain), presenting on `game_queue`
// - the presenter's own - with `fg_queues` (present, image acquire, async
// compute) for its own work, and the frame generation context for it. False:
// nothing made, the presenter makes its own.
bool framegen_create_swapchain(VkPhysicalDevice phys, VkDevice device, const VkSwapchainCreateInfoKHR& sci, VkQueue game_queue,
                               std::uint32_t family, void* const fg_queues[3], VkSwapchainKHR* out);
// Its swapchain is the presenter's now.
bool framegen_active();
// The replacements, for its swapchain.
VkResult framegen_get_images(VkDevice device, VkSwapchainKHR swapchain, std::uint32_t* count, VkImage* images);
VkResult framegen_acquire(VkDevice device, VkSwapchainKHR swapchain, std::uint64_t timeout, VkSemaphore sem, VkFence fence,
                          std::uint32_t* index);
// The frame's settings first (frameID counting each presented game frame by
// one), then AMD's present.
VkResult framegen_present(VkQueue queue, const VkPresentInfoKHR* info);
// The contexts and AMD's swapchain gone; the presenter makes a swapchain anew.
void framegen_destroy_swapchain(VkDevice device);
