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
// Each flip it generates for needs three things, all of that flip: AMD's
// prepare (the 3D's depth, the motion the upscaler made and the camera),
// recorded in the game's command buffer after the upscale; the HUD-less
// picture, the display buffer copied before the HUD's first draw; and that
// picture brought to the swapchain's size as the presenter brings the frame.
// A flip missing one - no upscaler, no HUD-less copy, a flip skipped - is
// presented as it is.

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
// Frames shown for each of the game's, measured over the last 60 presents:
// about 2 while generating, 1 otherwise (the FPS counter's factor).
float framegen_shown_per_frame();
// The contexts and AMD's swapchain gone; the presenter makes a swapchain anew.
void framegen_destroy_swapchain(VkDevice device);

// The game's side, under the renderer's lock: AMD's prepare for `flip`,
// recorded into `cb` after the upscale. Depth and motion at the render size;
// motion in render pixels, the way the upscaler takes it.
struct FramegenInputs {
    VkImage depth = VK_NULL_HANDLE;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkImageLayout depth_layout = VK_IMAGE_LAYOUT_GENERAL;
    VkImage motion = VK_NULL_HANDLE;  // R16G16_SFLOAT, GENERAL
    std::uint32_t render_w = 0, render_h = 0;
    float jitter_x = 0.0f, jitter_y = 0.0f, motion_scale = 1.0f;
    float frame_ms = 16.7f, near_m = 0.1f, far_m = 10000.0f, fov_y = 1.0f;
};
// Frames are numbered by the game's side, one a frame prepared: `id`. False:
// not prepared (no generation for this frame).
bool framegen_prepare(VkCommandBuffer cb, std::uint64_t id, const FramegenInputs& in);
// The HUD-less picture of frame `id` (GENERAL, the display buffer's size and
// format), copied from the display buffer at `display_va`, which the
// presenter finds again by that buffer when it presents it.
// `display_w` x `display_h` is the display buffer's size, which the copy -
// the HUD's target, the output size - covers whole (render scale's back
// buffer can be smaller than the display buffers).
void framegen_note_hudless(std::uint64_t display_va, std::uint64_t id, VkImage image, VkFormat format, std::uint32_t w,
                           std::uint32_t h, std::uint32_t display_w, std::uint32_t display_h);
// The presenter's side, in its command buffer after the frame's blit: the
// HUD-less picture of the frame in `display_va` brought to the swapchain's
// size over the same rectangles, for AMD to take the UI out of the frame with.
void framegen_frame(VkCommandBuffer cmd, std::uint64_t display_va, VkOffset2D dst, VkExtent2D dst_size, std::int32_t src_x,
                    std::int32_t src_y, std::uint32_t src_w, std::uint32_t src_h);
