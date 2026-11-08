#pragma once

// The presenter's one pass over the swapchain image (host/window.cpp). The
// game's display image is drawn into the picture's rectangle - a full-screen
// triangle sampling it, which is what vkCmdBlitImage does with a linear filter
// - and the host overlay (the FPS counter, the pointer, the text box, F10's
// screen) and the plugin menu (F9) go into the same rendering. The bars beside
// a picture of another shape are the rendering's clear.
//
// What it replaces, per presented frame: a transfer-layout change, a clear of
// the whole image when there are bars, the blit, and - with anything on the
// overlay - a second layout change and a second rendering that loaded the
// image the blit had just stored. Now: one layout change in, one rendering,
// one out. A magnified picture keeps the blit (or FSR 1's compute passes, host/fsr.cpp, when
// BBHOST_UPSCALE=fsr1 asks for them).

#include <vulkan/vulkan.h>

#include <cstdint>

// Presenter-owned, for the swapchain's format; safe to call repeatedly.
bool present_pass_init(VkDevice device, VkPhysicalDevice phys, VkFormat colour_format);
void present_pass_shutdown(VkDevice device);
bool present_pass_ready();

// The game's display image and the region of it shown.
struct PresentSource {
    VkImage image = VK_NULL_HANDLE;  // in GENERAL, written by earlier submissions on the renderer's queue
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0, height = 0;  // the image's size
    std::uint32_t x = 0, y = 0, w = 0, h = 0;  // the region
};

// Records the layout change of `dst` (from UNDEFINED, ordered after
// COLOR_ATTACHMENT_OUTPUT, the stage the acquire is waited at), begins a
// rendering over `dst_view` that clears to `clear` unless `shown` covers the
// whole image, and draws `src` (null: nothing, the clear is the frame) into
// `shown`. The rendering stays open for the overlay and the menu; end it with
// present_pass_end. False, with nothing recorded, when the pass cannot sample
// the source (the caller blits instead). Call after the presenter's fence: the
// last frame's view of the source and its descriptor are reused here.
bool present_pass_begin(VkCommandBuffer cmd, VkImage dst, VkImageView dst_view, VkExtent2D extent, VkRect2D shown,
                        const PresentSource* src, const float clear[4]);
// Ends the rendering and moves `dst` to PRESENT_SRC_KHR.
void present_pass_end(VkCommandBuffer cmd, VkImage dst);
