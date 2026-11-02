// The menus over the game: F9's patches and plugins, and F10's settings
// (host/options.h), in the setup window's look (host/launcher_ui.h) - a
// glass panel over the game with its pages down the left. F9 has the patch
// list, each patch turned on or off at once where it can be
// (engine/patch_manifest.h, patch_manifest_set_live) and at the next start
// where it cannot, and the plugin manager (host/plugin_ui.h): every plugin's
// settings, changed at once where a plugin says they can be, and its actions
// (Start the boss rush). Dear ImGui drawn into the presented frame with its
// Vulkan backend, after the host overlay. While one is open it takes the
// keyboard, the mouse and the pad; the pad and the arrow keys move through
// it (L1 and R1 the pages), Escape or the pad's B closes it.
//
// Two threads: the window's event pump hands its SDL events here
// (ingame_menu_event), and the presenter builds and records the menu
// (ingame_menu_record) - the ImGui context lives on the presenter, fed from a
// queue.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

union SDL_Event;

// Either menu: F9's or F10's (host_options_open).
bool ingame_menu_open();
// Pump thread. F9 opens and closes the patch and plugin menu, F10 the
// settings; while one is open every input event is taken. `pixel_ratio`: the
// window's pixels per point. Returns true when the event was the menu's.
bool ingame_menu_event(const SDL_Event& e, float pixel_ratio);
// Presenter. Once, with the swapchain's device; again after a swapchain
// rebuild changes the image count or the format.
bool ingame_menu_init(VkInstance instance, VkPhysicalDevice phys, VkDevice device, std::uint32_t family, VkQueue queue,
                      VkFormat format, std::uint32_t image_count);
void ingame_menu_shutdown();
// Pump thread, every pump: BBHOST_MENU_TEST="<s>:<step>,..." works the menus
// as a player would, seconds after the first pump - f9 and f10 (the keys),
// page:N (the open menu's page), patch:<name>=on|off (its box clicked), close.
// The log says what each step did. A test aid; nothing without the variable.
void ingame_menu_pump();
// Records the menu over `view` (an image in the colour-attachment layout).
// Nothing when it is closed.
void ingame_menu_record(VkCommandBuffer cmd, VkImageView view, VkExtent2D extent);
// The same draws into a rendering the caller has begun over an image of
// `extent` (the presenter's one pass); false, and nothing, when it is closed.
bool ingame_menu_record_in_pass(VkCommandBuffer cmd, VkExtent2D extent);
