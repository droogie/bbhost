/* The plugin API's version 29 structures as a C compiler lays them out - the
 * layout plugins are built against, which a host may append to and never
 * move: their sizes, the places of the fields read most, and the API table's
 * version 27 calls right after version 26's camera_zoom - the frame's three,
 * then the shaders' five, followed by version 28's camera target hook and
 * version 29's world pause, controller and world state.
 * Built as C99, so the header stays C. */
#include "bbhost_plugin.h"

#include <stddef.h>
#include <stdio.h>

_Static_assert(sizeof(BbCamera) == 80, "BbCamera");
_Static_assert(sizeof(BbFrameCamera) == 336, "BbFrameCamera");
_Static_assert(offsetof(BbFrameCamera, view) == 80, "BbFrameCamera.view");
_Static_assert(offsetof(BbFrameCamera, inv_view_proj) == 272, "BbFrameCamera.inv_view_proj");
_Static_assert(sizeof(BbFrameInfo) == 736, "BbFrameInfo");
_Static_assert(offsetof(BbFrameInfo, frame) == 8, "BbFrameInfo.frame");
_Static_assert(offsetof(BbFrameInfo, render_width) == 24, "BbFrameInfo.render_width");
_Static_assert(offsetof(BbFrameInfo, method) == 40, "BbFrameInfo.method");
_Static_assert(offsetof(BbFrameInfo, jitter_x) == 48, "BbFrameInfo.jitter_x");
_Static_assert(offsetof(BbFrameInfo, reset) == 60, "BbFrameInfo.reset");
_Static_assert(offsetof(BbFrameInfo, now) == 64, "BbFrameInfo.now");
_Static_assert(offsetof(BbFrameInfo, prev) == 400, "BbFrameInfo.prev");
_Static_assert(sizeof(BbRenderImage) == 48, "BbRenderImage");
_Static_assert(offsetof(BbRenderImage, format) == 24, "BbRenderImage.format");
_Static_assert(offsetof(BbRenderImage, flags) == 44, "BbRenderImage.flags");
_Static_assert(sizeof(BbRenderDevice) == 56, "BbRenderDevice");
_Static_assert(offsetof(BbRenderDevice, get_device_proc_addr) == 40, "BbRenderDevice.get_device_proc_addr");
_Static_assert(offsetof(BbRenderDevice, frames_in_flight) == 52, "BbRenderDevice.frames_in_flight");
_Static_assert(sizeof(BbRenderCall) == 272, "BbRenderCall");
_Static_assert(offsetof(BbRenderCall, command_buffer) == 8, "BbRenderCall.command_buffer");
_Static_assert(offsetof(BbRenderCall, scene) == 32, "BbRenderCall.scene");
_Static_assert(offsetof(BbRenderCall, picture) == 224, "BbRenderCall.picture");
_Static_assert(offsetof(BbHostApi, frame_info) == offsetof(BbHostApi, camera_zoom) + sizeof(void*), "frame_info after camera_zoom");
_Static_assert(offsetof(BbHostApi, on_render) == offsetof(BbHostApi, frame_info) + sizeof(void*), "on_render after frame_info");
_Static_assert(offsetof(BbHostApi, aa_claim) == offsetof(BbHostApi, on_render) + sizeof(void*), "aa_claim after on_render");
_Static_assert(BB_PLUGIN_AA_LEAVE == 16u && BB_PLUGIN_AA_TAKE == 32u, "the claim's flags");
_Static_assert(BB_AA_NONE == 0 && BB_AA_LEAVE == 1 && BB_AA_TAKE == 2, "the claims");
_Static_assert(offsetof(BbHostApi, shader_override) == offsetof(BbHostApi, aa_claim) + sizeof(void*), "shader_override after aa_claim");
_Static_assert(offsetof(BbHostApi, shader_folder) == offsetof(BbHostApi, shader_override) + sizeof(void*), "shader_folder after shader_override");
_Static_assert(offsetof(BbHostApi, shader_state) == offsetof(BbHostApi, shader_folder) + sizeof(void*), "shader_state after shader_folder");
_Static_assert(offsetof(BbHostApi, shader_source) == offsetof(BbHostApi, shader_state) + sizeof(void*), "shader_source after shader_state");
_Static_assert(offsetof(BbHostApi, shader_data) == offsetof(BbHostApi, shader_source) + sizeof(void*), "shader_data after shader_source");
_Static_assert(offsetof(BbHostApi, on_camera_target) == offsetof(BbHostApi, shader_data) + sizeof(void*), "on_camera_target after shader_data");
_Static_assert(offsetof(BbHostApi, world_pause) == offsetof(BbHostApi, on_camera_target) + sizeof(void*), "world_pause after on_camera_target");
_Static_assert(offsetof(BbHostApi, world_step) == offsetof(BbHostApi, world_pause) + sizeof(void*), "world_step after world_pause");
_Static_assert(offsetof(BbHostApi, pad) == offsetof(BbHostApi, world_step) + sizeof(void*), "pad after world_step");
_Static_assert(offsetof(BbHostApi, pad_take) == offsetof(BbHostApi, pad) + sizeof(void*), "pad_take after pad");
_Static_assert(offsetof(BbHostApi, world_state) == offsetof(BbHostApi, pad_take) + sizeof(void*), "world_state after pad_take");
_Static_assert(offsetof(BbHostApi, hud_hide) == offsetof(BbHostApi, world_state) + sizeof(void*), "hud_hide after world_state");
_Static_assert(offsetof(BbHostApi, post_look) == offsetof(BbHostApi, hud_hide) + sizeof(void*), "post_look after hud_hide");
_Static_assert(offsetof(BbHostApi, panel_show) == offsetof(BbHostApi, post_look) + sizeof(void*), "panel_show after post_look");
_Static_assert(offsetof(BbHostApi, panel_state) == offsetof(BbHostApi, panel_show) + sizeof(void*), "panel_state after panel_show");
_Static_assert(offsetof(BbHostApi, option_set) == offsetof(BbHostApi, panel_state) + sizeof(void*), "option_set after panel_state");
_Static_assert(sizeof(BbHostApi) == offsetof(BbHostApi, option_set) + sizeof(void*), "option_set ends the API table");
_Static_assert(BB_OPT_PANEL == 0x400u && BB_PANEL_DEFAULTS == 1u, "the panel's flags");
_Static_assert(offsetof(BbPanel, page) == 8 && sizeof(BbPanel) == 8 + 3 * sizeof(void*), "BbPanel");
_Static_assert(sizeof(BbPanelState) == 44 && offsetof(BbPanelState, page) == 12, "BbPanelState");
_Static_assert(sizeof(BbLook) == 112, "BbLook");
_Static_assert(offsetof(BbLook, vignette) == 28 && offsetof(BbLook, dof) == 32 && offsetof(BbLook, aberration) == 52, "BbLook's groups");
_Static_assert(offsetof(BbLook, motion_blur) == 72 && offsetof(BbLook, blur_samples) == 88 && offsetof(BbLook, fog_distance) == 92 &&
                   offsetof(BbLook, fog_height_offset) == 108,
               "BbLook's last groups");
_Static_assert(sizeof(BbPad) == 40, "BbPad");
_Static_assert(offsetof(BbPad, pressed) == 8 && offsetof(BbPad, connected) == 12, "BbPad's presses");
_Static_assert(offsetof(BbPad, lx) == 16 && offsetof(BbPad, r2) == 36, "BbPad's axes");
_Static_assert(sizeof(BbWorldState) == 20, "BbWorldState");
_Static_assert(offsetof(BbWorldState, others) == 8 && offsetof(BbWorldState, session) == 16, "BbWorldState's fields");
_Static_assert(BB_PAD_CROSS == 0x4000u && BB_PAD_CIRCLE == 0x2000u && BB_PAD_L1 == 0x400u && BB_PAD_R1 == 0x800u &&
                   BB_PAD_L3 == 0x2u && BB_PAD_R3 == 0x4u && BB_PAD_OPTIONS == 0x8u && BB_PAD_TOUCHPAD == 0x100000u,
               "the controller's buttons");
_Static_assert(BB_SESSION_NONE == 0 && BB_SESSION_HOST == 3 && BB_SESSION_GUEST == 6 && BB_SESSION_LEAVING == 7, "the session roles");
_Static_assert(BB_SHADER_NONE == 0 && BB_SHADER_WAITING == 1 && BB_SHADER_TAKEN == 2 && BB_SHADER_REFUSED == 3 && BB_SHADER_SHADOWED == 4 &&
                   BB_SHADER_OFF == 5,
               "shader_state's values");
_Static_assert(BB_SHADER_DATA_BYTES == 65536u, "a plugin's shader data");

int main(void) {
    printf("plugin_abi_test: version %d's structures as plugins see them\n", BB_PLUGIN_API_VERSION);
    return BB_PLUGIN_API_VERSION >= 29 ? 0 : 1;
}
