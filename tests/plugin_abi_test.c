/* The plugin API's version 27 structures as a C compiler lays them out - the
 * layout plugins are built against, which a host may append to and never
 * move: their sizes, the places of the fields read most, and the API table's
 * version 27 calls right after version 26's camera_zoom. Built as C99, so the
 * header stays C. */
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

int main(void) {
    printf("plugin_abi_test: version %d's structures as plugins see them\n", BB_PLUGIN_API_VERSION);
    return BB_PLUGIN_API_VERSION >= 27 ? 0 : 1;
}
