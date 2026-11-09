#pragma once

// Plugin API 27's frame part (include/bbhost_plugin.h frame_info): each 3D
// frame as host/upscale_pass.cpp's call point describes it - its sizes, the
// upscaler and its jitter, this frame's and the last frame's cameras - kept
// for the plugins and published for frame_info from any thread. Nothing is
// described until a plugin asks (render_hooks_on): until then the call point
// is as it was.

#include "bbhost_plugin.h"

#include <cstdint>

// Whether a plugin takes the frames: frame_info has been called. The draws
// read it (render.cpp, host/upscale_pass.cpp): any thread, one relaxed load.
bool render_hooks_on();
// frame_info (host/plugins.cpp): the frames described from its first call
// on; the last one described into *out, as much as both know (out->size).
// 0 on success; 1 before a frame is described, or for a size under 8.
int render_frame_info(BbFrameInfo* out);

namespace gpu {

// What the call point knows of a 3D frame (host/upscale_pass.cpp): its sizes,
// what takes the 3D to the picture (BB_UPSCALE_*), the quality and jitter the
// upscaler is told, and the view block's terms (camera_of's) when the frame's
// camera was seen.
struct RenderFrameDesc {
    std::uint32_t render_w = 0, render_h = 0, output_w = 0, output_h = 0;
    std::uint32_t method = BB_UPSCALE_NONE, quality = BB_QUALITY_NATIVE;
    float jitter_x = 0.0f, jitter_y = 0.0f;
    bool camera = false;
    float tan_x = 0.0f, tan_y = 0.0f, a = 0.0f, r2w = 0.0f;
    float rows[12] = {};
};
// The frame described - once a call point - and published. Under g.mu.
void render_frame_describe_locked(const RenderFrameDesc& d);

}  // namespace gpu
