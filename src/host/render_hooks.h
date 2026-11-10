#pragma once

// Plugin API 27's frame part (include/bbhost_plugin.h frame_info, on_render):
// each 3D frame as host/upscale_pass.cpp's call point describes it - its
// sizes, the upscaler and its jitter, this frame's and the last frame's
// cameras - kept for the plugins and published for frame_info from any
// thread, and the render hooks the plugins register, which the call point
// calls (host/upscale_pass.cpp: the barriers around them, the images). No
// Vulkan here: the images and the device come in as their handles. Nothing is
// described or called until a plugin asks (render_hooks_on): until then the
// call point is as it was.

#include "bbhost_plugin.h"

#include <cstdint>
#include <string>
#include <vector>

// Whether a plugin takes the frames: frame_info has been called, or a plugin
// that is on has a hook. The draws read it (render.cpp, host/upscale_pass.cpp,
// gpu.cpp's deferred destruction): any thread, one relaxed load.
bool render_hooks_on();
// Whether a plugin that is on has a render hook (frame_info alone describes
// the frames, and calls nothing). The same load.
bool render_hooks_registered();
// Whether a hook of a plugin that is on asks for the frame's motion
// (BB_RENDER_MOTION): the upscaler's motion is made then without an upscaler
// too (upscale_motion_on). The same load.
bool render_hooks_want_motion();
// frame_info (host/plugins.cpp): the frames described from its first call
// on; the frame being recorded inside a hook, else the last one described,
// into *out, as much as both know (out->size). 0 on success; 1 before a frame
// is described, or for a size under 8.
int render_frame_info(BbFrameInfo* out);

// on_render (host/plugins.cpp): fn's points for `name`'s plugin, replacing
// what fn and user had; no points (BB_RENDER_BEFORE_UPSCALE | _AFTER_UPSCALE)
// take it away. 0 on success; 1 for a bad argument. *motion_changed: whether
// render_hooks_want_motion changed with it.
int render_hooks_add(const char* name, std::uint32_t points, BbRenderFn fn, void* user, bool* motion_changed);
// A plugin turned on or off (a live switch; a visitor is off): its hooks
// called or not. Main thread. True when render_hooks_want_motion changed.
bool render_hooks_plugin_on(const std::string& name, bool on);
// A plugin unloaded: its hooks go.
void render_hooks_drop(const std::string& name);
// Inside a hook call, on its thread: frame_info's frame is the one being
// recorded, and aa_claim refuses.
bool render_hooks_in_call();

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
// The frame being recorded: the last one described. Under g.mu.
const BbFrameInfo* render_frame_current_locked();

// A hook at a point, for the call point to call: fn, user, and its name for
// the GPU profile ("plugin <name> before|after", kept for the run).
struct RenderHook {
    BbRenderFn fn = nullptr;
    void* user = nullptr;
    const std::string* profile = nullptr;
};
// The hooks of plugins that are on at `point` (BB_RENDER_BEFORE_UPSCALE or
// _AFTER_UPSCALE), in the order they were registered.
void render_hooks_take(std::uint32_t point, std::vector<RenderHook>* out);
// Around the calls, on their thread (render_hooks_in_call, frame_info).
void render_hooks_enter(const BbFrameInfo* frame);
void render_hooks_leave();
// The images a call hands over, by their place in BbRenderCall: their
// generation now (render_api.h image_generation). Under g.mu.
enum RenderImageSlot { kSlotScene, kSlotDepth, kSlotMotion, kSlotGlow, kSlotPicture, kSlotCount };
std::uint64_t render_hooks_generation_locked(RenderImageSlot slot, std::uint64_t image, std::uint64_t view, std::uint32_t format,
                                             std::uint32_t width, std::uint32_t height);
// The host lets go of an image or a view (gpu.cpp's deferred destruction):
// a slot that handed it out gives a new generation. Under g.mu.
void render_hooks_gone_locked(std::uint64_t handle);

}  // namespace gpu
