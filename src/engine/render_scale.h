#pragma once

// Render scale: the world drawn smaller than the picture and the temporal
// upscaler (host/upscaler.h) bringing it back to the full size before the
// post chain - YEBIS, the HUD and the menus stay at the Resolution
// (engine/render_scale.cpp has the game's side, host/upscale_pass.cpp and
// host/render.cpp the renderer's).

#include "engine/patch_manifest_file.h"

#include <cstdint>
#include <vector>

struct ElfImage;

struct RenderScale {
    std::uint32_t render_w = 0, render_h = 0;  // the 3D: the scene contexts' size
    std::uint32_t output_w = 0, output_h = 0;  // the picture: the Resolution, YEBIS, the HUD
    int quality = 0;                           // Upscaling quality: 1 Quality, 2 Balanced, 3 Performance
};

// What the settings ask for at the start: the Resolution, and the render
// size of the Upscaling quality when an upscaler is chosen (the full size
// otherwise). False only when render scale is off (BBHOST_RENDER_SCALE=0, or
// no live resolution: its changes are the live resolution's rebuilds).
// BBHOST_UPSCALE_QUALITY stands for the setting in tests.
bool render_scale_plan(RenderScale* out);

// The render size of an Upscaling quality for an output size: the output
// over 1, 1.5, 1.7 or 2, down to a multiple of 4 each way.
void render_scale_size(int quality, std::uint32_t output_w, std::uint32_t output_h, std::uint32_t* render_w, std::uint32_t* render_h);

// The sites of the built-in manifest "render-scale" (engine/patch_manifest.cpp)
// for a plan.
std::vector<patch_file::Site> render_scale_sites(const RenderScale& rs);

// The manifest went in: the scene contexts' sizes are this file's to write.
bool render_scale_applied();

// At load, after live resolution's own code edits: the scene's back-buffer
// hooks, and the settings listener that asks for a rebuild when the upscaler
// or its quality changes (engine/live_resolution.h live_resolution_rebuild).
void render_scale_install(ElfImage* image);

// Right after the GX init (which makes the GPU device and attaches the
// upscaler), before the graphics manager is built: the sizes for the settings
// as they are, the display buffers being display_w x display_h.
void render_scale_gx_init(std::uint32_t display_w, std::uint32_t display_h, std::uint64_t flip);

// Before a rebuild at an output size (live resolution's quiet point, nothing
// recording): the render size the settings ask for there - the full size
// with no upscaler running - written into the code, and the scene's back
// buffer chosen. The sizes before and after in *before / *after.
bool render_scale_retarget(std::uint32_t output_w, std::uint32_t output_h, std::uint32_t display_w, std::uint32_t display_h, std::uint64_t flip,
                           RenderScale* before, RenderScale* after);

// Each frame's end: back buffers the scene let go of, a few flips on.
void render_scale_flip(std::uint64_t flip);

// The 3D is drawn at a render size smaller than the output now: the sizes in
// *out when given.
bool render_scale_on(RenderScale* out = nullptr);

// HD motion blur's share of the 3D for its velocity map and depth (`shift`
// right: 2 a quarter, 1 half, 0 full) while a plugin's look draws the 3D
// smaller than the window (host/scene_look.h): the share of a power of two
// nearest half the window's height when the asked one is below that -
// negative a shift left, a map larger than the 3D. `shift` itself otherwise.
int render_scale_blur_shift(int shift);
