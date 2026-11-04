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

// What this run's settings ask for, decided once at the start: an upscaler
// (Graphics > Upscaling, or BBHOST_UPSCALE) and an Upscaling quality above
// Native AA (or BBHOST_UPSCALE_QUALITY). False: the 3D at the full size.
// BBHOST_RENDER_SCALE=0 turns it off for a test.
bool render_scale_plan(RenderScale* out);

// The render size of an Upscaling quality for an output size: the output
// over 1, 1.5, 1.7 or 2, down to a multiple of 4 each way.
void render_scale_size(int quality, std::uint32_t output_w, std::uint32_t output_h, std::uint32_t* render_w, std::uint32_t* render_h);

// The sites of the built-in manifest "render-scale" (engine/patch_manifest.cpp)
// for a plan.
std::vector<patch_file::Site> render_scale_sites(const RenderScale& rs);

// The manifest went in: the scene contexts are made at the render size and
// the display buffers have to be exactly the output size
// (engine/live_resolution.cpp).
bool render_scale_applied();

// Right after the GX init (which makes the GPU device and attaches the
// upscaler), before the graphics manager is built: with no upscaler
// running here (no bridge, or a method this GPU lacks), the 3D goes back to the
// full size, so the picture is never a plain stretch of a smaller one.
void render_scale_gx_init(ElfImage* image);

// The 3D is drawn at the render size now and the upscaler brings it back:
// the sizes in *out when given.
bool render_scale_on(RenderScale* out = nullptr);
