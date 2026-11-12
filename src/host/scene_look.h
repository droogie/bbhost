#pragma once

// The 3D's look a plugin asks for (plugin API 18, include/bbhost_plugin.h:
// scene_snap, scene_filter, scene_scale, scene_colour_bits, grading_filter) -
// the Bloodborne PS1 plugin's (plugins/ps1). Only the state: render.cpp snaps the
// camera draws' vertices (gcn/vertex_snap.h) and samples their textures as it
// says, engine/render_scale.cpp and host/options.cpp size the 3D by it,
// engine/patch_manifest.cpp's scene-rgba8 makes the colour targets RGBA8.
// Any thread.

#include <cstdint>

// Vertex snap: the grid's lines, 0 off, kSnapPixels the 3D's own pixels.
constexpr std::uint32_t kSnapPixels = 0xffffffffu;
void scene_look_set_snap(std::uint32_t lines);
std::uint32_t scene_look_snap();
// Called (the plugin's thread) after the snap changes: render.cpp makes its
// twins of the manifest's pipelines ahead (look_prewarm_start).
void scene_look_on_snap_change(void (*fn)());
// The grid for a 3D of w x h: its cells across and down (false: no snap).
bool scene_look_snap_cells(std::uint32_t w, std::uint32_t h, float* cells_x, float* cells_y);

// The cutaway (plugin API 22, host/vertex_cutaway.h): the camera draws clipped
// within `radius` of the view's axis nearer than `near`; radius 0 off. The
// getter fills tan_x, tan_y, near, radius (when `out` is given) and says
// whether it is on.
void scene_look_set_cutaway(float tan_x, float tan_y, float near_z, float radius);
bool scene_look_cutaway(float* out);

// The cutaway per pixel (plugin API 23, host/pixel_cutaway.h): its numbers
// but the viewport, which each draw gives; radius 0 off. The getter fills
// them (when `out` is given) and says whether it is on.
struct PixelCut;
void scene_look_set_pixel_cut(const PixelCut& cut);
bool scene_look_pixel_cut(PixelCut* out);
// The objects' cone (plugin API 24): the same cut for the map's objects, its
// tip `start` of the way from the point looked at toward the camera; below 0
// the objects are never cut.
void scene_look_set_object_cut(float start);
float scene_look_object_cut();

// The camera draws' textures sampled nearest.
void scene_look_set_nearest(bool on);
bool scene_look_nearest();

// The 3D at a size of its own, nearest up to the window's (scene_scale):
// `lines` high, 0 the Resolution. Not while an upscaler is chosen and there
// (upscaler_method): FSR or DLSS then brings the 3D up from its own size.
void scene_look_set_scale(bool nearest, std::uint32_t lines);
bool scene_look_scale(std::uint32_t* lines = nullptr);

// The finished picture dithered to `bits` a channel before the HUD (plugin
// API 25, host/upscale_pass.cpp scene_dither_locked): 0 off.
void scene_look_set_dither(std::uint32_t bits);
std::uint32_t scene_look_dither();
// The game's HUD, menus and prompts left out of the picture (plugin API 29
// hud_hide), held by each plugin that asks, by name: render.cpp keeps the
// picture as it was before the game's first 2D draw into its target (frame
// generation's HUD-less point) and puts it back before the display buffer's
// pass composites that target (host/upscale_pass.cpp hud_hide_*_locked). The
// game's own state - its HUD option - is untouched.
void scene_look_set_hud_hidden(const char* owner, bool hidden);
bool scene_look_hud_hidden();
// Every hold of `owner` let go (the plugin turned off).
void scene_look_drop_hud(const char* owner);

// The scene's colour targets RGBA8.
void scene_look_set_colour_8bit(bool on);
bool scene_look_colour_8bit();

// The colour grading's table sampled nearest (grading_filter): the 16x16x16
// table the area's CGrading textures are copied into each frame, which the
// grading pass samples (render.cpp look_samplers).
void scene_look_set_grading_nearest(bool on);
bool scene_look_grading_nearest();

// Whether any of it takes the camera's draws (the snap, the nearest
// textures, the 3D's own size): render.cpp then learns the scene and its depth
// without an upscaler.
bool scene_look_camera();

// Called (main thread or the menus') after a change the scene's targets
// follow - the scale or the colour bits - once the game runs: they are made
// again at the next quiet point. Set by engine/render_scale.cpp.
void scene_look_on_targets_change(void (*fn)());
