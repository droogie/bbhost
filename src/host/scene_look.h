#pragma once

// The 3D's look a plugin asks for (plugin API 18, include/bbhost_plugin.h:
// scene_snap, scene_filter, scene_scale, scene_colour_bits, grading_filter) -
// the PS1 Look plugin's (plugins/ps1). Only the state: render.cpp snaps the
// camera draws' vertices (gcn/vertex_snap.h) and samples their textures as it
// says, engine/render_scale.cpp and host/options.cpp size the 3D by it,
// engine/patch_manifest.cpp's scene-rgba8 makes the colour targets RGBA8.
// Any thread.

#include <cstdint>

// Vertex snap: the grid's lines, 0 off, kSnapPixels the 3D's own pixels.
constexpr std::uint32_t kSnapPixels = 0xffffffffu;
void scene_look_set_snap(std::uint32_t lines);
std::uint32_t scene_look_snap();
// The grid for a 3D of w x h: its cells across and down (false: no snap).
bool scene_look_snap_cells(std::uint32_t w, std::uint32_t h, float* cells_x, float* cells_y);

// The camera draws' textures sampled nearest.
void scene_look_set_nearest(bool on);
bool scene_look_nearest();

// The 3D at a size of its own, nearest up to the window's (scene_scale):
// `lines` high, 0 the Resolution.
void scene_look_set_scale(bool nearest, std::uint32_t lines);
bool scene_look_scale(std::uint32_t* lines = nullptr);

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
