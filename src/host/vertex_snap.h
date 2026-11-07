#pragma once

// PS1 vertex snapping (plugin API 18, BbHostApi::scene_snap): the PS1's
// geometry engine wrote each vertex's screen position in whole pixels, with no
// fraction, so a polygon's corners jumped from pixel to pixel as it moved - the
// wobble of its 3D. vertex_snap gives a stage that ends in the rasterizer (a
// vertex, tessellation-evaluation or geometry stage) the same: the clip-space
// position it writes is moved, as it leaves the stage, to the nearest corner of
// a grid of cells_x by cells_y cells over the viewport, its corners on the
// viewport's edges (window space, as the PS1's pixels were),
//
//   t = (xy / w + 1) * cells / 2;   xy = (floor(t + 0.5) / (cells / 2) - 1) * w      when w > 0,
//
// z and w as they were (the depth the same, the perspective the same). A
// vertex behind the eye (w <= 0) or so far off the screen that its place has
// no fraction left is kept. The rewrite is made where the position is final:
// before each return of a vertex or evaluation stage's entry point, before each
// emit of a geometry stage - so it meets the lifter's SPIR-V and DXC's alike,
// however they wrote the position (a variable decorated BuiltIn Position or a
// member of the gl_PerVertex block). Its arithmetic is NoContraction: two
// pipelines of one vertex shader (the depth-only and the shaded draw of a mesh)
// snap a vertex to the same corner.
//
// Host-side (not src/gcn): the translator's sources are fingerprinted, and a
// change there throws every player's translation and pipeline caches away.

#include <cstdint>
#include <string>
#include <vector>

// Rewrites `spirv` in place; false, the module untouched and `why` set, when it
// writes no position or is in a form this does not take.
bool vertex_snap(std::vector<std::uint32_t>& spirv, float cells_x, float cells_y, std::string* why = nullptr);
