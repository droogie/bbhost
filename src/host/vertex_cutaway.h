#pragma once

// A camera cutaway (plugin API 22, BbHostApi::view_cutaway): what stands
// between a high camera and the character it looks at - a roof, a ceiling, the
// near wall - left out, as an isometric game cuts its walls away. vertex_cutaway
// gives a stage that ends in the rasterizer (a vertex, tessellation-evaluation
// or geometry stage) a clip distance from its clip-space position as it leaves
// the stage. The cylinder cut is about the view's axis, which runs through the
// character at the screen's middle, in view space:
//
//   v.xy = clip.xy * tan;  d = max(|v.xy| - radius, clip.w - near, 1.5 - clip.w)
//
// clipped where d < 0: within `radius` of the axis and nearer than `near` (the
// character's distance less a margin) - clip.w is the view depth and clip.xy *
// tan(half field of view) its place across - but not nearer than 1.5: the
// full-screen passes drawn with the camera's depth bound (the lighting, the
// composite, the HUD) are triangles at w = 1. Camera-relative, so the numbers
// change only with the view's distance and field of view, not as it moves:
// baked into the stage as constants, a twin per set. The distance is per
// vertex and interpolated across each triangle, so a large triangle wholly
// outside the cylinder at its corners stays even where it crosses it.
//
// A stage writing clip distances of its own is refused. Host-side (not
// src/gcn): the translator's sources are fingerprinted.

#include <cstdint>
#include <string>
#include <vector>

// `cut`: tan_x, tan_y (half the field of view's tangents), near (view depth),
// radius (view-space units, metres). Rewrites `spirv` in place; false, the
// module untouched and `why` set, when it writes no position or is in a form
// this does not take.
bool vertex_cutaway(std::vector<std::uint32_t>& spirv, const float cut[4], std::string* why = nullptr);
