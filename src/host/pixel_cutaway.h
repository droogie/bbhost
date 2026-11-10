#pragma once

// The camera cutaway per pixel (plugin API 23, BbHostApi::view_cutaway2): a
// camera draw's pixel stage leaves out the pixels that stand between a high
// camera and the character it looks at, fading them out over a band - a
// screen-door dither, since the G-buffer has no blending. Each pixel's place
// in view space comes back from its window position and depth:
//
//   n = (frag.xy - offset) / scale             (the viewport's, so n.y is up)
//   w = 1 / (depth_z frag.z + depth_w);  v = (n.x tan_x w, n.y tan_y w, w)
//
// (the game's own decode of its depth, read with the tangents from the
// camera's view block - upscale_view_terms; frag.w is not the view depth in
// the host's translated stages)
//
// and it is cut where all of these hold, each by the feather or more (less
// fades it):
//
//   |v.xy| < radius + widen * max(0, distance - w)   about the view's axis, wider toward the camera
//   v.y cos(pitch) - (w - distance) sin(pitch) > height   that high over the point looked at
//   w < distance - margin                       in front of it
//
// (a pixel not in front of it skips the rest: a branch the ground beyond the
// hunter, most of a frame, takes).
//
// The camera only turns and moves, it never tilts, so these are the same
// numbers whatever it does: baked in as constants, a twin per set - or, with
// a live buffer (PixelCutLive at a device address), the ones a zoom or a
// field of view changes are read from it each pixel, so the twin stays the
// same however the view moves: no pipelines built while it zooms.
//
// EarlyFragmentTests is dropped from the twin: a pixel left out must not
// have written its depth first. Host-side (not src/gcn): the translator's
// sources are fingerprinted.

#include <cstdint>
#include <string>
#include <vector>

struct PixelCut {
    float scale_x, offset_x, scale_y, offset_y;  // the viewport (window = n * scale + offset)
    float tan_x, tan_y;                          // half the field of view's tangents
    float distance;                              // the view depth of the point looked at
    float sin_pitch, cos_pitch;                  // the view's angle below the horizon
    float radius, widen, height, margin, feather;
    float depth_z, depth_w;  // the game's depth decode: view depth = 1 / (depth_z frag.z + depth_w)
};

// The numbers a twin reads at `live` (a device address, 4-aligned) instead of
// baking them in. Its PixelCut then gives the viewport, height, margin and
// feather; its distance says which distance the cone's tip stands at - 1 the
// map's, 2 the objects' - and the rest is 0.
struct PixelCutLive {
    float tan_x, tan_y;
    float distance;
    float sin_pitch, cos_pitch;
    float radius, widen;
    float depth_z, depth_w;
    float object_distance;  // the objects' cone's tip (scene_look_object_cut)
};

// Rewrites `spirv` (a fragment stage) in place; false, the module untouched
// and `why` set, when it is in a form this does not take. `live` 0: every
// number baked in.
bool pixel_cutaway(std::vector<std::uint32_t>& spirv, const PixelCut& cut, std::string* why = nullptr, std::uint64_t live = 0);

// A fragment stage of the cutaway alone - no outputs, only the test - for a
// camera draw with no pixel stage of its own (the map's depth-only passes,
// which would otherwise leave the cut surfaces' depth behind for the lighting).
// Empty when the rewrite refuses it.
std::vector<std::uint32_t> pixel_cutaway_only(const PixelCut& cut, std::uint64_t live = 0);
