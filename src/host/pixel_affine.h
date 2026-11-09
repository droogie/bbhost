#pragma once

// PS1 affine texture mapping (plugin API 27, BbHostApi::scene_affine): the
// PS1's GPU had no depth per pixel, so it stepped a polygon's texture
// coordinates evenly across the screen - straight lines between its corners,
// not the perspective's - and a texture slid and bent over a polygon as it
// turned toward or away from the camera, the more the larger it was on the
// screen. pixel_affine gives a pixel stage the same: its inputs that are its
// texture coordinates are decorated NoPerspective, interpolated linearly in
// window space.
//
// Which inputs are the texture coordinates: those whose values reach the
// coordinate of an image sample, read or gather - each instruction's result
// carries the inputs its operands came from (through the stage's own
// variables too), forward to a fixed point. Not those divided by an input on
// the way (a screen position, xy / w: the soft particles' depth read, the
// light volumes' G-buffer reads): those keep their perspective, as the PS1's
// screen had none of them. A sample's result carries none: a texture read
// from the coordinates is not one. Flat, built-in and already linear inputs,
// and inputs that are not floats, are left alone.
//
// Only the fragment stage's decoration decides the interpolation (Vulkan does
// not ask the vertex stage's outputs to match): the vertex stage stays as it
// is. Host-side (not src/gcn): the translator's sources are fingerprinted.

#include <cstdint>
#include <string>
#include <vector>

// Rewrites `spirv` (a fragment stage) in place; false, the module untouched
// and `why` set, when it has no texture coordinate to take or is in a form
// this does not take. `count` (when given): the inputs made linear.
bool pixel_affine(std::vector<std::uint32_t>& spirv, std::string* why = nullptr, std::uint32_t* count = nullptr);
