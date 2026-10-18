#pragma once

// Host changes to the game's own GCN programs, made where the host reads a
// program's words - gpu::extract_program (every stage, at the draw or the
// dispatch) and the pixel-shader precompile (the container, at creation) - so
// both see the same program and the stage caches, keyed by the words, agree.
// A program is named by its OrbShdr footer hash ("%08x"), and every change
// is checked against the exact words it replaces.
//
// Motion blur: the game's own switch for it (the +0x2969 bool the renderer's
// constructor sets, which the shadPS4 community patch clears) blows the frame
// out in this renderer, for a reason not yet found. So the game keeps its
// motion blur pass, and when the
// setting is off that pass writes every pixel through unblurred: YEBIS's blur
// (e0305cef) blurs only where |vx| + |vy| >= C#[0x2f2], and the patch makes
// that compare always false.
//
// Depth of field: its composite (111fce32) blends the blurred copy in by a
// clamped factor, and the patch makes that factor 0.
//
// Both follow their settings live: the programs are read again when the mask
// below changes (render.cpp's program_at - both are pixel shaders), and
// the patched and unpatched words are simply two programs, each with its own
// pipelines.

#include <cstdint>
#include <string>
#include <vector>

void shader_patch_apply(const std::string& name, std::vector<std::uint32_t>& words);

// Which patches the settings call for now, one bit each; kept by a settings
// listener (host_settings_listen), told each change, so it is one load a draw.
std::uint32_t shader_patch_mask();
// Whether a program with this name is one a patch applies to.
bool shader_patch_touches(const std::string& name);
