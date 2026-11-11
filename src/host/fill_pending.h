// A fill that met no image, and the texture made over its memory afterwards.
//
// GX clears a texture it has just made with a fill (GX's clear 0x2ab4ef0 for
// one without CMASK, a fill token here). When no image is at that memory yet
// - nothing has drawn into or sampled the texture - render.cpp
// (render_handle_fill_locked) keeps the fill pending for a render target made
// there later and runs nothing, so the memory keeps what it held before. A
// texture made there afterwards (textures.cpp) was uploaded from that memory.
// The water's interaction height maps are such textures: three R16G16 float
// maps for each water object near the player (0x12bb960), made 2x2 and
// cleared to 0, 0, 0, 1, then read by the water's domain shader for the
// height of its waves. Made over memory an area visited before had used,
// they came out as garbage and NaN, and every vertex of Moonside Lake's water
// was displaced off the screen for the whole visit (2026-11-10).
//
// The texture takes the fill's colour instead, as the console's fill left the
// memory, when the fill was made for the resource the texture is: the live GX
// texture whose memory begins at the fill's base, named by its generation (a
// resource made later over the same memory has another, and its own data),
// which GX has not written since (the registry's write sequence: a Map puts
// the CPU's data there after the fill), a texture at that base no longer than
// the fill, and a colour fill - an HTILE word's fill is not one (its first
// channel is not finite).
// BBHOST_FILL_AT_MAKE=0: such a texture is uploaded from the memory, as before.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gpu {

// A pending fill as the rule reads it.
struct PendingFillFor {
    std::uint64_t base = 0;  // the fill's first byte
    std::size_t bytes = 0;   // its length
    std::uint64_t gen = 0;   // the generation of the GX texture whose memory began at `base` when it came; 0 for none
    std::uint64_t write_seq = 0;  // that resource's write sequence then (GxResourceInfo::write_seq)
    float rgba[4] = {};      // its colour
};

// Whether a texture made at `base`, `bytes` long, of the live GX resource of
// generation `gen` and write sequence `write_seq` now, takes the pending fill `f`.
inline bool pending_fill_takes(const PendingFillFor& f, std::uint64_t base, std::size_t bytes, std::uint64_t gen, std::uint64_t write_seq) {
    if (f.gen == 0 || gen != f.gen || base != f.base || write_seq != f.write_seq) return false;
    if (bytes == 0 || bytes > f.bytes) return false;
    for (float c : f.rgba) {
        if (!std::isfinite(c)) return false;
    }
    return true;
}

}  // namespace gpu
