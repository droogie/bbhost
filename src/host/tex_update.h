// A YEBIS texture's UpdateSubresource, kept until the texture cache makes
// the texture.
//
// YEBIS writes the textures it makes with data - CTextureUtil's black and
// gray 1x1s and 4x4 dither, its white 1x1, ramp, falloff, curve and noise -
// through its UpdateSubresource (0x15d8d70, from IGPUSurface::SetImage2D):
// the texels go into a staging block and a compute copy (baad65ce) from it
// into the texture is recorded in YEBIS's command buffer. The game starts
// YEBIS again twice an area load (the scene context's update, sub_26c53e0 ->
// yebis_startup 0x25d2dd0 -> IPfxBaseContext::Initialize), and the copies
// recorded by some of those starts never ran: in about a third of the area
// loads checked (33 of 90) the set in use was made from memory nothing had
// written, and stayed so for the whole visit. That memory holds what was
// there before: YEBIS's private heap as yebis_startup fills it (0xff, white -
// no difference in the frames checked) or an earlier resource's bytes. The
// sets left so in play showed no difference either, but a gray made from
// other bytes can: one of (1, 0.5, 1, 1) tints YEBIS's final pass - a milky
// veil over the Forbidden Woods, Hunter's Nightmare's blood floor grey.
//
// So each update is kept with its texels, and the texture of that resource
// (its generation) made at that memory takes them as it is made, before
// anything reads it - what the copy writes. A copy that does run writes the
// same texels again.
// BBHOST_YEBIS_UPDATE_AT_MAKE=0: as before (the update tells the cache its
// memory changed, and the texture is made from the memory).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gpu {

struct TexUpdate {
    std::uint64_t memory = 0;  // the texture's memory (its first byte)
    std::uint64_t gen = 0;     // its resource's generation (GxResourceInfo::gen); 0 for none
    std::uint32_t mip = 0, slice = 0;
    std::uint32_t x = 0, y = 0, w = 0, h = 0;  // the box, in texels
    std::uint32_t pitch = 0;                   // bytes from one row of `rows` to the next
    // The submissions made when the game called the update (hle_gnm_submitted_total): its copy was recorded for the
    // next one, which the texels taken for it count as written by - not the one the command processor runs as it
    // makes the texture, which can come after the resource died and its memory went to the next one.
    std::uint64_t submitted = 0;
    std::vector<std::uint8_t> rows;            // h rows of `pitch` bytes
};

// Whether the texture of generation `gen` made at `base` - `levels` levels,
// `layers` layers, level `mip` mip_w x mip_h texels of `texel_bytes` each,
// block 1 - takes `u`.
inline bool tex_update_takes(const TexUpdate& u, std::uint64_t base, std::uint64_t gen, std::uint32_t levels, std::uint32_t layers,
                             std::uint32_t mip_w, std::uint32_t mip_h, std::uint32_t texel_bytes) {
    if (!u.gen || u.gen != gen || u.memory != base) return false;
    if (u.mip >= levels || u.slice >= layers || !u.w || !u.h || !texel_bytes) return false;
    if (u.x >= mip_w || u.y >= mip_h || u.w > mip_w - u.x || u.h > mip_h - u.y) return false;
    const std::size_t row = static_cast<std::size_t>(u.w) * texel_bytes;
    if (u.pitch < row) return false;
    return u.rows.size() >= static_cast<std::size_t>(u.pitch) * (u.h - 1) + row;
}

// The update's texels as the image takes them: h rows of w * texel_bytes.
inline std::vector<std::uint8_t> tex_update_packed(const TexUpdate& u, std::uint32_t texel_bytes) {
    const std::size_t row = static_cast<std::size_t>(u.w) * texel_bytes;
    std::vector<std::uint8_t> out(row * u.h);
    for (std::uint32_t r = 0; r < u.h; ++r) std::memcpy(out.data() + row * r, u.rows.data() + static_cast<std::size_t>(u.pitch) * r, row);
    return out;
}

}  // namespace gpu
