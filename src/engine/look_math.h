#pragma once

// The arithmetic of a plugin's look (plugin API 29's post_look) that
// engine/graphics_patch.cpp's apply_look writes into YEBIS's post-process
// block, apart from the block so that tests/look_math_test.cpp checks it
// without the game.

#include <cstdint>

namespace look_math {

// The motion blur's sample count a look writes (the block's +0x1ec): 2 to 8,
// 7 where the area interleaves its samples (+0x204). YEBIS's techniques are
// built for 1 to 8 samples and a larger count draws with shaders that are not
// the blur's; interleaving makes the count odd, so 8 would be 9; 1 and below
// are taken as YEBIS's own 4 or 5 (graphics_patch.cpp says where).
constexpr int blur_samples(int wanted, bool interleaved) {
    const int most = interleaved ? 7 : 8;
    return wanted < 2 ? 2 : wanted > most ? most : wanted;
}

// The depth of field by distance (BB_LOOK_DOF_RANGE): the game's own custom
// depth of field, as the areas' GPARAM and its debug menu set it (DofNear and
// DofFar DepthStart, DepthEnd, CocSize and CocScale), in the block's fields.
// Each view the apply fills the table YEBIS's circle of confusion is read
// through (the loop at 0x25d9760) with, at a view distance z,
//   z < near_end:  -(1 - t) near_size near_scale,  t = (z - near_start) / (near_end - near_start), in 0..1
//   z > far_start:       t  far_size  far_scale,   t = (z - far_start) / (far_end - far_start), in 0..1
// and 0 between them (with the near ramp on, the lens never reaches the
// table), and 0 past `cut`. The levels are YEBIS's blurred levels in front
// and behind (+0x1d0, +0x1d4): their count is the blur's strength.
struct DofRange {
    std::int32_t near_on = 1, far_on = 0;                                         // +0x2d4, +0x2d8
    float near_start = 0, near_end = 0, near_size = 0, near_scale = 1;           // +0x2dc, +0x2e0, +0x2e4, +0x2e8
    float far_start = 0, far_end = 0, far_size = 0, far_scale = 1;               // +0x2ec, +0x2f0, +0x2f4, +0x2f8
    float cut = 1.0e9f;                                                           // +0x2fc
    std::int32_t near_levels = 0, far_levels = 0;                                 // +0x1d0, +0x1d4
};

// The circle of confusion a strength's blur is drawn with at its fullest:
// twice as wide a level (`scale` 1 but for a test's BBHOST_DOF_COC_SCALE).
constexpr float dof_coc(int levels, float scale = 1.0f) {
    float c = 0.0f;
    if (levels > 0) {
        c = 0.0025f;
        for (int i = 1; i < levels && i < 7; ++i) c *= 2.0f;
    }
    return c * scale;
}

constexpr DofRange dof_range(float near_end, float far_start, float near_fade, float far_fade, int near_levels, int far_levels,
                             float coc_scale = 1.0f) {
    DofRange r;
    const auto pos = [](float v) { return v > 0.0f ? v : 0.0f; };
    near_levels = near_levels < 0 ? 0 : near_levels > 7 ? 7 : near_levels;
    far_levels = far_levels < 0 ? 0 : far_levels > 7 ? 7 : far_levels;
    r.near_end = pos(near_end);
    r.near_start = pos(r.near_end - pos(near_fade));
    r.near_size = dof_coc(near_levels, coc_scale);
    r.near_scale = near_levels > 0 ? 1.0f : 0.0f;
    r.far_on = far_levels > 0 ? 1 : 0;
    r.far_start = pos(far_start);
    r.far_end = r.far_start + pos(far_fade);
    r.far_size = dof_coc(far_levels, coc_scale);
    r.far_scale = far_levels > 0 ? 1.0f : 0.0f;
    r.near_levels = near_levels;
    r.far_levels = far_levels;
    return r;
}

// The table's entry at view distance z, as the apply's loop makes it from
// those fields (with the near ramp on): for the tests, and for the log.
constexpr float dof_entry(const DofRange& r, float z) {
    float v = 0.0f;
    if (r.near_on && z < r.near_end) {
        const float span = r.near_end - r.near_start > 1e-7f ? r.near_end - r.near_start : 1e-7f;
        float t = (z - r.near_start) / span;
        t = t < 0.0f ? 0.0f : t > 1.0f ? 1.0f : t;
        v = -(1.0f - t) * r.near_size;
    }
    if (v < 0.0f) v *= r.near_scale;
    if (r.far_on && z > r.far_start) {
        const float span = r.far_end - r.far_start > 1e-7f ? r.far_end - r.far_start : 1e-7f;
        float t = (z - r.far_start) / span;
        t = t < 0.0f ? 0.0f : t > 1.0f ? 1.0f : t;
        v = t * r.far_size;
    }
    if (v > 0.0f) v *= r.far_scale;
    if (r.cut < z) v = 0.0f;
    return v;
}

}  // namespace look_math
