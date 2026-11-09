#pragma once

// Plugin API 27's rules as plain functions (include/bbhost_plugin.h frame_info,
// on_render, aa_claim): no Vulkan and no state of their own. A frame's cameras
// from the view block's terms, the frame's reset and its cut test, and the
// generations of the images the render hooks are handed.
// host/render_hooks.cpp uses them; tests/render_api_test.cpp checks them.

#include "bbhost_plugin.h"

#include <cmath>
#include <cstdint>

namespace render_api {

// The view block's terms as host/upscale_pass.cpp's camera_of keeps them: the
// tangents of half the field of view across and up, the depth's decode (raw =
// a + r2w / z, z the view distance) and the view-to-world rows (rotation |
// eye, three rows of four).
struct ViewTerms {
    float tan_x = 0.0f, tan_y = 0.0f, a = 0.0f, r2w = 0.0f;
    float rows[12] = {};
};

// out = a * b, column-major.
inline void mul(const double (&a)[16], const double (&b)[16], double (&out)[16]) {
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) s += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = s;
        }
    }
}

// The inverse of m by its cofactors; false (out the identity) when m has none.
inline bool invert(const double (&m)[16], double (&out)[16]) {
    double inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] -
             m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] +
             m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];
    const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (!(std::fabs(det) > 1e-30) || !std::isfinite(det)) {
        for (int k = 0; k < 16; ++k) out[k] = k % 5 == 0 ? 1.0 : 0.0;
        return false;
    }
    for (int k = 0; k < 16; ++k) out[k] = inv[k] / det;
    return true;
}

// The view distances the depth's 0 and 1 stand for, as the upscalers are
// told them (upscale_pass.cpp camera_terms): raw r is 1 / (z r + w) away, z =
// 1 / r2w and w = -a / r2w - nearer than 1 mm read 1 mm, farther than 1e6 m
// (an infinite far plane) 1e6 m. False, and the stand-ins 0.1 m and 10 km,
// for a decode that gives no distance.
inline bool depth_range(float a, float r2w, float* near_z, float* far_z, bool* reversed) {
    *near_z = 0.1f;
    *far_z = 10000.0f;
    *reversed = false;
    if (r2w == 0.0f) return false;
    const double z = 1.0 / r2w, w = -static_cast<double>(a) * z;
    const auto dist = [&](double raw) {
        const double q = z * raw + w;
        return q > 1e-7 ? 1.0 / q : 1e6;
    };
    const double d0 = dist(0.0), d1 = dist(1.0);
    if (!std::isfinite(d0) || !std::isfinite(d1) || d0 <= 0.0 || d1 <= 0.0) return false;
    *reversed = d1 < d0;
    *near_z = static_cast<float>(std::fmax(std::fmin(d0, d1), 0.001));
    *far_z = static_cast<float>(std::fmin(std::fmax(d0, d1), 1e6));
    return true;
}

// A frame's camera from its view block's terms (bbhost_plugin.h BbFrameCamera):
// its basis and eye (the view-to-world matrix's columns), the field of view
// and shape from the tangents, the clip planes from the depth's decode, and
// the four matrices - the view the inverse of view to world, the projection
// Vulkan's over the images (y down, z the depth's raw value: a + r2w / z, as
// motion_at.glsl decodes it), worked out in double. `reversed`: the depth
// holds 1 at the near plane.
inline void frame_camera(const ViewTerms& t, BbFrameCamera* out, bool* reversed) {
    const float* r = t.rows;
    BbCamera& c = out->camera;
    for (int k = 0; k < 3; ++k) {
        c.right[k] = r[4 * k + 0];
        c.up[k] = r[4 * k + 1];
        c.forward[k] = r[4 * k + 2];
        c.position[k] = r[4 * k + 3];
    }
    c.right[3] = c.up[3] = c.forward[3] = 0.0f;
    c.position[3] = 1.0f;
    c.fov_y = t.tan_y > 0.0f ? 2.0f * std::atan(t.tan_y) : 0.0f;
    c.aspect = t.tan_y > 0.0f ? t.tan_x / t.tan_y : 0.0f;
    depth_range(t.a, t.r2w, &c.near_z, &c.far_z, reversed);
    double to_world[16];
    for (int k = 0; k < 4; ++k) {
        to_world[0 + k] = c.right[k];
        to_world[4 + k] = c.up[k];
        to_world[8 + k] = c.forward[k];
        to_world[12 + k] = c.position[k];
    }
    double view[16], proj[16] = {}, vp[16], inv_vp[16];
    invert(to_world, view);
    proj[0] = t.tan_x != 0.0f ? 1.0 / t.tan_x : 0.0;
    proj[5] = t.tan_y != 0.0f ? -1.0 / t.tan_y : 0.0;
    proj[10] = t.a;
    proj[11] = 1.0;
    proj[14] = t.r2w;
    mul(proj, view, vp);
    invert(vp, inv_vp);
    for (int k = 0; k < 16; ++k) {
        out->view[k] = static_cast<float>(view[k]);
        out->proj[k] = static_cast<float>(proj[k]);
        out->view_proj[k] = static_cast<float>(vp[k]);
        out->inv_view_proj[k] = static_cast<float>(inv_vp[k]);
    }
}

// A cut between two frames' cameras (BB_FRAME_CUT): the eye moved more than
// 4 m, the view turned more than 75 degrees, or the vertical field of view
// changed by more than 10 degrees.
inline bool frame_cut(const BbCamera& now, const BbCamera& prev) {
    const double dx = static_cast<double>(now.position[0]) - prev.position[0], dy = static_cast<double>(now.position[1]) - prev.position[1],
                 dz = static_cast<double>(now.position[2]) - prev.position[2];
    if (dx * dx + dy * dy + dz * dz > 4.0 * 4.0) return true;
    const double ln = std::sqrt(static_cast<double>(now.forward[0]) * now.forward[0] + static_cast<double>(now.forward[1]) * now.forward[1] +
                                static_cast<double>(now.forward[2]) * now.forward[2]),
                 lp = std::sqrt(static_cast<double>(prev.forward[0]) * prev.forward[0] + static_cast<double>(prev.forward[1]) * prev.forward[1] +
                                static_cast<double>(prev.forward[2]) * prev.forward[2]);
    if (ln > 0.0 && lp > 0.0) {
        const double cosine = (static_cast<double>(now.forward[0]) * prev.forward[0] + static_cast<double>(now.forward[1]) * prev.forward[1] +
                               static_cast<double>(now.forward[2]) * prev.forward[2]) / (ln * lp);
        if (cosine < std::cos(75.0 * 3.14159265358979323846 / 180.0)) return true;
    }
    return std::fabs(static_cast<double>(now.fov_y) - prev.fov_y) > 10.0 * 3.14159265358979323846 / 180.0;
}

// What decides whether a frame continues the last one (BbFrameInfo.reset):
// its sizes and its method.
struct FrameShape {
    std::uint32_t render_w = 0, render_h = 0, output_w = 0, output_h = 0, method = 0;
    bool operator==(const FrameShape& o) const {
        return render_w == o.render_w && render_h == o.render_h && output_w == o.output_w && output_h == o.output_h && method == o.method;
    }
};

// BbFrameInfo.reset: nothing to blend with - the first frame, another size or
// method than the last frame's, no camera this frame or the last, a cut, or
// more than a quarter of a second since the last frame.
inline bool frame_reset(bool first, const FrameShape& last, const FrameShape& now, bool camera, bool prev, bool cut, double seconds) {
    return first || !(last == now) || !camera || !prev || cut || seconds > 0.25;
}

// One of the images the render hooks are handed (BbRenderImage), as it was
// handed last: its handles and shape, its generation, and whether the host
// let go of the image or its view since (render_hooks_gone).
struct ImageSlot {
    std::uint64_t image = 0, view = 0, generation = 0;
    std::uint32_t format = 0, width = 0, height = 0;
    bool gone = false;
};

// The generation of the image handed in `s` now: a new one from `serial` (one
// counter for every slot, so none repeats) when anything of it differs from
// the last time or the host let go of what it had - a driver can make a new
// image under a handle it has freed - else the same. 0 for no image.
inline std::uint64_t image_generation(ImageSlot& s, std::uint64_t image, std::uint64_t view, std::uint32_t format, std::uint32_t width,
                                      std::uint32_t height, std::uint64_t& serial) {
    if (!image) return 0;
    if (s.gone || image != s.image || view != s.view || format != s.format || width != s.width || height != s.height || !s.generation) {
        s.image = image;
        s.view = view;
        s.format = format;
        s.width = width;
        s.height = height;
        s.generation = ++serial;
        s.gone = false;
    }
    return s.generation;
}

// The host lets go of an image or a view (its deferred destruction): a slot
// that handed it out gives the next one a new generation.
inline void image_gone(ImageSlot& s, std::uint64_t handle) {
    if (handle && (handle == s.image || handle == s.view)) s.gone = true;
}

}  // namespace render_api
