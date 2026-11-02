// Whether a draw writes every pixel of one of its colour targets - the
// question a copy left unmade (host/copy_elide.h) needs answered before its
// source is drawn into: if the draw overwrites all of the source, what the
// source held before is dead, and the source and the copy's destination can
// exchange images instead of the copy being made. Pure: the caller fills an
// Input from the draw's state, its pixel shader and its vertices.
//
// Every pixel of the target is written when all of these hold:
//   - the attachment's write mask covers every channel of its format, with no
//     blending (what lands does not depend on what was there);
//   - nothing can reject a fragment: no depth test other than ALWAYS, no
//     stencil test, no depth bounds, no alpha to coverage, no rasterizer
//     discard, no face culling, solid fill, and a pixel shader that cannot
//     discard and writes its output on every path (straight-line code);
//   - one layer, and the viewport, the scissor and the render area (every
//     attachment of the draw at least the target's size) cover the target;
//   - the geometry covers the viewport's whole square: one or two triangles
//     (a rectangle list's one rectangle, a strip or list of two triangles that
//     share a diagonal, or one triangle), every vertex in front of the eye and
//     inside the depth range (or depth clamped), whose union - convex - holds
//     the four corners of [-1, 1]^2. A pixel's centre lies strictly inside
//     that square, so inside the union, and the rasterizer's fill rule gives
//     each such pixel to exactly one of the triangles;
//   - and for copy elision's decision (render.cpp copy_elide_decide_locked):
//     those positions are the bytes the GPU reads, the draw samples none of
//     its own targets, and the pipeline variant it may be bound with is known.
// tests/full_write_test.cpp checks each rule and the geometry's edge cases.
#pragma once

#include <cmath>
#include <cstdint>

namespace full_write {

enum Fail : int {
    kNone,
    kMask,          // the write mask leaves a channel out
    kBlend,         // blending on
    kDepth,         // a depth test that can fail
    kStencil,       // a stencil test
    kBounds,        // a depth bounds test
    kCoverage,      // alpha to coverage
    kDiscard,       // the pixel shader may discard
    kBranches,      // the pixel shader has control flow (its output might not be written on every path)
    kRasterKill,    // rasterizer discard
    kCull,          // face culling on
    kFill,          // not solid fill
    kLayers,        // a layered target
    kViewport,      // the viewport leaves part of the target out
    kScissor,       // the scissor leaves part of the target out
    kShader,        // a vertex shader whose position is not known (not the screen-space quad's)
    kPrimitive,     // not one or two triangles of the kinds above (or indexed, indirect, instanced)
    kGeometry,      // the triangles leave part of the square out
    kClip,          // a vertex behind the eye or outside the depth range
    kArea,          // an attachment smaller than the target: the render area leaves part of it out
    kBytes,         // the positions read are not known to be the bytes the GPU reads
    kFeedback,      // the draw samples its own target
    kVariant,       // a pipeline variant the draw may be bound with is not built yet (its pixel shader not known)
    kFails
};
inline const char* fail_name(int f) {
    static const char* const names[kFails] = {"proven", "write mask", "blending", "depth test", "stencil test", "depth bounds",
                                              "alpha to coverage", "may discard", "pixel shader branches", "rasterizer discard",
                                              "culling", "fill mode", "layers", "viewport", "scissor", "vertex shader",
                                              "primitive", "geometry", "clipped vertex", "render area", "vertex bytes",
                                              "samples its target", "pipeline variant"};
    return f >= 0 && f < kFails ? names[f] : "?";
}

enum Prim : int { kPrimOther, kPrimTriangleList, kPrimTriangleStrip, kPrimRectList };

struct Input {
    std::uint32_t width = 0, height = 0, layers = 1;  // the target
    std::uint32_t mask = 0, channels = 0xf;           // the write mask, the format's channels (bits)
    bool blend = false;
    bool depth_rejects = false, stencil = false, bounds = false, coverage = false;
    bool discard = false, branches = false, raster_kill = false, cull = false, solid = true, depth_clamp = false;
    float vp_x = 0, vp_y = 0, vp_w = 0, vp_h = 0;  // the viewport in pixels (vp_h may be negative)
    std::int32_t sc_x = 0, sc_y = 0;
    std::uint32_t sc_w = 0, sc_h = 0;
    std::uint32_t area_w = ~0u, area_h = ~0u;  // the render area: the smallest of the draw's attachments
    bool bytes_unknown = false;  // the positions are not known to be the bytes the GPU reads
    bool feedback = false;       // a texture of the draw is its target
    bool variant_unknown = false;  // a pipeline variant it may be bound with is not built
    bool known_vs = false;  // the positions below are what the vertex shader outputs
    int prim = kPrimOther;
    bool simple = true;     // not indexed, not indirect, one instance
    std::uint32_t vertices = 0;
    float pos[6][4] = {};   // clip-space positions (x, y, z, w) of vertices 0..5
};

namespace detail {
// Twice the signed area of (a, b, c).
inline double cross(const double* a, const double* b, const double* c) {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}
// Whether convex polygon `p` (n corners, either winding) holds point q (its edges included).
inline bool convex_holds(const double (*p)[2], int n, const double* q) {
    int sign = 0;
    for (int i = 0; i < n; ++i) {
        const double c = cross(p[i], p[(i + 1) % n], q);
        if (c == 0) continue;
        const int s = c > 0 ? 1 : -1;
        if (sign && s != sign) return false;
        sign = s;
    }
    return true;
}
inline bool holds_square(const double (*p)[2], int n) {
    static const double corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    for (const auto& c : corners) {
        if (!convex_holds(p, n, c)) return false;
    }
    return true;
}
// Whether p (n corners in order) is a convex polygon with no three corners on a line.
inline bool strictly_convex(const double (*p)[2], int n) {
    int sign = 0;
    for (int i = 0; i < n; ++i) {
        const double c = cross(p[i], p[(i + 1) % n], p[(i + 2) % n]);
        if (c == 0) return false;
        const int s = c > 0 ? 1 : -1;
        if (sign && s != sign) return false;
        sign = s;
    }
    return true;
}
}  // namespace detail

// The geometry's part: whether the triangles cover [-1, 1]^2 (kNone) or why not.
inline int geometry(const Input& in) {
    double v[6][2];
    if (!in.simple) return kPrimitive;
    const std::uint32_t n = in.vertices;
    if (n > 6) return kPrimitive;
    for (std::uint32_t i = 0; i < n; ++i) {
        const float w = in.pos[i][3];
        if (!(w > 0) || !std::isfinite(in.pos[i][0]) || !std::isfinite(in.pos[i][1]) || !std::isfinite(in.pos[i][2])) return kClip;
        // Vulkan's clip volume: 0 <= z <= w, unless the depth is clamped.
        if (!in.depth_clamp && (in.pos[i][2] < 0 || in.pos[i][2] > w)) return kClip;
        v[i][0] = static_cast<double>(in.pos[i][0]) / w;
        v[i][1] = static_cast<double>(in.pos[i][1]) / w;
    }
    // One triangle (a list of 3): it holds the square.
    if (in.prim == kPrimTriangleList && n == 3) {
        if (detail::cross(v[0], v[1], v[2]) == 0) return kGeometry;
        return detail::holds_square(v, 3) ? kNone : kGeometry;
    }
    // A rectangle list's one rectangle: corners v0, v1, v2 and v1 + v2 - v0.
    if (in.prim == kPrimRectList && n == 3) {
        const double q[4][2] = {{v[0][0], v[0][1]}, {v[1][0], v[1][1]}, {v[1][0] + v[2][0] - v[0][0], v[1][1] + v[2][1] - v[0][1]}, {v[2][0], v[2][1]}};
        if (!detail::strictly_convex(q, 4)) return kGeometry;
        return detail::holds_square(q, 4) ? kNone : kGeometry;
    }
    // Two triangles that share an edge which is a diagonal of the quadrilateral
    // their four corners make (a strip of 4: (0,1,2), (1,2,3); a list of 6).
    int tri[2][3];
    if (in.prim == kPrimTriangleStrip && n == 4) {
        tri[0][0] = 0, tri[0][1] = 1, tri[0][2] = 2;
        tri[1][0] = 1, tri[1][1] = 2, tri[1][2] = 3;
    } else if (in.prim == kPrimTriangleList && n == 6) {
        for (int k = 0; k < 6; ++k) tri[k / 3][k % 3] = k;
    } else {
        return kPrimitive;
    }
    // The shared corners (by position) and the two others.
    const auto same = [&](int a, int b) { return v[a][0] == v[b][0] && v[a][1] == v[b][1]; };
    int shared[2], ns = 0, other[2] = {-1, -1};
    for (int i = 0; i < 3; ++i) {
        bool found = false;
        for (int j = 0; j < 3 && !found; ++j) found = same(tri[0][i], tri[1][j]);
        if (found) {
            if (ns < 2) shared[ns] = tri[0][i];
            ++ns;
        } else {
            other[0] = tri[0][i];
        }
    }
    if (ns != 2 || other[0] < 0) return kGeometry;
    for (int j = 0; j < 3; ++j) {
        if (!same(tri[1][j], shared[0]) && !same(tri[1][j], shared[1])) other[1] = tri[1][j];
    }
    if (other[1] < 0) return kGeometry;
    // The quadrilateral other0, shared0, other1, shared1 - convex, so the shared edge is its diagonal and the
    // union of the two triangles is the quadrilateral.
    const double q[4][2] = {{v[other[0]][0], v[other[0]][1]}, {v[shared[0]][0], v[shared[0]][1]}, {v[other[1]][0], v[other[1]][1]},
                            {v[shared[1]][0], v[shared[1]][1]}};
    if (!detail::strictly_convex(q, 4)) return kGeometry;
    return detail::holds_square(q, 4) ? kNone : kGeometry;
}

// Copy elision's B1 (an exact point copy): whether texture coordinates uv[v] - vertex v's, interpolated across the
// draw - put every pixel centre of the target on its own texel's centre under nearest sampling: w = 1 at every vertex
// (linear interpolation) and each vertex's coordinates its window position (the viewport transform) over the target's
// size, so at the pixel centre (x + 0.5, y + 0.5) they are ((x + 0.5) / W, (y + 0.5) / H), a quarter texel and more from
// any edge however the interpolation rounds.
inline bool texel_exact(const Input& in, const float (*uv)[2]) {
    if (!in.vertices || !in.width || !in.height) return false;
    for (std::uint32_t v = 0; v < in.vertices; ++v) {
        if (in.pos[v][3] != 1.0f) return false;
        const double xw = in.vp_x + (static_cast<double>(in.pos[v][0]) + 1.0) * in.vp_w / 2.0;
        const double yw = in.vp_y + (static_cast<double>(in.pos[v][1]) + 1.0) * in.vp_h / 2.0;
        if (std::fabs(uv[v][0] - xw / in.width) > 1e-6 || std::fabs(uv[v][1] - yw / in.height) > 1e-6) return false;
    }
    return true;
}

inline int check(const Input& in) {
    if ((in.mask & in.channels) != in.channels) return kMask;
    if (in.blend) return kBlend;
    if (in.depth_rejects) return kDepth;
    if (in.stencil) return kStencil;
    if (in.bounds) return kBounds;
    if (in.coverage) return kCoverage;
    if (in.discard) return kDiscard;
    if (in.branches) return kBranches;
    if (in.raster_kill) return kRasterKill;
    if (in.cull) return kCull;
    if (!in.solid) return kFill;
    if (in.layers != 1) return kLayers;
    const float vy0 = in.vp_h < 0 ? in.vp_y + in.vp_h : in.vp_y, vh = std::fabs(in.vp_h);
    if (in.vp_x > 0 || vy0 > 0 || in.vp_x + in.vp_w < static_cast<float>(in.width) || vy0 + vh < static_cast<float>(in.height)) return kViewport;
    if (in.sc_x > 0 || in.sc_y > 0 || in.sc_x + static_cast<std::int64_t>(in.sc_w) < in.width || in.sc_y + static_cast<std::int64_t>(in.sc_h) < in.height) {
        return kScissor;
    }
    if (in.area_w < in.width || in.area_h < in.height) return kArea;
    if (in.feedback) return kFeedback;
    if (in.variant_unknown) return kVariant;
    if (!in.known_vs) return kShader;
    if (const int g = geometry(in); g != kNone) return g;
    return in.bytes_unknown ? kBytes : kNone;
}

}  // namespace full_write
