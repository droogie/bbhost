// The engine's per-view block, as the G-buffer's vertex stage binds it: 864
// bytes, 216 floats (Numity1504's handoff, appendix A). The upscalers take
// their camera from it (host/upscale_pass.cpp: the motion, the characters'
// capture, the jitter and the history all wait on it), and so does the
// cutaway's depth decode. In floats:
//   36-51   the inverse projection, row by row: row 0's x the tangent of half
//           the horizontal field of view, row 1's y the vertical one's, row 2
//           (0, 0, 0, 1), row 3 the depth terms (x and y 0, z and w the
//           decode: view distance = 1 / (z raw + w));
//   52-67   the projection: its diagonal's first two the tangents' reciprocals;
//   180-195 the camera, view to world: each row's w the eye, row 3 (0, 0, 0, 1).
// Found by its shape among a camera draw's constant buffers (refusal below).
//
// The rows that are (0, 0, 0, 1) on paper the game computes in floats: they
// land exactly on 16:9 at its own field of view, and a rounding off at other
// shapes and fields of view - the inverse projection's w was 0.999999881 at
// 2560x1080, 0.99999994 at 1280x800 and 1.00000012 at 1920x1080 with the
// field of view 20% wider. Tested exactly, the block was refused there on
// every frame: no motion, no jitter and no history for the upscalers. So
// those rows are tested within a rounding, and the tangents are checked
// against the projection beside them, which no other buffer of the same shape
// matches.
#pragma once

#include <cmath>
#include <cstddef>

namespace gpu::view_block {

constexpr std::size_t kFloats = 216;  // 864 bytes
constexpr std::size_t kInvProj = 36;  // the inverse projection's 16 floats
constexpr std::size_t kProj = 52;     // the projection's
constexpr std::size_t kCamera = 180;  // the camera's 16 (view to world)

// How far a row that is (0, 0, 0, 1) on paper may be from it: a few ulps of
// 1.0 are 1e-7; the rest of the block's tests bound what else could pass.
constexpr float kRowTolerance = 1e-5f;
// How far a tangent times the projection's diagonal may be from 1.
constexpr float kProjectionTolerance = 1e-4f;

inline bool near_zero(float v) { return std::fabs(v) <= kRowTolerance; }
inline bool near_one(float v) { return std::fabs(v - 1.0f) <= kRowTolerance; }

// Cheaply first (most constant buffers are not one): the two rows' w.
inline bool may_be(const float* f) { return near_one(f[kInvProj + 11]) && near_one(f[kCamera + 15]); }

// Why `f` (kFloats floats) is not the view block, or null when it is.
inline const char* refusal(const float* f) {
    for (std::size_t i = 0; i < kFloats; ++i) {
        if (!std::isfinite(f[i])) return "a number that is not finite";
    }
    const float tan_x = f[kInvProj + 0], tan_y = f[kInvProj + 5];
    if (!(tan_x > 0.1f && tan_x < 10.0f && tan_y > 0.1f && tan_y < 10.0f)) return "tangents out of range";
    const float* r2 = f + kInvProj + 8;
    if (!near_zero(r2[0]) || !near_zero(r2[1]) || !near_zero(r2[2]) || !near_one(r2[3])) return "the inverse projection's third row is not (0, 0, 0, 1)";
    const float* c3 = f + kCamera + 12;
    if (!near_zero(c3[0]) || !near_zero(c3[1]) || !near_zero(c3[2]) || !near_one(c3[3])) return "the camera's last row is not (0, 0, 0, 1)";
    const float* r3 = f + kInvProj + 12;
    if (std::fabs(r3[0]) > 1e-4f || std::fabs(r3[1]) > 1e-4f) return "the depth terms' x or y is not 0";
    if (!(std::fabs(r3[2]) > 1e-6f)) return "no depth term";
    if (std::fabs(tan_x * f[kProj + 0] - 1.0f) > kProjectionTolerance || std::fabs(tan_y * f[kProj + 5] - 1.0f) > kProjectionTolerance)
        return "the tangents are not the projection's";
    return nullptr;
}

}  // namespace gpu::view_block
