// NS_SPRJ::ChrExFollowCam::Update (0x183ac60) as our source (docs/decomp.md):
// the player-follow camera's step, once a frame on the main thread, from the
// camera manager's blend - (the camera, the followed character, the
// collision world, dt). It runs in 23 phases, each a function below, called
// in order by follow_cam_update at the end: the LockCamParam row, the moving
// floor, the stick, the character's frame and the points that follow it,
// the angles (lock-on, the auto turn, the stick), where the camera wants to
// be, the walls, the chase and the basis, the collisions, the keep-out
// spheres.
//
// The game's function is 20,077 bytes of hand-scheduled SIMD: rotations,
// lengths from a reciprocal square root refined by two Newton steps, sine
// and cosine series summed with horizontal adds, angle wraps through integer
// conversions. Only the same operations in the same order give the same
// bits, so ours keeps them: each of the game's idioms is a helper below that
// performs its operations in its order (how sums and products associate is
// what decides the bits; which operand comes first is not), and the rest is
// plain C++. Built with -ffp-contract=off (no multiply fused into an add) and
// -mavx (the game's own instruction set).
//
// What it calls, it calls as the game does: the sine, arctangent and tangent
// through the game's import stubs (bound to bbhost's libm), the sphere cast,
// the quaternion and matrix helpers and the debug draw in the eboot. Three
// places in the game's code are rewritten at load, and ours reads them where
// the game would: the fov-uncap patch (0x183af4e) and the 60 fps jump
// (0x183dc97) and immediate (0x183dcfe). The two 60 fps constants are read
// from memory as the game reads them; the game's other constants are
// literals here, and the body check hashes them with the code.
//
// At the frame rates bbhost converts (engine/frame_rate.h) the weights the
// camera eases by each frame are converted to the frame's length, so it turns,
// chases and settles as far in a second at any rate: the ones the Camera
// tuning option sets (the chase rates +0x198, +0x1a4, +0x1ac/+0x1b8 and their
// lock-on pair, +0x1c4, the direct pitch rate +0x23c, the lock chase +0x240,
// the pitch return's kept share +0x1a8) from a frame of the rate their values
// are for - 60 with FPS++, whose values the 60 FPS++ sites store, 30 with
// Original, the game's own - and the LockCamParam blend +0x330, which neither
// changes, from a 30 fps one (frame_rate_step.h; Kyo's 1.3 lists and
// yogesh1239's AI/Camera fix do the same in caves). The distance and
// chase-rate eases are constants the frame-rate code writes every frame. At
// the game's own 30 and in a compare run nothing is converted: the step is the
// game's to the bit - and with Original at a frame of 1/30 s, every weight is
// the game's own too.
//
// Two things the PC port adds, neither in the game's function: the mouse
// turns the free camera after the stick, where Dark Souls III turns it
// (engine/mouse_camera.h), and while the keyboard and mouse were used last the
// camera's own turns as the character moves are held - the four stores the
// community patch "Disable Camera Auto Rotation via Movement" removes, which
// the body check accepts as nops too (engine/mouse_camera_step.h). Without
// mouse movement or the hold, the step is the game's to the bit.
//
// tests/decomp/follow_camera_test.cpp runs the game's version and ours in the
// eboot kit over generated cameras, frame after frame - with the community
// patch's nops in the game's code against our hold, and with the mouse
// camera's hook in the game's code against our mouse step.
#include "decomp/decomp.h"
#include "decomp/guest.h"
#include "engine/camera_target.h"
#include "engine/frame_rate_step.h"
#include "engine/mouse_camera.h"
#include "engine/mouse_camera_step.h"

#include "log.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <immintrin.h>

using namespace decomp;

namespace {

constexpr u64 kUpdate = 0x183ac60, kUpdateEnd = 0x183facd;

inline float ldf(u64 bn) { return load<float>(game_address(bn)); }

std::atomic<u64> g_frames{0};

// ---- The camera -------------------------------------------------------------

// ChrExFollowCam, 0x350 bytes. Names from what the step does with each field;
// the phase that uses one most is noted. Offsets checked below.
struct FollowCam {
    u8 head[0x10];
    __m128 basis_x, basis_y, basis_z;      // +0x10: the view basis (P18)
    __m128 position;                       // +0x40: where the camera is
    float fov_y;                           // +0x50: vertical field of view, radians (P2)
    float aspect;                          // +0x54
    u8 pad_58[8];
    __m128 chr_neg_x, chr_y, chr_neg_z;    // +0x60: the followed character's frame, -X, Y, -Z (P8)
    __m128 chr_position;                   // +0x90: its position, chased (P8)
    __m128 origin_a, origin_b;             // +0xa0: the look-at offset from the frame, from the chase (P9)
    __m128 orbit_reference;                // +0xc0: what the camera keeps its distance from (P10)
    __m128 focus;                          // +0xd0: what it looks at (P10)
    __m128 wanted;                         // +0xe0: where it wants to be (P15)
    __m128 wanted_tested;                  // +0xf0: the same, tested against walls (P16, P17)
    __m128 point;                          // +0x100: the point the chase moves (P18)
    __m128 floor_velocity;                 // +0x110: a moving floor's velocity (P5)
    u8 pad_120[4];
    float floor_yaw_rate;                  // +0x124: and its turn
    u8 pad_128[8];
    float return_weight;                   // +0x130: the pitch return's weight (P12, P18)
    float return_wait;                     // +0x134: its wait
    float fast_turn;                       // +0x138: the fast turn's ramp, -1 to 1 (P13)
    u8 reset_request;                      // +0x13c: one frame's reset (P11)
    u8 reset_to_target;                    // +0x13d: to the target angles, once
    u8 pad_13e[2];
    float pitch, yaw;                      // +0x140: the camera's angles
    float target_pitch, target_yaw;        // +0x148: the angles a reset takes
    float pitch_reference;                 // +0x150: the pitch the return eases back to (P14)
    float follow_weight;                   // +0x154: how fully the wanted point keeps up (P7, P10)
    float follow_end_wait;                 // +0x158
    u8 input_was_active;                   // +0x15c: the stick turned the camera last frame
    u8 pad_15d[3];
    __m128 look_at_offset;                 // +0x160: in the character's frame; lane 1 from the param (P9)
    float shrink_divisor, shrink_min, shrink_max;  // +0x170: the wanted distance's sideways shrink (P15)
    float distance_factor;                 // +0x17c: the wanted distance's share of the camera's (P6)
    float wanted_distance;                 // +0x180: how far from the orbit reference it wants to be
    float distance;                        // +0x184: the wall-tested point's (P3, P16)
    float eased_distance;                  // +0x188: toward the param's distance (P2)
    float sphere_radius;                   // +0x18c: the casts' sphere (P16, P19)
    float near_rate, near_max;             // +0x190: the near clamp (P23)
    float chr_chase_rate, chr_chase_rate_normal, chr_chase_rate_escape;  // +0x198: the character's chase (P8, P17)
    float target_chase_rate;               // +0x1a4: the wanted point's chase (P10)
    float pitch_chase_rate;                // +0x1a8: the share of its offset the pitch keeps (P14)
    float chase_rate_xz, chase_rate_xz_normal, chase_rate_xz_escape;  // +0x1ac: the point's chase (P17, P18)
    float chase_rate_y;                    // +0x1b8
    float chase_min_angle, chase_max_angle;            // +0x1bc
    float chase_distance_rate;                         // +0x1c4
    float lock_chase_rate_xz, lock_chase_rate_y;       // +0x1c8: the same under lock-on
    float lock_chase_min_angle, lock_chase_max_angle;  // +0x1d0
    float follow_end_wait_time, follow_release_time;   // +0x1d8 (P7)
    float return_wait_time, return_release_time;       // +0x1e0 (P12, P13)
    float fast_turn_ramp_time;                         // +0x1e8 (P13)
    float pitch_max, pitch_min;                        // +0x1ec: the pitch limits
    float lock_pitch_max, lock_pitch_min;              // +0x1f4: under lock-on, at no height difference (P11)
    float lock_height_low, lock_height_high;           // +0x1fc: the height difference that gives them up
    float pitch_speed, yaw_speed;                      // +0x204: the stick's speeds, from the option (P3, P13)
    float pitch_speed_min, yaw_speed_min, pitch_speed_max, yaw_speed_max;  // +0x20c
    float pitch_high_speed, yaw_high_speed;                                // +0x21c: the fast turn's
    float pitch_high_speed_min, yaw_high_speed_min, pitch_high_speed_max, yaw_high_speed_max;  // +0x224
    float option_speed_rate;               // +0x234: the player's camera speed, written by the blend
    float direct_pitch_angle;              // +0x238: how near sideways a full stick turns fast (P13)
    float direct_pitch_rate;               // +0x23c: the fast turn's pitch ease
    float lock_chase_rate;                 // +0x240: lock-on's and the auto turn's angle chase (P11)
    float play_angle;                      // +0x244: lock-on's dead zone
    float lock_shift_ratio;                // +0x248: where in the view lock-on puts the target
    u8 pad_24c[4];
    __m128 lock_target;                    // +0x250: the locked-on target
    u8 lock_on;                            // +0x260: one-frame requests (P20 clears them)
    u8 alternate_param;                    // +0x261
    u8 turn_behind;                        // +0x262
    u8 auto_turn;                          // +0x263: an auto turn runs (P11)
    float auto_turn_timer, auto_turn_duration;  // +0x264
    float auto_pitch, auto_yaw;            // +0x26c: its angles
    u8 lock_sets_reference;                // +0x274
    u8 turn_behind_sets_reference;         // +0x275
    u8 lock_stick_offsets;                 // +0x276: under lock-on the stick offsets the view (P13)
    u8 pad_277;
    float lock_pitch_offset, lock_yaw_offset;  // +0x278
    float lock_pitch_gain, lock_yaw_gain;      // +0x280
    float lock_offset_ease;                    // +0x288
    u8 wall_escape;                        // +0x28c: the yaw correction holds (P17)
    u8 pad_28d[3];
    float escape_turn;                     // +0x290: cleared with the correction
    float yaw_correction;                  // +0x294: the turn away from a wall
    u8 pad_298[4];
    float escape_rate_near;                // +0x29c: its rate, a wall at a probe's start
    u8 pad_2a0[4];
    float escape_rate_far, escape_near_fraction;  // +0x2a4: at its end; how far "near" reaches
    u8 pad_2ac[4];
    float escape_max_rate, escape_decay;                         // +0x2b0
    float probe_angle, probe_reach, probe_radius, probe_offset;  // +0x2b8: the side probes
    float probe_a, probe_b;                // +0x2c8: their hit fractions
    u8 keep_out;                           // +0x2d0: stay out of the keep-out spheres (P22)
    u8 keep_out_debug_draw;                // +0x2d1: and draw them
    u8 pad_2d2[0xe];
    __m128 work_basis_x, work_basis_y, work_basis_z, work_position;  // +0x2e0: the keep-out's working copy
    s32 param_override, param_id, alternate_param_id, default_param_id;  // +0x320: LockCamParam ids (P1)
    float param_ease;                      // +0x330: the ease toward its row (P2)
    u8 input_enabled;                      // +0x334: the camera takes the stick (P7)
    u8 pad_335[0xf];
    u8 framing;                            // +0x344: origin b's height follows the view (P9)
    u8 pad_345[0xb];
};
static_assert(sizeof(FollowCam) == 0x350);
#define AT(field, offset) static_assert(offsetof(FollowCam, field) == (offset), #field)
AT(basis_x, 0x10); AT(position, 0x40); AT(fov_y, 0x50); AT(aspect, 0x54); AT(chr_neg_x, 0x60); AT(chr_position, 0x90);
AT(origin_a, 0xa0); AT(orbit_reference, 0xc0); AT(focus, 0xd0); AT(wanted, 0xe0); AT(wanted_tested, 0xf0);
AT(point, 0x100); AT(floor_velocity, 0x110); AT(floor_yaw_rate, 0x124); AT(return_weight, 0x130);
AT(return_wait, 0x134); AT(fast_turn, 0x138); AT(reset_request, 0x13c); AT(reset_to_target, 0x13d); AT(pitch, 0x140);
AT(target_pitch, 0x148); AT(pitch_reference, 0x150); AT(follow_weight, 0x154); AT(follow_end_wait, 0x158);
AT(input_was_active, 0x15c); AT(look_at_offset, 0x160); AT(shrink_divisor, 0x170); AT(distance_factor, 0x17c);
AT(wanted_distance, 0x180); AT(distance, 0x184); AT(eased_distance, 0x188); AT(sphere_radius, 0x18c);
AT(near_rate, 0x190); AT(chr_chase_rate, 0x198); AT(target_chase_rate, 0x1a4); AT(pitch_chase_rate, 0x1a8);
AT(chase_rate_xz, 0x1ac); AT(chase_rate_y, 0x1b8); AT(chase_min_angle, 0x1bc); AT(chase_distance_rate, 0x1c4);
AT(lock_chase_rate_xz, 0x1c8); AT(lock_chase_min_angle, 0x1d0); AT(follow_end_wait_time, 0x1d8);
AT(return_wait_time, 0x1e0); AT(fast_turn_ramp_time, 0x1e8); AT(pitch_max, 0x1ec); AT(lock_pitch_max, 0x1f4);
AT(lock_height_low, 0x1fc); AT(pitch_speed, 0x204); AT(pitch_speed_min, 0x20c); AT(pitch_high_speed, 0x21c);
AT(pitch_high_speed_min, 0x224); AT(option_speed_rate, 0x234); AT(direct_pitch_angle, 0x238);
AT(direct_pitch_rate, 0x23c); AT(lock_chase_rate, 0x240); AT(play_angle, 0x244); AT(lock_shift_ratio, 0x248);
AT(lock_target, 0x250); AT(lock_on, 0x260); AT(alternate_param, 0x261); AT(turn_behind, 0x262); AT(auto_turn, 0x263);
AT(auto_turn_timer, 0x264); AT(auto_pitch, 0x26c); AT(lock_sets_reference, 0x274);
AT(turn_behind_sets_reference, 0x275); AT(lock_stick_offsets, 0x276); AT(lock_pitch_offset, 0x278);
AT(lock_pitch_gain, 0x280); AT(lock_offset_ease, 0x288); AT(wall_escape, 0x28c); AT(escape_turn, 0x290);
AT(yaw_correction, 0x294); AT(escape_rate_near, 0x29c); AT(escape_rate_far, 0x2a4); AT(escape_max_rate, 0x2b0);
AT(probe_angle, 0x2b8); AT(probe_a, 0x2c8); AT(keep_out, 0x2d0); AT(keep_out_debug_draw, 0x2d1);
AT(work_basis_x, 0x2e0); AT(param_override, 0x320); AT(param_ease, 0x330); AT(input_enabled, 0x334); AT(framing, 0x344);
#undef AT

// ---- Vectors, the game's way ------------------------------------------------

// Lane-wise arithmetic on __m128 is the compilers' vector extension (GCC and
// Clang): a + b is one addps, and with -ffp-contract=off a * b + c stays a
// multiply and an add.
using Vec = __m128;

inline Vec splat_x(Vec v) { return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), 0x00)); }
inline Vec splat_y(Vec v) { return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), 0x55)); }
inline Vec splat_z(Vec v) { return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), 0xaa)); }
inline Vec splat_w(Vec v) { return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), 0xff)); }

const Vec kZero = {0.0f, 0.0f, 0.0f, 0.0f};
const Vec kE0 = {1.0f, 0.0f, 0.0f, 0.0f}, kE1 = {0.0f, 1.0f, 0.0f, 0.0f};
const Vec kE2 = {0.0f, 0.0f, 1.0f, 0.0f}, kE3 = {0.0f, 0.0f, 0.0f, 1.0f};

// v as a direction (w 0) or a point (w 1).
inline Vec xyz(Vec v) { return _mm_blend_ps(v, kZero, 0x8); }
inline Vec point(Vec v) { return _mm_blend_ps(v, kE3, 0x8); }

// Four lanes summed as two horizontal adds sum them: (a + b) + (c + d).
inline float hadd_sum(Vec v) {
    const Vec pairs = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}

// Matrices are rows, and a point is a row times one.
struct Matrix {
    Vec r0, r1, r2, r3;
};
const Matrix kIdentity = {kE0, kE1, kE2, kE3};

// A row times a matrix, summed as the game's matrix product sums it:
// (x r0 + z r2) + (y r1 + w r3).
inline Vec row_times(Vec v, const Matrix& m) {
    return (splat_x(v) * m.r0 + splat_z(v) * m.r2) + (splat_y(v) * m.r1 + splat_w(v) * m.r3);
}
inline Matrix product(const Matrix& a, const Matrix& b) {
    return {row_times(a.r0, b), row_times(a.r1, b), row_times(a.r2, b), row_times(a.r3, b)};
}
// A point through a matrix, summed as the game's transform sums it:
// (x r0 + y r1) + (z r2 + w r3).
inline Vec transform(Vec p, const Matrix& m) {
    return (splat_x(p) * m.r0 + splat_y(p) * m.r1) + (splat_z(p) * m.r2 + splat_w(p) * m.r3);
}
// A matrix times a column: each row's dot product with it, summed by
// horizontal adds.
inline Vec rows_dot(const Matrix& m, Vec v) {
    return _mm_setr_ps(hadd_sum(v * m.r0), hadd_sum(v * m.r1), hadd_sum(v * m.r2), hadd_sum(v * m.r3));
}
inline Matrix translation(Vec offset) { return {kE0, kE1, kE2, point(offset)}; }
// Rotations about X and Y as the game builds them from a sine and a cosine.
inline Matrix rotation_x(float s, float c) { return {kE0, _mm_setr_ps(0.0f, c, s, 0.0f), _mm_setr_ps(0.0f, -s, c, 0.0f), kE3}; }
inline Matrix rotation_y(float s, float c) { return {_mm_setr_ps(c, 0.0f, -s, 0.0f), kE1, _mm_setr_ps(s, 0.0f, c, 0.0f), kE3}; }

// The game's square root, in every lane: a reciprocal square root estimate
// refined by two Newton steps, times the value - or the value itself where it
// or the estimate is 0 (so 0 stays 0 and infinity infinity).
inline Vec game_sqrt(Vec s) {
    const Vec half = _mm_set1_ps(0.5f), three = _mm_set1_ps(3.0f);
    const Vec r0 = _mm_rsqrt_ps(s);
    const Vec keep = _mm_or_ps(_mm_cmpeq_ps(s, kZero), _mm_cmpeq_ps(r0, kZero));
    const Vec r1 = (r0 * half) * (three - r0 * (r0 * s));
    const Vec r2 = (r1 * half) * (three - r1 * (s * r1));
    return _mm_blendv_ps(s * r2, s, keep);
}
// x, y and z squared and summed, z + (x + y), in every lane; and its root.
inline Vec length_sq3(Vec v) {
    const Vec sq = v * v;
    return splat_z(sq) + (splat_x(sq) + splat_y(sq));
}
inline Vec length3(Vec v) { return game_sqrt(length_sq3(v)); }
// All four squared and summed (x + z) + (y + w), in every lane.
inline Vec length_sq4(Vec v) {
    const Vec sq = v * v;
    return _mm_set1_ps((sq[0] + sq[2]) + (sq[1] + sq[3]));
}
// The game's normalize: v over its length (of all four lanes); a zero vector
// stays zero, and one whose squares sum to infinity becomes NaN.
inline Vec normalize4(Vec v) {
    const Vec sum = length_sq4(v);
    const Vec length = game_sqrt(sum);
    const Vec nonzero = _mm_cmpneq_ps(kZero, length);
    const Vec finite = _mm_cmpneq_ps(sum, _mm_set1_ps(__builtin_inff()));
    const Vec unit = _mm_and_ps(_mm_and_ps(v / length, nonzero), finite);
    return _mm_or_ps(_mm_andnot_ps(finite, _mm_castsi128_ps(_mm_set1_epi32(0x7fc00000))), unit);
}
// a x b, lane by lane as the game's cross product computes it, w 0.
inline Vec cross3(Vec a, Vec b) {
    return _mm_setr_ps(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0.0f);
}
// Whether any lane is a NaN.
inline bool any_nan(Vec v) { return _mm_movemask_ps(_mm_cmpunord_ps(v, v)) != 0; }

// ---- Scalars and angles, the game's way -------------------------------------

// sqrtss, minss and maxss. minss and maxss give the first operand when it
// compares less (greater), otherwise the second - so a NaN in either gives
// the second - and under the game's MXCSR (denormals are zero) a denormal
// comes out of them as 0.
inline float sqrt_ss(float x) { return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))); }
inline float min_ss(float a, float b) { return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(a), _mm_set_ss(b))); }
inline float max_ss(float a, float b) { return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(a), _mm_set_ss(b))); }

// Where the game chooses between two values with a branch or a cmov instead,
// the chosen one keeps its bits, a denormal too. The empty asm hides the
// values from the compiler, which would otherwise make a choice shaped like
// a min or a max into minss or maxss.
inline float choose(bool first, float a, float b) {
    __asm__("" : "+x"(a), "+x"(b));
    return first ? a : b;
}

constexpr float kPi = 3.14159274f, k2Pi = 6.28318548f, kHalfPi = 1.57079637f;
constexpr float kOneDegree = 0.0174532924f;  // pi / 180, which the game also converts degrees by

// An angle wrapped into [-pi, pi] as the game wraps one: less the multiple of
// 2 pi that (x + pi with x's sign) / 2 pi truncates to.
inline float wrap_angle(float x) {
    const s32 turns = _mm_cvttss_si32(_mm_set_ss((__builtin_copysignf(kPi, x) + x) / k2Pi));
    return x - static_cast<float>(turns) * k2Pi;
}
// The same where the game wraps two angles at once in a vector register:
// there the truncation is to 64 bits, of which the low 32 count.
inline float wrap_angle_wide(float x) {
    const s32 turns = static_cast<s32>(_mm_cvttss_si64(_mm_set_ss((x + __builtin_copysignf(kPi, x)) / k2Pi)));
    return x - static_cast<float>(turns) * k2Pi;
}
// Angles wrapped four at a time, the vector way: less the nearest multiple
// of 2 pi (cvtps2dq rounds) - except where x / 2 pi is 2^23 or more (or NaN),
// which is whole already and is taken as it is. The game makes that test an
// integer compare of -|x / 2 pi| with -2^23, kept here.
inline Vec wrap_angles(Vec v) {
    const Vec turns = v * _mm_set1_ps(0.15915494f);
    const Vec abs_turns = _mm_and_ps(turns, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)));
    const __m128i minus_abs = _mm_castps_si128(_mm_xor_ps(abs_turns, _mm_set1_ps(-0.0f)));
    const Vec small = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_castps_si128(_mm_set1_ps(-8388608.0f)), minus_abs));
    const Vec rounded = _mm_cvtepi32_ps(_mm_cvtps_epi32(turns));
    const Vec whole = _mm_or_ps(_mm_and_ps(small, rounded), _mm_andnot_ps(small, turns));
    return v - whole * _mm_set1_ps(k2Pi);
}

// The game's sine of an angle in [-pi, pi]: the Taylor series to w^23. The
// odd powers come four at a time - (1, w, w^2, w^3) squared and times w, then
// twice times w^8 - and each four, times their coefficients, are summed by
// horizontal adds, the three sums added in order.
inline float series_sin(float w) {
    const Vec k0 = {1.0f, -0.16666667f, 0.008333334f, -0.0001984127f};
    const Vec k1 = {2.7557319e-06f, -2.5052108e-08f, 1.6059044e-10f, -7.6471636e-13f};
    const Vec k2 = {2.8114574e-15f, -8.220635e-18f, 1.9572942e-20f, -3.8681703e-23f};
    const float w2 = w * w, w3 = w * w2;
    const Vec powers = _mm_setr_ps(1.0f, w, w2, w3);
    const Vec ws = splat_y(powers);
    const Vec odd = (powers * powers) * ws;  // w, w^3, w^5, w^7
    const Vec w8 = ws * splat_w(odd);
    const Vec odd2 = odd * w8;               // w^9 .. w^15
    const Vec odd3 = w8 * odd2;              // w^17 .. w^23
    return (hadd_sum(odd * k0) + hadd_sum(odd2 * k1)) + hadd_sum(odd3 * k2);
}

// The game's cosine of an angle in [-pi, pi], the sine's twin: the series to
// w^22, the even powers four at a time ((1, w, w^2, w^3) squared, then twice
// times w^8).
inline float series_cos(float w) {
    const Vec k0 = {1.0f, -0.5f, 0.041666668f, -0.0013888889f};
    const Vec k1 = {2.4801588e-05f, -2.755732e-07f, 2.0876758e-09f, -1.1470745e-11f};
    const Vec k2 = {4.7794773e-14f, -1.5619207e-16f, 4.1103176e-19f, -8.896791e-22f};
    const float w2 = w * w, w3 = w * w2;
    const Vec powers = _mm_setr_ps(1.0f, w, w2, w3);
    const Vec even = powers * powers;  // 1, w^2, w^4, w^6
    const Vec w8 = splat_z(powers) * splat_w(even);
    const Vec even2 = even * w8;       // w^8 .. w^14
    const Vec even3 = w8 * even2;      // w^16 .. w^22
    return (hadd_sum(even * k0) + hadd_sum(even2 * k1)) + hadd_sum(even3 * k2);
}

// The game's arcsine for |r| < 1: four terms in r, |r| and sqrt(1.0000001 -
// |r|), each times a quadratic in |r|, summed by horizontal adds.
inline float series_asin(float r) {
    const Vec ka = {-0.058063675f, -0.41861972f, 0.22480115f, 2.1733725f};
    const Vec kb = {0.61657274f, 4.296965f, -1.1894282f, -6.5378485f};
    const Vec kc = {-1.3692656f, -4.481793f, 1.4181067f, 5.4817924f};
    const float a = __builtin_fabsf(r);
    const float ar = a * r, arr = ar * r;
    const float q = (r - ar) / sqrt_ss(1.0000001f - a);
    const Vec terms = _mm_setr_ps(arr, 1.0f, arr, 1.0f) * _mm_setr_ps(q, q, r, r);
    const Vec as = _mm_set1_ps(a);
    return hadd_sum((as * (as * ka + kb) + kc) * terms);
}

// ---- What it calls in the game ----------------------------------------------

// The game's sine and cosine, _FSin(x, quadrant offset), and its tangent and
// arctangent, through their import stubs.
inline float game_fsin(u32 quadrant, float x) {
    return reinterpret_cast<float(GUEST_ABI*)(u32, u32, float)>(game_address(0x2fbe518))(quadrant, 0, x);
}
inline float game_sin(float x) { return game_fsin(0, x); }
inline float game_cos(float x) { return game_fsin(1, x); }
inline float game_tanf(float x) { return reinterpret_cast<float(GUEST_ABI*)(float)>(game_address(0x2fbe648))(x); }
inline float game_atan2f(float y, float x) { return reinterpret_cast<float(GUEST_ABI*)(float, float)>(game_address(0x2fbe698))(y, x); }

// The game's vector library: a quaternion from an axis and an angle, one from
// Euler angles, and a matrix from a position, a rotation and a scale.
struct Transform {
    Vec position, rotation, scale;
};
inline void quat_from_axis_angle(Vec* out, const Vec* axis, float angle) {
    reinterpret_cast<void(GUEST_ABI*)(Vec*, const Vec*, float)>(game_address(0x848e80))(out, axis, angle);
}
inline void quat_from_euler(Vec* out, Vec angles) { reinterpret_cast<void(GUEST_ABI*)(Vec*, Vec)>(game_address(0x1e4d420))(out, angles); }
inline void matrix_from_transform(const Transform* transform, Matrix* out) {
    reinterpret_cast<void(GUEST_ABI*)(const Transform*, Matrix*)>(game_address(0x83d4d0))(transform, out);
}
constexpr u64 kYAxis = 0x597be80;  // the engine's Y axis, set at start-up

// The world's sphere cast, the game's own (0x1c090e0): a sphere of the radius
// swept from `from` along `delta` against the filter's shapes. On a hit it
// gives the fraction of delta it got (and the hit point and normal, when
// asked for).
constexpr u32 kCameraCastFilter = 0x25;
inline bool sphere_cast(void* world, const Vec* from, const Vec* delta, Vec* hit, Vec* normal, float radius, float* fraction) {
    using Cast = u8(GUEST_ABI*)(void*, u32, const Vec*, const Vec*, Vec*, Vec*, float, float*);
    return reinterpret_cast<Cast>(game_address(0x1c090e0))(world, kCameraCastFilter, from, delta, hit, normal, radius, fraction) != 0;
}

// DL_PANIC for a missing singleton, as the game's GetInstance does it.
constexpr u64 kDlPanic = 0x24b55b0, kSingletonFile = 0x4d3b369, kSingletonFormat = 0x4d3b3bd;
void panic_singleton(u64 name) {
    using Panic = void(GUEST_ABI*)(u64, u64, u64, u64, ...);
    reinterpret_cast<Panic>(game_address(kDlPanic))(game_address(kSingletonFile), 0xb1, game_address(kSingletonFormat), game_address(name));
}

// ---- The world it reads -----------------------------------------------------

constexpr u64 kGameStateMan = 0x5956678;        // +0x14, +0x18: one-frame camera-param overrides
constexpr u64 kSoloParamRepository = 0x5940340, kSoloParamRepositoryName = 0x4d38ad0;  // slot 33: LockCamParam

// The debug menu's "Enable New Camera" (ChrCam), on in the game as shipped:
// on, the camera steers around walls with its side probes (P17); off, it
// casts back from them (P16).
constexpr u64 kEnableNewCamera = 0x5527a94;
inline bool new_camera() { return load<u8>(game_address(kEnableNewCamera)) != 0; }

// What the run-time patches rewrite, read where the game reads it.
constexpr u64 kFovUncapSite = 0x183af4e;          // cmovbe rax, rsi; the fov-uncap patch makes it mov rax, rsi
constexpr u64 kSixtyFpsJumpSite = 0x183dc97;      // the older camera's jump past the chase-rate pick
constexpr u64 kWallEscapeOnMissSite = 0x183dcfe;  // the wall escape when neither probe hits: 1, 0 at 60 fps
constexpr u64 kDistanceEase = 0x4d25e68;          // 0.1, 0.025 at 60 fps
constexpr u64 kChaseRateEase = 0x4d25e94;         // 1/30, rewritten at 60 fps

// A param row's address in its file (the repository's three layouts).
const u8* param_row(const u8* file, u32 index) {
    const u8 layout = file[0x2d];
    u64 offset;
    if (layout == 2)
        offset = load<u32>(reinterpret_cast<u64>(file) + 0x34 + index * 0xc);
    else if (layout <= 3 || !(file[0x2e] & 2))
        offset = load<u32>(reinterpret_cast<u64>(file) + 0x40 + index * 0xc + 4);
    else
        offset = load<u64>(reinterpret_cast<u64>(file) + 0x40 + index * 0x18 + 8);
    return reinterpret_cast<const u8*>(reinterpret_cast<u64>(file) + offset);
}

// ---- What the phases hand on ------------------------------------------------

// The camera as the moving floor leaves it (P5, P6), before the rest of the
// step: what the NaN guard (P21) puts back, and where the phases between
// measure this frame's movement from.
struct Saved {
    Vec position, chr_position, origin_b, orbit_reference, focus, wanted, wanted_tested, point;
    Vec basis_x, basis_y, basis_z, chr_neg_x, chr_y, chr_neg_z;
};

// The camera's own turns as the character moves, each held or not in this
// update: the three the mouse holds (all but R3's recentre) while
// auto-rotation is held, and each whose store in the game's code is a nop - a
// patch's, or a compare run's holding the game's update beside ours
// (engine/mouse_camera_step.h).
struct Hold {
    bool read_back = false;      // the yaw read back from where the camera is (P11, the game's 0x183c6e8)
    bool auto_turn = false;      // the auto turn's step of the yaw (P11, 0x183c870)
    bool return_weight = false;  // the pitch return's weight ramping down (P12, 0x183c984)
    bool wall_turn = false;      // the turn away from a wall (P17, 0x183dde6)
};

Hold held_turns() {
    const bool mouse = mouse_camera::held();
    const auto held = [mouse](int i) {
        return (mouse && (mouse_camera::kHeldWithMouse >> i & 1)) ||
               load<u8>(game_address(mouse_camera::kHoldSites[i].bn)) == 0x90;
    };
    Hold h;
    h.read_back = held(0);
    h.auto_turn = held(1);  // R3's recentre: held only by the patch's own nop
    h.return_weight = held(2);
    h.wall_turn = held(3);
    return h;
}

// What the phases hand each other.
struct Step {
    float dt;
    Vec dt4;  // dt in every lane
    // This frame's length in frames of the rate the camera's weights are
    // tuned for when the frame rate is converted - 60 with the FPS++ tuning,
    // 30 with the game's own (frame_rate_step.h) - for the per-frame weights;
    // 0: the game's own arithmetic (30, or a compare run).
    float frames = 0.0f;
    void* world;
    Saved saved;
    Vec stick;         // the right stick: x turns the pitch, y the yaw (P7)
    bool stick_turns;  // and it turns the camera this frame
    Hold hold;
};

// ---- P1, P2: the camera's LockCamParam row, eased toward --------------------

// Which LockCamParam row: the default (+0x32c); the normal id (+0x324, or
// GameStateMan's one-frame override when that is unset) when set; the
// alternate (+0x328, or the other override) when the alternate is asked for
// and set; the top override (+0x320) when set. Both overrides are consumed.
s32 camera_param_id(FollowCam& cam) {
    const u64 gsm = load<u64>(game_address(kGameStateMan));
    s32 id = cam.default_param_id;
    s32 normal = cam.param_id;
    if (normal < 0) normal = load<s32>(gsm + 0x14);
    if (normal >= 0) id = normal;
    if (cam.alternate_param) {
        s32 alternate = cam.alternate_param_id;
        if (alternate < 0) alternate = load<s32>(load<u64>(game_address(kGameStateMan)) + 0x18);
        if (alternate >= 0) id = alternate;
    }
    if (cam.param_override >= 0) id = cam.param_override;
    store<u64>(load<u64>(game_address(kGameStateMan)) + 0x14, ~0ull);
    return id;
}

// The row, by a binary search of the file's id index. Missing, the game falls
// back to a row with id 0 found by an odd walk - halving the count until an
// entry's id is 0 - which ours keeps.
const u8* lock_cam_row(s32 id) {
    u64 repo = load<u64>(game_address(kSoloParamRepository));
    if (!repo) {
        panic_singleton(kSoloParamRepositoryName);
        repo = load<u64>(game_address(kSoloParamRepository));
    }
    if (!load<u32>(repo + 0x9b8)) return nullptr;
    const u64 cap = load<u64>(repo + 0x9c0);
    if (!cap) return nullptr;
    const u64 file = load<u64>(load<u64>(cap + 0x70) + 0x70);
    if (!file) return nullptr;
    const u64 index = file + static_cast<u64>(static_cast<s64>(static_cast<s32>((load<u32>(file - 0x10) + 0xf) & 0xfffffff0u)));
    const s32 count = load<u16>(file + 0xa);
    const auto at = [&](s32 i) { return index + static_cast<u64>(static_cast<s64>(i)) * 8; };
    s32 lo = 0, hi = count - 1;
    while (lo <= hi) {
        const s32 mid = (lo + hi) >> 1;
        const u32 key = load<u32>(at(mid));
        if (key == static_cast<u32>(id)) {
            const s32 row = load<s32>(at(mid) + 4);
            if (row >= 0) {
                const u8* p = param_row(reinterpret_cast<const u8*>(file), static_cast<u32>(row));
                if (p) return p;
            }
            break;
        }
        if (key < static_cast<u32>(id)) lo = mid + 1;
        else hi = mid - 1;
    }
    for (s32 n = count; n > 0;) {
        n = (n - 1) >> 1;
        if (load<u32>(at(n)) != 0) continue;
        const s32 row = load<s32>(at(n) + 4);
        if (row < 0) return nullptr;
        return param_row(reinterpret_cast<const u8*>(file), static_cast<u32>(row));
    }
    return nullptr;
}

float ease(float from, float to, float k) { return from + (to - from) * k; }

// The distance, the pitch floor, the lock-on pitch shift, the look-at height
// and the field of view, each eased toward the row by +0x330 a frame. The
// field of view is camFovY in degrees, floored at 38 (below, a fixed
// 0.6632251 radians) and capped at 48 - unless the fov-uncap patch is in,
// which ours sees at the cap's own instruction.
void ease_toward_lock_cam_param(FollowCam& cam, const u8* row, float frames30) {
    const auto field = [row](u64 at) { return load<float>(reinterpret_cast<u64>(row) + at); };
    const float k = frame_rate_step::weight(cam.param_ease, frames30);  // the game's own 30 fps weight under either tuning
    cam.eased_distance = ease(cam.eased_distance, field(0x0), k);
    cam.pitch_min = ease(cam.pitch_min, field(0x4) * kOneDegree, k);
    cam.lock_shift_ratio = ease(cam.lock_shift_ratio, field(0x8), k);
    cam.look_at_offset[1] = ease(cam.look_at_offset[1], field(0xc), k);
    const float fov = field(0x14);
    float wanted;
    if (!(38.0f > fov)) {  // NaN takes the capped path
        const bool uncapped = load<u32>(game_address(kFovUncapSite)) == 0x90f08948u;
        const float capped = (uncapped || !(fov > 48.0f)) ? fov : 48.0f;
        wanted = capped * kOneDegree;
    } else {
        wanted = 0.6632251f;
    }
    cam.fov_y = ease(cam.fov_y, wanted, k);
}

// ---- P3: the distance and the rotation speeds -------------------------------

// The distance eases toward the eased param distance by 0.1 a frame (0.025 at
// 60 fps), and the stick's speeds sit between their minimum and maximum by the
// square of the player's camera-speed option.
void ease_distance_and_speeds(FollowCam& cam) {
    cam.distance = ease(cam.distance, cam.eased_distance, ldf(kDistanceEase));
    const float t = cam.option_speed_rate * cam.option_speed_rate;
    cam.pitch_speed = cam.pitch_speed_min + t * (cam.pitch_speed_max - cam.pitch_speed_min);
    cam.yaw_speed = cam.yaw_speed_min + t * (cam.yaw_speed_max - cam.yaw_speed_min);
    cam.pitch_high_speed = cam.pitch_high_speed_min + t * (cam.pitch_high_speed_max - cam.pitch_high_speed_min);
    cam.yaw_high_speed = cam.yaw_high_speed_min + t * (cam.yaw_high_speed_max - cam.yaw_high_speed_min);
}

// ---- P5: a moving floor carries the camera ----------------------------------

// Where the character stands, from its physics: ChrIns +0x58 -> +0x8 -> +0x3b0
// -> +0x68, the vector at +0x1e0.
Vec chr_physics_position(const u8* chr) {
    u64 p = load<u64>(reinterpret_cast<u64>(chr) + 0x58);
    p = load<u64>(p + 0x8);
    p = load<u64>(p + 0x3b0);
    p = load<u64>(p + 0x68);
    return *reinterpret_cast<const Vec*>(p + 0x1e0);
}

// A turn about Y around a pivot, as the game composes it: to the pivot,
// about Y, back. The game makes -pivot as 0 - pivot.
Matrix turn_about(Vec pivot, float angle) {
    const float s = game_sin(angle), c = game_cos(angle);
    return product(product(translation(kZero - pivot), rotation_y(s, c)), translation(pivot));
}

// A floor that moves (a lift, a turning platform) passes its velocity (+0x110)
// and its turn (+0x124) to the camera: everything the camera keeps - its
// basis and position, the character's frame, the points - turns with the
// floor about where the character stands, then moves with it. Without a turn
// the matrix is the identity, multiplied through all the same (a NaN in one
// lane reaches the others, as in the game).
void ride_moving_floor(FollowCam& cam, const u8* chr, Step& step) {
    const Vec move = step.dt4 * cam.floor_velocity;
    const float turn = step.dt * cam.floor_yaw_rate;
    const Matrix floor = turn != 0.0f ? turn_about(chr_physics_position(chr), turn) : kIdentity;

    const Matrix camera = product({cam.basis_x, cam.basis_y, cam.basis_z, cam.position}, floor);
    cam.basis_x = camera.r0;
    cam.basis_y = camera.r1;
    cam.basis_z = camera.r2;
    cam.position = move + camera.r3;
    const Matrix chr_frame = product({cam.chr_neg_x, cam.chr_y, cam.chr_neg_z, cam.chr_position}, floor);
    cam.chr_neg_x = chr_frame.r0;
    cam.chr_y = chr_frame.r1;
    cam.chr_neg_z = chr_frame.r2;
    cam.chr_position = move + chr_frame.r3;
    for (Vec* p : {&cam.origin_a, &cam.origin_b, &cam.orbit_reference, &cam.focus, &cam.wanted, &cam.wanted_tested, &cam.point})
        *p = move + transform(*p, floor);

    step.saved.position = cam.position;
    step.saved.chr_position = cam.chr_position;
    step.saved.origin_b = cam.origin_b;
    step.saved.orbit_reference = cam.orbit_reference;
    step.saved.focus = cam.focus;
    step.saved.wanted = cam.wanted;
    step.saved.wanted_tested = cam.wanted_tested;
    step.saved.point = cam.point;
}

// ---- P6: the rest of the restore point, and the wanted distance -------------

// With a distance factor (+0x17c), the distance the camera wants is how far
// it is from the orbit reference, times the factor, and at least 0.5.
void take_wanted_distance(FollowCam& cam, Step& step) {
    step.saved.basis_x = cam.basis_x;
    step.saved.basis_y = cam.basis_y;
    step.saved.basis_z = cam.basis_z;
    step.saved.chr_neg_x = cam.chr_neg_x;
    step.saved.chr_y = cam.chr_y;
    step.saved.chr_neg_z = cam.chr_neg_z;
    if (cam.distance_factor > 0.0f) {
        cam.wanted_distance = length3(step.saved.position - step.saved.orbit_reference)[0] * cam.distance_factor;
        if (0.5f > cam.wanted_distance) cam.wanted_distance = 0.5f;
    }
}

// ---- P7: the stick, and the follow weight -----------------------------------

// The right stick as the camera reads it: from the character's pad (ChrIns
// +0x58 -> +0x280, or +0x80 without one, at +0x40) when the camera takes
// input (+0x334), a zero vector otherwise.
Vec read_stick(const FollowCam& cam, const u8* chr) {
    if (!cam.input_enabled) return kZero;
    const u64 owner = load<u64>(reinterpret_cast<u64>(chr) + 0x58);
    u64 pad = load<u64>(owner + 0x280);
    if (!pad) pad = load<u64>(owner + 0x80);
    return *reinterpret_cast<const Vec*>(pad + 0x40);
}

// Whether the stick turns the camera this frame (it moves, and neither a
// reset nor a lock-on without stick offsets takes it). The follow weight is
// how fully the wanted point keeps up with the character (P10): 0 while the
// stick turns, 1 again when it lets go - at once when something else took
// the stick, otherwise for the follow-end wait, after which it ramps down
// over the release time.
bool stick_turns_camera(FollowCam& cam, Vec stick, float dt) {
    const bool taken = cam.reset_request || (cam.lock_on && !cam.lock_stick_offsets);
    if (!taken) {
        if (stick[0] != 0.0f || stick[1] != 0.0f) {
            cam.input_was_active = 1;
            cam.follow_weight = 0.0f;
            cam.follow_end_wait = 0.0f;
            return true;
        }
        const bool let_go_now = cam.input_was_active;
        cam.input_was_active = 0;
        if (!let_go_now) {
            cam.follow_end_wait = cam.follow_end_wait - dt;
            if (0.0f > cam.follow_end_wait) {
                cam.follow_end_wait = 0.0f;
                cam.follow_weight = cam.follow_weight - dt / cam.follow_release_time;
                if (0.0f > cam.follow_weight) cam.follow_weight = 0.0f;
            } else {
                cam.follow_weight = 1.0f;
            }
            return false;
        }
    } else {
        cam.input_was_active = 0;
    }
    cam.follow_weight = 1.0f;
    cam.follow_end_wait = cam.follow_end_wait_time;
    return false;
}

// ---- P8: the character's frame, chased --------------------------------------

// The character's frame from its physics (ChrIns +0x3b0 -> +0x68): its
// rotation at +0x1d0 as Euler angles - or, with +0x326 set, only the yaw,
// about Y - and its position at +0x1e0 (the w added, where the moving floor's
// pivot had it masked), made a matrix by the game's own helpers. The camera
// keeps the rows as -X (0 - X), Y, -Z and the position; the position, unless
// a reset is asked for, goes only part of the way there, by the character
// chase rate (+0x198). Returns where the frame itself is.
Vec follow_character_frame(FollowCam& cam, const u8* chr, const Step& step) {
    const u64 physics = load<u64>(load<u64>(reinterpret_cast<u64>(chr) + 0x3b0) + 0x68);
    const Vec angles = *reinterpret_cast<const Vec*>(physics + 0x1d0);
    Transform transform;
    if (load<u8>(physics + 0x326))
        quat_from_axis_angle(&transform.rotation, reinterpret_cast<const Vec*>(game_address(kYAxis)), angles[1]);
    else
        quat_from_euler(&transform.rotation, angles);
    transform.position = xyz(*reinterpret_cast<const Vec*>(physics + 0x1e0)) + kE3;
    transform.scale = _mm_set1_ps(1.0f);
    Matrix frame;
    matrix_from_transform(&transform, &frame);

    cam.chr_neg_x = kZero - xyz(frame.r0);
    cam.chr_y = xyz(frame.r1);
    cam.chr_neg_z = kZero - xyz(frame.r2);
    const Vec frame_position = point(frame.r3);
    cam.chr_position = frame_position;
    if (!cam.reset_request)
        cam.chr_position = step.saved.chr_position +
                           (frame_position - step.saved.chr_position) * _mm_set1_ps(frame_rate_step::weight(cam.chr_chase_rate, step.frames));
    return frame_position;
}

// ---- P9: the origin points --------------------------------------------------

// The look-at offset (+0x160) along the character's frame, from where the frame
// is (origin a) and from its chased position (origin b). With framing
// (+0x344), when the view at the camera's distance is taller than the offset's
// height, origin b's height goes halfway to it.
void place_origins(FollowCam& cam, Vec frame_position, const Step& step) {
    const Vec offset = cam.look_at_offset;
    const Vec along_x = cam.chr_neg_x * splat_x(offset);
    const Vec along_y = cam.chr_y * splat_y(offset);
    const Vec along_z = cam.chr_neg_z * splat_z(offset);
    cam.origin_a = along_z + (along_y + (frame_position + along_x));
    cam.origin_b = along_z + (along_y + (cam.chr_position + along_x));
    if (cam.framing) {
        const float distance = length3(step.saved.position - step.saved.orbit_reference)[0];
        const float half_view = distance * game_tanf(cam.fov_y * 0.5f);
        if (half_view > offset[1]) {
            const float height = offset[1] + (half_view - offset[1]) * 0.5f;
            cam.origin_b = along_z + ((cam.chr_position + along_x) + cam.chr_y * _mm_set1_ps(height));
        }
    }
}

// ---- P10: the orbit and focus points, the wanted point's chase --------------

// The orbit reference and the focus start the frame at origin b, the distance
// at the eased param distance. The wanted point (and its wall-tested copy)
// follow the character's chased position by the target chase rate, raised
// toward fully by the follow weight.
void chase_wanted_point(FollowCam& cam, const Step& step) {
    cam.orbit_reference = cam.origin_b;
    cam.focus = cam.origin_b;
    cam.distance = cam.eased_distance;
    const float rate = frame_rate_step::weight(cam.target_chase_rate + (1.0f - cam.target_chase_rate) * cam.follow_weight, step.frames);
    const Vec move = (cam.chr_position - step.saved.chr_position) * _mm_set1_ps(rate);
    cam.wanted = move + cam.wanted;
    cam.wanted_tested = move + cam.wanted_tested;
}

// ---- P11: the angles --------------------------------------------------------

// The pitch that looks along v (-atan2 of its height over its reach across)
// and the yaw (atan2 of x over z); a v with neither keeps the current one.
float pitch_along(Vec v, float current) {
    const float across = sqrt_ss(v[0] * v[0] + v[2] * v[2]);
    if (v[1] != 0.0f || across != 0.0f) return -game_atan2f(v[1], across);
    return current;
}
float yaw_along(Vec v, float current) {
    if (v[0] != 0.0f || v[2] != 0.0f) return game_atan2f(v[0], v[2]);
    return current;
}

// Lock-on: the camera turns toward the target by the lock chase rate (+0x240).
// The pitch it wants puts the target where the lock shift ratio (+0x248) of the
// view puts it - the arcsine of the triangle the camera, the focus and the
// target make, pi/2 when the target is nearer than that - and stays within
// limits that ease from the lock-on ones (+0x1f4, +0x1f8) to the normal ones
// (+0x1ec, +0x1f0) as the target's height difference grows from +0x1fc to
// +0x200. Nothing moves while the angles are within the play angle (+0x244)
// of where they should be; a step that would end inside it stops at its edge.
void turn_toward_lock_target(FollowCam& cam, float frames) {
    const Vec to_target = cam.lock_target - cam.focus;
    const Vec to_focus = cam.focus - cam.wanted;
    const float pitch = cam.pitch, yaw = cam.yaw;
    const float target_pitch = pitch_along(to_target, pitch);
    const float target_yaw = yaw_along(to_target, yaw);
    const float from_pitch = pitch_along(to_focus, pitch);
    const float from_yaw = yaw_along(to_focus, yaw);

    const float shift = cam.fov_y * cam.lock_shift_ratio * 0.5f;
    const float target_distance = length3(to_target)[0];
    const float across = length3(to_focus)[0] * series_sin(wrap_angle(shift));
    const float tilt = target_distance > across ? series_asin(across / target_distance) : kHalfPi;
    const float height = (__builtin_fabsf(to_target[1]) - cam.lock_height_low) / (cam.lock_height_high - cam.lock_height_low);
    const float t = 0.0f > height ? 0.0f : min_ss(1.0f, height);
    float want = target_pitch + (shift + tilt);
    const float lowest = cam.lock_pitch_min + t * (cam.pitch_min - cam.lock_pitch_min);
    if (lowest > want)
        want = lowest;
    else
        want = min_ss(cam.lock_pitch_max + t * (cam.pitch_max - cam.lock_pitch_max), want);

    const float rate = frame_rate_step::weight(cam.lock_chase_rate, frames);
    const float pitch_error = wrap_angle(want - from_pitch);
    const float new_pitch = wrap_angle(from_pitch + rate * pitch_error);
    const float yaw_error = wrap_angle(target_yaw - from_yaw);
    const float new_yaw = wrap_angle(from_yaw + rate * yaw_error);
    cam.pitch = new_pitch;
    cam.yaw = new_yaw;
    const float error_sq = pitch_error * pitch_error + yaw_error * yaw_error;
    const float play_sq = cam.play_angle * cam.play_angle;
    if (play_sq > error_sq) {
        cam.pitch = from_pitch;
        cam.yaw = from_yaw;
    } else {
        const float step_pitch = wrap_angle_wide(new_pitch - from_pitch), step_yaw = wrap_angle_wide(new_yaw - from_yaw);
        if (play_sq > step_pitch * step_pitch + step_yaw * step_yaw) {
            const float inverse = 1.0f / sqrt_ss(error_sq);
            const float to_edge_pitch = pitch_error - cam.play_angle * (pitch_error * inverse);
            const float to_edge_yaw = yaw_error - cam.play_angle * (yaw_error * inverse);
            cam.pitch = wrap_angle(from_pitch + rate * to_edge_pitch);
            cam.yaw = wrap_angle(from_yaw + to_edge_yaw * rate);
        }
    }

    cam.auto_turn = 0;
    cam.auto_pitch = cam.pitch;
    cam.auto_yaw = cam.yaw;
    if (cam.lock_sets_reference)
        cam.pitch_reference = cam.pitch;
    else
        cam.auto_pitch = cam.pitch_reference;
}

// A running auto turn steps the angles toward its own (+0x26c, +0x270) by the
// lock chase rate, and ends when its time (+0x264) runs out, when both angles
// are within a degree of it, or when the stick turns the camera after its
// time is up. Whether it still runs.
bool step_auto_turn(FollowCam& cam, const Step& step) {
    bool running = true;
    float left = cam.auto_turn_timer - step.dt;
    cam.auto_turn_timer = left;
    if (0.0f > left) {
        cam.auto_turn = 0;
        cam.auto_turn_timer = 0.0f;
        running = false;
        left = 0.0f;
    }
    const float rate = frame_rate_step::weight(cam.lock_chase_rate, step.frames);
    const float auto_pitch = cam.auto_pitch;
    const float pitch = wrap_angle(cam.pitch + rate * wrap_angle(auto_pitch - cam.pitch));
    cam.pitch = pitch;
    const float auto_yaw = cam.auto_yaw;
    const float yaw = wrap_angle(cam.yaw + rate * wrap_angle(auto_yaw - cam.yaw));
    if (!step.hold.auto_turn) cam.yaw = yaw;  // held, the end test below still takes this yaw, as the game's does
    if (cam.turn_behind_sets_reference) cam.pitch_reference = pitch;
    const bool pitch_there = kOneDegree > __builtin_fabsf(wrap_angle(pitch - auto_pitch));
    if (pitch_there && kOneDegree > __builtin_fabsf(wrap_angle(yaw - auto_yaw))) {
        cam.auto_turn = 0;
        cam.auto_turn_timer = 0.0f;
        running = false;
        left = 0.0f;
    }
    if (step.stick_turns && 0.0f >= left) {
        cam.auto_turn = 0;
        cam.auto_turn_timer = 0.0f;
        running = false;
    }
    return running;
}

// Who sets the angles this frame. A reset (+0x13c) puts them at the target
// angles (+0x148, when the one-shot +0x13d asks for them) or level behind the
// character; lock-on (+0x260) turns them toward the target; a turn-behind
// request (+0x262) starts an auto turn to behind the character, at the pitch
// reference or (with +0x275) taking the pitch as the reference; otherwise they
// are read back from where the camera is - from the wanted point to the
// focus. A running auto turn then steps them. Whether it still runs.
bool choose_angles(FollowCam& cam, const Step& step) {
    if (cam.reset_request) {
        if (cam.reset_to_target) {
            cam.reset_to_target = 0;
            cam.pitch = cam.target_pitch;
            cam.yaw = cam.target_yaw;
            cam.auto_turn = 0;
            cam.auto_pitch = cam.target_pitch;
            cam.auto_yaw = cam.target_yaw;
            cam.pitch_reference = cam.target_pitch;
        } else {
            cam.pitch = 0.0f;
            cam.yaw = yaw_along(cam.chr_neg_z, cam.yaw);
            cam.auto_turn = 0;
            cam.auto_pitch = 0.0f;
            cam.auto_yaw = cam.yaw;
            cam.pitch_reference = 0.0f;
        }
        return false;
    }
    if (cam.lock_on) {
        turn_toward_lock_target(cam, step.frames);
        return false;
    }
    if (cam.turn_behind) {
        cam.auto_turn = 1;
        cam.auto_pitch = 0.0f;
        cam.auto_yaw = yaw_along(cam.chr_neg_z, cam.auto_yaw);
        if (cam.turn_behind_sets_reference)
            cam.pitch_reference = cam.pitch;
        else
            cam.auto_pitch = cam.pitch_reference;
        cam.auto_turn_timer = cam.auto_turn_duration;
    } else {
        cam.pitch = pitch_along(cam.focus - cam.wanted, cam.pitch);
        if (!step.hold.read_back) cam.yaw = yaw_along(cam.focus - cam.wanted, cam.yaw);
        if (!cam.auto_turn) return false;
    }
    return step_auto_turn(cam, step);
}

// ---- P12: the pitch return's timers -----------------------------------------

// After the stick lets go (P13 restarts them), the pitch return waits (+0x134,
// from +0x1e0), then its weight (+0x130) ramps from 1 down to 0 over the
// release time (+0x1e4). Held, the ramp is not stored - but the clamp to 0
// still tests the value it would have had, as the game's does.
void run_return_timers(FollowCam& cam, const Step& step) {
    const float dt = step.dt;
    cam.return_wait = max_ss(0.0f, cam.return_wait - dt);
    if (0.0f >= cam.return_wait) {
        const float weight = cam.return_weight - dt / cam.return_release_time;
        if (!step.hold.return_weight) cam.return_weight = weight;
        if (0.0f > weight) cam.return_weight = 0.0f;
    }
}

// ---- P13: the stick ---------------------------------------------------------

// The pitch and yaw speeds, raised from the normal ones toward the high ones
// by t (the fast turn's ramp), as (pitch, yaw, 0, 0).
inline Vec turn_speeds(const FollowCam& cam, float t) {
    return _mm_setr_ps(cam.pitch_speed + t * (cam.pitch_high_speed - cam.pitch_speed),
                       cam.yaw_speed + t * (cam.yaw_high_speed - cam.yaw_speed), 0.0f, 0.0f);
}

// Under lock-on the stick only offsets the view: the offsets (+0x278, +0x27c)
// ease (+0x288) toward the stick times their gains (+0x280, +0x284) and are
// added to the angles.
void offset_lock_view(FollowCam& cam, Vec stick) {
    const float ease = cam.lock_offset_ease;
    cam.lock_pitch_offset = cam.lock_pitch_offset + (stick[0] * cam.lock_pitch_gain - cam.lock_pitch_offset) * ease;
    cam.lock_yaw_offset = cam.lock_yaw_offset + (stick[1] * cam.lock_yaw_gain - cam.lock_yaw_offset) * ease;
    cam.pitch = wrap_angle(cam.lock_pitch_offset + cam.pitch);
    cam.yaw = wrap_angle(cam.lock_yaw_offset + cam.yaw);
}

// The fast turn's ramp (+0x138) for a stick pushed all the way (past 0.998)
// and within the direct-pitch angle (+0x238) of sideways: toward 1 over
// +0x1e8 while it pushes right, toward -1 while it pushes left, from 0 when
// it was going the other way. Any other stick drops it to 0. Returns it.
float fast_turn_ramp(FollowCam& cam, Vec stick, float dt) {
    if (!(length3(stick)[0] > 0.998f) || !(__builtin_fabsf(stick[1]) > series_cos(wrap_angle(cam.direct_pitch_angle)))) {
        cam.fast_turn = 0.0f;
        return 0.0f;
    }
    float ramp = cam.fast_turn;
    if (stick[1] > 0.0f) {
        if (0.0f >= ramp) {
            cam.fast_turn = 0.0f;
            ramp = 0.0f;
        }
        ramp = dt / cam.fast_turn_ramp_time + ramp;
        cam.fast_turn = ramp;
        if (ramp > 1.0f) {
            cam.fast_turn = 1.0f;
            ramp = 1.0f;
        }
    } else {
        if (ramp >= 0.0f) {
            cam.fast_turn = 0.0f;
            ramp = 0.0f;
        }
        ramp = ramp - dt / cam.fast_turn_ramp_time;
        cam.fast_turn = ramp;
        if (-1.0f > ramp) {
            cam.fast_turn = -1.0f;
            ramp = -1.0f;
        }
    }
    return ramp;
}

// The fast turn: the yaw turns the stick's way at the speeds the ramp raises
// (the stick's own deflection does not count), and the pitch eases (+0x23c)
// toward the one the stick's tilt asks for - its share of the sine of the
// direct-pitch angle, of the pitch limit on that side.
void fast_turn(FollowCam& cam, const Step& step, float ramp) {
    const Vec stick = step.stick;
    const Vec angles = _mm_setr_ps(cam.pitch, cam.yaw, 0.0f, 0.0f);
    const Vec turn = step.dt4 * turn_speeds(cam, __builtin_fabsf(ramp));
    const float pitch = cam.pitch;
    cam.yaw = wrap_angles(stick[1] > 0.0f ? angles + turn : angles - turn)[1];
    // The pitch is held within its limits as the game holds it: the minimum
    // by a branch; the maximum by a cmov when the stick's x is positive and
    // by minss when not - so only there does a denormal maximum survive.
    if (stick[0] > 0.0f) {
        const float direct = stick[0] / series_sin(wrap_angle(cam.direct_pitch_angle)) * cam.pitch_max;
        const float target = 0.0f > direct ? 0.0f : min_ss(cam.pitch_max, direct);
        const float eased = pitch + (target - pitch) * frame_rate_step::weight(cam.direct_pitch_rate, step.frames);
        cam.pitch = choose(cam.pitch_min > eased, cam.pitch_min, choose(eased > cam.pitch_max, cam.pitch_max, eased));
    } else {
        const float direct = stick[0] / series_sin(wrap_angle(-cam.direct_pitch_angle)) * cam.pitch_min;
        const float target = cam.pitch_min > direct ? cam.pitch_min : min_ss(0.0f, direct);
        const float eased = pitch + (target - pitch) * frame_rate_step::weight(cam.direct_pitch_rate, step.frames);
        cam.pitch = choose(cam.pitch_min > eased, cam.pitch_min, min_ss(cam.pitch_max, eased));
    }
}

// The stick turns both angles by its deflection at the normal speeds (the
// ramp, 0 here, still multiplied in), the pitch kept within its limits.
void turn_by_stick(FollowCam& cam, const Step& step, float ramp) {
    const Vec turn = step.dt4 * (step.stick * turn_speeds(cam, __builtin_fabsf(ramp)));
    const Vec angles = wrap_angles(_mm_setr_ps(cam.pitch, cam.yaw, 0.0f, 0.0f) + turn);
    cam.pitch = choose(cam.pitch_min > angles[0], cam.pitch_min, min_ss(cam.pitch_max, angles[0]));
    cam.yaw = angles[1];
}

// The stick turns the camera - unless a reset, an auto turn or lock-on
// without stick offsets has it. Before the stick turns the camera, its
// angles are read back from where it is (the point toward the focus). A
// stick that turned it restarts the pitch return (P12, P14), with this pitch
// as the reference.
void steer_by_stick(FollowCam& cam, const Step& step, bool auto_turning) {
    if (cam.reset_request) return;
    if (cam.lock_on) {
        if (cam.lock_stick_offsets && !auto_turning) offset_lock_view(cam, step.stick);
        return;
    }
    if (auto_turning) return;
    if (step.stick_turns) {
        const Vec to_focus = cam.focus - cam.point;
        cam.pitch = pitch_along(to_focus, cam.pitch);
        cam.yaw = yaw_along(to_focus, cam.yaw);
    }
    const float ramp = fast_turn_ramp(cam, step.stick, step.dt);
    mouse_camera::free_camera_ran();
    bool turned = step.stick_turns;
    if (ramp != 0.0f) {
        fast_turn(cam, step, ramp);  // the stick has the camera: the mouse waits
    } else {
        turn_by_stick(cam, step, ramp);
        // The mouse, as Dark Souls III turns it (engine/mouse_camera.h): an
        // angle a count, only when the stick has not turned the camera -
        // after the stick's angles are stored, where the game's code goes on
        // to 0x183ce67 - and a turn restarts the pitch return as the stick's does.
        if (mouse_camera::turn(step.stick[0], step.stick[1], cam.pitch, cam.yaw, cam.pitch_min, cam.pitch_max))
            turned = true;
    }
    if (turned) {
        cam.pitch_reference = cam.pitch;
        cam.return_weight = 1.0f;
        cam.return_wait = cam.return_wait_time;
    }
}

// ---- P14: the pitch return --------------------------------------------------

// Without lock-on or an auto turn, the pitch eases back toward its reference:
// it keeps +0x1a8 of its difference from it each frame.
void return_pitch(FollowCam& cam, float frames) {
    if (cam.lock_on || cam.auto_turn) return;
    cam.pitch = wrap_angle(cam.pitch_reference +
                           wrap_angle(cam.pitch - cam.pitch_reference) * frame_rate_step::keep(cam.pitch_chase_rate, frames));
}

// ---- P15: the wanted position -----------------------------------------------

// The wanted point goes where the angles look from the focus, as far from it
// as it was (1e-6 when that was under 1e-5).
void look_along_angles(FollowCam& cam) {
    const float pitch_sin = game_sin(cam.pitch), pitch_cos = game_cos(cam.pitch);
    const float yaw_sin = game_sin(cam.yaw), yaw_cos = game_cos(cam.yaw);
    const Vec facing = xyz(row_times(row_times(kE2, rotation_x(pitch_sin, pitch_cos)), rotation_y(yaw_sin, yaw_cos)));
    const float away = length3(cam.wanted - cam.focus)[0];
    cam.wanted = cam.focus - normalize4(facing) * _mm_set1_ps(away < 1e-5f ? 1e-6f : away);
}

// The wanted point along the angles. With a shrink divisor (+0x170),
// the wanted distance becomes the camera's distance from the orbit reference,
// less the share (+0x174..+0x178) the character's sideways move this frame
// (along the frame-start X axis, over the divisor) takes off, and at least
// 0.5. Then the wanted point is moved to the wanted distance from the orbit
// reference.
void place_wanted(FollowCam& cam, const Step& step) {
    look_along_angles(cam);

    if (cam.shrink_divisor > 0.0f) {
        const Vec moved = cam.chr_position - step.saved.chr_position;
        const Vec side = step.saved.basis_x;
        const float across = __builtin_fabsf(side[0] * moved[0] + side[2] * moved[2]) / cam.shrink_divisor;
        const float shrink = cam.shrink_min > across ? cam.shrink_min : min_ss(cam.shrink_max, across);
        cam.wanted_distance = (1.0f - shrink) * length3(cam.position - cam.orbit_reference)[0];
        if (0.5f > cam.wanted_distance) cam.wanted_distance = 0.5f;
    }
    const Vec out = cam.wanted - cam.orbit_reference;
    const float out_length = length3(out)[0];
    const float scale = out_length > 1e-5f ? cam.wanted_distance / out_length : 1.0f;
    cam.wanted = cam.orbit_reference + out * _mm_set1_ps(scale);
}

// ---- P16: back from the walls -----------------------------------------------

// The older camera's walls (with "Enable New Camera" off). From the orbit
// reference a sphere (+0x18c) is cast toward the wanted point;
// on a hit the wanted point comes back to where the sphere got (at least 0.01
// out). The wall-tested point is the same direction at the eased distance
// (+0x184), cast the same way, and on a hit it comes back too (at least 1e-5
// out).
void cast_back_from_walls(FollowCam& cam, void* world) {
    if (!new_camera() && world) {
        const Vec from = cam.orbit_reference;
        const Vec delta = cam.wanted - from;
        Vec hit;
        float fraction;
        if (sphere_cast(world, &from, &delta, &hit, nullptr, cam.sphere_radius, &fraction)) {
            const float length = length3(delta)[0];
            float share = length * fraction / length;
            if (0.01f > length * share) share = 0.01f / length;
            cam.wanted = from + delta * _mm_set1_ps(share);
        }
    }
    const Vec out = cam.wanted - cam.orbit_reference;
    const float out_length = length3(out)[0];
    const float scale = out_length > 1e-5f ? cam.distance / out_length : 1.0f;
    cam.wanted_tested = cam.orbit_reference + out * _mm_set1_ps(scale);
    if (!new_camera() && world) {
        const Vec from = cam.orbit_reference;
        const Vec delta = cam.wanted_tested - from;
        Vec hit;
        float fraction;
        if (sphere_cast(world, &from, &delta, &hit, nullptr, cam.sphere_radius, &fraction)) {
            const Vec length = length3(delta);
            cam.wanted_tested = from + (_mm_set1_ps(1.0f) / length) * (delta * _mm_set1_ps(max_ss(1e-5f, length[0] * fraction)));
        }
    }
}

// ---- P17: around the walls, and the chase rates -----------------------------

// With one probe hitting, the character and XZ chase rates (+0x198, +0x1ac)
// jump to their escape values; otherwise they ease back to the normal ones by
// 1/30 a frame.
void pick_chase_rates(FollowCam& cam, bool escaping) {
    if (escaping) {
        cam.chr_chase_rate = cam.chr_chase_rate_escape;
        cam.chase_rate_xz = cam.chase_rate_xz_escape;
    } else {
        const float ease = ldf(kChaseRateEase);
        cam.chr_chase_rate = cam.chr_chase_rate + (cam.chr_chase_rate_normal - cam.chr_chase_rate) * ease;
        cam.chase_rate_xz = cam.chase_rate_xz + (cam.chase_rate_xz_normal - cam.chase_rate_xz) * ease;
    }
}

// How fast a probe's hit turns the view: +0x29c when the wall is within
// +0x2a8 of the probe's start, easing to +0x2a4 at its end.
float escape_rate(const FollowCam& cam, float fraction) {
    if (cam.escape_near_fraction > fraction) return cam.escape_rate_near;
    const float t = (fraction - cam.escape_near_fraction) / (1.0f - cam.escape_near_fraction);
    const float clamped = 0.0f > t ? 0.0f : min_ss(1.0f, t);
    return cam.escape_rate_near + clamped * (cam.escape_rate_far - cam.escape_rate_near);
}

// The new camera steers around walls. Two spheres (+0x2c0) are cast from
// either side of origin a (+0x2c4 out), each along the direction to the
// wall-tested point turned by the probe angle (+0x2b8) its way, +0x2bc long.
// When exactly one hits, the yaw correction (+0x294) turns away from that
// side - from 0 if it was turning the other way. While the wall escape (+0x28c,
// set when neither probe hits; dropped by the stick, an auto turn, lock-on or
// a reset) holds, the correction turns the view: at most +0x2b0 a second,
// fading by +0x2b4 a frame when no single probe hits, nothing under a degree.
// The wall-tested point turns about origin a, the wanted point follows at
// the wanted distance, the yaw turns, and the wanted point goes along the
// angles again.
void steer_around_walls(FollowCam& cam, const Step& step, void* world) {
    cam.probe_a = 0.0f;
    cam.probe_b = 0.0f;
    if (!new_camera()) {
        cam.wall_escape = 0;
        // With the 60 fps patch this path reaches the chase-rate pick with
        // the flag's address where the probes' result is kept; its low byte
        // (0x94) is not 0, so the escape rates are taken.
        if (load<u16>(game_address(kSixtyFpsJumpSite)) == 0x05d6) pick_chase_rates(cam, true);
        return;
    }
    if (cam.input_was_active || cam.auto_turn || cam.lock_on || cam.reset_request) cam.wall_escape = 0;
    bool one_side = false;
    if (world) {
        const Vec ahead = normalize4(xyz(cam.wanted_tested - cam.origin_a));
        const float back = cam.probe_offset / game_tanf(cam.probe_angle);
        const float slant = back / series_cos(wrap_angle(cam.probe_angle));
        const Vec behind = cam.origin_a - ahead * _mm_set1_ps(back);
        const float angle = cam.probe_angle;
        const float sin_a = game_sin(angle), cos_a = game_cos(angle), sin_b = game_sin(-angle), cos_b = game_cos(-angle);
        const Matrix turn_a = rotation_y(sin_a, cos_a), turn_b = rotation_y(sin_b, cos_b);
        const Vec from_a = behind + _mm_set1_ps(slant) * rows_dot(turn_a, ahead);
        const Vec from_b = behind + _mm_set1_ps(slant) * rows_dot(turn_b, ahead);
        const Vec delta_a = rows_dot(turn_a, ahead) * _mm_set1_ps(cam.probe_reach);
        const Vec delta_b = rows_dot(turn_b, ahead) * _mm_set1_ps(cam.probe_reach);
        float fraction_a = 0.0f, fraction_b = 0.0f;
        const bool hit_a = sphere_cast(world, &from_a, &delta_a, nullptr, nullptr, cam.probe_radius, &fraction_a);
        const bool hit_b = sphere_cast(world, &from_b, &delta_b, nullptr, nullptr, cam.probe_radius, &fraction_b);
        cam.probe_a = fraction_a;
        cam.probe_b = fraction_b;
        if (!hit_a && !hit_b) {
            cam.wall_escape = load<u8>(game_address(kWallEscapeOnMissSite));
        } else if (hit_a != hit_b) {
            float correction = cam.yaw_correction;
            if (hit_a) {
                if (correction > 0.0f) {
                    cam.yaw_correction = 0.0f;
                    correction = 0.0f;
                }
                correction = correction - escape_rate(cam, fraction_a) * step.dt;
            } else {
                if (0.0f > correction) {
                    cam.yaw_correction = 0.0f;
                    correction = 0.0f;
                }
                correction = correction + escape_rate(cam, fraction_b) * step.dt;
            }
            if (!step.hold.wall_turn) cam.yaw_correction = correction;
            one_side = true;
        }
    }
    if (!cam.wall_escape) {
        cam.escape_turn = 0.0f;
        cam.yaw_correction = 0.0f;
    } else if (cam.yaw_correction != 0.0f) {
        const float scaled = (one_side ? 1.0f : cam.escape_decay) * cam.yaw_correction;
        const float limit = step.dt * cam.escape_max_rate;
        float turn = -limit > scaled ? -limit : min_ss(limit, scaled);
        if (__builtin_fabsf(turn) < kOneDegree) turn = 0.0f;
        cam.yaw_correction = turn;
        const float turn_sin = game_sin(turn), turn_cos = game_cos(turn);
        const Matrix rotation = rotation_y(turn_sin, turn_cos);
        cam.wanted_tested = cam.origin_a + rows_dot(rotation, cam.wanted_tested - cam.origin_a);
        const Vec out = cam.wanted_tested - cam.orbit_reference;
        const float out_length = length3(out)[0];
        cam.wanted = cam.orbit_reference + out * _mm_set1_ps(out_length > 1e-5f ? cam.wanted_distance / out_length : 1.0f);
        cam.yaw = cam.yaw - cam.yaw_correction;
        look_along_angles(cam);
    }
    pick_chase_rates(cam, one_side);
}

// ---- P18: the chase, and the basis ------------------------------------------

// The camera point chases the wall-tested point. How far, per axis: the XZ and
// Y chase rates (+0x1ac, +0x1b8; +0x1c8, +0x1cc under lock-on or an auto
// turn). While the point is within the min chase angle (+0x1bc, or +0x1d0) of
// the target as the focus sees them, only Y chases; up to the max angle
// (+0x1c0, or +0x1d4 - first raised to the min when below it), XZ chases in
// part. The pitch return's weight (+0x130) pulls all three toward 1. Then the
// point's distance from the orbit reference chases the target's by +0x1c4.
// A reset puts the point on the target.
Vec chase_point(FollowCam& cam, float frames) {
    const Vec target = cam.wanted_tested;
    if (cam.reset_request) return target;
    const bool lock = cam.auto_turn || cam.lock_on;
    const float rate_xz = lock ? cam.lock_chase_rate_xz : cam.chase_rate_xz;
    const float rate_y = lock ? cam.lock_chase_rate_y : cam.chase_rate_y;
    if (cam.chase_min_angle > cam.chase_max_angle) cam.chase_max_angle = cam.chase_min_angle;
    if (cam.lock_chase_min_angle > cam.lock_chase_max_angle) cam.lock_chase_max_angle = cam.lock_chase_min_angle;
    const float min_angle = lock ? cam.lock_chase_min_angle : cam.chase_min_angle;
    const float max_angle = lock ? cam.lock_chase_max_angle : cam.chase_max_angle;

    const Vec point = cam.point;
    const Vec to_target = target - cam.focus;
    float cos_apart = 1.0f;
    if (length3(to_target)[0] > 1e-5f) {
        const Vec to_point = point - cam.focus;
        if (length3(to_point)[0] > 1e-5f) {
            const Vec both = normalize4(xyz(to_target)) * normalize4(xyz(to_point));
            cos_apart = (both[0] + both[1]) + both[2];
        }
    }
    Vec rates = _mm_setr_ps(rate_xz, rate_y, rate_xz, 0.0f);
    const float cos_min = series_cos(wrap_angle(min_angle));
    if (cos_apart > cos_min) {
        rates = rates * kE1;
    } else {
        const float cos_max = series_cos(wrap_angle(max_angle));
        if (cos_apart > cos_max && cos_min > cos_max) {
            const float t = (cos_apart - cos_min) / (cos_max - cos_min);
            rates = rates * _mm_setr_ps(t, 1.0f, t, 0.0f);
        }
    }
    Vec share = rates + (_mm_setr_ps(1.0f, 1.0f, 1.0f, 0.0f) - rates) * _mm_set1_ps(cam.return_weight);
    if (frames > 0.0f) {
        share = _mm_setr_ps(frame_rate_step::weight(share[0], frames), frame_rate_step::weight(share[1], frames),
                            frame_rate_step::weight(share[2], frames), share[3]);
    }
    const Vec chased = point + (target - point) * share;

    const Vec out = chased - cam.orbit_reference;
    const float out_length = length3(out)[0];
    float scale = 1.0f;
    if (out_length > 1e-5f) {
        const float target_distance = length3(target - cam.orbit_reference)[0];
        const float point_distance = length3(point - cam.orbit_reference)[0];
        scale = (point_distance + (target_distance - point_distance) * frame_rate_step::weight(cam.chase_distance_rate, frames)) /
                out_length;
    }
    return cam.orbit_reference + out * _mm_set1_ps(scale);
}

// The basis from the point toward the orbit reference (the old Z when that is
// straight up or down), X across it and world up, Y from both; a basis with a
// NaN anywhere goes back to the frame-start one. The camera is at the point.
void place_camera(FollowCam& cam, const Step& step, Vec point) {
    Vec forward = normalize4(xyz(cam.orbit_reference - point));
    if (1e-5f > forward[0] * forward[0] + forward[2] * forward[2]) forward = cam.basis_z;
    const Vec side = normalize4(xyz(cross3(kE1, forward)));
    const Vec up = normalize4(xyz(cross3(forward, side)));
    cam.basis_x = side;
    cam.basis_y = up;
    cam.basis_z = forward;
    if (any_nan(side) || any_nan(up) || any_nan(forward)) {
        cam.basis_x = step.saved.basis_x;
        cam.basis_y = step.saved.basis_y;
        cam.basis_z = step.saved.basis_z;
        cam.position = step.saved.position;
    }
    cam.point = point;
    cam.position = point;
}

// ---- P19: the camera's own collision ----------------------------------------

// A sphere (+0x18c) cast from the orbit reference to `to`: on a hit, where
// the sphere got - at least 1e-5 out, along the wanted direction when `to` is
// right at the reference; otherwise `to` itself.
Vec pulled_in(const FollowCam& cam, void* world, Vec to) {
    if (!world) return to;
    const Vec from = cam.orbit_reference;
    const Vec delta = to - from;
    Vec hit;
    float fraction;
    if (!sphere_cast(world, &from, &delta, &hit, nullptr, cam.sphere_radius, &fraction)) return to;
    const Vec length = length3(delta);
    Vec offset;
    if (1e-5f > length[0])
        offset = normalize4(xyz(cam.wanted_tested - cam.orbit_reference)) * _mm_set1_ps(1e-5f);
    else
        offset = (_mm_set1_ps(1.0f) / length) * (delta * _mm_set1_ps(max_ss(1e-5f, length[0] * fraction)));
    return from + offset;
}

// The camera point, pulled in from walls.
void collide_camera(FollowCam& cam, void* world) { cam.position = pulled_in(cam, world, cam.point); }

// ---- P20: the one-shot requests ---------------------------------------------

// The reset, lock-on, alternate-param and turn-behind requests last a frame;
// whoever wants them sets them again.
void clear_requests(FollowCam& cam) {
    cam.reset_request = 0;
    cam.lock_on = 0;
    cam.alternate_param = 0;
    cam.turn_behind = 0;
}

// ---- P21: the NaN guard -----------------------------------------------------

// A NaN anywhere in the basis or the position puts the camera back as the
// frame found it (after the moving floor): the basis and position, the
// character's frame, and every point but origin a.
void guard_against_nan(FollowCam& cam, const Saved& saved) {
    if (!any_nan(cam.basis_x) && !any_nan(cam.basis_y) && !any_nan(cam.basis_z) && !any_nan(cam.position)) return;
    cam.basis_x = saved.basis_x;
    cam.basis_y = saved.basis_y;
    cam.basis_z = saved.basis_z;
    cam.position = saved.position;
    cam.chr_neg_x = saved.chr_neg_x;
    cam.chr_y = saved.chr_y;
    cam.chr_neg_z = saved.chr_neg_z;
    cam.chr_position = saved.chr_position;
    cam.origin_b = saved.origin_b;
    cam.orbit_reference = saved.orbit_reference;
    cam.focus = saved.focus;
    cam.wanted = saved.wanted;
    cam.wanted_tested = saved.wanted_tested;
    cam.point = saved.point;
}

// ---- P22: the keep-out spheres ----------------------------------------------

// The camera keep-out spheres, a list (0x593e858 -> +0x8 the first) the
// camera reads: each a center, a radius in whole units, bit 1 of its flags
// set when it counts, and the next.
struct KeepOutSphere {
    Vec center;
    u8 pad_10[0x50];
    u8 radius;  // +0x60
    u8 flags;   // +0x61
    u8 pad_62[6];
    const KeepOutSphere* next;  // +0x68
};
static_assert(offsetof(KeepOutSphere, radius) == 0x60 && offsetof(KeepOutSphere, flags) == 0x61);
static_assert(offsetof(KeepOutSphere, next) == 0x68);
constexpr u64 kKeepOutSpheres = 0x593e858;

// RendMan, whose current draw list takes the debug shapes: +0x20 the lists,
// their [+0x20] the current one, at +0x10 + 8 * that, its shape object at
// +0x40.
constexpr u64 kRendMan = 0x5940298, kRendManName = 0x4d3b1e1;
constexpr u64 kDrawShape = 0x135d810;  // (the lists, the shape's 3x4 matrix)

// The debug draw of a keep-out sphere: the current shape is begun (its
// vtable +0x28), set to the camera's look - colours (0.25, 0, 0.25, 0.4) at
// +0x20 and +0x30, modes 1 (+0x1c) and 2 (+0x18) and +0x70 of the current
// list's shape 2, each change flagged at +0x8 - and drawn scaled by the
// radius at the center.
void draw_keep_out_sphere(Vec center, float radius) {
    u64 rendman = load<u64>(game_address(kRendMan));
    if (!rendman) {
        panic_singleton(kRendManName);
        rendman = load<u64>(game_address(kRendMan));
    }
    const u64 lists = load<u64>(rendman + 0x20);
    const auto current_shape = [lists] {
        const u64 list = load<u64>(lists + 0x10 + 8 * static_cast<s64>(load<s32>(lists + 0x20)));
        return load<u64>(list + 0x40);
    };
    const u64 shape = current_shape();
    reinterpret_cast<void(GUEST_ABI*)(u64, s32)>(load<u64>(load<u64>(shape) + 0x28))(shape, -1);
    const u64 now = current_shape();
    if (load<s32>(now + 0x70) != 2) {
        store<s32>(now + 0x70, 2);
        store<u8>(now + 0x8, load<u8>(now + 0x8) | 0x80);
    }
    const Vec colour = {0.25f, 0.0f, 0.25f, 0.4f};
    for (const auto& [at, flag] : {std::pair{0x20, 0x10}, std::pair{0x30, 0x20}}) {
        if (_mm_movemask_ps(_mm_cmpeq_ps(load<Vec>(shape + at), colour)) == 0xf) continue;
        store<Vec>(shape + at, colour);
        store<u8>(shape + 0x8, load<u8>(shape + 0x8) | flag);
    }
    if (load<s32>(shape + 0x1c) != 1) {
        store<s32>(shape + 0x1c, 1);
        store<u8>(shape + 0x8, load<u8>(shape + 0x8) | 0x8);
    }
    if (load<s32>(shape + 0x18) != 2) {
        store<s32>(shape + 0x18, 2);
        store<u8>(shape + 0x8, load<u8>(shape + 0x8) | 0x4);
    }
    const Vec rows[3] = {_mm_setr_ps(0.0f, 0.0f, 0.0f, center[0]) + _mm_setr_ps(radius, 0.0f, 0.0f, 0.0f),
                         _mm_setr_ps(0.0f, 0.0f, 0.0f, center[1]) + _mm_setr_ps(0.0f, radius, 0.0f, 0.0f),
                         _mm_setr_ps(0.0f, 0.0f, 0.0f, center[2]) + _mm_setr_ps(0.0f, 0.0f, radius, 0.0f)};
    reinterpret_cast<void(GUEST_ABI*)(u64, const Vec*)>(game_address(kDrawShape))(lists, rows);
}

// With keep-out on (+0x2d0) the camera stays out of the keep-out spheres.
// Working on a copy of its basis and position (+0x2e0), each sphere it is in
// pushes it back along its view to the sphere's surface, pulled in from walls
// like the camera itself (P19). With +0x2d1 every sphere that counts is drawn.
void keep_out_of_spheres(FollowCam& cam, void* world) {
    if (!cam.keep_out) return;
    cam.work_basis_x = cam.basis_x;
    cam.work_basis_y = cam.basis_y;
    cam.work_basis_z = cam.basis_z;
    cam.work_position = cam.position;
    const auto* sphere = reinterpret_cast<const KeepOutSphere*>(load<u64>(load<u64>(game_address(kKeepOutSpheres)) + 0x8));
    for (; sphere; sphere = sphere->next) {
        if (!(sphere->flags & 2)) continue;
        const Vec center = sphere->center;
        const float radius = static_cast<float>(sphere->radius);
        const Vec position = cam.work_position;
        const Vec to_center = center - position;
        const float distance = length3(to_center)[0];
        if (radius > distance) {
            const Vec forward = normalize4(xyz(cam.work_basis_z));
            const Vec along = to_center * forward;
            const float ahead = (along[0] + along[1]) + along[2];
            const float back = sqrt_ss(radius * radius - (distance * distance - ahead * ahead)) - ahead;
            cam.work_position = pulled_in(cam, world, position - forward * _mm_set1_ps(back));
        }
        if (cam.keep_out_debug_draw) draw_keep_out_sphere(center, radius);
    }
    cam.basis_x = cam.work_basis_x;
    cam.basis_y = cam.work_basis_y;
    cam.basis_z = cam.work_basis_z;
    cam.position = cam.work_position;
}

// ---- P23: the near clamp ----------------------------------------------------

// The camera comes in toward the orbit reference by +0x190 of how far it is
// past 1e-5 from it, at most +0x194.
void pull_toward_reference(FollowCam& cam) {
    const Vec out = cam.position - cam.orbit_reference;
    const Vec length = length3(out);
    if (!(length[0] > 1e-5f)) return;
    const float in = min_ss((length[0] - 1e-5f) * cam.near_rate, cam.near_max);
    cam.position = cam.orbit_reference + (_mm_set1_ps(1.0f) / length) * (out * _mm_set1_ps(length[0] - in));
}

// The step, once the update has taken the mouse's counts. At a converted
// frame rate `frames` is this frame's length in frames of the rate the
// tuning's weights are for (60 FPS++, 30 Original) and `frames30` in 30 fps
// frames, for the LockCamParam blend, which neither tuning changes; both 0
// for the game's own arithmetic.
void follow_cam_step(u8* cam_bytes, u8* chr, void* world, float dt, float frames, float frames30) {
    g_frames.fetch_add(1, std::memory_order_relaxed);
    FollowCam& cam = *reinterpret_cast<FollowCam*>(cam_bytes);
    if (const u8* row = lock_cam_row(camera_param_id(cam))) ease_toward_lock_cam_param(cam, row, frames30);  // P1, P2
    ease_distance_and_speeds(cam);                                                                         // P3
    if (!chr) return;                                                                                      // P4
    const auto* target = static_cast<const u8*>(camera_target::get(chr));
    const u8* floor_chr = chr;
    const u8* model_chr = chr;
    if (target) {
        const u64 owner = load<u64>(reinterpret_cast<u64>(target) + 0x58);
        if (owner && load<u64>(owner + 0x8)) floor_chr = target;
        if (load<u64>(reinterpret_cast<u64>(target) + 0x3b0)) model_chr = target;
    }
    Step step{};
    step.dt = dt;
    step.dt4 = _mm_set1_ps(dt);
    step.frames = frames;
    step.world = world;
    step.hold = held_turns();
    ride_moving_floor(cam, floor_chr, step);                                 // P5
    take_wanted_distance(cam, step);                                         // P6
    step.stick = read_stick(cam, chr);                                       // P7
    step.stick_turns = stick_turns_camera(cam, step.stick, dt);
    place_origins(cam, follow_character_frame(cam, model_chr, step), step);  // P8, P9
    chase_wanted_point(cam, step);                                           // P10
    const bool auto_turning = choose_angles(cam, step);                      // P11
    run_return_timers(cam, step);                                            // P12
    steer_by_stick(cam, step, auto_turning);                                 // P13
    return_pitch(cam, step.frames);                                        // P14
    place_wanted(cam, step);                                                 // P15
    cast_back_from_walls(cam, world);                                        // P16
    steer_around_walls(cam, step, world);                                    // P17
    place_camera(cam, step, chase_point(cam, step.frames));                // P18
    collide_camera(cam, world);                                              // P19
    clear_requests(cam);                                                     // P20
    guard_against_nan(cam, step.saved);                                      // P21
    keep_out_of_spheres(cam, world);                                         // P22
    pull_toward_reference(cam);                                              // P23
}

}  // namespace

// Every update first takes the mouse's counts, as the mouse camera's hook on
// the game's prologue does where ours is not in place.
DECOMP_LEAF void follow_cam_update(u8* cam_bytes, u8* chr, void* world, float dt) {
    mouse_camera::begin_update();
    const float frame = frame_rate_step::g_frame_seconds.load(std::memory_order_relaxed);
    const int tuned = frame_rate_step::g_camera_tuned_fps.load(std::memory_order_relaxed);
    const bool converted = frame > 0.0f && tuned > 0;
    follow_cam_step(cam_bytes, chr, world, dt, converted ? frame * static_cast<float>(tuned) : 0.0f,
                    converted ? frame * 30.0f : 0.0f);
}

// ---- Placing it -------------------------------------------------------------

namespace {

// The body ours mirrors, hashed (FNV-1a) with the run-time sites masked; each
// site must hold the game's bytes or the known patch's - the three below, and
// the four stores holding auto-rotation removes (mouse_camera::kHoldSites),
// which ours mirrors as nops whoever wrote them.
constexpr u64 kBodyHash = 0xc106b99ce56d67baull;
constexpr struct Site {
    u64 at;
    unsigned n;
    u32 game, patched;
} kSites[] = {
    {0x183af4e, 4, 0xc6460f48u, 0x90f08948u},  // fov-uncap: cmovbe rax, rsi / mov rax, rsi; nop
    {0x183dc97, 2, 0x0657u, 0x05d6u},          // 60 fps: jmp 0x183e2f2 / jmp 0x183e271
    {0x183dcfe, 1, 0x01u, 0x00u},              // 60 fps: mov byte [r13 + 0x28c], 1 / 0
};

// The constants the body reads, which ours writes as literals, hashed the
// same way - all but the two 60 fps ones, which ours reads where they are.
// A patch to any of them leaves the game's code in place.
constexpr u64 kConstantsHash = 0xdf647bba5cc4d74bull;
constexpr struct Span {
    u64 at, n;
} kConstants[] = {{0x4cf24a0, 0x200}, {0x4d24e50, 0x10}, {0x4d25e58, 0x44}, {0x2fd5280, 0x10}, {0x4b38000, 0x10}};
constexpr u64 kReadAtRunTime[] = {kDistanceEase, kChaseRateEase};

u64 fnv1a(u64 h, u8 b) { return (h ^ b) * 0x100000001b3ull; }

bool masked(u64 a) {
    for (const Site& s : kSites)
        if (a >= s.at && a < s.at + s.n) return true;
    for (const mouse_camera::HoldSite& s : mouse_camera::kHoldSites)
        if (a >= s.bn && a < s.bn + sizeof s.game) return true;
    return false;
}

}  // namespace

// The body check, for decomp_add and the test.
bool follow_camera_body_ok(const std::uint8_t* entry) {
    u64 h = 0xcbf29ce484222325ull;
    for (u64 a = kUpdate; a < kUpdateEnd; ++a) h = fnv1a(h, masked(a) ? 0 : entry[a - kUpdate]);
    if (h != kBodyHash) return false;
    for (const Site& s : kSites) {
        u32 v = 0;
        std::memcpy(&v, entry + (s.at - kUpdate), s.n);
        if (v != s.game && v != s.patched) return false;
    }
    for (const mouse_camera::HoldSite& s : mouse_camera::kHoldSites) {
        const u8* at = entry + (s.bn - kUpdate);
        if (std::memcmp(at, s.game, sizeof s.game) != 0 && std::memcmp(at, mouse_camera::kNop9, sizeof s.game) != 0)
            return false;
    }
    h = 0xcbf29ce484222325ull;
    for (const Span& span : kConstants) {
        for (u64 a = span.at; a < span.at + span.n; ++a) {
            u8 b = load<u8>(game_address(a));
            for (const u64 at : kReadAtRunTime)
                if (a >= at && a < at + 4) b = 0;
            h = fnv1a(h, b);
        }
    }
    return h == kConstantsHash;
}

namespace {

// ---- Compare runs (BBHOST_DECOMP_COMPARE=1) ---------------------------------
//
// The SprjFlipper model: the game's step runs on the camera and keeps its
// result; ours runs on a copy of the camera as it was, with the
// GameStateMan overrides as they were (the game's run consumes them), and
// the two are compared - the whole object, every NaN the same. A frame with
// the debug draw on (+0x2d1) is not compared: it would draw twice.

void* g_update_game = nullptr;
DecompCompare g_counts;
std::atomic<u64> g_not_compared{0};

bool same_object(const u8* a, const u8* b, std::size_t n, std::size_t* where) {
    for (std::size_t k = 0; k < n; k += 4) {
        if (std::memcmp(a + k, b + k, 4) == 0) continue;
        float x, y;
        std::memcpy(&x, a + k, 4);
        std::memcpy(&y, b + k, 4);
        if (x != x && y != y) continue;
        *where = k;
        return false;
    }
    return true;
}

GUEST_ABI void update_compare(u8* cam, u8* chr, void* world, float dt) {
    const auto game = reinterpret_cast<void(GUEST_ABI*)(u8*, u8*, void*, float)>(g_update_game);
    // The mouse's counts once, for both: the game's update turns by them
    // through the mouse camera's hook at its free camera's store, ours in its
    // own step - and the game's holds auto-rotation when ours does.
    mouse_camera_compare_begin();
    if (cam[0x2d1]) {
        g_not_compared.fetch_add(1, std::memory_order_relaxed);
        game(cam, chr, world, dt);
        return;
    }
    alignas(16) u8 ours[0x350];
    std::memcpy(ours, cam, sizeof ours);
    const u64 gsm = load<u64>(game_address(0x5956678));
    u64 overrides = 0, after_game = 0, after_ours = 0;
    if (gsm) overrides = load<u64>(gsm + 0x14);
    game(cam, chr, world, dt);
    if (gsm) {
        after_game = load<u64>(gsm + 0x14);
        store<u64>(gsm + 0x14, overrides);
    }
    follow_cam_step(ours, chr, world, dt, 0.0f, 0.0f);  // the game's arithmetic: compared at any frame rate
    if (gsm) {
        after_ours = load<u64>(gsm + 0x14);
        store<u64>(gsm + 0x14, after_game);
    }
    g_counts.calls.fetch_add(1, std::memory_order_relaxed);
    std::size_t at = 0;
    if (!same_object(cam, ours, sizeof ours, &at) || after_game != after_ours) {
        if (g_counts.differ.fetch_add(1, std::memory_order_relaxed) < 8) {
            u32 x = 0, y = 0;
            std::memcpy(&x, cam + at, 4);
            std::memcpy(&y, ours + at, 4);
            host_log("decomp: the follow camera differs at +0x%zx: the game's %08x, ours %08x", at, x, y);
        }
    }
}

void report() {
    const u64 n = g_frames.load(), skipped = g_not_compared.load();
    host_log("decomp: the follow camera's update ran %llu frames as ours%s", static_cast<unsigned long long>(n),
             skipped ? " (and some with its debug draw, not compared)" : "");
}

}  // namespace

void decomp_follow_camera_add() {
    // push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp, 0x3d8
    static const std::uint8_t kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                          0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xd8, 0x03, 0x00, 0x00};
    decomp_add({"NS_SPRJ::ChrExFollowCam::Update", "camera", kUpdate, kEntry, sizeof kEntry,
                reinterpret_cast<void*>(&follow_cam_update), DecompKind::Leaf, &g_update_game,
                reinterpret_cast<void*>(&update_compare), &g_counts, &report, &follow_camera_body_ok});
}

