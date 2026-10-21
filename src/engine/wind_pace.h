#pragma once

// The wind's pace above 30 fps (engine/frame_rate.cpp), as pure functions the
// frame-rate code and tests/wind_pace_test.cpp share.
//
// sub_2713870 moves every swaying model's wind - its gusts' and point winds'
// times (the w lane of WindForceAndTime, which the grass and tree shaders sway
// by), the point winds' lives (a push's fade) and their 0.1 s ramps - by one
// step a frame, and with the GX device's +0x100 at 1, as the game sets it, that
// step is a fixed 1/60 (0x4d29e70, loaded at 0x2713d8d): at 30 the wind runs
// 1/60 s each 1/30 s of game time, half the game's pace. That is the wind the
// game was made and shown at. Above 30 the frame-time site there gave it the
// frame's step, a whole 1/60 s each 1/60 s at 60 - twice 30's pace (measured
// in the Hunter's Dream: slot ages 0.500 game s a game second at 30, 0.999 at
// 60 and 120; a push's fade 1.5 s at 30, 0.75 s above). Half the frame's step keeps 30's pace at
// every rate and still moves every frame.
//
// The same step weighed the force the function uploads (WindForceAndTime.xyz =
// V x step, PointWindForce.x = speed x step) until a site made it the authored
// 1/30 at 0x27149ea; at 30 the game weighs it by the same 1/60 as the time, so
// 1/60 is 30's bend at every rate.

#include <cstdint>

#ifdef WIND_PACE_MUTANTS
extern int g_wind_pace_mutant;
#define WIND_PACE_MUTANT(n) (g_wind_pace_mutant == (n))
#else
#define WIND_PACE_MUTANT(n) false
#endif

namespace wind_pace {

constexpr float kStepAtThirty = 1.0f / 60.0f;  // the game's wind step a frame at 30 (0x3c888889)
constexpr float kFrameAtThirty = 1.0f / 30.0f;

// The wind's step for a frame whose task step is `step` (at most 1/30 s):
// 1/60 s of wind for each 1/30 s of game time, as at 30.
inline float step_for(float step) {
    if (WIND_PACE_MUTANT(1)) return step;           // the frame's step: twice 30's pace
    if (WIND_PACE_MUTANT(2)) return kStepAtThirty;  // the game's fixed 1/60 a frame: the frame rate's pace
    return step * (kStepAtThirty / kFrameAtThirty);
}

// The 32-bit displacement that points an instruction at `insn`, `len` bytes
// long with its rip displacement in its last 4 bytes, at `target` (Binary Ninja
// addresses); false when it does not fit.
inline bool rel32_to(std::uint64_t insn, unsigned len, std::uint64_t target, std::int32_t* disp) {
    const std::int64_t d = static_cast<std::int64_t>(target) - static_cast<std::int64_t>(insn + (WIND_PACE_MUTANT(3) ? 0 : len));
    if (d < INT32_MIN || d > INT32_MAX) return false;
    *disp = static_cast<std::int32_t>(d);
    return true;
}

}  // namespace wind_pace
