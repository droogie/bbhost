#pragma once

// The effects' step above 30 fps (engine/frame_rate.cpp), as a pure function
// the frame-rate code and tests/sfx_schedule_test.cpp share: each frame's time
// (the step the game takes, at most 1/30 s; 1/N under BBHOST_SFX_SIM_FPS) is
// added to a carry and taken out in whole steps of 1/rate s, rate 30 or 60.
// A frame takes at most the one 1/30 s the game steps a frame below 30 fps -
// one whole step at 30, two at 60 - and the rest stays in the carry. A frame
// that takes none is one bbhost put between whole steps: the game runs its
// effect update with a step of 0 there.

#include <cmath>

namespace sfx_schedule {

struct Frame {
    int whole = 0;        // whole steps this frame (0: a frame between whole steps)
    float seconds = 0.f;  // their time, the step the effect manager is handed
};

// The whole steps a frame takes at most: one 1/30 s.
constexpr int most(int rate) { return rate >= 60 ? 2 : 1; }

// `rate` is 30 or 60. A frame time above 1/30 s counts as 1/30 s (the game's
// own step there) and one that is not above 0 (or NaN) as none, so the carry
// stays under a step and a thirtieth.
inline Frame next(double& carry, double frame_seconds, int rate) {
    if (!(frame_seconds > 0.0)) frame_seconds = 0.0;
    if (frame_seconds > 1.0 / 30.0) frame_seconds = 1.0 / 30.0;
    carry += frame_seconds;
    // The 1e-6 keeps a carry a rounding short of a step from waiting a frame;
    // what it takes early leaves the carry that much below 0.
    int n = static_cast<int>(std::floor(carry * rate + 1e-6));
    if (n > most(rate)) n = most(rate);
    carry -= static_cast<double>(n) / rate;
    return {n, static_cast<float>(static_cast<double>(n) / rate)};
}

}  // namespace sfx_schedule
