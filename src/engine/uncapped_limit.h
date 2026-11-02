#pragma once

// The Uncapped frame rate (engine/frame_rate.cpp), as pure functions the
// frame-rate code and tests/uncapped_limit_test.cpp share. Uncapped is the
// display's refresh rate, read when the game starts and run as a fixed rate:
// the target, the steps, the code immediates and the frame counts all at that
// rate, as if it were one of the Frame rate option's own - 1/180 s each on a
// 180 Hz display. What counts frames cannot follow the frame's time, so a
// rate that moved with the frame time (the old Uncapped, 120's counts at any
// rate up to 240) ran those too fast or too slow.

#include <algorithm>

namespace uncapped_limit {

// The rates a run converts to: the game's own 30 up to the option's fastest.
constexpr int kMinFps = 30;
constexpr int kMaxFps = 960;
// Uncapped's rate when no display says one (a headless run).
constexpr int kNoDisplayFps = 60;

// BBHOST_UNCAPPED_FPS=N as the rate it sets: 30..960, 0 unset.
inline int forced_fps(int env_value) { return env_value <= 0 ? 0 : std::clamp(env_value, kMinFps, kMaxFps); }

// Uncapped's rate: `forced` when set; else the display's refresh rate to the
// nearest whole frame a second (59.94 Hz runs 60, 143.98 runs 144), within
// 30..960; else kNoDisplayFps.
inline int rate_fps(int forced, float refresh_hz) {
    if (forced) return forced;
    if (!(refresh_hz >= static_cast<float>(kMinFps) - 0.5f)) return kNoDisplayFps;  // also NaN
    return std::clamp(static_cast<int>(refresh_hz + 0.5f), kMinFps, kMaxFps);
}

}  // namespace uncapped_limit
