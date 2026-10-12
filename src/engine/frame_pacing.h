#pragma once

#include <algorithm>

namespace frame_pacing {

// Keep pacing, simulation calibration and audio scheduling independent.
constexpr int kStepFloorFps = 1000;
constexpr int kSoundHz = 240;
// A positive target selects the converted game mode. Zero selects the
// original 30 Hz mode in SprjFlipper::Update, so use the existing source
// snapshot's effectively unlimited 50 us target instead.
constexpr float kNoLimitTarget = 1.0f / 20000.0f;

inline float uncapped_limit(int forced, bool vsync, float refresh_hz) {
    if (forced > 0) return static_cast<float>(std::clamp(forced, 30, 1000));
    return vsync && refresh_hz >= 30.0f ? refresh_hz : 0.0f;
}

inline float target_for_limit(float limit) {
    return limit > 0.0f ? 1.0f / limit : kNoLimitTarget;
}

inline float step_floor(float target) {
    return std::max(target, 1.0f / static_cast<float>(kStepFloorFps));
}

}  // namespace frame_pacing
