#pragma once

// The uncapped mode's frame limit (engine/frame_rate.cpp), as pure functions
// the frame-rate code and tests/uncapped_limit_test.cpp share.

#include <algorithm>

namespace uncapped_limit {

// The limit in the world with V-Sync off: its code immediates and frame
// counts hold 120's values (engine/frame_rate.h), so above this they run
// more than twice as fast as real time.
constexpr int kWorldMaxFps = 240;

// BBHOST_UNCAPPED_FPS=N as the limit it sets: 30..kWorldMaxFps, 0 unset.
inline int forced_fps(int env_value) { return env_value <= 0 ? 0 : std::clamp(env_value, 30, kWorldMaxFps); }

// Frames a second at most while a limit applies: `forced` when set; else the
// display's refresh rate with V-Sync on (when it says one, 30 or more); else
// kWorldMaxFps.
inline float limit_fps(int forced, bool vsync, float refresh_hz) {
    if (forced) return static_cast<float>(forced);
    float limit = static_cast<float>(kWorldMaxFps);
    if (vsync && refresh_hz >= 30.0f) limit = std::min(limit, refresh_hz);
    return limit;
}

// No limit at all: the game runs uncapped (`game_fps` 0) with V-Sync off and
// no BBHOST_UNCAPPED_FPS, no world loaded, no loading screen up (that has its
// own rule, engine/loading.h) and no character creation (its model preview
// turns a fixed step a frame) - the title and the main menu, where nothing
// counts frames.
inline bool unlimited(int game_fps, int forced, bool vsync, bool in_world, bool loading_screen, bool character_creation) {
    return game_fps == 0 && !forced && !vsync && !in_world && !loading_screen && !character_creation;
}

}  // namespace uncapped_limit
