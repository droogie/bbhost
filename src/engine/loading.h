#pragma once

#include <cstdint>

// The game's loading screens, read once a frame on the main loop (the
// frame-time manager's hook): the first in-game frame is logged ("world:")
// and so is each load's length ("loading:") - what soak summaries start
// their window at. Reads only.
//
// A load after a death or lamp travel is held to at least 12 s by the game
// (the in-game step's +0x278 countdown); patches/quick-reentry.toml takes
// that minimum away (loading.quick_reentry).
//
// A load is the NowLoading request (byte 0x596286b, set by sub_1b6d380,
// cleared by sub_1b6d3c0) without an in-game frame: the in-game update
// (sub_1d3ac10) sets CSNowLoadingHelper (*0x593e8b8) +0x50 every frame it
// runs, and a per-frame callback (0x1dc8450) moves it to +0x51.
void loading_bind(std::uint64_t slide);
void loading_tick(std::uint64_t flip);
// The last tick's answer: a loading screen is up.
bool loading_screen_up();
// The last tick's answer: the in-game update ran (CSNowLoadingHelper +0x50 or
// +0x51), so a world is loaded and simulated. False at the title, the main
// menu and a loading screen.
bool loading_in_world();
// A loading screen is up, the game runs above 30 (60, 90 or uncapped: its
// steps follow the frame time through the FPS++ cave) and no Remo cutscene is
// requested or running: the frame-time manager waits for nothing
// (engine/frame_rate.cpp) and flips complete at once
// (hle_video_set_loading_uncapped), so the loaders, which step once a frame,
// run as fast as the machine allows - the community's fast-loading patch,
// done natively. BBHOST_FAST_LOADING=0 turns it off.
bool loading_fast_now();
