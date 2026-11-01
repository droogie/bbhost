#pragma once

// The sound thread, woken once a frame (with the frame clock on: engine/frame_clock.h).
//
// MagicOrchestra, the game's sound system over FMOD, runs a thread of its own
// (its loop, sub_2c00e30). A pass updates FMOD's event system and takes the
// sound requests the game posted - the loop runs passes while its flag +0x359
// or one of its two request queues (+0x128, +0x230) holds work - and then
// sleeps MainThreadUpdateTime (+0x8d4: 33333 us, read from the sound
// system's config text at 0x4b33b40) less the pass's own time (the call to
// the usleep wrapper sub_248bac0 at 0x2c0137d). The pass's time is taken in
// milliseconds and the period is in microseconds, so the sleep is all but
// the whole period: a pass every 33.3 ms plus the pass, a little slower than
// the game's own 30 frames a second. Faster frames hand one pass the
// requests of two frames or more, and some sounds were lost that way: at 60
// with the game's 33333, some attack sounds played half as often as at 30
// (2026-09-30, counted by sample signature). The frame-rate tables write a
// period for the rate chosen at start (1/N s, plus the pass), which is a pass
// a frame only while the frames take 1/N s and the pass next to nothing.
//
// Placed, the sleep is ours: it waits until a frame has begun since the pass
// began (the frame-time manager says so, sound_wake_frame, as each frame's
// work ends), or for the game's own sleep if no frame comes first (a long
// frame, a loading screen's stall, a paused window) - so the passes follow
// the frames at any frame rate, one for each frame's requests, and the
// period stays the game's 33333. A pass that ends after the next frame has
// begun is followed by the next at once.
//
// BBHOST_FRAME_CLOCK=1 places it at a converted rate (frame_rate.cpp's
// kCountsHandedOver); BBHOST_SOUND_WAKE=0 leaves it out, with the rate's
// period at start as with the clock off. BBHOST_SOUND_CENSUS=1 (a test
// switch) counts the passes the same way without the wake - the game's own
// sleep, or the rate's - at any rate, 30 included.

#include <string>

struct ElfImage;

// The wake, on the sleep's call: false (and nothing changed) when the bytes
// there are not the loop's.
bool sound_wake_install(ElfImage* image);
// BBHOST_SOUND_CENSUS=1: the census alone, the sleep left as it is (any rate).
void sound_wake_census_install(ElfImage* image);
// Once a frame on the main thread, from the frame-time manager, as the
// frame's work ends: before the limiter's wait.
void sound_wake_frame();
// For the 300-flip report: the passes since the last call - how many frames
// had begun between consecutive passes' ends (0: a pass no frame woke, 1: a
// frame's requests, 2 or more: two frames' requests in one pass), how long
// the passes took and how many waits ran out, stamped with the monotonic
// clock's milliseconds (the AT9 census's, hle/ajm.cpp); empty when neither
// is placed.
std::string sound_wake_report();
