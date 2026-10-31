#pragma once

// The frame clock: one step a frame, the time the frame-rate mechanisms read
// (engine/frame_rate.cpp). Its each_frame publishes it once a frame, before
// the game's frame runs, at every converted rate: the game's task step - the
// measured frame time clamped to [target, 1/30], the game's own rule (a long
// frame slows the game, it never jumps), times a plugin's time scale.
//
// BBHOST_FRAME_CLOCK=1 turns it on (off by default while it is being proven):
// what the frame-rate code would otherwise convert once, for the rate chosen
// at start, it converts from this clock every frame instead, so a frame of any
// length - a rate the PC does not hold, a variable refresh rate, a dip - is
// converted as the frame it is. Off, nothing publishes or reads it, and the
// frame-rate code works as it does without it. Each reader says so where it
// reads it.
//
// At the game's own 30 nothing publishes, and the clock holds the game's 1/30:
// whatever converts a value by it gets the game's own value back there.
//
// Plain atomics, read relaxed and defined here: decomp leaves read it on the
// game's threads (no host thread-locals, no libm), and the eboot kit's tests
// link without frame_rate.cpp.

#include <atomic>
#include <bit>
#include <cstdint>

namespace frame_clock {

// 1/30 as the game writes it (0x2418e36's step-info, the ragdoll's +0x60).
inline constexpr std::uint32_t kThirtiethBits = 0x3d088889u;
inline constexpr float kThirtieth = std::bit_cast<float>(kThirtiethBits);

// A step in 30 fps frames: 30 x step in double, rounded once - and exactly 1
// for the game's own 1/30 (30 x 0x3d088889 rounds to 1.0f anyway; said here
// rather than left to the rounding), so a value tuned for a 30 fps frame and
// multiplied by it is the game's own at 30.
[[gnu::always_inline]] inline float frames30_of(float dt) {
    return std::bit_cast<std::uint32_t>(dt) == kThirtiethBits ? 1.0f : static_cast<float>(30.0 * static_cast<double>(dt));
}

// BBHOST_FRAME_CLOCK=1, as frame_rate_install read it (before the decomps are
// added, which ask it).
inline std::atomic<bool> g_on{false};
// This frame's step in game seconds.
inline std::atomic<float> g_dt{kThirtieth};
// The same in 30 fps frames.
inline std::atomic<float> g_frames30{1.0f};
// Frames published: 0 while the game runs its own 30.
inline std::atomic<std::uint64_t> g_seq{0};

[[gnu::always_inline]] inline bool on() { return g_on.load(std::memory_order_relaxed); }
inline void set_on(bool on) { g_on.store(on, std::memory_order_relaxed); }

// Once a frame, from the frame-time manager (engine/frame_rate.cpp), on the
// main thread before the frame's tasks run.
inline void publish(float dt) {
    g_dt.store(dt, std::memory_order_relaxed);
    g_frames30.store(frames30_of(dt), std::memory_order_relaxed);
    g_seq.fetch_add(1, std::memory_order_relaxed);
}

[[gnu::always_inline]] inline float dt() { return g_dt.load(std::memory_order_relaxed); }
[[gnu::always_inline]] inline float frames30() { return g_frames30.load(std::memory_order_relaxed); }

}  // namespace frame_clock
