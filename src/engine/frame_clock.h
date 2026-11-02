#pragma once

// The frame clock: one step a frame, the time the frame-rate mechanisms read
// (engine/frame_rate.cpp). Its each_frame publishes it once a frame, before
// the game's frame runs, at every converted rate: the game's task step - the
// measured frame time clamped to [target, 1/30], the game's own rule (a long
// frame slows the game, it never jumps), times a plugin's time scale.
//
// The Native frame clock setting turns it on (F10, the setup window;
// bbhost.toml's [video] frame_clock its default; BBHOST_FRAME_CLOCK=0|1 over
// both - core/config.h's config_frame_clock), experimental and off by default:
// what the frame-rate code would otherwise convert once, for the rate chosen
// at start, it converts from this clock every frame instead, so a frame of any
// length - a rate the PC does not hold, a variable refresh rate, a dip - is
// converted as the frame it is. Off, nothing publishes or reads it, and the
// frame-rate code works as it does without it. Each reader says so where it
// reads it.
//
// At the game's own 30 nothing publishes, and the clock holds the game's 1/30:
// whatever converts a value by it gets the game's own value back there. With
// the switch off it is published at a converted rate all the same for what
// counts game time by default (publishing(), below).
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

// On, as frame_rate_install decided it (config_frame_clock: the setting or
// BBHOST_FRAME_CLOCK) before the decomps are added, which ask it.
inline std::atomic<bool> g_on{false};
// Whether the frame-time manager publishes the clock every frame although the
// switch is off: at a converted rate, for what counts game time by default -
// the replay ghosts and the local player's character sync
// (decomp/chr/replay_manipulator.cpp, decomp/chr/chr_sync.cpp; their own
// switches, BBHOST_REPLAY_STEP=0 and BBHOST_CHR_SYNC_STEP=0, turn them off).
// Publishing alone converts nothing: everything else that reads the clock
// asks on(). Set by frame_rate_install, before the decomps are added.
inline std::atomic<bool> g_publishing{false};
// This frame's step in game seconds.
inline std::atomic<float> g_dt{kThirtieth};
// The same in 30 fps frames.
inline std::atomic<float> g_frames30{1.0f};
// Frames published: 0 while the game runs its own 30.
inline std::atomic<std::uint64_t> g_seq{0};
// The 30 fps frames of game time this frame begins: what counts the game's
// own frames - a countdown, a cadence - takes a step for each, so it counts
// 30 of them a game second at any frame length, each in the frame where the
// game at 30 would take it (the one in which a 30 fps frame's time starts).
// The frames' steps in 30 fps frames are summed: a frame begins a 30 fps
// frame when a whole number lies in [the sum before it, the sum after it) -
// the first frame and every 1/30 one, every other frame of 1/60 (the first,
// the third...), one in 4.8 of 1/144. At most 1 a frame, since a step is at
// most 1/30; 1 while the game runs its own 30.
inline std::atomic<int> g_ticks30{1};
// How far the sum is past the last whole number begun, less 1 (in (-1, 0]):
// the publisher's (the main thread's) alone. 0 before the first frame.
inline double g_tick_carry = 0.0;

[[gnu::always_inline]] inline bool on() { return g_on.load(std::memory_order_relaxed); }
inline void set_on(bool on) { g_on.store(on, std::memory_order_relaxed); }
[[gnu::always_inline]] inline bool publishing() { return g_publishing.load(std::memory_order_relaxed); }
inline void set_publishing(bool on) { g_publishing.store(on, std::memory_order_relaxed); }

// Once a frame, from the frame-time manager (engine/frame_rate.cpp), on the
// main thread before the frame's tasks run.
inline void publish(float dt) {
    const float frames = frames30_of(dt);
    g_dt.store(dt, std::memory_order_relaxed);
    g_frames30.store(frames, std::memory_order_relaxed);
    const double sum = g_tick_carry + static_cast<double>(frames);
    const int ticks = sum > 0.0 ? 1 : 0;
    g_tick_carry = sum - ticks;
    g_ticks30.store(ticks, std::memory_order_relaxed);
    g_seq.fetch_add(1, std::memory_order_relaxed);
}

[[gnu::always_inline]] inline float dt() { return g_dt.load(std::memory_order_relaxed); }
[[gnu::always_inline]] inline float frames30() { return g_frames30.load(std::memory_order_relaxed); }
[[gnu::always_inline]] inline int ticks30() { return g_ticks30.load(std::memory_order_relaxed); }

}  // namespace frame_clock
