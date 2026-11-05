// The pool's rest while a frame cap holds the frame (host/gxvk_lists.h
// resting): the decision alone, a function of the main loop's frame limiter
// over half-second windows, so tests/gxvk_rest_test.cpp drives it without the
// game. Two windows in a row with a cap set and the limiter waiting 15% of the
// window or more begin a rest; one with no cap, the limiter waiting under 8%,
// or the command processor - drawing everything in order meanwhile - busy 85%
// of it ends the rest. A rest that ended within 30 s makes the next one wait
// longer: 5 s at first, doubling to 160 s; a longer one sets the wait back to
// 5 s.
#pragma once

#include <cstdint>

namespace gxvk_rest {

constexpr double kRestSlack = 0.15;  // the limiter's wait, as a share of a window, that counts toward a rest
constexpr double kWakeSlack = 0.08;  // under it a rest ends
constexpr double kWakeCp = 0.85;     // the command processor busy this share of a window ends a rest too
constexpr int kQuietWindows = 2;     // windows in a row with the cap holding the frame
constexpr std::uint64_t kBackoffFirstNs = 5'000'000'000ull, kBackoffMostNs = 160'000'000'000ull, kShortRestNs = 30'000'000'000ull;

struct State {
    bool resting = false;
    int quiet = 0;                 // windows in a row with the cap holding the frame
    std::uint64_t rest_at = 0;     // when the rest under way began
    std::uint64_t allowed_at = 0;  // no rest begins before
    std::uint64_t backoff_ns = kBackoffFirstNs;
};

// A window: the limiter's wait and the command processor's busy time as shares
// of it, and the cap the limiter held the frame to (0: none - a loading screen,
// or no limiter).
struct Window {
    double slack = 0, cp = 0;
    float cap_fps = 0;
};

enum Step { kStay, kRest, kWake };

// One window's verdict at `now` (ns): the state moved on, and whether a rest began or ended.
inline Step step(State& s, const Window& w, std::uint64_t now) {
    if (!s.resting) {
        s.quiet = w.cap_fps > 0 && w.slack >= kRestSlack ? s.quiet + 1 : 0;
        if (s.quiet < kQuietWindows || now < s.allowed_at) return kStay;
        s.resting = true;
        s.rest_at = now;
        return kRest;
    }
    if (w.cap_fps > 0 && w.slack >= kWakeSlack && w.cp < kWakeCp) return kStay;
    s.resting = false;
    s.quiet = 0;
    const std::uint64_t rested = now - s.rest_at;
    s.backoff_ns = rested < kShortRestNs ? (s.backoff_ns * 2 < kBackoffMostNs ? s.backoff_ns * 2 : kBackoffMostNs) : kBackoffFirstNs;
    s.allowed_at = now + s.backoff_ns;
    return kWake;
}

}  // namespace gxvk_rest
