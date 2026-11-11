#pragma once

// Frame generation's pace (engine/frame_rate.cpp), as pure functions the
// frame-rate code and tests/fg_pace_test.cpp share.
//
// AMD's swapchain shows a generated frame half way between two real ones
// (host/framegen.h). While it generates, the real frames are held to half the
// display's refresh rate at most, whatever the cap - Display-Limited, the
// refresh rate itself, among them - so real and generated frames together
// never pass the display: 72 real and 144 shown at 144 Hz, 30 and 60 at 60 Hz
// (cap_s). Not on a loading screen, whose frames run uncapped and where
// nothing is generated, and not where no display says a rate of kLeastHz or
// more (a headless run, RDP's 32 Hz): there the cap paces alone.
//
// A cap the frames do not reach is replaced by a held pace 3% under what they
// manage (Pace), so the limiter - not the GPU's queue - starts each frame:
// checked every 32 frames; with the limiter no longer waiting (the scene got
// heavier, or the pace was tried too fast) it is set again from what the
// frames took, every 8th check it tries 2% faster, and a pace the limiter
// waits on for a tenth of each frame or more comes down at once to what the
// frames take. A held pace is never slower than the game's own 1/30 s and
// goes as soon as the cap is reached again, frames stop being generated or a
// loading screen comes up. A change of the upscaler's settings lets it go and
// skips the next 64 frames (the rebuild's hitch is not a pace to hold), and a
// frame that took longer than kStallNs (a load, a hitch) is not one to pace by.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace fg_pace {

// The game's own frame: the longest a held pace may be.
constexpr float kThirtieth = 0.0333333351f;
// The slowest display the half-refresh limit believes: below it the cap paces alone.
constexpr float kLeastHz = 48.0f;
// Frames a window of the held pace counts, and those skipped after a change
// of the upscaler's settings.
constexpr std::uint32_t kWindow = 32, kSkipAfterChange = 64;
// A frame longer than this is a stall, not one to pace by.
constexpr std::uint64_t kStallNs = 200000000ull;

// The real frames' interval the cap asks for: `rate_s`, the rate's (1/fps; 0
// at the game's own 30, which has no target of ours), and while frames are
// generated outside a loading screen at least two refreshes of a display of
// `hz` - which 0, NaN or anything under kLeastHz leaves to the rate.
inline float cap_s(float rate_s, bool generating, bool loading, float hz) {
    if (!(rate_s > 0.0f)) return 0.0f;  // also NaN
    if (!generating || loading || !(hz >= kLeastHz)) return rate_s;
    return std::max(rate_s, 2.0f / hz);
}

// The limiter's target: the cap's interval, or a held pace longer than it.
inline float target_s(float cap, float held_s) { return cap > 0.0f && held_s > 0.0f ? std::max(cap, held_s) : cap; }

// The held pace's state, the frame-rate code's one (g_fg_pace).
struct Pace {
    float held_s = 0.0f;  // the held real frame interval, 0 when not holding
    std::uint32_t n = 0;  // this window's frames
    std::uint32_t windows = 0;
    std::uint32_t skip = 0;  // frames still to let pass after a change
    double interval_s = 0.0, wait_s = 0.0;  // this window's sums
    std::uint64_t last_ns = 0;
    std::uint64_t settings_key = 0;  // the upscaler's settings the pace was learned under
};

// What one frame's note did, for the log.
struct Step {
    enum Kind : std::uint8_t {
        None,
        LetGo,          // pacing stopped (no generation, a loading screen, no cap): `before` let go
        LetGoSettings,  // the upscaler's settings changed: `before` let go
        Window,         // a window ended: `before` -> held_s, from interval_s and wait_s
    };
    Kind kind = None;
    bool window_start = false;      // this frame began a window (from window_from_ns)
    std::uint64_t window_from_ns = 0;
    float before = 0.0f;
    double interval_s = 0.0, wait_s = 0.0;  // the window's means
};

// The frame just paced, at `now_ns` with the limiter's wait `wait_ns` in it,
// under `cap` (cap_s). `pacing`: the pace is on, frames are generated, no
// loading screen is up and there is a cap. `settings_key`: the upscaler's
// settings now. `headroom`: the held interval over the one the frames manage.
inline Step note(Pace& p, std::uint64_t now_ns, std::uint64_t wait_ns, float cap, bool pacing, std::uint64_t settings_key,
                 double headroom) {
    Step s;
    const std::uint64_t was = p.last_ns;
    p.last_ns = now_ns;
    if (!pacing || !(cap > 0.0f)) {
        if (p.held_s > 0.0f) {
            s.kind = Step::LetGo;
            s.before = p.held_s;
        }
        p.held_s = 0.0f;
        p.n = 0;
        return s;
    }
    if (!was || now_ns < was || now_ns - was > kStallNs) return s;
    if (settings_key != p.settings_key) {
        if (p.settings_key && p.held_s > 0.0f) {
            s.kind = Step::LetGoSettings;
            s.before = p.held_s;
        }
        p.settings_key = settings_key;
        p.held_s = 0.0f;
        p.n = 0;
        p.skip = kSkipAfterChange;
    }
    if (p.skip) {
        --p.skip;
        return s;
    }
    if (p.n == 0) {
        p.interval_s = p.wait_s = 0.0;
        s.window_start = true;
        s.window_from_ns = was;
    }
    p.interval_s += static_cast<double>(now_ns - was) * 1e-9;
    p.wait_s += static_cast<double>(wait_ns) * 1e-9;
    if (++p.n < kWindow) return s;
    const double interval = p.interval_s / p.n, wait = p.wait_s / p.n;
    p.n = 0;
    ++p.windows;
    s.kind = Step::Window;
    s.before = p.held_s;
    s.interval_s = interval;
    s.wait_s = wait;
    if (p.held_s <= 0.0f) {
        // Not holding: only a cap the frames do not reach is replaced.
        if (interval > cap * 1.05) p.held_s = static_cast<float>(interval * headroom);
    } else if (wait < interval * 0.02) {
        p.held_s = static_cast<float>(interval * headroom);  // the limiter no longer holds them: as fast as they go, the pace follows
    } else if (wait > interval * 0.10) {
        // The limiter waits a tenth of each frame or more: the frames take
        // less than the pace - it comes down to what they take.
        p.held_s = static_cast<float>(std::max((interval - wait) * headroom, static_cast<double>(cap)));
    } else if (p.windows % 8 == 0) {
        p.held_s *= 0.98f;  // a little faster now and then: the scene may have got lighter
    }
    if (!(p.held_s >= 0.0f)) p.held_s = 0.0f;  // NaN
    p.held_s = std::min(p.held_s, kThirtieth);
    if (p.held_s > 0.0f && p.held_s <= cap * 1.01f) p.held_s = 0.0f;  // the cap is reached again: the cap paces
    return s;
}

}  // namespace fg_pace
