// The draw pool's duty (host/gxvk_lists.h resting): whether it works - commits
// the GX-native draws ahead of the command processor - or rests, the command
// processor drawing them in order with the constant buffers' homes as without
// the pool. The decision alone, a function of half-second windows of the main
// loop's frame, so tests/gxvk_rest_test.cpp drives it without the game.
//
// The pool works only where it shortens the frame. Its two threads beside the
// game's own make the game's code slower (on two Xeon Gold 6230 the main
// loop's work a frame was 0.4-0.5 ms longer at Central Yharnam and in the
// sewers under it, which ran 3.8% and 5.6% slower with the pool than without
// it, uncapped), so where the command processor drawing in order holds no
// frame back the pool's work is a cost:
//  - At a frame cap holding the frame (two windows in a row with the limiter
//    waiting 15% of the window or more) it rests. It works again when the
//    command processor, drawing everything, is busy 85% of a window at the cap,
//    or the limiter waits under 8% and the main loop is held back by the command
//    processor's backlog 20% of the time or more (a frame that misses its cap
//    for the main loop's own work or the GPU's gains nothing from the pool). A
//    rest that ended within 30 s makes the next one wait longer: 5 s, doubling
//    to 160 s; a longer one sets the wait back to 5 s, and so does one a
//    loading screen came into (a tour of the five census places at 60, a
//    lamp's travel between each two, woke the pool at each arrival 27 s into
//    its rest, and the doubled waits kept it working 40 s into the third
//    place).
//  - With no cap holding the frame (uncapped, or a frame slower than its cap)
//    it rests unless the command processor holds the main loop back. Resting,
//    the main loop held back 20% or more of the time - the median of the last
//    six windows - wakes it (the homes alone, without the pool: Old Yharnam 54%,
//    Cainhurst's courtyard 44%, where the pool gained 12% and 2.5%; Central
//    Yharnam 1.2%, the sewer pit 1.4%, where it lost 1.3% and 3.9%). The median,
//    not the mean: one second in twenty at Central Yharnam holds the game back
//    40% while the rest hold it back 0-8%. Working, the held-back share is the
//    pool's to hide, so a probe asks: a rest of five windows, 10 s after it
//    began to work and then twice as far each time it answers yes (to 4 minutes).
//    It wakes the pool again when the median of its last three windows holds
//    the game back 20% or more and the frame is slower than while the pool
//    worked (the median of the four windows before the probe) by 3% or more;
//    else the rest goes on. Its first two windows are the switch's: the command
//    processor's caches are cold for the lookups the pool made, and the sewer
//    pit's first seconds of a rest held the game back 20-40% before it settled
//    to 0-3%. The frame rate decides what the held-back share alone cannot: at
//    the sewer pit a second held back 36% still ran 110 fps, as fast as with
//    the pool (the game's render thread waited, not its main loop), and under
//    Windows on an RTX 3070 Ti Laptop GPU the game was held back 73-87% at
//    Cainhurst's courtyard, Central Yharnam and the sewer pit with the command
//    processor busy 52-59% and the frame as fast without the pool, while at Old
//    Yharnam, the Nightmare of Mensis and Cainhurst's courtyard on the Xeon a
//    probe ran 7-23% slower than the windows before it. So the rest a probe
//    began is woken by the held-back share only with the frame 10% slower than
//    the probe found it too (the median of the same six windows: the place grew
//    heavier), and the pool works again to ask anew - five windows, then a
//    probe - once the probe's interval has passed, twice as far each time a
//    probe says no (the held-back share alone woke the pool within 3 s of each
//    probe there). The held-back share waking a rest leaves the interval as it
//    is, not twice as far: after a loading screen (which sets it back to 10 s)
//    the place is one no probe has judged - in the sewers under Central
//    Yharnam, reached from the lamp's spot the command processor held back, the
//    doubled interval left the pool working 20 s into a place where it cost 4%.
//  - A GPU-bound frame (the GPU busy 90% of two windows in a row) rests it: the
//    pool cannot shorten it, and on an APU its CPU time is the GPU's power. The
//    held-back share ends that rest only with the GPU busy under 80% (the
//    median of the same six windows), and the command processor's share at a
//    cap only with the GPU under 80% in its window: on a Steam Deck at Old
//    Yharnam, 88-97% from one window to the next, the pool rested and woke four
//    times in 20 s uncapped and five at 60.
//  - A loading screen neither wakes it nor counts: the windows around one are
//    not the place's, and the probes start again at 10 s after it.
#pragma once

#include <cstdint>

namespace gxvk_rest {

constexpr double kRestSlack = 0.15;  // the limiter's wait, as a share of a window, that counts toward a rest at a cap
constexpr double kWakeSlack = 0.08;  // under it the cap no longer holds the frame
constexpr double kWakeCp = 0.85;     // the command processor busy this share of a window ends a rest
constexpr double kWorkHeld = 0.20;   // the main loop held back by the command processor this share: the pool shortens the frame
constexpr double kSlower = 0.03;     // a probe's frame rate this much under the working windows': the pool shortened the frame
constexpr double kHeavier = 0.10;    // since a probe said no: the frame this much slower than at the probe, the place grew heavier
constexpr double kGpuBound = 0.90;   // the GPU busy this share of a window: the frame is the GPU's
constexpr double kGpuFree = 0.80;    // a rest the GPU began ends only with the GPU busy under this (the median of six windows)
constexpr int kQuietWindows = 2;     // windows in a row with the cap holding the frame, or GPU-bound
constexpr int kHeldWindows = 6;      // resting: the held-back share is the median of this many windows
constexpr int kProbeWindows = 5;     // a probe's rest; its first kProbeSkip windows are the switch's, the rest are counted
constexpr int kProbeSkip = 2;
constexpr int kProbeCounted = kProbeWindows - kProbeSkip;
constexpr int kWorkWindows = 4;      // working: the frame rate's reference for a probe is the median of this many windows
constexpr int kRecheckWindows = kWorkWindows + 1;  // a rest a probe began, asked anew: the pool works this many windows (the first the switch's), then a probe
constexpr std::uint64_t kBackoffFirstNs = 5'000'000'000ull, kBackoffMostNs = 160'000'000'000ull, kShortRestNs = 30'000'000'000ull;
constexpr std::uint64_t kProbeFirstNs = 10'000'000'000ull, kProbeMostNs = 240'000'000'000ull;

// Why the rest under way began (the lines say).
enum Why : int { kWhyNone, kWhyCap, kWhyGpu, kWhyProbe, kWhyHeld };

struct State {
    bool resting = false;
    Why why = kWhyNone;
    int quiet = 0;                 // windows in a row with the cap holding the frame (working)
    int gpu_bound = 0;             // windows in a row GPU-bound
    int probe_left = 0;            // resting for a probe: its windows still to come
    int probe_counted = 0;         // ... and those counted, with their held-back shares and frame rates
    double probe_held[kProbeCounted] = {}, probe_fps[kProbeCounted] = {};
    double work_fps[kWorkWindows] = {};  // working: the last windows' frame rates (a probe's reference)
    int work_n = 0;
    double probe_ref = 0;            // the reference a probe under way compares with (0: none)
    double held[kHeldWindows] = {};  // resting: the last windows' held-back shares ...
    double rest_fps[kHeldWindows] = {};  // ... frame rates
    double rest_gpu[kHeldWindows] = {};  // ... and the GPU's busy shares
    int held_n = 0;
    double rest_ref = 0;             // resting since a probe said no: the frame rate the probe found (0: none - the held-back share alone)
    std::uint64_t recheck_at = 0;    // ... and when the pool works again to ask anew (0: never)
    int recheck_left = 0;            // working to ask anew: the windows before its probe
    std::uint64_t rest_at = 0;       // when the rest under way began
    std::uint64_t allowed_at = 0;    // no rest at a cap begins before (the backoff)
    std::uint64_t backoff_ns = kBackoffFirstNs;
    bool travelled = false;          // a loading screen since the rest under way began: the backoff starts over at its end
    std::uint64_t probe_at = 0;      // working with no cap holding the frame: the next probe (0: not set yet)
    std::uint64_t probe_ns = kProbeFirstNs;
};

// A window: the limiter's wait, the command processor's busy time, the main
// loop's wait for the command processor's backlog and the GPU's busy time as
// shares of it (gpu < 0: not known), its frames a second (0: not known), the cap
// the limiter held the frame to (0: none - a loading screen, or no limiter), and
// whether a loading screen was up.
struct Window {
    double slack = 0, cp = 0, held = 0, gpu = -1, fps = 0;
    float cap_fps = 0;
    bool loading = false;
};

enum Step { kStay, kRest, kWake };

namespace detail {
// The median of n values (n at most kHeldWindows): one burst of a second moves it no more than the windows around it.
inline double median(const double* v, int n) {
    double a[kHeldWindows > kWorkWindows ? kHeldWindows : kWorkWindows];
    for (int k = 0; k < n; ++k) a[k] = v[k];
    for (int i = 1; i < n; ++i)  // (an insertion sort: six values)
        for (int j = i; j > 0 && a[j - 1] > a[j]; --j) {
            const double t = a[j];
            a[j] = a[j - 1];
            a[j - 1] = t;
        }
    if (!n) return 0;
    return n % 2 ? a[n / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
}
inline Step begin_rest(State& s, Why why, std::uint64_t now) {
    s.resting = true;
    s.why = why;
    s.rest_at = now;
    s.travelled = false;
    s.held_n = 0;
    s.probe_left = why == kWhyProbe ? kProbeWindows : 0;
    s.probe_counted = 0;
    s.probe_ref = why == kWhyProbe && s.work_n == kWorkWindows ? median(s.work_fps, s.work_n) : 0;
    s.rest_ref = 0;
    s.recheck_at = 0;
    s.recheck_left = 0;
    return kRest;
}
// How a rest ended: a probe found the pool shortens the frame, the held-back share woke it, or a cap no longer holds.
enum By { kByProbe, kByHeld, kByCap };
inline Step wake(State& s, std::uint64_t now, By by) {
    s.resting = false;
    s.quiet = 0;
    s.probe_left = 0;
    if (by == kByProbe) {
        // The pool shortens the frame here: the next probe twice as far.
        s.probe_ns = s.probe_ns * 2 < kProbeMostNs ? s.probe_ns * 2 : kProbeMostNs;
    } else if (by == kByHeld) {
        // Held back at a place no probe has judged since (a loading screen's, which set the interval back to 10 s, or a
        // view grown heavier): the interval as it is - not twice as far.
    } else {
        // A rest at the cap that ended: the backoff (a short one makes the next wait longer - but not across a loading
        // screen: the frame missing its cap in the next place's first seconds is no flapping).
        const std::uint64_t rested = now - s.rest_at;
        s.backoff_ns = rested < kShortRestNs && !s.travelled ? (s.backoff_ns * 2 < kBackoffMostNs ? s.backoff_ns * 2 : kBackoffMostNs)
                                                             : kBackoffFirstNs;
        s.allowed_at = now + s.backoff_ns;
    }
    s.probe_at = now + s.probe_ns;
    s.why = kWhyNone;
    s.work_n = 0;
    s.rest_ref = 0;
    s.recheck_at = 0;
    return kWake;
}
// A rest a probe began, asked anew: the pool works kRecheckWindows windows, then a probe compares the frame with them.
inline Step recheck(State& s, std::uint64_t now) {
    s.resting = false;
    s.quiet = 0;
    s.probe_left = 0;
    s.why = kWhyNone;
    s.work_n = 0;
    s.rest_ref = 0;
    s.recheck_at = 0;
    s.recheck_left = kRecheckWindows;
    s.probe_at = now + s.probe_ns;  // (its own probe comes first)
    return kWake;
}
}  // namespace detail

// One window's verdict at `now` (ns): the state moved on, and whether a rest began or ended.
inline Step step(State& s, const Window& w, std::uint64_t now) {
    if (w.loading) {
        // Not the place's: nothing counts, a probe under way is dropped (the rest goes on as one), and the probes
        // start again at the first interval after it.
        s.quiet = s.gpu_bound = 0;
        s.held_n = 0;
        s.work_n = 0;
        s.probe_left = 0;
        if (s.resting && s.why == kWhyProbe) s.why = kWhyHeld;
        s.travelled = true;
        s.rest_ref = 0;  // (another place's)
        s.recheck_at = 0;
        s.recheck_left = 0;
        s.probe_ns = kProbeFirstNs;
        s.probe_at = 0;
        return kStay;
    }
    const bool capped = w.cap_fps > 0 && w.slack >= kRestSlack;  // the cap holds the frame
    const bool at_cap = w.cap_fps > 0 && w.slack >= kWakeSlack;  // ... or nearly
    s.gpu_bound = w.gpu >= kGpuBound ? s.gpu_bound + 1 : 0;
    const bool gpu_bound = s.gpu_bound >= kQuietWindows;
    if (!s.resting) {
        if (w.fps > 0) {  // the working windows' frame rates: a probe's reference
            if (s.work_n == kWorkWindows) {
                for (int k = 1; k < kWorkWindows; ++k) s.work_fps[k - 1] = s.work_fps[k];
                s.work_fps[kWorkWindows - 1] = w.fps;
            } else {
                s.work_fps[s.work_n++] = w.fps;
            }
        }
        s.quiet = capped ? s.quiet + 1 : 0;
        if (s.quiet >= kQuietWindows && now >= s.allowed_at) return detail::begin_rest(s, kWhyCap, now);
        if (gpu_bound) return detail::begin_rest(s, kWhyGpu, now);
        if (at_cap) {  // the cap's rule decides there
            s.recheck_left = 0;
            return kStay;
        }
        if (s.recheck_left) {  // asking anew: the probe after the windows the pool worked
            if (--s.recheck_left) return kStay;
            return detail::begin_rest(s, kWhyProbe, now);
        }
        if (!s.probe_at) s.probe_at = now + s.probe_ns;
        if (now >= s.probe_at) return detail::begin_rest(s, kWhyProbe, now);
        return kStay;
    }
    if (s.probe_left) {
        // A probe: its first windows are the switch's; the main loop held back over the rest says whether the pool
        // shortened the frame.
        if (s.probe_left <= kProbeCounted) {
            s.probe_held[s.probe_counted] = w.held;
            s.probe_fps[s.probe_counted++] = w.fps;
        }
        if (--s.probe_left) return kStay;
        const double held = detail::median(s.probe_held, s.probe_counted);
        const double fps = detail::median(s.probe_fps, s.probe_counted);
        // Slower than while the pool worked (no reference - a probe right after the start or a loading screen - or no
        // frame rate: the held-back share alone).
        const bool slower = s.probe_ref <= 0 || fps <= 0 || fps < s.probe_ref * (1.0 - kSlower);
        if (held >= kWorkHeld && slower && !capped && !gpu_bound) return detail::wake(s, now, detail::kByProbe);
        // The probe says no: the held-back share wakes this rest only with the frame 10% slower than the probe found it
        // too (the place grew heavier), and the pool works again to ask anew once the next interval has passed (twice as
        // far as this one).
        s.why = kWhyHeld;
        s.rest_ref = fps;
        s.probe_ns = s.probe_ns * 2 < kProbeMostNs ? s.probe_ns * 2 : kProbeMostNs;
        s.recheck_at = now + s.probe_ns;
        return kStay;
    }
    if (gpu_bound) return kStay;
    if (capped || at_cap) {
        // At the cap: the command processor, drawing everything, near a core's whole time is about to hold the frame.
        // (Uncapped its busy share says nothing: it keeps up with a fast main loop at 90% without holding it back.) Not
        // in a rest the GPU began while the GPU is still busy 80%: on a Steam Deck at 60 at Old Yharnam the command
        // processor, busy 90-95% there, woke it five times in 20 s, the GPU between 88 and 97%.
        s.held_n = 0;
        if (s.why == kWhyGpu && w.gpu >= kGpuFree) return kStay;
        return w.cp >= kWakeCp ? detail::wake(s, now, detail::kByCap) : kStay;
    }
    // No cap holding the frame: a rest a probe began asks anew when its interval has passed ...
    if (s.recheck_at && now >= s.recheck_at) return detail::recheck(s, now);
    // ... and the held-back share over the last windows wakes it.
    if (s.held_n == kHeldWindows) {
        for (int k = 1; k < kHeldWindows; ++k) {
            s.held[k - 1] = s.held[k];
            s.rest_fps[k - 1] = s.rest_fps[k];
            s.rest_gpu[k - 1] = s.rest_gpu[k];
        }
        s.held[kHeldWindows - 1] = w.held;
        s.rest_fps[kHeldWindows - 1] = w.fps;
        s.rest_gpu[kHeldWindows - 1] = w.gpu;
    } else {
        s.held[s.held_n] = w.held;
        s.rest_fps[s.held_n] = w.fps;
        s.rest_gpu[s.held_n++] = w.gpu;
    }
    if (s.held_n == kHeldWindows && detail::median(s.held, s.held_n) >= kWorkHeld) {
        // A rest the GPU began: only once the GPU has room (on a Steam Deck at Old Yharnam, busy 88-97% from one window to
        // the next, the pool rested and woke four times in 20 s).
        if (s.why == kWhyGpu && detail::median(s.rest_gpu, s.held_n) >= kGpuFree) return kStay;
        // Since a probe said no, only with the frame 10% slower than the probe found it as well (no frame rate: the
        // share alone).
        const double fps = detail::median(s.rest_fps, s.held_n);
        if (s.rest_ref > 0 && fps > 0 && fps >= s.rest_ref * (1.0 - kHeavier)) return kStay;
        // Woken by the frame at a cap that it no longer meets (the cap's backoff), or by the place (the next probe after
        // the interval as it is).
        return detail::wake(s, now, s.why == kWhyCap ? detail::kByCap : detail::kByHeld);
    }
    return kStay;
}

}  // namespace gxvk_rest
