// The draw pool's duty (host/gxvk_lists.h resting): whether it works - commits
// the GX-native draws ahead of the command processor - or rests, the command
// processor drawing them in order with the constant buffers' homes as without
// the pool. The decision alone, a function of half-second windows of the main
// loop's frame, so tests/gxvk_rest_test.cpp drives it without the game.
//
// The pool works only where it shortens the frame. Its two threads beside the
// game's own can make the game's code slower (on two Xeon Gold 6230 the main
// loop's work a frame was 0.4-0.5 ms longer at Central Yharnam and in the
// sewers under it, which ran 3.8% and 5.6% slower with the pool than without
// it, uncapped), so where the command processor drawing in order holds no
// frame back the pool's work is a cost:
//  - At a frame cap holding the frame (two windows in a row with the limiter
//    waiting 15% of the window or more) it rests. It works again when the
//    command processor, drawing everything, is busy 85% with the frame meeting
//    its cap narrowly (the limiter waiting under 15%; the medians of the last
//    three windows at the cap, a window just off it in between not starting
//    the count again - with more the cap holds the frame with room: until
//    2026-11-10 any one window 85% busy woke the pool, and at Central Yharnam
//    at a 90 cap on two Xeon Gold 6230 with Windows' default acted out, the
//    command processor 71-75% busy at rest, one such window kept it working
//    through a measure window the cap held either way, 1.2 cores), or busy 92%
//    (the medians of the same three windows) however long the limiter waits: a
//    command processor that saturated drops frames resting - at Old Yharnam at
//    the same 90 cap, 92-97% busy at rest with the limiter waiting 15-21%,
//    resting left 4-15 frames of a 30 s window over 16.7 ms against 2 working,
//    and the 99th percentile frame 1.7 ms longer (15.4 against 13.7 ms), for
//    0.8 of a core and 4 W saved,
//    or the limiter waits under 8% and the main loop is held back by the command
//    processor's backlog 20% of the time or more (a frame that misses its cap
//    for the main loop's own work or the GPU's gains nothing from the pool); a
//    frame under 85% of its cap is held by no cap, however the limiter waits:
//    with uneven frames the fast ones still meet it while the frame rate sits
//    well under it. Such a window counts toward no rest at the cap, and resting
//    at a cap the pool works again once the median of the same three windows'
//    frame rates is under 85% of the cap - unless the GPU is busy 90% over the
//    last three windows (their mean: a Steam Deck's GPU moves between 88 and
//    97% from one window to the next), where the pool's work buys no frame and
//    on an APU its CPU time is the GPU's power. At Cainhurst's roof at a 120
//    cap on two Xeon Gold 6230
//    the place's event takes the frame to 80-97 fps for a few seconds with the
//    limiter still waiting 15-24% and the command processor 88-94% busy; with
//    the limiter's wait alone deciding, the pool rested through it for 7-9 s
//    in four runs of six once the drains were gone from the command processor
//    (v0.2.33's woke it after 2 s), and the window ran 6.6% slower. Six windows
//    with no cap at all - or the limiter waiting under 8% - end the rest the
//    same way, the median of their frame rates under 85% of the cap, to the
//    place's experiments (below: uncapped play after the title's menus met the
//    960 cap - v0.2.30 rested through the first places on that rest, as they
//    held the game back by little). A
//    rest that ended within 30 s makes the next one wait longer: 5 s, doubling
//    to 160 s; a longer one sets the wait back to 5 s, and so does one a
//    loading screen came into (a tour of the five census places at 60, a
//    lamp's travel between each two, woke the pool at each arrival 27 s into
//    its rest, and the doubled waits kept it working 40 s into the third
//    place) - and so does a wake for a frame far from its cap with the command
//    processor under 85% busy (the place's own hitch; with it busier the rest
//    itself holds the frame down, and the wait doubles).
//  - Near its cap (two windows in a row with the limiter waiting 8% of the
//    window or more, but not 15% twice) the frame meets the cap, narrowly: the
//    pool's work buys a frame there only if the frame would miss the cap
//    without it. A trial rest asks - the switch's three windows, then three
//    counted (two when they agree, which fixes the medians of three): where the
//    counted ones would keep a rest at the cap (their medians: the limiter still
//    waiting 8% or more, the command processor's share not ending it) the rest
//    goes on as a cap's; else the pool works
//    again, and the next trial waits as after a short rest at a cap (the same
//    backoff). Until 2026-11-10 a working pool never rested there ("no
//    experiment"), and since every loading screen starts a place working, a
//    place whose frame takes 85-92% of its cap's interval - a cap a machine
//    nearly makes - kept the pool working for good.
//  - With no cap holding the frame (uncapped, or a frame slower than its cap)
//    the frame rate decides, measured both ways at the place: an experiment
//    switches to the other state and back, and compares the frame rate there
//    with the windows on either side of it (a b a: a place growing heavier or
//    lighter meanwhile cancels out). The pool rests where resting costs nothing
//    - the frame under 0.5% slower resting than working - and works anywhere
//    else. The game held back by the command processor while resting only
//    rules the pool out: the pool shortens the frame by keeping the command
//    processor's backlog from holding the game back, so where resting holds it
//    back under 10% of the time (the median of the experiment's resting
//    windows) the pool rests unless the frame resting is 8% slower. There its
//    work is a cost or nothing - on two Xeon Gold 6230 the main loop's own
//    work 0.35 ms a frame longer beside its threads at Central Yharnam (held
//    back 0.3-0.7% resting), and the census found the homes alone 1.1% ahead
//    of the pool at the Nightmare of Mensis (12%) - and the frame rate alone,
//    4 s each way, can be 2-3% off: two probes at Central Yharnam found the
//    frame 0.7-1.9% faster working, and resting ran 2.4% faster than working
//    through the window. Where the share is high it says little on its own: under
//    Windows on an RTX 3070 Ti Laptop GPU it is 73-87% at Central Yharnam,
//    Cainhurst's courtyard and the sewer pit with the pool or without it, and
//    the pool there gains 2-4%; v0.2.30's rule, a probe of 2.5 s against the
//    windows before it that woke the pool only with the game held back 20% and
//    the frame 3% slower, forwent those gains, and a rest that crossed a
//    loading screen into a place that holds the game back by little
//    (Byrgenwerth's lake) was never asked again.
//    Experiments come 10 s after the pool began to work at a place (a loading
//    screen's end), then twice as far each time (to 4 minutes): from working a
//    probe (rest, then work again), from a rest an experiment decided an ask
//    (work, then rest again). Each phase is eight windows after the switch's
//    three: a switch takes 1-2 s to settle either way (resting 5 s at a time,
//    then working 5 s, the sewer pit's first seconds of a rest ran 2-3% slower
//    than its last - the command processor's caches cold for the lookups the
//    pool made - and Old Yharnam's and Cainhurst's courtyard's first second
//    ran faster, before the command processor's backlog filled); a probe's
//    rest 8% slower than the pool's windows before it after
//    three windows, or an ask's work 8% faster, decides at once (Old Yharnam's
//    probes ran 7-23% slower), and the next experiment comes four times as far
//    instead of twice. A rest an experiment decided asks again at once
//    when the game is held back 20% or more of the time (the median of six
//    windows) with the frame 5% slower than the experiment found it resting
//    (the place grew heavier); an ask that comes due while the rest holds the
//    game back under 10% is not made (the next comes twice as far).
//  - A GPU-bound frame (the GPU busy 90% of two windows in a row) rests it: the
//    pool cannot shorten it, and on an APU its CPU time is the GPU's power. The
//    rest ends once the GPU has room - busy under 80% over the last six windows
//    (the median) - and the place's experiments decide from there; at a cap the
//    command processor's share ends it only with the GPU under 80% in its
//    window: on a Steam Deck at Old Yharnam, 88-97% from one window to the
//    next, the pool rested and woke four times in 20 s uncapped and five at 60
//    with a line at 90%.
//  - A loading screen: its windows are not the place's and count for nothing;
//    an experiment under way is dropped. The next place is one no experiment
//    has judged, so a rest an experiment decided or the GPU began ends with it
//    and the experiments start again 10 s after it; a rest at a cap goes on (at
//    60 the cap's rule decides at the next place too).
#pragma once

#include <cstdint>

namespace gxvk_rest {

constexpr double kRestSlack = 0.15;  // the limiter's wait, as a share of a window, that counts toward a rest at a cap
constexpr double kWakeSlack = 0.08;  // under it the cap no longer holds the frame
constexpr double kWakeCp = 0.85;     // the command processor busy this share of a window ends a rest at a cap
constexpr double kSaturated = 0.92;  // ... and this share over three windows' medians, however long the limiter waits
constexpr double kWorkHeld = 0.20;   // the main loop held back by the command processor this share (the median of six windows)
constexpr double kFar = 0.85;        // a frame under this share of its cap: no cap holds it, however the limiter waits
constexpr double kCost = 0.005;      // resting no more than this much slower than working costs nothing: the pool rests
constexpr double kClear = 0.08;      // an experiment's other state this much slower (a probe) or faster (an ask): decided at once
constexpr double kHeavier = 0.05;    // a rest an experiment decided, held back 20%: this much slower than it found it asks again
constexpr double kQuiet = 0.10;      // resting holds the game back under this share (the median of an experiment's resting windows): the pool cannot shorten the frame
constexpr double kGpuBound = 0.90;   // the GPU busy this share of a window: the frame is the GPU's
constexpr double kGpuFree = 0.80;    // a rest the GPU began ends with the GPU busy under this (the median of six windows)
constexpr int kQuietWindows = 2;     // windows in a row with the cap holding the frame, or GPU-bound
constexpr int kCapWindows = 3;       // resting at a cap: the command processor's busy share and the wait are the median of this many windows
constexpr int kTrialWindows = 3;     // a trial rest near the cap: its counted windows, after the switch's kSkip
constexpr int kHeldWindows = 6;      // resting: the held-back share is the median of this many windows
constexpr int kSkip = 3;             // an experiment's phase (and a state after any switch): its first windows are the switch's
constexpr int kPhase = 8;            // ... then this many are counted: the phase's frame rate is their median
constexpr int kEarly = 3;            // an experiment's middle phase decides at once after this many counted windows (kClear)
constexpr std::uint64_t kBackoffFirstNs = 5'000'000'000ull, kBackoffMostNs = 160'000'000'000ull, kShortRestNs = 30'000'000'000ull;
constexpr std::uint64_t kProbeFirstNs = 10'000'000'000ull, kProbeMostNs = 240'000'000'000ull;

// Why the rest under way began (the lines and the counts say it). kWhyProbe: an experiment's rest (a probe's, or an
// ask's second); kWhyJudged: a rest an experiment decided.
enum Why : int { kWhyNone, kWhyCap, kWhyGpu, kWhyProbe, kWhyJudged, kWhyNear, kWhys };

// The experiment under way, by its phase: a probe (from working: rest, then work again) or an ask (from a rest an
// experiment decided: work, then rest again).
enum Phase : int { kNoPhase, kProbeRest, kProbeWork, kAskWork, kAskRest };

// What the last step did, for the lines.
enum Event : int {
    kEvNone,
    kEvCapRest,   // a rest at a cap began
    kEvGpuRest,   // a GPU-bound rest began
    kEvProbe,     // a probe began: the pool rests
    kEvProbeBack, // the probe's rest measured: the pool works again to compare
    kEvAsk,       // an ask began: the pool works (held: the game held back and the frame slower - a heavier place)
    kEvAskBack,   // the ask's work measured: the pool rests again to compare
    kEvRests,     // an experiment decided: the pool rests
    kEvWorks,     // an experiment decided: the pool works
    kEvCapWake,   // a rest at a cap (or GPU-bound) ended
    kEvLoadWake,  // a rest an experiment decided or the GPU began ended with a loading screen
    kEvQuiet,     // a decided rest's ask came due with the rest holding the game back under kQuiet: not made
    kEvNearTrial, // near the cap: a trial rest began
    kEvNearRests, // the trial: the cap holds the frame without the pool - a cap's rest goes on
    kEvNearWorks, // the trial: without the pool the frame does not keep its cap - the pool works again
};

struct State {
    bool resting = false;
    Why why = kWhyNone;
    Phase phase = kNoPhase;
    Event event = kEvNone;
    bool held_ask = false;            // the ask under way began with the game held back (kHeavier)
    int quiet = 0;                    // windows in a row with the cap holding the frame (working)
    int near = 0;                     // ... and with the frame at its cap or nearly (the limiter waiting kWakeSlack or more)
    int gpu_bound = 0;                // windows in a row GPU-bound
    int settle = kSkip;               // the state's own windows (outside an experiment) start after this many more
    double own[kPhase] = {};          // ... and its last kPhase windows' frame rates: an experiment's first phase
    int own_n = 0;
    int skip = 0;                     // an experiment's phase: the switch's windows still to come
    double cur[kPhase] = {};          // ... and its counted windows' frame rates
    double cur_held[kPhase] = {};     // ... and held-back shares
    int cur_n = 0;
    double before = 0, middle = 0, after = 0;  // the experiment's phases' frame rates (the medians; 0: not measured)
    double rest_held = -1;            // the game held back while it rested in the experiment (the median; -1: not measured)
    double rest_ref = 0;              // a rest an experiment decided: the frame rate it found resting
    double held[kHeldWindows] = {};   // resting: the last windows' held-back shares ...
    double rest_fps[kHeldWindows] = {};  // ... frame rates
    double rest_gpu[kHeldWindows] = {};  // ... and the GPU's busy shares
    int held_n = 0;
    double cap_cp[kCapWindows] = {}, cap_slack[kCapWindows] = {};  // resting at a cap: the last windows' busy shares, waits
    double cap_fps[kCapWindows] = {};                              // ... and frame rates
    int cap_n = 0;
    double gpu_last[kCapWindows] = {};  // the GPU's busy shares of the last windows of the place, whatever the state
    int gpu_last_n = 0;
    double trial_cp[kTrialWindows] = {}, trial_slack[kTrialWindows] = {};  // a trial near the cap: its counted windows'
    int trial_n = 0;
    std::uint64_t rest_at = 0;        // when the rest under way began
    std::uint64_t allowed_at = 0;     // no rest at a cap begins before (the backoff)
    std::uint64_t backoff_ns = kBackoffFirstNs;
    bool travelled = false;           // a loading screen since the rest under way began: the backoff starts over at its end
    std::uint64_t next_at = 0;        // the next experiment (0: not set yet - the interval from the next window of the place)
    std::uint64_t interval_ns = kProbeFirstNs;
    // Whether the frame rate's experiments decide (the pool's duty). Without them (the run commit's, list_resolve.h)
    // only a cap holding the frame or a GPU-bound frame rests it, and it works anywhere else, as it did before a duty.
    bool experiments = true;
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
constexpr int kMostValues = kHeldWindows > kPhase ? kHeldWindows : kPhase;
// The median of n values (n at most kMostValues): one window's hitch moves it no more than the windows around it.
inline double median(const double* v, int n) {
    double a[kMostValues];
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
// The mean of n values (0: none) - a busy share over windows of one length.
inline double mean(const double* v, int n) {
    double sum = 0;
    for (int k = 0; k < n; ++k) sum += v[k];
    return n ? sum / n : 0;
}
// The last `most` values, oldest first.
inline void keep(double* ring, int& n, int most, double v) {
    if (n == most) {
        for (int k = 1; k < most; ++k) ring[k - 1] = ring[k];
        ring[most - 1] = v;
    } else {
        ring[n++] = v;
    }
}
// After a switch: the state's own windows start again past the switch's.
inline void own_again(State& s) {
    s.own_n = 0;
    s.settle = kSkip;
    s.held_n = 0;
    s.quiet = 0;
    s.near = 0;
    s.cap_n = 0;
}
// A frame well under its cap - under kFar of it: no cap holds it, however the limiter waits (with uneven frames the
// fast ones still meet the limiter). A frame rate not known (0), or no cap, is not.
inline bool far_from_cap(double fps, double cap_fps) { return cap_fps > 0 && fps > 0 && fps < kFar * cap_fps; }
// At a cap: whether the command processor's busy share wakes a rest, over windows' medians - kWakeCp or more with
// the frame meeting its cap narrowly (the limiter waiting under kRestSlack), or kSaturated or more however it waits.
inline bool cap_wakes(double cp, double slack) { return (cp >= kWakeCp && slack < kRestSlack) || cp >= kSaturated; }
// Whether a rest at the cap holds over windows' medians: the frame at its cap (the limiter waiting kWakeSlack or more)
// and the command processor's share not waking it.
inline bool cap_holds(double cp, double slack) { return slack >= kWakeSlack && !cap_wakes(cp, slack); }
inline void phase(State& s, Phase p) {
    s.phase = p;
    s.skip = kSkip;
    s.cur_n = 0;
}
inline void no_experiment(State& s) {
    s.phase = kNoPhase;
    s.skip = 0;
    s.cur_n = 0;
    s.held_ask = false;
}
inline Step rest(State& s, Why why, std::uint64_t now) {
    s.resting = true;
    s.why = why;
    s.rest_at = now;
    s.travelled = false;
    own_again(s);
    return kRest;
}
inline Step work(State& s) {
    s.resting = false;
    s.why = kWhyNone;
    own_again(s);
    return kWake;
}
// An experiment decided: the next one twice as far as the last - four times after a clear answer (decided at once:
// at Old Yharnam, where a probe's rest costs the frame 15-20%, the next probe 20 s after one fell inside the harness's
// 20-s window).
inline void judged(State& s, std::uint64_t now, bool clear = false) {
    for (int k = clear ? 2 : 1; k > 0; --k) s.interval_ns = s.interval_ns * 2 < kProbeMostNs ? s.interval_ns * 2 : kProbeMostNs;
    s.next_at = now + s.interval_ns;
    no_experiment(s);
}
// A rest at a cap ended (the cap no longer holds the frame, or the command processor is about to): the backoff - a
// short one makes the next wait longer, but not across a loading screen (the frame missing its cap in the next place's
// first seconds is no flapping) - and the place's experiments from the next window.
// A frame that fell well under its cap with the command processor under kWakeCp busy over the cap windows (`hitch`) is
// the place's own hitch, no flapping of the cap's rule: the next rest at the cap may come kBackoffFirstNs on, and the
// backoff starts over - at Cathedral Ward at a 120 cap an arrival's second of 50-100 fps (the command processor 63-64%
// busy over the cap windows) woke the rest, and a backoff the tour's earlier wakes had doubled to 40 s kept the pool
// working through the 20-s window that followed at the cap (the process 26 points over v0.2.33's, which rested there).
// With the command processor busier the rest itself holds the frame down, and the backoff doubles as after any short
// rest: at Cainhurst's roof each rest at the cap brought the dip back within 2 s, and a backoff starting over at each
// such wake rested and dipped again every 6.5 s.
inline Step cap_wake(State& s, std::uint64_t now, bool hitch = false) {
    if (s.why == kWhyCap) {
        const std::uint64_t rested = now - s.rest_at;
        s.backoff_ns = rested < kShortRestNs && !s.travelled && !hitch ? (s.backoff_ns * 2 < kBackoffMostNs ? s.backoff_ns * 2 : kBackoffMostNs)
                                                                       : kBackoffFirstNs;
        s.allowed_at = now + s.backoff_ns;
    }
    s.next_at = now + s.interval_ns;
    s.event = kEvCapWake;
    no_experiment(s);
    return work(s);
}
// Resting with no cap holding the frame: the last windows' held-back shares, frame rates and GPU.
inline void keep_rest(State& s, const Window& w) {
    if (s.held_n == kHeldWindows) {
        for (int k = 1; k < kHeldWindows; ++k) {
            s.held[k - 1] = s.held[k];
            s.rest_fps[k - 1] = s.rest_fps[k];
            s.rest_gpu[k - 1] = s.rest_gpu[k];
        }
        s.held_n = kHeldWindows - 1;
    }
    s.held[s.held_n] = w.held;
    s.rest_fps[s.held_n] = w.fps;
    s.rest_gpu[s.held_n++] = w.gpu;
}
// The state's own window (outside an experiment): its frame rate kept past the switch's windows.
inline void own_window(State& s, const Window& w) {
    if (s.settle) {
        --s.settle;
        return;
    }
    if (w.fps > 0) keep(s.own, s.own_n, kPhase, w.fps);
}
// A phase's window: true once its kPhase windows are counted.
inline bool count(State& s, const Window& w) {
    if (s.skip) {
        --s.skip;
        return false;
    }
    if (s.cur_n < kPhase) {
        s.cur_held[s.cur_n] = w.held;
        s.cur[s.cur_n++] = w.fps;
    }
    return s.cur_n == kPhase;
}
// Whether resting at `rest` fps against working at `work` costs nothing (an unknown frame rate: not shown to) - or
// resting held the game back under kQuiet (`held`, -1: not known) and the frame is less than kClear slower: the pool
// cannot shorten a frame the command processor's backlog does not hold.
inline bool rest_free(double rest, double work, double held = -1) {
    if (rest <= 0 || work <= 0) return false;
    if (rest >= work * (1.0 - kCost)) return true;
    return held >= 0 && held < kQuiet && rest > work * (1.0 - kClear);
}
}  // namespace detail

// One window's verdict at `now` (ns): the state moved on, and whether a rest began or ended (s.event says why).
inline Step step(State& s, const Window& w, std::uint64_t now) {
    using namespace detail;
    s.event = kEvNone;
    if (w.loading) {
        // Not the place's: nothing counts, an experiment under way is dropped, and the next place is one no experiment
        // has judged - a rest an experiment decided (or began) or the GPU began ends here; one at a cap goes on.
        s.quiet = s.gpu_bound = 0;
        s.near = 0;
        s.held_n = 0;
        s.cap_n = 0;
        s.gpu_last_n = 0;
        s.own_n = 0;
        s.settle = kSkip;
        s.rest_ref = 0;
        s.interval_ns = kProbeFirstNs;
        s.next_at = 0;
        no_experiment(s);
        if (!s.resting) return kStay;
        s.travelled = true;
        if (s.why == kWhyNear) s.why = kWhyCap;  // (a trial under way: its windows were not the place's - it goes on as a cap's)
        if (s.why == kWhyCap) return kStay;
        s.event = kEvLoadWake;
        return work(s);
    }
    // The limiter waits (the frame at its cap, or nearly - or uneven frames under it, the fast ones meeting it); the cap
    // holds the frame only with the frame at kFar of it or more: a window well under its cap counts toward no rest at it.
    const bool far = far_from_cap(w.fps, w.cap_fps);
    const bool limited = w.cap_fps > 0 && w.slack >= kWakeSlack;
    const bool capped = w.cap_fps > 0 && w.slack >= kRestSlack && !far;  // the cap holds the frame
    const bool at_cap = limited && !far;                                 // ... or nearly
    s.gpu_bound = w.gpu >= kGpuBound ? s.gpu_bound + 1 : 0;
    keep(s.gpu_last, s.gpu_last_n, kCapWindows, w.gpu);
    const bool gpu_bound = s.gpu_bound >= kQuietWindows;
    if (!s.resting) {
        s.quiet = capped ? s.quiet + 1 : 0;
        s.near = at_cap ? s.near + 1 : 0;
        if (s.quiet >= kQuietWindows && now >= s.allowed_at) {
            no_experiment(s);
            s.event = kEvCapRest;
            return rest(s, kWhyCap, now);
        }
        if (gpu_bound) {
            no_experiment(s);
            s.event = kEvGpuRest;
            return rest(s, kWhyGpu, now);
        }
        if (at_cap) {  // the cap's rule decides there: no experiment, and its windows are not the place's uncapped ones
            no_experiment(s);
            s.next_at = 0;
            s.own_n = 0;
            s.settle = kSkip;
            // Near the cap: a trial whether the cap holds the frame without the pool.
            if (s.near >= kQuietWindows && now >= s.allowed_at) {
                s.event = kEvNearTrial;
                const Step r = rest(s, kWhyNear, now);
                s.skip = kSkip;
                s.trial_n = 0;
                return r;
            }
            return kStay;
        }
        if (s.phase == kProbeWork) {  // a probe's last phase: the pool's frame rate after its rest
            if (!count(s, w)) return kStay;
            s.after = median(s.cur, s.cur_n);
            const double work_fps = s.before > 0 ? (s.before + s.after) / 2 : s.after;
            if (rest_free(s.middle, work_fps, s.rest_held)) {
                s.rest_ref = s.middle;
                judged(s, now);
                s.event = kEvRests;
                return rest(s, kWhyJudged, now);
            }
            judged(s, now);
            s.event = kEvWorks;
            return kStay;
        }
        if (s.phase == kAskWork) {  // an ask's middle phase: the pool's frame rate
            if (!count(s, w)) {
                if (s.cur_n == kEarly && s.before > 0 && median(s.cur, s.cur_n) >= s.before * (1.0 + kClear)) {
                    s.middle = median(s.cur, s.cur_n);
                    s.after = 0;
                    judged(s, now, true);
                    s.event = kEvWorks;  // the pool shortens the frame clearly: it works on
                }
                return kStay;
            }
            s.middle = median(s.cur, s.cur_n);
            phase(s, kAskRest);
            s.event = kEvAskBack;
            return rest(s, kWhyProbe, now);
        }
        if (!s.experiments) return kStay;
        own_window(s, w);
        if (!s.next_at) s.next_at = now + s.interval_ns;
        if (now < s.next_at || s.own_n < kPhase) return kStay;
        // A probe: the pool's last windows, then a rest.
        s.before = median(s.own, s.own_n);
        s.middle = s.after = 0;
        s.rest_held = -1;
        phase(s, kProbeRest);
        s.event = kEvProbe;
        return rest(s, kWhyProbe, now);
    }
    if (s.why == kWhyNear) {
        // A trial near the cap: past the switch's windows, the counted ones would keep a rest at the cap (their medians)
        // - the cap holds the frame without the pool: the rest goes on as a cap's. Else the pool's work was what met the
        // cap: it works again, and the next trial (or rest at the cap) waits the backoff, doubled as after any short rest
        // at a cap.
        if (s.skip) {
            --s.skip;
            return kStay;
        }
        s.trial_cp[s.trial_n] = w.cp;
        s.trial_slack[s.trial_n++] = w.cap_fps > 0 ? w.slack : 0.0;
        // Two counted windows that agree fix the medians of three: both off the cap, both busy at a narrow
        // cap, or both with room - the third would not change the verdict. Half a second less per trial.
        int decided = 0;  // 1 holds, -1 fails
        if (s.trial_n == 2) {
            const double c0 = s.trial_cp[0], c1 = s.trial_cp[1], w0 = s.trial_slack[0], w1 = s.trial_slack[1];
            if (w0 < kWakeSlack && w1 < kWakeSlack) decided = -1;
            else if (cap_wakes(c0, w0) && cap_wakes(c1, w1)) decided = -1;
            else if (w0 >= kRestSlack && w1 >= kRestSlack && c0 < kSaturated && c1 < kSaturated) decided = 1;
        }
        if (!decided && s.trial_n < kTrialWindows) return kStay;
        s.why = kWhyCap;
        if (decided > 0 || (!decided && cap_holds(median(s.trial_cp, s.trial_n), median(s.trial_slack, s.trial_n)))) {
            s.event = kEvNearRests;
            return kStay;
        }
        const Step r = cap_wake(s, now);
        s.event = kEvNearWorks;
        return r;
    }
    if (gpu_bound) {
        if (s.phase != kNoPhase || s.why == kWhyJudged || s.why == kWhyProbe) {  // the frame is the GPU's now, whatever an experiment says
            no_experiment(s);
            s.why = kWhyGpu;
        }
        return kStay;
    }
    if (at_cap || (limited && s.why == kWhyCap)) {
        // At the cap: the command processor, drawing everything, near a core's whole time is about to hold the frame.
        // (Uncapped its busy share says nothing: it keeps up with a fast main loop at 90% without holding it back.) Not
        // in a rest the GPU began while the GPU is still busy 80%: on a Steam Deck at 60 at Old Yharnam the command
        // processor, busy 90-95% there, woke it five times in 20 s, the GPU between 88 and 97%. A rest an experiment
        // began or decided that meets the cap is the cap's from here on. A rest at a cap counts the windows the limiter
        // waits in well under the cap too: their frame rates end it.
        if (s.phase != kNoPhase || s.why == kWhyJudged || s.why == kWhyProbe) {
            no_experiment(s);
            s.why = kWhyCap;
        }
        s.held_n = 0;
        if (s.cap_n == kCapWindows) {
            for (int k = 1; k < kCapWindows; ++k) {
                s.cap_cp[k - 1] = s.cap_cp[k];
                s.cap_slack[k - 1] = s.cap_slack[k];
                s.cap_fps[k - 1] = s.cap_fps[k];
            }
            s.cap_n = kCapWindows - 1;
        }
        s.cap_cp[s.cap_n] = w.cp;
        s.cap_slack[s.cap_n] = w.slack;
        s.cap_fps[s.cap_n++] = w.fps;
        if (s.why == kWhyGpu && w.gpu >= kGpuFree) return kStay;
        if (s.cap_n < kCapWindows) return kStay;
        // The frame well under its cap over the cap windows (the median of their frame rates under kFar of it): no cap
        // holds it, whatever the limiter waits - unless the GPU is the limit (busy kGpuBound over the last windows, their
        // mean: a Steam Deck's GPU moves between 88 and 97% from one window to the next), where the pool's work buys no
        // frame and on an APU its CPU time is the GPU's power.
        const bool far_rest = far_from_cap(median(s.cap_fps, s.cap_n), w.cap_fps) && mean(s.gpu_last, s.gpu_last_n) < kGpuBound;
        if (cap_wakes(median(s.cap_cp, s.cap_n), median(s.cap_slack, s.cap_n))) return cap_wake(s, now);
        return far_rest ? cap_wake(s, now, median(s.cap_cp, s.cap_n) < kWakeCp) : kStay;
    }
    if (w.cap_fps <= 0) s.cap_n = 0;  // (no cap: its count starts again; a window just off it leaves the count as it is - a frame
                                      // hovering at the line is one at the cap, and its windows at the cap still make three)
    if (s.phase == kProbeRest) {  // a probe's middle phase: the frame resting
        if (!count(s, w)) {
            if (s.cur_n == kEarly && s.before > 0 && median(s.cur, s.cur_n) <= s.before * (1.0 - kClear)) {
                s.middle = median(s.cur, s.cur_n);
                s.after = 0;
                judged(s, now, true);
                s.event = kEvWorks;  // the pool shortens the frame clearly
                return work(s);
            }
            return kStay;
        }
        s.middle = median(s.cur, s.cur_n);
        s.rest_held = median(s.cur_held, s.cur_n);
        phase(s, kProbeWork);
        s.event = kEvProbeBack;
        return work(s);
    }
    if (s.phase == kAskRest) {  // an ask's last phase: the frame resting again
        if (!count(s, w)) return kStay;
        s.after = median(s.cur, s.cur_n);
        const double rest_fps = s.before > 0 ? (s.before + s.after) / 2 : s.after;
        const double after_held = median(s.cur_held, s.cur_n);
        if (s.rest_held < after_held) s.rest_held = after_held;  // (the higher of the two resting sides)
        if (rest_free(rest_fps, s.middle, s.rest_held)) {
            s.rest_ref = rest_fps;
            s.why = kWhyJudged;
            judged(s, now);
            s.event = kEvRests;
            own_again(s);
            return kStay;
        }
        judged(s, now);
        s.event = kEvWorks;
        return work(s);
    }
    keep_rest(s, w);
    if (s.why == kWhyJudged) {
        own_window(s, w);
        if (s.own_n < kPhase) return kStay;
        // An ask when its time has come, or at once when the place grew heavier: the game held back 20% or more with the
        // frame 5% slower than the experiment found it resting.
        const bool heavier = s.held_n == kHeldWindows && median(s.held, s.held_n) >= kWorkHeld && s.rest_ref > 0 &&
                             median(s.rest_fps, s.held_n) <= s.rest_ref * (1.0 - kHeavier);
        const bool due = s.next_at && now >= s.next_at;
        if (!due && !heavier) return kStay;
        // An ask that came due while the rest holds the game back under kQuiet is not made: no work of the pool's could
        // shorten this frame (the next comes twice as far).
        const double held_now = s.held_n == kHeldWindows ? median(s.held, s.held_n) : 1.0;
        if (!heavier && held_now < kQuiet) {
            s.interval_ns = s.interval_ns * 2 < kProbeMostNs ? s.interval_ns * 2 : kProbeMostNs;
            s.next_at = now + s.interval_ns;
            s.event = kEvQuiet;
            return kStay;
        }
        s.before = median(s.own, s.own_n);
        s.middle = s.after = 0;
        s.rest_held = s.held_n ? median(s.held, s.held_n) : -1;
        phase(s, kAskWork);
        s.held_ask = heavier && !due;
        s.event = kEvAsk;
        return work(s);
    }
    if (s.held_n < kHeldWindows) return kStay;
    // A GPU-bound rest ends once the GPU has room - under 80% over the last six windows (on a Steam Deck at Old Yharnam,
    // busy 88-97% from one window to the next, the pool rested and woke four times in 20 s at 90%): the place's
    // experiments decide from there.
    if (s.why == kWhyGpu) return median(s.rest_gpu, s.held_n) < kGpuFree ? cap_wake(s, now) : kStay;
    // A rest at a cap the frame no longer comes near - under 85% of it over the last six windows (the median): the cap
    // holds nothing there, and the place's experiments decide from there (uncapped play after the title's menus had met
    // the 960 cap: v0.2.30 rested through the first places on that rest, as their game was not held back 20%).
    if (w.cap_fps <= 0 || median(s.rest_fps, s.held_n) < kFar * w.cap_fps) return cap_wake(s, now);
    // One the frame misses by less: the held-back share over the last windows wakes it (a frame that misses its cap
    // for the main loop's own work or the GPU's gains nothing from the pool).
    return median(s.held, s.held_n) >= kWorkHeld ? cap_wake(s, now) : kStay;
}

}  // namespace gxvk_rest
