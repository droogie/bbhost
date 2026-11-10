// The draw pool's duty (host/gxvk_rest.h), without the game: windows of the
// main loop's frame fed to the rule, and when a rest begins and ends checked.
// At a cap: two windows in a row with the limiter waiting 15% or more begin a
// rest; the command processor busy 85% ends it, and so does a wait under 8%
// with the game held back by the command processor 20% or more (the median of
// six windows) - not a wait under 8% alone; a rest that ended within 30 s
// makes the next wait (5 s, doubling to 160 s; a long rest, or one a loading
// screen came into, sets it back). With no cap holding the frame the frame
// rate decides, both ways at the place: a probe 10 s after the pool began to
// work (rest three windows of the switch's and eight counted, then work as long)
// rests it when the rest's frame rate is under 0.5% slower than the median of
// the work on either side - or under 8% slower with the game held back under 10%
// while it rested - and keeps it working otherwise; 8% slower after three
// counted windows decides at once. A rest an experiment decided asks
// anew (work, then rest again) after the interval, twice as far each time (20
// s to 240 s), and at once with the game held back 20% and the frame 5%
// slower than it found it resting. GPU-bound (90%, two windows): a rest until
// the GPU's median of six windows is under 80%. A loading screen drops an
// experiment, ends any rest but one at a cap, and starts the place's
// experiments again 10 s after it.
#include "host/gxvk_rest.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>

namespace {

int g_failures = 0;
void check(bool ok, const char* what, int at) {
    if (ok) return;
    if (++g_failures <= 24) std::printf("FAIL: %s (window %d)\n", what, at);
}

constexpr std::uint64_t kSec = 1'000'000'000ull, kWin = kSec / 2;

gxvk_rest::Window win(double slack, float cap = 60.0f, double cp = 0.6, double held = 0.0, double gpu = 0.5, bool loading = false) {
    gxvk_rest::Window w;
    w.slack = slack;
    w.cp = cp;
    w.held = held;
    w.gpu = gpu;
    w.cap_fps = cap;
    w.loading = loading;
    w.fps = cap > 0 && slack > 0 ? cap : 100.0;
    return w;
}
gxvk_rest::Window loading_window() { return win(0.0, 0.0f, 0.1, 0.0, 0.2, true); }
// Uncapped (the limiter waits for nothing) at `fps`, the game held back `held` of the window.
gxvk_rest::Window openf(double held, double fps, double gpu = 0.5) {
    gxvk_rest::Window w = win(0.0, 960.0f, 0.6, held, gpu);
    w.fps = fps;
    return w;
}

// A place, uncapped: its frame rate working and resting, the game held back in each, and the switch's windows (the
// first two after the pool's state changed) at `switch_fps`.
struct Place {
    double work = 100, rest = 100;
    double held_work = 0, held_rest = 0;
    double switch_fps = 0;  // 0: as the state's own
};
struct Sim {
    gxvk_rest::State s;
    std::uint64_t t = 100 * kSec;
    bool was = false;
    int since = 0, changes = 0, rested = 0, windows = 0;
    // One window at place p in the state the pool is in (the switch's kSkip windows at p.switch_fps).
    gxvk_rest::Step one(const Place& p, double drift = 0.0) {
        const bool r = s.resting;
        if (r != was) {
            was = r;
            since = 0;
        }
        double fps = (r ? p.rest : p.work) * (1.0 + drift);
        if (since++ < gxvk_rest::kSkip && p.switch_fps > 0) fps = p.switch_fps;
        const gxvk_rest::Step k = gxvk_rest::step(s, openf(r ? p.held_rest : p.held_work, fps), t += kWin);
        if (k != gxvk_rest::kStay) ++changes;
        if (s.resting) ++rested;
        ++windows;
        return k;
    }
    void run(const Place& p, int n) {
        for (int k = 0; k < n; ++k) one(p);
    }
};
// Steps `w` until a rest begins, at most `most` windows (a rule that never rests fails, not hangs).
bool until_rest(gxvk_rest::State& s, const gxvk_rest::Window& w, std::uint64_t& t, int most = 2000) {
    for (int k = 0; k < most; ++k)
        if (gxvk_rest::step(s, w, t += kWin) == gxvk_rest::kRest) return true;
    check(false, "no rest in time", most);
    return false;
}

}  // namespace

int main() {
    using gxvk_rest::kRest;
    using gxvk_rest::kStay;
    using gxvk_rest::kWake;
    using gxvk_rest::State;
    using gxvk_rest::step;
    std::uint64_t t = 100 * kSec;
    constexpr int kPhaseWindows = gxvk_rest::kSkip + gxvk_rest::kPhase;  // an experiment's phase: the switch's and the counted
    static_assert(gxvk_rest::kSkip == 3 && gxvk_rest::kPhase == 8 && gxvk_rest::kEarly == 3, "the phases' windows the cases count on");
    // A cap holding the frame: the second window in a row begins the rest, not the first.
    {
        State s;
        check(step(s, win(0.3), t += kWin) == kStay, "a rest after one window", 0);
        check(step(s, win(0.3), t += kWin) == kRest && s.resting && s.why == gxvk_rest::kWhyCap, "no rest after two", 1);
        // Resting on while the limiter waits 8% or more and the command processor keeps up - the game held back or not:
        // the cap still holds the frame.
        check(step(s, win(0.10), t += kWin) == kStay && s.resting, "woken at a 10% wait", 2);
        for (int k = 0; k < 6; ++k) check(step(s, win(0.10, 60.0f, 0.6, 0.5), t += kWin) == kStay && s.resting, "woken held back at a 10% wait", k);
        check(step(s, win(0.3, 60.0f, 0.84), t += kWin) == kStay && s.resting, "woken at 84% busy", 3);
        // The command processor busy 85% of a window ends it.
        check(step(s, win(0.3, 60.0f, 0.86), t += kWin) == kWake && !s.resting && s.event == gxvk_rest::kEvCapWake, "not woken at 86% busy", 4);
    }
    // Just under the rest's wait never rests (and runs no experiment: the cap's rule decides there).
    {
        State s;
        for (int k = 0; k < 60; ++k)
            check(step(s, win(0.149), t += kWin) == kStay && !s.resting && s.phase == gxvk_rest::kNoPhase, "a rest or a probe at a 14.9% wait", k);
        step(s, win(0.15), t += kWin);
        check(step(s, win(0.15), t += kWin) == kRest, "no rest at 15%", 60);
        // A wait under 8% alone does not end it: the frame misses its cap, but not for the command processor.
        for (int k = 0; k < 8; ++k) check(step(s, win(0.05, 60.0f, 0.6, 0.0), t += kWin) == kStay && s.resting, "woken at a 5% wait alone", k);
        // With the game held back 25% the fourth such window ends it: the median of the last six reaches 25%.
        for (int k = 0; k < 3; ++k) check(step(s, win(0.05, 60.0f, 0.6, 0.25), t += kWin) == kStay, "woken before four held windows", k);
        check(step(s, win(0.05, 60.0f, 0.6, 0.25), t += kWin) == kWake, "not woken by the game held back at a missed cap", 4);
        // A loading screen (no cap) does not end a rest at a cap.
        State u;
        step(u, win(0.3), t += kWin);
        step(u, win(0.3), t += kWin);
        for (int k = 0; k < 10; ++k) check(u.resting && step(u, loading_window(), t += kWin) == kStay && u.resting, "a cap's rest woken by a loading screen", k);
    }
    // A window without the wait between two with it starts the count again.
    {
        State s;
        step(s, win(0.3), t += kWin);
        check(step(s, win(0.01), t += kWin) == kStay && step(s, win(0.3), t += kWin) == kStay, "a rest across a window without the wait", 0);
        check(step(s, win(0.3), t += kWin) == kRest, "no rest after two again", 1);
    }
    // A short rest at the cap makes the next wait: 10 s after the first (5 doubled), then 20, 40, 80, 160, 160.
    {
        State s;
        std::uint64_t want = gxvk_rest::kBackoffFirstNs;
        for (int round = 0; round < 7; ++round) {
            step(s, win(0.3), t += kWin);
            while (step(s, win(0.3), t += kWin) != kRest) {
                check(t < s.allowed_at + kSec, "no rest once allowed", round);
                if (t > s.allowed_at + kSec) break;
            }
            check(t >= s.allowed_at, "a rest before its wait", round);
            t += 2 * kSec;  // a 2 s rest
            check(step(s, win(0.3, 60.0f, 0.9), t) == kWake, "not woken", round);
            want = want * 2 < gxvk_rest::kBackoffMostNs ? want * 2 : gxvk_rest::kBackoffMostNs;
            check(s.backoff_ns == want && s.allowed_at == t + want, "the wait after a short rest", round);
            // Two windows with the wait inside it: no rest yet.
            check(step(s, win(0.3), t += kWin) == kStay && step(s, win(0.3), t += kWin) == kStay, "a rest inside the wait", round);
        }
        check(s.backoff_ns == gxvk_rest::kBackoffMostNs, "the wait past 160 s", 7);
        // A long rest sets the wait back to 5 s.
        while (step(s, win(0.3), t += kWin) != kRest) {
            if (t > s.allowed_at + kSec) break;
        }
        t += 40 * kSec;
        check(step(s, win(0.3, 60.0f, 0.9), t) == kWake && s.backoff_ns == gxvk_rest::kBackoffFirstNs && s.allowed_at == t + gxvk_rest::kBackoffFirstNs,
              "the wait after a long rest", 8);
        // A short rest a loading screen came into (a lamp's travel; the next place's first seconds miss the cap) sets the
        // wait back to 5 s too, after any number of short ones.
        for (int round = 0; round < 3; ++round) {
            while (step(s, win(0.3), t += kWin) != kRest) {
                if (t > s.allowed_at + kSec) break;
            }
            t += 2 * kSec;
            step(s, win(0.3, 60.0f, 0.9), t);
        }
        check(s.backoff_ns > gxvk_rest::kBackoffFirstNs, "short rests did not make the wait longer", 9);
        while (step(s, win(0.3), t += kWin) != kRest) {
            if (t > s.allowed_at + kSec) break;
        }
        for (int k = 0; k < 6; ++k) step(s, loading_window(), t += kWin);
        t += 5 * kSec;
        // (The next place's first seconds: 20-26 fps at the cap's limiter, the command processor busy 86-89%.)
        check(step(s, win(0.10, 60.0f, 0.9), t) == kWake && s.backoff_ns == gxvk_rest::kBackoffFirstNs && s.allowed_at == t + gxvk_rest::kBackoffFirstNs,
              "the wait after a short rest across a loading screen", 10);
        // ... and the next short rest, with no loading screen in it, doubles it again.
        while (step(s, win(0.3), t += kWin) != kRest) {
            if (t > s.allowed_at + kSec) break;
        }
        t += 2 * kSec;
        check(step(s, win(0.3, 60.0f, 0.9), t) == kWake && s.backoff_ns == 2 * gxvk_rest::kBackoffFirstNs, "a loading screen kept past its rest", 11);
    }
    // Uncapped where the pool's work costs the frame (two Xeon Gold 6230 at Central Yharnam and the sewer pit: 3% faster
    // resting), the game held back 15% resting: the first probe 10 s after the start rests three windows of the switch's
    // and eight counted, works as long again, and rests the pool; it asks anew 20 s later (work, then rest, each as long)
    // and rests on, then 40, 80, 160, 240 and 240 s later.
    {
        Sim m;
        const Place p{100.0, 103.0, 0.0, 0.15};
        int k = 0;
        for (; k < 20; ++k) check(m.one(p) == kStay && !m.s.resting, "a probe before 10 s", k);
        check(m.one(p) == kRest && m.s.phase == gxvk_rest::kProbeRest && m.s.event == gxvk_rest::kEvProbe && m.s.before == 100.0, "no probe at 10 s", k);
        for (int w = 0; w < kPhaseWindows - 1; ++w) check(m.one(p) == kStay && m.s.resting, "a probe's rest ended early", w);
        check(m.one(p) == kWake && m.s.phase == gxvk_rest::kProbeWork && m.s.event == gxvk_rest::kEvProbeBack && m.s.middle == 103.0,
              "the probe's rest not measured over its windows", k);
        for (int w = 0; w < kPhaseWindows - 1; ++w) check(m.one(p) == kStay && !m.s.resting, "the probe's work ended early", w);
        check(m.one(p) == kRest && m.s.why == gxvk_rest::kWhyJudged && m.s.event == gxvk_rest::kEvRests && m.s.rest_ref == 103.0,
              "the probe did not rest the pool 3% faster resting", k);
        std::uint64_t want = 2 * gxvk_rest::kProbeFirstNs;
        for (int round = 0; round < 6; ++round) {
            check(m.s.next_at == m.t + want, "the next time it asks anew", round);
            const std::uint64_t at = m.s.next_at;
            while (m.t + kWin < at) check(m.one(p) == kStay && m.s.resting, "woken before its ask", round);
            check(m.one(p) == kWake && m.s.phase == gxvk_rest::kAskWork && m.s.event == gxvk_rest::kEvAsk && m.s.before == 103.0, "not asked anew", round);
            for (int w = 0; w < kPhaseWindows - 1; ++w) check(m.one(p) == kStay && !m.s.resting, "the ask's work ended early", w);
            check(m.one(p) == kRest && m.s.phase == gxvk_rest::kAskRest && m.s.middle == 100.0, "the ask's work not measured", round);
            for (int w = 0; w < kPhaseWindows - 1; ++w) check(m.one(p) == kStay && m.s.resting, "the ask's rest ended early", w);
            check(m.one(p) == kStay && m.s.resting && m.s.why == gxvk_rest::kWhyJudged && m.s.event == gxvk_rest::kEvRests,
                  "the ask did not rest on", round);
            want = want * 2 < gxvk_rest::kProbeMostNs ? want * 2 : gxvk_rest::kProbeMostNs;
        }
    }
    // Uncapped where the pool gains a little (under Windows on an RTX 3070 Ti Laptop GPU at Central Yharnam and the sewer
    // pit: 113 fps resting against 116 working, the game held back 84% either way): the probe keeps it working, and the
    // next twice as far (20, 40, 80, 160, 240 s) - each a rest of 5.5 s, so in 20 minutes the pool rests 3.7% of the time
    // (eight probes: the frame 0.1% slower for them).
    {
        Sim m;
        const Place p{116.0, 113.0, 0.84, 0.84};
        m.run(p, 20);
        check(m.one(p) == kRest && m.s.phase == gxvk_rest::kProbeRest, "no probe at 10 s", 0);
        for (int w = 0; w < kPhaseWindows; ++w) m.one(p);
        check(!m.s.resting && m.s.phase == gxvk_rest::kProbeWork, "the probe's rest not followed by work", 1);
        for (int w = 0; w < kPhaseWindows; ++w) m.one(p);
        check(!m.s.resting && m.s.phase == gxvk_rest::kNoPhase && m.s.event == gxvk_rest::kEvWorks && m.s.next_at == m.t + 2 * gxvk_rest::kProbeFirstNs,
              "the probe 2.6% slower resting did not keep it working", 2);
        Sim n;
        n.run(p, 2400);
        check(n.rested <= 8 * kPhaseWindows, "resting more than eight probes' rests in 20 minutes where the pool gains 2.6%", n.rested);
        check(!n.s.resting, "resting at the end", 3);
    }
    // The line (the game held back 30% resting): resting 0.4% slower costs nothing (it rests), 0.6% slower does not (it
    // works), and as fast rests.
    {
        for (const double rest : {99.6, 99.4, 100.0}) {
            Sim m;
            const Place p{100.0, rest, 0.3, 0.3};
            m.run(p, 21 + 2 * kPhaseWindows);
            const bool want_rest = rest >= 99.5;
            check(m.s.resting == want_rest && m.s.phase == gxvk_rest::kNoPhase, want_rest ? "worked 0.4% slower or as fast resting" : "rested 0.6% slower",
                  static_cast<int>(rest * 10));
        }
    }
    // Where the pool shortens the frame clearly (Old Yharnam: the probes ran 7-23% slower resting): three counted windows
    // 8% slower decide at once - the probe rests 3 s, not 5.5 - and the next probe is 40 s away, not 20.
    {
        Sim m;
        const Place p{117.0, 98.0, 0.0, 0.5};
        m.run(p, 20);
        check(m.one(p) == kRest, "no probe at 10 s", 0);
        for (int w = 0; w < gxvk_rest::kSkip + gxvk_rest::kEarly - 1; ++w) check(m.one(p) == kStay && m.s.resting, "decided before three windows", w);
        check(m.one(p) == kWake && m.s.event == gxvk_rest::kEvWorks && m.s.phase == gxvk_rest::kNoPhase && m.s.after == 0 &&
                  m.s.next_at == m.t + 4 * gxvk_rest::kProbeFirstNs,
              "not decided at once 16% slower, or the next probe not 40 s on", 1);
        // 7% slower is not clear at once: the probe runs its full phases and keeps it working (the game held back 30%).
        Sim n;
        const Place q{100.0, 93.0, 0.0, 0.3};
        n.run(q, 21 + gxvk_rest::kSkip + gxvk_rest::kEarly);
        check(n.s.resting && n.s.phase == gxvk_rest::kProbeRest, "decided at once 7% slower", 2);
        n.run(q, 2 * kPhaseWindows);
        check(!n.s.resting && n.s.phase == gxvk_rest::kNoPhase && n.s.event != gxvk_rest::kEvRests, "rested 7% slower", 3);
    }
    // The switch's windows count for nothing: the first three of a rest at 60% of the frame (the command processor's
    // caches cold for the lookups the pool made; the sewer pit's first seconds of a rest ran slower), and the place resting
    // as fast as working: the probe rests it.
    {
        Sim m;
        Place p{110.0, 110.0, 0.0, 0.02};
        p.switch_fps = 66.0;
        m.run(p, 21 + 2 * kPhaseWindows);
        check(m.s.resting && m.s.why == gxvk_rest::kWhyJudged, "the switch's windows counted", 0);
    }
    // One hitch in a phase (a window at half the frame rate) is one window of eight: the medians do not move.
    {
        State s;
        for (int k = 0; k < 20; ++k) step(s, openf(0.0, 100.0), t += kWin);
        check(step(s, openf(0.0, 100.0), t += kWin) == kRest, "no probe", 0);
        for (int w = 0; w < kPhaseWindows; ++w) step(s, openf(0.0, w == 4 ? 50.0 : 101.0), t += kWin);
        for (int w = 0; w < kPhaseWindows; ++w) step(s, openf(0.0, w == 5 ? 50.0 : 100.0), t += kWin);
        check(s.resting && s.why == gxvk_rest::kWhyJudged && s.middle == 101.0 && s.after == 100.0, "a hitch moved a phase", 1);
    }
    // a b a: a place growing lighter 1% a phase (the frame faster from window to window) with the pool's work worth
    // nothing - the probe's rest is faster than the work before it and slower than the work after it: it rests, as the
    // average of the two says (with the work before it alone it would judge the rest 1% faster: the same); growing
    // heavier 1% a phase, the rest looks 1% slower than the work before it, and the work after it cancels that: it rests.
    {
        for (const double slope : {0.01, -0.01}) {
            State s;
            double fps = 100.0;
            const double per_window = slope / kPhaseWindows;
            for (int k = 0; k < 21; ++k) step(s, openf(0.0, fps *= 1.0 + per_window), t += kWin);
            check(s.phase == gxvk_rest::kProbeRest, "no probe", 0);
            for (int k = 0; k < 2 * kPhaseWindows; ++k) step(s, openf(0.0, fps *= 1.0 + per_window), t += kWin);
            check(s.resting && s.why == gxvk_rest::kWhyJudged, slope > 0 ? "worked as a place grew lighter" : "worked as a place grew heavier",
                  static_cast<int>(slope * 100));
        }
    }
    // The same where resting holds the game back 30%, so the frame rate decides by the 0.5% line: growing lighter 1% a
    // phase the probe's rest is 1% slower than the work after it (that alone would keep the pool working), growing
    // heavier 1% slower than the work before it (that alone too) - the average of the two sides says it costs nothing.
    {
        for (const double slope : {0.01, -0.01}) {
            State s;
            double fps = 100.0;
            const double per_window = slope / kPhaseWindows;
            for (int k = 0; k < 21; ++k) step(s, openf(0.3, fps *= 1.0 + per_window), t += kWin);
            check(s.phase == gxvk_rest::kProbeRest, "no probe (held 30%)", 0);
            for (int k = 0; k < 2 * kPhaseWindows; ++k) step(s, openf(0.3, fps *= 1.0 + per_window), t += kWin);
            check(s.resting && s.why == gxvk_rest::kWhyJudged,
                  slope > 0 ? "worked as a place grew lighter (held 30%)" : "worked as a place grew heavier (held 30%)",
                  static_cast<int>(slope * 100));
        }
    }
    // An ask where resting holds the game back 30% and the place grows lighter 1% a phase through it, the pool's work
    // worth nothing: its work is 1% faster than the rest before it and 1% slower than the rest after it - it rests on (the
    // rest before it alone would say the pool gains 1%).
    {
        Sim m;
        Place p{100.0, 100.0, 0.3, 0.3};
        m.run(p, 21 + 2 * kPhaseWindows);
        check(m.s.resting && m.s.why == gxvk_rest::kWhyJudged, "no rest decided (held 30%)", 0);
        const std::uint64_t at = m.s.next_at;
        const double per_window = 0.01 / kPhaseWindows;
        double d = 0.0;
        while (m.t + kWin < at) m.one(p, d += per_window);
        check(m.one(p, d += per_window) == kWake && m.s.phase == gxvk_rest::kAskWork, "no ask (held 30%)", 1);
        for (int k = 0; k < 2 * kPhaseWindows; ++k) m.one(p, d += per_window);
        check(m.s.resting && m.s.why == gxvk_rest::kWhyJudged && m.s.event == gxvk_rest::kEvRests, "an ask worked as the place grew lighter",
              2);
    }
    // Resting holds the game back 2% but the frame 15% slower once the rest has settled - its first three counted windows
    // only 5% slower, too little to decide at once: the phase's median decides, and the pool works (a quiet rest is free
    // only within 8%).
    {
        State s;
        for (int k = 0; k < 20; ++k) step(s, openf(0.02, 100.0), t += kWin);
        check(step(s, openf(0.02, 100.0), t += kWin) == kRest && s.phase == gxvk_rest::kProbeRest, "no probe (quiet, deepening)", 0);
        for (int w = 0; w < kPhaseWindows; ++w) step(s, openf(0.02, w < gxvk_rest::kSkip + gxvk_rest::kEarly ? 95.0 : 85.0), t += kWin);
        check(!s.resting && s.phase == gxvk_rest::kProbeWork, "the probe's rest did not run its phase", 1);
        for (int w = 0; w < kPhaseWindows; ++w) step(s, openf(0.02, 100.0), t += kWin);
        check(!s.resting && s.phase == gxvk_rest::kNoPhase && s.event == gxvk_rest::kEvWorks, "rested 15% slower (quiet)", 2);
    }
    // An ask where the pool now shortens the frame (the place changed while it rested: 6% faster working): the ask's work
    // 6% faster after its full phase, its rest again as slow - it works. 9% faster: decided at once, after three windows.
    {
        Sim m;
        Place p{100.0, 100.0, 0.3, 0.3};
        m.run(p, 21 + 2 * kPhaseWindows);
        check(m.s.resting && m.s.why == gxvk_rest::kWhyJudged, "no rest decided", 0);
        p.work = 106.0;
        const std::uint64_t at = m.s.next_at;
        while (m.t + kWin < at) m.one(p);
        check(m.one(p) == kWake && m.s.phase == gxvk_rest::kAskWork, "no ask", 1);
        m.run(p, 2 * kPhaseWindows);
        check(!m.s.resting && m.s.phase == gxvk_rest::kNoPhase && m.s.event == gxvk_rest::kEvWorks, "the ask did not keep it working 6% faster", 2);
        Sim n;
        Place q{100.0, 100.0, 0.3, 0.3};
        n.run(q, 21 + 2 * kPhaseWindows);
        q.work = 109.0;
        const std::uint64_t at2 = n.s.next_at;
        while (n.t + kWin < at2) n.one(q);
        n.one(q);
        n.run(q, gxvk_rest::kSkip + gxvk_rest::kEarly - 1);
        check(n.s.phase == gxvk_rest::kAskWork, "decided before three windows", 3);
        n.one(q);
        check(!n.s.resting && n.s.phase == gxvk_rest::kNoPhase && n.s.event == gxvk_rest::kEvWorks && n.s.after == 0, "not decided at once 9% faster", 4);
    }
    // Where resting holds the game back under 10% (on two Xeon Gold 6230 at Central Yharnam: 0.3-0.7%; the Nightmare of
    // Mensis 6-10%), the pool cannot shorten the frame: a probe 1.9% slower resting (the frame rate's own noise there, 4 s
    // each way) rests it, so does one 7% slower; 9% slower decides at once that it works. A rest held back 8% asks
    // nothing when its asks come due - in 10 minutes the pool's state never changes - and one held back 12% asks.
    {
        Sim m;
        const Place p{101.6, 99.7, 0.0, 0.005};
        m.run(p, 21 + 2 * kPhaseWindows);
        check(m.s.resting && m.s.why == gxvk_rest::kWhyJudged && m.s.event == gxvk_rest::kEvRests && m.s.rest_held < 0.01,
              "worked 1.9% slower resting with the game held back 0.5%", 0);
        Sim n;
        n.run(Place{100.0, 93.0, 0.0, 0.02}, 21 + 2 * kPhaseWindows);
        check(n.s.resting && n.s.why == gxvk_rest::kWhyJudged, "worked 7% slower resting with the game held back 2%", 1);
        Sim o;
        o.run(Place{100.0, 91.0, 0.0, 0.02}, 21 + gxvk_rest::kSkip + gxvk_rest::kEarly);
        check(!o.s.resting && o.s.event == gxvk_rest::kEvWorks && o.s.after == 0, "not decided at once 9% slower, held back 2%", 2);
        Sim q;
        const Place quiet{100.0, 101.0, 0.0, 0.08};
        q.run(quiet, 21 + 2 * kPhaseWindows);
        check(q.s.resting && q.s.why == gxvk_rest::kWhyJudged, "no rest decided", 3);
        const int changes = q.changes;
        int skipped = 0;
        for (int k = 0; k < 1200; ++k) {
            q.one(quiet);
            if (q.s.event == gxvk_rest::kEvQuiet) ++skipped;
        }
        check(q.changes == changes && q.s.resting && skipped >= 4, "asked while resting held the game back 8%", skipped);
        Sim u;
        const Place six{100.0, 101.0, 0.0, 0.12};
        u.run(six, 21 + 2 * kPhaseWindows);
        const std::uint64_t at = u.s.next_at;
        while (u.t + kWin < at) u.one(six);
        check(u.one(six) == kWake && u.s.phase == gxvk_rest::kAskWork, "not asked held back 12%", 4);
    }
    // A decided rest at a place grown heavier: the game held back 20% or more (the median of six windows) with the frame
    // 5% slower than the rest was found asks at once; held back 85% at the frame it found (the laptop's places) asks
    // nothing, nor 8% slower with the game held back 5%.
    {
        Sim m;
        Place p{110.0, 110.0, 0.0, 0.0};
        m.run(p, 21 + 2 * kPhaseWindows);
        check(m.s.resting && m.s.rest_ref == 110.0, "no rest decided", 0);
        p.held_rest = 0.85;
        for (int k = 0; k < 20; ++k) check(m.one(p) == kStay && m.s.resting, "asked held back at the frame it found", k);
        p.held_rest = 0.05;
        p.rest = 101.0;
        for (int k = 0; k < 6; ++k) check(m.one(p) == kStay && m.s.resting, "asked 8% slower held back 5%", k);
        p.held_rest = 0.30;
        for (int k = 0; k < 3; ++k) check(m.one(p) == kStay, "asked before the median", k);
        check(m.one(p) == kWake && m.s.phase == gxvk_rest::kAskWork && m.s.held_ask, "not asked held back 30% and 8% slower", 1);
    }
    // A loading screen: an experiment under way is dropped; a rest an experiment began or decided ends with it (the next
    // place is one no experiment judged); the next probe 10 s after it, the interval back at its first.
    {
        Sim m;
        const Place p{100.0, 103.0};
        m.run(p, 21 + 2 * kPhaseWindows);
        m.run(p, 200);  // (asks: 20 s and 40 s on)
        check(m.s.resting && m.s.interval_ns > 2 * gxvk_rest::kProbeFirstNs, "no doubled interval", 0);
        check(step(m.s, loading_window(), m.t += kWin) == kWake && !m.s.resting && m.s.event == gxvk_rest::kEvLoadWake, "a decided rest kept across a loading screen", 1);
        for (int k = 0; k < 10; ++k) check(step(m.s, loading_window(), m.t += kWin) == kStay && !m.s.resting, "a loading screen rested it", k);
        check(m.s.interval_ns == gxvk_rest::kProbeFirstNs && m.s.next_at == 0 && m.s.phase == gxvk_rest::kNoPhase, "the experiments not started again", 2);
        const std::uint64_t from = m.t;
        until_rest(m.s, openf(0.0, 100.0), m.t);
        check(m.t >= from + gxvk_rest::kProbeFirstNs && m.t <= from + gxvk_rest::kProbeFirstNs + kSec && m.s.phase == gxvk_rest::kProbeRest,
              "the probe not 10 s after the loading screen", 3);
        // A probe's rest under way: dropped, the pool works.
        check(step(m.s, loading_window(), m.t += kWin) == kWake && !m.s.resting && m.s.phase == gxvk_rest::kNoPhase, "a probe across a loading screen", 4);
        // A probe's work under way: dropped, the pool works on.
        until_rest(m.s, openf(0.0, 100.0), m.t);
        for (int w = 0; w < kPhaseWindows; ++w) step(m.s, openf(0.0, 100.0), m.t += kWin);
        check(!m.s.resting && m.s.phase == gxvk_rest::kProbeWork, "no probe's work", 5);
        check(step(m.s, loading_window(), m.t += kWin) == kStay && !m.s.resting && m.s.phase == gxvk_rest::kNoPhase, "a probe's work across a loading screen", 6);
    }
    // The title's menus meet an uncapped game's 960 cap: a rest at the cap, which goes on across the loading screen into
    // the world; at 100 fps there (under 85% of the cap) the sixth window ends it, the game held back or not, and the
    // place's probe comes 10 s later. A rest at a 60 cap with the frame at 55 (92% of it) stays the held-back share's.
    {
        State s;
        check(step(s, win(0.3, 960.0f), t += kWin) == kStay && step(s, win(0.3, 960.0f), t += kWin) == kRest && s.why == gxvk_rest::kWhyCap,
              "no rest at the title's cap", 0);
        for (int k = 0; k < 6; ++k) check(step(s, loading_window(), t += kWin) == kStay && s.resting, "a cap's rest ended by a loading screen", k);
        for (int k = 0; k < 5; ++k) check(step(s, openf(0.0, 100.0), t += kWin) == kStay && s.resting, "woken before six windows far from the cap", k);
        check(step(s, openf(0.0, 100.0), t += kWin) == kWake && !s.resting && s.event == gxvk_rest::kEvCapWake &&
                  s.next_at == t + gxvk_rest::kProbeFirstNs,
              "a rest at a cap the frame is far from not ended", 1);
        State u;
        step(u, win(0.3), t += kWin);
        check(step(u, win(0.3), t += kWin) == kRest, "no rest at 60", 2);
        for (int k = 0; k < 60; ++k) {
            gxvk_rest::Window w = win(0.02, 60.0f, 0.6, 0.05);
            w.fps = 55.0;
            check(step(u, w, t += kWin) == kStay && u.resting, "woken at 92% of the cap, not held back", k);
        }
        gxvk_rest::Window far = win(0.0, 60.0f, 0.6, 0.0);
        far.fps = 50.0;  // 83%: the median of the six windows under 85% from the fourth such window
        for (int k = 0; k < 3; ++k) check(step(u, far, t += kWin) == kStay, "woken before the median under 85%", k);
        check(step(u, far, t += kWin) == kWake, "not woken at 83% of the cap", 3);
    }
    // At a cap: no experiment. A probe's rest that meets the cap is the cap's rest from there on.
    {
        State s;
        for (int k = 0; k < 20; ++k) step(s, openf(0.0, 100.0), t += kWin);
        check(step(s, openf(0.0, 100.0), t += kWin) == kRest && s.phase == gxvk_rest::kProbeRest, "no probe", 0);
        check(step(s, win(0.10, 120.0f), t += kWin) == kStay && s.resting && s.why == gxvk_rest::kWhyCap && s.phase == gxvk_rest::kNoPhase,
              "a probe's rest at the cap not the cap's", 1);
        for (int k = 0; k < 100; ++k) check(step(s, win(0.10, 120.0f), t += kWin) == kStay && s.resting, "woken at the cap", k);
        // The frame slower than the cap from there (110 fps at 120: 92% of it), the game held back 0%: it rests on (the
        // cap's rule: the held-back share wakes it), and nothing asks.
        for (int k = 0; k < 100; ++k) {
            gxvk_rest::Window w = win(0.0, 120.0f, 0.6, 0.0);
            w.fps = 110.0;
            check(step(s, w, t += kWin) == kStay && s.resting, "woken off the cap, not held back", k);
        }
    }
    // GPU-bound: two windows in a row rest the pool, whatever the game is held back or the frame rate; no experiment
    // while it lasts, nor while the GPU's median of six windows is 80% or more (the Steam Deck at Old Yharnam: 88-97%
    // from one window to the next); under it the pool works, and the place's probe comes later.
    {
        State s;
        check(step(s, openf(0.6, 90.0, 0.95), t += kWin) == kStay, "a rest after one GPU-bound window", 0);
        check(step(s, openf(0.6, 90.0, 0.95), t += kWin) == kRest && s.why == gxvk_rest::kWhyGpu && s.event == gxvk_rest::kEvGpuRest, "no rest GPU-bound", 1);
        for (int k = 0; k < 200; ++k) check(step(s, openf(0.6, 90.0, 0.95), t += kWin) == kStay && s.resting, "woken GPU-bound", k);
        for (int k = 0; k < 40; ++k) check(step(s, openf(0.0, 90.0, k % 2 ? 0.97 : 0.88), t += kWin) == kStay && s.resting, "woken at 88-97% busy", k);
        for (int k = 0; k < 2; ++k) check(step(s, openf(0.0, 90.0, 0.6), t += kWin) == kStay, "woken before the median", k);
        check(step(s, openf(0.0, 90.0, 0.6), t += kWin) == kWake && s.event == gxvk_rest::kEvCapWake && s.next_at == t + s.interval_ns,
              "not woken once the GPU had room", 6);
        // At a cap (the Steam Deck at 60): the command processor busy 90% does not end a rest the GPU began while the GPU
        // is busy 80% or more; with the GPU under it, it does.
        State c;
        check(step(c, win(0.10, 60.0f, 0.9, 0.0, 0.95), t += kWin) == kStay && step(c, win(0.10, 60.0f, 0.9, 0.0, 0.95), t += kWin) == kRest &&
                  c.why == gxvk_rest::kWhyGpu,
              "no rest GPU-bound at the cap", 8);
        for (int k = 0; k < 40; ++k)
            check(step(c, win(0.10, 60.0f, 0.9, 0.0, k % 2 ? 0.97 : 0.88), t += kWin) == kStay && c.resting, "woken at the cap at 88-97% busy", k);
        check(step(c, win(0.10, 60.0f, 0.9, 0.0, 0.7), t += kWin) == kWake, "not woken at the cap once the GPU had room", 9);
        // A rest an experiment decided, then GPU-bound: the GPU's rest from there (no ask while it lasts).
        Sim g;
        const Place p{105.0, 105.0};
        g.run(p, 21 + 2 * kPhaseWindows);
        check(g.s.resting && g.s.why == gxvk_rest::kWhyJudged, "no rest decided", 10);
        for (int k = 0; k < 400; ++k)
            check(step(g.s, openf(0.0, 105.0, 0.95), g.t += kWin) == kStay && g.s.resting && g.s.phase == gxvk_rest::kNoPhase, "asked anew GPU-bound", k);
        check(g.s.why == gxvk_rest::kWhyGpu, "a decided rest GPU-bound not the GPU's", 11);
        // 89% is not GPU-bound.
        State u;
        for (int k = 0; k < 10; ++k) check(step(u, openf(0.0, 100.0, 0.89), t += kWin) == kStay && !u.resting, "GPU-bound at 89%", k);
        // A GPU-bound rest ends at a loading screen.
        check(step(s, openf(0.6, 90.0, 0.95), t += kWin) == kStay && step(s, openf(0.6, 90.0, 0.95), t += kWin) == kRest, "no rest GPU-bound again", 12);
        check(step(s, loading_window(), t += kWin) == kWake && !s.resting, "a GPU-bound rest kept across a loading screen", 13);
    }
    // Uncapped the game held back alone wakes nothing and rests nothing: the frame rate decides (the sewer pit: the command
    // processor 92% busy with the homes alone, the game held back 1.4%; the laptop: 84% either way).
    {
        Sim m;
        const Place p{100.0, 103.0, 0.0, 0.95};
        m.run(p, 21 + 2 * kPhaseWindows);
        check(m.s.resting && m.s.why == gxvk_rest::kWhyJudged, "held back 95% resting and 3% faster: not rested", 0);
        for (int k = 0; k < 30; ++k) check(m.one(p) == kStay && m.s.resting, "woken held back at the frame it found", k);
    }
    // No flapping: where resting and working are within 0.5% of each other (a coin toss at each experiment), and where
    // the frame moves 4% from window to window (a 1-in-3 pattern), after the first two minutes at most one change of the
    // pool's state a minute; and the rests and works an experiment brings are two changes - 20 minutes.
    {
        for (const double gain : {0.004, 0.006, 0.0}) {
            Sim m;
            int changes_late = 0;
            for (int k = 0; k < 2400; ++k) {
                const double wobble = (k % 3 == 0) ? 1.04 : 1.0;
                Place p{100.0 * (1.0 + gain) * wobble, 100.0 * wobble};
                if (m.one(p) != kStay && k >= 240) ++changes_late;
            }
            // Each experiment after the first two minutes is at most two switches out and, at most, one back - with the
            // interval at 240 s, five experiments in 18 minutes.
            check(changes_late <= 3 * 6, "more than one experiment's changes a four minutes after the first two", changes_late);
        }
    }
    if (g_failures) {
        std::printf("gxvk_rest_test: %d failures\n", g_failures);
        return 1;
    }
    std::printf("gxvk_rest_test: ok\n");
    return 0;
}
