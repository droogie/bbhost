// The draw pool's duty (host/gxvk_rest.h), without the game: windows of the
// main loop's frame fed to the rule, and when a rest begins and ends checked.
// At a cap: two windows in a row with the limiter waiting 15% or more begin a
// rest; the command processor busy 85% ends it, and so does a wait under 8%
// with the game held back by the command processor 20% or more (the median of
// six windows) - not a wait under 8% alone; a rest that ended within 30 s
// makes the next wait (5 s, doubling to 160 s; a long rest, or one a loading
// screen came into, sets it back). With
// no cap holding the frame: a probe (five windows of rest, the first two the
// switch's) 10 s after the pool began to work, twice as far each time the game
// is held back again (to 240 s); a probe with the game held back under 20%
// (the median of its last three windows), or no slower than the windows before
// it by 3%, rests on, and a rest ends when the
// median of the last six windows reaches 20% - a burst of a second does not -
// with the frame 3% slower than the pool's as well once a probe said no; that
// rest asks anew after the probe's interval (the pool works five windows, then
// a probe), the interval doubling each time. GPU-bound (90%, two windows): a
// rest, and no wake while it lasts. A loading screen changes nothing.
#include "host/gxvk_rest.h"

#include <cstdint>
#include <cstdio>

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
    return w;
}
// Uncapped (the limiter waits for nothing), the game held back `held` of the window.
gxvk_rest::Window open(double held, double gpu = 0.5) { return win(0.0, 960.0f, 0.6, held, gpu); }
// ... at `fps` frames a second.
gxvk_rest::Window openf(double held, double fps) {
    gxvk_rest::Window w = open(held);
    w.fps = fps;
    return w;
}
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
        check(step(s, win(0.3, 60.0f, 0.86), t += kWin) == kWake && !s.resting, "not woken at 86% busy", 4);
    }
    // Just under the rest's wait never rests (and asks no probe: the cap's rule decides there).
    {
        State s;
        for (int k = 0; k < 60; ++k) check(step(s, win(0.149), t += kWin) == kStay && !s.resting, "a rest at a 14.9% wait", k);
        step(s, win(0.15), t += kWin);
        check(step(s, win(0.15), t += kWin) == kRest, "no rest at 15%", 60);
        // A wait under 8% alone does not end it: the frame misses its cap, but not for the command processor.
        for (int k = 0; k < 8; ++k) check(step(s, win(0.05, 60.0f, 0.6, 0.0), t += kWin) == kStay && s.resting, "woken at a 5% wait alone", k);
        // With the game held back 25% the fourth such window ends it: the median of the last six reaches 25%.
        for (int k = 0; k < 3; ++k) check(step(s, win(0.05, 60.0f, 0.6, 0.25), t += kWin) == kStay, "woken before four held windows", k);
        check(step(s, win(0.05, 60.0f, 0.6, 0.25), t += kWin) == kWake, "not woken by the game held back at a missed cap", 4);
        // A loading screen (no cap) does not end a rest.
        State u;
        step(u, win(0.3), t += kWin);
        step(u, win(0.3), t += kWin);
        for (int k = 0; k < 10; ++k)
            check(u.resting && step(u, win(0.0, 0.0f, 0.2, 0.0, 0.3, true), t += kWin) == kStay && u.resting, "woken by a loading screen", k);
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
        for (int k = 0; k < 6; ++k) step(s, win(0.0, 0.0f, 0.1, 0.0, 0.2, true), t += kWin);
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
    // Uncapped, the game never held back (Central Yharnam, the sewer pit): the first probe 10 s after the start, and it
    // rests on; it asks anew 20 s later (five windows of work, then a probe that rests on again), then 40, 80, 160, 240
    // and 240 s later - nothing else wakes it.
    {
        State s;
        int k = 0;
        for (; k < 20; ++k) check(step(s, openf(0.0, 105.0), t += kWin) == kStay && !s.resting, "a probe before 10 s", k);
        check(step(s, openf(0.0, 105.0), t += kWin) == kRest && s.resting && s.why == gxvk_rest::kWhyProbe, "no probe at 10 s", k);
        for (int w = 0; w < gxvk_rest::kProbeWindows - 1; ++w) check(step(s, openf(0.0, 105.0), t += kWin) == kStay, "a probe ended early", w);
        check(s.probe_left == 1, "the probe's windows", k);
        check(step(s, openf(0.0, 105.0), t += kWin) == kStay && s.resting && s.why == gxvk_rest::kWhyHeld, "the probe did not rest on", k);
        std::uint64_t want = 2 * gxvk_rest::kProbeFirstNs;
        for (int round = 0; round < 6; ++round) {
            check(s.recheck_at == t + want && s.rest_ref > 104.0 && s.rest_ref < 106.0, "the next time it asks anew", round);
            const std::uint64_t at = s.recheck_at;
            while (t + kWin < at) check(step(s, openf(0.05, 105.0), t += kWin) == kStay && s.resting, "woken with the game held back 5%", round);
            check(step(s, openf(0.05, 105.0), t += kWin) == kWake && !s.resting && s.recheck_left == gxvk_rest::kRecheckWindows &&
                      gxvk_rest::kRecheckWindows == 5,
                  "not asked anew, or for other than five windows (2.5 s: the switch's and four the probe compares with)", round);
            for (int w = 0; w < gxvk_rest::kRecheckWindows - 1; ++w)
                check(step(s, openf(0.0, 105.0), t += kWin) == kStay && !s.resting, "a probe before the windows that ask anew", w);
            check(step(s, openf(0.0, 105.0), t += kWin) == kRest && s.why == gxvk_rest::kWhyProbe && s.probe_ref > 104.0, "no probe when asked anew",
                  round);
            for (int w = 0; w < gxvk_rest::kProbeWindows; ++w) step(s, openf(0.0, 105.0), t += kWin);
            check(s.resting && s.why == gxvk_rest::kWhyHeld, "the probe asked anew did not rest on", round);
            want = want * 2 < gxvk_rest::kProbeMostNs ? want * 2 : gxvk_rest::kProbeMostNs;
        }
    }
    // Held back with the frame as fast as with the pool (under Windows on an RTX 3070 Ti Laptop GPU: 73-87% at Central
    // Yharnam, 113 fps against 115): the rest a probe began is not woken by the held-back share, nor by a frame 8%
    // slower than the probe found it; 10% slower (the place grew heavier) wakes it, with the held-back share - not
    // without it.
    {
        State s;
        until_rest(s, openf(0.8, 115.0), t);
        for (int w = 0; w < gxvk_rest::kProbeWindows; ++w) step(s, openf(0.84, 113.0), t += kWin);
        check(s.resting && s.why == gxvk_rest::kWhyHeld && s.rest_ref > 112.0 && s.rest_ref < 114.0, "the probe 2% slower did not rest on", 0);
        // (Within the 20 s before it asks anew: 40 windows.)
        for (int k = 0; k < 8; ++k) check(step(s, openf(0.85, 113.0), t += kWin) == kStay && s.resting, "woken held back at the pool's frame rate", k);
        for (int k = 0; k < 6; ++k) check(step(s, openf(0.85, 104.0), t += kWin) == kStay && s.resting, "woken held back 8% slower", k);
        for (int k = 0; k < 6; ++k) check(step(s, openf(0.1, 113.0), t += kWin) == kStay && s.resting, "woken", k);
        for (int k = 0; k < 8; ++k) check(step(s, openf(0.1, 100.0), t += kWin) == kStay && s.resting, "woken slower but not held back", k);
        for (int k = 0; k < 2; ++k) check(step(s, openf(0.6, 100.0), t += kWin) == kStay, "woken before the median", k);
        check(step(s, openf(0.6, 100.0), t += kWin) == kWake && !s.resting && s.recheck_left == 0, "not woken held back and 11.5% slower", 1);
    }
    // Uncapped where the pool shortens the frame (Old Yharnam, the game held back 54% with it resting): each probe wakes
    // it again, the next twice as far (20, 40, 80, 160, 240, 240 s); its first window, the switch's, is not counted.
    {
        State s;
        until_rest(s, open(0.0), t);
        std::uint64_t want = gxvk_rest::kProbeFirstNs;
        for (int round = 0; round < 7; ++round) {
            check(s.why == gxvk_rest::kWhyProbe, "a rest that is not a probe", round);
            for (int w = 0; w < gxvk_rest::kProbeSkip; ++w)  // (the switch's windows: their held share is not the place's)
                check(step(s, open(0.0), t += kWin) == kStay, "the switch's window counted", round);
            for (int w = 0; w < gxvk_rest::kProbeCounted - 1; ++w) check(step(s, open(0.5), t += kWin) == kStay, "a probe ended early", round);
            check(step(s, open(0.5), t += kWin) == kWake && !s.resting, "not woken by the probe", round);
            want = want * 2 < gxvk_rest::kProbeMostNs ? want * 2 : gxvk_rest::kProbeMostNs;
            check(s.probe_ns == want && s.probe_at == t + want, "the next probe", round);
            const std::uint64_t at = s.probe_at;
            until_rest(s, open(0.0), t);
            check(t == at || (t > at && t < at + kWin), "a probe not when due", round);
            check(t >= at, "a probe before it was due", round);
        }
    }
    // A probe's first two windows are the switch's (the command processor's caches cold for the lookups the pool made):
    // the game held back 40% there and 2% after it rests on (the sewer pit's first second of a rest); one window of
    // three counted at 40% does not wake it either (the median).
    {
        State s;
        until_rest(s, open(0.0), t);
        check(step(s, open(0.4), t += kWin) == kStay && step(s, open(0.4), t += kWin) == kStay, "a probe ended early", 0);
        check(step(s, open(0.02), t += kWin) == kStay && step(s, open(0.4), t += kWin) == kStay, "a probe ended early", 1);
        check(step(s, open(0.02), t += kWin) == kStay && s.resting && s.why == gxvk_rest::kWhyHeld, "the switch's windows counted", 2);
    }
    // The frame rate decides with the held-back share: a probe holding the game back 36% at the frame rate the pool
    // worked at (the sewer pit: 110 fps either way) rests on; one 7% slower (the Nightmare of Mensis: 93 against 100)
    // wakes it; one 3% slower with the game held back 10% rests on (something else slowed the frame).
    {
        State s;
        for (int k = 0; k < 20; ++k) step(s, openf(0.0, 110.0), t += kWin);
        check(step(s, openf(0.0, 110.0), t += kWin) == kRest && s.probe_ref > 109.0 && s.probe_ref < 111.0, "a probe without its reference", 0);
        for (int w = 0; w < gxvk_rest::kProbeSkip; ++w) step(s, openf(0.3, 104.0), t += kWin);
        for (int w = 0; w < gxvk_rest::kProbeCounted; ++w) step(s, openf(0.36, 110.0), t += kWin);
        check(s.resting && s.why == gxvk_rest::kWhyHeld, "woken at the frame rate the pool worked at", 1);
        State m;
        until_rest(m, openf(0.0, 100.0), t);
        check(m.resting && m.why == gxvk_rest::kWhyProbe, "no probe", 2);
        for (int w = 0; w < gxvk_rest::kProbeSkip; ++w) step(m, openf(0.5, 90.0), t += kWin);
        for (int w = 0; w < gxvk_rest::kProbeCounted - 1; ++w) check(step(m, openf(0.45, 93.0), t += kWin) == kStay, "a probe ended early", 3);
        check(step(m, openf(0.45, 93.0), t += kWin) == kWake, "not woken 7% slower and held back", 4);
        State o;
        until_rest(o, openf(0.0, 100.0), t);
        for (int w = 0; w < gxvk_rest::kProbeWindows; ++w) step(o, openf(0.10, 97.0), t += kWin);
        check(o.resting && o.why == gxvk_rest::kWhyHeld, "woken 3% slower with the game held back 10%", 5);
        // 2% slower and held back 40%: not slower enough - rests on.
        State q;
        until_rest(q, openf(0.0, 100.0), t);
        for (int w = 0; w < gxvk_rest::kProbeWindows; ++w) step(q, openf(0.40, 98.0), t += kWin);
        check(q.resting && q.why == gxvk_rest::kWhyHeld, "woken 2% slower", 6);
    }
    // Resting with no cap: the game held back 20% or more of the time (the median of six windows) wakes it, and the next
    // probe comes after the interval as it was (the probe's "no" doubled it), not twice as far again.
    {
        State s;
        until_rest(s, open(0.0), t);
        for (int k = 0; k < gxvk_rest::kProbeWindows; ++k) step(s, open(0.0), t += kWin);  // the probe rests on
        check(s.resting && s.why == gxvk_rest::kWhyHeld, "resting after the probe", 0);
        for (int k = 0; k < 5; ++k) check(step(s, open(0.5), t += kWin) == kStay, "woken before six windows", k);
        check(step(s, open(0.0), t += kWin) == kWake && s.probe_ns == 2 * gxvk_rest::kProbeFirstNs && s.probe_at == t + s.probe_ns,
              "not woken at a 50% median", 5);
        // Just under 20% rests on; 20% wakes (0.19 0.19 0.19 0.22 0.22 0.22: a 20.5% median).
        State u;
        until_rest(u, open(0.0), t);
        for (int k = 0; k < gxvk_rest::kProbeWindows; ++k) step(u, open(0.0), t += kWin);
        for (int k = 0; k < 12; ++k) check(step(u, open(0.19), t += kWin) == kStay && u.resting, "woken at 19%", k);
        check(step(u, open(0.22), t += kWin) == kStay && step(u, open(0.22), t += kWin) == kStay && u.resting, "woken at a 19% median", 12);
        check(step(u, open(0.22), t += kWin) == kWake, "not woken at a 20.5% median", 14);
        // A burst of a second (Central Yharnam's one second in twenty at 40%) among windows at 0-8% wakes nothing: in
        // 60 s only the ask 20 s after the probe's "no" changes the state (the next is 40 s after its own "no").
        State b;
        until_rest(b, open(0.0), t);
        for (int k = 0; k < gxvk_rest::kProbeWindows; ++k) step(b, open(0.0), t += kWin);
        int asks = 0;
        for (int k = 0; k < 120; ++k) {
            const double held = (k % 40) < 2 ? 0.4 : (k % 3) * 0.04;
            const gxvk_rest::Step r = step(b, open(held), t += kWin);
            if (r == kWake) ++asks;
            check(r != kWake || b.recheck_left == gxvk_rest::kRecheckWindows, "woken by a burst of a second", k);
        }
        check(asks == 1 && b.resting, "not asked anew once in 60 s", asks);
    }
    // GPU-bound: two windows in a row rest the pool, whatever the game is held back; no wake while it lasts, nor while
    // the GPU stays busy 80% or more (the Steam Deck at Old Yharnam: 88-97% from one window to the next); then the
    // held-back share decides.
    {
        State s;
        check(step(s, open(0.6, 0.95), t += kWin) == kStay, "a rest after one GPU-bound window", 0);
        check(step(s, open(0.6, 0.95), t += kWin) == kRest && s.why == gxvk_rest::kWhyGpu, "no rest GPU-bound", 1);
        for (int k = 0; k < 20; ++k) check(step(s, open(0.6, 0.95), t += kWin) == kStay && s.resting, "woken GPU-bound", k);
        for (int k = 0; k < 40; ++k) check(step(s, open(0.6, k % 2 ? 0.97 : 0.88), t += kWin) == kStay && s.resting, "woken at 88-97% busy", k);
        for (int k = 0; k < 2; ++k) check(step(s, open(0.6, 0.6), t += kWin) == kStay, "woken before the median", k);
        check(step(s, open(0.6, 0.6), t += kWin) == kWake, "not woken once the GPU had room", 6);
        // At a cap (the Steam Deck at 60): the command processor busy 90% does not end a rest the GPU began while the GPU
        // is busy 80% or more; with the GPU under it, it does.
        State c;
        check(step(c, win(0.10, 60.0f, 0.9, 0.0, 0.95), t += kWin) == kStay && step(c, win(0.10, 60.0f, 0.9, 0.0, 0.95), t += kWin) == kRest &&
                  c.why == gxvk_rest::kWhyGpu,
              "no rest GPU-bound at the cap", 8);
        for (int k = 0; k < 40; ++k)
            check(step(c, win(0.10, 60.0f, 0.9, 0.0, k % 2 ? 0.97 : 0.88), t += kWin) == kStay && c.resting, "woken at the cap at 88-97% busy", k);
        check(step(c, win(0.10, 60.0f, 0.9, 0.0, 0.7), t += kWin) == kWake, "not woken at the cap once the GPU had room", 9);
        // A rest a probe began is not asked anew while the frame is GPU-bound, however long.
        State g;
        until_rest(g, openf(0.0, 105.0), t);
        for (int w = 0; w < gxvk_rest::kProbeWindows; ++w) step(g, openf(0.0, 105.0), t += kWin);
        check(g.resting && g.recheck_at > 0, "the probe did not say no", 7);
        for (int k = 0; k < 200; ++k) check(step(g, open(0.6, 0.95), t += kWin) == kStay && g.resting, "asked anew GPU-bound", k);
        // 89% is not GPU-bound.
        State u;
        for (int k = 0; k < 10; ++k) check(step(u, open(0.0, 0.89), t += kWin) == kStay && !u.resting, "GPU-bound at 89%", k);
    }
    // Uncapped the command processor's busy share wakes nothing (the sewer pit: 92% busy with the homes alone, the game
    // held back 1.4%): only the held-back share does.
    {
        State s;
        until_rest(s, open(0.0), t);
        for (int k = 0; k < 40; ++k)
            check(step(s, win(0.0, 960.0f, 0.95, 0.014), t += kWin) == kStay && s.resting, "woken by a busy command processor uncapped", k);
    }
    // A loading screen: a probe under way is dropped (the rest goes on), nothing counts, and the probes start again at
    // 10 s after it - an interval doubled before it too.
    {
        State d;
        until_rest(d, open(0.0), t);
        for (int w = 0; w < gxvk_rest::kProbeWindows; ++w) step(d, open(0.5), t += kWin);  // the probe wakes it: 20 s
        check(!d.resting && d.probe_ns == 2 * gxvk_rest::kProbeFirstNs, "the probe's yes did not double the interval", 45);
        for (int k = 0; k < 4; ++k) step(d, win(0.0, 0.0f, 0.1, 0.0, 0.2, true), t += kWin);
        check(d.probe_ns == gxvk_rest::kProbeFirstNs && d.probe_at == 0, "a doubled interval kept across a loading screen", 46);
        State s;
        until_rest(s, open(0.0), t);
        check(step(s, win(0.0, 0.0f, 0.1, 0.9, 0.2, true), t += kWin) == kStay && s.resting && s.probe_left == 0 && s.why == gxvk_rest::kWhyHeld,
              "a probe across a loading screen", 0);
        for (int k = 0; k < 40; ++k) check(step(s, win(0.0, 0.0f, 0.1, 0.9, 0.2, true), t += kWin) == kStay && s.resting, "a loading screen counted", k);
        check(s.probe_ns == gxvk_rest::kProbeFirstNs && s.probe_at == 0, "the probes not started again", 41);
        // Working when one begins: still working after it, and the first probe 10 s after it.
        State u;
        check(step(u, open(0.0), t += kWin) == kStay, "a rest at once", 42);
        for (int k = 0; k < 40; ++k) step(u, win(0.0, 0.0f, 0.1, 0.0, 0.2, true), t += kWin);
        check(!u.resting && u.probe_at == 0, "a rest across a loading screen", 43);
        const std::uint64_t from = t;
        until_rest(u, open(0.0), t);
        check(t >= from + gxvk_rest::kProbeFirstNs && t <= from + gxvk_rest::kProbeFirstNs + kSec, "the probe 10 s after the loading screen", 44);
    }
    // Held back at the pool's frame rate (the laptop's places above), 20 minutes: the state changes only to ask anew,
    // twice each time - 20, 40, 80, 160, 240 s apart -, so after the first five minutes at most two changes in four.
    {
        State s;
        int changes = 0, changes_late = 0;
        for (int k = 0; k < 2400; ++k) {
            const gxvk_rest::Step r = step(s, openf(0.84, 114.0), t += kWin);
            if (r != kStay) {
                ++changes;
                if (k >= 600) ++changes_late;
            }
        }
        check(changes_late <= 2 * (15 * 60 / 240 + 1), "more than two changes in four minutes after the first five", changes_late);
        check(changes >= 6, "never asked anew", changes);
    }
    // A loading screen drops a probe's "no" (another place's): after it the held-back share alone wakes the rest (the
    // laptop's 85% at the frame the probe found), and the ask that was due does not come.
    {
        State s;
        until_rest(s, openf(0.8, 115.0), t);
        for (int w = 0; w < gxvk_rest::kProbeWindows; ++w) step(s, openf(0.84, 113.0), t += kWin);
        check(s.resting && s.rest_ref > 112.0 && s.recheck_at > t, "the probe did not say no", 0);
        for (int k = 0; k < 4; ++k) step(s, win(0.0, 0.0f, 0.1, 0.0, 0.2, true), t += kWin);
        check(s.resting && s.rest_ref == 0 && s.recheck_at == 0, "a probe's no kept across a loading screen", 1);
        for (int k = 0; k < 5; ++k) check(step(s, openf(0.85, 113.0), t += kWin) == kStay, "woken before six windows", k);
        check(step(s, openf(0.85, 113.0), t += kWin) == kWake && s.recheck_left == 0, "not woken by the share after a loading screen", 2);
        State u;
        until_rest(u, openf(0.0, 105.0), t);
        for (int w = 0; w < gxvk_rest::kProbeWindows; ++w) step(u, openf(0.0, 105.0), t += kWin);
        const std::uint64_t due = u.recheck_at;
        for (int k = 0; k < 4; ++k) step(u, win(0.0, 0.0f, 0.1, 0.0, 0.2, true), t += kWin);
        while (t < due + 10 * kSec) check(step(u, openf(0.0, 105.0), t += kWin) == kStay && u.resting, "asked anew across a loading screen", 3);
    }
    // No flapping: the game held back on and off every few seconds (the Nightmare of Mensis' 0-42%) changes the state
    // less and less often - after the first two minutes, at most one change a minute.
    {
        State s;
        int changes_late = 0;
        for (int k = 0; k < 2400; ++k) {  // 20 minutes
            const double held = (k / 6) % 2 ? 0.42 : 0.0;
            const gxvk_rest::Step r = step(s, open(held), t += kWin);
            if (r != kStay && k >= 240) ++changes_late;
        }
        check(changes_late <= 18 * 2, "more than one change a minute after the first two", changes_late);
    }
    if (g_failures) {
        std::printf("gxvk_rest_test: %d failures\n", g_failures);
        return 1;
    }
    std::printf("gxvk_rest_test: ok\n");
    return 0;
}
