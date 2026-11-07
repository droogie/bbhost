// The draw pool's rest while a frame cap holds the frame (host/gxvk_rest.h),
// without the game: windows of the main loop's limiter fed to the rule, and
// when a rest begins and ends checked - two windows in a row with a cap and
// the limiter waiting 15% or more begin one, a window with no cap, a wait under
// 8% or the command processor busy 85% ends it, and a rest that ended within
// 30 s makes the next wait (5 s, doubling to 160 s; a long rest sets it back).
#include "host/gxvk_rest.h"

#include <cstdint>
#include <cstdio>

namespace {

int g_failures = 0;
void check(bool ok, const char* what, int at) {
    if (ok) return;
    if (++g_failures <= 16) std::printf("FAIL: %s (window %d)\n", what, at);
}

constexpr std::uint64_t kSec = 1'000'000'000ull, kWin = kSec / 2;

gxvk_rest::Window win(double slack, float cap = 60.0f, double cp = 0.6) {
    gxvk_rest::Window w;
    w.slack = slack;
    w.cp = cp;
    w.cap_fps = cap;
    return w;
}

}  // namespace

int main() {
    using gxvk_rest::kRest;
    using gxvk_rest::kStay;
    using gxvk_rest::kWake;
    using gxvk_rest::State;
    using gxvk_rest::step;
    std::uint64_t t = 100 * kSec;
    // Uncapped, or no limiter: never a rest, however long the limiter waits.
    {
        State s;
        for (int k = 0; k < 40; ++k) check(step(s, win(0.6, 0.0f), t += kWin) == kStay && !s.resting, "a rest with no cap", k);
    }
    // A cap holding the frame: the second window in a row begins the rest, not the first.
    {
        State s;
        check(step(s, win(0.3), t += kWin) == kStay, "a rest after one window", 0);
        check(step(s, win(0.3), t += kWin) == kRest && s.resting, "no rest after two", 1);
        // Resting on while the limiter waits 8% or more and the command processor keeps up.
        check(step(s, win(0.10), t += kWin) == kStay && s.resting, "woken at a 10% wait", 2);
        check(step(s, win(0.3, 60.0f, 0.84), t += kWin) == kStay && s.resting, "woken at 84% busy", 3);
        // The command processor busy 85% of a window ends it.
        check(step(s, win(0.3, 60.0f, 0.86), t += kWin) == kWake && !s.resting, "not woken at 86% busy", 4);
    }
    // Just under the rest's wait never rests; the wake's wait ends one; a loading screen (no cap) ends one.
    {
        State s;
        for (int k = 0; k < 10; ++k) check(step(s, win(0.149), t += kWin) == kStay, "a rest at a 14.9% wait", k);
        step(s, win(0.15), t += kWin);
        check(step(s, win(0.15), t += kWin) == kRest, "no rest at 15%", 10);
        check(step(s, win(0.079), t += kWin) == kWake, "not woken at a 7.9% wait", 11);
        State u;
        step(u, win(0.3), t += kWin);
        step(u, win(0.3), t += kWin);
        check(u.resting && step(u, win(0.3, 0.0f), t += kWin) == kWake, "not woken with no cap", 12);
    }
    // A window without the wait between two with it starts the count again.
    {
        State s;
        step(s, win(0.3), t += kWin);
        check(step(s, win(0.01), t += kWin) == kStay && step(s, win(0.3), t += kWin) == kStay, "a rest across a window without the wait", 0);
        check(step(s, win(0.3), t += kWin) == kRest, "no rest after two again", 1);
    }
    // A short rest makes the next wait: 10 s after the first (5 doubled), then 20, 40, 80, 160, 160.
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
            check(step(s, win(0.01), t) == kWake, "not woken", round);
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
        check(step(s, win(0.01), t) == kWake && s.backoff_ns == gxvk_rest::kBackoffFirstNs && s.allowed_at == t + gxvk_rest::kBackoffFirstNs,
              "the wait after a long rest", 8);
    }
    if (g_failures) {
        std::printf("gxvk_rest_test: %d failures\n", g_failures);
        return 1;
    }
    std::printf("gxvk_rest_test: ok\n");
    return 0;
}
