// Frame generation's pace (engine/fg_pace.h): the real frames held to half
// the display's refresh rate while frames are generated - Display-Limited
// included - and the held pace for a cap the frames do not reach, across the
// refresh rates players have (and the ones no display should say), the
// loading screen's edge, a change of the upscaler's settings and frames of
// every length. Whatever the inputs, the limiter's target is a finite length
// no longer than the game's own 1/30 s (or the rate's own below 30), so the
// pace cannot hold a frame for ever.
#include "engine/fg_pace.h"
#include "engine/uncapped_limit.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {
int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);      \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

// The rates a player can pick: the list's, a Custom limit's 10 and 45, and
// 0 (the game's own 30, which has no target of ours).
const int kRates[] = {0, 10, 30, 45, 60, 90, 120, 150, 180, 240, 360, 480, 960};
// Displays: none (headless), RDP's 32 Hz, under and at the limit's floor, the
// common ones, and what no display should say.
const float kHz[] = {0.0f, 30.0f, 32.0f, 47.9f, 48.0f, 59.94f, 60.0f, 75.0f, 119.88f, 120.0f, 143.98f, 144.0f, 165.0f,
                     180.0f, 240.0f, 360.0f, 1000.0f, kNaN, kInf, -1.0f};

bool finite(float v) { return std::isfinite(v); }

// The cap for every rate, display and edge.
void cap_sweep() {
    for (const float hz : kHz) {
        // Display-Limited is the display's rate (engine/uncapped_limit.h), 60 with no display that says one.
        std::vector<float> rates;
        rates.push_back(1.0f / static_cast<float>(uncapped_limit::rate_fps(0, hz)));
        for (const int r : kRates) rates.push_back(r ? 1.0f / static_cast<float>(r) : 0.0f);
        for (const float rate : rates) {
            for (int g = 0; g < 2; ++g) {
                for (int l = 0; l < 2; ++l) {
                    const bool generating = g != 0, loading = l != 0;
                    const float cap = fg_pace::cap_s(rate, generating, loading, hz);
                    CHECK(finite(cap) && cap >= 0.0f);
                    if (rate <= 0.0f) {
                        CHECK(cap == 0.0f);  // the game's own 30: no target of ours
                        continue;
                    }
                    CHECK(cap >= rate);                                 // never faster than the rate
                    CHECK(cap <= std::max(rate, 2.0f / 48.0f));         // never slower than two refreshes of the slowest display believed
                    const bool holds = generating && !loading && hz >= fg_pace::kLeastHz;
                    if (!holds) CHECK(cap == rate);  // off on a loading screen, without generation, without a display
                    if (holds && finite(hz)) {
                        CHECK(cap == std::max(rate, 2.0f / hz));
                        CHECK(1.0f / cap <= hz / 2.0f * 1.0001f);  // real frames at most half the refresh rate
                    }
                }
            }
        }
    }
}

// Display-Limited with frame generation: real frames at half the refresh
// rate, shown at the refresh rate - and a rate under half left as it is.
void display_limited() {
    const float hz_list[] = {59.94f, 60.0f, 75.0f, 120.0f, 143.98f, 144.0f, 165.0f, 240.0f, 360.0f};
    for (const float hz : hz_list) {
        const float rate = 1.0f / static_cast<float>(uncapped_limit::rate_fps(0, hz));
        const float cap = fg_pace::cap_s(rate, true, false, hz);
        const float shown = 2.0f / cap;
        CHECK(std::fabs(shown - hz) <= hz * 0.0001f);  // the refresh rate, half of it generated
        CHECK(fg_pace::cap_s(rate, true, true, hz) == rate);  // a loading screen: the rate's own
        CHECK(fg_pace::cap_s(rate, false, false, hz) == rate);  // no generation: the rate's own
    }
    CHECK(fg_pace::cap_s(1.0f / 60.0f, true, false, 144.0f) == 1.0f / 60.0f);  // 60 at 144 Hz: 60 real, 120 shown
    CHECK(fg_pace::cap_s(1.0f / 960.0f, true, false, 144.0f) == 2.0f / 144.0f);  // 960 at 144 Hz: 72 real
    CHECK(fg_pace::cap_s(1.0f / 60.0f, true, false, 32.0f) == 1.0f / 60.0f);  // RDP's 32 Hz: the cap alone
    CHECK(fg_pace::cap_s(1.0f / 60.0f, true, false, 0.0f) == 1.0f / 60.0f);   // headless: the cap alone
}

void target() {
    CHECK(fg_pace::target_s(0.0f, 0.02f) == 0.0f);  // no cap, no target, whatever is held
    CHECK(fg_pace::target_s(0.01f, 0.0f) == 0.01f);
    CHECK(fg_pace::target_s(0.01f, 0.02f) == 0.02f);
    CHECK(fg_pace::target_s(0.02f, 0.01f) == 0.02f);
    CHECK(fg_pace::target_s(0.01f, kNaN) == 0.01f);
    CHECK(fg_pace::target_s(0.01f, -1.0f) == 0.01f);
}

// A machine that takes `work_s` a frame: the limiter waits out the rest of
// the target, so the frame is the longer of the two.
struct Machine {
    std::uint64_t now_ns = 1000000000ull;
    double work_s = 0.010;
    // One frame under `target`; returns the limiter's wait.
    std::uint64_t frame(float target) {
        const double interval = std::max(static_cast<double>(target), work_s);
        const double wait = std::max(0.0, static_cast<double>(target) - work_s);
        now_ns += static_cast<std::uint64_t>(interval * 1e9);
        return static_cast<std::uint64_t>(wait * 1e9);
    }
};

// What every frame must keep, whatever came before.
void invariants(const fg_pace::Pace& p, float cap) {
    CHECK(finite(p.held_s) && p.held_s >= 0.0f && p.held_s <= fg_pace::kThirtieth);
    if (p.held_s > 0.0f) CHECK(p.held_s > cap * 1.01f);  // a pace held only where the cap is not reached
    const float t = fg_pace::target_s(cap, p.held_s);
    CHECK(finite(t) && t >= cap && t <= std::max(cap, fg_pace::kThirtieth));
}

// Runs `n` frames at `hz` under the cap Display-Limited makes, the loading
// screen up or not; returns the last target.
float run(fg_pace::Pace& p, Machine& m, int n, float hz, bool generating, bool loading, std::uint64_t key = 1,
          int* windows = nullptr, int* let_go = nullptr) {
    const float rate = 1.0f / static_cast<float>(uncapped_limit::rate_fps(0, hz));
    float t = 0.0f;
    for (int k = 0; k < n; ++k) {
        const float cap = fg_pace::cap_s(rate, generating, loading, hz);
        t = loading ? 0.0f : fg_pace::target_s(cap, p.held_s);  // a loading screen runs with no target
        const std::uint64_t wait = m.frame(t);
        const bool pacing = cap > 0.0f && generating && !loading;
        const fg_pace::Step s = fg_pace::note(p, m.now_ns, wait, loading ? 0.0f : cap, pacing, key, 1.03);
        if (windows && s.kind == fg_pace::Step::Window) ++*windows;
        if (let_go && (s.kind == fg_pace::Step::LetGo || s.kind == fg_pace::Step::LetGoSettings)) ++*let_go;
        invariants(p, cap);
    }
    return t;
}

void held_pace() {
    // Reached: 144 Hz, frames of 8 ms under the 1/72 s cap - nothing held.
    {
        fg_pace::Pace p;
        Machine m;
        m.work_s = 0.008;
        const float t = run(p, m, 2000, 144.0f, true, false);
        CHECK(p.held_s == 0.0f);
        CHECK(std::fabs(t - 2.0f / 144.0f) < 1e-7f);
    }
    // Not reached: 1/72 s asked, 20 ms taken - held a little over what they
    // take (3% over, tried 2% faster now and then, and brought down at once
    // when the limiter waits a tenth of the frame).
    {
        fg_pace::Pace p;
        Machine m;
        m.work_s = 0.020;
        run(p, m, 2000, 144.0f, true, false);
        CHECK(p.held_s >= 0.020f * 0.97f && p.held_s <= 0.020f * 1.12f);
    }
    // Slower than the game's own frame: held at 1/30 s at most, the frames as they come.
    {
        fg_pace::Pace p;
        Machine m;
        m.work_s = 0.050;
        const float t = run(p, m, 2000, 144.0f, true, false);
        CHECK(p.held_s == fg_pace::kThirtieth);
        CHECK(t <= fg_pace::kThirtieth);
    }
    // The loading screen's edge: a held pace goes as the screen comes up, a
    // load's long first frame is no frame to pace by, and the pace comes back
    // after it; no target while it is up.
    {
        fg_pace::Pace p;
        Machine m;
        m.work_s = 0.020;
        run(p, m, 1000, 144.0f, true, false);
        CHECK(p.held_s > 0.0f);
        int let_go = 0;
        const float during = run(p, m, 300, 144.0f, true, true, 1, nullptr, &let_go);
        CHECK(during == 0.0f && p.held_s == 0.0f && p.n == 0 && let_go == 1);
        m.now_ns += 5000000000ull;  // the load's last frame and the world's first, 5 s apart
        m.work_s = 0.008;           // the world after it is lighter
        // Its first window: the 5 s between the two is not one of its frames
        // (held as a 1/30 s pace it would have been).
        run(p, m, static_cast<int>(fg_pace::kWindow) + 1, 144.0f, true, false);
        CHECK(p.held_s == 0.0f);
        int windows = 0;
        const float after = run(p, m, 1000, 144.0f, true, false, 1, &windows);
        CHECK(windows > 0 && p.held_s == 0.0f && std::fabs(after - 2.0f / 144.0f) < 1e-7f);
        // The same with the world heavier after the load: held again.
        run(p, m, 300, 144.0f, true, true);
        m.now_ns += 5000000000ull;
        m.work_s = 0.025;
        run(p, m, 1000, 144.0f, true, false);
        CHECK(p.held_s >= 0.025f * 0.97f && p.held_s <= 0.025f * 1.12f);
    }
    // Frame generation stops (the swapchain made again, a toggle): let go at once.
    {
        fg_pace::Pace p;
        Machine m;
        m.work_s = 0.020;
        run(p, m, 1000, 144.0f, true, false);
        int let_go = 0;
        run(p, m, 10, 144.0f, false, false, 1, nullptr, &let_go);
        CHECK(p.held_s == 0.0f && let_go == 1);
    }
    // A change of the upscaler's settings: let go, the next 64 frames skipped.
    {
        fg_pace::Pace p;
        Machine m;
        m.work_s = 0.020;
        run(p, m, 1000, 144.0f, true, false, 1);
        int let_go = 0, windows = 0;
        run(p, m, static_cast<int>(fg_pace::kSkipAfterChange), 144.0f, true, false, 2, &windows, &let_go);
        CHECK(let_go == 1 && windows == 0 && p.held_s == 0.0f);
        run(p, m, 200, 144.0f, true, false, 2);
        CHECK(p.held_s > 0.0f);
    }
}

// Frames of any length, loading screens, generation and settings coming and
// going, at every display: the invariants hold on every frame.
void fuzz() {
    std::uint64_t rng = 0x9e3779b97f4a7c15ull;
    const auto next = [&rng] {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };
    for (const float hz : kHz) {
        fg_pace::Pace p;
        Machine m;
        bool loading = false, generating = true;
        std::uint64_t key = 1;
        for (int k = 0; k < 20000; ++k) {
            if (next() % 500 == 0) loading = !loading;
            if (next() % 2000 == 0) generating = !generating;
            if (next() % 3000 == 0) ++key;
            // 0.1 ms to 0.3 s, now and then a stall of seconds.
            m.work_s = 0.0001 + static_cast<double>(next() % 300000) * 1e-6;
            if (next() % 1000 == 0) m.work_s = 3.0;
            run(p, m, 1, hz, generating, loading, key);
        }
    }
}
}  // namespace

int main() {
    cap_sweep();
    display_limited();
    target();
    held_pace();
    fuzz();
    if (!g_fail) std::printf("fg_pace_test: ok\n");
    return g_fail ? 1 : 0;
}
