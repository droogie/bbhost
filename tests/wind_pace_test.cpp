// The wind's pace above 30 fps (engine/wind_pace.h): the wind time a game
// second and a push's fade at 60, 90, 120, 144 and uncapped's uneven frames
// against the game's at 30 (its fixed 1/60 a 1/30 s frame); the displacements
// the frame-rate code writes, against the game's own bytes at the two loads;
// and each mutant caught.
#define WIND_PACE_MUTANTS
int g_wind_pace_mutant = 0;
#include "engine/wind_pace.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

int g_fail = 0;
bool g_quiet = false;
void check(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        if (!g_quiet) std::printf("FAIL: %s\n", what);
    }
}

std::uint32_t g_rng = 0x2713870u;
double uniform(double lo, double hi) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return lo + (hi - lo) * ((g_rng >> 8) / double(1u << 24));
}

// 20 game seconds of frames: the wind time and the frames a push's 0.75 s of
// life lasts, the life falling by the wind's step a frame as sub_2713870's does.
struct Pace {
    double wind_s = 0.0, game_s = 0.0, fade_s = 0.0;
};
template <class Dt>
Pace run(Dt dt_of, int frames, bool game_at_30) {
    Pace p;
    double life = 0.75;
    bool faded = false;
    for (int i = 0; i < frames; ++i) {
        const float step = static_cast<float>(dt_of(i));
        const float wind = game_at_30 ? wind_pace::kStepAtThirty : wind_pace::step_for(step);
        p.wind_s += wind;
        p.game_s += step;
        life -= wind;
        if (!faded && life <= 0.0) {
            faded = true;
            p.fade_s = p.game_s;
        }
    }
    return p;
}

void paces() {
    // the game at 30: 1/60 of wind a frame - half a second of wind a game second, a push fading in 1.5 s
    const Pace ref = run([](int) { return 1.0 / 30; }, 600, true);
    if (!g_quiet) std::printf("  the game at 30: %.4f s of wind a game second, a push fades in %.3f game s\n", ref.wind_s / ref.game_s, ref.fade_s);
    const double rates[] = {60, 90, 120, 144};
    for (double fps : rates) {
        const Pace p = run([fps](int) { return 1.0 / fps; }, static_cast<int>(20 * fps), false);
        if (!g_quiet)
            std::printf("  ours at %g: %.4f s of wind a game second, a push fades in %.3f game s\n", fps, p.wind_s / p.game_s, p.fade_s);
        check(std::fabs(p.wind_s / p.game_s - ref.wind_s / ref.game_s) < 1e-4, "the wind's pace as at 30");
        check(std::fabs(p.fade_s - ref.fade_s) <= 1.0 / fps + 1e-6, "a push's fade as at 30, within a frame");
    }
    const Pace u = run([](int) { return uniform(1.0 / 240, 1.0 / 30); }, 5000, false);
    check(std::fabs(u.wind_s / u.game_s - ref.wind_s / ref.game_s) < 1e-4, "the wind's pace as at 30 at uneven frames");
}

void displacements() {
    std::int32_t d = 0;
    // the game's own load at 0x2713d8d (c5fa1025 db606102): its 1/60 at 0x4d29e70
    check(wind_pace::rel32_to(0x2713d8d, 8, 0x4d29e70, &d) && d == 0x026160db, "0x2713d8d's displacement to the game's 1/60");
    // the frame-time site there (c5fa1025 736b0403): the slot 0x575a908
    check(wind_pace::rel32_to(0x2713d8d, 8, 0x575a908, &d) && d == 0x03046b73, "0x2713d8d's displacement to the frame-time slot");
    // the force's site at 0x27149ea (c57a1035 72196102): the authored 1/30 at 0x4d26364; ours, the game's 1/60
    check(wind_pace::rel32_to(0x27149ea, 8, 0x4d26364, &d) && d == 0x02611972, "0x27149ea's displacement to 1/30");
    check(wind_pace::rel32_to(0x27149ea, 8, 0x4d29e70, &d) && d == 0x0261547e, "0x27149ea's displacement to the game's 1/60");
    check(!wind_pace::rel32_to(0x400000, 8, 0x400000ull + (1ull << 33), &d), "a target past 2 GiB refused");
}

int run_all() {
    g_fail = 0;
    paces();
    displacements();
    return g_fail;
}

}  // namespace

int main() {
    std::printf("wind_pace_test: the wind at 30's pace above 30\n");
    const int fails = run_all();
    static const char* const kMutants[] = {"", "the frame's step", "the game's fixed 1/60 a frame", "a displacement from the instruction's start"};
    int caught = 0;
    g_quiet = true;
    for (int m = 1; m <= 3; ++m) {
        g_wind_pace_mutant = m;
        const int f = run_all();
        caught += f > 0;
        std::printf("  mutant %d (%s): %s\n", m, kMutants[m], f ? "caught" : "NOT CAUGHT");
    }
    g_wind_pace_mutant = 0;
    std::printf("wind_pace_test: %s (%d failures; %d of 3 mutants caught)\n", fails == 0 && caught == 3 ? "pass" : "FAIL", fails, caught);
    return fails == 0 && caught == 3 ? 0 : 1;
}
