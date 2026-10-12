#include "engine/frame_pacing.h"

#include <cmath>
#include <cstdio>

namespace {
int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
bool near(float a, float b) { return std::fabs(a - b) < 1e-7f; }
}

int main() {
    using namespace frame_pacing;
    // The reported title-menu workload takes ~357 us at 2800 fps. Uncapped
    // with V-Sync off must not request the old 4167 us (240 fps) interval.
    const float unlimited = target_for_limit(uncapped_limit(0, false, 240.0f));
    CHECK(unlimited > 0.0f);  // zero would select the game's 30 Hz mode
    CHECK(unlimited < 1.0f / 2800.0f);
    CHECK(near(target_for_limit(uncapped_limit(0, true, 144.0f)), 1.0f / 144.0f));
    CHECK(near(target_for_limit(uncapped_limit(0, true, 360.0f)), 1.0f / 360.0f));
    CHECK(near(target_for_limit(uncapped_limit(0, true, 0.0f)), unlimited));
    CHECK(near(target_for_limit(uncapped_limit(240, false, 0.0f)), 1.0f / 240.0f));
    CHECK(near(target_for_limit(uncapped_limit(500, true, 60.0f)), 1.0f / 500.0f));
    CHECK(uncapped_limit(1, false, 0.0f) == 30.0f);
    CHECK(uncapped_limit(3000, false, 0.0f) == 1000.0f);
    CHECK(uncapped_limit(-1, false, 0.0f) == 0.0f);
    // A floor for calibrated simulation values is not a pacing limit.
    CHECK(near(step_floor(unlimited), 0.001f));
    CHECK(near(step_floor(1.0f / 60.0f), 1.0f / 60.0f));
    CHECK(kSoundHz == 240);
    return g_fail ? 1 : 0;
}
