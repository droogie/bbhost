// The uncapped mode's frame limit (engine/uncapped_limit.h): the V-Sync
// setting, the world and BBHOST_UNCAPPED_FPS each decide it as documented.
#include "engine/uncapped_limit.h"

#include <cstdio>

namespace {
int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);      \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)
}  // namespace

int main() {
    using namespace uncapped_limit;
    // The title and the main menu: uncapped, V-Sync off, no world, no load.
    CHECK(unlimited(0, 0, false, false, false, false));
    // V-Sync turned on (live, from F10 or PC Settings): a limit again.
    CHECK(!unlimited(0, 0, true, false, false, false));
    // In the world, or on a loading screen (its own rule): a limit.
    CHECK(!unlimited(0, 0, false, true, false, false));
    CHECK(!unlimited(0, 0, false, false, true, false));
    // Character creation (at the title, no world): its preview turns a step a frame.
    CHECK(!unlimited(0, 0, false, false, false, true));
    // BBHOST_UNCAPPED_FPS sets the limit everywhere.
    CHECK(!unlimited(0, 144, false, false, false, false));
    // The fixed rates never run unlimited.
    for (const int fps : {30, 60, 90, 120}) CHECK(!unlimited(fps, 0, false, false, false, false));

    // The limit where one applies.
    CHECK(limit_fps(0, false, 144.0f) == 240.0f);  // V-Sync off: the world's 240
    CHECK(limit_fps(0, true, 144.0f) == 144.0f);   // V-Sync on: the refresh rate
    CHECK(limit_fps(0, true, 360.0f) == 240.0f);   // ... no more than 240
    CHECK(limit_fps(0, true, 0.0f) == 240.0f);     // a display that does not say
    CHECK(limit_fps(100, true, 60.0f) == 100.0f);  // BBHOST_UNCAPPED_FPS wins
    CHECK(forced_fps(0) == 0 && forced_fps(-5) == 0);
    CHECK(forced_fps(10) == 30 && forced_fps(1000) == kWorldMaxFps && forced_fps(144) == 144);
    if (!g_fail) std::printf("uncapped_limit_test: ok\n");
    return g_fail ? 1 : 0;
}
