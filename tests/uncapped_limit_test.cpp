// The Uncapped frame rate (engine/uncapped_limit.h): the display's refresh
// rate, BBHOST_UNCAPPED_FPS over it, 60 with no display.
#include "engine/uncapped_limit.h"

#include <cstdio>
#include <limits>

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
    // The display's rate, to the nearest whole frame a second.
    CHECK(rate_fps(0, 180.0f) == 180);
    CHECK(rate_fps(0, 179.98f) == 180);
    CHECK(rate_fps(0, 59.94f) == 60);
    CHECK(rate_fps(0, 143.98f) == 144);
    CHECK(rate_fps(0, 119.88f) == 120);
    CHECK(rate_fps(0, 165.0f) == 165);
    CHECK(rate_fps(0, 50.0f) == 50);
    // Within the rates the frame-rate code converts to.
    CHECK(rate_fps(0, 29.97f) == 30);
    CHECK(rate_fps(0, 24.0f) == kNoDisplayFps);  // below 30: as no rate
    CHECK(rate_fps(0, 1000.0f) == kMaxFps);
    // No display (headless), or one that does not say.
    CHECK(rate_fps(0, 0.0f) == kNoDisplayFps);
    CHECK(rate_fps(0, -1.0f) == kNoDisplayFps);
    CHECK(rate_fps(0, std::numeric_limits<float>::quiet_NaN()) == kNoDisplayFps);
    // BBHOST_UNCAPPED_FPS wins over the display.
    CHECK(rate_fps(100, 180.0f) == 100);
    CHECK(rate_fps(240, 0.0f) == 240);
    CHECK(forced_fps(0) == 0 && forced_fps(-5) == 0);
    CHECK(forced_fps(10) == 30 && forced_fps(2000) == kMaxFps && forced_fps(144) == 144);
    if (!g_fail) std::printf("uncapped_limit_test: ok\n");
    return g_fail ? 1 : 0;
}
