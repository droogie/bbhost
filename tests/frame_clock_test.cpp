// The frame clock (engine/frame_clock.h): off until switched on; before
// anything publishes - the game's own 30 - it holds the game's 1/30 and
// exactly one 30 fps frame; each publish holds that frame's step and the step
// in 30 fps frames, rounded once from double, and counts the frame; and the
// game's 1/30 is exactly 1 frame whoever publishes it, so a value converted by
// the clock is the game's own there.
#include "engine/frame_clock.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>

namespace {
int g_fail = 0;
#define CHECK(c)                                                     \
    do {                                                             \
        if (!(c)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                \
        }                                                            \
    } while (0)
}  // namespace

int main() {
    using namespace frame_clock;
    // Off until BBHOST_FRAME_CLOCK=1 turns it on.
    CHECK(!on());
    set_on(true);
    CHECK(on());
    set_on(false);
    CHECK(!on());
    // The game's own 30: nothing published.
    CHECK(std::bit_cast<std::uint32_t>(dt()) == kThirtiethBits);
    CHECK(frames30() == 1.0f);
    CHECK(g_seq.load() == 0);

    // 1/30 is one frame exactly, published or not.
    publish(kThirtieth);
    CHECK(frames30() == 1.0f && dt() == kThirtieth && g_seq.load() == 1);
    // The rates' frames: 30 x dt, rounded once.
    const float kSteps[] = {1.0f / 60.0f,  1.0f / 90.0f,  1.0f / 120.0f, 1.0f / 144.0f, 1.0f / 150.0f, 1.0f / 165.0f,
                            1.0f / 240.0f, 1.0f / 360.0f, 1.0f / 480.0f, 1.0f / 960.0f, 1.0f / 2000.0f};
    for (const float s : kSteps) {
        publish(s);
        CHECK(dt() == s);
        CHECK(frames30() == static_cast<float>(30.0 * static_cast<double>(s)));
    }
    publish(1.0f / 60.0f);
    CHECK(frames30() == 0.5f);  // 30 x f32(1/60) is within half an ulp of 0.5
    // Any step between 1/20000 and 1/30.
    std::mt19937 g(0xc10c);
    std::uniform_real_distribution<float> u(1.0f / 20000.0f, kThirtieth);
    for (int i = 0; i < 100000; ++i) {
        const float s = u(g);
        CHECK(frames30_of(s) == (std::bit_cast<std::uint32_t>(s) == kThirtiethBits ? 1.0f : static_cast<float>(30.0 * static_cast<double>(s))));
        CHECK(frames30_of(s) > 0.0f && frames30_of(s) <= 1.0f);
    }
    CHECK(frames30_of(kThirtieth) == 1.0f);
    CHECK(g_seq.load() == 1 + sizeof kSteps / sizeof kSteps[0] + 1);
    if (!g_fail) std::printf("frame_clock_test: ok\n");
    return g_fail ? 1 : 0;
}
