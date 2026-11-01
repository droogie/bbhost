// The frame clock (engine/frame_clock.h): off until switched on; before
// anything publishes - the game's own 30 - it holds the game's 1/30 and
// exactly one 30 fps frame; each publish holds that frame's step and the step
// in 30 fps frames, rounded once from double, and counts the frame; and the
// game's 1/30 is exactly 1 frame whoever publishes it, so a value converted by
// the clock is the game's own there; and the 30 fps frames a frame begins are
// 1 a frame at 30 and, at any other frame length, one in each frame where a
// 30 fps frame's time starts, never two.
#include "engine/frame_clock.h"

#include <bit>
#include <cmath>
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
    // Off until BBHOST_FRAME_CLOCK=1 turns it on; not published without it
    // until the frame-rate code says something counts game time by default.
    CHECK(!on());
    set_on(true);
    CHECK(on());
    set_on(false);
    CHECK(!on());
    CHECK(!publishing());
    set_publishing(true);
    CHECK(publishing() && !on());
    set_publishing(false);
    CHECK(!publishing());
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

    // The 30 fps frames a frame begins. The game's own 30, and any run of
    // 1/30 published: one every frame, nothing carried.
    g_ticks30.store(1);
    g_tick_carry = 0.0;
    for (int i = 0; i < 100000; ++i) {
        publish(kThirtieth);
        CHECK(ticks30() == 1);
    }
    CHECK(g_tick_carry == 0.0);
    // A rate's frames: the first frame begins one, then one each 30 fps frame
    // of steps, never two in a frame - 1/60 on the first, the third, ...
    g_tick_carry = 0.0;
    for (int i = 0; i < 1000; ++i) {
        publish(1.0f / 60.0f);
        CHECK(ticks30() == (i % 2 == 0 ? 1 : 0));
    }
    // Every whole number the steps' sum passes, counted from 0, ticked once,
    // in the frame whose sum before it is at or below it and after above it
    // (a sum within 1e-9 of a whole number may tick either side of it).
    const auto begun = [](double before, double after, int ticks) {
        const auto near_whole = [](double x) { return std::fabs(x - std::round(x)) < 1e-9; };
        if (near_whole(before) || near_whole(after)) return ticks == 0 || ticks == 1;  // either side of it
        return ticks == static_cast<int>(std::ceil(after) - std::ceil(before));       // the whole numbers in [before, after)
    };
    for (const float s : kSteps) {
        g_tick_carry = 0.0;
        double frames = 0.0;
        long ticks = 0;
        for (int i = 0; i < 20000; ++i) {
            publish(s);
            const double before = frames;
            frames += static_cast<double>(frames30());
            CHECK(ticks30() == 0 || ticks30() == 1);
            CHECK(begun(before, frames, ticks30()));
            ticks += ticks30();
        }
        CHECK(std::fabs(static_cast<double>(ticks) - std::ceil(frames)) <= 1.0);
    }
    // Frames of any length to 1/30, mixed: the same.
    {
        g_tick_carry = 0.0;
        double frames = 0.0;
        long ticks = 0;
        for (int i = 0; i < 200000; ++i) {
            publish(i % 7 == 0 ? kThirtieth : u(g));
            const double before = frames;
            frames += static_cast<double>(frames30());
            CHECK(ticks30() == 0 || ticks30() == 1);
            CHECK(begun(before, frames, ticks30()));
            ticks += ticks30();
        }
        CHECK(std::fabs(static_cast<double>(ticks) - std::ceil(frames)) <= 1.0);
    }
    g_dt.store(kThirtieth);
    g_frames30.store(1.0f);
    g_ticks30.store(1);
    g_tick_carry = 0.0;
    if (!g_fail) std::printf("frame_clock_test: ok\n");
    return g_fail ? 1 : 0;
}
