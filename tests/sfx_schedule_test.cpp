// The effects' step schedule (engine/sfx_schedule.h): whole 1/30 s or 1/60 s
// steps at each rate the game runs, the frames between them, the long-run
// count against the time that passed, and the carry's bounds.
#include "engine/sfx_schedule.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

namespace {
int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);      \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

std::uint32_t bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// The whole steps of `frames` frames of `seconds` each, as a string of digits.
std::string pattern(double seconds, int rate, int frames, double carry = 0.0) {
    std::string s;
    for (int i = 0; i < frames; ++i) s += static_cast<char>('0' + sfx_schedule::next(carry, seconds, rate).whole);
    return s;
}

// Over `frames` frames of `seconds`: every frame within its bounds, and the
// whole steps the time that passed holds, give or take the one in the carry.
void steady(double seconds, int rate, int frames) {
    double carry = 0.0, time = 0.0;
    long steps = 0;
    for (int i = 0; i < frames; ++i) {
        const sfx_schedule::Frame f = sfx_schedule::next(carry, seconds, rate);
        CHECK(f.whole >= 0 && f.whole <= sfx_schedule::most(rate));
        CHECK(f.seconds == static_cast<float>(static_cast<double>(f.whole) / rate));
        CHECK(carry > -1e-6 && carry < 1.0 / rate + 1.0 / 30.0);
        steps += f.whole;
        time += seconds > 1.0 / 30.0 ? 1.0 / 30.0 : seconds;
    }
    const double want = time * rate;
    CHECK(steps >= static_cast<long>(want) - 1 && steps <= static_cast<long>(want) + 1);
}
}  // namespace

int main() {
    using sfx_schedule::next;
    // 30 fps: a whole 1/30 s every frame, the game's own 1/30 (0x3d088889).
    CHECK(pattern(1.0 / 30, 30, 8) == "11111111");
    {
        double c = 0.0;
        CHECK(bits(next(c, 1.0 / 30, 30).seconds) == 0x3d088889u);
        CHECK(bits(next(c, static_cast<float>(1.0 / 30), 30).seconds) == 0x3d088889u);  // the game's float step
    }
    // 60 fps: every other frame (the frames between: 0); 120: every fourth.
    CHECK(pattern(1.0 / 60, 30, 8) == "01010101");
    CHECK(pattern(1.0 / 120, 30, 8) == "00010001");
    // Whole 1/60 s steps: every frame at 60, every other at 120, two at 30.
    CHECK(pattern(1.0 / 60, 60, 6) == "111111");
    CHECK(pattern(1.0 / 120, 60, 6) == "010101");
    CHECK(pattern(1.0 / 30, 60, 4) == "2222");
    {
        double c = 0.0;
        CHECK(bits(next(c, 1.0 / 60, 60).seconds) == 0x3c888889u);  // 1/60
    }
    // 144: a whole 1/30 s every 4.8 frames - 30 a second, never two in a row.
    {
        const std::string p = pattern(1.0 / 144, 30, 144 * 10);
        long ones = 0;
        for (char ch : p) ones += ch == '1';
        CHECK(ones == 300);
        CHECK(p.find("11") == std::string::npos);
    }
    // Longer than 1/30 s (a slow frame; lavapipe's): one whole step, nothing carried.
    CHECK(pattern(0.25, 30, 4) == "1111");
    CHECK(pattern(0.25, 60, 3) == "222");
    steady(0.25, 30, 1000);
    steady(0.1, 60, 1000);
    // Nothing for a frame that is not above 0, and the steps go on after it.
    CHECK(pattern(0.0, 30, 3) == "000");
    for (const double bad : {-1.0, std::nan("")}) {
        double c = 0.0;
        CHECK(next(c, bad, 30).whole == 0);
        CHECK(pattern(1.0 / 30, 30, 3, c) == "111");
    }
    // A carry a rounding short of a step and a whole 1/30 s frame: still the
    // one 1/30 s a frame (the 1e-6 would make it two), the rest carried.
    {
        double c = 1.0 / 30 - 1e-9;
        CHECK(next(c, 1.0 / 30, 30).whole == 1);
        CHECK(next(c, 1.0 / 30, 30).whole == 1);
        c = 1.0 / 60 - 1e-9;
        CHECK(next(c, 1.0 / 30, 60).whole == 2);
    }
    // The long-run count at steady rates, one rounding short of a step each.
    for (const double fps : {30.0, 45.0, 60.0, 90.0, 120.0, 144.0, 165.0, 240.0, 1000.0}) {
        steady(1.0 / fps, 30, 200000);
        steady(1.0 / fps, 60, 200000);
    }
    // BBHOST_SFX_SIM_FPS's float frame time (1/N as a double of a float).
    steady(static_cast<float>(1.0 / 60), 30, 200000);
    steady(static_cast<float>(1.0 / 144), 60, 200000);
    // Uncapped with jitter: frame times from 1/240 s to 1/20 s, a million of them.
    for (const int rate : {30, 60}) {
        std::mt19937 rng(20261019u + rate);
        std::uniform_real_distribution<double> ft(1.0 / 240, 1.0 / 20);
        double carry = 0.0, time = 0.0;
        long steps = 0, between = 0, run = 0, longest = 0;
        for (int i = 0; i < 1000000; ++i) {
            const double s = ft(rng);
            const sfx_schedule::Frame f = next(carry, s, rate);
            CHECK(f.whole <= sfx_schedule::most(rate));
            CHECK(carry > -1e-6 && carry < 1.0 / rate + 1.0 / 30.0);
            steps += f.whole;
            time += s > 1.0 / 30 ? 1.0 / 30 : s;
            if (f.whole == 0) {
                ++between;
                if (++run > longest) longest = run;
            } else {
                run = 0;
            }
        }
        CHECK(steps >= static_cast<long>(time * rate) - 1 && steps <= static_cast<long>(time * rate) + 1);
        CHECK(between > 0);
        CHECK(longest <= (rate == 30 ? 7 : 3));  // at 240 fps the most frames between whole steps
    }
    if (!g_fail) std::printf("sfx_schedule_test: ok\n");
    return g_fail ? 1 : 0;
}
