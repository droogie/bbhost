// Mouse sensitivity's scale (engine/mouse_camera_step.h): 0.005 to 0.4
// degrees a count, each step of the 0-100 setting the same factor further than
// the one below, 0.045 at the default 50 - and a setting saved on Dark Souls
// III's old scale (0.1 to 0.4) read back as the same angle a count.
#include "engine/mouse_camera_step.h"

#include <cmath>
#include <cstdio>

namespace {
int g_fail = 0;
#define CHECK(c)                                                     \
    do {                                                             \
        if (!(c)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                \
        }                                                            \
    } while (0)

bool near(float a, float b, float tolerance) { return std::fabs(a - b) <= tolerance; }
}  // namespace

int main() {
    using namespace mouse_camera;
    // The two ends and the default.
    CHECK(near(degrees_per_count(0.0f), 0.005f, 1e-6f));
    CHECK(near(degrees_per_count(10.0f), 0.4f, 1e-5f));
    CHECK(near(degrees_per_count(5.0f), 0.0447214f, 1e-5f));
    // Held to the ends outside 0..10.
    CHECK(degrees_per_count(-3.0f) == degrees_per_count(0.0f));
    CHECK(degrees_per_count(42.0f) == degrees_per_count(10.0f));
    // Each step of the setting the same factor further: 80 ^ 0.01, about 4.5%.
    float below = degrees_per_count(0.0f);
    for (int i = 1; i <= 100; ++i) {
        const float d = degrees_per_count(static_cast<float>(i) * 0.1f);
        CHECK(d > below);
        CHECK(near(d / below, 1.044794f, 2e-4f));
        below = d;
    }
    // Each whole number of the System menu's 0-10 slider about half again as far.
    CHECK(near(degrees_per_count(6.0f) / degrees_per_count(5.0f), 1.54987f, 1e-3f));
    // The way back.
    for (int i = 0; i <= 100; ++i) {
        const float s = static_cast<float>(i) * 0.1f;
        CHECK(near(sensitivity_for_degrees(degrees_per_count(s)), s, 1e-3f));
    }
    CHECK(sensitivity_for_degrees(0.0f) == 0.0f);
    CHECK(sensitivity_for_degrees(0.001f) == 0.0f);
    CHECK(sensitivity_for_degrees(1.0f) == 10.0f);
    // Dark Souls III's scale: 0.1 at 0, 0.25 at its default 5, 0.4 at 10.
    CHECK(near(ds3_degrees_per_count(0.0f), 0.1f, 1e-6f));
    CHECK(near(ds3_degrees_per_count(5.0f), 0.25f, 1e-6f));
    CHECK(near(ds3_degrees_per_count(10.0f), 0.4f, 1e-6f));
    // A setting saved on it turns as far as before, to the nearest step (half a
    // step is 2.2%): its slowest is today's 68, its fastest today's 100, and
    // its order is kept.
    CHECK(setting_from_ds3(0) == 68);
    CHECK(setting_from_ds3(50) == 89);
    CHECK(setting_from_ds3(100) == 100);
    for (int i = 0; i <= 100; ++i) {
        const int now = setting_from_ds3(i);
        const float was = ds3_degrees_per_count(static_cast<float>(i) * 0.1f);
        CHECK(now >= 68 && now <= 100);
        CHECK(near(degrees_per_count(static_cast<float>(now) * 0.1f), was, was * 0.023f));
        if (i > 0) CHECK(now >= setting_from_ds3(i - 1));
    }
    if (g_fail) {
        std::printf("mouse_sensitivity_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("mouse_sensitivity_test: ok\n");
    return 0;
}
