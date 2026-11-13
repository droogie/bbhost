// engine/look_math.h: what a plugin's look (post_look) writes into YEBIS's
// post-process block, without the game. The motion blur's sample count stays
// where YEBIS has shaders for it: 2 to 8, 7 where the area interleaves its
// samples (8 interleaved is 9).
#include "engine/look_math.h"

#include <cstdio>
#include <initializer_list>

static int g_failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

int main() {
    using look_math::blur_samples;
    // Inside: as asked, interleaved or not.
    for (int n = 2; n <= 7; ++n) {
        CHECK(blur_samples(n, false) == n);
        CHECK(blur_samples(n, true) == n);
    }
    CHECK(blur_samples(8, false) == 8);
    // The ceiling: 8, and 7 interleaved (YEBIS would make 8 a 9).
    CHECK(blur_samples(8, true) == 7);
    for (const int n : {9, 10, 16, 1000, 0x7fffffff}) {
        CHECK(blur_samples(n, false) == 8);
        CHECK(blur_samples(n, true) == 7);
    }
    // The floor: 1 and below are YEBIS's own 4 or 5, not one sample.
    for (const int n : {1, 0, -1, -0x7fffffff}) {
        CHECK(blur_samples(n, false) == 2);
        CHECK(blur_samples(n, true) == 2);
    }
    if (g_failures) {
        std::fprintf(stderr, "look_math_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("look_math_test: ok\n");
    return 0;
}
