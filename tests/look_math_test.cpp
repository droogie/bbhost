// engine/look_math.h: what a plugin's look (post_look) writes into YEBIS's
// post-process block, without the game. The motion blur's sample count stays
// where YEBIS has shaders for it: 2 to 8, 7 where the area interleaves its
// samples (8 interleaved is 9). The depth of field by distance makes the
// game's own ramps: sharp between the two distances, blurred in front and
// behind as strongly as asked, as the apply's loop fills its table.
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
    // The depth of field by distance: sharp between the two distances, the
    // blur in front negative (YEBIS's near side) and behind positive, full at
    // the fade's end, and as the apply's loop (0x25d9760) makes its table.
    {
        using look_math::dof_entry;
        using look_math::dof_range;
        const look_math::DofRange r = dof_range(3.0f, 10.0f, 1.0f, 5.0f, 4, 6);
        CHECK(r.near_on == 1 && r.far_on == 1);
        CHECK(r.near_start == 2.0f && r.near_end == 3.0f && r.far_start == 10.0f && r.far_end == 15.0f);
        CHECK(r.near_levels == 4 && r.far_levels == 6 && r.near_scale == 1.0f && r.far_scale == 1.0f);
        CHECK(r.near_size > 0.0f && r.far_size > r.near_size);  // a stronger blur is a wider circle
        CHECK(dof_entry(r, 1.0f) == -r.near_size);              // full in front of the fade
        CHECK(dof_entry(r, 2.5f) < 0.0f && dof_entry(r, 2.5f) > -r.near_size);  // fading out to Sharp from
        for (const float z : {3.0f, 5.0f, 9.99f, 10.0f}) CHECK(dof_entry(r, z) == 0.0f);  // sharp between
        CHECK(dof_entry(r, 12.5f) > 0.0f && dof_entry(r, 12.5f) < r.far_size);  // fading in past Sharp to
        CHECK(dof_entry(r, 15.0f) == r.far_size && dof_entry(r, 5000.0f) == r.far_size);  // full from the fade's end
        // Moved: the same circle, its edges where asked.
        const look_math::DofRange m = dof_range(1.0f, 40.0f, 0.0f, 0.0f, 4, 6);
        CHECK(dof_entry(m, 0.9f) == -m.near_size && dof_entry(m, 1.0f) == 0.0f);
        CHECK(dof_entry(m, 39.0f) == 0.0f && dof_entry(m, 40.5f) == m.far_size);
        CHECK(dof_entry(m, 12.5f) == 0.0f);  // sharp now where r blurred
        // No blur on a side at 0: none there, and nothing of the lens either
        // (the near ramp stays on, so the table never takes the lens's circle).
        const look_math::DofRange none = dof_range(2.0f, 10.0f, 0.5f, 5.0f, 0, 0);
        CHECK(none.near_on == 1 && none.far_on == 0 && none.near_size == 0.0f && none.near_scale == 0.0f);
        for (const float z : {0.1f, 1.0f, 5.0f, 50.0f, 5000.0f}) CHECK(dof_entry(none, z) == 0.0f);
        // Out of range asks: levels kept to 0..7, distances to 0 and up.
        const look_math::DofRange odd = dof_range(-5.0f, -1.0f, -2.0f, -3.0f, 99, -4);
        CHECK(odd.near_levels == 7 && odd.far_levels == 0 && odd.near_end == 0.0f && odd.near_start == 0.0f);
        CHECK(odd.far_start == 0.0f && odd.far_end == 0.0f && odd.cut > 1.0e6f);
        // Each level twice the last's circle, up to 7.
        for (int l = 2; l <= 7; ++l) CHECK(look_math::dof_coc(l) == 2.0f * look_math::dof_coc(l - 1));
        CHECK(look_math::dof_coc(0) == 0.0f && look_math::dof_coc(8) == look_math::dof_coc(7));
    }
    if (g_failures) {
        std::fprintf(stderr, "look_math_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("look_math_test: ok\n");
    return 0;
}
