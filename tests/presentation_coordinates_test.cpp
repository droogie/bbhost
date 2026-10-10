#include "host/presentation_coordinates.h"
#include "host/display_settings.h"

#include <cassert>
#include <cmath>
#include <cstdio>

static void near(float a, float b) { assert(std::abs(a - b) < 0.002f); }

// Start at a known menu-row point, draw it into the centered stage and then
// into the swapchain/window. The cursor must recover both that picture pixel
// and that same menu row, independently of input resolution and DPI.
static void menu_point(float dw, float dh, float ww, float wh, float sw, float sh,
                       float sx, float sy) {
    const auto stage = host::fit_picture(dw, dh, 1920, 1080);
    const float px = stage.x + sx * stage.w / 1920;
    const float py = stage.y + sy * stage.h / 1080;
    const auto picture = host::fit_picture(sw, sh, dw, dh);
    const float wx = (picture.x + px * picture.w / dw) * ww / sw;
    const float wy = (picture.y + py * picture.h / dh) * wh / sh;
    float x = 0, y = 0, mx = 0, my = 0;
    assert(host::window_to_picture(wx, wy, ww, wh, sw, sh, dw, dh, x, y));
    near(x, px); near(y, py);
    assert(host::picture_to_menu_stage(x, y, dw, dh, mx, my));
    near(mx, sx); near(my, sy);
}

int main() {
    for (int mode = 0; mode < 6; ++mode) {
        const auto d = host::display_dimensions(3440, 1440, mode);
        for (float row : {150.0f, 540.0f, 975.0f}) {
            menu_point(d.output_width, d.output_height, 1920, 1080, 3840, 2160, 400, row);
            menu_point(d.output_width, d.output_height, 2560, 1600, 1920, 1080, 1500, row);
        }
        // F10 and the host cursor use the output canvas, not an SR input.
        float x = 0, y = 0;
        assert(host::window_to_picture(960, 540, 1920, 1080, 3840, 2160,
                                      d.output_width, d.output_height, x, y));
        near(x, 1720); near(y, 720);
    }
    menu_point(1920, 1080, 1920, 1080, 1920, 1080, 300, 800);
    menu_point(1280, 800, 1280, 800, 1280, 800, 500, 900);
    menu_point(3440, 1640, 1920, 1080, 3840, 2160, 1500, 950);
    float x = 0, y = 0;
    assert(!host::window_to_picture(960, 10, 1920, 1080, 3840, 2160, 3440, 1440, x, y));
    assert(!host::window_to_picture(10, 10, 1280, 800, 1280, 800, 1920, 1080, x, y));
    assert(!host::window_to_picture(0, 0, 0, 0, 0, 0, 1920, 1080, x, y));
    assert(!host::picture_to_menu_stage(0, 0, 0, 0, x, y));
    assert(host::window_to_picture(960, 540, 1920, 1080, 0, 0, 1920, 1080, x, y));
    near(x, 960); near(y, 540);
    // Outside the centered 16:9 stage, but still inside the ultrawide picture.
    assert(host::picture_to_menu_stage(0, 720, 3440, 1440, x, y) && x < 0);
    near(y, 540);
    std::puts("presentation_coordinates_test: SR output, menu rows, letterboxing, DPI and resize passed");
}
