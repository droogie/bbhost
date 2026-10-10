#include "host/display_settings.h"
#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    int w = 0, h = 0;
    assert(host::parse_resolution("3440x1640", &w, &h) && w == 3440 && h == 1640);
    assert(!host::parse_resolution("3440x1640junk", &w, &h));
    assert(!host::parse_resolution("-1x1080", &w, &h));
    assert(!host::parse_resolution("7681x2160", &w, &h));
    assert(!host::parse_resolution("1280x0", &w, &h));
    for (int mode = 0; mode < 6; ++mode) {
        auto d = host::display_dimensions(3440, 1640, mode);
        assert(d.output_width == 3440 && d.output_height == 1640);
        if (mode < 2) assert(d.render_width == 3440 && d.render_height == 1640);
        else {
            assert(d.render_width < d.output_width && d.render_height < d.output_height);
            assert(d.render_width % 2 == 0 && d.render_height % 2 == 0);
            assert(std::abs(double(d.render_width) / d.render_height - 3440.0 / 1640.0) < .005);
        }
    }
    auto quality = host::display_dimensions(3440, 1640, 2);
    assert(quality.render_width == 2294 && quality.render_height == 1094);
    auto perf = host::display_dimensions(3440, 1640, 4);
    assert(perf.render_width == 1720 && perf.render_height == 820);
    // Existing profiles still force a developer-selected input explicitly.
    auto legacy = host::display_dimensions(1920, 1080, 2, 1920, 1080, 1280, 720);
    assert(legacy.render_width == 1280 && legacy.render_height == 720);
    assert(legacy.output_width == 1920 && legacy.output_height == 1080);
    auto override_output = host::display_dimensions(1920, 1080, 2, 3440, 1440);
    assert(override_output.output_width == 3440 && override_output.output_height == 1440);
    auto native = host::display_dimensions(1920, 1080, 1, 3440, 1440);
    assert(native.output_width == 1920 && native.render_width == 1920);

    assert(host::parse_upscaler_backend("dlss") == 0);
    assert(host::parse_upscaler_backend("fsr3") == 1);
    assert(host::parse_upscaler_backend("fsr4") == 2);
    assert(host::parse_upscaler_backend("unknown") == 0);
    assert(std::string(host::upscaler_backend_name(0)) == "dlss");
    assert(std::string(host::upscaler_backend_name(1)) == "fsr3");
    assert(std::string(host::upscaler_backend_name(2)) == "fsr4");

    assert(host::parse_fg_backend("dlss") == 0);
    assert(host::parse_fg_backend("fsr3") == 1);
    assert(host::parse_fg_backend("other") == 0);
    assert(std::string(host::fg_backend_name(0)) == "dlss");
    assert(std::string(host::fg_backend_name(1)) == "fsr3");

    std::puts("display_settings_test: custom sizes, SR presets, native modes and overrides passed");
}
