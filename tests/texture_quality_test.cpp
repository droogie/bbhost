// Texture quality's level arithmetic (host/texture_quality.h): which of a
// texture's levels a reduced image leaves out, and where a view of the
// guest's levels lands in it.
#include "host/texture_quality.h"

#include <cstdint>
#include <cstdio>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

constexpr std::uint32_t kFloor = 256;

void dropped() {
    using texture_quality::levels_dropped;
    // 0.5x and 0.25x of a 1024x2048 chain (12 levels): 512x1024, 256x512.
    CHECK(levels_dropped(1, 1024, 2048, 12, kFloor) == 1);
    CHECK(levels_dropped(2, 1024, 2048, 12, kFloor) == 2);
    CHECK(levels_dropped(0, 1024, 2048, 12, kFloor) == 0);
    // The longer side stays at the floor: 512 goes to 256 at 0.25x, 256 stays.
    CHECK(levels_dropped(2, 512, 512, 10, kFloor) == 1);
    CHECK(levels_dropped(1, 256, 256, 9, kFloor) == 0);
    CHECK(levels_dropped(2, 2048, 64, 12, kFloor) == 2);
    // One level is always kept, and a texture without levels below its first stays.
    CHECK(levels_dropped(2, 2048, 2048, 2, kFloor) == 1);
    CHECK(levels_dropped(2, 4096, 4096, 1, kFloor) == 0);
    CHECK(levels_dropped(1, 4096, 4096, 0, kFloor) == 0);
}

void views() {
    using texture_quality::view_levels;
    const auto is = [](texture_quality::Levels l, std::uint32_t base, std::uint32_t count) { return l.base == base && l.count == count; };
    // Nothing left out: the view as it is.
    CHECK(is(view_levels(0, 12, 0), 0, 12));
    CHECK(is(view_levels(3, 2, 0), 3, 2));
    // The whole chain of a 12-level texture at 0.5x: the image's 11.
    CHECK(is(view_levels(0, 12, 1), 0, 11));
    // A view of the first level alone reads the image's first.
    CHECK(is(view_levels(0, 1, 1), 0, 1));
    CHECK(is(view_levels(0, 1, 2), 0, 1));
    CHECK(is(view_levels(1, 1, 2), 0, 1));
    // A view past the levels left out moves down by them.
    CHECK(is(view_levels(3, 2, 1), 2, 2));
    CHECK(is(view_levels(2, 3, 2), 0, 3));
    // One that starts inside them keeps its end.
    CHECK(is(view_levels(1, 4, 2), 0, 3));
}

}  // namespace

int main() {
    dropped();
    views();
    if (g_fail) {
        std::printf("texture_quality_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("texture_quality_test: ok\n");
    return 0;
}
