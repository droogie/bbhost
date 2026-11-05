#pragma once
// Texture quality (host/textures.cpp quality_drop): a reduced texture's image
// leaves out the guest's first levels - its level 0 is the guest's level
// `drop`, at width >> drop by height >> drop - and its own smaller levels,
// already in its memory, take their place. How many it leaves out follows
// from the texture's shape; a view of the guest's levels is moved into the
// image's.
#include <algorithm>
#include <cstdint>

namespace texture_quality {

// How many of a `width` x `height` texture's `levels` its image leaves out,
// `want` at most: one level is always kept, and the longer side stays at
// least `floor` texels.
inline std::uint32_t levels_dropped(std::uint32_t want, std::uint32_t width, std::uint32_t height, std::uint32_t levels, std::uint32_t floor) {
    if (levels < 2) return 0;
    std::uint32_t d = std::min(want, levels - 1);
    while (d && (std::max(width, height) >> d) < floor) --d;
    return d;
}

// A view of the guest's levels [base, base + count), in an image that left
// out the first `drop`: from the first level the image has where the view
// starts before it, and at least that one.
struct Levels {
    std::uint32_t base, count;
};
inline Levels view_levels(std::uint32_t base, std::uint32_t count, std::uint32_t drop) {
    if (!drop) return {base, count};
    const std::uint32_t end = std::max(base + count, drop + 1) - drop;
    const std::uint32_t first = std::max(base, drop) - drop;
    return {first, end - first};
}

}  // namespace texture_quality
