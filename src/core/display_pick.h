// Which display the display buffers are made for (core/portable.cpp
// host_desktop_largest_size, engine/live_resolution.cpp).
//
// The game makes its display buffers once, at its graphics start, so a live
// change of size goes only as far as they do. They are made for one display's
// own width and height - the one with the most pixels - not the widest and the
// tallest of all of them, a size no display may have: a landscape 2560x1440
// beside a portrait 1440x2560 made 2560x2560, so every frame was drawn into a
// target of its own and copied into them. On a tie (that pair: one screen,
// turned) the one shaped like the starting size, landscape or portrait, when
// one is given, then the primary.
#pragma once

#include <cstdint>

struct LargestDisplay {
    int like_w = 0, like_h = 0;  // the starting size; 0 for none
    int w = 0, h = 0;            // the display taken so far; 0 for none
    std::uint64_t px = 0;
    int rank = -1;

    // One display's mode in physical pixels; a size of 0 or less is skipped.
    constexpr void take(int dw, int dh, bool primary) {
        if (dw <= 0 || dh <= 0) return;
        const std::uint64_t dpx = static_cast<std::uint64_t>(dw) * static_cast<std::uint64_t>(dh);
        const bool like = like_w > 0 && like_h > 0 && (dw >= dh) == (like_w >= like_h);
        const int drank = (like ? 2 : 0) + (primary ? 1 : 0);
        if (dpx > px || (dpx == px && drank > rank)) {
            px = dpx;
            rank = drank;
            w = dw;
            h = dh;
        }
    }
};
