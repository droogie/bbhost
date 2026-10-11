#pragma once

#include <cstdint>
#include <cstdlib>

namespace gpu {

// One switch covers the upload, its GPU detile, and the following carry.
inline bool texture_stream_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("BBHOST_TEXTURE_STREAM");
        return !(e && e[0] == '0');
    }();
    return enabled;
}

// Conservative admission for skipping carried levels. run_prep clears the
// WHOLE upload if ANY slice fails, including a level the carry overwrites.
// Every admitted level must therefore be unable to fail the CPU detile. This
// examines layout only: guest writes cannot change the result. Keep unknown,
// thick, and possibly truncated thin/linear layouts on the whole-upload path.
template <class Levels>
bool texture_upload_layouts_supported(const Levels& levels, std::uint32_t esize) {
    if (!esize || esize > 16) return false;
    for (const auto& lv : levels) {
        if (!lv.w_e || !lv.h_e) return false;
        switch (lv.tiling) {
        case 8: case 9: {
            const std::uint64_t end_elements = static_cast<std::uint64_t>(lv.h_e - 1) * lv.pitch_e + lv.w_e;
            if (end_elements > lv.src_slice_bytes / esize) return false;
            break;
        }
        case 13: case 5: case 0: {
            const std::uint64_t tiles_per_row = (static_cast<std::uint64_t>(lv.pitch_e) + 7) / 8;
            const std::uint64_t last_tile = static_cast<std::uint64_t>((lv.h_e - 1) / 8) * tiles_per_row + (lv.w_e - 1) / 8;
            // A complete final microtile is stronger than the CPU requires,
            // but proves every element it can visit fits in the slice.
            if (last_tile + 1 > lv.src_slice_bytes / 64 / esize) return false;
            break;
        }
        case 14: case 10: case 2: case 3: case 4:
            // untile_rows returns success and writes black for out-of-range
            // elements of a 2D tiled slice, including reused staging bytes.
            break;
        default: return false;
        }
    }
    return !levels.empty();
}

}  // namespace gpu
