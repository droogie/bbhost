// Textures and samplers for the renderer: T#/S# words resolved from the
// stage's user data become Vulkan images (untiled on upload) and samplers.
// Render targets sampled as textures alias the render target image.
#include "host/gpu_internal.h"
#include "host/texture_upload_plan.h"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#include "core/image_file.h"
#include "core/portable.h"
#include "engine/gx_resources.h"

#include "core/write_watch.h"
#include "hle/modules.h"
#include "log.h"

#include <atomic>
#include <limits>
#include <memory>
#include <deque>
#include <bit>
#include <cstdarg>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <set>
#include <unordered_map>

namespace gpu {

namespace {

struct FormatInfo {
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t bytes_per_element = 0;  // bytes per texel, or per 4x4 block for BC
    std::uint32_t block = 1;              // texels per block side
};

// T# BASE_ADDRESS is word0 plus word1 bits 7:0 shifted up, all in 256-byte
// units. This title writes 0x40 into those high bits on some descriptors whose
// real base is entirely in word0: the two compute shaders 9a9cf8a9 and
// 10bc83cb resolve a T# reading dfmt 12 / nfmt 7 (RGBA16F), 960x540, pitch
// 1024, tiling 14 - which is exactly the render target at 0x15dcb0000 - and
// with the high byte the address is 0x40015dcb0000, which is in no mapping.
// Taking the low interpretation when the full one is unmapped recovers the
// texture; it can only ever turn a guaranteed failure into a hit.
}  // namespace

std::uint64_t tsharp_base(const std::uint32_t* w) {
    const std::uint64_t full = (static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32)) << 8;
    if (hle_kernel_va_mapped(full, 16)) return full;
    const std::uint64_t low = static_cast<std::uint64_t>(w[0]) << 8;
    if (hle_kernel_va_mapped(low, 16)) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 8) {
            host_log("texture: T# base 0x%llx is unmapped; BASE_ADDRESS_HI 0x%02x is not part of the address, using 0x%llx",
                     static_cast<unsigned long long>(full), w[1] & 0xff, static_cast<unsigned long long>(low));
        }
        return low;
    }
    // A V# is a 4-dword byte address (type bits 31:30 of word3 are 0). A T#
    // is 8 dwords, type 8-15, base in 256-byte units. The fields overlap, so
    // a V# sitting in a T# slot has an unmapped <<8 address and a mapped
    // unshifted one in the GFX heap. Binding that as a tiled 1D texture
    // (NUM_RECORDS+1 wide, pitch from the next dword, 8x8 micro-tile pad)
    // writeback-stomps neighbouring heap headers; Dantelion free_ then dies
    // in coalesce. shadPS4: Image::Valid is (type & 8) != 0 and Address is
    // always << 8; V#s go to the buffer cache, never the texture cache.
    const std::uint64_t vsharp = static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32);
    if (vsharp && hle_kernel_va_mapped(vsharp, 16)) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 8) {
            host_log("texture: T# base 0x%llx unmapped; descriptor looks like a V# at 0x%llx (not a T#, skipping)",
                     static_cast<unsigned long long>(full), static_cast<unsigned long long>(vsharp));
        }
    }
    return full;
}

namespace {

FormatInfo image_format(std::uint32_t dfmt, std::uint32_t nfmt) {
    const bool srgb = nfmt == 9;
    const bool flt = nfmt == 7;
    const bool uint_ = nfmt == 4;
    const bool sint = nfmt == 5;
    const bool snorm = nfmt == 1;
    switch (dfmt) {
    case 1: return {uint_ ? VK_FORMAT_R8_UINT : sint ? VK_FORMAT_R8_SINT : snorm ? VK_FORMAT_R8_SNORM : srgb ? VK_FORMAT_R8_SRGB : VK_FORMAT_R8_UNORM, 1, 1};
    case 2: return {flt ? VK_FORMAT_R16_SFLOAT : uint_ ? VK_FORMAT_R16_UINT : sint ? VK_FORMAT_R16_SINT : snorm ? VK_FORMAT_R16_SNORM : VK_FORMAT_R16_UNORM, 2, 1};
    case 3: return {uint_ ? VK_FORMAT_R8G8_UINT : sint ? VK_FORMAT_R8G8_SINT : snorm ? VK_FORMAT_R8G8_SNORM : srgb ? VK_FORMAT_R8G8_SRGB : VK_FORMAT_R8G8_UNORM, 2, 1};
    case 4: return {flt ? VK_FORMAT_R32_SFLOAT : sint ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_UINT, 4, 1};
    case 5: return {flt ? VK_FORMAT_R16G16_SFLOAT : uint_ ? VK_FORMAT_R16G16_UINT : sint ? VK_FORMAT_R16G16_SINT : snorm ? VK_FORMAT_R16G16_SNORM : VK_FORMAT_R16G16_UNORM, 4, 1};
    case 6:
    case 7: return {VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4, 1};
    case 9: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4, 1};
    case 10: return {srgb ? VK_FORMAT_R8G8B8A8_SRGB : uint_ ? VK_FORMAT_R8G8B8A8_UINT : sint ? VK_FORMAT_R8G8B8A8_SINT : snorm ? VK_FORMAT_R8G8B8A8_SNORM : VK_FORMAT_R8G8B8A8_UNORM, 4, 1};
    case 11: return {flt ? VK_FORMAT_R32G32_SFLOAT : sint ? VK_FORMAT_R32G32_SINT : VK_FORMAT_R32G32_UINT, 8, 1};
    case 12: return {flt ? VK_FORMAT_R16G16B16A16_SFLOAT : uint_ ? VK_FORMAT_R16G16B16A16_UINT : sint ? VK_FORMAT_R16G16B16A16_SINT : snorm ? VK_FORMAT_R16G16B16A16_SNORM : VK_FORMAT_R16G16B16A16_UNORM, 8, 1};
    case 13: return {flt ? VK_FORMAT_R32G32B32_SFLOAT : sint ? VK_FORMAT_R32G32B32_SINT : VK_FORMAT_R32G32B32_UINT, 12, 1};
    case 14: return {flt ? VK_FORMAT_R32G32B32A32_SFLOAT : sint ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_UINT, 16, 1};
    case 35: return {srgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 8, 4};
    case 36: return {srgb ? VK_FORMAT_BC2_SRGB_BLOCK : VK_FORMAT_BC2_UNORM_BLOCK, 16, 4};
    case 37: return {srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK, 16, 4};
    case 38: return {snorm ? VK_FORMAT_BC4_SNORM_BLOCK : VK_FORMAT_BC4_UNORM_BLOCK, 8, 4};
    case 39: return {snorm ? VK_FORMAT_BC5_SNORM_BLOCK : VK_FORMAT_BC5_UNORM_BLOCK, 16, 4};
    case 40: return {snorm ? VK_FORMAT_BC6H_SFLOAT_BLOCK : VK_FORMAT_BC6H_UFLOAT_BLOCK, 16, 4};
    case 41: return {srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK, 16, 4};
    default: return {};
    }
}

VkComponentSwizzle swizzle(std::uint32_t sel) {
    switch (sel & 7) {
    case 0: return VK_COMPONENT_SWIZZLE_ZERO;
    case 1: return VK_COMPONENT_SWIZZLE_ONE;
    case 4: return VK_COMPONENT_SWIZZLE_R;
    case 5: return VK_COMPONENT_SWIZZLE_G;
    case 6: return VK_COMPONENT_SWIZZLE_B;
    case 7: return VK_COMPONENT_SWIZZLE_A;
    default: return VK_COMPONENT_SWIZZLE_IDENTITY;
    }
}

// Element index inside an 8x8 micro tile for the "thin" (non-displayable)
// micro tile mode: x0 y0 x1 y1 x2 y2.
inline std::uint32_t thin_index(std::uint32_t x, std::uint32_t y) {
    return (x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) | ((x & 4) << 2) | ((y & 4) << 3);
}

inline std::uint32_t thick_index(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return (x & 1) | ((y & 1) << 1) | ((z & 1) << 2) | ((x & 2) << 2) | ((y & 2) << 3) | ((z & 2) << 4) | ((x & 4) << 4) |
           ((y & 4) << 5);
}

// 2D tiled thin (ARRAY_2D_TILED_THIN1) on the PS4's GPU: 8 pipes in the
// P8_32x32_8x16 configuration, 256-byte pipe interleave, and the macro tile
// mode picked from the micro tile size (tile split 1 KiB). Bank width is 1 in
// every mode used, bank height/aspect/banks follow the macro tile table.
struct MacroMode {
    std::uint32_t bank_height, aspect, banks;
};
MacroMode macro_mode_for(std::uint32_t esize) {
    const std::uint32_t micro_bytes = 64 * esize;
    const std::uint32_t split = std::min<std::uint32_t>(1024, micro_bytes);
    switch (split / 64) {  // 1, 2, 4, 8, 16
    case 1: return {4, 4, 16};   // 1x4_16
    case 2: return {2, 2, 16};   // 1x2_16
    case 4: return {1, 2, 16};   // 1x1_16
    case 8: return {1, 2, 16};   // 1x1_16_dup
    default: return {1, 1, 8};   // 1x1_8
    }
}

std::size_t tiled2d_offset(std::uint32_t x, std::uint32_t y, std::uint32_t pitch_e, std::uint32_t esize, const MacroMode& mm) {
    const std::uint32_t num_pipes = 8;
    const std::uint32_t micro_bytes = 64 * esize;
    const std::uint32_t macro_pitch = 8 * 1 * num_pipes * mm.aspect;               // elements
    const std::uint32_t macro_height = 8 * mm.bank_height * mm.banks / mm.aspect;  // elements
    const std::size_t macro_bytes = static_cast<std::size_t>(micro_bytes) * (macro_pitch / 8) * (macro_height / 8);
    const std::uint32_t macros_per_row = std::max(1u, (pitch_e + macro_pitch - 1) / macro_pitch);
    const std::size_t macro_offset = (static_cast<std::size_t>(y / macro_height) * macros_per_row + x / macro_pitch) * macro_bytes;
    // pipe from the micro tile coordinates
    const std::uint32_t tx = x >> 3, ty = y >> 3;
    const std::uint32_t x3 = tx & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1;
    const std::uint32_t y3 = ty & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1;
    const std::uint32_t pipe = (x4 ^ y3 ^ x5) | ((x3 ^ y4) << 1) | ((x5 ^ y5) << 2);
    // bank from the bank-tile coordinates
    const std::uint32_t bx = x >> (3 + 3);                   // bank width 1, 8 pipes
    std::uint32_t bh_shift = 0;
    while ((1u << bh_shift) < mm.bank_height) ++bh_shift;
    const std::uint32_t by = y >> (3 + bh_shift);
    const std::uint32_t bx3 = bx & 1, bx4 = (bx >> 1) & 1, bx5 = (bx >> 2) & 1, bx6 = (bx >> 3) & 1;
    const std::uint32_t by3 = by & 1, by4 = (by >> 1) & 1, by5 = (by >> 2) & 1, by6 = (by >> 3) & 1;
    std::uint32_t bank = 0, bank_bits = 0;
    switch (mm.banks) {
    case 16: bank = (bx3 ^ by6) | ((bx4 ^ by5 ^ by6) << 1) | ((bx5 ^ by4) << 2) | ((bx6 ^ by3) << 3); bank_bits = 4; break;
    case 8: bank = (bx3 ^ by5) | ((bx4 ^ by4 ^ by5) << 1) | ((bx5 ^ by3) << 2); bank_bits = 3; break;
    case 4: bank = (bx3 ^ by4) | ((bx4 ^ by3 ^ by4) << 1); bank_bits = 2; break;
    default: bank = bx3 ^ by3; bank_bits = 1; break;
    }
    // element within the micro tile, plus the micro tile within the bank tile
    const std::size_t elem = static_cast<std::size_t>(thin_index(x & 7, y & 7)) * esize +
                             static_cast<std::size_t>(ty & (mm.bank_height - 1)) * micro_bytes;
    const std::size_t total = macro_offset + elem;
    const std::size_t low = total & 255;
    const std::size_t high = (total >> 8) << (8 + 3 + bank_bits);
    return high | (static_cast<std::size_t>(bank) << (8 + 3)) | (static_cast<std::size_t>(pipe) << 8) | low;
}

// 1D thick tiling: micro tiles of 8x8x4 elements; `src` points at the start
// of the 4-slice group and `z` is the slice within it.
bool untile_thick_slice(std::uint8_t* dst, const std::uint8_t* src, std::size_t src_avail, std::uint32_t width_e,
                        std::uint32_t height_e, std::uint32_t pitch_e, std::uint32_t z, std::uint32_t esize) {
    const std::uint32_t tiles_per_row = (pitch_e + 7) / 8;
    const std::size_t tile_bytes = 256ull * esize;
    for (std::uint32_t y = 0; y < height_e; ++y) {
        for (std::uint32_t x = 0; x < width_e; ++x) {
            const std::size_t tile = static_cast<std::size_t>(y / 8) * tiles_per_row + (x / 8);
            const std::size_t off = tile * tile_bytes + static_cast<std::size_t>(thick_index(x & 7, y & 7, z & 3)) * esize;
            if (off + esize > src_avail) return false;
            std::memcpy(dst + (static_cast<std::size_t>(y) * width_e + x) * esize, src + off, esize);
        }
    }
    return true;
}

// The per-element untiling this file used before the row-band version below;
// BBHOST_UNTILE_CHECK=1 compares every slice against it.
bool untile_slice_reference(std::uint8_t* dst, const std::uint8_t* src, std::size_t src_avail, std::uint32_t tiling,
                  std::uint32_t width_e, std::uint32_t height_e, std::uint32_t pitch_e, std::uint32_t esize) {
    if (tiling == 8 || tiling == 9) {  // linear (aligned / general)
        for (std::uint32_t y = 0; y < height_e; ++y) {
            const std::size_t off = static_cast<std::size_t>(y) * pitch_e * esize;
            if (off + static_cast<std::size_t>(width_e) * esize > src_avail) return false;
            std::memcpy(dst + static_cast<std::size_t>(y) * width_e * esize, src + off, static_cast<std::size_t>(width_e) * esize);
        }
        return true;
    }
    if (tiling == 13 || tiling == 5 || tiling == 0) {  // 1D tiled thin (micro tiles only)
        const std::uint32_t tiles_per_row = (pitch_e + 7) / 8;
        const std::size_t tile_bytes = 64ull * esize;
        for (std::uint32_t y = 0; y < height_e; ++y) {
            for (std::uint32_t x = 0; x < width_e; ++x) {
                const std::size_t tile = static_cast<std::size_t>(y / 8) * tiles_per_row + (x / 8);
                const std::size_t off = tile * tile_bytes + static_cast<std::size_t>(thin_index(x & 7, y & 7)) * esize;
                if (off + esize > src_avail) return false;
                std::memcpy(dst + (static_cast<std::size_t>(y) * width_e + x) * esize, src + off, esize);
            }
        }
        return true;
    }
    if (tiling == 14 || tiling == 10 || tiling == 2 || tiling == 3 || tiling == 4) {  // 2D tiled thin
        const MacroMode mm = macro_mode_for(esize);
        for (std::uint32_t y = 0; y < height_e; ++y) {
            for (std::uint32_t x = 0; x < width_e; ++x) {
                const std::size_t off = tiled2d_offset(x, y, pitch_e, esize, mm);
                if (off + esize > src_avail) continue;  // outside the padded surface: leave black
                std::memcpy(dst + (static_cast<std::size_t>(y) * width_e + x) * esize, src + off, esize);
            }
        }
        return true;
    }
    return false;
}

// Untiling was per element on the command processor's thread: every element
// recomputed the slice's macro tile geometry (a bank-height loop included) and
// went through a variable-size memcpy, and large textures were 350-530 ms of a
// world-load stall. Rows are now untiled in bands with the slice constants
// hoisted, the micro tile's pipe, bank and macro offset computed once per tile,
// the element order from a table and fixed-size copies. Large slices split
// their bands across a small worker pool (BBHOST_UNTILE_THREADS, 0 = none);
// the command processor runs a band itself and waits, so an upload still lands
// when it did. BBHOST_UNTILE_CHECK=1 (checks) also runs the reference on the
// same snapshot of the source and counts slices whose bytes differ.
const bool g_untile_check = [] {
    const char* e = std::getenv("BBHOST_UNTILE_CHECK");
    return e && e[0] == '1';
}();
std::atomic<std::uint64_t> g_untile_checked{0}, g_untile_mismatched{0};

constexpr std::uint8_t kThin[8][8] = {
    {0, 1, 4, 5, 16, 17, 20, 21},     {2, 3, 6, 7, 18, 19, 22, 23},     {8, 9, 12, 13, 24, 25, 28, 29},  {10, 11, 14, 15, 26, 27, 30, 31},
    {32, 33, 36, 37, 48, 49, 52, 53}, {34, 35, 38, 39, 50, 51, 54, 55}, {40, 41, 44, 45, 56, 57, 60, 61}, {42, 43, 46, 47, 58, 59, 62, 63}};

inline void copy_element(std::uint8_t* dst, const std::uint8_t* src, std::uint32_t esize) {
    switch (esize) {
    case 1: *dst = *src; break;
    case 2: std::memcpy(dst, src, 2); break;
    case 4: std::memcpy(dst, src, 4); break;
    case 8: std::memcpy(dst, src, 8); break;
    case 16: std::memcpy(dst, src, 16); break;
    default: std::memcpy(dst, src, esize); break;
    }
}

// Rows [y0, y1) of one slice. Same results as untile_slice_reference.
bool untile_rows(std::uint8_t* dst, const std::uint8_t* src, std::size_t src_avail, std::uint32_t tiling, std::uint32_t width_e,
                 std::uint32_t pitch_e, std::uint32_t esize, std::uint32_t y0, std::uint32_t y1) {
    const std::size_t row_bytes = static_cast<std::size_t>(width_e) * esize;
    if (tiling == 8 || tiling == 9) {
        for (std::uint32_t y = y0; y < y1; ++y) {
            const std::size_t off = static_cast<std::size_t>(y) * pitch_e * esize;
            if (off + row_bytes > src_avail) return false;
            std::memcpy(dst + static_cast<std::size_t>(y) * row_bytes, src + off, row_bytes);
        }
        return true;
    }
    const std::uint32_t tiles_wide = (width_e + 7) / 8;
    if (tiling == 13 || tiling == 5 || tiling == 0) {
        const std::size_t tiles_per_row = (pitch_e + 7) / 8;
        const std::size_t tile_bytes = 64ull * esize;
        for (std::uint32_t y = y0; y < y1; ++y) {
            const std::uint32_t ly = y & 7;
            const std::size_t row_tiles = static_cast<std::size_t>(y >> 3) * tiles_per_row;
            std::uint8_t* out = dst + static_cast<std::size_t>(y) * row_bytes;
            for (std::uint32_t tx = 0; tx < tiles_wide; ++tx) {
                const std::size_t base = (row_tiles + tx) * tile_bytes;
                const std::uint32_t xs = tx * 8, xe = std::min(width_e, xs + 8);
                for (std::uint32_t x = xs; x < xe; ++x) {
                    const std::size_t off = base + static_cast<std::size_t>(kThin[ly][x - xs]) * esize;
                    if (off + esize > src_avail) return false;
                    copy_element(out + static_cast<std::size_t>(x) * esize, src + off, esize);
                }
            }
        }
        return true;
    }
    if (tiling == 14 || tiling == 10 || tiling == 2 || tiling == 3 || tiling == 4) {
        const MacroMode mm = macro_mode_for(esize);
        const std::size_t micro_bytes = 64ull * esize;
        const std::uint32_t macro_pitch = 8 * 8 * mm.aspect;
        const std::uint32_t macro_height = 8 * mm.bank_height * mm.banks / mm.aspect;
        const std::size_t macro_bytes = micro_bytes * (macro_pitch / 8) * (macro_height / 8);
        const std::size_t macros_per_row = std::max(1u, (pitch_e + macro_pitch - 1) / macro_pitch);
        std::uint32_t bh_shift = 0;
        while ((1u << bh_shift) < mm.bank_height) ++bh_shift;
        const std::uint32_t bank_bits = mm.banks == 16 ? 4 : mm.banks == 8 ? 3 : mm.banks == 4 ? 2 : 1;
        for (std::uint32_t y = y0; y < y1; ++y) {
            const std::uint32_t ly = y & 7, ty = y >> 3;
            const std::uint32_t y3 = ty & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1;
            const std::uint32_t by = y >> (3 + bh_shift);
            const std::uint32_t by3 = by & 1, by4 = (by >> 1) & 1, by5 = (by >> 2) & 1, by6 = (by >> 3) & 1;
            const std::size_t macro_row = static_cast<std::size_t>(y / macro_height) * macros_per_row;
            const std::size_t tile_row_bytes = static_cast<std::size_t>(ty & (mm.bank_height - 1)) * micro_bytes;
            std::uint8_t* out = dst + static_cast<std::size_t>(y) * row_bytes;
            for (std::uint32_t tx = 0; tx < tiles_wide; ++tx) {
                const std::uint32_t xs = tx * 8, xe = std::min(width_e, xs + 8);
                const std::uint32_t x3 = tx & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1;
                const std::size_t pipe = (x4 ^ y3 ^ x5) | ((x3 ^ y4) << 1) | ((x5 ^ y5) << 2);
                const std::uint32_t bx = xs >> 6;
                const std::uint32_t bx3 = bx & 1, bx4 = (bx >> 1) & 1, bx5 = (bx >> 2) & 1, bx6 = (bx >> 3) & 1;
                std::size_t bank;
                switch (mm.banks) {
                case 16: bank = (bx3 ^ by6) | ((bx4 ^ by5 ^ by6) << 1) | ((bx5 ^ by4) << 2) | ((bx6 ^ by3) << 3); break;
                case 8: bank = (bx3 ^ by5) | ((bx4 ^ by4 ^ by5) << 1) | ((bx5 ^ by3) << 2); break;
                case 4: bank = (bx3 ^ by4) | ((bx4 ^ by3 ^ by4) << 1); break;
                default: bank = bx3 ^ by3; break;
                }
                const std::size_t macro_offset = (macro_row + xs / macro_pitch) * macro_bytes + tile_row_bytes;
                const std::size_t fixed = (bank << (8 + 3)) | (pipe << 8);
                for (std::uint32_t x = xs; x < xe; ++x) {
                    const std::size_t total = macro_offset + static_cast<std::size_t>(kThin[ly][x - xs]) * esize;
                    const std::size_t off = ((total >> 8) << (8 + 3 + bank_bits)) | fixed | (total & 255);
                    if (off + esize > src_avail) {
                        // Outside the padded surface: black. Staging chunks are reused, so
                        // write the zeros rather than leave another upload's bytes.
                        std::memset(out + static_cast<std::size_t>(x) * esize, 0, esize);
                        continue;
                    }
                    copy_element(out + static_cast<std::size_t>(x) * esize, src + off, esize);
                }
            }
        }
        return true;
    }
    return false;
}

// A fixed pool for untiling bands. run() hands out indices [0, count) to the
// workers and the calling thread and returns once every one finished.
class UntilePool {
public:
    static UntilePool& get() {
        static UntilePool pool;
        return pool;
    }
    unsigned workers() const { return static_cast<unsigned>(threads_.size()); }
    void run(unsigned count, const std::function<void(unsigned)>& fn) {
        std::lock_guard<std::mutex> one(run_mu_);
        {
            std::lock_guard<std::mutex> lk(mu_);
            fn_ = &fn;
            count_ = count;
            next_ = 0;
            pending_ = count;
            ++generation_;
        }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
        fn_ = nullptr;
    }
    ~UntilePool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread& t : threads_) t.join();
    }

private:
    UntilePool() {
        const char* e = std::getenv("BBHOST_UNTILE_THREADS");
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const unsigned n = e ? static_cast<unsigned>(std::strtoul(e, nullptr, 10)) : std::min(7u, hw > 2 ? hw / 2 - 1 : 0u);
        for (unsigned i = 0; i < n; ++i) {
            threads_.emplace_back([this] {
                std::uint64_t seen = 0;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lk(mu_);
                        cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
                        if (stop_) return;
                        seen = generation_;
                    }
                    work();
                }
            });
        }
    }
    void work() {
        for (;;) {
            unsigned i;
            const std::function<void(unsigned)>* fn;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!fn_ || next_ >= count_) return;
                i = next_++;
                fn = fn_;
            }
            (*fn)(i);
            std::lock_guard<std::mutex> lk(mu_);
            if (--pending_ == 0) done_cv_.notify_all();
        }
    }
    std::mutex run_mu_, mu_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(unsigned)>* fn_ = nullptr;
    unsigned count_ = 0, next_ = 0, pending_ = 0;
    std::uint64_t generation_ = 0;
    bool stop_ = false;
    std::vector<std::thread> threads_;
};

// Untiles one 2D slice into `dst` (row-major, tightly packed). Returns false
// for tiling modes that are not supported yet.
bool untile_slice(std::uint8_t* dst, const std::uint8_t* src, std::size_t src_avail, std::uint32_t tiling,
                  std::uint32_t width_e, std::uint32_t height_e, std::uint32_t pitch_e, std::uint32_t esize) {
    const std::size_t bytes = static_cast<std::size_t>(width_e) * height_e * esize;
    // The check compares both implementations on one snapshot: guest memory
    // (render-target memory especially) can change between two passes.
    std::vector<std::uint8_t> snapshot;
    if (g_untile_check) {
        snapshot.assign(src, src + src_avail);
        src = snapshot.data();
    }
    UntilePool& pool = UntilePool::get();
    bool ok = true;
    if (bytes >= (256u << 10) && height_e >= 32 && pool.workers() > 0) {
        const unsigned bands = std::min<unsigned>(pool.workers() + 1, height_e / 16);
        const std::uint32_t rows = ((height_e + bands - 1) / bands + 7) & ~7u;
        std::atomic<bool> all_ok{true};
        pool.run(bands, [&](unsigned b) {
            const std::uint32_t y0 = std::min(height_e, b * rows), y1 = std::min(height_e, y0 + rows);
            if (y0 < y1 && !untile_rows(dst, src, src_avail, tiling, width_e, pitch_e, esize, y0, y1)) all_ok = false;
        });
        ok = all_ok;
    } else {
        ok = untile_rows(dst, src, src_avail, tiling, width_e, pitch_e, esize, 0, height_e);
    }
    if (g_untile_check) {
        std::vector<std::uint8_t> ref(bytes);
        const bool ref_ok = untile_slice_reference(ref.data(), src, src_avail, tiling, width_e, height_e, pitch_e, esize);
        g_untile_checked.fetch_add(1, std::memory_order_relaxed);
        if (ref_ok != ok || (ok && std::memcmp(ref.data(), dst, bytes) != 0)) {
            if (g_untile_mismatched.fetch_add(1, std::memory_order_relaxed) < 8) {
                host_log("texture: untile check: tiling %u %ux%u pitch %u esize %u differs from the reference (ok %d, reference %d)", tiling,
                         width_e, height_e, pitch_e, esize, ok, ref_ok);
            }
        }
    }
    return ok;
}

std::atomic<std::uint64_t> g_uploads{0}, g_upload_bytes{0}, g_unsupported{0};
std::map<std::uint32_t, std::uint32_t> g_tiling_hist;

// One guest image (by base address): every T# that names this address shares
// the VkImage, so compute stores and later samples see the same pixels.
struct Surface {
    VkImage image = VK_NULL_HANDLE;
    VkImageCreateInfo info{};  // as created (draw capture describes views with it)
    ImageMemory memory;                      // gpu.cpp image heap
    VkFormat format = VK_FORMAT_UNDEFINED;   // storage-capable base format (UNORM for sRGB)
    std::uint32_t width = 0, height = 0, depth = 1, layers = 1;
    std::uint32_t type = 0;
    std::uint64_t base = 0;
    std::map<std::uint64_t, VkImageView> views;  // by (view format, swizzle, storage)
    bool failed = false;
    // Source description for re-uploads when the guest memory changes.
    std::uint32_t tiling = 0, width_e = 0, height_e = 0, pitch_e = 0, esize = 0, slices = 1;
    bool thick = false;
    // T# POW2_PAD (word 3 bit 25): each level's width and height, and the
    // slice count, are padded to powers of two in memory (see compute_levels).
    bool pow2pad = false;
    std::size_t slice_src_bytes = 0, slice_dst_bytes = 0, total_src = 0;
    // Mip levels (level-major in memory: every slice of level 0, then level 1, ...).
    struct Level {
        std::uint32_t w_e, h_e, pitch_e, tiling;
        std::size_t src_off, src_slice_bytes, dst_off, dst_slice_bytes;
    };
    std::vector<Level> levels;
    std::size_t total_dst = 0;
    std::uint64_t content_hash = 0;   // the sampled hash, checked on every bind
    std::uint64_t exact_hash = 0;     // the whole surface; the slow catch-up
    std::uint64_t exact_at_ms = 0;
    std::uint64_t checked_at_ms = 0;
    // The command buffer (Gpu::record_serial) this surface was last brought
    // up to date in; refresh_surface_locked() runs once per surface per
    // command buffer, and a CP write over it (mark_surfaces_dirty_locked)
    // clears this so the next bind looks again.
    std::uint64_t refreshed_serial = 0;
    // The flip and the time (steady clock, ms) it was made at or a shader, a
    // fill, a copy or an upload last wrote its image: its age and order for
    // the reuse retire (textures_retire_reused_locked).
    std::uint64_t written_flip = 0, written_ms = 0;
    std::uint32_t hot = 0;  // checks left before backing off again; see below
    std::uint32_t unchanged = 0;  // consecutive checks that found the memory unchanged (check backoff)
    std::uint32_t uploads = 0;
    std::uint32_t block = 1;  // texel block edge: 4 for BC, 1 otherwise
    // A shader stored into it (surface_queue_writeback): the image holds what
    // the guest memory underneath does not.
    bool gpu_written = false;
    // (mip, layer 0) pairs a copy token has written, one bit a mip: the first
    // copy into each goes into the texture's history for F12.
    std::uint32_t copied_mips = 0;
    // Its memory is write-protected (core/write_watch.h): a CPU write marks it,
    // and the next bind uploads it. Armed by every upload that can.
    WriteWatch watch;
    bool watched = false;
    // A surface the CPU keeps rewriting (a movie frame) would fault on every
    // page every frame: past kWatchPagesPerSecond it goes back to the hashes
    // for a while.
    std::uint64_t watch_window_ms = 0, watch_pause_until_ms = 0;
    std::uint32_t watch_window_pages = 0;
    // The census: the GX registry's write sequence for the resource
    // under this surface at its last upload, and whether a command-processor
    // write (mark_surfaces_dirty_locked) has landed on it since.
    std::uint64_t gx_seq = 0;
    bool cp_dirty = false;
    bool gx_registered = false;  // its memory is a live GX resource's
    // Created at the resource's creation, ahead of its first bind.
    bool ahead = false, ahead_used = false;
    // The last upload's untile and hashes, while the prep pool still has them
    // (TexturePrep); settle_prep() takes the hashes.
    std::shared_ptr<struct TexturePrep> prep;
};

std::uint64_t now_ms() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::atomic<std::uint64_t> g_hashes{0}, g_hash_us{0}, g_surface_replacements{0};
// The exact catch-up hash (refresh_surface_locked): what it applies to, how
// often a surface may have one, and the ceiling on how much it may read.
constexpr std::size_t kSampledAbove = 256u << 10;   // content_hash_of hashes whole below this
// A first cut capped this at 4 MiB and glyphs still went missing in other
// menus: the in-game screens use bigger atlases and were simply excluded. With
// a four-lane hash a whole surface is cheap enough that the cap can be where
// texture streaming actually starts rather than where the old hash got slow.
constexpr std::size_t kExactMaxBytes = 8u << 20;
// How often a surface may be hashed whole. A single interval of 100 ms was
// measured at half a second from a glyph being rasterised to it appearing -
// text popped in wrong and corrected itself, which for a menu is *exactly*
// when you are looking at it. A surface the sampled hash has just caught a
// write in is marked `hot`, and hot is precisely the set being written into,
// so that set gets checked about once a frame and everything else stays slow.
constexpr std::uint64_t kExactHotMs = 16;
constexpr std::uint64_t kExactColdMs = 250;
constexpr std::uint64_t kExactMs = 100;  // the budget window
// The only limiter. Every gate tried instead of this was a hole: a 4 MiB cap
// left the bigger atlases out, and "only surfaces uploaded twice" leaves out
// any surface whose every write since its first upload the sampling missed -
// which is the exact case being fixed. The budget starves nobody, because a
// surface that is served steps aside for kExactMs while a surface that is
// denied keeps its old timestamp and stays due, so a busy frame spreads the
// sweep over several windows rather than dropping anyone out of it.
constexpr std::size_t kExactBudgetPerWindow = 16u << 20;  // 160 MB/s ceiling
// BBHOST_TEXTURE_EXACT=0 turns the catch-up off, which is both the kill switch
// and how its cost was measured rather than asserted.
const bool g_exact_enabled = [] {
    const char* e = std::getenv("BBHOST_TEXTURE_EXACT");
    return !e || e[0] != '0';
}();
std::atomic<std::uint64_t> g_exact_catches{0}, g_exact_hashes{0}, g_exact_too_big{0}, g_exact_no_budget{0};

// Everything here runs under the render lock, so a plain window and counter.
bool exact_budget_take(std::size_t bytes) {
    static std::uint64_t window_at = 0;
    static std::size_t spent = 0;
    const std::uint64_t now = now_ms();
    if (now - window_at >= kExactMs) {
        window_at = now;
        spent = 0;
    }
    if (spent + bytes > kExactBudgetPerWindow) {
        g_exact_no_budget.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    spent += bytes;
    g_exact_hashes.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Hash of the source memory (whole for small textures, strided otherwise).
// Every bound texture is hashed at least five times a second and a streaming
// one on every bind, so this reads 64-bit words: the byte-at-a-time FNV-1a it
// replaces was a third of the command processor's time during a world load.
//
// **The strided form samples 64 bytes out of every `step` and is therefore
// blind to a small change that lands in the gaps.** That is not theoretical:
// the menu glyph atlas is 1 MiB with tiling 8, so a freshly rasterised glyph
// is about nine scattered 64-byte micro-tiles, each the size of one sample.
// When none of them fell in a sampled window the surface was judged unchanged
// and never re-uploaded, and that glyph was missing from the screen for good -
// characters dropping out of menu text, the game's own included. Callers that
// need certainty pass `exact`; refresh_surface_locked() below does so on a
// slow cadence so the fast path keeps its coverage and its cost.
std::uint64_t content_hash_of(std::uint64_t base, std::size_t bytes, bool exact = false) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto* p = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(base));
    // Four independent FNV lanes. One chain is a dependent multiply per 8
    // bytes - about five cycles each - so a whole 1 MiB surface cost ~0.7 ms
    // and the hash was latency-bound, not memory-bound. Four lanes read the
    // same bytes four times as fast, which is what makes hashing a surface
    // whole affordable enough to do for real.
    std::uint64_t h0 = 1469598103934665603ull ^ bytes, h1 = h0 ^ 0x9e3779b97f4a7c15ull,
                  h2 = h0 ^ 0xc2b2ae3d27d4eb4full, h3 = h0 ^ 0x165667b19e3779f9ull;
    auto mix = [&](const std::uint8_t* q, std::size_t n) {
        constexpr std::uint64_t kP = 1099511628211ull;
        std::size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            std::uint64_t v0, v1, v2, v3;
            std::memcpy(&v0, q + i, 8);
            std::memcpy(&v1, q + i + 8, 8);
            std::memcpy(&v2, q + i + 16, 8);
            std::memcpy(&v3, q + i + 24, 8);
            h0 = (h0 ^ v0) * kP;
            h1 = (h1 ^ v1) * kP;
            h2 = (h2 ^ v2) * kP;
            h3 = (h3 ^ v3) * kP;
        }
        for (; i + 8 <= n; i += 8) {
            std::uint64_t v;
            std::memcpy(&v, q + i, 8);
            h0 = (h0 ^ v) * kP;
        }
        for (; i < n; ++i) h0 = (h0 ^ q[i]) * kP;
    };
    if (exact || bytes <= (256u << 10)) {
        mix(p, bytes);
    } else {
        const std::size_t step = bytes / 4096;  // ~4096 samples of 64 bytes
        for (std::size_t off = 0; off + 64 <= bytes; off += step) mix(p + off, 64);
    }
    const std::uint64_t h = (h0 ^ (h1 * 1099511628211ull)) ^ ((h2 ^ (h3 * 1099511628211ull)) * 0x9e3779b97f4a7c15ull);
    g_hashes.fetch_add(1, std::memory_order_relaxed);
    g_hash_us.fetch_add(static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()),
                        std::memory_order_relaxed);
    return h;
}

// Tiles a linear slice back into guest memory (inverse of untile_slice).
bool tile_slice(std::uint8_t* dst, std::size_t dst_avail, const std::uint8_t* src, std::uint32_t tiling, std::uint32_t width_e,
                std::uint32_t height_e, std::uint32_t pitch_e, std::uint32_t esize) {
    if (tiling == 8 || tiling == 9) {
        for (std::uint32_t y = 0; y < height_e; ++y) {
            const std::size_t off = static_cast<std::size_t>(y) * pitch_e * esize;
            if (off + static_cast<std::size_t>(width_e) * esize > dst_avail) return false;
            std::memcpy(dst + off, src + static_cast<std::size_t>(y) * width_e * esize, static_cast<std::size_t>(width_e) * esize);
        }
        return true;
    }
    if (tiling == 13 || tiling == 5 || tiling == 0) {
        const std::uint32_t tiles_per_row = (pitch_e + 7) / 8;
        const std::size_t tile_bytes = 64ull * esize;
        for (std::uint32_t y = 0; y < height_e; ++y) {
            for (std::uint32_t x = 0; x < width_e; ++x) {
                const std::size_t tile = static_cast<std::size_t>(y / 8) * tiles_per_row + (x / 8);
                const std::size_t off = tile * tile_bytes + static_cast<std::size_t>(thin_index(x & 7, y & 7)) * esize;
                if (off + esize > dst_avail) return false;
                std::memcpy(dst + off, src + (static_cast<std::size_t>(y) * width_e + x) * esize, esize);
            }
        }
        return true;
    }
    if (tiling == 14 || tiling == 10 || tiling == 2 || tiling == 3 || tiling == 4) {
        const MacroMode mm = macro_mode_for(esize);
        for (std::uint32_t y = 0; y < height_e; ++y) {
            for (std::uint32_t x = 0; x < width_e; ++x) {
                const std::size_t off = tiled2d_offset(x, y, pitch_e, esize, mm);
                if (off + esize > dst_avail) continue;
                std::memcpy(dst + off, src + (static_cast<std::size_t>(y) * width_e + x) * esize, esize);
            }
        }
        return true;
    }
    return false;
}

struct WriteBack {
    std::uint64_t base;
    DevBuffer staging;
};
std::vector<WriteBack> g_writebacks;             // recorded into the current command buffer
std::vector<WriteBack> g_slot_writebacks[kSlots];  // by submission slot

// Lays out `count` mip levels of a surface as the GPU addresses them: 1D
// levels are padded to 8x8 micro tiles, 2D levels to macro tiles, and 2D
// levels smaller than one macro tile fall back to 1D tiling. Every level
// starts on a 256-byte boundary; slices of a level are contiguous.
void compute_levels(Surface& sf, std::uint32_t count, std::uint32_t block) {
    sf.levels.clear();
    std::size_t src = 0, dst = 0;
    const bool is2d = sf.tiling == 14 || sf.tiling == 10 || sf.tiling == 2 || sf.tiling == 3 || sf.tiling == 4;
    const MacroMode mm = macro_mode_for(sf.esize);
    const std::uint32_t macro_pitch = 8 * 8 * mm.aspect, macro_height = 8 * mm.bank_height * mm.banks / mm.aspect;
    // POW2_PAD: a cube's six faces take eight slices of memory per level,
    // and every level's extent is padded to a power of two before tiling
    // alignment. Stepping six faces a level read the Hunter's Dream's BC6H
    // reflection cubes' second level from two slices early: every level
    // past the first decoded as random HDR blocks, and the materials that
    // sample a rough (blurry) level lit up green in the thousands.
    const std::uint32_t mem_slices = sf.pow2pad && sf.type != 10 ? std::bit_ceil(std::max(sf.slices, 1u)) : sf.slices;
    const auto pad = [&](std::uint32_t e) { return sf.pow2pad ? std::bit_ceil(std::max(e, 1u)) : e; };
    for (std::uint32_t l = 0; l < count; ++l) {
        Surface::Level lv{};
        const std::uint32_t wt = std::max(1u, sf.width >> l), ht = std::max(1u, (sf.type == 8 || sf.type == 12) ? 1u : sf.height >> l);
        lv.w_e = (wt + block - 1) / block;
        lv.h_e = (ht + block - 1) / block;
        const std::uint32_t pw_e = pad(lv.w_e), ph_e = pad(lv.h_e);
        lv.tiling = sf.tiling;
        if (l == 0) {
            lv.pitch_e = std::max(sf.pitch_e, pw_e);
        } else if (is2d) {
            lv.pitch_e = (pw_e + macro_pitch - 1) / macro_pitch * macro_pitch;
        } else if (sf.tiling == 8 || sf.tiling == 9) {
            lv.pitch_e = (pw_e + 7) / 8 * 8;
        } else {
            lv.pitch_e = (pw_e + 7) / 8 * 8;
        }
        if (is2d && (pw_e < macro_pitch || ph_e < macro_height)) {
            lv.tiling = 13;  // small levels are 1D tiled
            lv.pitch_e = (pw_e + 7) / 8 * 8;
        }
        std::uint32_t ph;
        if (lv.tiling == 8 || lv.tiling == 9) {
            ph = ph_e;
        } else if (lv.tiling == 13 || lv.tiling == 19 || lv.tiling == 5 || lv.tiling == 0) {
            ph = (ph_e + 7) / 8 * 8;
        } else {
            lv.pitch_e = (lv.pitch_e + macro_pitch - 1) / macro_pitch * macro_pitch;
            ph = (ph_e + macro_height - 1) / macro_height * macro_height;
        }
        lv.src_slice_bytes = static_cast<std::size_t>(lv.pitch_e) * ph * sf.esize;
        lv.src_slice_bytes = (lv.src_slice_bytes + 255) & ~static_cast<std::size_t>(255);
        lv.dst_slice_bytes = static_cast<std::size_t>(lv.w_e) * lv.h_e * sf.esize;
        lv.src_off = src;
        lv.dst_off = dst;
        const std::size_t slice_groups = lv.tiling == 19 ? (mem_slices + 3) / 4 * 4 : mem_slices;
        src += lv.src_slice_bytes * slice_groups;
        dst += lv.dst_slice_bytes * sf.slices;
        sf.levels.push_back(lv);
    }
    sf.total_src = src;
    sf.total_dst = dst;
}

// Where texture work goes on the command processor, microseconds: creating a
// new surface's image and memory, allocating an upload's staging buffer,
// untiling into it, recording the copy, and creating views.
std::atomic<std::uint64_t> g_tex_image_us{0}, g_tex_staging_us{0}, g_tex_untile_us{0}, g_tex_record_us{0}, g_tex_view_us{0};
std::uint64_t us_since(std::chrono::steady_clock::time_point t0) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
}

// BBHOST_TEX_PROBE=0xbase: the surface at that address, as it is created
// (texture_view_uncached) and every time it is uploaded again.
std::uint64_t tex_probe_base() {
    static const std::uint64_t probe = [] {
        const char* e = std::getenv("BBHOST_TEX_PROBE");
        return e ? std::strtoull(e, nullptr, 0) : 0ull;
    }();
    return probe;
}

// An upload's CPU half - untiling the guest memory into its staging span and
// hashing the memory for the next checks. The command processor did it as it
// recorded the copy, and a streaming burst (~190 textures entering an area or
// an enemy's first appearance) was 60-110 ms of untiling and 30 ms of hashing
// on its thread: past its backlog of ten submits, which stalled the game's
// render thread and with it the main loop. Now it records the copy and hands
// this to a pool; the submission that carries the copy waits for it
// (textures_before_submit), and the hashes are taken before anything reads
// them (settle_prep). The watch is armed when the copy is recorded, before
// the read: a write in between marks the surface and it is uploaded again.
//
// On by default (BBHOST_TEX_ASYNC=0 turns it off; BBHOST_TEX_PREP_THREADS,
// default 4). First measured 2026-09-19 it shortened the area-entry stall but
// steady play looked no better, across too few runs to tell the pool from
// run-to-run variance, and it stayed off. Three alternating soak rounds on a
// frozen seed (2026-09-27), once the command processor's translations were
// gone, settled it: the stalls it removes are the streaming bursts in play -
// ~200 uploads at once, 65-96 ms of untiling - 1-3 a soak (130-165 ms each)
// without it and none with it, the area-entry stall ~400 -> ~280 ms,
// frame rate within noise.
struct TexturePrep {
    const std::uint8_t* src = nullptr;
    std::uint8_t* dst = nullptr;
    std::uint64_t base = 0;
    std::vector<Surface::Level> levels;
    std::uint32_t slices = 1, esize = 0, width = 0, height = 0, tiling = 0;
    std::size_t total_src = 0, total_dst = 0;
    std::uint64_t content_hash = 0, exact_hash = 0;
    bool hash_only = false;     // the GPU untiles this upload; the prep only hashes the source
    std::atomic<int> state{0};  // 0 queued, 1 running, 2 done; changed under PrepPool's mutex
};

const bool g_tex_async = [] {
    const char* e = std::getenv("BBHOST_TEX_ASYNC");
    return !(e && e[0] == '0');
}();
// The creation-ahead path (textures_create_ahead_locked) always preps off
// the calling thread: it has frames of lead time, the command processor none.
thread_local bool t_force_async = false;
std::atomic<std::uint64_t> g_prep_waits{0}, g_prep_wait_us{0}, g_prep_inline{0};
// Untiles run on the thread that binds (the command processor, when the
// upload is not created ahead or prepped off it), against the prep pool's.
std::atomic<std::uint64_t> g_tex_untile_inline_us{0}, g_tex_untile_inline{0};

void run_prep(TexturePrep& p) {
    auto t0 = std::chrono::steady_clock::now();
    bool untiled = true;
    if (p.hash_only) {
        p.content_hash = content_hash_of(p.base, p.total_src);
        p.exact_hash = content_hash_of(p.base, p.total_src, true);
        return;
    }
    for (std::size_t l = 0; l < p.levels.size() && untiled; ++l) {
        const Surface::Level& lv = p.levels[l];
        for (std::uint32_t sl = 0; sl < p.slices && untiled; ++sl) {
            std::uint8_t* dst = p.dst + lv.dst_off + sl * lv.dst_slice_bytes;
            untiled = lv.tiling == 19
                          ? untile_thick_slice(dst, p.src + lv.src_off + (sl / 4) * lv.src_slice_bytes * 4, lv.src_slice_bytes * 4, lv.w_e, lv.h_e,
                                               lv.pitch_e, sl & 3, p.esize)
                          : untile_slice(dst, p.src + lv.src_off + sl * lv.src_slice_bytes, lv.src_slice_bytes, lv.tiling, lv.w_e, lv.h_e,
                                         lv.pitch_e, p.esize);
        }
    }
    g_tex_untile_us.fetch_add(us_since(t0), std::memory_order_relaxed);
    if (!untiled) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 8) host_log("texture: tiling mode %u not supported yet (%ux%u); uploading zeros", p.tiling, p.width, p.height);
        g_unsupported.fetch_add(1);
        std::memset(p.dst, 0, p.total_dst);
    }
    p.content_hash = content_hash_of(p.base, p.total_src);
    p.exact_hash = content_hash_of(p.base, p.total_src, true);
}

class PrepPool {
public:
    static PrepPool& get() {
        static PrepPool pool;
        return pool;
    }
    void submit(const std::shared_ptr<TexturePrep>& p) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push_back(p);
        }
        cv_.notify_one();
    }
    // Until p is done; a job no worker has taken yet runs here instead.
    void wait(TexturePrep& p) {
        std::unique_lock<std::mutex> lk(mu_);
        if (p.state == 2) return;
        const auto t0 = std::chrono::steady_clock::now();
        if (p.state == 0) {
            p.state = 1;
            lk.unlock();
            run_prep(p);
            lk.lock();
            p.state = 2;
            done_cv_.notify_all();
            g_prep_inline.fetch_add(1, std::memory_order_relaxed);
        } else {
            done_cv_.wait(lk, [&] { return p.state == 2; });
        }
        g_prep_waits.fetch_add(1, std::memory_order_relaxed);
        g_prep_wait_us.fetch_add(us_since(t0), std::memory_order_relaxed);
    }

private:
    PrepPool() {
        const char* e = std::getenv("BBHOST_TEX_PREP_THREADS");
        const unsigned n = e ? std::max(1u, static_cast<unsigned>(std::strtoul(e, nullptr, 10))) : 4u;
        for (unsigned i = 0; i < n; ++i) {
            std::thread([this, i] {
                char name[16];
                std::snprintf(name, sizeof(name), "bb-texprep%u", i);
                host_thread_set_name(name);
                for (;;) {
                    std::shared_ptr<TexturePrep> p;
                    {
                        std::unique_lock<std::mutex> lk(mu_);
                        cv_.wait(lk, [this] { return !queue_.empty(); });
                        p = std::move(queue_.front());
                        queue_.pop_front();
                        if (p->state != 0) continue;  // a waiter ran it
                        p->state = 1;
                    }
                    run_prep(*p);
                    std::lock_guard<std::mutex> lk(mu_);
                    p->state = 2;
                    done_cv_.notify_all();
                }
            }).detach();
        }
    }
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<std::shared_ptr<TexturePrep>> queue_;
};

// Preps the current recording's copies read; textures_before_submit() waits
// for them. Under the render lock.
// A sampled view in a format that cannot be a storage image, of an image made
// with storage use (its UNORM format can; the mutable-format flag gives the
// sRGB sibling its view): without a narrower usage the view inherits the
// image's, and a view may not carry a use its format lacks
// (VUID-VkImageViewCreateInfo-usage-02275; an sRGB view, every run, under the
// validation layer). Narrowed to sampling, which is all such a view does.
// Under g.mu, where views are made.
void narrow_sampled_view_usage(VkImageViewCreateInfo& vci, VkImageViewUsageCreateInfo& usage) {
    static std::unordered_map<int, bool> storable;  // by format: can be a storage image
    auto it = storable.find(static_cast<int>(vci.format));
    if (it == storable.end()) {
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(g.phys, vci.format, &fp);
        it = storable.emplace(static_cast<int>(vci.format), (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0).first;
    }
    if (it->second) return;
    usage = VkImageViewUsageCreateInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    usage.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    usage.pNext = vci.pNext;
    vci.pNext = &usage;
}

std::vector<std::shared_ptr<TexturePrep>> g_batch_preps;

bool prep_done(const TexturePrep& p) { return p.state.load(std::memory_order_acquire) == 2; }

// The surface's last upload, finished: its hashes are the surface's.
void settle_prep(Surface& sf) {
    if (!sf.prep) return;
    PrepPool::get().wait(*sf.prep);
    sf.content_hash = sf.prep->content_hash;
    sf.exact_hash = sf.prep->exact_hash;
    sf.prep.reset();
}

// Untiles the guest memory into a staging buffer and copies it into the image.
// BBHOST_GPU_UNTILE: 1 (default) untiles uploads on the GPU where it can
// (gpu_untile.cpp), 0 on the CPU always, 2 both - the GPU's feeds the image,
// the CPU's is kept beside it and the two are compared once the command
// buffer has retired (10,168 uploads, 0 differ).
const int g_gpu_untile = [] {
    const char* e = std::getenv("BBHOST_GPU_UNTILE");
    return e && *e ? std::atoi(e) : 1;
}();
// BBHOST_UNTILE_VRAM (on unless 0): the GPU untile read the surface's tiled
// bytes where they live - guest memory, imported host memory across the
// bus, element by element in tile order - and wrote its linear result to the
// host-visible staging, which the image copy then read back across the bus.
// In a Central Yharnam warp tour the untile was 4.5-24 ms of every slow
// command buffer, the image copy of the same bytes 0.3-1.3: the render
// thread waited for the command processor, which waited for the GPU, which
// was untiling. Now the tiled bytes cross the bus once, in one copy, into
// device-local scratch; the untile works scratch to scratch and the image
// copy reads scratch.
const bool g_untile_vram = [] {
    const char* e = std::getenv("BBHOST_UNTILE_VRAM");
    return !(e && e[0] == '0');
}();
std::atomic<std::uint64_t> g_untiled_vram{0}, g_untiled_vram_bytes{0};
std::atomic<std::uint64_t> g_gpu_untiled{0}, g_gpu_untiled_bytes{0}, g_gpu_untile_declined{0};

// The passes that untile `sf` on the GPU into `dst`, or false when a level
// cannot be: thick or unknown tiling (and 3D surfaces), an element size the
// shader does not copy, source bytes outside one imported mapping, or a
// linear or 1D level whose elements would reach past its slice (the CPU makes
// that surface zeros; it keeps those).
bool gpu_untile_plan(const Surface& sf, VkDeviceAddress dst, std::vector<UntileGpuPass>& passes, VkDeviceAddress src_copy = 0,
                     std::uint32_t first_level = 0) {
    if (sf.type == 10 || sf.thick || (sf.esize != 4 && sf.esize != 8 && sf.esize != 16) || (sf.base & 3)) return false;
    // src_copy: the surface's bytes copied whole to device-local memory, the
    // same layout from there; else read in place through the import.
    const VkDeviceAddress src = src_copy ? src_copy : guest_device_address(sf.base, sf.total_src);
    if (!src) return false;
    const MacroMode mm = macro_mode_for(sf.esize);
    const std::size_t dst_base = sf.levels[first_level].dst_off;
    for (std::size_t l = first_level; l < sf.levels.size(); ++l) {
        const Surface::Level& lv = sf.levels[l];
        UntileGpuPass pass;
        switch (lv.tiling) {
        case 8: case 9: pass.mode = 0; break;
        case 13: case 5: case 0: pass.mode = 1; break;
        case 14: case 10: case 2: case 3: case 4: pass.mode = 2; break;
        default: return false;
        }
        const std::uint64_t es = sf.esize;
        if (pass.mode == 0 && lv.h_e && ((static_cast<std::uint64_t>(lv.h_e) - 1) * lv.pitch_e + lv.w_e) * es > lv.src_slice_bytes) return false;
        if (pass.mode == 1 && lv.h_e && lv.w_e) {
            const std::uint64_t tpr = (lv.pitch_e + 7) / 8;
            const std::uint64_t last_tile = static_cast<std::uint64_t>((lv.h_e - 1) / 8) * tpr + (lv.w_e - 1) / 8;
            if ((last_tile + 1) * 64 * es > lv.src_slice_bytes) return false;
        }
        pass.src = src + lv.src_off;
        pass.dst = dst + lv.dst_off - dst_base;
        pass.src_slice_bytes = lv.src_slice_bytes;
        pass.dst_slice_bytes = lv.dst_slice_bytes;
        pass.width_e = lv.w_e;
        pass.height_e = lv.h_e;
        pass.pitch_e = lv.pitch_e;
        pass.esize = sf.esize;
        pass.slices = sf.slices;
        pass.bank_height = mm.bank_height;
        pass.aspect = mm.aspect;
        pass.banks = mm.banks;
        passes.push_back(pass);
    }
    return !passes.empty();
}

void check_stale_upload(const Surface& sf);

// The raw fallback is retained for production A/B with profiling disabled.
void texture_upload_barrier(VkPipelineStageFlags src, VkPipelineStageFlags dst, std::uint32_t nmem,
                            const VkMemoryBarrier* mem, std::uint32_t nimg, const VkImageMemoryBarrier* img) {
    if (texture_stream_enabled()) rec().pipeline_barrier(src, dst, 0, nmem, mem, 0, nullptr, nimg, img);
    else vkCmdPipelineBarrier(g_cmd(), src, dst, 0, nmem, mem, 0, nullptr, nimg, img);
}

bool upload_surface(Surface& sf, VkImageAspectFlags aspect, bool first, std::uint32_t first_level = 0) {
    // Only carry_surface_locked requests a suffix, after whole-plan admission.
    const std::size_t dst_base = sf.levels[first_level].dst_off;
    const std::size_t upload_bytes = sf.total_dst - dst_base;
    // An earlier upload still being prepared fills its own staging span (the
    // submission waits for it); its hashes are superseded by this one's.
    sf.prep.reset();
    if (!hle_kernel_va_mapped(sf.base, sf.total_src)) return false;
    if (sf.width >= 16 && sf.height >= 16) {
        tex_event(sf.base, sf.total_src, "upload 0x%llx %ux%u vkformat %d %zu level(s)%s%s", static_cast<unsigned long long>(sf.base), sf.width,
                  sf.height, static_cast<int>(sf.format), sf.levels.size(), first ? " (new)" : "",
                  sf.gpu_written ? " OVER GPU-WRITTEN" : "");
    }
    check_stale_upload(sf);
    // Watch before reading: a write after this faults and marks the surface,
    // and one before it is in what is read below (core/write_watch.h).
    sf.watched = !sf.gpu_written && write_watch_enabled() && now_ms() >= sf.watch_pause_until_ms &&
                 hle_kernel_write_watch(sf.base, sf.total_src, sf.watch);
    if (!first && sf.base == tex_probe_base()) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 24) {
            std::uint64_t head[4] = {};
            std::memcpy(head, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(sf.base)), sizeof(head));
            host_log("texture probe: 0x%llx uploaded again (upload %u, flip %llu); first %016llx %016llx %016llx %016llx",
                     static_cast<unsigned long long>(sf.base), sf.uploads + 1,
                     static_cast<unsigned long long>(hle_video_flip_count()), static_cast<unsigned long long>(head[0]),
                     static_cast<unsigned long long>(head[1]), static_cast<unsigned long long>(head[2]),
                     static_cast<unsigned long long>(head[3]));
        }
    }
    const auto* src = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(sf.base));
    DevBuffer staging;
    auto t_part = std::chrono::steady_clock::now();
    VkDeviceSize staging_offset = 0;
    // The device-local route (g_untile_vram): the guest bytes as a copy
    // source, and one scratch span holding the tiled copy then the linear
    // result. Taken only where the GPU untile would run anyway.
    Located vram_src;
    DevBuffer vram;
    VkDeviceSize vram_off = 0;
    const std::uint64_t vram_src_span = (sf.total_src + 255) & ~std::uint64_t{255};
    std::vector<UntileGpuPass> passes;
    bool in_vram = false;
    if (g_gpu_untile == 1 && g_untile_vram && !g_untile_check && sf.base != tex_probe_base() && untile_gpu_available_locked() &&
        guest_device_address(sf.base, sf.total_src)) {
        vram_src = locate(sf.base, sf.total_src);
        if (vram_src.buffer && vram_src.avail >= sf.total_src && acquire_scratch_locked(vram, vram_src_span + upload_bytes, vram_off)) {
            in_vram = gpu_untile_plan(sf, vram.address + vram_off + vram_src_span, passes, vram.address + vram_off, first_level);
            if (!in_vram) passes.clear();
        }
    }
    if (!in_vram && !acquire_staging_locked(staging, upload_bytes, staging_offset)) return false;
    g_tex_staging_us.fetch_add(us_since(t_part), std::memory_order_relaxed);
    auto prep = std::make_shared<TexturePrep>();
    prep->src = src;
    prep->dst = static_cast<std::uint8_t*>(staging.map);  // null on the device-local route: the prep only hashes
    prep->base = sf.base;
    prep->levels.assign(sf.levels.begin() + first_level, sf.levels.end());
    for (Surface::Level& lv : prep->levels) lv.dst_off -= dst_base;
    prep->slices = sf.slices;
    prep->esize = sf.esize;
    prep->width = sf.width;
    prep->height = sf.height;
    prep->tiling = sf.tiling;
    prep->total_src = sf.total_src;
    prep->total_dst = upload_bytes;
    // On the GPU where it can: the prep only hashes the source, off this
    // thread, and nothing waits for it before the submission.
    const bool on_gpu = in_vram || (g_gpu_untile && !g_untile_check && sf.base != tex_probe_base() && untile_gpu_available_locked() &&
                                    gpu_untile_plan(sf, staging.address + staging_offset, passes, 0, first_level));
    if (g_gpu_untile && !on_gpu) g_gpu_untile_declined.fetch_add(1, std::memory_order_relaxed);
    DevBuffer cpu_copy;  // BBHOST_GPU_UNTILE=2: the CPU's untile of the same upload, compared later
    if (on_gpu && g_gpu_untile == 2) {
        VkDeviceSize cpu_offset = 0;
        if (acquire_staging_locked(cpu_copy, upload_bytes, cpu_offset)) {
            std::memset(cpu_copy.map, 0, upload_bytes);
            prep->dst = static_cast<std::uint8_t*>(cpu_copy.map);
            run_prep(*prep);  // untiles into the copy and hashes
            sf.content_hash = prep->content_hash;
            sf.exact_hash = prep->exact_hash;
        } else {
            cpu_copy = DevBuffer{};
        }
    }
    if (on_gpu) {
        if (!cpu_copy.map) {
            prep->hash_only = true;
            PrepPool::get().submit(prep);
            sf.prep = prep;
        }
    } else
    // The checks compare two snapshots of the memory; the probe reads it
    // here: both as the command processor always did.
    if ((g_tex_async || t_force_async) && !g_untile_check && sf.base != tex_probe_base()) {
        PrepPool::get().submit(prep);
        g_batch_preps.push_back(prep);
        sf.prep = prep;
    } else {
        const auto t_inline = std::chrono::steady_clock::now();
        run_prep(*prep);
        g_tex_untile_inline_us.fetch_add(us_since(t_inline), std::memory_order_relaxed);
        g_tex_untile_inline.fetch_add(1, std::memory_order_relaxed);
        sf.content_hash = prep->content_hash;
        sf.exact_hash = prep->exact_hash;
    }
    t_part = std::chrono::steady_clock::now();
    begin_recording_locked();
    render_end_pass_locked();
    if (in_vram) {
        // Whatever wrote the guest bytes on the GPU before this is done, then
        // they cross the bus once.
        VkMemoryBarrier rb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        rb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        rb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        texture_upload_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 1, &rb, 0, nullptr);
        static const std::string fetch_name = "tex-fetch";
        profile_begin_locked(&fetch_name);
        const VkBufferCopy fetch{vram_src.offset, vram_off, sf.total_src};
        if (texture_stream_enabled()) rec().copy_buffer(vram_src.buffer, vram.buffer, 1, &fetch);
        else vkCmdCopyBuffer(g_cmd(), vram_src.buffer, vram.buffer, 1, &fetch);
        profile_end_locked();
        VkMemoryBarrier fb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        fb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        texture_upload_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1, &fb, 0, nullptr);
        g_untiled_vram.fetch_add(1, std::memory_order_relaxed);
        g_untiled_vram_bytes.fetch_add(sf.total_src, std::memory_order_relaxed);
    }
    if (on_gpu) {
        if (cpu_copy.map) {
            // A compare: the GPU's span starts from zeros as the CPU's copy
            // did, so elements neither writes (past a 2D slice) agree.
            if (texture_stream_enabled()) rec().fill_buffer(staging.buffer, staging_offset, (upload_bytes + 3) & ~std::size_t{3}, 0);
            else vkCmdFillBuffer(g_cmd(), staging.buffer, staging_offset, (upload_bytes + 3) & ~std::size_t{3}, 0);
            VkMemoryBarrier fb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            fb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            fb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            texture_upload_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1, &fb, 0, nullptr);
            g.slots[g.slot].untile_checks.push_back(
                {static_cast<const std::uint8_t*>(staging.map), static_cast<const std::uint8_t*>(cpu_copy.map), upload_bytes, sf.base});
        }
        // BBHOST_GPU_PROFILE: the untile and the copy as entries of their own.
        static const std::string untile_name = "tex-untile";
        profile_begin_locked(&untile_name);
        for (const UntileGpuPass& pass : passes) untile_gpu_record_locked(pass);
        profile_end_locked();
        VkMemoryBarrier ub{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        ub.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ub.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        texture_upload_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 1, &ub, 0, nullptr);
        g_gpu_untiled.fetch_add(1, std::memory_order_relaxed);
        g_gpu_untiled_bytes.fetch_add(upload_bytes, std::memory_order_relaxed);
    }
    const std::uint32_t layers = sf.type == 10 ? 1 : sf.layers;
    const std::uint32_t nlevels = static_cast<std::uint32_t>(sf.levels.size());
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = first ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = sf.image;
    b.subresourceRange = {aspect, first_level, nlevels - first_level, 0, layers};
    b.srcAccessMask = first ? 0 : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    texture_upload_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, nullptr, 1, &b);
    std::vector<VkBufferImageCopy> regions;
    for (std::uint32_t l = first_level; l < nlevels; ++l) {
        const Surface::Level& lv = sf.levels[l];
        for (std::uint32_t sl = 0; sl < layers; ++sl) {
            VkBufferImageCopy r{};
            r.bufferOffset = (in_vram ? vram_off + vram_src_span : staging_offset) + lv.dst_off - dst_base + sl * lv.dst_slice_bytes;
            r.imageSubresource = {aspect, l, sl, 1};
            r.imageExtent = {std::max(1u, sf.width >> l), (sf.type == 8 || sf.type == 12) ? 1u : std::max(1u, sf.height >> l),
                             sf.type == 10 ? std::max(1u, sf.depth >> l) : 1u};
            regions.push_back(r);
        }
    }
    static const std::string upload_copy_name = "tex-upload-copy";
    profile_begin_locked(&upload_copy_name);
    // Rec deep-copies these transient upload regions.
    if (texture_stream_enabled()) {
        rec().copy_buffer_to_image(in_vram ? vram.buffer : staging.buffer, sf.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    static_cast<std::uint32_t>(regions.size()), regions.data());
    } else {
        vkCmdCopyBufferToImage(g_cmd(), in_vram ? vram.buffer : staging.buffer, sf.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               static_cast<std::uint32_t>(regions.size()), regions.data());
    }
    profile_end_locked();
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    texture_upload_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, nullptr, 1, &b);
    // The staging span is free for reuse once this command buffer retired.
    g_tex_record_us.fetch_add(us_since(t_part), std::memory_order_relaxed);
    sf.exact_at_ms = now_ms();
    sf.checked_at_ms = now_ms();
    ++sf.uploads;
    g_uploads.fetch_add(1);
    g_upload_bytes.fetch_add(upload_bytes);
    return true;
}
std::map<std::uint64_t, Surface> g_surfaces;        // by base VA

// texture_view() is called for every image binding of every stage of every
// draw - 6.1M times in a 1800-flip run - and each call decodes the T# and then
// walks g_rts, g_snapshots (twice: the T# base and the V# base) and g_surfaces
// before hashing a view key and walking sf.views. The descriptors repeat: the
// whole run used about a thousand distinct ones, so a memo on the descriptor
// words answers 100% of the calls after the first.
//
// What the memo may *not* skip is the part of texture_view() that is not a
// function of the T#: the surface's content check, which re-uploads a texture
// the game streamed into. The entry keeps the surface so a hit still runs it.
// A render-target alias has no surface and needs no check.
//
// Anything that could change what a descriptor resolves to - a view or image
// destroyed, a surface replaced, a render target created or dropped - bumps
// g_view_epoch, and the memo drops everything it holds.
struct ViewMemoEntry {
    std::uint64_t key;    // hash of the descriptor words and flags; 0 if unused
    std::uint64_t epoch;  // live only while this is the current g_view_epoch
    VkImageView view;
    Surface* sf;  // the surface to keep fresh, or null for a render-target alias
    std::uint32_t w[8];
    std::uint32_t dim;
    std::uint8_t flags;
    bool arrayed;
    bool dims_ok;
};
// Open-addressed and power-of-two so a lookup is a mask and a linear probe over
// a flat array: an unordered_map costs a prime modulo and a node pointer-chase,
// which was most of what the memo saved. The epoch stamp means invalidation
// writes nothing - a stale entry simply reads as a free slot.
constexpr std::size_t kViewMemoSlots = 4096;  // a world run keeps about a thousand descriptors live
constexpr int kViewMemoProbes = 8;
std::vector<ViewMemoEntry> g_view_memo(kViewMemoSlots);
Surface* g_last_view_surface = nullptr;
// Set by texture_view_uncached when its view must be resolved again at every
// bind (a render target's region copy, rt_region_view): the memos skip it.
bool g_last_view_no_memo = false;
// Every caller of texture_view() is already serialised by the render lock (it
// mutates g_surfaces), so these need not be atomic.
std::uint64_t g_view_memo_hits = 0, g_view_memo_misses = 0, g_view_memo_evictions = 0, g_view_memo_mismatch = 0;
// 0 off, 1 on, 2 verify (resolve both ways and report any disagreement).
const int g_view_memo_mode = [] {
    const char* e = std::getenv("BBHOST_VIEW_MEMO");
    if (!e) return 1;
    if (*e == 'v') return 2;
    return std::atoi(e) ? 1 : 0;
}();
// The dirty walk (mark_surfaces_dirty_locked) takes g_surfaces from
// kLargeSurfaceSrc below a write, and the surfaces larger than that from this
// short map, from the largest one's size below it. With one bound for all, a
// single big surface widened every write's window for good: ~5,000 surfaces,
// ~50,000 fills and copies a second, and the walk was 6% of the command
// processor.
constexpr std::size_t kLargeSurfaceSrc = 4u << 20;
std::map<std::uint64_t, Surface*> g_large_surfaces;  // by base; Surface nodes in g_surfaces never move
std::size_t g_max_large_src = 0;
// Before either walk, a count of the surfaces over each 64 KiB page: most of
// those writes are GX's dynamic-buffer write-backs and constant-buffer fills,
// nowhere near a texture, and a write none of whose pages has a surface skips
// the walks (2.2% of the command processor, 2026-09-27). The counts go in
// 4 GiB blocks, allocated when a surface first lands in one; a surface past
// the last block turns the check off.
constexpr unsigned kOccupancyShift = 16;
constexpr std::uint64_t kOccupancyBlocks = 1024;  // 4 TiB of guest addresses
std::unique_ptr<std::uint32_t[]> g_occupancy[kOccupancyBlocks];
bool g_occupancy_off = false;
std::uint64_t g_dirty_walks = 0, g_dirty_walks_skipped = 0;
void occupancy_add(const Surface& sf, std::uint32_t delta) {
    if (!sf.total_src) return;
    const std::uint64_t last = (sf.base + sf.total_src - 1) >> kOccupancyShift;
    for (std::uint64_t p = sf.base >> kOccupancyShift; p <= last; ++p) {
        if ((p >> 16) >= kOccupancyBlocks) {
            g_occupancy_off = true;
            return;
        }
        std::unique_ptr<std::uint32_t[]>& block = g_occupancy[p >> 16];
        if (!block) block = std::make_unique<std::uint32_t[]>(std::size_t{1} << 16);
        block[p & 0xffff] += delta;
    }
}
bool surface_pages(std::uint64_t va, std::uint64_t end) {
    if (g_occupancy_off) return true;
    const std::uint64_t last = (end - 1) >> kOccupancyShift;
    for (std::uint64_t p = va >> kOccupancyShift; p <= last; ++p) {
        if ((p >> 16) >= kOccupancyBlocks) return true;
        const std::unique_ptr<std::uint32_t[]>& block = g_occupancy[p >> 16];
        if (block && block[p & 0xffff]) return true;
    }
    return false;
}
void surface_written(Surface& sf) {
    sf.written_flip = hle_video_flip_count();
    sf.written_ms = now_ms();
}
void note_surface_added(Surface& sf) {
    surface_written(sf);
    occupancy_add(sf, 1);
    if (sf.total_src <= kLargeSurfaceSrc) return;
    g_large_surfaces[sf.base] = &sf;
    g_max_large_src = std::max(g_max_large_src, sf.total_src);
}
void note_surface_erased(const Surface& sf) {
    occupancy_add(sf, ~0u);
    g_large_surfaces.erase(sf.base);
}

// The stale-read check. A drawn render target or a surface a shader or copy
// token wrote (gpu_written) holds bits the guest memory underneath does not -
// nothing is written back (surface_queue_writeback). A surface uploaded from
// that memory therefore reads what was there before the GPU wrote it: another
// view of a mip chain or layer the GPU built, memory a target was drawn into
// and then read as a texture at another address or format. Each such upload is
// counted, the first ones logged, and every one goes into the F12 history -
// the pattern behind the blood layers' noise and, likely, the new-character
// cutscenes' magenta. BBHOST_STALE_LOG=N logs N (default 40).
std::atomic<std::uint64_t> g_stale_over_rt{0}, g_stale_over_surface{0};
const int g_stale_log = [] {
    const char* e = std::getenv("BBHOST_STALE_LOG");
    return e && *e ? std::atoi(e) : 40;
}();
std::atomic<int> g_stale_logged{0};

void check_stale_upload(const Surface& sf) {
    if (!sf.total_src) return;
    const std::uint64_t lo = sf.base, hi = sf.base + sf.total_src;
    if (const RtImage* r = rt_overlapping_locked(lo, sf.total_src)) {
        g_stale_over_rt.fetch_add(1, std::memory_order_relaxed);
        tex_event(lo, sf.total_src, "STALE? upload 0x%llx %ux%u vkformat %d from memory render target 0x%llx %ux%u format %d holds (drawn, never written back)",
                  static_cast<unsigned long long>(lo), sf.width, sf.height, static_cast<int>(sf.format), static_cast<unsigned long long>(r->base),
                  r->width, r->height, static_cast<int>(r->format));
        if (g_stale_logged.fetch_add(1, std::memory_order_relaxed) < g_stale_log) {
            host_log("texture: STALE? 0x%llx %ux%u vkformat %d uploaded from memory render target 0x%llx %ux%u format %d was drawn into (flip %llu)",
                     static_cast<unsigned long long>(lo), sf.width, sf.height, static_cast<int>(sf.format), static_cast<unsigned long long>(r->base),
                     r->width, r->height, static_cast<int>(r->format), static_cast<unsigned long long>(hle_video_flip_count()));
        }
        return;
    }
    const Surface* owner = nullptr;
    auto consider = [&](const Surface& o) {
        if (&o == &sf || owner || !o.gpu_written || o.failed || !o.total_src) return;
        if (o.base < hi && o.base + o.total_src > lo) owner = &o;
    };
    for (auto it = g_surfaces.lower_bound(hi); it != g_surfaces.begin() && !owner;) {
        --it;
        if (it->first + kLargeSurfaceSrc <= lo) break;
        consider(it->second);
    }
    for (auto it = g_large_surfaces.lower_bound(hi); it != g_large_surfaces.begin() && !owner;) {
        --it;
        if (it->first + g_max_large_src <= lo) break;
        consider(*it->second);
    }
    if (!owner) return;
    g_stale_over_surface.fetch_add(1, std::memory_order_relaxed);
    tex_event(lo, sf.total_src, "STALE? upload 0x%llx %ux%u vkformat %d from memory GPU-written texture 0x%llx %ux%u vkformat %d holds",
              static_cast<unsigned long long>(lo), sf.width, sf.height, static_cast<int>(sf.format), static_cast<unsigned long long>(owner->base),
              owner->width, owner->height, static_cast<int>(owner->format));
    if (g_stale_logged.fetch_add(1, std::memory_order_relaxed) < g_stale_log) {
        host_log("texture: STALE? 0x%llx %ux%u vkformat %d uploaded from memory GPU-written texture 0x%llx %ux%u vkformat %d holds (flip %llu)",
                 static_cast<unsigned long long>(lo), sf.width, sf.height, static_cast<int>(sf.format), static_cast<unsigned long long>(owner->base),
                 owner->width, owner->height, static_cast<int>(owner->format), static_cast<unsigned long long>(hle_video_flip_count()));
    }
}


VkFormat unorm_sibling(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_SRGB: return VK_FORMAT_R8_UNORM;
    case VK_FORMAT_R8G8_SRGB: return VK_FORMAT_R8G8_UNORM;
    case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    // The compressed formats too: a GX texture is created UNORM (typeless)
    // and viewed sRGB, and a surface created at the resource's creation
    // must be the one the sRGB view then finds.
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK: return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case VK_FORMAT_BC2_SRGB_BLOCK: return VK_FORMAT_BC2_UNORM_BLOCK;
    case VK_FORMAT_BC3_SRGB_BLOCK: return VK_FORMAT_BC3_UNORM_BLOCK;
    case VK_FORMAT_BC7_SRGB_BLOCK: return VK_FORMAT_BC7_UNORM_BLOCK;
    default: return f;
    }
}
// Whether an image of this (UNORM) format may be viewed as sRGB later, so it
// is created mutable whichever view asks first.
bool has_srgb_sibling(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC7_UNORM_BLOCK:
        return true;
    default: return false;
    }
}
std::unordered_map<std::uint64_t, VkSampler> g_samplers;  // by S# hash
// Sampled views of render targets: base -> (swizzle, view type, layer range) -> view.
std::map<std::uint64_t, std::map<std::uint64_t, VkImageView>> g_rt_views;
std::uint64_t g_view_epoch = 1;
// How every sampled view and sampler handed to a draw was created, so a draw
// capture records what was bound rather than re-deriving it from the T#/S#.
std::unordered_map<VkImageView, ViewRecord> g_view_records;
std::unordered_map<VkSampler, VkSamplerCreateInfo> g_sampler_records;

void note_view(VkImageView view, VkImage image, VkImageCreateInfo image_info, VkImageViewCreateInfo view_info,
               std::uint64_t base, bool render_target) {
    image_info.pNext = nullptr;
    image_info.queueFamilyIndexCount = 0;
    image_info.pQueueFamilyIndices = nullptr;
    view_info.pNext = nullptr;
    g_view_records[view] = ViewRecord{image, image_info, view_info, base, render_target};
}

VkImageViewType view_type(std::uint32_t type, bool& arrayed, bool& cube) {
    arrayed = false;
    cube = false;
    switch (type) {
    case 8: return VK_IMAGE_VIEW_TYPE_1D;
    case 9: return VK_IMAGE_VIEW_TYPE_2D;
    case 10: return VK_IMAGE_VIEW_TYPE_3D;
    // A cube T# is six faces per cube, sampled by face id as array layers
    // (translate.cpp image_for); a Vulkan cube view would take a direction.
    case 11: cube = true; arrayed = true; return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case 12: arrayed = true; return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case 13: arrayed = true; return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    default: return VK_IMAGE_VIEW_TYPE_2D;
    }
}

}  // namespace

std::uint32_t tsharp_dim(std::uint32_t type, bool& arrayed) {
    bool cube;
    const VkImageViewType vt = view_type(type, arrayed, cube);
    if (cube) {  // tells the translator the T# is a cube; it binds a 2D array of faces
        arrayed = false;
        return 3;
    }
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D: case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return 0;
    case VK_IMAGE_VIEW_TYPE_3D: return 2;
    default: return 1;
    }
}

std::uint32_t tsharp_kind(const std::uint32_t* w) {
    switch (image_format((w[1] >> 20) & 0x3f, (w[1] >> 26) & 0xf).format) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32A32_UINT:
        return 1;
    case VK_FORMAT_R8_SINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32G32B32_SINT: case VK_FORMAT_R32G32B32A32_SINT:
        return 2;
    default:
        return 0;
    }
}

bool tsharp_sample_as_2d(std::uint32_t type, std::uint32_t base_array, std::uint32_t last_array,
                         bool shader_arrayed) {
    // shadPS4 Image::GetViewType(is_array): Color2DArray + !DA → Color2D.
    // ImageType: 9 Color2D, 13 Color2DArray, 14 Color2DMsaa, 15 Color2DMsaaArray.
    if (type == 9 || type == 14 || type == 15) return true;
    if (type != 13) return false;
    if (!shader_arrayed) return true;
    return last_array <= base_array;
}

// Sampled view of a render target (colour or depth) for aliasing.
// `first_layer` / `layer_count` pick the slices of a layered target; `arrayed`
// makes a 2D-array view (a cube's faces, an array T#).
//
// A T# with an integer format reads the target's bytes as integers, and the
// view is in that format: the characters' blood is copied from a UNORM
// target into its sRGB texture by a shader that loads and stores through
// R8G8B8A8_UINT T#s, a byte-for-byte copy. Through the target's own UNORM
// view the load returned 67/255 as a float, whose bits the integer store
// clamped to 255 - every bloodied texel came out full red and blue.
VkImageView rt_copied_view(RtImage& rt, const std::uint32_t* w, std::uint32_t width, std::uint32_t height);  // below
std::atomic<std::uint64_t> g_rt_unlisted_reads{0};

VkImageView render_target_view(RtImage& r, const std::uint32_t* tsharp, std::uint32_t first_layer, std::uint32_t layer_count,
                               bool arrayed) {
    first_layer = std::min(first_layer, r.layers - 1);
    layer_count = std::max(1u, std::min(layer_count, r.layers - first_layer));
    VkFormat format = r.format;
    if (!r.depth && r.mutable_format && tsharp_kind(tsharp) != 0) {
        const VkFormat as = image_format((tsharp[1] >> 20) & 0x3f, (tsharp[1] >> 26) & 0xf).format;
        if (format_bytes_per_pixel(as) == format_bytes_per_pixel(r.format)) format = as;
        if (r.format_listed && format != r.format) {
            // Outside the formats the target was made for (its compression
            // would be read wrongly): a copy of it, in an image of its own.
            VkFormat listed[5];
            const std::uint32_t n = rt_view_formats(r.format, listed);
            bool in = false;
            for (std::uint32_t i = 0; i < n; ++i) in |= listed[i] == format;
            if (!in) {
                g_rt_unlisted_reads.fetch_add(1, std::memory_order_relaxed);
                static std::atomic<int> logs{0};
                if (logs.fetch_add(1) < 16) {
                    host_log("texture: render target 0x%llx (format %d %ux%u) read as format %d, outside its view list: through a copy",
                             static_cast<unsigned long long>(r.base), static_cast<int>(r.format), r.width, r.height, static_cast<int>(format));
                }
                if (r.layers == 1 && first_layer == 0 && !arrayed) {
                    if (VkImageView v = rt_copied_view(r, tsharp, r.width, r.height)) return v;
                }
            }
        }
    }
    const std::uint64_t key = (tsharp[3] & 0xfff) | (static_cast<std::uint64_t>(first_layer) << 12) |
                              (static_cast<std::uint64_t>(layer_count) << 24) | (arrayed ? 1ull << 40 : 0) |
                              (static_cast<std::uint64_t>(format) << 41);
    auto& views = g_rt_views[r.base];
    if (auto it = views.find(key); it != views.end()) return it->second;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = r.image;
    vci.viewType = arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    // BGRA by the view's format: an R8G8B8A8_UINT view of a BGRA target
    // presents memory byte 0 as .r, which is what DST_SEL names.
    const bool bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_B8G8R8A8_UINT ||
                      format == VK_FORMAT_B8G8R8A8_SNORM;
    auto sel = [&](std::uint32_t v) {
        // DST_SEL names a memory component; a BGRA image already presents
        // memory byte 0 as .b and byte 2 as .r.
        if (bgra && v == 4) v = 6;
        else if (bgra && v == 6) v = 4;
        return swizzle(v);
    };
    vci.components = {sel(tsharp[3] & 7), sel((tsharp[3] >> 3) & 7), sel((tsharp[3] >> 6) & 7), sel((tsharp[3] >> 9) & 7)};
    vci.subresourceRange = {static_cast<VkImageAspectFlags>(r.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1,
                            first_layer, layer_count};
    if (r.depth) vci.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE};
    VkImageViewUsageCreateInfo narrowed{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    if (!r.depth) narrow_sampled_view_usage(vci, narrowed);
    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(g.device, &vci, nullptr, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
    views[key] = view;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = r.format;
    ici.extent = {r.width, r.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = r.layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                (r.depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    note_view(view, r.image, ici, vci, r.base, true);
    return view;
}

// The write watch's side of refresh_surface_locked (core/write_watch.h).
constexpr std::uint32_t kWatchPagesPerSecond = 2048;  // ~2k faults, a few ms of a thread
constexpr std::uint64_t kWatchPauseMs = 3000;
std::atomic<std::uint64_t> g_watch_uploads{0}, g_watch_sooner_ms{0}, g_watch_missed{0}, g_watch_paused{0};

// The census: what explains each upload. A surface's memory belongs
// to a GX resource (engine/gx_resources.h) or not; a re-upload is explained
// when GX wrote the resource since the last upload (a creation with initial
// data, a Map), when the command processor did (a fill or copy dispatch, the
// dirty mark), or by nothing the hooks saw - the writers the content hashes
// exist for. Zero unexplained over a walk is what lets the hashes go for
// registered resources.
std::atomic<std::uint64_t> g_census_new_registered{0}, g_census_new_registered_nodata{0}, g_census_new_unregistered{0};
std::atomic<std::uint64_t> g_census_gx{0}, g_census_cp{0}, g_census_unexplained{0}, g_census_unregistered{0};
// Unexplained re-uploads of registered surfaces by what noticed them: the
// write watch, or a hash (sampled, or the whole-surface one) - the hashes'
// catches are what removing them for registered surfaces would lose.
std::atomic<std::uint64_t> g_census_unexplained_watch{0}, g_census_unexplained_hash{0}, g_census_unexplained_exact{0};
// Re-uploads asked for because the memory was unmapped (the watch reports a
// forgotten page as written): not a write, and the upload cannot read it.
std::atomic<std::uint64_t> g_census_unmapped{0};

void census_upload(Surface& sf, const char* why) {
    if (why && !hle_kernel_va_mapped(sf.base, sf.total_src)) {
        g_census_unmapped.fetch_add(1, std::memory_order_relaxed);
        sf.gx_seq = 0;
        sf.gx_registered = false;
        sf.cp_dirty = false;
        return;
    }
    GxResourceInfo r;
    const bool registered = gx_resource_at(sf.base, sf.total_src, &r);
    if (!why) {
        if (!registered) g_census_new_unregistered.fetch_add(1, std::memory_order_relaxed);
        else if (r.initial_data) g_census_new_registered.fetch_add(1, std::memory_order_relaxed);
        else g_census_new_registered_nodata.fetch_add(1, std::memory_order_relaxed);
    } else if (!registered) {
        g_census_unregistered.fetch_add(1, std::memory_order_relaxed);
    } else if (r.write_seq > sf.gx_seq) {
        g_census_gx.fetch_add(1, std::memory_order_relaxed);
    } else if (sf.cp_dirty) {
        g_census_cp.fetch_add(1, std::memory_order_relaxed);
    } else {
        if (why[0] == 'w') g_census_unexplained_watch.fetch_add(1, std::memory_order_relaxed);
        else if (why[0] == 'c') g_census_unexplained_hash.fetch_add(1, std::memory_order_relaxed);
        else g_census_unexplained_exact.fetch_add(1, std::memory_order_relaxed);
        if (g_census_unexplained.fetch_add(1, std::memory_order_relaxed) < 12) {
            // The guest addresses among the recorded frames (host ones are the
            // HLE memcpy and its thunk), as Binary Ninja shows them.
            std::uint64_t pcs[24];
            const int n = write_watch_recent_faults(sf.base, sf.total_src, pcs, 24);
            std::string who;
            for (int i = 0; i < n; ++i) {
                if (pcs[i] > 0x100000000000ull) continue;
                const std::uint64_t bn = gx_guest_to_bn(pcs[i]);
                if (bn < 0x400000 || bn > 0x6000000) continue;
                char buf[32];
                std::snprintf(buf, sizeof(buf), " 0x%llx", static_cast<unsigned long long>(bn));
                who += buf;
            }
            std::uint64_t first = 0, last = 0;
            const int faults = write_watch_recent_fault_span(sf.base, sf.total_src, &first, &last);
            const std::uint64_t now_seq = write_watch_fault_seq();
            host_log("texture: census: 0x%llx %ux%u (%zu KiB) re-uploaded (%s) at flip %llu with no GX write seen: resource id %u type %u "
                     "usage %u [0x%llx, +0x%x) created at flip %llu%s; writers%s (%d faults in the ring, %llu to %llu faults ago; last upload %u)",
                     static_cast<unsigned long long>(sf.base), sf.width, sf.height, sf.total_src / 1024, why,
                     static_cast<unsigned long long>(hle_video_flip_count()), r.id, r.type, r.usage, static_cast<unsigned long long>(r.memory),
                     r.bytes, static_cast<unsigned long long>(r.created_flip), r.initial_data ? ", with data" : "", who.empty() ? " unknown" : who.c_str(),
                     faults, static_cast<unsigned long long>(faults ? now_seq - last : 0), static_cast<unsigned long long>(faults ? now_seq - first : 0),
                     sf.uploads);
        }
    }
    sf.gx_seq = registered ? r.write_seq : 0;
    sf.gx_registered = registered;
    sf.cp_dirty = false;
}

// A surface on a registered GX resource that is watched
// is not hashed. Over two soaks every change to one was noticed by the write
// watch (93 and 186) or explained by a GX creation over the same memory, and
// the sampled and whole-surface hashes caught none - they cost 2.6 s of
// command-processor time a walk for the surfaces they were hashing. One in
// 64 refreshes of such a surface still hashes it, as the check that this
// stays true (g_census_check_hashed / g_census_check_missed).
// BBHOST_TEX_HASH_REGISTERED=1 hashes them all again.
const bool g_hash_registered = [] {
    const char* e = std::getenv("BBHOST_TEX_HASH_REGISTERED");
    return e && e[0] == '1';
}();
std::atomic<std::uint64_t> g_census_hash_skipped{0}, g_census_check_hashed{0}, g_census_check_missed{0};
std::atomic<std::uint64_t> g_ahead_used{0}, g_ahead_replaced{0}, g_ahead_bind_before_prep{0};

// Keeps a surface in step with the guest memory behind it. Split out of
// texture_view() so a memo hit runs it too: it is the one part of resolving a
// T# that is not a function of the descriptor.
// BBHOST_REFRESH_ONCE=0: check a surface on every bind, as before. The check
// (write-watch pages, then a content hash when due) is per texture per draw,
// and a streaming surface marked hot is hashed on every one of them: a
// texture bound thirty times in a frame was hashed thirty times. Once per
// command buffer catches the same CPU writes, a few milliseconds later at
// most, and CP writes still land at once through the dirty mark.
const bool g_refresh_once = [] {
    const char* e = std::getenv("BBHOST_REFRESH_ONCE");
    return !e || e[0] != '0';
}();

void refresh_surface_locked(Surface& sf) {
    if (g_refresh_once) {
        const std::uint64_t stamp = g.record_serial + 1;  // 0 stays "never"
        if (sf.refreshed_serial == stamp) return;
        sf.refreshed_serial = stamp;
    }
    // A write the watch saw: upload now, whatever the hash schedule below
    // says. That schedule backs off to 2 s for a surface that sits still, so
    // a glyph drawn into a menu atlas that had been idle took up to that long
    // to appear.
    if (sf.watched && !sf.gpu_written) {
        if (const std::uint32_t pages = write_watch_dirty(sf.base, sf.total_src, sf.watch)) {
            const std::uint64_t now = now_ms();
            if (now - sf.watch_window_ms >= 1000) {
                sf.watch_window_ms = now;
                sf.watch_window_pages = 0;
            }
            sf.watch_window_pages += pages;
            if (sf.watch_window_pages > kWatchPagesPerSecond) {
                sf.watch_pause_until_ms = now + kWatchPauseMs;
                g_watch_paused.fetch_add(1, std::memory_order_relaxed);
                static std::atomic<int> logs{0};
                if (logs.fetch_add(1) < 6) {
                    host_log("texture: 0x%llx %ux%u is rewritten too often to watch (%u pages in a second); "
                             "hashing it instead for %llu ms",
                             static_cast<unsigned long long>(sf.base), sf.width, sf.height, sf.watch_window_pages,
                             static_cast<unsigned long long>(kWatchPauseMs));
                }
            }
            // When the hashes would next have looked, for the report.
            const std::uint64_t due =
                sf.hot ? 0u : std::min<std::uint32_t>(2000u, 200u << std::min<std::uint32_t>(sf.unchanged / 10, 4u));
            const std::uint64_t waited = now - sf.checked_at_ms;
            const std::uint64_t sooner = due > waited ? due - waited : 0;
            g_watch_sooner_ms.fetch_add(sooner, std::memory_order_relaxed);
            g_watch_uploads.fetch_add(1, std::memory_order_relaxed);
            census_upload(sf, "write watch");
            upload_surface(sf, VK_IMAGE_ASPECT_COLOR_BIT, false);
            sf.unchanged = 0;
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 12) {
                host_log("texture: 0x%llx %ux%u written (%u pages) - uploaded on this bind, %llu ms before a hash "
                         "was due (upload %u)",
                         static_cast<unsigned long long>(sf.base), sf.width, sf.height, pages,
                         static_cast<unsigned long long>(sooner), sf.uploads);
            }
            return;
        }
    }
    // A watched surface's CPU writes arrive through the watch, at once; the
    // hashes are its backstop for writes the watch cannot see (the command
    // processor's, other mappings'), so being hot - recently written - does
    // not put it on the every-bind cadence the way it does an unwatched one.
    // The glyph atlas is the case: written every frame in menus, watched, and
    // hashed whole on every check, ~3% of the command processor.
    const bool hot = sf.hot && !(sf.watched && !sf.gpu_written);
    if (now_ms() - sf.checked_at_ms <
        (hot ? 0u : std::min<std::uint32_t>(2000u, 200u << std::min<std::uint32_t>(sf.unchanged / 10, 4u)))) {
        return;
    }
    bool check_only = false;
    // Not while the command processor or the GPU has written the memory
    // since the last upload (cp_dirty, from mark_surfaces_dirty_locked): the
    // write watch sees only the CPU. The opening movie's frames reach its
    // luma and chroma textures by copy tokens, and a registered chroma plane
    // skipped its checks and stayed at the zeros it was first uploaded with -
    // the whole movie green.
    if (!g_hash_registered && sf.gx_registered && sf.watched && !sf.gpu_written && !sf.cp_dirty) {
        if (g_census_hash_skipped.fetch_add(1, std::memory_order_relaxed) % 64 != 63) {
            sf.checked_at_ms = now_ms();
            ++sf.unchanged;
            if (sf.hot) --sf.hot;
            return;
        }
        check_only = true;
        g_census_check_hashed.fetch_add(1, std::memory_order_relaxed);
    }
    // A surface that keeps hashing the same backs off - 200 ms, doubling
    // every ten unchanged checks, up to 2 s. Rehashing every bound texture
    // five times a second was a third of the command processor in the
    // world (~100k hashes per 300 flips); new and changing surfaces stay
    // on the short interval.
    // Streamed textures land after their first use: re-upload when the
    // guest memory changed since the last upload. Hashing every bound
    // texture is not free, so a surface that has been sitting still is only
    // checked five times a second - but one that keeps changing (a movie
    // frame, an atlas the game streams into) is checked on every bind.
    // Throttling those to 5 Hz was why cutscenes played at a few frames a
    // second while every other counter said 30.
    sf.checked_at_ms = now_ms();
    if (!hle_kernel_va_mapped(sf.base, sf.total_src)) {
        ++sf.unchanged;
        if (sf.hot) --sf.hot;
        return;
    }
    // Uploaded so recently that its prep has not finished: nothing to check
    // against yet, and waiting for it here serialized a surface rewritten
    // every frame back onto the command processor.
    if (sf.prep && !prep_done(*sf.prep)) return;
    settle_prep(sf);
    bool changed = content_hash_of(sf.base, sf.total_src) != sf.content_hash;
    // What the sampled hash cannot see: a change small enough to fall between
    // its samples (see content_hash_of). Nothing else would ever notice it, so
    // the whole surface is hashed on a slow cadence and compared against its
    // own recorded value. A miss then costs a fraction of a second rather than
    // being permanent - which is what a glyph vanishing from a menu was.
    //
    // Only surfaces the sampled hash is actually blind to (over its 256 KiB
    // whole-hash threshold) and small enough to read cheaply, at most one pass
    // per kExactMs each, under a global byte budget so a world load streaming
    // hundreds of them cannot turn this into the cost the sampling removed.
    bool exact_caught = false;
    if (!changed && g_exact_enabled && sf.total_src > kExactMaxBytes) {
        // Recorded rather than silently skipped: this is the set the fix does
        // not cover, and "it still happens in some menus" is what a growing
        // count here looks like.
        g_exact_too_big.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 6) {
            host_log("texture: 0x%llx is %zu KiB, too big for a whole-surface check; it keeps the sampled hash only",
                     static_cast<unsigned long long>(sf.base), sf.total_src / 1024);
        }
    }
    const std::uint64_t exact_every = hot ? kExactHotMs : kExactColdMs;
    if (!changed && g_exact_enabled && sf.total_src > kSampledAbove && sf.total_src <= kExactMaxBytes &&
        now_ms() - sf.exact_at_ms >= exact_every && exact_budget_take(sf.total_src)) {
        sf.exact_at_ms = now_ms();
        const std::uint64_t h = content_hash_of(sf.base, sf.total_src, true);
        if (h != sf.exact_hash) {
            changed = exact_caught = true;
            g_exact_catches.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (changed && check_only) g_census_check_missed.fetch_add(1, std::memory_order_relaxed);
    if (changed) {
        // On a watched surface this is a write that did not fault: through
        // another mapping of the same memory, or by the command processor.
        if (sf.watched) {
            g_watch_missed.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 8) {
                host_log("texture: 0x%llx %ux%u changed under its write watch - the hashes caught it",
                         static_cast<unsigned long long>(sf.base), sf.width, sf.height);
            }
        }
        census_upload(sf, exact_caught ? "whole-surface hash" : "content hash");
        upload_surface(sf, VK_IMAGE_ASPECT_COLOR_BIT, false);
        sf.hot = 30;  // about a second of frames before backing off again
        sf.unchanged = 0;
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 12) {
            host_log("texture: re-uploaded 0x%llx %ux%u after its memory changed%s (upload %u)",
                     static_cast<unsigned long long>(sf.base), sf.width, sf.height,
                     exact_caught ? " - the sampled hash had missed it" : "", sf.uploads);
        }
    } else {
        ++sf.unchanged;
        if (sf.hot) --sf.hot;
    }
}

// Returns the image view for a T#, uploading the texture when needed.
// A shader-written surface seen again through another format with the same
// block size and the same block grid keeps its pixels. The debug font is the
// case that found it: its loader stores the 17 pages of a BC1 512x512 array
// with a compute shader, through an R16G16B16A16_UINT 128x128 view of the same
// memory, and the glyph draw then samples it as BC1. The GPU image is the only
// copy of what the shader stored (surface_queue_writeback leaves guest memory
// alone), so replacing the surface by re-uploading the memory underneath
// uploaded the zeros that were there before the shader ran, and every glyph
// sampled nothing. The new image gets the old one's texels instead, block for
// block - Vulkan copies between an uncompressed and a compressed format whose
// texel size is the block size. shadPS4 handles a compressed view of an
// uncompressed image with the same block size the same way.
//
// The same holds when the new surface only has more mip levels. A shader that
// fills a mip chain one level at a time names levels 0..0, then 1..1, and so
// on: each T# asks for one more level than the surface has, so each replaced
// it. Carrying only a surface with no more levels than the old one uploaded the
// memory underneath every time instead, and every level stored before was lost
// - the characters' 512x512 blood masks came out as whatever that memory held
// before, rainbow noise on everyone a hit bloodied. Now the levels both images
// have are carried, and the ones the old image never had come from memory.
std::atomic<std::uint64_t> g_surface_carries{0};
const bool g_carry_added_mips = [] {
    const char* e = std::getenv("BBHOST_CARRY_ADDED_MIPS");
    return !(e && e[0] == '0');
}();
const bool g_texture_upload_stats = [] {
    const char* e = std::getenv("BBHOST_TEXTURE_UPLOAD_STATS");
    return e && e[0] == '1';
}();
std::atomic<std::uint64_t> g_carry_added_uploads{0}, g_carry_omitted_bytes{0}, g_carry_whole_uploads{0};

bool carry_surface_locked(Surface& dst, const Surface& src) {
    if (!src.gpu_written || src.block != 1 || src.esize != dst.esize || src.type == 10 || dst.type == 10 || src.thick ||
        dst.thick || dst.levels.empty() || src.levels.empty() || dst.info.arrayLayers > src.info.arrayLayers) {
        return false;
    }
    const std::size_t nlevels = std::min(dst.levels.size(), src.levels.size());
    for (std::size_t l = 0; l < nlevels; ++l) {
        if (src.levels[l].w_e != dst.levels[l].w_e || src.levels[l].h_e != dst.levels[l].h_e) return false;
    }
    // Levels the old image does not have: from memory first, then the rest on top.
    const bool partial = dst.levels.size() > nlevels;
    static const bool no_partial = [] { const char* e = std::getenv("BBHOST_CARRY_LEVELS"); return e && *e == '0'; }();
    if (partial && no_partial) return false;
    // The omitted common levels must not hide run_prep's whole-upload failure.
    // Every layout, including those levels, is admitted before compacting the
    // added levels. Otherwise retain the original upload and its all-zero path.
    const bool added_only = partial && g_carry_added_mips && texture_upload_layouts_supported(dst.levels, dst.esize);
    if (partial && !upload_surface(dst, VK_IMAGE_ASPECT_COLOR_BIT, true, added_only ? static_cast<std::uint32_t>(nlevels) : 0)) return false;
    if (partial && g_texture_upload_stats) {
        (added_only ? g_carry_added_uploads : g_carry_whole_uploads).fetch_add(1, std::memory_order_relaxed);
        if (added_only) g_carry_omitted_bytes.fetch_add(dst.levels[nlevels].dst_off, std::memory_order_relaxed);
    }
    begin_recording_locked();
    render_end_pass_locked();
    auto barrier = [](VkImage img, VkImageLayout from, VkAccessFlags src_access, VkAccessFlags dst_access,
                      std::uint32_t levels = VK_REMAINING_MIP_LEVELS) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, VK_REMAINING_ARRAY_LAYERS};
        b.srcAccessMask = src_access;
        b.dstAccessMask = dst_access;
        texture_upload_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, nullptr, 1, &b);
    };
    barrier(src.image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    if (added_only) {
        // Only added levels have left UNDEFINED so far; do not discard them
        // when initializing the common levels that this copy fills completely.
        barrier(dst.image, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT, static_cast<std::uint32_t>(nlevels));
    } else if (partial) {
        barrier(dst.image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    } else {
        barrier(dst.image, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    }
    std::vector<VkImageCopy> regions;
    for (std::uint32_t l = 0; l < nlevels; ++l) {
        VkImageCopy r{};
        r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, dst.info.arrayLayers};
        r.dstSubresource = r.srcSubresource;
        // In the source's texels, which (block 1) are the destination's blocks.
        r.extent = {src.levels[l].w_e, src.levels[l].h_e, 1};
        regions.push_back(r);
    }
    if (texture_stream_enabled()) {
        rec().copy_image(src.image, VK_IMAGE_LAYOUT_GENERAL, dst.image, VK_IMAGE_LAYOUT_GENERAL,
                         static_cast<std::uint32_t>(regions.size()), regions.data());
    } else {
        vkCmdCopyImage(g_cmd(), src.image, VK_IMAGE_LAYOUT_GENERAL, dst.image, VK_IMAGE_LAYOUT_GENERAL,
                       static_cast<std::uint32_t>(regions.size()), regions.data());
    }
    barrier(dst.image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    // The memory underneath is what it was; only a change to it from here on
    // is a new upload.
    settle_prep(dst);
    dst.content_hash = content_hash_of(dst.base, dst.total_src);
    dst.exact_hash = content_hash_of(dst.base, dst.total_src, true);
    dst.exact_at_ms = now_ms();
    dst.checked_at_ms = now_ms();
    dst.gpu_written = true;
    if (g_surface_carries.fetch_add(1) < 16) {
        host_log("texture: 0x%llx seen again (%ux%u x%u, block %u, %zu levels) keeps the texels a shader stored "
                 "through the old view (%ux%u, %zu levels)",
                 static_cast<unsigned long long>(dst.base), dst.width, dst.height, dst.info.arrayLayers, dst.block,
                 dst.levels.size(), src.width, src.height, src.levels.size());
    }
    return true;
}

// A T# smaller than the render target it names samples the target's top-left
// width x height, as the hardware addresses it: the game draws a smaller
// level into memory it allocated for a larger one and reads it back through a
// T# of the smaller size. The new-character cutscenes allocate the glare
// pyramid's levels at their size (a 392x331 base) and the game's own pyramid
// (314x265) reuses them: a view of the whole image stretched the rest - pixels
// left from the cutscene - over each level, the error compounded down the
// pyramid, and the frame went magenta until a reload.
// So the region is copied into an image of the T#'s size and that is sampled.
// The target is drawn again every frame and such a T# is bound a handful of
// times a frame, so the copy is made at every bind (the memos skip the view).
// BBHOST_RT_REGIONS=0: the whole image, as before.
struct RtRegion {
    RtImage img;             // not in g_rts; img.base is a key of its own for g_rt_views
    VkImage source = VK_NULL_HANDLE;  // the target image it was made for
};
std::map<std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, int>, RtRegion> g_rt_regions;
std::atomic<std::uint64_t> g_rt_region_copies{0}, g_rt_region_packets{0}, g_rt_smaller_than_tsharp{0}, g_rt_reused_as_texture{0};

void drop_rt_region(RtRegion& reg) {
    invalidate_rt_views(reg.img.base);
    defer_destroy_view(reg.img.view);
    defer_destroy_image(reg.img.image, reg.img.memory);
}

void textures_drop_rt_regions_locked(std::uint64_t base) {
    for (auto it = g_rt_regions.lower_bound(std::make_tuple(base, 0u, 0u, std::numeric_limits<int>::min()));
         it != g_rt_regions.end() && std::get<0>(it->first) == base;) {
        drop_rt_region(it->second);
        it = g_rt_regions.erase(it);
    }
}

VkImageView rt_region_view(RtImage& rt, const std::uint32_t* w, std::uint32_t width, std::uint32_t height) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_RT_REGIONS");
        return !(e && e[0] == '0');
    }();
    if (!on || rt.depth || rt.layers != 1 || width > rt.width || height > rt.height || (width == rt.width && height == rt.height)) {
        return VK_NULL_HANDLE;
    }
    return rt_copied_view(rt, w, width, height);
}

void record_region_copy(VkCommandBuffer cmd, VkImage src, VkImage dst, std::uint32_t width, std::uint32_t height, bool dst_initialised) {
    const auto barrier = [cmd](VkImage img, VkImageLayout from, VkAccessFlags src_access, VkAccessFlags dst_access) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.srcAccessMask = src_access;
        b.dstAccessMask = dst_access;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    };
    barrier(src, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
    barrier(dst, dst_initialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy r{};
    r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.dstSubresource = r.srcSubresource;
    r.extent = {width, height, 1};
    vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    barrier(dst, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}

// The target's top-left width x height copied into an image of its own, made
// at every bind, and the T#'s view of that (rt_region_view; a format the
// target's view list leaves out, render_target_view).
VkImageView rt_copied_view(RtImage& rt, const std::uint32_t* w, std::uint32_t width, std::uint32_t height) {
    const auto key = std::make_tuple(rt.base, width, height, static_cast<int>(rt.format));
    auto it = g_rt_regions.find(key);
    if (it != g_rt_regions.end() && it->second.source != rt.image) {
        drop_rt_region(it->second);
        g_rt_regions.erase(it);
        it = g_rt_regions.end();
    }
    if (it == g_rt_regions.end()) {
        if (g_rt_regions.size() > 256) {  // targets come and go with areas; the regions follow them out
            for (auto& [k, old] : g_rt_regions) drop_rt_region(old);
            g_rt_regions.clear();
        }
        RtRegion nr;
        nr.source = rt.image;
        nr.img.base = (rt.base ^ (0x5ull << 60)) + (static_cast<std::uint64_t>(width) << 44) + (static_cast<std::uint64_t>(height) << 32);
        nr.img.format = rt.format;
        nr.img.width = width;
        nr.img.height = height;
        nr.img.layers = 1;
        nr.img.depth = false;
        nr.img.mutable_format = rt.mutable_format;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.flags = rt.mutable_format ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = rt.format;
        ici.extent = {width, height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(g.device, &ici, nullptr, &nr.img.image) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(g.device, nr.img.image, &req);
        if (!image_memory_alloc(req, nr.img.memory)) {
            vkDestroyImage(g.device, nr.img.image, nullptr);
            return VK_NULL_HANDLE;
        }
        vkBindImageMemory(g.device, nr.img.image, nr.img.memory.memory, nr.img.memory.offset);
        it = g_rt_regions.emplace(key, nr).first;
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 16) {
            host_log("texture: T# %ux%u over render target 0x%llx %ux%u format %d samples its top-left region (copied at every bind)", width,
                     height, static_cast<unsigned long long>(rt.base), rt.width, rt.height, static_cast<int>(rt.format));
        }
    }
    RtRegion& reg = it->second;
    begin_recording_locked();
    // Into the draw's packet, started for it now, with the pass end before it:
    // recorded in place, the copy waited for the recorder to finish the draws
    // before it (~11 copies a frame on a Steam Deck at 960x600, 2.5% of its
    // command processor). BBHOST_REGION_COPY_PACKETS=0: in place.
    static const bool packets = [] {
        const char* e = std::getenv("BBHOST_REGION_COPY_PACKETS");
        return !(e && e[0] == '0');
    }();
    DrawCmds* open = packets ? draw_packet_early_locked() : nullptr;
    render_end_pass_locked();
    const bool initialised = reg.img.initialised;
    reg.img.initialised = true;
    if (open && open->copy_region(rt.image, reg.img.image, width, height, initialised)) {
        g_rt_region_packets.fetch_add(1, std::memory_order_relaxed);
    } else {
        record_region_copy(g_cmd(), rt.image, reg.img.image, width, height, initialised);
    }
    g_rt_region_copies.fetch_add(1, std::memory_order_relaxed);
    g_last_view_no_memo = true;
    return render_target_view(reg.img, w, 0, 1, false);
}

// A T# that starts where a mip level of a shader-written surface starts reads
// that level as the hardware does: its top-left width x height, from the
// memory the shader stored into. The GPU image is the only copy of what the
// shader stored (surface_queue_writeback), so uploading the memory under the
// T# read what was there before. The tone map's exposure is this: a compute
// shader (baad65ce) reduces the scene's luminance into level 5 of a 256x1
// chain at 0x157169d00, and the tone map and the YEBIS composite read the
// result through a 1x1 T# at that level's address (0x15716db00) - which read
// stale memory every frame, so the exposure never followed the scene. The
// level is copied into an image of the T#'s size at every bind, as a render
// target's region is (rt_region_view). BBHOST_LEVEL_ALIAS=0: memory, as before.
struct LevelAlias {
    RtImage img;                      // not in g_rts; img.base is a key of its own for g_rt_views
    VkImage source = VK_NULL_HANDLE;  // the surface image it was made for
};
std::map<std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, int>, LevelAlias> g_level_aliases;
std::atomic<std::uint64_t> g_level_alias_copies{0};

VkImageView level_alias_view(std::uint64_t base, const std::uint32_t* w, VkFormat image_fmt, VkFormat view_fmt, std::uint32_t esize,
                             std::uint32_t block, std::uint32_t tiling, std::uint32_t width, std::uint32_t height) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_LEVEL_ALIAS");
        return !(e && e[0] == '0');
    }();
    if (!on || block != 1) return VK_NULL_HANDLE;
    const Surface* owner = nullptr;
    std::uint32_t level = 0;
    auto consider = [&](const Surface& o) {
        if (owner || !o.gpu_written || o.failed || !o.image || o.thick || o.block != 1 || o.esize != esize || o.type == 10 ||
            o.base >= base || o.base + o.total_src <= base) {
            return;
        }
        for (std::uint32_t l = 1; l < o.levels.size(); ++l) {
            if (o.base + o.levels[l].src_off != base) continue;
            if (o.levels[l].tiling != tiling || width > std::max(1u, o.width >> l) || height > std::max(1u, o.height >> l)) return;
            owner = &o;
            level = l;
            return;
        }
    };
    for (auto it = g_surfaces.lower_bound(base); it != g_surfaces.begin() && !owner;) {
        --it;
        if (it->first + kLargeSurfaceSrc <= base) break;
        consider(it->second);
    }
    for (auto it = g_large_surfaces.lower_bound(base); it != g_large_surfaces.begin() && !owner;) {
        --it;
        if (it->first + g_max_large_src <= base) break;
        consider(*it->second);
    }
    if (!owner) return VK_NULL_HANDLE;
    const auto key = std::make_tuple(owner->base, level, width, height, static_cast<int>(image_fmt));
    auto it = g_level_aliases.find(key);
    if (it != g_level_aliases.end() && it->second.source != owner->image) {
        invalidate_rt_views(it->second.img.base);
        defer_destroy_view(it->second.img.view);
        defer_destroy_image(it->second.img.image, it->second.img.memory);
        g_level_aliases.erase(it);
        it = g_level_aliases.end();
    }
    if (it == g_level_aliases.end()) {
        LevelAlias na;
        na.source = owner->image;
        na.img.base = (base ^ (0x6ull << 60)) + (static_cast<std::uint64_t>(width) << 44) + (static_cast<std::uint64_t>(height) << 32);
        na.img.format = image_fmt;
        na.img.width = width;
        na.img.height = height;
        na.img.layers = 1;
        na.img.depth = false;
        na.img.mutable_format = image_fmt != view_fmt || has_srgb_sibling(image_fmt);
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.flags = na.img.mutable_format ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = image_fmt;
        ici.extent = {width, height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(g.device, &ici, nullptr, &na.img.image) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(g.device, na.img.image, &req);
        if (!image_memory_alloc(req, na.img.memory)) {
            vkDestroyImage(g.device, na.img.image, nullptr);
            return VK_NULL_HANDLE;
        }
        vkBindImageMemory(g.device, na.img.image, na.img.memory.memory, na.img.memory.offset);
        it = g_level_aliases.emplace(key, na).first;
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 16) {
            host_log("texture: T# %ux%u at 0x%llx reads level %u of shader-written 0x%llx %ux%u (copied at every bind)", width, height,
                     static_cast<unsigned long long>(base), level, static_cast<unsigned long long>(owner->base), owner->width,
                     owner->height);
        }
        tex_event(base, static_cast<std::uint64_t>(width) * height * esize, "level alias 0x%llx %ux%u: level %u of shader-written 0x%llx %ux%u",
                  static_cast<unsigned long long>(base), width, height, level, static_cast<unsigned long long>(owner->base), owner->width,
                  owner->height);
    }
    LevelAlias& la = it->second;
    begin_recording_locked();
    render_end_pass_locked();
    auto barrier = [](VkImage img, std::uint32_t mip, VkImageLayout from, VkAccessFlags src_access, VkAccessFlags dst_access) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
        b.srcAccessMask = src_access;
        b.dstAccessMask = dst_access;
        vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    };
    barrier(owner->image, level, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
    barrier(la.img.image, 0, la.img.initialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT);
    la.img.initialised = true;
    VkImageCopy r{};
    r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
    r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.extent = {width, height, 1};
    vkCmdCopyImage(g_cmd(), owner->image, VK_IMAGE_LAYOUT_GENERAL, la.img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    barrier(la.img.image, 0, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    g_level_alias_copies.fetch_add(1, std::memory_order_relaxed);
    g_last_view_no_memo = true;
    return render_target_view(la.img, w, 0, 1, false);
}

VkImageView texture_view_uncached(const std::uint32_t* w, bool storage, bool& dims_ok, std::uint32_t& dim, bool& arrayed,
                         bool shader_arrayed, bool r128) {
    dims_ok = true;
    std::uint64_t base = tsharp_base(w);
    std::uint32_t dfmt = (w[1] >> 20) & 0x3f, nfmt = (w[1] >> 26) & 0xf;
    std::uint32_t width = (w[2] & 0x3fff) + 1, height = ((w[2] >> 14) & 0x3fff) + 1;
    std::uint32_t type = (w[3] >> 28) & 0xf, tiling = (w[3] >> 20) & 0x1f;
    std::uint32_t base_level = (w[3] >> 12) & 0xf, last_level = (w[3] >> 16) & 0xf;
    std::uint32_t depth = (w[4] & 0x1fff) + 1, pitch = ((w[4] >> 13) & 0x3fff) + 1;
    std::uint32_t base_array = w[5] & 0x1fff, last_array = (w[5] >> 13) & 0x1fff;
    const std::uint64_t vbase = static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32);
    // Alias before the V# rewrite: YEBIS samples its HDR targets as type 13/14
    // (and sometimes a V# whose byte address is the colour target). Requiring
    // type==9 uploaded 2D-thin guest memory instead, which is why 0x158010000
    // had black macrotile columns while 0x16ab40000 was a clean clinic.
    auto alias_rt = [&](std::uint64_t addr) -> VkImageView {
        RtImage* rt = find_render_target(addr);
        if (!rt || !rt->initialised || storage) return VK_NULL_HANDLE;
        // A cube T# over its target samples the faces as layers (the render
        // target is layered when the game draws faces through CB_COLOR*_VIEW);
        // an arrayed T# samples its slice range.
        if (type == 11) {
            dim = 1;
            arrayed = true;
            static std::set<std::uint64_t> logged;
            if (logged.size() < 16 && logged.insert(addr).second) {
                host_log("texture: cube T# at 0x%llx samples %u of %u layers of its render target (%ux%u)",
                         static_cast<unsigned long long>(addr), std::min(rt->layers, 6 * depth), rt->layers, rt->width, rt->height);
            }
            return render_target_view(*rt, w, 0, 6 * depth, true);
        }
        if ((type == 12 || type == 13) && shader_arrayed && last_array > base_array && rt->layers > 1) {
            dim = 1;
            arrayed = true;
            return render_target_view(*rt, w, base_array, last_array - base_array + 1, true);
        }
        const bool as_2d = tsharp_sample_as_2d(type, base_array, last_array, shader_arrayed) ||
                           ((type & 8u) == 0 && !r128 && !rt->depth && rt->width > 1 && rt->height > 1);
        if (!as_2d) {
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 12) {
                host_log("texture: RT 0x%llx not aliased (type %u array %u..%u storage %d %ux%u)",
                         static_cast<unsigned long long>(addr), type, base_array, last_array, storage ? 1 : 0, rt->width,
                         rt->height);
            }
            return VK_NULL_HANDLE;
        }
        dim = 1;
        arrayed = false;
        if (type != 9) {
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 16) {
                host_log("texture: aliased RT 0x%llx as 2D (T# type %u tiling %u %ux%u)",
                         static_cast<unsigned long long>(addr), type, tiling, rt->width, rt->height);
            }
        }
        if (VkImageView v = rt_region_view(*rt, w, width, height)) return v;
        if (width > rt->width || height > rt->height) {
            // Memory a target was drawn into and the game has since given to a
            // texture of the T#'s size (the registry's live resource there):
            // the texture, not the old target (a 1024x1024 texture read a
            // 256x256 blood layer's target).
            std::uint32_t live[8];
            if (gx_resource_tsharp(addr, live) && (live[2] & 0x0fffffff) == (w[2] & 0x0fffffff)) {
                g_rt_reused_as_texture.fetch_add(1, std::memory_order_relaxed);
                return VK_NULL_HANDLE;
            }
            // The target is smaller than the T# that reads it: what the game
            // drew past its edge was clipped away (a size learned for another
            // target at this address), and the sample stretches what is left.
            g_rt_smaller_than_tsharp.fetch_add(1, std::memory_order_relaxed);
            static std::set<std::uint64_t> logged;
            if (logged.size() < 24 && logged.insert(addr ^ (static_cast<std::uint64_t>(width) << 40) ^ (static_cast<std::uint64_t>(height) << 52)).second) {
                host_log("texture: T# %ux%u samples render target 0x%llx of only %ux%u (drawn past its edge, clipped)", width, height,
                         static_cast<unsigned long long>(addr), rt->width, rt->height);
            }
        }
        return render_target_view(*rt, w, rt->layers > 1 ? base_array : 0, 1, false);  // one slice (BASE_ARRAY)
    };
    if (VkImageView v = alias_rt(base)) return v;
    if (vbase && vbase != base) {
        if (VkImageView v = alias_rt(vbase)) return v;
    }
    // MIMG r128: 4-dword V# sampled as an image. shadPS4 binds these as
    // formatted buffers. Upload a linear 1D of NUM_RECORDS texels at the byte
    // address, exact size, no 8x8 tile pad.
    // Without r128, type&8==0 is an invalid T# (Image::Valid is type&8).
    // YEBIS 7d668276 puts a 257-record V# in an 8-dword slot (logged as
    // 258x1 type 2); treating that as 1D compiled the 2D sample as Dim1D
    // and fed a LUT into a blur. shadPS4 substitutes Image::Null.
    if (!r128 && (type & 8u) == 0) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 16) {
            host_log("texture: T# type %u at 0x%llx is not a valid image (r128=0, V# at 0x%llx)", type,
                     static_cast<unsigned long long>(base), static_cast<unsigned long long>(vbase));
        }
        g_unsupported.fetch_add(1);
        return VK_NULL_HANDLE;
    }
    if (r128) {
        const std::uint32_t stride = (w[1] >> 16) & 0x3fff;
        const std::uint32_t records = w[2];
        const std::size_t bytes = stride ? static_cast<std::size_t>(stride) * records : records;
        if (!vbase || bytes == 0 || bytes > (16u << 20) || !hle_kernel_va_mapped(vbase, 16)) {
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 8) {
                host_log("texture: descriptor at 0x%llx is not a T# (type %u); V# unusable",
                         static_cast<unsigned long long>(base), type);
            }
            g_unsupported.fetch_add(1);
            return VK_NULL_HANDLE;
        }
        base = vbase;
        type = 8;
        tiling = 8;
        height = 1;
        depth = 1;
        base_level = 0;
        last_level = 0;
        base_array = 0;
        last_array = 0;
        dfmt = (w[3] >> 15) & 0xf;
        nfmt = (w[3] >> 12) & 7;
        const FormatInfo probe = image_format(dfmt, nfmt);
        std::uint32_t es = stride ? stride : (probe.bytes_per_element ? probe.bytes_per_element : 4u);
        if (probe.format == VK_FORMAT_UNDEFINED || (stride && probe.bytes_per_element != stride)) {
            if (es == 16) {
                dfmt = 14;
                nfmt = 7;
            } else if (es == 8) {
                dfmt = 11;
                nfmt = 7;
            } else if (es == 4) {
                dfmt = 4;
                nfmt = 7;
            } else if (es == 2) {
                dfmt = 2;
                nfmt = 0;
            } else {
                dfmt = 1;
                nfmt = 0;
                es = 1;
            }
        }
        width = stride ? records : static_cast<std::uint32_t>(bytes / es);
        if (width == 0) width = 1;
        pitch = width;
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 12) {
            host_log("texture: V# at 0x%llx %u records stride %u as linear 1D %ux1 (%zu bytes)",
                     static_cast<unsigned long long>(vbase), records, stride, width, bytes);
        }
    }
    const bool flat_2d = tsharp_sample_as_2d(type, base_array, last_array, shader_arrayed);
    dim = flat_2d ? 1u : tsharp_dim(type, arrayed);
    if (flat_2d) arrayed = false;
    if (type == 11) {  // bound as the 2D array of faces the translator declares for a cube
        dim = 1;
        arrayed = true;
    }
    const FormatInfo fi = image_format(dfmt, nfmt);
    bool cube = false;
    // view_type(13) sets arrayed=true and would undo the flatten: that is
    // a040875f "want dim 1, got dim 1 array" on a type-13 T# whose LAST_ARRAY
    // is 0 (a Color2D view of layer 0; DEPTH=6 is the resource, not the view).
    VkImageViewType vt;
    std::uint32_t layers;
    if (flat_2d) {
        vt = VK_IMAGE_VIEW_TYPE_2D;
        // The view is one layer, but the image is the whole array: views of
        // different layers must be different layers. With a one-layer image
        // they were all layer 0, so a loader writing an array a layer at a
        // time (the debug font: 17 compute stores, one per layer) left only
        // its last layer, and two views of one array replaced each other's
        // surface on every alternation.
        layers = type == 13 ? std::max(depth, last_array + 1) : 1;
        if (type != 9) {
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 16) {
                host_log("texture: flattened T# type %u at 0x%llx to 2D %ux%u (array %u..%u depth %u da %d)", type,
                         static_cast<unsigned long long>(base), width, height, base_array, last_array, depth,
                         shader_arrayed ? 1 : 0);
            }
        }
    } else {
        vt = view_type(type, arrayed, cube);
        layers = (type == 12 || type == 13) ? std::max(depth, last_array >= base_array ? last_array - base_array + 1 : 1u)
                                            : cube ? 6 * depth : 1;  // DEPTH counts whole cubes
    }
    const std::uint32_t slices = type == 10 ? depth : layers;
    if (fi.format == VK_FORMAT_UNDEFINED || width == 0 || height == 0 || slices == 0 || base == 0) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 8) host_log("texture: unsupported T# dfmt=%u nfmt=%u type=%u %ux%u", dfmt, nfmt, type, width, height);
        g_unsupported.fetch_add(1);
        return VK_NULL_HANDLE;
    }
    // Formats whose reading differs from ours on hardware, not yet handled
    // (shadPS4 facts): 11_11_10 swaps red and blue relative to 10_11_11,
    // and a 32-bit UNORM T# reads as value / (2^32 - 1), not the raw bits.
    if (dfmt == 7 || (nfmt == 0 && (dfmt == 4 || dfmt == 11 || dfmt == 13 || dfmt == 14))) {
        static std::set<std::uint32_t> logged;
        if (logged.size() < 8 && logged.insert(dfmt | (nfmt << 8)).second) {
            host_log("texture: T# dfmt %u nfmt %u at 0x%llx (%ux%u) reads differently on hardware (%s)", dfmt, nfmt,
                     static_cast<unsigned long long>(base), width, height,
                     dfmt == 7 ? "11_11_10: red and blue swapped" : "32-bit UNORM: normalized, not raw");
        }
    }
    const VkFormat image_fmt = unorm_sibling(fi.format);
    const VkFormat view_fmt = storage ? image_fmt : fi.format;
    if (!storage && !r128 && type == 9 && flat_2d && base_level == 0 && last_level == 0) {
        if (VkImageView v = level_alias_view(base, w, image_fmt, fi.format, fi.bytes_per_element, fi.block, tiling, width, height)) return v;
    }
    auto it = g_surfaces.find(base);
    Surface replaced;  // what this lookup replaces, for carry_surface_locked
    bool have_replaced = false;
    if (it != g_surfaces.end()) {
        Surface& sf = it->second;
        if (sf.failed) return VK_NULL_HANDLE;
        const std::uint32_t want_levels = (type == 10 || tiling == 19) ? 1u : std::min<std::uint32_t>(last_level + 1, 16u);
        if (sf.format != image_fmt || sf.width != width || sf.height != height || sf.type != type || sf.layers != layers ||
            sf.levels.size() < want_levels) {
            // Memory re-used for a different image: replace the surface. The
            // old image stays alive until the current submission retires, so
            // this costs nothing - idling the device here was most of a world
            // load's GPU wait, at a couple of thousand replacements a minute.
            g_surface_replacements.fetch_add(1);
            if (sf.ahead && !sf.ahead_used) {
                if (g_ahead_replaced.fetch_add(1, std::memory_order_relaxed) < 8) {
                    host_log("texture: ahead surface 0x%llx (vkformat %d %ux%u type %u layers %u %zu levels) replaced at its first bind by vkformat %d "
                             "%ux%u type %u layers %u %u levels",
                             static_cast<unsigned long long>(base), static_cast<int>(sf.format), sf.width, sf.height, sf.type, sf.layers,
                             sf.levels.size(), static_cast<int>(image_fmt), width, height, type, layers, want_levels);
                }
            }
            tex_event(base, sf.total_src, "replace 0x%llx: was vkformat %d %ux%u %zu level(s)%s, now vkformat %d %ux%u %u level(s) (T# dfmt %u "
                      "type %u levels %u..%u%s)",
                      static_cast<unsigned long long>(base), static_cast<int>(sf.format), sf.width, sf.height, sf.levels.size(),
                      sf.gpu_written ? " gpu-written" : "", static_cast<int>(image_fmt), width, height, want_levels, dfmt, type,
                      base_level, last_level, storage ? " storage" : "");
            replaced = sf;  // its image lives until this submission retires
            have_replaced = true;
            bump_view_epoch();
            for (auto& v : sf.views) defer_destroy_view(v.second);
            defer_destroy_image(sf.image, sf.memory);
            note_surface_erased(it->second);
            g_surfaces.erase(it);
            it = g_surfaces.end();
        }
    }
    if (it == g_surfaces.end()) {
        Surface sf;
        sf.base = base;
        sf.format = image_fmt;
        sf.width = width;
        sf.height = height;
        sf.depth = type == 10 ? depth : 1;
        sf.layers = layers;
        sf.type = type;
        sf.tiling = tiling;
        sf.width_e = (width + fi.block - 1) / fi.block;
        sf.height_e = (height + fi.block - 1) / fi.block;
        sf.pitch_e = std::max(sf.width_e, (pitch + fi.block - 1) / fi.block);
        sf.esize = fi.bytes_per_element;
        sf.block = fi.block;
        sf.slices = slices;
        sf.thick = tiling == 19;
        sf.pow2pad = ((w[3] >> 25) & 1) != 0;
        // 3D textures: level 0 only for now.
        const std::uint32_t nlevels = (type == 10 || tiling == 19) ? 1u : std::min<std::uint32_t>(last_level + 1, 16u);
        compute_levels(sf, nlevels, fi.block);
        sf.slice_src_bytes = sf.levels[0].src_slice_bytes;
        sf.slice_dst_bytes = sf.levels[0].dst_slice_bytes;
        if (!hle_kernel_va_mapped(base, sf.total_src)) {
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 8) host_log("texture: T# at 0x%llx (%zu bytes) not mapped", static_cast<unsigned long long>(base), sf.total_src);
            return VK_NULL_HANDLE;
        }
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.flags = (image_fmt != fi.format || has_srgb_sibling(image_fmt)) ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
        ici.imageType = type == 10 ? VK_IMAGE_TYPE_3D : (type == 8 || type == 12) ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
        ici.format = image_fmt;
        ici.extent = {width, (type == 8 || type == 12) ? 1u : height, type == 10 ? depth : 1u};
        ici.mipLevels = static_cast<std::uint32_t>(sf.levels.size());
        ici.arrayLayers = type == 10 ? 1 : layers;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(g.phys, image_fmt, &fp);
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ici.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        const auto t_image = std::chrono::steady_clock::now();
        if (vkCreateImage(g.device, &ici, nullptr, &sf.image) != VK_SUCCESS) {
            host_log("texture: image creation failed (format %d %ux%u)", image_fmt, width, height);
            sf.failed = true;
            note_surface_added(g_surfaces[base] = sf);
            return VK_NULL_HANDLE;
        }
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(g.device, sf.image, &req);
        if (!image_memory_alloc(req, sf.memory)) {
            vkDestroyImage(g.device, sf.image, nullptr);
            sf.image = VK_NULL_HANDLE;
            sf.failed = true;
            note_surface_added(g_surfaces[base] = sf);
            return VK_NULL_HANDLE;
        }
        vkBindImageMemory(g.device, sf.image, sf.memory.memory, sf.memory.offset);
        g_tex_image_us.fetch_add(us_since(t_image), std::memory_order_relaxed);
        sf.info = ici;
        const bool carried = have_replaced && carry_surface_locked(sf, replaced);
        if (have_replaced && replaced.gpu_written) {
            tex_event(base, sf.total_src, "  0x%llx: the GPU-written texels were %s (%zu of %zu levels)",
                      static_cast<unsigned long long>(base), carried ? "carried over" : "LOST - uploaded from guest memory instead",
                      std::min(sf.levels.size(), replaced.levels.size()), sf.levels.size());
        }
        census_upload(sf, nullptr);
        if (!carried && !upload_surface(sf, VK_IMAGE_ASPECT_COLOR_BIT, true)) {
            sf.failed = true;
            note_surface_added(g_surfaces[base] = sf);
            return VK_NULL_HANDLE;
        }
        // BBHOST_TEX_PROBE=0xbase: every surface created at that address, with
        // what the guest memory it was uploaded from held.
        const std::uint64_t probe = tex_probe_base();
        if (probe && base == probe) {
            const auto* g8 = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(base));
            std::size_t nonzero = 0;
            for (std::size_t i = 0; i < sf.slice_src_bytes; ++i) nonzero += g8[i] != 0;
            std::uint64_t head[4] = {};
            std::memcpy(head, g8, sizeof(head));
            host_log("texture probe: 0x%llx %ux%u x%u dfmt=%u nfmt=%u tiling=%u type=%u; layer 0 is %zu bytes, %zu non-zero; "
                     "first %016llx %016llx %016llx %016llx",
                     static_cast<unsigned long long>(base), width, height, slices, dfmt, nfmt, tiling, type, sf.slice_src_bytes,
                     nonzero, static_cast<unsigned long long>(head[0]), static_cast<unsigned long long>(head[1]),
                     static_cast<unsigned long long>(head[2]), static_cast<unsigned long long>(head[3]));
        }
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 40 && width > 1) {
            host_log("texture: uploaded 0x%llx %ux%ux%u dfmt=%u nfmt=%u tiling=%u type=%u (%zu KiB)", static_cast<unsigned long long>(base),
                     width, height, slices, dfmt, nfmt, tiling, type, sf.slice_dst_bytes * slices / 1024);
        }
        it = g_surfaces.emplace(base, sf).first;
        note_surface_added(it->second);
        ++g_tiling_hist[tiling];
        if (sf.base < 0x157010000ull && sf.base + sf.total_src > 0x15700c000ull) {
            host_log("texture: surface 0x%llx %ux%u type=%u tiling=%u src=%zu overlaps 0x15700c000-0x157010000 (end 0x%llx)",
                     static_cast<unsigned long long>(sf.base), sf.width, sf.height, sf.type, sf.tiling, sf.total_src,
                     static_cast<unsigned long long>(sf.base + sf.total_src));
        } else if (sf.base >= 0x155400000ull && sf.base < 0x165400000ull) {
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 24) {
                host_log("texture: GFX-heap surface 0x%llx %ux%u type=%u tiling=%u src=%zu (end 0x%llx)",
                         static_cast<unsigned long long>(sf.base), sf.width, sf.height, sf.type, sf.tiling, sf.total_src,
                         static_cast<unsigned long long>(sf.base + sf.total_src));
            }
        }
    } else {
        refresh_surface_locked(it->second);
    }
    Surface& sf = it->second;
    g_last_view_surface = &sf;
    if (sf.ahead && !sf.ahead_used && !t_force_async) {
        sf.ahead_used = true;
        g_ahead_used.fetch_add(1, std::memory_order_relaxed);
        if (sf.prep && !prep_done(*sf.prep)) g_ahead_bind_before_prep.fetch_add(1, std::memory_order_relaxed);
    }
    // Array views start at BASE_ARRAY; the surface holds every slice.
    const std::uint32_t view_first = (type == 12 || type == 13) ? std::min(base_array, sf.layers - 1) : 0;
    const std::uint32_t view_layers = flat_2d ? 1u : type == 10 ? 1u : (type == 12 || type == 13) ? std::max(1u, std::min(sf.layers - view_first, last_array >= base_array ? last_array - base_array + 1 : 1u)) : sf.layers;
    std::uint64_t vkey = fnv1a(&view_fmt, sizeof(view_fmt));
    const std::uint32_t swz = storage ? 0xfffu : (w[3] & 0xfff);
    vkey = fnv1a(&swz, 4, vkey);
    vkey = fnv1a(&storage, 1, vkey);
    vkey = fnv1a(&view_first, 4, vkey);
    vkey = fnv1a(&view_layers, 4, vkey);
    const std::uint32_t nlv = static_cast<std::uint32_t>(sf.levels.size());
    const std::uint32_t view_base_level = std::min(base_level, nlv - 1);
    const std::uint32_t view_levels = storage ? 1u : std::max(1u, std::min(nlv - view_base_level, last_level >= base_level ? last_level - base_level + 1 : 1u));
    vkey = fnv1a(&view_base_level, 4, vkey);
    vkey = fnv1a(&view_levels, 4, vkey);
    auto vit = sf.views.find(vkey);
    if (vit != sf.views.end()) return vit->second;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = sf.image;
    vci.viewType = vt;
    vci.format = view_fmt;
    if (!storage) vci.components = {swizzle(w[3] & 7), swizzle((w[3] >> 3) & 7), swizzle((w[3] >> 6) & 7), swizzle((w[3] >> 9) & 7)};
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, view_base_level, view_levels, view_first, view_layers};
    VkImageViewUsageCreateInfo narrowed{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    if (!storage) narrow_sampled_view_usage(vci, narrowed);
    VkImageView view = VK_NULL_HANDLE;
    const auto t_view = std::chrono::steady_clock::now();
    if (vkCreateImageView(g.device, &vci, nullptr, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
    g_tex_view_us.fetch_add(us_since(t_view), std::memory_order_relaxed);
    sf.views[vkey] = view;
    note_view(view, sf.image, sf.info, vci, sf.base, false);
    return view;
}

// Who moved the epoch, by return address (BBHOST_EPOCH_SITES=1): every bump
// empties both view memos, so a bump every frame costs every binding a resolve.
namespace {
struct EpochSite { void* site; std::uint64_t n; };
EpochSite g_epoch_sites[16];
const bool g_epoch_sites_on = [] { const char* e = std::getenv("BBHOST_EPOCH_SITES"); return e && e[0] == '1'; }();
}
__attribute__((noinline)) void bump_view_epoch() {
    ++g_view_epoch;
    if (!g_epoch_sites_on) return;
    void* site = __builtin_return_address(0);
    for (EpochSite& e : g_epoch_sites) {
        if (e.site == site || !e.site) { e.site = site; ++e.n; return; }
    }
}
std::string view_epoch_sites_report() {
    if (!g_epoch_sites_on) return std::string();
    std::string out = "texture: view epoch bumps by site:";
    char buf[64];
    for (const EpochSite& e : g_epoch_sites) {
        if (!e.site) break;
#if !defined(_WIN32)
        Dl_info info{};
        const std::uintptr_t off = dladdr(e.site, &info) && info.dli_fbase
                                       ? reinterpret_cast<std::uintptr_t>(e.site) - reinterpret_cast<std::uintptr_t>(info.dli_fbase)
                                       : reinterpret_cast<std::uintptr_t>(e.site);
#else
        const std::uintptr_t off = reinterpret_cast<std::uintptr_t>(e.site);  // the address: no dladdr
#endif
        std::snprintf(buf, sizeof(buf), " +0x%llx:%llu", static_cast<unsigned long long>(off), static_cast<unsigned long long>(e.n));
        out += buf;
    }
    return out;
}
std::uint64_t view_epoch() { return g_view_epoch; }

// The memo in front of texture_view_uncached(). See ViewMemoEntry above for
// what it may and may not skip.
VkImageView texture_view(const std::uint32_t* w, bool storage, bool& dims_ok, std::uint32_t& dim, bool& arrayed,
                         bool shader_arrayed, bool r128) {
    if (!g_view_memo_mode) return texture_view_uncached(w, storage, dims_ok, dim, arrayed, shader_arrayed, r128);
    const int nw = r128 ? 4 : 8;
    const std::uint8_t flags = static_cast<std::uint8_t>((storage ? 1 : 0) | (shader_arrayed ? 2 : 0) | (r128 ? 4 : 0));
    std::uint64_t key = fnv1a(w, static_cast<std::size_t>(nw) * 4);
    key = fnv1a(&flags, 1, key) | 1;
    std::size_t idx = key & (kViewMemoSlots - 1);
    ViewMemoEntry* free_slot = nullptr;
    for (int probe = 0; probe < kViewMemoProbes; ++probe, idx = (idx + 1) & (kViewMemoSlots - 1)) {
        ViewMemoEntry& e = g_view_memo[idx];
        if (e.epoch != g_view_epoch) {
            if (!free_slot) free_slot = &e;
            continue;
        }
        // The hash is verified against the words, so a collision costs a
        // resolve rather than the wrong texture.
        if (e.key != key || e.flags != flags || std::memcmp(e.w, w, static_cast<std::size_t>(nw) * 4) != 0) continue;
        if (e.sf) refresh_surface_locked(*e.sf);
        ++g_view_memo_hits;
        if (g_view_memo_mode == 2) {
            bool real_ok = true;
            std::uint32_t real_dim = 0;
            bool real_arrayed = false;
            const VkImageView real = texture_view_uncached(w, storage, real_ok, real_dim, real_arrayed, shader_arrayed, r128);
            if (real != e.view || real_dim != e.dim || real_arrayed != e.arrayed || real_ok != e.dims_ok) {
                ++g_view_memo_mismatch;
                static std::atomic<int> logs{0};
                if (logs.fetch_add(1) < 16) {
                    host_log("texture: view memo disagreed for T# at 0x%llx (memo dim %u arrayed %d, resolved dim %u arrayed %d, %s view)",
                             static_cast<unsigned long long>(tsharp_base(w)), e.dim, e.arrayed ? 1 : 0, real_dim,
                             real_arrayed ? 1 : 0, real == e.view ? "same" : "a different");
                }
                dims_ok = real_ok;
                dim = real_dim;
                arrayed = real_arrayed;
                return real;
            }
        }
        dims_ok = e.dims_ok;
        dim = e.dim;
        arrayed = e.arrayed;
        return e.view;
    }
    g_last_view_surface = nullptr;
    g_last_view_no_memo = false;
    const VkImageView view = texture_view_uncached(w, storage, dims_ok, dim, arrayed, shader_arrayed, r128);
    ++g_view_memo_misses;
    if (g_last_view_no_memo) return view;
    // Only a descriptor that resolved is remembered. A T# whose memory is not
    // mapped yet, or whose surface failed to upload, has to be tried again:
    // the game streams the texture in later, and a remembered null would never
    // come back.
    if (view == VK_NULL_HANDLE) return view;
    if (!free_slot) {
        free_slot = &g_view_memo[key & (kViewMemoSlots - 1)];
        ++g_view_memo_evictions;
    }
    free_slot->key = key;
    free_slot->epoch = g_view_epoch;
    free_slot->view = view;
    free_slot->sf = g_last_view_surface;
    free_slot->dim = dim;
    free_slot->flags = flags;
    free_slot->arrayed = arrayed;
    free_slot->dims_ok = dims_ok;
    std::memcpy(free_slot->w, w, static_cast<std::size_t>(nw) * 4);
    if (nw < 8) std::memset(free_slot->w + nw, 0, (8 - nw) * 4);
    return view;
}

// The memo keyed by a shader-resource view's id. Direct-mapped by
// id; an entry lives while the view epoch it was made in is current, like the
// words memo's. A miss resolves through the words (texture_view_uncached),
// which also names the surface to keep fresh.
struct IdViewEntry {
    std::uint32_t id = 0;
    std::uint8_t flags = 0;
    bool arrayed = false, dims_ok = false;
    std::uint64_t epoch = 0;
    VkImageView view = VK_NULL_HANDLE;
    Surface* sf = nullptr;
    std::uint32_t dim = 0;
    // The recording its surface was last kept fresh in (record_serial + 1):
    // under BBHOST_REFRESH_ONCE a surface is checked once a recording, and the
    // Surface itself is a cold line to read just to learn that it was.
    std::uint64_t refreshed = 0;
};
constexpr std::size_t kIdViewSlots = 8192;
IdViewEntry g_id_views[kIdViewSlots];
std::uint64_t g_id_view_hits = 0, g_id_view_misses = 0;

VkImageView texture_view_by_id(std::uint32_t id, const std::uint32_t* w, bool storage, bool& dims_ok, std::uint32_t& dim, bool& arrayed,
                               bool shader_arrayed, bool r128) {
    const std::uint8_t flags = static_cast<std::uint8_t>((storage ? 1 : 0) | (shader_arrayed ? 2 : 0) | (r128 ? 4 : 0));
    IdViewEntry& e = g_id_views[id & (kIdViewSlots - 1)];
    if (e.id == id && e.epoch == g_view_epoch && e.flags == flags) {
        const std::uint64_t stamp = g.record_serial + 1;
        if (e.sf && (!g_refresh_once || e.refreshed != stamp)) {
            refresh_surface_locked(*e.sf);
            e.refreshed = stamp;
        }
        ++g_id_view_hits;
        dims_ok = e.dims_ok;
        dim = e.dim;
        arrayed = e.arrayed;
        return e.view;
    }
    ++g_id_view_misses;
    g_last_view_surface = nullptr;
    g_last_view_no_memo = false;
    const VkImageView view = texture_view_uncached(w, storage, dims_ok, dim, arrayed, shader_arrayed, r128);
    // As the words memo: only a view that resolved is remembered - a texture
    // not streamed in yet has to be tried again (and a region copy is made
    // again at every bind).
    if (view != VK_NULL_HANDLE && !g_last_view_no_memo) e = IdViewEntry{id, flags, arrayed, dims_ok, g_view_epoch, view, g_last_view_surface, dim, g.record_serial + 1};
    return view;
}

void texture_view_by_id_prefetch(std::uint32_t id) { __builtin_prefetch(&g_id_views[id & (kIdViewSlots - 1)]); }

std::string texture_view_by_id_report() {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "views found by their shader-resource view's id %llu, resolved from the T# for a new id %llu",
                  static_cast<unsigned long long>(g_id_view_hits), static_cast<unsigned long long>(g_id_view_misses));
    return buf;
}

VkSampler sampler_for(const std::uint32_t* s, bool compare) {
    // The last sampler made for these words, in front of the map (samplers
    // are never destroyed).
    struct Memo {
        std::uint32_t w[4];
        bool compare;
        VkSampler sampler = VK_NULL_HANDLE;
    };
    static Memo memo[64];  // under g.mu
    Memo& m = memo[(s[0] ^ (s[1] * 31u) ^ (s[2] * 17u) ^ (s[3] * 7u) ^ (compare ? 1u : 0u)) & 63];
    if (m.sampler && m.compare == compare && std::memcmp(m.w, s, 16) == 0) return m.sampler;
    std::uint64_t key = fnv1a(s, 16);
    key = fnv1a(&compare, 1, key);
    auto it = g_samplers.find(key);
    if (it != g_samplers.end()) {
        std::memcpy(m.w, s, 16);
        m.compare = compare;
        m.sampler = it->second;
        return it->second;
    }
    auto address = [](std::uint32_t c) {
        switch (c & 7) {
        case 0: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case 1: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case 2: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 3: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
        case 4: case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        }
    };
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.addressModeU = address(s[0] & 7);
    sci.addressModeV = address((s[0] >> 3) & 7);
    sci.addressModeW = address((s[0] >> 6) & 7);
    const std::uint32_t aniso = (s[0] >> 9) & 7;
    const std::uint32_t mag = (s[2] >> 20) & 3, min = (s[2] >> 22) & 3, mip = (s[2] >> 26) & 3;
    sci.magFilter = (mag & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    sci.minFilter = (min & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    sci.mipmapMode = mip == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.anisotropyEnable = (mag >= 2 || min >= 2) && aniso ? VK_TRUE : VK_FALSE;
    // MAX_ANISO_RATIO goes past what the device supports (up to 128 here
    // against a limit of 16), so clamp instead of failing sampler creation.
    sci.maxAnisotropy = std::min(static_cast<float>(1u << aniso), g.max_anisotropy);
    sci.minLod = static_cast<float>(s[1] & 0xfff) / 256.0f;
    sci.maxLod = static_cast<float>((s[1] >> 12) & 0xfff) / 256.0f;
    if (mip == 0) sci.maxLod = 0.25f;
    std::int32_t bias = static_cast<std::int32_t>(s[2] & 0x3fff);
    if (bias & 0x2000) bias -= 0x4000;
    sci.mipLodBias = static_cast<float>(bias) / 256.0f;
    // The comparison only applies to samplers the shader uses with
    // image_sample_c* (Vulkan needs compareEnable for Dref ops and forbids it
    // for plain sampling); DEPTH_COMPARE_FUNC gives the function.
    const std::uint32_t cmp = (s[0] >> 12) & 7;
    static const VkCompareOp ops[8] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_LESS, VK_COMPARE_OP_EQUAL, VK_COMPARE_OP_LESS_OR_EQUAL,
                                       VK_COMPARE_OP_GREATER, VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_ALWAYS};
    sci.compareEnable = compare ? VK_TRUE : VK_FALSE;
    sci.compareOp = ops[cmp];
    sci.borderColor = ((s[3] >> 30) & 3) == 2 ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : ((s[3] >> 30) & 3) == 1 ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    sci.unnormalizedCoordinates = VK_FALSE;
    if ((s[0] >> 15) & 1) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 8) {
            host_log("texture: S# force_unnormalized (bit 15); translator converts texel coordinates");
        }
    }
    VkSampler out = VK_NULL_HANDLE;
    if (vkCreateSampler(g.device, &sci, nullptr, &out) != VK_SUCCESS) {
        sci.anisotropyEnable = VK_FALSE;
        sci.compareEnable = VK_FALSE;
        vkCreateSampler(g.device, &sci, nullptr, &out);
    }
    g_samplers[key] = out;
    if (out) g_sampler_records[out] = sci;
    return out;
}

// After a storage-image dispatch the VkImage is the source of truth. Tiling
// it back into guest memory writes the GFX heap the Dantelion allocator
// lives in: a 1920x1080 with 2D-thin pad, or any pitch taken from the next
// descriptor, overruns into neighbouring headers (free_ then dies on
// 0x9f828d6f918a8c6f). shadPS4 marks the image GPU-dirty
// (InvalidateMemoryFromGPU) and does not memcpy it into CPU memory.
void surface_queue_writeback(std::uint64_t base) {
    auto it = g_surfaces.find(base);
    if (it == g_surfaces.end() || it->second.failed || it->second.thick) return;
    it->second.gpu_written = true;
    surface_written(it->second);
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 16) {
        const Surface& sf = it->second;
        host_log("texture: skip guest writeback of 0x%llx %ux%u (%zu src bytes); GPU image is the storage",
                 static_cast<unsigned long long>(sf.base), sf.width, sf.height, sf.total_src);
    }
}

// A texture created ahead of its first bind, from the T# GX
// built for it. Resolved the way a bind resolves it (so the first bind finds
// the surface, its upload already queued to the prep pool).
std::atomic<std::uint64_t> g_ahead_surfaces{0}, g_ahead_already{0}, g_ahead_failed{0};

bool textures_create_ahead_locked(const std::uint32_t* w) {
    const std::uint64_t base = tsharp_base(w);
    if (g_surfaces.count(base)) {
        g_ahead_already.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    bool dims_ok = true, arrayed = false;
    std::uint32_t dim = 0;
    t_force_async = true;
    const VkImageView v = texture_view_uncached(w, false, dims_ok, dim, arrayed, false, false);
    t_force_async = false;
    auto it = g_surfaces.find(base);
    if (v == VK_NULL_HANDLE || it == g_surfaces.end() || it->second.failed) {
        g_ahead_failed.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    it->second.ahead = true;
    g_ahead_surfaces.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// A fill over a sampled texture's memory where no render target is (the
// pending branch of render_handle_fill_locked). The guest's fill wrote that
// memory on the GPU and the texture then read it; here the fill reached no
// image and the memory under the texture was never written, so the texture
// kept what it was first uploaded with - Cainhurst's 960x540 R16G16F
// reflection-offset map, cleared every frame, held +-65504 and NaN and the
// floors' reflections broke into blocks. So each texture the fill covers
// whole is cleared to the fill's colour, as a target would be, and is
// GPU-written from then on: the image is the storage, nothing re-uploads it
// from the memory the fill never reached. Formats a float clear colour
// suits only; block-compressed and integer ones are left (and counted).
std::atomic<std::uint64_t> g_fill_surfaces{0}, g_fill_surfaces_skipped{0};
bool float_clearable(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R16_SNORM: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16B16A16_SNORM:
        case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32B32A32_SFLOAT:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        case VK_FORMAT_R5G6B5_UNORM_PACK16: case VK_FORMAT_B5G6R5_UNORM_PACK16:
            return true;
        default:
            return false;
    }
}
int textures_fill_surfaces_locked(std::uint64_t va, std::size_t bytes, const float rgba[4]) {
    if (!bytes || !surface_pages(va, va + bytes)) return 0;
    int cleared = 0;
    const std::uint64_t end = va + bytes;
    for (auto kv = g_surfaces.lower_bound(va); kv != g_surfaces.end() && kv->first < end; ++kv) {
        Surface& sf = kv->second;
        if (sf.failed || !sf.image || !sf.total_src || sf.base + sf.total_src > end) continue;
        if (sf.block != 1 || !float_clearable(sf.format)) {
            g_fill_surfaces_skipped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        begin_recording_locked();
        render_end_pass_locked();
        VkClearColorValue v{};
        const bool bgra = sf.format == VK_FORMAT_B8G8R8A8_UNORM;
        v.float32[0] = bgra ? rgba[2] : rgba[0];
        v.float32[1] = rgba[1];
        v.float32[2] = bgra ? rgba[0] : rgba[2];
        v.float32[3] = rgba[3];
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = sf.image;
        b.subresourceRange = range;
        b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        rec().clear_color_image(sf.image, VK_IMAGE_LAYOUT_GENERAL, v, 1, &range);
        surface_written(sf);
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        rec().pipeline_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        if (!sf.gpu_written) {
            sf.gpu_written = true;
            sf.watched = false;
            tex_event(sf.base, sf.total_src, "fill over texture 0x%llx %ux%u: cleared, the image is the storage from here",
                      static_cast<unsigned long long>(sf.base), sf.width, sf.height);
        }
        ++cleared;
        g_fill_surfaces.fetch_add(1, std::memory_order_relaxed);
    }
    return cleared;
}

// A subresource region from an upload token, copied into the
// image. The guest's path tiled the rows into its memory with a compute
// dispatch and the cache re-untiled them at the next bind; here the image
// takes the rows directly and is marked GPU-written, so nothing re-uploads
// it from the memory the rows never reached.
std::atomic<std::uint64_t> g_region_uploads{0}, g_region_bytes{0}, g_region_created{0}, g_region_no_surface{0}, g_region_out_of_range{0};

bool textures_upload_region_locked(std::uint64_t base, const std::uint32_t* tsharp, std::uint32_t mip, std::uint32_t layer, std::uint32_t x,
                                   std::uint32_t y, std::uint32_t w, std::uint32_t h, const void* data, std::size_t bytes, std::uint32_t row_bytes) {
    auto it = g_surfaces.find(base);
    if (it == g_surfaces.end() && tsharp) {
        if (textures_create_ahead_locked(tsharp)) g_region_created.fetch_add(1, std::memory_order_relaxed);
        it = g_surfaces.find(base);
    }
    if (it == g_surfaces.end() || it->second.failed || !it->second.image) {
        g_region_no_surface.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    Surface& sf = it->second;
    const std::uint32_t layers = sf.type == 10 ? 1 : sf.layers;
    const std::uint32_t mw = std::max(1u, sf.width >> mip), mh = std::max(1u, sf.height >> mip);
    if (mip >= sf.levels.size() || layer >= layers || !w || !h || x + w > mw || y + h > mh || (x % sf.block) || (y % sf.block) ||
        (w % sf.block && x + w != mw) || (h % sf.block && y + h != mh)) {
        g_region_out_of_range.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const std::uint32_t wb = (w + sf.block - 1) / sf.block, hb = (h + sf.block - 1) / sf.block;
    if (row_bytes != wb * sf.esize || bytes < static_cast<std::size_t>(row_bytes) * hb) {
        g_region_out_of_range.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    DevBuffer staging;
    VkDeviceSize staging_offset = 0;
    if (!acquire_staging_locked(staging, bytes, staging_offset)) return false;
    std::memcpy(staging.map, data, bytes);
    begin_recording_locked();
    render_end_pass_locked();
    const VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = sf.image;
    b.subresourceRange = {aspect, mip, 1, layer, 1};
    b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy r{};
    r.bufferOffset = staging_offset;
    r.imageSubresource = {aspect, mip, layer, 1};
    r.imageOffset = {static_cast<std::int32_t>(x), static_cast<std::int32_t>(y), 0};
    r.imageExtent = {w, h, 1};
    rec().copy_buffer_to_image(staging.buffer, sf.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    surface_written(sf);
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    if (!sf.gpu_written) {
        sf.gpu_written = true;
        sf.watched = false;
        tex_event(sf.base, sf.total_src, "upload token 0x%llx %ux%u: the image is the storage from here", static_cast<unsigned long long>(sf.base), sf.width,
                  sf.height);
    }
    g_region_uploads.fetch_add(1, std::memory_order_relaxed);
    g_region_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

// A texture-to-texture copy from a copy-image token. Both sides
// resolve as a bind would (texture_view_uncached: the surface made or
// refreshed, or a render target aliased), then vkCmdCopyImage between them;
// the destination surface is GPU-written from here.
std::atomic<std::uint64_t> g_image_copies{0}, g_image_copy_texels{0}, g_image_copy_no_src{0}, g_image_copy_no_dst{0},
    g_image_copy_format{0}, g_image_copy_range{0}, g_image_copy_rt_src{0}, g_image_copy_rt_dst{0};

struct CopySide {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0, height = 0, levels = 1, layers = 1, block = 1;
    std::size_t esize = 0;
    Surface* surface = nullptr;
    RtImage* rt = nullptr;
};

bool copy_side_locked(const std::uint32_t* w, CopySide& out) {
    const std::uint64_t base = tsharp_base(w);
    if (RtImage* rt = find_render_target(base); rt && rt->initialised) {
        out.image = rt->image;
        out.format = rt->format;
        out.width = rt->width;
        out.height = rt->height;
        out.layers = rt->layers;
        out.esize = format_bytes_per_pixel(rt->format);
        out.rt = rt;
        return true;
    }
    bool dims_ok = true, arrayed = false;
    std::uint32_t dim = 0;
    const VkImageView v = texture_view_uncached(w, false, dims_ok, dim, arrayed, false, false);
    auto it = g_surfaces.find(base);
    if (v == VK_NULL_HANDLE || it == g_surfaces.end() || it->second.failed || !it->second.image) return false;
    Surface& sf = it->second;
    out.image = sf.image;
    out.format = sf.format;
    out.width = sf.width;
    out.height = sf.height;
    out.levels = static_cast<std::uint32_t>(sf.levels.size());
    out.layers = sf.type == 10 ? 1 : sf.layers;
    out.block = sf.block;
    out.esize = sf.esize;
    out.surface = &sf;
    return true;
}

// A colour target created where a shader stored a texture of the same size
// and texel size takes that texture's texels: on the console they are the
// same memory. The game clears its post-processing targets with a compute
// fill (9a9cf8a9) at the start of a scene, before any draw has made them
// render targets here, so the fills went to cached textures, and the targets
// created a moment later by the first draws started as whatever the image
// heap held: 72 of them across the two scene changes of the new-character
// cutscenes. BBHOST_RT_FROM_SURFACE=0: as before.
std::atomic<std::uint64_t> g_rt_from_surface{0};

bool textures_carry_into_target_locked(std::uint64_t base, std::uint32_t width, std::uint32_t height, std::size_t bytes_per_pixel,
                                       VkImage target) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_RT_FROM_SURFACE");
        return !(e && e[0] == '0');
    }();
    if (!on) return false;
    auto it = g_surfaces.find(base);
    if (it == g_surfaces.end()) return false;
    const Surface& sf = it->second;
    if (!sf.gpu_written || sf.failed || !sf.image || sf.block != 1 || sf.esize != bytes_per_pixel || sf.width != width ||
        sf.height != height || sf.type == 10 || sf.thick) {
        return false;
    }
    begin_recording_locked();
    render_end_pass_locked();
    auto barrier = [](VkImage img, VkImageLayout from, VkAccessFlags src_access, VkAccessFlags dst_access) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.srcAccessMask = src_access;
        b.dstAccessMask = dst_access;
        vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    };
    barrier(sf.image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    barrier(target, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy r{};
    r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.dstSubresource = r.srcSubresource;
    r.extent = {width, height, 1};
    vkCmdCopyImage(g_cmd(), sf.image, VK_IMAGE_LAYOUT_GENERAL, target, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    // Transfers too: the game's clear of the new target (a fill) can come next.
    barrier(target, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    tex_event(base, sf.total_src, "render target 0x%llx %ux%u takes the texels a shader stored there", static_cast<unsigned long long>(base),
              width, height);
    if (g_rt_from_surface.fetch_add(1, std::memory_order_relaxed) < 12) {
        host_log("texture: render target 0x%llx %ux%u created over a shader-written texture of its size; its texels carried",
                 static_cast<unsigned long long>(base), width, height);
    }
    return true;
}

bool textures_copy_image_region_locked(const GpuImageCopy& c) {
    CopySide src, dst;
    // A refused token is a copy that did not happen (the game's method did
    // not run either): the destination's history says so, for F12.
    const auto refused = [&](const char* why) {
        if (Surface* sf = dst.surface) {
            tex_event(sf->base, sf->total_src, "copy token into 0x%llx mip %u layer %u (%ux%u at %u,%u) from 0x%llx mip %u layer %u REFUSED: %s",
                      static_cast<unsigned long long>(sf->base), c.dst_mip, c.dst_layer, c.w, c.h, c.dst_x, c.dst_y,
                      static_cast<unsigned long long>(tsharp_base(c.src_tsharp)), c.src_mip, c.src_layer, why);
        }
    };
    if (!copy_side_locked(c.src_tsharp, src)) {
        g_image_copy_no_src.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (!copy_side_locked(c.dst_tsharp, dst)) {
        g_image_copy_no_dst.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // vkCmdCopyImage wants size-compatible formats: the same one, or the same
    // bytes per texel with no compression on either side.
    if (src.format != dst.format && (src.esize != dst.esize || src.block != 1 || dst.block != 1)) {
        g_image_copy_format.fetch_add(1, std::memory_order_relaxed);
        refused("formats");
        return false;
    }
    const std::uint32_t sw = std::max(1u, src.width >> c.src_mip), sh = std::max(1u, src.height >> c.src_mip);
    const std::uint32_t dw = std::max(1u, dst.width >> c.dst_mip), dh = std::max(1u, dst.height >> c.dst_mip);
    if (!c.w || !c.h || c.src_mip >= src.levels || c.dst_mip >= dst.levels || c.src_layer >= src.layers || c.dst_layer >= dst.layers ||
        c.src_x + c.w > sw || c.src_y + c.h > sh || c.dst_x + c.w > dw || c.dst_y + c.h > dh) {
        if (g_image_copy_range.fetch_add(1, std::memory_order_relaxed) < 8) {
            host_log("texture: copy token refused: src 0x%llx %ux%u levels %u layers %u%s mip %u layer %u at %u,%u; dst 0x%llx %ux%u levels %u layers %u%s "
                     "mip %u layer %u at %u,%u; %ux%u",
                     static_cast<unsigned long long>(tsharp_base(c.src_tsharp)), src.width, src.height, src.levels, src.layers, src.rt ? " (render target)" : "",
                     c.src_mip, c.src_layer, c.src_x, c.src_y, static_cast<unsigned long long>(tsharp_base(c.dst_tsharp)), dst.width, dst.height, dst.levels,
                     dst.layers, dst.rt ? " (render target)" : "", c.dst_mip, c.dst_layer, c.dst_x, c.dst_y, c.w, c.h);
        }
        refused(c.src_mip >= src.levels || c.src_layer >= src.layers ? "source mip or layer out of range"
                                                                     : "destination mip, layer or region out of range");
        return false;
    }
    if (src.rt) g_image_copy_rt_src.fetch_add(1, std::memory_order_relaxed);
    if (dst.rt) {
        g_image_copy_rt_dst.fetch_add(1, std::memory_order_relaxed);
        dst.rt->fill_last = false;  // written: a fill is no longer its last write
    }
    begin_recording_locked();
    render_end_pass_locked();
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    const VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageCopy r{};
    r.srcSubresource = {aspect, c.src_mip, c.src_layer, 1};
    r.srcOffset = {static_cast<std::int32_t>(c.src_x), static_cast<std::int32_t>(c.src_y), 0};
    r.dstSubresource = {aspect, c.dst_mip, c.dst_layer, 1};
    r.dstOffset = {static_cast<std::int32_t>(c.dst_x), static_cast<std::int32_t>(c.dst_y), 0};
    r.extent = {c.w, c.h, 1};
    vkCmdCopyImage(g_cmd(), src.image, VK_IMAGE_LAYOUT_GENERAL, dst.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    if (dst.surface) surface_written(*dst.surface);
    if (dst.surface && !dst.surface->gpu_written) {
        dst.surface->gpu_written = true;
        dst.surface->watched = false;
        tex_event(dst.surface->base, dst.surface->total_src, "copy token into 0x%llx %ux%u: the image is the storage from here",
                  static_cast<unsigned long long>(dst.surface->base), dst.surface->width, dst.surface->height);
    }
    if (dst.surface && c.dst_layer == 0 && c.dst_mip < 32 && !(dst.surface->copied_mips & (1u << c.dst_mip))) {
        dst.surface->copied_mips |= 1u << c.dst_mip;
        tex_event(dst.surface->base, dst.surface->total_src, "copy token into 0x%llx mip %u (%ux%u at %u,%u) from %s 0x%llx mip %u layer %u",
                  static_cast<unsigned long long>(dst.surface->base), c.dst_mip, c.w, c.h, c.dst_x, c.dst_y,
                  src.rt ? "render target" : "texture", static_cast<unsigned long long>(tsharp_base(c.src_tsharp)), c.src_mip, c.src_layer);
    }
    g_image_copies.fetch_add(1, std::memory_order_relaxed);
    g_image_copy_texels.fetch_add(static_cast<std::uint64_t>(c.w) * c.h, std::memory_order_relaxed);
    return true;
}

// Surfaces of a resource whose memory is gone: retired once the command
// processor has retired everything submitted when the resource died, unless
// a live resource has taken the memory since (its surface is then the new
// one's, refreshed by the usual path).
struct RetireEntry {
    std::uint64_t base;
    std::size_t bytes;
    std::uint64_t after;
};
std::mutex g_retire_mu;
std::vector<RetireEntry> g_retire_pending;  // under g_retire_mu
std::atomic<std::uint64_t> g_retired_surfaces{0}, g_retired_bytes{0}, g_retire_kept_live{0};

void textures_retire(std::uint64_t base, std::size_t bytes, std::uint64_t after_submitted) {
    std::lock_guard<std::mutex> lk(g_retire_mu);
    g_retire_pending.push_back({base, bytes, after_submitted});
}

void retire_surfaces_locked() {
    std::vector<RetireEntry> due;
    {
        std::lock_guard<std::mutex> lk(g_retire_mu);
        if (g_retire_pending.empty()) return;
        const std::uint64_t retired = hle_gnm_retired_total();
        auto keep = std::remove_if(g_retire_pending.begin(), g_retire_pending.end(), [&](const RetireEntry& e) {
            if (e.after > retired) return false;
            due.push_back(e);
            return true;
        });
        g_retire_pending.erase(keep, g_retire_pending.end());
    }
    bool any = false;
    for (const RetireEntry& e : due) {
        for (auto it = g_surfaces.lower_bound(e.base); it != g_surfaces.end() && it->first < e.base + e.bytes;) {
            Surface& sf = it->second;
            if (gx_resource_at(sf.base, std::max<std::size_t>(sf.total_src, 1), nullptr)) {
                g_retire_kept_live.fetch_add(1, std::memory_order_relaxed);
                ++it;
                continue;
            }
            g_retired_surfaces.fetch_add(1, std::memory_order_relaxed);
            g_retired_bytes.fetch_add(sf.total_dst, std::memory_order_relaxed);
            for (auto& v : sf.views) defer_destroy_view(v.second);
            if (sf.image) defer_destroy_image(sf.image, sf.memory);
            if (g_last_view_surface == &sf) g_last_view_surface = nullptr;
            note_surface_erased(it->second);
            it = g_surfaces.erase(it);
            any = true;
        }
    }
    if (any) bump_view_epoch();
}

void textures_before_submit() {
    for (const std::shared_ptr<TexturePrep>& p : g_batch_preps) PrepPool::get().wait(*p);
    g_batch_preps.clear();
    retire_surfaces_locked();
    image_reuse_sweep_locked();
}

ReuseRetired textures_retire_reused_locked(std::uint64_t now, std::uint64_t idle_ms, std::chrono::steady_clock::time_point deadline) {
    ReuseRetired out;
    static std::uint64_t cursor = 0;  // the next base the scan looks at
    int k = 0;
    auto it = g_surfaces.lower_bound(cursor);
    for (; it != g_surfaces.end();) {
        if ((++k & 15) == 0 && std::chrono::steady_clock::now() >= deadline) break;
        Surface& sf = it->second;
        const std::size_t span = std::max<std::size_t>(sf.total_src, 1);
        if (!sf.image || !sf.gpu_written || sf.written_ms + idle_ms > now || gx_resource_at(sf.base, span, nullptr) ||
            (!gx_resource_newer_overlapping(sf.base, span, sf.written_flip) &&
             !render_target_newer_overlapping_locked(sf.base, span, sf.written_flip, nullptr))) {
            ++it;
            continue;
        }
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 8) {
            host_log("texture: shader-written 0x%llx %ux%u (%llu KiB) retired: a newer object holds its memory, last written %llu s ago",
                     static_cast<unsigned long long>(sf.base), sf.width, sf.height, static_cast<unsigned long long>(sf.memory.size >> 10),
                     static_cast<unsigned long long>((now - sf.written_ms) / 1000));
        }
        ++out.n;
        out.bytes += sf.memory.size;
        // The copies made of its levels go with it: keyed by its base, they
        // would hold their memory until a surface at that base was read again.
        for (auto la = g_level_aliases.lower_bound(std::make_tuple(sf.base, 0u, 0u, 0u, std::numeric_limits<int>::min()));
             la != g_level_aliases.end() && std::get<0>(la->first) == sf.base;) {
            invalidate_rt_views(la->second.img.base);
            defer_destroy_view(la->second.img.view);
            defer_destroy_image(la->second.img.image, la->second.img.memory);
            la = g_level_aliases.erase(la);
        }
        for (auto& v : sf.views) defer_destroy_view(v.second);
        defer_destroy_image(sf.image, sf.memory);
        if (g_last_view_surface == &sf) g_last_view_surface = nullptr;
        note_surface_erased(sf);
        it = g_surfaces.erase(it);
    }
    cursor = it == g_surfaces.end() ? 0 : it->first;  // at the end the next scan starts over
    if (out.n) bump_view_epoch();
    return out;
}

// The wait above, made first without the renderer's mutex, for the jobs
// queued so far (a copy of the list, taken under it): a submission then finds
// them done, and waits under the mutex only for what was queued in between.
// Waited for under it, the untiling of a burst's new textures - a waiter runs
// a job no worker has taken yet - held up every other thread's draws, copies
// and presents: in the frame after the first fight, the command processor
// waited ~80 ms in five seconds for the mutex, ~40 ms of it held in here.
void textures_prewait_for_submit() {
    std::vector<std::shared_ptr<TexturePrep>> preps;
    {
        std::lock_guard<GpuMutex> lock(g.mu);
        if (g_batch_preps.empty()) return;
        preps = g_batch_preps;
    }
    for (const std::shared_ptr<TexturePrep>& p : preps) PrepPool::get().wait(*p);
}

void textures_submitted(int slot) {
    for (WriteBack& wb : g_writebacks) g_slot_writebacks[slot].push_back(wb);
    g_writebacks.clear();
}

void textures_retired(int slot) {
    for (WriteBack& wb : g_slot_writebacks[slot]) {
        auto it = g_surfaces.find(wb.base);
        if (it != g_surfaces.end() && hle_kernel_va_mapped(it->second.base, it->second.total_src)) {
            Surface& sf = it->second;
            auto* dst = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(sf.base));
            bool ok = true;
            for (std::uint32_t sl = 0; sl < sf.slices && ok; ++sl) {
                ok = tile_slice(dst + sl * sf.slice_src_bytes, sf.slice_src_bytes, static_cast<const std::uint8_t*>(wb.staging.map) + sl * sf.slice_dst_bytes,
                                sf.tiling, sf.width_e, sf.height_e, sf.pitch_e, sf.esize);
            }
            if (ok) {
                settle_prep(sf);
                sf.content_hash = content_hash_of(sf.base, sf.total_src);
                sf.exact_hash = content_hash_of(sf.base, sf.total_src, true);
            }
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 6) {
                const auto* f = static_cast<const float*>(wb.staging.map);
                const std::size_t n = sf.slice_dst_bytes * sf.slices / 4;
                host_log("texture: wrote back storage image 0x%llx %ux%ux%u to guest memory%s; floats[0,1,2,mid,last]=%g %g %g %g %g",
                         static_cast<unsigned long long>(sf.base), sf.width, sf.height, sf.slices, ok ? "" : " (tiling unsupported)",
                         n > 0 ? f[0] : 0.f, n > 1 ? f[1] : 0.f, n > 2 ? f[2] : 0.f, n > 2 ? f[n / 2] : 0.f, n > 0 ? f[n - 1] : 0.f);
            }
        }
        vkDestroyBuffer(g.device, wb.staging.buffer, nullptr);
        vkFreeMemory(g.device, wb.staging.memory, nullptr);
    }
    g_slot_writebacks[slot].clear();
}

namespace {
struct TexEvent {
    std::uint64_t flip, lo, hi;
    const char* fmt;
    std::uint64_t args[TexEventArgs::kArgs];
    char strings[TexEventArgs::kStrings];
};
constexpr std::size_t kTexEvents = 32768;  // ~8 MiB; tens of seconds of a world frame's events
std::vector<TexEvent> g_tex_events(kTexEvents);
std::uint64_t g_tex_event_next = 0;
std::mutex g_tex_event_mu;

// An event's text: its format with each conversion given the argument kept
// for it, read as the conversion says (printf's own rules for the length).
std::string tex_event_text(const TexEvent& e) {
    std::string out;
    unsigned next = 0;
    char spec[40], piece[256];
    for (const char* f = e.fmt; f && *f;) {
        if (*f != '%') {
            out += *f++;
            continue;
        }
        if (f[1] == '%') {
            out += '%';
            f += 2;
            continue;
        }
        const char* start = f++;
        while (*f && std::strchr("-+ #0", *f)) ++f;
        while (*f && ((*f >= '0' && *f <= '9') || *f == '.')) ++f;
        const char* length = f;
        while (*f && std::strchr("hlLjzt", *f)) ++f;
        const std::size_t length_n = static_cast<std::size_t>(f - length);
        const char conv = *f;
        if (!conv) break;
        ++f;
        const std::size_t head = std::min<std::size_t>(static_cast<std::size_t>(length - start), sizeof(spec) - 4);
        std::memcpy(spec, start, head);
        const std::uint64_t v = next < TexEventArgs::kArgs ? e.args[next++] : 0;
        const bool wide = length_n && (length[0] == 'l' || length[0] == 'j' || length[0] == 'z' || length[0] == 't');
        const bool half = length_n == 1 && length[0] == 'h', byte = length_n == 2 && length[0] == 'h';
        int n = 0;
        switch (conv) {
            case 'd':
            case 'i': {
                const long long x = wide ? static_cast<long long>(v) : half ? static_cast<short>(v) : byte ? static_cast<signed char>(v)
                                                                            : static_cast<int>(v);
                std::memcpy(spec + head, "lld", 4);
                n = std::snprintf(piece, sizeof(piece), spec, x);
                break;
            }
            case 'u':
            case 'o':
            case 'x':
            case 'X': {
                const unsigned long long x = wide ? v : half ? (v & 0xffff) : byte ? (v & 0xff) : (v & 0xffffffffu);
                spec[head] = 'l';
                spec[head + 1] = 'l';
                spec[head + 2] = conv;
                spec[head + 3] = 0;
                n = std::snprintf(piece, sizeof(piece), spec, x);
                break;
            }
            case 'c':
                spec[head] = 'c';
                spec[head + 1] = 0;
                n = std::snprintf(piece, sizeof(piece), spec, static_cast<int>(v));
                break;
            case 'e':
            case 'E':
            case 'f':
            case 'F':
            case 'g':
            case 'G':
            case 'a':
            case 'A': {
                double d;
                std::memcpy(&d, &v, sizeof(d));
                spec[head] = conv;
                spec[head + 1] = 0;
                n = std::snprintf(piece, sizeof(piece), spec, d);
                break;
            }
            case 's':
                spec[head] = 's';
                spec[head + 1] = 0;
                n = std::snprintf(piece, sizeof(piece), spec, e.strings + (v < TexEventArgs::kStrings ? v : TexEventArgs::kStrings - 1));
                break;
            case 'p':
                spec[head] = 'p';
                spec[head + 1] = 0;
                n = std::snprintf(piece, sizeof(piece), spec, reinterpret_cast<void*>(static_cast<std::uintptr_t>(v)));
                break;
            default:
                out.append(start, f);
                continue;
        }
        if (n > 0) out.append(piece, std::min<std::size_t>(static_cast<std::size_t>(n), sizeof(piece) - 1));
    }
    return out;
}
}  // namespace

void tex_event_keep(std::uint64_t lo, std::uint64_t bytes, const char* fmt, const TexEventArgs& args) {
    std::lock_guard<std::mutex> lock(g_tex_event_mu);
    TexEvent& e = g_tex_events[g_tex_event_next++ % kTexEvents];
    e.flip = hle_video_flip_count();
    e.lo = lo;
    e.hi = lo + std::max<std::uint64_t>(bytes, 1);
    e.fmt = fmt;
    std::memcpy(e.args, args.args, args.n * sizeof(args.args[0]));
    std::memcpy(e.strings, args.strings, TexEventArgs::kStrings);
}

void tex_events_for(std::FILE* f, std::uint64_t lo, std::uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_tex_event_mu);
    const std::uint64_t hi = lo + std::max<std::uint64_t>(bytes, 1);
    const std::uint64_t first = g_tex_event_next > kTexEvents ? g_tex_event_next - kTexEvents : 0;
    std::vector<std::uint64_t> hits;
    for (std::uint64_t i = first; i < g_tex_event_next; ++i) {
        const TexEvent& e = g_tex_events[i % kTexEvents];
        if (e.lo < hi && lo < e.hi) hits.push_back(i);
    }
    // The first few (what made it) and the most recent (what keeps it).
    constexpr std::size_t kHead = 12, kTail = 28;
    for (std::size_t k = 0; k < hits.size(); ++k) {
        if (hits.size() > kHead + kTail && k == kHead) {
            std::fprintf(f, "      ... %zu more\n", hits.size() - kHead - kTail);
            k = hits.size() - kTail;
        }
        const TexEvent& e = g_tex_events[hits[k] % kTexEvents];
        std::fprintf(f, "      flip %llu: %s\n", static_cast<unsigned long long>(e.flip), tex_event_text(e).c_str());
    }
}

void tex_events_write(std::FILE* f) {
    std::lock_guard<std::mutex> lock(g_tex_event_mu);
    const std::uint64_t first = g_tex_event_next > kTexEvents ? g_tex_event_next - kTexEvents : 0;
    std::fprintf(f, "\n%llu texture events, the last %llu:\n", static_cast<unsigned long long>(g_tex_event_next),
                 static_cast<unsigned long long>(g_tex_event_next - first));
    for (std::uint64_t i = first; i < g_tex_event_next; ++i) {
        const TexEvent& e = g_tex_events[i % kTexEvents];
        std::fprintf(f, "flip %llu: %s\n", static_cast<unsigned long long>(e.flip), tex_event_text(e).c_str());
    }
}

std::size_t texture_src_bytes_locked(std::uint64_t base) {
    auto it = g_surfaces.find(base);
    return it == g_surfaces.end() ? 0 : it->second.total_src;
}

std::string texture_describe_locked(std::uint64_t base) {
    auto it = g_surfaces.find(base);
    if (it == g_surfaces.end()) return {};
    const Surface& sf = it->second;
    char line[400];
    std::snprintf(line, sizeof(line),
                  "surface vkformat %d %ux%u x%u layers, type %u, tiling %u, %zu level(s), block %u, %zu source bytes; "
                  "%u upload(s), last checked %llu ms ago, unchanged %u, hot %u, gpu_written %d, watched %d%s",
                  static_cast<int>(sf.format), sf.width, sf.height, sf.layers, sf.type, sf.tiling, sf.levels.size(),
                  sf.block, sf.total_src, sf.uploads, static_cast<unsigned long long>(now_ms() - sf.checked_at_ms),
                  sf.unchanged, sf.hot, sf.gpu_written ? 1 : 0, sf.watched ? 1 : 0, sf.failed ? ", FAILED" : "");
    return line;
}

// Blits level 0 / layer 0 of the surface into an RGBA8 image and writes it
// as a PNG or PPM, by the path's extension (any sampled format the driver can
// blit from, BC included).
bool texture_dump_locked(std::uint64_t base, const char* path, std::uint32_t level, std::uint32_t layer) {
    auto it = g_surfaces.find(base);
    if (it == g_surfaces.end() || it->second.failed) return false;
    Surface& sf = it->second;
    if (level >= sf.levels.size() || layer >= std::max(1u, sf.info.arrayLayers)) return false;
    const std::uint32_t lw = std::max(1u, sf.width >> level), lh = std::max(1u, sf.height >> level);
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {lw, lh, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImage tmp = VK_NULL_HANDLE;
    VkDeviceMemory tmp_mem = VK_NULL_HANDLE;
    if (vkCreateImage(g.device, &ici, nullptr, &tmp) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g.device, tmp, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(g.device, &mai, nullptr, &tmp_mem) != VK_SUCCESS) {
        vkDestroyImage(g.device, tmp, nullptr);
        return false;
    }
    vkBindImageMemory(g.device, tmp, tmp_mem, 0);
    DevBuffer staging;
    if (!create_dev_buffer(staging, static_cast<std::uint64_t>(lw) * lh * 4, true)) {
        vkDestroyImage(g.device, tmp, nullptr);
        vkFreeMemory(g.device, tmp_mem, nullptr);
        return false;
    }
    begin_recording_locked();
    render_end_pass_locked();
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.image = tmp;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
    blit.srcOffsets[1] = {static_cast<std::int32_t>(lw), static_cast<std::int32_t>(lh), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {static_cast<std::int32_t>(lw), static_cast<std::int32_t>(lh), 1};
    vkCmdBlitImage(g_cmd(), sf.image, VK_IMAGE_LAYOUT_GENERAL, tmp, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_NEAREST);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {lw, lh, 1};
    vkCmdCopyImageToBuffer(g_cmd(), tmp, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &region);
    flush_locked();
    ImageFile f;
    bool ok = f.open(path, lw, lh);
    if (ok) {
        const auto* px = static_cast<const std::uint8_t*>(staging.map);
        std::vector<std::uint8_t> row(lw * 3);
        for (std::uint32_t y = 0; y < lh; ++y) {
            for (std::uint32_t x = 0; x < lw; ++x) {
                const std::uint8_t* p = px + (static_cast<std::size_t>(y) * lw + x) * 4;
                row[x * 3] = p[0];
                row[x * 3 + 1] = p[1];
                row[x * 3 + 2] = p[2];
            }
            f.row(row.data());
        }
        ok = f.close();
    }
    if (ok) {
        host_log("texture: wrote %s (%ux%u, format %d, tiling %u, level %u layer %u)", path, lw, lh, sf.format, sf.tiling, level, layer);
    } else {
        host_log("texture: cannot write %s", path);
    }
    vkDestroyBuffer(g.device, staging.buffer, nullptr);
    vkFreeMemory(g.device, staging.memory, nullptr);
    vkDestroyImage(g.device, tmp, nullptr);
    vkFreeMemory(g.device, tmp_mem, nullptr);
    return ok;
}

// GPU work wrote over this range: any cached surface it covers has to be read
// back from guest memory before its next use. Without this the only thing that
// notices a change is the periodic content hash, and a texture the game rewrites
// every frame - a movie frame, most visibly - was picked up five times a second.
// Drop the cached sampled views of a render target whose image is about to be
// destroyed. Without this a later draw binds a view of a freed image, which is
// a GPU fault - the caller has already waited for the queue to go idle.
void invalidate_rt_views(std::uint64_t base) {
    auto it = g_rt_views.find(base);
    if (it == g_rt_views.end()) return;
    for (auto& [key, view] : it->second) defer_destroy_view(view);
    g_rt_views.erase(it);
    bump_view_epoch();
}

bool surfaces_may_cover_locked(std::uint64_t va, std::size_t bytes) {
    if (!bytes || !surface_pages(va, va + bytes)) return false;
    // The pages hold a surface: whether one overlaps these bytes themselves,
    // found as mark_surfaces_dirty_locked finds them.
    const std::uint64_t end = va + bytes;
    const std::uint64_t first = va > kLargeSurfaceSrc ? va - kLargeSurfaceSrc : 0;
    for (auto kv = g_surfaces.lower_bound(first); kv != g_surfaces.end() && kv->first < end; ++kv) {
        const Surface& sf = kv->second;
        if (sf.total_src <= kLargeSurfaceSrc && sf.total_src && sf.base + sf.total_src > va) return true;
    }
    const std::uint64_t large_first = va > g_max_large_src ? va - g_max_large_src : 0;
    for (auto kv = g_large_surfaces.lower_bound(large_first); kv != g_large_surfaces.end() && kv->first < end; ++kv) {
        if (kv->second->base + kv->second->total_src > va) return true;
    }
    return false;
}

int mark_surfaces_dirty_locked(std::uint64_t va, std::size_t bytes) {
    if (!bytes) return 0;
    shadow_written_locked(va, bytes, true);  // GPU copies and fills land here too
    if (!surface_pages(va, va + bytes)) {
        ++g_dirty_walks_skipped;
        return 0;
    }
    ++g_dirty_walks;
    int touched = 0;
    const std::uint64_t end = va + bytes;
    // Every copy and fill lands here. A surface overlaps [va, end) if it
    // starts below `end` and no further below `va` than its own size: the
    // ordinary ones from g_surfaces within kLargeSurfaceSrc, the large ones
    // from their own map (g_large_surfaces, above).
    static std::vector<Surface*> hit;
    hit.clear();
    const std::uint64_t first = va > kLargeSurfaceSrc ? va - kLargeSurfaceSrc : 0;
    for (auto kv = g_surfaces.lower_bound(first); kv != g_surfaces.end() && kv->first < end; ++kv) {
        Surface& sf = kv->second;
        if (sf.total_src > kLargeSurfaceSrc) continue;  // taken below
        if (sf.total_src && sf.base + sf.total_src > va) hit.push_back(&sf);
    }
    const std::uint64_t large_first = va > g_max_large_src ? va - g_max_large_src : 0;
    for (auto kv = g_large_surfaces.lower_bound(large_first); kv != g_large_surfaces.end() && kv->first < end; ++kv) {
        if (kv->second->base + kv->second->total_src > va) hit.push_back(kv->second);
    }
    for (Surface* sp : hit) {
        Surface& sf = *sp;
        ++touched;
        // Only lift the throttle - the content hash still decides whether the
        // surface really changed. Forcing the upload instead made a single
        // 16 MiB copy re-upload every texture it covered, every frame.
        sf.checked_at_ms = 0;
        sf.refreshed_serial = 0;
        sf.unchanged = 0;
        sf.cp_dirty = true;
        if (sf.hot < 2) sf.hot = 2;
        if (sf.gpu_written && sf.width >= 16) {
            tex_event(sf.base, sf.total_src, "dirty 0x%llx (gpu-written %ux%u) by a write to [0x%llx,+0x%zx)", static_cast<unsigned long long>(sf.base),
                      sf.width, sf.height, static_cast<unsigned long long>(va), bytes);
        }
    }
    return touched;
}

bool describe_view_locked(VkImageView view, ViewRecord& out) {
    auto it = g_view_records.find(view);
    if (it == g_view_records.end()) return false;
    out = it->second;
    return true;
}

bool describe_sampler_locked(VkSampler sampler, VkSamplerCreateInfo& out) {
    auto it = g_sampler_records.find(sampler);
    if (it == g_sampler_records.end()) return false;
    out = it->second;
    return true;
}

void forget_view_locked(VkImageView view) { g_view_records.erase(view); }

// Called by --stream-selftest under g.mu. Local source snapshots exercise the
// production CPU prep without pretending to be kernel-admitted guest mappings.
bool texture_upload_plan_selftest() {
    const std::uint64_t validation_before = g_validation_messages.load();
    bool ok = true;
    unsigned checks = 0, cases = 0;
    auto check = [&](bool result, const char* what, std::uint32_t tiling = 0, std::uint32_t esize = 0) {
        ++checks;
        if (!result) {
            ok = false;
            host_log("texture upload selftest: FAIL %s (tiling %u, esize %u)", what, tiling, esize);
        }
    };
    auto prep_for = [](TexturePrep& p, const Surface& sf, const std::vector<std::uint8_t>& source,
                       std::uint8_t* dst, std::size_t first, bool hash_only = false) {
        p.src = source.data();
        p.base = reinterpret_cast<std::uintptr_t>(source.data());
        p.dst = dst;
        p.levels.assign(sf.levels.begin() + first, sf.levels.end());
        const std::size_t dst_base = sf.levels[first].dst_off;
        for (auto& lv : p.levels) lv.dst_off -= dst_base;
        p.slices = sf.slices;
        p.esize = sf.esize;
        p.width = sf.width;
        p.height = sf.height;
        p.tiling = sf.tiling;
        p.total_src = sf.total_src;
        p.total_dst = sf.total_dst - dst_base;
        p.hash_only = hash_only;
    };
    constexpr std::uint32_t modes[] = {8, 9, 13, 5, 0, 14, 10, 2, 3, 4};
    constexpr std::uint32_t sizes[] = {1, 2, 4, 8, 12, 16};
    for (const auto tiling : modes) {
        for (const auto esize : sizes) {
            Surface sf;
            sf.type = 13;
            sf.esize = esize;
            sf.tiling = tiling;
            sf.slices = sf.layers = 3;
            sf.pow2pad = true;
            const MacroMode mm = macro_mode_for(esize);
            const bool macro = tiling == 14 || tiling == 10 || tiling == 2 || tiling == 3 || tiling == 4;
            sf.width = macro ? 64 * mm.aspect + 1 : 35;
            sf.height = macro ? 8 * mm.bank_height * mm.banks / mm.aspect + 1 : 19;
            sf.pitch_e = sf.width + 7;
            compute_levels(sf, 5, 1);
            std::vector<std::uint8_t> source(sf.total_src);
            for (std::size_t i = 0; i < source.size(); ++i)
                source[i] = static_cast<std::uint8_t>((i * 37) ^ (i >> 7) ^ (i >> 15) ^ 0x6d);
            std::vector<std::uint8_t> reference(sf.total_dst, 0), full(sf.total_dst, 0xcd);
            bool ref_ok = true;
            for (const auto& lv : sf.levels) {
                for (std::uint32_t sl = 0; sl < sf.slices; ++sl) {
                    ref_ok = untile_slice_reference(reference.data() + lv.dst_off + sl * lv.dst_slice_bytes,
                                                     source.data() + lv.src_off + sl * lv.src_slice_bytes,
                                                     lv.src_slice_bytes, lv.tiling, lv.w_e, lv.h_e, lv.pitch_e, esize) && ref_ok;
                }
            }
            check(texture_upload_layouts_supported(sf.levels, esize), "valid whole layout admitted", tiling, esize);
            check(ref_ok, "reference layout supported", tiling, esize);
            TexturePrep whole;
            prep_for(whole, sf, source, full.data(), 0);
            run_prep(whole);
            check(full == reference, "production full detile equals per-element reference", tiling, esize);
            for (const std::size_t first : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
                const std::size_t omitted = sf.levels[first].dst_off;
                const std::size_t bytes = sf.total_dst - omitted;
                std::vector<std::uint8_t> compact(bytes + 32, 0xcd);
                TexturePrep suffix;
                prep_for(suffix, sf, source, compact.data() + 16, first);
                run_prep(suffix);
                check(std::all_of(compact.begin(), compact.begin() + 16, [](auto v) { return v == 0xcd; }) &&
                          std::all_of(compact.end() - 16, compact.end(), [](auto v) { return v == 0xcd; }),
                      "compact detile stays inside staging span", tiling, esize);
                check(std::equal(reference.begin() + omitted, reference.end(), compact.begin() + 16),
                      "added mip bytes equal full upload", tiling, esize);
                // Model the carry overwrite with bytes distinct from guest memory.
                std::vector<std::uint8_t> carried(omitted, 0xa7), before = full, after(sf.total_dst);
                std::copy(carried.begin(), carried.end(), before.begin());
                std::copy(carried.begin(), carried.end(), after.begin());
                std::copy(compact.begin() + 16, compact.end() - 16, after.begin() + omitted);
                check(before == after, "full upload plus carry equals compact upload plus carry", tiling, esize);
                check(suffix.content_hash == whole.content_hash && suffix.exact_hash == whole.exact_hash &&
                          suffix.total_src == sf.total_src && suffix.base == whole.base,
                      "suffix prep retains whole source identity and hashes", tiling, esize);
                if (esize == 4 || esize == 8 || esize == 16) {
                    std::vector<UntileGpuPass> passes;
                    // Address arithmetic only: src_copy never dereferences guest VA.
                    constexpr VkDeviceAddress src_addr = 0x100000, dst_addr = 0x800000;
                    check(gpu_untile_plan(sf, dst_addr, passes, src_addr, static_cast<std::uint32_t>(first)),
                          "GPU suffix plan admitted", tiling, esize);
                    bool exact = passes.size() == sf.levels.size() - first;
                    for (std::size_t i = 0; i < passes.size(); ++i) {
                        const auto& lv = sf.levels[first + i];
                        const auto& pass = passes[i];
                        exact = exact && pass.src == src_addr + lv.src_off && pass.dst == dst_addr + lv.dst_off - omitted &&
                                pass.src_slice_bytes == lv.src_slice_bytes && pass.dst_slice_bytes == lv.dst_slice_bytes &&
                                pass.width_e == lv.w_e && pass.height_e == lv.h_e && pass.pitch_e == lv.pitch_e &&
                                pass.esize == esize && pass.slices == sf.slices;
                    }
                    check(exact, "GPU suffix offsets retain absolute source and compact destination", tiling, esize);
                }
            }
            ++cases;
        }
    }
    // A failure in an older mip must still zero valid added mips. The gate
    // must inspect the whole plan, rather than only the suffix it would upload.
    for (const std::uint32_t bad : {8u, 13u, 7u, 19u}) {
        Surface sf;
        sf.width = 16;
        sf.height = 16;
        sf.pitch_e = 16;
        sf.type = 9;
        sf.esize = 4;
        sf.tiling = 8;
        compute_levels(sf, 3, 1);
        sf.levels[0].tiling = bad;
        sf.levels[0].src_slice_bytes = 4;  // linear/thin/thick fail after at most one element
        std::vector<std::uint8_t> source(sf.total_src, 0x5b), full(sf.total_dst, 0xcd);
        check(!texture_upload_layouts_supported(sf.levels, sf.esize), "invalid older mip declines suffix", bad, sf.esize);
        TexturePrep whole;
        prep_for(whole, sf, source, full.data(), 0);
        run_prep(whole);
        check(std::all_of(full.begin(), full.end(), [](auto v) { return v == 0; }),
              "invalid older mip zeros whole upload including additions", bad, sf.esize);
        std::vector<Surface::Level> added(sf.levels.begin() + 1, sf.levels.end());
        check(texture_upload_layouts_supported(added, sf.esize), "valid additions cannot mask invalid older mip", bad, sf.esize);
    }
    // Macro detile's out-of-range elements are defined black, even when staging
    // is reused; this layout is safe to admit despite a truncated source slice.
    {
        Surface sf;
        sf.width = sf.height = 16;
        sf.pitch_e = 128;
        sf.type = 9;
        sf.esize = 4;
        sf.tiling = 14;
        sf.levels = {{16, 16, 128, 14, 0, 4, 0, 1024}};
        sf.total_src = 4;
        sf.total_dst = 1024;
        std::vector<std::uint8_t> source(4, 0x5b), full(1024, 0xcd), reference(1024, 0);
        TexturePrep whole;
        prep_for(whole, sf, source, full.data(), 0);
        run_prep(whole);
        const bool ref_ok = untile_slice_reference(reference.data(), source.data(), 4, 14, 16, 16, 128, 4);
        check(texture_upload_layouts_supported(sf.levels, 4) && ref_ok && full == reference && full[0] == 0x5b && full[4] == 0,
              "truncated macro slice writes defined black into reused staging", 14, 4);
        sf.levels[0].w_e = 0;
        check(!texture_upload_layouts_supported(sf.levels, 4), "zero extent declines suffix");
        sf.levels.clear();
        check(!texture_upload_layouts_supported(sf.levels, 4), "empty plan declines suffix");
    }
    // A carried level's source bytes and the sampled hash's blind gaps remain
    // in the hash-only prep used by GPU detile and in the CPU suffix prep.
    {
        Surface sf;
        sf.width = sf.height = sf.pitch_e = 512;
        sf.type = 9;
        sf.esize = 4;
        sf.tiling = 8;
        compute_levels(sf, 3, 1);
        std::vector<std::uint8_t> source(sf.total_src, 0x35), dst(sf.total_dst - sf.levels[2].dst_off);
        TexturePrep initial, changed, cpu;
        prep_for(initial, sf, source, nullptr, 2, true);
        run_prep(initial);
        source[96] ^= 0x7f;  // older mip, outside the first 64-byte sampled window
        prep_for(changed, sf, source, nullptr, 2, true);
        run_prep(changed);
        prep_for(cpu, sf, source, dst.data(), 2);
        run_prep(cpu);
        check(initial.content_hash == changed.content_hash && initial.exact_hash != changed.exact_hash,
              "whole-source exact hash detects carried mip change outside sampled windows");
        source[0] ^= 0x3f;
        TexturePrep sampled;
        prep_for(sampled, sf, source, nullptr, 2, true);
        run_prep(sampled);
        check(sampled.content_hash != changed.content_hash && cpu.content_hash == changed.content_hash &&
                  cpu.exact_hash == changed.exact_hash,
              "CPU and GPU hash-only suffix prep both hash carried source levels");
    }
    // Real device buffers let the production GPU detile run without importing
    // a made-up guest VA. Exercise the production common-level image carry,
    // then upload added mips explicitly: upload_surface's guest mapping/watch
    // gate and partial-carry ordering are deliberately outside this fixture.
    const std::uint64_t mappings_before = hle_kernel_maps_generation();
    if (g.ok && g.device) {
        const bool gpu_ready = untile_gpu_available_locked();
        check(gpu_ready, "production GPU detile pipeline available");
        for (const auto tiling : {8u, 13u, 14u}) {
            for (const bool on_gpu : {false, true}) {
                if (on_gpu && !gpu_ready) continue;
                Surface dst;
                dst.width = 256;
                dst.height = 128;
                dst.pitch_e = 256;
                dst.type = 13;
                dst.tiling = tiling;
                dst.esize = 4;
                dst.slices = dst.layers = 2;
                dst.format = VK_FORMAT_R32_UINT;
                compute_levels(dst, 3, 1);
                const auto all_levels = dst.levels;
                const std::size_t omitted = dst.levels[1].dst_off;
                const std::size_t suffix_bytes = dst.total_dst - omitted;
                std::vector<std::uint8_t> source(dst.total_src), expected(dst.total_dst, 0);
                for (std::size_t i = 0; i < source.size(); ++i)
                    source[i] = static_cast<std::uint8_t>((i * 17) ^ (i >> 11) ^ 0x53);
                TexturePrep cpu;
                prep_for(cpu, dst, source, expected.data(), 0);
                run_prep(cpu);
                // Distinguish the shader-written common level from guest bytes.
                for (std::size_t i = 0; i < omitted; ++i) expected[i] ^= 0xb6;
                dst.base = reinterpret_cast<std::uintptr_t>(source.data());
                Surface src = dst;
                src.gpu_written = true;
                src.levels.resize(1);
                dst.levels.resize(1);  // common-only carry needs no admitted guest mapping
                VkDeviceMemory src_mem = VK_NULL_HANDLE, dst_mem = VK_NULL_HANDLE;
                DevBuffer tiled, linear, readback;
                auto make_image = [&](Surface& sf, std::uint32_t levels, VkDeviceMemory& memory) {
                    auto& ci = sf.info;
                    ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                    ci.imageType = VK_IMAGE_TYPE_2D;
                    ci.format = sf.format;
                    ci.extent = {sf.width, sf.height, 1};
                    ci.mipLevels = levels;
                    ci.arrayLayers = sf.layers;
                    ci.samples = VK_SAMPLE_COUNT_1_BIT;
                    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
                    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                    if (vkCreateImage(g.device, &ci, nullptr, &sf.image) != VK_SUCCESS) return false;
                    VkMemoryRequirements req{};
                    vkGetImageMemoryRequirements(g.device, sf.image, &req);
                    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                    ai.allocationSize = req.size;
                    ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                    return ai.memoryTypeIndex != UINT32_MAX && vkAllocateMemory(g.device, &ai, nullptr, &memory) == VK_SUCCESS &&
                           vkBindImageMemory(g.device, sf.image, memory, 0) == VK_SUCCESS;
                };
                const bool setup = make_image(src, 1, src_mem) && make_image(dst, 3, dst_mem) &&
                                   create_dev_buffer(tiled, dst.total_src, true, true) &&
                                   create_dev_buffer(linear, dst.total_dst + 32, true, true) &&
                                   create_dev_buffer(readback, dst.total_dst, true, true) && tiled.map && linear.map && readback.map;
                check(setup, "GPU carry/upload fixture allocated", tiling, 4);
                if (setup) {
                    std::memcpy(tiled.map, source.data(), source.size());
                    std::memset(linear.map, 0xcd, linear.size);
                    std::memcpy(static_cast<std::uint8_t*>(linear.map) + 16, expected.data(), expected.size());
                    std::memset(readback.map, 0xcd, readback.size);
                    begin_recording_locked();
                    render_end_pass_locked();
                    auto memory_barrier = [&](VkAccessFlags from, VkAccessFlags to, VkPipelineStageFlags a, VkPipelineStageFlags b) {
                        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        mb.srcAccessMask = from;
                        mb.dstAccessMask = to;
                        rec().pipeline_barrier(a, b, 0, 1, &mb, 0, nullptr, 0, nullptr);
                    };
                    auto image_barrier = [&](VkImage image, VkImageLayout from, VkImageLayout to, std::uint32_t first,
                                             std::uint32_t count, VkAccessFlags a, VkAccessFlags b) {
                        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                        ib.oldLayout = from;
                        ib.newLayout = to;
                        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        ib.image = image;
                        ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, first, count, 0, dst.layers};
                        ib.srcAccessMask = a;
                        ib.dstAccessMask = b;
                        rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                               0, 0, nullptr, 0, nullptr, 1, &ib);
                    };
                    memory_barrier(VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
                                   VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                    image_barrier(src.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, 1,
                                  0, VK_ACCESS_TRANSFER_WRITE_BIT);
                    std::vector<VkBufferImageCopy> common;
                    for (std::uint32_t sl = 0; sl < dst.layers; ++sl) {
                        VkBufferImageCopy r{};
                        r.bufferOffset = 16 + sl * all_levels[0].dst_slice_bytes;
                        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, sl, 1};
                        r.imageExtent = {dst.width, dst.height, 1};
                        common.push_back(r);
                    }
                    rec().copy_buffer_to_image(linear.buffer, src.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                              static_cast<std::uint32_t>(common.size()), common.data());
                    image_barrier(src.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, 0, 1,
                                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                    const bool carried = carry_surface_locked(dst, src);
                    check(carried, "production common mip carry recorded", tiling, 4);
                    dst.levels = all_levels;
                    // Finish the common copy before reusing the linear buffer.
                    memory_barrier(VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                    std::vector<UntileGpuPass> passes;
                    const bool planned = gpu_untile_plan(dst, linear.address + 16, passes, tiled.address, 1);
                    check(planned, "real device-address suffix plan admitted", tiling, 4);
                    if (on_gpu && planned) {
                        for (const auto& pass : passes) untile_gpu_record_locked(pass);
                        memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                    }
                    // CPU compact output occupies a different span, so it cannot
                    // race the earlier common-level upload's read of linear.
                    DevBuffer cpu_suffix;
                    bool suffix_ready = on_gpu;
                    if (!on_gpu) {
                        suffix_ready = create_dev_buffer(cpu_suffix, suffix_bytes, true, true) && cpu_suffix.map;
                        if (suffix_ready) {
                            TexturePrep suffix;
                            prep_for(suffix, dst, source, static_cast<std::uint8_t*>(cpu_suffix.map), 1);
                            run_prep(suffix);
                            memory_barrier(VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                           VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                        }
                    }
                    check(suffix_ready, "suffix staging ready", tiling, 4);
                    if (carried && suffix_ready && (!on_gpu || planned)) {
                        image_barrier(dst.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 2,
                                      0, VK_ACCESS_TRANSFER_WRITE_BIT);
                        std::vector<VkBufferImageCopy> added, reads;
                        for (std::uint32_t l = 0; l < all_levels.size(); ++l) {
                            for (std::uint32_t sl = 0; sl < dst.layers; ++sl) {
                                VkBufferImageCopy r{};
                                r.bufferOffset = all_levels[l].dst_off + sl * all_levels[l].dst_slice_bytes;
                                r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, sl, 1};
                                r.imageExtent = {std::max(1u, dst.width >> l), std::max(1u, dst.height >> l), 1};
                                reads.push_back(r);
                                if (l) {
                                    r.bufferOffset = (on_gpu ? 16 : 0) + r.bufferOffset - omitted;
                                    added.push_back(r);
                                }
                            }
                        }
                        rec().copy_buffer_to_image(on_gpu ? linear.buffer : cpu_suffix.buffer, dst.image,
                                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                  static_cast<std::uint32_t>(added.size()), added.data());
                        image_barrier(dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, 1, 2,
                                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                        image_barrier(dst.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, 0, 3,
                                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                        rec().copy_image_to_buffer(dst.image, VK_IMAGE_LAYOUT_GENERAL, readback.buffer,
                                                  static_cast<std::uint32_t>(reads.size()), reads.data());
                        memory_barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
                        flush_locked();
                        check(g.ok && std::memcmp(readback.map, expected.data(), expected.size()) == 0,
                              on_gpu ? "GPU suffix upload plus actual carry matches all mip/layer bytes"
                                     : "CPU suffix upload plus actual carry matches all mip/layer bytes", tiling, 4);
                    } else {
                        flush_locked();
                    }
                    if (cpu_suffix.map) vkUnmapMemory(g.device, cpu_suffix.memory);
                    if (cpu_suffix.buffer) vkDestroyBuffer(g.device, cpu_suffix.buffer, nullptr);
                    if (cpu_suffix.memory) vkFreeMemory(g.device, cpu_suffix.memory, nullptr);
                }
                for (auto* b : {&tiled, &linear, &readback}) {
                    if (b->map) vkUnmapMemory(g.device, b->memory);
                    if (b->buffer) vkDestroyBuffer(g.device, b->buffer, nullptr);
                    if (b->memory) vkFreeMemory(g.device, b->memory, nullptr);
                }
                if (src.image) vkDestroyImage(g.device, src.image, nullptr);
                if (dst.image) vkDestroyImage(g.device, dst.image, nullptr);
                if (src_mem) vkFreeMemory(g.device, src_mem, nullptr);
                if (dst_mem) vkFreeMemory(g.device, dst_mem, nullptr);
            }
        }
    } else {
        host_log("texture upload selftest: GPU fixtures skipped (no initialized device)");
    }
    check(hle_kernel_maps_generation() == mappings_before, "selftest preserves guest mapping registry");
    check(g_validation_messages.load() == validation_before, "GPU fixtures emit no Vulkan validation messages");
    host_log("texture upload selftest: %s, %u checks across %u mip-chain layouts (CPU detile/reference, compact carry model, fallback, whole-source hashes, GPU plan offsets; device-buffer carry/upload)",
             ok ? "PASS" : "FAIL", checks, cases);
    return ok;
}

std::uint64_t textures_upload_count() { return g_uploads.load(); }
void textures_time_us(std::uint64_t out[5]) {
    out[0] = g_tex_image_us.load();
    out[1] = g_tex_staging_us.load();
    out[2] = g_tex_untile_us.load();
    out[3] = g_tex_record_us.load();
    out[4] = g_tex_view_us.load();
}
std::uint64_t textures_hash_count() { return g_hashes.load(); }
std::uint64_t textures_hash_us() { return g_hash_us.load(); }
std::uint64_t textures_replacements() { return g_surface_replacements.load(); }

void textures_report() {
    if (const std::uint64_t n = g_view_memo_hits + g_view_memo_misses) {
        host_log("texture: view memo %llu of %llu descriptor lookups (%.1f%%), %llu resolved, %llu evicted%s",
                 static_cast<unsigned long long>(g_view_memo_hits), static_cast<unsigned long long>(n),
                 100.0 * g_view_memo_hits / n, static_cast<unsigned long long>(g_view_memo_misses),
                 static_cast<unsigned long long>(g_view_memo_evictions),
                 g_view_memo_mode == 2 ? (g_view_memo_mismatch ? " - DISAGREED with the full resolve"
                                                               : " - agreed with every full resolve")
                                       : "");
    }
    host_log("texture: upload staging buffers %s", staging_pool_stats().c_str());
    if (g_untile_check) {
        host_log("texture: untile check: %llu slices compared, %llu differed", static_cast<unsigned long long>(g_untile_checked.load()),
                 static_cast<unsigned long long>(g_untile_mismatched.load()));
    }
    std::string hist;
    for (const auto& kv : g_tiling_hist) hist += " " + std::to_string(kv.first) + ":" + std::to_string(kv.second);
    host_log("texture: surfaces by tiling mode:%s", hist.c_str());
    host_log("texture: GPU writes looked for surfaces over %llu times, skipped %llu with no surface on their pages%s",
             static_cast<unsigned long long>(g_dirty_walks), static_cast<unsigned long long>(g_dirty_walks_skipped),
             g_occupancy_off ? " (page counts off: a surface past 4 TiB)" : "");
    host_log("texture: census: new surfaces on GX resources %llu (%llu created without data), on no resource %llu; "
             "re-uploads explained by a GX write %llu, by a command-processor write %llu, unexplained %llu (noticed by the write watch %llu, "
             "the sampled hash %llu, the whole-surface hash %llu), on no resource %llu, asked for by an unmap %llu",
             static_cast<unsigned long long>(g_census_new_registered.load()), static_cast<unsigned long long>(g_census_new_registered_nodata.load()),
             static_cast<unsigned long long>(g_census_new_unregistered.load()), static_cast<unsigned long long>(g_census_gx.load()),
             static_cast<unsigned long long>(g_census_cp.load()), static_cast<unsigned long long>(g_census_unexplained.load()),
             static_cast<unsigned long long>(g_census_unexplained_watch.load()), static_cast<unsigned long long>(g_census_unexplained_hash.load()),
             static_cast<unsigned long long>(g_census_unexplained_exact.load()), static_cast<unsigned long long>(g_census_unregistered.load()),
             static_cast<unsigned long long>(g_census_unmapped.load()));
    host_log("texture: upload tokens: regions copied into images %llu (%llu MiB), surfaces created for them %llu, without a surface %llu, "
             "out of range %llu",
             static_cast<unsigned long long>(g_region_uploads.load()), static_cast<unsigned long long>(g_region_bytes.load() >> 20),
             static_cast<unsigned long long>(g_region_created.load()), static_cast<unsigned long long>(g_region_no_surface.load()),
             static_cast<unsigned long long>(g_region_out_of_range.load()));
    host_log("texture: copy-image tokens: copies %llu (%llu Mtexels; from a render target %llu, into one %llu), refused: no source %llu, "
             "no destination %llu, formats %llu, range %llu",
             static_cast<unsigned long long>(g_image_copies.load()), static_cast<unsigned long long>(g_image_copy_texels.load() >> 20),
             static_cast<unsigned long long>(g_image_copy_rt_src.load()), static_cast<unsigned long long>(g_image_copy_rt_dst.load()),
             static_cast<unsigned long long>(g_image_copy_no_src.load()), static_cast<unsigned long long>(g_image_copy_no_dst.load()),
             static_cast<unsigned long long>(g_image_copy_format.load()), static_cast<unsigned long long>(g_image_copy_range.load()));
    host_log("texture: render targets sampled through a smaller T# (their top-left region copied): %zu regions, %llu copies (%llu in "
             "a draw's packet); through a "
             "larger T# (the target too small, drawing clipped): %llu binds; memory a target was drawn into read as the live texture the game "
             "put there since: %llu binds; targets created over a shader-written texture of their size, its texels carried: %llu; read in a "
             "format outside the target's view list (through a copy): %llu",
             g_rt_regions.size(), static_cast<unsigned long long>(g_rt_region_copies.load()),
             static_cast<unsigned long long>(g_rt_region_packets.load()),
             static_cast<unsigned long long>(g_rt_smaller_than_tsharp.load()), static_cast<unsigned long long>(g_rt_reused_as_texture.load()),
             static_cast<unsigned long long>(g_rt_from_surface.load()), static_cast<unsigned long long>(g_rt_unlisted_reads.load()));
    host_log("texture: uploads from memory the GPU holds newer bits for (stale reads): %llu over drawn render targets, %llu over GPU-written "
             "textures; T#s at a shader-written level read from the GPU image instead: %zu, %llu copies",
             static_cast<unsigned long long>(g_stale_over_rt.load()), static_cast<unsigned long long>(g_stale_over_surface.load()),
             g_level_aliases.size(), static_cast<unsigned long long>(g_level_alias_copies.load()));
    host_log("texture: untiles on the binding thread %llu in %llu ms (the rest on the prep pool)",
             static_cast<unsigned long long>(g_tex_untile_inline.load()), static_cast<unsigned long long>(g_tex_untile_inline_us.load() / 1000));
    host_log("texture: ahead: surfaces created at their resource's creation %llu (%llu the cache already had, %llu failed); "
             "found by their first bind %llu (%llu before their untile was done), replaced at it %llu; surfaces retired with their resource %llu "
             "(%llu MiB), kept for a live resource %llu",
             static_cast<unsigned long long>(g_ahead_surfaces.load()), static_cast<unsigned long long>(g_ahead_already.load()),
             static_cast<unsigned long long>(g_ahead_failed.load()), static_cast<unsigned long long>(g_ahead_used.load()),
             static_cast<unsigned long long>(g_ahead_bind_before_prep.load()), static_cast<unsigned long long>(g_ahead_replaced.load()),
             static_cast<unsigned long long>(g_retired_surfaces.load()), static_cast<unsigned long long>(g_retired_bytes.load() >> 20),
             static_cast<unsigned long long>(g_retire_kept_live.load()));
    host_log("texture: census: hashes skipped on registered watched surfaces %llu; of the 1-in-64 check hashes %llu, %llu found a change the watch had "
             "not reported",
             static_cast<unsigned long long>(g_census_hash_skipped.load()), static_cast<unsigned long long>(g_census_check_hashed.load()),
             static_cast<unsigned long long>(g_census_check_missed.load()));
    host_log("texture: content hashes=%llu in %llu ms; %llu whole-surface, catching %llu the sampling missed; skipped %llu too big, %llu out of budget",
             static_cast<unsigned long long>(g_hashes.load()), static_cast<unsigned long long>(g_hash_us.load() / 1000),
             static_cast<unsigned long long>(g_exact_hashes.load()), static_cast<unsigned long long>(g_exact_catches.load()),
             static_cast<unsigned long long>(g_exact_too_big.load()), static_cast<unsigned long long>(g_exact_no_budget.load()));
    host_log("texture: uploads prepared off the command processor %s; waited for %llu times (%llu ms in all), %llu of them run on "
             "the waiting thread",
             g_tex_async ? "(BBHOST_TEX_ASYNC)" : "off", static_cast<unsigned long long>(g_prep_waits.load()),
             static_cast<unsigned long long>(g_prep_wait_us.load() / 1000), static_cast<unsigned long long>(g_prep_inline.load()));
    if (g_gpu_untile) {
        host_log("texture: untiled on the GPU (BBHOST_GPU_UNTILE=%d): %llu uploads (%llu MiB), %llu left to the CPU, %llu (%llu MiB) "
                 "through device-local scratch (BBHOST_UNTILE_VRAM); %s",
                 g_gpu_untile, static_cast<unsigned long long>(g_gpu_untiled.load()), static_cast<unsigned long long>(g_gpu_untiled_bytes.load() >> 20),
                 static_cast<unsigned long long>(g_gpu_untile_declined.load()), static_cast<unsigned long long>(g_untiled_vram.load()),
                 static_cast<unsigned long long>(g_untiled_vram_bytes.load() >> 20), untile_gpu_report().c_str());
    }
    if (g_texture_upload_stats) {
        host_log("texture: carry uploads added-mips-only=%llu whole=%llu omitted detile/staging/upload=%llu bytes (BBHOST_CARRY_ADDED_MIPS=%d)",
                 static_cast<unsigned long long>(g_carry_added_uploads.load()), static_cast<unsigned long long>(g_carry_whole_uploads.load()),
                 static_cast<unsigned long long>(g_carry_omitted_bytes.load()), g_carry_added_mips ? 1 : 0);
    }
    {
        // Who takes the faults still in the ring (the last 16384), as Binary
        // Ninja shows them: writer < its caller.
        std::uint64_t pcs[10], callers[10];
        std::uint32_t counts[10];
        const int n = write_watch_fault_writers(pcs, callers, counts, 10);
        std::string top;
        for (int i = 0; i < n; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), " 0x%llx<0x%llx x%u;", static_cast<unsigned long long>(pcs[i] ? gx_guest_to_bn(pcs[i]) : 0),
                          static_cast<unsigned long long>(callers[i] ? gx_guest_to_bn(callers[i]) : 0), counts[i]);
            top += buf;
        }
        if (n) host_log("texture: write watch: the writers behind the last faults:%s", top.c_str());
    }
    const WriteWatchStats ws = write_watch_stats();
    host_log("texture: write watch: %llu arms, %llu write faults (%llu pages more released ahead of them), %llu reads released, %llu remaps, %llu refused; "
             "%llu uploads on a watched write (%llu ms sooner in all than the hashes would have looked), %llu caught "
             "by the hashes instead, %llu paused",
             static_cast<unsigned long long>(ws.arms), static_cast<unsigned long long>(ws.faults),
             static_cast<unsigned long long>(ws.ahead_pages), static_cast<unsigned long long>(ws.releases), static_cast<unsigned long long>(ws.forgets),
             static_cast<unsigned long long>(ws.refused), static_cast<unsigned long long>(g_watch_uploads.load()),
             static_cast<unsigned long long>(g_watch_sooner_ms.load()),
             static_cast<unsigned long long>(g_watch_missed.load()), static_cast<unsigned long long>(g_watch_paused.load()));
    host_log("texture: uploads=%llu (%llu KiB) unsupported=%llu samplers=%zu", static_cast<unsigned long long>(g_uploads.load()),
             static_cast<unsigned long long>(g_upload_bytes.load() / 1024), static_cast<unsigned long long>(g_unsupported.load()),
             g_samplers.size());
}

}  // namespace gpu
