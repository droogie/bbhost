#pragma once
// Where a plugin's read() reads (plugins.cpp api_read). Plugins name the
// eboot's own bytes by Binary Ninja address - the image at its preferred base
// 0x400000 - and everything else (heaps, stacks, the pointers they read out of
// guest memory) by the runtime address the game sees. Under Linux the image
// sits at its preferred base, so the two kinds agree and the guest's heaps
// lie above the image. Under Windows the image is slid (to 0x300000000 when
// 0x400000 is taken) and the guest's heaps lie in windows below it as well as
// above ([0x100000000, +0x1f0000000) and [0xff00000000, +4 GiB)): a read of
// a heap below the slid image found neither rule and failed. Kept apart from
// the host so a test can drive it.
#include <cstddef>
#include <cstdint>

namespace plugin_address {

constexpr std::uint64_t kPreferredBase = 0x400000;

// The runtime address read(addr, n) copies from, 0 when it may not: a Binary
// Ninja address inside the image moves by the image's slide; any other
// address is taken as a runtime one when it lies above the image's runtime
// range or the guest has the whole range mapped (`mapped(addr, n)`, the
// kernel's map). `image_size` 0: no image yet, nothing is read.
template <typename Mapped>
inline std::uint64_t read_address(std::uint64_t addr, std::size_t n, std::uint64_t slide, std::uint64_t image_size, Mapped&& mapped) {
    if (!image_size) return 0;
    if (addr >= kPreferredBase && addr - kPreferredBase + n <= image_size) return slide + (addr - kPreferredBase);
    if (addr >= slide + image_size) return addr;
    return mapped(addr, n) ? addr : 0;
}

}  // namespace plugin_address
