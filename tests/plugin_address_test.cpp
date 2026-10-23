// Where a plugin's read() reads (host/plugin_address.h): a Binary Ninja
// address inside the image moves by the slide; any other address is a runtime
// one when it lies above the image or the guest has it mapped - under Linux
// (the image at 0x400000) and under Windows (the image slid to 0x300000000,
// the guest's heaps below it as well as above).
#include "host/plugin_address.h"

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

using plugin_address::read_address;

constexpr std::uint64_t kImageSize = 0x5ad4000;  // the 1.09 eboot's image
constexpr std::uint64_t kLinuxSlide = 0x400000, kWindowsSlide = 0x300000000;

// The guest's mapped ranges in each layout, as the kernel's map would answer.
struct Range {
    std::uint64_t lo, hi;
};
bool in(const Range* r, int count, std::uint64_t a, std::size_t n) {
    for (int i = 0; i < count; ++i) {
        if (a >= r[i].lo && a + n <= r[i].hi) return true;
    }
    return false;
}
const Range kWindowsMaps[] = {{0x100000000, 0x2f0000000}, {kWindowsSlide, kWindowsSlide + kImageSize}, {0xff00000000, 0x10000000000}};
const Range kLinuxMaps[] = {{kLinuxSlide, kLinuxSlide + kImageSize}, {0x200000000, 0x400000000}};
auto windows_mapped = [](std::uint64_t a, std::size_t n) { return in(kWindowsMaps, 3, a, n); };
auto linux_mapped = [](std::uint64_t a, std::size_t n) { return in(kLinuxMaps, 2, a, n); };
auto nothing_mapped = [](std::uint64_t, std::size_t) { return false; };

void test_windows() {
    // The image's own bytes by Binary Ninja address: slid.
    CHECK(read_address(0x58b0000, 8, kWindowsSlide, kImageSize, windows_mapped) == 0x3054b0000);
    CHECK(read_address(0x400000, 1, kWindowsSlide, kImageSize, windows_mapped) == kWindowsSlide);
    CHECK(read_address(0x400000 + kImageSize - 8, 8, kWindowsSlide, kImageSize, windows_mapped) == kWindowsSlide + kImageSize - 8);
    // A heap below the slid image: read where it is (what failed before), and only while the guest has it mapped.
    CHECK(read_address(0x134731000, 0x70, kWindowsSlide, kImageSize, windows_mapped) == 0x134731000);
    CHECK(read_address(0x134731000, 0x70, kWindowsSlide, kImageSize, nothing_mapped) == 0);
    CHECK(read_address(0x2f0000000 - 4, 8, kWindowsSlide, kImageSize, windows_mapped) == 0);  // runs past the mapped window
    // Above the image: a runtime address as before, mapped or not (the safe copy catches a bad one).
    CHECK(read_address(0xff00001000, 16, kWindowsSlide, kImageSize, windows_mapped) == 0xff00001000);
    CHECK(read_address(kWindowsSlide + kImageSize, 16, kWindowsSlide, kImageSize, nothing_mapped) == kWindowsSlide + kImageSize);
    // A runtime address inside the slid image (a pointer read from guest memory): itself, being mapped.
    CHECK(read_address(kWindowsSlide + 0x1000, 8, kWindowsSlide, kImageSize, windows_mapped) == kWindowsSlide + 0x1000);
    CHECK(read_address(kWindowsSlide + 0x1000, 8, kWindowsSlide, kImageSize, nothing_mapped) == 0);
    // Below the image's Binary Ninja range, unmapped: refused.
    CHECK(read_address(0x1000, 8, kWindowsSlide, kImageSize, windows_mapped) == 0);
    // A Binary Ninja range running past the image's end: not the image's; unmapped there, refused.
    CHECK(read_address(0x400000 + kImageSize - 4, 8, kWindowsSlide, kImageSize, windows_mapped) == 0);
}

void test_linux() {
    // Binary Ninja and runtime addresses agree: the image unmoved, the heaps above it.
    CHECK(read_address(0x58b0000, 8, kLinuxSlide, kImageSize, linux_mapped) == 0x58b0000);
    CHECK(read_address(0x200001000, 64, kLinuxSlide, kImageSize, linux_mapped) == 0x200001000);
    CHECK(read_address(0x200001000, 64, kLinuxSlide, kImageSize, nothing_mapped) == 0x200001000);  // above the image
    CHECK(read_address(0x1000, 8, kLinuxSlide, kImageSize, linux_mapped) == 0);
}

void test_no_image() {
    CHECK(read_address(0x58b0000, 8, kWindowsSlide, 0, windows_mapped) == 0);
    CHECK(read_address(0x134731000, 8, kWindowsSlide, 0, windows_mapped) == 0);
}

}  // namespace

int main() {
    test_windows();
    test_linux();
    test_no_image();
    if (g_fail) {
        std::printf("plugin_address_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("plugin_address_test: ok\n");
    return 0;
}
