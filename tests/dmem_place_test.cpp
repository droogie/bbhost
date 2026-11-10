// Where a direct mapping goes when the kernel chooses its address (core/dmem_place.h): the view of whole blocks
// around [phys, phys + len), the mapping's offset in it, and whether an alignment can be met - checked against
// addresses worked out by hand and against every base a view could get, and the checks shown to catch wrong
// placements (mutants).
#include "core/dmem_place.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>

namespace {

constexpr std::uint64_t kWin = 0x10000;  // Windows: a view's offset and base on the 64 KiB allocation granularity
constexpr std::uint64_t kLin = 0x1000;   // Linux: the page
constexpr std::uint64_t kKiB = 0x400;

using PlaceFn = DmemPlacement (*)(std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);

int g_fail = 0;
bool g_quiet = false;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            if (!g_quiet) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

// The placements worked out by hand.
void test_known(PlaceFn place) {
    // The AvPlayer's four 16 KiB audio buffers on Windows, 32 KiB apart in the game's CPU-only heap (b1 on the
    // laptop: 0x13bc8000 first; the wine runs: 0x13118000, 0x11204000, 0x12490000 ...): each a 64 KiB view of the
    // block holding it, the mapping 0, 16, 32 or 48 KiB into it.
    for (std::uint64_t phys : {0x13bc8000ull, 0x13bd0000ull, 0x13bd8000ull, 0x13be0000ull, 0x11204000ull, 0x1311c000ull}) {
        const DmemPlacement p = place(phys, 16 * kKiB, 16 * kKiB, kWin);
        CHECK(p.ok);
        CHECK(p.view_offset == (phys & ~0xffffull));
        CHECK(p.view_len == kWin);
        CHECK(p.va_offset == (phys & 0xffff));
        CHECK(p.base_align == kWin);
    }
    // Across a block boundary: two blocks.
    DmemPlacement p = place(0x1311c000, 32 * kKiB, 16 * kKiB, kWin);
    CHECK(p.ok && p.view_offset == 0x13110000 && p.view_len == 2 * kWin && p.va_offset == 0xc000);
    // Whole blocks: the view is the mapping.
    p = place(0x7400000, 0x20c00000, 16 * kKiB, kWin);
    CHECK(p.ok && p.view_offset == 0x7400000 && p.view_len == 0x20c00000 && p.va_offset == 0);
    // 64 KiB alignment: only memory on a block boundary can have it.
    CHECK(place(0x12490000, 16 * kKiB, kWin, kWin).ok);
    CHECK(!place(0x12498000, 16 * kKiB, kWin, kWin).ok);
    CHECK(!place(0x12494000, 16 * kKiB, 32 * kKiB, kWin).ok);
    CHECK(place(0x12490000, 16 * kKiB, 32 * kKiB, kWin).ok);
    CHECK(place(0x12498000, 16 * kKiB, 32 * kKiB, kWin).ok);
    // Above the block: the base takes the alignment, the memory must start a block.
    p = place(0x12400000, 16 * kKiB, 0x200000, kWin);
    CHECK(p.ok && p.base_align == 0x200000 && p.va_offset == 0);
    CHECK(place(0x12410000, 16 * kKiB, 0x200000, kWin).ok);  // on a block boundary: the base carries the alignment
    CHECK(!place(0x12404000, 16 * kKiB, 0x200000, kWin).ok);
    // No alignment asked for.
    CHECK(place(0x12494000, 16 * kKiB, 0, kWin).ok);
    // Linux: page views - the view is the mapping, any alignment met by the base.
    p = place(0x13bc8000, 16 * kKiB, 16 * kKiB, kLin);
    CHECK(p.ok && p.view_offset == 0x13bc8000 && p.view_len == 16 * kKiB && p.va_offset == 0 && p.base_align == 16 * kKiB);
    p = place(0x13bc8000, 16 * kKiB, 0x200000, kLin);
    CHECK(p.ok && p.base_align == 0x200000 && p.va_offset == 0);
    p = place(0x13bc8000, 16 * kKiB, 0, kLin);
    CHECK(p.ok && p.base_align == kLin);
    // Nothing to map.
    CHECK(!place(0x13bc8000, 0, 16 * kKiB, kWin).ok);
}

// For every base a view could get, the address the game would see: it lies on its alignment, keeps phys's offset in
// its block, and the view holds the whole mapping in as few blocks as can hold it.
void test_bases(PlaceFn place) {
    const std::uint64_t blocks[] = {kWin, kLin};
    const std::uint64_t aligns[] = {0, 4 * kKiB, 16 * kKiB, 32 * kKiB, 64 * kKiB, 128 * kKiB, 0x200000};
    const std::uint64_t lens[] = {16 * kKiB, 32 * kKiB, 48 * kKiB, 64 * kKiB, 80 * kKiB, 0x100000};
    int placed = 0, refused = 0;
    for (std::uint64_t block : blocks) {
        for (std::uint64_t phys = 0x13bc0000; phys < 0x13bc0000 + 4 * kWin; phys += 16 * kKiB) {
            for (std::uint64_t len : lens) {
                for (std::uint64_t align : aligns) {
                    const DmemPlacement p = place(phys, len, align, block);
                    // Some base gives an aligned address exactly when phys's offset in its block is a multiple of
                    // the alignment (a base is always a multiple of the block).
                    const std::uint64_t in_block = phys % block;
                    const bool can = align == 0 || in_block % align == 0;
                    CHECK(p.ok == can);
                    if (!p.ok) {
                        ++refused;
                        continue;
                    }
                    ++placed;
                    CHECK(p.base_align % block == 0 && p.base_align >= align);
                    CHECK(p.view_offset % block == 0 && p.view_len % block == 0);
                    CHECK(p.view_offset <= phys && phys + len <= p.view_offset + p.view_len);
                    CHECK(p.view_len - len < 2 * block);                     // not a block more than needed
                    CHECK(phys - p.view_offset < block);                      // starts in the view's first block
                    CHECK((p.view_offset + p.view_len) - (phys + len) < block);  // ends in its last
                    const std::uint64_t first = (0x7190000 + p.base_align - 1) & ~(p.base_align - 1);
                    for (std::uint64_t base = first; base < first + 8 * 0x200000; base += p.base_align) {
                        const std::uint64_t va = base + p.va_offset;
                        CHECK(align == 0 || va % align == 0);
                        CHECK(va % block == phys % block);
                        CHECK(va - base == phys - p.view_offset);  // the mapping's bytes are the memory's
                    }
                }
            }
        }
    }
    CHECK(placed > 0 && refused > 0);
}

// Wrong placements the checks must catch.
DmemPlacement m_no_offset(std::uint64_t phys, std::uint64_t len, std::uint64_t align, std::uint64_t block) {
    DmemPlacement p = dmem_placement(phys, len, align, block);
    p.va_offset = 0;  // the view's base handed out: the mapping 16-48 KiB off its memory
    return p;
}
DmemPlacement m_short_view(std::uint64_t phys, std::uint64_t len, std::uint64_t align, std::uint64_t block) {
    DmemPlacement p = dmem_placement(phys, len, align, block);
    p.view_len = (len + block - 1) & ~(block - 1);  // the length rounded, not the end: a block short across a boundary
    return p;
}
DmemPlacement m_ignore_align(std::uint64_t phys, std::uint64_t len, std::uint64_t align, std::uint64_t block) {
    DmemPlacement p = dmem_placement(phys, len, 0, block);
    (void)align;  // every alignment "met"
    return p;
}
DmemPlacement m_small_base(std::uint64_t phys, std::uint64_t len, std::uint64_t align, std::uint64_t block) {
    DmemPlacement p = dmem_placement(phys, len, align, block);
    p.base_align = block;  // a larger alignment not passed on to the base
    return p;
}
DmemPlacement m_phys_view(std::uint64_t phys, std::uint64_t len, std::uint64_t align, std::uint64_t block) {
    DmemPlacement p = dmem_placement(phys, len, align, block);
    p.view_offset = phys;  // the view at phys itself (what Windows refuses off a boundary)
    p.va_offset = 0;
    return p;
}

int run(PlaceFn place) {
    const int before = g_fail;
    test_known(place);
    test_bases(place);
    return g_fail - before;
}

}  // namespace

int main() {
    const int fails = run(&dmem_placement);
    const struct {
        const char* name;
        PlaceFn fn;
    } mutants[] = {{"no offset", m_no_offset},   {"short view", m_short_view}, {"alignment ignored", m_ignore_align},
                   {"small base", m_small_base}, {"view at phys", m_phys_view}};
    int caught = 0;
    g_quiet = true;
    for (const auto& m : mutants) {
        if (run(m.fn) > 0) {
            ++caught;
        } else {
            std::printf("FAIL mutant not caught: %s\n", m.name);
        }
    }
    g_quiet = false;
    const int total = static_cast<int>(sizeof(mutants) / sizeof(mutants[0]));
    if (fails || caught != total) {
        std::printf("dmem_place_test: %d failure(s), %d of %d mutants caught\n", fails, caught, total);
        return 1;
    }
    std::printf("ok - dmem placement, %d mutants caught\n", caught);
    return 0;
}
