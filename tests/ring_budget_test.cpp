// The rings the CPU writes and the GPU reads every draw go in video memory the
// CPU can write only while that heap's budget leaves room (host/ring_budget.h):
// on a GPU without resizable BAR the window is 256 MiB that every program
// shares, and all three rings in it ran it over its budget.
#include "host/ring_budget.h"

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

void test_ring_budget() {
    const std::uint64_t M = 1ull << 20;
    // An RTX 4070 without resizable BAR (Linux, 2026-10-16): a 256 MiB window,
    // 218 MiB of budget with 27 MiB in use. The params ring (32 MiB) fits, then
    // the constant-buffer copies (64 MiB) - and NVIDIA's budget fell by each
    // allocation (218 -> 186 -> 122) - but not the vertex snapshots' ring.
    CHECK(ring_budget_fits(256 * M, true, 218 * M, 27 * M, 32 * M).fits);
    CHECK(ring_budget_fits(256 * M, true, 186 * M, 59 * M, 64 * M).fits);
    CHECK(!ring_budget_fits(256 * M, true, 122 * M, 123 * M, 64 * M).fits);
    // The budget it ran over before: 187 MiB in use against 58.
    CHECK(!ring_budget_fits(256 * M, true, 58 * M, 187 * M, 1 * M).fits);
    // A driver whose budget counts its own use (the extension's meaning): all
    // three rings fit a 256 MiB window with 230 MiB of budget, room kept.
    CHECK(ring_budget_fits(256 * M, true, 230 * M, 20 * M, 32 * M).fits);
    CHECK(ring_budget_fits(256 * M, true, 230 * M, 52 * M, 64 * M).fits);
    CHECK(ring_budget_fits(256 * M, true, 230 * M, 116 * M, 64 * M).fits);
    CHECK(!ring_budget_fits(256 * M, true, 230 * M, 180 * M, 64 * M).fits);
    // The margin: 32 MiB or an eighth of the budget.
    CHECK(ring_budget_fits(256 * M, true, 230 * M, 0, 64 * M).keep == 32 * M);
    CHECK(ring_budget_fits(12288 * M, true, 11000 * M, 0, 64 * M).keep == 1375 * M);
    // Resizable BAR (the whole of video memory): every ring fits.
    CHECK(ring_budget_fits(12288 * M, true, 11000 * M, 3000 * M, 64 * M).fits);
    // No budget from the driver: a heap of 1 GiB or more only.
    CHECK(!ring_budget_fits(256 * M, false, 0, 0, 32 * M).fits);
    CHECK(ring_budget_fits(16384 * M, false, 0, 0, 64 * M).fits);
}

}  // namespace

int main() {
    test_ring_budget();
    if (g_fail) {
        std::printf("ring_budget_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("ring_budget_test: ok\n");
    return 0;
}
