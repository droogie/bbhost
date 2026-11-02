// GX's buffer copies within a budget (host/gxvk_budget.h): the budget from the
// environment or the device, when a copy is idle, and which copies go first.
#include "host/gxvk_budget.h"

#include <cstdio>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

using namespace gxvk_budget;

void test_budget() {
    // Small video memory: an eighth of what is free, from 256 to 512 MiB.
    CHECK(budget_bytes(nullptr, true, 2950 * kMiB) == 2950 * kMiB / 8);  // a 4 GB card's ~368
    CHECK(budget_bytes(nullptr, true, 1024 * kMiB) == 256 * kMiB);       // clamped up
    CHECK(budget_bytes(nullptr, true, 4700 * kMiB) == 512 * kMiB);       // a 6 GB card: clamped down
    CHECK(budget_bytes("", true, 2048 * kMiB) == 256 * kMiB);            // empty: unset
    // Other cards: no budget.
    CHECK(budget_bytes(nullptr, false, 2950 * kMiB) == 0);
    CHECK(budget_bytes(nullptr, false, 12000 * kMiB) == 0);
    // The environment on any card, 0 for none.
    CHECK(budget_bytes("300", false, 12000 * kMiB) == 300 * kMiB);
    CHECK(budget_bytes("300", true, 2950 * kMiB) == 300 * kMiB);
    CHECK(budget_bytes("0", true, 2950 * kMiB) == 0);
}

void test_idle() {
    CHECK(!idle(1000, 1000));
    CHECK(!idle(1000, 1000 + kIdleFlips - 1));
    CHECK(idle(1000, 1000 + kIdleFlips));
    CHECK(idle(0xffffffa0u, 0xffffffa0u + kIdleFlips));  // across the wrap (the sum wraps to 0x18)
    CHECK(!idle(0xfffffff0u, 0x10u));                     // 32 flips across the wrap
}

void test_victims() {
    const std::uint32_t now = 10000;
    // Hot copies never go; idle ones go oldest first until enough is freed.
    std::vector<Copy> c = {{now - 5, 64 * kMiB}, {now - 500, 8 * kMiB}, {now - 300, 16 * kMiB}, {now - 900, 4 * kMiB}, {now, 32 * kMiB}};
    std::vector<std::size_t> v = victims(c, now, 10 * kMiB);
    CHECK(v.size() == 2 && v[0] == 3 && v[1] == 1);  // 900 flips idle (4 MiB), then 500 (8): 12 MiB >= 10
    v = victims(c, now, 1);
    CHECK(v.size() == 1 && v[0] == 3);
    v = victims(c, now, 1000 * kMiB);  // more than all idle ones hold: every idle one, nothing hot
    CHECK(v.size() == 3 && v[0] == 3 && v[1] == 1 && v[2] == 2);
    v = victims(c, now, 0);
    CHECK(v.empty());
    // Ties keep their order.
    std::vector<Copy> t = {{now - 200, kMiB}, {now - 200, kMiB}, {now - 200, kMiB}};
    v = victims(t, now, 2 * kMiB);
    CHECK(v.size() == 2 && v[0] == 0 && v[1] == 1);
    // Nothing idle: nothing.
    std::vector<Copy> hot = {{now - 1, kMiB}, {now - kIdleFlips + 1, kMiB}};
    CHECK(victims(hot, now, kMiB).empty());
    // A shorter idle time (BBHOST_TEST_GXVK_IDLE_FLIPS) makes them idle; the oldest still first.
    v = victims(hot, now, kMiB, 1);
    CHECK(v.size() == 1 && v[0] == 1);
    CHECK(idle(now - 2, now, 2) && !idle(now - 1, now, 2));
}

}  // namespace

int main() {
    test_budget();
    test_idle();
    test_victims();
    if (g_fail) {
        std::printf("gxvk_budget_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("gxvk_budget_test: ok\n");
    return 0;
}
