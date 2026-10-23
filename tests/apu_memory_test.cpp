// What an integrated GPU gives back of the memory it holds in play
// (host/apu_memory.h): the three parts' switches - the environment, then the
// config, then the device's default, off everywhere until
// kWindowsApuDefaults - and the budget, drain and spare arithmetic the memory
// keeper and the textures' eviction and drain use.
#include "host/apu_memory.h"

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

using namespace apu_memory;

constexpr std::uint64_t kMiB = 1ull << 20;

void test_defaults() {
    // Neither set: only an integrated GPU under Windows can take the defaults, and only once they are on.
    CHECK(staging_return(nullptr, "", 1, true) == kWindowsApuDefaults);
    CHECK(heap_lean(nullptr, "", 1, true) == kWindowsApuDefaults);
    CHECK(cache_budget_mib(nullptr, "", 1, true) == (kWindowsApuDefaults ? kWindowsApuBudgetMiB : 0));
    // Linux, a discrete GPU, the device not chosen yet: off.
    for (int integrated = -1; integrated <= 1; ++integrated) {
        for (int host = 0; host < 2; ++host) {
            if (integrated == 1 && host) continue;
            CHECK(!staging_return(nullptr, "", integrated, host));
            CHECK(!heap_lean("", nullptr, integrated, host));
            CHECK(cache_budget_mib(nullptr, "", integrated, host) == 0);
            CHECK(!apu_defaults(integrated, host));
        }
    }
    CHECK(apu_defaults(1, true) == kWindowsApuDefaults);
    // Which device takes them: an integrated GPU under Windows, not one under Linux, a discrete one or none chosen yet.
    CHECK(windows_apu(1, true));
    CHECK(!windows_apu(1, false));
    CHECK(!windows_apu(0, true));
    CHECK(!windows_apu(-1, true));
}

void test_environment() {
    // It wins over the config and the device, everywhere.
    for (int integrated = -1; integrated <= 1; ++integrated) {
        for (int host = 0; host < 2; ++host) {
            CHECK(staging_return("1", "false", integrated, host));
            CHECK(!staging_return("0", "true", integrated, host));
            CHECK(staging_return("2", "", integrated, host));  // any number but 0 is on, as BBHOST_STAGING_RETURN's old parse
            CHECK(heap_lean("1", "false", integrated, host));
            CHECK(!heap_lean("0", "true", integrated, host));
            CHECK(cache_budget_mib("1536", "4096", integrated, host) == 1536);
            CHECK(cache_budget_mib("0", "4096", integrated, host) == 0);  // explicitly none
        }
    }
}

void test_config() {
    for (int integrated = -1; integrated <= 1; ++integrated) {
        for (int host = 0; host < 2; ++host) {
            CHECK(staging_return(nullptr, "true", integrated, host));
            CHECK(!staging_return(nullptr, "false", integrated, host));
            CHECK(!staging_return("", "1", integrated, host));  // only true means on
            CHECK(heap_lean(nullptr, "true", integrated, host));
            CHECK(!heap_lean(nullptr, "false", integrated, host));
            CHECK(cache_budget_mib(nullptr, "3072", integrated, host) == 3072);
            CHECK(cache_budget_mib("", "0", integrated, host) == 0);
        }
    }
    // An integrated GPU under Windows keeps them off with the config, whatever the default.
    CHECK(!staging_return(nullptr, "false", 1, true));
    CHECK(!heap_lean(nullptr, "false", 1, true));
    CHECK(cache_budget_mib(nullptr, "0", 1, true) == 0);
}

void test_flag() {
    CHECK(flag(nullptr, nullptr, true));
    CHECK(!flag(nullptr, nullptr, false));
    CHECK(flag("", "", true));
    CHECK(!flag("", "", false));
    CHECK(!flag("0", nullptr, true));
    CHECK(flag("1", nullptr, false));
    CHECK(flag(nullptr, "true", false));
    CHECK(!flag(nullptr, "yes", true));  // a config value other than true is off, not the fallback
}

void test_budget() {
    // No budget: never over, nothing in excess.
    CHECK(!over_budget(0, 10000 * kMiB, 256 * kMiB));
    CHECK(excess(0, 10000 * kMiB) == 0);
    // Under, at and over 2 GiB, with what is about to be made.
    CHECK(!over_budget(2048 * kMiB, 1500 * kMiB, 0));
    CHECK(!over_budget(2048 * kMiB, 1792 * kMiB, 256 * kMiB));  // exactly at it is not over
    CHECK(over_budget(2048 * kMiB, 1793 * kMiB, 256 * kMiB));
    CHECK(over_budget(2048 * kMiB, 2049 * kMiB, 0));
    CHECK(excess(2048 * kMiB, 2048 * kMiB) == 0);
    CHECK(excess(2048 * kMiB, 1000 * kMiB) == 0);
    CHECK(excess(2048 * kMiB, 2300 * kMiB) == 252 * kMiB);
}

void test_drain_and_spare() {
    const DrainLimits normal = drain_limits(false), lean = drain_limits(true);
    CHECK(normal.holes_at == 384 * kMiB && normal.most_used == 128 * kMiB && normal.hot_per_turn == 16 * kMiB);
    CHECK(lean.holes_at == 128 * kMiB && lean.most_used == 192 * kMiB && lean.hot_per_turn == 32 * kMiB);
    // The spare blocks in play: the environment first, else none when few or lean, else one.
    CHECK(spare_play_blocks(nullptr, false, false) == 1);
    CHECK(spare_play_blocks(nullptr, true, false) == 0);
    CHECK(spare_play_blocks(nullptr, false, true) == 0);
    CHECK(spare_play_blocks("", false, false) == 1);
    CHECK(spare_play_blocks("2", true, true) == 2);
    CHECK(spare_play_blocks("0", false, false) == 0);
}

}  // namespace

int main() {
    test_defaults();
    test_environment();
    test_config();
    test_flag();
    test_budget();
    test_drain_and_spare();
    if (g_fail) {
        std::printf("apu_memory_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("apu_memory_test: ok\n");
    return 0;
}
