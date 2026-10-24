// What an integrated GPU gives back of the memory it holds in play
// (host/apu_memory.h): the three parts' switches - the environment, then the
// config, then the device's default, off everywhere until
// kWindowsApuDefaults - the budget, drain and spare arithmetic the memory
// keeper and the textures' eviction and drain use, and when the keeper gives
// staging back (the world settled, counted from the latest load's end).
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

void test_staging_settle() {
    // The world settles 30 s after the latest load's end, with no loading screen up.
    constexpr std::uint64_t kSettle = 30000;
    CHECK(!settled(0, 100000, false, kSettle));  // no load has ended yet
    CHECK(!settled(10000, 39999, false, kSettle));
    CHECK(settled(10000, 40000, false, kSettle));
    CHECK(!settled(10000, 50000, true, kSettle));   // a loading screen is up
    CHECK(!settled(50000, 40000, false, kSettle));  // a clock read before the end: not yet
    // A load that ends restarts the count; one that begins, or no change, keeps it.
    CHECK(latest_load_end(10000, true, false, 27000) == 27000);
    CHECK(latest_load_end(0, true, false, 9000) == 9000);
    CHECK(latest_load_end(10000, false, true, 26000) == 10000);
    CHECK(latest_load_end(10000, false, false, 26000) == 10000);
    CHECK(latest_load_end(10000, true, true, 26000) == 10000);
    // The 680M's case: the world at 10 s, a death's loading screen from 26 to 28 s, staging over its floor all along.
    // Counted from when staging went over the floor the frees began at 40 s, 12 s into the next arrival; counted
    // from the latest load's end they wait until 58 s.
    const std::uint64_t floor = 256 * kMiB, over = 768 * kMiB, pace = 2000;
    std::uint64_t end = latest_load_end(0, true, false, 10000);
    bool loading = false;
    CHECK(!staging_give_now(true, over, floor, settled(end, 25000, loading, kSettle), false, 25000, 0, pace));
    end = latest_load_end(end, loading, true, 26000);
    loading = true;
    CHECK(end == 10000);
    CHECK(!staging_give_now(true, over, floor, settled(end, 27000, loading, kSettle), false, 27000, 0, pace));
    end = latest_load_end(end, loading, false, 28000);
    loading = false;
    CHECK(end == 28000);
    CHECK(!staging_give_now(true, over, floor, settled(end, 40000, loading, kSettle), false, 40000, 0, pace));
    CHECK(!staging_give_now(true, over, floor, settled(end, 57999, loading, kSettle), false, 57999, 0, pace));
    CHECK(staging_give_now(true, over, floor, settled(end, 58000, loading, kSettle), false, 58000, 0, pace));
    // Then a chunk at a time, `pace` apart - sooner only when video memory runs short.
    CHECK(!staging_give_now(true, over, floor, true, false, 59000, 58000, pace));
    CHECK(staging_give_now(true, over, floor, true, true, 59000, 58000, pace));
    CHECK(staging_give_now(true, over, floor, true, false, 60000, 58000, pace));
    // Short of video memory is no reason to give it back before the world has settled.
    CHECK(!staging_give_now(true, over, floor, false, true, 90000, 0, pace));
    // Down to the floor and no further; off gives nothing back.
    CHECK(!staging_give_now(true, floor, floor, true, false, 90000, 0, pace));
    CHECK(staging_give_now(true, floor + 1, floor, true, false, 90000, 0, pace));
    CHECK(!staging_give_now(false, over, floor, true, true, 90000, 0, pace));
}

}  // namespace

int main() {
    test_defaults();
    test_environment();
    test_config();
    test_flag();
    test_budget();
    test_drain_and_spare();
    test_staging_settle();
    if (g_fail) {
        std::printf("apu_memory_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("apu_memory_test: ok\n");
    return 0;
}
