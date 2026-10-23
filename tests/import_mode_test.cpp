// How direct memory reaches the GPU (host/import_mode.h): the environment
// first, then the config, then the device and the platform - windows on an
// integrated GPU under Windows, whole spans everywhere else - and undecided
// until the device is known.
#include "host/import_mode.h"

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

using import_mode::Mode;
using import_mode::decide;

void test_device_and_platform() {
    // Neither set: an integrated GPU under Windows takes windows; under Linux, and a discrete GPU anywhere, whole spans.
    CHECK(decide(nullptr, "", 1, true) == Mode::kWindows);
    CHECK(decide(nullptr, "", 1, false) == Mode::kWhole);
    CHECK(decide(nullptr, "", 0, true) == Mode::kWhole);
    CHECK(decide(nullptr, "", 0, false) == Mode::kWhole);
    CHECK(decide("", nullptr, 1, true) == Mode::kWindows);
    CHECK(decide("", nullptr, 1, false) == Mode::kWhole);
    // The device not chosen yet: undecided, on either platform.
    CHECK(decide(nullptr, "", -1, true) == Mode::kUndecided);
    CHECK(decide(nullptr, "", -1, false) == Mode::kUndecided);
}

void test_environment() {
    // It wins over the config and the device, on every platform, before the device is known too.
    for (int integrated = -1; integrated <= 1; ++integrated) {
        for (int host = 0; host < 2; ++host) {
            CHECK(decide("1", "", integrated, host) == Mode::kWindows);
            CHECK(decide("0", "", integrated, host) == Mode::kWhole);
            CHECK(decide("1", "false", integrated, host) == Mode::kWindows);
            CHECK(decide("0", "true", integrated, host) == Mode::kWhole);
            CHECK(decide("yes", "true", integrated, host) == Mode::kWhole);  // only 1 means windows
        }
    }
}

void test_config() {
    for (int integrated = -1; integrated <= 1; ++integrated) {
        for (int host = 0; host < 2; ++host) {
            CHECK(decide(nullptr, "true", integrated, host) == Mode::kWindows);
            CHECK(decide(nullptr, "false", integrated, host) == Mode::kWhole);
            CHECK(decide("", "true", integrated, host) == Mode::kWindows);
            CHECK(decide(nullptr, "1", integrated, host) == Mode::kWhole);  // only true means windows
        }
    }
    // A Linux integrated GPU turns windows on with the config, as the Steam Deck kit's toml does.
    CHECK(decide(nullptr, "true", 1, false) == Mode::kWindows);
    // A Windows integrated GPU keeps whole spans with false.
    CHECK(decide(nullptr, "false", 1, true) == Mode::kWhole);
}

}  // namespace

int main() {
    test_device_and_platform();
    test_environment();
    test_config();
    if (g_fail) {
        std::printf("import_mode_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("import_mode_test: ok\n");
    return 0;
}
