// The memory policy (host/memory_policy.h): the device's class from its kind,
// each part's default on every class and platform, every switch winning on
// any device, and the log line naming what decided each part.
#include "host/memory_policy.h"

#include <cstdio>
#include <string>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

using memory_policy::decide;
using memory_policy::Device;
using memory_policy::device_of;
using memory_policy::Policy;
using memory_policy::Switches;
using gxvk::Buffers;
constexpr std::uint64_t kMiB = 1ull << 20;
constexpr std::uint64_t kSmallFree = 2950 * kMiB;  // a 4 GB card's budget less the driver's own (BBHOST_TEST_VRAM_MB=2950)

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

void test_device_of() {
    // Integrated first: with room and not tight a roomy one; tight or small, a smaller one, whatever else is said.
    CHECK(device_of(true, true, false, false) == Device::kIntegratedRoomy);
    CHECK(device_of(true, true, true, false) == Device::kIntegrated);
    CHECK(device_of(true, false, false, false) == Device::kIntegrated);
    CHECK(device_of(true, false, true, true) == Device::kIntegrated);  // a Steam Deck (tight), small forced or not
    CHECK(device_of(true, true, false, true) == Device::kIntegratedRoomy);
    // A discrete GPU: tight memory, then small video memory, else room.
    CHECK(device_of(false, false, true, false) == Device::kTight);
    CHECK(device_of(false, false, true, true) == Device::kTight);
    CHECK(device_of(false, false, false, true) == Device::kSmallVram);
    CHECK(device_of(false, false, false, false) == Device::kRoomy);
    CHECK(device_of(false, true, false, false) == Device::kRoomy);  // room means nothing off an integrated GPU
}

void test_defaults() {
    const Switches none;
    for (int host = 0; host < 2; ++host) {
        const bool win = host == 1;
        // A card with room: copies with no budget, whole imports, the staging kept.
        Policy p = decide(Device::kRoomy, win, 10800 * kMiB, false, none);
        CHECK(p.buffers == Buffers::kOn && p.budget == 0 && !p.windows && !p.staging_return);
        // A card with small video memory: copies within a budget, windows, the staging given back - on either platform.
        p = decide(Device::kSmallVram, win, kSmallFree, false, none);
        CHECK(p.buffers == Buffers::kOn && p.budget == kSmallFree / 8 && p.windows && p.staging_return);
        // The budget clamped to 256-512 MiB.
        CHECK(decide(Device::kSmallVram, win, 1024 * kMiB, false, none).budget == 256 * kMiB);
        CHECK(decide(Device::kSmallVram, win, 7000 * kMiB, false, none).budget == 512 * kMiB);
        // An integrated GPU with room: copies with no budget; windows and the staging back under Windows only.
        p = decide(Device::kIntegratedRoomy, win, 8192 * kMiB, false, none);
        CHECK(p.buffers == Buffers::kOn && p.budget == 0 && p.windows == win);
        CHECK(p.staging_return == (win && apu_memory::kWindowsApuDefaults));
        // A smaller integrated GPU: today's path; the platform as above.
        p = decide(Device::kIntegrated, win, 2048 * kMiB, false, none);
        CHECK(p.buffers == Buffers::kOff && p.budget == 0 && p.windows == win);
        CHECK(p.staging_return == (win && apu_memory::kWindowsApuDefaults));
        // NVIDIA's driver: a small card keeps its staging (each chunk's free stalls a frame) and the rest as above; nothing
        // else changes for it.
        p = decide(Device::kSmallVram, win, kSmallFree, true, none);
        CHECK(p.buffers == Buffers::kOn && p.budget == kSmallFree / 8 && p.windows && !p.staging_return && p.nvidia);
        p = decide(Device::kRoomy, win, 10800 * kMiB, true, none);
        CHECK(p.buffers == Buffers::kOn && p.budget == 0 && !p.windows && !p.staging_return);
        CHECK(decide(Device::kIntegrated, win, 2048 * kMiB, true, none).staging_return == (win && apu_memory::kWindowsApuDefaults));
        // Tight memory: today's path, whole imports, the staging kept.
        p = decide(Device::kTight, win, kSmallFree, false, none);
        CHECK(p.buffers == Buffers::kOff && p.budget == 0 && !p.windows && !p.staging_return);
        CHECK(p.device == Device::kTight && p.windows_host == win && p.free_at_start == kSmallFree);
    }
}

void test_switches() {
    for (int host = 0; host < 2; ++host) {
        const bool win = host == 1;
        Switches s;
        // GX's buffers: off takes the budget with it; compared keeps it; on anywhere, budgeted only off an integrated GPU.
        s.buffers = "0";
        Policy p = decide(Device::kSmallVram, win, kSmallFree, false, s);
        CHECK(p.buffers == Buffers::kOff && p.budget == 0 && p.windows && p.staging_return);
        s.buffers = "2";
        p = decide(Device::kSmallVram, win, kSmallFree, false, s);
        CHECK(p.buffers == Buffers::kCompare && p.budget == kSmallFree / 8);
        s.buffers = "1";
        CHECK(decide(Device::kIntegrated, win, kSmallFree, false, s).buffers == Buffers::kOn);
        CHECK(decide(Device::kIntegrated, win, kSmallFree, false, s).budget == 0);
        CHECK(decide(Device::kTight, win, kSmallFree, false, s).buffers == Buffers::kOn);
        CHECK(decide(Device::kTight, win, kSmallFree, false, s).budget == 0);
        s.buffers = "census";
        CHECK(decide(Device::kRoomy, win, kSmallFree, false, s).buffers == Buffers::kOff);
        s.buffers = "";
        CHECK(decide(Device::kRoomy, win, kSmallFree, false, s).buffers == Buffers::kOn);
        // The budget's switch: any discrete GPU with the copies, never an integrated one; 0 none.
        s = Switches{};
        s.budget = "300";
        CHECK(decide(Device::kRoomy, win, 10800 * kMiB, false, s).budget == 300 * kMiB);
        CHECK(decide(Device::kSmallVram, win, kSmallFree, false, s).budget == 300 * kMiB);
        CHECK(decide(Device::kIntegratedRoomy, win, 8192 * kMiB, false, s).budget == 0);
        s.budget = "0";
        CHECK(decide(Device::kSmallVram, win, kSmallFree, false, s).budget == 0);
        // The imports: the environment, then the config, on any device.
        s = Switches{};
        s.imports_env = "0";
        CHECK(!decide(Device::kSmallVram, win, kSmallFree, false, s).windows);
        CHECK(!decide(Device::kIntegrated, win, kSmallFree, false, s).windows);
        s.imports_env = "1";
        s.imports_cfg = "false";
        CHECK(decide(Device::kRoomy, win, kSmallFree, false, s).windows);
        s.imports_env = nullptr;
        CHECK(!decide(Device::kSmallVram, win, kSmallFree, false, s).windows);
        s.imports_cfg = "true";  // the Steam Deck kit's toml
        CHECK(decide(Device::kIntegrated, win, kSmallFree, false, s).windows);
        CHECK(decide(Device::kTight, win, kSmallFree, false, s).windows);
        // The staging: the environment, then the config, on any device - and apart from the imports.
        s = Switches{};
        s.staging_env = "0";
        p = decide(Device::kSmallVram, win, kSmallFree, false, s);
        CHECK(!p.staging_return && p.windows);
        s.staging_env = nullptr;
        s.staging_cfg = "true";
        CHECK(decide(Device::kRoomy, win, kSmallFree, false, s).staging_return);
        s.staging_cfg = "false";
        CHECK(!decide(Device::kIntegrated, true, kSmallFree, false, s).staging_return);
        // The switches give it back on NVIDIA's driver too.
        s.staging_cfg = "true";
        CHECK(decide(Device::kSmallVram, win, kSmallFree, true, s).staging_return);
        s.staging_cfg = nullptr;
        s.staging_env = "1";
        CHECK(decide(Device::kSmallVram, win, kSmallFree, true, s).staging_return);
    }
}

void test_retire() {
    const Switches none;
    for (int host = 0; host < 2; ++host) {
        const bool win = host == 1;
        // Idle windows let go wherever windows are imported: a small card's by default...
        CHECK(decide(Device::kSmallVram, win, kSmallFree, false, none).retire == import_retire::kOn);
        // ... but NVIDIA's under Windows, whose driver takes 23-57 ms to import a window again (under Linux: let go) ...
        CHECK(decide(Device::kSmallVram, win, kSmallFree, true, none).retire == (win ? import_retire::kOff : import_retire::kOn));
        // ... nothing to let go with whole imports ...
        CHECK(decide(Device::kRoomy, win, 10800 * kMiB, false, none).retire == import_retire::kOff);
        CHECK(decide(Device::kTight, win, kSmallFree, false, none).retire == import_retire::kOff);
        // ... an integrated GPU's under Windows, where its windows are the default (Linux imports whole) ...
        CHECK(decide(Device::kIntegrated, win, 2048 * kMiB, false, none).retire == (win ? import_retire::kOn : import_retire::kOff));
        CHECK(decide(Device::kIntegratedRoomy, win, 8192 * kMiB, false, none).retire == (win ? import_retire::kOn : import_retire::kOff));
        // ... and one under Linux whose windows are asked for (gpu.import_windows).
        Switches s;
        s.imports_cfg = "true";  // the Steam Deck kit's toml
        CHECK(decide(Device::kIntegrated, win, 2048 * kMiB, false, s).windows);
        CHECK(decide(Device::kIntegrated, win, 2048 * kMiB, false, s).retire == import_retire::kOn);
        // Windows asked for on a card with room: let go too, but on NVIDIA's driver under Windows.
        s = Switches{};
        s.imports_env = "1";
        CHECK(decide(Device::kRoomy, win, 10800 * kMiB, false, s).retire == import_retire::kOn);
        CHECK(decide(Device::kRoomy, win, 10800 * kMiB, true, s).retire == (win ? import_retire::kOff : import_retire::kOn));
        // The switch wins wherever windows are imported, and means nothing with whole ones.
        s.retire = "0";
        CHECK(decide(Device::kRoomy, win, 10800 * kMiB, false, s).retire == import_retire::kOff);
        s.retire = "2";
        CHECK(decide(Device::kRoomy, win, 10800 * kMiB, false, s).retire == import_retire::kCount);
        s.imports_env = nullptr;
        s.retire = "1";
        CHECK(decide(Device::kRoomy, win, 10800 * kMiB, false, s).retire == import_retire::kOff);  // whole: nothing to let go
        CHECK(decide(Device::kIntegrated, true, 2048 * kMiB, false, s).retire == import_retire::kOn);   // Windows' windows
        CHECK(decide(Device::kIntegrated, false, 2048 * kMiB, false, s).retire == import_retire::kOff);  // Linux: whole
        s.imports_cfg = "true";
        CHECK(decide(Device::kIntegrated, false, 2048 * kMiB, false, s).retire == import_retire::kOn);   // the Deck's, asked
        s.retire = "0";
        CHECK(decide(Device::kSmallVram, win, kSmallFree, false, s).retire == import_retire::kOff);
        s = Switches{};
        s.retire = "1";  // NVIDIA's under Windows, asked for
        CHECK(decide(Device::kSmallVram, win, kSmallFree, true, s).retire == import_retire::kOn);
    }
    // The log's words.
    std::string d = memory_policy::describe(decide(Device::kSmallVram, true, kSmallFree, false, none), none);
    CHECK(has(d, "a window import a minute without a use let go") && !has(d, "(BBHOST_IMPORT_RETIRE)"));
    d = memory_policy::describe(decide(Device::kRoomy, true, 10800 * kMiB, false, none), none);
    CHECK(!has(d, "let go") && !has(d, "window import kept"));  // whole imports: no part said
    d = memory_policy::describe(decide(Device::kIntegrated, true, 2048 * kMiB, false, none), none);
    CHECK(has(d, "a window import a minute without a use let go") && !has(d, "(BBHOST_IMPORT_RETIRE)") && !has(d, "every window import kept"));
    d = memory_policy::describe(decide(Device::kSmallVram, true, kSmallFree, true, none), none);
    CHECK(has(d, "every window import kept, as NVIDIA's driver under Windows takes 23-57 ms to import a window wanted again, on the "
                 "command processor"));
    d = memory_policy::describe(decide(Device::kSmallVram, false, kSmallFree, true, none), none);
    CHECK(has(d, "a window import a minute without a use let go") && !has(d, "NVIDIA's driver under Windows"));
    Switches s;
    s.retire = "2";
    d = memory_policy::describe(decide(Device::kSmallVram, true, kSmallFree, false, s), s);
    CHECK(has(d, "idle window imports counted, none let go (BBHOST_IMPORT_RETIRE)"));
    s.retire = "0";
    d = memory_policy::describe(decide(Device::kSmallVram, true, kSmallFree, false, s), s);
    CHECK(has(d, "every window import kept (BBHOST_IMPORT_RETIRE)"));
    d = memory_policy::describe(decide(Device::kSmallVram, true, kSmallFree, true, s), s);
    CHECK(has(d, "every window import kept (BBHOST_IMPORT_RETIRE)") && !has(d, "NVIDIA's driver under Windows takes"));
    s.retire = "1";
    d = memory_policy::describe(decide(Device::kSmallVram, true, kSmallFree, true, s), s);
    CHECK(has(d, "a window import a minute without a use let go (BBHOST_IMPORT_RETIRE)"));
}

void test_describe() {
    const Switches none;
    std::string d = memory_policy::describe(decide(Device::kSmallVram, true, kSmallFree, false, none), none);
    CHECK(has(d, "small video memory (2950 MiB free at the start"));
    CHECK(has(d, "bound by handle, their copies within a budget of 368 MiB (an eighth"));
    CHECK(has(d, "a 16 MB window at a time"));
    CHECK(has(d, "given back down to its floor"));
    CHECK(!has(d, "(BBHOST_IMPORT_WINDOWS)") && !has(d, "(gpu.import_windows)") && !has(d, "(BBHOST_STAGING_RETURN)"));
    CHECK(has(d, "under 7680"));
    Switches forced;
    forced.small = "1";  // lavapipe's tests take the card as small
    CHECK(has(memory_policy::describe(decide(Device::kSmallVram, false, kSmallFree, false, forced), forced), "at the start, BBHOST_SMALL_VRAM)"));
    CHECK(!has(d, "NVIDIA"));
    d = memory_policy::describe(decide(Device::kSmallVram, true, kSmallFree, true, none), none);
    CHECK(has(d, "the upload staging kept, as NVIDIA's driver stalls the frame a staging chunk's free falls in"));
    d = memory_policy::describe(decide(Device::kRoomy, false, 10800 * kMiB, true, none), none);
    CHECK(!has(d, "NVIDIA"));
    d = memory_policy::describe(decide(Device::kRoomy, false, 10800 * kMiB, false, none), none);
    CHECK(has(d, "room in its video memory (10800 MiB"));
    CHECK(has(d, "GX's buffers bound by handle;"));
    CHECK(has(d, "imported whole") && has(d, "the upload staging kept"));
    CHECK(!has(memory_policy::describe(decide(Device::kRoomy, false, 0, false, none), none), "MiB free"));  // no device asked
    d = memory_policy::describe(decide(Device::kIntegratedRoomy, true, 8192 * kMiB, false, none), none);
    CHECK(has(d, "8 GiB of video memory or more") && has(d, "from their copies in its video memory"));
    d = memory_policy::describe(decide(Device::kIntegrated, false, 1024 * kMiB, false, none), none);
    CHECK(has(d, "an integrated GPU;") && has(d, "today's path, read where they are") && has(d, "imported whole"));
    // What decided a part, when a switch did.
    Switches s;
    s.buffers = "2";
    s.budget = "64";
    s.imports_cfg = "false";
    s.staging_env = "0";
    d = memory_policy::describe(decide(Device::kSmallVram, true, kSmallFree, false, s), s);
    CHECK(has(d, "compared with today's path (BBHOST_GXVK_BUFFERS)") && has(d, "budget of 64 MiB (BBHOST_GXVK_BUFFERS_BUDGET_MB)"));
    CHECK(has(d, "imported whole (gpu.import_windows)") && has(d, "staging kept (BBHOST_STAGING_RETURN)"));
    s = Switches{};
    s.buffers = "0";
    s.imports_env = "1";
    s.imports_cfg = "false";
    d = memory_policy::describe(decide(Device::kRoomy, false, 10800 * kMiB, false, s), s);
    CHECK(has(d, "today's path (BBHOST_GXVK_BUFFERS)") && has(d, "at a time as the GPU first needs it (BBHOST_IMPORT_WINDOWS)"));
}

}  // namespace

int main() {
    test_device_of();
    test_defaults();
    test_switches();
    test_retire();
    test_describe();
    if (g_fail) {
        std::printf("memory_policy_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("memory_policy_test: ok\n");
    return 0;
}
