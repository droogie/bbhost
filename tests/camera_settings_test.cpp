// The camera settings (engine/camera.cpp) against a small LockCamParam laid
// out as the game has it: applied when they change, left alone while they do
// not - a view then reads one flag and touches neither the settings nor the
// table - put back when they return to the game's own, taken from the param
// repository when it names the table (no search), and written again when it
// reloads it, in place or into a new buffer, without scaling twice.
#include "core/elf.h"
#include "engine/addr.h"
#include "engine/camera.h"
#include "hle/modules.h"
#include "host/settings_store.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {
int g_fail = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++g_fail;                                                  \
        }                                                              \
    } while (0)

HostSettingsStore& store() {
    static HostSettingsStore& s = *new HostSettingsStore;
    return s;
}

using Table = std::array<std::uint8_t, 512>;
Table g_table{}, g_moved{}, g_repo{}, g_reloaded{};
std::mutex g_maps_mu;
std::vector<GuestMapInfo> g_maps;
std::atomic<int> g_scans{0};

// The file as the game keeps it: row count at +0xa, the type name at +0xc,
// 0x18-byte records from +0x40 (data offset at +8), 0x20-byte rows with
// camDistTarget at +0, chrOrgOffset_Y at +0xc and camFovY at +0x14.
void make_table(Table& t, float distance, float height, float fov) {
    t.fill(0);
    const std::uint16_t rows = 2;
    std::memcpy(t.data() + 0xa, &rows, 2);
    const char type[] = "LOCK_CAM_PARAM_ST";
    std::memcpy(t.data() + 0xc, type, sizeof(type));
    for (std::uint64_t r = 0; r < 2; ++r) {
        const std::uint64_t off = 0x80 + r * 0x20;
        std::memcpy(t.data() + 0x40 + r * 0x18 + 8, &off, 8);
        std::memcpy(t.data() + off, &distance, 4);
        std::memcpy(t.data() + off + 0xc, &height, 4);
        std::memcpy(t.data() + off + 0x14, &fov, 4);
    }
}

float field(const Table& t, int row, int at) {
    float v = 0;
    std::memcpy(&v, t.data() + 0x80 + row * 0x20 + at, 4);
    return v;
}
void set_field(Table& t, int row, int at, float v) { std::memcpy(t.data() + 0x80 + row * 0x20 + at, &v, 4); }

void map_only(Table& t) {
    std::lock_guard<std::mutex> lock(g_maps_mu);
    g_maps = {{reinterpret_cast<std::uint64_t>(t.data()), t.size(), -1, false, 3}};
}

// Views until the rows read `want` (the search runs on a thread of its own).
bool views_until(const Table& t, float want) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < end) {
        camera_tick();
        if (field(t, 0, 0) == want) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

void publish_camera(float distance, float height, float fov) {
    HostSettings h = store().now();
    h.camera_distance_scale = distance;
    h.camera_height_scale = height;
    h.fov_scale = fov;
    store().publish(h);
}
}  // namespace

std::uint64_t at(Table& t) { return reinterpret_cast<std::uint64_t>(t.data()); }

// Views until a field of row 0 reads `want`.
bool views_until_field(const Table& t, int off, float want) {
    for (int i = 0; i < 200; ++i) {
        camera_tick();
        if (field(t, 0, off) == want) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

// What camera.cpp links against.
void host_settings_listen(HostSettingsListener fn) { store().listen(fn); }
bool host_read_safe(const void* addr, void* out, std::size_t n) {
    std::memcpy(out, addr, n);  // the tables here are all ours
    return true;
}
void hle_kernel_snapshot_maps(std::vector<GuestMapInfo>& out) {
    std::lock_guard<std::mutex> lock(g_maps_mu);
    out = g_maps;
    ++g_scans;
}
void hle_watch_arm_reads(std::uint64_t, std::uint64_t) {}
void hle_watch_report() {}
const void* host_memmem(const void* hay, std::size_t n, const void* needle, std::size_t m) {
    const auto* h = static_cast<const std::uint8_t*>(hay);
    const auto* k = static_cast<const std::uint8_t*>(needle);
    const auto* at = std::search(h, h + n, k, k + m);
    return at == h + n ? nullptr : at;
}

int main() {
    unsetenv("BBHOST_FOV_WATCH");
    // camera_install reads the field-of-view clamp's bytes in the image and
    // registers the settings' listener.
    constexpr std::uint64_t kFovClamp = 0x183af4e;
    std::vector<std::uint8_t> image_bytes(kFovClamp - kPreferredGuestSlide + 16, 0);
    ElfImage image;
    image.mem.base = image_bytes.data();
    image.mem.size = image_bytes.size();
    image.mem.slide = kPreferredGuestSlide;
    camera_install(&image);

    make_table(g_table, 4, 2, 43);
    map_only(g_table);

    // The game's own settings: views never look for the table or touch it.
    for (int i = 0; i < 100; ++i) camera_tick();
    CHECK(g_scans.load() == 0);
    CHECK(field(g_table, 0, 0) == 4);

    // A change: the next view looks for the table, then writes every row.
    publish_camera(2.0f, 1.5f, 1.25f);
    CHECK(views_until(g_table, 8));
    CHECK(g_scans.load() == 1);
    CHECK(field(g_table, 0, 0xc) == 3 && field(g_table, 0, 0x14) == 53.75f);
    CHECK(field(g_table, 1, 0) == 8 && field(g_table, 1, 0x14) == 53.75f);

    // Nothing changed: a thousand views write nothing and look for nothing.
    // (A value moved behind the camera's back stays moved: no view walks the
    // rows to check them.)
    set_field(g_table, 1, 0, 7);
    for (int i = 0; i < 1000; ++i) camera_tick();
    CHECK(field(g_table, 1, 0) == 7);
    CHECK(g_scans.load() == 1);

    // Another setting changed: still nothing to do for the camera.
    {
        HostSettings h = store().now();
        h.fps_counter = !h.fps_counter;
        store().publish(h);
    }
    for (int i = 0; i < 100; ++i) camera_tick();
    CHECK(field(g_table, 1, 0) == 7);

    // The param repository names the same buffer (reloaded in place): the
    // rows are written again - the one the game changed from its new value
    // (7 is not what was written: the game's own now), the rest left as
    // they were, not scaled twice - and nothing is searched for.
    camera_params_changed(at(g_table));
    camera_tick();
    CHECK(field(g_table, 1, 0) == 14);
    CHECK(field(g_table, 0, 0) == 8 && field(g_table, 0, 0x14) == 53.75f);
    CHECK(g_scans.load() == 1);

    // A new value scales the game's own, not what was written before.
    publish_camera(1.5f, 1.0f, 1.0f);
    for (int i = 0; i < 5; ++i) camera_tick();
    CHECK(field(g_table, 0, 0) == 6 && field(g_table, 0, 0xc) == 2 && field(g_table, 0, 0x14) == 43);

    // Back to the game's own: the rows go back too, from a change made on
    // another thread (the window's, where the menus run).
    std::thread menu([] { publish_camera(1.0f, 1.0f, 1.0f); });
    menu.join();
    for (int i = 0; i < 5; ++i) camera_tick();
    CHECK(field(g_table, 0, 0) == 4 && field(g_table, 0, 0xc) == 2 && field(g_table, 0, 0x14) == 43);
    CHECK(field(g_table, 1, 0) == 7);
    const int scans = g_scans.load();
    for (int i = 0; i < 100; ++i) camera_tick();
    CHECK(g_scans.load() == scans);

    // The table moved (its old address no longer reads as the file, and the
    // repository has said nothing): the next change searches and finds it.
    make_table(g_moved, 5, 2, 43);
    map_only(g_moved);
    std::memset(g_table.data() + 0xc, 0, 18);
    publish_camera(2.0f, 1.0f, 1.0f);
    CHECK(views_until(g_moved, 10));
    CHECK(field(g_moved, 1, 0) == 10);
    const int searched = g_scans.load();
    CHECK(searched == scans + 1);

    // The param repository names the buffer the game reads: taken as it is,
    // with no search, and written at once.
    make_table(g_repo, 3, 1, 40);
    camera_params_changed(at(g_repo));
    CHECK(views_until_field(g_repo, 0, 6));
    CHECK(field(g_repo, 1, 0) == 6 && field(g_repo, 0, 0xc) == 1);
    CHECK(g_scans.load() == searched);

    // The same memory seen at another address (the direct memory's second
    // mapping) is not scaled twice: what was written carries over.
    Table alias = g_repo;  // a copy standing in for the second view of it
    camera_params_changed(at(alias));
    for (int i = 0; i < 5; ++i) camera_tick();
    CHECK(field(alias, 0, 0) == 6 && field(alias, 1, 0) == 6);

    // A reload into a new buffer has the game's own values: they are scaled.
    make_table(g_reloaded, 4.5f, 1, 40);
    camera_params_changed(at(g_reloaded));
    CHECK(views_until_field(g_reloaded, 0, 9));
    CHECK(field(g_reloaded, 1, 0) == 9);
    CHECK(g_scans.load() == searched);

    // And with nothing pending, views leave even the repository's table be.
    set_field(g_reloaded, 1, 0, 1);
    for (int i = 0; i < 1000; ++i) camera_tick();
    CHECK(field(g_reloaded, 1, 0) == 1);

    if (g_fail) return 1;
    std::printf("camera_settings_test: ok\n");
    return 0;
}
