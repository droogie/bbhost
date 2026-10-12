// Exercise the actual camera consumer with a small guest parameter table.
#include "engine/camera.h"
#include "hle/modules.h"
#include "host/settings_snapshot.h"

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
HostSettingsSnapshot g_settings;
std::array<std::uint8_t, 512> g_table{}, g_replacement{};
std::mutex g_maps_mu;
std::vector<GuestMapInfo> g_maps;
std::atomic<std::uint64_t> g_maps_generation{1};
std::atomic<int> g_scans{0}, g_header_reads{0};
std::atomic<bool> g_hold_scan{false};
int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

void make_table(std::array<std::uint8_t, 512>& table, float distance, float height, float fov) {
    table.fill(0);
    const std::uint16_t rows = 2;
    std::memcpy(table.data() + 0xa, &rows, 2);
    const char type[] = "LOCK_CAM_PARAM_ST";
    std::memcpy(table.data() + 0xc, type, sizeof(type));
    for (std::uint64_t r = 0; r < 2; ++r) {
        const std::uint64_t offset = 0x80 + r * 0x20;
        std::memcpy(table.data() + 0x40 + r * 0x18 + 8, &offset, 8);
        std::memcpy(table.data() + offset, &distance, 4);
        std::memcpy(table.data() + offset + 0xc, &height, 4);
        std::memcpy(table.data() + offset + 0x14, &fov, 4);
    }
}

float field(const std::array<std::uint8_t, 512>& table, int off) {
    float v = 0;
    std::memcpy(&v, table.data() + 0x80 + off, 4);
    return v;
}

void map_table(std::array<std::uint8_t, 512>& table) {
    std::lock_guard<std::mutex> lock(g_maps_mu);
    g_maps = {{reinterpret_cast<std::uint64_t>(table.data()), table.size(), -1, false, 3}};
    ++g_maps_generation;
}

bool wait_for(std::array<std::uint8_t, 512>& table, float value) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
        camera_tick();
        if (field(table, 0) == value) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}
}

HostSettings host_settings() { return g_settings.read(); }
std::uint64_t host_opt_serial() { return g_settings.serial(); }
std::uint64_t hle_kernel_maps_generation() { return g_maps_generation.load(); }
void hle_kernel_snapshot_maps(std::vector<GuestMapInfo>& maps) {
    {
        std::lock_guard<std::mutex> lock(g_maps_mu);
        maps = g_maps;
    }
    ++g_scans;
    while (g_hold_scan.load()) std::this_thread::yield();
}
void hle_watch_arm_reads(std::uint64_t, std::uint64_t) {}
void hle_watch_report() {}

const void* host_memmem(const void* data, std::size_t size, const void* needle, std::size_t n) {
    const auto* first = static_cast<const std::uint8_t*>(data);
    const auto* key = static_cast<const std::uint8_t*>(needle);
    const auto* found = std::search(first, first + size, key, key + n);
    return found == first + size ? nullptr : found;
}

bool host_read_safe(const void* addr, void* out, std::size_t size) {
    ++g_header_reads;
    const auto at = reinterpret_cast<std::uint64_t>(addr);
    std::lock_guard<std::mutex> lock(g_maps_mu);
    for (const auto& map : g_maps) {
        if (at >= map.va && at + size <= map.va + map.len) {
            std::memcpy(out, addr, size);
            return true;
        }
    }
    return false;
}

int main() {
    unsetenv("BBHOST_FOV_WATCH");
    for (int i = 0; i < 20; ++i) camera_tick();
    CHECK(g_scans.load() == 0);  // defaults do not discover or touch parameters

    // A setting changed before parameters exist. An empty discovery must
    // wait for a lifecycle signal instead of scanning every scene view.
    HostSettings h;
    h.camera_distance_scale = 2;
    h.camera_height_scale = 1.5f;
    h.fov_scale = 1.25f;
    g_settings.publish(h);
    camera_tick();
    while (g_scans.load() == 0) std::this_thread::yield();
    for (int i = 0; i < 20; ++i) {
        camera_tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(g_scans.load() == 1);

    // The table arrives during an empty discovery. Finishing that old scan
    // must not consume the new mapping generation and suppress its retry.
    g_hold_scan.store(true);
    camera_refresh();
    camera_tick();
    while (g_scans.load() < 2) std::this_thread::yield();
    make_table(g_table, 4, 2, 48);
    map_table(g_table);
    g_hold_scan.store(false);
    CHECK(wait_for(g_table, 8));
    CHECK(field(g_table, 0xc) == 3);
    CHECK(field(g_table, 0x14) == 60);
    const int reads = g_header_reads.load(), scans = g_scans.load();
    for (int i = 0; i < 500; ++i) camera_tick();
    CHECK(g_header_reads.load() == reads);
    CHECK(g_scans.load() == scans);
    CHECK(field(g_table, 0) == 8);  // no compounding

    // Unrelated settings and mapping churn do not walk camera rows.
    h.fps_counter = true;
    g_settings.publish(h);
    camera_tick();
    CHECK(g_header_reads.load() == reads);
    const float edited = 5;
    std::memcpy(g_table.data() + 0x80, &edited, 4);
    ++g_maps_generation;
    camera_tick();
    CHECK(field(g_table, 0) == 5);
    camera_refresh();  // an in-place reload / existing maintenance pass
    camera_tick();
    CHECK(field(g_table, 0) == 10);
    camera_tick();
    CHECK(field(g_table, 0) == 10);

    // Restoring defaults on a different view thread must restore the table,
    // even if that thread has never applied a camera override itself.
    g_settings.publish(HostSettings{});
    std::thread restore([] { camera_tick(); });
    restore.join();
    CHECK(field(g_table, 0) == 5);
    CHECK(field(g_table, 0xc) == 2);
    CHECK(field(g_table, 0x14) == 48);
    camera_tick();
    CHECK(field(g_table, 0) == 5);

    // Rediscover after the old address stops being readable.
    g_settings.publish(h);
    camera_tick();
    CHECK(field(g_table, 0) == 10);
    make_table(g_replacement, 6, 2, 48);
    map_table(g_replacement);
    CHECK(wait_for(g_replacement, 12));
    CHECK(field(g_replacement, 0xc) == 3);
    CHECK(field(g_replacement, 0x14) == 60);
    // No asynchronous discovery remains in flight after successful apply.
    return g_fail ? 1 : 0;
}
