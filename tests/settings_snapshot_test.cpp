#include "host/settings_snapshot.h"
#include "host/shader_patch.h"

#include <atomic>
#include <cstdio>
#include <new>
#include <thread>
#include <vector>

namespace {
HostSettingsSnapshot g_settings;
int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
}

// The shader cache reads the same snapshot/generation API as the host.
HostSettings host_settings() { return g_settings.read(); }
std::uint64_t host_opt_serial() { return g_settings.serial(); }

int main() {
    const auto first = g_settings.read();
    CHECK(first.serial == g_settings.serial());
    CHECK(first.motion_blur && first.depth_of_field);
    HostSettings next;
    next.mouse_sens = 9;
    next.motion_blur = next.depth_of_field = false;
    g_settings.publish(next);
    const auto changed = g_settings.read();
    CHECK(changed.serial > first.serial);
    CHECK(changed.serial == g_settings.serial());
    CHECK(changed.mouse_sens == 9);
    CHECK(shader_patch_mask() == 3);
    for (int i = 0; i < 100; ++i) CHECK(g_settings.read().serial == changed.serial);

    // A cache belongs to a store and its lifetime, even at a reused address.
    HostSettingsSnapshot other;
    CHECK(other.read().mouse_sens == 5);
    CHECK(g_settings.read().mouse_sens == 9);
    alignas(HostSettingsSnapshot) unsigned char storage[sizeof(HostSettingsSnapshot)];
    auto* reused = new (storage) HostSettingsSnapshot;
    reused->publish(next);
    CHECK(reused->read().mouse_sens == 9);
    const auto old_serial = reused->serial();
    reused->~HostSettingsSnapshot();
    reused = new (storage) HostSettingsSnapshot;
    CHECK(reused->serial() != old_serial);
    CHECK(reused->read().mouse_sens == 5);
    reused->~HostSettingsSnapshot();

    // Concurrent publication must never expose a partially updated tuple or
    // a shader mask assembled from different snapshots. All readers must
    // observe the last publication once the writer has finished.
    HostSettings initial;
    initial.mouse_sens = 0;
    initial.fov_scale = 0;
    initial.camera_distance_scale = 0;
    g_settings.publish(initial);
    std::atomic<bool> start{false}, done{false};
    std::atomic<int> ready{0};
    std::atomic<int> errors{0};
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&] {
            ++ready;
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            std::uint64_t seen = 0;
            do {
                const auto published = host_opt_serial();
                const auto h = host_settings();
                if (h.serial < published || h.serial < seen || h.fov_scale != h.mouse_sens ||
                    h.camera_distance_scale != 2.0f * h.mouse_sens || h.motion_blur != h.depth_of_field) ++errors;
                seen = h.serial;
                const auto mask = shader_patch_mask();
                if (mask != 0 && mask != 3) ++errors;
            } while (!done.load(std::memory_order_acquire));
            if (host_settings().mouse_sens != 2000 || shader_patch_mask() != 0) ++errors;
        });
    }
    while (ready.load() != 4) std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (int i = 1; i <= 2000; ++i) {
        HostSettings h;
        h.mouse_sens = i;
        h.fov_scale = static_cast<float>(i);
        h.camera_distance_scale = 2.0f * i;
        h.motion_blur = h.depth_of_field = (i % 2 == 0);
        g_settings.publish(h);
    }
    done.store(true, std::memory_order_release);
    for (auto& t : readers) t.join();
    CHECK(errors.load() == 0);
    return g_fail ? 1 : 0;
}
