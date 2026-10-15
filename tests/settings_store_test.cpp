// The settings object's publication (host/settings_store.h): a change is a
// new snapshot, never edited after, told once to every listener; readers take
// the newest without a lock. And a consumer built on it, the shader patches'
// mask (host/shader_patch.cpp), follows a change at once without asking.
#include "host/settings_store.h"
#include "host/shader_patch.h"

#include <atomic>
#include <cstdio>
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

int g_calls_a = 0, g_calls_b = 0, g_order = 0, g_a_at = 0, g_b_at = 0;
std::uint64_t g_last_serial = 0;
void listener_a(const HostSettings& h) {
    ++g_calls_a;
    g_a_at = ++g_order;
    g_last_serial = h.serial;
}
void listener_b(const HostSettings&) {
    ++g_calls_b;
    g_b_at = ++g_order;
}
void listener_nop(const HostSettings&) {}

std::atomic<std::uint64_t> g_seen{0};
std::atomic<int> g_out_of_order{0};
void listener_count(const HostSettings& h) {
    const std::uint64_t was = g_seen.exchange(h.serial);
    if (h.serial != was + 1 && was != 0) g_out_of_order.fetch_add(1);
}
}  // namespace

// What the shader patches link against (host/settings.h), served by this store.
void host_settings_listen(HostSettingsListener fn) { store().listen(fn); }

int main() {
    // The first snapshot: the defaults, numbered 1.
    CHECK(store().now().serial == 1);
    CHECK(store().now().motion_blur && store().now().depth_of_field);

    // A listener hears the settings at once, then every change once, in the
    // order listeners registered.
    CHECK(store().listen(&listener_a));
    CHECK(g_calls_a == 1 && g_last_serial == 1);
    CHECK(store().listen(&listener_b));
    CHECK(g_calls_b == 1);
    const HostSettings& before = store().now();
    HostSettings next;
    next.mouse_sens = 9;
    next.fov_scale = 1.25f;
    store().publish(next);
    CHECK(g_calls_a == 2 && g_calls_b == 2 && g_a_at < g_b_at);
    CHECK(g_last_serial == 2 && store().now().serial == 2);
    CHECK(store().now().mouse_sens == 9 && store().now().fov_scale == 1.25f);
    // A snapshot already handed out is never edited: the reference still
    // reads the settings as they were.
    CHECK(before.serial == 1 && before.mouse_sens == 5 && before.fov_scale == 1.0f);

    // The shader patches' mask: listening from the first ask, and following
    // a change made after that at once - nothing polls for it.
    CHECK(shader_patch_mask() == 0);
    HostSettings blur_off = store().now();
    blur_off.motion_blur = false;
    store().publish(blur_off);
    CHECK(shader_patch_mask() == 1);
    blur_off.depth_of_field = false;
    store().publish(blur_off);
    CHECK(shader_patch_mask() == 3);
    store().publish(HostSettings{});
    CHECK(shader_patch_mask() == 0);

    // A full table refuses the next listener rather than dropping one.
    int added = 0;
    while (store().listen(&listener_nop)) ++added;
    CHECK(added > 0 && added < HostSettingsStore::kMaxListeners);
    CHECK(!store().listen(&listener_nop));

    // Readers on other threads while a writer publishes: every snapshot a
    // reader sees is whole (fields published together read together), the
    // numbers never go back, and the last one is seen once the writer is done.
    HostSettingsStore& s2 = *new HostSettingsStore;
    s2.listen(&listener_count);
    std::atomic<bool> start{false}, done{false};
    std::atomic<int> ready{0}, torn{0}, back{0};
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&] {
            ++ready;
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            std::uint64_t seen = 0;
            do {
                const HostSettings& h = s2.now();
                if (h.serial > 1 && (h.fov_scale != static_cast<float>(h.mouse_sens) ||
                                     h.camera_distance_scale != 2.0f * static_cast<float>(h.mouse_sens) ||
                                     h.motion_blur != h.depth_of_field))
                    ++torn;
                if (h.serial < seen) ++back;
                seen = h.serial;
            } while (!done.load(std::memory_order_acquire));
            if (s2.now().mouse_sens != 20000) ++torn;
        });
    }
    while (ready.load() != 4) std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (int i = 1; i <= 20000; ++i) {
        HostSettings h;
        h.mouse_sens = i;
        h.fov_scale = static_cast<float>(i);
        h.camera_distance_scale = 2.0f * static_cast<float>(i);
        h.motion_blur = h.depth_of_field = (i % 2 == 0);
        s2.publish(h);
    }
    done.store(true, std::memory_order_release);
    for (auto& t : readers) t.join();
    CHECK(torn.load() == 0);
    CHECK(back.load() == 0);
    CHECK(g_out_of_order.load() == 0 && g_seen.load() == s2.now().serial);
    CHECK(s2.now().serial == 20001 && s2.snapshots() == 20001);
    if (g_fail) return 1;
    std::printf("settings_store_test: ok\n");
    return 0;
}
