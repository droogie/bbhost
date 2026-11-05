// The 3D's look a plugin asks for (host/scene_look.h).
#include "host/scene_look.h"

#include "host/upscaler.h"

#include <atomic>
#include <cmath>

namespace {

std::atomic<std::uint32_t> g_snap{0};
std::atomic<bool> g_nearest{false};
std::atomic<bool> g_scale{false};
std::atomic<std::uint32_t> g_scale_lines{0};
std::atomic<bool> g_colour_8bit{false};
std::atomic<bool> g_grading_nearest{false};
std::atomic<void (*)()> g_targets_change{nullptr};

void targets_changed() {
    if (void (*fn)() = g_targets_change.load(std::memory_order_acquire)) fn();
}

}  // namespace

void scene_look_set_snap(std::uint32_t lines) { g_snap.store(lines, std::memory_order_relaxed); }
std::uint32_t scene_look_snap() { return g_snap.load(std::memory_order_relaxed); }

bool scene_look_snap_cells(std::uint32_t w, std::uint32_t h, float* cells_x, float* cells_y) {
    const std::uint32_t lines = scene_look_snap();
    if (!lines || !w || !h) return false;
    if (lines == kSnapPixels) {
        *cells_x = static_cast<float>(w);
        *cells_y = static_cast<float>(h);
    } else {
        // As many across as the 3D's shape gives, whole cells.
        *cells_y = static_cast<float>(lines);
        *cells_x = std::round(static_cast<float>(lines) * static_cast<float>(w) / static_cast<float>(h));
    }
    return true;
}

void scene_look_set_nearest(bool on) { g_nearest.store(on, std::memory_order_relaxed); }
bool scene_look_nearest() { return g_nearest.load(std::memory_order_relaxed); }

void scene_look_set_scale(bool nearest, std::uint32_t lines) {
    const bool was = g_scale.exchange(nearest, std::memory_order_acq_rel);
    const std::uint32_t was_lines = g_scale_lines.exchange(nearest ? lines : 0, std::memory_order_acq_rel);
    if (was != nearest || was_lines != (nearest ? lines : 0)) targets_changed();
}
bool scene_look_scale(std::uint32_t* lines) {
    // An upscaler chosen and there takes the 3D's size, the plugin's yields.
    const bool on = g_scale.load(std::memory_order_acquire) && !upscaler_method();
    if (lines) *lines = on ? g_scale_lines.load(std::memory_order_acquire) : 0;
    return on;
}

void scene_look_set_colour_8bit(bool on) {
    if (g_colour_8bit.exchange(on, std::memory_order_acq_rel) != on) targets_changed();
}
bool scene_look_colour_8bit() { return g_colour_8bit.load(std::memory_order_acquire); }

void scene_look_set_grading_nearest(bool on) { g_grading_nearest.store(on, std::memory_order_relaxed); }
bool scene_look_grading_nearest() { return g_grading_nearest.load(std::memory_order_relaxed); }

bool scene_look_camera() { return scene_look_snap() || scene_look_nearest() || scene_look_scale(); }

void scene_look_on_targets_change(void (*fn)()) { g_targets_change.store(fn, std::memory_order_release); }
