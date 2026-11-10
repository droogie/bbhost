// The 3D's look a plugin asks for (host/scene_look.h).
#include "host/scene_look.h"

#include "host/pixel_cutaway.h"
#include "host/upscaler.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>

namespace {

std::atomic<std::uint32_t> g_snap{0};
std::atomic<bool> g_nearest{false};
std::atomic<bool> g_scale{false};
std::atomic<std::uint32_t> g_scale_lines{0};
std::atomic<bool> g_colour_8bit{false};
std::atomic<bool> g_grading_nearest{false};
std::atomic<void (*)()> g_targets_change{nullptr};
std::atomic<void (*)()> g_snap_change{nullptr};
std::mutex g_cut_mu;
float g_cut[4] = {};  // under g_cut_mu
std::atomic<bool> g_cut_on{false};
PixelCut g_pixel_cut{};  // under g_cut_mu
std::atomic<bool> g_pixel_cut_on{false};

void targets_changed() {
    if (void (*fn)() = g_targets_change.load(std::memory_order_acquire)) fn();
}

}  // namespace

void scene_look_set_snap(std::uint32_t lines) {
    if (g_snap.exchange(lines, std::memory_order_relaxed) == lines) return;
    if (void (*fn)() = g_snap_change.load(std::memory_order_acquire)) fn();
}
void scene_look_on_snap_change(void (*fn)()) { g_snap_change.store(fn, std::memory_order_release); }
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

void scene_look_set_cutaway(float tan_x, float tan_y, float near_z, float radius) {
    const bool on = tan_x > 0.0f && tan_y > 0.0f && radius > 0.0f && near_z > 0.0f;
    std::lock_guard<std::mutex> lk(g_cut_mu);
    // Rounded: each new set of numbers is a new twin of every camera draw's
    // vertex stage, so a drift in the last bits must not make one.
    const auto round_to = [](float v, float step) { return std::round(v / step) * step; };
    g_cut[0] = on ? round_to(tan_x, 1.0f / 1024) : 0.0f;
    g_cut[1] = on ? round_to(tan_y, 1.0f / 1024) : 0.0f;
    g_cut[2] = on ? round_to(near_z, 0.25f) : 0.0f;
    g_cut[3] = on ? round_to(radius, 0.25f) : 0.0f;
    g_cut_on.store(on, std::memory_order_release);
}
bool scene_look_cutaway(float* out) {
    if (!g_cut_on.load(std::memory_order_acquire)) return false;
    if (out) {
        std::lock_guard<std::mutex> lk(g_cut_mu);
        std::memcpy(out, g_cut, sizeof(g_cut));
        return out[3] > 0.0f;
    }
    return true;
}

void scene_look_set_pixel_cut(const PixelCut& cut) {
    const bool on = cut.radius > 0.0f && cut.distance > 0.0f;
    std::lock_guard<std::mutex> lk(g_cut_mu);
    // As given: the twins read them live (render.cpp pixel_cut_live_write),
    // or, where they are baked in, round them there.
    g_pixel_cut = cut;
    g_pixel_cut_on.store(on, std::memory_order_release);
}
std::atomic<float> g_object_cut{-1.0f};
void scene_look_set_object_cut(float start) {
    // Read live by the twins (render.cpp pixel_cut_live_write); where the
    // numbers are baked in, the objects' distance is rounded there.
    g_object_cut.store(start >= 0.0f && start < 1.0f ? start : -1.0f, std::memory_order_relaxed);
}
float scene_look_object_cut() { return g_object_cut.load(std::memory_order_relaxed); }
std::atomic<std::uint32_t> g_dither{0};
void scene_look_set_dither(std::uint32_t bits) { g_dither.store(bits >= 1 && bits <= 8 ? bits : 0, std::memory_order_relaxed); }
std::uint32_t scene_look_dither() { return g_dither.load(std::memory_order_relaxed); }

bool scene_look_pixel_cut(PixelCut* out) {
    if (!g_pixel_cut_on.load(std::memory_order_acquire)) return false;
    if (out) {
        std::lock_guard<std::mutex> lk(g_cut_mu);
        *out = g_pixel_cut;
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
    // An upscaler chosen and there takes the 3D's size, the plugin's yields
    // (TAA does not: it runs at whatever size the 3D is).
    const bool on = g_scale.load(std::memory_order_acquire) && !upscaler_method_scales(upscaler_method());
    if (lines) *lines = on ? g_scale_lines.load(std::memory_order_acquire) : 0;
    return on;
}

void scene_look_set_colour_8bit(bool on) {
    if (g_colour_8bit.exchange(on, std::memory_order_acq_rel) != on) targets_changed();
}
bool scene_look_colour_8bit() { return g_colour_8bit.load(std::memory_order_acquire); }

void scene_look_set_grading_nearest(bool on) { g_grading_nearest.store(on, std::memory_order_relaxed); }
bool scene_look_grading_nearest() { return g_grading_nearest.load(std::memory_order_relaxed); }

bool scene_look_camera() {
    return scene_look_snap() || scene_look_nearest() || scene_look_scale() || scene_look_cutaway(nullptr) || scene_look_pixel_cut(nullptr);
}

void scene_look_on_targets_change(void (*fn)()) { g_targets_change.store(fn, std::memory_order_release); }
