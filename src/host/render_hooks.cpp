#include "host/render_hooks.h"

#include "hle/modules.h"
#include "host/render_api.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <mutex>

namespace {

// What the draws read (render_hooks_on, render_hooks_want_motion): frame_info
// asked for; a hook of a plugin that is on; one of them asking for the motion.
constexpr std::uint32_t kInfoWanted = 1u, kHooks = 2u, kMotion = 4u;
std::atomic<std::uint32_t> g_state{0};

constexpr std::uint32_t kPoints = BB_RENDER_BEFORE_UPSCALE | BB_RENDER_AFTER_UPSCALE;

// The hooks plugins registered (on_render), in the order they were.
struct Entry {
    std::string name;
    std::uint32_t points = 0;
    BbRenderFn fn = nullptr;
    void* user = nullptr;
    const std::string* profile[2] = {};  // the GPU profile's names, before and after
};
std::mutex g_reg_mu;
std::vector<Entry> g_entries;            // under g_reg_mu
std::vector<std::string> g_off;          // the plugins that are off (live switch, visitor), under g_reg_mu
std::deque<std::string> g_profile_names;  // under g_reg_mu; kept for the run: a slot's profile names them until it retires

bool plugin_on_locked(const std::string& name) { return std::find(g_off.begin(), g_off.end(), name) == g_off.end(); }

// The state from the entries (under g_reg_mu); true when the motion's bit changed.
bool restate_locked() {
    std::uint32_t bits = 0;
    for (const Entry& e : g_entries) {
        if (!(e.points & kPoints) || !plugin_on_locked(e.name)) continue;
        bits |= kHooks;
        if (e.points & BB_RENDER_MOTION) bits |= kMotion;
    }
    std::uint32_t was = g_state.load(std::memory_order_relaxed);
    while (!g_state.compare_exchange_weak(was, (was & kInfoWanted) | bits, std::memory_order_relaxed)) {
    }
    return ((was ^ bits) & kMotion) != 0;
}

const char* points_name(std::uint32_t points) {
    const std::uint32_t p = points & kPoints;
    return p == kPoints ? "before and after the upscale" : p == BB_RENDER_BEFORE_UPSCALE ? "before the upscale" : "after the upscale";
}

// The last frame described, for frame_info from any thread.
std::mutex g_info_mu;
BbFrameInfo g_published{};  // under g_info_mu
bool g_have = false;        // under g_info_mu

// The call point's own (under g.mu): the frame being recorded, the frames so
// far and what the last one was, which the next one's prev and reset come
// from; and the images the hooks were handed.
BbFrameInfo g_now{};
std::uint64_t g_frames = 0;
render_api::FrameShape g_last_shape;
bool g_last_camera = false, g_last_reversed = false;
BbFrameCamera g_last_cam{};
std::chrono::steady_clock::time_point g_last_time;
render_api::ImageSlot g_slots[gpu::kSlotCount];
std::uint64_t g_serial = 0;

// Inside a hook call: the frame being recorded (render_hooks_enter).
thread_local const BbFrameInfo* t_call_frame = nullptr;

const char* method_name(std::uint32_t m) {
    return m == BB_UPSCALE_DLSS ? "DLSS" : m == BB_UPSCALE_FSR ? "FSR" : m == BB_UPSCALE_TAA ? "TAA" : "none";
}

}  // namespace

bool render_hooks_on() { return g_state.load(std::memory_order_relaxed) != 0; }
bool render_hooks_registered() { return (g_state.load(std::memory_order_relaxed) & kHooks) != 0; }
bool render_hooks_want_motion() { return (g_state.load(std::memory_order_relaxed) & kMotion) != 0; }
bool render_hooks_in_call() { return t_call_frame != nullptr; }

int render_frame_info(BbFrameInfo* out) {
    if (!out || out->size < 8) return 1;
    if (t_call_frame) {
        const std::uint32_t n = std::min<std::uint32_t>(out->size, sizeof(BbFrameInfo));
        std::memcpy(out, t_call_frame, n);
        out->size = n;
        return 0;
    }
    if (!(g_state.fetch_or(kInfoWanted, std::memory_order_relaxed) & kInfoWanted))
        host_log("plugin: frame info asked for: each 3D frame described from now on");
    std::lock_guard<std::mutex> lk(g_info_mu);
    if (!g_have) return 1;
    const std::uint32_t n = std::min<std::uint32_t>(out->size, sizeof(BbFrameInfo));
    std::memcpy(out, &g_published, n);
    out->size = n;
    return 0;
}

int render_hooks_add(const char* name, std::uint32_t points, BbRenderFn fn, void* user, bool* motion_changed) {
    if (motion_changed) *motion_changed = false;
    if (!name || !name[0] || !fn || (points & ~(kPoints | BB_RENDER_MOTION))) return 1;
    std::lock_guard<std::mutex> lk(g_reg_mu);
    auto it = std::find_if(g_entries.begin(), g_entries.end(), [&](const Entry& e) { return e.fn == fn && e.user == user; });
    if (!(points & kPoints)) {
        if (it == g_entries.end()) return 0;
        host_log("plugin: %s: a render hook taken away", it->name.c_str());
        g_entries.erase(it);
    } else if (it == g_entries.end()) {
        Entry e;
        e.name = name;
        e.points = points;
        e.fn = fn;
        e.user = user;
        g_profile_names.push_back("plugin " + e.name + " before");
        e.profile[0] = &g_profile_names.back();
        g_profile_names.push_back("plugin " + e.name + " after");
        e.profile[1] = &g_profile_names.back();
        host_log("plugin: %s: a render hook %s%s", name, points_name(points), points & BB_RENDER_MOTION ? ", with the motion" : "");
        g_entries.push_back(std::move(e));
    } else {
        it->points = points;
        host_log("plugin: %s: a render hook now %s%s", it->name.c_str(), points_name(points), points & BB_RENDER_MOTION ? ", with the motion" : "");
    }
    const bool changed = restate_locked();
    if (motion_changed) *motion_changed = changed;
    return 0;
}

bool render_hooks_plugin_on(const std::string& name, bool on) {
    std::lock_guard<std::mutex> lk(g_reg_mu);
    const auto it = std::find(g_off.begin(), g_off.end(), name);
    if (on && it != g_off.end()) g_off.erase(it);
    if (!on && it == g_off.end()) g_off.push_back(name);
    return restate_locked();
}

void render_hooks_drop(const std::string& name) {
    std::lock_guard<std::mutex> lk(g_reg_mu);
    const auto before = g_entries.size();
    g_entries.erase(std::remove_if(g_entries.begin(), g_entries.end(), [&](const Entry& e) { return e.name == name; }), g_entries.end());
    if (g_entries.size() != before) host_log("plugin: %s: its render hooks go with it", name.c_str());
    g_off.erase(std::remove(g_off.begin(), g_off.end(), name), g_off.end());
    restate_locked();
}

namespace gpu {

void render_frame_describe_locked(const RenderFrameDesc& d) {
    BbFrameInfo f{};
    f.size = sizeof(f);
    f.frame = ++g_frames;
    f.flip = hle_video_flip_count();
    f.render_width = d.render_w;
    f.render_height = d.render_h;
    f.output_width = d.output_w;
    f.output_height = d.output_h;
    f.method = d.method;
    f.quality = d.quality;
    f.jitter_x = d.jitter_x;
    f.jitter_y = d.jitter_y;
    const bool first = f.frame == 1;
    const auto now = std::chrono::steady_clock::now();
    const double seconds = first ? 0.0 : std::chrono::duration<double>(now - g_last_time).count();
    g_last_time = now;
    f.frame_seconds = static_cast<float>(seconds);
    bool reversed = g_last_reversed;
    if (d.camera) {
        render_api::ViewTerms t;
        t.tan_x = d.tan_x;
        t.tan_y = d.tan_y;
        t.a = d.a;
        t.r2w = d.r2w;
        std::memcpy(t.rows, d.rows, sizeof(t.rows));
        render_api::frame_camera(t, &f.now, &reversed);
        f.flags |= BB_FRAME_CAMERA;
    } else {
        f.now = g_last_cam;  // the last one known
    }
    f.prev = g_last_camera ? g_last_cam : f.now;
    if (g_last_camera) f.flags |= BB_FRAME_PREV;
    const bool cut = d.camera && g_last_camera && render_api::frame_cut(f.now.camera, g_last_cam.camera);
    if (cut) f.flags |= BB_FRAME_CUT;
    if (reversed) f.flags |= BB_FRAME_DEPTH_REVERSED;
    const render_api::FrameShape shape{d.render_w, d.render_h, d.output_w, d.output_h, d.method};
    f.reset = render_api::frame_reset(first, g_last_shape, shape, d.camera, g_last_camera, cut, seconds) ? 1u : 0u;
    // Said when the shape changes (the first frame, a Resolution, an upscaler).
    static int logs = 0;
    if ((first || !(shape == g_last_shape)) && logs < 16) {
        ++logs;
        host_log("plugin: frame info: frame %llu, %ux%u to %ux%u, method %s quality %u, camera %s", static_cast<unsigned long long>(f.frame),
                 f.render_width, f.render_height, f.output_width, f.output_height, method_name(f.method), f.quality,
                 d.camera ? "seen" : "not seen");
    }
    g_last_shape = shape;
    g_last_camera = d.camera;
    g_last_reversed = reversed;
    if (d.camera) g_last_cam = f.now;
    g_now = f;
    std::lock_guard<std::mutex> lk(g_info_mu);
    g_published = f;
    g_have = true;
}

const BbFrameInfo* render_frame_current_locked() { return &g_now; }

void render_hooks_take(std::uint32_t point, std::vector<RenderHook>* out) {
    out->clear();
    std::lock_guard<std::mutex> lk(g_reg_mu);
    for (const Entry& e : g_entries) {
        if (!(e.points & point) || !plugin_on_locked(e.name)) continue;
        out->push_back(RenderHook{e.fn, e.user, e.profile[point == BB_RENDER_AFTER_UPSCALE ? 1 : 0]});
    }
}

void render_hooks_enter(const BbFrameInfo* frame) { t_call_frame = frame; }
void render_hooks_leave() { t_call_frame = nullptr; }

std::uint64_t render_hooks_generation_locked(RenderImageSlot slot, std::uint64_t image, std::uint64_t view, std::uint32_t format,
                                             std::uint32_t width, std::uint32_t height) {
    return render_api::image_generation(g_slots[slot], image, view, format, width, height, g_serial);
}

void render_hooks_gone_locked(std::uint64_t handle) {
    for (render_api::ImageSlot& s : g_slots) render_api::image_gone(s, handle);
}

}  // namespace gpu
