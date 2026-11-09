#include "host/render_hooks.h"

#include "hle/modules.h"
#include "host/render_api.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>

namespace {

// What the draws read (render_hooks_on): bit 0, frame_info asked for.
constexpr std::uint32_t kInfoWanted = 1u;
std::atomic<std::uint32_t> g_state{0};

// The last frame described, for frame_info from any thread.
std::mutex g_info_mu;
BbFrameInfo g_published{};  // under g_info_mu
bool g_have = false;        // under g_info_mu

// The call point's own (under g.mu): the frames so far and what the last one
// was, which the next one's prev and reset come from.
std::uint64_t g_frames = 0;
render_api::FrameShape g_last_shape;
bool g_last_camera = false, g_last_reversed = false;
BbFrameCamera g_last_cam{};
std::chrono::steady_clock::time_point g_last_time;

const char* method_name(std::uint32_t m) {
    return m == BB_UPSCALE_DLSS ? "DLSS" : m == BB_UPSCALE_FSR ? "FSR" : m == BB_UPSCALE_TAA ? "TAA" : "none";
}

}  // namespace

bool render_hooks_on() { return g_state.load(std::memory_order_relaxed) != 0; }

int render_frame_info(BbFrameInfo* out) {
    if (!out || out->size < 8) return 1;
    if (!(g_state.fetch_or(kInfoWanted, std::memory_order_relaxed) & kInfoWanted))
        host_log("plugin: frame info asked for: each 3D frame described from now on");
    std::lock_guard<std::mutex> lk(g_info_mu);
    if (!g_have) return 1;
    const std::uint32_t n = std::min<std::uint32_t>(out->size, sizeof(BbFrameInfo));
    std::memcpy(out, &g_published, n);
    out->size = n;
    return 0;
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
    std::lock_guard<std::mutex> lk(g_info_mu);
    g_published = f;
    g_have = true;
}

}  // namespace gpu
