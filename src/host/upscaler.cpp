#include "host/upscaler.h"

#include "bbhost_version.h"
#include "core/config.h"
#include "host/settings.h"
#include "log.h"

#if defined(_WIN32)
#include <windows.h>

#include "upscaler_bridge.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <atomic>
#include <mutex>
#include <string>

namespace {

const ub_api* g_ub = nullptr;
ub_context* g_ctx = nullptr;
bool g_attached = false;
std::mutex g_caps_mu;
UpscalerCaps g_caps;  // under g_caps_mu

void UB_CALL bridge_log(void*, int32_t level, const char* line) {
    host_log("upscaler: %s%s", level == UB_LOG_ERROR ? "error: " : level == UB_LOG_WARNING ? "warning: " : "", line);
}

}  // namespace

bool upscaler_open() {
    if (const char* e = std::getenv("BBHOST_UPSCALER"); e && e[0] == '0') {
        host_log("upscaler: off (BBHOST_UPSCALER=0)");
        return false;
    }
    const std::filesystem::path dll = std::filesystem::path(config_exe_dir()) / "upscaler_bridge.dll";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(dll, ec)) {
        host_log("upscaler: no upscaler_bridge.dll beside bbhost.exe: DLSS and FSR 3/4 are not available");
        return false;
    }
    // Its own folder first for what it loads (the vendors' files beside it).
    HMODULE h = LoadLibraryExW(dll.wstring().c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!h) {
        host_log("upscaler: %s did not load (error %lu)", dll.string().c_str(), GetLastError());
        return false;
    }
    auto get_api = reinterpret_cast<PFN_ub_get_api>(reinterpret_cast<void*>(GetProcAddress(h, "ub_get_api")));
    const ub_api* api = get_api ? get_api(UB_API_VERSION) : nullptr;
    if (!api) {
        host_log("upscaler: %s is not a bridge of API %u", dll.string().c_str(), UB_API_VERSION);
        return false;
    }
    const std::string folder = dll.parent_path().string();
    ub_open_desc od{};
    od.struct_size = sizeof(od);
    od.vendor_folder = folder.c_str();
    od.engine_version = BBHOST_VERSION;
    od.log = &bridge_log;
    if (api->open_context(&od, &g_ctx) != UB_OK) {
        host_log("upscaler: the bridge did not open: %s", api->last_error(nullptr));
        return false;
    }
    g_ub = api;
    host_log("upscaler: %s loaded", dll.string().c_str());
    return true;
}

const VkInstanceCreateInfo* upscaler_prepare_instance(const VkInstanceCreateInfo* in) {
    if (!g_ub) return in;
    const VkInstanceCreateInfo* out = in;
    g_ub->prepare_instance(g_ctx, vkGetInstanceProcAddr, in, &out);
    return out ? out : in;
}

void upscaler_instance_done(VkResult r) {
    if (g_ub) g_ub->instance_done(g_ctx, r);
}

bool upscaler_instance_retry(const VkInstanceCreateInfo* in, const VkInstanceCreateInfo* used) { return g_ub && used != in; }

const VkDeviceCreateInfo* upscaler_prepare_device(VkInstance instance, VkPhysicalDevice phys, const VkDeviceCreateInfo* in) {
    if (!g_ub) return in;
    const VkDeviceCreateInfo* out = in;
    g_ub->prepare_device(g_ctx, instance, phys, in, &out);
    return out ? out : in;
}

void upscaler_device_done(VkResult r) {
    if (g_ub) g_ub->device_done(g_ctx, r);
}

void upscaler_attach(VkInstance instance, VkPhysicalDevice phys, VkDevice device, std::uint32_t frames_in_flight) {
    if (!g_ub) return;
    ub_device_desc dd{};
    dd.struct_size = sizeof(dd);
    dd.frames_in_flight = frames_in_flight;
    dd.instance = instance;
    dd.physical_device = phys;
    dd.device = device;
    dd.get_instance_proc_addr = vkGetInstanceProcAddr;
    dd.get_device_proc_addr = vkGetDeviceProcAddr;
    if (g_ub->attach_device(g_ctx, &dd) != UB_OK) {
        host_log("upscaler: attach failed: %s", g_ub->last_error(g_ctx));
        return;
    }
    g_attached = true;
    ub_caps caps{};
    caps.struct_size = sizeof(caps);
    if (g_ub->get_caps(g_ctx, &caps) != UB_OK) return;
    UpscalerCaps c;
    c.loaded = true;
    c.dlss = caps.dlss_available != 0;
    c.fsr = caps.fsr_available != 0;
    c.fsr4 = caps.fsr4_available != 0;
    c.dlss_version = caps.dlss_version;
    c.fsr_versions = caps.fsr_versions;
    c.dlss_reason = caps.dlss_reason;
    c.fsr_reason = caps.fsr_reason;
    host_log("upscaler: bridge %s; DLSS %s%s%s; FSR %s%s%s%s", caps.bridge_version, c.dlss ? "available " : "not available",
             c.dlss ? caps.dlss_version : "", c.dlss || !caps.dlss_reason[0] ? "" : (std::string(" (") + caps.dlss_reason + ")").c_str(),
             c.fsr ? "available: " : "not available", c.fsr ? caps.fsr_versions : "", c.fsr4 ? " (FSR 4 here)" : "",
             c.fsr || !caps.fsr_reason[0] ? "" : (std::string(" (") + caps.fsr_reason + ")").c_str());
    std::lock_guard<std::mutex> lk(g_caps_mu);
    g_caps = c;
}

UpscalerCaps upscaler_caps() {
    std::lock_guard<std::mutex> lk(g_caps_mu);
    return g_caps;
}

namespace {
std::atomic<bool> g_failed{false};  // an evaluate failed: the plain picture from then on
}  // namespace

int upscaler_method() {
    static const int env = [] {
        const char* e = std::getenv("BBHOST_UPSCALE");
        if (!e || !*e) return -1;
        const std::string v = e;
        return v == "dlss" ? 1 : v == "fsr" || v == "fsr3" ? 2 : 0;
    }();
    const int setting = host_settings_ref().upscaler;
    const int asked = env >= 0 ? env : setting == 3 ? 2 : setting;
    if (!asked || !g_attached || g_failed.load(std::memory_order_relaxed)) return 0;
    const UpscalerCaps c = upscaler_caps();
    return (asked == 1 && c.dlss) || (asked == 2 && c.fsr) ? asked : 0;
}

void upscaler_jitter(int method, std::uint32_t quality, std::uint32_t output_w, std::uint32_t output_h, std::uint32_t frame, float* x,
                     float* y) {
    *x = *y = 0.0f;
    if (!g_ub || !method) return;
    ub_render_info ri{};
    ri.struct_size = sizeof(ri);
    if (g_ub->get_render_info(g_ctx, method == 1 ? UB_METHOD_DLSS : UB_METHOD_FSR, quality, output_w, output_h, &ri) != UB_OK ||
        !ri.jitter_phases)
        return;
    g_ub->get_jitter(frame % ri.jitter_phases, ri.jitter_phases, x, y);
}

bool upscaler_fsr3_override() {
    const char* e = std::getenv("BBHOST_UPSCALE");
    if (e && *e) return std::strcmp(e, "fsr3") == 0;
    return host_settings_ref().upscaler == 3;
}

bool upscaler_evaluate(VkCommandBuffer cb, const UpscaleCall& c) {
    if (!g_ub || !g_attached || g_failed.load(std::memory_order_relaxed)) return false;
    auto img = [](const UpscaleImage& i) {
        ub_image u{};
        u.image = i.image;
        u.view = i.view;
        u.format = i.format;
        u.layout = i.layout;
        return u;
    };
    ub_evaluate_desc d{};
    d.struct_size = sizeof(d);
    d.method = c.method == 1 ? UB_METHOD_DLSS : UB_METHOD_FSR;
    d.quality = c.quality;
    d.flags = (c.reset ? UB_EVAL_RESET : 0u) | (c.method == 2 && upscaler_fsr3_override() ? UB_EVAL_FSR3 : 0u);
    d.render_width = c.render_w;
    d.render_height = c.render_h;
    d.output_width = c.output_w;
    d.output_height = c.output_h;
    d.color = img(c.color);
    d.depth = img(c.depth);
    d.motion = img(c.motion);
    d.output = img(c.output);
    d.jitter_x = c.jitter_x;
    d.jitter_y = c.jitter_y;
    d.motion_scale_x = d.motion_scale_y = c.motion_scale;
    d.frame_time_ms = c.frame_ms;
    d.camera_near = c.near_m;
    d.camera_far = c.far_m;
    d.camera_fov_y = c.fov_y;
    if (g_ub->evaluate(g_ctx, cb, &d) == UB_OK) return true;
    g_failed.store(true, std::memory_order_relaxed);
    host_log("upscaler: evaluate failed, the plain picture from now on: %s", g_ub->last_error(g_ctx));
    return false;
}

#else  // no bridge outside Windows

bool upscaler_open() { return false; }
const VkInstanceCreateInfo* upscaler_prepare_instance(const VkInstanceCreateInfo* in) { return in; }
void upscaler_instance_done(VkResult) {}
bool upscaler_instance_retry(const VkInstanceCreateInfo*, const VkInstanceCreateInfo*) { return false; }
const VkDeviceCreateInfo* upscaler_prepare_device(VkInstance, VkPhysicalDevice, const VkDeviceCreateInfo* in) { return in; }
void upscaler_device_done(VkResult) {}
void upscaler_attach(VkInstance, VkPhysicalDevice, VkDevice, std::uint32_t) {}
UpscalerCaps upscaler_caps() { return {}; }
int upscaler_method() { return 0; }
bool upscaler_fsr3_override() { return false; }
bool upscaler_evaluate(VkCommandBuffer, const UpscaleCall&) { return false; }
void upscaler_jitter(int, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, float* x, float* y) { *x = *y = 0.0f; }

#endif
