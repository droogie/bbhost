#include "host/upscaler.h"

#include "bbhost_version.h"
#include "core/config.h"
#include "host/settings.h"
#include "log.h"

#include <cmath>
#include <cstdlib>
#include <string>

namespace {

// The method asked for, run here or not: BBHOST_UPSCALE over Graphics >
// Upscaling (the settings' 0 off, 1 DLSS, 2 FSR, 3 FSR 3, 4 TAA).
int method_asked() {
    static const int env = [] {
        const char* e = std::getenv("BBHOST_UPSCALE");
        if (!e || !*e) return -1;
        const std::string v = e;
        return v == "dlss" ? 1 : v == "fsr" || v == "fsr3" ? 2 : v == "taa" ? kUpscaleTaa : 0;
    }();
    if (env >= 0) return env;
    const int setting = host_settings_ref().upscaler;
    return setting == 3 ? 2 : setting == 4 ? kUpscaleTaa : setting;
}

float halton(std::uint32_t i, std::uint32_t base) {
    float f = 1.0f, r = 0.0f;
    for (; i; i /= base) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(i % base);
    }
    return r;
}

// TAA's jitter: Halton (2, 3) over 8 phases, centred, never phase 0 (the
// centre itself); and its LOD bias, none (Numity1504's step 9: the negative
// bias is for upscaling only).
void taa_jitter(std::uint32_t frame, float* x, float* y) {
    const std::uint32_t phase = frame % 8 + 1;
    *x = halton(phase, 2) - 0.5f;
    *y = halton(phase, 3) - 0.5f;
}
constexpr float kTaaMipBias = 0.0f;

}  // namespace

#if defined(_WIN32) || defined(__linux__)
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "upscaler_bridge.h"

#include <algorithm>
#include <cmath>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace {

const ub_api* g_ub = nullptr;
ub_context* g_ctx = nullptr;
bool g_attached = false;
std::mutex g_caps_mu;
UpscalerCaps g_caps;  // under g_caps_mu
// Its two methods' availability, read at every draw (upscaler_method) without
// the lock or a copy of the caps' four strings: the copy's allocations and the
// lock's contention between the run parts were ~4% of the command processor.
std::atomic<bool> g_caps_dlss{false}, g_caps_fsr{false};

void UB_CALL bridge_log(void*, int32_t level, const char* line) {
    host_log("upscaler: %s%s", level == UB_LOG_ERROR ? "error: " : level == UB_LOG_WARNING ? "warning: " : "", line);
}

// upscaler_open has run: from then on the device's caps say what can run.
std::atomic<bool> g_tried{false};

#if !defined(_WIN32)
// The bridges bbhost loads on Linux, beside the executable, the first found:
// the upscaler bridge's own source built for Linux under the Windows DLL's
// name (upscaler_bridge.so), Numity1504's own under a library's name, should
// he make one, else bbhost's NGX bridge (src/ngx_bridge: DLSS through NVIDIA's
// NGX). The same API, so each takes the place of those after it.
constexpr const char* kNgxBridge = "libbbhost_ngx_bridge.so";
constexpr const char* kLinuxBridges[] = {"upscaler_bridge.so", "libupscaler_bridge.so", kNgxBridge};

bool bridge_beside_exe() {
    static const bool beside = [] {
        for (const char* name : kLinuxBridges) {
            std::error_code ec;
            if (std::filesystem::is_regular_file(std::filesystem::path(config_exe_dir()) / name, ec)) return true;
        }
        return false;
    }();
    return beside;
}
#endif

}  // namespace

bool upscaler_open() {
    g_tried.store(true, std::memory_order_release);
    if (const char* e = std::getenv("BBHOST_UPSCALER"); e && e[0] == '0') {
        host_log("upscaler: off (BBHOST_UPSCALER=0)");
        return false;
    }
#if defined(_WIN32)
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
#else
    // From the executable's own folder (/proc/self/exe's, never a library
    // path), RTLD_LOCAL: nothing of the bridge's reaches anything else.
    const std::string dir = config_exe_dir();
    void* h = nullptr;
    std::string path;
    bool seen = false;
    for (const char* name : kLinuxBridges) {
        const std::string p = dir + "/" + name;
        std::error_code ec;
        if (!std::filesystem::is_regular_file(p, ec)) continue;
        seen = true;
        if ((h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL))) {
            path = p;
            break;
        }
        host_log("upscaler: %s did not load: %s", p.c_str(), dlerror());
    }
    if (!h) {
        if (!seen) host_log("upscaler: no %s beside bbhost (%s): DLSS is not available", kNgxBridge, dir.c_str());
        return false;
    }
    auto get_api = reinterpret_cast<PFN_ub_get_api>(dlsym(h, "ub_get_api"));
    const ub_api* api = get_api ? get_api(UB_API_VERSION) : nullptr;
    if (!api) {
        host_log("upscaler: %s is not a bridge of API %u", path.c_str(), UB_API_VERSION);
        return false;
    }
    // NGX's data folder (its logs, when it logs) under the run's own settings
    // folder: BBHOST_CONFIG_DIR's in a test, never the player's then.
    const std::string data = config_user_dir() + "/ngx";
    ub_open_desc od{};
    od.struct_size = sizeof(od);
    od.vendor_folder = dir.c_str();
    od.data_folder = data.c_str();
    od.engine_version = BBHOST_VERSION;
    od.log = &bridge_log;
    if (api->open_context(&od, &g_ctx) != UB_OK) {
        host_log("upscaler: the bridge did not open: %s", api->last_error(nullptr));
        return false;
    }
    g_ub = api;
    host_log("upscaler: %s loaded", path.c_str());
    return true;
#endif
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
    g_caps_dlss.store(c.dlss, std::memory_order_relaxed);
    g_caps_fsr.store(c.fsr, std::memory_order_relaxed);
#if !defined(_WIN32)
    host_log("upscaler: the Upscaling list offers Off, TAA%s", c.dlss ? ", DLSS" : "");
#endif
}

UpscalerCaps upscaler_caps() {
    std::lock_guard<std::mutex> lk(g_caps_mu);
    return g_caps;
}

namespace {
std::atomic<bool> g_failed{false};  // an evaluate failed: the plain picture from then on
}  // namespace

// BBsuite's layer beside the exe (a vulkan-1.dll whose version resource's
// ProductName does not start with "Vulkan": not the loader). It runs its own
// TAA and jitter, so ours stays off with it - the bridge's UB_ERR_BLOCKED for
// TAA, which has no bridge to ask (Numity1504's step 9, "Only one AA").
#if defined(_WIN32)
bool bbsuite_layer_present() {
    static const bool present = [] {
        wchar_t exe[MAX_PATH];
        const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
        if (!n || n >= MAX_PATH) return false;
        const std::filesystem::path dll = std::filesystem::path(exe).parent_path() / L"vulkan-1.dll";
        std::error_code ec;
        if (!std::filesystem::exists(dll, ec)) return false;
        DWORD handle = 0;
        const DWORD size = GetFileVersionInfoSizeW(dll.c_str(), &handle);
        std::wstring product;
        if (size) {
            std::vector<unsigned char> info(size);
            struct Lang {
                WORD lang, page;
            }* langs = nullptr;
            UINT len = 0;
            if (GetFileVersionInfoW(dll.c_str(), 0, size, info.data()) &&
                VerQueryValueW(info.data(), L"\\VarFileInfo\\Translation", reinterpret_cast<void**>(&langs), &len) && len >= sizeof(Lang)) {
                wchar_t key[64];
                swprintf(key, 64, L"\\StringFileInfo\\%04x%04x\\ProductName", langs[0].lang, langs[0].page);
                wchar_t* name = nullptr;
                if (VerQueryValueW(info.data(), key, reinterpret_cast<void**>(&name), &len) && name && len) product = name;
            }
        }
        const bool other = product.rfind(L"Vulkan", 0) != 0;
        host_log("upscaler: a vulkan-1.dll beside bbhost.exe (%ls): %s", product.empty() ? L"no product name" : product.c_str(),
                 other ? "BBsuite's or another layer - TAA stays off (it runs its own)" : "the Vulkan loader");
        return other;
    }();
    return present;
}
#else
bool bbsuite_layer_present() { return false; }  // (BBsuite is a Windows layer)
#endif

bool upscaler_offered(int index) {
#if defined(_WIN32)
    return index >= 0 && index <= 4;
#else
    if (index == 0 || index == 1) return true;  // Off, TAA
    if (index != 4) return false;               // FSR 3, FSR 4: AMD's runtime is Windows' alone
    // DLSS: the device's answer, or before there is one (the setup window) a bridge beside bbhost.
    return g_tried.load(std::memory_order_acquire) ? g_caps_dlss.load(std::memory_order_relaxed) : bridge_beside_exe();
#endif
}

int upscaler_method() {
    // Chosen, it wins over a plugin's 3D at a size of its own (host/scene_look.h).
    // TAA is bbhost's own: it needs neither the bridge nor its caps - only no
    // BBsuite beside it.
    const int asked = method_asked();
    if (asked == kUpscaleTaa) return bbsuite_layer_present() ? 0 : kUpscaleTaa;
    if (!asked || !g_attached || g_failed.load(std::memory_order_relaxed)) return 0;
    return (asked == 1 && g_caps_dlss.load(std::memory_order_relaxed)) || (asked == 2 && g_caps_fsr.load(std::memory_order_relaxed)) ? asked : 0;
}

void upscaler_jitter(int method, std::uint32_t quality, std::uint32_t output_w, std::uint32_t output_h, std::uint32_t frame, float* x,
                     float* y) {
    *x = *y = 0.0f;
    if (method == kUpscaleTaa) return taa_jitter(frame, x, y);
    if (!g_ub || !method) return;
    ub_render_info ri{};
    ri.struct_size = sizeof(ri);
    if (g_ub->get_render_info(g_ctx, method == 1 ? UB_METHOD_DLSS : UB_METHOD_FSR, quality, output_w, output_h, &ri) != UB_OK ||
        !ri.jitter_phases)
        return;
    g_ub->get_jitter(frame % ri.jitter_phases, ri.jitter_phases, x, y);
}

float upscaler_mip_bias(int method, std::uint32_t quality, std::uint32_t output_w, std::uint32_t output_h) {
    if (method == kUpscaleTaa) return kTaaMipBias;
    if (!g_ub || !method) return 0.0f;
    ub_render_info ri{};
    ri.struct_size = sizeof(ri);
    if (g_ub->get_render_info(g_ctx, method == 1 ? UB_METHOD_DLSS : UB_METHOD_FSR, quality, output_w, output_h, &ri) != UB_OK) return 0.0f;
    return ri.mip_bias;
}

void upscaler_jitter_sized(int method, std::uint32_t render_w, std::uint32_t output_w, std::uint32_t frame, float* x, float* y) {
    *x = *y = 0.0f;
    if (method == kUpscaleTaa) return taa_jitter(frame, x, y);
    if (!g_ub || !method || !render_w || !output_w) return;
    const double r = static_cast<double>(output_w) / render_w;
    const auto phases = static_cast<std::uint32_t>(std::max(8.0, std::ceil(8.0 * r * r)));
    g_ub->get_jitter(frame % phases, phases, x, y);
}

float upscaler_mip_bias_sized(int method, std::uint32_t render_w, std::uint32_t output_w) {
    // TAA's own, and as the others below the output: by the ratio.
    if (method == kUpscaleTaa)
        return kTaaMipBias + (render_w && output_w ? static_cast<float>(std::log2(static_cast<double>(render_w) / output_w)) : 0.0f);
    if (!g_ub || !method || !render_w || !output_w) return 0.0f;
    return static_cast<float>(std::log2(static_cast<double>(render_w) / output_w) - 1.0);
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
    d.flags = (c.reset ? UB_EVAL_RESET : 0u) | (c.method == 2 && upscaler_fsr3_override() ? UB_EVAL_FSR3 : 0u) |
              (c.depth_inverted ? UB_EVAL_DEPTH_INVERTED : 0u);
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
    d.dlss_preset = c.dlss_preset;  // a 1.0.0 bridge (no such field) reads its struct only as far as its own
    if (g_ub->evaluate(g_ctx, cb, &d) == UB_OK) return true;
    g_failed.store(true, std::memory_order_relaxed);
    host_log("upscaler: evaluate failed, the plain picture from now on: %s", g_ub->last_error(g_ctx));
    return false;
}

#else  // no bridge outside Windows and Linux

bool upscaler_open() { return false; }
bool upscaler_offered(int index) { return index == 0 || index == 1; }
const VkInstanceCreateInfo* upscaler_prepare_instance(const VkInstanceCreateInfo* in) { return in; }
void upscaler_instance_done(VkResult) {}
bool upscaler_instance_retry(const VkInstanceCreateInfo*, const VkInstanceCreateInfo*) { return false; }
const VkDeviceCreateInfo* upscaler_prepare_device(VkInstance, VkPhysicalDevice, const VkDeviceCreateInfo* in) { return in; }
void upscaler_device_done(VkResult) {}
void upscaler_attach(VkInstance, VkPhysicalDevice, VkDevice, std::uint32_t) {}
UpscalerCaps upscaler_caps() { return {}; }
int upscaler_method() { return method_asked() == kUpscaleTaa ? kUpscaleTaa : 0; }
bool upscaler_fsr3_override() { return false; }
void upscaler_jitter_sized(int method, std::uint32_t, std::uint32_t, std::uint32_t frame, float* x, float* y) {
    *x = *y = 0.0f;
    if (method == kUpscaleTaa) taa_jitter(frame, x, y);
}
float upscaler_mip_bias_sized(int method, std::uint32_t render_w, std::uint32_t output_w) {
    if (method != kUpscaleTaa) return 0.0f;
    return kTaaMipBias + (render_w && output_w ? static_cast<float>(std::log2(static_cast<double>(render_w) / output_w)) : 0.0f);
}
bool upscaler_evaluate(VkCommandBuffer, const UpscaleCall&) { return false; }
void upscaler_jitter(int method, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t frame, float* x, float* y) {
    *x = *y = 0.0f;
    if (method == kUpscaleTaa) taa_jitter(frame, x, y);
}
float upscaler_mip_bias(int method, std::uint32_t, std::uint32_t, std::uint32_t) { return method == kUpscaleTaa ? kTaaMipBias : 0.0f; }

#endif
