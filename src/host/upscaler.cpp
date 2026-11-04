#include "host/upscaler.h"

#include "bbhost_version.h"
#include "core/config.h"
#include "log.h"

#if defined(_WIN32)
#include <windows.h>

#include "upscaler_bridge.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>

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

#else  // no bridge outside Windows

bool upscaler_open() { return false; }
const VkInstanceCreateInfo* upscaler_prepare_instance(const VkInstanceCreateInfo* in) { return in; }
void upscaler_instance_done(VkResult) {}
bool upscaler_instance_retry(const VkInstanceCreateInfo*, const VkInstanceCreateInfo*) { return false; }
const VkDeviceCreateInfo* upscaler_prepare_device(VkInstance, VkPhysicalDevice, const VkDeviceCreateInfo* in) { return in; }
void upscaler_device_done(VkResult) {}
void upscaler_attach(VkInstance, VkPhysicalDevice, VkDevice, std::uint32_t) {}
UpscalerCaps upscaler_caps() { return {}; }

#endif
