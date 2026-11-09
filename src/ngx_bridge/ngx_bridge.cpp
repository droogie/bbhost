/*
 * ngx_bridge.cpp -- bbhost's NGX bridge: NVIDIA DLSS on Linux behind the C API of Numity1504's upscaler bridge.
 *
 * Copyright (c) 2026 the bbhost authors. MIT licence: see LICENSE in this folder.
 *
 * The API is Numity1504's (BBsuite): third_party/upscaler_bridge/upscaler_bridge.h, MIT - the function table
 * ub_get_api returns, the order of the calls and what each one promises. This file implements it for Linux, so
 * bbhost's host/upscaler.cpp runs one path on Windows (his upscaler_bridge.dll) and here (libbbhost_ngx_bridge.so); his
 * own Linux build, should there be one, can take this one's place unchanged.
 *
 * DLSS runs through NVIDIA's NGX SDK - its static library linked into this library and nowhere else - and NGX's
 * Vulkan API. NGX's core is part of NVIDIA's Linux driver; the DLSS runtime (libnvidia-ngx-dlss.so.<version>) lies in
 * the vendor folder, which is the feature path NGX is given. NGX is asked nothing until an NVIDIA GPU with NVIDIA's own
 * driver is in sight: on any other machine (an AMD card, lavapipe in the tests) the instance and the device are made
 * exactly as the host asks and DLSS reports why it cannot run. FSR is not here: on Linux bbhost runs it through its own build
 * of AMD's FidelityFX library (host/upscaler_ffx.h), and get_caps says so.
 *
 * Where it differs from the Windows bridge: no crash inside NVIDIA's code is caught (UB_ERR_CRASH is never returned -
 * the host's own fault handler sees it), and no vulkan-1.dll check (UB_ERR_BLOCKED). NGX's log lines come to the log
 * function when NGX logs (BBHOST_NGX_LOG=1 or 2, or NVIDIA's __NGX_LOG_LEVEL), and NGX writes its own log files into
 * the data folder then. BBHOST_NGX_JITTER=sx,sy scales the jitter handed to DLSS (-1,-1 negates it: a check of the sign).
 * BBHOST_NGX_OFF=1 keeps NGX out altogether (DLSS off, saying so): the tests' guard, which they also check is never
 * what stops DLSS on a GPU that is not NVIDIA's.
 */
#include "bridge_core.h"
#include "upscaler_bridge.h"

#include <vulkan/vulkan.h>

#include "nvsdk_ngx_helpers_vk.h"

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#ifndef NGX_BRIDGE_SDK
#define NGX_BRIDGE_SDK "unknown"
#endif

using namespace ngx_bridge;

namespace {

constexpr char kVersionNumber[] = "1.0.0";
constexpr char kVersion[] = "ngx-bridge 1.0.0";  // get_caps' bridge_version: which bridge ran, in the host's log
// NGX's project id when the host gives none (ub_open_desc.project_id): this bridge's own.
constexpr char kProjectId[] = "55df37c7-823f-4257-a144-58eed328a47f";
constexpr char kFsrReason[] = "FSR on Linux is bbhost's own FidelityFX library's (libbbhost_fidelityfx_vk.so), not this bridge's";
constexpr std::uint32_t kDefaultPreset = NVSDK_NGX_DLSS_Hint_Render_Preset_K;

struct Feature {
    NVSDK_NGX_Handle* handle = nullptr;
    std::uint32_t rw = 0, rh = 0, ow = 0, oh = 0, quality = 0, flags = 0, preset = 0;
};
struct Retired {
    NVSDK_NGX_Handle* handle;
    std::uint32_t evaluates_left;
};

}  // namespace

struct ub_context {
    // open_context
    std::uint32_t flags = 0;
    std::string vendor_folder, data_folder, project_id, engine_version;
    ub_log_fn log = nullptr;
    void* log_user = nullptr;
    std::string runtime_name, runtime_version;  // the DLSS runtime in the vendor folder
    float jitter_sx = 1.0f, jitter_sy = 1.0f;
    bool ngx_off = false;  // BBHOST_NGX_OFF=1
    bool data_made = false;  // the data folder made (before NGX's first call)
    NVSDK_NGX_Logging_Level ngx_log_level = NVSDK_NGX_LOGGING_LEVEL_OFF;
    std::string last_error;
    std::string reason;  // why DLSS cannot run; "" while it still may
    // NGX's common information: the feature path (the vendor folder), the data path and the log callback.
    std::wstring vendor_w, data_w;
    const wchar_t* paths[1] = {nullptr};
    NVSDK_NGX_FeatureCommonInfo common{};
    // prepare_instance / instance_done
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    bool instance_prepared = false, instance_pending = false, instance_copied = false, instance_has_ngx = false;
    InstanceCopy instance_copy;
    // prepare_device / device_done
    bool device_pending = false, device_planned = false, device_ready = false;
    VkPhysicalDevice prepared_phys = VK_NULL_HANDLE;
    DeviceCopy device_copy;
    // attach_device
    bool attached = false, ngx_started = false, dlss = false;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier = nullptr;
    std::uint32_t frames_in_flight = 8;
    NVSDK_NGX_Parameter* params = nullptr;
    Feature feature;
    std::vector<Retired> retired;
    // get_render_info: the last answer NGX gave
    std::uint32_t ri_quality = ~0u, ri_ow = 0, ri_oh = 0, ri_w = 0, ri_h = 0;
};

namespace {

std::mutex g_mu;            // one lock: every call takes it
ub_context* g_ctx = nullptr;  // one context a process
std::string g_open_error;   // why the last open_context failed
// NGX may log from any thread at any time: the log function it reaches, set while a context is open.
std::atomic<ub_log_fn> g_ngx_log{nullptr};
std::atomic<void*> g_ngx_log_user{nullptr};

void say(ub_context* c, std::int32_t level, const char* fmt, ...) __attribute__((format(printf, 3, 4)));
void say(ub_context* c, std::int32_t level, const char* fmt, ...) {
    if (!c || !c->log) return;
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    c->log(c->log_user, level, line);
}

void NVSDK_CONV ngx_log(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    const ub_log_fn fn = g_ngx_log.load(std::memory_order_acquire);
    if (!fn || !message) return;
    std::string m = message;
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r')) m.pop_back();
    if (m.empty()) return;
    fn(g_ngx_log_user.load(std::memory_order_acquire), UB_LOG_INFO, ("NGX: " + m).c_str());
}

// An error of c's (last_error) with the code it goes with.
ub_result fail(ub_context* c, ub_result r, const std::string& why) {
    if (c) c->last_error = why;
    return r;
}

// DLSS off, with why (the first reason stands).
void dlss_off(ub_context* c, const std::string& why) {
    if (c->reason.empty()) c->reason = why;
}

std::string result_text(NVSDK_NGX_Result r) {
    char hex[16];
    std::snprintf(hex, sizeof(hex), "0x%08x", static_cast<unsigned>(r));
    const std::string name = narrow(GetNGXResultAsString(r));
    return name.empty() ? hex : name + " " + hex;
}

bool make_dirs(const std::string& path) {
    std::string cur;
    for (std::size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty() && ::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
        if (i < path.size()) cur += path[i];
    }
    return true;
}

std::string own_folder() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&own_folder), &info) && info.dli_fname) {
        const std::string p = info.dli_fname;
        const std::size_t slash = p.find_last_of('/');
        if (slash != std::string::npos) return slash ? p.substr(0, slash) : "/";
    }
    return ".";
}

NVSDK_NGX_FeatureDiscoveryInfo discovery(ub_context* c) {
    NVSDK_NGX_FeatureDiscoveryInfo d{};
    d.SDKVersion = NVSDK_NGX_Version_API;
    d.FeatureID = NVSDK_NGX_Feature_SuperSampling;
    d.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    d.Identifier.v.ProjectDesc.ProjectId = c->project_id.c_str();
    d.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
    d.Identifier.v.ProjectDesc.EngineVersion = c->engine_version.c_str();
    d.ApplicationDataPath = c->data_w.c_str();
    d.FeatureInfo = &c->common;
    return d;
}

// BBHOST_NGX_OFF=1: NGX is never called (the reason says so). Otherwise NGX
// is about to be called: its data folder is made first, once - NGX may fail
// without a word on one it cannot write, and its discovery calls are given it
// too, before attach_device.
bool ngx_kept_out(ub_context* c) {
    if (c->ngx_off) {
        dlss_off(c, "NGX is kept out (BBHOST_NGX_OFF=1)");
        return true;
    }
    if (!c->data_made) {
        c->data_made = true;
        if (!make_dirs(c->data_folder)) say(c, UB_LOG_WARNING, "DLSS: NGX's data folder %s could not be made", c->data_folder.c_str());
    }
    return false;
}

// The machine has NVIDIA's own kernel driver: without it no NVIDIA GPU runs NGX.
bool nvidia_driver_loaded() { return ::access("/proc/driver/nvidia/version", F_OK) == 0; }

// Whether the instance the host is about to make sees an NVIDIA GPU run by NVIDIA's own driver: a throwaway instance
// with the host's application info and no layers or extensions, its devices listed (`seen`: their names). 1 yes, 0
// no, -1 no instance could be made.
int probe_nvidia(PFN_vkGetInstanceProcAddr gipa, const VkInstanceCreateInfo* in, std::string* seen) {
    auto create = reinterpret_cast<PFN_vkCreateInstance>(gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!create) return -1;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = in->pApplicationInfo ? in->pApplicationInfo : &app;
    VkInstance probe = VK_NULL_HANDLE;
    if (create(&ici, nullptr, &probe) != VK_SUCCESS || !probe) return -1;
    auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(gipa(probe, "vkDestroyInstance"));
    auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gipa(probe, "vkEnumeratePhysicalDevices"));
    auto props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(gipa(probe, "vkGetPhysicalDeviceProperties"));
    auto props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(gipa(probe, "vkGetPhysicalDeviceProperties2"));
    const std::uint32_t instance_api = ici.pApplicationInfo->apiVersion ? ici.pApplicationInfo->apiVersion : VK_API_VERSION_1_0;
    int found = 0;
    std::uint32_t n = 0;
    if (enumerate && props && enumerate(probe, &n, nullptr) == VK_SUCCESS && n) {
        std::vector<VkPhysicalDevice> devs(n);
        if (enumerate(probe, &n, devs.data()) >= 0) {
            devs.resize(n);
            for (VkPhysicalDevice d : devs) {
                VkPhysicalDeviceProperties p{};
                props(d, &p);
                VkDriverId driver = static_cast<VkDriverId>(0);
                if (props2 && p.apiVersion >= VK_API_VERSION_1_2 && instance_api >= VK_API_VERSION_1_1) {
                    VkPhysicalDeviceDriverProperties dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
                    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                    p2.pNext = &dp;
                    props2(d, &p2);
                    driver = dp.driverID;
                }
                if (!seen->empty()) *seen += ", ";
                *seen += p.deviceName;
                if (p.vendorID == kVendorNvidia && (!driver || driver == VK_DRIVER_ID_NVIDIA_PROPRIETARY)) found = 1;
            }
        }
    }
    if (seen->empty()) *seen = "no Vulkan devices";
    if (destroy) destroy(probe, nullptr);
    return found;
}

void release_feature(ub_context* c, NVSDK_NGX_Handle* h, const char* what) {
    if (!h) return;
    const NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_ReleaseFeature(h);
    if (NVSDK_NGX_FAILED(r)) say(c, UB_LOG_WARNING, "DLSS failed at %s (%s)", what, result_text(r).c_str());
}

void detach_locked(ub_context* c) {
    if (!c->attached) return;
    release_feature(c, c->feature.handle, "its release");
    c->feature = Feature{};
    for (const Retired& r : c->retired) release_feature(c, r.handle, "its release");
    c->retired.clear();
    if (c->params) NVSDK_NGX_VULKAN_DestroyParameters(c->params);
    c->params = nullptr;
    if (c->ngx_started) {
        const NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Shutdown1(c->device);
        if (NVSDK_NGX_FAILED(r)) say(c, UB_LOG_WARNING, "DLSS failed at its end (%s)", result_text(r).c_str());
    }
    c->ngx_started = false;
    c->dlss = false;
    c->attached = false;
    c->device_ready = false;
    c->device = VK_NULL_HANDLE;
    c->ri_quality = ~0u;
}

// --- the API ------------------------------------------------------------------------------------------------------

ub_result UB_CALL open_context(const ub_open_desc* desc, ub_context** out) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (out) *out = nullptr;
    if (!desc || !out || desc->struct_size < sizeof(ub_open_desc)) {
        g_open_error = "open_context: a NULL pointer or an old struct_size";
        return UB_ERR_ARGUMENT;
    }
    if (g_ctx) {
        g_open_error = "open_context: one context a process, and one is open";
        return UB_ERR_STATE;
    }
    try {
        auto* c = new ub_context;
        c->flags = desc->flags;
        c->vendor_folder = desc->vendor_folder && *desc->vendor_folder ? desc->vendor_folder : own_folder();
        if (desc->data_folder && *desc->data_folder) {
            c->data_folder = desc->data_folder;
        } else {
            const char* t = std::getenv("TMPDIR");
            c->data_folder = t && *t ? t : "/tmp";
        }
        c->project_id = desc->project_id && *desc->project_id ? desc->project_id : kProjectId;
        c->engine_version = desc->engine_version && *desc->engine_version ? desc->engine_version : "1.0";
        c->log = desc->log;
        c->log_user = desc->log_user;
        if (const char* e = std::getenv("BBHOST_NGX_LOG"); e && (e[0] == '1' || e[0] == '2'))
            c->ngx_log_level = e[0] == '2' ? NVSDK_NGX_LOGGING_LEVEL_VERBOSE : NVSDK_NGX_LOGGING_LEVEL_ON;
        else if (desc->flags & UB_OPEN_DEBUG_CHECKS)
            c->ngx_log_level = NVSDK_NGX_LOGGING_LEVEL_ON;
        if (const char* e = std::getenv("BBHOST_NGX_OFF"); e && e[0] == '1') c->ngx_off = true;
        if (const char* e = std::getenv("BBHOST_NGX_JITTER"); e && *e) {
            float sx = 1.0f, sy = 1.0f;
            if (std::sscanf(e, "%f,%f", &sx, &sy) == 2) {
                c->jitter_sx = sx;
                c->jitter_sy = sy;
            }
        }
        c->vendor_w = widen(c->vendor_folder);
        c->data_w = widen(c->data_folder);
        c->paths[0] = c->vendor_w.c_str();
        c->common.PathListInfo.Path = c->paths;
        c->common.PathListInfo.Length = 1;
        c->common.LoggingInfo.LoggingCallback = &ngx_log;
        c->common.LoggingInfo.MinimumLoggingLevel = c->ngx_log_level;
        c->common.LoggingInfo.DisableOtherLoggingSinks = false;
        if (!find_runtime(c->vendor_folder, &c->runtime_name, &c->runtime_version))
            c->reason = "libnvidia-ngx-dlss.so is not in the vendor folder (" + c->vendor_folder + ")";
        g_ngx_log_user.store(c->log_user, std::memory_order_release);
        g_ngx_log.store(c->log, std::memory_order_release);
        g_ctx = c;
        *out = c;
        say(c, UB_LOG_INFO, "bbhost's NGX bridge %s for Linux (Numity1504's upscaler bridge API %u): DLSS through NVIDIA's NGX SDK %s, %s%s",
            kVersionNumber, UB_API_VERSION, NGX_BRIDGE_SDK, c->runtime_name.empty() ? "no DLSS runtime in " : "the runtime ",
            c->runtime_name.empty() ? c->vendor_folder.c_str() : (c->vendor_folder + "/" + c->runtime_name).c_str());
        if (c->jitter_sx != 1.0f || c->jitter_sy != 1.0f)
            say(c, UB_LOG_WARNING, "DLSS: its jitter scaled by %.2f, %.2f (BBHOST_NGX_JITTER, a test)", static_cast<double>(c->jitter_sx),
                static_cast<double>(c->jitter_sy));
        return UB_OK;
    } catch (...) {
        g_open_error = "open_context: no memory";
        return UB_ERR_MEMORY;
    }
}

void UB_CALL close_context(ub_context* c) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!c || c != g_ctx) return;
    detach_locked(c);
    g_ngx_log.store(nullptr, std::memory_order_release);
    g_ngx_log_user.store(nullptr, std::memory_order_release);
    g_ctx = nullptr;
    delete c;
}

const char* UB_CALL last_error(ub_context* c) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!c || c != g_ctx) return g_open_error.c_str();
    return c->last_error.c_str();
}

ub_result UB_CALL prepare_instance(ub_context* c, PFN_vkGetInstanceProcAddr gipa, const VkInstanceCreateInfo* in,
                                   const VkInstanceCreateInfo** out) {
    if (out) *out = in;
    std::lock_guard<std::mutex> lk(g_mu);
    if (!c || c != g_ctx || !gipa || !in || !out) return fail(c == g_ctx ? c : nullptr, UB_ERR_ARGUMENT, "prepare_instance: a NULL pointer");
    if (c->instance_pending) return fail(c, UB_ERR_STATE, "prepare_instance: instance_done of the last one is missing");
    if (c->attached) return fail(c, UB_ERR_STATE, "prepare_instance: a device is attached");
    try {
        c->gipa = gipa;
        c->instance_prepared = c->instance_pending = true;
        c->instance_copied = c->instance_has_ngx = false;
        if (!c->reason.empty()) return UB_OK;  // DLSS is off already: the instance as the host asks
        if (!nvidia_driver_loaded()) {
            dlss_off(c, "DLSS needs an NVIDIA RTX card and NVIDIA's driver, which is not loaded");
            return UB_OK;
        }
        std::string seen;
        const int nv = probe_nvidia(gipa, in, &seen);
        if (nv <= 0) {
            dlss_off(c, nv < 0 ? std::string("DLSS needs an NVIDIA RTX card (the Vulkan devices could not be listed)")
                               : "DLSS needs an NVIDIA RTX card (" + seen + ")");
            return UB_OK;
        }
        if (ngx_kept_out(c)) return UB_OK;
        NVSDK_NGX_FeatureDiscoveryInfo d = discovery(c);
        std::uint32_t n = 0;
        VkExtensionProperties* props = nullptr;
        const NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&d, &n, &props);
        if (NVSDK_NGX_FAILED(r)) {
            const std::string why = "NGX could not name its instance extensions (" + result_text(r) + ")";
            say(c, UB_LOG_WARNING, "DLSS: %s", why.c_str());
            dlss_off(c, why);
            return fail(c, UB_ERR_VENDOR, why);
        }
        std::vector<std::string> need;
        for (std::uint32_t i = 0; props && i < n; ++i) need.emplace_back(props[i].extensionName);
        // What the loader offers (its own, the drivers' and the implicit layers').
        std::vector<std::string> have;
        if (auto list = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(gipa(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"))) {
            std::uint32_t m = 0;
            if (list(nullptr, &m, nullptr) == VK_SUCCESS && m) {
                std::vector<VkExtensionProperties> e(m);
                if (list(nullptr, &m, e.data()) >= 0)
                    for (std::uint32_t i = 0; i < m; ++i) have.emplace_back(e[i].extensionName);
            }
        }
        std::vector<std::string> lacking;
        for (const std::string& e : need)
            if (!contains(have, e)) lacking.push_back(e);
        if (!lacking.empty()) {
            const std::string why = "the Vulkan loader lacks " + join(lacking);
            say(c, UB_LOG_WARNING, "DLSS: %s", why.c_str());
            dlss_off(c, why);
            return UB_OK;
        }
        if (copy_instance(*in, need, &c->instance_copy)) {
            c->instance_copied = true;
            *out = &c->instance_copy.info;
            say(c, UB_LOG_INFO, "DLSS: NGX's instance extensions added: %s", join(c->instance_copy.added).c_str());
        } else {
            say(c, UB_LOG_INFO, "DLSS: NGX's instance extensions: %s", need.empty() ? "none" : "the host's instance has them");
        }
        return UB_OK;
    } catch (...) {
        *out = in;
        return fail(c, UB_ERR_MEMORY, "prepare_instance: no memory");
    }
}

void UB_CALL instance_done(ub_context* c, VkResult result) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!c || c != g_ctx || !c->instance_pending) return;
    c->instance_pending = false;
    if (result != VK_SUCCESS && c->instance_copied) {
        char why[96];
        std::snprintf(why, sizeof(why), "the instance could not take NGX's extensions (VkResult %d)", static_cast<int>(result));
        say(c, UB_LOG_WARNING, "DLSS: %s", why);
        dlss_off(c, why);
    }
    c->instance_has_ngx = result == VK_SUCCESS && c->reason.empty();
    c->instance_copied = false;
    c->instance_copy = InstanceCopy{};
}

ub_result UB_CALL prepare_device(ub_context* c, VkInstance instance, VkPhysicalDevice phys, const VkDeviceCreateInfo* in,
                                 const VkDeviceCreateInfo** out) {
    if (out) *out = in;
    std::lock_guard<std::mutex> lk(g_mu);
    if (!c || c != g_ctx || !instance || !phys || !in || !out) return fail(c == g_ctx ? c : nullptr, UB_ERR_ARGUMENT, "prepare_device: a NULL pointer");
    if (!c->instance_prepared || !c->gipa) {
        dlss_off(c, "prepare_device: prepare_instance was not used");
        return fail(c, UB_ERR_STATE, "prepare_device: prepare_instance was not used");
    }
    if (c->device_pending) return fail(c, UB_ERR_STATE, "prepare_device: device_done of the last one is missing");
    if (c->attached) return fail(c, UB_ERR_STATE, "prepare_device: a device is attached");
    try {
        c->device_pending = true;
        c->device_planned = c->device_ready = false;
        c->prepared_phys = phys;
        auto props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(c->gipa(instance, "vkGetPhysicalDeviceProperties"));
        if (!props) {
            dlss_off(c, "prepare_device: no vkGetPhysicalDeviceProperties");
            return fail(c, UB_ERR_STATE, "prepare_device: no vkGetPhysicalDeviceProperties");
        }
        VkPhysicalDeviceProperties p{};
        props(phys, &p);
        VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        auto props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(c->gipa(instance, "vkGetPhysicalDeviceProperties2"));
        if (props2 && p.apiVersion >= VK_API_VERSION_1_2) {
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            p2.pNext = &driver;
            props2(phys, &p2);
        }
        // The device itself first: its own reason is the one that holds there.
        if (p.vendorID != kVendorNvidia) {
            c->reason = std::string("DLSS needs an NVIDIA RTX card (this GPU is ") + p.deviceName + ")";
            return UB_OK;
        }
        if (driver.driverID && driver.driverID != VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
            c->reason = std::string("DLSS needs NVIDIA's own driver (") + p.deviceName + " runs " + driver.driverName + ")";
            return UB_OK;
        }
        if (!c->reason.empty()) return UB_OK;
        if (!c->instance_has_ngx) {
            dlss_off(c, "the instance was made without NGX's extensions");
            return UB_OK;
        }
        if (ngx_kept_out(c)) return UB_OK;
        NVSDK_NGX_FeatureDiscoveryInfo d = discovery(c);
        NVSDK_NGX_FeatureRequirement req{};
        NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, phys, &d, &req);
        if (NVSDK_NGX_FAILED(r)) {
            const std::string why = "NGX could not check this card (" + result_text(r) + ")";
            dlss_off(c, why);
            return fail(c, UB_ERR_VENDOR, why);
        }
        if (req.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported) {
            const unsigned s = static_cast<unsigned>(req.FeatureSupported);
            char why[320];
            if (s & NVSDK_NGX_FeatureSupportResult_AdapterUnsupported)
                std::snprintf(why, sizeof(why), "DLSS does not run on this card (it needs an RTX card; this is %s)", p.deviceName);
            else if (s & NVSDK_NGX_FeatureSupportResult_DriverVersionUnsupported)
                std::snprintf(why, sizeof(why), "the NVIDIA driver is too old for DLSS");
            else if (s & NVSDK_NGX_FeatureSupportResult_OSVersionBelowMinimumSupported)
                std::snprintf(why, sizeof(why), "this system is too old for DLSS (NGX asks for %.200s)", req.MinOSVersion);
            else
                std::snprintf(why, sizeof(why), "NGX says DLSS is not supported here (0x%x)", s);
            dlss_off(c, why);
            return UB_OK;
        }
        std::uint32_t n = 0;
        VkExtensionProperties* need_props = nullptr;
        r = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, phys, &d, &n, &need_props);
        if (NVSDK_NGX_FAILED(r)) {
            const std::string why = "NGX could not name its Vulkan extensions (" + result_text(r) + ")";
            dlss_off(c, why);
            return fail(c, UB_ERR_VENDOR, why);
        }
        std::vector<std::string> need, have;
        for (std::uint32_t i = 0; need_props && i < n; ++i) need.emplace_back(need_props[i].extensionName);
        if (auto list = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(c->gipa(instance, "vkEnumerateDeviceExtensionProperties"))) {
            std::uint32_t m = 0;
            if (list(phys, nullptr, &m, nullptr) == VK_SUCCESS && m) {
                std::vector<VkExtensionProperties> e(m);
                if (list(phys, nullptr, &m, e.data()) >= 0)
                    for (std::uint32_t i = 0; i < m; ++i) have.emplace_back(e[i].extensionName);
            }
        }
        bool bda = false;
        if (auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(c->gipa(instance, "vkGetPhysicalDeviceFeatures2"));
            features2 && p.apiVersion >= VK_API_VERSION_1_2) {
            VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &f12;
            features2(phys, &f2);
            bda = f12.bufferDeviceAddress != VK_FALSE;
        }
        // (The host's chain is its own and writable, the API's rule: a field may be turned on for the call.)
        plan_device(const_cast<VkDeviceCreateInfo*>(in), need, have, p.apiVersion, bda, &c->device_copy);
        if (!c->device_copy.lacking.empty()) {
            dlss_off(c, "this card or driver lacks " + join(c->device_copy.lacking));
            return UB_OK;
        }
        c->device_planned = true;
        if (c->device_copy.copied) *out = &c->device_copy.info;
        say(c, UB_LOG_INFO, "DLSS: NGX's needs on the device: %s",
            c->device_copy.added.empty() ? "the host's device has them" : (join(c->device_copy.added) + " added").c_str());
        return UB_OK;
    } catch (...) {
        *out = in;
        restore_device(&c->device_copy);
        return fail(c, UB_ERR_MEMORY, "prepare_device: no memory");
    }
}

void UB_CALL device_done(ub_context* c, VkResult result) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!c || c != g_ctx || !c->device_pending) return;
    c->device_pending = false;
    const bool changed = c->device_copy.copied || !c->device_copy.restore.empty();
    restore_device(&c->device_copy);
    if (result != VK_SUCCESS && c->device_planned && changed) {
        char why[96];
        std::snprintf(why, sizeof(why), "the device could not be made with NGX's extensions (VkResult %d)", static_cast<int>(result));
        say(c, UB_LOG_WARNING, "DLSS: %s", why);
        dlss_off(c, why);
    }
    c->device_ready = result == VK_SUCCESS && c->device_planned && c->reason.empty();
    c->device_planned = false;
    c->device_copy.copied = false;
}

ub_result UB_CALL attach_device(ub_context* c, const ub_device_desc* desc) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!c || c != g_ctx) return UB_ERR_ARGUMENT;
    if (!desc || desc->struct_size < sizeof(ub_device_desc) || !desc->instance || !desc->physical_device || !desc->device ||
        !desc->get_instance_proc_addr || !desc->get_device_proc_addr)
        return fail(c, UB_ERR_ARGUMENT, "attach_device: a NULL handle or function, or an old struct_size");
    if (c->attached) return fail(c, UB_ERR_STATE, "attach_device: a device is attached");
    auto barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(desc->get_device_proc_addr(desc->device, "vkCmdPipelineBarrier"));
    if (!barrier) return fail(c, UB_ERR_ARGUMENT, "attach_device: no vkCmdPipelineBarrier from the get_device_proc_addr given");
    try {
        c->attached = true;
        c->instance = desc->instance;
        c->phys = desc->physical_device;
        c->device = desc->device;
        c->cmd_pipeline_barrier = barrier;
        c->frames_in_flight = desc->frames_in_flight ? desc->frames_in_flight : 8;
        if (c->reason.empty() && (!c->device_ready || c->prepared_phys != desc->physical_device)) dlss_off(c, "prepare_device was not used for this device");
        if (c->reason.empty() && !ngx_kept_out(c)) {
            NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(c->project_id.c_str(), NVSDK_NGX_ENGINE_TYPE_CUSTOM, c->engine_version.c_str(),
                                                                      c->data_w.c_str(), c->instance, c->phys, c->device,
                                                                      desc->get_instance_proc_addr, desc->get_device_proc_addr, &c->common,
                                                                      NVSDK_NGX_Version_API);
            if (NVSDK_NGX_FAILED(r)) {
                dlss_off(c, "NVIDIA's NGX did not start (" + result_text(r) + ")");
            } else {
                c->ngx_started = true;
                r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&c->params);
                if (NVSDK_NGX_FAILED(r) || !c->params) {
                    c->params = nullptr;
                    dlss_off(c, "NGX gave no parameters (" + result_text(r) + ")");
                } else {
                    int available = 0, needs_driver = 0, init_result = 0;
                    unsigned major = 0, minor = 0;
                    NVSDK_NGX_Parameter_GetI(c->params, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
                    NVSDK_NGX_Parameter_GetI(c->params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs_driver);
                    NVSDK_NGX_Parameter_GetUI(c->params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &major);
                    NVSDK_NGX_Parameter_GetUI(c->params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minor);
                    NVSDK_NGX_Parameter_GetI(c->params, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &init_result);
                    if (available) {
                        c->dlss = true;
                    } else if (needs_driver) {
                        char why[96];
                        std::snprintf(why, sizeof(why), "the NVIDIA driver is too old for DLSS (it needs %u.%u or newer)", major, minor);
                        dlss_off(c, why);
                    } else {
                        dlss_off(c, "DLSS is not available on this card or driver (" + result_text(static_cast<NVSDK_NGX_Result>(init_result)) + ")");
                    }
                }
            }
            if (!c->dlss) {
                // Nothing of NGX's is wanted: it goes now.
                if (c->params) NVSDK_NGX_VULKAN_DestroyParameters(c->params);
                c->params = nullptr;
                if (c->ngx_started) NVSDK_NGX_VULKAN_Shutdown1(c->device);
                c->ngx_started = false;
            }
        }
        say(c, UB_LOG_INFO, "attached: DLSS %s%s; FSR off: %s", c->dlss ? "on, " : "off: ",
            c->dlss ? (c->runtime_name + " " + c->runtime_version).c_str() : c->reason.c_str(), kFsrReason);
        return UB_OK;
    } catch (...) {
        return fail(c, UB_ERR_MEMORY, "attach_device: no memory");
    }
}

void UB_CALL detach_device(ub_context* c) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!c || c != g_ctx) return;
    detach_locked(c);
}

ub_result UB_CALL get_caps(ub_context* c, ub_caps* out) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!c || c != g_ctx || !out || out->struct_size < 2 * sizeof(std::uint32_t)) return fail(c == g_ctx ? c : nullptr, UB_ERR_ARGUMENT, "get_caps: a NULL pointer or an old struct_size");
    ub_caps caps{};
    caps.struct_size = out->struct_size;
    caps.dlss_available = c->attached && c->dlss ? 1u : 0u;
    std::snprintf(caps.bridge_version, sizeof(caps.bridge_version), "%s", kVersion);
    std::snprintf(caps.dlss_version, sizeof(caps.dlss_version), "%s", c->runtime_version.c_str());
    const std::string why = caps.dlss_available ? std::string() : !c->reason.empty() ? c->reason : std::string("no device attached yet");
    std::snprintf(caps.dlss_reason, sizeof(caps.dlss_reason), "%s", why.c_str());
    std::snprintf(caps.fsr_reason, sizeof(caps.fsr_reason), "%s", kFsrReason);
    // A struct_size that ends before a field: the fields after it are not written.
    std::memcpy(out, &caps, std::min<std::size_t>(out->struct_size, sizeof(caps)));
    return UB_OK;
}

ub_result UB_CALL get_render_info(ub_context* c, std::uint32_t method, std::uint32_t quality, std::uint32_t output_width,
                                  std::uint32_t output_height, ub_render_info* out) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!c || c != g_ctx || !out || out->struct_size < sizeof(std::uint32_t)) return fail(c == g_ctx ? c : nullptr, UB_ERR_ARGUMENT, "get_render_info: a NULL pointer or an old struct_size");
    if (method != UB_METHOD_DLSS && method != UB_METHOD_FSR) return fail(c, UB_ERR_ARGUMENT, "get_render_info: an unknown method");
    if (quality > UB_QUALITY_PERFORMANCE) return fail(c, UB_ERR_ARGUMENT, "get_render_info: an unknown quality");
    if (!output_width || !output_height) return fail(c, UB_ERR_ARGUMENT, "get_render_info: a size of 0");
    const double ratio = plain_ratio(method, quality);
    std::uint32_t w = render_extent(method, output_width, ratio), h = render_extent(method, output_height, ratio);
    // DLSS's own sizes once NGX runs (its optimal settings; the native one is the output's).
    if (method == UB_METHOD_DLSS && quality != UB_QUALITY_NATIVE && c->dlss && c->params) {
        if (c->ri_quality == quality && c->ri_ow == output_width && c->ri_oh == output_height) {
            w = c->ri_w;
            h = c->ri_h;
        } else {
            void* cb = nullptr;
            NVSDK_NGX_Parameter_GetVoidPointer(c->params, NVSDK_NGX_Parameter_DLSSOptimalSettingsCallback, &cb);
            if (cb) {
                static const NVSDK_NGX_PerfQuality_Value kPerf[4] = {NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
                                                                     NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf};
                NVSDK_NGX_Parameter_SetUI(c->params, NVSDK_NGX_Parameter_Width, output_width);
                NVSDK_NGX_Parameter_SetUI(c->params, NVSDK_NGX_Parameter_Height, output_height);
                NVSDK_NGX_Parameter_SetI(c->params, NVSDK_NGX_Parameter_PerfQualityValue, kPerf[quality]);
                NVSDK_NGX_Parameter_SetI(c->params, NVSDK_NGX_Parameter_RTXValue, 0);
                unsigned ngx_w = 0, ngx_h = 0;
                if (NVSDK_NGX_SUCCEED(reinterpret_cast<PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback>(cb)(c->params)) &&
                    NVSDK_NGX_SUCCEED(NVSDK_NGX_Parameter_GetUI(c->params, NVSDK_NGX_Parameter_OutWidth, &ngx_w)) &&
                    NVSDK_NGX_SUCCEED(NVSDK_NGX_Parameter_GetUI(c->params, NVSDK_NGX_Parameter_OutHeight, &ngx_h)) && ngx_w && ngx_h &&
                    ngx_w <= output_width && ngx_h <= output_height) {
                    w = ngx_w;
                    h = ngx_h;
                } else {
                    say(c, UB_LOG_WARNING, "DLSS failed at its render size for %ux%u: the plain ratio's %ux%u", output_width, output_height, w, h);
                }
            }
            c->ri_quality = quality;
            c->ri_ow = output_width;
            c->ri_oh = output_height;
            c->ri_w = w;
            c->ri_h = h;
        }
    }
    ub_render_info ri{};
    ri.struct_size = out->struct_size;
    ri.width = w;
    ri.height = h;
    ri.jitter_phases = jitter_phase_count(w, output_width);
    ri.mip_bias = mip_bias(w, output_width);
    std::memcpy(out, &ri, std::min<std::size_t>(out->struct_size, sizeof(ri)));
    return UB_OK;
}

void UB_CALL get_jitter(std::uint32_t frame, std::uint32_t phases, float* x, float* y) {
    float jx = 0.0f, jy = 0.0f;
    jitter(frame, phases, &jx, &jy);
    if (x) *x = jx;
    if (y) *y = jy;
}

VkImageAspectFlags aspects(VkFormat f) {
    switch (f) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT: return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT: return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        case VK_FORMAT_S8_UINT: return VK_IMAGE_ASPECT_STENCIL_BIT;
        default: return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

// The four images into the layouts DLSS reads and writes them in (the inputs sampled, the output a storage image), or
// back into the host's, every write before visible after.
void move_images(ub_context* c, VkCommandBuffer cb, const ub_evaluate_desc* d, bool back) {
    const ub_image* imgs[4] = {&d->color, &d->depth, &d->motion, &d->output};
    VkImageMemoryBarrier b[4]{};
    std::uint32_t n = 0;
    for (int i = 0; i < 4; ++i) {
        const bool output = i == 3;
        const VkImageLayout ours = output ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        const VkImageLayout theirs = imgs[i]->layout;
        VkImageMemoryBarrier& m = b[n++];
        m.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        m.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        m.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        m.oldLayout = back ? ours : theirs;
        // The output given UNDEFINED is left in GENERAL.
        m.newLayout = back ? (output && theirs == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : theirs) : ours;
        m.srcQueueFamilyIndex = m.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        m.image = imgs[i]->image;
        m.subresourceRange = {aspects(imgs[i]->format), 0, 1, 0, 1};
    }
    c->cmd_pipeline_barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, n, b);
}

NVSDK_NGX_Resource_VK resource(const ub_image& i, std::uint32_t w, std::uint32_t h, bool write) {
    const VkImageAspectFlags a = aspects(i.format);
    // A depth image is read by its depth alone.
    const VkImageSubresourceRange range{a & VK_IMAGE_ASPECT_DEPTH_BIT ? static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT) : a, 0, 1, 0, 1};
    return NVSDK_NGX_Create_ImageView_Resource_VK(i.view, i.image, range, i.format, w, h, write);
}

ub_result UB_CALL evaluate(ub_context* c, VkCommandBuffer cb, const ub_evaluate_desc* d) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!c || c != g_ctx) return UB_ERR_ARGUMENT;
    if (!d || d->struct_size < offsetof(ub_evaluate_desc, dlss_preset)) return fail(c, UB_ERR_ARGUMENT, "evaluate: a NULL pointer or an old struct_size");
    if (!cb) return fail(c, UB_ERR_ARGUMENT, "evaluate: no command buffer");
    if (!c->attached) return fail(c, UB_ERR_STATE, "evaluate: no device attached");
    if (d->method != UB_METHOD_DLSS && d->method != UB_METHOD_FSR) return fail(c, UB_ERR_ARGUMENT, "evaluate: an unknown method");
    if (d->quality > UB_QUALITY_PERFORMANCE) return fail(c, UB_ERR_ARGUMENT, "evaluate: an unknown quality");
    if (!d->render_width || !d->render_height || !d->output_width || !d->output_height) return fail(c, UB_ERR_ARGUMENT, "evaluate: a size of 0");
    if (d->render_width > d->output_width || d->render_height > d->output_height)
        return fail(c, UB_ERR_ARGUMENT, "evaluate: the render size is above the output size");
    if (d->quality == UB_QUALITY_NATIVE && (d->render_width != d->output_width || d->render_height != d->output_height))
        return fail(c, UB_ERR_ARGUMENT, "evaluate: native quality needs the render size = the output size");
    if (!d->color.image || !d->depth.image || !d->motion.image || !d->output.image) return fail(c, UB_ERR_ARGUMENT, "evaluate: an image is missing");
    if (d->color.layout == VK_IMAGE_LAYOUT_UNDEFINED || d->depth.layout == VK_IMAGE_LAYOUT_UNDEFINED || d->motion.layout == VK_IMAGE_LAYOUT_UNDEFINED)
        return fail(c, UB_ERR_ARGUMENT, "evaluate: an input image is given in layout UNDEFINED");
    if (d->method == UB_METHOD_FSR) return fail(c, UB_ERR_UNAVAILABLE, "FSR is not available");
    if (!c->dlss || !c->params) return fail(c, UB_ERR_UNAVAILABLE, "DLSS is not available");
    if (!d->color.view || !d->depth.view || !d->motion.view || !d->output.view) return fail(c, UB_ERR_ARGUMENT, "evaluate: DLSS needs an image view of each image");
    const std::uint32_t preset_asked =
        d->struct_size >= offsetof(ub_evaluate_desc, dlss_preset) + sizeof(d->dlss_preset) ? d->dlss_preset : 0;
    const std::uint32_t preset = preset_asked ? preset_asked : kDefaultPreset;
    const std::uint32_t flags = d->flags & (UB_EVAL_LDR | UB_EVAL_DEPTH_INVERTED);
    try {
        // The replaced contexts the GPU may no longer use go (frames_in_flight + 2 evaluates after their last).
        for (std::size_t i = 0; i < c->retired.size();) {
            if (c->retired[i].evaluates_left-- == 0) {
                release_feature(c, c->retired[i].handle, "its release");
                c->retired.erase(c->retired.begin() + static_cast<std::ptrdiff_t>(i));
            } else {
                ++i;
            }
        }
        Feature& f = c->feature;
        if (!f.handle || f.rw != d->render_width || f.rh != d->render_height || f.ow != d->output_width || f.oh != d->output_height ||
            f.quality != d->quality || f.flags != flags || f.preset != preset) {
            if (f.handle) c->retired.push_back({f.handle, c->frames_in_flight + 2});
            f = Feature{};
            static const NVSDK_NGX_PerfQuality_Value kPerf[4] = {NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
                                                                 NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf};
            static const char* const kPresetHints[] = {
                NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality,
            };
            for (const char* hint : kPresetHints) NVSDK_NGX_Parameter_SetUI(c->params, hint, preset);
            const bool ldr = (flags & UB_EVAL_LDR) != 0;
            NVSDK_NGX_DLSS_Create_Params cp{};
            cp.Feature.InWidth = d->render_width;
            cp.Feature.InHeight = d->render_height;
            cp.Feature.InTargetWidth = d->output_width;
            cp.Feature.InTargetHeight = d->output_height;
            cp.Feature.InPerfQualityValue = kPerf[d->quality];
            cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                      (ldr ? 0 : NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure) |
                                      (flags & UB_EVAL_DEPTH_INVERTED ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0);
            cp.InEnableOutputSubrects = false;
            NVSDK_NGX_Handle* h = nullptr;
            const NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT1(c->device, cb, 1, 1, &h, c->params, &cp);
            if (NVSDK_NGX_FAILED(r) || !h) {
                const std::string why = "DLSS failed at its start (" + result_text(r) + ")";
                say(c, UB_LOG_ERROR, "%s", why.c_str());
                return fail(c, UB_ERR_VENDOR, why);
            }
            f.handle = h;
            f.rw = d->render_width;
            f.rh = d->render_height;
            f.ow = d->output_width;
            f.oh = d->output_height;
            f.quality = d->quality;
            f.flags = flags;
            f.preset = preset;
            say(c, UB_LOG_INFO, "DLSS: made at %ux%u -> %ux%u, %s, preset %c", f.rw, f.rh, f.ow, f.oh, ldr ? "LDR" : "HDR with NGX's auto exposure",
                preset >= 1 && preset <= 26 ? static_cast<char>('A' + preset - 1) : '?');
        }
        NVSDK_NGX_Resource_VK color = resource(d->color, d->render_width, d->render_height, false);
        NVSDK_NGX_Resource_VK depth = resource(d->depth, d->render_width, d->render_height, false);
        NVSDK_NGX_Resource_VK motion = resource(d->motion, d->render_width, d->render_height, false);
        NVSDK_NGX_Resource_VK output = resource(d->output, d->output_width, d->output_height, true);
        NVSDK_NGX_VK_DLSS_Eval_Params ep{};
        ep.Feature.pInColor = &color;
        ep.Feature.pInOutput = &output;
        ep.pInDepth = &depth;
        ep.pInMotionVectors = &motion;
        // The API's jitter is the picture's move in render pixels, +x right and +y down: NGX's own convention (the
        // jitter applied to the projection, in the motion vectors' directions).
        ep.InJitterOffsetX = d->jitter_x * c->jitter_sx;
        ep.InJitterOffsetY = d->jitter_y * c->jitter_sy;
        ep.InRenderSubrectDimensions.Width = d->render_width;
        ep.InRenderSubrectDimensions.Height = d->render_height;
        ep.InReset = d->flags & UB_EVAL_RESET ? 1 : 0;
        ep.InMVScaleX = d->motion_scale_x != 0.0f ? d->motion_scale_x : 1.0f;
        ep.InMVScaleY = d->motion_scale_y != 0.0f ? d->motion_scale_y : 1.0f;
        ep.InFrameTimeDeltaInMsec = d->frame_time_ms;
        move_images(c, cb, d, false);
        const NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cb, f.handle, c->params, &ep);
        move_images(c, cb, d, true);
        if (NVSDK_NGX_FAILED(r)) {
            const std::string why = "DLSS failed at its evaluation (" + result_text(r) + ")";
            say(c, UB_LOG_ERROR, "%s", why.c_str());
            return fail(c, UB_ERR_VENDOR, why);
        }
        return UB_OK;
    } catch (...) {
        return fail(c, UB_ERR_MEMORY, "evaluate: no memory");
    }
}

const ub_api kApi = {
    sizeof(ub_api), UB_API_VERSION, &open_context, &close_context, &last_error, &prepare_instance, &instance_done,
    &prepare_device, &device_done, &attach_device, &detach_device, &get_caps, &get_render_info, &get_jitter, &evaluate,
};

}  // namespace

extern "C" __attribute__((visibility("default"))) const ub_api* UB_CALL ub_get_api(std::uint32_t api_version) {
    return api_version == UB_API_VERSION ? &kApi : nullptr;
}
