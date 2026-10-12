#include "host/reflex.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <unordered_map>

#include "core/host_clock.h"
#include "log.h"
#include "host/settings.h"

namespace host::reflex {

namespace {

#if defined(BBHOST_HAVE_VULKAN)
typedef VkResult (VKAPI_PTR *PFN_vkSetLatencySleepModeNV)(VkDevice device, VkSwapchainKHR swapchain, const VkLatencySleepModeInfoNV* pSleepModeInfo);
typedef VkResult (VKAPI_PTR *PFN_vkLatencySleepNV)(VkDevice device, VkSwapchainKHR swapchain, const VkLatencySleepInfoNV* pSleepInfo);
typedef void (VKAPI_PTR *PFN_vkSetLatencyMarkerNV)(VkDevice device, VkSwapchainKHR swapchain, const VkSetLatencyMarkerInfoNV* pLatencyMarkerInfo);
typedef void (VKAPI_PTR *PFN_vkGetLatencyTimingsNV)(VkDevice device, VkSwapchainKHR swapchain, VkGetLatencyMarkerInfoNV* pLatencyMarkerInfo);
typedef void (VKAPI_PTR *PFN_vkQueueNotifyOutOfBandNV)(VkQueue queue, const VkOutOfBandQueueTypeInfoNV* pQueueTypeInfo);

PFN_vkSetLatencySleepModeNV g_pfn_set_sleep_mode = nullptr;
PFN_vkLatencySleepNV g_pfn_latency_sleep = nullptr;
PFN_vkSetLatencyMarkerNV g_pfn_set_marker = nullptr;
PFN_vkGetLatencyTimingsNV g_pfn_get_timings = nullptr;
PFN_vkQueueNotifyOutOfBandNV g_pfn_queue_notify_oob = nullptr;

VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_phys = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
VkSwapchainKHR g_swapchain = VK_NULL_HANDLE;
VkQueue g_present_queue = VK_NULL_HANDLE;

VkSemaphore g_sleep_semaphore = VK_NULL_HANDLE;
uint64_t g_sleep_value = 0;
std::unordered_map<VkQueue, uint64_t> g_queue_ids;
uint64_t g_render_started = 0;
uint64_t g_report_logged = 0;
uint64_t g_input_sampled = 0;
#endif
std::mutex g_mu;
std::mutex g_sleep_mu; // serializes sleeps and swapchain/semaphore destruction
std::mutex g_swapchain_api_mu;
bool g_supported = false;
Mode g_mode = Mode::kOn;
uint32_t g_min_interval_us = 0;

std::atomic<uint64_t> g_sim_frame_id{0};
thread_local uint64_t t_cp_frame_id = 0;

// Diagnostic throttle state for BBHOST_TEST_SOURCE_FPS
struct TestPacer {
    bool enabled = false;
    std::vector<int> targets;
    size_t current_idx = 0;
    std::chrono::steady_clock::time_point last_cycle_switch{};
    std::chrono::steady_clock::time_point last_frame_time{};
    int active_fps = 0;
};
TestPacer g_test_pacer;
std::mutex g_pacer_mu;

void init_test_pacer() {
    const char* env = std::getenv("BBHOST_TEST_SOURCE_FPS");
    if (!env || !env[0]) return;
    std::string s(env);
    std::vector<int> rates;
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) comma = s.size();
        std::string token = s.substr(start, comma - start);
        int v = std::atoi(token.c_str());
        if (v > 0) rates.push_back(std::clamp(v, 30, 60));
        start = comma + 1;
    }
    if (!rates.empty()) {
        g_test_pacer.enabled = true;
        g_test_pacer.targets = rates;
        g_test_pacer.current_idx = 0;
        g_test_pacer.active_fps = rates[0];
        g_test_pacer.last_cycle_switch = std::chrono::steady_clock::now();
        g_test_pacer.last_frame_time = std::chrono::steady_clock::now();
        host_log("reflex: BBHOST_TEST_SOURCE_FPS active; target %d fps", g_test_pacer.active_fps);
    }
}

void pace_diagnostic_frame() {
    static std::once_flag once;
    std::call_once(once, init_test_pacer);
    if (!g_test_pacer.enabled) return;
    std::unique_lock<std::mutex> lk(g_pacer_mu);
    const auto now = std::chrono::steady_clock::now();
    if (g_test_pacer.targets.size() > 1) {
        if (now - g_test_pacer.last_cycle_switch >= std::chrono::seconds(10)) {
            g_test_pacer.current_idx = (g_test_pacer.current_idx + 1) % g_test_pacer.targets.size();
            g_test_pacer.active_fps = g_test_pacer.targets[g_test_pacer.current_idx];
            g_test_pacer.last_cycle_switch = now;
            host_log("reflex: BBHOST_TEST_SOURCE_FPS cycling to %d fps", g_test_pacer.active_fps);
        }
    }
    if (g_test_pacer.active_fps > 0) {
        const uint64_t target_interval_ns = 1000000000ull / static_cast<uint64_t>(g_test_pacer.active_fps);
        const auto target_time = g_test_pacer.last_frame_time + std::chrono::nanoseconds(target_interval_ns);
        if (target_time > now) {
            std::this_thread::sleep_until(target_time);
        }
        const auto finished = std::chrono::steady_clock::now();
        g_test_pacer.last_frame_time = finished - target_time < std::chrono::nanoseconds(target_interval_ns)
            ? target_time : finished;
    }
}

Mode parse_env_or_config_mode() {
    // Check environment override first
    if (const char* env = std::getenv("BBHOST_REFLEX")) {
        if (std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0) return Mode::kOff;
        if (std::strcmp(env, "1") == 0 || std::strcmp(env, "on") == 0) return Mode::kOn;
        if (std::strcmp(env, "2") == 0 || std::strcmp(env, "boost") == 0) return Mode::kBoost;
    }
    // Check HostSettings startup configuration
    const int host_val = host_startup_settings().reflex_mode;
    if (host_val == 0) return Mode::kOff;
    if (host_val == 2) return Mode::kBoost;
    return Mode::kOn;
}

#if defined(BBHOST_HAVE_VULKAN)
void disable_locked(const char* operation, VkResult result) {
    host_log("reflex: %s failed (%d); driver pacing and markers disabled", operation, int(result));
    g_supported = false;
}
void apply_mode_locked() {
    if (!g_supported || !g_device || !g_swapchain || !g_pfn_set_sleep_mode) return;
    VkLatencySleepModeInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV;
    info.pNext = nullptr;
    info.lowLatencyMode = (g_mode != Mode::kOff) ? VK_TRUE : VK_FALSE;
    info.lowLatencyBoost = (g_mode == Mode::kBoost) ? VK_TRUE : VK_FALSE;
    info.minimumIntervalUs = g_min_interval_us;
    const VkResult r = g_pfn_set_sleep_mode(g_device, g_swapchain, &info);
    if (r != VK_SUCCESS) disable_locked("vkSetLatencySleepModeNV", r);
    host_log("reflex: set latency sleep mode: mode=%d boost=%d min_interval=%u us (result=%d)",
             static_cast<int>(info.lowLatencyMode), static_cast<int>(info.lowLatencyBoost),
             info.minimumIntervalUs, static_cast<int>(r));
}

void emit_marker_locked(uint64_t frame_id, VkLatencyMarkerNV marker) {
    if (!g_supported || !g_device || !g_swapchain || !g_pfn_set_marker) return;
    VkSetLatencyMarkerInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV;
    info.pNext = nullptr;
    info.presentID = frame_id;
    info.marker = marker;
    g_pfn_set_marker(g_device, g_swapchain, &info);
}

#endif
}  // namespace

std::mutex& swapchain_api_mutex() { return g_swapchain_api_mu; }

#if defined(BBHOST_HAVE_VULKAN)
bool configure_device(VkPhysicalDevice phys, std::vector<const char*>& exts,
                      VkPhysicalDeviceVulkan12Features& features12,
                      VkPhysicalDevicePresentIdFeaturesKHR& present_id) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, nullptr) != VK_SUCCESS) return false;
    std::vector<VkExtensionProperties> available(count);
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, available.data()) != VK_SUCCESS) return false;
    bool latency = false, present = false;
    for (const auto& prop : available) {
        latency |= std::strcmp(prop.extensionName, VK_NV_LOW_LATENCY_2_EXTENSION_NAME) == 0 && prop.specVersion >= 3;
        present |= std::strcmp(prop.extensionName, VK_KHR_PRESENT_ID_EXTENSION_NAME) == 0;
    }
    if (!latency || !present) return false;
    VkPhysicalDeviceVulkan12Features q12{}; q12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDevicePresentIdFeaturesKHR qp{}; qp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 query{}; query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    query.pNext = &q12; q12.pNext = &qp;
    vkGetPhysicalDeviceFeatures2(phys, &query);
    if (!q12.timelineSemaphore || !qp.presentId) return false;
    features12.timelineSemaphore = VK_TRUE;
    present_id = {};
    present_id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;
    present_id.presentId = VK_TRUE;
    for (auto name : {VK_NV_LOW_LATENCY_2_EXTENSION_NAME, VK_KHR_PRESENT_ID_EXTENSION_NAME}) {
        if (std::none_of(exts.begin(), exts.end(), [&](const char* x) { return std::strcmp(x, name) == 0; })) exts.push_back(name);
    }
    host_log("reflex: low_latency2 revision 3, present IDs and timeline semaphores enabled");
    return true;
}

void init(VkInstance instance, VkPhysicalDevice phys, VkDevice device, bool extensions_enabled) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_instance = instance;
    g_phys = phys;
    g_device = device;

    g_mode = parse_env_or_config_mode();
    if (const char* interval = std::getenv("BBHOST_REFLEX_INTERVAL_US"))
        g_min_interval_us = static_cast<uint32_t>(std::clamp(std::atoi(interval), 0, 100000));

    if (!device || !extensions_enabled) {
        g_supported = false;
        if (g_mode != Mode::kOff)
            host_log("reflex: required driver extension/features unavailable; ordinary pacing retained");
        return;
    }

    g_pfn_set_sleep_mode = reinterpret_cast<PFN_vkSetLatencySleepModeNV>(vkGetDeviceProcAddr(device, "vkSetLatencySleepModeNV"));
    g_pfn_latency_sleep = reinterpret_cast<PFN_vkLatencySleepNV>(vkGetDeviceProcAddr(device, "vkLatencySleepNV"));
    g_pfn_set_marker = reinterpret_cast<PFN_vkSetLatencyMarkerNV>(vkGetDeviceProcAddr(device, "vkSetLatencyMarkerNV"));
    g_pfn_get_timings = reinterpret_cast<PFN_vkGetLatencyTimingsNV>(vkGetDeviceProcAddr(device, "vkGetLatencyTimingsNV"));
    g_pfn_queue_notify_oob = reinterpret_cast<PFN_vkQueueNotifyOutOfBandNV>(vkGetDeviceProcAddr(device, "vkQueueNotifyOutOfBandNV"));

    g_supported = g_pfn_set_sleep_mode && g_pfn_latency_sleep && g_pfn_set_marker && g_pfn_get_timings && g_pfn_queue_notify_oob;
    if (g_supported) {
        VkSemaphoreTypeCreateInfo type{}; type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo ci{}; ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO; ci.pNext = &type;
        const auto r = vkCreateSemaphore(device, &ci, nullptr, &g_sleep_semaphore);
        if (r != VK_SUCCESS) disable_locked("timeline semaphore creation", r);
    }
    host_log("reflex: device initialized: supported=%d (mode=%s)",
             g_supported, g_mode == Mode::kOff ? "off" : g_mode == Mode::kBoost ? "on+boost" : "on");
}

#endif
void shutdown() {
    std::lock_guard<std::mutex> sleep(g_sleep_mu);
    std::lock_guard<std::mutex> lk(g_mu);
#if defined(BBHOST_HAVE_VULKAN)
    // Device shutdown must already have drained its outstanding work.
    if (g_sleep_semaphore) vkDestroySemaphore(g_device, g_sleep_semaphore, nullptr);
    g_sleep_semaphore = VK_NULL_HANDLE;
    g_swapchain = VK_NULL_HANDLE;
    g_present_queue = VK_NULL_HANDLE;
    g_device = VK_NULL_HANDLE;
    g_queue_ids.clear();
#endif
    g_supported = false;
}

bool is_supported() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_supported;
}

bool is_active() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_supported && (g_mode != Mode::kOff);
}

Mode current_mode() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_mode;
}

void set_mode(Mode mode, uint32_t minimum_interval_us) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_mode = mode;
    g_min_interval_us = minimum_interval_us;
    #if defined(BBHOST_HAVE_VULKAN)
    apply_mode_locked();
    #endif
}

#if defined(BBHOST_HAVE_VULKAN)
void on_swapchain_created(VkSwapchainKHR swapchain) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_swapchain = swapchain;
    apply_mode_locked();
}

void on_swapchain_destroying(VkSwapchainKHR old_swapchain) {
    std::lock_guard<std::mutex> sleep(g_sleep_mu);
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_swapchain == old_swapchain) {
        g_swapchain = VK_NULL_HANDLE;
    }
}

void on_present_queue_created(VkQueue present_queue) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_present_queue = present_queue;
    if (g_supported && present_queue && g_pfn_queue_notify_oob) {
        VkOutOfBandQueueTypeInfoNV info{};
        info.sType = VK_STRUCTURE_TYPE_OUT_OF_BAND_QUEUE_TYPE_INFO_NV;
        info.pNext = nullptr;
        info.queueType = VK_OUT_OF_BAND_QUEUE_TYPE_PRESENT_NV;
        g_pfn_queue_notify_oob(present_queue, &info);
        host_log("reflex: present queue marked OUT_OF_BAND (present)");
    }
}

uint64_t prepare_submit(VkQueue queue, uint64_t frame_id, VkSubmitInfo& submit,
                        VkLatencySubmissionPresentIdNV& attribution) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_supported) return 0;
    auto& last = g_queue_ids[queue];
    if (frame_id && frame_id < last) {
        static unsigned logs = 0;
        if (logs++ < 8) host_log("reflex: stale submission ID %llu after %llu; clearing queue attribution",
                               (unsigned long long)frame_id, (unsigned long long)last);
        frame_id = 0;
    }
    if (frame_id) last = frame_id; // keep historical maximum even after zero clears
    attribution = {};
    attribution.sType = VK_STRUCTURE_TYPE_LATENCY_SUBMISSION_PRESENT_ID_NV;
    attribution.presentID = frame_id;
    attribution.pNext = submit.pNext; submit.pNext = &attribution;
    return frame_id;
}
#endif
uint64_t current_sim_frame_id() {
    return g_sim_frame_id.load(std::memory_order_relaxed);
}

void set_cp_thread_frame_id(uint64_t frame_id) {
    t_cp_frame_id = frame_id;
}

uint64_t cp_thread_frame_id() {
    return t_cp_frame_id;
}

uint64_t on_frame_begin() {
    pace_diagnostic_frame();
    std::lock_guard<std::mutex> sleep(g_sleep_mu);
    std::unique_lock<std::mutex> lk(g_mu);
    #if defined(BBHOST_HAVE_VULKAN)
    static const bool sleep_enabled = [] {
        const char* e = std::getenv("BBHOST_REFLEX_SLEEP");
        return !(e && e[0] == '0');
    }();
    if (g_supported && g_device && g_swapchain && sleep_enabled && g_pfn_latency_sleep) {
        VkLatencySleepInfoNV sleep_info{};
        sleep_info.sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV;
        sleep_info.pNext = nullptr;
        sleep_info.signalSemaphore = g_sleep_semaphore;
        sleep_info.value = ++g_sleep_value;
        const auto device = g_device;
        const auto swapchain = g_swapchain;
        const auto started = std::chrono::steady_clock::now();
        // Neither driver sleep nor its CPU wait may hold the marker/submit
        // mutex. g_sleep_mu keeps these handles alive throughout both calls.
        lk.unlock();
        VkResult r;
        {
            std::lock_guard<std::mutex> access(g_swapchain_api_mu);
            r = g_pfn_latency_sleep(device, swapchain, &sleep_info);
        }
        if (r == VK_SUCCESS) {
            VkSemaphoreWaitInfo wait{}; wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            wait.semaphoreCount = 1; wait.pSemaphores = &g_sleep_semaphore;
            wait.pValues = &sleep_info.value;
            // Allow the submit/present threads to progress while the CPU is
            // sleeping. g_sleep_mu protects the semaphore and swapchain lifetime.
            r = vkWaitSemaphores(device, &wait, 100000000ull);
        }
        lk.lock();
        const auto sleep_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
        // Ordinary driver pacing can sleep for a full source interval. Report
        // unusually long waits at most once a second, not every GPU-bound frame.
        static auto last_slow_sleep = std::chrono::steady_clock::time_point{};
        const auto finished = std::chrono::steady_clock::now();
        if (sleep_us > 50000 && finished - last_slow_sleep >= std::chrono::seconds(1)) {
            last_slow_sleep = finished;
            host_log("reflex: latency sleep took %lld us", static_cast<long long>(sleep_us));
        }
        if (r != VK_SUCCESS) disable_locked("latency sleep/wait", r);
    }

    #endif
    const uint64_t fid = g_sim_frame_id.fetch_add(1, std::memory_order_relaxed) + 1;
    #if defined(BBHOST_HAVE_VULKAN)
    emit_marker_locked(fid, VK_LATENCY_MARKER_SIMULATION_START_NV);
    #endif
    return fid;
}

void on_input_sampled() {
    const uint64_t fid = current_sim_frame_id();
    if (!fid) return;
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(BBHOST_HAVE_VULKAN)
    if (fid <= g_input_sampled) return;
    g_input_sampled = fid;
    emit_marker_locked(fid, VK_LATENCY_MARKER_INPUT_SAMPLE_NV);
    #endif
}

void on_simulation_end() {
    const uint64_t fid = current_sim_frame_id();
    if (!fid) return;
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(BBHOST_HAVE_VULKAN)
    emit_marker_locked(fid, VK_LATENCY_MARKER_SIMULATION_END_NV);
    #endif
}

void on_render_submit_begin(uint64_t frame_id) {
    if (!frame_id) return;
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(BBHOST_HAVE_VULKAN)
    if (frame_id > g_render_started) {
        g_render_started = frame_id;
        emit_marker_locked(frame_id, VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
    }
    #endif
}

void on_render_submit_end(uint64_t frame_id) {
    if (!frame_id) return;
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(BBHOST_HAVE_VULKAN)
    emit_marker_locked(frame_id, VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    #endif
}

void on_foreign_submit_begin(uint64_t frame_id) {
    if (!frame_id) return;
#if defined(BBHOST_HAVE_VULKAN)
    std::lock_guard<std::mutex> lk(g_mu);
    emit_marker_locked(frame_id, VK_LATENCY_MARKER_OUT_OF_BAND_RENDERSUBMIT_START_NV);
#endif
}
void on_foreign_submit_end(uint64_t frame_id) {
    if (!frame_id) return;
#if defined(BBHOST_HAVE_VULKAN)
    std::lock_guard<std::mutex> lk(g_mu);
    emit_marker_locked(frame_id, VK_LATENCY_MARKER_OUT_OF_BAND_RENDERSUBMIT_END_NV);
#endif
}
void on_present_begin(uint64_t frame_id, bool is_real) {
    if (!frame_id) return;
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(BBHOST_HAVE_VULKAN)
    emit_marker_locked(frame_id, is_real ? VK_LATENCY_MARKER_PRESENT_START_NV
                                         : VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_START_NV);
    #endif
}

void on_present_end(uint64_t frame_id, bool is_real) {
    if (!frame_id) return;
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(BBHOST_HAVE_VULKAN)
    emit_marker_locked(frame_id, is_real ? VK_LATENCY_MARKER_PRESENT_END_NV
                                         : VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_END_NV);
    #endif
}

bool query_latency_timings(LatencyReport* out_report) {
    if (!out_report) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(BBHOST_HAVE_VULKAN)
    if (!g_supported || !g_device || !g_swapchain || !g_pfn_get_timings) return false;

    const auto query_start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> access(g_swapchain_api_mu);
    VkGetLatencyMarkerInfoNV info{}; info.sType = VK_STRUCTURE_TYPE_GET_LATENCY_MARKER_INFO_NV;
    g_pfn_get_timings(g_device, g_swapchain, &info);
    if (!info.timingCount || info.timingCount > 4096) return false;
    std::vector<VkLatencyTimingsFrameReportNV> reports(info.timingCount);
    for (auto& report : reports) report.sType = VK_STRUCTURE_TYPE_LATENCY_TIMINGS_FRAME_REPORT_NV;
    info.pTimings = reports.data();
    g_pfn_get_timings(g_device, g_swapchain, &info);
    const auto query_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - query_start).count();
    if (query_us > 5000) host_log("reflex: latency query took %lld us", static_cast<long long>(query_us));
    const VkLatencyTimingsFrameReportNV* latest = nullptr;
    for (uint32_t i = 0; i < std::min(info.timingCount, uint32_t(reports.size())); ++i) {
        const auto& r = reports[i];
        if (r.presentID && r.simStartTimeUs && r.simEndTimeUs && r.renderSubmitStartTimeUs &&
            r.renderSubmitEndTimeUs && r.presentStartTimeUs && r.presentEndTimeUs &&
            r.gpuRenderStartTimeUs && r.gpuRenderEndTimeUs && (!latest || r.presentID > latest->presentID)) latest = &r;
    }
    if (!latest) return false;
    const auto& report = *latest;
    static const bool diagnostic = [] { const char* e = std::getenv("BBHOST_REFLEX_TIMING"); return e && e[0] == '1'; }();
    if (diagnostic && report.presentID > g_report_logged) {
        g_report_logged = report.presentID;
        host_log("reflex timing: id=%llu input=%llu sim=%llu..%llu render=%llu..%llu present=%llu..%llu gpu=%llu..%llu (driver timestamps us)",
                 (unsigned long long)report.presentID, (unsigned long long)report.inputSampleTimeUs,
                 (unsigned long long)report.simStartTimeUs, (unsigned long long)report.simEndTimeUs,
                 (unsigned long long)report.renderSubmitStartTimeUs, (unsigned long long)report.renderSubmitEndTimeUs,
                 (unsigned long long)report.presentStartTimeUs, (unsigned long long)report.presentEndTimeUs,
                 (unsigned long long)report.gpuRenderStartTimeUs, (unsigned long long)report.gpuRenderEndTimeUs);
    }
    out_report->present_id = report.presentID;
    out_report->input_sample_us = report.inputSampleTimeUs;
    out_report->sim_start_us = report.simStartTimeUs;
    out_report->sim_end_us = report.simEndTimeUs;
    out_report->render_submit_start_us = report.renderSubmitStartTimeUs;
    out_report->render_submit_end_us = report.renderSubmitEndTimeUs;
    out_report->present_start_us = report.presentStartTimeUs;
    out_report->present_end_us = report.presentEndTimeUs;
    out_report->gpu_render_start_us = report.gpuRenderStartTimeUs;
    out_report->gpu_render_end_us = report.gpuRenderEndTimeUs;
    return true;
    #else
    return false;
    #endif
}

}  // namespace host::reflex
