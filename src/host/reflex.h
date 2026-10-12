#pragma once

// NVIDIA Reflex (VK_NV_low_latency2 rev3) integration for bbhost.
//
// Provides low-latency mode (Reflex On / On+Boost), driver frame pacing
// (vkLatencySleepNV), and precise per-frame latency tracking markers across
// guest simulation, input sampling, command processor recording, Vulkan
// queue submission, and custom presentation (including Frame Generation).

#include <cstdint>
#include <mutex>
#include <vector>

#if defined(BBHOST_HAVE_VULKAN)
#include <vulkan/vulkan.h>
#endif

namespace host::reflex {

// Serialize acquire/present with sleep/timing queries. Marker calls have an
// explicit specification exception and must remain free to progress.
std::mutex& swapchain_api_mutex();

enum class Mode : int {
    kOff = 0,
    kOn = 1,
    kBoost = 2
};

struct LatencyReport {
    uint64_t present_id = 0;
    uint64_t input_sample_us = 0;
    uint64_t sim_start_us = 0;
    uint64_t sim_end_us = 0;
    uint64_t render_submit_start_us = 0;
    uint64_t render_submit_end_us = 0;
    uint64_t present_start_us = 0;
    uint64_t present_end_us = 0;
    uint64_t gpu_render_start_us = 0;
    uint64_t gpu_render_end_us = 0;
};

// Device extension inspection and registration (called before vkCreateDevice).
#if defined(BBHOST_HAVE_VULKAN)
// The caller keeps present_id alive through vkCreateDevice; chain it only if true.
bool configure_device(VkPhysicalDevice phys, std::vector<const char*>& exts,
                      VkPhysicalDeviceVulkan12Features& features12,
                      VkPhysicalDevicePresentIdFeaturesKHR& present_id);

// Lifecycle on Vulkan device (called after vkCreateDevice).
void init(VkInstance instance, VkPhysicalDevice phys, VkDevice device, bool extensions_enabled);
#endif
void shutdown();

bool is_supported();
bool is_active();
Mode current_mode();
void set_mode(Mode mode, uint32_t minimum_interval_us = 0);

// Swapchain registration and serialization.
// on_swapchain_destroying must be called before vkDestroySwapchainKHR to serialize
// with in-flight on_frame_begin sleep and marker operations.
#if defined(BBHOST_HAVE_VULKAN)
void on_swapchain_created(VkSwapchainKHR swapchain);
void on_swapchain_destroying(VkSwapchainKHR old_swapchain);

// Dedicated presenter queue registration: marked OUT_OF_BAND so presenter blits
// are ignored for latency evaluation by the driver. OOB_PRESENT affects queue
// submissions, while real vkQueuePresentKHR calls remain attributed by markers.
void on_present_queue_created(VkQueue present_queue);
// Caller holds its queue mutex. Zero clears prior attribution. Stale IDs are
// cleared instead of violating the extension's monotonic submission rule.
// Zero clears this queue, but is not an exclusion flag: other explicit queues
// can still supply device-wide fallback attribution. Shared-queue foreign work
// should carry its captured source ID; only a dedicated OOB_PRESENT queue
// excludes compositor submissions from driver timing.
uint64_t prepare_submit(VkQueue queue, uint64_t frame_id, VkSubmitInfo& submit,
                        VkLatencySubmissionPresentIdNV& attribution);
#endif

// Thread Attribution & Context Tracking
uint64_t current_sim_frame_id();
void set_cp_thread_frame_id(uint64_t frame_id);
uint64_t cp_thread_frame_id();

// Simulation & Input Tracing (Guest Main Thread)
// - Handles diagnostic pacing (BBHOST_TEST_SOURCE_FPS)
// - Calls vkLatencySleepNV before input sampling
// - Increments monotonic frame ID
// - Emits VK_LATENCY_MARKER_SIMULATION_START_NV
uint64_t on_frame_begin();

// Emits VK_LATENCY_MARKER_INPUT_SAMPLE_NV for the current simulation frame.
void on_input_sampled();

// Emits VK_LATENCY_MARKER_SIMULATION_END_NV for the current simulation frame.
void on_simulation_end();

// Render Queue Attribution (Submit Thread / Renderer Queue)
// Emits RENDERSUBMIT_START on first submit for a frame, and RENDERSUBMIT_END
// on each completed submit for that frame.
void on_render_submit_begin(uint64_t frame_id);
void on_render_submit_end(uint64_t frame_id);
void on_foreign_submit_begin(uint64_t frame_id);
void on_foreign_submit_end(uint64_t frame_id);

// Present Tracking (Presenter Thread)
// - is_real = true: emits VK_LATENCY_MARKER_PRESENT_START/END_NV.
//   VkPresentIdKHR values are independent WSI IDs, not Reflex attribution.
// - is_real = false (synthetic / generated frame): emits OUT_OF_BAND markers with frame_id.
//   This permits multiple generated presents to share the real simulation ID.
void on_present_begin(uint64_t frame_id, bool is_real);
void on_present_end(uint64_t frame_id, bool is_real);

// Diagnostic Query
bool query_latency_timings(LatencyReport* out_report);

}  // namespace host::reflex
