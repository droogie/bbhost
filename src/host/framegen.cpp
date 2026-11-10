#include "host/framegen.h"

#include "core/config.h"
#include "engine/loading.h"
#include "hle/modules.h"
#include "host/gpu.h"
#include "host/settings.h"
#include "log.h"

#include "ffx_api.h"
#include "ffx_framegeneration.h"
#include "vk/ffx_api_framegeneration_vk.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

PfnFfxCreateContext g_create = nullptr;
PfnFfxDestroyContext g_destroy = nullptr;
PfnFfxConfigure g_configure = nullptr;
PfnFfxQuery g_query = nullptr;
PfnFfxDispatch g_dispatch = nullptr;

ffxContext g_swap_ctx = nullptr;  // AMD's swapchain
ffxContext g_fg_ctx = nullptr;    // frame generation
ffxQueryDescSwapchainReplacementFunctionsVK g_fns{};
VkSwapchainKHR g_swapchain = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
VkPhysicalDevice g_phys = VK_NULL_HANDLE;
VkFormat g_sc_format = VK_FORMAT_UNDEFINED;
VkExtent2D g_sc_extent{};
// The largest render size the frame generation context was made for: the
// prepare of a larger one is refused (the frame presented as it is).
VkExtent2D g_max_render{};

// The game's side: each display buffer's last HUD-less copy and the frame it
// is of (the game draws into at most three in turn).
struct Hudless {
    std::uint64_t display_va = 0, id = 0;
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t w = 0, h = 0;
    std::uint32_t display_w = 0, display_h = 0;
};
std::mutex g_hudless_mu;
// AMD's frame generation context is used from two threads - the prepare on
// the renderer's, the configure on the presenter's - where AMD's sample uses
// it from one: one at a time.
std::mutex g_ctx_mu;
// The call into AMD's swapchain the presenting thread is in, and since when
// (steady clock, ns), for the watchdog (framegen_in_call); null when none.
std::atomic<const char*> g_amd_call{nullptr};
std::atomic<std::uint64_t> g_amd_call_ns{0};
struct AmdCall {
    explicit AmdCall(const char* what) {
        g_amd_call_ns.store(static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()),
                            std::memory_order_relaxed);
        g_amd_call.store(what, std::memory_order_release);
    }
    ~AmdCall() { g_amd_call.store(nullptr, std::memory_order_release); }
};
const char* present_mode_text(VkPresentModeKHR m) {
    switch (m) {
        case VK_PRESENT_MODE_IMMEDIATE_KHR: return "IMMEDIATE";
        case VK_PRESENT_MODE_MAILBOX_KHR: return "MAILBOX";
        case VK_PRESENT_MODE_FIFO_KHR: return "FIFO";
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "FIFO_RELAXED";
        default: return "another mode";
    }
}
// Generation switched on or off in AMD's swapchain (framegen_present): each
// switch waits inside the configure for everything AMD has in flight and
// stops or starts its presenting threads. Counted, the first ones told with
// their time; the presenting thread's.
std::uint64_t g_switches = 0, g_switches_base = 0;
double g_switch_ms_max = 0.0;
bool g_generating_now = false;
// AMD's swapchain drains the device in some calls - a configure that switches
// generation on or off, its swapchain's end (FrameInterpolationSwapchainVK.cpp
// setFrameGenerationConfig -> waitForPresents: vkQueueWaitIdle on its four
// queues, then vkDeviceWaitIdle; destroySwapchain the same) - and
// vkDeviceWaitIdle needs every queue of the device kept from other threads
// while it runs: bb-submit submitting the game's frames to queue 0 meanwhile
// broke that (one laptop run lost the GPU in exactly such a drain, at the
// switch on right after a loading screen).
// Those calls run with the submission thread's queue lock held
// (host_gpu_queue_lock: its pending submissions first, then no submit until
// released); every value AMD's drain waits for is submitted by then - see
// framegen_present. BBHOST_TEST_FG_QUEUE_LOCK=off (a test switch): without it,
// as v0.2.30.
const bool g_queue_lock = [] {
    const char* e = std::getenv("BBHOST_TEST_FG_QUEUE_LOCK");
    const bool off = e && std::strcmp(e, "off") == 0;
    if (off) host_log("framegen: BBHOST_TEST_FG_QUEUE_LOCK=off - AMD's device drains run beside the game's submissions (a test switch)");
    return !off;
}();
struct DrainLock {
    explicit DrainLock(bool take) : held(take && g_queue_lock) {
        if (held) host_gpu_queue_lock();
    }
    ~DrainLock() {
        if (held) host_gpu_queue_unlock();
    }
    const bool held;
};
// Frames AMD's swapchain presented for each of ours over the last 60 of ours:
// 0 until this swapchain has been measured once (framegen_present).
std::atomic<float> g_shown_per_frame{0.0f};
// Its counts, this swapchain's (framegen_present; reset as one is made).
std::uint64_t g_count_ours = 0, g_count_base = 0, g_count_base60 = 0;
// The frames that went without generation and why, this swapchain's too
// (framegen_frame; reset with the counts above).
std::uint64_t g_skip_frames = 0, g_skip_none = 0, g_skip_again = 0;
Hudless g_hudless[4];  // under g_hudless_mu

// The presenter's side: the HUD-less picture at the swapchain's size, and the
// flip being presented with whether everything for it is there.
VkImage g_sc_hudless = VK_NULL_HANDLE;
VkDeviceMemory g_sc_hudless_mem = VK_NULL_HANDLE;
std::uint64_t g_present_id = 0;
bool g_present_ready = false;

// AMD's loader beside the exe, once; it finds the frame generation DLL itself.
bool load() {
    static const bool ok = [] {
#if defined(_WIN32)
        const std::filesystem::path dir = config_exe_dir();
        if (!std::filesystem::exists(dir / "amd_fidelityfx_framegeneration_vk.dll")) {
            host_log("framegen: no amd_fidelityfx_framegeneration_vk.dll beside bbhost.exe: frame generation is not available");
            return false;
        }
        const std::filesystem::path dll = dir / "amd_fidelityfx_loader_vk.dll";
        HMODULE h = LoadLibraryExW(dll.wstring().c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!h) {
            host_log("framegen: %s did not load (error %lu)", dll.string().c_str(), static_cast<unsigned long>(GetLastError()));
            return false;
        }
        const auto get = [h](const char* name) { return reinterpret_cast<void*>(GetProcAddress(h, name)); };
        g_create = reinterpret_cast<PfnFfxCreateContext>(get("ffxCreateContext"));
        g_destroy = reinterpret_cast<PfnFfxDestroyContext>(get("ffxDestroyContext"));
        g_configure = reinterpret_cast<PfnFfxConfigure>(get("ffxConfigure"));
        g_query = reinterpret_cast<PfnFfxQuery>(get("ffxQuery"));
        g_dispatch = reinterpret_cast<PfnFfxDispatch>(get("ffxDispatch"));
        if (!g_create || !g_destroy || !g_configure || !g_query || !g_dispatch) {
            host_log("framegen: %s lacks the FidelityFX API", dll.string().c_str());
            return false;
        }
        host_log("framegen: AMD FidelityFX frame generation loaded");
        return true;
#else
        return false;
#endif
    }();
    return ok;
}

// The swapchain-sized HUD-less image, device-local, made with the swapchain.
bool make_sc_hudless() {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = g_sc_format;
    ici.extent = {g_sc_extent.width, g_sc_extent.height, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (vkCreateImage(g_device, &ici, nullptr, &g_sc_hudless) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g_device, g_sc_hudless, &req);
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
    std::uint32_t type = ~0u;
    for (std::uint32_t i = 0; i < mp.memoryTypeCount && type == ~0u; ++i) {
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) type = i;
    }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (type == ~0u || vkAllocateMemory(g_device, &mai, nullptr, &g_sc_hudless_mem) != VK_SUCCESS ||
        vkBindImageMemory(g_device, g_sc_hudless, g_sc_hudless_mem, 0) != VK_SUCCESS) {
        vkDestroyImage(g_device, g_sc_hudless, nullptr);
        g_sc_hudless = VK_NULL_HANDLE;
        if (g_sc_hudless_mem) vkFreeMemory(g_device, g_sc_hudless_mem, nullptr);
        g_sc_hudless_mem = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

FfxApiResource resource(VkImage image, VkFormat format, std::uint32_t w, std::uint32_t h, VkImageUsageFlags usage, std::uint32_t state) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {w, h, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.usage = usage;
    return ffxApiGetResourceVK(image, ffxApiGetImageResourceDescriptionVK(image, ici, 0), state);
}

ffxReturnCode_t dispatch_generation(ffxDispatchDescFrameGeneration* params, void* ctx) {
    return g_dispatch(static_cast<ffxContext*>(ctx), &params->header);
}

VkQueueInfoFFXAPI queue_info(void* q, std::uint32_t family) {
    VkQueueInfoFFXAPI i{};
    i.queue = static_cast<VkQueue>(q);
    i.familyIndex = family;
    return i;
}

}  // namespace

bool framegen_wanted() {
    // Live: the presenter asks each time it makes its swapchain, and a change
    // of the setting makes it again (host/options.cpp).
    return host_settings().frame_generation && load();
}

bool framegen_create_swapchain(VkPhysicalDevice phys, VkDevice device, const VkSwapchainCreateInfoKHR& sci, VkQueue game_queue,
                               std::uint32_t family, void* const fg_queues[3], VkSwapchainKHR* out) {
    if (!framegen_wanted() || !game_queue || !fg_queues[0] || !fg_queues[1] || !fg_queues[2]) return false;
    ffxCreateContextDescFrameGenerationSwapChainVK sc{};
    sc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
    sc.physicalDevice = phys;
    sc.device = device;
    VkSwapchainKHR made = VK_NULL_HANDLE;
    sc.swapchain = &made;
    sc.createInfo = sci;
    sc.createInfo.oldSwapchain = VK_NULL_HANDLE;
    sc.gameQueue = queue_info(game_queue, family);
    sc.presentQueue = queue_info(fg_queues[0], family);
    sc.imageAcquireQueue = queue_info(fg_queues[1], family);
    sc.asyncComputeQueue = queue_info(fg_queues[2], family);
    if (g_create(&g_swap_ctx, &sc.header, nullptr) != FFX_API_RETURN_OK || !made) {
        host_log("framegen: AMD's swapchain was not made; the presenter makes its own");
        g_swap_ctx = nullptr;
        return false;
    }
    g_fns = {};
    g_fns.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    if (g_query(&g_swap_ctx, &g_fns.header) != FFX_API_RETURN_OK || !g_fns.pOutAcquireNextImageKHR || !g_fns.pOutQueuePresentKHR ||
        !g_fns.pOutGetSwapchainImagesKHR || !g_fns.pOutDestroySwapchainFFXAPI) {
        host_log("framegen: AMD's swapchain gave no replacement functions");
        g_destroy(&g_swap_ctx, nullptr);
        g_swap_ctx = nullptr;
        return false;
    }
    // The frame generation context, on the backend of this device.
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.deviceContext.vkDevice = device;
    backend.deviceContext.vkPhysicalDevice = phys;
    backend.deviceContext.vkDeviceProcAddr = vkGetDeviceProcAddr;
    ffxCreateContextDescFrameGeneration fg{};
    fg.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    fg.header.pNext = &backend.header;
    fg.flags = FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT;
    fg.displaySize = {sci.imageExtent.width, sci.imageExtent.height};
    // The render size can be above the display's: a Resolution larger than the
    // window (2560x1440 on a 1080p screen). With the display's size as the
    // most, AMD read only the top-left 1920x1080 of the 2560x1440 depth and
    // motion - the rest of the picture (the right and bottom quarter on
    // screen) generated from nothing, blurred blocks on every generated frame.
    const HostSettings& hs = host_settings_ref();
    fg.maxRenderSize = {std::max<std::uint32_t>(sci.imageExtent.width, static_cast<std::uint32_t>(std::max(hs.res_width, 0))),
                        std::max<std::uint32_t>(sci.imageExtent.height, static_cast<std::uint32_t>(std::max(hs.res_height, 0)))};
    fg.backBufferFormat = ffxApiGetSurfaceFormatVK(sci.imageFormat);
    if (g_create(&g_fg_ctx, &fg.header, nullptr) != FFX_API_RETURN_OK) {
        host_log("framegen: the frame generation context was not made; AMD's swapchain presents every frame as it comes");
        g_fg_ctx = nullptr;
    }
    g_max_render = {fg.maxRenderSize.width, fg.maxRenderSize.height};
    // The counts start again with this swapchain: the last one's factor shown
    // for its first ticks was the FPS counter's bracket after a toggle.
    g_shown_per_frame.store(0.0f, std::memory_order_relaxed);
    g_count_ours = g_count_base = g_count_base60 = 0;
    g_skip_frames = g_skip_none = g_skip_again = 0;
    g_swapchain = made;
    g_device = device;
    g_phys = phys;
    g_sc_format = sci.imageFormat;
    g_sc_extent = sci.imageExtent;
    if (g_fg_ctx && !make_sc_hudless()) host_log("framegen: no image for the HUD-less picture; frames are presented as they are");
    *out = made;
#if defined(_WIN32)
    for (const wchar_t* m : {L"amd_fidelityfx_framegeneration_vk.dll", L"amd_fidelityfx_loader_vk.dll", L"amdvlk64.dll", L"amdxc64.dll",
                             L"vulkan-1.dll", L"upscaler_bridge.dll", L"amd_fidelityfx_upscaler_vk.dll"})
        if (HMODULE h = GetModuleHandleW(m)) host_log("framegen: module %ls at %p", m, static_cast<void*>(h));
#endif
    host_log("framegen: AMD's swapchain %ux%u in place, presenting %s, render sizes up to %ux%u (queues 2-4 of family %u)%s",
             sci.imageExtent.width, sci.imageExtent.height, present_mode_text(sci.presentMode), g_max_render.width, g_max_render.height,
             family, g_fg_ctx ? ", frame generation on for the flips whose motion was prepared" : "");
    g_generating_now = false;
    g_switches = g_switches_base = 0;
    g_switch_ms_max = 0.0;
    return true;
}

bool framegen_active() { return g_swap_ctx != nullptr; }

const char* framegen_in_call(double& seconds) {
    const char* const what = g_amd_call.load(std::memory_order_acquire);
    if (!what) return nullptr;
    const std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::duration(g_amd_call_ns.load(std::memory_order_relaxed))};
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return what;
}

float framegen_shown_per_frame() { return g_swap_ctx ? g_shown_per_frame.load(std::memory_order_relaxed) : 1.0f; }  // 0: not measured yet

VkResult framegen_get_images(VkDevice device, VkSwapchainKHR swapchain, std::uint32_t* count, VkImage* images) {
    return g_fns.pOutGetSwapchainImagesKHR(device, swapchain, count, images);
}

VkResult framegen_acquire(VkDevice device, VkSwapchainKHR swapchain, std::uint64_t timeout, VkSemaphore sem, VkFence fence,
                          std::uint32_t* index) {
    return g_fns.pOutAcquireNextImageKHR(device, swapchain, timeout, sem, fence, index);
}

VkResult framegen_present(VkQueue queue, const VkPresentInfoKHR* info) {
    if (g_fg_ctx) {
        // Generated only for a flip with its prepare and its HUD-less picture:
        // AMD takes the UI out of the frame against that picture and puts it
        // over every frame it shows.
        // Not while a loading screen is up: its frames run uncapped to load
        // faster, and there is nothing to smooth.
        const bool ready = g_present_ready && !loading_screen_up();
        // Switched on only after that many such flips in a row, off at the
        // first without (BBHOST_FG_DEBOUNCE=N; 0: on at the first, as
        // v0.2.30). A switch waits inside AMD's configure for all it has in
        // flight and starts or stops its presenting threads; at a loading
        // screen's end that was the world's first flip, its frame times still
        // uneven, where a player saw "Not Responding" (Display-Limited with
        // frame generation). Now the switch on comes 30 steady flips later. (A
        // loading screen still makes one switch off and one on: the flips that
        // show a display buffer again fall inside it, generation off already.)
        static const int need = [] {
            const char* e = std::getenv("BBHOST_FG_DEBOUNCE");
            const int n = e && *e ? std::atoi(e) : 30;
            if (n > 1)
                host_log("framegen: generation switched on after %d new frames in a row (BBHOST_FG_DEBOUNCE=0: at the first, as v0.2.30)", n);
            else
                host_log("framegen: BBHOST_FG_DEBOUNCE=%s - generation switched on at the first new frame, as v0.2.30", e ? e : "");
            return n > 1 ? n : 0;
        }();
        // BBHOST_TEST_FG_SWITCH=N (a test switch): one present in N taken as not
        // ready - generation switched off there and on again by the rule above,
        // hundreds of switches a run.
        static const int test_switch = [] {
            const char* e = std::getenv("BBHOST_TEST_FG_SWITCH");
            const int n = e ? std::atoi(e) : 0;
            if (n > 1) host_log("framegen: BBHOST_TEST_FG_SWITCH=%d - one present in %d taken as not ready (a test switch)", n, n);
            return n > 1 ? n : 0;
        }();
        static std::uint64_t presents = 0;
        const bool forced_off = test_switch && ++presents % static_cast<std::uint64_t>(test_switch) == 0;
        static int in_a_row = 0;
        in_a_row = ready && !forced_off ? std::min(in_a_row + 1, need) : 0;  // held at need: no overflow in a long session
        const bool on = ready && !forced_off && (g_generating_now || in_a_row >= need);
        ffxConfigureDescFrameGeneration cfg{};
        cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cfg.swapChain = g_swapchain;
        cfg.frameGenerationEnabled = on;
        // The generation on AMD's async compute queue (queue 4), beside the
        // game's next frame. On the present queue it waited behind that frame's
        // work whenever the GPU was full, and the generated frame came out late:
        // shown 9.5 / 14.5 ms in turn where 12 / 12 was asked (2026-11-06, 960
        // cap, ~95 real frames). With it: real and generated frames' screen
        // time 0.45 -> 0.16 apart (the median pair's |a - b| / (a + b)), display
        // latency p95 16 -> 10.5 ms. BBHOST_FG_ASYNC=0: on the present queue.
        static const bool async = [] {
            const char* e = std::getenv("BBHOST_FG_ASYNC");
            return !(e && e[0] == '0');
        }();
        cfg.allowAsyncWorkloads = async;
        cfg.frameGenerationCallback = dispatch_generation;
        cfg.frameGenerationCallbackUserContext = &g_fg_ctx;
        if (on)
            cfg.HUDLessColor = resource(g_sc_hudless, g_sc_format, g_sc_extent.width, g_sc_extent.height,
                                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, FFX_API_RESOURCE_STATE_COMMON);
        cfg.generationRect = {0, 0, static_cast<std::int32_t>(g_sc_extent.width), static_cast<std::int32_t>(g_sc_extent.height)};
        cfg.frameID = g_present_id;
        static std::atomic<int> told{0};
        ffxReturnCode_t rc;
        const auto t0 = std::chrono::steady_clock::now();
        auto t_locked = t0;  // when the queue lock was held (a switch's)
        bool locked = false;
        {
            // A switch drains the device inside the configure: the game's
            // submissions held off meanwhile (g_queue_lock). Everything it
            // waits for is submitted by now, so the lock cannot hold up a
            // value the drain needs: AMD's game, interpolation and present
            // semaphores are signalled by its own submits - made inside our
            // earlier presents on this thread, or by its presenting thread,
            // which takes none of our locks - and the game's work they wait on,
            // our blit's `rendered`, went in before the last present
            // (host_gpu_wait_submitted in vk_present); what queue 0 holds waits
            // only on queue 0's earlier work and on AMD's acquire semaphore,
            // signalled by a submit made at that acquire. The queue lock first,
            // outside g_ctx_mu: its wait for the submission thread's pending
            // jobs needs nothing the context's lock guards.
            const DrainLock drain(on != g_generating_now);
            locked = drain.held;
            t_locked = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lk(g_ctx_mu);
            const AmdCall in("configuring frame generation");
            rc = g_configure(&g_fg_ctx, &cfg.header);
        }
        // A switch - generation on or off in AMD's swapchain - waits inside the
        // configure for all AMD has in flight (its presents, its queues, the
        // device) and starts or stops its presenting threads: the first 24 told
        // with their time and why, and after them any that held this thread
        // over 40 ms (48 more at most), then a count every 600 of our frames
        // with the longest hold (the queue lock's wait and the configure).
        if (rc == FFX_API_RETURN_OK && on != g_generating_now) {
            g_generating_now = on;
            ++g_switches;
            const auto t1 = std::chrono::steady_clock::now();
            const double held = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double amd = std::chrono::duration<double, std::milli>(t1 - t_locked).count();
            g_switch_ms_max = std::max(g_switch_ms_max, held);
            static int said = 0, said_slow = 0;
            if (said < 24 || (held > 40.0 && said_slow < 48)) {
                if (said < 24)
                    ++said;
                else
                    ++said_slow;
                char wait[64] = "";
                if (locked) std::snprintf(wait, sizeof(wait), ", after the queue lock's wait of %.1f ms", held - amd);
                host_log("framegen: generation switched %s at flip %llu, frame %llu (%s), AMD's configure %.1f ms%s", on ? "on" : "off",
                         static_cast<unsigned long long>(hle_video_flip_count()), static_cast<unsigned long long>(g_present_id),
                         on ? (need ? "the new frames in a row" : "prepared, with its HUD-less picture")
                            : (forced_off ? "BBHOST_TEST_FG_SWITCH" : loading_screen_up() ? "a loading screen" : "no fresh HUD-less picture"),
                         amd, wait);
            }
        }
        if (rc != FFX_API_RETURN_OK) {
            if (told.fetch_add(1) < 4) host_log("framegen: configuring frame %llu failed", static_cast<unsigned long long>(g_present_id));
        } else if (on) {
            static bool first = true;
            if (first) host_log("framegen: generating, first at frame %llu", static_cast<unsigned long long>(g_present_id));
            first = false;
        }
    }
    // The generation runs inside the present (its callback, on this thread):
    // not while the renderer's prepare is in the context.
    VkResult r;
    {
        std::lock_guard<std::mutex> lk(g_ctx_mu);
        const AmdCall in("presenting");
        r = g_fns.pOutQueuePresentKHR(queue, info);
    }
    // What reaches the screen, real and generated, against the game's frames:
    // the FPS counter's factor every 60, the log's line every 600.
    // (AMD's count of its presents, real and generated - what it handed the
    // display, not a measure of what the display showed - against ours; this
    // swapchain's alone: the counts start again as one is made.)
    if (g_fns.pOutGetLastPresentCountFFXAPI && (g_count_ours + 1) % 60 == 0) {
        const std::uint64_t shown = g_fns.pOutGetLastPresentCountFFXAPI(g_swapchain);
        if (g_count_base60 && shown > g_count_base60)
            g_shown_per_frame.store(static_cast<float>(shown - g_count_base60) / 60.0f, std::memory_order_relaxed);
        g_count_base60 = shown ? shown : 1;
    }
    if (g_fns.pOutGetLastPresentCountFFXAPI && ++g_count_ours % 600 == 0) {
        const std::uint64_t shown = g_fns.pOutGetLastPresentCountFFXAPI(g_swapchain);
        if (g_switches > g_switches_base)
            host_log("framegen: AMD presented %llu frames for the last 600 of ours; generation switched %llu times, the longest switch %.1f ms",
                     static_cast<unsigned long long>(shown - g_count_base), static_cast<unsigned long long>(g_switches - g_switches_base),
                     g_switch_ms_max);
        else
            host_log("framegen: AMD presented %llu frames for the last 600 of ours", static_cast<unsigned long long>(shown - g_count_base));
        g_count_base = shown;
        g_switches_base = g_switches;
        g_switch_ms_max = 0.0;
    }
    return r;
}

bool framegen_prepare(VkCommandBuffer cb, std::uint64_t id, const FramegenInputs& in) {
    if (!g_fg_ctx || !in.depth || !in.motion) return false;
    if (in.render_w > g_max_render.width || in.render_h > g_max_render.height) {
        // A Resolution raised past the context's size since it was made: no
        // generation until the swapchain is made again.
        static std::atomic<int> told{0};
        if (told.fetch_add(1) < 2)
            host_log("framegen: render size %ux%u is above the %ux%u frame generation was made for: frames presented as they are",
                     in.render_w, in.render_h, g_max_render.width, g_max_render.height);
        return false;
    }
    // AMD's backend reads an image in the layout its state names: depth goes
    // to GENERAL for it (COMMON) and back.
    const bool move = in.depth_layout != VK_IMAGE_LAYOUT_GENERAL;
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = in.depth;
    b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    if (in.depth_format == VK_FORMAT_D32_SFLOAT_S8_UINT || in.depth_format == VK_FORMAT_D24_UNORM_S8_UINT)
        b.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    if (move) {
        b.oldLayout = in.depth_layout;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
    ffxDispatchDescFrameGenerationPrepareV2 p{};
    p.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
    p.frameID = id;
    p.commandList = cb;
    p.renderSize = {in.render_w, in.render_h};
    p.jitterOffset = {in.jitter_x, in.jitter_y};
    p.motionVectorScale = {in.motion_scale, in.motion_scale};  // render pixels already
    p.frameTimeDelta = in.frame_ms;
    p.cameraNear = in.near_m;
    p.cameraFar = in.far_m;
    p.cameraFovAngleVertical = in.fov_y;
    p.viewSpaceToMetersFactor = 1.0f;
    p.depth = resource(in.depth, in.depth_format, in.render_w, in.render_h, VK_IMAGE_USAGE_SAMPLED_BIT, FFX_API_RESOURCE_STATE_COMMON);
    p.motionVectors = resource(in.motion, VK_FORMAT_R16G16_SFLOAT, in.render_w, in.render_h,
                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, FFX_API_RESOURCE_STATE_COMMON);
    bool ok;
    {
        std::lock_guard<std::mutex> lk(g_ctx_mu);
        ok = g_dispatch(&g_fg_ctx, &p.header) == FFX_API_RETURN_OK;
    }
    if (move) {
        std::swap(b.oldLayout, b.newLayout);
        b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
    static std::atomic<int> told{0};
    if (!ok && told.fetch_add(1) < 4) host_log("framegen: the prepare for frame %llu failed", static_cast<unsigned long long>(id));
    return ok;
}

void framegen_note_hudless(std::uint64_t display_va, std::uint64_t id, VkImage image, VkFormat format, std::uint32_t w,
                           std::uint32_t h, std::uint32_t display_w, std::uint32_t display_h) {
    std::lock_guard<std::mutex> lk(g_hudless_mu);
    Hudless* slot = &g_hudless[0];
    for (Hudless& s : g_hudless) {
        if (s.display_va == display_va) { slot = &s; break; }
        if (s.id < slot->id) slot = &s;  // else the oldest
    }
    *slot = Hudless{display_va, id, image, format, w, h, display_w ? display_w : w, display_h ? display_h : h};
    static bool told = false;
    if (!told) host_log("framegen: first HUD-less copy, frame %llu, display buffer 0x%llx %ux%u", static_cast<unsigned long long>(id),
                        static_cast<unsigned long long>(display_va), w, h);
    told = true;
}

void framegen_frame(VkCommandBuffer cmd, std::uint64_t display_va, VkOffset2D dst, VkExtent2D dst_size, std::int32_t src_x,
                    std::int32_t src_y, std::uint32_t src_w, std::uint32_t src_h, bool fsr) {
    g_present_ready = false;
    if (!g_fg_ctx || !g_sc_hudless) return;
    Hudless h;
    {
        std::lock_guard<std::mutex> lk(g_hudless_mu);
        for (const Hudless& s : g_hudless) {
            if (s.display_va == display_va) h = s;
        }
    }
    // Why frames go without generation, every 600 presents (beside the log's
    // "AMD presented" line, over this swapchain's frames as it is).
    if (++g_skip_frames % 600 == 0 && (g_skip_none || g_skip_again)) {
        host_log("framegen: of the last 600 frames %llu had no HUD-less copy, %llu were shown before (last prepared %llu, presented %llu)",
                 static_cast<unsigned long long>(g_skip_none), static_cast<unsigned long long>(g_skip_again),
                 static_cast<unsigned long long>(h.id), static_cast<unsigned long long>(g_present_id));
        g_skip_none = g_skip_again = 0;
    }
    // A frame seen before (the same buffer shown again) is not one to generate after.
    if (!h.image || h.id <= g_present_id) {
        ++(h.image ? g_skip_again : g_skip_none);
        static std::atomic<int> told{0};
        if (told.fetch_add(1) < 3)
            host_log("framegen: no HUD-less copy to present display buffer 0x%llx with (%s)", static_cast<unsigned long long>(display_va),
                     h.image ? "shown before" : "none for it");
        return;
    }
    g_present_id = h.id;
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = g_sc_hudless;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    // Black around the picture, as the frame's own edges are.
    const VkClearColorValue black{{0.0f, 0.0f, 0.0f, 1.0f}};
    vkCmdClearColorImage(cmd, g_sc_hudless, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &b.subresourceRange);
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    // The rectangles are the display buffer's; the copy is the HUD's target,
    // which the display pass stretches over the whole display buffer.
    const double kx = static_cast<double>(h.w) / h.display_w, ky = static_cast<double>(h.h) / h.display_h;
    VkImageBlit r{};
    r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.srcOffsets[0] = {static_cast<std::int32_t>(src_x * kx), static_cast<std::int32_t>(src_y * ky), 0};
    r.srcOffsets[1] = {static_cast<std::int32_t>((src_x + static_cast<std::int32_t>(src_w)) * kx),
                       static_cast<std::int32_t>((src_y + static_cast<std::int32_t>(src_h)) * ky), 1};
    r.dstOffsets[0] = {dst.x, dst.y, 0};
    r.dstOffsets[1] = {dst.x + static_cast<std::int32_t>(dst_size.width), dst.y + static_cast<std::int32_t>(dst_size.height), 1};
    const bool same = r.srcOffsets[1].x - r.srcOffsets[0].x == static_cast<std::int32_t>(dst_size.width) &&
                      r.srcOffsets[1].y - r.srcOffsets[0].y == static_cast<std::int32_t>(dst_size.height);
    // A frame FSR 1 upscaled (a picture smaller than its rectangle: Upscaling
    // Off or TAA at a Resolution below the window's): its HUD-less picture by
    // FSR 1 too, from the same rectangle into the same one. AMD takes whatever
    // differs between the frame and this picture for the UI and puts it back
    // as it is over the generated frames; stretched by a blit, the picture
    // differed at every edge EASU drew and every detail RCAS sharpened - most
    // of the 3D, which so moved at the real frames' rate (the generated frames
    // presented, the screen as smooth as without them).
    bool by_fsr = false;
    if (fsr && !same) {
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        by_fsr = host_gpu_fsr_upscale(cmd, h.image, static_cast<std::uint32_t>(h.format), h.w, h.h,
                                      static_cast<std::uint32_t>(r.srcOffsets[0].x), static_cast<std::uint32_t>(r.srcOffsets[0].y),
                                      static_cast<std::uint32_t>(r.srcOffsets[1].x - r.srcOffsets[0].x),
                                      static_cast<std::uint32_t>(r.srcOffsets[1].y - r.srcOffsets[0].y), g_sc_hudless, dst.x, dst.y,
                                      dst_size.width, dst_size.height);
        static bool told = false;
        if (!told)
            host_log("framegen: the frame is upscaled by FSR 1, and its HUD-less picture %s (%dx%d to %ux%u)",
                     by_fsr ? "too" : "could not be: stretched", r.srcOffsets[1].x - r.srcOffsets[0].x,
                     r.srcOffsets[1].y - r.srcOffsets[0].y, dst_size.width, dst_size.height);
        told = true;
    }
    if (!by_fsr)
        vkCmdBlitImage(cmd, h.image, VK_IMAGE_LAYOUT_GENERAL, g_sc_hudless, b.oldLayout, 1, &r, same ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    g_present_ready = true;
}

void framegen_destroy_swapchain(VkDevice device) {
    if (!g_swap_ctx) return;
    {
        // AMD's swapchain's end drains the device (destroySwapchain ->
        // waitForPresents) before it stops its threads: the game's submissions
        // held off meanwhile (g_queue_lock; what it waits for is submitted, as
        // for a switching configure). Its callers hold the renderer's lock, so
        // no prepare is in the frame generation context either.
        const DrainLock drain(true);
        if (g_fg_ctx) g_destroy(&g_fg_ctx, nullptr);
        g_fg_ctx = nullptr;
        {
            const AmdCall in("ending its swapchain");  // its threads stopped, its presents waited for
            g_destroy(&g_swap_ctx, nullptr);
        }
        g_swap_ctx = nullptr;
        // After the contexts, as AMD's replacement asks.
        g_fns.pOutDestroySwapchainFFXAPI(device, g_swapchain, nullptr, nullptr);
    }
    g_swapchain = VK_NULL_HANDLE;
    if (g_sc_hudless) vkDestroyImage(device, g_sc_hudless, nullptr);
    if (g_sc_hudless_mem) vkFreeMemory(device, g_sc_hudless_mem, nullptr);
    g_sc_hudless = VK_NULL_HANDLE;
    g_sc_hudless_mem = VK_NULL_HANDLE;
    host_log("framegen: AMD's swapchain gone");
}
