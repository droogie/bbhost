#include "host/framegen.h"

#include "core/config.h"
#include "host/settings.h"
#include "log.h"

#include "ffx_api.h"
#include "ffx_framegeneration.h"
#include "vk/ffx_api_framegeneration_vk.h"

#include <cstring>
#include <filesystem>

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
std::uint64_t g_frame_id = 0;

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

VkQueueInfoFFXAPI queue_info(void* q, std::uint32_t family) {
    VkQueueInfoFFXAPI i{};
    i.queue = static_cast<VkQueue>(q);
    i.familyIndex = family;
    return i;
}

}  // namespace

bool framegen_wanted() {
    static const bool want = host_settings().frame_generation;
    return want && load();
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
    fg.maxRenderSize = fg.displaySize;
    fg.backBufferFormat = ffxApiGetSurfaceFormatVK(sci.imageFormat);
    if (g_create(&g_fg_ctx, &fg.header, nullptr) != FFX_API_RETURN_OK) {
        host_log("framegen: the frame generation context was not made; AMD's swapchain presents every frame as it comes");
        g_fg_ctx = nullptr;
    }
    g_swapchain = made;
    g_frame_id = 0;
    *out = made;
    host_log("framegen: AMD's swapchain %ux%u in place (queues 2-4 of family %u)%s", sci.imageExtent.width, sci.imageExtent.height,
             family, g_fg_ctx ? ", frame generation context made - generation off in this step" : "");
    return true;
}

bool framegen_active() { return g_swap_ctx != nullptr; }

VkResult framegen_get_images(VkDevice device, VkSwapchainKHR swapchain, std::uint32_t* count, VkImage* images) {
    return g_fns.pOutGetSwapchainImagesKHR(device, swapchain, count, images);
}

VkResult framegen_acquire(VkDevice device, VkSwapchainKHR swapchain, std::uint64_t timeout, VkSemaphore sem, VkFence fence,
                          std::uint32_t* index) {
    return g_fns.pOutAcquireNextImageKHR(device, swapchain, timeout, sem, fence, index);
}

VkResult framegen_present(VkQueue queue, const VkPresentInfoKHR* info) {
    if (g_fg_ctx) {
        // Generation off for now: AMD's swapchain shows each frame as it is.
        ffxConfigureDescFrameGeneration cfg{};
        cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cfg.swapChain = g_swapchain;
        cfg.frameGenerationEnabled = false;
        cfg.frameID = g_frame_id++;
        g_configure(&g_fg_ctx, &cfg.header);
    }
    return g_fns.pOutQueuePresentKHR(queue, info);
}

void framegen_destroy_swapchain(VkDevice device) {
    if (!g_swap_ctx) return;
    if (g_fg_ctx) g_destroy(&g_fg_ctx, nullptr);
    g_fg_ctx = nullptr;
    g_destroy(&g_swap_ctx, nullptr);
    g_swap_ctx = nullptr;
    // After the contexts, as AMD's replacement asks.
    g_fns.pOutDestroySwapchainFFXAPI(device, g_swapchain, nullptr, nullptr);
    g_swapchain = VK_NULL_HANDLE;
    host_log("framegen: AMD's swapchain gone");
}
