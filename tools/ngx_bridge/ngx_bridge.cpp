// See ngx_bridge.h.
#define NGXB_BUILD
#include "ngx_bridge.h"

#include <cstdarg>
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <cstring>
#include <limits>

#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_helpers_dlssg_vk.h"

namespace {

// Any stable id: NGX keys its per-application data on it.
constexpr char kProjectId[] = "b7d1e0a4-6c2f-4f3e-9a58-2b0dd7c1b109";

std::mutex g_mu;
NgxbLogFn g_log = nullptr;
VkDevice g_device = VK_NULL_HANDLE;
NVSDK_NGX_Parameter* g_params = nullptr;
NVSDK_NGX_Handle* g_feature = nullptr;
NVSDK_NGX_Parameter* g_fg_params = nullptr;
NVSDK_NGX_Handle* g_fg_feature = nullptr;
bool g_fg_available = false;

constexpr NVSDK_NGX_PerfQuality_Value kQualities[] = {
    NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
    NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf,
    NVSDK_NGX_PerfQuality_Value_UltraPerformance};

void fg_release() {
    if (g_fg_feature) NVSDK_NGX_VULKAN_ReleaseFeature(g_fg_feature);
    if (g_fg_params) NVSDK_NGX_VULKAN_DestroyParameters(g_fg_params);
    g_fg_feature = nullptr;
    g_fg_params = nullptr;
}

void logf(const char* fmt, ...) {
    if (!g_log) return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_log(buf);
}

void NVSDK_CONV ngx_log(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    if (!g_log || !message) return;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "ngx: %s", message);
    // NGX ends its lines with a newline; the host's log adds its own.
    for (char* p = buf; *p; ++p) {
        if ((*p == '\n' || *p == '\r') && !p[1]) *p = 0;
    }
    g_log(buf);
}

NVSDK_NGX_Resource_VK resource(const NgxbImage& im, bool rw) {
    VkImageSubresourceRange range{};
    range.aspectMask = im.aspect ? im.aspect : VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    return NVSDK_NGX_Create_ImageView_Resource_VK(im.view, im.image, range, im.format, im.width, im.height, rw);
}

}  // namespace

extern "C" {

int32_t ngxb_version(void) { return NGXB_VERSION; }

uint32_t ngxb_fg_extensions(VkInstance instance, VkPhysicalDevice physical, const wchar_t* dll_dir,
                            uint32_t* count, const VkExtensionProperties** extensions) {
    NVSDK_NGX_FeatureCommonInfo common{};
    const wchar_t* paths[] = {dll_dir};
    common.PathListInfo = {paths, 1};
    NVSDK_NGX_FeatureDiscoveryInfo discovery{};
    discovery.SDKVersion = NVSDK_NGX_Version_API;
    discovery.FeatureID = NVSDK_NGX_Feature_FrameGeneration;
    discovery.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    discovery.Identifier.v.ProjectDesc = {kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0"};
    discovery.FeatureInfo = &common;
    discovery.ApplicationDataPath = dll_dir;
    VkExtensionProperties* required = nullptr;
    const auto r = physical
        ? NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, physical, &discovery, count, &required)
        : NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&discovery, count, &required);
    *extensions = required;
    return static_cast<uint32_t>(r);
}

uint32_t ngxb_required_extensions(uint32_t* instance_count, const char* const** instance_exts, uint32_t* device_count,
                                  const char* const** device_exts) {
    unsigned int ni = 0, nd = 0;
    const char** inst = nullptr;
    const char** dev = nullptr;
    const NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_RequiredExtensions(&ni, &inst, &nd, &dev);
    if (instance_count) *instance_count = NVSDK_NGX_SUCCEED(r) ? ni : 0;
    if (instance_exts) *instance_exts = NVSDK_NGX_SUCCEED(r) ? inst : nullptr;
    if (device_count) *device_count = NVSDK_NGX_SUCCEED(r) ? nd : 0;
    if (device_exts) *device_exts = NVSDK_NGX_SUCCEED(r) ? dev : nullptr;
    return static_cast<uint32_t>(r);
}

uint32_t ngxb_init(VkInstance instance, VkPhysicalDevice physical, VkDevice device, PFN_vkGetInstanceProcAddr gipa,
                   PFN_vkGetDeviceProcAddr gdpa, const wchar_t* dll_dir, const wchar_t* data_dir, NgxbLogFn log) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_log = log;
    static const wchar_t* paths[1];
    paths[0] = dll_dir;
    NVSDK_NGX_FeatureCommonInfo info = {};
    info.PathListInfo.Path = paths;
    info.PathListInfo.Length = 1;
    info.LoggingInfo.LoggingCallback = &ngx_log;
    info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", data_dir, instance,
                                                              physical, device, gipa, gdpa, &info);
    if (NVSDK_NGX_FAILED(r)) {
        logf("NGX init failed: 0x%08X", static_cast<unsigned>(r));
        return static_cast<uint32_t>(r);
    }
    g_device = device;
    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&g_params);
    if (NVSDK_NGX_FAILED(r) || !g_params) {
        logf("NGX capability parameters failed: 0x%08X", static_cast<unsigned>(r));
        NVSDK_NGX_VULKAN_Shutdown1(device);
        g_device = VK_NULL_HANDLE;
        g_params = nullptr;
        return static_cast<uint32_t>(NVSDK_NGX_FAILED(r) ? r : NVSDK_NGX_Result_Fail);
    }
    int available = 0, needs_driver = 0;
    unsigned int major = 0, minor = 0;
    NVSDK_NGX_Parameter_GetI(g_params, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    NVSDK_NGX_Parameter_GetI(g_params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs_driver);
    NVSDK_NGX_Parameter_GetUI(g_params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &major);
    NVSDK_NGX_Parameter_GetUI(g_params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minor);
    if (!available) {
        logf("DLSS not available on this GPU/driver (needs a newer driver: %d, minimum %u.%u)", needs_driver, major, minor);
        NVSDK_NGX_VULKAN_DestroyParameters(g_params);
        NVSDK_NGX_VULKAN_Shutdown1(device);
        g_params = nullptr;
        g_device = VK_NULL_HANDLE;
        return static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_FeatureNotSupported);
    }
    int fg = 0;
    NVSDK_NGX_Parameter_GetI(g_params, NVSDK_NGX_Parameter_FrameGeneration_Available, &fg);
    g_fg_available = fg != 0;
    logf("NGX ready (DLSS available; frame generation %s)", g_fg_available ? "available" : "unavailable");
    return static_cast<uint32_t>(NVSDK_NGX_Result_Success);
}

uint32_t ngxb_create(VkCommandBuffer cmd, uint32_t width, uint32_t height, uint32_t preset, int32_t flags) {
    return ngxb_create_sr(cmd, width, height, width, height, 0, preset, flags);
}

uint32_t ngxb_optimal_settings(uint32_t output_width, uint32_t output_height, int32_t quality, NgxbSettings* settings) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_params) return static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_NotInitialized);
    if (!settings || !output_width || !output_height || quality < 0 || quality >= 5)
        return static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_InvalidParameter);
    *settings = {};
    float sharpness = 0;
    return static_cast<uint32_t>(NGX_DLSS_GET_OPTIMAL_SETTINGS(g_params, output_width, output_height, kQualities[quality],
        &settings->width, &settings->height, &settings->max_width, &settings->max_height,
        &settings->min_width, &settings->min_height, &sharpness));
}

uint32_t ngxb_create_sr(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                        uint32_t output_width, uint32_t output_height,
                        int32_t quality, uint32_t preset, int32_t flags) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_params) return static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_NotInitialized);
    if (g_feature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(g_feature);
        g_feature = nullptr;
    }
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, preset);
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, preset);
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, preset);
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, preset);
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, preset);
    NVSDK_NGX_DLSS_Create_Params p = {};
    p.Feature.InWidth = width;
    p.Feature.InHeight = height;
    p.Feature.InTargetWidth = output_width;
    p.Feature.InTargetHeight = output_height;
    p.Feature.InPerfQualityValue = kQualities[quality >= 0 && quality < 5 ? quality : 0];
    p.InFeatureCreateFlags = flags;
    const NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT1(g_device, cmd, 1, 1, &g_feature, g_params, &p);
    if (NVSDK_NGX_FAILED(r)) {
        logf("creating the DLAA feature (%ux%u, flags 0x%x) failed: 0x%08X", width, height, flags, static_cast<unsigned>(r));
        g_feature = nullptr;
    } else {
        logf("DLSS feature created: %ux%u -> %ux%u, quality %d, preset %u, flags 0x%x",
             width, height, output_width, output_height, quality, preset, flags);
    }
    return static_cast<uint32_t>(r);
}

uint32_t ngxb_evaluate(VkCommandBuffer cmd, const NgxbEval* e) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_feature || !e) return static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_FeatureNotFound);
    NVSDK_NGX_Resource_VK color = resource(e->color, false), output = resource(e->output, true),
                          depth = resource(e->depth, false), motion = resource(e->motion, false);
    NVSDK_NGX_VK_DLSS_Eval_Params p = {};
    p.Feature.pInColor = &color;
    p.Feature.pInOutput = &output;
    p.pInDepth = &depth;
    p.pInMotionVectors = &motion;
    p.InJitterOffsetX = e->jitter_x;
    p.InJitterOffsetY = e->jitter_y;
    p.InRenderSubrectDimensions.Width = e->color.width;
    p.InRenderSubrectDimensions.Height = e->color.height;
    p.InReset = e->reset;
    p.InMVScaleX = e->mv_scale_x;
    p.InMVScaleY = e->mv_scale_y;
    p.InFrameTimeDeltaInMsec = e->frame_ms;
    return static_cast<uint32_t>(NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, g_feature, g_params, &p));
}

void ngxb_release(void) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_feature) NVSDK_NGX_VULKAN_ReleaseFeature(g_feature);
    g_feature = nullptr;
}

void ngxb_shutdown(void) {
    std::lock_guard<std::mutex> lk(g_mu);
    fg_release();
    g_fg_available = false;
    if (g_feature) NVSDK_NGX_VULKAN_ReleaseFeature(g_feature);
    g_feature = nullptr;
    if (g_params) NVSDK_NGX_VULKAN_DestroyParameters(g_params);
    g_params = nullptr;
    if (g_device) NVSDK_NGX_VULKAN_Shutdown1(g_device);
    g_device = VK_NULL_HANDLE;
}

int32_t ngxb_fg_available(void) { return g_device && g_fg_available; }

uint32_t ngxb_fg_max_generated(void) {
    if (!g_device || !g_fg_available || !g_params) return 0;
    unsigned int maximum = 1;
    NVSDK_NGX_Parameter_GetUI(g_params, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &maximum);
    return std::max(1u, maximum);
}

uint32_t ngxb_fg_create(VkCommandBuffer cmd, uint32_t width, uint32_t height, VkFormat format) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_fg_available) return static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_FeatureNotSupported);
    fg_release();
    auto r = NVSDK_NGX_VULKAN_AllocateParameters(&g_fg_params);
    if (NVSDK_NGX_FAILED(r)) return static_cast<uint32_t>(r);
    NVSDK_NGX_DLSSG_Create_Params p{};
    p.Width = width;
    p.Height = height;
    p.NativeBackbufferFormat = format;
    r = NGX_VK_CREATE_DLSSG(cmd, 1, 1, &g_fg_feature, g_fg_params, &p);
    logf("FG create %ux%u: 0x%08x", width, height, static_cast<unsigned>(r));
    return static_cast<uint32_t>(r);
}

// Vulkan NGX FG resource/camera wiring follows AwesomeObserver/bloodborne_pc's
// MIT bridge. Presentation/pacing belongs to the host, not this DLL.
uint32_t ngxb_fg_evaluate(VkCommandBuffer cmd, const NgxbGenerate* f) {
    return ngxb_fg_evaluate_index(cmd, f, 1, 1, 0);
}

uint32_t ngxb_fg_evaluate_index(VkCommandBuffer cmd, const NgxbGenerate* f,
                               uint32_t count, uint32_t index, uint64_t render_frame) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_fg_feature || !f) return static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_FeatureNotFound);
    if (!count || count > ngxb_fg_max_generated() || !index || index > count)
        return static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_InvalidParameter);
    auto color = resource(f->color, false), output = resource(f->output, true),
         real = resource(f->real, true), depth = resource(f->depth, false),
         motion = resource(f->motion, false), hudless = resource(f->hudless, false), ui = resource(f->ui, false);
    auto disable = NVSDK_NGX_Create_Buffer_Resource_VK(f->disable_interpolation, 4, true);
    NVSDK_NGX_VK_DLSSG_Eval_Params images{};
    images.pBackbuffer = &color;
    images.pDepth = &depth;
    images.pMVecs = &motion;
    images.pOutputInterpFrame = &output;
    images.pOutputRealFrame = f->real.image ? &real : nullptr;
    images.pOutputDisableInterpolation = &disable;
    images.pHudless = f->hudless.image ? &hudless : nullptr;
    images.pUI = f->ui.image ? &ui : nullptr;
    NVSDK_NGX_DLSSG_Opt_Eval_Params p{};
    std::memcpy(p.cameraViewToClip, f->camera.view_to_clip, 64);
    std::memcpy(p.clipToCameraView, f->camera.clip_to_view, 64);
    std::memcpy(p.clipToPrevClip, f->camera.clip_to_previous, 64);
    std::memcpy(p.prevClipToClip, f->camera.previous_to_clip, 64);
    for (int i = 0; i < 4; ++i) p.clipToLensClip[i][i] = 1;
    std::memcpy(p.cameraPos, f->camera.position, 12);
    std::memcpy(p.cameraUp, f->camera.up, 12);
    std::memcpy(p.cameraRight, f->camera.right, 12);
    std::memcpy(p.cameraFwd, f->camera.forward, 12);
    p.jitterOffset[0] = f->camera.jitter[0];
    p.jitterOffset[1] = f->camera.jitter[1];
    p.mvecScale[0] = p.mvecScale[1] = 1;
    p.cameraNear = f->camera.near_plane;
    p.cameraFar = f->camera.far_plane;
    p.cameraFOV = f->camera.vertical_fov;
    p.cameraAspectRatio = static_cast<float>(f->motion.width) / f->motion.height;
    p.cameraMotionIncluded = true;
    p.reset = f->reset != 0;
    p.depthInverted = false;
    p.colorBuffersHDR = false;
    p.motionVectorsInvalidValue = std::numeric_limits<float>::max();
    p.multiFrameCount = count;
    p.multiFrameIndex = index;
    if (render_frame) NVSDK_NGX_Parameter_SetULL(g_fg_params, NVSDK_NGX_DLSSG_Parameter_BackbufferFrameID, render_frame);
    return static_cast<uint32_t>(NGX_VK_EVALUATE_DLSSG(cmd, g_fg_feature, g_fg_params, &images, &p));
}

void ngxb_fg_release(void) {
    std::lock_guard<std::mutex> lk(g_mu);
    fg_release();
}

}  // extern "C"
