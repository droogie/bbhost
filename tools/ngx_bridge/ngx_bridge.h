// ngx_bridge: NVIDIA DLSS SR, DLAA and Frame Generation for a Vulkan host.
//
// NVIDIA's NGX static library (nvsdk_ngx_s.lib) is built with MSVC, while
// bbhost is built with Clang against mingw-w64, so the two cannot be linked
// together. This DLL is built with MSVC, links the SDK, and exposes only C
// functions with plain structs, which both ABIs agree on. bbhost loads it with
// LoadLibrary (src/host/dlaa.cpp); every function is optional at run time.
//
// Results are NVSDK_NGX_Result values: success has bit 0 set and no 0xBAD
// prefix; ngxb_failed() tests the same way NVSDK_NGX_FAILED does.
#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NGXB_VERSION 2

typedef struct NgxbImage {
    VkImage image;
    VkImageView view;  // a 2D view of mip 0, layer 0
    VkFormat format;   // the view's format
    uint32_t width, height;
    uint32_t aspect;  // VkImageAspectFlags of the view (depth inputs: VK_IMAGE_ASPECT_DEPTH_BIT)
} NgxbImage;

typedef struct NgxbEval {
    NgxbImage color;   // input, render resolution, VK_IMAGE_LAYOUT_GENERAL
    NgxbImage output;  // storage image, output resolution, VK_IMAGE_LAYOUT_GENERAL
    NgxbImage depth;
    NgxbImage motion;   // RG16F/RG32F, pixels, current -> previous
    float jitter_x, jitter_y;  // render-pixel offsets of this frame's projection
    float mv_scale_x, mv_scale_y;
    int32_t reset;
    float frame_ms;
} NgxbEval;

typedef void (*NgxbLogFn)(const char* message);

// Same camera convention as NVIDIA's FG API: row-major, row vectors, no jitter
// in the matrices. Motion vectors are render-resolution pixels, current to previous.
typedef struct NgxbCamera {
    float view_to_clip[16], clip_to_view[16], clip_to_previous[16], previous_to_clip[16];
    float position[3], up[3], right[3], forward[3];
    float jitter[2], near_plane, far_plane, vertical_fov;
} NgxbCamera;

typedef struct NgxbGenerate {
    NgxbImage color, output, real, depth, motion, hudless, ui;
    NgxbCamera camera;
    VkBuffer disable_interpolation;  // four-byte host-readable flag; wait for GPU completion
    int32_t reset;
} NgxbGenerate;

typedef struct NgxbSettings {
    uint32_t width, height, min_width, min_height, max_width, max_height;
} NgxbSettings;

#if defined(NGXB_BUILD)
#define NGXB_API __declspec(dllexport)
#else
#define NGXB_API
#endif

// The bridge's NGXB_VERSION.
NGXB_API int32_t ngxb_version(void);
// The Vulkan instance and device extensions NGX needs (arrays owned by NGX).
NGXB_API uint32_t ngxb_required_extensions(uint32_t* instance_count, const char* const** instance_exts, uint32_t* device_count,
                                           const char* const** device_exts);
// Feature-specific FG requirements, added before creating the Vulkan device.
// Instance query uses a null physical device. Returned arrays live until the next query.
NGXB_API uint32_t ngxb_fg_extensions(VkInstance instance, VkPhysicalDevice physical, const wchar_t* dll_dir,
                                    uint32_t* count, const VkExtensionProperties** extensions);
// NGX on this device. dll_dir holds nvngx_dlss.dll; data_dir gets NGX's logs.
// Fails (no DLSS) on a GPU without it; `log` gets NGX's messages and ours.
NGXB_API uint32_t ngxb_init(VkInstance instance, VkPhysicalDevice physical, VkDevice device, PFN_vkGetInstanceProcAddr gipa,
                            PFN_vkGetDeviceProcAddr gdpa, const wchar_t* dll_dir, const wchar_t* data_dir, NgxbLogFn log);
// The DLAA feature at width x height (input = output). preset: 0 = default,
// else NVSDK_NGX_DLSS_Hint_Render_Preset_* (10 = J, 11 = K). flags:
// NVSDK_NGX_DLSS_Feature_Flags_*. Records into cmd; releases any earlier one
// (the caller has waited for it to be idle).
NGXB_API uint32_t ngxb_create(VkCommandBuffer cmd, uint32_t width, uint32_t height, uint32_t preset, int32_t flags);
NGXB_API uint32_t ngxb_create_sr(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                uint32_t output_width, uint32_t output_height,
                                int32_t quality, uint32_t preset, int32_t flags);
// quality: 0 DLAA, 1 Quality, 2 Balanced, 3 Performance, 4 Ultra Performance.
// Query NVIDIA's supported render sizes after ngxb_init.
NGXB_API uint32_t ngxb_optimal_settings(uint32_t output_width, uint32_t output_height,
                                       int32_t quality, NgxbSettings* settings);
NGXB_API uint32_t ngxb_evaluate(VkCommandBuffer cmd, const NgxbEval* eval);
NGXB_API int32_t ngxb_fg_available(void);
NGXB_API uint32_t ngxb_fg_create(VkCommandBuffer cmd, uint32_t width, uint32_t height, VkFormat format);
NGXB_API uint32_t ngxb_fg_evaluate(VkCommandBuffer cmd, const NgxbGenerate* frame);
// Optional ABI-2 extensions. Each index requests a distinct temporal position
// from NGX; count is factor-1, index runs from 1 through count for one source pair.
NGXB_API uint32_t ngxb_fg_max_generated(void);
NGXB_API uint32_t ngxb_fg_evaluate_index(VkCommandBuffer cmd, const NgxbGenerate* frame,
                                        uint32_t count, uint32_t index, uint64_t render_frame);
NGXB_API void ngxb_fg_release(void);
NGXB_API void ngxb_release(void);
NGXB_API void ngxb_shutdown(void);

static inline int ngxb_failed(uint32_t r) { return (r & 0xFFF00000u) == 0xBAD00000u; }

typedef int32_t (*PFN_ngxb_version)(void);
typedef uint32_t (*PFN_ngxb_required_extensions)(uint32_t*, const char* const**, uint32_t*, const char* const**);
typedef uint32_t (*PFN_ngxb_init)(VkInstance, VkPhysicalDevice, VkDevice, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr,
                                  const wchar_t*, const wchar_t*, NgxbLogFn);
typedef uint32_t (*PFN_ngxb_create)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, int32_t);
typedef uint32_t (*PFN_ngxb_evaluate)(VkCommandBuffer, const NgxbEval*);
typedef void (*PFN_ngxb_release)(void);
typedef void (*PFN_ngxb_shutdown)(void);
typedef uint32_t (*PFN_ngxb_create_sr)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, uint32_t, int32_t);
typedef uint32_t (*PFN_ngxb_optimal_settings)(uint32_t, uint32_t, int32_t, NgxbSettings*);
typedef uint32_t (*PFN_ngxb_fg_extensions)(VkInstance, VkPhysicalDevice, const wchar_t*, uint32_t*, const VkExtensionProperties**);
typedef int32_t (*PFN_ngxb_fg_available)(void);
typedef uint32_t (*PFN_ngxb_fg_create)(VkCommandBuffer, uint32_t, uint32_t, VkFormat);
typedef uint32_t (*PFN_ngxb_fg_evaluate)(VkCommandBuffer, const NgxbGenerate*);
typedef uint32_t (*PFN_ngxb_fg_max_generated)(void);
typedef uint32_t (*PFN_ngxb_fg_evaluate_index)(VkCommandBuffer, const NgxbGenerate*, uint32_t, uint32_t, uint64_t);
typedef void (*PFN_ngxb_fg_release)(void);

#ifdef __cplusplus
}
#endif
