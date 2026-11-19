/*
 * upscaler_bridge.h -- the public C API of upscaler_bridge.dll (NVIDIA DLSS and AMD FSR behind one small C API).
 *
 * Copyright (c) 2026 Numity1504. MIT licence: see LICENSE.txt. The credit must stay.
 * By Numity1504 (BBsuite), using Claude by Anthropic as assistant.
 *
 * What it is: a Windows x64 dll, built with MSVC and a static CRT, that holds only the calls into NVIDIA's and AMD's code.
 * The host loads it at run time (LoadLibrary + GetProcAddress of one function, ub_get_api), so the host needs no vendor SDK
 * to build, and it can be built with any compiler (Clang or GCC with mingw-w64 included). This header needs only the
 * Vulkan headers. Everything is plain C: fixed-size integers, floats, Vulkan handles and pointers to structures.
 *
 * The order of the calls (INTEGRATION_NOTES.md has the whole story, test/upscaler_bridge_test.cpp is a working example):
 *
 *   ub_get_api                      the one exported function: returns the table of functions below
 *   open_context                    before Vulkan
 *   prepare_instance                right before vkCreateInstance; then instance_done
 *   prepare_device                  right before vkCreateDevice; then device_done
 *   attach_device                   after vkCreateDevice: starts NGX, loads AMD's file
 *   get_caps                        what can run here, the versions, why not
 *   get_render_info / get_jitter    the render size, mip bias and jitter for a method and quality
 *   evaluate                        once a frame, into your command buffer, outside a render pass
 *   detach_device                   after vkDeviceWaitIdle, before vkDestroyDevice
 *   close_context                   at the end
 *
 * Nothing here is fatal for the host: a failure returns an error code and a short reason (last_error), and a crash inside
 * NVIDIA's or AMD's code is caught and turns that method off. On any failure keep your plain path.
 * All functions may be called from any thread, one lock inside; do not call the bridge from inside your log function.
 */
#ifndef UPSCALER_BRIDGE_H
#define UPSCALER_BRIDGE_H

#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UB_API_VERSION 1u

#if defined(_WIN32)
#define UB_CALL __cdecl
#elif defined(__x86_64__)
#define UB_CALL __attribute__((ms_abi))
#else
#define UB_CALL
#endif

typedef struct ub_context ub_context; /* one a process: NGX and FSR keep state per process */

/* results */
typedef int32_t ub_result;
#define UB_OK              0
#define UB_ERR_ARGUMENT    1 /* a bad value from the caller */
#define UB_ERR_STATE       2 /* called in the wrong order */
#define UB_ERR_UNAVAILABLE 3 /* the method cannot run here; get_caps says why */
#define UB_ERR_VENDOR      4 /* NVIDIA's or AMD's code returned an error */
#define UB_ERR_CRASH       5 /* NVIDIA's or AMD's code crashed: caught, that method stays off until close_context */
#define UB_ERR_MEMORY      6
#define UB_ERR_BLOCKED     7 /* a vulkan-1.dll that is not Khronos's loader sits beside the exe: another upscaler, stay off */

/* methods */
#define UB_METHOD_DLSS 1u
#define UB_METHOD_FSR  2u

/* quality: the method's name for a render size */
#define UB_QUALITY_NATIVE      0u /* DLAA / FSR native AA: render size = output size */
#define UB_QUALITY_QUALITY     1u /* 1.5 */
#define UB_QUALITY_BALANCED    2u /* FSR 1.7, DLSS about 1.72 */
#define UB_QUALITY_PERFORMANCE 3u /* 2.0 */

/* ub_open_desc.flags */
#define UB_OPEN_DEBUG_CHECKS (1u << 0) /* AMD's run-time checks of what it is handed; their warnings go to your log function */

/* ub_evaluate_desc.flags. Changing LDR, DEPTH_INVERTED or FSR3 makes a new context, as a new size does */
#define UB_EVAL_RESET          (1u << 0) /* no history this frame */
#define UB_EVAL_LDR            (1u << 1) /* colour is tone-mapped 0..1. Off: linear HDR, the vendor's own auto exposure */
#define UB_EVAL_DEPTH_INVERTED (1u << 2) /* depth: near = 1, far = 0. Off: near = 0, far = 1 */
#define UB_EVAL_FSR3           (1u << 3) /* FSR: the newest 3.x even where FSR 4 could run (AMD's version override) */

/* log levels */
#define UB_LOG_INFO    0
#define UB_LOG_WARNING 1
#define UB_LOG_ERROR   2
typedef void(UB_CALL *ub_log_fn)(void *user, int32_t level, const char *line);

typedef struct ub_open_desc {
    uint32_t struct_size;       /* sizeof(ub_open_desc) */
    uint32_t flags;             /* UB_OPEN_* */
    const char *vendor_folder;  /* UTF-8; holds nvngx_dlss.dll and AMD's two dlls. NULL: the folder of upscaler_bridge.dll */
    const char *data_folder;    /* UTF-8; a folder NGX may write to. NULL: the Windows temp folder */
    const char *project_id;     /* NGX's project GUID. NULL: the bridge's own. Give your own before you ship */
    const char *engine_version; /* NULL: "1.0" */
    ub_log_fn log;              /* may be NULL; may be called from any thread, also while a bridge call runs */
    void *log_user;
} ub_open_desc;

typedef struct ub_device_desc {
    uint32_t struct_size; /* sizeof(ub_device_desc) */
    uint32_t frames_in_flight; /* the most frames you may have on the GPU; a context that was replaced is freed after this many
                                  + 2 evaluate calls. 0: 8 */
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr;
    PFN_vkGetDeviceProcAddr get_device_proc_addr;
} ub_device_desc;

typedef struct ub_caps {
    uint32_t struct_size;    /* sizeof(ub_caps) */
    uint32_t dlss_available; /* 1: evaluate with UB_METHOD_DLSS can run */
    uint32_t fsr_available;  /* 1: evaluate with UB_METHOD_FSR can run */
    uint32_t fsr4_available; /* 1: FSR runs as FSR 4 here */
    uint32_t fsr3_choice;    /* 1: FSR 4 runs and UB_EVAL_FSR3 can pick the newest 3.x instead */
    char bridge_version[32]; /* "1.0.0" */
    char dlss_version[32];   /* nvngx_dlss.dll's version, "310.9.16.0"; "" unknown */
    char fsr_versions[64];   /* what AMD's loader offers this GPU, newest first: "4.1.2, 3.1.6, 2.3.4" */
    char fsr_running[32];    /* the version of the FSR context in use: "3.1.6"; "" none yet */
    char dlss_reason[160];   /* why DLSS cannot run; "" it can */
    char fsr_reason[160];    /* why FSR cannot run, or why it runs as 3.x on a GPU that could run 4 */
} ub_caps;

typedef struct ub_render_info {
    uint32_t struct_size;   /* sizeof(ub_render_info) */
    uint32_t width, height; /* the size to render the 3D at for this output size */
    uint32_t jitter_phases; /* the length of the jitter sequence: hand it to get_jitter */
    float mip_bias;         /* add it to the texture LOD bias of the 3D's materials (never of shadows or post effects) */
} ub_render_info;

typedef struct ub_image {
    VkImage image;
    VkImageView view;     /* a view of one mip and one layer: sampled (output: storage). Depth: the depth aspect only. FSR makes its
                             own views from image and format, DLSS uses this one */
    VkFormat format;      /* the image's format */
    VkImageLayout layout; /* the layout it is in when the command buffer reaches evaluate; the bridge moves it, and puts it back
                             afterwards. Output only: UNDEFINED is allowed, it is left in GENERAL then */
} ub_image;

typedef struct ub_evaluate_desc {
    uint32_t struct_size; /* sizeof(ub_evaluate_desc) */
    uint32_t method;      /* UB_METHOD_* */
    uint32_t quality;     /* UB_QUALITY_* */
    uint32_t flags;       /* UB_EVAL_* */
    uint32_t render_width, render_height; /* color, depth and motion are exactly this size */
    uint32_t output_width, output_height; /* output is exactly this size */
    ub_image color;  /* sampled. RGBA16F or B10G11R11F for HDR, also 8-bit for LDR */
    ub_image depth;  /* sampled. D32_SFLOAT or R32_SFLOAT work for both methods; see INTEGRATION_NOTES.md for D32_SFLOAT_S8_UINT */
    ub_image motion; /* sampled. R16G16_SFLOAT, one vector for each render pixel */
    ub_image output; /* storage. RGBA16F. The bridge writes the whole image */
    float jitter_x, jitter_y; /* render pixels, each in -0.5 .. 0.5. +x: the picture moved right, +y: the picture moved down,
                                 measured on the image you hand in (so a flipped viewport does not change the sign) */
    float motion_scale_x, motion_scale_y; /* motion texel * scale = render pixels, from this pixel to where it was in the last frame.
                                             0 counts as 1 */
    float frame_time_ms; /* since the last evaluate */
    float camera_near, camera_far, camera_fov_y; /* metres, metres, radians; FSR needs them */
    /* Added after 1.0.0 (a struct_size that ends before it reads as 0). DLSS's render preset for every quality: 0 the bridge's own
       (K), else NGX's preset number - 10 J, 11 K, 12 L, 13 M. A change makes a new context. */
    uint32_t dlss_preset;
} ub_evaluate_desc;

/* The table of functions. Every pointer is always set. */
typedef struct ub_api {
    uint32_t struct_size; /* the bridge's sizeof(ub_api) */
    uint32_t api_version; /* UB_API_VERSION */

    /* Before Vulkan. Reads what lies in the vendor folder. */
    ub_result(UB_CALL *open_context)(const ub_open_desc *desc, ub_context **out);
    void(UB_CALL *close_context)(ub_context *c);

    /* The reason for the last error of c, short English. c == NULL: why the last open_context failed. Valid until the next call. */
    const char *(UB_CALL *last_error)(ub_context *c);

    /* Right before vkCreateInstance. *out is what to give vkCreateInstance: in itself, or the bridge's copy with NGX's extensions
       added. Valid until instance_done. It is set on every return, so an error never stops your start. */
    ub_result(UB_CALL *prepare_instance)(ub_context *c, PFN_vkGetInstanceProcAddr gipa, const VkInstanceCreateInfo *in,
                                         const VkInstanceCreateInfo **out);
    /* Right after vkCreateInstance, with its result. If it failed and *out was not in, try again with in. */
    void(UB_CALL *instance_done)(ub_context *c, VkResult result);

    /* Right before vkCreateDevice. *out is what to give vkCreateDevice: in itself, or the bridge's copy with what DLSS needs (an
       NVIDIA card) or FSR 4 needs (an AMD card). A field in a feature structure of your own chain may be turned on in place for the
       call, so your structures must be writable; device_done puts them back. Valid until device_done, set on every return. */
    ub_result(UB_CALL *prepare_device)(ub_context *c, VkInstance instance, VkPhysicalDevice physical_device, const VkDeviceCreateInfo *in,
                                       const VkDeviceCreateInfo **out);
    /* Right after vkCreateDevice, with its result. If it failed and *out was not in, try again with in: the device then has no DLSS or
       FSR 4, and get_caps says so. */
    void(UB_CALL *device_done)(ub_context *c, VkResult result);

    /* After vkCreateDevice. A method that cannot run is no error: get_caps tells. */
    ub_result(UB_CALL *attach_device)(ub_context *c, const ub_device_desc *desc);
    /* Call vkDeviceWaitIdle first. All vendor objects go. */
    void(UB_CALL *detach_device)(ub_context *c);

    ub_result(UB_CALL *get_caps)(ub_context *c, ub_caps *out);

    /* The render size, mip bias and jitter length of a method and quality. Works before attach_device with the plain ratios. */
    ub_result(UB_CALL *get_render_info)(ub_context *c, uint32_t method, uint32_t quality, uint32_t output_width, uint32_t output_height,
                                        ub_render_info *out);

    /* The jitter of a frame: Halton (2, 3), in the units and sign evaluate wants. frame counts up, wrapped by phases. */
    void(UB_CALL *get_jitter)(uint32_t frame, uint32_t phases, float *x, float *y);

    /* Records the upscale into cb: outside a render pass, on a queue that can do compute. The first call for a new size, quality or
       flag also makes the vendor's context, which takes a while the first time (shaders). On return the four images are in the
       layouts you gave, with all writes visible. The bound pipeline, descriptor sets and push constants of cb are gone: set yours
       again. On an error output holds nothing: use your plain path for this frame. */
    ub_result(UB_CALL *evaluate)(ub_context *c, VkCommandBuffer cb, const ub_evaluate_desc *desc);
} ub_api;

/* The dll's one export. NULL when api_version is not one this dll knows. */
typedef const ub_api *(UB_CALL *PFN_ub_get_api)(uint32_t api_version);

#ifdef __cplusplus
}
#endif

#endif /* UPSCALER_BRIDGE_H */
