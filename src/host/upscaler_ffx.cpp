#include "host/upscaler_ffx.h"

#include <algorithm>
#include <cmath>

namespace upscaler_ffx {

void render_size(std::uint32_t quality, std::uint32_t output_w, std::uint32_t output_h, std::uint32_t* w, std::uint32_t* h) {
    static const float kRatio[4] = {1.0f, 1.5f, 1.7f, 2.0f};
    const float ratio = kRatio[std::min<std::uint32_t>(quality, 3)];
    *w = std::max(1u, static_cast<std::uint32_t>(static_cast<float>(output_w) / ratio));
    *h = std::max(1u, static_cast<std::uint32_t>(static_cast<float>(output_h) / ratio));
}

std::uint32_t jitter_phases(std::uint32_t render_w, std::uint32_t output_w) {
    if (!render_w || !output_w) return 8;
    const float scale = static_cast<float>(output_w) / static_cast<float>(render_w);
    return std::max(8u, static_cast<std::uint32_t>(std::ceil(8.0f * scale * scale)));
}

namespace {
float halton(std::uint32_t i, std::uint32_t base) {
    float f = 1.0f, r = 0.0f;
    while (i > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(i % base);
        i /= base;
    }
    return r;
}
}  // namespace

void jitter(std::uint32_t frame, std::uint32_t phases, float* x, float* y) {
    const std::uint32_t i = (phases ? frame % phases : frame) + 1u;
    if (x) *x = halton(i, 2) - 0.5f;
    if (y) *y = halton(i, 3) - 0.5f;
}

}  // namespace upscaler_ffx

#if !defined(__linux__)

const ub_api* upscaler_ffx_api() { return nullptr; }

#else

#include "ffx_api.h"
#include "ffx_api_types.h"
#include "ffx_upscale.h"
#include "vk/ffx_api_vk.h"

#include <dlfcn.h>

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Ctx {
    std::mutex mu;
    ub_log_fn log = nullptr;
    void* log_user = nullptr;
    std::string error;
    // The library.
    void* lib = nullptr;
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    std::string versions;  // what it offers for upscaling, newest first
    // The device.
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;
    PFN_vkCmdPipelineBarrier cmd_barrier = nullptr;
    std::uint32_t in_flight = 8;
    bool available = false;
    std::string reason;
    bool dead = false;  // an error in AMD's code: FSR stays off until close_context
    // The context in use, what it was made for, and the ones replaced, with the evaluate they went at.
    ffxContext fsr = nullptr;
    std::uint32_t rw = 0, rh = 0, ow = 0, oh = 0, flags = 0;
    std::string running;
    std::vector<std::pair<ffxContext, std::uint64_t>> retired;
    std::uint64_t frame = 0;
};

Ctx* g_ctx = nullptr;  // one a process, as the bridge's
std::mutex g_open_mu;
std::string g_open_error;

void say(Ctx& c, std::int32_t level, const char* fmt, ...) {
    if (!c.log) return;
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    c.log(c.log_user, level, line);
}

ub_result fail(Ctx& c, ub_result r, const std::string& why) {
    c.error = why;
    return r;
}

// FSR's own messages (a context's fpMessage, which carries no user pointer): to the log of the open context.
void on_fsr_message(uint32_t type, const wchar_t* message) {
    Ctx* c = g_ctx;
    if (!c || !c->log || !message) return;
    std::string text;
    for (const wchar_t* p = message; *p && text.size() < 400; ++p) text += *p >= 32 && *p < 127 ? static_cast<char>(*p) : '?';
    c->log(c->log_user, type == FFX_API_MESSAGE_TYPE_ERROR ? UB_LOG_ERROR : UB_LOG_WARNING, ("FSR: " + text).c_str());
}

// What the library offers for upscaling (no device needed): "3.1.6".
std::string offered_versions(Ctx& c) {
    uint64_t count = 0;
    ffxQueryDescGetVersions q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.outputCount = &count;
    if (c.query(nullptr, &q.header) != FFX_API_RETURN_OK || count == 0 || count > 16) return "";
    std::vector<uint64_t> ids(count);
    std::vector<const char*> names(count);
    q.versionIds = ids.data();
    q.versionNames = names.data();
    if (c.query(nullptr, &q.header) != FFX_API_RETURN_OK) return "";
    std::string out;
    for (uint64_t i = 0; i < count; ++i) out += std::string(i ? ", " : "") + (names[i] ? names[i] : "?");
    return out;
}

ub_result UB_CALL open_context(const ub_open_desc* desc, ub_context** out) {
    std::lock_guard<std::mutex> lk(g_open_mu);
    g_open_error.clear();
    if (out) *out = nullptr;
    if (!desc || desc->struct_size < sizeof(std::uint32_t) * 2 || !out) {
        g_open_error = "open_context: a NULL pointer or an old struct_size";
        return UB_ERR_ARGUMENT;
    }
    if (g_ctx) {
        g_open_error = "open_context: one context a process, and one is open";
        return UB_ERR_STATE;
    }
    const std::string folder = desc->vendor_folder && *desc->vendor_folder ? desc->vendor_folder : ".";
    const std::string path = folder + "/" + kFfxLibrary;
    // From the executable's own folder, RTLD_LOCAL: nothing of the library's reaches anything else.
    void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        const char* e = dlerror();
        g_open_error = std::string(kFfxLibrary) + " did not load: " + (e ? e : "?");
        return UB_ERR_UNAVAILABLE;
    }
    auto* c = new Ctx();
    c->log = desc->log;
    c->log_user = desc->log_user;
    c->lib = lib;
    c->create = reinterpret_cast<PfnFfxCreateContext>(dlsym(lib, "ffxCreateContext"));
    c->destroy = reinterpret_cast<PfnFfxDestroyContext>(dlsym(lib, "ffxDestroyContext"));
    c->query = reinterpret_cast<PfnFfxQuery>(dlsym(lib, "ffxQuery"));
    c->dispatch = reinterpret_cast<PfnFfxDispatch>(dlsym(lib, "ffxDispatch"));
    if (!c->create || !c->destroy || !c->query || !c->dispatch) {
        g_open_error = path + " lacks the FidelityFX API's functions";
        dlclose(lib);
        delete c;
        return UB_ERR_UNAVAILABLE;
    }
    c->versions = offered_versions(*c);
    if (c->versions.empty()) {
        g_open_error = path + " offers no upscaler";
        dlclose(lib);
        delete c;
        return UB_ERR_UNAVAILABLE;
    }
    g_ctx = c;
    say(*c, UB_LOG_INFO, "bbhost's FidelityFX library %s: FSR %s (AMD FidelityFX SDK, MIT)", path.c_str(), c->versions.c_str());
    *out = reinterpret_cast<ub_context*>(c);
    return UB_OK;
}

Ctx* ctx_of(ub_context* c) { return c && reinterpret_cast<Ctx*>(c) == g_ctx ? g_ctx : nullptr; }

void UB_CALL detach_device(ub_context* cx);

void UB_CALL close_context(ub_context* cx) {
    Ctx* c = ctx_of(cx);
    if (!c) return;
    detach_device(cx);
    std::lock_guard<std::mutex> lk(g_open_mu);
    g_ctx = nullptr;
    // (The library stays loaded: AMD's code keeps state of its own, and nothing is gained by unloading it.)
    delete c;
}

const char* UB_CALL last_error(ub_context* cx) {
    if (!cx) return g_open_error.c_str();
    Ctx* c = ctx_of(cx);
    return c ? c->error.c_str() : "not this library's context";
}

ub_result UB_CALL prepare_instance(ub_context*, PFN_vkGetInstanceProcAddr, const VkInstanceCreateInfo* in, const VkInstanceCreateInfo** out) {
    if (out) *out = in;
    return UB_OK;
}
void UB_CALL instance_done(ub_context*, VkResult) {}
ub_result UB_CALL prepare_device(ub_context*, VkInstance, VkPhysicalDevice, const VkDeviceCreateInfo* in, const VkDeviceCreateInfo** out) {
    if (out) *out = in;
    return UB_OK;
}
void UB_CALL device_done(ub_context*, VkResult) {}

// Whether FSR 3 can run on a device: Vulkan 1.1 (its shaders are SPIR-V 1.3) and subgroup quad operations in compute
// shaders (its two pyramid passes reduce four pixels with them).
std::string device_why_not(const ub_device_desc& d) {
    const auto props2 = d.get_instance_proc_addr
                            ? reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(d.get_instance_proc_addr(d.instance, "vkGetPhysicalDeviceProperties2"))
                            : nullptr;
    if (!props2) return "FSR 3 needs Vulkan 1.1";
    VkPhysicalDeviceSubgroupProperties sg{};
    sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sg;
    props2(d.physical_device, &p2);
    if (p2.properties.apiVersion < VK_API_VERSION_1_1) return "FSR 3 needs Vulkan 1.1";
    if (!(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) || !(sg.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT))
        return "FSR 3 needs subgroup quad operations in compute shaders, which this device does not have";
    return "";
}

ub_result UB_CALL attach_device(ub_context* cx, const ub_device_desc* desc) {
    Ctx* c = ctx_of(cx);
    if (!c) return UB_ERR_ARGUMENT;
    std::lock_guard<std::mutex> lk(c->mu);
    if (!desc || !desc->device || !desc->physical_device || !desc->get_device_proc_addr) return fail(*c, UB_ERR_ARGUMENT, "attach_device: no device");
    if (c->device) return fail(*c, UB_ERR_STATE, "attach_device: a device is attached");
    c->device = desc->device;
    c->phys = desc->physical_device;
    c->gdpa = desc->get_device_proc_addr;
    c->in_flight = desc->frames_in_flight ? desc->frames_in_flight : 8;
    c->cmd_barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(c->gdpa(c->device, "vkCmdPipelineBarrier"));
    c->reason = c->cmd_barrier ? device_why_not(*desc) : "the device has no vkCmdPipelineBarrier";
    c->available = c->reason.empty() && !c->dead;
    return UB_OK;
}

void retire(Ctx& c, bool all) {
    const std::uint64_t wait = c.in_flight + 2;
    for (std::size_t i = 0; i < c.retired.size();) {
        if (all || c.frame > c.retired[i].second + wait) {
            ffxContext h = c.retired[i].first;
            c.destroy(&h, nullptr);
            c.retired.erase(c.retired.begin() + static_cast<std::ptrdiff_t>(i));
        } else {
            ++i;
        }
    }
}

void UB_CALL detach_device(ub_context* cx) {
    Ctx* c = ctx_of(cx);
    if (!c) return;
    std::lock_guard<std::mutex> lk(c->mu);
    if (!c->device) return;
    retire(*c, true);
    if (c->fsr) c->destroy(&c->fsr, nullptr);
    c->fsr = nullptr;
    c->running.clear();
    c->device = VK_NULL_HANDLE;
    c->phys = VK_NULL_HANDLE;
    c->available = false;
}

void copy_text(char* dst, std::size_t n, const std::string& s) {
    std::snprintf(dst, n, "%s", s.c_str());
}

ub_result UB_CALL get_caps(ub_context* cx, ub_caps* out) {
    Ctx* c = ctx_of(cx);
    if (!c || !out || out->struct_size < sizeof(ub_caps)) return UB_ERR_ARGUMENT;
    std::lock_guard<std::mutex> lk(c->mu);
    const std::uint32_t size = out->struct_size;
    std::memset(out, 0, sizeof(ub_caps));
    out->struct_size = size;
    out->fsr_available = c->available && !c->dead;
    copy_text(out->bridge_version, sizeof(out->bridge_version), "bbhost-ffx");
    copy_text(out->fsr_versions, sizeof(out->fsr_versions), c->versions);
    copy_text(out->fsr_running, sizeof(out->fsr_running), c->running);
    copy_text(out->dlss_reason, sizeof(out->dlss_reason), "DLSS is the NGX bridge's");
    copy_text(out->fsr_reason, sizeof(out->fsr_reason), !c->device ? "no device attached" : c->dead ? c->error : c->reason);
    return UB_OK;
}

ub_result UB_CALL get_render_info(ub_context* cx, std::uint32_t method, std::uint32_t quality, std::uint32_t ow, std::uint32_t oh,
                                  ub_render_info* out) {
    if (!ctx_of(cx) || !out || out->struct_size < sizeof(ub_render_info) || method != UB_METHOD_FSR || quality > UB_QUALITY_PERFORMANCE || !ow ||
        !oh)
        return UB_ERR_ARGUMENT;
    std::uint32_t w = 0, h = 0;
    upscaler_ffx::render_size(quality, ow, oh, &w, &h);
    out->width = w;
    out->height = h;
    out->jitter_phases = upscaler_ffx::jitter_phases(w, ow);
    // AMD's docs and NVIDIA's guide give the same formula.
    out->mip_bias = std::log2(static_cast<float>(w) / static_cast<float>(ow)) - 1.0f;
    return UB_OK;
}

void UB_CALL get_jitter(std::uint32_t frame, std::uint32_t phases, float* x, float* y) { upscaler_ffx::jitter(frame, phases, x, y); }

VkImageAspectFlags aspect_of(VkFormat f) {
    if (!ffxApiIsDepthFormat(f)) return VK_IMAGE_ASPECT_COLOR_BIT;
    return ffxApiIsStencilFormat(f) ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT);
}

VkImageMemoryBarrier barrier_of(const ub_image& im, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.image;
    b.subresourceRange = {aspect_of(im.format), 0, 1, 0, 1};
    return b;
}

// What FSR wants: the inputs read-only, the output GENERAL; and back to the caller's layouts after.
void to_fsr(Ctx& c, VkCommandBuffer cb, const ub_evaluate_desc& d) {
    const VkImageLayout ro = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkImageMemoryBarrier b[4] = {
        barrier_of(d.color, d.color.layout, ro, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
        barrier_of(d.depth, d.depth.layout, ro, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
        barrier_of(d.motion, d.motion.layout, ro, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
        barrier_of(d.output, d.output.layout, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT),
    };
    c.cmd_barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 4, b);
}
void from_fsr(Ctx& c, VkCommandBuffer cb, const ub_evaluate_desc& d) {
    const VkImageLayout ro = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkAccessFlags all = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    const VkImageMemoryBarrier b[4] = {
        barrier_of(d.color, ro, d.color.layout, VK_ACCESS_SHADER_READ_BIT, all),
        barrier_of(d.depth, ro, d.depth.layout, VK_ACCESS_SHADER_READ_BIT, all),
        barrier_of(d.motion, ro, d.motion.layout, VK_ACCESS_SHADER_READ_BIT, all),
        barrier_of(d.output, VK_IMAGE_LAYOUT_GENERAL, d.output.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : d.output.layout,
                   VK_ACCESS_MEMORY_WRITE_BIT, all),
    };
    c.cmd_barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 4, b);
}

FfxApiResource fsr_resource(const ub_image& im, std::uint32_t w, std::uint32_t h, std::uint32_t usage, std::uint32_t state) {
    FfxApiResource r{};
    r.resource = reinterpret_cast<void*>(im.image);
    r.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format = ffxApiGetSurfaceFormatVK(im.format);
    r.description.width = w;
    r.description.height = h;
    r.description.depth = 1;
    r.description.mipCount = 1;
    r.description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    r.description.usage = usage;
    r.state = state;
    return r;
}

ub_result fsr_evaluate(Ctx& c, VkCommandBuffer cb, const ub_evaluate_desc& d) {
    // A new size or depth direction: a new context, the old one freed once the GPU is surely past it.
    const std::uint32_t flags = d.flags & (UB_EVAL_LDR | UB_EVAL_DEPTH_INVERTED);
    const std::uint32_t rw = d.render_width, rh = d.render_height, ow = d.output_width, oh = d.output_height;
    if (c.fsr && (c.rw != rw || c.rh != rh || c.ow != ow || c.oh != oh || c.flags != flags)) {
        c.retired.push_back({c.fsr, c.frame});
        c.fsr = nullptr;
    }
    if (!c.fsr) {
        ffxCreateBackendVKDesc be{};
        be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        be.deviceContext.vkDevice = c.device;
        be.deviceContext.vkPhysicalDevice = c.phys;
        be.deviceContext.vkDeviceProcAddr = c.gdpa;
        // (The API version the host is built against, as the Windows bridge names it; FSR 3's provider reads none.)
        ffxCreateContextDescUpscaleVersion uv{};
        uv.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
        uv.version = FFX_UPSCALER_VERSION;
        be.header.pNext = &uv.header;
        ffxCreateContextDescUpscale cd{};
        cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        cd.header.pNext = &be.header;
        cd.flags = (d.flags & UB_EVAL_LDR) ? 0u : static_cast<std::uint32_t>(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE);
        if (d.flags & UB_EVAL_DEPTH_INVERTED) cd.flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
        cd.maxRenderSize = {rw, rh};
        cd.maxUpscaleSize = {ow, oh};
        cd.fpMessage = &on_fsr_message;
        ffxContext made = nullptr;
        const ffxReturnCode_t rc = c.create(&made, &cd.header, nullptr);
        if (rc != FFX_API_RETURN_OK || !made) {
            c.dead = true;
            c.available = false;
            return fail(c, UB_ERR_VENDOR,
                        "FSR could not be made at " + std::to_string(rw) + "x" + std::to_string(rh) + " -> " + std::to_string(ow) + "x" +
                            std::to_string(oh) + " (FidelityFX API error " + std::to_string(rc) + ")");
        }
        c.fsr = made;
        c.rw = rw;
        c.rh = rh;
        c.ow = ow;
        c.oh = oh;
        c.flags = flags;
        ffxQueryGetProviderVersion pv{};
        pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        c.running = c.query(&c.fsr, &pv.header) == FFX_API_RETURN_OK && pv.versionName ? pv.versionName : "?";
        say(c, UB_LOG_INFO, "FSR: made %s at %ux%u -> %ux%u, %s%s", c.running.c_str(), rw, rh, ow, oh,
            (d.flags & UB_EVAL_LDR) ? "LDR" : "HDR with FSR's auto exposure", (d.flags & UB_EVAL_DEPTH_INVERTED) ? ", depth inverted" : "");
    }
    const std::uint32_t read = FFX_API_RESOURCE_STATE_COMPUTE_READ;
    ffxDispatchDescUpscale dd{};
    dd.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    dd.commandList = reinterpret_cast<void*>(cb);
    dd.color = fsr_resource(d.color, rw, rh, FFX_API_RESOURCE_USAGE_READ_ONLY, read);
    std::uint32_t depth_usage = FFX_API_RESOURCE_USAGE_READ_ONLY;
    if (ffxApiIsDepthFormat(d.depth.format)) depth_usage |= FFX_API_RESOURCE_USAGE_DEPTHTARGET;
    if (ffxApiIsStencilFormat(d.depth.format)) depth_usage |= FFX_API_RESOURCE_USAGE_STENCILTARGET;
    dd.depth = fsr_resource(d.depth, rw, rh, depth_usage, read);
    dd.motionVectors = fsr_resource(d.motion, rw, rh, FFX_API_RESOURCE_USAGE_READ_ONLY, read);
    dd.output = fsr_resource(d.output, ow, oh, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    dd.jitterOffset.x = d.jitter_x;
    dd.jitterOffset.y = d.jitter_y;
    dd.motionVectorScale.x = d.motion_scale_x != 0.0f ? d.motion_scale_x : 1.0f;
    dd.motionVectorScale.y = d.motion_scale_y != 0.0f ? d.motion_scale_y : 1.0f;
    dd.renderSize = {rw, rh};
    dd.upscaleSize = {ow, oh};
    dd.enableSharpening = false;  // (FSR's own sharpening stays off: Sharpness is bbhost's, after any upscaler)
    dd.sharpness = 0.0f;
    dd.frameTimeDelta = d.frame_time_ms;
    dd.preExposure = 1.0f;
    dd.reset = (d.flags & UB_EVAL_RESET) != 0;
    dd.cameraNear = d.camera_near;
    dd.cameraFar = d.camera_far;
    dd.cameraFovAngleVertical = d.camera_fov_y;
    dd.viewSpaceToMetersFactor = 1.0f;
    dd.flags = 0;
    const ffxReturnCode_t rc = c.dispatch(&c.fsr, &dd.header);
    if (rc != FFX_API_RETURN_OK) {
        c.dead = true;
        c.available = false;
        return fail(c, UB_ERR_VENDOR, "FSR's dispatch failed at " + std::to_string(rw) + "x" + std::to_string(rh) + " -> " + std::to_string(ow) +
                                          "x" + std::to_string(oh) + " (FidelityFX API error " + std::to_string(rc) + ")");
    }
    return UB_OK;
}

ub_result UB_CALL evaluate(ub_context* cx, VkCommandBuffer cb, const ub_evaluate_desc* d) {
    Ctx* c = ctx_of(cx);
    if (!c) return UB_ERR_ARGUMENT;
    std::lock_guard<std::mutex> lk(c->mu);
    if (!d || d->struct_size < offsetof(ub_evaluate_desc, dlss_preset)) return fail(*c, UB_ERR_ARGUMENT, "evaluate: a NULL pointer or an old struct_size");
    if (!cb) return fail(*c, UB_ERR_ARGUMENT, "evaluate: no command buffer");
    if (!c->device) return fail(*c, UB_ERR_STATE, "evaluate: no device attached");
    if (d->method != UB_METHOD_FSR) return fail(*c, UB_ERR_UNAVAILABLE, "evaluate: only FSR is here");
    if (!c->available) return fail(*c, c->dead ? UB_ERR_VENDOR : UB_ERR_UNAVAILABLE, c->dead ? c->error : c->reason);
    if (!d->render_width || !d->render_height || !d->output_width || !d->output_height) return fail(*c, UB_ERR_ARGUMENT, "evaluate: a size of 0");
    if (d->render_width > d->output_width || d->render_height > d->output_height)
        return fail(*c, UB_ERR_ARGUMENT, "evaluate: the render size is above the output size");
    if (!d->color.image || !d->depth.image || !d->motion.image || !d->output.image) return fail(*c, UB_ERR_ARGUMENT, "evaluate: an image is missing");
    if (d->color.layout == VK_IMAGE_LAYOUT_UNDEFINED || d->depth.layout == VK_IMAGE_LAYOUT_UNDEFINED || d->motion.layout == VK_IMAGE_LAYOUT_UNDEFINED)
        return fail(*c, UB_ERR_ARGUMENT, "evaluate: an input image is given in layout UNDEFINED");
    if (ffxApiGetSurfaceFormatVK(d->color.format) == FFX_API_SURFACE_FORMAT_UNKNOWN || ffxApiGetSurfaceFormatVK(d->depth.format) == FFX_API_SURFACE_FORMAT_UNKNOWN ||
        ffxApiGetSurfaceFormatVK(d->motion.format) == FFX_API_SURFACE_FORMAT_UNKNOWN || ffxApiGetSurfaceFormatVK(d->output.format) == FFX_API_SURFACE_FORMAT_UNKNOWN)
        return fail(*c, UB_ERR_ARGUMENT, "evaluate: a format FSR does not know");
    ++c->frame;
    to_fsr(*c, cb, *d);
    const ub_result r = fsr_evaluate(*c, cb, *d);
    retire(*c, false);
    from_fsr(*c, cb, *d);  // (on an error too: the host's layouts stay true)
    return r;
}

}  // namespace

const ub_api* upscaler_ffx_api() {
    static const ub_api table = {sizeof(ub_api), UB_API_VERSION, &open_context, &close_context, &last_error, &prepare_instance, &instance_done,
                                 &prepare_device, &device_done, &attach_device, &detach_device, &get_caps, &get_render_info, &get_jitter,
                                 &evaluate};
    return &table;
}

#endif
