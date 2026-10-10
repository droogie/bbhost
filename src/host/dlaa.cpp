// See dlaa.h.
#include "host/dlaa.h"

#include "core/config.h"
#include "hle/modules.h"
#include "host/gpu_internal.h"
#include "host/object_motion.h"
#include "host/settings.h"
#include "host/display_settings.h"
#include "host/fsr_upscaler.h"
#include "host/shaders/dlaa_mv.spv.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <map>
#include <set>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "../../tools/ngx_bridge/ngx_bridge.h"

namespace gpu {
namespace {

[[maybe_unused]] constexpr int kBridgeVersion = NGXB_VERSION;
bool ngx_failed(std::uint32_t r) { return (r & 0xFFF00000u) == 0xBAD00000u; }

// NVSDK_NGX_DLSS_Feature_Flags: IsHDR | MVLowRes | AutoExposure. Depth is
// standard (the main depth target clears to 1), so no DepthInverted.
constexpr std::int32_t kFlagIsHdr = 1 << 0, kFlagMvLowRes = 1 << 1, kFlagAutoExposure = 1 << 6;

struct Bridge {
    bool tried = false;
#if defined(_WIN32)
    HMODULE dll = nullptr;
#endif
    PFN_ngxb_version version = nullptr;
    PFN_ngxb_required_extensions required = nullptr;
    PFN_ngxb_init init = nullptr;
    PFN_ngxb_create create = nullptr;
    PFN_ngxb_create_sr create_sr = nullptr;
    PFN_ngxb_evaluate evaluate = nullptr;
    PFN_ngxb_release release = nullptr;
    PFN_ngxb_fg_extensions fg_extensions = nullptr;
} g_bridge;

std::string g_dir;  // the exe's folder: the bridge and nvngx_dlss.dll

std::string dlss_mode() {
    const int mode = host_startup_settings().dlss_mode;
    return host::DlssModes[mode >= 0 && mode < 6 ? mode : 0];
}

int sr_quality() {
    // Without an explicit larger output, retain native DLAA rather than
    // jittering the scene and then skipping reconstruction entirely.
    const auto mode = dlss_mode();
    if (mode == "quality") return 1;
    if (mode == "balanced") return 2;
    if (mode == "performance") return 3;
    if (mode == "ultra_performance") return 4;
    return 0;
}

bool config_on() {
    static const bool on = [] {
        if (const char* e = std::getenv("BBHOST_DLAA")) return e[0] != '0';
        const auto mode = dlss_mode();
        if (!mode.empty()) return mode != "off";
        const std::string v = config_value("dlaa.enabled");
        return !(v == "false" || v == "0" || v == "off");
    }();
    return on;
}

bool fsr_selected() { return host_startup_settings().upscaler_backend != 0; }

bool ngx_requested() {
    const auto& settings = host_startup_settings();
    return (config_on() && !fsr_selected()) ||
           (settings.frame_generation && settings.frame_generation_backend == 0);
}

bool guides_only() {
    return dlss_sr_is_active() || fsr_selected() || !config_on();
}

std::uint32_t config_preset() {
    std::string v = config_value("dlaa.preset");
    if (const char* e = std::getenv("BBHOST_DLAA_PRESET")) v = e;
    if (v.size() == 1 && v[0] >= 'A' && v[0] <= 'Z') return static_cast<std::uint32_t>(v[0] - 'A' + 1);  // NVSDK_NGX_DLSS_Hint_Render_Preset_<letter>
    if (v.size() == 1 && v[0] >= 'a' && v[0] <= 'z') return static_cast<std::uint32_t>(v[0] - 'a' + 1);
    return 0;  // default: the driver's (K for DLAA with the 310 DLL)
}

bool load_bridge() {
    if (g_bridge.tried) return g_bridge.evaluate != nullptr;
    g_bridge.tried = true;
    if (!ngx_requested()) {
        host_log("temporal: NGX is not requested for the selected providers");
        return false;
    }
#if defined(_WIN32)
    g_dir = config_exe_dir();
    const std::string path = g_dir + "/ngx_bridge.dll";
    std::wstring wpath(path.size() + 1, L'\0');
    wpath.resize(static_cast<std::size_t>(MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), static_cast<int>(wpath.size()))));
    if (!wpath.empty() && wpath.back() == L'\0') wpath.pop_back();
    g_bridge.dll = LoadLibraryExW(wpath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g_bridge.dll) {
        host_log("dlaa: %s not found or not loadable (error %lu); off", path.c_str(), GetLastError());
        return false;
    }
    auto get = [](const char* name) { return reinterpret_cast<void*>(GetProcAddress(g_bridge.dll, name)); };
    g_bridge.version = reinterpret_cast<PFN_ngxb_version>(get("ngxb_version"));
    g_bridge.required = reinterpret_cast<PFN_ngxb_required_extensions>(get("ngxb_required_extensions"));
    g_bridge.init = reinterpret_cast<PFN_ngxb_init>(get("ngxb_init"));
    g_bridge.create = reinterpret_cast<PFN_ngxb_create>(get("ngxb_create"));
    g_bridge.create_sr = reinterpret_cast<PFN_ngxb_create_sr>(get("ngxb_create_sr"));
    g_bridge.evaluate = reinterpret_cast<PFN_ngxb_evaluate>(get("ngxb_evaluate"));
    g_bridge.release = reinterpret_cast<PFN_ngxb_release>(get("ngxb_release"));
    g_bridge.fg_extensions = reinterpret_cast<PFN_ngxb_fg_extensions>(get("ngxb_fg_extensions"));
    if (!g_bridge.version || g_bridge.version() != kBridgeVersion || !g_bridge.required || !g_bridge.init || !g_bridge.create ||
        !g_bridge.evaluate || !g_bridge.release || !g_bridge.create_sr) {
        host_log("dlaa: ngx_bridge.dll is not version %d; off", kBridgeVersion);
        g_bridge.evaluate = nullptr;
        return false;
    }
    host_log("dlaa: ngx_bridge.dll loaded from %s", g_dir.c_str());
    return true;
#else
    host_log("dlaa: Windows only; off");
    return false;
#endif
}

void add_missing(std::vector<const char*>& exts, const char* name, const std::vector<VkExtensionProperties>& avail, const char* what) {
    for (const char* e : exts) {
        if (std::strcmp(e, name) == 0) return;
    }
    for (const VkExtensionProperties& p : avail) {
        if (std::strcmp(p.extensionName, name) == 0) {
            exts.push_back(name);
            host_log("dlaa: %s extension %s added for NGX", what, name);
            return;
        }
    }
    host_log("dlaa: %s extension %s that NGX asks for is not available", what, name);
}

// ---- state on the command processor (under g.mu) ----

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool general = false;
};

struct State {
    bool ngx_tried = false, ngx_ok = false;
    bool failed = false;  // for the session
    std::atomic<bool> enabled{true};  // Ctrl+F1
    std::uint32_t width = 0, height = 0;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    bool feature = false;
    bool reset = true;
    Image output, depth, motion, object_fallback;
    bool output_same_format = true;
    // A new frame at the anchor: depth-writing draws since the last one. The
    // command processor can lag the CPU's flips, so the flip count alone
    // could put two frames' anchors in one flip.
    std::uint64_t depth_writes = 0, depth_writes_at_anchor = ~0ull;
    std::uint64_t anchors_this_frame = 0, anchors_last_frame = 0;
    std::uint64_t anchor_calls = 0, first_depth_flip = 0;
    bool anchor_missing_said = false;
    std::uint64_t evaluations = 0, failures = 0;
    std::chrono::steady_clock::time_point last_eval{};
    // The depth targets drawn with depth writes on, least recently written
    // replaced first. Some views draw eight a frame (shadows, the half-size
    // pass, a probe's mip chain): too few slots and the main depth is pushed
    // out before the anchor. Some views draw eight a frame (shadows, the
    // half-size pass, a chain of 256 to 32 pixel targets).
    struct Depth {
        std::uint64_t base = 0;
        std::uint32_t w = 0, h = 0;
        std::uint64_t flip = 0;
        std::uint64_t last = 0;    // depth_writes at its last write
        std::uint64_t writes = 0;  // depth-writing draws since the last anchor
    } depths[16];
    // Anchors skipped (no scene image with a matching depth) in a row, and
    // in all: said when it starts and when it ends.
    std::uint64_t skipped_run = 0, skipped = 0;
    std::uint64_t logged_scene = 0, logged_depth = 0;
    // The main depth (the scene's, from the last anchor): the draws
    // that test against it are the scene's geometry. They are jittered by a
    // sub-pixel shift of their viewport, and their vertex shaders bind the
    // camera buffer.
    std::uint64_t main_depth = 0;
    std::uint32_t main_w = 0, main_h = 0;
    float jx = 0, jy = 0;  // the jitter of the draws since the last anchor (pixels, x right, y down)
    std::uint32_t jitter_index = 0;
    std::uint64_t jittered_draws = 0;  // draws given a non-zero shift, since the last log
    // Ctrl+F2: 0 on, 1 off, 2 test: DLSS skipped and the jitter 8x, so the
    // raw scene visibly shakes by up to 4 pixels when the jitter reaches it.
    std::atomic<int> jitter_mode{0};
    std::atomic<int> jitter_sign{1};    // Ctrl+F3: the sign DLSS is given
    std::atomic<bool> mv_zero{false};   // Ctrl+F4: zero motion instead of the camera's
    // The camera: the 864-byte per-view buffer (rows, M*v): +0x010 the
    // target size, +0x0D0 the projection, +0x2D0 the inverse view (camera
    // position in column 3), +0x320 ViewProj without the translation. The
    // first one a main-depth draw binds in a frame.
    float cam_next[216] = {}, cam_cur[216] = {}, cam_prev[216] = {};
    bool cam_got = false, cam_cur_ok = false, cam_prev_ok = false, cam_checked = false;
    std::uint64_t cam_frames = 0, cam_missing = 0, cuts = 0;
    VkDescriptorSetLayout mv_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout mv_layout = VK_NULL_HANDLE;
    VkPipeline mv_pipeline = VK_NULL_HANDLE;
    bool mv_tried = false;
} g_s;

struct SrFeature {
    std::uint32_t input_w = 0, input_h = 0, output_w = 0, output_h = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    int quality = 0;
    bool ready = false;
    std::uint64_t evaluations = 0;
} g_sr;
Image g_sr_input, g_sr_output;
std::uint32_t g_sr_image_iw = 0, g_sr_image_ih = 0, g_sr_image_ow = 0, g_sr_image_oh = 0;
std::uint64_t g_sr_hud_anchor = 0;
bool g_sr_render_failed = false;
std::map<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>> g_sr_display_sizes;

// Immutable CPU metadata plus owned GPU copies for each guest display buffer.
// The presenter checks the token queued with its flip before using a snapshot.
struct FgSnapshot {
    Image depth, motion, hudless;
    DlssFgGuides guides{};
};
std::map<std::uint64_t, FgSnapshot> g_fg_snapshots;
DlssFgGuides g_fg_current{};
Image g_fg_hudless;
std::uint32_t g_fg_hudless_w = 0, g_fg_hudless_h = 0;
std::uint64_t g_fg_hudless_anchor = 0, g_fg_latched_anchor = 0, g_fg_next_frame = 0;

bool fg_requested() {
    return host_startup_settings().frame_generation;
}

// Halton(2, 3), 16 phases, centred: the jitter sequence.
float halton(std::uint32_t i, std::uint32_t b) {
    float f = 1.0f, r = 0.0f;
    while (i) {
        f /= static_cast<float>(b);
        r += f * static_cast<float>(i % b);
        i /= b;
    }
    return r;
}

bool camera_signature(const float* f, std::uint32_t w, std::uint32_t h) {
    for (int i = 0; i < 216; ++i) {
        if (!std::isfinite(f[i])) return false;
    }
    // Size, a perspective projection (w row 0 0 1 0, the screen's aspect,
    // z' = a z - b with b > 0): shadow views are orthographic.
    return f[4] == static_cast<float>(w) && f[5] == static_cast<float>(h) && f[52] > 0.05f && f[52] < 20.0f && f[57] > 0.05f &&
           f[57] < 20.0f && std::fabs(f[57] / f[52] - static_cast<float>(w) / static_cast<float>(h)) < 0.02f * static_cast<float>(w) / static_cast<float>(h) &&
           f[64] == 0.0f && f[65] == 0.0f && std::fabs(f[66] - 1.0f) < 1e-3f && f[67] == 0.0f && f[63] < 0.0f;
}

using Mat4 = double[4][4];
void mat_mul(const Mat4 a, const Mat4 b, Mat4 out) {
    Mat4 t;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double v = 0;
            for (int k = 0; k < 4; ++k) v += a[i][k] * b[k][j];
            t[i][j] = v;
        }
    std::memcpy(out, t, sizeof(t));
}
bool mat_inverse(const Mat4 m, Mat4 out) {
    double a[4][8];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j) a[i][j] = j < 4 ? m[i][j] : (j - 4 == i ? 1.0 : 0.0);
    for (int c = 0; c < 4; ++c) {
        int p = c;
        for (int r = c + 1; r < 4; ++r) {
            if (std::fabs(a[r][c]) > std::fabs(a[p][c])) p = r;
        }
        if (std::fabs(a[p][c]) < 1e-12) return false;
        if (p != c)
            for (int j = 0; j < 8; ++j) std::swap(a[p][j], a[c][j]);
        const double d = a[c][c];
        for (int j = 0; j < 8; ++j) a[c][j] /= d;
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double k = a[r][c];
            for (int j = 0; j < 8; ++j) a[r][j] -= k * a[c][j];
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out[i][j] = a[i][j + 4];
    return true;
}
void cam_vp(const float* f, Mat4 out) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out[i][j] = f[200 + 4 * i + j];
}
void cam_pos(const float* f, double out[3]) {
    out[0] = f[183];
    out[1] = f[187];
    out[2] = f[191];
}

// Once, on the first camera: +0x320 is the projection times the view's
// rotation, as the layout says.
void camera_self_check(const float* f) {
    // P * (R | 0) with R the view's 3x3 rotation (+0x020 rows): column 3 is P's own.
    double worst = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double v = f[52 + 4 * i + 3];
            if (j < 3) {
                v = 0;
                for (int k = 0; k < 3; ++k) v += static_cast<double>(f[52 + 4 * i + k]) * f[8 + 4 * k + j];
            }
            worst = std::max(worst, std::fabs(v - f[200 + 4 * i + j]));
        }
    host_log("dlaa: camera found: projection x %.5f y %.5f, near %.4g, position (%.2f %.2f %.2f); ViewProj check %s (worst %.2g)", f[52], f[57],
             -f[63] / f[62], f[183], f[187], f[191], worst < 1e-3 ? "ok" : "FAILED", worst);
}

bool make_mv_pipeline_locked() {
    const VkDescriptorSetLayoutBinding binds[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 3;
    sli.pBindings = binds;
    if (vkCreateDescriptorSetLayout(g.device, &sli, nullptr, &g_s.mv_set_layout) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 96};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_s.mv_set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(g.device, &pli, nullptr, &g_s.mv_layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(k_dlaa_mv_spv);
    smi.pCode = k_dlaa_mv_spv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.device, &smi, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = g_s.mv_layout;
    const bool ok = vkCreateComputePipelines(g.device, g.cache, 1, &ci, nullptr, &g_s.mv_pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(g.device, module, nullptr);
    return ok;
}

void destroy_image(Image& im) {
    if (im.view) vkDestroyImageView(g.device, im.view, nullptr);
    if (im.image) vkDestroyImage(g.device, im.image, nullptr);
    if (im.memory) vkFreeMemory(g.device, im.memory, nullptr);
    im = Image{};
}

bool make_image(Image& im, std::uint32_t w, std::uint32_t h, VkFormat format, VkImageUsageFlags usage) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g.device, &ici, nullptr, &im.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g.device, im.image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &im.memory) != VK_SUCCESS) return false;
    vkBindImageMemory(g.device, im.image, im.memory, 0);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = im.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(g.device, &vci, nullptr, &im.view) != VK_SUCCESS) return false;
    im.format = format;
    im.general = false;
    return true;
}

bool storage_ok(VkFormat f) {
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(g.phys, f, &fp);
    return (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
}

void memory_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void to_general(VkCommandBuffer cmd, Image& im) {
    if (im.general) return;
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    im.general = true;
}

bool fg_resize_image_locked(Image& image, std::uint32_t width, std::uint32_t height, VkFormat format) {
    return make_image(image, width, height, format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}

void fg_copy_locked(VkCommandBuffer cmd, VkImage source, Image& destination, std::uint32_t width, std::uint32_t height) {
    to_general(cmd, destination);
    VkImageCopy copy{};
    copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {width, height, 1};
    vkCmdCopyImage(cmd, source, VK_IMAGE_LAYOUT_GENERAL, destination.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
}

void fg_latch_locked(const DrawRec& draw) {
    if (!fg_requested() || std::strncmp(draw.name, "89cda9e4+", 9) || !draw.rt0) return;
    auto& snapshot = g_fg_snapshots[draw.rt0];
    snapshot.guides.valid = false;
    snapshot.guides.frame_index = ++g_fg_next_frame;
    if (!g_fg_current.valid || !g_fg_current.frame_index || g_fg_current.frame_index == g_fg_latched_anchor ||
        g_fg_hudless_anchor != g_fg_current.frame_index || !g_fg_hudless.image) return;
    g_fg_latched_anchor = g_fg_current.frame_index;
    const auto w = g_fg_current.width, h = g_fg_current.height;
    const auto hw = g_fg_hudless_w, hh = g_fg_hudless_h;
    if (!snapshot.depth.image || !snapshot.motion.image || !snapshot.hudless.image ||
        snapshot.guides.width != w || snapshot.guides.height != h ||
        snapshot.guides.hudless_width != hw || snapshot.guides.hudless_height != hh ||
        snapshot.hudless.format != g_fg_hudless.format) {
        flush_locked();
        destroy_image(snapshot.depth); destroy_image(snapshot.motion); destroy_image(snapshot.hudless);
        if (!fg_resize_image_locked(snapshot.depth, w, h, VK_FORMAT_R32_SFLOAT) ||
            !fg_resize_image_locked(snapshot.motion, w, h, VK_FORMAT_R16G16_SFLOAT) ||
            !fg_resize_image_locked(snapshot.hudless, hw, hh, g_fg_hudless.format)) {
            destroy_image(snapshot.depth); destroy_image(snapshot.motion); destroy_image(snapshot.hudless);
            return;
        }
    }
    begin_recording_locked(); render_end_pass_locked();
    auto cmd = g_cmd(); memory_barrier(cmd);
    fg_copy_locked(cmd, g_s.depth.image, snapshot.depth, w, h);
    fg_copy_locked(cmd, g_s.motion.image, snapshot.motion, w, h);
    fg_copy_locked(cmd, g_fg_hudless.image, snapshot.hudless, hw, hh);
    memory_barrier(cmd);
    const auto frame = snapshot.guides.frame_index;
    snapshot.guides = g_fg_current;
    snapshot.guides.frame_index = frame;
    snapshot.guides.depth = snapshot.depth.image; snapshot.guides.depth_view = snapshot.depth.view;
    snapshot.guides.motion = snapshot.motion.image; snapshot.guides.motion_view = snapshot.motion.view;
    snapshot.guides.hudless = snapshot.hudless.image; snapshot.guides.hudless_view = snapshot.hudless.view;
    snapshot.guides.hudless_format = snapshot.hudless.format;
    snapshot.guides.hudless_width = hw; snapshot.guides.hudless_height = hh;
    static std::uint64_t captured = 0;
    if (++captured == 1 || captured % 300 == 0)
        host_log("framegen: captured %llu complete guide snapshots, scene %ux%u, HUD-less %ux%u",
                 static_cast<unsigned long long>(captured), w, h, hw, hh);
}

[[maybe_unused]] void ngx_log(const char* message) { host_log("dlaa: %s", message); }

bool init_ngx_locked() {
    if (g_s.ngx_tried) return g_s.ngx_ok;
    g_s.ngx_tried = true;
    if (!load_bridge()) return false;
#if defined(_WIN32)
    auto widen = [](const std::string& s) {
        std::wstring w(s.size() + 1, L'\0');
        const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), static_cast<int>(w.size()));
        w.resize(n > 0 ? static_cast<std::size_t>(n - 1) : 0);
        return w;
    };
    const std::wstring dll_dir = widen(g_dir);
    std::error_code ec;
    std::filesystem::create_directories(g_dir + "/logs/ngx", ec);
    const std::wstring data_dir = widen(g_dir + "/logs/ngx");
    const std::uint32_t r = g_bridge.init(g.instance, g.phys, g.device, vkGetInstanceProcAddr, vkGetDeviceProcAddr, dll_dir.c_str(),
                                          data_dir.c_str(), &ngx_log);
    g_s.ngx_ok = !ngx_failed(r);
    host_log("dlaa: NGX init %s (0x%08x)%s", g_s.ngx_ok ? "ok" : "failed", r,
             g_s.ngx_ok ? "" : " - needs an NVIDIA RTX GPU and nvngx_dlss.dll beside bbhost.exe; the game renders unchanged");
#endif
    return g_s.ngx_ok;
}

void release_locked() {
    if (g_s.feature || g_sr.ready || g_s.output.image) flush_locked();  // NGX's and our images idle before they go
    if (g_s.feature || g_sr.ready) g_bridge.release();
    fsr_shutdown_locked();
    g_sr = {};
    g_s.feature = false;
    destroy_image(g_s.output);
    destroy_image(g_s.depth);
    destroy_image(g_s.motion);
    g_s.width = g_s.height = 0;
}

void fail_locked(const char* why) {
    host_log("dlaa: %s - off for this session; the game renders unchanged", why);
    release_locked();
    g_s.failed = true;
}

// Images and the DLSS feature for a scene of this size and format. False
// when they are not ready to evaluate this frame (just created, or failed).
bool ensure_locked(std::uint32_t w, std::uint32_t h, VkFormat color_format) {
    const bool sr = guides_only();
    if ((sr ? bool(g_s.depth.image) : g_s.feature) && g_s.width == w && g_s.height == h && g_s.color_format == color_format) return true;
    release_locked();
    g_s.output_same_format = storage_ok(color_format);
    const VkFormat out_format = g_s.output_same_format ? color_format : VK_FORMAT_R16G16B16A16_SFLOAT;
    if (!make_image(g_s.output, w, h, out_format,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
        !make_image(g_s.depth, w, h, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
        !make_image(g_s.motion, w, h, VK_FORMAT_R16G16_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        fail_locked("creating DLAA's images failed");
        return false;
    }
    begin_recording_locked();
    render_end_pass_locked();
    VkCommandBuffer cmd = g_cmd();
    to_general(cmd, g_s.output);
    to_general(cmd, g_s.depth);
    to_general(cmd, g_s.motion);
    // Zero motion until the motion-vector pass first writes it (and if its pipeline fails).
    const VkClearColorValue zero{};
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, g_s.motion.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    memory_barrier(cmd);
    const std::uint32_t preset = config_preset();
    const std::uint32_t r = sr ? 1u : g_bridge.create(cmd, w, h, preset, kFlagIsHdr | kFlagMvLowRes | kFlagAutoExposure);
    if (ngx_failed(r)) {
        char why[96];
        std::snprintf(why, sizeof(why), "creating the DLSS feature at %ux%u failed (0x%08x)", w, h, r);
        fail_locked(why);
        return false;
    }
    g_s.feature = !sr;
    g_s.width = w;
    g_s.height = h;
    g_s.color_format = color_format;
    g_s.reset = true;
    host_log("dlaa: ready at %ux%u, scene format %d, output %s, preset %u", w, h, static_cast<int>(color_format),
             g_s.output_same_format ? "copied back" : "RGBA16F blitted back", preset);
    return false;  // evaluate from the next frame: the feature's setup is in this command buffer
}

// ---- the constant-buffer capture ----

struct CbCapture {
    std::mutex mu;  // dir and the request, from the window thread
    std::string dir;
    bool requested = false;
    // On the command processor:
    int phase = 0;  // 0 idle, 1 waiting for a flip edge, 2 capturing
    std::uint64_t flip = 0;
    std::FILE* txt = nullptr;
    std::FILE* bin = nullptr;
    std::uint64_t bin_bytes = 0, draws = 0, buffers = 0;
    struct Pending {
        std::uint64_t seq;  // g_draw_rec_next while its draw resolved: a draw that never reached its record drops them
        int stage;
        std::size_t slot;
        std::uint64_t base, bytes, at, len;
    };
    std::vector<Pending> pending;  // the current draw's
    std::string path;
} g_cb;

void cb_close_locked() {
    if (g_cb.txt) std::fclose(g_cb.txt);
    if (g_cb.bin) std::fclose(g_cb.bin);
    g_cb.txt = g_cb.bin = nullptr;
    g_dlaa_cb_capturing.store(false, std::memory_order_relaxed);
    host_log("dlaa: constant buffers of flip %llu written: %llu draws, %llu buffers, %llu KiB (%s.txt/.bin)",
             static_cast<unsigned long long>(g_cb.flip), static_cast<unsigned long long>(g_cb.draws),
             static_cast<unsigned long long>(g_cb.buffers), static_cast<unsigned long long>(g_cb.bin_bytes >> 10), g_cb.path.c_str());
    g_cb.phase = 0;
}

void cb_tick_locked() {
    const std::uint64_t flip = hle_video_flip_count();
    if (g_cb.phase == 0) {
        std::lock_guard<std::mutex> lk(g_cb.mu);
        if (!g_cb.requested) return;
        g_cb.requested = false;
        g_cb.phase = 1;
        g_cb.flip = flip;
        return;
    }
    if (g_cb.phase == 1 && flip != g_cb.flip) {
        std::string dir;
        {
            std::lock_guard<std::mutex> lk(g_cb.mu);
            dir = g_cb.dir;
        }
        g_cb.flip = flip;
        g_cb.path = dir + "/cbs-" + std::to_string(flip);
        g_cb.txt = std::fopen((g_cb.path + ".txt").c_str(), "w");
        g_cb.bin = std::fopen((g_cb.path + ".bin").c_str(), "wb");
        g_cb.bin_bytes = g_cb.draws = g_cb.buffers = 0;
        g_cb.pending.clear();
        if (!g_cb.txt || !g_cb.bin) {
            host_log("dlaa: cannot write %s.txt/.bin", g_cb.path.c_str());
            if (g_cb.txt) std::fclose(g_cb.txt);
            if (g_cb.bin) std::fclose(g_cb.bin);
            g_cb.txt = g_cb.bin = nullptr;
            g_cb.phase = 0;
            return;
        }
        std::fprintf(g_cb.txt, "# one frame's constant buffers: draw lines, then each bound buffer (stage 0 VS / 1 PS, slot,\n"
                               "# guest base, bound bytes, offset and length of its bytes in the .bin)\n");
        g_cb.phase = 2;
        g_dlaa_cb_capturing.store(true, std::memory_order_relaxed);
        return;
    }
    if (g_cb.phase == 2 && flip != g_cb.flip) cb_close_locked();
}

}  // namespace

std::atomic<bool> g_dlaa_cb_capturing{false};
thread_local int t_dlaa_cb_stage = -1;
thread_local bool t_dlaa_main_draw = false;

std::uint64_t dlaa_fg_frame_id_locked(std::uint64_t display_va) {
    const auto found = g_fg_snapshots.find(display_va);
    return found == g_fg_snapshots.end() ? 0 : found->second.guides.frame_index;
}

bool dlaa_get_fg_guides_locked(std::uint64_t display_va, std::uint64_t expected_frame_id, DlssFgGuides* out) {
    if (!out || !expected_frame_id) return false;
    const auto found = g_fg_snapshots.find(display_va);
    if (found == g_fg_snapshots.end() || !found->second.guides.valid ||
        found->second.guides.frame_index != expected_frame_id) return false;
    *out = found->second.guides;
    return true;
}

void dlaa_capture_hudless_locked(RtImage* source, std::uint32_t logical_width, std::uint32_t logical_height) {
    if (!fg_requested() || !g_fg_current.valid || !source || !source->initialised || source->depth ||
        !logical_width || !logical_height || logical_width > source->width || logical_height > source->height ||
        g_fg_hudless_anchor == g_fg_current.frame_index) return;
    if (!g_fg_hudless.image || g_fg_hudless_w != logical_width || g_fg_hudless_h != logical_height ||
        g_fg_hudless.format != source->format) {
        flush_locked(); destroy_image(g_fg_hudless);
        if (!fg_resize_image_locked(g_fg_hudless, logical_width, logical_height, source->format)) {
            destroy_image(g_fg_hudless);
            return;
        }
        g_fg_hudless_w = logical_width; g_fg_hudless_h = logical_height;
    }
    begin_recording_locked(); render_end_pass_locked();
    const auto cmd = g_cmd(); memory_barrier(cmd);
    fg_copy_locked(cmd, source->image, g_fg_hudless, logical_width, logical_height);
    memory_barrier(cmd);
    g_fg_hudless_anchor = g_fg_current.frame_index;
}

// SR's renderer module supplies the output override; native DLAA retains the
// existing display target. This is replaced when the SR renderer is wired.
bool dlaa_get_display_override_locked(std::uint64_t, void**, std::uint32_t*, std::uint32_t*, std::uint32_t*) { return false; }

bool dlss_sr_is_active() {
    // FSR native AA uses the same pre-HUD reconstruction stage as upscaling.
    // The public name remains for the renderer's existing resolution hooks.
    if (fsr_selected())
        return config_on() && !g_sr_render_failed &&
               g_s.enabled.load(std::memory_order_relaxed) && !g_s.failed;
    return config_on() && sr_quality() != 0 && !g_sr_render_failed &&
           static_cast<std::uint32_t>(host_startup_settings().output_width) > g_s.main_w &&
           static_cast<std::uint32_t>(host_startup_settings().output_height) > g_s.main_h &&
           g_s.enabled.load(std::memory_order_relaxed) && !g_s.failed;
}

void dlss_sr_get_dimensions(std::uint32_t in_w, std::uint32_t in_h, std::uint32_t* out_w, std::uint32_t* out_h) {
    *out_w = host_startup_settings().output_width > 0 ? static_cast<std::uint32_t>(host_startup_settings().output_width) : in_w;
    *out_h = host_startup_settings().output_height > 0 ? static_cast<std::uint32_t>(host_startup_settings().output_height) : in_h;
}

bool dlss_sr_evaluate_locked(VkCommandBuffer, VkImage input, VkImageView input_view, VkFormat input_format,
                             std::uint32_t in_w, std::uint32_t in_h, VkImage output, VkImageView output_view,
                             VkFormat output_format, std::uint32_t out_w, std::uint32_t out_h) {
    if (!dlss_sr_is_active() || !g_fg_current.valid || !input || !output ||
        g_fg_current.width != in_w || g_fg_current.height != in_h) return false;
    if (fsr_selected()) {
        auto guides = g_fg_current;
        guides.depth = g_s.depth.image; guides.depth_view = g_s.depth.view;
        guides.motion = g_s.motion.image; guides.motion_view = g_s.motion.view;
        const NgxbImage color{input, input_view, input_format, in_w, in_h, VK_IMAGE_ASPECT_COLOR_BIT};
        const NgxbImage target{output, output_view, output_format, out_w, out_h, VK_IMAGE_ASPECT_COLOR_BIT};
        render_end_pass_locked();
        memory_barrier(g_cmd());
        const bool ok = fsr_upscale_locked(g.phys, g.device, g_cmd(), color, target,
                                           guides, g.flushes, g.completed_submits);
        memory_barrier(g_cmd());
        if (!ok) g_s.reset = true;
        return ok;
    }
    if (!init_ngx_locked()) return false;
    const int quality = sr_quality();
    if (!g_sr.ready || g_sr.input_w != in_w || g_sr.input_h != in_h || g_sr.output_w != out_w ||
        g_sr.output_h != out_h || g_sr.format != input_format || g_sr.quality != quality) {
        // NGX owns only one SR feature. Its old command submissions must finish
        // before replacement; acquire the current recording buffer after flushing.
        flush_locked();
        begin_recording_locked();
        render_end_pass_locked();
        const auto result = g_bridge.create_sr(g_cmd(), in_w, in_h, out_w, out_h, quality,
                                               config_preset(), kFlagMvLowRes | kFlagAutoExposure);
        if (ngx_failed(result)) {
            host_log("dlss: SR create failed (0x%08x)", result);
            g_sr.ready = false;
            g_sr_render_failed = true;
            g_s.reset = true;
            return false;
        }
        g_sr = {in_w, in_h, out_w, out_h, input_format, quality, true, 0};
        g_fg_current.reset = 1;
        host_log("dlss: SR ready, %ux%u -> %ux%u, quality %d", in_w, in_h, out_w, out_h, quality);
    }
    render_end_pass_locked();
    memory_barrier(g_cmd());
    NgxbEval evaluation{};
    evaluation.color = {input, input_view, input_format, in_w, in_h, VK_IMAGE_ASPECT_COLOR_BIT};
    evaluation.output = {output, output_view, output_format, out_w, out_h, VK_IMAGE_ASPECT_COLOR_BIT};
    evaluation.depth = {g_s.depth.image, g_s.depth.view, g_s.depth.format, in_w, in_h, VK_IMAGE_ASPECT_COLOR_BIT};
    evaluation.motion = {g_s.motion.image, g_s.motion.view, g_s.motion.format, in_w, in_h, VK_IMAGE_ASPECT_COLOR_BIT};
    evaluation.jitter_x = g_fg_current.jitter_x;
    evaluation.jitter_y = g_fg_current.jitter_y;
    evaluation.mv_scale_x = evaluation.mv_scale_y = 1.0f;
    evaluation.reset = g_fg_current.reset;
    evaluation.frame_ms = g_fg_current.frame_ms;
    const auto result = g_bridge.evaluate(g_cmd(), &evaluation);
    memory_barrier(g_cmd());
    if (ngx_failed(result)) {
        host_log("dlss: SR evaluate failed (0x%08x)", result);
        g_s.reset = true;
        g_sr_render_failed = true;
        return false;
    }
    if (++g_sr.evaluations == 1 || g_sr.evaluations % 300 == 0)
        host_log("dlss: SR evaluated %llu frames", static_cast<unsigned long long>(g_sr.evaluations));
    return true;
}

bool dlss_sr_upscale_hud_locked(RtImage* source, std::uint32_t in_w, std::uint32_t in_h,
                                std::uint32_t* out_w, std::uint32_t* out_h) {
    if (!source || !source->initialised || source->depth || !dlss_sr_is_active() ||
        !g_fg_current.valid || g_fg_current.width != in_w || g_fg_current.height != in_h) return false;
    dlss_sr_get_dimensions(in_w, in_h, out_w, out_h);
    if (!in_w || !in_h || *out_w < in_w || *out_h < in_h ||
        *out_w > source->width || *out_h > source->height || !storage_ok(source->format)) {
        host_log("dlss: requested output does not fit the HUD target; falling back to native DLAA");
        g_sr_render_failed = true; g_s.reset = true;
        return false;
    }
    if (g_sr_hud_anchor == g_fg_current.frame_index) return true;
    const auto sr_format = fsr_selected() && source->format == VK_FORMAT_B8G8R8A8_UNORM ?
                           VK_FORMAT_R8G8B8A8_UNORM : source->format;
    // The pinned FSR accumulation shader declares rgba16f storage output.
    // Keep SDR input as SDR; only the owned reconstruction output needs FP16.
    const auto output_format = fsr_selected() ? VK_FORMAT_R16G16B16A16_SFLOAT : sr_format;
    if (!g_sr_input.image || !g_sr_output.image || g_sr_image_iw != in_w || g_sr_image_ih != in_h ||
        g_sr_image_ow != *out_w || g_sr_image_oh != *out_h || g_sr_input.format != sr_format ||
        g_sr_output.format != output_format) {
        flush_locked(); destroy_image(g_sr_input); destroy_image(g_sr_output);
        if (!fg_resize_image_locked(g_sr_input, in_w, in_h, sr_format) ||
            !fg_resize_image_locked(g_sr_output, *out_w, *out_h, output_format)) {
            destroy_image(g_sr_input); destroy_image(g_sr_output);
            return false;
        }
        g_sr_image_iw = in_w; g_sr_image_ih = in_h;
        g_sr_image_ow = *out_w; g_sr_image_oh = *out_h;
    }
    begin_recording_locked(); render_end_pass_locked();
    memory_barrier(g_cmd());
    if (sr_format == source->format) {
        fg_copy_locked(g_cmd(), source->image, g_sr_input, in_w, in_h);
    } else {
        to_general(g_cmd(), g_sr_input);
        VkImageBlit blit{};
        blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = blit.dstOffsets[1] = {static_cast<int>(in_w), static_cast<int>(in_h), 1};
        vkCmdBlitImage(g_cmd(), source->image, VK_IMAGE_LAYOUT_GENERAL, g_sr_input.image,
                       VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_NEAREST);
    }
    to_general(g_cmd(), g_sr_output);
    memory_barrier(g_cmd());
    if (!dlss_sr_evaluate_locked(g_cmd(), g_sr_input.image, g_sr_input.view, g_sr_input.format,
                                 in_w, in_h, g_sr_output.image, g_sr_output.view, g_sr_output.format,
                                 *out_w, *out_h)) return false;
    VkImageCopy copy{};
    copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {*out_w, *out_h, 1};
    if (output_format == source->format) {
        vkCmdCopyImage(g_cmd(), g_sr_output.image, VK_IMAGE_LAYOUT_GENERAL, source->image,
                       VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    } else {
        VkImageBlit blit{};
        blit.srcSubresource = blit.dstSubresource = copy.srcSubresource;
        blit.srcOffsets[1] = blit.dstOffsets[1] = {static_cast<int>(*out_w), static_cast<int>(*out_h), 1};
        vkCmdBlitImage(g_cmd(), g_sr_output.image, VK_IMAGE_LAYOUT_GENERAL, source->image,
                       VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_NEAREST);
    }
    memory_barrier(g_cmd());
    source->fill_last = false;
    g_sr_hud_anchor = g_fg_current.frame_index;
    return true;
}

bool dlss_sr_display_dimensions_locked(std::uint64_t display_va, std::uint32_t* width, std::uint32_t* height) {
    const auto it = g_sr_display_sizes.find(display_va);
    if (!dlss_sr_is_active() || it == g_sr_display_sizes.end()) return false;
    *width = it->second.first; *height = it->second.second;
    return true;
}

bool dlaa_main_draw_locked(std::uint64_t depth, std::uint32_t depth_ctl, std::uint32_t prim, std::uint32_t count) {
    // Depth-tested draws into the scene's depth (Z_ENABLE), but not the
    // screen-space quads that test it too (the deferred lights: 4-vertex
    // strips, rect lists): shifting those would resample what they read.
    return depth && depth == g_s.main_depth && (depth_ctl & 0x2) && !((prim == 6 || prim == 0x11) && count <= 4);
}

bool dlaa_motion_active_locked() {
    return (config_on() || fg_requested()) &&
           g_s.enabled.load(std::memory_order_relaxed) && !g_s.failed;
}

void dlaa_jitter_locked(float* x, float* y) {
    *x = g_s.jx;
    *y = g_s.jy;
    if (g_s.jx != 0.0f || g_s.jy != 0.0f) ++g_s.jittered_draws;
}

void dlaa_camera_note_locked(std::uint64_t base, std::uint64_t bytes) {
    if (g_s.cam_got || g_s.failed || !base || (base & 3) || bytes < 256 || !hle_kernel_va_mapped(base, sizeof(g_s.cam_next))) return;
    const float* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(base));
    if (!camera_signature(f, g_s.main_w, g_s.main_h)) return;
    std::memcpy(g_s.cam_next, f, sizeof(g_s.cam_next));
    g_s.cam_got = true;
}

void dlaa_add_instance_extensions(std::vector<const char*>& exts) {
    if (!load_bridge()) return;
    std::uint32_t ni = 0, nd = 0;
    const char* const* inst = nullptr;
    const char* const* dev = nullptr;
    if (ngx_failed(g_bridge.required(&ni, &inst, &nd, &dev))) {
        host_log("dlaa: NGX's extension list failed");
        return;
    }
    std::uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> avail(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, avail.data());
    for (std::uint32_t i = 0; i < ni; ++i) add_missing(exts, inst[i], avail, "instance");
    std::uint32_t nf = 0;
    const VkExtensionProperties* fg = nullptr;
    const std::wstring directory = std::filesystem::path(g_dir).wstring();
    if (g_bridge.fg_extensions && !ngx_failed(g_bridge.fg_extensions(VK_NULL_HANDLE, VK_NULL_HANDLE,
            directory.c_str(), &nf, &fg)))
        for (std::uint32_t i = 0; i < nf; ++i) add_missing(exts, fg[i].extensionName, avail, "FG instance");
}

void dlaa_add_device_extensions(VkPhysicalDevice phys, std::vector<const char*>& exts) {
    if (!load_bridge()) return;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    if (props.vendorID != 0x10DE) {
        host_log("dlaa: not an NVIDIA GPU; off");
        g_bridge.evaluate = nullptr;
        return;
    }
    std::uint32_t ni = 0, nd = 0;
    const char* const* inst = nullptr;
    const char* const* dev = nullptr;
    if (ngx_failed(g_bridge.required(&ni, &inst, &nd, &dev))) return;
    std::uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> avail(n);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, avail.data());
    for (std::uint32_t i = 0; i < nd; ++i) add_missing(exts, dev[i], avail, "device");
    std::uint32_t nf = 0;
    const VkExtensionProperties* fg = nullptr;
    const std::wstring directory = std::filesystem::path(g_dir).wstring();
    if (g_bridge.fg_extensions && !ngx_failed(g_bridge.fg_extensions(g.instance, phys,
            directory.c_str(), &nf, &fg)))
        for (std::uint32_t i = 0; i < nf; ++i) add_missing(exts, fg[i].extensionName, avail, "FG device");
    // The renderer enables Vulkan 1.2 bufferDeviceAddress. NGX's extension
    // union also names the incompatible pre-core EXT version; enabling both
    // violates device creation requirements. NGX uses the core address entry.
    exts.erase(std::remove_if(exts.begin(), exts.end(), [](const char* name) {
        return !std::strcmp(name, "VK_EXT_buffer_device_address");
    }), exts.end());
}

// The first post-process draw: YEBIS's copy of the lit HDR scene (RGBA16F, the
// target the main depth was drawn with) into its working image. The scene is
// read there and nowhere else. Not the depth-of-field composite after it
// (d3ca03f3+..., its PS hash changes with the settings): it reads the copy,
// whose alpha a draw in between fills, and DLSS does not keep alpha.
// BBHOST_DLAA_ANCHOR=<name prefix> picks another.
bool dlaa_is_anchor(const std::string& name) {
    static const std::string anchor = [] {
        const char* e = std::getenv("BBHOST_DLAA_ANCHOR");
        return std::string(e && *e ? e : "0b0acf50+ccbf44a6");
    }();
    const bool hit = name.compare(0, anchor.size(), anchor) == 0;
    if (hit) host_log("dlaa: anchor pipeline %s", name.c_str());
    return hit;
}

void dlaa_after_draw_locked(const DrawRec& r) {
    if (r.rt0 && !std::strncmp(r.name, "89cda9e4+", 9)) {
        g_sr_display_sizes.erase(r.rt0);
        if (dlss_sr_is_active() && g_fg_current.valid && g_sr_hud_anchor == g_fg_current.frame_index)
            g_sr_display_sizes[r.rt0] = {g_sr_image_ow, g_sr_image_oh};
    }
    fg_latch_locked(r);
    // A final copy consumes this scene. Menus/loading frames without a new
    // HDR anchor must not reuse its SR success or camera/depth guides.
    if (r.rt0 && !std::strncmp(r.name, "89cda9e4+", 9)) g_fg_current.valid = false;
    // Z_ENABLE and Z_WRITE_ENABLE (DB_DEPTH_CONTROL bits 1, 2): a draw into the depth.
    if (r.depth && (r.depth_ctl & 0x6) == 0x6) {
        State::Depth* slot = nullptr;
        for (State::Depth& d : g_s.depths) {
            if (d.base == r.depth) slot = &d;
        }
        if (!slot) {
            slot = &g_s.depths[0];
            for (State::Depth& d : g_s.depths) {
                if (d.last < slot->last) slot = &d;
            }
            const RtImage* rt = find_render_target(r.depth);
            slot->base = r.depth;
            slot->w = rt ? rt->width : 0;
            slot->h = rt ? rt->height : 0;
            slot->writes = 0;
        }
        slot->flip = hle_video_flip_count();
        ++slot->writes;
        slot->last = ++g_s.depth_writes;
        // Said once: 3D frames for ten seconds and never the anchor.
        if (!g_s.anchor_calls && !g_s.anchor_missing_said && g_bridge.evaluate) {
            if (!g_s.first_depth_flip) g_s.first_depth_flip = slot->flip;
            if (slot->flip > g_s.first_depth_flip + 600) {
                g_s.anchor_missing_said = true;
                host_log("dlaa: 600 flips of 3D frames without the anchor pipeline; DLAA is not running (BBHOST_DLAA_ANCHOR names another)");
            }
        }
    }
    if (g_cb.phase != 2 || !g_dlaa_cb_capturing.load(std::memory_order_relaxed)) {
        cb_tick_locked();
        return;
    }
    std::fprintf(g_cb.txt, "draw %llu %s n=%u vp=%.0fx%.0f rt0=0x%llx depth=0x%llx dctl=%08x tex0=0x%llx\n",
                 static_cast<unsigned long long>(g_cb.draws), r.name, r.count, r.vp[2], r.vp[3],
                 static_cast<unsigned long long>(r.rt0), static_cast<unsigned long long>(r.depth), r.depth_ctl,
                 static_cast<unsigned long long>(r.tex[0]));
    for (const CbCapture::Pending& p : g_cb.pending) {
        if (p.seq + 1 != g_draw_rec_next) continue;  // its record was just written: g_draw_rec_next moved on by one
        std::fprintf(g_cb.txt, "  cb %d %zu 0x%llx %llu %llu %llu\n", p.stage, p.slot, static_cast<unsigned long long>(p.base),
                     static_cast<unsigned long long>(p.bytes), static_cast<unsigned long long>(p.at),
                     static_cast<unsigned long long>(p.len));
    }
    g_cb.pending.clear();
    ++g_cb.draws;
    cb_tick_locked();
}

void dlaa_cb_note_locked(std::size_t slot, std::uint64_t base, std::uint64_t bytes) {
    if (t_dlaa_cb_stage < 0 || g_cb.phase != 2 || !g_cb.bin || !base) return;
    const std::uint64_t len = std::min<std::uint64_t>(bytes, 4096);
    if (!g_cb.pending.empty() && g_cb.pending.front().seq != g_draw_rec_next) g_cb.pending.clear();
    CbCapture::Pending p{g_draw_rec_next, t_dlaa_cb_stage, slot, base, bytes, g_cb.bin_bytes, 0};
    if (len && hle_kernel_va_mapped(base, len)) {
        std::fwrite(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(base)), 1, len, g_cb.bin);
        g_cb.bin_bytes += len;
        p.len = len;
    }
    g_cb.pending.push_back(p);
    ++g_cb.buffers;
}

void dlaa_request_cb_capture(const std::string& dir) {
    std::lock_guard<std::mutex> lk(g_cb.mu);
    g_cb.dir = dir;
    g_cb.requested = true;
}

bool dlaa_hotkey(int fkey) {
    if (fkey == 1) {
        const bool on = !g_s.enabled.load();
        g_s.enabled.store(on);
        host_log("dlaa: %s (Ctrl+F1)", on ? "on" : "off");
        return true;
    }
    if (fkey == 2) {
        const int mode = (g_s.jitter_mode.load() + 1) % 3;
        g_s.jitter_mode.store(mode);
        host_log("dlaa: jitter %s (Ctrl+F2 / Ctrl+F6)",
                 mode == 0 ? "on" : mode == 1 ? "off" : "TEST: DLSS skipped, jitter x8 - the scene should shake");
        return true;
    }
    if (fkey == 3) {
        const int sign = -g_s.jitter_sign.load();
        g_s.jitter_sign.store(sign);
        host_log("dlaa: jitter sign given to DLSS %+d (Ctrl+F3 / Ctrl+F7)", sign);
        return true;
    }
    if (fkey == 4) {
        const bool zero = !g_s.mv_zero.load();
        g_s.mv_zero.store(zero);
        host_log("dlaa: motion vectors %s (Ctrl+F4)", zero ? "zero" : "from the camera");
        return true;
    }
    return false;
}

namespace {

// This frame's camera into cam_cur (cam_prev keeps the last one). A frame
// whose draws bound no camera keeps the last, with no camera motion.
void take_camera_locked() {
    if (!g_s.cam_got) {
        if (g_s.cam_cur_ok) {
            std::memcpy(g_s.cam_prev, g_s.cam_cur, sizeof(g_s.cam_prev));
            g_s.cam_prev_ok = true;
            ++g_s.cam_missing;
        }
        return;
    }
    g_s.cam_got = false;
    if (g_s.cam_cur_ok) {
        std::memcpy(g_s.cam_prev, g_s.cam_cur, sizeof(g_s.cam_prev));
        g_s.cam_prev_ok = true;
    }
    std::memcpy(g_s.cam_cur, g_s.cam_next, sizeof(g_s.cam_cur));
    g_s.cam_cur_ok = true;
    ++g_s.cam_frames;
    if (!g_s.cam_checked) {
        g_s.cam_checked = true;
        camera_self_check(g_s.cam_cur);
    }
}

// The motion vectors' matrix: previous ViewProj * the camera's move *
// inverse of this frame's ViewProj (both camera-relative), in double, as
// GLSL's column-major mat4. *cut when the camera jumped or turned too far
// for one frame.
bool reprojection_locked(float out[16], bool* cut) {
    *cut = false;
    Mat4 m{};
    for (int i = 0; i < 4; ++i) m[i][i] = 1.0;
    if (g_s.cam_prev_ok) {
        Mat4 vp_cur, vp_prev, inv, move{};
        cam_vp(g_s.cam_cur, vp_cur);
        cam_vp(g_s.cam_prev, vp_prev);
        double pc[3], pp[3];
        cam_pos(g_s.cam_cur, pc);
        cam_pos(g_s.cam_prev, pp);
        const double d[3] = {pc[0] - pp[0], pc[1] - pp[1], pc[2] - pp[2]};
        // The w rows are the views' forward axes.
        const double turn = vp_cur[3][0] * vp_prev[3][0] + vp_cur[3][1] * vp_prev[3][1] + vp_cur[3][2] * vp_prev[3][2];
        if (std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) > 4.0 || turn < 0.9) *cut = true;
        if (!mat_inverse(vp_cur, inv)) return false;
        for (int i = 0; i < 4; ++i) move[i][i] = 1.0;
        for (int i = 0; i < 3; ++i) move[i][3] = d[i];
        mat_mul(vp_prev, move, m);
        mat_mul(m, inv, m);
    }
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) out[c * 4 + r] = static_cast<float>(m[r][c]);
    return true;
}

bool record_motion_locked(VkCommandBuffer cmd, const float reproject[16], float jx, float jy, bool zero) {
    if (!g_s.mv_tried) {
        g_s.mv_tried = true;
        if (!make_mv_pipeline_locked()) g_s.mv_pipeline = VK_NULL_HANDLE;
        host_log("dlaa: camera motion vectors %s", g_s.mv_pipeline ? "ready" : "unavailable (pipeline creation failed); zero motion");
    }
    if (!g_s.mv_pipeline) return false;
    if (!g_s.object_fallback.image) {
        if (!make_image(g_s.object_fallback, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
            destroy_image(g_s.object_fallback);
            return false;
        }
        to_general(cmd, g_s.object_fallback);
        const VkClearColorValue clear{};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, g_s.object_fallback.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        memory_barrier(cmd);
    }
    const VkImageView objects = object_motion_read_locked(g_s.main_depth, g_s.width, g_s.height, g_s.reset);
    // Optional readback diagnostics can submit this buffer; bind the compute
    // pass into the current recording rather than an already-submitted one.
    begin_recording_locked();
    cmd = g_cmd();
    const VkDescriptorSet set = alloc_set_locked(g_s.mv_set_layout);
    if (!set) return false;
    const VkDescriptorImageInfo depth_info{VK_NULL_HANDLE, g_s.depth.view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo motion_info{VK_NULL_HANDLE, g_s.motion.view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo object_info{VK_NULL_HANDLE, objects ? objects : g_s.object_fallback.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[3] = {};
    for (int k = 0; k < 3; ++k) {
        w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k].dstSet = set;
        w[k].dstBinding = static_cast<std::uint32_t>(k);
        w[k].descriptorCount = 1;
        w[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    w[0].pImageInfo = &depth_info;
    w[1].pImageInfo = &motion_info;
    w[2].pImageInfo = &object_info;
    vkUpdateDescriptorSets(g.device, 3, w, 0, nullptr);
    struct Push {
        float reproject[16];
        float jitter[2];
        std::uint32_t size[2];
        std::uint32_t zero;
        std::uint32_t objects;
        float projection_z;
        std::uint32_t pad;
    } push{};
    static_assert(sizeof(Push) == 96, "the shader's push block");
    std::memcpy(push.reproject, reproject, sizeof(push.reproject));
    push.jitter[0] = jx;
    push.jitter[1] = jy;
    push.size[0] = g_s.width;
    push.size[1] = g_s.height;
    push.zero = zero ? 1u : 0u;
    push.objects = objects && !zero ? 1u : 0u;
    push.projection_z = g_s.cam_cur_ok ? g_s.cam_cur[62] : 1.0f;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_s.mv_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_s.mv_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, g_s.mv_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (g_s.width + 7) / 8, (g_s.height + 7) / 8, 1);
    return true;
}

}  // namespace

void dlaa_anchor_locked(const std::uint64_t* sampled, int count) {
    const std::uint64_t flip = hle_video_flip_count();
    ++g_s.anchor_calls;
    if (g_s.depth_writes == g_s.depth_writes_at_anchor) {
        ++g_s.anchors_this_frame;  // the same frame's again
        return;
    }
    g_s.depth_writes_at_anchor = g_s.depth_writes;
    g_s.anchors_last_frame = g_s.anchors_this_frame;
    g_s.anchors_this_frame = 1;
    // Advance vertex history even if the temporal pass cannot evaluate this frame.
    // Otherwise a later frame could match geometry from before a loading screen.
    struct MotionFrameEnd {
        bool invalidate = true;
        ~MotionFrameEnd() { object_motion_finish_frame_locked(invalidate); }
    } motion_frame;
    g_fg_current.valid = false;
    // The jitter this frame's draws were given. The next frame's is set only
    // when this one evaluates.
    const float jx = g_s.jx, jy = g_s.jy;
    g_s.jx = g_s.jy = 0.0f;
    take_camera_locked();
    // The scene: the first sampled colour target with a depth target of its
    // size; of those depths, the one most drawn into this frame (else the latest).
    RtImage* scene = nullptr;
    const State::Depth* depth = nullptr;
    for (int i = 0; i < count && !scene; ++i) {
        RtImage* rt = sampled[i] ? find_render_target(sampled[i]) : nullptr;
        if (!rt || rt->depth || !rt->initialised || rt->layers != 1) continue;
        for (const State::Depth& d : g_s.depths) {
            if (!d.base || d.w != rt->width || d.h != rt->height || flip - d.flip > 2) continue;
            if (!depth || d.writes > depth->writes || (d.writes == depth->writes && d.flip > depth->flip)) depth = &d;
        }
        if (depth) scene = rt;
    }
    for (State::Depth& d : g_s.depths) d.writes = 0;
    if (g_s.failed) return;
    if (!scene) {
        ++g_s.skipped;
        if (g_s.skipped_run++ == 0) {
            host_log("dlaa: anchor at flip %llu without a scene image and a matching depth (sampled 0x%llx 0x%llx); skipped until there is one",
                     static_cast<unsigned long long>(flip), static_cast<unsigned long long>(count > 0 ? sampled[0] : 0),
                     static_cast<unsigned long long>(count > 1 ? sampled[1] : 0));
        }
        g_s.reset = true;
        return;
    }
    if (g_s.skipped_run) {
        host_log("dlaa: a scene and its depth again at flip %llu, after %llu skipped anchors",
                 static_cast<unsigned long long>(flip), static_cast<unsigned long long>(g_s.skipped_run));
        g_s.skipped_run = 0;
    }
    RtImage* depth_rt = find_render_target(depth->base);
    if (!depth_rt || !depth_rt->depth || !depth_rt->initialised) {
        g_s.reset = true;
        return;
    }
    if (g_s.main_depth != depth_rt->base) g_s.reset = true;  // a new depth target: its frame was drawn without jitter
    g_s.main_depth = depth_rt->base;
    g_s.main_w = depth_rt->width;
    g_s.main_h = depth_rt->height;
    if (g_s.logged_scene != scene->base || g_s.logged_depth != depth_rt->base) {
        g_s.logged_scene = scene->base;
        g_s.logged_depth = depth_rt->base;
        host_log("dlaa: scene 0x%llx %ux%u format %d, depth 0x%llx format %d (flip %llu)", static_cast<unsigned long long>(scene->base),
                 scene->width, scene->height, static_cast<int>(scene->format), static_cast<unsigned long long>(depth_rt->base),
                 static_cast<int>(depth_rt->format), static_cast<unsigned long long>(flip));
    }
    if (!g_s.enabled.load(std::memory_order_relaxed)) {
        g_s.reset = true;
        return;
    }
    if (g_s.jitter_mode.load(std::memory_order_relaxed) == 2) {
        g_s.jitter_index = g_s.jitter_index % 16 + 1;
        g_s.jx = 8.0f * (halton(g_s.jitter_index, 2) - 0.5f);
        g_s.jy = 8.0f * (halton(g_s.jitter_index, 3) - 0.5f);
        g_s.reset = true;
        return;
    }
    if (!config_on() && !fg_requested()) return;
    if (ngx_requested() && !init_ngx_locked() && !fsr_selected()) {
        g_s.failed = true;
        return;
    }
    if (!depth_copy_available_locked()) {
        fail_locked("the depth copy pass is unavailable");
        return;
    }
    if (!ensure_locked(scene->width, scene->height, scene->format)) return;

    begin_recording_locked();
    render_end_pass_locked();
    VkCommandBuffer cmd = g_cmd();
    memory_barrier(cmd);
    if (!depth_copy_record_locked(depth_rt->image, depth_rt->format, g_s.depth.image, scene->width, scene->height)) {
        fail_locked("copying the depth failed");
        return;
    }
    cmd = g_cmd();
    memory_barrier(cmd);
    float reproject[16] = {};
    bool cut = false;
    const bool camera = g_s.cam_cur_ok && reprojection_locked(reproject, &cut);
    if (cut) {
        g_s.reset = true;
        if (++g_s.cuts <= 20) host_log("dlaa: camera cut at flip %llu; history reset", static_cast<unsigned long long>(flip));
    }
    // Without the pass the motion image keeps its zero clear.
    record_motion_locked(cmd, reproject, jx, jy, !camera || g_s.mv_zero.load(std::memory_order_relaxed));
    motion_frame.invalidate = !camera || cut;
    cmd = g_cmd();
    memory_barrier(cmd);
    const auto now = std::chrono::steady_clock::now();
    const float ms = g_s.last_eval.time_since_epoch().count() ? std::chrono::duration<float, std::milli>(now - g_s.last_eval).count() : 16.7f;
    g_s.last_eval = now;
    const float sign = static_cast<float>(g_s.jitter_sign.load(std::memory_order_relaxed));
    NgxbEval e{};
    e.color = {scene->image, scene->view, scene->format, scene->width, scene->height, VK_IMAGE_ASPECT_COLOR_BIT};
    e.output = {g_s.output.image, g_s.output.view, g_s.output.format, scene->width, scene->height, VK_IMAGE_ASPECT_COLOR_BIT};
    e.depth = {g_s.depth.image, g_s.depth.view, VK_FORMAT_R32_SFLOAT, scene->width, scene->height, VK_IMAGE_ASPECT_COLOR_BIT};
    e.motion = {g_s.motion.image, g_s.motion.view, VK_FORMAT_R16G16_SFLOAT, scene->width, scene->height, VK_IMAGE_ASPECT_COLOR_BIT};
    e.jitter_x = jx * sign;
    e.jitter_y = jy * sign;
    e.mv_scale_x = e.mv_scale_y = 1.0f;
    e.reset = g_s.reset ? 1 : 0;
    e.frame_ms = ms;
    const bool sr = guides_only();
    const std::uint32_t r = sr ? 1u : g_bridge.evaluate(cmd, &e);
    if (ngx_failed(r)) {
        if (++g_s.failures <= 5) host_log("dlaa: evaluate failed (0x%08x)", r);
        if (g_s.failures >= 30) fail_locked("30 evaluations failed");
        return;
    }
    if (camera && g_s.mv_pipeline) {
        auto& guides = g_fg_current;
        guides = {};
        guides.valid = true;
        guides.frame_index = g_s.evaluations + 1;
        guides.render_frame_index = guides.frame_index;
        guides.width = scene->width; guides.height = scene->height;
        guides.depth_format = VK_FORMAT_R32_SFLOAT; guides.motion_format = VK_FORMAT_R16G16_SFLOAT;
        guides.frame_ms = ms; guides.reset = e.reset;
        guides.jitter_x = jx; guides.jitter_y = jy;
        auto& camera_out = guides.camera;
        const float* f = g_s.cam_cur;
        Mat4 projection{}, inverse{};
        for (int row = 0; row < 4; ++row) for (int col = 0; col < 4; ++col)
            projection[row][col] = f[52 + row * 4 + col];
        if (!mat_inverse(projection, inverse)) guides.valid = false;
        for (int row = 0; row < 4; ++row) for (int col = 0; col < 4; ++col) {
            camera_out.view_to_clip[row * 4 + col] = static_cast<float>(projection[col][row]);
            camera_out.clip_to_view[row * 4 + col] = static_cast<float>(inverse[col][row]);
        }
        for (int row = 0; row < 4; ++row) for (int col = 0; col < 4; ++col)
            projection[row][col] = reproject[row * 4 + col];
        std::memcpy(camera_out.clip_to_previous, reproject, sizeof(camera_out.clip_to_previous));
        if (!mat_inverse(projection, inverse)) guides.valid = false;
        for (int row = 0; row < 4; ++row) for (int col = 0; col < 4; ++col)
            camera_out.previous_to_clip[row * 4 + col] = static_cast<float>(inverse[row][col]);
        for (int i = 0; i < 3; ++i) {
            camera_out.position[i] = f[183 + i * 4];
            camera_out.right[i] = f[180 + i * 4];
            camera_out.up[i] = f[181 + i * 4];
            camera_out.forward[i] = f[182 + i * 4];
        }
        camera_out.jitter[0] = jx; camera_out.jitter[1] = jy;
        camera_out.near_plane = -f[63] / f[62];
        camera_out.far_plane = f[0] > camera_out.near_plane ? f[0] : 3000.0f;
        camera_out.vertical_fov = 2.0f * std::atan(1.0f / f[57]);
    }
    cmd = g_cmd();
    memory_barrier(cmd);
    if (!sr && g_s.output_same_format) {
        VkImageCopy c{};
        c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.dstSubresource = c.srcSubresource;
        c.extent = {scene->width, scene->height, 1};
        vkCmdCopyImage(cmd, g_s.output.image, VK_IMAGE_LAYOUT_GENERAL, scene->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    } else if (!sr) {
        VkImageBlit b{};
        b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.dstSubresource = b.srcSubresource;
        b.srcOffsets[1] = {static_cast<std::int32_t>(scene->width), static_cast<std::int32_t>(scene->height), 1};
        b.dstOffsets[1] = b.srcOffsets[1];
        vkCmdBlitImage(cmd, g_s.output.image, VK_IMAGE_LAYOUT_GENERAL, scene->image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
    }
    memory_barrier(cmd);
    scene->fill_last = false;
    g_s.reset = false;
    // FSR recommends eight phases times the square of the upscale ratio.
    // Keep native DLSS's existing sequence; FG-only guide capture has no jitter.
    if (config_on() && g_s.jitter_mode.load(std::memory_order_relaxed) == 0) {
        const double ratio = static_cast<double>(host_startup_settings().output_width) /
                             std::max(1u, scene->width);
        const auto phases = fsr_selected() ? std::max(8u, static_cast<unsigned>(std::lround(8.0 * ratio * ratio))) : 16u;
        g_s.jitter_index = g_s.jitter_index % phases + 1;
        g_s.jx = halton(g_s.jitter_index, 2) - 0.5f;
        g_s.jy = halton(g_s.jitter_index, 3) - 0.5f;
    }
    if (++g_s.evaluations == 1 || g_s.evaluations == 300 || g_s.evaluations % 3600 == 0) {
        double pc[3] = {};
        if (g_s.cam_cur_ok) cam_pos(g_s.cam_cur, pc);
        host_log("dlaa: %llu frames evaluated (flip %llu, anchor draws in the frame before: %llu, anchors skipped: %llu); camera in %llu "
                 "frames, kept from the last in %llu, at (%.2f %.2f %.2f); jitter (%.3f %.3f), %llu jittered draws since the last line",
                 static_cast<unsigned long long>(g_s.evaluations), static_cast<unsigned long long>(flip),
                 static_cast<unsigned long long>(g_s.anchors_last_frame), static_cast<unsigned long long>(g_s.skipped),
                 static_cast<unsigned long long>(g_s.cam_frames), static_cast<unsigned long long>(g_s.cam_missing), pc[0], pc[1], pc[2], jx,
                 jy, static_cast<unsigned long long>(g_s.jittered_draws));
        g_s.jittered_draws = 0;
    }
}

}  // namespace gpu
