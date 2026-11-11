// bbhost's NGX bridge (src/ngx_bridge): DLSS on Linux behind Numity1504's
// upscaler bridge API.
//   ngx_bridge_test                the parts without NVIDIA's code: render sizes
//                                  for each quality, the jitter sequence, the mip
//                                  bias, the runtime's file, the create infos
//   ngx_bridge_test LIBRARY        the built bridge loaded as bbhost loads it,
//                                  through the whole call order on a pretend
//                                  Vulkan (tests/fake_vulkan.h) whose GPU is not
//                                  NVIDIA's: one symbol out, the host's instance
//                                  and device untouched, DLSS off with its
//                                  reason, FSR not the bridge's, and evaluate
//                                  refused - never a call into NGX (BBHOST_NGX_OFF
//                                  guards it, and must never be the reason).
#include "fake_vulkan.h"
#include "ngx_bridge/bridge_core.h"
#include "upscaler_bridge.h"

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}
bool near(double a, double b, double eps = 1e-4) { return std::fabs(a - b) < eps; }

std::string temp_dir(const char* tag) {
    const char* base = std::getenv("TMPDIR");
    std::string t = std::string(base && *base ? base : "/tmp") + "/ngx_bridge_test_" + tag + "_XXXXXX";
    std::vector<char> buf(t.begin(), t.end());
    buf.push_back(0);
    return mkdtemp(buf.data()) ? std::string(buf.data()) : std::string();
}
void touch(const std::string& p) { std::ofstream(p) << "x"; }

void test_core() {
    using namespace ngx_bridge;
    // Render sizes: DLSS's rounded (NGX's own), FSR's cut down (AMD's).
    const struct {
        std::uint32_t method, quality, ow, oh, w, h;
    } kSizes[] = {
        {UB_METHOD_DLSS, UB_QUALITY_NATIVE, 1920, 1080, 1920, 1080},      {UB_METHOD_DLSS, UB_QUALITY_QUALITY, 1920, 1080, 1280, 720},
        {UB_METHOD_DLSS, UB_QUALITY_BALANCED, 1920, 1080, 1114, 626},     {UB_METHOD_DLSS, UB_QUALITY_PERFORMANCE, 1920, 1080, 960, 540},
        {UB_METHOD_DLSS, UB_QUALITY_QUALITY, 2560, 1440, 1707, 960},      {UB_METHOD_DLSS, UB_QUALITY_PERFORMANCE, 3840, 2160, 1920, 1080},
        {UB_METHOD_FSR, UB_QUALITY_QUALITY, 2560, 1440, 1706, 960},       {UB_METHOD_FSR, UB_QUALITY_BALANCED, 1920, 1080, 1129, 635},
        {UB_METHOD_FSR, UB_QUALITY_PERFORMANCE, 1920, 1080, 960, 540},    {UB_METHOD_FSR, UB_QUALITY_NATIVE, 1280, 800, 1280, 800},
    };
    for (const auto& s : kSizes) {
        const double r = plain_ratio(s.method, s.quality);
        char what[96];
        std::snprintf(what, sizeof(what), "render size, method %u quality %u at %ux%u", s.method, s.quality, s.ow, s.oh);
        check(render_extent(s.method, s.ow, r) == s.w && render_extent(s.method, s.oh, r) == s.h, what);
    }
    check(plain_ratio(UB_METHOD_DLSS, 4) == 0.0 && plain_ratio(7, UB_QUALITY_QUALITY) == 0.0, "an unknown quality or method has no ratio");
    check(render_extent(UB_METHOD_DLSS, 1, 2.0) == 1, "a render extent is at least 1");
    // The phases: 8 x (output / render)^2, at least 8 (NVIDIA's table: DLAA 8, Quality 18, Balanced 24, Performance 32, Ultra Performance 72).
    check(jitter_phase_count(1920, 1920) == 8, "DLAA's phases");
    check(jitter_phase_count(1280, 1920) == 18, "Quality's phases");
    check(jitter_phase_count(1114, 1920) == 24, "Balanced's phases");
    check(jitter_phase_count(960, 1920) == 32, "Performance's phases");
    check(jitter_phase_count(640, 1920) == 72, "Ultra Performance's phases");
    // The mip bias: log2(render / output) - 1.
    check(near(mip_bias(1920, 1920), -1.0), "native's mip bias is -1");
    check(near(mip_bias(960, 1920), -2.0), "Performance's mip bias is -2");
    check(near(mip_bias(1280, 1920), std::log2(2.0 / 3.0) - 1.0), "Quality's mip bias");
    // The jitter: Halton (2, 3) from its first element, centred, wrapped by the phases.
    float x = 0, y = 0;
    jitter(0, 8, &x, &y);
    check(near(x, 0.0) && near(y, 1.0 / 3.0 - 0.5), "frame 0's jitter is Halton's first element");
    jitter(1, 8, &x, &y);
    check(near(x, 0.25 - 0.5) && near(y, 2.0 / 3.0 - 0.5), "frame 1's jitter");
    jitter(2, 8, &x, &y);
    check(near(x, 0.75 - 0.5) && near(y, 1.0 / 9.0 - 0.5), "frame 2's jitter");
    for (std::uint32_t phases : {8u, 18u, 24u, 32u}) {
        std::set<std::pair<float, float>> seen;
        double sx = 0, sy = 0;
        for (std::uint32_t f = 0; f < phases; ++f) {
            jitter(f, phases, &x, &y);
            check(x > -0.5f && x < 0.5f && y > -0.5f && y < 0.5f, "a jitter inside the pixel");
            seen.insert({x, y});
            sx += x;
            sy += y;
            float x2 = 0, y2 = 0;
            jitter(f + phases, phases, &x2, &y2);
            check(x2 == x && y2 == y, "the sequence wraps at its phases");
        }
        check(seen.size() == phases, "every phase is a sample of its own");
        check(std::fabs(sx / phases) < 0.07 && std::fabs(sy / phases) < 0.07, "the samples are centred on the pixel");
    }
    jitter(5, 0, &x, &y);
    check(x == 0 && y == 0, "no phases, no jitter");
    // TAA's own (host/upscaler.cpp taa_jitter: Halton over 8, from element 1) is the same sequence.
    for (std::uint32_t f = 0; f < 16; ++f) {
        jitter(f, 8, &x, &y);
        const std::uint32_t phase = f % 8 + 1;
        check(x == halton(phase, 2) - 0.5f && y == halton(phase, 3) - 0.5f, "TAA's eight phases are the bridge's");
    }
    // The runtime's file: the newest versioned libnvidia-ngx-dlss.so.*, nothing else.
    const std::string dir = temp_dir("runtime");
    check(!dir.empty(), "a temporary folder");
    std::string name, version;
    check(!find_runtime(dir, &name, &version), "an empty folder has no runtime");
    for (const char* f : {"libnvidia-ngx-dlss.so.310.9.16", "libnvidia-ngx-dlss.so.310.10.0", "libnvidia-ngx-dlss.so.310.2.1",
                          "libnvidia-ngx-dlssd.so.310.11.0", "libnvidia-ngx-dlss.so.bad", "libnvidia-ngx-dlss.so"})
        touch(dir + "/" + f);
    check(find_runtime(dir, &name, &version) && name == "libnvidia-ngx-dlss.so.310.10.0" && version == "310.10.0", "the newest runtime");
    check(compare_versions("310.9.16", "310.10.0") < 0 && compare_versions("310.9.16", "310.9.16") == 0 &&
              compare_versions("310.9.16.1", "310.9.16") > 0 && compare_versions("2.4", "10.0") < 0,
          "versions compared by number");
    std::system(("rm -rf '" + dir + "'").c_str());
    // Paths for NGX: UTF-8 there and back.
    const std::string path = "/opt/bbhost-t\xc3\xab" "st/\xe3\x83\x87\xe3\x83\xbc\xe3\x82\xbf/\xf0\x9f\x8e\xae";
    const std::wstring w = widen(path);
    check(w.size() == 22 && w[13] == 0xeb && w[17] == 0x30c7 && w[19] == 0x30bf && w[21] == 0x1f3ae, "a UTF-8 path widened");
    check(narrow(w.c_str()) == path, "a path back to UTF-8 unchanged");
    check(widen("a\xff" "b") == std::wstring(L"a\xfffd" L"b"), "a bad byte becomes U+FFFD");
    // The instance: NGX's extensions added where the host's lacks them, its own kept.
    const char* host_ext[] = {"VK_KHR_surface", "VK_KHR_xlib_surface"};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = host_ext;
    InstanceCopy ic;
    check(copy_instance(ici, {"VK_KHR_surface", "VK_KHR_get_physical_device_properties2"}, &ic), "an instance copy");
    check(ic.info.enabledExtensionCount == 3 && std::strcmp(ic.info.ppEnabledExtensionNames[2], "VK_KHR_get_physical_device_properties2") == 0 &&
              ic.info.ppEnabledExtensionNames[0] == host_ext[0] && ic.added.size() == 1,
          "the instance copy keeps the host's and adds NGX's");
    InstanceCopy none;
    check(!copy_instance(ici, {"VK_KHR_surface"}, &none) && !copy_instance(ici, {}, &none), "no copy when the host's instance has them");
    // The device: extensions added, buffer device address as the core feature.
    const std::vector<std::string> ngx = {"VK_NVX_binary_import", "VK_NVX_image_view_handle", "VK_EXT_buffer_device_address", "VK_KHR_push_descriptor"};
    const std::vector<std::string> avail = {"VK_NVX_binary_import", "VK_NVX_image_view_handle", "VK_EXT_buffer_device_address",
                                            "VK_KHR_buffer_device_address", "VK_KHR_push_descriptor", "VK_KHR_swapchain"};
    const char* dev_ext[] = {"VK_KHR_swapchain", "VK_KHR_push_descriptor"};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = dev_ext;
    {
        DeviceCopy dc;
        f12.bufferDeviceAddress = VK_FALSE;
        plan_device(&dci, ngx, avail, VK_API_VERSION_1_3, true, &dc);
        check(dc.copied && dc.lacking.empty(), "a device copy");
        check(dc.info.enabledExtensionCount == 4 && dc.info.pNext == &f2, "two NVX extensions added, the host's chain kept");
        bool ext_bda = false;
        for (std::uint32_t i = 0; i < dc.info.enabledExtensionCount; ++i)
            ext_bda |= std::strstr(dc.info.ppEnabledExtensionNames[i], "buffer_device_address") != nullptr;
        check(!ext_bda, "no buffer device address extension beside Vulkan 1.2's feature");
        check(f12.bufferDeviceAddress == VK_TRUE && dc.restore.size() == 1, "bufferDeviceAddress turned on in the host's own structure");
        restore_device(&dc);
        check(f12.bufferDeviceAddress == VK_FALSE, "and put back");
    }
    {
        DeviceCopy dc;
        f12.bufferDeviceAddress = VK_TRUE;
        const char* all[] = {"VK_KHR_swapchain", "VK_KHR_push_descriptor", "VK_NVX_binary_import", "VK_NVX_image_view_handle"};
        VkDeviceCreateInfo full = dci;
        full.enabledExtensionCount = 4;
        full.ppEnabledExtensionNames = all;
        plan_device(&full, ngx, avail, VK_API_VERSION_1_3, true, &dc);
        check(!dc.copied && dc.added.empty() && dc.restore.empty(), "the host's device has them: no copy, nothing changed");
    }
    {
        DeviceCopy dc;
        VkDeviceCreateInfo bare{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        plan_device(&bare, {"VK_EXT_buffer_device_address"}, avail, VK_API_VERSION_1_3, true, &dc);
        check(dc.copied && dc.info.pNext == &dc.bda && dc.bda.bufferDeviceAddress == VK_TRUE && dc.bda.pNext == nullptr,
              "no structure of the host's: the copy's own at the chain's head");
    }
    {
        DeviceCopy dc;
        f12.bufferDeviceAddress = VK_FALSE;
        plan_device(&dci, {"VK_NVX_binary_import", "VK_NVX_made_up", "VK_EXT_buffer_device_address"}, avail, VK_API_VERSION_1_3, true, &dc);
        check(!dc.copied && dc.lacking == std::vector<std::string>{"VK_NVX_made_up"} && f12.bufferDeviceAddress == VK_FALSE,
              "a device that lacks one: nothing added, nothing left changed");
    }
    {
        DeviceCopy dc;
        plan_device(&dci, {"VK_EXT_buffer_device_address"}, avail, VK_API_VERSION_1_3, false, &dc);
        check(!dc.copied && dc.lacking == std::vector<std::string>{"bufferDeviceAddress"}, "a device without buffer device address");
    }
}

// --- the library, as bbhost loads it --------------------------------------------------------------------------------

std::vector<std::string> g_lines;
void UB_CALL log_line(void*, std::int32_t, const char* line) { g_lines.push_back(line); }
bool logged(const char* part) {
    for (const std::string& l : g_lines)
        if (l.find(part) != std::string::npos) return true;
    return false;
}

struct Run {
    const ub_api* api;
    ub_context* ctx = nullptr;
};

// One pass of bbhost's order (host/gpu.cpp init_locked with host/upscaler.cpp)
// on the pretend GPU: the reason DLSS gives.
std::string run_order(const ub_api* api, const std::string& vendor_folder, const std::string& data_folder) {
    ub_open_desc od{};
    od.struct_size = sizeof(od);
    od.vendor_folder = vendor_folder.c_str();
    od.data_folder = data_folder.c_str();
    od.engine_version = "test";
    od.log = &log_line;
    ub_context* c = nullptr;
    check(api->open_context(&od, &c) == UB_OK && c, "open_context");
    ub_context* second = nullptr;
    check(api->open_context(&od, &second) == UB_ERR_STATE && !second, "one context a process");
    check(std::strstr(api->last_error(nullptr), "one context") != nullptr, "last_error(NULL) says why open_context failed");
    ub_caps caps{};
    caps.struct_size = sizeof(caps);
    check(api->get_caps(c, &caps) == UB_OK && !caps.dlss_available && caps.dlss_reason[0], "before a device: no DLSS, and why");
    // The instance as bbhost makes it.
    const char* iext[] = {"VK_KHR_surface"};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = iext;
    const VkInstanceCreateInfo* ici_use = nullptr;
    api->prepare_instance(c, &fake_vk::get_instance_proc_addr, &ici, &ici_use);
    check(ici_use == &ici, "the instance as the host asks it (nothing added off NVIDIA)");
    check(fake_vk::g_live_instances == 0, "a probe instance is destroyed");
    api->instance_done(c, VK_SUCCESS);
    // The device: bbhost's chain, which must come back as it went.
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    const char* dext[] = {"VK_EXT_external_memory_host"};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dext;
    const VkPhysicalDeviceVulkan12Features before = f12;
    const VkDeviceCreateInfo* dci_use = nullptr;
    api->prepare_device(c, fake_vk::instance(), fake_vk::physical_device(), &dci, &dci_use);
    check(dci_use == &dci, "the device as the host asks it (nothing added off NVIDIA)");
    api->device_done(c, VK_SUCCESS);
    check(std::memcmp(&before, &f12, sizeof(f12)) == 0 && dci.enabledExtensionCount == 1 && dci.ppEnabledExtensionNames == dext,
          "the host's structures unchanged");
    // Attach: refused without its handles, then made.
    ub_device_desc dd{};
    dd.struct_size = sizeof(dd);
    check(api->attach_device(c, &dd) == UB_ERR_ARGUMENT, "attach_device refuses NULL handles");
    dd.frames_in_flight = 3;
    dd.instance = fake_vk::instance();
    dd.physical_device = fake_vk::physical_device();
    dd.device = fake_vk::device();
    dd.get_instance_proc_addr = &fake_vk::get_instance_proc_addr;
    dd.get_device_proc_addr = &fake_vk::get_device_proc_addr;
    check(api->attach_device(c, &dd) == UB_OK, "attach_device");
    check(api->attach_device(c, &dd) == UB_ERR_STATE, "one device at a time");
    caps = ub_caps{};
    caps.struct_size = sizeof(caps);
    check(api->get_caps(c, &caps) == UB_OK, "get_caps");
    check(!caps.dlss_available && !caps.fsr_available && !caps.fsr4_available && !caps.fsr3_choice, "nothing runs on this GPU");
    check(std::strncmp(caps.bridge_version, "ngx-bridge ", 11) == 0, "the bridge says which it is");
    check(std::strstr(caps.fsr_reason, "libbbhost_fidelityfx_vk.so") != nullptr, "FSR's reason: the FidelityFX library's, not the bridge's");
    const std::string reason = caps.dlss_reason;
    check(!reason.empty() && reason.find("BBHOST_NGX_OFF") == std::string::npos, "DLSS's reason is the GPU's, never the tests' guard");
    // A short struct: only what it holds is written.
    unsigned char raw[sizeof(ub_caps)];
    std::memset(raw, 0xab, sizeof(raw));
    const std::uint32_t short_size = 3 * sizeof(std::uint32_t);
    std::memcpy(raw, &short_size, sizeof(short_size));
    check(api->get_caps(c, reinterpret_cast<ub_caps*>(raw)) == UB_OK && raw[short_size] == 0xab && raw[sizeof(raw) - 1] == 0xab,
          "a short ub_caps is written only as far as it goes");
    // Render sizes, mip bias and jitter: the plain ratios here.
    ub_render_info ri{};
    ri.struct_size = sizeof(ri);
    check(api->get_render_info(c, UB_METHOD_DLSS, UB_QUALITY_QUALITY, 1920, 1080, &ri) == UB_OK && ri.width == 1280 && ri.height == 720 &&
              ri.jitter_phases == 18 && near(ri.mip_bias, std::log2(2.0 / 3.0) - 1.0),
          "DLSS Quality at 1080p");
    check(api->get_render_info(c, UB_METHOD_DLSS, UB_QUALITY_NATIVE, 2560, 1440, &ri) == UB_OK && ri.width == 2560 && ri.jitter_phases == 8 &&
              near(ri.mip_bias, -1.0),
          "DLAA");
    check(api->get_render_info(c, UB_METHOD_FSR, UB_QUALITY_BALANCED, 1920, 1080, &ri) == UB_OK && ri.width == 1129 && ri.height == 635,
          "FSR's plain ratio");
    check(api->get_render_info(c, 9, UB_QUALITY_QUALITY, 1920, 1080, &ri) == UB_ERR_ARGUMENT, "an unknown method");
    check(api->get_render_info(c, UB_METHOD_DLSS, 9, 1920, 1080, &ri) == UB_ERR_ARGUMENT, "an unknown quality");
    float jx = 1, jy = 1, kx = 0, ky = 0;
    api->get_jitter(3, 18, &jx, &jy);
    ngx_bridge::jitter(3, 18, &kx, &ky);
    check(jx == kx && jy == ky, "get_jitter is the core's sequence");
    // evaluate: refused, its images untouched (no barrier recorded).
    ub_evaluate_desc ed{};
    ed.struct_size = sizeof(ed);
    ed.method = UB_METHOD_DLSS;
    ed.quality = UB_QUALITY_QUALITY;
    ed.render_width = 1280;
    ed.render_height = 720;
    ed.output_width = 1920;
    ed.output_height = 1080;
    const auto img = [](std::uintptr_t i) {
        ub_image u{};
        u.image = reinterpret_cast<VkImage>(i);
        u.view = reinterpret_cast<VkImageView>(i + 1);
        u.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        u.layout = VK_IMAGE_LAYOUT_GENERAL;
        return u;
    };
    ed.color = img(0x10);
    ed.depth = img(0x20);
    ed.depth.format = VK_FORMAT_D32_SFLOAT;
    ed.motion = img(0x30);
    ed.output = img(0x40);
    const auto cb = reinterpret_cast<VkCommandBuffer>(static_cast<std::uintptr_t>(0x4000));
    const int barriers = fake_vk::g_barriers;
    check(api->evaluate(c, cb, &ed) == UB_ERR_UNAVAILABLE && std::strstr(api->last_error(c), "DLSS is not available"), "evaluate: DLSS refused");
    ed.method = UB_METHOD_FSR;
    check(api->evaluate(c, cb, &ed) == UB_ERR_UNAVAILABLE && std::strstr(api->last_error(c), "FSR is not available"), "evaluate: FSR refused");
    ed.method = UB_METHOD_DLSS;
    ed.quality = UB_QUALITY_NATIVE;
    check(api->evaluate(c, cb, &ed) == UB_ERR_ARGUMENT, "evaluate: native needs the render size = the output size");
    ed.quality = UB_QUALITY_QUALITY;
    ed.color.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    check(api->evaluate(c, cb, &ed) == UB_ERR_ARGUMENT, "evaluate: an input in UNDEFINED");
    ed.color.layout = VK_IMAGE_LAYOUT_GENERAL;
    ed.struct_size = 8;
    check(api->evaluate(c, cb, &ed) == UB_ERR_ARGUMENT, "evaluate: an old struct_size");
    ed.struct_size = sizeof(ed);
    check(api->evaluate(c, VK_NULL_HANDLE, &ed) == UB_ERR_ARGUMENT, "evaluate: no command buffer");
    check(fake_vk::g_barriers == barriers, "nothing recorded by a refused evaluate");
    api->detach_device(c);
    check(api->evaluate(c, cb, &ed) == UB_ERR_STATE, "evaluate after detach_device");
    api->close_context(c);
    return reason;
}

int test_library(const char* path) {
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::fprintf(stderr, "FAIL: %s did not load: %s\n", path, dlerror());
        return 1;
    }
    auto get_api = reinterpret_cast<PFN_ub_get_api>(dlsym(h, "ub_get_api"));
    check(get_api != nullptr, "ub_get_api is exported");
    if (!get_api) return 1;
    check(get_api(UB_API_VERSION + 1) == nullptr && get_api(0) == nullptr, "an API version it does not know: NULL");
    const ub_api* api = get_api(UB_API_VERSION);
    check(api && api->struct_size == sizeof(ub_api) && api->api_version == UB_API_VERSION, "the table of API 1");
    if (!api) return 1;
    const void* const* fns = reinterpret_cast<const void* const*>(&api->open_context);
    for (int i = 0; i < 13; ++i) check(fns[i] != nullptr, "every pointer of the table is set");
    // NVIDIA's library stays inside: none of its functions are exported.
    for (const char* sym : {"NVSDK_NGX_VULKAN_Init", "NVSDK_NGX_VULKAN_Init_with_ProjectID", "NVSDK_NGX_VULKAN_CreateFeature", "GetNGXResultAsString"})
        check(dlsym(h, sym) == nullptr, "an NGX function exported");
    ub_context* c = nullptr;
    ub_open_desc old{};
    old.struct_size = 8;
    check(api->open_context(&old, &c) == UB_ERR_ARGUMENT && !c && std::strstr(api->last_error(nullptr), "struct_size"), "an old ub_open_desc");

    const std::string vendor = temp_dir("vendor"), data = temp_dir("data") + "/ngx";
    // 1. No runtime in the vendor folder.
    std::string why = run_order(api, vendor, data);
    check(why.find("libnvidia-ngx-dlss.so is not in the vendor folder") != std::string::npos || why.find("NVIDIA RTX card") != std::string::npos,
          "no runtime: DLSS off and why");
    // 2. The runtime there (a pretend file) and an AMD GPU.
    touch(vendor + "/libnvidia-ngx-dlss.so.310.9.16");
    g_lines.clear();
    why = run_order(api, vendor, data);
    check(why.find("DLSS needs an NVIDIA RTX card (this GPU is Pretend AMD GPU)") != std::string::npos, "an AMD GPU: DLSS needs an NVIDIA RTX card");
    check(logged("libnvidia-ngx-dlss.so.310.9.16") && logged("attached: DLSS off"), "the bridge logs its runtime and what is off");
    // 3. An NVIDIA GPU run by NVK (Mesa): not NVIDIA's driver.
    fake_vk::g_vendor = 0x10de;
    fake_vk::g_name = "Pretend NVIDIA GPU";
    fake_vk::g_driver = VK_DRIVER_ID_MESA_NVK;
    why = run_order(api, vendor, data);
    check(why.find("NVIDIA's own driver") != std::string::npos || why.find("NVIDIA's driver, which is not loaded") != std::string::npos,
          "an NVIDIA GPU on NVK: DLSS needs NVIDIA's own driver");
    struct stat st{};
    check(stat(data.c_str(), &st) != 0, "NGX's data folder made only when NGX starts");
    std::system(("rm -rf '" + vendor + "' '" + data.substr(0, data.size() - 4) + "'").c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // The guard: whatever happens, this test never reaches NVIDIA's code.
    setenv("BBHOST_NGX_OFF", "1", 1);
    test_core();
    if (argc > 1) test_library(argv[1]);
    if (g_fail) {
        std::fprintf(stderr, "ngx_bridge_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("ngx_bridge_test: ok%s\n", argc > 1 ? " (the library too)" : " (the core; no library given)");
    return 0;
}
