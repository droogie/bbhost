// host/upscaler.cpp's Linux path, as bbhost runs it, on a pretend Vulkan
// (tests/fake_vulkan.h stands in for the loader's vkGetInstanceProcAddr):
//   upscaler_load_test none              no bridge or FidelityFX library beside
//                                        the executable: nothing loads, nothing
//                                        changes, DLSS and FSR are not offered
//                                        and fall back, TAA runs
//   upscaler_load_test bridge LIBRARY    bbhost's NGX bridge beside it: it loads
//                                        from the executable's folder, the
//                                        instance and device go as asked, DLSS
//                                        stays off with its reason on this GPU
//                                        (not NVIDIA's) and leaves the
//                                        Upscaling list, evaluate refuses
//   upscaler_load_test numity LIBRARY    the same library under Numity1504's
//                                        Linux name: that one is the one loaded
//   upscaler_load_test order LIBRARY     the same library under all three
//                                        names: the upscaler bridge's own
//                                        (upscaler_bridge.so) is the one loaded
//   upscaler_load_test fsr FFX [BRIDGE]  bbhost's FidelityFX library beside it
//                                        (and the NGX bridge): FSR 3.1.6 is
//                                        offered before and after the device,
//                                        runs when FSR 3 or FSR 4 is chosen,
//                                        with the Windows bridge's jitter and
//                                        LOD bias, adds nothing to the instance
//                                        or device but compute derivatives
//                                        (host/gpu.cpp), and an evaluate it
//                                        refuses turns it off
//   upscaler_load_test fsr-noquad FFX    the same on a device without subgroup
//                                        quad operations in compute shaders:
//                                        not offered, with the reason
// The executable's folder is a temporary one the test fills (config_exe_dir
// is the test's), and BBHOST_NGX_OFF=1 keeps NVIDIA's code out whatever
// happens. A plugin's anti-aliasing claim (plugin API 27) is the test's own:
// taking it over turns TAA off, leaving it to bbhost lets TAA run beside a
// Vulkan layer - BBHOST_UPSCALE_LAYER=1 pretends one (upscaler_load_layer).
#include "fake_vulkan.h"
#include "host/plugins.h"
#include "host/settings.h"
#include "host/upscaler.h"
#include "log.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace {
std::string g_exe_dir, g_user_dir;
HostSettings g_settings;
int g_claim = kAaNone;
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}
std::string temp_dir(const char* tag) {
    const char* base = std::getenv("TMPDIR");
    std::string t = std::string(base && *base ? base : "/tmp") + "/upscaler_load_test_" + tag + "_XXXXXX";
    std::string buf = t;
    return mkdtemp(buf.data()) ? buf : std::string();
}
bool mapped(const char* name) {
    std::ifstream maps("/proc/self/maps");
    std::stringstream ss;
    ss << maps.rdbuf();
    return ss.str().find(name) != std::string::npos;
}
}  // namespace

// What upscaler.cpp asks of the rest of bbhost.
std::string config_exe_dir() { return g_exe_dir; }
std::string config_user_dir() { return g_user_dir; }
const HostSettings& host_settings_ref() { return g_settings; }
int plugins_aa_claim() { return g_claim; }
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance i, const char* name) {
    return fake_vk::get_instance_proc_addr(i, name);
}
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice d, const char* name) {
    return fake_vk::get_device_proc_addr(d, name);
}

int main(int argc, char** argv) {
    setenv("BBHOST_NGX_OFF", "1", 1);
    unsetenv("BBHOST_UPSCALE");
    unsetenv("BBHOST_UPSCALER");
    host_log_decide(true);
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool fsr_mode = mode == "fsr" || mode == "fsr-noquad";
    if (mode != "none" && (argc < 3 || (mode != "bridge" && mode != "numity" && mode != "order" && !fsr_mode))) {
        std::fprintf(stderr, "usage: upscaler_load_test none | bridge LIBRARY | numity LIBRARY | order LIBRARY | fsr FFX [BRIDGE] | fsr-noquad FFX\n");
        return 2;
    }
    g_exe_dir = temp_dir("exe");
    g_user_dir = temp_dir("cfg");
    check(!g_exe_dir.empty() && !g_user_dir.empty(), "temporary folders");
    // (The FSR modes: the FidelityFX library, and the NGX bridge when a second library is given.)
    const char* bridge_file = fsr_mode ? (argc > 3 ? argv[3] : nullptr) : mode == "none" ? nullptr : argv[2];
    const bool fsr_lib = fsr_mode;
    const bool bridge = bridge_file != nullptr;
    const char* name = mode == "numity" ? "libupscaler_bridge.so" : mode == "order" ? "/upscaler_bridge.so" : "libbbhost_ngx_bridge.so";
    // A copy, not a link: /proc/self/maps names the file the loader opened.
    const auto copy = [&](const char* from, const char* as) {
        std::ifstream src(from, std::ios::binary);
        std::ofstream dst(g_exe_dir + "/" + as, std::ios::binary);
        dst << src.rdbuf();
        check(src.good() && dst.good(), "a library beside the executable");
    };
    if (bridge) {
        if (mode == "order") {
            for (const char* as : {"upscaler_bridge.so", "libupscaler_bridge.so", "libbbhost_ngx_bridge.so"}) copy(bridge_file, as);
        } else {
            copy(bridge_file, name);
        }
        std::ofstream(g_exe_dir + "/libnvidia-ngx-dlss.so.310.9.16") << "x";
    }
    if (fsr_lib) copy(argv[2], "libbbhost_fidelityfx_vk.so");
    if (mode == "fsr-noquad") fake_vk::g_quad_compute = false;
    // The setup window comes first: no device yet.
    check(upscaler_offered(0) && upscaler_offered(1), "Off and TAA are always offered");
    check(upscaler_offered(2) == fsr_lib, "before the device, FSR 3 is offered where bbhost's FidelityFX library is beside bbhost");
    check(!upscaler_offered(3), "FSR 4 is never offered on Linux");
    check(upscaler_offered(4) == bridge, "before the device, DLSS is offered where a bridge is beside bbhost");

    const bool opened = upscaler_open();
    check(opened == (bridge || fsr_lib), !bridge && !fsr_lib ? "no bridge or library: nothing opens" : "the bridge or library opens");
    if (bridge) check(mapped(name), "the bridge loaded from the executable's folder under its own name");
    if (fsr_lib) check(mapped("libbbhost_fidelityfx_vk.so"), "the FidelityFX library loaded from the executable's folder");
    check(upscaler_wants_compute_derivatives() == fsr_lib, "compute derivatives asked of the device only for FSR 3");
    if (mode == "numity") check(!mapped("libbbhost_ngx_bridge.so"), "Numity1504's name first: the NGX bridge's not loaded beside it");
    if (mode == "order")
        check(!mapped("libupscaler_bridge.so") && !mapped("libbbhost_ngx_bridge.so"),
              "the upscaler bridge's own name first: neither file after it is loaded");

    // bbhost's instance and device, as host/gpu.cpp makes them.
    const char* iext[] = {"VK_KHR_surface"};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = iext;
    const VkInstanceCreateInfo* ici_use = upscaler_prepare_instance(&ici);
    check(ici_use == &ici, "the instance as bbhost asks it");
    upscaler_instance_done(VK_SUCCESS);
    check(!upscaler_instance_retry(&ici, ici_use), "no second instance");
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    const char* dext[] = {"VK_EXT_external_memory_host"};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dext;
    check(upscaler_prepare_device(fake_vk::instance(), fake_vk::physical_device(), &dci) == &dci, "the device as bbhost asks it");
    upscaler_device_done(VK_SUCCESS);
    check(dci.enabledExtensionCount == 1 && f12.bufferDeviceAddress == VK_TRUE && f2.pNext == &f12, "bbhost's structures as they were");
    upscaler_attach(fake_vk::instance(), fake_vk::physical_device(), fake_vk::device(), 6);

    const UpscalerCaps caps = upscaler_caps();
    const bool fsr_runs = mode == "fsr";
    check(caps.loaded == bridge, "the caps say whether the bridge runs");
    check(caps.fsr_loaded == fsr_lib, "... and whether the FidelityFX library does");
    check(!caps.dlss && !caps.fsr4, "no DLSS (not NVIDIA's GPU) and no FSR 4 here");
    check(caps.fsr == fsr_runs, fsr_runs ? "FSR 3 runs on this GPU" : "FSR does not run here");
    if (bridge) check(!caps.dlss_reason.empty() && caps.dlss_reason.find("BBHOST_NGX_OFF") == std::string::npos, "DLSS says why, for the GPU's reason");
    if (!fsr_lib) check(caps.fsr_reason.find("libbbhost_fidelityfx_vk.so") != std::string::npos, "FSR says what is missing");
    if (fsr_lib) check(caps.fsr_versions == "3.1.6", "the library offers FSR 3.1.6, as AMD's Windows DLL");
    if (mode == "fsr-noquad") check(caps.fsr_reason.find("quad") != std::string::npos, "FSR says what the device lacks");
    check(upscaler_offered(0) && upscaler_offered(1) && upscaler_offered(2) == fsr_runs && !upscaler_offered(3) && !upscaler_offered(4),
          fsr_runs ? "with the device up, the Upscaling list offers Off, TAA and FSR 3" : "with the device up, the Upscaling list offers Off and TAA alone");

    // The methods: DLSS falls back to the game's own, FSR 3 runs where it can (FSR 4 chosen runs it too), TAA runs.
    g_settings.upscaler = 1;
    check(upscaler_method() == 0, "DLSS chosen: the game's anti-aliasing instead");
    g_settings.upscaler = 3;
    check(upscaler_method() == (fsr_runs ? 2 : 0), fsr_runs ? "FSR 3 chosen: FSR runs" : "FSR 3 chosen: the game's anti-aliasing instead");
    g_settings.upscaler = 2;
    check(upscaler_method() == (fsr_runs ? 2 : 0), fsr_runs ? "FSR 4 chosen: FSR 3 runs" : "FSR 4 chosen: the game's anti-aliasing instead");
    if (fsr_runs) {
        // The Windows bridge's numbers: Quality at 1920x1080 draws 1280x720 over 18 phases, LOD bias log2(1280/1920) - 1;
        // frame 1 is Halton's second element.
        float fx = 9, fy = 9;
        upscaler_jitter(2, 1, 1920, 1080, 1, &fx, &fy);
        check(fx == -0.25f && fy > 0.1666f && fy < 0.1667f, "FSR's jitter: Halton (2, 3), centred");
        upscaler_jitter(2, 1, 1920, 1080, 18 + 1, &fx, &fy);
        check(fx == -0.25f && fy > 0.1666f && fy < 0.1667f, "... over 18 phases at Quality");
        const float bias = upscaler_mip_bias(2, 1, 1920, 1080);
        check(bias > -1.5850f && bias < -1.5849f, "FSR's LOD bias at Quality");
        upscaler_jitter_sized(2, 1280, 1920, 1, &fx, &fy);
        check(fx == -0.25f && fy > 0.1666f && fy < 0.1667f, "the same by size");
        // An evaluate the library refuses (an input in no layout): nothing recorded, the plain picture from then on.
        UpscaleCall fc;
        fc.method = 2;
        fc.quality = 1;
        fc.render_w = 1280;
        fc.render_h = 720;
        fc.output_w = 1920;
        fc.output_h = 1080;
        const auto img = [](std::uintptr_t h) {
            UpscaleImage i;
            i.image = reinterpret_cast<VkImage>(h);
            i.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            return i;
        };
        fc.color = img(0x5000);
        fc.color.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        fc.depth = img(0x5001);
        fc.depth.format = VK_FORMAT_R32_SFLOAT;
        fc.motion = img(0x5002);
        fc.motion.format = VK_FORMAT_R16G16_SFLOAT;
        fc.output = img(0x5003);
        check(!upscaler_evaluate(reinterpret_cast<VkCommandBuffer>(static_cast<std::uintptr_t>(0x4000)), fc), "FSR's evaluate refused");
        check(fake_vk::g_barriers == 0, "... before anything was recorded");
        check(upscaler_method() == 0, "... and the plain picture from then on");
    }
    g_settings.upscaler = 4;
    const char* layer_env = std::getenv("BBHOST_UPSCALE_LAYER");
    const bool layer = layer_env && layer_env[0] == '1';
    check(upscaler_method() == (layer ? 0 : kUpscaleTaa), layer ? "TAA off beside a Vulkan layer" : "TAA runs");
    check(upscaler_layer_blocks_taa() == layer, "the layer's note: only beside one");
    // A plugin's claim: taking the anti-aliasing over turns TAA off; leaving it
    // to bbhost lets TAA run, beside a layer too.
    g_claim = kAaTake;
    check(upscaler_method() == 0, "a plugin takes the anti-aliasing over: no TAA");
    g_settings.upscaler = 1;
    check(upscaler_method() == 0, "... nor DLSS");
    g_settings.upscaler = 4;
    g_claim = kAaLeave;
    check(upscaler_method() == kUpscaleTaa, "a plugin leaves the anti-aliasing to bbhost: TAA runs");
    check(!upscaler_layer_blocks_taa(), "... and no layer keeps it off");
    g_claim = kAaNone;
    check(upscaler_method() == (layer ? 0 : kUpscaleTaa), "no claim: as before");
    float x = 9, y = 9;
    upscaler_jitter(kUpscaleTaa, 0, 1920, 1080, 1, &x, &y);
    check(x == -0.25f && y > 0.16f && y < 0.17f, "TAA's jitter");
    upscaler_jitter(0, 0, 1920, 1080, 1, &x, &y);
    check(x == 0 && y == 0 && upscaler_mip_bias(0, 0, 1920, 1080) == 0.0f, "no method, no jitter or bias");
    UpscaleCall c;
    c.method = 1;
    c.render_w = c.output_w = 1920;
    c.render_h = c.output_h = 1080;
    check(!upscaler_evaluate(reinterpret_cast<VkCommandBuffer>(static_cast<std::uintptr_t>(0x4000)), c), "evaluate refused: the plain picture");
    check(fake_vk::g_barriers == 0, "nothing recorded into the command buffer");

    std::system(("rm -rf '" + g_exe_dir + "' '" + g_user_dir + "'").c_str());
    if (g_fail) {
        std::fprintf(stderr, "upscaler_load_test %s: %d failure(s)\n", mode.c_str(), g_fail);
        return 1;
    }
    std::printf("upscaler_load_test %s%s: ok\n", mode.c_str(), layer ? " (a Vulkan layer pretended)" : "");
    return 0;
}
