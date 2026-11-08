// host/upscaler.cpp's Linux path, as bbhost runs it, on a pretend Vulkan
// (tests/fake_vulkan.h stands in for the loader's vkGetInstanceProcAddr):
//   upscaler_load_test none              no bridge beside the executable:
//                                        nothing loads, nothing changes, DLSS
//                                        is not offered and falls back, TAA runs
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
// The executable's folder is a temporary one the test fills (config_exe_dir
// is the test's), and BBHOST_NGX_OFF=1 keeps NVIDIA's code out whatever
// happens.
#include "fake_vulkan.h"
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
    if (mode != "none" && (argc < 3 || (mode != "bridge" && mode != "numity" && mode != "order"))) {
        std::fprintf(stderr, "usage: upscaler_load_test none | bridge LIBRARY | numity LIBRARY | order LIBRARY\n");
        return 2;
    }
    g_exe_dir = temp_dir("exe");
    g_user_dir = temp_dir("cfg");
    check(!g_exe_dir.empty() && !g_user_dir.empty(), "temporary folders");
    const char* name = mode == "numity" ? "libupscaler_bridge.so" : mode == "order" ? "/upscaler_bridge.so" : "libbbhost_ngx_bridge.so";
    if (mode != "none") {
        // A copy, not a link: /proc/self/maps names the file the loader opened.
        const auto copy = [&](const char* as) {
            std::ifstream src(argv[2], std::ios::binary);
            std::ofstream dst(g_exe_dir + "/" + as, std::ios::binary);
            dst << src.rdbuf();
            check(src.good() && dst.good(), "the bridge beside the executable");
        };
        if (mode == "order") {
            for (const char* as : {"upscaler_bridge.so", "libupscaler_bridge.so", "libbbhost_ngx_bridge.so"}) copy(as);
        } else {
            copy(name);
        }
        std::ofstream(g_exe_dir + "/libnvidia-ngx-dlss.so.310.9.16") << "x";
    }
    // The setup window comes first: no device yet.
    check(upscaler_offered(0) && upscaler_offered(1), "Off and TAA are always offered");
    check(!upscaler_offered(2) && !upscaler_offered(3), "FSR 3 and FSR 4 are not offered on Linux");
    check(upscaler_offered(4) == (mode != "none"), "before the device, DLSS is offered where a bridge is beside bbhost");

    const bool opened = upscaler_open();
    check(opened == (mode != "none"), mode == "none" ? "no bridge: nothing opens" : "the bridge opens");
    if (mode != "none") check(mapped(name), "the bridge loaded from the executable's folder under its own name");
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
    check(caps.loaded == opened, "the caps say whether the bridge runs");
    check(!caps.dlss && !caps.fsr && !caps.fsr4, "nothing of the bridge's runs on this GPU");
    if (opened) {
        check(!caps.dlss_reason.empty() && caps.dlss_reason.find("BBHOST_NGX_OFF") == std::string::npos, "DLSS says why, for the GPU's reason");
        check(caps.fsr_reason.find("Windows") != std::string::npos, "FSR says it is Windows'");
    }
    check(upscaler_offered(0) && upscaler_offered(1) && !upscaler_offered(2) && !upscaler_offered(3) && !upscaler_offered(4),
          "with the device up, the Upscaling list offers Off and TAA alone");

    // The methods: DLSS and FSR fall back to the game's own, TAA runs.
    g_settings.upscaler = 1;
    check(upscaler_method() == 0, "DLSS chosen: the game's anti-aliasing instead");
    g_settings.upscaler = 2;
    check(upscaler_method() == 0, "FSR chosen: the game's anti-aliasing instead");
    g_settings.upscaler = 4;
    check(upscaler_method() == kUpscaleTaa, "TAA runs");
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
    std::printf("upscaler_load_test %s: ok\n", mode.c_str());
    return 0;
}
