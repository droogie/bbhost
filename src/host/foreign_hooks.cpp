#include "host/foreign_hooks.h"

#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <link.h>
#endif

namespace {

struct Hook {
    const char* module;  // the file a process holds when the hook is in (Linux: part of the name)
    const char* what;
    bool obs;
};

#if defined(_WIN32)
constexpr Hook kHooks[] = {
    {"graphics-hook64.dll", "OBS game capture", true},
    {"RTSSHooks64.dll", "RivaTuner Statistics Server", false},
    {"GameOverlayRenderer64.dll", "the Steam overlay", false},
    {"DiscordHook64.dll", "the Discord overlay", false},
    {"EOSOverlayRenderer-Win64-Shipping.dll", "the Epic overlay", false},
    {"ReShade64.dll", "ReShade", false},
    {"fraps64.dll", "Fraps", false},
};
#else
constexpr Hook kHooks[] = {
    {"libobs_vkcapture", "OBS game capture (obs-vkcapture)", true},
    {"libMangoHud", "MangoHud", false},
    {"gameoverlayrenderer", "the Steam overlay", false},
    {"libVkLayer_MESA_overlay", "Mesa's overlay layer", false},
};
#endif

bool loaded(const char* module) {
#if defined(_WIN32)
    return GetModuleHandleA(module) != nullptr;
#elif defined(__APPLE__)
    const std::uint32_t n = _dyld_image_count();
    for (std::uint32_t i = 0; i < n; ++i) {
        const char* name = _dyld_get_image_name(i);
        if (name && std::strstr(name, module)) return true;
    }
    return false;
#else
    struct Find {
        const char* name;
        bool found;
    } f{module, false};
    dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* user) {
            auto* f = static_cast<Find*>(user);
            if (info->dlpi_name && std::strstr(info->dlpi_name, f->name)) {
                f->found = true;
                return 1;
            }
            return 0;
        },
        &f);
    return f.found;
#endif
}

}  // namespace

std::string host_foreign_hooks() {
    std::string out;
    for (const Hook& h : kHooks) {
        if (!loaded(h.module)) continue;
        if (!out.empty()) out += ", ";
        out += std::string(h.what) + " (" + h.module + ")";
    }
    return out;
}

bool host_foreign_hooks_obs() {
    for (const Hook& h : kHooks) {
        if (h.obs && loaded(h.module)) return true;
    }
    return false;
}

std::string host_graphics_modules() {
#if defined(_WIN32)
    // D3D12Core.dll comes with d3d12.dll once a D3D12 device is made.
    static const char* const kModules[] = {"dxgi.dll", "d3d11.dll", "d3d12.dll", "D3D12Core.dll", "dcomp.dll"};
    std::string out;
    for (const char* m : kModules) {
        if (!loaded(m)) continue;
        if (!out.empty()) out += " ";
        out += m;
    }
    return out;
#else
    return {};
#endif
}
