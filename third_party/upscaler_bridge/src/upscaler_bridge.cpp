// upscaler_bridge.cpp -- upscaler_bridge.dll: NVIDIA DLSS and AMD FSR behind one small C API (include/upscaler_bridge.h).
//
// Copyright (c) 2026 Numity1504. MIT licence: see LICENSE.txt. The credit must stay.
// By Numity1504 (BBsuite), using Claude by Anthropic as assistant.
//
// Built with MSVC and a static CRT by build.cmd. With UB_WITH_DLSS it links NVIDIA's nvsdk_ngx_s.lib (not part of this source);
// without it the dll does FSR only. AMD's code is loaded at run time, its structures come from the MIT headers in third_party.
//
// This software contains source code provided by NVIDIA Corporation (the inline helpers of the DLSS SDK's headers, in a build with DLSS).
//
// Linux (bbhost's port, tools/build_upscaler_bridge_linux.sh): upscaler_bridge.so, built with gcc or clang against NVIDIA's
// libnvsdk_ngx.a. NGX's runtime there is libnvidia-ngx-dlss.so.<version> in the vendor folder (the driver brings the rest), and
// FSR is libamd_fidelityfx_vk.so - one library with AMD's loader and providers, built from the FidelityFX SDK's MIT source,
// since AMD ships its signed files for Windows only. A crash inside the vendors' code is not caught there (no SEH).

#define VK_NO_PROTOTYPES
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "upscaler_bridge.h"

#include "ffx_api/ffx_api.h"
#include "ffx_api/ffx_upscale.h"
#include "ffx_api/vk/ffx_api_vk.h"

#if defined(UB_WITH_DLSS)
#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_vk.h"
#endif

#define UB_VERSION_TEXT "1.1.0"

namespace {

#if defined(_WIN32)
constexpr const char *kDlssFile = "nvngx_dlss.dll";
constexpr const char *kFsrLoaderFile = "amd_fidelityfx_loader_vk.dll";
constexpr const char *kFsrUpscalerFile = "amd_fidelityfx_upscaler_vk.dll";
constexpr const wchar_t *kSep = L"\\";
#else
constexpr const char *kDlssFile = "libnvidia-ngx-dlss.so"; // with its version after it: libnvidia-ngx-dlss.so.310.9.16
constexpr const char *kFsrLoaderFile = "libamd_fidelityfx_vk.so";
constexpr const char *kFsrUpscalerFile = "libamd_fidelityfx_vk.so";
constexpr const wchar_t *kSep = L"/";
#endif
constexpr uint32_t kVendorNvidia = 0x10DE;
constexpr uint32_t kVendorAmd = 0x1002;

// NGX wants a project GUID when there is no application id from NVIDIA. The host can give its own (ub_open_desc.project_id).
constexpr const char *kDefaultProject = "104f5f16-0223-43d3-a6d7-8e0181364408";

// FSR SDK 2.x's own structure types, from the public headers of FidelityFX SDK 2.4.0. third_party holds the 1.1.4 headers (MIT), whose
// structures match 2.4's field for field; these three values are what changed. The 1.1.4 backend type 0x3 makes no context at all with
// the 2.x loader. FFX_API_MAKE_BACKEND_SUB_ID(FFX_API_BACKEND_ID_VK, 0x03):
constexpr uint64_t kFfxBackendVk = 0x01000003u;
// FFX_API_MAKE_EFFECT_SUB_ID(FFX_API_EFFECT_ID_UPSCALE, 0x0b): names the header version the structures follow
constexpr uint64_t kFfxUpscaleVersionDesc = 0x0001000bu;
constexpr uint32_t kFfxUpscalerVersion = (4u << 22) | (1u << 12) | 2u; // FFX_UPSCALER_VERSION 4.1.2
struct FfxUpscaleVersionDesc {
    ffxCreateContextDescHeader header;
    uint32_t version;
};
// the version list's "device" on Vulkan is this, not the VkDevice
struct FfxVkDeviceContext {
    VkDevice vkDevice;
    VkPhysicalDevice vkPhysicalDevice;
    PFN_vkGetDeviceProcAddr vkDeviceProcAddr;
};

std::mutex g_open_mx;
char g_open_error[256] = {};

template <class T> bool size_ok(const T *p) { return p && p->struct_size >= sizeof(T); }

void copy_text(char *to, size_t n, const std::string &s) {
    const size_t k = std::min(n - 1, s.size());
    std::memcpy(to, s.data(), k);
    to[k] = 0;
}

std::string hex(uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof(b), "0x%08x", v);
    return b;
}

#if defined(_WIN32)
using CrashCode = DWORD;

std::wstring widen(const std::string &s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(size_t(n > 0 ? n - 1 : 0), L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

std::wstring folder_of(const std::wstring &path) {
    const size_t k = path.find_last_of(L"\\/");
    return k == std::wstring::npos ? std::wstring() : path.substr(0, k);
}

std::wstring module_path(HMODULE m) {
    std::wstring p(1024, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(m, &p[0], DWORD(p.size()));
        if (n == 0) return {};
        if (n < p.size()) {
            p.resize(n);
            return p;
        }
        p.resize(p.size() * 2);
    }
}

bool file_there(const std::wstring &p) {
    const DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

std::string file_version(const std::wstring &p) {
    DWORD h = 0;
    const DWORD n = GetFileVersionInfoSizeW(p.c_str(), &h);
    if (!n) return {};
    std::vector<unsigned char> b(n);
    if (!GetFileVersionInfoW(p.c_str(), 0, n, b.data())) return {};
    VS_FIXEDFILEINFO *fi = nullptr;
    UINT len = 0;
    if (!VerQueryValueW(b.data(), L"\\", reinterpret_cast<void **>(&fi), &len) || !fi) return {};
    char t[32];
    std::snprintf(t, sizeof(t), "%u.%u.%u.%u", unsigned(HIWORD(fi->dwFileVersionMS)), unsigned(LOWORD(fi->dwFileVersionMS)),
                  unsigned(HIWORD(fi->dwFileVersionLS)), unsigned(LOWORD(fi->dwFileVersionLS)));
    return t;
}

// the version resource's ProductName ("" none)
std::wstring file_product(const std::wstring &p) {
    DWORD h = 0;
    const DWORD n = GetFileVersionInfoSizeW(p.c_str(), &h);
    if (!n) return {};
    std::vector<unsigned char> b(n);
    if (!GetFileVersionInfoW(p.c_str(), 0, n, b.data())) return {};
    struct Lang {
        WORD lang, page;
    } *t = nullptr;
    UINT len = 0;
    if (!VerQueryValueW(b.data(), L"\\VarFileInfo\\Translation", reinterpret_cast<void **>(&t), &len) || !t || len < sizeof(Lang)) return {};
    wchar_t key[64];
    swprintf(key, 64, L"\\StringFileInfo\\%04x%04x\\ProductName", unsigned(t->lang), unsigned(t->page));
    wchar_t *v = nullptr;
    if (!VerQueryValueW(b.data(), key, reinterpret_cast<void **>(&v), &len) || !v) return {};
    return v;
}

// Only one upscaler can run. A vulkan-1.dll beside the exe that is not Khronos's own loader is a wrapper with an upscaler of its own.
bool other_vulkan_dll() {
    const std::wstring p = folder_of(module_path(nullptr)) + L"\\vulkan-1.dll";
    return file_there(p) && file_product(p).find(L"Vulkan") != 0;
}

// the vendor files by name, as they are
std::wstring vendor_file(const std::wstring &folder, const char *name) { return folder + kSep + widen(name); }

void *load_library(const std::wstring &p) { return LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH); }
void *library_symbol(void *lib, const char *name) { return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(lib), name)); }
std::string load_error() { return "Windows error " + std::to_string(GetLastError()); }

std::wstring self_folder() {
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&self_folder), &self);
    return folder_of(module_path(self));
}

std::wstring full_path(const std::wstring &p) {
    wchar_t full[2048];
    const DWORD n = GetFullPathNameW(p.c_str(), 2048, full, nullptr); // (a dll by a relative path is not loaded reliably)
    return n > 0 && n < 2048 ? std::wstring(full) : p;
}

std::wstring temp_folder() {
    wchar_t t[MAX_PATH + 1] = {};
    const DWORD n = GetTempPathW(MAX_PATH + 1, t);
    return std::wstring(t, n);
}

using GuardFn = void (*)(void *);
int guard_filter(EXCEPTION_POINTERS *ep, DWORD *code) {
    *code = ep->ExceptionRecord->ExceptionCode;
    return EXCEPTION_EXECUTE_HANDLER;
}
bool guard_call(GuardFn fn, void *arg, DWORD *code) {
    __try {
        fn(arg);
        return true;
    } __except (guard_filter(GetExceptionInformation(), code)) {
        return false;
    }
}

#else // Linux
using CrashCode = uint32_t;

// UTF-8 to wchar_t (UTF-32 here), which NGX's paths and AMD's messages use on Linux as well
std::wstring widen(const std::string &s) {
    std::wstring w;
    for (size_t i = 0; i < s.size();) {
        const unsigned char b = static_cast<unsigned char>(s[i]);
        const int n = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 1;
        uint32_t cp = n == 1 ? b : n == 2 ? (b & 0x1fu) : n == 3 ? (b & 0x0fu) : (b & 0x07u);
        for (int k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3fu);
        w += static_cast<wchar_t>(cp);
        i += n;
    }
    return w;
}

std::string narrow(const std::wstring &w) {
    std::string s;
    for (wchar_t c : w) {
        const uint32_t cp = static_cast<uint32_t>(c);
        if (cp < 0x80) {
            s += char(cp);
        } else if (cp < 0x800) {
            s += char(0xc0 | (cp >> 6));
            s += char(0x80 | (cp & 0x3f));
        } else if (cp < 0x10000) {
            s += char(0xe0 | (cp >> 12));
            s += char(0x80 | ((cp >> 6) & 0x3f));
            s += char(0x80 | (cp & 0x3f));
        } else {
            s += char(0xf0 | (cp >> 18));
            s += char(0x80 | ((cp >> 12) & 0x3f));
            s += char(0x80 | ((cp >> 6) & 0x3f));
            s += char(0x80 | (cp & 0x3f));
        }
    }
    return s;
}

std::wstring folder_of(const std::wstring &path) {
    const size_t k = path.find_last_of(L'/');
    return k == std::wstring::npos ? std::wstring() : path.substr(0, k);
}

bool file_there(const std::wstring &p) {
    struct stat st{};
    return stat(narrow(p).c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// NGX's runtime carries its version in its name (libnvidia-ngx-dlss.so.310.9.16), and NGX looks for it by that name: the newest
// there, "" none.
std::string dlss_file_in(const std::wstring &folder) {
    const std::string prefix = std::string(kDlssFile) + ".";
    std::string best;
    std::vector<int> best_v;
    if (DIR *d = opendir(narrow(folder).c_str())) {
        while (dirent *e = readdir(d)) {
            const std::string n = e->d_name;
            if (n.compare(0, prefix.size(), prefix) != 0 || n.size() == prefix.size()) continue;
            std::vector<int> v;
            for (const char *p = n.c_str() + prefix.size(); *p;) {
                v.push_back(std::atoi(p));
                while (*p && *p != '.') ++p;
                if (*p) ++p;
            }
            if (best.empty() || v > best_v) best = n, best_v = v;
        }
        closedir(d);
    }
    return best;
}

std::wstring vendor_file(const std::wstring &folder, const char *name) {
    if (std::strcmp(name, kDlssFile) == 0) {
        const std::string f = dlss_file_in(folder);
        return f.empty() ? std::wstring() : folder + kSep + widen(f);
    }
    return folder + kSep + widen(name);
}

// "310.9.16.0" from libnvidia-ngx-dlss.so.310.9.16, the way the dll's version resource reads on Windows
std::string file_version(const std::wstring &p) {
    const std::string n = narrow(p.substr(p.find_last_of(L'/') + 1));
    const std::string prefix = std::string(kDlssFile) + ".";
    if (n.compare(0, prefix.size(), prefix) != 0) return {};
    std::string v = n.substr(prefix.size());
    for (int dots = int(std::count(v.begin(), v.end(), '.')); dots < 3; ++dots) v += ".0";
    return v;
}

// (no wrapper loader beside a Linux binary to keep the upscaler from)
bool other_vulkan_dll() { return false; }

void *load_library(const std::wstring &p) { return dlopen(narrow(p).c_str(), RTLD_NOW | RTLD_LOCAL); }
void *library_symbol(void *lib, const char *name) { return dlsym(lib, name); }
std::string load_error() {
    const char *e = dlerror();
    return e ? e : "dlopen failed";
}

std::wstring self_folder() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(&self_folder), &info) || !info.dli_fname) return {};
    return folder_of(widen(info.dli_fname));
}

std::wstring full_path(const std::wstring &p) {
    char full[PATH_MAX];
    return realpath(narrow(p).c_str(), full) ? widen(full) : p;
}

std::wstring temp_folder() {
    const char *t = std::getenv("TMPDIR");
    return widen(t && *t ? t : "/tmp");
}

using GuardFn = void (*)(void *);
bool guard_call(GuardFn fn, void *arg, CrashCode *) {
    fn(arg);
    return true;
}
#endif

enum { kDlss = 0, kFsr = 1 };

// ---- what the device needs ------------------------------------------------------------------------------------------------------

// where a feature lives: in a Vulkan 1.x features structure (0: none) and in a structure of its own. The two may not sit side by side in
// one chain, so the host's chain is searched for either, and a structure of ours is added only when the host has neither.
struct Spot {
    const char *name;
    VkStructureType core;
    size_t core_off;
    VkStructureType own;
    size_t own_off;
    size_t own_size;
};

struct DeviceMerge {
    const VkDeviceCreateInfo *host = nullptr;
    VkDeviceCreateInfo ci{};
    VkPhysicalDeviceFeatures core{}; // (shaderInt16 where the host gives no Features2)
    std::vector<std::string> names;  // the extensions added
    std::vector<const char *> exts;
    std::vector<std::pair<VkBool32 *, VkBool32>> touched; // the host's fields set for the call, with their values
    std::vector<std::unique_ptr<unsigned char[]>> ours;   // structures of ours in the chain
    bool changed = false;

    void start(const VkDeviceCreateInfo *in) {
        host = in;
        ci = *in;
        names.clear();
        exts.clear();
        touched.clear();
        ours.clear();
        changed = false;
    }
    bool has_ext(const char *n) const {
        for (uint32_t i = 0; i < host->enabledExtensionCount && host->ppEnabledExtensionNames; ++i)
            if (host->ppEnabledExtensionNames[i] && std::strcmp(host->ppEnabledExtensionNames[i], n) == 0) return true;
        for (const std::string &s : names)
            if (s == n) return true;
        return false;
    }
    void add_ext(const char *n) {
        if (has_ext(n)) return;
        names.push_back(n);
        changed = true;
    }
    // true: it was off and is on now; false: the host had it on
    bool turn_on(const Spot &s) {
        VkBool32 *at = nullptr;
        for (auto *p = static_cast<const VkBaseInStructure *>(host->pNext); p && !at; p = p->pNext) {
            auto *b = reinterpret_cast<char *>(const_cast<VkBaseInStructure *>(p));
            if (s.core && p->sType == s.core) at = reinterpret_cast<VkBool32 *>(b + s.core_off);
            else if (p->sType == s.own) at = reinterpret_cast<VkBool32 *>(b + s.own_off);
        }
        if (at) {
            if (*at == VK_TRUE) return false;
            touched.push_back({at, *at});
            *at = VK_TRUE;
            changed = true;
            return true;
        }
        for (auto &o : ours) {
            auto *b = reinterpret_cast<VkBaseOutStructure *>(o.get());
            if (b->sType == s.own) {
                *reinterpret_cast<VkBool32 *>(o.get() + s.own_off) = VK_TRUE;
                return true;
            }
        }
        std::unique_ptr<unsigned char[]> o(new unsigned char[s.own_size]());
        auto *b = reinterpret_cast<VkBaseOutStructure *>(o.get());
        b->sType = s.own;
        b->pNext = static_cast<VkBaseOutStructure *>(const_cast<void *>(ci.pNext));
        *reinterpret_cast<VkBool32 *>(o.get() + s.own_off) = VK_TRUE;
        ci.pNext = b;
        ours.push_back(std::move(o));
        changed = true;
        return true;
    }
    bool turn_on_shader_int16() {
        VkBool32 *at = nullptr;
        for (auto *p = static_cast<const VkBaseInStructure *>(host->pNext); p && !at; p = p->pNext)
            if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
                at = &reinterpret_cast<VkPhysicalDeviceFeatures2 *>(const_cast<VkBaseInStructure *>(p))->features.shaderInt16;
        if (at) {
            if (*at == VK_TRUE) return false;
            touched.push_back({at, *at});
            *at = VK_TRUE;
            changed = true;
            return true;
        }
        if (ci.pEnabledFeatures && ci.pEnabledFeatures->shaderInt16 == VK_TRUE) return false;
        core = ci.pEnabledFeatures ? *ci.pEnabledFeatures : VkPhysicalDeviceFeatures{};
        core.shaderInt16 = VK_TRUE;
        ci.pEnabledFeatures = &core;
        changed = true;
        return true;
    }
    const VkDeviceCreateInfo *finish() {
        if (!changed) return host;
        exts.clear();
        for (uint32_t i = 0; i < host->enabledExtensionCount && host->ppEnabledExtensionNames; ++i) exts.push_back(host->ppEnabledExtensionNames[i]);
        for (const std::string &s : names) exts.push_back(s.c_str());
        ci.enabledExtensionCount = uint32_t(exts.size());
        ci.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();
        return &ci;
    }
    void restore() {
        for (auto it = touched.rbegin(); it != touched.rend(); ++it) *it->first = it->second;
        touched.clear();
    }
};

// ---- the context ----------------------------------------------------------------------------------------------------------------

struct DlssState {
    bool file = false;
    bool ready = false;  // started, evaluate may run
    bool device_ok = false; // the device was made with what NGX named
    bool asked = false;  // prepare_device asked for something
    int major = 0;       // nvngx_dlss.dll's first number; presets exist from 310 on
    std::string version, reason, device_why;
    VkDevice started_on = VK_NULL_HANDLE;
    VkPhysicalDevice asked_gpu = VK_NULL_HANDLE; // NGX answers its device query only until its first shutdown: the answer is kept
    std::vector<std::string> need;
#if defined(UB_WITH_DLSS)
    NVSDK_NGX_Parameter *params = nullptr;
    NVSDK_NGX_Handle *handle = nullptr;
    NVSDK_NGX_FeatureCommonInfo common{};
    const wchar_t *paths[1] = {nullptr};
    std::vector<std::pair<NVSDK_NGX_Handle *, uint64_t>> retired;
#endif
    uint32_t rw = 0, rh = 0, ow = 0, oh = 0, quality = 0, flags = 0, preset = 0;
};

struct FsrState {
    bool files = false;
    void *dll = nullptr; // (never closed: a driver thread may still run its code)
    bool load_tried = false;
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    bool f4_device_ok = false; // the device was made with FSR 4's features
    bool f4_asked = false;
    std::string f4_why;
    bool newest_is_4 = false;
    uint64_t v3_id = 0; // AMD's id of the newest version below 4 (0: none offered)
    std::string versions, reason, running;
    ffxContext ctx = nullptr;
    uint64_t ctx_override = 0;
    uint32_t rw = 0, rh = 0, ow = 0, oh = 0, flags = 0;
    std::vector<std::pair<ffxContext, uint64_t>> retired;
    std::vector<std::string> told; // AMD's messages said once a context
    std::vector<std::string> core_names;
};

struct Ctx {
    std::mutex mx;
    uint32_t open_flags = 0;
    std::wstring vendor, data;
    std::string project_id, engine_version;
    ub_log_fn log = nullptr;
    void *log_user = nullptr;
    char error[256] = {};
    bool dead[2] = {false, false}; // a crash inside the vendor's code: never called again

    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;
    uint32_t instance_api = VK_API_VERSION_1_0;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkCmdPipelineBarrier cmd_barrier = nullptr;
    uint32_t in_flight = 8;
    uint64_t frame = 0;

    VkInstanceCreateInfo ici{};
    std::vector<std::string> inst_names;
    std::vector<const char *> inst_exts;
    bool inst_asked = false;

    DeviceMerge merge;
    bool merge_open = false;
    DlssState dlss;
    FsrState fsr;
};

// NGX and AMD call back with no user pointer: the one context of the process
std::atomic<Ctx *> g_ctx{nullptr};
std::atomic<int> g_ngx_lines{0};
PFN_vkGetDeviceProcAddr g_fsr_next = nullptr;

void logf(Ctx &c, int level, const char *fmt, ...) {
    if (!c.log) return;
    char b[512];
    va_list a;
    va_start(a, fmt);
    std::vsnprintf(b, sizeof(b), fmt, a);
    va_end(a);
    c.log(c.log_user, level, b);
}

ub_result fail(Ctx &c, ub_result r, const std::string &why) {
    copy_text(c.error, sizeof(c.error), why);
    logf(c, UB_LOG_ERROR, "%s", why.c_str());
    return r;
}

template <class F> bool guarded(Ctx &c, int method, const char *step, F &&f) {
    using Fn = std::remove_reference_t<F>;
    CrashCode code = 0;
    if (guard_call([](void *p) { (*static_cast<Fn *>(p))(); }, const_cast<void *>(static_cast<const void *>(&f)), &code)) return true;
    c.dead[method] = true;
    std::string why = std::string(method == kDlss ? "NVIDIA's" : "AMD's") + " code crashed in " + step + " (" + hex(code) + "): it stays off until close_context";
    fail(c, UB_ERR_CRASH, why);
    (method == kDlss ? c.dlss.reason : c.fsr.reason) = why;
    return false;
}

// ---- NVIDIA ---------------------------------------------------------------------------------------------------------------------

#if defined(UB_WITH_DLSS)
void NVSDK_CONV ngx_log(const char *message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    Ctx *c = g_ctx.load();
    if (!c || !message || g_ngx_lines.fetch_add(1) >= 60) return; // (its start says much)
    std::string s(message);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    logf(*c, UB_LOG_INFO, "NGX: %s", s.c_str());
}

void ngx_common(Ctx &c) {
    DlssState &x = c.dlss;
    x.paths[0] = c.vendor.c_str();
    x.common = NVSDK_NGX_FeatureCommonInfo{};
    x.common.PathListInfo.Path = x.paths;
    x.common.PathListInfo.Length = 1;
    x.common.LoggingInfo.LoggingCallback = &ngx_log;
    x.common.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
    x.common.LoggingInfo.DisableOtherLoggingSinks = true;
}

NVSDK_NGX_FeatureDiscoveryInfo ngx_discovery(Ctx &c) {
    ngx_common(c);
    NVSDK_NGX_FeatureDiscoveryInfo di{};
    di.SDKVersion = NVSDK_NGX_Version_API;
    di.FeatureID = NVSDK_NGX_Feature_SuperSampling;
    di.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    di.Identifier.v.ProjectDesc.ProjectId = c.project_id.c_str();
    di.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
    di.Identifier.v.ProjectDesc.EngineVersion = c.engine_version.c_str();
    di.ApplicationDataPath = c.data.c_str();
    di.FeatureInfo = &c.dlss.common;
    return di;
}

std::string support_text(uint32_t f) {
    if (f & NVSDK_NGX_FeatureSupportResult_AdapterUnsupported) return "DLSS does not run on this card (it needs an RTX card)";
    if (f & NVSDK_NGX_FeatureSupportResult_DriverVersionUnsupported) return "the NVIDIA driver is too old for DLSS";
    if (f & NVSDK_NGX_FeatureSupportResult_OSVersionBelowMinimumSupported) return "this Windows version is too old for DLSS";
    return "NGX says DLSS is not supported here (" + hex(f) + ")";
}

// what NGX wants of the device: added to the merge. reason set when DLSS cannot run on this device
void dlss_prepare_device(Ctx &c, VkPhysicalDevice phys, const VkPhysicalDeviceProperties &props) {
    DlssState &x = c.dlss;
    x.asked = false;
    x.device_why.clear();
    if (!x.file) {
        x.device_why = std::string(kDlssFile) + " is not in the vendor folder";
        return;
    }
    if (c.dead[kDlss]) return;
    if (props.vendorID != kVendorNvidia) {
        x.device_why = "DLSS needs an NVIDIA RTX card";
        return;
    }
    if (x.asked_gpu != phys) {
        NVSDK_NGX_FeatureDiscoveryInfo di = ngx_discovery(c);
        NVSDK_NGX_FeatureRequirement req{};
        NVSDK_NGX_Result r = NVSDK_NGX_Result_Fail, rx = NVSDK_NGX_Result_Fail;
        std::vector<std::string> need;
        if (!guarded(c, kDlss, "its device query", [&] {
                r = NVSDK_NGX_VULKAN_GetFeatureRequirements(c.instance, phys, &di, &req);
                if (NVSDK_NGX_SUCCEED(r) && req.FeatureSupported == NVSDK_NGX_FeatureSupportResult_Supported) {
                    uint32_t n = 0;
                    VkExtensionProperties *ext = nullptr;
                    rx = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(c.instance, phys, &di, &n, &ext);
                    if (NVSDK_NGX_SUCCEED(rx) && ext)
                        for (uint32_t i = 0; i < n; ++i) need.push_back(ext[i].extensionName);
                }
            })) {
            x.device_why = c.dlss.reason;
            return;
        }
        if (NVSDK_NGX_FAILED(r)) {
            x.device_why = "NGX could not check this card (" + hex(uint32_t(r)) + ")";
            return;
        }
        if (req.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported) {
            x.device_why = support_text(uint32_t(req.FeatureSupported));
            return;
        }
        if (NVSDK_NGX_FAILED(rx)) {
            x.device_why = "NGX could not name its Vulkan extensions (" + hex(uint32_t(rx)) + ")";
            return;
        }
        x.need = need;
        x.asked_gpu = phys;
    }
    auto enum_ext = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(c.gipa(c.instance, "vkEnumerateDeviceExtensionProperties"));
    std::vector<VkExtensionProperties> have;
    if (enum_ext) {
        uint32_t k = 0;
        enum_ext(phys, nullptr, &k, nullptr);
        have.resize(k);
        if (k) enum_ext(phys, nullptr, &k, have.data());
        have.resize(k);
    }
    auto gpu_has = [&](const std::string &n) {
        for (const VkExtensionProperties &e : have)
            if (n == e.extensionName) return true;
        return false;
    };
    // a Vulkan 1.2+ device has buffer device addresses in its core; the EXT extension beside the core feature is not allowed
    const bool core12 = props.apiVersion >= VK_API_VERSION_1_2 && c.instance_api >= VK_API_VERSION_1_2;
    bool bda = false;
    std::vector<std::string> add;
    std::string added;
    for (const std::string &n : x.need) {
        if (n == "VK_KHR_buffer_device_address" || n == "VK_EXT_buffer_device_address") bda = true;
        if (core12 && n == "VK_EXT_buffer_device_address") continue;
        if (c.merge.has_ext(n.c_str())) continue;
        if (!gpu_has(n)) {
            x.device_why = "this card or driver lacks " + n;
            return;
        }
        add.push_back(n);
        added += (added.empty() ? "" : ", ") + n;
    }
    for (const std::string &n : add) c.merge.add_ext(n.c_str());
    if (bda) {
        const Spot s{"bufferDeviceAddress",
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                     offsetof(VkPhysicalDeviceVulkan12Features, bufferDeviceAddress),
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
                     offsetof(VkPhysicalDeviceBufferDeviceAddressFeatures, bufferDeviceAddress),
                     sizeof(VkPhysicalDeviceBufferDeviceAddressFeatures)};
        if (c.merge.turn_on(s)) added += std::string(added.empty() ? "" : ", ") + "bufferDeviceAddress";
    }
    x.asked = true;
    logf(c, UB_LOG_INFO, "DLSS: NGX's needs on the device: %s", added.empty() ? "the host's device has them" : added.c_str());
}

void dlss_attach(Ctx &c) {
    DlssState &x = c.dlss;
    if (c.dead[kDlss]) return; // (reason kept)
    if (!x.device_ok) {
        x.reason = x.device_why.empty() ? "prepare_device was not used for this device" : x.device_why;
        return;
    }
    ngx_common(c);
    g_ngx_lines.store(0);
    NVSDK_NGX_Result r = NVSDK_NGX_Result_Fail;
    int avail = 0, needs = 0, init = 0;
    unsigned int maj = 0, min = 0;
    if (!guarded(c, kDlss, "its start", [&] {
            r = NVSDK_NGX_VULKAN_Init_with_ProjectID(c.project_id.c_str(), NVSDK_NGX_ENGINE_TYPE_CUSTOM, c.engine_version.c_str(), c.data.c_str(), c.instance,
                                                     c.phys, c.device, c.gipa, c.gdpa, &x.common, NVSDK_NGX_Version_API);
            if (NVSDK_NGX_FAILED(r)) return;
            x.started_on = c.device;
            r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&x.params);
            if (NVSDK_NGX_FAILED(r) || !x.params) return;
            NVSDK_NGX_Parameter_GetI(x.params, NVSDK_NGX_Parameter_SuperSampling_Available, &avail);
            if (!avail) {
                NVSDK_NGX_Parameter_GetI(x.params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs);
                NVSDK_NGX_Parameter_GetUI(x.params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &maj);
                NVSDK_NGX_Parameter_GetUI(x.params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &min);
                NVSDK_NGX_Parameter_GetI(x.params, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &init);
            }
        }))
        return;
    if (NVSDK_NGX_FAILED(r) || !x.params) {
        x.reason = x.started_on ? "NGX gave no parameters (" + hex(uint32_t(r)) + ")" : "NVIDIA's NGX did not start (" + hex(uint32_t(r)) + ")";
    } else if (!avail) {
        x.reason = needs ? "the NVIDIA driver is too old for DLSS (it needs " + std::to_string(maj) + "." + std::to_string(min) + " or newer)"
                         : "DLSS is not available on this card or driver (" + hex(uint32_t(init)) + ")";
    } else {
        x.ready = true;
        x.reason.clear();
        return;
    }
    guarded(c, kDlss, "its end", [&] {
        if (x.params) NVSDK_NGX_VULKAN_DestroyParameters(x.params);
        x.params = nullptr;
        if (x.started_on) NVSDK_NGX_VULKAN_Shutdown1(x.started_on);
        x.started_on = VK_NULL_HANDLE;
    });
}

ub_result dlss_evaluate(Ctx &c, VkCommandBuffer cb, const ub_evaluate_desc &d) {
    DlssState &x = c.dlss;
    const uint32_t kept = d.flags & (UB_EVAL_LDR | UB_EVAL_DEPTH_INVERTED);
    const uint32_t preset = d.dlss_preset >= 10 && d.dlss_preset <= 13 ? d.dlss_preset : uint32_t(NVSDK_NGX_DLSS_Hint_Render_Preset_K);
    if (x.handle && (x.rw != d.render_width || x.rh != d.render_height || x.ow != d.output_width || x.oh != d.output_height || x.quality != d.quality ||
                     x.flags != kept || x.preset != preset)) {
        x.retired.push_back({x.handle, c.frame});
        x.handle = nullptr;
    }
    NVSDK_NGX_Result r = NVSDK_NGX_Result_Success;
    const VkImageSubresourceRange col{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkImageSubresourceRange dep{VkImageAspectFlags(ffxApiIsDepthFormat(d.depth.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
    const uint32_t rw = d.render_width, rh = d.render_height, ow = d.output_width, oh = d.output_height;
    bool made = false;
    const bool ok = guarded(c, kDlss, "its evaluation", [&] {
        if (!x.handle) {
            // the preset asked for (K, the transformer model, by default) for each quality; a file older than 310 has no such preset
            // and runs its own model
            if (x.major == 0 || x.major >= 310) {
                NVSDK_NGX_Parameter_SetUI(x.params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, preset);
                NVSDK_NGX_Parameter_SetUI(x.params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, preset);
                NVSDK_NGX_Parameter_SetUI(x.params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, preset);
                NVSDK_NGX_Parameter_SetUI(x.params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, preset);
            }
            static const NVSDK_NGX_PerfQuality_Value kQ[4] = {NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
                                                              NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf};
            NVSDK_NGX_DLSS_Create_Params cp{};
            cp.Feature.InWidth = rw;
            cp.Feature.InHeight = rh;
            cp.Feature.InTargetWidth = ow;
            cp.Feature.InTargetHeight = oh;
            cp.Feature.InPerfQualityValue = kQ[d.quality];
            int flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
            if (!(d.flags & UB_EVAL_LDR)) flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
            if (d.flags & UB_EVAL_DEPTH_INVERTED) flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
            cp.InFeatureCreateFlags = flags;
            cp.InEnableOutputSubrects = false;
            r = NGX_VULKAN_CREATE_DLSS_EXT1(c.device, cb, 1, 1, &x.handle, x.params, &cp);
            if (NVSDK_NGX_FAILED(r) || !x.handle) {
                x.handle = nullptr;
                return;
            }
            made = true;
        }
        NVSDK_NGX_Resource_VK rc = NVSDK_NGX_Create_ImageView_Resource_VK(d.color.view, d.color.image, col, d.color.format, rw, rh, false);
        NVSDK_NGX_Resource_VK rd = NVSDK_NGX_Create_ImageView_Resource_VK(d.depth.view, d.depth.image, dep, d.depth.format, rw, rh, false);
        NVSDK_NGX_Resource_VK rm = NVSDK_NGX_Create_ImageView_Resource_VK(d.motion.view, d.motion.image, col, d.motion.format, rw, rh, false);
        NVSDK_NGX_Resource_VK ro = NVSDK_NGX_Create_ImageView_Resource_VK(d.output.view, d.output.image, col, d.output.format, ow, oh, true);
        NVSDK_NGX_VK_DLSS_Eval_Params ep{};
        ep.Feature.pInColor = &rc;
        ep.Feature.pInOutput = &ro;
        ep.pInDepth = &rd;
        ep.pInMotionVectors = &rm;
        ep.InJitterOffsetX = d.jitter_x;
        ep.InJitterOffsetY = d.jitter_y;
        ep.InRenderSubrectDimensions.Width = rw;
        ep.InRenderSubrectDimensions.Height = rh;
        ep.InReset = (d.flags & UB_EVAL_RESET) ? 1 : 0;
        ep.InMVScaleX = d.motion_scale_x != 0.0f ? d.motion_scale_x : 1.0f;
        ep.InMVScaleY = d.motion_scale_y != 0.0f ? d.motion_scale_y : 1.0f;
        ep.InFrameTimeDeltaInMsec = d.frame_time_ms;
        r = NGX_VULKAN_EVALUATE_DLSS_EXT(cb, x.handle, x.params, &ep);
    });
    if (!ok) return UB_ERR_CRASH;
    if (made) {
        x.rw = rw;
        x.rh = rh;
        x.ow = ow;
        x.oh = oh;
        x.quality = d.quality;
        x.flags = kept;
        x.preset = preset;
        logf(c, UB_LOG_INFO, "DLSS: made at %ux%u -> %ux%u, %s, preset %c", rw, rh, ow, oh, (d.flags & UB_EVAL_LDR) ? "LDR" : "HDR with NGX's auto exposure",
             char('A' + preset - 1));
    }
    if (NVSDK_NGX_FAILED(r)) return fail(c, UB_ERR_VENDOR, "DLSS failed at " + std::to_string(rw) + "x" + std::to_string(rh) + " -> " + std::to_string(ow) + "x" + std::to_string(oh) + " (" + hex(uint32_t(r)) + ")");
    return UB_OK;
}

void dlss_retire(Ctx &c, bool all) {
    DlssState &x = c.dlss;
    const uint64_t wait = c.in_flight + 2;
    for (size_t i = 0; i < x.retired.size();) {
        if (all || c.frame > x.retired[i].second + wait) {
            NVSDK_NGX_Handle *h = x.retired[i].first;
            guarded(c, kDlss, "its release", [&] { NVSDK_NGX_VULKAN_ReleaseFeature(h); });
            x.retired.erase(x.retired.begin() + std::ptrdiff_t(i));
        } else {
            ++i;
        }
    }
}

void dlss_detach(Ctx &c) {
    DlssState &x = c.dlss;
    if (x.started_on && !c.dead[kDlss]) {
        dlss_retire(c, true);
        guarded(c, kDlss, "its end", [&] {
            if (x.handle) NVSDK_NGX_VULKAN_ReleaseFeature(x.handle);
            x.handle = nullptr;
            if (x.params) NVSDK_NGX_VULKAN_DestroyParameters(x.params);
            x.params = nullptr;
            NVSDK_NGX_VULKAN_Shutdown1(x.started_on);
        });
    }
    x.retired.clear();
    x.handle = nullptr;
    x.params = nullptr;
    x.started_on = VK_NULL_HANDLE;
    x.ready = false;
    x.device_ok = false;
}
#endif // UB_WITH_DLSS

// ---- AMD ------------------------------------------------------------------------------------------------------------------------

// AMD's Vulkan backend asks for vkGetBufferMemoryRequirements2KHR by its old name and calls it unchecked. A Vulkan 1.1+ device that did not
// name that extension gives NULL for the old name. A NULL for a name ending in KHR or EXT is asked again without the suffix: the same
// function, core since the Vulkan version that took the extension in.
PFN_vkVoidFunction VKAPI_CALL fsr_gdpa(VkDevice dev, const char *name) {
    const PFN_vkGetDeviceProcAddr next = g_fsr_next;
    if (!next || !name) return nullptr;
    if (const PFN_vkVoidFunction f = next(dev, name)) return f;
    const size_t n = std::strlen(name);
    for (const char *sfx : {"KHR", "EXT"}) {
        const size_t k = std::strlen(sfx);
        if (n <= k + 2 || std::strcmp(name + n - k, sfx) != 0) continue;
        const std::string core(name, n - k);
        const PFN_vkVoidFunction f = next(dev, core.c_str());
        if (f)
            if (Ctx *c = g_ctx.load()) c->fsr.core_names.push_back(std::string(name) + " as " + core);
        return f;
    }
    return nullptr;
}

void fsr_message(uint32_t type, const wchar_t *message) {
    Ctx *c = g_ctx.load();
    if (!c || !message) return;
    std::string s;
    for (const wchar_t *p = message; *p && s.size() < 400; ++p) s += (*p >= 32 && *p < 127) ? char(*p) : ' ';
    for (const std::string &t : c->fsr.told)
        if (t == s) return;
    if (c->fsr.told.size() < 32) c->fsr.told.push_back(s);
    logf(*c, type == FFX_API_MESSAGE_TYPE_ERROR ? UB_LOG_ERROR : UB_LOG_WARNING, "FSR: %s", s.c_str());
}

// FSR 4's machine-learning passes use device features a host seldom asks for. AMD's loader checks only what the GPU can do, never what the
// device turned on, so FSR 4 may run only on a device made with them; anywhere else the newest 3.x runs by AMD's version override.
void fsr_prepare_device(Ctx &c, VkPhysicalDevice phys, const VkPhysicalDeviceProperties &props) {
    FsrState &x = c.fsr;
    x.f4_asked = false;
    x.f4_why.clear();
    if (!x.files || c.dead[kFsr] || props.vendorID != kVendorAmd) return;
    if (props.apiVersion < VK_API_VERSION_1_3 || c.instance_api < VK_API_VERSION_1_3) {
        x.f4_why = "FSR 4 needs a Vulkan 1.3 instance and device";
        return;
    }
    auto feats2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(c.gipa(c.instance, "vkGetPhysicalDeviceFeatures2"));
    auto enum_ext = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(c.gipa(c.instance, "vkEnumerateDeviceExtensionProperties"));
    if (!feats2 || !enum_ext) return;
    std::vector<VkExtensionProperties> have;
    uint32_t k = 0;
    enum_ext(phys, nullptr, &k, nullptr);
    have.resize(k);
    if (k) enum_ext(phys, nullptr, &k, have.data());
    have.resize(k);
    auto gpu_has = [&](const char *n) {
        for (const VkExtensionProperties &e : have)
            if (std::strcmp(n, e.extensionName) == 0) return true;
        return false;
    };
    const bool has_cm = gpu_has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    const bool has_f8 = has_cm && gpu_has(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    const bool has_csd = gpu_has(VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features s11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features s12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features s13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR csd{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    f2.pNext = &s11;
    s11.pNext = &s12;
    s12.pNext = &s13;
    void **tail = &s13.pNext;
    if (has_cm) {
        *tail = &cm;
        tail = &cm.pNext;
    }
    if (has_f8) {
        *tail = &f8;
        tail = &f8.pNext;
    }
    if (has_csd) {
        *tail = &csd;
        tail = &csd.pNext;
    }
    feats2(phys, &f2);

    using S11 = VkPhysicalDeviceVulkan11Features;
    using S12 = VkPhysicalDeviceVulkan12Features;
    using S13 = VkPhysicalDeviceVulkan13Features;
    const VkStructureType c11 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, c12 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                          c13 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, none = VkStructureType(0);
    using T16 = VkPhysicalDevice16BitStorageFeatures;
    using TF16 = VkPhysicalDeviceShaderFloat16Int8Features;
    using T8 = VkPhysicalDevice8BitStorageFeatures;
    using TSG = VkPhysicalDeviceSubgroupSizeControlFeatures;
    using TDOT = VkPhysicalDeviceShaderIntegerDotProductFeatures;
    using TCM = VkPhysicalDeviceCooperativeMatrixFeaturesKHR;
    using TF8 = VkPhysicalDeviceShaderFloat8FeaturesEXT;
    using TCSD = VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR;
    struct Want {
        Spot spot;
        bool can;
        const char *ext; // (nullptr: core in Vulkan 1.3)
    };
    const Want want[] = {
        {{"storageBuffer16BitAccess", c11, offsetof(S11, storageBuffer16BitAccess), VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, offsetof(T16, storageBuffer16BitAccess), sizeof(T16)},
         s11.storageBuffer16BitAccess == VK_TRUE, nullptr},
        {{"shaderFloat16", c12, offsetof(S12, shaderFloat16), VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, offsetof(TF16, shaderFloat16), sizeof(TF16)},
         s12.shaderFloat16 == VK_TRUE, nullptr},
        {{"shaderInt8", c12, offsetof(S12, shaderInt8), VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, offsetof(TF16, shaderInt8), sizeof(TF16)},
         s12.shaderInt8 == VK_TRUE, nullptr},
        {{"storageBuffer8BitAccess", c12, offsetof(S12, storageBuffer8BitAccess), VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, offsetof(T8, storageBuffer8BitAccess), sizeof(T8)},
         s12.storageBuffer8BitAccess == VK_TRUE, nullptr},
        {{"subgroupSizeControl", c13, offsetof(S13, subgroupSizeControl), VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, offsetof(TSG, subgroupSizeControl), sizeof(TSG)},
         s13.subgroupSizeControl == VK_TRUE, nullptr},
        {{"computeFullSubgroups", c13, offsetof(S13, computeFullSubgroups), VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, offsetof(TSG, computeFullSubgroups), sizeof(TSG)},
         s13.computeFullSubgroups == VK_TRUE, nullptr},
        {{"shaderIntegerDotProduct", c13, offsetof(S13, shaderIntegerDotProduct), VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, offsetof(TDOT, shaderIntegerDotProduct), sizeof(TDOT)},
         s13.shaderIntegerDotProduct == VK_TRUE, nullptr},
        {{"cooperativeMatrix", none, 0, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, offsetof(TCM, cooperativeMatrix), sizeof(TCM)},
         has_cm && cm.cooperativeMatrix == VK_TRUE, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME},
        {{"shaderFloat8", none, 0, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, offsetof(TF8, shaderFloat8), sizeof(TF8)},
         has_f8 && f8.shaderFloat8 == VK_TRUE, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME},
        {{"shaderFloat8CooperativeMatrix", none, 0, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, offsetof(TF8, shaderFloat8CooperativeMatrix), sizeof(TF8)},
         has_f8 && cm.cooperativeMatrix == VK_TRUE && f8.shaderFloat8CooperativeMatrix == VK_TRUE, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME},
        {{"computeDerivativeGroupQuads", none, 0, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR, offsetof(TCSD, computeDerivativeGroupQuads), sizeof(TCSD)},
         has_csd && csd.computeDerivativeGroupQuads == VK_TRUE, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME},
        {{"computeDerivativeGroupLinear", none, 0, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR, offsetof(TCSD, computeDerivativeGroupLinear), sizeof(TCSD)},
         has_csd && csd.computeDerivativeGroupLinear == VK_TRUE, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME},
    };
    std::string added, lacks;
    auto note = [](std::string &s, const char *n) { s += (s.empty() ? "" : ", ") + std::string(n); };
    for (const Want &w : want) {
        if (!w.can) {
            note(lacks, w.spot.name);
            continue;
        }
        if (c.merge.turn_on(w.spot)) note(added, w.spot.name);
        if (w.ext) c.merge.add_ext(w.ext);
    }
    if (f2.features.shaderInt16 == VK_TRUE) {
        if (c.merge.turn_on_shader_int16()) note(added, "shaderInt16");
    } else {
        note(lacks, "shaderInt16");
    }
    x.f4_asked = true;
    logf(c, UB_LOG_INFO, "FSR 4's device features: %s%s%s", added.empty() ? "the host's device has them" : added.c_str(), lacks.empty() ? "" : "; not on this GPU: ", lacks.c_str());
}

void fsr_versions(Ctx &c) {
    FsrState &x = c.fsr;
    x.versions.clear();
    x.newest_is_4 = false;
    x.v3_id = 0;
    FfxVkDeviceContext dc{c.device, c.phys, &fsr_gdpa};
    ffxQueryDescGetVersions q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.device = &dc;
    uint64_t n = 0;
    q.outputCount = &n;
    ffxReturnCode_t rc = x.query(nullptr, &q.header);
    if (rc != FFX_API_RETURN_OK || n == 0 || n > 32) {
        logf(c, UB_LOG_WARNING, "FSR: AMD's loader names no versions for this GPU (%u, %llu)", unsigned(rc), static_cast<unsigned long long>(n));
        return;
    }
    std::vector<uint64_t> ids(size_t(n), 0);
    std::vector<const char *> names(size_t(n), nullptr);
    q.versionIds = ids.data();
    q.versionNames = names.data();
    rc = x.query(nullptr, &q.header);
    if (rc != FFX_API_RETURN_OK) return;
    const size_t k = std::min(size_t(n), ids.size());
    for (size_t i = 0; i < k; ++i) {
        std::string nm;
        for (const char *p = names[i]; p && *p && nm.size() < 24; ++p) nm += (*p >= 32 && *p < 127) ? *p : ' ';
        x.versions += (x.versions.empty() ? "" : ", ") + nm;
        const bool four = !nm.empty() && nm[0] == '4';
        if (i == 0) x.newest_is_4 = four;
        else if (!four && !x.v3_id) x.v3_id = ids[i];
    }
}

void fsr_attach(Ctx &c) {
    FsrState &x = c.fsr;
    if (c.dead[kFsr]) return; // (reason kept)
    if (!x.files) {
        x.reason = std::string(kFsrLoaderFile) + " and " + kFsrUpscalerFile + " are not in the vendor folder";
        return;
    }
    if (!x.load_tried) {
        x.load_tried = true;
        const std::wstring p = vendor_file(c.vendor, kFsrLoaderFile);
        // (by its full path: the loader finds the upscaler file in its own folder. It stays loaded: a driver thread may still run its code)
        x.dll = load_library(p);
        if (!x.dll) {
            x.reason = std::string(kFsrLoaderFile) + " did not load (" + load_error() + ")";
            return;
        }
        x.create = reinterpret_cast<PfnFfxCreateContext>(library_symbol(x.dll, "ffxCreateContext"));
        x.destroy = reinterpret_cast<PfnFfxDestroyContext>(library_symbol(x.dll, "ffxDestroyContext"));
        x.query = reinterpret_cast<PfnFfxQuery>(library_symbol(x.dll, "ffxQuery"));
        x.dispatch = reinterpret_cast<PfnFfxDispatch>(library_symbol(x.dll, "ffxDispatch"));
    }
    if (!x.create || !x.destroy || !x.query || !x.dispatch) {
        x.reason = std::string(kFsrLoaderFile) + " lacks the FSR API's functions";
        return;
    }
    x.reason.clear();
    g_fsr_next = c.gdpa;
    guarded(c, kFsr, "its version list", [&] { fsr_versions(c); });
    if (x.newest_is_4 && !x.f4_device_ok)
        x.reason = x.f4_why.empty() ? "the device was made without FSR 4's features: FSR 3 runs" : x.f4_why + ": FSR 3 runs";
}

uint64_t fsr_override(const FsrState &x, uint32_t flags) {
    if (!x.newest_is_4 || !x.v3_id) return 0;
    return !x.f4_device_ok || (flags & UB_EVAL_FSR3) ? x.v3_id : 0;
}

FfxApiResource fsr_resource(const ub_image &im, uint32_t w, uint32_t h, uint32_t usage, uint32_t state) {
    FfxApiResource r{};
    r.resource = reinterpret_cast<void *>(im.image);
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

ub_result fsr_evaluate(Ctx &c, VkCommandBuffer cb, const ub_evaluate_desc &d) {
    FsrState &x = c.fsr;
    const uint32_t kept = d.flags & (UB_EVAL_LDR | UB_EVAL_DEPTH_INVERTED);
    const uint64_t ov_id = fsr_override(x, d.flags);
    const uint32_t rw = d.render_width, rh = d.render_height, ow = d.output_width, oh = d.output_height;
    if (x.ctx && (x.rw != rw || x.rh != rh || x.ow != ow || x.oh != oh || x.flags != kept || x.ctx_override != ov_id)) {
        x.retired.push_back({x.ctx, c.frame});
        x.ctx = nullptr;
    }
    ffxReturnCode_t rc = FFX_API_RETURN_OK;
    bool made = false;
    const bool ok = guarded(c, kFsr, "its evaluation", [&] {
        if (!x.ctx) {
            ffxCreateBackendVKDesc be{};
            be.header.type = kFfxBackendVk;
            be.vkDevice = c.device;
            be.vkPhysicalDevice = c.phys;
            be.vkDeviceProcAddr = &fsr_gdpa;
            FfxUpscaleVersionDesc uv{};
            uv.header.type = kFfxUpscaleVersionDesc;
            uv.version = kFfxUpscalerVersion;
            be.header.pNext = &uv.header;
            ffxOverrideVersion ov{};
            if (ov_id) {
                ov.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
                ov.versionId = ov_id;
                uv.header.pNext = &ov.header;
            }
            ffxCreateContextDescUpscale cd{};
            cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
            cd.header.pNext = &be.header;
            cd.flags = (d.flags & UB_EVAL_LDR) ? 0u : uint32_t(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE);
            if (d.flags & UB_EVAL_DEPTH_INVERTED) cd.flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
            if (c.open_flags & UB_OPEN_DEBUG_CHECKS) cd.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
            cd.maxRenderSize = {rw, rh};
            cd.maxUpscaleSize = {ow, oh};
            cd.fpMessage = &fsr_message;
            ffxContext ctx = nullptr;
            x.told.clear();
            x.core_names.clear();
            rc = x.create(&ctx, &cd.header, nullptr);
            if (rc != FFX_API_RETURN_OK || !ctx) return;
            x.ctx = ctx;
            made = true;
            ffxQueryGetProviderVersion pv{};
            pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
            x.running.clear();
            if (x.query(&x.ctx, &pv.header) == FFX_API_RETURN_OK && pv.versionName)
                for (const char *p = pv.versionName; *p && x.running.size() < 24; ++p) x.running += (*p >= 32 && *p < 127) ? *p : ' ';
        }
        const uint32_t read = FFX_API_RESOURCE_STATE_COMPUTE_READ;
        ffxDispatchDescUpscale dd{};
        dd.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        dd.commandList = reinterpret_cast<void *>(cb);
        dd.color = fsr_resource(d.color, rw, rh, FFX_API_RESOURCE_USAGE_READ_ONLY, read);
        uint32_t depth_usage = FFX_API_RESOURCE_USAGE_READ_ONLY;
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
        dd.enableSharpening = false; // (FSR's own sharpening stays off: the host's picture stage decides)
        dd.sharpness = 0.0f;
        dd.frameTimeDelta = d.frame_time_ms;
        dd.preExposure = 1.0f;
        dd.reset = (d.flags & UB_EVAL_RESET) != 0;
        dd.cameraNear = d.camera_near;
        dd.cameraFar = d.camera_far;
        dd.cameraFovAngleVertical = d.camera_fov_y;
        dd.viewSpaceToMetersFactor = 1.0f;
        dd.flags = 0;
        rc = x.dispatch(&x.ctx, &dd.header);
    });
    if (!ok) return UB_ERR_CRASH;
    if (made) {
        x.rw = rw;
        x.rh = rh;
        x.ow = ow;
        x.oh = oh;
        x.flags = kept;
        x.ctx_override = ov_id;
        logf(c, UB_LOG_INFO, "FSR: made %s at %ux%u -> %ux%u, %s%s", x.running.empty() ? "?" : x.running.c_str(), rw, rh, ow, oh,
             (d.flags & UB_EVAL_LDR) ? "LDR" : "HDR with FSR's auto exposure", ov_id ? ", AMD's version override" : "");
        for (const std::string &s : x.core_names) logf(c, UB_LOG_INFO, "FSR: its Vulkan function %s (the old name gives NULL on this device)", s.c_str());
    }
    if (rc != FFX_API_RETURN_OK)
        return fail(c, UB_ERR_VENDOR, std::string(x.ctx ? "FSR's dispatch failed" : "FSR could not be made") + " at " + std::to_string(rw) + "x" + std::to_string(rh) + " -> " +
                                          std::to_string(ow) + "x" + std::to_string(oh) + " (FSR API error " + std::to_string(rc) + ")");
    return UB_OK;
}

void fsr_retire(Ctx &c, bool all) {
    FsrState &x = c.fsr;
    const uint64_t wait = c.in_flight + 2;
    for (size_t i = 0; i < x.retired.size();) {
        if (all || c.frame > x.retired[i].second + wait) {
            ffxContext h = x.retired[i].first;
            guarded(c, kFsr, "its release", [&] { x.destroy(&h, nullptr); });
            x.retired.erase(x.retired.begin() + std::ptrdiff_t(i));
        } else {
            ++i;
        }
    }
}

void fsr_detach(Ctx &c) {
    FsrState &x = c.fsr;
    if (!c.dead[kFsr]) {
        fsr_retire(c, true);
        if (x.ctx) guarded(c, kFsr, "its end", [&] { x.destroy(&x.ctx, nullptr); });
    }
    x.retired.clear();
    x.ctx = nullptr;
    x.running.clear();
    x.f4_device_ok = false;
    x.newest_is_4 = false;
    x.v3_id = 0;
    x.versions.clear();
}

// ---- layouts --------------------------------------------------------------------------------------------------------------------

VkImageAspectFlags barrier_aspect(VkFormat f) {
    if (!ffxApiIsDepthFormat(f)) return VK_IMAGE_ASPECT_COLOR_BIT;
    return ffxApiIsStencilFormat(f) ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT);
}

VkImageMemoryBarrier barrier_of(const ub_image &im, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.image;
    b.subresourceRange = {barrier_aspect(im.format), 0, 1, 0, 1};
    return b;
}

// what NGX and FSR want: the inputs read-only, the output GENERAL
void to_vendor(Ctx &c, VkCommandBuffer cb, const ub_evaluate_desc &d) {
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
void from_vendor(Ctx &c, VkCommandBuffer cb, const ub_evaluate_desc &d) {
    const VkImageLayout ro = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkAccessFlags all = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    const VkImageMemoryBarrier b[4] = {
        barrier_of(d.color, ro, d.color.layout, VK_ACCESS_SHADER_READ_BIT, all),
        barrier_of(d.depth, ro, d.depth.layout, VK_ACCESS_SHADER_READ_BIT, all),
        barrier_of(d.motion, ro, d.motion.layout, VK_ACCESS_SHADER_READ_BIT, all),
        barrier_of(d.output, VK_IMAGE_LAYOUT_GENERAL, d.output.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : d.output.layout, VK_ACCESS_MEMORY_WRITE_BIT, all),
    };
    c.cmd_barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 4, b);
}

// ---- the API --------------------------------------------------------------------------------------------------------------------

Ctx &ctx_of(ub_context *c) { return *reinterpret_cast<Ctx *>(c); }

ub_result UB_CALL open_context(const ub_open_desc *desc, ub_context **out) {
    std::lock_guard<std::mutex> lock(g_open_mx);
    g_open_error[0] = 0;
    auto refuse = [](ub_result r, const char *why) {
        copy_text(g_open_error, sizeof(g_open_error), why);
        return r;
    };
    if (out) *out = nullptr;
    if (!size_ok(desc) || !out) return refuse(UB_ERR_ARGUMENT, "open_context: a NULL pointer or an old struct_size");
    if (g_ctx.load()) return refuse(UB_ERR_STATE, "open_context: one context a process, and one is open");
    if (other_vulkan_dll()) return refuse(UB_ERR_BLOCKED, "a vulkan-1.dll that is not Khronos's loader sits beside the exe: it keeps its own upscaler, the bridge stays off");
    std::unique_ptr<Ctx> c(new (std::nothrow) Ctx());
    if (!c) return refuse(UB_ERR_MEMORY, "open_context: no memory");
    c->open_flags = desc->flags;
    c->log = desc->log;
    c->log_user = desc->log_user;
    c->project_id = desc->project_id && *desc->project_id ? desc->project_id : kDefaultProject;
    c->engine_version = desc->engine_version && *desc->engine_version ? desc->engine_version : "1.0";
    if (desc->vendor_folder && *desc->vendor_folder) {
        c->vendor = widen(desc->vendor_folder);
    } else {
        c->vendor = self_folder();
    }
    c->vendor = full_path(c->vendor); // (a dll by a relative path is not loaded reliably)
    while (!c->vendor.empty() && (c->vendor.back() == L'\\' || c->vendor.back() == L'/')) c->vendor.pop_back();
    if (desc->data_folder && *desc->data_folder) {
        c->data = widen(desc->data_folder);
    } else {
        c->data = temp_folder();
    }
    const std::wstring dlss_path = vendor_file(c->vendor, kDlssFile);
    c->dlss.file = !dlss_path.empty() && file_there(dlss_path);
    if (c->dlss.file) {
        c->dlss.version = file_version(dlss_path);
        c->dlss.major = std::atoi(c->dlss.version.c_str());
    }
#if !defined(UB_WITH_DLSS)
    c->dlss.file = false; // (a build without NVIDIA's library)
#endif
    c->fsr.files = file_there(vendor_file(c->vendor, kFsrLoaderFile)) && file_there(vendor_file(c->vendor, kFsrUpscalerFile));
    Ctx *p = c.release();
    g_ctx.store(p);
    *out = reinterpret_cast<ub_context *>(p);
    return UB_OK;
}

void UB_CALL close_context(ub_context *cx) {
    if (!cx) return;
    std::lock_guard<std::mutex> open_lock(g_open_mx);
    Ctx *c = reinterpret_cast<Ctx *>(cx);
    {
        std::lock_guard<std::mutex> lock(c->mx);
#if defined(UB_WITH_DLSS)
        if (c->device) dlss_detach(*c);
#endif
        if (c->device) fsr_detach(*c);
        c->device = VK_NULL_HANDLE;
        c->merge.restore();
    }
    g_ctx.store(nullptr);
    g_fsr_next = nullptr;
    delete c;
}

const char *UB_CALL last_error(ub_context *cx) { return cx ? ctx_of(cx).error : g_open_error; }

ub_result UB_CALL prepare_instance(ub_context *cx, PFN_vkGetInstanceProcAddr gipa, const VkInstanceCreateInfo *in, const VkInstanceCreateInfo **out) {
    if (out) *out = in;
    if (!cx || !in || !out || !gipa) return UB_ERR_ARGUMENT;
    Ctx &c = ctx_of(cx);
    std::lock_guard<std::mutex> lock(c.mx);
    c.gipa = gipa;
    c.instance_api = in->pApplicationInfo && in->pApplicationInfo->apiVersion ? in->pApplicationInfo->apiVersion : VK_API_VERSION_1_0;
    c.inst_asked = false;
#if defined(UB_WITH_DLSS)
    if (!c.dlss.file || c.dead[kDlss]) return UB_OK;
    NVSDK_NGX_FeatureDiscoveryInfo di = ngx_discovery(c);
    NVSDK_NGX_Result r = NVSDK_NGX_Result_Fail;
    std::vector<std::string> need;
    if (!guarded(c, kDlss, "its instance query", [&] {
            uint32_t n = 0;
            VkExtensionProperties *props = nullptr;
            r = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&di, &n, &props);
            if (NVSDK_NGX_SUCCEED(r) && props)
                for (uint32_t i = 0; i < n; ++i) need.push_back(props[i].extensionName);
        }))
        return UB_ERR_CRASH;
    if (NVSDK_NGX_FAILED(r)) {
        logf(c, UB_LOG_WARNING, "DLSS: NGX could not name its instance extensions (%s)", hex(uint32_t(r)).c_str());
        return UB_OK;
    }
    std::vector<VkExtensionProperties> have;
    if (auto enum_ext = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(gipa(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"))) {
        uint32_t k = 0;
        enum_ext(nullptr, &k, nullptr);
        have.resize(k);
        if (k) enum_ext(nullptr, &k, have.data());
        have.resize(k);
    }
    c.inst_names.clear();
    std::string added;
    for (const std::string &n : need) {
        bool named = false;
        for (uint32_t i = 0; i < in->enabledExtensionCount && in->ppEnabledExtensionNames; ++i)
            if (in->ppEnabledExtensionNames[i] && n == in->ppEnabledExtensionNames[i]) named = true;
        if (named) continue;
        bool there = false;
        for (const VkExtensionProperties &e : have)
            if (n == e.extensionName) there = true;
        if (!there) {
            logf(c, UB_LOG_WARNING, "DLSS: the Vulkan loader lacks %s", n.c_str());
            continue;
        }
        c.inst_names.push_back(n);
        added += (added.empty() ? "" : ", ") + n;
    }
    if (c.inst_names.empty()) return UB_OK;
    c.ici = *in;
    c.inst_exts.clear();
    for (uint32_t i = 0; i < in->enabledExtensionCount && in->ppEnabledExtensionNames; ++i) c.inst_exts.push_back(in->ppEnabledExtensionNames[i]);
    for (const std::string &s : c.inst_names) c.inst_exts.push_back(s.c_str());
    c.ici.enabledExtensionCount = uint32_t(c.inst_exts.size());
    c.ici.ppEnabledExtensionNames = c.inst_exts.data();
    c.inst_asked = true;
    *out = &c.ici;
    logf(c, UB_LOG_INFO, "DLSS: NGX's instance extensions added: %s", added.c_str());
#endif
    return UB_OK;
}

void UB_CALL instance_done(ub_context *cx, VkResult result) {
    if (!cx) return;
    Ctx &c = ctx_of(cx);
    std::lock_guard<std::mutex> lock(c.mx);
    if (c.inst_asked && result != VK_SUCCESS) logf(c, UB_LOG_WARNING, "DLSS: the instance could not take NGX's extensions (VkResult %d)", int(result));
    c.inst_asked = false;
}

ub_result UB_CALL prepare_device(ub_context *cx, VkInstance instance, VkPhysicalDevice phys, const VkDeviceCreateInfo *in, const VkDeviceCreateInfo **out) {
    if (out) *out = in;
    if (!cx || !in || !out || !instance || !phys) return UB_ERR_ARGUMENT;
    Ctx &c = ctx_of(cx);
    std::lock_guard<std::mutex> lock(c.mx);
    if (!c.gipa) return fail(c, UB_ERR_STATE, "prepare_device: prepare_instance was not used");
    if (c.merge_open) return fail(c, UB_ERR_STATE, "prepare_device: device_done of the last one is missing");
    c.instance = instance;
    auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(c.gipa(instance, "vkGetPhysicalDeviceProperties"));
    if (!get_props) return fail(c, UB_ERR_STATE, "prepare_device: no vkGetPhysicalDeviceProperties");
    VkPhysicalDeviceProperties props{};
    get_props(phys, &props);
    c.merge.start(in);
    c.merge_open = true;
#if defined(UB_WITH_DLSS)
    dlss_prepare_device(c, phys, props);
#else
    c.dlss.device_why = "this bridge was built without NVIDIA's library";
#endif
    fsr_prepare_device(c, phys, props);
    *out = c.merge.finish();
    return UB_OK;
}

void UB_CALL device_done(ub_context *cx, VkResult result) {
    if (!cx) return;
    Ctx &c = ctx_of(cx);
    std::lock_guard<std::mutex> lock(c.mx);
    if (!c.merge_open) return;
    c.merge_open = false;
    c.merge.restore();
    if (c.dlss.asked) {
        c.dlss.device_ok = result == VK_SUCCESS;
        if (result != VK_SUCCESS) {
            c.dlss.device_why = "the device could not be made with NGX's extensions (VkResult " + std::to_string(int(result)) + ")";
            logf(c, UB_LOG_WARNING, "%s", c.dlss.device_why.c_str());
        }
    }
    if (c.fsr.f4_asked) {
        c.fsr.f4_device_ok = result == VK_SUCCESS;
        if (result != VK_SUCCESS) {
            c.fsr.f4_why = "the device could not be made with FSR 4's features (VkResult " + std::to_string(int(result)) + ")";
            logf(c, UB_LOG_WARNING, "%s", c.fsr.f4_why.c_str());
        }
    }
    c.dlss.asked = c.fsr.f4_asked = false;
}

ub_result UB_CALL attach_device(ub_context *cx, const ub_device_desc *d) {
    if (!cx || !size_ok(d) || !d->instance || !d->physical_device || !d->device || !d->get_instance_proc_addr || !d->get_device_proc_addr)
        return cx ? fail(ctx_of(cx), UB_ERR_ARGUMENT, "attach_device: a NULL handle or function, or an old struct_size") : UB_ERR_ARGUMENT;
    Ctx &c = ctx_of(cx);
    std::lock_guard<std::mutex> lock(c.mx);
    if (c.device) return fail(c, UB_ERR_STATE, "attach_device: a device is attached");
    c.instance = d->instance;
    c.phys = d->physical_device;
    c.device = d->device;
    c.gipa = d->get_instance_proc_addr;
    c.gdpa = d->get_device_proc_addr;
    c.in_flight = d->frames_in_flight ? d->frames_in_flight : 8;
    c.frame = 0;
    c.cmd_barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(c.gdpa(c.device, "vkCmdPipelineBarrier"));
    if (!c.cmd_barrier) {
        c.device = VK_NULL_HANDLE;
        return fail(c, UB_ERR_STATE, "attach_device: no vkCmdPipelineBarrier from the get_device_proc_addr given");
    }
#if defined(UB_WITH_DLSS)
    dlss_attach(c);
#else
    c.dlss.reason = c.dlss.device_why.empty() ? "this bridge was built without NVIDIA's library" : c.dlss.device_why;
#endif
    fsr_attach(c);
    logf(c, UB_LOG_INFO, "attached: DLSS %s%s; FSR %s%s", c.dlss.ready ? c.dlss.version.c_str() : "off: ", c.dlss.ready ? "" : c.dlss.reason.c_str(),
         c.fsr.dll && c.fsr.create ? c.fsr.versions.c_str() : "off: ", c.fsr.dll && c.fsr.create ? "" : c.fsr.reason.c_str());
    return UB_OK;
}

void UB_CALL detach_device(ub_context *cx) {
    if (!cx) return;
    Ctx &c = ctx_of(cx);
    std::lock_guard<std::mutex> lock(c.mx);
    if (!c.device) return;
#if defined(UB_WITH_DLSS)
    dlss_detach(c);
#endif
    fsr_detach(c);
    c.device = VK_NULL_HANDLE;
    c.phys = VK_NULL_HANDLE;
    c.gdpa = nullptr;
}

ub_result UB_CALL get_caps(ub_context *cx, ub_caps *out) {
    if (!cx || !size_ok(out)) return UB_ERR_ARGUMENT;
    Ctx &c = ctx_of(cx);
    std::lock_guard<std::mutex> lock(c.mx);
    const uint32_t size = out->struct_size;
    std::memset(out, 0, sizeof(*out));
    out->struct_size = size;
    const FsrState &f = c.fsr;
    const bool fsr_ready = f.dll && f.create && !c.dead[kFsr] && c.device;
    out->dlss_available = c.dlss.ready && !c.dead[kDlss] ? 1u : 0u;
    out->fsr_available = fsr_ready ? 1u : 0u;
    out->fsr4_available = fsr_ready && f.newest_is_4 && (f.f4_device_ok || !f.v3_id) ? 1u : 0u;
    out->fsr3_choice = out->fsr4_available && f.v3_id ? 1u : 0u;
    copy_text(out->bridge_version, sizeof(out->bridge_version), UB_VERSION_TEXT);
    copy_text(out->dlss_version, sizeof(out->dlss_version), c.dlss.version);
    copy_text(out->fsr_versions, sizeof(out->fsr_versions), f.versions);
    copy_text(out->fsr_running, sizeof(out->fsr_running), f.ctx ? f.running : std::string());
    std::string dr = c.dlss.reason;
    if (dr.empty() && !c.dlss.ready) dr = c.device ? c.dlss.device_why : "no device attached yet";
    copy_text(out->dlss_reason, sizeof(out->dlss_reason), dr);
    std::string fr = f.reason;
    if (fr.empty() && !fsr_ready) fr = "no device attached yet";
    copy_text(out->fsr_reason, sizeof(out->fsr_reason), fr);
    return UB_OK;
}

ub_result UB_CALL get_render_info(ub_context *cx, uint32_t method, uint32_t quality, uint32_t ow, uint32_t oh, ub_render_info *out) {
    if (!cx || !size_ok(out) || !ow || !oh || (method != UB_METHOD_DLSS && method != UB_METHOD_FSR) || quality > UB_QUALITY_PERFORMANCE) return UB_ERR_ARGUMENT;
    Ctx &c = ctx_of(cx);
    std::lock_guard<std::mutex> lock(c.mx);
    static const float kFsrRatio[4] = {1.0f, 1.5f, 1.7f, 2.0f};
    static const float kDlssRatio[4] = {1.0f, 1.5f, 1.0f / 0.58f, 2.0f};
    const float ratio = (method == UB_METHOD_DLSS ? kDlssRatio : kFsrRatio)[quality];
    uint32_t w = std::max(1u, uint32_t(float(ow) / ratio)), h = std::max(1u, uint32_t(float(oh) / ratio));
#if defined(UB_WITH_DLSS)
    if (method == UB_METHOD_DLSS && quality != UB_QUALITY_NATIVE && c.dlss.ready && c.dlss.params && !c.dead[kDlss]) {
        static const NVSDK_NGX_PerfQuality_Value kQ[4] = {NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
                                                          NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf};
        unsigned int nw = 0, nh = 0;
        NVSDK_NGX_Result r = NVSDK_NGX_Result_Fail;
        NVSDK_NGX_Parameter *p = c.dlss.params;
        // (the steps of the SDK's NGX_DLSS_GET_OPTIMAL_SETTINGS: its header also wants the D3D helpers)
        if (!guarded(c, kDlss, "its render size", [&] {
                void *callback = nullptr;
                NVSDK_NGX_Parameter_GetVoidPointer(p, NVSDK_NGX_Parameter_DLSSOptimalSettingsCallback, &callback);
                if (!callback) return;
                NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_Width, ow);
                NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_Height, oh);
                NVSDK_NGX_Parameter_SetI(p, NVSDK_NGX_Parameter_PerfQualityValue, kQ[quality]);
                NVSDK_NGX_Parameter_SetI(p, NVSDK_NGX_Parameter_RTXValue, 0);
                r = reinterpret_cast<PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback>(callback)(p);
                if (NVSDK_NGX_SUCCEED(r)) {
                    NVSDK_NGX_Parameter_GetUI(p, NVSDK_NGX_Parameter_OutWidth, &nw);
                    NVSDK_NGX_Parameter_GetUI(p, NVSDK_NGX_Parameter_OutHeight, &nh);
                }
            }))
            return UB_ERR_CRASH;
        if (NVSDK_NGX_SUCCEED(r) && nw && nh) {
            w = nw;
            h = nh;
        }
    }
#endif
    out->width = w;
    out->height = h;
    const float scale = float(ow) / float(w);
    out->jitter_phases = std::max(8u, uint32_t(std::ceil(8.0f * scale * scale)));
    // NVIDIA's guide and AMD's docs give the same formula
    out->mip_bias = std::log2(float(w) / float(ow)) - 1.0f;
    return UB_OK;
}

float halton(uint32_t i, uint32_t base) {
    float f = 1.0f, r = 0.0f;
    while (i > 0) {
        f /= float(base);
        r += f * float(i % base);
        i /= base;
    }
    return r;
}

void UB_CALL get_jitter(uint32_t frame, uint32_t phases, float *x, float *y) {
    const uint32_t i = (phases ? frame % phases : frame) + 1u;
    if (x) *x = halton(i, 2) - 0.5f;
    if (y) *y = halton(i, 3) - 0.5f;
}

ub_result UB_CALL evaluate(ub_context *cx, VkCommandBuffer cb, const ub_evaluate_desc *d) {
    if (!cx) return UB_ERR_ARGUMENT;
    Ctx &c = ctx_of(cx);
    std::lock_guard<std::mutex> lock(c.mx);
    // A 1.0.0 caller's struct ends before dlss_preset: read as far as it goes, the rest 0.
    if (!d || d->struct_size < offsetof(ub_evaluate_desc, dlss_preset)) return fail(c, UB_ERR_ARGUMENT, "evaluate: a NULL pointer or an old struct_size");
    ub_evaluate_desc full{};
    std::memcpy(&full, d, std::min<size_t>(d->struct_size, sizeof(full)));
    d = &full;
    if (!cb) return fail(c, UB_ERR_ARGUMENT, "evaluate: no command buffer");
    if (!c.device) return fail(c, UB_ERR_STATE, "evaluate: no device attached");
    if (d->method != UB_METHOD_DLSS && d->method != UB_METHOD_FSR) return fail(c, UB_ERR_ARGUMENT, "evaluate: an unknown method");
    if (d->quality > UB_QUALITY_PERFORMANCE) return fail(c, UB_ERR_ARGUMENT, "evaluate: an unknown quality");
    if (!d->render_width || !d->render_height || !d->output_width || !d->output_height) return fail(c, UB_ERR_ARGUMENT, "evaluate: a size of 0");
    if (d->render_width > d->output_width || d->render_height > d->output_height) return fail(c, UB_ERR_ARGUMENT, "evaluate: the render size is above the output size");
    const bool same_size = d->render_width == d->output_width && d->render_height == d->output_height;
    if (d->quality == UB_QUALITY_NATIVE && !same_size) return fail(c, UB_ERR_ARGUMENT, "evaluate: native quality needs the render size = the output size");
    if (!d->color.image || !d->depth.image || !d->motion.image || !d->output.image) return fail(c, UB_ERR_ARGUMENT, "evaluate: an image is missing");
    if (d->color.layout == VK_IMAGE_LAYOUT_UNDEFINED || d->depth.layout == VK_IMAGE_LAYOUT_UNDEFINED || d->motion.layout == VK_IMAGE_LAYOUT_UNDEFINED)
        return fail(c, UB_ERR_ARGUMENT, "evaluate: an input image is given in layout UNDEFINED");
    const int m = d->method == UB_METHOD_DLSS ? kDlss : kFsr;
    if (c.dead[m]) return fail(c, UB_ERR_CRASH, m == kDlss ? c.dlss.reason : c.fsr.reason);
    if (m == kDlss) {
        if (!c.dlss.ready) return fail(c, UB_ERR_UNAVAILABLE, c.dlss.reason.empty() ? "DLSS is not available" : c.dlss.reason);
        if (!d->color.view || !d->depth.view || !d->motion.view || !d->output.view) return fail(c, UB_ERR_ARGUMENT, "evaluate: DLSS needs an image view of each image");
    } else {
        if (!c.fsr.dll || !c.fsr.create) return fail(c, UB_ERR_UNAVAILABLE, c.fsr.reason.empty() ? "FSR is not available" : c.fsr.reason);
        if (ffxApiGetSurfaceFormatVK(d->color.format) == FFX_API_SURFACE_FORMAT_UNKNOWN || ffxApiGetSurfaceFormatVK(d->depth.format) == FFX_API_SURFACE_FORMAT_UNKNOWN ||
            ffxApiGetSurfaceFormatVK(d->motion.format) == FFX_API_SURFACE_FORMAT_UNKNOWN || ffxApiGetSurfaceFormatVK(d->output.format) == FFX_API_SURFACE_FORMAT_UNKNOWN)
            return fail(c, UB_ERR_ARGUMENT, "evaluate: a format FSR does not know");
    }
    ++c.frame;
    // the other method's context goes: only one runs
#if defined(UB_WITH_DLSS)
    if (m == kFsr && c.dlss.handle) {
        c.dlss.retired.push_back({c.dlss.handle, c.frame});
        c.dlss.handle = nullptr;
    }
#endif
    if (m == kDlss && c.fsr.ctx) {
        c.fsr.retired.push_back({c.fsr.ctx, c.frame});
        c.fsr.ctx = nullptr;
    }
    ub_evaluate_desc e = *d;
    if (same_size) e.quality = UB_QUALITY_NATIVE; // (the render size decides: no upscale is native AA, whatever quality was named)
    to_vendor(c, cb, e);
#if defined(UB_WITH_DLSS)
    const ub_result r = m == kDlss ? dlss_evaluate(c, cb, e) : fsr_evaluate(c, cb, e);
    dlss_retire(c, false);
#else
    const ub_result r = fsr_evaluate(c, cb, e);
#endif
    fsr_retire(c, false);
    from_vendor(c, cb, e); // (on an error too: the host's layouts stay true)
    return r;
}

} // namespace

#if defined(_WIN32)
#define UB_EXPORT __declspec(dllexport)
#else
#define UB_EXPORT __attribute__((visibility("default")))
#endif
extern "C" UB_EXPORT const ub_api *UB_CALL ub_get_api(uint32_t api_version) {
    static const ub_api table = {sizeof(ub_api),  UB_API_VERSION, &open_context,   &close_context, &last_error,      &prepare_instance, &instance_done, &prepare_device,
                                 &device_done,    &attach_device, &detach_device,  &get_caps,      &get_render_info, &get_jitter,       &evaluate};
    return api_version == UB_API_VERSION ? &table : nullptr;
}
