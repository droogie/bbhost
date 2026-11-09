/*
 * bridge_core.cpp -- see bridge_core.h.
 *
 * Copyright (c) 2026 the bbhost authors. MIT licence: see LICENSE in this folder.
 */
#include "bridge_core.h"

#include "upscaler_bridge.h"

#include <dirent.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace ngx_bridge {

double plain_ratio(std::uint32_t method, std::uint32_t quality) {
    if (method != UB_METHOD_DLSS && method != UB_METHOD_FSR) return 0.0;
    switch (quality) {
        case UB_QUALITY_NATIVE: return 1.0;
        case UB_QUALITY_QUALITY: return 1.5;
        case UB_QUALITY_BALANCED: return method == UB_METHOD_DLSS ? 1.0 / 0.58 : 1.7;
        case UB_QUALITY_PERFORMANCE: return 2.0;
        default: return 0.0;
    }
}

std::uint32_t render_extent(std::uint32_t method, std::uint32_t output, double ratio) {
    if (!output || ratio <= 0.0) return 0;
    const double r = static_cast<double>(output) / ratio;
    const auto v = static_cast<std::uint32_t>(method == UB_METHOD_DLSS ? r + 0.5 : r + 1e-9);
    return v ? v : 1;
}

std::uint32_t jitter_phase_count(std::uint32_t render_w, std::uint32_t output_w) {
    if (!render_w || !output_w) return 8;
    const double s = static_cast<double>(output_w) / render_w;
    const auto n = static_cast<std::uint32_t>(std::ceil(8.0 * s * s - 1e-6));
    return n < 8 ? 8 : n;
}

float mip_bias(std::uint32_t render_w, std::uint32_t output_w) {
    if (!render_w || !output_w) return 0.0f;
    return static_cast<float>(std::log2(static_cast<double>(render_w) / output_w) - 1.0);
}

float halton(std::uint32_t index, std::uint32_t base) {
    float f = 1.0f, r = 0.0f;
    for (; index; index /= base) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
    }
    return r;
}

void jitter(std::uint32_t frame, std::uint32_t phases, float* x, float* y) {
    if (!phases) {
        *x = *y = 0.0f;
        return;
    }
    const std::uint32_t i = frame % phases + 1;
    *x = halton(i, 2) - 0.5f;
    *y = halton(i, 3) - 0.5f;
}

int compare_versions(const std::string& a, const std::string& b) {
    const char* p = a.c_str();
    const char* q = b.c_str();
    while (*p || *q) {
        char* pe = nullptr;
        char* qe = nullptr;
        const unsigned long x = *p ? std::strtoul(p, &pe, 10) : 0;
        const unsigned long y = *q ? std::strtoul(q, &qe, 10) : 0;
        if (x != y) return x < y ? -1 : 1;
        p = *p ? pe : p;
        q = *q ? qe : q;
        if (*p == '.') ++p;
        if (*q == '.') ++q;
    }
    return 0;
}

bool find_runtime(const std::string& dir, std::string* name, std::string* version) {
    static const char kPrefix[] = "libnvidia-ngx-dlss.so.";
    DIR* d = opendir(dir.empty() ? "." : dir.c_str());
    if (!d) return false;
    std::string best_name, best_version;
    while (const dirent* e = readdir(d)) {
        const char* n = e->d_name;
        if (std::strncmp(n, kPrefix, sizeof(kPrefix) - 1) != 0) continue;
        const std::string v = n + sizeof(kPrefix) - 1;
        // Digits and dots alone, starting with a digit: "310.9.16".
        if (v.empty() || v.find_first_not_of("0123456789.") != std::string::npos || v[0] == '.') continue;
        if (best_name.empty() || compare_versions(v, best_version) > 0) {
            best_name = n;
            best_version = v;
        }
    }
    closedir(d);
    if (best_name.empty()) return false;
    *name = best_name;
    *version = best_version;
    return true;
}

std::wstring widen(const std::string& s) {
    std::wstring w;
    w.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        std::uint32_t cp = 0xfffd;
        std::size_t len = 1;
        if (c < 0x80) {
            cp = c;
        } else if ((c >> 5) == 0x6 || (c >> 4) == 0xe || (c >> 3) == 0x1e) {
            len = (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : 4;
            cp = c & (len == 2 ? 0x1f : len == 3 ? 0x0f : 0x07);
            bool ok = i + len <= s.size();
            for (std::size_t k = 1; ok && k < len; ++k) {
                const auto t = static_cast<unsigned char>(s[i + k]);
                ok = (t >> 6) == 0x2;
                cp = (cp << 6) | (t & 0x3f);
            }
            if (!ok) {
                cp = 0xfffd;
                len = 1;
            }
        }
        w.push_back(static_cast<wchar_t>(cp));
        i += len;
    }
    return w;
}

std::string narrow(const wchar_t* w) {
    std::string s;
    if (!w) return s;
    for (; *w; ++w) {
        const auto cp = static_cast<std::uint32_t>(*w);
        if (cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            s.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else if (cp < 0x10000) {
            s.push_back(static_cast<char>(0xe0 | (cp >> 12)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else if (cp < 0x110000) {
            s.push_back(static_cast<char>(0xf0 | (cp >> 18)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
            s += "\xef\xbf\xbd";
        }
    }
    return s;
}

std::string join(const std::vector<std::string>& names, const char* sep) {
    std::string s;
    for (const std::string& n : names) {
        if (!s.empty()) s += sep;
        s += n;
    }
    return s;
}

bool contains(const std::vector<std::string>& names, const std::string& name) {
    for (const std::string& n : names)
        if (n == name) return true;
    return false;
}

namespace {
bool enabled(const char* const* names, std::uint32_t n, const std::string& name) {
    for (std::uint32_t i = 0; i < n; ++i)
        if (names && names[i] && name == names[i]) return true;
    return false;
}
}  // namespace

bool copy_instance(const VkInstanceCreateInfo& in, const std::vector<std::string>& need, InstanceCopy* out) {
    out->added.clear();
    out->names.clear();
    for (const std::string& e : need)
        if (!enabled(in.ppEnabledExtensionNames, in.enabledExtensionCount, e) && !contains(out->added, e)) out->added.push_back(e);
    if (out->added.empty()) return false;
    for (std::uint32_t i = 0; i < in.enabledExtensionCount; ++i) out->names.push_back(in.ppEnabledExtensionNames[i]);
    for (const std::string& e : out->added) out->names.push_back(e.c_str());
    out->info = in;
    out->info.enabledExtensionCount = static_cast<std::uint32_t>(out->names.size());
    out->info.ppEnabledExtensionNames = out->names.data();
    return true;
}

void restore_device(DeviceCopy* c) {
    for (auto it = c->restore.rbegin(); it != c->restore.rend(); ++it) *it->first = it->second;
    c->restore.clear();
}

void plan_device(VkDeviceCreateInfo* in, const std::vector<std::string>& need, const std::vector<std::string>& available,
                 std::uint32_t api_version, bool bda_supported, DeviceCopy* out) {
    out->added.clear();
    out->lacking.clear();
    out->storage.clear();
    out->names.clear();
    out->restore.clear();
    out->copied = false;
    bool head_bda = false, bda_on = false;
    for (const std::string& e : need) {
        const bool bda = e == "VK_EXT_buffer_device_address" || e == "VK_KHR_buffer_device_address";
        if (bda && api_version >= VK_API_VERSION_1_2) {
            // The core feature: the host's own structure turned on where it has one.
            if (!bda_supported) {
                if (!contains(out->lacking, "bufferDeviceAddress")) out->lacking.push_back("bufferDeviceAddress");
                continue;
            }
            if (bda_on || head_bda) continue;
            bool found = false;
            for (auto* s = static_cast<VkBaseOutStructure*>(const_cast<void*>(in->pNext)); s && !found; s = s->pNext) {
                VkBool32* field = nullptr;
                if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                    field = &reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(s)->bufferDeviceAddress;
                else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)
                    field = &reinterpret_cast<VkPhysicalDeviceBufferDeviceAddressFeatures*>(s)->bufferDeviceAddress;
                if (!field) continue;
                found = true;
                if (!*field) {
                    out->restore.emplace_back(field, *field);
                    *field = VK_TRUE;
                    bda_on = true;
                }
            }
            if (!found) head_bda = true;
            continue;
        }
        if (!contains(available, e)) {
            if (!contains(out->lacking, e)) out->lacking.push_back(e);
            continue;
        }
        if (!enabled(in->ppEnabledExtensionNames, in->enabledExtensionCount, e) && !contains(out->storage, e)) out->storage.push_back(e);
    }
    if (!out->lacking.empty()) {
        restore_device(out);
        out->storage.clear();
        return;
    }
    out->added = out->storage;
    if (bda_on || head_bda) out->added.push_back("bufferDeviceAddress");
    if (out->storage.empty() && !head_bda) return;  // nothing to copy (perhaps a field turned on in place)
    out->info = *in;
    if (!out->storage.empty()) {
        for (std::uint32_t i = 0; i < in->enabledExtensionCount; ++i) out->names.push_back(in->ppEnabledExtensionNames[i]);
        for (const std::string& e : out->storage) out->names.push_back(e.c_str());
        out->info.enabledExtensionCount = static_cast<std::uint32_t>(out->names.size());
        out->info.ppEnabledExtensionNames = out->names.data();
    }
    if (head_bda) {
        out->bda = VkPhysicalDeviceBufferDeviceAddressFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
        out->bda.bufferDeviceAddress = VK_TRUE;
        out->bda.pNext = const_cast<void*>(in->pNext);
        out->info.pNext = &out->bda;
    }
    out->copied = true;
}

}  // namespace ngx_bridge
