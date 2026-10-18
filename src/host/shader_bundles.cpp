#include "host/shader_bundles.h"

#include "gcn/container.h"
#include "hle/fs.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace gpu {

namespace {

// The game's bundles, the debug menu's last (its shaders repeat others').
constexpr const char* kBundles[] = {
    "gxshader.shaderbnd.dcx",         "gxrenderershader.shaderbnd.dcx", "gxposteffect.shaderbnd.dcx", "gxgui.shaderbnd.dcx",
    "gxffxshader.shaderbnd.dcx",      "gxdecal.shaderbnd.dcx",          "gxflvershader.shaderbnd.dcx",
    "gxflvershader.shaderbnddebug.dcx",
};

struct Bundles {
    std::once_flag once;
    bool ok = false;
    std::vector<BundleShader> list;
    std::unordered_map<std::string, std::size_t> by_name;  // index into list
    std::string summary;
};
Bundles g_bundles;

void read_bundles() {
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t files = 0, bytes = 0, entries = 0, repeats = 0;
    std::string missing;
    for (const char* b : kBundles) {
        const std::string guest = std::string("/app0/dvdroot_ps4/shader/") + b;
        const std::string path = hle_fs_map_path(guest.c_str());
        std::vector<std::uint8_t> raw;
        if (path.empty() || !gcn::read_file(path, raw)) {
            missing += std::string(" ") + b;
            continue;
        }
        std::string err;
        const std::vector<std::uint8_t> plain = gcn::dcx_decompress(raw, &err);
        std::vector<gcn::BundleEntry> found;
        if (plain.empty() || !gcn::bnd4_entries(plain, found, &err)) {
            host_log("render: shader bundle %s: %s", path.c_str(), err.c_str());
            continue;
        }
        ++files;
        bytes += raw.size();
        for (gcn::BundleEntry& e : found) {
            ++entries;
            const std::string name = shader_container_name(e.data.data(), e.data.size());
            if (name.empty()) continue;
            if (g_bundles.by_name.count(name)) {
                ++repeats;
                continue;
            }
            BundleShader s;
            s.name = name;
            s.type = shader_container_type(e.data.data(), e.data.size());
            s.debug = std::strstr(b, "debug") != nullptr;
            s.file = e.name.substr(e.name.find_last_of("\\/") == std::string::npos ? 0 : e.name.find_last_of("\\/") + 1);
            s.container = std::make_shared<const std::vector<std::uint8_t>>(std::move(e.data));
            g_bundles.by_name.emplace(name, g_bundles.list.size());
            g_bundles.list.push_back(std::move(s));
        }
    }
    g_bundles.ok = !g_bundles.list.empty();
    const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::size_t by_type[8] = {};
    for (const BundleShader& s : g_bundles.list) by_type[s.type >= 0 && s.type < 8 ? s.type : 7] += 1;
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "%zu shaders in %zu bundles (%.1f MiB; %zu entries, %zu names repeated): vertex %zu, pixel %zu, compute %zu, other %zu; "
                  "read in %lld ms%s%s",
                  g_bundles.list.size(), files, bytes / 1048576.0, entries, repeats, by_type[1], by_type[2], by_type[4],
                  g_bundles.list.size() - by_type[1] - by_type[2] - by_type[4], ms, missing.empty() ? "" : "; not found:", missing.c_str());
    g_bundles.summary = buf;
}

}  // namespace

std::string shader_container_name(const std::uint8_t* container, std::size_t size) {
    for (std::size_t i = size >= 20 ? size - 20 : 0;; --i) {  // the 28-byte footer sits at the end
        if (i + 20 <= size && std::memcmp(container + i, "OrbShdr", 7) == 0) {
            std::uint32_t h0;
            std::memcpy(&h0, container + i + 16, 4);
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%08x", h0);
            return buf;
        }
        if (i == 0) break;
    }
    return {};
}

int shader_container_type(const std::uint8_t* container, std::size_t size) {
    for (std::size_t i = 0; i + 16 <= size && i < 0x100; ++i) {
        if (std::memcmp(container + i, "Shdr", 4) == 0) return container[i + 8] & 0xf;
    }
    return -1;
}

bool shader_bundles_load() {
    std::call_once(g_bundles.once, read_bundles);
    return g_bundles.ok;
}

std::shared_ptr<const std::vector<std::uint8_t>> shader_bundle_container(const std::string& name) {
    if (!shader_bundles_load()) return nullptr;
    const auto it = g_bundles.by_name.find(name);
    return it == g_bundles.by_name.end() ? nullptr : g_bundles.list[it->second].container;
}

std::string shader_bundle_file_name(const std::string& name) {
    if (!shader_bundles_load()) return {};
    const auto it = g_bundles.by_name.find(name);
    return it == g_bundles.by_name.end() ? std::string() : g_bundles.list[it->second].file;
}

const std::vector<BundleShader>& shader_bundles_list() {
    shader_bundles_load();
    return g_bundles.list;
}

std::vector<BundleShader> eboot_shaders(const std::string& path) {
    std::vector<BundleShader> out;
    std::vector<std::uint8_t> d;
    if (path.empty() || !gcn::read_file(path, d)) return out;
    std::unordered_map<std::string, bool> seen;
    // Each container: a ShaderFileHeader ('Shdr', not the tail of a footer's
    // 'OrbShdr'), its code, and the ShaderBinaryInfo footer after it, 28 bytes.
    static const char kShdr[] = "Shdr";
    static const char kFooter[] = "OrbShdr";
    for (std::size_t at = 0; at + 0x60 <= d.size(); ++at) {
        const std::uint8_t* p = static_cast<const std::uint8_t*>(std::memchr(d.data() + at, 'S', d.size() - at));
        if (!p) break;
        at = static_cast<std::size_t>(p - d.data());
        if (at + 0x60 > d.size() || std::memcmp(p, kShdr, 4) != 0 || (at >= 3 && std::memcmp(p - 3, "Orb", 3) == 0)) continue;
        const int type = p[8] & 0xf;
        if (type != 1 && type != 2 && type != 4) continue;
        // The footer: the first one after the header, within 1 MiB.
        const std::size_t limit = std::min(d.size(), at + (1u << 20));
        std::size_t foot = std::string::npos;
        for (std::size_t k = at + 16; k + 28 <= limit; ++k) {
            if (d[k] == 'O' && std::memcmp(d.data() + k, kFooter, 7) == 0) {
                foot = k;
                break;
            }
        }
        if (foot == std::string::npos) continue;
        BundleShader s;
        s.name = shader_container_name(d.data() + at, foot + 28 - at);
        s.type = type;
        if (s.name.empty() || !seen.emplace(s.name, true).second) continue;
        s.container = std::make_shared<const std::vector<std::uint8_t>>(d.begin() + static_cast<std::ptrdiff_t>(at),
                                                                        d.begin() + static_cast<std::ptrdiff_t>(foot + 28));
        out.push_back(std::move(s));
    }
    return out;
}

namespace {
struct EbootPrograms {
    std::mutex mu;
    std::string path;  // under mu
    bool read = false;  // under mu
    std::unordered_map<std::string, std::shared_ptr<const std::vector<std::uint8_t>>> by_name;  // under mu, once read
};
EbootPrograms g_eboot;
}  // namespace

void shader_programs_set_eboot(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_eboot.mu);
    g_eboot.path = path;
}

std::shared_ptr<const std::vector<std::uint8_t>> shader_program_container(const std::string& name) {
    if (auto c = shader_bundle_container(name)) return c;
    std::lock_guard<std::mutex> lk(g_eboot.mu);
    if (!g_eboot.read) {
        g_eboot.read = true;
        for (BundleShader& s : eboot_shaders(g_eboot.path)) g_eboot.by_name.emplace(s.name, std::move(s.container));
    }
    const auto it = g_eboot.by_name.find(name);
    return it == g_eboot.by_name.end() ? nullptr : it->second;
}

std::string shader_bundles_summary() {
    shader_bundles_load();
    return g_bundles.summary;
}

}  // namespace gpu
