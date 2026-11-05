// A texture's origin by the file its texels came out of (engine/texture_origin.h).
//
// The game reads a TPF (the texture pack: "TPF\0", the data's size, the
// texture count, then a header and a name a texture, then the texels) out of
// its file, decompressed into memory, and hands GX a pointer into it to make
// each texture with. So the TPF's header lies before the texels: on its own
// where the TPF was a file of its own or an entry of a map tpfbdt, or inside
// a BND4 (parts, sfx, chr, obj), whose file table names every file in it by
// its path on the developers' disk (N:\SPRJ\data\INTERROOT_ps4\parts\...).
// The header is found by looking back from the texels, the BND4 by looking
// back from the TPF and finding the TPF among its files; a TPF of its own is
// placed by its first texture's name: a map tpfbhd names each of the areas'
// textures, the menus' begin MENU_, the colour grading tables are mNN_..._
// CGrading. A texture's own name (its entry in the TPF) then says whether it
// is baked lighting or a grading table, whatever pack holds it. A TPF's textures are made one after another, so the last few
// TPFs found are kept and the next texture is looked for in them first.
#include "engine/texture_origin.h"

#include "core/portable.h"
#include "hle/fs.h"
#include "hle/modules.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::uint32_t kTpfMagic = 0x00465054u;   // "TPF\0"
constexpr std::uint32_t kBndMagic = 0x34444e42u;   // "BND4"
// How far before the texels a TPF's header is looked for. The areas' TPFs
// hold one texture each and the parts' and effects' a few small ones; a
// texture deeper in a pack than this is a character's or an object's, whose
// rule is the same as no origin's - and a look that finds nothing (texels that
// are in no TPF) costs this much reading.
constexpr std::uint64_t kTpfReach = 8ull << 20;
constexpr std::uint64_t kBndReach = 96ull << 20;   // how far before a TPF its BND4's header may lie

bool read(std::uint64_t at, void* out, std::size_t n) {
    return host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), out, n);
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// The printable UTF-16 string at `at` (ASCII only), up to `most` characters.
std::string wide_at(std::uint64_t at, std::size_t most = 260) {
    std::vector<std::uint16_t> buf(most);
    std::size_t n = most;
    while (n && !read(at, buf.data(), n * 2)) n /= 2;  // (the end of a mapping)
    std::string out;
    for (std::size_t i = 0; i < n; ++i) {
        if (buf[i] < 0x20 || buf[i] > 0x7e) break;
        out += static_cast<char>(buf[i]);
    }
    return out;
}

// The highest 4-aligned `magic` at or below `from` and above `from - reach`,
// read in place within the guest mappings that hold it (one after another
// down, while they meet); 0 when there is none.
std::uint64_t find_back(std::uint64_t from, std::uint32_t magic, std::uint64_t reach) {
    const std::uint64_t floor = from > reach ? from - reach : 0;
    std::uint64_t top = (from & ~3ull) + 4;  // one past the highest word looked at
    while (top > floor) {
        std::uint64_t lo = 0, hi = 0;
        if (!hle_kernel_va_mapped_in(top - 4, 4, &lo, &hi)) return 0;
        const std::uint64_t bottom = (std::max(lo, floor) + 3) & ~3ull;
        for (std::uint64_t a = top - 4; a >= bottom; a -= 4) {
            if (*reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(a)) == magic) return a;
            if (a < 4) break;
        }
        if (bottom <= floor || lo == 0) return 0;
        top = lo;
    }
    return 0;
}

// A TPF's header at `t`: its texture count and data size are a pack's, and
// `texels` lies within what it can hold. Its extent: [t, end).
bool tpf_at(std::uint64_t t, std::uint64_t texels, std::uint64_t* end) {
    std::uint32_t h[4] = {};
    if (!read(t, h, sizeof(h)) || h[0] != kTpfMagic) return false;
    const std::uint32_t data = h[1], count = h[2];
    if (!count || count > 8192 || data > (1u << 30)) return false;
    const std::uint64_t e = t + 0x10 + static_cast<std::uint64_t>(count) * 0x80 + data + 0x10000;  // headers, names, data, a margin
    if (texels >= e) return false;
    *end = e;
    return true;
}

// The first texture's name after a TPF's header, read in place (the guest
// mapping that holds the header, 32 KiB at most).
std::string tpf_name(std::uint64_t t) {
    std::uint64_t lo = 0, hi = 0;
    if (!hle_kernel_va_mapped_in(t, 0x10, &lo, &hi)) return {};
    const auto* p = reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(t + 0x10));
    const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(32u << 10, (hi - t - 0x10) / 2));
    for (std::size_t i = 0; i + 3 < n; ++i) {
        std::size_t k = i;
        while (k < n && p[k] >= 0x20 && p[k] <= 0x7e) ++k;
        if (k - i >= 3 && k < n && p[k] == 0) {
            std::string s;
            for (std::size_t m = i; m < k; ++m) s += static_cast<char>(p[m]);
            return s;
        }
        i = k;
    }
    return {};
}

// The name of the TPF at `t`'s texture whose texels start at `texels`, or "":
// the PS4's entries (0x24 bytes each after the 0x10 header) hold the data's
// offset from the TPF at +0 and the name's at +0x18.
std::string tpf_entry_name(std::uint64_t t, std::uint64_t texels) {
    std::uint8_t h[0x10];
    if (!read(t, h, sizeof(h)) || h[12] != 4 || texels <= t || texels - t > 0xffffffffull) return {};
    std::uint32_t count = 0;
    std::memcpy(&count, h + 8, 4);
    if (!count || count > 8192) return {};
    std::vector<std::uint8_t> e(static_cast<std::size_t>(count) * 0x24);
    if (!read(t + 0x10, e.data(), e.size())) return {};
    const std::uint32_t at = static_cast<std::uint32_t>(texels - t);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint32_t off = 0, name = 0;
        std::memcpy(&off, e.data() + i * 0x24, 4);
        std::memcpy(&name, e.data() + i * 0x24 + 0x18, 4);
        if (off == at) return name ? wide_at(t + name, 128) : std::string{};
    }
    return {};
}

// What a texture's own name says over its pack's origin: baked lighting
// (the gi_env packs' *_gi_*, their atlases and their _l lower levels) and
// grading tables are never scaled, wherever they lie.
// BBHOST_TEXTURE_ORIGIN_LOG=1 logs each texture's name and origin.
GxTextureOrigin by_own_name(const std::string& name, GxTextureOrigin pack) {
    const std::string n = lower(name);
    GxTextureOrigin o = pack;
    if (n.find("cgrading") != std::string::npos) o = kOriginGrading;
    else if (n.find("_gi_") != std::string::npos || n.rfind("gi_", 0) == 0 || n.find("lightmap") != std::string::npos) o = kOriginLighting;
    static const bool log = [] {
        const char* e = std::getenv("BBHOST_TEXTURE_ORIGIN_LOG");
        return e && *e == '1';
    }();
    if (log) host_log("texture-origin: %s -> %s", name.empty() ? "(no name)" : name.c_str(), texture_origin_name(o));
    return o;
}

// A BND4's file table: its headers (from +0x40, fileHeaderSize each, the
// name's offset in their last word). The last few BND4s read are kept, so the
// next TPF in one is found without looking back for its header again.
struct Bnd {
    std::uint64_t base = 0, fhs = 0;
    std::uint32_t files = 0;
    std::vector<std::uint8_t> table;
};
thread_local Bnd t_bnds[4];
thread_local unsigned t_bnd_next = 0;

bool bnd_at(std::uint64_t b, Bnd& out) {
    std::uint8_t h[0x40];
    if (!read(b, h, sizeof(h))) return false;
    std::uint32_t files = 0;
    std::uint64_t fhs = 0;
    std::memcpy(&files, h + 0x0c, 4);
    std::memcpy(&fhs, h + 0x20, 8);
    if (!files || files > 65536 || fhs < 0x10 || fhs > 0x80) return false;
    out.table.resize(static_cast<std::size_t>(files) * fhs);
    if (out.table.size() > (4u << 20) || !read(b + 0x40, out.table.data(), out.table.size())) return false;
    out.base = b;
    out.fhs = fhs;
    out.files = files;
    return true;
}

// The path of the file of `d` whose data starts at `t`, or "".
std::string path_in(const Bnd& d, std::uint64_t t) {
    if (!d.files || t <= d.base || t - d.base > 0xffffffffull) return {};
    const std::uint32_t at = static_cast<std::uint32_t>(t - d.base);
    for (std::uint32_t f = 0; f < d.files; ++f) {
        const std::uint8_t* fh = d.table.data() + static_cast<std::size_t>(f) * d.fhs;
        bool holds = false;
        for (std::uint64_t j = 0; j + 4 <= d.fhs - 4 && !holds; j += 4) {
            std::uint32_t v = 0;
            std::memcpy(&v, fh + j, 4);
            holds = v == at;
        }
        if (!holds) continue;
        std::uint32_t name = 0;
        std::memcpy(&name, fh + d.fhs - 4, 4);
        std::string path = wide_at(d.base + name);
        if (path.find('\\') != std::string::npos || path.find(':') != std::string::npos) return path;
    }
    return {};
}

// The path of the file the TPF at `t` is in a BND4 before it, or "".
std::string bnd_path(std::uint64_t t) {
    for (const Bnd& d : t_bnds) {
        std::string p = path_in(d, t);
        if (!p.empty()) return p;
    }
    std::uint64_t from = t - 4;
    for (int tries = 0; tries < 4; ++tries) {
        const std::uint64_t b = find_back(from, kBndMagic, kBndReach - (t - from));
        if (!b) return {};
        from = b - 4;
        Bnd d;
        if (!bnd_at(b, d)) continue;
        std::string p = path_in(d, t);
        if (p.empty()) continue;
        t_bnds[t_bnd_next++ & 3] = std::move(d);
        return p;
    }
    return {};
}

// The areas' texture names: every entry of every map/mNN/*.tpfbhd (its TPFs,
// each named for its texture: m24_00_add_a.tpf.dcx), read from the dump - the
// area folders' own files only (the mNN_NN_NN_NN folders hold the models,
// tens of thousands of files, and no tpfbhd).
const std::unordered_set<std::string>& map_names() {
    static const std::unordered_set<std::string> names = [] {
        std::unordered_set<std::string> out;
        std::error_code ec;
        const std::filesystem::path root = hle_fs_map_path("/app0/dvdroot_ps4/map");
        std::vector<std::filesystem::path> bhds;
        for (auto area = std::filesystem::directory_iterator(root, ec); !ec && area != std::filesystem::directory_iterator(); area.increment(ec)) {
            const std::string dir = area->path().filename().string();
            if (dir.size() != 3 || dir[0] != 'm' || !area->is_directory(ec)) continue;  // mNN, not mNN_NN_NN_NN
            std::error_code ec2;
            for (auto f = std::filesystem::directory_iterator(area->path(), ec2); !ec2 && f != std::filesystem::directory_iterator(); f.increment(ec2)) {
                if (f->path().extension() == ".tpfbhd") bhds.push_back(f->path());
            }
        }
        for (const std::filesystem::path& bhd : bhds) {
            std::ifstream in(bhd, std::ios::binary | std::ios::ate);
            std::string b(static_cast<std::size_t>(std::max<std::streamoff>(0, in.tellg())), '\0');
            in.seekg(0);
            in.read(b.data(), static_cast<std::streamsize>(b.size()));
            // Each name ends ".tpf" (".tpf.dcx"): from there back to its start.
            static const std::string kExt(".\0t\0p\0f\0", 8);
            for (std::size_t at = b.find(kExt); at != std::string::npos; at = b.find(kExt, at + kExt.size())) {
                std::size_t start = at;
                while (start >= 2 && b[start - 1] == 0 && b[start - 2] >= 0x20 && b[start - 2] <= 0x7e) start -= 2;
                if (at - start < 2) continue;
                std::string s;
                for (std::size_t k = start; k < at; k += 2) s += static_cast<char>(std::tolower(static_cast<unsigned char>(b[k])));
                out.insert(std::move(s));
            }
        }
        host_log("texture-origin: %zu texture names in the map folders' tpfbhd files", out.size());
        return out;
    }();
    return names;
}

GxTextureOrigin by_path(const std::string& path) {
    const std::string p = lower(path);
    if (p.find("cgrading") != std::string::npos) return kOriginGrading;
    if (p.find("\\map\\") != std::string::npos) return kOriginMap;
    if (p.find("\\parts\\") != std::string::npos) return kOriginParts;
    if (p.find("\\sfx\\") != std::string::npos) return kOriginSfx;
    if (p.find("\\menu\\") != std::string::npos) return kOriginMenu;
    return kOriginOther;
}

GxTextureOrigin by_name(const std::string& name) {
    const std::string n = lower(name);
    if (n.empty()) return kOriginOther;
    if (n.find("cgrading") != std::string::npos) return kOriginGrading;
    if (n.rfind("menu_", 0) == 0) return kOriginMenu;
    const bool area = n.size() > 4 && n[0] == 'm' && std::isdigit(static_cast<unsigned char>(n[1])) && std::isdigit(static_cast<unsigned char>(n[2])) && n[3] == '_';
    if (area || map_names().count(n)) return kOriginMap;
    return kOriginOther;
}

struct Known {
    std::uint64_t lo = 0, hi = 0;
    GxTextureOrigin origin = kOriginUnknown;
};
thread_local Known t_known[4];
thread_local unsigned t_next = 0;
std::atomic<std::uint64_t> g_textures{0}, g_packs{0}, g_us{0}, g_phase_us[4] = {};
std::uint64_t us_from(std::chrono::steady_clock::time_point t) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count());
}

}  // namespace

GxTextureOrigin texture_origin_of(std::uint64_t texels) {
    if (!texels) return kOriginUnknown;
    g_textures.fetch_add(1, std::memory_order_relaxed);
    for (const Known& k : t_known) {
        if (texels >= k.lo && texels < k.hi) return by_own_name(tpf_entry_name(k.lo, texels), k.origin);
    }
    const auto t0 = std::chrono::steady_clock::now();
    struct Timed {
        std::chrono::steady_clock::time_point t0;
        ~Timed() {
            g_us.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()),
                           std::memory_order_relaxed);
        }
    } timed{t0};
    auto tp = std::chrono::steady_clock::now();
    std::uint64_t from = texels, t = 0, end = 0;
    for (int tries = 0; tries < 8; ++tries) {
        t = find_back(from, kTpfMagic, kTpfReach - (texels - from));
        if (!t || tpf_at(t, texels, &end)) break;
        from = t - 4;  // texel bytes that read "TPF\0", or another pack below this one
        t = 0;
    }
    g_phase_us[0].fetch_add(us_from(tp));
    if (!t) return kOriginUnknown;
    tp = std::chrono::steady_clock::now();
    // Placed by its name where that says (an area's, the menus', a grading
    // table): those are TPFs of their own, and no BND4 is looked for.
    GxTextureOrigin origin = by_name(tpf_name(t));
    g_phase_us[1].fetch_add(us_from(tp));
    tp = std::chrono::steady_clock::now();
    if (origin != kOriginMap && origin != kOriginMenu && origin != kOriginGrading) {
        const std::string path = bnd_path(t);
        if (!path.empty()) origin = by_path(path);
    }
    g_phase_us[2].fetch_add(us_from(tp));
    t_known[t_next++ & 3] = Known{t, end, origin};
    g_packs.fetch_add(1, std::memory_order_relaxed);
    return by_own_name(tpf_entry_name(t, texels), origin);
}

void texture_origin_prepare() {
    std::thread([] { map_names(); }).detach();
}

void texture_origin_stats(std::uint64_t* textures, std::uint64_t* packs, std::uint64_t* us) {
    *textures = g_textures.load();
    *packs = g_packs.load();
    *us = g_us.load();
    host_log("texture-origin: TPF look-ups %.1f ms, names %.1f ms, BND4 look-ups %.1f ms", g_phase_us[0].load() / 1000.0, g_phase_us[1].load() / 1000.0,
             g_phase_us[2].load() / 1000.0);
}

const char* texture_origin_name(std::uint8_t origin) {
    static const char* const kNames[kOriginCount] = {"unknown", "map", "parts", "sfx", "menu", "grading", "lighting", "other"};
    return origin < kOriginCount ? kNames[origin] : "?";
}
