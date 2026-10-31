#include "engine/patch_manifest.h"

#include "core/config.h"
#include "core/elf.h"
#include "core/memory.h"
#include "core/portable.h"
#include "engine/addr.h"
#include "gcn/container.h"
#include "hle/fs.h"
#include "host/ruleset.h"
#include "host/settings.h"
#include "log.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Site {
    std::string id;
    std::uint64_t address = 0;  // guest VA at the preferred base
    std::vector<std::uint8_t> expect, bytes;
    bool code = true;
    // Patches this site steps aside for: while any of them is on, this site
    // is neither checked nor written - two patches that write the same bytes
    // differently, the one named here having the last word.
    std::vector<std::string> yields;
};

struct Manifest {
    std::string path, name, title, eboot, description, option, group;
    bool enabled = true;
    bool gameplay = false;  // [patch] gameplay: changes the game (the session's online rules say so)
    bool live = true;       // [patch] live: it may change while the game runs (patch_manifest_set_live)
    int order = 0;  // [patch] order: its place in the list it came from (later wins)
    std::vector<Site> sites;
};

// What the applied manifests wrote, by guest address at the preferred base:
// the manifest each byte is now, and the eboot's own byte under it. A later
// manifest's site is checked against those, so it may write over an earlier
// one's bytes (patch_manifest_owner answers the host's own patches).
// The applied manifests that change the game, by name (patch_manifests_gameplay),
// and whether patch_manifests_apply has run.
std::mutex g_gameplay_mu;
std::vector<std::string> g_gameplay;  // under g_gameplay_mu
std::atomic<bool> g_manifests_done{false};

struct Written {
    std::string owner;
    std::uint8_t original = 0;
};
std::map<std::uint64_t, Written> g_written;  // under g_live_mu once the game runs

// Every manifest read for this eboot at the start, in the order they apply,
// and whether its bytes are in now: what the patch menu turns on and off while
// the game runs (patch_manifest_set_live). The image's memory, which lives as
// long as the run. A manifest bbhost's own patches left a value to
// (patch_manifest_owner: the frame-rate conversion skips what one wrote) keeps
// its bytes until the next start: taken out, that value would be the eboot's,
// not the conversion's.
struct Loaded {
    Manifest m;
    bool applied = false;
};
std::mutex g_live_mu;
std::vector<Loaded> g_loaded;         // under g_live_mu
GuestMemory* g_mem = nullptr;         // under g_live_mu
std::set<std::string> g_deferred_to;  // under g_live_mu

std::string unquote(const std::string& v) {
    if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') && v.back() == v.front()) return v.substr(1, v.size() - 2);
    return v;
}

// A TOML array of strings, ["a", "b"] (or a lone string).
std::vector<std::string> parse_string_list(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false;
    char q = 0;
    for (char c : text) {
        if (quoted) {
            if (c == q) {
                quoted = false;
                out.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        } else if (c == '"' || c == '\'') {
            quoted = true;
            q = c;
        }
    }
    if (out.empty() && !text.empty() && text.front() != '[') out.push_back(unquote(text));
    return out;
}

bool parse_hex_bytes(const std::string& text, std::vector<std::uint8_t>* out) {
    out->clear();
    std::string s;
    for (char c : text) {
        if (c == ' ' || c == ',' || c == '\t') continue;
        s.push_back(c);
    }
    if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
    if (s.empty() || s.size() % 2) return false;
    for (std::size_t i = 0; i < s.size(); i += 2) {
        char* end = nullptr;
        const std::string pair = s.substr(i, 2);
        const long v = std::strtol(pair.c_str(), &end, 16);
        if (!end || *end) return false;
        out->push_back(static_cast<std::uint8_t>(v));
    }
    return true;
}

bool parse_manifest(const std::string& path, Manifest* m, std::string* err) {
    std::map<std::string, std::string> kv;
    if (!config_parse_toml(path, &kv, err)) return false;
    m->path = path;
    auto get = [&](const std::string& k) {
        auto it = kv.find(k);
        return it == kv.end() ? std::string() : unquote(it->second);
    };
    m->name = get("patch.name");
    m->title = get("patch.title");
    m->eboot = get("patch.eboot");
    m->description = get("patch.description");
    m->option = get("patch.option");
    m->group = get("patch.group");
    m->order = std::atoi(get("patch.order").c_str());
    // `default`: the name an earlier port of Kyo's list gave the same switch.
    std::string en = get("patch.enabled");
    if (en.empty()) en = get("patch.default");
    m->enabled = en.empty() || en == "true" || en == "1";
    const std::string gp = get("patch.gameplay");
    m->gameplay = gp == "true" || gp == "1";
    const std::string lv = get("patch.live");
    m->live = !(lv == "false" || lv == "0");
    if (m->name.empty() || m->eboot.empty()) {
        *err = "[patch] needs name and eboot";
        return false;
    }
    std::map<std::string, Site> sites;
    for (const auto& [k, v] : kv) {
        if (k.rfind("site.", 0) != 0) continue;
        const std::size_t dot = k.find('.', 5);
        if (dot == std::string::npos) continue;
        Site& s = sites[k.substr(5, dot - 5)];
        s.id = k.substr(5, dot - 5);
        const std::string field = k.substr(dot + 1), val = unquote(v);
        if (field == "address") {
            s.address = std::strtoull(val.c_str(), nullptr, 16);
        } else if (field == "expect") {
            if (!parse_hex_bytes(val, &s.expect)) {
                *err = "site " + s.id + ": bad expect";
                return false;
            }
        } else if (field == "bytes") {
            if (!parse_hex_bytes(val, &s.bytes)) {
                *err = "site " + s.id + ": bad bytes";
                return false;
            }
        } else if (field == "kind") {
            s.code = val != "data";
        } else if (field == "yields") {
            s.yields = parse_string_list(v);
        }
    }
    for (auto& [id, s] : sites) {
        if (!s.address || s.expect.empty() || s.bytes.size() != s.expect.size()) {
            *err = "site " + id + ": needs address, expect and bytes of the same length";
            return false;
        }
        m->sites.push_back(s);
    }
    if (m->sites.empty()) {
        *err = "no [site.*]";
        return false;
    }
    return true;
}

// The config switches a manifest may name.
bool option_on(const std::string& option, bool* known) {
    *known = true;
    if (option == "startup.skip_intro") return config().skip_intro;
    if (option == "loading.quick_reentry") return config().quick_reentry;
    if (option == "streaming.all_post_processors") return config().all_post_processors;
    if (option == "debug.free_camera") return host_settings().debug_camera;  // F10 > STARTUP, read once
    if (option == "startup.performance_mode") return host_settings().performance_mode;  // F10 > STARTUP, read once
    *known = false;
    return false;
}

// Whether a site steps aside for a patch that is on (Site::yields).
bool yielded(const Site& s, const std::set<std::string>& on) {
    for (const std::string& y : s.yields) {
        if (on.count(y)) return true;
    }
    return false;
}

// The eboot's own byte at a guest address (preferred base): under whatever an
// applied manifest wrote there.
std::uint8_t original_byte(std::uint64_t address, std::uint8_t now) {
    const auto it = g_written.find(address);
    return it == g_written.end() ? now : it->second.original;
}

// Whether it applied: every site written, already as it writes it, or left to
// a patch it yields to (false: refused, or a page could not be unprotected).
bool apply(ElfImage* image, const Manifest& m, const std::set<std::string>& on) {
    GuestMemory& mem = image->mem;
    // Every site checked before any is written - against the eboot's own
    // bytes, under any an earlier manifest wrote: manifests apply in their
    // list's order (order, then name), and where two write the same bytes the
    // later one's stand, as shadPS4 applies Kyo's list.
    for (const Site& s : m.sites) {
        if (yielded(s, on)) continue;
        if (s.address < kPreferredGuestSlide) {
            host_log("patch: %s refused: site %s address 0x%llx is below the image", m.name.c_str(), s.id.c_str(),
                     static_cast<unsigned long long>(s.address));
            return false;
        }
        const std::uint64_t off = s.address - kPreferredGuestSlide;
        if (off >= mem.size || s.expect.size() > mem.size - off) {
            host_log("patch: %s refused: site %s at 0x%llx is outside the image", m.name.c_str(), s.id.c_str(),
                     static_cast<unsigned long long>(s.address));
            return false;
        }
        const auto* p = static_cast<const std::uint8_t*>(guest_ptr(mem, mem.slide + off));
        // Already applied (the bytes are what the patch writes) is not a
        // refusal: the same file from two directories, or a rerun.
        if (std::memcmp(p, s.bytes.data(), s.bytes.size()) == 0) continue;
        for (std::size_t i = 0; i < s.expect.size(); ++i) {
            if (p[i] == s.expect[i] || p[i] == s.bytes[i] || original_byte(s.address + i, p[i]) == s.expect[i]) continue;
            host_log("patch: %s refused: unexpected bytes at 0x%llx (site %s)", m.name.c_str(),
                     static_cast<unsigned long long>(s.address), s.id.c_str());
            return false;
        }
    }
    int written = 0, stepped_aside = 0, already = 0;
    std::string aside, over;
    for (const Site& s : m.sites) {
        if (yielded(s, on)) {
            ++stepped_aside;
            for (const std::string& y : s.yields) {
                if (on.count(y) && aside.find(y) == std::string::npos) aside += (aside.empty() ? "" : ", ") + y;
            }
            continue;
        }
        const std::uint64_t at = mem.slide + (s.address - kPreferredGuestSlide);
        auto* p = static_cast<std::uint8_t*>(guest_ptr(mem, at));
        if (std::memcmp(p, s.bytes.data(), s.bytes.size()) == 0) {
            already += s.bytes != s.expect;  // another patch wrote the same (a site that writes what is there counts neither way)
            continue;
        }
        const std::uint64_t lo = at & ~0xfffull, hi = (at + s.bytes.size() + 0xfff) & ~0xfffull;
        const bool ok = s.code ? guest_protect_rwx(&mem, lo, hi - lo) : guest_protect_rw(&mem, lo, hi - lo);
        if (!ok) {
            host_log("patch: %s: cannot unprotect 0x%llx (site %s); the rest is left alone", m.name.c_str(),
                     static_cast<unsigned long long>(s.address), s.id.c_str());
            return false;
        }
        for (std::size_t i = 0; i < s.bytes.size(); ++i) {
            if (p[i] == s.bytes[i]) continue;
            const auto [it, fresh] = g_written.try_emplace(s.address + i, Written{m.name, p[i]});
            if (!fresh && it->second.owner != m.name) {
                if (over.find(it->second.owner) == std::string::npos) over += (over.empty() ? "" : ", ") + it->second.owner;
                it->second.owner = m.name;
            }
        }
        std::memcpy(p, s.bytes.data(), s.bytes.size());
        if (s.code && !guest_protect_rx(&mem, lo, hi - lo)) {
            host_log("patch: %s: cannot restore the protection of 0x%llx", m.name.c_str(),
                     static_cast<unsigned long long>(s.address));
            std::abort();
        }
        ++written;
    }
    std::string notes;
    if (already) notes += " (" + std::to_string(already) + " already as it writes them)";
    if (stepped_aside) notes += " (" + std::to_string(stepped_aside) + " left to " + aside + ")";
    if (!over.empty()) notes += " (over " + over + "'s bytes, as it comes after in the list)";
    host_log("patch: %s: %d of %zu sites written%s%s%s", m.name.c_str(), written, m.sites.size(), notes.c_str(),
             m.description.empty() ? "" : " - ", m.description.c_str());
    return true;
}

// The directories manifests come from, in order: the package's, a build
// tree's, the player's own (beside the per-user bbhost.toml: patches of their
// own that no copy of bbhost ships, and every copy reads), the mods
// directory's.
std::vector<std::string> manifest_dirs(const std::string& mods) {
    std::vector<std::string> dirs = {config_exe_dir() + "/patches", config_exe_dir() + "/../patches", config_user_dir() + "/patches"};
    if (!mods.empty()) dirs.push_back(mods + "/patches");
    return dirs;
}

// Whether a texture pack (a TPF, inflated) names a texture: its name table
// holds each name NUL-terminated, in UTF-16 or in single bytes.
bool tpf_names(const std::vector<std::uint8_t>& tpf, const std::string& name) {
    std::vector<std::uint8_t> wide, narrow(name.begin(), name.end());
    for (const char c : name) wide.insert(wide.end(), {static_cast<std::uint8_t>(c), 0});
    wide.insert(wide.end(), {0, 0});
    narrow.push_back(0);
    return std::search(tpf.begin(), tpf.end(), wide.begin(), wide.end()) != tpf.end() ||
           std::search(tpf.begin(), tpf.end(), narrow.begin(), narrow.end()) != tpf.end();
}

// The Old Hunters edition looks for the DLC's own files, which came with the
// update that brought it (1.07), so a game folder holding the base game's
// versions has none of them: the edition's title art, MENU_Title_00004 in
// menu/title.tpf.dcx - without it the title movie fills its picture flat
// green (a player's game folder, 2026-10-07) - and the three areas' maps,
// m34-m36 (their lamps are the Old Hunters' headstone's, slots 44-54).
// What is missing, or "" when it is all there; read as the game will read it,
// update folder and mods included.
std::string old_hunters_missing() {
    const std::string pack = hle_fs_map_path("/app0/dvdroot_ps4/menu/title.tpf.dcx");
    std::ifstream in(pack, std::ios::binary);
    if (!in) return "menu/title.tpf.dcx";
    const std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::vector<std::uint8_t> tpf = gcn::dcx_decompress(raw);
    if (!tpf_names(tpf.empty() ? raw : tpf, "MENU_Title_00004")) return "Old Hunters title art (MENU_Title_00004 in menu/title.tpf.dcx)";
    for (const char* map : {"m34_00_00_00", "m35_00_00_00", "m36_00_00_00"}) {
        std::error_code ec;
        if (!std::filesystem::is_directory(hle_fs_map_path(("/app0/dvdroot_ps4/map/" + std::string(map)).c_str()), ec))
            return std::string("map/") + map + " (an Old Hunters area)";
    }
    return {};
}

// bbhost's own byte patches, carried in the binary rather than as files: a
// setting switches each, and no patch list shows one beside its setting. A
// manifest file of the same name replaces it (an older package's).
//
// performance-mode, Performance mode (startup.performance_mode): Kyo's Giga
// Performance Patch's post-effects. The scene's constructor sets its feature
// flags with one `mov dword [rbx+0x2968], 0x01010101` (SSAO +0x2968, motion
// blur +0x2969, anti-aliasing +0x296a) and the dynamic lights' shadows at
// +0x296e (imm at 0x26c2538) - each 0. The directional shadow (+0x296c, the
// sun's and the moon's: the buildings' shadows) stays. YEBIS's apply (sub_25d5f80) jumps past
// every depth-of-field pass (0x25d7bbc) and reads GlareShape as 0, no bloom
// (0x25d72f0). The map's fog is left as the game draws it (the Fog setting
// still scales it). Chromatic aberration is its own setting's mask
// (engine/graphics_patch.cpp), cleared while this is on.
Manifest performance_mode_manifest() {
    Manifest m;
    m.name = "performance-mode";
    m.title = "Performance mode";
    m.eboot = kEboot109Sha256;
    m.description = "every post-effect and YEBIS pass off: SSAO, motion blur, anti-aliasing, depth of field, chromatic "
                    "aberration, bloom and the dynamic lights' shadows";
    m.option = "startup.performance_mode";
    m.order = 11;
    const std::vector<std::uint8_t> one = {0x01}, zero = {0x00};
    m.sites.push_back(Site{"ssao", 0x026c2548, one, zero, true, {}});
    m.sites.push_back(Site{"motion-blur", 0x026c2549, one, zero, true, {}});
    m.sites.push_back(Site{"anti-aliasing", 0x026c254a, one, zero, true, {}});
    m.sites.push_back(Site{"dynamic-light-shadow", 0x026c2538, one, zero, true, {}});
    m.sites.push_back(Site{"dof", 0x025d7bbc, {0x0f, 0x84, 0xde, 0x00, 0x00, 0x00}, {0xe9, 0xdf, 0x00, 0x00, 0x00, 0x90}, true, {}});
    m.sites.push_back(Site{"bloom", 0x025d72f0, {0x8b, 0xb3, 0xd8, 0x00, 0x00, 0x00}, {0x31, 0xf6, 0x0f, 0x1f, 0x40, 0x00}, true, {}});
    return m;
}

// skip-intro, Skip company logos (startup.skip_intro): illusion's 1.09 Skip
// Intro. The three company logos (FromSoftware, Japan Studio, SCE) are
// entries the startup flow plays by name; emptying the first two UTF-16
// characters of each name skips them. The warning that the last session may
// not have ended with Exit Game - shown after any session that ended another
// way, which on PC is closing the window - is a step the flow builds only in
// its state 6 (sub_202edf0: a dialog of message 400003, group 204); the jne
// there becoming a jmp builds the step after it, as every other state does
// (the community list's 0x0202EE1C EB73). Story movies are untouched.
std::vector<Manifest> built_in_manifests() {
    Manifest m;
    m.name = "skip-intro";
    m.title = "Skip company logos";
    m.eboot = kEboot109Sha256;
    m.description = "the three company logos and the Exit Game warning are skipped at startup";
    m.option = "startup.skip_intro";
    m.order = 10;
    const std::vector<std::uint8_t> name = {0x4c, 0x00, 0x6f, 0x00}, none(4, 0);
    m.sites.push_back(Site{"logo-fromsoft", 0x04d99138, name, none, true, {}});
    m.sites.push_back(Site{"logo-jstudio", 0x04d99154, name, none, true, {}});
    m.sites.push_back(Site{"logo-sce", 0x04d9916e, name, none, true, {}});
    m.sites.push_back(Site{"startup-warning", 0x0202ee1c, {0x75, 0x73}, {0xeb, 0x73}, true, {}});
    std::vector<Manifest> out;
    out.push_back(std::move(m));
    out.push_back(performance_mode_manifest());
    return out;
}

// By file name, a later directory's file replacing an earlier one's.
std::map<std::string, std::string> manifest_files(const std::vector<std::string>& dirs) {
    std::map<std::string, std::string> files;
    for (const std::string& dir : dirs) {
        std::vector<std::string> names;
        if (!host_list_dir(dir.c_str(), &names)) continue;
        for (const std::string& n : names) {
            if (n.size() > 5 && n.compare(n.size() - 5, 5, ".toml") == 0) files[n] = dir + "/" + n;
        }
    }
    return files;
}

}  // namespace

bool patch_manifest_option_known(const std::string& option) {
    bool known = false;
    option_on(option, &known);
    return known;
}

bool patch_manifest_on(const std::string& name, bool enabled) {
    const std::string v = config_value("patches." + name);
    if (!v.empty()) return !(v == "false" || v == "0");
    return enabled && !patch_manifest_off(name);
}

bool patch_manifest_off(const std::string& name) {
    const std::string v = config_value("patches." + name);
    if (!v.empty()) return v == "false" || v == "0";
    // The Old Hunters was a PC enhancement in v0.2.9, `old_hunters` under
    // [world]: a bbhost.toml (a package's, a player's) that turned it off there
    // keeps it off until [patches] says otherwise.
    if (name == "old-hunters") {
        const std::string w = config_value("world.old_hunters");
        return w == "false" || w == "0";
    }
    return false;
}

std::vector<PatchListing> patch_manifests_list(const std::string& mods) {
    std::vector<PatchListing> out;
    for (const auto& [file, path] : manifest_files(manifest_dirs(mods))) {
        Manifest m;
        PatchListing l;
        l.path = path;
        if (!parse_manifest(path, &m, &l.error)) {
            l.name = file.substr(0, file.size() - 5);
        } else {
            l.name = m.name;
            l.title = m.title;
            l.description = m.description;
            l.option = m.option;
            l.eboot = m.eboot;
            l.enabled = m.enabled;
            l.gameplay = m.gameplay;
            l.live = m.live;
            l.group = m.group;
            l.order = m.order;
        }
        out.push_back(l);
    }
    // As they apply: the list they came from, in its order.
    std::stable_sort(out.begin(), out.end(),
                     [](const PatchListing& a, const PatchListing& b) { return std::tie(a.order, a.name) < std::tie(b.order, b.name); });
    return out;
}

const char* patch_manifest_owner(std::uint64_t address, std::size_t n) {
    std::lock_guard<std::mutex> lk(g_live_mu);
    const auto it = g_written.lower_bound(address);
    if (it == g_written.end() || it->first >= address + n) return nullptr;
    // Whoever asks leaves this value to the manifest: it keeps its bytes until
    // the next start (patch_manifest_set_live), so its entries, and the string
    // returned, stay put.
    g_deferred_to.insert(it->second.owner);
    return it->second.owner.c_str();
}

std::vector<std::string> patch_manifests_gameplay() {
    std::lock_guard<std::mutex> lk(g_gameplay_mu);
    return g_gameplay;
}

bool patch_manifests_done() { return g_manifests_done.load(std::memory_order_acquire); }

void patch_manifests_apply(ElfImage* image) {
    struct Done {
        ~Done() { g_manifests_done.store(true, std::memory_order_release); }
    } done;
    if (const char* e = std::getenv("BBHOST_PATCHES"); e && e[0] == '0') {
        host_log("patch: manifests off (BBHOST_PATCHES=0)");
        return;
    }
    const char* mods = hle_fs_mods_root();
    const std::vector<std::string> dirs = manifest_dirs(mods ? mods : "");
    const std::map<std::string, std::string> files = manifest_files(dirs);
    if (files.empty()) {
        host_log("patch: no manifest files (looked in %s, %s, %s%s); bbhost's own still apply", dirs[0].c_str(), dirs[1].c_str(),
                 dirs[2].c_str(), dirs.size() > 3 ? ", the mods directory" : "");
    }
    std::lock_guard<std::mutex> lk(g_live_mu);
    g_mem = &image->mem;
    // The files for this eboot, then bbhost's own that no file replaces.
    std::vector<Manifest> read;
    for (const auto& [name, path] : files) {
        Manifest m;
        std::string err;
        if (!parse_manifest(path, &m, &err)) {
            host_log("patch: %s: %s", path.c_str(), err.c_str());
            continue;
        }
        if (m.eboot != image->sha256) {
            host_log("patch: %s skipped: for eboot %.12s..., this is %.12s...", m.name.c_str(), m.eboot.c_str(),
                     image->sha256.c_str());
            continue;
        }
        read.push_back(std::move(m));
    }
    for (Manifest& b : built_in_manifests()) {
        if (b.eboot != image->sha256) continue;
        if (std::none_of(read.begin(), read.end(), [&](const Manifest& m) { return m.name == b.name; })) read.push_back(std::move(b));
    }
    // Which are on, first: a site may step aside for another patch (yields).
    std::vector<Manifest> chosen;
    for (Manifest& m : read) {
        // Every one for this eboot, on or not, for the patch menu.
        g_loaded.push_back({m, false});
        if (!patch_manifest_on(m.name, m.enabled)) {
            // Off by the player, or off until the player turns it on (enabled = false).
            if (m.enabled) host_log("patch: %s off ([patches] in bbhost.toml: the setup window's Patches tab)", m.name.c_str());
            continue;
        }
        if (!m.option.empty()) {
            bool known = false;
            const bool on = option_on(m.option, &known);
            if (!known) {
                host_log("patch: %s skipped: unknown option %s", m.name.c_str(), m.option.c_str());
                continue;
            }
            if (!on) continue;
        }
        if (m.name == "old-hunters") {
            if (const std::string missing = old_hunters_missing(); !missing.empty()) {
                host_log("patch: old-hunters skipped - the game folder has no %s. Its files are older than the 1.09 "
                         "update's, which bring The Old Hunters: copy the update's files over the game folder, replacing "
                         "the ones there (or keep them in a CUSA00900-UPDATE folder beside it). The game stays the edition "
                         "without it.",
                         missing.c_str());
                continue;
            }
        }
        chosen.push_back(std::move(m));
    }
    // In their list's order; one of a group (variants of one patch: the Model
    // LODs, the hunter sizes, Performance Patch v3's two) - the setup window
    // and the patch menu turn the others off, and a bbhost.toml with two on
    // keeps the later.
    std::stable_sort(chosen.begin(), chosen.end(), [](const Manifest& a, const Manifest& b) { return a.order < b.order; });
    std::map<std::string, std::string> group_on;
    for (auto it = chosen.rbegin(); it != chosen.rend(); ++it) {
        if (!it->group.empty()) group_on.try_emplace(it->group, it->name);
    }
    std::vector<Manifest> apply_list;
    for (Manifest& m : chosen) {
        if (!m.group.empty() && group_on[m.group] != m.name) {
            host_log("patch: %s off: %s, another of group %s, is on (one at a time)", m.name.c_str(), group_on[m.group].c_str(),
                     m.group.c_str());
            continue;
        }
        apply_list.push_back(std::move(m));
    }
    std::set<std::string> on;
    for (const Manifest& m : apply_list) on.insert(m.name);
    std::vector<std::string> gameplay;
    std::stable_sort(g_loaded.begin(), g_loaded.end(),
                     [](const Loaded& a, const Loaded& b) { return std::tie(a.m.order, a.m.name) < std::tie(b.m.order, b.m.name); });
    for (const Manifest& m : apply_list) {
        const bool applied = apply(image, m, on);
        if (applied && m.gameplay) gameplay.push_back(m.name);
        for (Loaded& l : g_loaded) {
            if (l.m.name == m.name) l.applied = applied;
        }
    }
    std::sort(gameplay.begin(), gameplay.end());
    if (!gameplay.empty()) {
        std::string names, tokens;
        for (const std::string& n : gameplay) {
            names += (names.empty() ? "" : ", ") + n;
            tokens += (tokens.empty() ? "" : ",") + patch_rule_token(n);
        }
        host_log("patch: %s change%s the game: the session's online rules carry %s, so the server keeps its bloodstains, "
                 "messages and statistics apart from vanilla ones",
                 names.c_str(), gameplay.size() == 1 ? "s" : "", tokens.c_str());
    }
    std::lock_guard<std::mutex> gk(g_gameplay_mu);
    g_gameplay = std::move(gameplay);
}

namespace {

// Whether a code site is new code where the eboot has zeros (a cave): put in
// before the jumps to it, and left in when they are taken out.
bool cave(const Site& s) {
    return s.code && std::all_of(s.expect.begin(), s.expect.end(), [](std::uint8_t b) { return b == 0; });
}

// One compare-and-swap of the aligned 8-byte word holding [p, p + n): no
// thread sees half of it. The word's other bytes are whatever is there.
void store_in_word(std::uint8_t* p, const std::uint8_t* bytes, std::size_t n) {
    auto* word = reinterpret_cast<std::uint64_t*>(reinterpret_cast<std::uintptr_t>(p) & ~std::uintptr_t{7});
    const std::size_t at = static_cast<std::size_t>(p - reinterpret_cast<std::uint8_t*>(word));
    std::uint64_t was = __atomic_load_n(word, __ATOMIC_ACQUIRE), now;
    do {
        now = was;
        std::memcpy(reinterpret_cast<std::uint8_t*>(&now) + at, bytes, n);
    } while (!__atomic_compare_exchange_n(word, &was, now, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE));
}

bool in_one_word(const std::uint8_t* p, std::size_t n) {
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p);
    return n && n <= 8 && (a & ~std::uintptr_t{7}) == ((a + n - 1) & ~std::uintptr_t{7});
}

// [p, p + n) one aligned word at a time, the last first: a thread sees each
// word old or new.
void store_words(std::uint8_t* p, const std::uint8_t* bytes, std::size_t n) {
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p);
    for (std::size_t end = n; end;) {
        const std::uintptr_t word = (a + end - 1) & ~std::uintptr_t{7};
        const std::size_t begin = word > a ? static_cast<std::size_t>(word - a) : 0;
        store_in_word(p + begin, bytes + begin, end - begin);
        end = begin;
    }
}

// An opcode a hook starts with (jmp, call, nop, ret, int3): the site starts
// an instruction, where a thread can arrive.
bool starts_insn(std::uint8_t b) { return b == 0xe9 || b == 0xe8 || b == 0xeb || b == 0x90 || b == 0xc3 || b == 0xcc; }

// Writes a site while the game runs (patch_manifest_set_live's rules). The
// page is writable for the length of it.
bool write_live(const Site& s, std::uint8_t* p, const std::uint8_t* bytes) {
    const std::size_t n = s.bytes.size();
    const std::uint64_t at = reinterpret_cast<std::uint64_t>(p);
    const std::uint64_t lo = at & ~0xfffull, hi = (at + n + 0xfff) & ~0xfffull;
    if (!(s.code ? guest_protect_rwx(g_mem, lo, hi - lo) : guest_protect_rw(g_mem, lo, hi - lo))) return false;
    if (in_one_word(p, n)) {
        store_in_word(p, bytes, n);
    } else if (s.code && n > 2 && in_one_word(p, 2) && (starts_insn(p[0]) || starts_insn(bytes[0]))) {
        // jmp $ while the rest goes in: a thread arriving waits on it.
        static constexpr std::uint8_t kSelf[2] = {0xeb, 0xfe};
        store_in_word(p, kSelf, 2);
        store_words(p + 2, bytes + 2, n - 2);
        store_in_word(p, bytes, 2);
    } else {
        store_words(p, bytes, n);
    }
    if (s.code && !guest_protect_rx(g_mem, lo, hi - lo)) {
        host_log("patch: cannot restore the protection of 0x%llx", static_cast<unsigned long long>(s.address));
        std::abort();
    }
    return true;
}

// Why a manifest may not change while the game runs, or "" when it may.
std::string held_locked(const Loaded& l) {
    if (!g_mem) return "the game's code is not loaded";
    if (!l.m.live) return "it takes effect as the game starts";
    if (!l.m.option.empty() && !patch_manifest_option_known(l.m.option)) return "it follows " + l.m.option + ", which this bbhost does not know";
    if (g_deferred_to.count(l.m.name)) return "bbhost's frame-rate conversion left its values to it at the start";
    return {};
}

Loaded* find_locked(const std::string& name) {
    for (Loaded& l : g_loaded) {
        if (l.m.name == name) return &l;
    }
    return nullptr;
}

bool take_out_locked(Loaded& l, std::string* why);

std::string join(const std::set<std::string>& names) {
    std::string out;
    for (const std::string& n : names) out += (out.empty() ? "" : ", ") + n;
    return out;
}

// The byte the applied manifests want at `a`, as they apply at the start: the
// last in order with a site over it that does not step aside, else the
// eboot's own. `who` is that manifest (null: the eboot's).
std::uint8_t layered_byte(std::uint64_t a, std::uint8_t now, const std::set<std::string>& on, const Loaded** who) {
    std::uint8_t b = original_byte(a, now);
    *who = nullptr;
    for (const Loaded& o : g_loaded) {
        if (!o.applied) continue;
        for (const Site& s : o.m.sites) {
            if (a < s.address || a >= s.address + s.bytes.size() || yielded(s, on)) continue;
            b = s.bytes[a - s.address];
            *who = &o;
        }
    }
    return b;
}

// A site's bytes made what the applied manifests want there now, and the
// owners noted: a manifest put in under a later one leaves the later one's
// bytes, and one taken out hands its bytes to the next under it, or the
// eboot's back. `keep_cave`: new code where the eboot had zeros stays (a
// thread may still be inside it). False when its page cannot be written.
bool relayer_locked(const Site& s, bool keep_cave, bool* wrote, const std::string& self, std::set<std::string>* others) {
    std::set<std::string> on;
    for (const Loaded& o : g_loaded) {
        if (o.applied) on.insert(o.m.name);
    }
    GuestMemory& mem = *g_mem;
    auto* p = static_cast<std::uint8_t*>(guest_ptr(mem, mem.slide + (s.address - kPreferredGuestSlide)));
    const std::size_t n = s.bytes.size();
    std::vector<std::uint8_t> want(p, p + n), original(n);
    std::vector<const Loaded*> owner(n, nullptr);
    for (std::size_t i = 0; i < n; ++i) {
        original[i] = original_byte(s.address + i, p[i]);
        want[i] = layered_byte(s.address + i, p[i], on, &owner[i]);
        if (keep_cave && cave(s) && !owner[i]) want[i] = p[i];
    }
    *wrote = std::memcmp(p, want.data(), n) != 0;
    if (*wrote && !write_live(s, p, want.data())) return false;
    for (std::size_t i = 0; i < n; ++i) {
        if (owner[i] && owner[i]->m.name != self) others->insert(owner[i]->m.name);
        if (owner[i] && want[i] != original[i]) {
            g_written[s.address + i] = Written{owner[i]->m.name, original[i]};
        } else {
            g_written.erase(s.address + i);
        }
    }
    return true;
}

// The manifest's bytes in, checked first as patch_manifests_apply checks them.
bool put_in_locked(Loaded& l, std::string* why) {
    if (l.applied) return true;
    if (std::string h = held_locked(l); !h.empty()) {
        *why = h;
        return false;
    }
    if (l.m.name == "old-hunters" && !old_hunters_missing().empty()) {
        *why = "the game folder lacks The Old Hunters' files";
        return false;
    }
    // One of a group: the one on goes first.
    if (!l.m.group.empty()) {
        for (Loaded& o : g_loaded) {
            if (&o == &l || !o.applied || o.m.group != l.m.group) continue;
            std::string w;
            if (!take_out_locked(o, &w)) {
                *why = (o.m.title.empty() ? o.m.name : o.m.title) + ", another of its kind, cannot be taken out now (" + w + ")";
                return false;
            }
        }
    }
    std::set<std::string> on;
    for (const Loaded& o : g_loaded) {
        if (o.applied) on.insert(o.m.name);
    }
    on.insert(l.m.name);
    GuestMemory& mem = *g_mem;
    for (const Site& s : l.m.sites) {
        if (yielded(s, on)) continue;
        const std::uint64_t off = s.address - kPreferredGuestSlide;
        if (s.address < kPreferredGuestSlide || off >= mem.size || s.expect.size() > mem.size - off) {
            *why = "a site is outside the image";
            return false;
        }
        const auto* p = static_cast<const std::uint8_t*>(guest_ptr(mem, mem.slide + off));
        for (std::size_t i = 0; i < s.expect.size(); ++i) {
            if (p[i] == s.expect[i] || p[i] == s.bytes[i] || original_byte(s.address + i, p[i]) == s.expect[i]) continue;
            char at[32];
            std::snprintf(at, sizeof(at), "0x%llx", static_cast<unsigned long long>(s.address + i));
            *why = std::string("the game's code at ") + at + " is not what it expects (bbhost or another patch changed it)";
            return false;
        }
    }
    // The new code first, the jumps to it after.
    l.applied = true;
    std::vector<const Site*> order;
    for (const Site& s : l.m.sites) order.push_back(&s);
    std::stable_partition(order.begin(), order.end(), [](const Site* s) { return cave(*s); });
    int written = 0;
    std::set<std::string> others;
    for (const Site* s : order) {
        bool wrote = false;
        if (!relayer_locked(*s, false, &wrote, l.m.name, &others)) {
            *why = "a page could not be made writable";
            take_out_locked(l, why);
            return false;
        }
        written += wrote;
    }
    host_log("patch: %s: on while the game runs, %d of %zu sites written%s%s", l.m.name.c_str(), written, l.m.sites.size(),
             others.empty() ? "" : " (bytes shared with ", others.empty() ? "" : (join(others) + ", layered in the list's order)").c_str());
    return true;
}

// Its bytes out: what the applied manifests under it want there, or the
// eboot's own.
bool take_out_locked(Loaded& l, std::string* why) {
    if (!l.applied) return true;
    if (std::string h = held_locked(l); !h.empty()) {
        *why = h;
        return false;
    }
    GuestMemory& mem = *g_mem;
    // Its bytes must still be its own: one the game or bbhost changed since
    // means a writer that would find the eboot's put back under it.
    for (const Site& s : l.m.sites) {
        const auto* p = static_cast<const std::uint8_t*>(guest_ptr(mem, mem.slide + (s.address - kPreferredGuestSlide)));
        for (std::size_t i = 0; i < s.bytes.size(); ++i) {
            const auto it = g_written.find(s.address + i);
            if (it == g_written.end() || it->second.owner != l.m.name || p[i] == s.bytes[i]) continue;
            char at[32];
            std::snprintf(at, sizeof(at), "0x%llx", static_cast<unsigned long long>(s.address + i));
            *why = std::string("its bytes at ") + at + " were changed after it wrote them";
            return false;
        }
    }
    // The jumps first; new code where the eboot had zeros stays.
    l.applied = false;
    std::vector<const Site*> order;
    for (const Site& s : l.m.sites) order.push_back(&s);
    std::stable_partition(order.begin(), order.end(), [](const Site* s) { return !cave(*s); });
    int restored = 0;
    std::set<std::string> others;
    for (const Site* s : order) {
        bool wrote = false;
        if (!relayer_locked(*s, true, &wrote, l.m.name, &others)) {
            *why = "a page could not be made writable";
            l.applied = true;
            return false;
        }
        restored += wrote;
    }
    host_log("patch: %s: off while the game runs, %d of %zu sites put back%s%s", l.m.name.c_str(), restored, l.m.sites.size(),
             others.empty() ? "" : " (bytes shared with ", others.empty() ? "" : (join(others) + ", layered in the list's order)").c_str());
    return true;
}

}  // namespace

PatchLiveState patch_manifest_live_state(const std::string& name) {
    std::lock_guard<std::mutex> lk(g_live_mu);
    PatchLiveState st;
    const Loaded* l = find_locked(name);
    if (!l) return st;
    st.known = true;
    st.applied = l->applied;
    st.why = held_locked(*l);
    st.can_change = st.why.empty();
    return st;
}

bool patch_manifest_set_live(const std::string& name, bool on, std::string* why) {
    std::lock_guard<std::mutex> lk(g_live_mu);
    Loaded* l = find_locked(name);
    std::string w;
    if (!l) w = "it was not read for this eboot at the start";
    const bool ok = l && (on ? put_in_locked(*l, &w) : take_out_locked(*l, &w));
    if (!ok) host_log("patch: %s: %s at the next start - %s", name.c_str(), on ? "on" : "off", w.c_str());
    if (why) *why = w;
    return ok;
}
