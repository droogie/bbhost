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
    std::string path, name, eboot, description, option, group;
    bool enabled = true;
    bool gameplay = false;  // [patch] gameplay: changes the game (the session's online rules say so)
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
std::map<std::uint64_t, Written> g_written;

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
// tree's, the mods directory's.
std::vector<std::string> manifest_dirs(const std::string& mods) {
    std::vector<std::string> dirs = {config_exe_dir() + "/patches", config_exe_dir() + "/../patches"};
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
            l.description = m.description;
            l.option = m.option;
            l.eboot = m.eboot;
            l.enabled = m.enabled;
            l.gameplay = m.gameplay;
            l.group = m.group;
            l.order = m.order;
        }
        out.push_back(l);
    }
    return out;
}

const char* patch_manifest_owner(std::uint64_t address, std::size_t n) {
    const auto it = g_written.lower_bound(address);
    return it != g_written.end() && it->first < address + n ? it->second.owner.c_str() : nullptr;
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
        host_log("patch: no manifests (looked in %s, %s%s)", dirs[0].c_str(), dirs[1].c_str(),
                 dirs.size() > 2 ? ", the mods directory" : "");
        return;
    }
    // Which are on, first: a site may step aside for another patch (yields).
    std::vector<Manifest> chosen;
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
    // In their list's order; one of a group (variants of one patch: Kyo's
    // analog run ranges, Performance Patch v3's two) - the setup window turns
    // the others off, and a bbhost.toml with two on keeps the later.
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
    for (const Manifest& m : apply_list) {
        if (apply(image, m, on) && m.gameplay) gameplay.push_back(m.name);
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
    std::lock_guard<std::mutex> lk(g_gameplay_mu);
    g_gameplay = std::move(gameplay);
}
