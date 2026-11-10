#include "hle/fs.h"
#include "core/config.h"
#include "core/write_watch.h"
#include "hle/common.h"
#include "host/frame_stats.h"
#include "core/portable.h"
#include "hle/platform.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <atomic>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef EISDIR
#define EISDIR 21
#endif
#ifndef ENOTDIR
#define ENOTDIR 20
#endif

#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#include <fcntl.h>
#else
#include <dirent.h>
#include <strings.h>
#include <fcntl.h>
#include <limits.h>
#include <unistd.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

namespace {

constexpr int kOrbisOCreat = 0x0200;
constexpr int kOrbisOTrunc = 0x0400;
constexpr int kOrbisOExcl = 0x0800;
constexpr int kOrbisOAppend = 0x0008;
constexpr int kOrbisODir = 0x20000;

// Host directories the guest may touch, canonical form. Set by
// hle_fs_set_roots from the config; nothing outside them is reachable.
std::vector<std::string> g_allowed_roots;

std::mutex g_fs_mu;
std::string g_app0;
// The game's update kept in a folder of its own beside the game folder
// (core/config.h, config_game_folders): its files are read in place of the
// game folder's, as the PS4 applies an update. Empty without one.
std::string g_update;
std::atomic<int> g_update_hits{0};
std::string g_data;
std::string g_tmp;
// The port's asset overlay: a directory that resolves *ahead* of the dump, so
// a modified file ships beside `dvdroot_ps4` instead of being written into it.
// The dump is read-only on purpose - it is the thing every run is diffed
// against - and a modification set kept on its own is reviewable on its own.
// Empty when there is no such directory.
std::string g_mods;
// The host's generated overlays are written at start, before they are
// mounted, and nothing writes them after (engine/menu_assets.h, and the PC
// enhancements' engine/change_appearance.h, rebirth.h, five_players.h). So
// each is listed once as it is mounted, and a path it does not have passes it
// by without touching the disk. Probing one cost a stat per path component
// from the data folder down until one was missing, and one more for the file:
// on Windows ~11 us a stat of a directory that is there and ~40 us of a path
// that is not, so ~0.1 ms a layer on every /app0 open and stat - and with the
// three PC enhancements on there are four such layers (the menus' and one
// each), ~0.4 ms on every file the game asked for, its main loop's included,
// for a handful of files in three maps. A path the index has is resolved on
// disk exactly as before. BBHOST_FS_OVERLAY_INDEX=0: every layer probed on
// disk for every path, as before.
struct OverlayIndex {
    bool valid = false;                     // listed; false: probe the disk
    std::unordered_set<std::string> paths;  // files and directories: lowercase, '/'-separated, from the layer
};
const bool g_overlay_index_on = [] {
    const char* e = std::getenv("BBHOST_FS_OVERLAY_INDEX");
    return !(e && e[0] == '0');
}();
std::atomic<std::uint64_t> g_app0_lookups{0}, g_overlay_probes{0}, g_overlay_skips{0};
std::string g_generated;  // hle_fs_set_generated_root
OverlayIndex g_generated_index;
// hle_fs_set_patched_root: tried before the player's mods.
std::string g_patched;
OverlayIndex g_patched_index;
// hle_fs_add_generated_root: fixed slots published by the count, as the
// plugins' layers are.
constexpr int kMaxGeneratedLayers = 4;
std::array<std::string, kMaxGeneratedLayers> g_generated_more;
std::array<OverlayIndex, kMaxGeneratedLayers> g_generated_more_index;
std::atomic<int> g_generated_more_count{0};
// The plugins' overlays (hle_fs_add_plugin_overlay), between mods and
// generated. A plugin's first write can come while the game's file threads
// read - a co-op join on the network thread, a re-seed on the main one - so
// the layers sit in fixed slots, published by the count: a reader never
// locks, and what it holds never moves.
constexpr int kMaxPluginLayers = 32;
std::array<std::string, kMaxPluginLayers> g_plugin_layers;
std::atomic<int> g_plugin_layer_count{0};
std::mutex g_plugin_layer_mu;  // writers

// The overlays in the order a read tries them: the player's mods, the
// plugins' (in the order they were added), then the host's generated files.
// The player's and the plugins' are read from disk every time (a plugin
// writes its own while the game runs); the generated ones carry their index.
struct OverlayLayer {
    const std::string* dir;
    const OverlayIndex* index;  // null: no index, probe the disk
};
thread_local bool t_without_plugins = false;  // hle_fs_map_path_base
std::vector<OverlayLayer> overlay_layers() {
    std::vector<OverlayLayer> v;
    if (!g_patched.empty()) v.push_back({&g_patched, &g_patched_index});  // a mod's file with the port's text added
    v.push_back({&g_mods, nullptr});
    if (!t_without_plugins) {
        const int n = g_plugin_layer_count.load(std::memory_order_acquire);
        for (int i = 0; i < n; ++i) v.push_back({&g_plugin_layers[static_cast<std::size_t>(i)], nullptr});
    }
    v.push_back({&g_generated, &g_generated_index});
    const int n = g_generated_more_count.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
        v.push_back({&g_generated_more[static_cast<std::size_t>(i)], &g_generated_more_index[static_cast<std::size_t>(i)]});
    }
    return v;
}

// The index's key for a path relative to a layer: lowercase (the game's file
// systems ignore case, and the generated files' names are lowercase),
// '/'-separated, without empty or "." components. False when it has a ".."
// - the disk decides those.
bool overlay_key(const std::string& rest, std::string* key) {
    key->clear();
    std::size_t pos = 0;
    while (pos <= rest.size()) {
        std::size_t end = rest.find_first_of("/\\", pos);
        if (end == std::string::npos) end = rest.size();
        const std::size_t len = end - pos;
        if (len == 2 && rest.compare(pos, 2, "..") == 0) return false;
        if (len && !(len == 1 && rest[pos] == '.')) {
            if (!key->empty()) key->push_back('/');
            for (std::size_t i = pos; i < end; ++i) {
                key->push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(rest[i]))));
            }
        }
        pos = end + 1;
    }
    return !key->empty();
}
std::atomic<int> g_mod_hits{0};

struct HostFile {
    int host_fd = -1;
    std::string host_path;
    bool is_dir = false;
    std::vector<std::pair<std::string, uint8_t>> dents;
    std::size_t dent_pos = 0;
};
std::unordered_map<int, HostFile> g_files;
int g_fd_next = 3;

bool is_dir_path(const char* p) {
#if defined(_WIN32)
    struct _stat64 st{};
    return _stat64(p, &st) == 0 && (st.st_mode & _S_IFDIR);
#else
    struct stat st{};
    return ::stat(p, &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

bool path_exists(const char* p) {
#if defined(_WIN32)
    struct _stat64 st{};
    return _stat64(p, &st) == 0;
#else
    struct stat st{};
    return ::stat(p, &st) == 0;
#endif
}

std::string join_path(std::string a, std::string b) {
    while (!a.empty() && (a.back() == '/' || a.back() == '\\')) {
        a.pop_back();
    }
    while (!b.empty() && (b.front() == '/' || b.front() == '\\')) {
        b.erase(b.begin());
    }
    if (b.empty()) {
        return a;
    }
    if (a.empty()) {
        return b;
    }
    return a + "/" + b;
}

void slashify(std::string& p) {
    for (char& c : p) {
        if (c == '\\') {
            c = '/';
        }
    }
}

bool has_dotdot(const std::string& p) {
    std::string cur;
    for (std::size_t i = 0; i <= p.size(); ++i) {
        if (i == p.size() || p[i] == '/' || p[i] == '\\') {
            if (cur == "..") {
                return true;
            }
            cur.clear();
        } else {
            cur.push_back(p[i]);
        }
    }
    return false;
}

bool guest_prefix(const std::string& g, const char* pre) {
    const std::size_t n = std::strlen(pre);
    if (g.size() < n || g.compare(0, n, pre) != 0) {
        return false;
    }
    return g.size() == n || g[n] == '/';
}

bool prefix_dir(const std::string& path, const char* root) {
    const std::size_t n = std::strlen(root);
    if (path.size() < n || path.compare(0, n, root) != 0) {
        return false;
    }
    return path.size() == n || path[n] == '/';
}

bool under_allowed(const std::string& path) {
    for (const auto& r : g_allowed_roots) {
        if (prefix_dir(path, r.c_str())) {
            return true;
        }
    }
    const int n = g_plugin_layer_count.load(std::memory_order_acquire);   // the plugins' overlays
    for (int i = 0; i < n; ++i) {
        if (prefix_dir(path, g_plugin_layers[static_cast<std::size_t>(i)].c_str())) return true;
    }
    return false;
}

// Windows: a drive letter ("Z:/...") is an absolute path and stays the
// first component of the normalized form ("Z:/home/..."), with no leading
// slash; a rooted path without a drive gets the current drive.
bool has_drive(const std::string& p) {
#if defined(_WIN32)
    return p.size() >= 2 && std::isalpha(static_cast<unsigned char>(p[0])) && p[1] == ':';
#else
    (void)p;
    return false;
#endif
}

std::string lex_normalize(std::string p) {
    slashify(p);
    if (p.empty()) {
        return {};
    }
    std::string drive;
    if (has_drive(p)) {
        drive = p.substr(0, 2);
        p = p.size() > 2 ? p.substr(2) : "/";
        if (p[0] != '/') p = "/" + p;
    }
    if (p[0] != '/') {
        char cwd[PATH_MAX];
#if defined(_WIN32)
        if (!_getcwd(cwd, sizeof(cwd))) {
            return {};
        }
        for (char* c = cwd; *c; ++c) {
            if (*c == '\\') {
                *c = '/';
            }
        }
#else
        if (!getcwd(cwd, sizeof(cwd))) {
            return {};
        }
#endif
        p = join_path(cwd, p);
        if (has_drive(p)) {
            drive = p.substr(0, 2);
            p = p.substr(2);
        }
    }
    std::vector<std::string> parts;
    std::string cur;
    for (std::size_t i = 0; i <= p.size(); ++i) {
        if (i == p.size() || p[i] == '/') {
            if (cur.empty() || cur == ".") {
            } else if (cur == "..") {
                if (!parts.empty()) {
                    parts.pop_back();
                }
            } else {
                parts.push_back(cur);
            }
            cur.clear();
        } else {
            cur.push_back(p[i]);
        }
    }
    std::string out = drive;
    for (const auto& s : parts) {
        out.push_back('/');
        out += s;
    }
    return out.size() == drive.size() ? drive + "/" : out;
}

void log_reject(const std::string& path) {
    static int logs;
    if (logs < 16) {
        host_log("FS reject %s", path.c_str());
        ++logs;
    }
}

std::vector<std::string> split_parts(const std::string& n) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : n) {
        if (c == '/') {
            if (!cur.empty()) {
                parts.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        parts.push_back(cur);
    }
    return parts;
}

const char* allowed_root_of(const std::string& n) {
    for (const auto& r : g_allowed_roots) {
        if (prefix_dir(n, r.c_str())) {
            return r.c_str();
        }
    }
    const int count = g_plugin_layer_count.load(std::memory_order_acquire);   // the plugins' overlays
    for (int i = 0; i < count; ++i) {
        const std::string& r = g_plugin_layers[static_cast<std::size_t>(i)];
        if (prefix_dir(n, r.c_str())) return r.c_str();
    }
    return nullptr;
}

std::string resolve_confined(const std::string& n) {
    const char* root = allowed_root_of(n);
    if (!root) {
        log_reject(n);
        return {};
    }
    const std::size_t root_len = std::strlen(root);
    const auto parts = split_parts(n.size() == root_len ? std::string() : n.substr(root_len));
    std::string acc = root;
#if !defined(_WIN32)
    {
        struct stat rst{};
        if (lstat(acc.c_str(), &rst) == 0 && S_ISLNK(rst.st_mode)) {
            char t[PATH_MAX];
            const ssize_t k = ::readlink(acc.c_str(), t, sizeof(t) - 1);
            if (k < 0) {
                log_reject(acc);
                return {};
            }
            t[k] = 0;
            std::string tgt = t;
            slashify(tgt);
            const std::string resolved = lex_normalize(tgt[0] == '/' ? tgt : join_path(acc, tgt));
            if (!under_allowed(resolved)) {
                log_reject(std::string(root) + " -> " + resolved);
                return {};
            }
            acc = resolved;
        }
    }
#endif
    for (std::size_t i = 0; i < parts.size(); ++i) {
        std::string next = lex_normalize(join_path(acc, parts[i]));
        if (!under_allowed(next)) {
            log_reject(next);
            return {};
        }
#if defined(_WIN32)
        struct _stat64 st{};
        if (_stat64(next.c_str(), &st) != 0) {
            for (std::size_t j = i + 1; j < parts.size(); ++j) {
                next = join_path(next, parts[j]);
            }
            return lex_normalize(next);
        }
        acc = next;
#else
        struct stat st{};
        if (lstat(next.c_str(), &st) != 0) {
            for (std::size_t j = i + 1; j < parts.size(); ++j) {
                next = join_path(next, parts[j]);
            }
            return lex_normalize(next);
        }
        if (S_ISLNK(st.st_mode)) {
            char t[PATH_MAX];
            const ssize_t k = ::readlink(next.c_str(), t, sizeof(t) - 1);
            if (k < 0) {
                log_reject(next);
                return {};
            }
            t[k] = 0;
            std::string tgt = t;
            slashify(tgt);
            const std::string resolved =
                lex_normalize(tgt.empty() ? acc : (tgt[0] == '/' ? tgt : join_path(acc, tgt)));
            if (!under_allowed(resolved)) {
                log_reject(next + " -> " + resolved);
                return {};
            }
            acc = resolved;
        } else {
            acc = next;
        }
#endif
    }
    return acc;
}

std::string confine(const std::string& path) {
    const std::string n = lex_normalize(path);
    if (n.empty() || !under_allowed(n)) {
        log_reject(path);
        return {};
    }
    return resolve_confined(n);
}

// The PS4's file systems ignore case, and the game lowercases what it asks for
// ("adhoc:/font/DbgFont14h.ccm" arrives as .../adhoc/font/dbgfont14h.ccm). The
// dump is on a Linux file system that does not, so a file whose name has
// capitals - DbgFont14h.ccm, the FontShader directory, anything a player adds -
// was simply not found. When the exact path is missing, this resolves it one
// component at a time, ignoring case, the way the console would. Only for the
// dump and the overlay, which nothing writes, so the answers are cached, the
// misses as well.
std::string resolve_nocase(const std::string& root, const std::string& rest) {
#if defined(_WIN32)
    (void)root;
    (void)rest;
    return {};
#else
    static std::mutex mu;
    static std::unordered_map<std::string, std::string> cache;
    const std::string key = root + '\0' + rest;
    {
        std::lock_guard<std::mutex> lock(mu);
        if (auto it = cache.find(key); it != cache.end()) return it->second;
    }
    std::string cur = root;
    std::size_t pos = 0;
    while (pos <= rest.size()) {
        std::size_t end = rest.find('/', pos);
        if (end == std::string::npos) end = rest.size();
        const std::string comp = rest.substr(pos, end - pos);
        pos = end + 1;
        if (comp.empty()) continue;
        std::string next = join_path(cur, comp);
        if (!path_exists(next.c_str())) {
            next.clear();
            if (DIR* d = opendir(cur.c_str())) {
                while (dirent* e = readdir(d)) {
                    if (strcasecmp(e->d_name, comp.c_str()) == 0) {
                        next = join_path(cur, e->d_name);
                        break;
                    }
                }
                closedir(d);
            }
            if (next.empty()) {
                cur.clear();
                break;
            }
        }
        cur = next;
    }
    std::lock_guard<std::mutex> lock(mu);
    cache.emplace(key, cur);
    return cur;
#endif
}

std::atomic<int> g_nocase_hits{0};

// Dynamic mounts (save data): guest prefix -> host directory under g_data.
std::vector<std::pair<std::string, std::string>> g_mounts;

void mkdir_confined(const std::string& path);

std::string guest_to_host(const char* guest) {
    if (!guest || !guest[0]) {
        return {};
    }
    std::string g = guest;
    slashify(g);
    std::string rest;
    std::string root = g_app0;
    bool mounted = false;
    {
        std::lock_guard<std::mutex> lock(g_fs_mu);
        for (const auto& m : g_mounts) {
            if (guest_prefix(g, m.first.c_str())) {
                root = m.second;
                rest = g.size() > m.first.size() ? g.substr(m.first.size() + 1) : "";
                mounted = true;
                break;
            }
        }
    }
    if (mounted) {
        // fall through to the checks below
    } else if (guest_prefix(g, "/app0")) {
        rest = g.size() > 5 ? g.substr(g[5] == '/' ? 6 : 5) : "";
        // The game names some of /app0 through its dvdroot: "capture:" is
        // "/app0/dvdroot_ps4/../capture". A ".." that stays inside /app0 is
        // resolved here; one that would leave it is still refused below.
        if (has_dotdot(rest)) {
            std::vector<std::string> parts;
            std::string cur;
            bool escapes = false;
            for (std::size_t i = 0; i <= rest.size(); ++i) {
                if (i == rest.size() || rest[i] == '/') {
                    if (cur == "..") {
                        if (parts.empty()) escapes = true;
                        else parts.pop_back();
                    } else if (!cur.empty() && cur != ".") {
                        parts.push_back(cur);
                    }
                    cur.clear();
                } else {
                    cur.push_back(rest[i]);
                }
            }
            if (!escapes) {
                rest.clear();
                for (const auto& s : parts) rest += (rest.empty() ? "" : "/") + s;
            }
        }
        // capture: is the dev kit's capture drive, where the debug menu keeps
        // its window layout (debugmenulayoutlist.xmllist, "SaveCurrentLayout")
        // and its bookmarks (bookmark.txt). It is written, so it lives in the
        // data directory, not the dump. Without it every window opened at its
        // built-in size - the sub-window a 200x200 box its lines overran.
        if (guest_prefix("/" + rest, "/capture") && !g_data.empty()) {
            root = join_path(g_data, "capture");
            rest = rest.size() > 7 ? rest.substr(8) : "";
            mounted = true;
            static std::once_flag made;
            std::call_once(made, [&] {
                mkdir_confined(root);
                mkdir_confined(join_path(root, "debugmenu"));
            });
        }
    } else if (guest_prefix(g, "/data")) {
        root = g_data;
        rest = g.size() > 5 ? g.substr(g[5] == '/' ? 6 : 5) : "";
    } else if (guest_prefix(g, "/temp")) {
        root = g_tmp;
        rest = g.size() > 5 ? g.substr(g[5] == '/' ? 6 : 5) : "";
    } else if (guest_prefix(g, "/tmp")) {
        root = g_tmp;
        rest = g.size() > 4 ? g.substr(g[4] == '/' ? 5 : 4) : "";
    } else if (g[0] != '/') {
        rest = g;
    } else {
        log_reject(g);
        return {};
    }
    if (root.empty() || has_dotdot(rest)) {
        log_reject(g);
        return {};
    }
    // The overlay shadows the dump and only the dump: a save or a temp file
    // still goes where it belongs. A path that is not in the overlay falls
    // through untouched, so a run without one behaves exactly as before.
    // A generated layer's index answers for it when it does not have the path
    // (OverlayIndex): no disk access for the layers the PC enhancements add.
    const bool app0_lookup = !mounted && root == g_app0 && !rest.empty();
    if (app0_lookup) g_app0_lookups.fetch_add(1, std::memory_order_relaxed);
    std::string key;
    int key_state = 0;  // 0 not made yet, 1 made, -1 none (the disk decides)
    for (const OverlayLayer& layer_of : overlay_layers()) {
        if (!app0_lookup) break;
        const std::string* layer = layer_of.dir;
        if (layer->empty()) continue;
        if (g_overlay_index_on && layer_of.index && layer_of.index->valid) {
            if (key_state == 0) key_state = overlay_key(rest, &key) ? 1 : -1;
            if (key_state == 1 && layer_of.index->paths.count(key) == 0) {
                g_overlay_skips.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
        }
        if (layer_of.index) g_overlay_probes.fetch_add(1, std::memory_order_relaxed);
        std::string over = confine(join_path(*layer, rest));
        if (!over.empty() && !path_exists(over.c_str())) {
            const std::string ci = resolve_nocase(*layer, rest);
            over = ci.empty() ? std::string() : confine(ci);
        }
        if (!over.empty() && path_exists(over.c_str())) {
            const int n = g_mod_hits.fetch_add(1);
            static const char* traced = std::getenv("BBHOST_FS_TRACE");
            if (n < 32 || (traced && *traced && g.find(traced) != std::string::npos)) {
                host_log("FS overlay: %s -> %s", g.c_str(), over.c_str());
            }
            return over;
        }
    }
    // BBHOST_FS_TRACE=<substring>: every guest path containing it, whether or
    // not the overlay had a file. Without this, a path the overlay *misses* is
    // invisible - and "the overlay served nothing" and "the game never asked"
    // look identical in the log, which cost a run.
    static const char* trace = std::getenv("BBHOST_FS_TRACE");
    if (trace && *trace && g.find(trace) != std::string::npos) {
        static int logs = 0;
        if (logs < 40) {
            ++logs;
            host_log("FS trace: %s (no overlay file)", g.c_str());
        }
    }
    // Then the update's file, in place of the game folder's. A directory the
    // game folder has as well stays the game folder's: a listing of it is
    // the game's, and the files in it still come from the update one by one.
    if (!mounted && root == g_app0 && !rest.empty() && !g_update.empty()) {
        std::string up = confine(join_path(g_update, rest));
        if (!up.empty() && !path_exists(up.c_str())) {
            const std::string ci = resolve_nocase(g_update, rest);
            up = ci.empty() ? std::string() : confine(ci);
        }
        if (!up.empty() && path_exists(up.c_str()) &&
            !(is_dir_path(up.c_str()) && is_dir_path(join_path(g_app0, rest).c_str()))) {
            if (g_update_hits.fetch_add(1) < 8) host_log("FS update: %s -> %s", g.c_str(), up.c_str());
            return up;
        }
    }
    std::string host = confine(join_path(root, rest));
    if (!mounted && root == g_app0 && !rest.empty() && !host.empty() && !path_exists(host.c_str())) {
        const std::string ci = resolve_nocase(root, rest);
        if (!ci.empty()) {
            const std::string confined = confine(ci);
            if (!confined.empty()) {
                if (g_nocase_hits.fetch_add(1) < 16) {
                    host_log("FS: %s found ignoring case: %s", g.c_str(), confined.c_str());
                }
                return confined;
            }
        }
    }
    return host;
}

std::string dirname_of(const char* path) {
    std::string p = path ? path : "";
    auto slash = p.find_last_of("/\\");
    if (slash == std::string::npos) {
        return ".";
    }
    if (slash == 0) {
        return "/";
    }
    return p.substr(0, slash);
}

void mkdir_confined(const std::string& path) {
    if (path.empty()) {
        return;
    }
#if defined(_WIN32)
    _mkdir(path.c_str());
#else
    ::mkdir(path.c_str(), 0777);
#endif
}

std::string canonical_dir(const std::string& p) {
    if (p.empty()) {
        return {};
    }
#if defined(_WIN32)
    char buf[4096];
    if (!_fullpath(buf, p.c_str(), sizeof(buf))) {
        return {};
    }
    std::string out = buf;
#else
    char buf[PATH_MAX];
    if (!realpath(p.c_str(), buf)) {
        return {};
    }
    std::string out = buf;
#endif
    slashify(out);
    while (out.size() > 1 && out.back() == '/') {
        out.pop_back();
    }
    return out;
}

void set_roots(const char* app0, const char* data, const char* tmp, const char* eboot,
               const char* mods) {
    std::string work = dirname_of(eboot ? eboot : ".");
    std::string data_dir = data ? data : join_path(work, "data");
    std::string tmp_dir = tmp ? tmp : join_path(work, "tmp");
    // paths.mods, or <data>/mods. Not created: an empty overlay directory and
    // no overlay directory should behave the same, and making one silently
    // would hide a mistyped path.
    std::string mods_dir = mods && mods[0] ? std::string(mods) : join_path(data_dir, "mods");
    mkdir_confined(data_dir);
    mkdir_confined(tmp_dir);
    // The game folder and, beside it, the update's (app0 may name either).
    const GameFolders game = config_game_folders(app0 ? app0 : "");
    g_allowed_roots.clear();
    for (const std::string& r : {game.base, game.update, data_dir, tmp_dir, mods_dir}) {
        std::string c = canonical_dir(r);
        if (!c.empty()) {
            g_allowed_roots.push_back(c);
        }
    }
    g_app0 = canonical_dir(game.base);
    g_update = canonical_dir(game.update);
    g_data = canonical_dir(data_dir);
    g_tmp = canonical_dir(tmp_dir);
    g_mods = is_dir_path(mods_dir.c_str()) ? canonical_dir(mods_dir) : std::string();
    host_log("FS app0=%s data=%s tmp=%s", g_app0.c_str(), g_data.c_str(), g_tmp.c_str());
    if (!g_update.empty()) {
        host_log("FS update=%s (the game's update: its files are read in place of the game folder's)", g_update.c_str());
    }
    if (!g_mods.empty()) {
        host_log("FS overlay=%s (shadows /app0)", g_mods.c_str());
    }
    if (g_app0.empty() || !is_dir_path(join_path(g_app0, "dvdroot_ps4").c_str())) {
        host_log("FS dump missing dvdroot_ps4 at %s", app0 ? app0 : "(unset)");
    }
}

void fill_orbis_stat(void* out, std::uint64_t size, bool is_dir, std::int64_t mtime) {
    unsigned char buf[120]{};
    auto wr16 = [&](int off, std::uint16_t v) { std::memcpy(buf + off, &v, 2); };
    auto wr32 = [&](int off, std::uint32_t v) { std::memcpy(buf + off, &v, 4); };
    auto wr64 = [&](int off, std::int64_t v) { std::memcpy(buf + off, &v, 8); };
    wr32(0, 1);
    wr32(4, 1);
    wr16(8, static_cast<std::uint16_t>(is_dir ? 0x41edu : 0x81a4u));
    wr16(10, 1);
    wr64(24, mtime);
    wr64(40, mtime);
    wr64(56, mtime);
    wr64(72, static_cast<std::int64_t>(size));
    wr64(80, static_cast<std::int64_t>((size + 511) / 512));
    wr32(88, 4096);
    wr64(104, mtime);
    std::memcpy(out, buf, sizeof(buf));
}

int stat_host(const char* host, void* sb) {
    if (!host || !host[0]) {
        return sce_err(ENOENT);
    }
    if (!sb) {
        return sce_err(EINVAL);
    }
#if defined(_WIN32)
    struct _stat64 st{};
    if (_stat64(host, &st) != 0) {
        return sce_err(errno);
    }
    fill_orbis_stat(sb, static_cast<std::uint64_t>(st.st_size), (st.st_mode & _S_IFDIR) != 0,
                    static_cast<std::int64_t>(st.st_mtime));
#else
    struct stat st{};
    if (::stat(host, &st) != 0) {
        return sce_err(errno);
    }
    fill_orbis_stat(sb, static_cast<std::uint64_t>(st.st_size), S_ISDIR(st.st_mode),
                    static_cast<std::int64_t>(st.st_mtime));
#endif
    return 0;
}

int host_open_flags(int orbis) {
#if defined(_WIN32)
    int f = _O_BINARY;
    switch (orbis & 3) {
        case 1:
            f |= _O_WRONLY;
            break;
        case 2:
            f |= _O_RDWR;
            break;
        default:
            f |= _O_RDONLY;
            break;
    }
    if (orbis & kOrbisOCreat) {
        f |= _O_CREAT;
    }
    if (orbis & kOrbisOTrunc) {
        f |= _O_TRUNC;
    }
    if (orbis & kOrbisOExcl) {
        f |= _O_EXCL;
    }
    if (orbis & kOrbisOAppend) {
        f |= _O_APPEND;
    }
    return f;
#else
    int f = orbis & 3;
    if (orbis & kOrbisOCreat) {
        f |= O_CREAT;
    }
    if (orbis & kOrbisOTrunc) {
        f |= O_TRUNC;
    }
    if (orbis & kOrbisOExcl) {
        f |= O_EXCL;
    }
    if (orbis & kOrbisOAppend) {
        f |= O_APPEND;
    }
    if (orbis & 4) {
        f |= O_NONBLOCK;
    }
    if (orbis & kOrbisODir) {
        f |= O_DIRECTORY;
    }
    return f;
#endif
}

void snapshot_dir(HostFile& hf) {
    hf.dents.clear();
    hf.dent_pos = 0;
#if defined(_WIN32)
    std::string pat = hf.host_path + "\\*";
    WIN32_FIND_DATAA fd{};
    HANDLE h = FindFirstFileA(pat.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        uint8_t ty = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? 4 : 8;
        hf.dents.emplace_back(fd.cFileName, ty);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR* d = opendir(hf.host_path.c_str());
    if (!d) {
        return;
    }
    while (dirent* e = readdir(d)) {
        hf.dents.emplace_back(e->d_name, e->d_type);
    }
    closedir(d);
#endif
}

bool fs_trace() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_TRACE_FS");
        return e && e[0] == '1';
    }();
    return on;
}

// Game files that would not open, or opened empty. The game's file loader
// stops on either far from the open (DL_PANIC FileTransferTask.cpp(865): a
// file of size 0), so each is logged (the first 64) and the last few are kept
// for the panic report (hle_fs_problem_files). A game that runs probes a few
// files it does without (menu/logo.tpf.dcx), so one here is a lead, not a
// fault.
std::mutex g_problem_mu;
std::vector<std::string> g_problems;  // newest last, each once
std::set<std::string> g_problem_seen;

void note_problem(const char* guest, const std::string& host, const char* what) {
    if (!guest || std::strncmp(guest, "/app0/", 6) != 0) return;
    std::lock_guard<std::mutex> lk(g_problem_mu);
    const std::string entry = std::string(guest) + " (" + what + ")";
    if (g_problem_seen.insert(entry).second) {
        if (g_problem_seen.size() <= 64) host_log("FS: %s %s (%s)", guest, what, host.empty() ? "no mapping" : host.c_str());
    } else {
        g_problems.erase(std::remove(g_problems.begin(), g_problems.end(), entry), g_problems.end());
    }
    g_problems.push_back(entry);
    if (g_problems.size() > 8) g_problems.erase(g_problems.begin());
}

// The last opens, for crash reports (hle_fs_log_recent_opens): fixed slots,
// written without a lock and read at a crash without one - a torn line in a
// report beats a deadlock in it.
constexpr unsigned kRecentOpens = 16;
char g_recent[kRecentOpens][240];
std::atomic<unsigned> g_recent_next{0};

void note_open(const char* guest, int result, long long size) {
    const unsigned i = g_recent_next.fetch_add(1) % kRecentOpens;
    if (result < 0) {
        std::snprintf(g_recent[i], sizeof(g_recent[i]), "%s -> error 0x%08x", guest ? guest : "?",
                      static_cast<unsigned>(result));
    } else if (size >= 0) {
        std::snprintf(g_recent[i], sizeof(g_recent[i]), "%s -> %lld bytes", guest ? guest : "?", size);
    } else {
        std::snprintf(g_recent[i], sizeof(g_recent[i]), "%s -> opened", guest ? guest : "?");
    }
}

int log_open(const char* path, const std::string& host, int flags, int result, long long size = -1) {
    note_open(path, result, size);
    static int logs;
    if (logs < 24 || fs_trace()) {
        host_log("sceKernelOpen %s -> %s flags=0x%x -> %d", path ? path : "", host.c_str(), flags,
                 result);
        ++logs;
    }
    if (result < 0) note_problem(path, host, "is not in the game folder");
    return result;
}

GUEST_ABI int hle_kernel_open(const char* path, int flags, int mode) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::string host = guest_to_host(path);
    if (host.empty()) {
        return log_open(path, host, flags, sce_err(ENOENT));
    }
    const bool want_dir = (flags & kOrbisODir) != 0;
#if defined(_WIN32)
    struct _stat64 st{};
    const int st_ok = _stat64(host.c_str(), &st);
    const bool is_dir = st_ok == 0 && (st.st_mode & _S_IFDIR) != 0;
#else
    struct stat st{};
    const int st_ok = ::stat(host.c_str(), &st);
    const bool is_dir = st_ok == 0 && S_ISDIR(st.st_mode);
#endif
    if (want_dir) {
        if (st_ok != 0) {
            return log_open(path, host, flags, sce_err(errno));
        }
        if (!is_dir) {
            return log_open(path, host, flags, sce_err(ENOTDIR));
        }
    }
    HostFile hf{};
    hf.host_path = host;
    hf.is_dir = is_dir;
    if (hf.is_dir) {
        snapshot_dir(hf);
    } else {
#if defined(_WIN32)
        int fd = _open(host.c_str(), host_open_flags(flags), mode ? mode : 0666);
#else
        int fd = ::open(host.c_str(), host_open_flags(flags), mode ? mode : 0666);
#endif
        if (fd < 0) {
            return log_open(path, host, flags, sce_err(errno));
        }
        hf.host_fd = fd;
        // No file of a PS4 game is empty: a dump with one is incomplete.
        if (st_ok == 0 && st.st_size == 0) note_problem(path, host, "is empty (0 bytes)");
    }
    const long long size = st_ok == 0 && !is_dir ? static_cast<long long>(st.st_size) : -1;
    std::lock_guard<std::mutex> lock(g_fs_mu);
    int id = g_fd_next++;
    g_files[id] = std::move(hf);
    return log_open(path, host, flags, id, size);
}

HostFile* file_get(int fd) {
    auto it = g_files.find(fd);
    return it == g_files.end() ? nullptr : &it->second;
}

GUEST_ABI int hle_kernel_close(int fd) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::lock_guard<std::mutex> lock(g_fs_mu);
    auto it = g_files.find(fd);
    if (it == g_files.end()) {
        return sce_err(EBADF);
    }
    if (it->second.host_fd >= 0) {
#if defined(_WIN32)
        _close(it->second.host_fd);
#else
        ::close(it->second.host_fd);
#endif
    }
    g_files.erase(it);
    return 0;
}

GUEST_ABI std::int64_t hle_kernel_read(int fd, void* buf, std::uint64_t n) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::lock_guard<std::mutex> lock(g_fs_mu);
    HostFile* f = file_get(fd);
    if (!f) {
        return sce_err(EBADF);
    }
    if (f->is_dir) {
        return sce_err(EISDIR);
    }
    if (!buf && n) {
        return sce_err(EINVAL);
    }
    if (!n) {
        return 0;
    }
    auto* p = static_cast<unsigned char*>(buf);
    std::uint64_t got = 0;
    // The kernel writes this buffer, and a page the texture cache has
    // write-protected (core/write_watch.h) fails the read with EFAULT rather
    // than faulting: release them first, and again if a texture upload
    // protected one in the meantime.
    write_watch_release(p, static_cast<std::size_t>(n));
    int efaults = 0;
    while (got < n) {
#if defined(_WIN32)
        unsigned chunk = static_cast<unsigned>(std::min<std::uint64_t>(n - got, 0x7fffffff));
        int r = _read(f->host_fd, p + got, chunk);
#else
        ssize_t r = ::read(f->host_fd, p + got, static_cast<std::size_t>(n - got));
#endif
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
#if defined(_WIN32)
            // ReadFile into a read-only page fails with ERROR_NOACCESS, which
            // the CRT maps to EINVAL.
            const bool fault = errno == EFAULT || GetLastError() == ERROR_NOACCESS;
#else
            const bool fault = errno == EFAULT;
#endif
            if (fault && efaults++ < 8 && write_watch_release(p + got, static_cast<std::size_t>(n - got))) {
                continue;
            }
            return got ? static_cast<std::int64_t>(got) : sce_err(errno);
        }
        if (r == 0) {
            break;
        }
        got += static_cast<std::uint64_t>(r);
    }
    return static_cast<std::int64_t>(got);
}

GUEST_ABI std::int64_t hle_kernel_write(int fd, const void* buf, std::uint64_t n) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::lock_guard<std::mutex> lock(g_fs_mu);
    HostFile* f = file_get(fd);
    if (!f) {
        return sce_err(EBADF);
    }
    if (f->is_dir) {
        return sce_err(EISDIR);
    }
    if (!n) {
        return 0;
    }
    auto* p = static_cast<const unsigned char*>(buf);
    std::uint64_t put = 0;
    while (put < n) {
#if defined(_WIN32)
        unsigned chunk = static_cast<unsigned>(std::min<std::uint64_t>(n - put, 0x7fffffff));
        int r = _write(f->host_fd, p + put, chunk);
#else
        ssize_t r = ::write(f->host_fd, p + put, static_cast<std::size_t>(n - put));
#endif
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return put ? static_cast<std::int64_t>(put) : sce_err(errno);
        }
        if (r == 0) {
            break;
        }
        put += static_cast<std::uint64_t>(r);
    }
    return static_cast<std::int64_t>(put);
}

GUEST_ABI std::int64_t hle_kernel_lseek(int fd, std::int64_t off, int whence) {
    std::lock_guard<std::mutex> lock(g_fs_mu);
    HostFile* f = file_get(fd);
    if (!f) {
        return sce_err(EBADF);
    }
    if (f->is_dir) {
        if (whence == 0) {
            f->dent_pos = off < 0 ? 0 : static_cast<std::size_t>(off);
        } else if (whence == 1) {
            f->dent_pos = static_cast<std::size_t>(static_cast<std::int64_t>(f->dent_pos) + off);
        } else {
            f->dent_pos = f->dents.size();
        }
        return static_cast<std::int64_t>(f->dent_pos);
    }
#if defined(_WIN32)
    std::int64_t r = _lseeki64(f->host_fd, off, whence);
#else
    std::int64_t r = ::lseek(f->host_fd, static_cast<off_t>(off), whence);
#endif
    if (r < 0) {
        return sce_err(errno);
    }
    return r;
}

struct OrbisDirent {
    std::uint32_t d_fileno;
    std::uint16_t d_reclen;
    std::uint8_t d_type;
    std::uint8_t d_namlen;
    char d_name[256];
};

GUEST_ABI int hle_kernel_getdirentries(int fd, char* buf, int nbytes, std::int64_t* basep) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::lock_guard<std::mutex> lock(g_fs_mu);
    HostFile* f = file_get(fd);
    if (!f) {
        return sce_err(EBADF);
    }
    if (!f->is_dir) {
        return sce_err(ENOTDIR);
    }
    if (!buf || nbytes <= 0) {
        return 0;
    }
    int written = 0;
    while (f->dent_pos < f->dents.size()) {
        const auto& e = f->dents[f->dent_pos];
        std::uint8_t namlen = static_cast<std::uint8_t>(std::min(e.first.size(), static_cast<std::size_t>(255)));
        std::uint16_t reclen = static_cast<std::uint16_t>((8 + namlen + 1 + 3) & ~3);
        if (written + reclen > nbytes) {
            break;
        }
        OrbisDirent d{};
        d.d_fileno = static_cast<std::uint32_t>(f->dent_pos + 1);
        d.d_reclen = reclen;
        d.d_type = e.second;
        d.d_namlen = namlen;
        std::memcpy(d.d_name, e.first.c_str(), namlen);
        std::memcpy(buf + written, &d, reclen);
        written += reclen;
        f->dent_pos += 1;
    }
    if (basep) {
        *basep = static_cast<std::int64_t>(f->dent_pos);
    }
    return written;
}

GUEST_ABI int hle_kernel_stat(const char* path, void* sb) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::string host = guest_to_host(path);
    static int logs;
    if (logs < 16) {
        host_log("sceKernelStat %s -> %s", path ? path : "", host.c_str());
        ++logs;
    }
    if (host.empty()) {
        return sce_err(ENOENT);
    }
    return stat_host(host.c_str(), sb);
}

GUEST_ABI int hle_kernel_fstat(int fd, void* sb) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::lock_guard<std::mutex> lock(g_fs_mu);
    HostFile* f = file_get(fd);
    if (!f) {
        return sce_err(EBADF);
    }
    int r = stat_host(f->host_path.c_str(), sb);
    static int logs;
    if (logs < 12) {
        std::uint64_t sz = 0;
        if (r == 0 && sb) {
            std::memcpy(&sz, static_cast<char*>(sb) + 0x48, 8);
        }
        host_log("sceKernelFstat fd=%d %s size=%llu r=%d", fd, f->host_path.c_str(),
                 static_cast<unsigned long long>(sz), r);
        ++logs;
    }
    return r;
}

GUEST_ABI int hle_kernel_mkdir(const char* path, int mode) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::string host = guest_to_host(path);
    if (host.empty()) {
        return sce_err(ENOENT);
    }
#if defined(_WIN32)
    (void)mode;
    return _mkdir(host.c_str()) == 0 ? 0 : sce_err(errno);
#else
    return ::mkdir(host.c_str(), static_cast<mode_t>(mode ? mode : 0777)) == 0 ? 0 : sce_err(errno);
#endif
}

GUEST_ABI int hle_kernel_rmdir(const char* path) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::string host = guest_to_host(path);
    if (host.empty()) {
        return sce_err(ENOENT);
    }
#if defined(_WIN32)
    return _rmdir(host.c_str()) == 0 ? 0 : sce_err(errno);
#else
    return ::rmdir(host.c_str()) == 0 ? 0 : sce_err(errno);
#endif
}

GUEST_ABI int hle_kernel_unlink(const char* path) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::string host = guest_to_host(path);
    if (host.empty()) {
        return sce_err(ENOENT);
    }
    return std::remove(host.c_str()) == 0 ? 0 : sce_err(errno);
}

GUEST_ABI int hle_kernel_rename(const char* a, const char* b) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::string ha = guest_to_host(a);
    std::string hb = guest_to_host(b);
    if (ha.empty() || hb.empty()) {
        return sce_err(ENOENT);
    }
    return std::rename(ha.c_str(), hb.c_str()) == 0 ? 0 : sce_err(errno);
}

GUEST_ABI int hle_kernel_truncate(const char* path, std::int64_t len) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::string host = guest_to_host(path);
    if (host.empty()) {
        return sce_err(ENOENT);
    }
#if defined(_WIN32)
    int fd = _open(host.c_str(), _O_RDWR | _O_BINARY);
    if (fd < 0) {
        return sce_err(errno);
    }
    int r = _chsize_s(fd, len) == 0 ? 0 : sce_err(errno);
    _close(fd);
    return r;
#else
    return ::truncate(host.c_str(), static_cast<off_t>(len)) == 0 ? 0 : sce_err(errno);
#endif
}

GUEST_ABI int hle_kernel_ftruncate(int fd, std::int64_t len) {
    MainThreadWait timed(2);  // the main loop's file I/O, for frame stats
    std::lock_guard<std::mutex> lock(g_fs_mu);
    HostFile* f = file_get(fd);
    if (!f || f->host_fd < 0) {
        return sce_err(EBADF);
    }
#if defined(_WIN32)
    return _chsize_s(f->host_fd, len) == 0 ? 0 : sce_err(errno);
#else
    return ::ftruncate(f->host_fd, static_cast<off_t>(len)) == 0 ? 0 : sce_err(errno);
#endif
}

}  // namespace

void hle_fs_set_roots(const char* app0, const char* data, const char* tmp, const char* eboot_path,
                      const char* mods) {
    set_roots(app0, data, tmp, eboot_path, mods);
}

const char* hle_fs_mods_root() { return g_mods.c_str(); }
const char* hle_fs_app0_root() { return g_app0.c_str(); }
const char* hle_fs_update_root() { return g_update.c_str(); }
std::string hle_fs_game_file(const std::string& rel) {
    const std::string r = !rel.empty() && rel[0] == '/' ? rel.substr(1) : rel;
    if (!g_update.empty()) {
        const std::string up = join_path(g_update, r);
        if (path_exists(up.c_str())) return up;
    }
    return join_path(g_app0, r);
}
std::string hle_fs_mods_file(const std::string& rel) {
    if (g_mods.empty()) return {};
    const std::string r = !rel.empty() && rel[0] == '/' ? rel.substr(1) : rel;
    const std::string p = join_path(g_mods, r);
    if (path_exists(p.c_str())) return p;
    return resolve_nocase(g_mods, r);  // empty when it is not there
}

namespace {

// A generated layer's files and directories, listed once as it is mounted
// (OverlayIndex). A layer too deep or too large to list (not one the host
// makes) is left without an index, and probed on disk as before.
void index_layer(const std::string& dir, OverlayIndex& idx) {
    idx.paths.clear();
    idx.valid = false;
    if (!g_overlay_index_on) return;
    std::vector<std::pair<std::string, int>> todo{{std::string(), 0}};  // relative directory, depth
    std::vector<std::string> names;
    while (!todo.empty()) {
        const auto [rel, depth] = todo.back();
        todo.pop_back();
        if (depth > 16 || !host_list_dir(join_path(dir, rel).c_str(), &names)) return;
        for (const std::string& name : names) {
            const std::string child = rel.empty() ? name : rel + "/" + name;
            std::string key;
            if (!overlay_key(child, &key)) return;
            idx.paths.insert(key);
            if (idx.paths.size() > 65536) return;
            if (is_dir_path(join_path(dir, child).c_str())) todo.emplace_back(child, depth + 1);
        }
    }
    idx.valid = true;
}

void log_generated_layer(const std::string& d, const OverlayIndex& idx) {
    if (idx.valid) {
        host_log("FS generated overlay=%s (behind the mods overlay; %zu files and folders, indexed: a path it does not have "
                 "passes it by without a disk lookup)",
                 d.c_str(), idx.paths.size());
    } else {
        host_log("FS generated overlay=%s (behind the mods overlay)", d.c_str());
    }
}

}  // namespace

void hle_fs_set_generated_root(const char* dir) {
    g_generated = dir && *dir && is_dir_path(dir) ? canonical_dir(dir) : std::string();
    if (!g_generated.empty()) {
        // Under <data>, which is already an allowed root; listed for confine() all the same.
        g_allowed_roots.push_back(g_generated);
        index_layer(g_generated, g_generated_index);
        log_generated_layer(g_generated, g_generated_index);
    }
}
void hle_fs_set_patched_root(const char* dir) {
    g_patched = dir && *dir && is_dir_path(dir) ? canonical_dir(dir) : std::string();
    if (!g_patched.empty()) {
        g_allowed_roots.push_back(g_patched);  // under <data>, as the generated one is
        index_layer(g_patched, g_patched_index);
        host_log("FS patched overlay=%s (ahead of the mods overlay: the mod's own files with the port's text added; %zu files and folders)",
                 g_patched.c_str(), g_patched_index.paths.size());
    }
}
void hle_fs_add_generated_root(const char* dir) {
    if (!dir || !*dir || !is_dir_path(dir)) return;
    const int n = g_generated_more_count.load(std::memory_order_relaxed);
    if (n >= kMaxGeneratedLayers) return;
    const std::string d = canonical_dir(dir);
    g_generated_more[static_cast<std::size_t>(n)] = d;
    index_layer(d, g_generated_more_index[static_cast<std::size_t>(n)]);  // before it is published
    g_allowed_roots.push_back(d);
    g_generated_more_count.store(n + 1, std::memory_order_release);
    log_generated_layer(d, g_generated_more_index[static_cast<std::size_t>(n)]);
}
std::string hle_fs_overlay_report() {
    static std::uint64_t last_lookups = 0, last_probes = 0, last_skips = 0;
    const std::uint64_t lookups = g_app0_lookups.load(std::memory_order_relaxed);
    const std::uint64_t probes = g_overlay_probes.load(std::memory_order_relaxed);
    const std::uint64_t skips = g_overlay_skips.load(std::memory_order_relaxed);
    if (lookups == last_lookups) return {};
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "fs: %llu /app0 lookups; the generated overlays probed on disk %llu times, passed by through their index %llu "
                  "(total %llu, %llu, %llu)",
                  static_cast<unsigned long long>(lookups - last_lookups), static_cast<unsigned long long>(probes - last_probes),
                  static_cast<unsigned long long>(skips - last_skips), static_cast<unsigned long long>(lookups),
                  static_cast<unsigned long long>(probes), static_cast<unsigned long long>(skips));
    last_lookups = lookups;
    last_probes = probes;
    last_skips = skips;
    return buf;
}
void hle_fs_add_plugin_overlay(const char* dir) {
    if (!dir || !*dir || !is_dir_path(dir)) return;
    const std::string d = canonical_dir(dir);
    std::lock_guard<std::mutex> lk(g_plugin_layer_mu);
    const int n = g_plugin_layer_count.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i) {
        if (g_plugin_layers[static_cast<std::size_t>(i)] == d) return;
    }
    if (n >= kMaxPluginLayers) {
        host_log("FS plugin overlay=%s not mounted: %d plugin overlays already", d.c_str(), kMaxPluginLayers);
        return;
    }
    g_plugin_layers[static_cast<std::size_t>(n)] = d;           // written before it is published
    g_plugin_layer_count.store(n + 1, std::memory_order_release);
    host_log("FS plugin overlay=%s (behind the mods overlay)", d.c_str());
}
std::string hle_fs_overlay_file(const char* rel) {
    for (const OverlayLayer& layer_of : overlay_layers()) {
        const std::string* layer = layer_of.dir;
        if (layer->empty() || !rel) continue;
        const std::string p = join_path(*layer, rel[0] == '/' ? rel + 1 : rel);
        if (path_exists(p.c_str())) return p;
        const std::string ci = resolve_nocase(*layer, rel[0] == '/' ? rel + 1 : rel);
        if (!ci.empty()) return ci;
    }
    return {};
}

const char* hle_fs_data_root() {
    return g_data.c_str();
}

bool hle_fs_mount(const char* guest_prefix_str, const char* data_subdir) {
    if (!guest_prefix_str || !data_subdir || g_data.empty()) {
        return false;
    }
    std::string host = confine(join_path(g_data, data_subdir));
    if (host.empty()) {
        return false;
    }
    mkdir_confined(host);
    std::lock_guard<std::mutex> lock(g_fs_mu);
    for (auto& m : g_mounts) {
        if (m.first == guest_prefix_str) {
            m.second = host;
            return true;
        }
    }
    g_mounts.emplace_back(guest_prefix_str, host);
    return true;
}

void hle_fs_umount(const char* guest_prefix_str) {
    std::lock_guard<std::mutex> lock(g_fs_mu);
    for (auto it = g_mounts.begin(); it != g_mounts.end(); ++it) {
        if (it->first == guest_prefix_str) {
            g_mounts.erase(it);
            return;
        }
    }
}

void hle_fs_log_recent_opens() {
    static std::atomic<bool> done{false};
    const unsigned n = g_recent_next.load();
    if (!n || done.exchange(true)) return;
    host_log("files opened last, oldest first:");
    for (unsigned k = n > kRecentOpens ? n - kRecentOpens : 0; k < n; ++k) host_log("  %s", g_recent[k % kRecentOpens]);
}

std::string hle_fs_problem_files() {
    std::lock_guard<std::mutex> lk(g_problem_mu);
    std::string out;
    for (const std::string& f : g_problems) out += (out.empty() ? "" : "; ") + f;
    return out;
}

std::string hle_fs_map_path(const char* guest) { return guest_to_host(guest); }
std::string hle_fs_map_path_base(const char* guest) {
    t_without_plugins = true;
    std::string out = guest_to_host(guest);
    t_without_plugins = false;
    return out;
}

int hle_fs_stat_path(const char* guest, void* orbis_sb) {
    return stat_host(guest_to_host(guest).c_str(), orbis_sb);
}

void hle_register_fs() {
#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
    REG("sceKernelOpen", hle_kernel_open);
    REG("sceKernelClose", hle_kernel_close);
    REG("sceKernelRead", hle_kernel_read);
    REG("sceKernelWrite", hle_kernel_write);
    REG("sceKernelLseek", hle_kernel_lseek);
    REG("sceKernelGetdirentries", hle_kernel_getdirentries);
    REG("sceKernelStat", hle_kernel_stat);
    REG("sceKernelFstat", hle_kernel_fstat);
    REG("sceKernelMkdir", hle_kernel_mkdir);
    REG("sceKernelRmdir", hle_kernel_rmdir);
    REG("sceKernelUnlink", hle_kernel_unlink);
    REG("sceKernelRename", hle_kernel_rename);
    REG("sceKernelTruncate", hle_kernel_truncate);
    REG("sceKernelFtruncate", hle_kernel_ftruncate);
#undef REG
}
