#include "core/config.h"
#include "core/sfo.h"
#include "log.h"
#include "bbhost_version.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#define BBHOST_CONFIG_VERSION BBHOST_VERSION

// tests/config_test.cpp's mutants of the play-log migration (CONFIG_MUTANTS
// there only).
#ifdef CONFIG_MUTANTS
int g_config_mutant = 0;
#define CONFIG_MUTANT(n) (g_config_mutant == (n))
#else
#define CONFIG_MUTANT(n) false
#endif

namespace {

HostConfig g_cfg;

std::string trim(const std::string& s) {
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return {};
    }
    std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::string unquote(const std::string& v) {
    if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') && v.back() == v.front()) {
        // A backslash is a character, not an escape - these values are mostly
        // Windows paths, and real TOML turns "C:\temp" into a tab and eats the
        // slash in "C:\stuff". Both spellings have to mean the folder the
        // player typed, so only a doubled backslash collapses to one.
        std::string out;
        for (std::size_t i = 1; i + 1 < v.size(); ++i) {
            if (v[i] == '\\' && i + 2 < v.size() && v[i + 1] == '\\') ++i;
            out += v[i];
        }
        return out;
    }
    return v;
}

// section.key -> raw value text
bool parse_file(const std::string& path, std::map<std::string, std::string>* kv, std::string* err) {
    std::ifstream in(path);
    if (!in) {
        *err = "cannot open " + path;
        return false;
    }
    std::string line;
    std::string section;
    int n = 0;
    while (std::getline(in, line)) {
        ++n;
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') {
            continue;
        }
        if (t.front() == '[' && t.back() == ']') {
            section = trim(t.substr(1, t.size() - 2));
            continue;
        }
        std::size_t eq = t.find('=');
        if (eq == std::string::npos) {
            *err = path + ":" + std::to_string(n) + ": expected key = value";
            return false;
        }
        std::string key = trim(t.substr(0, eq));
        std::string val = trim(t.substr(eq + 1));
        // strip trailing comment outside quotes
        if (!val.empty() && val[0] != '"' && val[0] != '\'') {
            std::size_t hash = val.find('#');
            if (hash != std::string::npos) {
                val = trim(val.substr(0, hash));
            }
        } else if (!val.empty()) {
            // After a quoted value too: upgrade_user_config's notes
            // (host = "x"  # added by bbhost ...) are not part of it. A
            // backslash is no escape here, so the next quote closes it.
            const std::size_t close = val.find(val[0], 1);
            if (close != std::string::npos) {
                const std::string rest = trim(val.substr(close + 1));
                if (!rest.empty() && rest[0] == '#') val = val.substr(0, close + 1);
            }
        }
        (*kv)[section.empty() ? key : section + "." + key] = val;
    }
    return true;
}

void apply_values(const std::map<std::string, std::string>& kv, HostConfig* c) {
    auto str = [&](const char* k, std::string* dst) {
        auto it = kv.find(k);
        if (it != kv.end()) {
            *dst = unquote(it->second);
        }
    };
    auto num = [&](const char* k, int* dst) {
        auto it = kv.find(k);
        if (it != kv.end()) {
            *dst = std::atoi(it->second.c_str());
        }
    };
    str("paths.app0", &c->app0);
    str("paths.data", &c->data);
    str("paths.tmp", &c->tmp);
    str("paths.mods", &c->mods);
    str("paths.eboot", &c->eboot);
    str("online.host", &c->online_host);
    str("online.scheme", &c->online_scheme);
    {
        auto it = kv.find("online.verify_tls");
        if (it != kv.end()) {
            c->online_verify_tls = it->second == "true" || it->second == "1";
        }
    }
    str("online.online_id", &c->online_id);
    {
        auto it = kv.find("online.require_account");
        if (it != kv.end()) {
            c->online_require_account = it->second == "true" || it->second == "1";
        }
    }
    str("online.np_server", &c->np_server);
    str("online.auth_server", &c->auth_server);
    str("online.p2p_addr", &c->p2p_addr);
    str("online.stun_server", &c->stun_server);
    num("online.p2p_port", &c->p2p_port);
    num("online.playlog_upload_seconds", &c->playlog_upload_seconds);
    num("online.playlog_sample_ms", &c->playlog_sample_ms);
    num("online.sign_timeout_seconds", &c->sign_timeout_seconds);
    {
        auto it = kv.find("update.check");
        if (it != kv.end()) c->update_check = it->second == "true" || it->second == "1";
    }
    str("update.source", &c->update_source);
    str("update.token_file", &c->update_token_file);
    str("player.name", &c->player_name);
    str("player.ime", &c->ime_mode);
    num("video.width", &c->width);
    num("video.height", &c->height);
    num("video.fps_cap", &c->fps_cap);
    str("video.resolution", &c->resolution);
    str("video.window_mode", &c->window_mode);
    str("video.model_detail", &c->model_detail);
    str("video.shaders", &c->shaders);
    num("video.vblank_hz", &c->vblank_hz);
    {
        auto it = kv.find("startup.setup_window");
        if (it != kv.end()) c->setup_always = it->second == "true" || it->second == "1";
    }
    {
        auto it = kv.find("debug.tooling");
        if (it != kv.end()) c->tooling = it->second == "true" || it->second == "1";
    }
    {
        auto it = kv.find("startup.skip_intro");
        if (it != kv.end()) {
            c->skip_intro = it->second == "true" || it->second == "1";
        }
    }
    {
        auto it = kv.find("loading.quick_reentry");
        if (it != kv.end()) c->quick_reentry = it->second == "true" || it->second == "1";
    }
    {
        auto it = kv.find("streaming.all_post_processors");
        if (it != kv.end()) c->all_post_processors = it->second == "true" || it->second == "1";
    }
    {
        auto it = kv.find("world.change_appearance");
        if (it != kv.end()) c->change_appearance = it->second == "true" || it->second == "1";
    }
    {
        auto it = kv.find("world.rebirth");
        if (it != kv.end()) c->rebirth = it->second == "true" || it->second == "1";
    }
    {
        auto it = kv.find("world.five_players");
        if (it != kv.end()) c->five_players = it->second == "true" || it->second == "1";
    }
    // Every [keys] entry, whatever it is called: the action names live in
    // window.cpp beside the defaults, and an unknown one is reported there
    // rather than silently dropped here.
    for (const auto& [k, v] : kv) {
        if (k.rfind("keys.", 0) == 0) {
            c->keys[k.substr(5)] = unquote(v);
        }
    }
    {
        auto it = kv.find("video.headless");
        if (it != kv.end()) {
            c->headless = it->second == "true" || it->second == "1";
        }
    }
}

std::map<std::string, std::string> g_values;  // the config file's raw values, for config_value

std::string exe_dir() {
#if defined(_WIN32)
    char buf[4096];
    const DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
    if (n == 0 || n >= sizeof(buf)) return ".";
    std::string p(buf, n);
    for (char& ch : p) {
        if (ch == '\\') ch = '/';
    }
    const std::size_t slash = p.find_last_of('/');
    return slash == std::string::npos ? "." : p.substr(0, slash);
#else
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return ".";
    }
    buf[n] = 0;
    std::string p(buf);
    std::size_t slash = p.find_last_of('/');
    return slash == std::string::npos ? "." : p.substr(0, slash);
#endif
}

bool file_exists(const std::string& p) {
    std::ifstream f(p);
    return static_cast<bool>(f);
}

bool dir_exists(const std::string& p) {
#if defined(_WIN32)
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

// The template's stand-in paths. A config still holding one has not been
// filled in, which reads better as "not set" than as "that path is missing".
bool is_placeholder(const std::string& v) {
    return v.find("/path/to/") != std::string::npos || v.find("PATH-TO-") != std::string::npos;
}

std::string env_dir(const char* name) {
    const char* e = std::getenv(name);
    if (!e || !e[0]) return {};
    std::string p(e);
    for (char& ch : p) {
        if (ch == '\\') ch = '/';
    }
    return p;
}

// The per-user folders (config_user_dir): what a player sets once and every
// download or update of bbhost then finds - the game's paths, the F10
// settings, the account. BBHOST_CONFIG_DIR replaces the config one (tests,
// a portable setup).
std::string user_config_dir() {
    if (std::string d = env_dir("BBHOST_CONFIG_DIR"); !d.empty()) return d;
#if defined(_WIN32)
    if (std::string d = env_dir("APPDATA"); !d.empty()) return d + "/bbhost";
#else
    if (std::string d = env_dir("XDG_CONFIG_HOME"); !d.empty()) return d + "/bbhost";
    if (std::string d = env_dir("HOME"); !d.empty()) return d + "/.config/bbhost";
#endif
    return exe_dir();
}

std::string user_data_dir() {
#if defined(_WIN32)
    if (std::string d = env_dir("LOCALAPPDATA"); !d.empty()) return d + "/bbhost/data";
#else
    if (std::string d = env_dir("XDG_DATA_HOME"); !d.empty()) return d + "/bbhost/data";
    if (std::string d = env_dir("HOME"); !d.empty()) return d + "/.local/share/bbhost/data";
#endif
    return exe_dir() + "/data";
}

bool make_dirs(const std::string& path) {
    std::string cur;
    for (std::size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty() && !(cur.size() == 2 && cur[1] == ':') && !dir_exists(cur)) {
#if defined(_WIN32)
                CreateDirectoryA(cur.c_str(), nullptr);
#else
                ::mkdir(cur.c_str(), 0755);
#endif
            }
        }
        if (i < path.size()) cur += path[i];
    }
    return dir_exists(path);
}

bool is_relative_path(const std::string& v) {
    if (v.empty() || v[0] == '/' || v[0] == '\\') return false;
    if (v.size() >= 2 && v[1] == ':') return false;  // a drive letter
    return true;
}

std::string dir_of(const std::string& file) {
    const std::size_t slash = file.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : file.substr(0, slash);
}

// The [paths] keys whose relative values are relative to their file.
constexpr const char* kPathKeys[] = {"paths.app0", "paths.data", "paths.tmp", "paths.mods", "paths.eboot"};

// One layer's values, made independent of where the file was: relative paths
// joined to its folder, the template's stand-ins dropped (a package's
// unfilled template must not hide the paths a lower layer has).
void normalize_layer(const std::string& file, std::map<std::string, std::string>* kv) {
    const std::string dir = dir_of(file);
    for (const char* k : kPathKeys) {
        auto it = kv->find(k);
        if (it == kv->end()) continue;
        std::string v = unquote(it->second);
        if (is_placeholder(v)) {
            kv->erase(it);
            continue;
        }
        if (!dir.empty() && dir != "." && is_relative_path(v)) {
            v = dir + "/" + (v.rfind("./", 0) == 0 ? v.substr(2) : v);
            it->second = "\"" + v + "\"";
        }
    }
}

// The keys every user config should carry, with the value bbhost uses when
// one is missing: a key added to bbhost later is written into the player's
// file at their next start (with a note), so the file always shows what is
// in effect. Required keys with no sensible default (paths.app0,
// paths.eboot) are not here: config_setup_help stops the run and names them.
struct KeySpec {
    const char* section;
    const char* key;
    const char* value;
    const char* note;
};
constexpr KeySpec kUserKeys[] = {
    {"online", "host", "\"thehuntersdream.com\"", "the private server the game's online traffic goes to"},
    // https for the live server; a file naming another server gets http, the
    // value it would have got before.
    {"online", "scheme", "\"https\"", "http or https for that server"},
};

// Adds what kUserKeys says is missing to the file, each under its section,
// leaving everything already there - comments included - as it was.
void upgrade_user_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) return;
    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);) {
        if (!l.empty() && l.back() == '\r') l.pop_back();
        lines.push_back(l);
    }
    in.close();
    std::map<std::string, std::string> kv;
    std::string err;
    if (!parse_file(path, &kv, &err)) return;
    bool changed = false;
    for (const KeySpec& k : kUserKeys) {
        const std::string full = std::string(k.section) + "." + k.key;
        if (kv.count(full)) continue;
        const char* value = k.value;
        if (full == "online.scheme" && kv.count("online.host") && unquote(kv["online.host"]) != "thehuntersdream.com")
            value = "\"http\"";
        const std::string entry = std::string(k.key) + " = " + value + "  # added by bbhost " BBHOST_CONFIG_VERSION ": " + k.note;
        // The end of the section's block: before the next [header], else the file's end.
        std::size_t at = lines.size();
        bool found = false;
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const std::string t = trim(lines[i]);
            if (t.size() >= 2 && t.front() == '[' && t.back() == ']') {
                if (found) {
                    at = i;
                    while (at > 0 && trim(lines[at - 1]).empty()) --at;
                    break;
                }
                found = trim(t.substr(1, t.size() - 2)) == k.section;
            }
        }
        if (!found) {
            lines.push_back("");
            lines.push_back(std::string("[") + k.section + "]");
            at = lines.size();
        }
        lines.insert(lines.begin() + static_cast<std::ptrdiff_t>(at), entry);
        host_log("config: %s had no %s; added %s = %s", path.c_str(), full.c_str(), full.c_str(), value);
        changed = true;
    }
    if (!changed) return;
    std::ofstream out(path, std::ios::trunc);
    for (const std::string& l : lines) out << l << "\n";
}

// The live server takes only https and signed-in players, as the playtest
// server does. A user config from before that (below config_version 3) that
// still names it with the old settings - plain http, no certificate check,
// no account - gets the new ones written into it, so the file shows what is
// in effect.
void move_live_server_to_https(const std::string& path) {
    std::map<std::string, std::string> kv;
    std::string err;
    if (!parse_file(path, &kv, &err)) return;
    const auto get = [&](const std::string& k) {
        auto it = kv.find(k);
        return it == kv.end() ? std::string() : unquote(it->second);
    };
    const std::string ver = get("bbhost.config_version");
    if ((ver.empty() ? 1 : std::atoi(ver.c_str())) >= 3 || get("online.host") != "thehuntersdream.com") return;
    struct Want {
        const char* key;
        const char* value;
    };
    constexpr Want kLive[] = {
        {"scheme", "\"https\""},
        {"verify_tls", "true"},
        {"require_account", "true"},
        {"auth_server", "\"https://thehuntersdream.com\""},
    };
    std::vector<ConfigEdit> edits;
    std::string keys;
    for (const Want& w : kLive) {
        if (get(std::string("online.") + w.key) == unquote(w.value)) continue;
        edits.push_back({"online", w.key, w.value});
        keys += (keys.empty() ? "" : ", ") + std::string(w.key);
    }
    if (edits.empty()) return;
    if (!config_set_values(path, edits)) {
        host_log("config: could not update %s for the live server's https settings", path.c_str());
        return;
    }
    host_log("config: %s: the live server takes https and signed-in players now; set to its values: %s", path.c_str(),
             keys.c_str());
}

// The first-start template wrote player.ime = "auto" (below config_version
// 4): the character's name was answered with player.name before it could be
// typed, and the chalice glyph and network password boxes were cancelled the
// moment they opened, so none of the three could be typed in. That was the
// template's value, not a choice, and a file that still has it is set to
// "type", the default.
void move_ime_to_type(const std::string& path) {
    std::map<std::string, std::string> kv;
    std::string err;
    if (!parse_file(path, &kv, &err)) return;
    const auto get = [&](const std::string& k) {
        auto it = kv.find(k);
        return it == kv.end() ? std::string() : unquote(it->second);
    };
    const std::string ver = get("bbhost.config_version");
    if ((ver.empty() ? 1 : std::atoi(ver.c_str())) >= 4 || get("player.ime") != "auto") return;
    if (!config_set_values(path, {{"player", "ime", "\"type\""}})) {
        host_log("config: could not set player.ime to \"type\" in %s", path.c_str());
        return;
    }
    host_log("config: %s: player.ime was the old template's \"auto\", which kept the game's text boxes from being typed in; "
             "set to \"type\"",
             path.c_str());
}

// The play log went out every 5 s, the default bbhost.example.toml spelled
// out, until the server asked for 30 s or more (every upload is a file it
// keeps). A 5 in the per-user file below config_version 5 is that copy (the
// first-start template never wrote the key): it is rewritten to 30, so the
// file shows what is in effect and the setup window's next save, which writes
// the version, finds nothing to bless.
void move_playlog_to_default(const std::string& path) {
    std::map<std::string, std::string> kv;
    std::string err;
    if (!parse_file(path, &kv, &err)) return;
    const auto get = [&](const std::string& k) {
        auto it = kv.find(k);
        return it == kv.end() ? std::string() : unquote(it->second);
    };
    const std::string ver = get("bbhost.config_version");
    const int version = ver.empty() ? 1 : std::atoi(ver.c_str());
    if (CONFIG_MUTANT(1) ? version > 5 : version >= 5) return;
    if (get("online.playlog_upload_seconds") != "5") return;
    if (CONFIG_MUTANT(2)) return;
    if (!config_set_values(path, {{"online", "playlog_upload_seconds", "30"}})) {
        host_log("config: could not set online.playlog_upload_seconds to 30 in %s", path.c_str());
        return;
    }
    host_log("config: %s: online.playlog_upload_seconds was the old example's 5; set to 30, the default now (the "
             "server keeps every upload as a file)",
             path.c_str());
}

// A bbhost.toml from before the per-user config (beside an older package's
// exe, or in the working directory) that names the game: copied to the user
// config once, its relative paths made absolute - the saves stay where they
// were - so the next download finds it all.
bool import_legacy(const std::string& from, const std::string& to, bool paths_only = false) {
    std::map<std::string, std::string> kv;
    std::string err;
    if (!parse_file(from, &kv, &err)) return false;
    const auto real = [&](const char* k) {
        auto it = kv.find(k);
        return it != kv.end() && !unquote(it->second).empty() && !is_placeholder(unquote(it->second));
    };
    if (!real("paths.app0") || !real("paths.eboot")) return false;
    std::ifstream in(from);
    std::ostringstream out;
    out << "# Imported by bbhost " BBHOST_CONFIG_VERSION " from " << from << ": bbhost reads this one now,\n"
        << "# from any copy of bbhost, so a new download or an update needs nothing set again.\n";
    const std::string dir = dir_of(from);
    std::string section;
    for (std::string l; std::getline(in, l);) {
        if (!l.empty() && l.back() == '\r') l.pop_back();
        const std::string t = trim(l);
        if (t.size() >= 2 && t.front() == '[' && t.back() == ']') section = trim(t.substr(1, t.size() - 2));
        if (paths_only && section != "paths") continue;  // a profile's own settings stay its own
        const std::size_t eq = t.find('=');
        if (section == "paths" && !t.empty() && t[0] != '#' && eq != std::string::npos) {
            const std::string key = trim(t.substr(0, eq));
            std::string v = unquote(trim(t.substr(eq + 1)));
            if (!dir.empty() && dir != "." && is_relative_path(v) && !is_placeholder(v)) {
                v = dir + "/" + (v.rfind("./", 0) == 0 ? v.substr(2) : v);
                out << key << " = \"" << v << "\"\n";
                continue;
            }
        }
        out << l << "\n";
    }
    if (!make_dirs(dir_of(to))) return false;
    std::ofstream o(to);
    o << out.str();
    if (!o) return false;
    host_log("config: imported %s into %s (the per-user config every copy of bbhost reads)", from.c_str(), to.c_str());
    return true;
}

}  // namespace

bool config_load(int argc, char** argv, HostConfig* out, std::string* error) {
    HostConfig c;
    std::string explicit_cfg;
    std::string cli_app0, cli_data, cli_eboot;
    std::string write_template;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](std::string* dst) -> bool {
            std::size_t eq = a.find('=');
            if (eq != std::string::npos) {
                *dst = a.substr(eq + 1);
                return true;
            }
            if (i + 1 < argc) {
                *dst = argv[++i];
                return true;
            }
            *error = "missing value for " + a;
            return false;
        };
        if (a.rfind("--config", 0) == 0) {
            if (!val(&explicit_cfg)) return false;
        } else if (a.rfind("--app0", 0) == 0) {
            if (!val(&cli_app0)) return false;
        } else if (a.rfind("--data", 0) == 0) {
            if (!val(&cli_data)) return false;
        } else if (a.rfind("--eboot", 0) == 0) {
            if (!val(&cli_eboot)) return false;
        } else if (a.rfind("--write-config", 0) == 0) {
            if (!val(&write_template)) return false;
        } else if (a == "--headless") {
            c.headless = true;
        } else if (a == "--setup") {
            c.setup_requested = true;
        } else if (a.rfind("--", 0) == 0) {
            *error = "unknown flag " + a;
            return false;
        } else if (cli_eboot.empty()) {
            cli_eboot = a;
        }
    }
    if (!write_template.empty()) {
        if (!config_write_template(write_template)) {
            *error = "cannot write " + write_template;
            return false;
        }
        host_log("wrote config template %s", write_template.c_str());
    }
    // The configuration in layers, each overriding only what it sets:
    //   the per-user bbhost.toml (config_user_dir: what a player sets once),
    //   a bbhost.toml in the working directory or beside the exe (a
    //   developer's tree, an older package), then --config (a package's own
    //   settings, as the playtest kit's dev server). A user config that does
    //   not exist yet is imported from the second when that names the game,
    //   and one that does is given the keys bbhost has since come to expect.
    const std::string user_cfg = user_config_dir() + "/bbhost.toml";
    make_dirs(user_config_dir());  // where the F10 settings are saved, whatever else exists
    std::string local_cfg;
    // Absolute, so an import into the user config can resolve its relative paths.
    std::string cwd;
    {
        std::error_code ec;
        cwd = std::filesystem::current_path(ec).generic_string();
    }
    for (const std::string& cand : {cwd.empty() ? std::string("bbhost.toml") : cwd + "/bbhost.toml", exe_dir() + "/bbhost.toml"}) {
        if (file_exists(cand) && cand != user_cfg) {
            local_cfg = cand;
            break;
        }
    }
    if (!file_exists(user_cfg) && !local_cfg.empty()) import_legacy(local_cfg, user_cfg);
    // An older playtest kit kept the game's paths in its --config file: those
    // (only those) carry over too.
    if (!file_exists(user_cfg) && !explicit_cfg.empty()) {
        std::string abs = explicit_cfg;
        if (is_relative_path(abs) && !cwd.empty()) abs = cwd + "/" + abs;
        import_legacy(abs, user_cfg, true);
    }
    if (file_exists(user_cfg)) {
        upgrade_user_config(user_cfg);
        move_live_server_to_https(user_cfg);
        move_ime_to_type(user_cfg);
        move_playlog_to_default(user_cfg);
    }
    std::map<std::string, std::string> merged;
    std::string layers;
    for (const std::string& f : {file_exists(user_cfg) ? user_cfg : std::string(), local_cfg, explicit_cfg}) {
        if (f.empty()) continue;
        std::map<std::string, std::string> kv;
        std::string err;
        if (!parse_file(f, &kv, &err)) {
            if (f == explicit_cfg) {
                *error = err;
                return false;
            }
            host_log("config: %s", err.c_str());
            continue;
        }
        normalize_layer(f, &kv);
        if (f == user_cfg) {
            // A user config from before the setup window wrote its version
            // (kUserConfigVersion): its setup_window is the old default it
            // saved, not a choice, and this build's default stands.
            auto ver = kv.find("bbhost.config_version");
            const int v = ver == kv.end() ? 1 : std::atoi(unquote(ver->second).c_str());
            if (v < 2 && kv.erase("startup.setup_window"))
                host_log("config: %s is from an older bbhost: its setup_window is set aside until the setup window saves it", f.c_str());
        }
        {
            // The old example's play-log period in a file bbhost does not
            // rewrite (beside the exe, --config), or a per-user one it could
            // not: set aside below version 5 (move_playlog_to_default).
            auto ver = kv.find("bbhost.config_version");
            const int v = ver == kv.end() ? 1 : std::atoi(unquote(ver->second).c_str());
            auto it = kv.find("online.playlog_upload_seconds");
            if (v < 5 && it != kv.end() && unquote(it->second) == "5" && !CONFIG_MUTANT(3)) {
                kv.erase(it);
                host_log("config: %s: online.playlog_upload_seconds = 5 was the old example's; set aside, the default "
                         "%d s stands (config_version 5 in the file keeps a 5)",
                         f.c_str(), HostConfig{}.playlog_upload_seconds);
            }
        }
        for (auto& [k, v] : kv) merged[k] = v;
        layers += (layers.empty() ? "" : " <- ") + f;
    }
#if defined(BBHOST_RELEASE_BUILD)
    c.setup_always = true;  // a player's build opens the setup window at every start unless told not to
#endif
    apply_values(merged, &c);
    g_values = merged;
    // The file a player edits: the user config, once there is one.
    c.config_path = file_exists(user_cfg) ? user_cfg : !explicit_cfg.empty() ? explicit_cfg : local_cfg;
    c.config_layers = layers;
    if (!explicit_cfg.empty()) {
        // A --config is a profile (the playtest kit's dev server, a second
        // account): its F10 settings and account are its own.
        std::string stem = explicit_cfg.substr(explicit_cfg.find_last_of("/\\") == std::string::npos ? 0 : explicit_cfg.find_last_of("/\\") + 1);
        if (stem.size() > 5 && stem.compare(stem.size() - 5, 5, ".toml") == 0) stem.resize(stem.size() - 5);
        for (char& ch : stem) {
            if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_') ch = '_';
        }
        c.profile = stem;
        c.profile_config = explicit_cfg;
    }
    if (!cli_app0.empty()) c.app0 = cli_app0;
    if (!cli_data.empty()) c.data = cli_data;
    if (!cli_eboot.empty()) c.eboot = cli_eboot;
    // A config that still holds the template's stand-in path is not configured.
    if (is_placeholder(c.app0)) c.app0.clear();
    if (is_placeholder(c.eboot)) c.eboot.clear();
    if (c.eboot.empty()) {
        // Next to the working directory first (a developer's tree), then next
        // to the executable (a package the player unzipped somewhere).
        const std::string dir = exe_dir();
        for (const std::string& p : {std::string("eboot-109-decrypted.bin"), std::string("eboot.elf"),
                                     dir + "/eboot-109-decrypted.bin", dir + "/eboot.elf"}) {
            if (file_exists(p)) {
                c.eboot = p;
                break;
            }
        }
    }
    if (c.eboot.empty() && !c.app0.empty()) {
        // Then the game's own eboot.bin, the update's first, when it is an
        // ELF already: a dump made for shadPS4 has it decrypted.
        const GameFolders g = config_game_folders(c.app0);
        for (const std::string& d : {g.update, g.base}) {
            if (d.empty()) continue;
            std::ifstream f(d + "/eboot.bin", std::ios::binary);
            char magic[4] = {};
            if (f.read(magic, 4) && std::memcmp(magic, "\x7f" "ELF", 4) == 0) {
                c.eboot = d + "/eboot.bin";
                break;
            }
        }
    }
    if (c.data.empty()) {
        // Saves and caches: where they were (a data folder beside the eboot,
        // the old default) if that exists, else the per-user data folder,
        // which outlives any copy of bbhost.
        // A package ships a data folder of its own (data/mods), so only one
        // with saves in it is the old one.
        const std::string beside = (c.eboot.empty() ? exe_dir() : (dir_of(c.eboot).empty() ? "." : dir_of(c.eboot))) + "/data";
        c.data = dir_exists(beside + "/saves") ? beside : user_data_dir();
    }
    if (c.mods.empty() && dir_exists(exe_dir() + "/data/mods")) {
        // The package's own asset overlay (the debug menu's font), wherever
        // the saves are.
        c.mods = exe_dir() + "/data/mods";
    }
    if (c.tmp.empty() && !c.data.empty()) {
        c.tmp = c.data + "/tmp";
    }
    // The file system layer makes the last level only; the per-user data
    // folder is several levels deep on a first run.
    if (!c.data.empty()) make_dirs(c.data);
    if (const char* e = std::getenv("BBHOST_HEADLESS")) {
        if (e[0] == '1') {
            c.headless = true;
        }
    }
    if (const char* e = std::getenv("BBHOST_SKIP_INTRO")) {
        if (std::strcmp(e, "1") == 0) c.skip_intro = true;
        else if (std::strcmp(e, "0") == 0) c.skip_intro = false;
    }
    if (const char* e = std::getenv("BBHOST_QUICK_REENTRY")) {
        if (std::strcmp(e, "1") == 0) c.quick_reentry = true;
        else if (std::strcmp(e, "0") == 0) c.quick_reentry = false;
    }
    if (const char* e = std::getenv("BBHOST_ALL_POST_PROCESSORS")) {
        if (std::strcmp(e, "1") == 0) c.all_post_processors = true;
        else if (std::strcmp(e, "0") == 0) c.all_post_processors = false;
    }
    if (const char* e = std::getenv("BBHOST_CHANGE_APPEARANCE")) {
        if (std::strcmp(e, "1") == 0) c.change_appearance = true;
        else if (std::strcmp(e, "0") == 0) c.change_appearance = false;
    }
    if (const char* e = std::getenv("BBHOST_REBIRTH")) {
        if (std::strcmp(e, "1") == 0) c.rebirth = true;
        else if (std::strcmp(e, "0") == 0) c.rebirth = false;
    }
    if (const char* e = std::getenv("BBHOST_FIVE_PLAYERS")) {
        if (std::strcmp(e, "1") == 0) c.five_players = true;
        else if (std::strcmp(e, "0") == 0) c.five_players = false;
    }
    if (const char* e = std::getenv("BBHOST_SETUP_WINDOW")) {
        if (std::strcmp(e, "1") == 0) c.setup_always = true;
        else if (std::strcmp(e, "0") == 0) c.setup_always = false;
    }
    if (const char* e = std::getenv("BBHOST_TOOLING")) {
        if (std::strcmp(e, "1") == 0) c.tooling = true;
        else if (std::strcmp(e, "0") == 0) c.tooling = false;
    }
    if (const char* e = std::getenv("BBHOST_IME")) {
        if (std::strcmp(e, "type") == 0 || std::strcmp(e, "auto") == 0) c.ime_mode = e;
    }
    if (const char* e = std::getenv("BBHOST_PLAYLOG_UPLOAD_SECONDS"); e && *e && !CONFIG_MUTANT(4)) {
        c.playlog_upload_seconds = std::atoi(e);
    }
    g_cfg = c;
    *out = c;
    return true;
}

const HostConfig& config() { return g_cfg; }

std::string config_setup_help(const HostConfig& c) {
    if (c.eboot.empty() || c.app0.empty()) {
        const std::string path = user_config_dir() + "/bbhost.toml";
        if (!file_exists(path) && make_dirs(user_config_dir())) config_write_template(path);
        return "bbhost: ゲームフォルダーと復号済み 1.09 eboot の場所を設定してください。\n"
               "設定ファイル: " + path + "\n[paths] の app0 と eboot を指定します。\n";
    }
    if (!file_exists(c.eboot)) return "bbhost: eboot が見つかりません: " + c.eboot + "\n";
    if (!dir_exists(c.app0)) return "bbhost: ゲームフォルダーが見つかりません: " + c.app0 + "\n";
    if (!dir_exists(c.app0 + "/dvdroot_ps4"))
        return "bbhost: dvdroot_ps4 を含む親フォルダーを指定してください: " + c.app0 + "\n";
    return {};
}

std::string config_eboot_help(const HostConfig& c, const std::string& sha256, const char* want_sha256) {
    return "bbhost: 対応する復号済み 1.09 eboot ではありません。\nファイル: " + c.eboot +
           "\n検出した SHA-256: " + sha256 + "\n必要な SHA-256: " + want_sha256 +
           "\nゲームに更新 1.09 を適用し、対応する復号済み ELF を paths.eboot に指定してください。\n";
}

void config_set_skip_intro(bool on) { g_cfg.skip_intro = on; }

void config_set_enhancements(bool change_appearance, bool rebirth, bool five_players) {
    const auto set = [](bool* field, bool on, const char* env) {
        const char* e = std::getenv(env);
        if (e && (std::strcmp(e, "0") == 0 || std::strcmp(e, "1") == 0)) return;  // the environment's, kept
        *field = on;
    };
    set(&g_cfg.change_appearance, change_appearance, "BBHOST_CHANGE_APPEARANCE");
    set(&g_cfg.rebirth, rebirth, "BBHOST_REBIRTH");
    set(&g_cfg.five_players, five_players, "BBHOST_FIVE_PLAYERS");
}

void config_set_online(const std::string& host, const std::string& scheme, bool verify_tls, bool require_account,
                       const std::string& auth_server) {
    g_cfg.online_host = host;
    g_cfg.online_scheme = scheme;
    g_cfg.online_verify_tls = verify_tls;
    g_cfg.online_require_account = require_account;
    g_cfg.auth_server = auth_server;
}

GameFolders config_game_folders(const std::string& app0) {
    GameFolders g;
    std::string a = app0;
    while (a.size() > 1 && (a.back() == '/' || a.back() == '\\')) a.pop_back();
    if (a.empty()) return g;
    static const char* const kSuffixes[] = {"-UPDATE", "-patch"};  // shadPS4 tries them in this order
    for (const char* s : kSuffixes) {
        const std::size_t n = std::strlen(s);
        if (a.size() > n && a.compare(a.size() - n, n, s) == 0 && dir_exists(a.substr(0, a.size() - n) + "/dvdroot_ps4")) {
            g.base = a.substr(0, a.size() - n);
            g.update = a;
            return g;
        }
    }
    g.base = a;
    for (const char* s : kSuffixes) {
        if (dir_exists(a + s)) {
            g.update = a + s;
            break;
        }
    }
    return g;
}

App0Version config_app0_version(const std::string& app0) {
    App0Version v;
    const GameFolders g = config_game_folders(app0);
    if (g.base.empty()) return v;
    std::map<std::string, SfoValue> sfo;
    std::string err;
    if (!g.update.empty() && sfo_read(g.update + "/sce_sys/param.sfo", &sfo, &err)) {
        v.update = g.update;
    } else if (!sfo_read(g.base + "/sce_sys/param.sfo", &sfo, &err)) {
        return v;
    }
    if (sfo.count("APP_VER")) v.app_ver = sfo["APP_VER"].text;
    if (sfo.count("CATEGORY")) v.category = sfo["CATEGORY"].text;
    return v;
}

bool config_write_template(const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << R"toml(# bbhost 日本語版の設定。ゲームと復号済み eboot の場所を指定してください。
# 相対パスはこのファイルの場所を基準にします。Windows でも / を使用できます。
[paths]
# 日本版 Old Hunters (CUSA03014)、更新 1.09、dvdroot_ps4 を含むフォルダー。
app0 = "PATH-TO-THE-GAME-DUMP"
# SHA-256: 941f887a562aae054fac35af8cc8f27cf075f3d4cc2e029fb5ae2a663aaa5ae7
eboot = "PATH-TO-THE-EBOOT"
# セーブとキャッシュの保存先。省略時は利用者ごとの標準フォルダーです。
# data = "D:/bbhost-data"
# mods = "./data/mods"

[online]
host = "thehuntersdream.com"
scheme = "https"
verify_tls = true
online_id = "Hunter"
require_account = true
auth_server = "https://thehuntersdream.com"
p2p_port = 9307
# np_server = "http://127.0.0.1:18671"
# stun_server = "thehuntersdream.com:3478"

[player]
name = "Hunter"
# type: 手入力、auto: 上の名前を自動入力。
ime = "type"

[video]
width = 1920
height = 1080
fps_cap = 30
headless = false

[startup]
skip_intro = false

[plugins]
region_switch = true

[region_switch]
# 日本版ゲームデータを維持して欧州オンラインを利用。OFF で無効。
region = "JP"

[update]
# この日本語版の更新先を設定するまでは自動確認しません。
check = false

# 操作の変更: ゲーム中の「キー設定」。値は SDL のキー名です。
# [keys]
# roll = "Space"
# interact = "E"
# attack = "Mouse1"
# firearm = "X, Mouse2"
)toml";
    return true;
}

std::string config_exe_dir() { return exe_dir(); }
std::string config_user_dir() { return user_config_dir(); }
std::string config_user_file() { return user_config_dir() + "/bbhost.toml"; }
std::string config_default_data_dir() { return user_data_dir(); }

bool config_set_values(const std::string& path, const std::vector<ConfigEdit>& edits) {
    std::vector<std::string> lines;
    {
        std::ifstream in(path);
        for (std::string l; std::getline(in, l);) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            lines.push_back(l);
        }
    }
    for (const ConfigEdit& e : edits) {
        // Where the key is now, and where its section's block ends.
        std::string section;
        std::size_t found = std::string::npos, block_end = std::string::npos;
        bool in_section = false;
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const std::string t = trim(lines[i]);
            if (t.size() >= 2 && t.front() == '[' && t.back() == ']') {
                if (in_section && block_end == std::string::npos) {
                    block_end = i;
                    while (block_end > 0 && trim(lines[block_end - 1]).empty()) --block_end;
                }
                section = trim(t.substr(1, t.size() - 2));
                in_section = section == e.section;
                continue;
            }
            if (!in_section || t.empty() || t[0] == '#') continue;
            const std::size_t eq = t.find('=');
            if (eq != std::string::npos && trim(t.substr(0, eq)) == e.key) found = i;
        }
        const std::string entry = e.key + " = " + e.value;
        if (found != std::string::npos) {
            lines[found] = entry;
        } else if (in_section || block_end != std::string::npos) {
            const std::size_t at = block_end == std::string::npos ? lines.size() : block_end;
            lines.insert(lines.begin() + static_cast<std::ptrdiff_t>(at), entry);
        } else {
            lines.push_back("");
            lines.push_back("[" + e.section + "]");
            lines.push_back(entry);
        }
    }
    if (!make_dirs(dir_of(path))) return false;
    std::ofstream out(path, std::ios::trunc);
    for (const std::string& l : lines) out << l << "\n";
    return static_cast<bool>(out);
}

bool config_parse_toml(const std::string& path, std::map<std::string, std::string>* kv, std::string* err) {
    return parse_file(path, kv, err);
}

void config_value_set(const std::string& key, const std::string& raw) { g_values[key] = raw; }

std::string config_value(const std::string& key) {
    auto it = g_values.find(key);
    return it == g_values.end() ? std::string() : unquote(it->second);
}

std::string config_effect_steps() {
    if (const char* e = std::getenv("BBHOST_SFX_STEP"); e && *e) return e;
    const std::string v = config_value("video.effect_steps");
    return v.empty() ? "smooth" : v;
}

std::string config_effect_emission() {
    if (const char* e = std::getenv("BBHOST_SFX_EMISSION"); e && *e) return e;
    const std::string v = config_value("video.effect_emission");
    return v.empty() ? "60" : v;
}
