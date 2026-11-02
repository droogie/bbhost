// core/config.cpp: a per-user bbhost.toml as older releases left it, loaded
// by this one - notes after quoted values, the live server's move to https,
// the play log's move from 5 s to 30 (with its mutants), and servers of the
// player's own left as they are. And the game folder with its update in a
// folder of its own beside it, as shadPS4 keeps one.
#include "core/config.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

namespace fs = std::filesystem;

fs::path g_dir;

std::string read(const fs::path& p) {
    std::ifstream in(p);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

// The file as the player's config, loaded the way bbhost starts.
HostConfig load(const std::string& toml) {
    {
        std::ofstream out(g_dir / "config" / "bbhost.toml", std::ios::trunc);
        out << toml;
    }
    char arg0[] = "config_test";
    char* argv[] = {arg0, nullptr};
    HostConfig c;
    std::string err;
    CHECK(config_load(1, argv, &c, &err));
    return c;
}

// A param.sfo with APP_VER and CATEGORY in it, laid out as the console's.
void write_sfo(const fs::path& p, const std::string& ver, const std::string& cat) {
    const std::vector<std::pair<std::string, std::string>> kv = {{"APP_VER", ver}, {"CATEGORY", cat}};
    std::string head, entries, keys, data;
    const auto put = [](std::string& s, std::uint32_t v, int bytes) {
        for (int i = 0; i < bytes; ++i) s.push_back(static_cast<char>(v >> (8 * i)));
    };
    for (const auto& [k, v] : kv) {
        const std::uint32_t len = static_cast<std::uint32_t>(v.size() + 1), cap = (len + 3) & ~3u;
        put(entries, static_cast<std::uint32_t>(keys.size()), 2);
        put(entries, 0x0204, 2);  // UTF-8 text
        put(entries, len, 4);
        put(entries, cap, 4);
        put(entries, static_cast<std::uint32_t>(data.size()), 4);
        keys += k + '\0';
        data += v + std::string(cap - v.size(), '\0');
    }
    while (keys.size() % 4) keys.push_back('\0');
    head = std::string("\0PSF", 4);
    put(head, 0x0101, 4);
    put(head, static_cast<std::uint32_t>(20 + entries.size()), 4);
    put(head, static_cast<std::uint32_t>(20 + entries.size() + keys.size()), 4);
    put(head, static_cast<std::uint32_t>(kv.size()), 4);
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << head << entries << keys << data;
}

bool is_live_https(const HostConfig& c) {
    return c.online_host == "thehuntersdream.com" && c.online_scheme == "https" && c.online_verify_tls &&
           c.online_require_account && c.auth_server == "https://thehuntersdream.com";
}

// The play-log period's move from 5 s to 30 (config_version 5), with checks
// of their own so its mutants can run quietly.
int g_pl_fail = 0;
bool g_pl_quiet = false;
void pl_check(bool ok, const char* what) {
    if (!ok) {
        ++g_pl_fail;
        if (!g_pl_quiet) std::printf("FAIL: play log: %s\n", what);
    }
}

// Loads with the per-user file `user`, a bbhost.toml in the working directory
// (`local`, none when empty) and a --config file (`profile`, none when empty).
HostConfig load_layers(const std::string& user, const std::string& local, const std::string& profile) {
    {
        std::ofstream out(g_dir / "config" / "bbhost.toml", std::ios::trunc);
        out << user;
    }
    const fs::path local_file = g_dir / "work" / "bbhost.toml", profile_file = g_dir / "profile.toml";
    std::error_code ec;
    fs::remove(local_file, ec);
    if (!local.empty()) std::ofstream(local_file, std::ios::trunc) << local;
    if (!profile.empty()) std::ofstream(profile_file, std::ios::trunc) << profile;
    std::string arg_config = "--config=" + profile_file.string();
    char arg0[] = "config_test";
    char* argv[] = {arg0, arg_config.data(), nullptr};
    HostConfig c;
    std::string err;
    const bool ok = config_load(profile.empty() ? 1 : 2, argv, &c, &err);
    pl_check(ok, "the config loads");
    fs::remove(local_file, ec);
    return c;
}

int playlog_migration() {
    g_pl_fail = 0;
    const fs::path user = g_dir / "config" / "bbhost.toml";
    // The old example's 5, copied into the per-user file (no version: an
    // older bbhost's): rewritten to 30, the rest of the file as it was.
    HostConfig c = load_layers("[online]\n# How often, in seconds\nplaylog_upload_seconds = 5\nplaylog_sample_ms = 500\n", "", "");
    pl_check(c.playlog_upload_seconds == 30, "a copied 5 reads as 30");
    std::string f = read(user);
    pl_check(f.find("playlog_upload_seconds = 30") != std::string::npos && f.find("playlog_upload_seconds = 5") == std::string::npos,
             "the per-user file is rewritten to 30");
    pl_check(f.find("# How often, in seconds") != std::string::npos && f.find("playlog_sample_ms = 500") != std::string::npos,
             "the rest of the file as it was");
    // Loaded again, nothing changes: the rewrite (and its log line) was once.
    c = load_layers(f, "", "");
    pl_check(c.playlog_upload_seconds == 30 && read(user) == f, "a second start finds nothing to move");
    // The setup window's save writes the version: what it finds then is 30.
    pl_check(config_set_values(user.string(), {{"bbhost", "config_version", "5"}}), "the setup window's save");
    c = load_layers(read(user), "", "");
    pl_check(c.playlog_upload_seconds == 30, "after the setup window's save: still 30");
    // Saved by an older setup window (version 4): the same.
    c = load_layers("[online]\nplaylog_upload_seconds = 5\n\n[bbhost]\nconfig_version = 4\n", "", "");
    pl_check(c.playlog_upload_seconds == 30 && read(user).find("playlog_upload_seconds = 30") != std::string::npos,
             "a 5 at config_version 4 is moved too");
    // From version 5 on a 5 is the player's own.
    c = load_layers("[online]\nplaylog_upload_seconds = 5\n\n[bbhost]\nconfig_version = 5\n", "", "");
    pl_check(c.playlog_upload_seconds == 5 && read(user).find("playlog_upload_seconds = 5") != std::string::npos,
             "a 5 at config_version 5 stays");
    // Any other value is the player's at any version, the file untouched.
    for (const char* v : {"10", "60", "0", "300"}) {
        const std::string text = std::string("[online]\nhost = \"192.168.1.50\"\nscheme = \"http\"\nplaylog_upload_seconds = ") + v + "\n";
        c = load_layers(text, "", "");
        pl_check(c.playlog_upload_seconds == std::atoi(v) && read(user) == text, "another value stays, the file untouched");
    }
    // No key: the new default.
    c = load_layers("[online]\nhost = \"192.168.1.50\"\n", "", "");
    pl_check(c.playlog_upload_seconds == 30, "the default is 30");
    // The example copied beside the exe or in the working directory: set
    // aside (the default stands), that file left as it is.
    c = load_layers("[online]\nhost = \"192.168.1.50\"\n", "[online]\nplaylog_upload_seconds = 5\n", "");
    pl_check(c.playlog_upload_seconds == 30, "a 5 in a bbhost.toml beside the per-user one is set aside");
    c = load_layers("[online]\nhost = \"192.168.1.50\"\n", "[online]\nplaylog_upload_seconds = 15\n", "");
    pl_check(c.playlog_upload_seconds == 15, "a 15 there stays");
    // ... and in a --config profile, unless the profile says version 5.
    c = load_layers("[online]\nhost = \"192.168.1.50\"\n", "", "[online]\nplaylog_upload_seconds = 5\n");
    pl_check(c.playlog_upload_seconds == 30, "a 5 in a --config file is set aside");
    pl_check(read(g_dir / "profile.toml").find("playlog_upload_seconds = 5") != std::string::npos, "the --config file left as it was");
    c = load_layers("[online]\nhost = \"192.168.1.50\"\n", "",
                    "[online]\nplaylog_upload_seconds = 5\n\n[bbhost]\nconfig_version = 5\n");
    pl_check(c.playlog_upload_seconds == 5, "a 5 in a --config file at version 5 stays");
    // A harness keeps its own whatever the files say.
    setenv("BBHOST_PLAYLOG_UPLOAD_SECONDS", "5", 1);
    c = load_layers("[online]\nplaylog_upload_seconds = 5\n", "", "");
    pl_check(c.playlog_upload_seconds == 5, "BBHOST_PLAYLOG_UPLOAD_SECONDS=5 wins");
    setenv("BBHOST_PLAYLOG_UPLOAD_SECONDS", "0", 1);
    c = load_layers("[online]\nplaylog_upload_seconds = 30\n", "", "");
    pl_check(c.playlog_upload_seconds == 0, "BBHOST_PLAYLOG_UPLOAD_SECONDS=0 keeps the game's own");
    unsetenv("BBHOST_PLAYLOG_UPLOAD_SECONDS");
    std::error_code ec;
    fs::remove(g_dir / "profile.toml", ec);
    return g_pl_fail;
}

}  // namespace

extern int g_config_mutant;  // core/config.cpp, built with CONFIG_MUTANTS here

int main() {
    char tmpl[] = "/tmp/bbhost-config-test-XXXXXX";
    if (!mkdtemp(tmpl)) return 1;
    g_dir = tmpl;
    fs::create_directories(g_dir / "config");
    fs::create_directories(g_dir / "work");
    setenv("BBHOST_CONFIG_DIR", (g_dir / "config").c_str(), 1);
    setenv("XDG_DATA_HOME", (g_dir / "data").c_str(), 1);
    // No bbhost.toml in the working directory to layer over the user's.
    if (chdir((g_dir / "work").c_str()) != 0) return 1;

    // A note after a quoted value is not part of it.
    {
        HostConfig c = load("[online]\nhost = \"192.168.1.50\"  # added by bbhost v0.2.0: the server\n"
                            "scheme = \"http\" # a comment\n");
        CHECK(c.online_host == "192.168.1.50");
        CHECK(c.online_scheme == "http");
    }
    // A Windows path keeps its backslashes, a trailing one included.
    {
        HostConfig c = load("[paths]\napp0 = \"C:\\Games\\Bloodborne\\\"  # the dump\n");
        CHECK(c.app0 == "C:\\Games\\Bloodborne\\");
    }
    // The first-start template of older releases: plain http to the live server.
    {
        HostConfig c = load("[online]\nhost = \"thehuntersdream.com\"\nscheme = \"http\"\nverify_tls = false\n"
                            "require_account = false\n# auth_server = \"https://thehuntersdream.com\"\n");
        CHECK(is_live_https(c));
        const std::string f = read(g_dir / "config" / "bbhost.toml");
        CHECK(f.find("scheme = \"https\"") != std::string::npos);
        CHECK(f.find("verify_tls = true") != std::string::npos);
        CHECK(f.find("require_account = true") != std::string::npos);
        CHECK(f.find("\nauth_server = \"https://thehuntersdream.com\"") != std::string::npos);
    }
    // Saved by an older setup window's "Live server".
    {
        HostConfig c = load("[online]\nhost = \"thehuntersdream.com\"\nscheme = \"http\"\nverify_tls = false\n"
                            "require_account = false\nauth_server = \"\"\n\n[bbhost]\nconfig_version = 2\n");
        CHECK(is_live_https(c));
    }
    // No online section at all: the keys bbhost adds, then the live server's.
    {
        HostConfig c = load("[paths]\napp0 = \"PATH-TO-THE-GAME-DUMP\"\n");
        CHECK(is_live_https(c));
    }
    // A server on the player's network: left alone, and a missing scheme is http.
    {
        HostConfig c = load("[online]\nhost = \"192.168.1.50\"\nscheme = \"http\"\nverify_tls = false\n");
        CHECK(c.online_host == "192.168.1.50" && c.online_scheme == "http" && !c.online_verify_tls);
        c = load("[online]\nhost = \"192.168.1.50\"\n");
        CHECK(c.online_scheme == "http");
        CHECK(read(g_dir / "config" / "bbhost.toml").find("scheme = \"http\"") != std::string::npos);
    }
    // The playtest server keeps its own settings.
    {
        HostConfig c = load("[online]\nhost = \"dev.thehuntersdream.com\"\nscheme = \"https\"\nverify_tls = true\n"
                            "require_account = true\nauth_server = \"https://dev.thehuntersdream.com\"\n");
        CHECK(c.online_host == "dev.thehuntersdream.com" && c.auth_server == "https://dev.thehuntersdream.com");
    }
    // Saved by this release's setup window: a choice, not rewritten.
    {
        HostConfig c = load("[online]\nhost = \"thehuntersdream.com\"\nscheme = \"http\"\n\n[bbhost]\nconfig_version = 3\n");
        CHECK(c.online_scheme == "http");
    }

    // The first-start template's player.ime = "auto", which kept the game's
    // text boxes from being typed in: rewritten to "type", in a file the
    // setup window never saved and in one it saved before version 4.
    {
        HostConfig c = load("[player]\nname = \"Hunter\"\n# auto: the dialog is answered with `name` at once.\nime = \"auto\"\n");
        CHECK(c.ime_mode == "type");
        std::string f = read(g_dir / "config" / "bbhost.toml");
        CHECK(f.find("ime = \"type\"") != std::string::npos && f.find("ime = \"auto\"") == std::string::npos);
        CHECK(f.find("# auto: the dialog is answered") != std::string::npos);  // the rest of the file as it was
        c = load("[player]\nime = \"auto\"\n\n[bbhost]\nconfig_version = 3\n");
        CHECK(c.ime_mode == "type");
        // From version 4 on, "auto" was set by the player: kept.
        c = load("[player]\nime = \"auto\"\n\n[bbhost]\nconfig_version = 4\n");
        CHECK(c.ime_mode == "auto");
        // And a harness keeps its own either way.
        setenv("BBHOST_IME", "auto", 1);
        c = load("[player]\nime = \"type\"\n");
        CHECK(c.ime_mode == "auto");
        unsetenv("BBHOST_IME");
    }
    // [video] effect_steps: smooth when the file has no such key (the
    // template ships it commented out and nothing saves it), and an explicit
    // value - 30 among them - kept as written at any config version, the file
    // untouched; BBHOST_SFX_STEP over either.
    {
        load("[player]\nname = \"Hunter\"\n");
        CHECK(config_effect_steps() == "smooth");
        load("[video]\n# whole steps\neffect_steps = 30\n");
        CHECK(config_effect_steps() == "30");
        CHECK(read(g_dir / "config" / "bbhost.toml").find("effect_steps = 30") != std::string::npos);
        for (const char* ver : {"2", "4", "5"}) {
            load(std::string("[video]\neffect_steps = 30\n\n[bbhost]\nconfig_version = ") + ver + "\n");
            CHECK(config_effect_steps() == "30");
            CHECK(read(g_dir / "config" / "bbhost.toml").find("effect_steps = 30") != std::string::npos);
        }
        load("[video]\neffect_steps = \"60\"\n");
        CHECK(config_effect_steps() == "60");
        load("[video]\neffect_steps = \"frame\"\n");
        CHECK(config_effect_steps() == "frame");
        load("[video]\neffect_steps = \"smooth\"\n");
        CHECK(config_effect_steps() == "smooth");
        setenv("BBHOST_SFX_STEP", "60", 1);
        load("[video]\neffect_steps = 30\n");
        CHECK(config_effect_steps() == "60");
        unsetenv("BBHOST_SFX_STEP");
    }
    // [video] frame_clock: the Native frame clock setting's default - off when
    // the file has no such key, true or false as written - and
    // config_frame_clock: the setting (the options file's when it holds one,
    // else that default), BBHOST_FRAME_CLOCK=1 or 0 over it either way, and
    // what decided, for the start line.
    {
        unsetenv("BBHOST_FRAME_CLOCK");
        const char* from = nullptr;
        HostConfig c = load("[player]\nname = \"Hunter\"\n");
        CHECK(!c.frame_clock);
        CHECK(!config_frame_clock(c.frame_clock, false, &from) && std::string(from) == "the default");
        c = load("[video]\nframe_clock = true\n");
        CHECK(c.frame_clock);
        CHECK(config_frame_clock(c.frame_clock, false, &from) && std::string(from) == "bbhost.toml's [video] frame_clock");
        c = load("[video]\nframe_clock = false\n");
        CHECK(!c.frame_clock);
        CHECK(!config_frame_clock(c.frame_clock, false, &from) && std::string(from) == "bbhost.toml's [video] frame_clock");
        // The options file's choice over the file's default, both ways.
        CHECK(config_frame_clock(true, true, &from) && std::string(from) == "the Native frame clock setting");
        c = load("[video]\nframe_clock = true\n");
        CHECK(!config_frame_clock(false, true, &from) && std::string(from) == "the Native frame clock setting");
        // The environment over both, either way; the field stays the file's.
        setenv("BBHOST_FRAME_CLOCK", "1", 1);
        c = load("[video]\nframe_clock = false\n");
        CHECK(!c.frame_clock);
        CHECK(config_frame_clock(false, true, &from) && std::string(from) == "BBHOST_FRAME_CLOCK=1");
        CHECK(config_frame_clock(false, false, &from) && std::string(from) == "BBHOST_FRAME_CLOCK=1");
        setenv("BBHOST_FRAME_CLOCK", "0", 1);
        c = load("[video]\nframe_clock = true\n");
        CHECK(c.frame_clock);
        CHECK(!config_frame_clock(true, true, &from) && std::string(from) == "BBHOST_FRAME_CLOCK=0");
        CHECK(!config_frame_clock(c.frame_clock, false, &from) && std::string(from) == "BBHOST_FRAME_CLOCK=0");
        // Anything else is not a switch: the setting stands.
        setenv("BBHOST_FRAME_CLOCK", "yes", 1);
        CHECK(config_frame_clock(true, true, &from) && std::string(from) == "the Native frame clock setting");
        setenv("BBHOST_FRAME_CLOCK", "", 1);
        CHECK(!config_frame_clock(false, true, &from));
        unsetenv("BBHOST_FRAME_CLOCK");
    }
    // The template a first start writes types in the window.
    {
        const fs::path t = g_dir / "template.toml";
        CHECK(config_write_template(t.string()));
        std::map<std::string, std::string> kv;
        std::string err;
        CHECK(config_parse_toml(t.string(), &kv, &err));
        CHECK(kv["player.ime"] == "\"type\"");
    }

    // The game folder, then an update folder beside it: "-patch", and
    // "-UPDATE", which shadPS4 looks for first. app0 may name either one.
    {
        const fs::path lib = g_dir / "games";
        const std::string base = (lib / "CUSA00900").string(), update = (lib / "CUSA00900-UPDATE").string();
        fs::create_directories(lib / "CUSA00900" / "dvdroot_ps4");
        GameFolders g = config_game_folders(base);
        CHECK(g.base == base && g.update.empty());
        fs::create_directories(lib / "CUSA00900-patch");
        g = config_game_folders(base + "/");
        CHECK(g.base == base && g.update == (lib / "CUSA00900-patch").string());
        fs::create_directories(lib / "CUSA00900-UPDATE" / "dvdroot_ps4");
        g = config_game_folders(base);
        CHECK(g.base == base && g.update == update);
        g = config_game_folders(update);
        CHECK(g.base == base && g.update == update);
        // An update folder with no game folder beside it is the game folder.
        fs::create_directories(lib / "other-UPDATE");
        g = config_game_folders((lib / "other-UPDATE").string());
        CHECK(g.base == (lib / "other-UPDATE").string() && g.update.empty());

        // The version is the update's when it has a param.sfo, else the game folder's.
        write_sfo(lib / "CUSA00900" / "sce_sys" / "param.sfo", "01.00", "gd");
        App0Version v = config_app0_version(base);
        CHECK(v.app_ver == "01.00" && v.category == "gd" && v.update.empty());
        write_sfo(lib / "CUSA00900-UPDATE" / "sce_sys" / "param.sfo", "01.09", "gp");
        v = config_app0_version(base);
        CHECK(v.app_ver == "01.09" && v.category == "gp" && v.update == update);
        CHECK(config_app0_version(update).app_ver == "01.09");

        // Without paths.eboot, the game's own eboot.bin when it is an ELF, the update's first.
        std::ofstream(lib / "CUSA00900" / "eboot.bin", std::ios::binary) << "SCE\0 an encrypted one";
        HostConfig c = load("[paths]\napp0 = \"" + base + "\"\n");
        CHECK(c.eboot.empty());
        std::ofstream(lib / "CUSA00900" / "eboot.bin", std::ios::binary) << "\x7f" "ELF base";
        c = load("[paths]\napp0 = \"" + base + "\"\n");
        CHECK(c.eboot == base + "/eboot.bin");
        std::ofstream(lib / "CUSA00900-UPDATE" / "eboot.bin", std::ios::binary) << "\x7f" "ELF update";
        c = load("[paths]\napp0 = \"" + base + "\"\n");
        CHECK(c.eboot == update + "/eboot.bin");
        c = load("[paths]\napp0 = \"" + base + "\"\neboot = \"" + base + "/eboot.bin\"\n");
        CHECK(c.eboot == base + "/eboot.bin");  // a path given is kept
    }

    // The play log's 5 s from the old example moved to 30 (config_version
    // 5), then four mutants of the move, each of which a check must catch.
    const int pl_fails = playlog_migration();
    g_fail += pl_fails;
    static const char* const kPlMutants[] = {"", "a 5 at version 5 moved too", "the per-user file not rewritten",
                                             "a 5 in another layer not set aside", "BBHOST_PLAYLOG_UPLOAD_SECONDS ignored"};
    int caught = 0;
    g_pl_quiet = true;
    for (int m = 1; m <= 4; ++m) {
        g_config_mutant = m;
        const int f = playlog_migration();
        caught += f > 0;
        std::printf("  play-log mutant %d (%s): %s\n", m, kPlMutants[m], f ? "caught" : "NOT CAUGHT");
    }
    g_config_mutant = 0;
    g_pl_quiet = false;
    std::printf("config_test: the play-log move: %d failures; %d of 4 mutants caught\n", pl_fails, caught);
    if (caught != 4) ++g_fail;

    std::error_code ec;
    fs::remove_all(g_dir, ec);
    if (g_fail) {
        std::printf("%d failed\n", g_fail);
        return 1;
    }
    std::printf("config_test: ok\n");
    return 0;
}
