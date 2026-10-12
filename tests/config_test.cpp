// core/config.cpp: a per-user bbhost.toml as older releases left it, loaded
// by this one - notes after quoted values, the live server's move to https,
// and servers of the player's own left as they are. And the game folder with
// its update in a folder of its own beside it, as shadPS4 keeps one.
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

}  // namespace

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

    CHECK(!load("[dlss]\nmode = \"quality\"\n").dlss_object_motion);
    CHECK(load("[dlss]\nobject_motion = true\n").dlss_object_motion);
    CHECK(!load("[dlss]\nobject_motion = false\n").dlss_object_motion);

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

    std::error_code ec;
    fs::remove_all(g_dir, ec);
    if (g_fail) {
        std::printf("%d failed\n", g_fail);
        return 1;
    }
    std::printf("config_test: ok\n");
    return 0;
}
