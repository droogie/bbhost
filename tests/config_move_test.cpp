// core/config_move.cpp: the per-user settings folder's move on Linux, from
// ~/.config/bbhost to ~/.local/share/bbhost, at a start (config_load) - in a
// temporary folder with a HOME, XDG_CONFIG_HOME and XDG_DATA_HOME of its own,
// checked before every start, so the player's folders are never reached. The
// old folder alone, the new one alone, both with names in each, a later file
// in the old one, the old folder a link (and the new one a link to it), a
// read-only old folder, failed copies (every change undone), files bbhost
// does not own, relative and absolute paths into the old folder, the XDG
// folders, BBHOST_CONFIG_DIR, and the notice's text.
#include "core/config.h"
#include "core/config_move.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
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

fs::path g_root;              // the test's own folder: everything happens in it
fs::path g_home, g_old, g_new;

void write(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary | std::ios::trunc) << text;
}

std::string read(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

bool there(const fs::path& p) {
    std::error_code ec;
    return fs::symlink_status(p, ec).type() != fs::file_type::not_found;
}

bool inside_root(const std::string& p) {
    const std::string r = g_root.string() + "/";
    return p.compare(0, r.size(), r) == 0;
}

// Every folder a start could reach is in the test's own: else stop before
// anything is moved.
void guard() {
    const char* home = std::getenv("HOME");
    bool ok = (!home || inside_root(home)) && inside_root(config_user_dir());
    for (const char* v : {"XDG_CONFIG_HOME", "XDG_DATA_HOME", "BBHOST_CONFIG_DIR"}) {
        const char* e = std::getenv(v);
        if (e && *e && !inside_root(e)) ok = false;
    }
    if (!ok) {
        std::printf("FAIL: a folder outside %s would be reached; stopping\n", g_root.c_str());
        std::exit(1);
    }
}

// A case's own HOME, the XDG folders and BBHOST_CONFIG_DIR unset (the ~
// defaults), and a process that has not started yet.
void fresh(const std::string& name) {
    config_move_test_reset();
    g_home = g_root / name;
    std::error_code ec;
    fs::remove_all(g_home, ec);
    fs::create_directories(g_home);
    setenv("HOME", g_home.c_str(), 1);
    unsetenv("XDG_CONFIG_HOME");
    unsetenv("XDG_DATA_HOME");
    unsetenv("BBHOST_CONFIG_DIR");
    g_old = g_home / ".config" / "bbhost";
    g_new = g_home / ".local" / "share" / "bbhost";
    guard();
}

// A start: config_load as main() calls it, once a process; `again` is the
// next start (a new process).
HostConfig start(bool again = false) {
    if (again) {
        // The hooks are the case's own: kept.
        const bool copy = g_config_move_test_copy, no_remove = g_config_move_test_no_remove;
        const int fail = g_config_move_test_fail_copy;
        config_move_test_reset();
        g_config_move_test_copy = copy;
        g_config_move_test_no_remove = no_remove;
        g_config_move_test_fail_copy = fail;
    }
    guard();
    char arg0[] = "config_move_test";
    char* argv[] = {arg0, nullptr};
    HostConfig c;
    std::string err;
    CHECK(config_load(1, argv, &c, &err));
    return c;
}

// A config the start leaves as it is: the keys bbhost adds are there.
std::string cfg(const std::string& name, const std::string& more = {}) {
    return "# the player's own comment\n[online]\nhost = \"127.0.0.1\"\nscheme = \"http\"\n\n[player]\nname = \"" + name + "\"\n" + more;
}

std::string today() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
    return buf;
}

std::vector<std::string> names(const fs::path& dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string n = it->path().filename().string();
        if (n.rfind(".nfs", 0) != 0) out.push_back(n);  // NFS's own, for a file removed while open
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool has(const std::vector<std::string>& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

// The notice's whole text, heading and lines.
std::string notice(const ConfigMove& m) {
    std::string s = config_move_heading(m) + "\n";
    for (const ConfigMoveLine& l : config_move_lines(m)) s += (l.warn ? "! " : "  ") + l.text + "\n";
    return s;
}
bool says(const ConfigMove& m, const std::string& what) { return notice(m).find(what) != std::string::npos; }

int mode_of(const fs::path& p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 ? static_cast<int>(st.st_mode & 07777) : -1;
}

}  // namespace

int main() {
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    std::string tmpl = (ec ? fs::path("/tmp") : base).string() + "/bbhost-config-move-test-XXXXXX";
    if (!mkdtemp(tmpl.data())) return 1;
    g_root = tmpl;
    fs::create_directories(g_root / "work");
    // No bbhost.toml in the working directory to layer over the per-user one.
    if (chdir((g_root / "work").c_str()) != 0) return 1;
    const std::string date = today();

    // The old folder alone, as every Linux player has it: what bbhost keeps
    // moves (renamed), the player's own stays, one record and one notice; the
    // next start finds nothing more, and Dismiss forgets it.
    {
        fresh("old-only");
        write(g_old / "bbhost.toml", cfg("Old only"));
        write(g_old / "bbhost-options.toml", "[options]\nversion = \"5\"\nframe_cap = \"60\"\n");
        fs::permissions(g_old / "bbhost-options.toml", fs::perms::owner_read | fs::perms::owner_write);
        write(g_old / "bbhost-options-kit.toml", "[options]\nversion = \"5\"\n");
        write(g_old / "plugins" / "quake.so", "a plugin");
        write(g_old / "plugins" / "quake.so.sig", "its signature");
        write(g_old / "patches" / "mine.toml", "[patch]\nname = \"mine\"\n");
        write(g_old / "notes.txt", "the player's notes");
        write(g_old / "stuff" / "file.txt", "the player's folder");
        const std::string toml = read(g_old / "bbhost.toml");
        HostConfig c = start();
        CHECK(c.player_name == "Old only");
        CHECK(config_user_dir() == g_new.string());
        CHECK(config_user_file() == (g_new / "bbhost.toml").string());
        CHECK(c.config_path == (g_new / "bbhost.toml").string());
        CHECK(read(g_new / "bbhost.toml") == toml);
        CHECK(read(g_new / "bbhost-options.toml").find("frame_cap = \"60\"") != std::string::npos);
        CHECK(mode_of(g_new / "bbhost-options.toml") == 0600);
        CHECK(there(g_new / "bbhost-options-kit.toml") && there(g_new / "plugins" / "quake.so") &&
              there(g_new / "plugins" / "quake.so.sig") && there(g_new / "patches" / "mine.toml"));
        CHECK(names(g_old) == (std::vector<std::string>{"notes.txt", "stuff"}));
        CHECK(read(g_old / "notes.txt") == "the player's notes" && read(g_old / "stuff" / "file.txt") == "the player's folder");
        std::vector<ConfigMove> shown = config_moves_to_show();
        CHECK(shown.size() == 1);
        if (shown.size() == 1) {
            const ConfigMove& m = shown[0];
            CHECK(m.failed.empty() && m.from == g_old.string() && m.to == g_new.string() && m.link.empty());
            CHECK(m.moved == (std::vector<std::string>{"bbhost-options-kit.toml", "bbhost-options.toml", "bbhost.toml", "patches", "plugins"}));
            CHECK(m.left == (std::vector<std::string>{"notes.txt", "stuff/"}));
            CHECK(m.beside.empty() && m.same.empty() && m.kept.empty() && m.paths.empty());
            CHECK(config_move_heading(m) == "Your settings moved to ~/.local/share/bbhost");
            CHECK(says(m, "Moved from there: bbhost-options-kit.toml, bbhost-options.toml, bbhost.toml, patches and plugins."));
            CHECK(says(m, "Left in ~/.config/bbhost, as they are not bbhost's: notes.txt and stuff/."));
            CHECK(says(m, "An older bbhost still looks in ~/.config/bbhost and does not find them; anything it leaves there is "
                          "moved here at the next start."));
            CHECK(notice(m).find("! ") == std::string::npos);  // nothing to worry about
            const std::string log = config_move_log_line(m);
            CHECK(log.find("config: the settings moved from " + g_old.string() + " to " + g_new.string()) == 0);
            CHECK(log.find("left (not bbhost's): notes.txt and stuff/") != std::string::npos);
        }
        // The next start: nothing more to move, the notice still there.
        c = start(true);
        CHECK(c.player_name == "Old only" && config_moves_to_show().size() == 1);
        CHECK(names(g_old) == (std::vector<std::string>{"notes.txt", "stuff"}));
        config_moves_dismiss();
        CHECK(config_moves_to_show().empty() && !there(g_new / kConfigMoveRecord));
        start(true);
        CHECK(config_moves_to_show().empty());
    }

    // The old folder emptied of all it held: gone. A move across file
    // systems (every rename refused) copies, compares and removes, and keeps
    // the files' modes and times.
    {
        fresh("across");
        write(g_old / "bbhost.toml", cfg("Across"));
        write(g_old / "bbhost-options.toml", "[options]\nversion = \"5\"\n");
        fs::permissions(g_old / "bbhost-options.toml", fs::perms::owner_read | fs::perms::owner_write);
        write(g_old / "plugins" / "lohit.so", std::string(200000, 'x'));
        const auto when = fs::last_write_time(g_old / "plugins" / "lohit.so");
        g_config_move_test_copy = true;
        HostConfig c = start();
        CHECK(c.player_name == "Across");
        CHECK(!there(g_old));
        CHECK(read(g_new / "plugins" / "lohit.so") == std::string(200000, 'x'));
        CHECK(fs::last_write_time(g_new / "plugins" / "lohit.so") == when);
        CHECK(mode_of(g_new / "bbhost-options.toml") == 0600);
        for (const std::string& n : names(g_new)) CHECK(n.find(".bbhost-move") == std::string::npos);
        const std::vector<ConfigMove> shown = config_moves_to_show();
        CHECK(shown.size() == 1 && shown[0].moved.size() == 3 && shown[0].kept.empty());
    }

    // The new folder alone: read from there, nothing moved or recorded. An old
    // folder with nothing of bbhost's in it is left as it is, untold.
    {
        fresh("new-only");
        write(g_new / "bbhost.toml", cfg("New only"));
        HostConfig c = start();
        CHECK(c.player_name == "New only" && config_moves_to_show().empty());
        CHECK(!there(g_old) && !there(g_new / kConfigMoveRecord));
        write(g_old / "notes.txt", "not bbhost's");
        start(true);
        CHECK(config_moves_to_show().empty() && read(g_old / "notes.txt") == "not bbhost's");
    }

    // Both folders, with names in each: the new folder's stay in use, the old
    // ones go beside them under a dated name (inside plugins/ too, and a
    // dated plugin is not a plugin bbhost loads), a file the same as the one
    // there is taken as it, and a name only the old folder has moves. Both
    // paths are in the notice. Then an older bbhost writes bbhost.toml in the
    // old folder again: moved beside the same way, and told again.
    {
        fresh("both");
        write(g_new / "bbhost.toml", cfg("New"));
        write(g_new / "bbhost-options.toml", "[options]\nversion = \"5\"\n");
        write(g_new / "plugins" / "quake.so", "the new quake");
        write(g_old / "bbhost.toml", cfg("Old"));
        write(g_old / "bbhost-options.toml", "[options]\nversion = \"5\"\n");
        write(g_old / "plugins" / "quake.so", "the old quake");
        write(g_old / "plugins" / "mutators.so", "mutators");
        HostConfig c = start();
        CHECK(c.player_name == "New");
        const std::string as = "bbhost.toml.from-config-" + date, plugin_as = "quake.so.from-config-" + date;
        CHECK(read(g_new / "bbhost.toml") == cfg("New"));
        CHECK(read(g_new / as) == cfg("Old"));
        CHECK(read(g_new / "plugins" / "quake.so") == "the new quake");
        CHECK(read(g_new / "plugins" / plugin_as) == "the old quake");
        CHECK(plugin_as.compare(plugin_as.size() - 3, 3, ".so") != 0);
        CHECK(read(g_new / "plugins" / "mutators.so") == "mutators");
        CHECK(read(g_new / "bbhost-options.toml") == "[options]\nversion = \"5\"\n");
        CHECK(!there(g_old));
        std::vector<ConfigMove> shown = config_moves_to_show();
        CHECK(shown.size() == 1);
        if (shown.size() == 1) {
            const ConfigMove& m = shown[0];
            CHECK(m.beside.size() == 2 && m.same == std::vector<std::string>{"bbhost-options.toml"} &&
                  m.moved == std::vector<std::string>{"plugins/mutators.so"});
            CHECK(says(m, "! ~/.local/share/bbhost/bbhost.toml was already there and stays in use. The one from ~/.config/bbhost is "
                          "beside it: ~/.local/share/bbhost/" + as + "."));
            CHECK(says(m, "~/.local/share/bbhost/plugins/quake.so was already there"));
            CHECK(says(m, "The same as the ones already in ~/.local/share/bbhost, so those are kept: bbhost-options.toml."));
            CHECK(says(m, "Moved from there: plugins."));
        }
        write(g_old / "bbhost.toml", cfg("Older again"));
        c = start(true);
        CHECK(c.player_name == "New");
        CHECK(read(g_new / (as + "-2")) == cfg("Older again"));
        shown = config_moves_to_show();
        CHECK(shown.size() == 2 && shown.back().beside.size() == 1 && shown.back().beside[0].as == as + "-2");
        config_moves_dismiss();
        CHECK(config_moves_to_show().empty());
    }

    // The old folder a link (a folder kept elsewhere): nothing is taken out
    // of it - its files are copied, the originals and the link stay - and the
    // next start leaves them alone. One changed there later is moved beside
    // the copy and told. A new folder that is a link to the old one is one
    // folder: nothing moves.
    {
        fresh("old-link");
        const fs::path real = g_home / "dotfiles" / "bbhost";
        write(real / "bbhost.toml", cfg("Linked"));
        write(real / "bbhost-options.toml", "[options]\nversion = \"5\"\n");
        write(real / "notes.txt", "mine");
        fs::create_directories(g_old.parent_path());
        fs::create_directory_symlink(real, g_old);
        HostConfig c = start();
        CHECK(c.player_name == "Linked");
        CHECK(fs::is_symlink(g_old));
        CHECK(names(real) == (std::vector<std::string>{"bbhost-options.toml", "bbhost.toml", "notes.txt"}));
        CHECK(read(real / "bbhost.toml") == cfg("Linked") && read(g_new / "bbhost.toml") == cfg("Linked"));
        std::vector<ConfigMove> shown = config_moves_to_show();
        CHECK(shown.size() == 1);
        if (shown.size() == 1) {
            CHECK(shown[0].link == real.string() && shown[0].kept.size() == 2);
            CHECK(says(shown[0], "Copied from there: bbhost-options.toml and bbhost.toml."));
            CHECK(says(shown[0], "! ~/.config/bbhost is a link to ~/dotfiles/bbhost, so nothing was taken out of it"));
            CHECK(says(shown[0], "An older bbhost still reads the originals in ~/.config/bbhost, which this one no longer changes."));
        }
        config_moves_dismiss();
        CHECK(there(g_new / kConfigMoveRecord));  // the originals it leaves alone
        c = start(true);
        CHECK(config_moves_to_show().empty() && c.player_name == "Linked");
        CHECK(names(g_new) == (std::vector<std::string>{"bbhost-moved.txt", "bbhost-options.toml", "bbhost.toml", "data"}));
        write(real / "bbhost.toml", cfg("Linked, changed later"));
        c = start(true);
        CHECK(c.player_name == "Linked");
        CHECK(read(g_new / ("bbhost.toml.from-config-" + date)) == cfg("Linked, changed later"));
        CHECK(read(real / "bbhost.toml") == cfg("Linked, changed later"));
        shown = config_moves_to_show();
        CHECK(shown.size() == 1 && shown[0].beside.size() == 1 && shown[0].kept.size() == 1);
        start(true);
        CHECK(config_moves_to_show().size() == 1);  // told once

        fresh("new-link");
        write(g_old / "bbhost.toml", cfg("One folder"));
        fs::create_directories(g_new.parent_path());
        fs::create_directory_symlink(g_old, g_new);
        c = start();
        CHECK(c.player_name == "One folder" && config_moves_to_show().empty());
        CHECK(names(g_old) == (std::vector<std::string>{"bbhost.toml", "data"}) && !there(g_new / kConfigMoveRecord));
    }

    // A read-only old folder: copied, the originals stay (the notice says
    // why), and later starts leave them alone. Where a mode does not stop
    // this test (run as root, a file system that ignores it), the folder's
    // refusals are the hook's.
    {
        fresh("read-only");
        write(g_old / "bbhost.toml", cfg("Read-only"));
        write(g_old / "bbhost-options.toml", "[options]\nversion = \"5\"\n");
        write(g_old / "plugins" / "quake.so", "a plugin");
        // Whether a folder's mode keeps a file in it here: one renamed out of
        // a read-only folder of its own.
        const fs::path probe = g_home / "probe";
        write(probe / "file", "x");
        fs::permissions(probe, fs::perms::owner_read | fs::perms::owner_exec);
        const bool ignored = ::rename((probe / "file").c_str(), (g_home / "probe-file").c_str()) == 0;
        fs::permissions(probe, fs::perms::owner_all);
        fs::remove_all(probe, ec);
        fs::remove(g_home / "probe-file", ec);
        if (ignored) {
            g_config_move_test_no_remove = true;
            std::printf("config_move_test: a folder's mode does not keep its files here; the read-only folder is the hook's\n");
        } else {
            fs::permissions(g_old / "plugins", fs::perms::owner_read | fs::perms::owner_exec);
            fs::permissions(g_old, fs::perms::owner_read | fs::perms::owner_exec);
        }
        HostConfig c = start();
        CHECK(c.player_name == "Read-only");
        CHECK(config_user_dir() == g_new.string());
        CHECK(read(g_new / "bbhost.toml") == cfg("Read-only") && read(g_new / "plugins" / "quake.so") == "a plugin");
        CHECK(names(g_old) == (std::vector<std::string>{"bbhost-options.toml", "bbhost.toml", "plugins"}));
        CHECK(there(g_old / "plugins" / "quake.so"));
        std::vector<ConfigMove> shown = config_moves_to_show();
        CHECK(shown.size() == 1);
        if (shown.size() == 1) {
            CHECK(shown[0].kept.size() == 3 && shown[0].moved.size() == 3);
            CHECK(says(shown[0], "! Copied, but still in ~/.config/bbhost as well (Permission denied): bbhost-options.toml, "
                                 "bbhost.toml and plugins. bbhost uses the copies; the ones there can be deleted."));
        }
        c = start(true);
        CHECK(config_moves_to_show().size() == 1 && c.player_name == "Read-only");
        CHECK(!there(g_new / ("bbhost.toml.from-config-" + date)));
        fs::permissions(g_old, fs::perms::owner_all);
        fs::permissions(g_old / "plugins", fs::perms::owner_all);
    }

    // A failed copy: everything the start did is undone - the renames put
    // back, the copies and folders it made removed - the old folder is used
    // as it is, the notice says so, nothing is recorded, and the next start
    // moves it all.
    {
        fresh("failed-copy");
        write(g_old / "bbhost.toml", cfg("Stays"));
        write(g_old / "bbhost-options.toml", "[options]\nversion = \"5\"\n");
        write(g_old / "plugins" / "quake.so", "a plugin");
        write(g_old / "plugins" / "lohit.so", "another");
        g_config_move_test_copy = true;
        g_config_move_test_fail_copy = 3;
        HostConfig c = start();
        CHECK(c.player_name == "Stays");
        CHECK(config_user_dir() == g_old.string() && config_move_fallback() == g_old.string());
        CHECK(c.config_path == (g_old / "bbhost.toml").string());
        CHECK(read(g_old / "bbhost.toml") == cfg("Stays") && read(g_old / "plugins" / "quake.so") == "a plugin" &&
              read(g_old / "plugins" / "lohit.so") == "another");
        CHECK(!there(g_new / "bbhost.toml") && !there(g_new / "bbhost-options.toml") && !there(g_new / "plugins"));
        for (const std::string& n : names(g_new)) CHECK(n == "data");
        CHECK(!there(g_new / kConfigMoveRecord));
        std::vector<ConfigMove> shown = config_moves_to_show();
        CHECK(shown.size() == 1);
        if (shown.size() == 1) {
            CHECK(shown[0].failed == "copying plugins/lohit.so failed: Input/output error");
            CHECK(config_move_heading(shown[0]) == "Your settings could not be moved");
            CHECK(says(shown[0], "! bbhost keeps its settings in ~/.local/share/bbhost now, but could not move yours from "
                                 "~/.config/bbhost: copying plugins/lohit.so failed: Input/output error. This start uses them "
                                 "where they are; the next one tries again."));
            CHECK(config_move_log_line(shown[0]).find("this start uses " + g_old.string()) != std::string::npos);
        }
        // Renamed first, then a copy that fails (plugins/ a link, copied):
        // the renames go back.
        fresh("failed-after-renames");
        write(g_old / "bbhost.toml", cfg("Put back"));
        write(g_old / "bbhost-options.toml", "[options]\nversion = \"5\"\n");
        write(g_home / "elsewhere" / "quake.so", "a plugin");
        fs::create_directory_symlink(g_home / "elsewhere", g_old / "plugins");
        g_config_move_test_fail_copy = 1;
        c = start();
        CHECK(c.player_name == "Put back" && config_user_dir() == g_old.string());
        CHECK(read(g_old / "bbhost.toml") == cfg("Put back") && there(g_old / "bbhost-options.toml") && fs::is_symlink(g_old / "plugins"));
        CHECK(!there(g_new / "bbhost.toml") && !there(g_new / "bbhost-options.toml") && !there(g_new / "plugins"));
        // The next start, nothing failing: moved, the link's plugin copied and the link left.
        g_config_move_test_fail_copy = 0;
        c = start(true);
        CHECK(c.player_name == "Put back" && config_user_dir() == g_new.string());
        CHECK(read(g_new / "plugins" / "quake.so") == "a plugin" && fs::is_symlink(g_old / "plugins"));
        shown = config_moves_to_show();
        CHECK(shown.size() == 1 && shown[0].failed.empty());
        if (shown.size() == 1)
            CHECK(says(shown[0], "! Links in ~/.config/bbhost, copied and left as they are: plugins. bbhost uses the copies."));
    }

    // The moved bbhost.toml's paths: one relative to the old folder (its
    // file's) is written out in full, the same place; one that points into
    // the old folder still works, since nothing of the player's moved.
    {
        fresh("paths");
        write(g_old / "eboot.bin", "\x7f" "ELF");
        fs::create_directories(g_old / "mods");
        fs::create_directories(g_old / "saves");
        write(g_old / "bbhost.toml", cfg("Paths", "\n[paths]\neboot = \"eboot.bin\"\nmods = \"./mods\"\ndata = \"" +
                                                       (g_old / "saves").string() + "\"\n"));
        HostConfig c = start();
        CHECK(c.eboot == (g_old / "eboot.bin").string());
        CHECK(c.mods == (g_old / "mods").string());
        CHECK(c.data == (g_old / "saves").string());
        const std::string f = read(g_new / "bbhost.toml");
        CHECK(f.find("eboot = \"" + (g_old / "eboot.bin").string() + "\"") != std::string::npos);
        CHECK(f.find("# the player's own comment") != std::string::npos);
        CHECK(names(g_old) == (std::vector<std::string>{"eboot.bin", "mods", "saves"}));
        std::vector<ConfigMove> shown = config_moves_to_show();
        CHECK(shown.size() == 1 && shown[0].paths.size() == 3);
        if (shown.size() == 1) {
            CHECK(says(shown[0], "paths.eboot in bbhost.toml was relative to ~/.config/bbhost, so it now says the same place in full: "
                                 "~/.config/bbhost/eboot.bin."));
            CHECK(says(shown[0], "paths.mods in bbhost.toml was relative"));
            CHECK(says(shown[0], "paths.data in bbhost.toml points into ~/.config/bbhost and still works: nothing of yours was moved "
                                 "from there."));
            CHECK(says(shown[0], "Left in ~/.config/bbhost, as they are not bbhost's: eboot.bin, mods/ and saves/."));
        }
    }

    // $XDG_CONFIG_HOME/bbhost to $XDG_DATA_HOME/bbhost; the two the same
    // folder is one folder.
    {
        fresh("xdg");
        setenv("XDG_CONFIG_HOME", (g_home / "xc").c_str(), 1);
        setenv("XDG_DATA_HOME", (g_home / "xd").c_str(), 1);
        guard();
        write(g_home / "xc" / "bbhost" / "bbhost.toml", cfg("XDG"));
        HostConfig c = start();
        CHECK(c.player_name == "XDG" && config_user_dir() == (g_home / "xd" / "bbhost").string());
        CHECK(there(g_home / "xd" / "bbhost" / "bbhost.toml") && !there(g_home / "xc" / "bbhost"));
        CHECK(c.data == (g_home / "xd" / "bbhost" / "data").string());
        CHECK(config_moves_to_show().size() == 1 && !there(g_old) && !there(g_new));

        fresh("xdg-same");
        setenv("XDG_CONFIG_HOME", (g_home / "x").c_str(), 1);
        setenv("XDG_DATA_HOME", (g_home / "x").c_str(), 1);
        guard();
        write(g_home / "x" / "bbhost" / "bbhost.toml", cfg("Same folder"));
        c = start();
        CHECK(c.player_name == "Same folder" && config_moves_to_show().empty());
    }

    // No home for the settings (neither XDG_DATA_HOME nor HOME): they stay
    // where they were, never beside the executable.
    {
        fresh("no-home");
        setenv("XDG_CONFIG_HOME", (g_home / "xc").c_str(), 1);
        write(g_home / "xc" / "bbhost" / "bbhost.toml", cfg("Nowhere else", "\n[paths]\ndata = \"" + (g_home / "data").string() + "\"\n"));
        unsetenv("HOME");
        guard();
        HostConfig c = start();
        CHECK(c.player_name == "Nowhere else" && config_user_dir() == (g_home / "xc" / "bbhost").string());
        CHECK(config_moves_to_show().empty() && there(g_home / "xc" / "bbhost" / "bbhost.toml"));
        setenv("HOME", g_home.c_str(), 1);
    }

    // A folder of its own (BBHOST_CONFIG_DIR): read from there, the old
    // folder left as it is, nothing told.
    {
        fresh("config-dir");
        write(g_old / "bbhost.toml", cfg("Old"));
        write(g_home / "own" / "bbhost.toml", cfg("Own"));
        setenv("BBHOST_CONFIG_DIR", (g_home / "own").c_str(), 1);
        guard();
        HostConfig c = start();
        CHECK(c.player_name == "Own" && config_user_dir() == (g_home / "own").string());
        CHECK(read(g_old / "bbhost.toml") == cfg("Old") && !there(g_new / "bbhost.toml") && !there(g_new / kConfigMoveRecord));
        CHECK(config_moves_to_show().empty());
        unsetenv("BBHOST_CONFIG_DIR");
    }

    // The record survives a reader that knows nothing of a line's kind, and
    // a move with nothing in it is no move.
    {
        fresh("record");
        write(g_old / "bbhost.toml", cfg("Record"));
        start();
        std::string r = read(g_new / kConfigMoveRecord);
        CHECK(r.find("move\t") != std::string::npos && r.find("moved\tbbhost.toml\n") != std::string::npos);
        write(g_new / kConfigMoveRecord, r + "something-new\twhat a later bbhost writes\n");
        start(true);
        const std::vector<ConfigMove> shown = config_moves_to_show();
        CHECK(shown.size() == 1 && shown[0].moved == std::vector<std::string>{"bbhost.toml"});
        const ConfigMove none = config_move_settings((g_home / "nothing").string(), g_new.string());
        CHECK(none.failed.empty() && none.moved.empty());
    }

    fs::remove_all(g_root, ec);
    if (g_fail) {
        std::printf("config_move_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("config_move_test: ok\n");
    return 0;
}
