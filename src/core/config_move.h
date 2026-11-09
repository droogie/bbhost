#pragma once

// The per-user settings folder's move on Linux. bbhost kept its settings in
// $XDG_CONFIG_HOME/bbhost or ~/.config/bbhost; they are kept in
// $XDG_DATA_HOME/bbhost or ~/.local/share/bbhost now, beside the data folder
// (core/config.h, config_user_dir). At every start, before anything reads the
// configuration (config_load), what bbhost keeps in the old folder -
// bbhost.toml, the F10 settings' bbhost-options*.toml, plugins/ and patches/ -
// goes to the new one, and the setup window says what moved until the player
// dismisses it (host/launcher.cpp). Anything else there is the player's and
// stays. Windows' %APPDATA%\bbhost and a BBHOST_CONFIG_DIR do not move.
//
// Nothing is overwritten or lost:
//   - each is renamed into the new folder where it can be, else copied,
//     compared byte for byte with its original and only then the original
//     removed;
//   - a name the new folder has already stays as it is there, and the old one
//     goes beside it as <name>.from-config-<date> (a folder's contents the
//     same way, one by one; a file the same as the one there is taken as that
//     one);
//   - through a link (the old folder, or an entry in it, a symbolic link)
//     nothing is removed: what it holds is copied and the originals stay, as
//     do originals the old folder will not give up (read-only);
//   - when anything cannot be put in the new folder, everything this start
//     moved goes back, and the start uses the old folder as it is.
// The originals that stay are remembered by their size and time, so later
// starts leave them alone. One that changes after, or a file an older bbhost
// writes there later, is moved the same way and told again.

#include <string>
#include <vector>

struct ConfigMove {
    std::string when;      // local time, "2026-11-06 14:30"
    std::string from, to;  // the old folder and the new one
    std::string link;      // the old folder is a symbolic link to this ("" when not)
    // The names below are relative to `from` (and to `to`).
    std::vector<std::string> moved;  // in `to` now, under their own names
    struct Beside {
        std::string name, as;  // `to` had `name` already, still in use; the old one is `as` there
    };
    std::vector<Beside> beside;
    std::vector<std::string> same;  // the same as `to`'s of that name: the old copy removed
    struct Kept {
        std::string name, why;  // copied, and the original still in `from`: why
    };
    std::vector<Kept> kept;
    std::vector<std::string> left;  // not bbhost's: left in `from` (a folder with a trailing /)
    struct Path {
        std::string key, value;      // a [paths] key of the moved bbhost.toml that points into `from`
        bool made_absolute = false;  // it was relative to `from`: written out in full, the same place
    };
    std::vector<Path> paths;
    std::string failed;  // nothing moved, and why ("" when it did)
};

// The move (config_load, once a process): what bbhost keeps in `from` into
// `to`, a record of it in `to` (kConfigMoveRecord, the setup window's notice)
// and one log line. What happened; a start with nothing to move returns every
// list empty, and nothing is recorded or logged.
ConfigMove config_move_settings(const std::string& from, const std::string& to);
// config_move_settings at start, once a process, unless the two are one
// folder (a link from one to the other). When it fails, config_move_fallback()
// names `from` for the rest of the run: config_user_dir() is that folder.
void config_move_at_start(const std::string& from, const std::string& to);
std::string config_move_fallback();
// What the setup window has to tell: the moves the record holds that are not
// dismissed yet, and this start's own failure (not recorded: the next start
// tries again). Empty when nothing moved, and on Windows.
std::vector<ConfigMove> config_moves_to_show();
// The notice dismissed: the record forgets the moves (not the originals it
// keeps track of).
void config_moves_dismiss();
// The record's name in the settings folder.
inline constexpr char kConfigMoveRecord[] = "bbhost-moved.txt";

// The notice's text for one move: a heading, then its lines - `warn` the ones
// the player should look at. Paths in the home folder are written ~/...
struct ConfigMoveLine {
    std::string text;
    bool warn = false;
};
std::string config_move_heading(const ConfigMove& m);
std::vector<ConfigMoveLine> config_move_lines(const ConfigMove& m);
// The log's line for one move (full paths).
std::string config_move_log_line(const ConfigMove& m);

#ifdef CONFIG_MOVE_TEST
// tests/config_move_test.cpp: every rename refused as across file systems
// (the copies are used), the nth copy (1 = the first) failing, every removal
// refused (the read-only folder for a test run as root), and the once a
// process forgotten.
extern bool g_config_move_test_copy;
extern int g_config_move_test_fail_copy;
extern bool g_config_move_test_no_remove;
void config_move_test_reset();
#endif
