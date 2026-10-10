#pragma once

// bbhost.toml: a small TOML subset ([section], key = "string" | integer |
// true/false, # comments). Values are addressed as "section.key".

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct HostConfig {
    std::string app0;          // paths.app0: the /app0 dump (dvdroot_ps4 inside)
    std::string data;          // paths.data: /data, saves, caches
    std::string tmp;           // paths.tmp
    std::string mods;          // paths.mods: the asset overlay, resolved ahead of /app0
    std::string eboot;         // paths.eboot
    std::string online_host;   // online.host: replaces *.scej-network.jp in game URLs
    std::string online_scheme; // online.scheme: "http"/"https" forced for those hosts ("" keeps)
    bool online_verify_tls = false;  // online.verify_tls
    std::string online_id;     // online.online_id
    bool online_require_account = false;  // online.require_account: signed out until the F10 screen has a token
    std::string auth_server;   // online.auth_server: where /auth/* goes (an https origin in production); default np_server
    std::string np_server;     // online.np_server: the private server's API base for the host's own calls
                               // (default "<scheme>://<host>:18671")
    std::string p2p_addr;      // online.p2p_addr: the address other players reach this host at (default 127.0.0.1)
    std::string stun_server;   // online.stun_server: "host[:port]" the P2P port asks for its reflexive address; "" = the server's host on 3478, "off" = never
    int p2p_port = 9307;       // online.p2p_port: this host's P2P UDP port; two instances on one machine differ here
    int playlog_upload_seconds = 5;   // online.playlog_upload_seconds: how often the game sends its play log (the game's own: 300; 0 keeps it)
    int playlog_sample_ms = 500;      // online.playlog_sample_ms: how often it logs the player's position (the game's own: 1500; 0 keeps it)
    int sign_timeout_seconds = 30;    // online.sign_timeout_seconds: how long a sign being answered may take (the game's own: 180; 0 keeps it)
    // [update] (host/updater.h): whether to look for a newer release at start
    // (-1: only a release build does), where the releases are, and a file
    // holding a GitHub token for a private repository's.
    int update_check = -1;           // update.check: true/false
    std::string update_source;       // update.source: a GitHub "releases/latest" API URL
    std::string update_token_file;   // update.token_file
    std::string player_name;   // player.name: what the name box starts from, or receives with "auto" (default online_id)
    std::string ime_mode;      // player.ime / BBHOST_IME: "type" (default: type in the window) or "auto" (the name is player.name at once)
    int width = 1920;          // video.width
    int height = 1080;         // video.height
    int fps_cap = 30;          // video.fps_cap
    std::string resolution;    // video.resolution: the Resolution setting's default entry ("1280x800"); F10's file has the last word
    std::string window_mode;   // video.window_mode: the Window mode setting's default ("Windowed" or "Fullscreen")
    std::string model_detail;  // video.model_detail: the Model detail setting's default ("Full", "Normal", "Low", "Lowest")
    int vblank_hz = 0;         // video.vblank_hz: the flip clock's rate; 0 = the display's own
    bool headless = false;     // video.headless / --headless / BBHOST_HEADLESS=1
    // [dlss]: output dimensions are independent of the game's render size.
    // Zero uses the selected game resolution. Frame generation affects only
    // presentation; it must not raise the game's simulation rate.
    std::string dlss_mode = "off"; // off, dlaa, quality, balanced, performance, ultra_performance
    int dlss_output_width = 0;
    int dlss_output_height = 0;
    std::string upscaler_backend = "dlss"; // dlss, fsr3, fsr4
    std::string frame_generation_backend = "dlss"; // dlss, fsr3
    bool dlss_frame_generation = false;
    int dlss_fg_factor = 2;
    bool dlss_object_motion = false; // animated mesh vectors; optional extra GPU work
    bool skip_intro = false;   // startup.skip_intro / BBHOST_SKIP_INTRO=0|1: the three company logos only
    // loading.quick_reentry / BBHOST_QUICK_REENTRY=0|1: a load after a death or
    // lamp travel ends when the area is ready, not after the game's 12-second
    // minimum (engine/loading.h).
    bool quick_reentry = true;
    // streaming.all_post_processors / BBHOST_ALL_POST_PROCESSORS=0|1: all three
    // resource post-processors in play, not two (patches/ingame-post-processors.toml).
    bool all_post_processors = false;
    // world.change_appearance / BBHOST_CHANGE_APPEARANCE=0|1: the Hunter's
    // Dream mirror opens the appearance editor (engine/change_appearance.h).
    bool change_appearance = true;
    // world.rebirth / BBHOST_REBIRTH=0|1: with the Yharnam Stone, the Altar of
    // Despair offers "Rebirth in the Nightmare" - back to the origin's level and
    // attributes, the echoes returned, the level-up menu (engine/rebirth.h).
    bool rebirth = true;
    // world.five_players / BBHOST_FIVE_PLAYERS=0|1: Mensis, Mergo's Loft, the
    // Nightmare Frontier and the Old Hunters' areas take two invaders, as the
    // chalice dungeons do (engine/five_players.h).
    bool five_players = true;
    // [keys] from the config, action name -> SDL key name. Empty unless the
    // player has rebound something; window.cpp holds the defaults and the
    // list of action names.
    std::map<std::string, std::string> keys;
    std::string config_path;   // the file to edit: the per-user config once there is one
    std::string config_layers; // every file read, lowest first ("a <- b <- c")
    std::string profile;       // --config's file name without .toml ("" without one): names its own F10 settings
    std::string profile_config; // that --config file
    bool setup_requested = false;  // --setup: open the setup window (host/launcher.h)
    // startup.setup_window: open it at every start. A release build's default
    // is true, a build from source's false (its harnesses never stop there);
    // BBHOST_SETUP_WINDOW=0|1 overrides.
    bool setup_always = false;
};

// Parse argv (--config, --app0, --data, --eboot, first positional = eboot),
// then the config file (next to the binary, cwd, or --config). Command line
// wins over file. Returns false on a hard error (bad flag, unreadable file
// that was explicitly requested).
bool config_load(int argc, char** argv, HostConfig* out, std::string* error);
const HostConfig& config();
// The options screen (host/options.h) owns this one once it has read its own
// file; the loader reads config().skip_intro after that.
void config_set_skip_intro(bool on);
// The per-user bbhost.toml's format, saved as [bbhost] config_version by the
// setup window. A file below it was written by an older bbhost:
//   2 - an older setup window wrote startup.setup_window = false whenever it
//       saved, its default then, so that value is set aside (this build's
//       default stands) until the file is saved again.
//   3 - the live server (thehuntersdream.com) takes only https and signed-in
//       players; a file that still has the old plain-http settings for it is
//       rewritten with the new ones (scheme, verify_tls, require_account,
//       auth_server).
//   4 - the first-start template wrote player.ime = "auto", which answered
//       the name before it could be typed and cancelled the chalice glyph
//       and network password boxes as they opened; a file that still has it
//       is rewritten to "type", the default (BBHOST_IME=auto keeps a
//       harness's).
constexpr int kUserConfigVersion = 4;
// The game folder and its update. The PS4 applies an update by putting its
// files in place of the game's; a dump can keep the update in a folder of its
// own beside the game's, named after it with "-UPDATE" or "-patch" appended
// (CUSA00900-UPDATE: how shadPS4 lays out a game library). Every game file
// the update has is then read from there, the rest from the game folder.
// `update` is empty when there is no such folder. A paths.app0 that names the
// update folder itself, with the game folder beside it, gives the same pair.
struct GameFolders {
    std::string base, update;
};
GameFolders config_game_folders(const std::string& app0);
// The game's version, from sce_sys/param.sfo (the update folder's when it has
// one): APP_VER ("01.09") and CATEGORY ("gp" the game with an update, "gd"
// without); both empty when the file cannot be read. `update` is the update
// folder it was read from, "" for the game folder's own. The 1.09 eboot needs
// the 1.09 update's files: without them the game stops when it reads one.
struct App0Version {
    std::string app_ver, category, update;
};
App0Version config_app0_version(const std::string& app0);
// The PC enhancements, the same way: the options file's choices (F10, the
// System menu's PC Enhancements, the setup window's tab) once it is read. A
// BBHOST_CHANGE_APPEARANCE, BBHOST_REBIRTH or BBHOST_FIVE_PLAYERS of 0 or 1
// still wins over the options file.
void config_set_enhancements(bool change_appearance, bool rebirth, bool five_players);
// The setup window's server, in memory before it is saved: its account calls
// (net::auth_server_base) go where its dropdown says. Main thread, before
// the game starts.
void config_set_online(const std::string& host, const std::string& scheme, bool verify_tls, bool require_account,
                       const std::string& auth_server);
// What to tell the player when the game's paths are missing or wrong: the
// file to edit, the two keys, and which path failed. "" when they are usable.
// Writes a template next to the executable when there is no config at all.
std::string config_setup_help(const HostConfig& c);
// What to tell the player when the eboot's SHA-256 is not the build bbhost
// runs (want_sha256): the eboot and its hash, the version app0's param.sfo
// gives, and where a 1.09 eboot comes from.
std::string config_eboot_help(const HostConfig& c, const std::string& sha256, const char* want_sha256);
// Writes a commented template to path. Used by --write-config.
bool config_write_template(const std::string& path);
// One key of a config file to set: its section, its name and the value as it
// is written (quotes included for a string).
struct ConfigEdit {
    std::string section, key, value;
};
// Sets each key in the file - in place where it is, else at the end of its
// section, else in a new section - leaving every other line, comments
// included, as it was. Creates the file (with its folder) when missing.
bool config_set_values(const std::string& path, const std::vector<ConfigEdit>& edits);
// The per-user bbhost.toml (config_user_dir()/bbhost.toml).
std::string config_user_file();
// The folder saves go to when paths.data is not set (the per-user data folder).
std::string config_default_data_dir();
// The per-user folder the config and the F10 settings live in: %APPDATA%/bbhost,
// $XDG_CONFIG_HOME/bbhost or ~/.config/bbhost; BBHOST_CONFIG_DIR replaces it.
std::string config_user_dir();
// The directory the running executable is in ("." when unknown).
std::string config_exe_dir();
// A raw value from the loaded config file by "section.key" (quotes stripped), "" when absent.
std::string config_value(const std::string& key);
// Changes a loaded value in memory (raw, as config_set_values writes it): a
// setting changed while running (the plugin menu), so config_value agrees
// with the file just written. Main thread.
void config_value_set(const std::string& key, const std::string& raw);
// The config file's own reader (sections and key = value, comments, quotes),
// for other TOML-shaped files: keys come back as "section.key".
bool config_parse_toml(const std::string& path, std::map<std::string, std::string>* kv, std::string* err);
