#pragma once

#include <string>

// app0 is required. data/tmp default to <dir of eboot>/data and /tmp. `mods`
// is the port's asset overlay - a directory resolved ahead of the dump for
// /app0 reads, so a modified asset never has to be written into the dump. It
// defaults to <data>/mods and is ignored when that directory does not exist.
void hle_fs_set_roots(const char* app0, const char* data, const char* tmp, const char* eboot_path,
                      const char* mods);
const char* hle_fs_data_root();  // host path of the data root; empty when none is set
const char* hle_fs_mods_root();  // host path of the asset overlay; empty when there is none
const char* hle_fs_app0_root();  // host path of the dump; empty when none is set
// The game's update beside it (<game folder>-UPDATE or -patch, core/config.h's
// config_game_folders), whose files /app0 reads find in place of the game
// folder's; empty when there is none.
const char* hle_fs_update_root();
// The game's own file at an /app0-relative path ("dvdroot_ps4/..."): the
// update's when it has the file, else the game folder's (which may not exist).
// What the host builds its generated files from - never an overlay's file.
std::string hle_fs_game_file(const std::string& rel);
// A second overlay, behind `mods`: files the host made from the dump itself
// (engine/menu_assets.h). /app0 reads find the player's mods first, then this.
void hle_fs_set_generated_root(const char* dir);
// The player's own file at an /app0-relative path ("dvdroot_ps4/..."), in the
// mods folder (ignoring case), or empty when the mods folder has none. What the
// host patches when a mod ships a bank the host adds its text to
// (engine/menu_assets.h).
std::string hle_fs_mods_file(const std::string& rel);
// A layer *ahead* of `mods`, for the few files the host made from a mod's own
// file: the mod's menu text with the port's strings added. Without it the mod's
// file, found first, hides the generated one. Empty dir: none.
void hle_fs_set_patched_root(const char* dir);
// More of the host's generated files, behind that one, in the order added
// (engine/change_appearance.h): a feature that is off adds none. At start,
// before the game reads a file.
void hle_fs_add_generated_root(const char* dir);
// The /app0 lookups since the last call and how many went to the disk for a
// generated overlay or were answered by its index (hle/fs.cpp, OverlayIndex),
// for the 300-flip report; empty when there were none.
std::string hle_fs_overlay_report();
// A plugin's overlay (plugins' overlay_file): read behind the player's mods
// and ahead of the generated files and the dump. Plugins added first win.
void hle_fs_add_plugin_overlay(const char* dir);
// The host file an /app0-relative path ("/dvdroot_ps4/...") resolves to in
// either overlay, or empty when neither has it.
std::string hle_fs_overlay_file(const char* rel);
void hle_register_fs();
std::string hle_fs_map_path(const char* guest);
// The game files most recently found missing or empty, newest last ("" for
// none): what a panic in the game's file loader most likely stopped on.
std::string hle_fs_problem_files();
// The last 16 files the game opened, with their sizes or errors, into the log
// (once): every crash report ends with them.
void hle_fs_log_recent_opens();
// The same without the plugins' overlays: the file a plugin starts from (the
// player's mods, the generated files, the dump), never its own output.
std::string hle_fs_map_path_base(const char* guest);
// Mount <data>/<subdir> at a guest prefix such as "/savedata0". Creates the dir.
bool hle_fs_mount(const char* guest_prefix, const char* data_subdir);
void hle_fs_umount(const char* guest_prefix);
int hle_fs_stat_path(const char* guest, void* orbis_sb);
