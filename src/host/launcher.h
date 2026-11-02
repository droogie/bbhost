// The setup window: a small window of its own (Dear ImGui on SDL3's 2D
// renderer, not the game's renderer) that opens before the game when the
// game's paths are missing or wrong, when bbhost starts with --setup, or at
// every start when startup.setup_window says so. A sidebar of pages (Game,
// Online, PC Enhancements, Patches, Shaders, Plugins, Settings) and Play, over
// a moonlit night (host/launcher_ui.h). It sets the game folder and the
// eboot (with the system's folder and file pickers, and a check of each), the
// data folder, the online server and account, a few start-up settings (the
// same ones the F10 screen has, through host_opt_*), the patches and update
// checks, and writes them to the per-user bbhost.toml (config_user_file) in
// place. Never in a headless run.
#pragma once

#include "core/config.h"

#include <string>

enum class LauncherResult { Play, Quit };

// Runs the window until the player presses Play (the settings saved; the
// caller reads the configuration again) or closes it. `reason` says why it
// opened ("" when asked for).
// `plugins_tab` opens it on the plugin manager (host/plugin_ui.h).
LauncherResult launcher_run(const HostConfig& cfg, const std::string& reason, bool plugins_tab = false);
