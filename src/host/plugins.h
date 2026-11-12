// Plugins: shared libraries in <exe dir>/plugins and
// <data>/plugins that get the C API in include/bbhost_plugin.h. Two phases:
// before the eboot is loaded (log, config, register_hle) and after it is
// loaded and patched (read, write, patch, hook). BBHOST_PLUGINS=0 loads
// none.
#pragma once

struct ElfImage;

// Loads every plugin and calls its bb_plugin_init. Before register_hle().
void plugins_load();
// Calls every loaded plugin's bb_plugin_image. After hle_patch_guest().
void plugins_image(ElfImage* image);
// The callbacks plugins registered with on_frame, from the frame-time
// manager's hook on the main thread.
void plugins_frame();
// A flag change for the callbacks registered with on_event_flag, from the
// same hook (engine/event_flags.cpp).
void plugins_event_flag(unsigned id, bool value);

// --- the plugin manager (host/plugin_ui.cpp) -------------------------------

#include <cstdint>
#include <string>
#include <vector>

// One setting a plugin declares (bbhost_plugin.h BbOption), copied.
struct PluginOptionDesc {
    std::uint32_t type = 0;  // BB_OPT_*, without BB_OPT_LIVE
    bool live = false;       // takes effect at once in game
    std::string key, label, help, def, choices;
    std::string when;        // BB_OPT_WHEN: "key=a|b", greyed out unless that option is one of them
    double min = 0, max = 0;
    std::string page;        // BB_OPT_PANEL: the key of the panel page's heading it is on (or is); empty in the menus
};
// A plugin file the manager shows: its info, settings and state.
struct PluginEntry {
    std::string name, title, version, author, description, path, file;
    std::uint32_t flags = 0;
    bool official = false;     // a valid release-key signature
    bool bad_signature = false;
    bool loaded = false;       // running in this session
    bool visitor = false;      // running though off: only to play other players' worlds
    bool enabled = false;      // what the config says (opt-in ones default off)
    int aa = 0;                // its anti-aliasing claim: kAaNone, kAaLeave or kAaTake (its flags', or the live one in game)
    std::vector<PluginOptionDesc> options;
};
// The directories plugins load from, in order (exe, per-user, data).
std::vector<std::string> plugins_dirs();
// Every plugin file in them, read without starting it (for the launcher;
// works before plugins_load). A later directory's file of the same name wins.
std::vector<PluginEntry> plugins_scan();
// What plugins_load found, with their live state (for the in-game menu).
std::vector<PluginEntry> plugins_catalog();
// Turns a plugin on or off in the per-user config (from the next start).
void plugins_set_enabled(const std::string& name, bool on);
// A plugin running in this session that turns on and off at once
// (BB_PLUGIN_LIVE_SWITCH): its switch needs no restart.
bool plugins_live_switch(const PluginEntry& p);
// A plugin setting: written to the per-user config, and - in game - handed
// to the plugin's on_option callback on the main thread.
std::string plugins_option_value(const PluginEntry& p, const PluginOptionDesc& o);
void plugins_set_option(const std::string& name, const std::string& key, const std::string& value);
// An action button: queued for the plugin's on_action on the main thread.
void plugins_run_action(const std::string& name, const std::string& action);
// The rules this session plays by, for the server (the X-BBHost-Ruleset
// header on every request the game makes): "vanilla", or each loaded plugin
// flagged BB_PLUGIN_GAMEPLAY, sorted, comma separated, with ";seed=<seed>"
// for one that has a seed setting - "boss_rush,randomizer;seed=1a2b3c4d".
// The server keeps those runs off the normal map and stats, and shows the
// player only the bloodstains and messages of the same ruleset.
std::string plugins_ruleset();
// The plugin of that name is loaded and on for this run (not a visitor):
// what host features a plugin turns on ask (engine/debug_menu.h).
bool plugins_active(const std::string& name);
// Plugin API 27's anti-aliasing claim (bbhost_plugin.h aa_claim): the
// strongest claim of the plugins that are on - kAaTake before kAaLeave - in
// one relaxed load (host/upscaler.cpp reads it at every draw).
constexpr int kAaNone = 0, kAaLeave = 1, kAaTake = 2;
int plugins_aa_claim();
// The title of the first loaded plugin that takes it over, which the menus
// name ("" when none does).
std::string plugins_aa_owner();
// The setup window's, before any plugin runs: the title of the first plugin
// turned on whose BbPluginInfo flags take it over (BB_PLUGIN_AA_TAKE), from
// the plugin files ("" when none).
std::string plugins_aa_owner_scanned();
// The loaded plugins that can play by a host's rules (BB_PLUGIN_ADOPTS_RULES),
// "randomizer" - the X-BBHost-Adopt header; "" for none.
std::string plugins_adopts();
// A guest joining a world whose host plays by other rules ("" when going
// back to its own): the adopting plugins rewrite what the next load reads,
// and the session's rules (plugins_ruleset) become the host's. Before that
// world loads; any thread but the main one.
void plugins_enter_world(const std::string& host_rules);
