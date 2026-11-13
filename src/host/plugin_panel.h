#pragma once

// A plugin's panel (plugin API 29's panel_show and panel_state): the pages of
// its options table it marks (BB_OPT_PANEL headings), drawn over the game at
// the bottom right with the menus' ImGui (host/ingame_menu.cpp) while no menu
// is up. The controller drives it - the d-pad up and down to a row, left and
// right to change it, Cross to set a switch or press a button, L1 and R1 the
// pages, Circle closes it - as do the keyboard (the arrows, Enter, Page Up and
// Page Down or Q and E, Escape) and the pointer, which is free while it is up
// (hle/system.cpp). While it is up the game reads none of those buttons and
// keys, and neither do the plugins' pad and input (host/window.cpp): the panel
// has them. One panel at a time.

#include <cstdint>
#include <string>

struct PluginPanelShow {
    std::string page;              // its heading's key; empty: the page shown last
    std::string guide, guide_keys;  // a line under the rows, for the controller and for the keyboard
    bool defaults = false;         // BB_PANEL_DEFAULTS: each page ends with a Defaults row
};
// The panel of the plugin named `name` up, at `how.page`: false when the
// plugin has no such page (or none at all). It replaces another plugin's.
bool plugin_panel_show(const std::string& name, const PluginPanelShow& how);
// Its panel down (panel_show with NULL, the plugin turned off). Not "closed":
// that is the player's.
void plugin_panel_hide(const std::string& name);

struct PluginPanelState {
    bool shown = false;
    bool closed = false;  // the player closed it since the last call
    std::string page;
};
// Whether `name`'s panel is up, whether the player closed it since the last
// call (which clears that), and the page shown or shown last.
void plugin_panel_state(const std::string& name, PluginPanelState& out);

// Any plugin's panel is up: the pad, the keys and the pointer go to it.
bool plugin_panel_shown();
// The controller's buttons the panel takes while up (the menus' bits, as
// host_pad_input_bit gives them): the d-pad, Cross, Circle, L1 and R1.
constexpr std::uint32_t kPanelPadButtons = 0x10u | 0x20u | 0x40u | 0x80u | 0x4000u | 0x2000u | 0x400u | 0x800u;
// A key the panel takes while up (SDL scancode): the arrows, Enter, Page Up,
// Page Down, Q, E and Escape.
bool plugin_panel_key(int scancode);

// Draws it into the current ImGui frame (the menus' context), at `scale`:
// the presenter's thread, while no menu is up. `pad` is the controller in use
// (the guide for it).
void plugin_panel_draw(float scale, bool pad);
