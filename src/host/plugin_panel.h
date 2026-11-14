#pragma once

// A plugin's in-game menu (plugin API 29's panel_show, panel_state,
// panel_rows, panel_value and panel_open; the model in host/plugin_menu.h):
// the pages of its options table it marks (BB_OPT_PANEL headings) and rows it
// sets at run time, drawn over the game with the menus' ImGui
// (host/ingame_menu.cpp) while no menu of the host's is up - centred as the
// game's dialogs are, the screen dimmed behind it, or at the bottom right
// (BB_PANEL_CORNER). The controller drives it - the d-pad up and down to a
// row, left and right to change it, Cross to set a switch or press a button,
// L1 and R1 the pages, Circle back or closed - as do the keyboard (the
// arrows, Enter, Page Up and Page Down or Q and E, Escape or Backspace) and
// the pointer, which is free while it is up. A centred menu has every input
// while it is up; a corner panel those buttons and keys, which the game and
// the plugins' pad and input then do not read (host/window.cpp). One menu at
// a time.

#include <cstdint>
#include <string>
#include <vector>

struct PluginOptionDesc;

struct PluginPanelShow {
    std::string page;              // its heading's key; empty: the page shown last
    std::string guide, guide_keys;  // a line under the rows, for the controller and for the keyboard; none: the host's
    bool defaults = false;         // BB_PANEL_DEFAULTS: each page ends with a Defaults row
    bool corner = false;           // BB_PANEL_CORNER: at the bottom right; else the game's dialog, centred
    std::vector<std::string> pages;  // BbPanel.pages: the menu's pages in L1/R1 order; none: the table's menu of `page`
};
// The panel of the plugin named `name` up, at `how.page`: false when the
// plugin has no such page (or none at all), or `how.pages` names one it has
// not. It replaces another plugin's.
bool plugin_panel_show(const std::string& name, const PluginPanelShow& how);
// Plugin API 29's run-time pages (panel_rows, panel_value, panel_open,
// host/plugin_menu.h): a page's rows set by the plugin (none: the table's
// again), one of their values, a page opened on top of the one shown. False
// for a plugin, page or row it has not got (and for panel_open, no panel of
// its up). Any thread.
bool plugin_panel_rows(const std::string& name, const std::string& page, const std::vector<PluginOptionDesc>& rows);
bool plugin_panel_value(const std::string& name, const std::string& page, const std::string& key, const std::string& value);
bool plugin_panel_open(const std::string& name, const std::string& page);
// The menus a plugin's table names (BB_OPT_MENU headings): the first page of
// each and its name, for the overlay's Open buttons.
struct PluginMenuEntry {
    std::string page, name;
};
std::vector<PluginMenuEntry> plugin_panel_menus(const std::string& name);
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
// The panel up is a centred menu, which takes every input: the game and the
// plugins read nothing of the controller, the keyboard or the mouse while it
// is (a corner panel takes the buttons and keys below only).
bool plugin_panel_exclusive();
// The controller's buttons the panel takes while up (the menus' bits, as
// host_pad_input_bit gives them): the d-pad, Cross, Circle, L1 and R1.
constexpr std::uint32_t kPanelPadButtons = 0x10u | 0x20u | 0x40u | 0x80u | 0x4000u | 0x2000u | 0x400u | 0x800u;
// A key the panel takes while up (SDL scancode): the arrows, Enter, Page Up,
// Page Down, Q, E, Escape and Backspace.
bool plugin_panel_key(int scancode);

// The game's menu art (host/game_menu_art.h) read now, on a thread of its
// own: a plugin with panel pages is loaded, and its panel's first frame then
// does not wait the fifth of a second the art takes to read.
void plugin_panel_preload();

// Draws it into the current ImGui frame (the menus' context), at `scale`:
// the presenter's thread, while no menu is up. `pad` is the controller in use
// (the guide for it).
void plugin_panel_draw(float scale, bool pad);

// The game's menu sound the panel asks for since the last call (the cursor
// moved, a value changed, a page turned, Defaults pressed, the panel up or
// closed: engine/menu_pointer.h's ids), the loudest of them, with its
// priority; -1 when none. For the game's thread to post (plugins_frame).
int plugin_panel_take_sound(int* priority);

// BBHOST_PANEL_PREVIEW=<plugin>[:<page>] (a test aid): that plugin's menu up
// from the first frame the menus draw, wherever the game is - at the title
// too - with BBHOST_PANEL_PREVIEW_GUIDE and BBHOST_PANEL_PREVIEW_GUIDE_KEYS
// as its guide; centred, or with BBHOST_PANEL_PREVIEW_CORNER=1 in the corner
// with Defaults rows (photo mode's). Nothing without the variable. The
// presenter's thread.
void plugin_panel_preview_tick();
