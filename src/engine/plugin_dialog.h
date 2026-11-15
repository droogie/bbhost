#pragma once

// A plugin's in-game menu as pages of the game's own options dialog: the
// game's rows, cursor, sounds, line help and key guide, built by the same
// calls as bbhost's PC Settings pages (engine/option_rows.h), in sections the
// generated movie adds (engine/menu_assets.cpp): PlugPage, centred as the
// game's, and PlugCorner, the same smaller at the bottom right with the world
// in view (BB_PANEL_CORNER, photo mode's).
//
// What a page holds comes from the host's panel (host/plugin_panel.h): its
// title and rows. A BOOL is the game's On/Off row, a CHOICE of two its
// left/right pair, an INT from 0 to 10 its slider; any other number is the
// game's slider too, its bar and number set by the host from the plugin's
// range and step; every other row - a longer CHOICE, INFO, TEXT, a button -
// is a game choice row whose label, value and line help are the host's own
// text, worked by the host: Left and Right step it (held, they repeat), Cross
// presses it.
// A page of more rows than the section's eight scrolls by the host moving its
// rows under them. Opening, closing and every page change go through the
// menu flow's own calls (sub_1fb4c70 makes the screen, sub_1fb27a0 pushes it,
// +0xb12 asks it to close) from the frame-time hook - no code of the game's
// is patched for it.
//
// The menu flow is the one the game opens its own menus on, so the dialog
// opens over a menu the game has up (it comes back after), and anywhere the
// flow does - the title, the world, the System list (its Plugins row). Where
// it cannot - another eboot, a movie without the sections, a flow that
// refuses - the host draws its own panel instead (host/plugin_panel.h).

#include <cstdint>
#include <string>
#include <vector>

struct ElfImage;

namespace plugin_dialog {

enum class Widget : int {
    Text,    // a choice row of one entry, its texts the host's: steppers, lists, info, text, buttons
    Toggle,  // the game's On/Off (1 On)
    Pair,    // the game's left/right of two entries the host names
    Slider,  // the game's 0..10 slider
};

struct Row {
    Widget widget = Widget::Text;
    std::string label, help, value;  // UTF-8; `value` a Text row's
    std::string pair[2];             // a Pair's two entries
    int on = 0;                      // a Toggle's, Pair's or Slider's value
    // A Slider of the host's own: 0..positions (one a step of the plugin's),
    // its number `value`, Left and Right the host's. 0: the game's 0..10.
    int positions = 0;
    bool grey = false;               // BB_OPT_WHEN not met: shown dimmed, does nothing
    bool steps = false;              // Left and Right change it
    bool presses = false;            // Cross does something (a button, a text, a page)
};

struct Page {
    std::string title;
    std::vector<Row> rows;
    bool corner = false;    // PlugCorner
    int cursor = 0;         // the row the cursor starts on
};

// The generated movie has the sections and option_menu is in (the 1.09
// eboot, its messages); from option_menu_install on.
bool available();

// From any thread. open: the host's page up as soon as the flow takes it -
// false when it cannot be (the host then draws its own). close: down, as the
// plugin or the host asks (not the player's close). changed: the page's rows
// (a page turned, rows set) or only their values changed since it was built.
bool open();
void close();
void changed(bool rows);

// Our dialog is the flow's open menu (world_state does not count it a game
// menu); and it is a corner one.
bool up();
bool corner_up();
// The flow's open menu is a page the host pushed over no menu of the game's -
// opening, up, or for a second after it went (world_state does not count it a
// game menu from the moment it is pushed until the flow has let it go: photo
// mode, which opens its page as it starts and hides it on Circle, would end
// itself).
bool alone();
// A list of a menu of the game's that had the input when the host pushed a
// page over it - the title's, which the menu flow does not stop: held, with
// no input and hidden, until the page is gone (engine/menu_pointer.cpp).
bool holds(void* comp);

// The game's thread, once a frame (plugins_frame): opens, closes and keeps
// track of the dialog.
void frame();
// Every menu list update, on the menu's thread (engine/option_menu.cpp):
// the host's rows' input, scrolling and texts.
void list_update(void* comp, bool focused);
// The System list being built: its Plugins row, while any plugin has a
// named menu (engine/option_menu.cpp, just before its Exit Game).
void system_row(void* builder);
// A click on row `index` of `list` is a press (engine/menu_pointer.cpp asks):
// our page's buttons, text rows and the Plugins list's menus, wherever on the
// row it lands - not a step, as on the rows Left and Right change.
bool click_decides(void* list, int index);
// Every pad read, before the game has it (hle/system.cpp): what our dialog
// keeps from the game - Circle on a page opened on top, Up and Down at the
// ends of the rows it scrolls, and with a corner page up everything but the
// buttons it moves by (the sticks and the rest fly photo mode's camera).
void pad_filter(std::uint32_t& buttons, std::uint8_t& lx, std::uint8_t& ly, std::uint8_t& rx, std::uint8_t& ry,
                std::uint8_t& l2, std::uint8_t& r2);

}  // namespace plugin_dialog

// From option_menu_install, once it is in: the generated movie has the
// pages' sections (a player's own older movie in paths.mods may not).
void plugin_dialog_install();
// engine/option_menu.cpp's hook name (C++ linkage, outside the namespace).
inline void plugin_dialog_system_row(void* builder) { plugin_dialog::system_row(builder); }
