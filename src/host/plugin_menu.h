#pragma once

// A plugin's in-game menus as the host keeps them (host/plugin_panel.cpp
// draws them): the pages its options table declares (BB_OPT_PANEL headings),
// the menus they make (BB_OPT_MENU), the sub-pages (BB_OPT_SUBPAGE) a row
// opens (BB_OPT_OPEN) or the plugin does (panel_open), the rows a plugin sets
// at run time (panel_rows) with their values (panel_value), and the menu up
// now: its pages in L1/R1 order and the pages opened on top of them. No
// drawing and no locks: the panel holds its own.

#include "host/plugins.h"

#include <map>
#include <string>
#include <vector>

namespace plugin_menu {

// A row of a page: one of the table's, or one panel_rows set (its value kept
// here, never in the config, its events "page/key").
struct Row {
    PluginOptionDesc desc;
    bool runtime = false;
};

struct Page {
    std::string key, title, help;
    std::string menu;      // the key of the BB_OPT_MENU heading it comes under; "" for a plugin's one unnamed menu
    bool subpage = false;  // BB_OPT_SUBPAGE: out of its menu's L1/R1 order
    std::vector<Row> rows;  // the table's
};

// A menu the table names (a BB_OPT_MENU heading): its name for the overlay's
// Open button and its pages in L1/R1 order (its sub-pages left out).
struct MenuDecl {
    std::string key, name;
    std::vector<std::string> pages;
};

struct Layout {
    std::vector<Page> pages;
    std::vector<MenuDecl> menus;
    const Page* page(const std::string& key) const;
};

// The pages and menus of a plugin's options table, as plugins.cpp reads it
// (PluginOptionDesc: the type without its flags, which are in `flags`).
Layout layout_of(const std::vector<PluginOptionDesc>& options);

// The L1/R1 order of the menu that holds `page`: that menu's pages but its
// sub-pages; just `page` for a sub-page; empty for a page the table has not.
std::vector<std::string> cycle_of(const Layout& l, const std::string& page);

// What a plugin set at run time: rows for some of its pages and their values.
struct Runtime {
    std::map<std::string, std::vector<Row>> rows;    // by page
    std::map<std::string, std::string> values;       // by "page/key"
};

// panel_rows: `page`'s rows from now on, each one's value its `def` (its
// value is kept); none gives the page the table's rows again. False for a
// page the table has not got.
bool set_rows(const Layout& l, Runtime& rt, const std::string& page, const std::vector<PluginOptionDesc>& rows);
// panel_value: a run-time row's value. False for a page or row it has not.
bool set_value(Runtime& rt, const std::string& page, const std::string& key, const std::string& value);
// The rows a page shows: those set at run time, else the table's.
const std::vector<Row>& rows_of(const Page& p, const Runtime& rt);
// A run-time row's value: what was set, else its default.
std::string value_of(const Runtime& rt, const std::string& page, const Row& row);
// The key an event of `row` on `page` comes to the plugin with (on_option,
// on_action): a table row's own key, "page/key" for a run-time one.
std::string event_key(const std::string& page, const Row& row);

// The menu up now: its pages in L1/R1 order, the one shown, and the pages
// opened on top of it, each with the row it was opened from.
struct Nav {
    std::vector<std::string> cycle;
    int at = 0;
    struct Opened {
        std::string page;
        int from_row = 0;  // the row of the page under it that opened it: the cursor goes back there
    };
    std::vector<Opened> stack;

    const std::string& shown() const;  // the top of the stack, else the cycle's page
    bool on_subpage() const { return !stack.empty(); }
    // L1/R1: the next or the previous page of the cycle, round; nothing on a
    // page opened on top. True when the page changed.
    bool turn(int d);
    // A page opened on top of the shown one, from its row `from_row`.
    void open(const std::string& page, int from_row);
    // Circle: back to the page under the shown one - the row the cursor goes
    // to, the one that opened it - or -1 when the shown page is the cycle's:
    // the menu closes.
    int back();
};

}  // namespace plugin_menu
