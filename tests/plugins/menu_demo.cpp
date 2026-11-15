// A test plugin that uses every part of plugin API 29's in-game menus
// (docs/plugins.md, In-game menus): two named menus, a sub-page, rows set at
// run time (a list of lamps with a sort order, a lamp's details, a count),
// INFO, TEXT, OPEN and CONFIRM rows, panel_value, panel_open, and panel_show
// with its own page list in the corner. tests/plugin_menus_test.cpp drives it
// on a pretend host; the lavapipe pictures load it into the game. Built
// beside the tests, never into plugins/ and in no package.
#include "bbhost_plugin.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

const BbHostApi* api = nullptr;
constexpr const char* kName = "menu_demo";

struct Lamp {
    std::string id, name, area;
    bool lit;
};
// More than a menu shows at once: the list scrolls.
std::vector<Lamp> lamps = {
    {"dream", "Hunter's Dream", "Hunter's Dream", true},
    {"plaza", "Central Yharnam", "Yharnam", true},
    {"ward", "Cathedral Ward", "Yharnam", false},
    {"woods", "Forbidden Woods", "Forbidden Woods", false},
    {"bridge", "Great Bridge", "Yharnam", false},
    {"oedon", "Tomb of Oedon", "Yharnam", false},
    {"grand", "Grand Cathedral", "Cathedral Ward", false},
    {"upper", "Upper Cathedral Ward", "Cathedral Ward", false},
    {"old", "Old Yharnam", "Old Yharnam", false},
    {"hemwick", "Hemwick Charnel Lane", "Hemwick", false},
    {"byrgenwerth", "Byrgenwerth", "Byrgenwerth", false},
    {"yahar", "Yahar'gul", "Yahar'gul", false},
};
std::string sort_by = "Story";
std::string shown_lamp;  // the lamp on the "lamp" page

int lit_count() { return static_cast<int>(std::count_if(lamps.begin(), lamps.end(), [](const Lamp& l) { return l.lit; })); }

// The "lamps" page: the list in the order asked, each lamp a button that
// opens the "lamp" page; a count of the lit ones; the order itself.
void fill_lamps() {
    std::vector<Lamp> order = lamps;
    if (sort_by == "Name") std::sort(order.begin(), order.end(), [](const Lamp& a, const Lamp& b) { return a.name < b.name; });
    std::vector<BbOption> rows;
    static std::string lit;
    lit = std::to_string(lit_count()) + " of " + std::to_string(lamps.size());
    rows.push_back({BB_OPT_INFO, "lit", "Lamps lit", "Lamps you have lit.", lit.c_str()});
    rows.push_back({BB_OPT_CHOICE, "sort", "Order", "Story: as you meet them. Name: A to Z.", sort_by.c_str(), 0, 0, "Story|Name"});
    static std::vector<std::string> keep;  // the rows' texts outlive the call only as long as the host needs: it copies
    keep.clear();
    for (const Lamp& l : order) keep.push_back(l.name + (l.lit ? "" : " (unlit)"));
    for (std::size_t i = 0; i < order.size(); ++i)
        rows.push_back({BB_OPT_ACTION | BB_OPT_OPEN, order[i].id.c_str(), keep[i].c_str(), "Its details.", "lamp"});
    rows.push_back({BB_OPT_ACTION, "random", "A random lamp", "Opens one of them, chosen by the plugin."});
    api->panel_rows(kName, "lamps", rows.data(), rows.size());
}

// The "lamp" page: one lamp's details, a name of the player's, and travel.
void fill_lamp(const std::string& id) {
    const auto it = std::find_if(lamps.begin(), lamps.end(), [&](const Lamp& l) { return l.id == id; });
    if (it == lamps.end()) return;
    shown_lamp = id;
    const BbOption rows[] = {
        {BB_OPT_INFO, "area", "Area", "Where it is.", it->area.c_str()},
        {BB_OPT_INFO, "state", "State", "Lit or not.", it->lit ? "Lit" : "Unlit"},
        {BB_OPT_TEXT, "rename", "Name", "A name of your own for it.", it->name.c_str()},
        {BB_OPT_ACTION | BB_OPT_CONFIRM, "travel", "Travel there", "The world loads again at that lamp."},
    };
    api->panel_rows(kName, "lamp", rows, sizeof(rows) / sizeof(rows[0]));
}

void on_option(const char* key, const char* value, void*) {
    if (std::strcmp(key, "lamps/sort") == 0) {
        sort_by = value;
        fill_lamps();
    } else if (std::strcmp(key, "lamp/rename") == 0) {
        for (Lamp& l : lamps) {
            if (l.id == shown_lamp) l.name = value;
        }
        fill_lamps();
    }
}

void on_action(const char* action, void*) {
    const std::string a = action;
    if (a.rfind("lamps/", 0) == 0) {
        const std::string id = a.substr(6);
        if (id == "random") {
            fill_lamp(lamps[2].id);  // chosen by the plugin: a page opened from code
            api->panel_open(kName, "lamp");
        } else {
            fill_lamp(id);  // the row's BB_OPT_OPEN opens the page
        }
    } else if (a == "lamp/travel") {
        for (Lamp& l : lamps) {
            if (l.id == shown_lamp) l.lit = true;
        }
        const std::string lit = std::to_string(lit_count()) + " of " + std::to_string(lamps.size());
        api->panel_value(kName, "lamps", "lit", lit.c_str());
        api->panel_value(kName, "lamp", "state", "Lit");
        api->log("menu_demo: travel to %s", shown_lamp.c_str());
    } else if (a == "reset") {
        for (Lamp& l : lamps) l.lit = l.id == "dream";
        fill_lamps();
    } else if (a == "corner") {
        // Its own page list, in the corner: panel_show with pages.
        static const char* const pages[] = {"tools", "lamps"};
        BbPanel p{};
        p.size = sizeof(p);
        p.flags = BB_PANEL_CORNER;
        p.page = "tools";
        p.pages = pages;
        p.page_count = 2;
        api->panel_show(kName, &p);
    }
}

}  // namespace

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {BB_PLUGIN_API_VERSION, 0, kName, "Menu demo", "1.0.0", "bbhost",
                                                      "Every part of the in-game menus, for the tests."};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_ACTION, "corner", "Show the tools in the corner", "panel_show with a page list of its own."},
    // The Travel menu: a list filled at run time, a page of the table's rows, a sub-page.
    {BB_OPT_HEADING | BB_OPT_PANEL | BB_OPT_MENU, "lamps", "Lamps", "Choose a lamp.", "Travel"},
    {BB_OPT_HEADING | BB_OPT_PANEL | BB_OPT_SUBPAGE, "lamp", "Lamp", "One lamp."},
    {BB_OPT_HEADING | BB_OPT_PANEL, "history", "History", "What the menu keeps."},
    {BB_OPT_TEXT | BB_OPT_LIVE, "note", "Note", "Text of your own, kept in the config.", "A note"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "fast", "Fast travel", "Kept in the config.", "true"},
    // The Tools menu: table rows only.
    {BB_OPT_HEADING | BB_OPT_PANEL | BB_OPT_MENU, "tools", "Tools", "Things to change.", "Tools"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "speed", "Speed", "How fast.", "Normal", 0, 0, "Slow|Normal|Fast"},
    {BB_OPT_INFO, "version", "Version", "This plugin's.", "1.0.0"},
    {BB_OPT_ACTION | BB_OPT_CONFIRM, "reset", "Forget every lamp", "Every lamp but the Dream's goes out."},
    {BB_OPT_HEADING | BB_OPT_PANEL, "more", "More", "Numbers."},
    {BB_OPT_INT | BB_OPT_LIVE, "count", "Count", "From 0 to 10.", "3", 0, 10},
    {BB_OPT_INT | BB_OPT_LIVE, "level", "Level", "From -50 to 50.", "10", -50, 50},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "mix", "Mix", "From 0 to 1.", "0.25", 0, 1},
    {0}};

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* host) {
    if (host->version < 29 || host->size < sizeof(BbHostApi)) return 1;
    api = host;
    api->on_option(kName, &on_option, nullptr);
    api->on_action(kName, &on_action, nullptr);
    fill_lamps();
    return 0;
}
