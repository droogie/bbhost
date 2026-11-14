// The in-game menus' test plugin (tests/plugins/menu_demo.cpp) on a host of
// our own: no game, no drawing. Every plugin API 29 menu call it makes is
// recorded. Checked: it refuses a host older than 29; its table declares two
// named menus, a sub-page, INFO, TEXT, OPEN and CONFIRM rows as the host
// reads them (host/plugin_menu.h); at the start it fills its list page; a
// lamp's button (an OPEN row) fills the sub-page and leaves the opening to
// the host, while its "random" button opens the page itself (panel_open); a
// run-time row's change comes with its page ("lamps/sort", "lamp/rename")
// and the list is set again in that order, by that name; travel (a CONFIRM
// row) updates two values (panel_value); its own page list goes in the
// corner (panel_show with pages and BB_PANEL_CORNER). Then its menus in the
// host's panel (host/plugin_panel.cpp, on a catalog of this test's) through
// the world's states frame by frame: a loading screen coming up, a cutscene
// starting and the character dying close a centred menu - panel_state reads
// shown 0 and closed 1, once - while one under way when it opened, the
// game's messages and menus, a pause and other players do not, and a corner
// panel stays through all of them.
//
//   plugin_menus_test <menu_demo.so>
#include "bbhost_plugin.h"
#include "host/game_menu_art.h"
#include "host/plugin_menu.h"
#include "host/plugin_panel.h"
#include "host/plugins.h"
#include "engine/plugin_dialog.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);    \
            ++g_failures;                                                \
        }                                                                \
    } while (0)

struct Rows {
    std::string page;
    std::vector<PluginOptionDesc> rows;
};
struct Host {
    void (*option_fn)(const char*, const char*, void*) = nullptr;
    void (*action_fn)(const char*, void*) = nullptr;
    std::vector<Rows> rows;                     // every panel_rows, in order
    std::vector<std::string> values;            // "page/key=value", every panel_value
    std::vector<std::string> opened;            // every panel_open
    std::vector<std::string> shown_pages;       // the last panel_show's page list
    std::string shown_page;
    std::uint32_t shown_flags = 0;
    int shows = 0;
} H;

const Rows* last_rows(const std::string& page) {
    for (auto it = H.rows.rbegin(); it != H.rows.rend(); ++it) {
        if (it->page == page) return &*it;
    }
    return nullptr;
}

void fake_log(const char*, ...) {}
const char* fake_config(const char*) { return ""; }
int fake_on_option(const char*, void (*fn)(const char*, const char*, void*), void*) {
    H.option_fn = fn;
    return 0;
}
int fake_on_action(const char*, void (*fn)(const char*, void*), void*) {
    H.action_fn = fn;
    return 0;
}
int fake_panel_rows(const char* name, const char* page, const BbOption* rows, size_t n) {
    if (std::strcmp(name, "menu_demo") != 0) return 1;
    Rows r;
    r.page = page;
    for (std::size_t i = 0; i < n; ++i) {
        PluginOptionDesc d;
        d.type = rows[i].type & 0xffu;
        d.flags = rows[i].type & ~0xffu;
        d.key = rows[i].key ? rows[i].key : "";
        d.label = rows[i].label ? rows[i].label : "";
        d.help = rows[i].help ? rows[i].help : "";
        d.def = rows[i].def ? rows[i].def : "";
        d.choices = rows[i].choices ? rows[i].choices : "";
        r.rows.push_back(d);
    }
    H.rows.push_back(r);
    return 0;
}
int fake_panel_value(const char*, const char* page, const char* key, const char* value) {
    H.values.push_back(std::string(page) + "/" + key + "=" + value);
    return 0;
}
int fake_panel_open(const char*, const char* page) {
    H.opened.push_back(page);
    return 0;
}
int fake_panel_show(const char*, const BbPanel* p) {
    ++H.shows;
    H.shown_page = p && p->page ? p->page : "";
    H.shown_flags = p ? p->flags : 0;
    H.shown_pages.clear();
    if (p && p->size >= offsetof(BbPanel, page_count) + sizeof(p->page_count) && p->pages) {
        for (std::uint32_t i = 0; i < p->page_count; ++i) H.shown_pages.push_back(p->pages[i]);
    }
    return 0;
}

struct Plugin {
#if defined(_WIN32)
    HMODULE handle = nullptr;
    void* sym(const char* name) { return reinterpret_cast<void*>(GetProcAddress(handle, name)); }
#else
    void* handle = nullptr;
    void* sym(const char* name) { return dlsym(handle, name); }
#endif
    int (*image)(const BbHostApi*) = nullptr;
    const BbOption* options = nullptr;
};

BbHostApi make_api(std::uint32_t version) {
    BbHostApi api;
    std::memset(&api, 0, sizeof api);
    api.version = version;
    api.size = sizeof(BbHostApi);
    api.log = &fake_log;
    api.config = &fake_config;
    api.on_option = &fake_on_option;
    api.on_action = &fake_on_action;
    api.panel_show = &fake_panel_show;
    api.panel_rows = &fake_panel_rows;
    api.panel_value = &fake_panel_value;
    api.panel_open = &fake_panel_open;
    if (version < 29) api.size = static_cast<std::uint32_t>(offsetof(BbHostApi, on_camera_target) + sizeof(void*));
    return api;
}

// The list's first four lamps (its OPEN rows), and how many it has.
std::string labels(const Rows* r, int* count = nullptr) {
    std::string s;
    int n = 0;
    if (r) {
        for (const PluginOptionDesc& d : r->rows) {
            if (d.type != BB_OPT_ACTION || !(d.flags & BB_OPT_OPEN)) continue;
            if (n++ < 4) s += (s.empty() ? "" : ",") + d.label;
        }
    }
    if (count) *count = n;
    return s;
}

std::vector<PluginEntry> g_catalog;  // what plugins_catalog gives host/plugin_panel.cpp

PluginPanelState panel_state() {
    PluginPanelState st;
    plugin_panel_state("menu_demo", st);
    return st;
}

}  // namespace

// host/plugin_panel.cpp's catalog: the test plugin's table.
std::vector<PluginEntry> plugins_catalog() { return g_catalog; }
namespace game_menu_art {
void preload() {}
}  // namespace game_menu_art
// The rest of the host it reaches: the options (a config of this test's)
// and the game's dialog (engine/plugin_dialog.h), absent at first - the
// host's own panel then, the fallback - and there for the native checks.
struct Native {
    bool available = false;
    int opens = 0, closes = 0, changes = 0;
    std::vector<std::string> options, actions;  // "key=value", the action keys
    std::map<std::string, std::string> config;
} N;
std::string plugins_option_value(const PluginEntry&, const PluginOptionDesc& o) {
    const auto it = N.config.find(o.key);
    return it != N.config.end() ? it->second : o.def;
}
void plugins_set_option(const std::string&, const std::string& key, const std::string& value) {
    N.options.push_back(key + "=" + value);
    N.config[key] = value;
}
void plugins_menu_option(const std::string&, const std::string& key, const std::string& value) { N.options.push_back(key + "=" + value); }
void plugins_run_action(const std::string&, const std::string& action) { N.actions.push_back(action); }
namespace plugin_dialog {
bool available() { return N.available; }
bool open() {
    ++N.opens;
    return N.available;
}
void close() { ++N.closes; }
void changed(bool) { ++N.changes; }
bool up() { return false; }
bool corner_up() { return false; }
}  // namespace plugin_dialog

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: plugin_menus_test <menu_demo.so>\n");
        return 2;
    }
    Plugin p;
#if defined(_WIN32)
    p.handle = LoadLibraryA(argv[1]);
#else
    p.handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
#endif
    if (!p.handle) {
        std::printf("FAIL: cannot load %s\n", argv[1]);
        return 1;
    }
    p.image = reinterpret_cast<int (*)(const BbHostApi*)>(p.sym("bb_plugin_image"));
    p.options = static_cast<const BbOption*>(p.sym("bb_plugin_options"));
    CHECK(p.image && p.options);
    if (!p.image || !p.options) return 1;

    // ---- an older host: refused before anything is asked
    {
        const BbHostApi old = make_api(28);
        CHECK(p.image(&old) != 0);
        CHECK(H.rows.empty() && !H.option_fn);
    }
    // ---- its table as the host reads it
    std::vector<PluginOptionDesc> table;
    {
        std::vector<PluginOptionDesc>& t = table;
        for (const BbOption* o = p.options; o->type; ++o) {
            PluginOptionDesc d;
            d.type = o->type & 0xffu;
            d.flags = o->type & ~0xffu;
            d.key = o->key ? o->key : "";
            d.label = o->label ? o->label : "";
            d.help = o->help ? o->help : "";
            d.def = o->def ? o->def : "";
            d.choices = o->choices ? o->choices : "";
            d.min = o->min;
            d.max = o->max;
            t.push_back(d);
        }
        const plugin_menu::Layout l = plugin_menu::layout_of(t);
        CHECK(l.menus.size() == 2);
        CHECK(l.menus.size() == 2 && l.menus[0].name == "Travel" && l.menus[1].name == "Tools");
        CHECK(l.page("lamp") && l.page("lamp")->subpage);
        CHECK(plugin_menu::cycle_of(l, "lamps").size() == 2);  // lamps, history: the sub-page left out
        CHECK(plugin_menu::cycle_of(l, "tools").size() == 2);  // tools, more
        bool info = false, text = false, confirm = false;
        for (const PluginOptionDesc& d : t) {
            info = info || d.type == BB_OPT_INFO;
            text = text || d.type == BB_OPT_TEXT;
            confirm = confirm || (d.type == BB_OPT_ACTION && (d.flags & BB_OPT_CONFIRM));
        }
        CHECK(info && text && confirm);
    }
    // ---- the start: its list filled
    const BbHostApi api = make_api(29);
    CHECK(p.image(&api) == 0);
    CHECK(H.option_fn && H.action_fn);
    const Rows* lamps = last_rows("lamps");
    CHECK(lamps && lamps->rows.size() == 15);  // lit, sort, twelve lamps, random: more than a menu shows
    CHECK(lamps && lamps->rows[0].type == BB_OPT_INFO && lamps->rows[0].def == "2 of 12");
    CHECK(lamps && lamps->rows[1].type == BB_OPT_CHOICE && lamps->rows[1].choices == "Story|Name");
    CHECK(lamps && lamps->rows[2].def == "lamp" && (lamps->rows[2].flags & BB_OPT_OPEN));  // a lamp opens the lamp page
    int count = 0;
    CHECK(labels(lamps, &count) == "Hunter's Dream,Central Yharnam,Cathedral Ward (unlit),Forbidden Woods (unlit)");
    CHECK(count == 12);
    // ---- a lamp's button: the sub-page filled, opened by the host (its OPEN flag)
    H.action_fn("lamps/plaza", nullptr);
    const Rows* lamp = last_rows("lamp");
    CHECK(lamp && lamp->rows.size() == 4 && lamp->rows[0].def == "Yharnam" && lamp->rows[1].def == "Lit");
    CHECK(lamp && lamp->rows[2].type == BB_OPT_TEXT && lamp->rows[2].def == "Central Yharnam");
    CHECK(lamp && lamp->rows[3].type == BB_OPT_ACTION && (lamp->rows[3].flags & BB_OPT_CONFIRM));
    CHECK(H.opened.empty());
    // ---- "random": the plugin opens the page itself
    H.action_fn("lamps/random", nullptr);
    CHECK(H.opened.size() == 1 && H.opened[0] == "lamp");
    lamp = last_rows("lamp");
    CHECK(lamp && lamp->rows[0].def == "Yharnam" && lamp->rows[1].def == "Unlit" && lamp->rows[2].def == "Cathedral Ward");
    // ---- a run-time row's change, with its page: the list again in that order
    H.option_fn("lamps/sort", "Name", nullptr);
    CHECK(labels(last_rows("lamps")) == "Byrgenwerth (unlit),Cathedral Ward (unlit),Central Yharnam,Forbidden Woods (unlit)");
    CHECK(last_rows("lamps") && last_rows("lamps")->rows[1].def == "Name");
    H.option_fn("sort", "Story", nullptr);  // a table row's key: not the list's
    CHECK(labels(last_rows("lamps")) == "Byrgenwerth (unlit),Cathedral Ward (unlit),Central Yharnam,Forbidden Woods (unlit)");
    // ---- a TEXT row on the sub-page: the lamp renamed, in the list too
    H.option_fn("lamp/rename", "A Cathedral", nullptr);
    CHECK(labels(last_rows("lamps")) == "A Cathedral (unlit),Byrgenwerth (unlit),Central Yharnam,Forbidden Woods (unlit)");
    // ---- travel (after its confirm): two values, no new rows
    const std::size_t rows_before = H.rows.size();
    H.action_fn("lamp/travel", nullptr);
    CHECK(H.values.size() == 2 && H.values[0] == "lamps/lit=3 of 12" && H.values[1] == "lamp/state=Lit");
    CHECK(H.rows.size() == rows_before);
    // ---- a table row's action: the list again
    H.action_fn("reset", nullptr);
    CHECK(last_rows("lamps") && last_rows("lamps")->rows[0].def == "1 of 12");
    // ---- its own page list, in the corner
    H.action_fn("corner", nullptr);
    CHECK(H.shows == 1 && H.shown_page == "tools" && (H.shown_flags & BB_PANEL_CORNER));
    CHECK(H.shown_pages.size() == 2 && H.shown_pages[0] == "tools" && H.shown_pages[1] == "lamps");
    // ---- the game's own interruptions (host/plugin_menu.h), frame by frame
    {
        constexpr std::uint32_t kIn = BB_WORLD_IN, kLoad = BB_WORLD_LOADING, kCut = BB_WORLD_CUTSCENE;
        plugin_menu::Interruptions w;
        const auto step = [&](std::uint32_t flags, std::int64_t hp) {
            const char* why = w.step({flags, hp});
            return std::string(why ? why : "-");
        };
        CHECK(step(kIn, 76) == "-");
        CHECK(step(kLoad, 76) == "a loading screen");
        CHECK(step(kLoad, 76) == "-");  // still up: once
        CHECK(step(kIn, 76) == "-");    // the world again
        CHECK(step(kIn | kCut, 76) == "a cutscene");
        CHECK(step(kIn | kCut, 76) == "-");
        CHECK(step(kIn, 76) == "-");
        CHECK(step(kIn | BB_WORLD_MENU, 40) == "-");  // a message, a hit
        CHECK(step(kIn | BB_WORLD_PAUSED | BB_WORLD_OPEN | BB_WORLD_VISITING, 40) == "-");
        CHECK(step(kIn, -1) == "-");  // no character read
        CHECK(step(kIn, 0) == "the character's death");
        CHECK(step(kIn, 0) == "-");  // dead still: once
        CHECK(step(kLoad, 0) == "a loading screen");  // the load after it
        CHECK(step(0, 0) == "-");    // out of the world, 0 HP is no death
        CHECK(step(kLoad | kCut, 76) == "a loading screen");  // both at once: one interruption
        CHECK(step(0, 0) == "-");    // the title, its record at 0 HP: no death
    }
    {
        PluginEntry e;
        e.name = "menu_demo";
        e.title = "Menu demo";
        e.loaded = e.enabled = true;
        e.options = table;
        g_catalog = {e};
        constexpr std::uint32_t kIn = BB_WORLD_IN;
        PluginPanelShow centred, corner;
        centred.page = "lamps";
        corner.page = "tools";
        corner.corner = true;
        // In the world, a centred menu up: the game's messages and menus, a
        // pause, a player on the way or visiting - none closes it.
        plugin_panel_world(kIn, 76);
        CHECK(plugin_panel_show("menu_demo", centred));
        for (const std::uint32_t f : {kIn | BB_WORLD_MENU, kIn, kIn | BB_WORLD_PAUSED, kIn | BB_WORLD_OPEN, kIn | BB_WORLD_VISITING, kIn})
            plugin_panel_world(f, 76);
        plugin_panel_world(kIn, 40);  // hit, not dead
        PluginPanelState st = panel_state();
        CHECK(st.shown && !st.closed && plugin_panel_exclusive());
        // A loading screen comes up (lamp travel): closed as the player closes it, read once.
        plugin_panel_world(BB_WORLD_LOADING, 40);
        st = panel_state();
        CHECK(!st.shown && st.closed && st.page == "lamps");
        CHECK(!plugin_panel_shown() && !plugin_panel_exclusive());
        st = panel_state();
        CHECK(!st.shown && !st.closed);
        // Opened during the load: it stays through the load and the world's coming in.
        CHECK(plugin_panel_show("menu_demo", centred));
        plugin_panel_world(BB_WORLD_LOADING, 40);
        plugin_panel_world(kIn, 76);
        st = panel_state();
        CHECK(st.shown && !st.closed);
        // A cutscene starts.
        plugin_panel_world(kIn | BB_WORLD_CUTSCENE, 76);
        st = panel_state();
        CHECK(!st.shown && st.closed);
        // Opened during it: it stays through the cutscene and its end.
        CHECK(plugin_panel_show("menu_demo", centred));
        plugin_panel_world(kIn | BB_WORLD_CUTSCENE, 76);
        plugin_panel_world(kIn, 76);
        CHECK(panel_state().shown);
        // Taken down by its plugin (not "closed"); a cutscene and a load come
        // with nothing up, and a menu opened during that load stays.
        plugin_panel_hide("menu_demo");
        st = panel_state();
        CHECK(!st.shown && !st.closed);
        plugin_panel_world(kIn | BB_WORLD_CUTSCENE, 76);
        plugin_panel_world(BB_WORLD_LOADING, 76);
        CHECK(plugin_panel_show("menu_demo", centred));
        plugin_panel_world(BB_WORLD_LOADING, 76);
        plugin_panel_world(kIn, 76);
        st = panel_state();
        CHECK(st.shown && !st.closed);
        // The character dies.
        plugin_panel_world(kIn, 0);
        st = panel_state();
        CHECK(!st.shown && st.closed);
        // Opened on the death's screen: it stays until the load after it comes up.
        CHECK(plugin_panel_show("menu_demo", centred));
        plugin_panel_world(kIn, 0);
        CHECK(panel_state().shown);
        plugin_panel_world(BB_WORLD_LOADING, 0);
        st = panel_state();
        CHECK(!st.shown && st.closed);
        // A corner panel stays through every one of them; its plugin takes it down (not "closed").
        plugin_panel_world(kIn, 76);
        CHECK(plugin_panel_show("menu_demo", corner));
        for (const std::uint32_t f : {BB_WORLD_LOADING, kIn, kIn | BB_WORLD_CUTSCENE, kIn}) plugin_panel_world(f, 76);
        plugin_panel_world(kIn, 0);
        st = panel_state();
        CHECK(st.shown && !st.closed && !plugin_panel_exclusive());
        plugin_panel_hide("menu_demo");
        st = panel_state();
        CHECK(!st.shown && !st.closed);
    }
    // ---- a row's help on the game's dialog's one line (host/plugin_menu.h one_line)
    {
        using plugin_menu::one_line;
        CHECK(one_line("How fast.", 96) == "How fast.");
        CHECK(one_line("How fast. Slow is half, fast twice.", 96) == "How fast.");
        CHECK(one_line("No full stop at all", 96) == "No full stop at all");
        CHECK(one_line("Version 1.0 of it. More", 96) == "Version 1.0 of it.");  // a stop inside a number is not one
        CHECK(one_line("Line one\nline two. Then", 96) == "Line one line two.");
        CHECK(one_line("Every lamp but the Dream's goes out, and the rest of it too", 30) == "Every lamp but the Dream's...");
        CHECK(one_line("Supercalifragilisticexpialidocious", 12) == "Supercali...");  // one word longer than the line
        CHECK(one_line("Short words, cut here, after", 20) == "Short words, cut...");  // no comma before the dots
        CHECK(one_line("", 96).empty());
        CHECK(one_line("Ça va très bien. Oui", 96) == "Ça va très bien.");  // code points, not bytes
        CHECK(one_line("Ééééé ééééé", 8) == "Ééééé...");
        CHECK(one_line("Ééééé ééé", 9) == "Ééééé ééé");  // nine characters in eighteen bytes: it fits
        CHECK(one_line("Alpha beta gamma delta", 15) == "Alpha beta...");  // the room ends inside a word: cut before it
    }
    // ---- the game's own dialog: the page the host gives it, and the player's presses
    {
        N.available = true;
        const Rows* list = last_rows("lamps");
        CHECK(list && plugin_panel_rows("menu_demo", "lamps", list->rows));
        PluginPanelShow how;
        how.page = "lamps";
        const int opens = N.opens;
        CHECK(plugin_panel_show("menu_demo", how) && N.opens == opens + 1);
        CHECK(!plugin_panel_overlay_shown() && plugin_panel_shown());  // the game's dialog, not the host's panel
        plugin_dialog::Page pg;
        CHECK(plugin_panel_native_page(pg));
        CHECK(pg.title == "Lamps   1/2" && !pg.corner && list && pg.rows.size() == list->rows.size());
        if (pg.rows.size() >= 4 && list) {
            CHECK(pg.rows[0].widget == plugin_dialog::Widget::Text && pg.rows[0].value == list->rows[0].def && !pg.rows[0].steps &&
                  !pg.rows[0].presses);  // INFO
            CHECK(pg.rows[1].widget == plugin_dialog::Widget::Pair && pg.rows[1].pair[0] == "Story" && pg.rows[1].pair[1] == "Name" &&
                  pg.rows[1].on == (list->rows[1].def == "Name" ? 1 : 0));  // a choice of two
            CHECK(pg.rows[2].widget == plugin_dialog::Widget::Text && pg.rows[2].presses && pg.rows[2].label == list->rows[2].label);
            CHECK(pg.rows[1].help.find("L1 R1: page") != std::string::npos);  // a menu of pages: its keys after the help
        }
        // The choice of two set by the game's widget: the run-time row's value, with its page.
        N.options.clear();
        CHECK(plugin_panel_native_set(1, 1) && !N.options.empty() && N.options.back() == "lamps/sort=Name");
        // A lamp's button (OPEN): the plugin told, the sub-page up, Circle there goes back.
        std::string question;
        N.actions.clear();
        CHECK(plugin_panel_native_press(2, &question) == 2 && N.actions.size() == 1 && list && N.actions[0] == "lamps/" + list->rows[2].key);
        const Rows* one = last_rows("lamp");  // what the plugin filled the sub-page with (its on_action, earlier)
        CHECK(one && plugin_panel_rows("menu_demo", "lamp", one->rows));
        CHECK(plugin_panel_native_page(pg) && pg.title == "Lamp");  // a page on top shows no page count
        CHECK(plugin_panel_native_on_subpage());
        CHECK(plugin_panel_native_back());
        CHECK(plugin_panel_native_page(pg) && pg.title == "Lamps   1/2" && pg.cursor == 2);  // back on the button
        CHECK(!plugin_panel_native_on_subpage() && !plugin_panel_native_back());  // the menu's own page: Circle closes
        // L1 / R1.
        CHECK(plugin_panel_native_turn(1) && plugin_panel_native_page(pg) && pg.title == "History   2/2");
        CHECK(plugin_panel_native_turn(-1) && plugin_panel_native_page(pg) && pg.title == "Lamps   1/2");
        // Tools: a choice of three steps (a text row), a CONFIRM asks first, Defaults asks then puts every row back.
        how.page = "tools";
        how.defaults = true;
        CHECK(plugin_panel_show("menu_demo", how) && plugin_panel_native_page(pg));
        CHECK(pg.rows.size() == 4 && pg.rows.back().label == "Defaults" && pg.rows.back().presses);
        if (pg.rows.size() == 4) {
            CHECK(pg.rows[0].widget == plugin_dialog::Widget::Text && pg.rows[0].steps && pg.rows[0].value == "Normal");
            N.options.clear();
            CHECK(plugin_panel_native_step(0, 1) && N.options.size() == 1 && N.options[0] == "speed=Fast");
            CHECK(plugin_panel_native_step(0, 1) && N.options.back() == "speed=Slow");  // round
            CHECK(plugin_panel_native_page(pg) && pg.rows[0].value == "Slow");
            CHECK(!plugin_panel_native_step(1, 1));  // INFO does not step
            N.actions.clear();
            question.clear();
            CHECK(plugin_panel_native_press(2, &question) == 3 && question.rfind("Forget every lamp?", 0) == 0 && N.actions.empty());
            CHECK(!plugin_panel_native_confirm(2) && N.actions.size() == 1 && N.actions[0] == "reset");
            question.clear();
            CHECK(plugin_panel_native_press(3, &question) == 3 && !question.empty());
            N.options.clear();
            CHECK(!plugin_panel_native_confirm(3) && N.options.size() == 1 && N.options[0] == "speed=Normal");
        }
        // Numbers: an INT from 0 to 10 is the game's own slider; any other range the game's slider with a
        // position a step (1 for an INT, a hundredth of the range for a FLOAT), its number the host's.
        how.page = "more";
        how.defaults = false;
        CHECK(plugin_panel_show("menu_demo", how) && plugin_panel_native_page(pg) && pg.title == "More   2/2");
        CHECK(pg.rows.size() == 3);
        if (pg.rows.size() == 3) {
            CHECK(pg.rows[0].widget == plugin_dialog::Widget::Slider && pg.rows[0].positions == 0 && pg.rows[0].on == 3);
            CHECK(pg.rows[1].widget == plugin_dialog::Widget::Slider && pg.rows[1].positions == 100 && pg.rows[1].on == 60 &&
                  pg.rows[1].value == "10" && pg.rows[1].steps);
            CHECK(pg.rows[2].widget == plugin_dialog::Widget::Slider && pg.rows[2].positions == 100 && pg.rows[2].on == 25 &&
                  pg.rows[2].value == "0.25");
            N.options.clear();
            CHECK(plugin_panel_native_set(1, 61) && N.options.size() == 1 && N.options[0] == "level=11");  // a position
            CHECK(plugin_panel_native_set(0, 7) && N.options.back() == "count=7");                         // the game's number
            CHECK(plugin_panel_native_step(2, 1) && N.options.back() == "mix=0.26");                     // a step of a hundredth
            CHECK(plugin_panel_native_page(pg) && pg.rows[1].on == 61 && pg.rows[1].value == "11" && pg.rows[2].on == 26 &&
                  pg.rows[2].value == "0.26");
            CHECK(plugin_panel_native_step(1, -1) && plugin_panel_native_page(pg) && pg.rows[1].value == "10");
            N.config.erase("level");
            N.config.erase("mix");
            N.config.erase("count");
        }
        // The help line: the row's first sentence and the guide, inside the line's characters (the
        // CONFIRM row's 36-character help is cut to leave the guide its share).
        how.page = "tools";
        how.guide = "L1 R1: page    R3: show again";
        CHECK(plugin_panel_show("menu_demo", how) && plugin_panel_native_page(pg));
        for (const plugin_dialog::Row& r : pg.rows) {
            CHECK(r.help.size() <= kPanelHelpLine);
            CHECK(r.help.size() >= how.guide.size() && r.help.compare(r.help.size() - how.guide.size(), std::string::npos, how.guide) == 0);
        }
        how.guide.clear();
        // The game's dialog's keys: the menus' own, not Q and E (photo mode's camera flies on them).
        CHECK(plugin_panel_native_key(82) && plugin_panel_native_key(40) && plugin_panel_native_key(41) && plugin_panel_native_key(42));
        CHECK(!plugin_panel_native_key(20) && !plugin_panel_native_key(8) && plugin_panel_key(20) && plugin_panel_key(8));
        // A corner page the game's menu flow took is still wanted (the host brings it back); a centred one
        // is not, and neither is one its plugin took down.
        CHECK(!plugin_panel_native_corner_wanted());
        how.corner = true;
        CHECK(plugin_panel_show("menu_demo", how) && plugin_panel_native_corner_wanted());
        plugin_panel_hide("menu_demo");
        CHECK(!plugin_panel_native_corner_wanted());
        how.corner = false;
        CHECK(plugin_panel_show("menu_demo", how) && !plugin_panel_native_corner_wanted());
        // The game's dialog closed by the player: closed, read once.
        plugin_panel_native_gone(true);
        PluginPanelState st = panel_state();
        CHECK(!st.shown && st.closed);
        CHECK(!panel_state().closed);
        // Taken by the game's menu flow (a centred page): closed all the same, read once.
        CHECK(plugin_panel_show("menu_demo", how));
        plugin_panel_native_gone(true, "the game's menu flow");
        st = panel_state();
        CHECK(!st.shown && st.closed);
        CHECK(!panel_state().closed);
        // The System list's Plugins row: the named menus as a page; one picked is on top of it.
        CHECK(plugin_panel_native_menus_any());
        plugin_panel_native_list();
        CHECK(plugin_panel_native_page(pg) && pg.title == "Plugins" && pg.rows.size() == 2);
        CHECK(pg.rows.size() == 2 && pg.rows[0].label == "Travel" && pg.rows[1].label == "Tools" && pg.rows[1].presses);
        CHECK(plugin_panel_native_press(1, nullptr) == 2 && plugin_panel_native_page(pg) && pg.title == "Tools   1/2");
        CHECK(plugin_panel_native_on_subpage());  // its own page: Circle goes back to the list
        CHECK(plugin_panel_native_back() && plugin_panel_native_page(pg) && pg.title == "Plugins");
        st = panel_state();
        CHECK(!st.shown && st.closed);  // the plugin: its menu closed by the player
        plugin_panel_native_gone(true);
        CHECK(!plugin_panel_shown());
        // No game dialog (another eboot, an old movie): the host's own panel, as before.
        N.available = false;
        how = PluginPanelShow{};
        how.page = "lamps";
        CHECK(plugin_panel_show("menu_demo", how) && plugin_panel_overlay_shown() && !plugin_panel_native_page(pg));
        plugin_panel_hide("menu_demo");
    }
    if (g_failures) {
        std::printf("plugin_menus_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("plugin_menus_test: ok\n");
    return 0;
}
