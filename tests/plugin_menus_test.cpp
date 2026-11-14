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
// corner (panel_show with pages and BB_PANEL_CORNER).
//
//   plugin_menus_test <menu_demo.so>
#include "bbhost_plugin.h"
#include "host/plugin_menu.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdarg>
#include <cstdio>
#include <cstring>
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

}  // namespace

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
    {
        std::vector<PluginOptionDesc> t;
        for (const BbOption* o = p.options; o->type; ++o) {
            PluginOptionDesc d;
            d.type = o->type & 0xffu;
            d.flags = o->type & ~0xffu;
            d.key = o->key ? o->key : "";
            d.label = o->label ? o->label : "";
            d.def = o->def ? o->def : "";
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
    if (g_failures) {
        std::printf("plugin_menus_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("plugin_menus_test: ok\n");
    return 0;
}
