// A plugin's in-game menus as the host keeps them (host/plugin_menu.h), with
// no game and no drawing: the pages and menus an options table declares - a
// plugin without BB_OPT_MENU (photo mode) one unnamed menu of all its pages,
// MENU headings each starting a named menu, SUBPAGE pages out of the L1/R1
// order - the rows a plugin sets at run time over a page's own and gives
// back, their values and the keys their events come with, and the menu up
// now: L1/R1 round its pages and not on a page opened on top, pages opened on
// top and Circle back to the row that opened each, then closing.
#include "host/plugin_menu.h"

#include "bbhost_plugin.h"

#include <cstdio>
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

PluginOptionDesc opt(std::uint32_t type, const char* key, const char* label, const char* def = "", std::uint32_t flags = 0) {
    PluginOptionDesc d;
    d.type = type;
    d.flags = flags;
    d.key = key;
    d.label = label;
    d.def = def;
    return d;
}

PluginOptionDesc heading(const char* key, const char* label, std::uint32_t flags, const char* def = "") {
    return opt(BB_OPT_HEADING, key, label, def, flags);
}

std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (const std::string& x : v) s += (s.empty() ? "" : ",") + x;
    return s;
}

}  // namespace

int main() {
    using namespace plugin_menu;
    // ---- a plugin without MENU headings (photo mode's table): one menu of every page
    {
        const std::vector<PluginOptionDesc> t = {
            opt(BB_OPT_HEADING, "about", "Photo mode"),  // a plain heading: no page
            opt(BB_OPT_CHOICE, "start", "Start with"),
            heading("camera", "Camera", BB_OPT_PANEL),
            opt(BB_OPT_CHOICE, "mode", "Mode"),
            opt(BB_OPT_INT, "fov", "Field of view"),
            heading("dof", "Depth of field", BB_OPT_PANEL),
            opt(BB_OPT_CHOICE, "dof", "Depth of field"),
        };
        const Layout l = layout_of(t);
        CHECK(l.pages.size() == 2);
        CHECK(l.menus.empty());
        CHECK(l.page("camera") && l.page("camera")->rows.size() == 2 && l.page("camera")->title == "Camera");
        CHECK(l.page("dof") && l.page("dof")->rows.size() == 1);
        CHECK(!l.page("about"));
        CHECK(join(cycle_of(l, "camera")) == "camera,dof");
        CHECK(join(cycle_of(l, "dof")) == "camera,dof");
        CHECK(cycle_of(l, "nowhere").empty());
    }
    // ---- named menus, sub-pages, the rows under each
    const std::vector<PluginOptionDesc> t = {
        heading("travel", "Lamps", BB_OPT_PANEL | BB_OPT_MENU, "Travel"),
        opt(BB_OPT_ACTION, "refresh", "Refresh"),
        heading("lamp", "Lamp", BB_OPT_PANEL | BB_OPT_SUBPAGE),
        opt(BB_OPT_INFO, "area", "Area"),
        heading("history", "History", BB_OPT_PANEL),
        heading("tools", "Tools", BB_OPT_PANEL | BB_OPT_MENU),  // no def: named by its label
        opt(BB_OPT_BOOL, "god", "Invincible"),
        heading("more", "More", BB_OPT_PANEL),
        heading("plain", "Plain", 0),  // not a page: its rows on no page
        opt(BB_OPT_BOOL, "x", "X"),
    };
    Layout l = layout_of(t);
    CHECK(l.pages.size() == 5);
    CHECK(l.menus.size() == 2);
    CHECK(l.menus.size() == 2 && l.menus[0].name == "Travel" && join(l.menus[0].pages) == "travel,history");
    CHECK(l.menus.size() == 2 && l.menus[1].name == "Tools" && join(l.menus[1].pages) == "tools,more");
    CHECK(join(cycle_of(l, "travel")) == "travel,history");
    CHECK(join(cycle_of(l, "history")) == "travel,history");
    CHECK(join(cycle_of(l, "more")) == "tools,more");
    CHECK(join(cycle_of(l, "lamp")) == "lamp");  // a sub-page: only itself
    CHECK(l.page("lamp") && l.page("lamp")->subpage && l.page("lamp")->menu == "travel");
    CHECK(l.page("more") && l.page("more")->rows.empty());
    CHECK(l.page("tools") && l.page("tools")->rows.size() == 1);  // "plain"'s row is on no page
    // ---- rows set at run time over a page's own, their values and events
    Runtime rt;
    const Page* travel = l.page("travel");
    CHECK(rows_of(*travel, rt).size() == 1 && !rows_of(*travel, rt)[0].runtime);
    CHECK(!set_rows(l, rt, "nowhere", {opt(BB_OPT_ACTION, "a", "A")}));
    CHECK(set_rows(l, rt, "travel", {opt(BB_OPT_ACTION, "hd", "Hunter's Dream"), opt(BB_OPT_HEADING, "h", "H"),
                                     opt(BB_OPT_INFO, "count", "Lamps lit", "3"), opt(BB_OPT_CHOICE, "sort", "Sort", "name")}));
    {
        const std::vector<Row>& r = rows_of(*travel, rt);
        CHECK(r.size() == 3);  // a heading is no row
        CHECK(r.size() == 3 && r[0].runtime && r[0].desc.key == "hd" && r[1].desc.key == "count");
        CHECK(r.size() == 3 && event_key("travel", r[0]) == "travel/hd");
        CHECK(r.size() == 3 && value_of(rt, "travel", r[1]) == "3");
        CHECK(set_value(rt, "travel", "count", "4"));
        CHECK(r.size() == 3 && value_of(rt, "travel", r[1]) == "4");
        CHECK(!set_value(rt, "travel", "nothing", "1"));
        CHECK(!set_value(rt, "history", "count", "1"));  // a page with no rows of its own
    }
    CHECK(event_key("tools", l.page("tools")->rows[0]) == "god");  // a table row: its own key
    // new rows start from their defaults again; none gives the table's back
    CHECK(set_rows(l, rt, "travel", {opt(BB_OPT_INFO, "count", "Lamps lit", "5")}));
    CHECK(rows_of(*travel, rt).size() == 1 && value_of(rt, "travel", rows_of(*travel, rt)[0]) == "5");
    CHECK(set_rows(l, rt, "travel", {}));
    CHECK(rows_of(*travel, rt).size() == 1 && rows_of(*travel, rt)[0].desc.key == "refresh");
    CHECK(rt.values.empty());
    // another page's values stay as they are
    CHECK(set_rows(l, rt, "lamp", {opt(BB_OPT_INFO, "area", "Area", "Yharnam")}));
    CHECK(set_value(rt, "lamp", "area", "Cathedral Ward"));
    CHECK(set_rows(l, rt, "history", {opt(BB_OPT_INFO, "n", "N", "0")}));
    CHECK(value_of(rt, "lamp", rows_of(*l.page("lamp"), rt)[0]) == "Cathedral Ward");
    // ---- the menu up: L1/R1 round its pages, pages opened on top, Circle back
    Nav nav;
    nav.cycle = cycle_of(l, "travel");
    CHECK(nav.shown() == "travel");
    CHECK(nav.turn(1) && nav.shown() == "history");
    CHECK(nav.turn(1) && nav.shown() == "travel");  // round
    CHECK(nav.turn(-1) && nav.shown() == "history");
    CHECK(!nav.turn(0));
    nav.open("lamp", 3);
    CHECK(nav.shown() == "lamp" && nav.on_subpage());
    CHECK(!nav.turn(1) && nav.shown() == "lamp");  // no L1/R1 on a page opened on top
    nav.open("lamp", 0);  // the same page again, from its own row 0
    CHECK(nav.back() == 0 && nav.shown() == "lamp");
    CHECK(nav.back() == 3 && nav.shown() == "history" && !nav.on_subpage());
    CHECK(nav.back() == -1);  // Circle on a menu page: it closes
    Nav one;
    one.cycle = {"only"};
    CHECK(!one.turn(1) && one.shown() == "only");
    Nav none;
    CHECK(none.shown().empty() && none.back() == -1);
    if (g_failures) {
        std::printf("plugin_menu_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("plugin_menu_test: ok\n");
    return 0;
}
