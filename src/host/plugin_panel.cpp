#define IMGUI_DEFINE_MATH_OPERATORS
#include "host/plugin_panel.h"

#include "bbhost_plugin.h"
#include "engine/menu_pointer.h"
#include "engine/plugin_dialog.h"
#include "host/game_menu_art.h"
#include "host/plugin_menu.h"
#include "host/plugins.h"
#include "log.h"

#include <algorithm>
#include <cfloat>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

#if defined(BBHOST_HAVE_SDL3)
#include "host/bindings.h"
#include "host/launcher_ui.h"
#include "host/text_entry.h"
#include "host/window.h"
#include "imgui.h"
#include "imgui_internal.h"
#endif

namespace {

struct Panel {
    std::string owner;                         // the plugin whose panel is up; empty: none
    std::shared_ptr<const PluginEntry> entry;  // its options, as they were when it was shown
    std::shared_ptr<const plugin_menu::Layout> layout;  // its pages and menus (host/plugin_menu.h)
    plugin_menu::Nav nav;                      // the menu up: its L1/R1 pages and the pages opened on top
    std::string guide, guide_keys;
    bool defaults = false;
    bool corner = false;     // BB_PANEL_CORNER; else the game's dialog, centred, the screen dimmed
    int focused_row = 0;     // the row the cursor is on (the draw's): a page opened on top comes back to it
    int cursor_to = -1;      // the row the cursor goes to on the next frame (back from a page on top)
    bool native = false;     // in the game's own options dialog (engine/plugin_dialog.h), not the host's panel
    bool from_list = false;  // opened from the Plugins list: Circle on its menu's own page goes back to it
};
std::mutex g_mu;
Panel g_panel;                                     // under g_mu
std::map<std::string, bool> g_closed;              // under g_mu: closed (the player, the game) since its plugin last asked
plugin_menu::Interruptions g_interruptions;        // under g_mu: the world's last frame (plugin_panel_world)
std::map<std::string, std::string> g_last_page;    // under g_mu: the menu page each plugin's panel showed last
std::map<std::string, plugin_menu::Runtime> g_runtime;  // under g_mu: what each plugin set at run time
std::map<std::string, std::shared_ptr<const plugin_menu::Layout>> g_layouts;  // under g_mu: each plugin's, read once
std::atomic<bool> g_shown{false};
std::atomic<bool> g_native{false};                 // the panel up is in the game's own dialog
bool g_list = false;                               // under g_mu: the game's dialog shows the Plugins list
std::atomic<bool> g_exclusive{false};              // the panel up is a centred menu: every input its own
std::atomic<bool> g_appeared{false};               // shown since the last frame drew it: give it the focus
std::atomic<bool> g_opened{false};                 // shown, not just a page turned: the game's menu-opening sound
// The menu sound asked for since plugins_frame last took one: (priority << 16) | id + 1, 0 none.
std::atomic<std::uint32_t> g_sound{0};

// The game's menu sound for something the player did on the panel: the
// loudest asked for since the game's thread last took one.
void ask_sound(int id) {
    const std::uint32_t priority = id == kMenuSoundDecide || id == kMenuSoundCancel ? 1 : 0;
    const std::uint32_t want = (priority << 16) | static_cast<std::uint32_t>(id + 1);
    std::uint32_t was = g_sound.load(std::memory_order_relaxed);
    while ((was == 0 || (was >> 16) <= priority) && !g_sound.compare_exchange_weak(was, want, std::memory_order_relaxed)) {
    }
}

// A loaded plugin's catalog entry (its options table as the menus have it).
bool catalog_entry(const std::string& name, PluginEntry& out) {
    for (PluginEntry& e : plugins_catalog()) {
        if (e.name == name && !e.visitor) {
            out = std::move(e);
            return true;
        }
    }
    return false;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream in(s);
    while (std::getline(in, cur, sep)) out.push_back(cur);
    return out;
}

// A number's step as the menus take it: 1 for an INT, a hundredth of its range for a FLOAT.
double number_step(const PluginOptionDesc& o) { return o.type == BB_OPT_INT ? 1.0 : (o.max - o.min) / 100.0; }

// A row's value as text for the config (plugins_set_option).
std::string number_text(const PluginOptionDesc& o, double v) {
    char t[32];
    if (o.type == BB_OPT_INT) std::snprintf(t, sizeof(t), "%d", static_cast<int>(std::lround(v)));
    else std::snprintf(t, sizeof(t), "%.3g", v);
    return t;
}

// Greyed out (BB_OPT_WHEN) unless another of the plugin's options holds one of its values.
bool greyed(const PluginEntry& p, const PluginOptionDesc& o) {
    const std::size_t eq = o.when.find('=');
    if (eq == std::string::npos) return false;
    const std::string key = o.when.substr(0, eq);
    const std::vector<std::string> want = split(o.when.substr(eq + 1), '|');
    for (const PluginOptionDesc& other : p.options) {
        if (other.key == key) return std::find(want.begin(), want.end(), plugins_option_value(p, other)) == want.end();
    }
    return false;
}

// Whether a plugin's menu goes into the game's own dialog: it can open
// there (engine/plugin_dialog.h), and no test asks for the host's panel
// (BBHOST_PANEL_OVERLAY=1).
bool native_wanted() {
    static const bool overlay = [] {
        const char* e = std::getenv("BBHOST_PANEL_OVERLAY");
        return e && e[0] == '1';
    }();
    return !overlay && plugin_dialog::available();
}

// The panel up in the host's own drawing after all: the game's dialog could
// not take it (logged once).
void fall_back(const char* why) {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_panel.native) return;
        g_panel.native = false;
        g_native.store(false);
        g_exclusive.store(!g_panel.corner && g_shown.load());
        g_opened.store(true);
    }
    static bool told = false;
    if (!told) {
        told = true;
        host_log("plugin: a plugin's menu is drawn by the host: %s", why);
    }
}

// A plugin's pages and menus, read from its table the first time (the table
// does not change while it runs). Not under g_mu: the catalog is read
// without it.
std::shared_ptr<const plugin_menu::Layout> layout_for(const std::string& name) {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const auto it = g_layouts.find(name);
        if (it != g_layouts.end()) return it->second;
    }
    PluginEntry e;
    if (!catalog_entry(name, e)) return nullptr;
    auto l = std::make_shared<const plugin_menu::Layout>(plugin_menu::layout_of(e.options));
    std::lock_guard<std::mutex> lk(g_mu);
    return g_layouts.emplace(name, l).first->second;
}

}  // namespace

int plugin_panel_take_sound(int* priority) {
    const std::uint32_t s = g_sound.exchange(0, std::memory_order_relaxed);
    if (priority) *priority = static_cast<int>(s >> 16);
    return s ? static_cast<int>(s & 0xffffu) - 1 : -1;
}

void plugin_panel_preload() { game_menu_art::preload(); }

void plugin_panel_preview_tick() {
    static const std::string want = [] {
        const char* e = std::getenv("BBHOST_PANEL_PREVIEW");
        return std::string(e ? e : "");
    }();
    static bool done = want.empty();
    static int tries = 0;
    if (done) return;
    if (++tries > 3600) {  // a minute of frames and no such plugin
        done = true;
        host_log("plugin: no panel to preview for BBHOST_PANEL_PREVIEW=%s", want.c_str());
        return;
    }
    PluginPanelShow how;
    const std::size_t colon = want.find(':');
    const std::string name = want.substr(0, colon);
    if (colon != std::string::npos) how.page = want.substr(colon + 1);
    const char* g = std::getenv("BBHOST_PANEL_PREVIEW_GUIDE");
    const char* gk = std::getenv("BBHOST_PANEL_PREVIEW_GUIDE_KEYS");
    const char* corner = std::getenv("BBHOST_PANEL_PREVIEW_CORNER");
    how.guide = g ? g : "";
    how.guide_keys = gk ? gk : "";
    how.corner = corner && corner[0] == '1';
    how.defaults = how.corner;  // photo mode's: its corner panel has Defaults rows
    if (plugin_panel_show(name, how)) {
        done = true;
        host_log("plugin: %s's panel shown for a preview (BBHOST_PANEL_PREVIEW)", name.c_str());
    }
}

namespace {

// The panel's state for `name` at `how` (plugin_panel_show), under g_mu.
bool set_panel(const std::string& name, const PluginPanelShow& how) {
    PluginEntry entry;
    if (!catalog_entry(name, entry)) return false;
    const std::shared_ptr<const plugin_menu::Layout> layout = layout_for(name);
    std::lock_guard<std::mutex> lk(g_mu);
    if (!layout || layout->pages.empty()) return false;
    for (const std::string& pg : how.pages) {
        if (!layout->page(pg)) return false;  // a page it has not got
    }
    if (!how.page.empty() && !layout->page(how.page)) return false;
    std::string want = how.page.empty() ? g_last_page[name] : how.page;
    if (!how.pages.empty() && std::find(how.pages.begin(), how.pages.end(), want) == how.pages.end()) {
        if (!how.page.empty()) return false;  // a page outside the menu it names
        want = how.pages.front();
    }
    if (want.empty() || !layout->page(want)) {
        want = layout->pages.front().key;
        for (const plugin_menu::Page& pg : layout->pages) {
            if (!pg.subpage) {
                want = pg.key;
                break;
            }
        }
    }
    plugin_menu::Nav nav;
    nav.cycle = how.pages.empty() ? plugin_menu::cycle_of(*layout, want) : how.pages;
    nav.at = static_cast<int>(std::find(nav.cycle.begin(), nav.cycle.end(), want) - nav.cycle.begin());
    if (nav.at >= static_cast<int>(nav.cycle.size())) nav.at = 0;
    const bool was = g_shown.load() && g_panel.owner == name;
    g_panel = Panel{};
    g_panel.owner = name;
    g_panel.entry = std::make_shared<const PluginEntry>(std::move(entry));
    g_panel.layout = layout;
    g_panel.nav = std::move(nav);
    g_panel.guide = how.guide;
    g_panel.guide_keys = how.guide_keys;
    g_panel.defaults = how.defaults;
    g_panel.corner = how.corner;
    g_panel.native = native_wanted();
    g_list = false;
    g_last_page[name] = g_panel.nav.shown();
    if (!was) {
        g_appeared.store(true);
        g_opened.store(!g_panel.native);  // the game's dialog plays its own
        host_log("plugin: %s's panel up at its page %s%s%s", name.c_str(), g_last_page[name].c_str(), how.corner ? "" : ", centred",
                 g_panel.native ? ", in the game's dialog" : "");
    }
    g_native.store(g_panel.native);
    g_exclusive.store(!how.corner && !g_panel.native);
    g_shown.store(true);
    return true;
}

}  // namespace

bool plugin_panel_show(const std::string& name, const PluginPanelShow& how) {
    if (!set_panel(name, how)) return false;
    bool native = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        native = g_panel.native && g_panel.owner == name;
    }
    if (native && !plugin_dialog::open()) fall_back("the game's dialog cannot open it");
    return true;
}

void plugin_panel_hide(const std::string& name) {
    bool native = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner != name) return;
        native = g_panel.native;
        g_panel.owner.clear();
        g_exclusive.store(false);
        g_native.store(false);
        if (g_shown.exchange(false)) host_log("plugin: %s's panel down", name.c_str());
    }
    if (native) plugin_dialog::close();
}

void plugin_panel_state(const std::string& name, PluginPanelState& out) {
    std::lock_guard<std::mutex> lk(g_mu);
    out.shown = g_shown.load() && g_panel.owner == name;
    auto it = g_closed.find(name);
    out.closed = it != g_closed.end() && it->second;
    if (it != g_closed.end()) it->second = false;
    out.page = out.shown ? g_panel.nav.shown() : g_last_page[name];
}

void plugin_panel_world(std::uint32_t flags, std::int64_t hp) {
    bool native = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const char* why = g_interruptions.step({flags, hp});
        if (!why || !g_shown.load() || g_panel.corner || (g_panel.owner.empty() && !g_list)) return;
        if (!g_panel.owner.empty()) g_closed[g_panel.owner] = true;
        host_log("plugin: %s's menu closed by %s", g_panel.owner.empty() ? "the Plugins list" : g_panel.owner.c_str(), why);
        native = g_panel.native || g_list;
        g_panel.owner.clear();
        g_list = false;
        g_exclusive.store(false);
        g_native.store(false);
        g_shown.store(false);
    }
    if (native) plugin_dialog::close();
}

bool plugin_panel_rows(const std::string& name, const std::string& page, const std::vector<PluginOptionDesc>& rows) {
    const std::shared_ptr<const plugin_menu::Layout> layout = layout_for(name);
    bool shown_native = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!layout || !plugin_menu::set_rows(*layout, g_runtime[name], page, rows)) return false;
        shown_native = g_panel.native && g_panel.owner == name && g_panel.nav.shown() == page;
    }
    if (shown_native) plugin_dialog::changed(true);
    return true;
}

bool plugin_panel_value(const std::string& name, const std::string& page, const std::string& key, const std::string& value) {
    bool shown_native = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const auto it = g_runtime.find(name);
        if (it == g_runtime.end() || !plugin_menu::set_value(it->second, page, key, value)) return false;
        shown_native = g_panel.native && g_panel.owner == name && g_panel.nav.shown() == page;
    }
    if (shown_native) plugin_dialog::changed(false);
    return true;
}

bool plugin_panel_open(const std::string& name, const std::string& page) {
    bool native = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_shown.load() || g_panel.owner != name || !g_panel.layout || !g_panel.layout->page(page)) return false;
        g_panel.nav.open(page, g_panel.focused_row);
        g_appeared.store(true);
        native = g_panel.native;
    }
    if (native) plugin_dialog::changed(true);
    return true;
}

std::vector<PluginMenuEntry> plugin_panel_menus(const std::string& name) {
    std::vector<PluginMenuEntry> out;
    const std::shared_ptr<const plugin_menu::Layout> layout = layout_for(name);
    if (!layout) return out;
    for (const plugin_menu::MenuDecl& m : layout->menus) {
        if (!m.pages.empty()) out.push_back({m.pages.front(), m.name});
    }
    return out;
}

// ---- the game's own dialog (engine/plugin_dialog.h) -------------------------------

namespace {

// The Plugins list's rows: each loaded plugin's named menus (host/plugin_menu.h).
struct ListRow {
    std::string plugin, page, name, help;
};
std::vector<ListRow> list_rows() {
    std::vector<ListRow> out;
    for (const PluginEntry& e : plugins_catalog()) {
        if (!e.loaded || e.visitor || !e.enabled) continue;
        const std::shared_ptr<const plugin_menu::Layout> layout = layout_for(e.name);
        if (!layout) continue;
        for (const plugin_menu::MenuDecl& m : layout->menus) {
            if (!m.pages.empty()) out.push_back({e.name, m.pages.front(), m.name, e.title.empty() ? e.name : e.title});
        }
    }
    return out;
}

// The controller in use, for which guide the help line gives.
bool keys_in_use() {
#if defined(BBHOST_HAVE_SDL3)
    return host_input_device() == InputDevice::KeyboardMouse;
#else
    return false;
#endif
}

// The keys that turn a menu's pages in the game's dialog: the player's own
// Menu Previous Page and Menu Next Page bindings (PageUp and PageDown unless
// moved), or L1 and R1.
std::string page_keys() {
#if defined(BBHOST_HAVE_SDL3)
    if (keys_in_use()) {
        const char* l = host_key_name(host_binding_key(kBindMenuPageL));
        const char* r = host_key_name(host_binding_key(kBindMenuPageR));
        if (l && r) return std::string(l) + " " + r + ": page";
    }
#endif
    return "L1 R1: page";
}

// A row's line in the game's dialog: its first sentence (host/plugin_menu.h
// one_line), with the plugin's guide - or the menu's pages' keys - after it,
// the two together in the characters the dialog's help line holds inside its
// frame (kPanelHelpLine; the guide's own share first, the help at least 20).
std::string native_help(const std::string& help, const std::string& guide) {
    const std::size_t room = guide.empty() ? kPanelHelpLine : std::max<std::size_t>(20, kPanelHelpLine - std::min(kPanelHelpLine, guide.size() + 4));
    const std::string line = plugin_menu::one_line(help, room);
    if (guide.empty()) return line;
    return line.empty() ? guide : line + "    " + guide;
}

constexpr const char* kDefaultsLabel = "Defaults";

// An INT from 0 to 10: the game's own slider, its number the game's.
bool game_slider(const PluginOptionDesc& o) { return o.type == BB_OPT_INT && o.min == 0.0 && o.max == 10.0; }

// A number as the host's slider shows it: a FLOAT to its step's decimals
// (a range of 1 in hundredths, of 10 in tenths, of 100 in whole numbers).
std::string shown_number(const PluginOptionDesc& o, double v) {
    char t[32];
    if (o.type == BB_OPT_INT) {
        std::snprintf(t, sizeof(t), "%d", static_cast<int>(std::lround(v)));
    } else {
        const double step = number_step(o);
        const int places = step > 0.0 ? std::clamp(static_cast<int>(std::ceil(-std::log10(step) - 1e-9)), 0, 4) : 2;
        std::snprintf(t, sizeof(t), "%.*f", places, v);
    }
    return t;
}

// The snapshot a native call works on (under g_mu).
struct NativeView {
    Panel p;
    plugin_menu::Runtime rt;
    std::string page_key;
    const plugin_menu::Page* page = nullptr;
    std::vector<plugin_menu::Row> rows;
};
bool native_view(NativeView& v) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_panel.owner.empty() || !g_panel.layout || !g_panel.native) return false;
    v.p = g_panel;
    const auto it = g_runtime.find(v.p.owner);
    if (it != g_runtime.end()) v.rt = it->second;
    v.page_key = v.p.nav.shown();
    v.page = v.p.layout->page(v.page_key);
    if (!v.page) return false;
    v.rows = plugin_menu::rows_of(*v.page, v.rt);
    return true;
}

std::string value_now(const NativeView& v, const plugin_menu::Row& r) {
    return r.runtime ? plugin_menu::value_of(v.rt, v.page_key, r) : plugins_option_value(*v.p.entry, r.desc);
}

// Greyed out (BB_OPT_WHEN) unless the row it names holds one of its values:
// a run-time row of this page, else one of the table's.
bool grey_now(const NativeView& v, const plugin_menu::Row& r) {
    const std::size_t eq = r.desc.when.find('=');
    if (eq == std::string::npos) return false;
    const std::string key = r.desc.when.substr(0, eq);
    const std::vector<std::string> want = split(r.desc.when.substr(eq + 1), '|');
    for (const plugin_menu::Row& other : v.rows) {
        if (other.runtime && other.desc.key == key) return std::find(want.begin(), want.end(), value_now(v, other)) == want.end();
    }
    return greyed(*v.p.entry, r.desc);
}

// A row's value set as the menus set it: the config's for a table row, the
// menu's own for a run-time one, the plugin told either way.
void apply_native(const NativeView& v, const plugin_menu::Row& r, const std::string& value) {
    if (r.runtime) {
        {
            std::lock_guard<std::mutex> lk(g_mu);
            plugin_menu::set_value(g_runtime[v.p.owner], v.page_key, r.desc.key, value);
        }
        plugins_menu_option(v.p.owner, plugin_menu::event_key(v.page_key, r), value);
    } else {
        plugins_set_option(v.p.owner, r.desc.key, value);
    }
}

// A TEXT row being typed in the host's text box, from the game's dialog.
struct NativeTyping {
    bool on = false;
    std::string owner, page, key;
    bool runtime = false;
} g_native_typing;  // under g_mu

}  // namespace

bool plugin_panel_native_page(plugin_dialog::Page& out) {
    out = plugin_dialog::Page{};
    bool list = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        list = g_list;
    }
    if (list) {
        // (list_rows reads the layouts, which take g_mu themselves.)
        out.title = "Plugins";
        for (const ListRow& l : list_rows()) {
            plugin_dialog::Row r;
            r.label = l.name;
            r.help = native_help(l.help + ".", "");
            r.presses = true;
            out.rows.push_back(std::move(r));
        }
        return !out.rows.empty();
    }
    NativeView v;
    if (!native_view(v)) return false;
    const int pages = static_cast<int>(v.p.nav.cycle.size());
    out.title = v.page->title;
    if (pages > 1 && !v.p.nav.on_subpage()) out.title += "   " + std::to_string(v.p.nav.at + 1) + "/" + std::to_string(pages);
    out.corner = v.p.corner;
    std::string guide = keys_in_use() ? v.p.guide_keys : v.p.guide;
    if (guide.empty() && pages > 1 && !v.p.nav.on_subpage()) guide = page_keys();
    for (const plugin_menu::Row& r : v.rows) {
        plugin_dialog::Row row;
        const PluginOptionDesc& o = r.desc;
        row.label = o.label;
        row.help = native_help(o.help, guide);
        row.grey = grey_now(v, r);
        const std::string value = value_now(v, r);
        switch (o.type) {
        case BB_OPT_BOOL:
            row.widget = plugin_dialog::Widget::Toggle;
            row.on = value == "true" || value == "1";
            row.value = row.on ? "On" : "Off";
            row.steps = true;  // as a text row (a long page): Left and Right switch it
            break;
        case BB_OPT_CHOICE: {
            const std::vector<std::string> choices = split(o.choices, '|');
            int cur = 0;
            for (std::size_t i = 0; i < choices.size(); ++i) {
                if (choices[i] == value) cur = static_cast<int>(i);
            }
            row.value = choices.empty() ? std::string() : choices[static_cast<std::size_t>(cur)];
            if (choices.size() == 2) {
                row.widget = plugin_dialog::Widget::Pair;
                row.pair[0] = choices[0];
                row.pair[1] = choices[1];
                row.on = cur;
            }
            row.steps = choices.size() > 1;
            break;
        }
        case BB_OPT_INT:
        case BB_OPT_FLOAT: {
            const double x = std::atof(value.c_str());
            row.value = shown_number(o, x);
            row.widget = plugin_dialog::Widget::Slider;
            if (game_slider(o)) {
                row.on = std::clamp(static_cast<int>(std::lround(x)), 0, 10);
            } else {
                // The game's slider with a position a step of the plugin's, and its number.
                const double step = number_step(o);
                const double span = o.max > o.min && step > 0.0 ? (o.max - o.min) / step : 1.0;
                row.positions = static_cast<int>(std::clamp(std::lround(span), 1L, 1L << 20));
                const long at = step > 0.0 ? std::lround((x - o.min) / step) : 0L;
                row.on = static_cast<int>(std::clamp(at, 0L, static_cast<long>(row.positions)));
            }
            row.steps = true;
            break;
        }
        case BB_OPT_TEXT:
            row.value = value;
            row.presses = true;
            break;
        case BB_OPT_ACTION:
            row.presses = true;
            break;
        default:  // BB_OPT_INFO, and a type this host does not know
            row.value = value;
            break;
        }
        out.rows.push_back(std::move(row));
    }
    if (v.p.defaults) {
        plugin_dialog::Row d;
        d.label = kDefaultsLabel;
        d.help = native_help("Every row here back to its default.", guide);
        d.presses = true;
        out.rows.push_back(std::move(d));
    }
    out.cursor = std::clamp(v.p.cursor_to >= 0 ? v.p.cursor_to : 0, 0, std::max(0, static_cast<int>(out.rows.size()) - 1));
    return !out.rows.empty();
}

bool plugin_panel_native_step(int row, int dir) {
    NativeView v;
    if (!native_view(v) || row < 0 || row >= static_cast<int>(v.rows.size()) || !dir) return false;
    const plugin_menu::Row& r = v.rows[static_cast<std::size_t>(row)];
    const PluginOptionDesc& o = r.desc;
    const std::string value = value_now(v, r);
    if (o.type == BB_OPT_BOOL) {
        apply_native(v, r, value == "true" || value == "1" ? "false" : "true");
        return true;
    }
    if (o.type == BB_OPT_CHOICE) {
        const std::vector<std::string> choices = split(o.choices, '|');
        const int n = static_cast<int>(choices.size());
        if (n < 2) return false;
        int cur = 0;
        for (int i = 0; i < n; ++i) {
            if (choices[static_cast<std::size_t>(i)] == value) cur = i;
        }
        apply_native(v, r, choices[static_cast<std::size_t>(((cur + dir) % n + n) % n)]);
        return true;
    }
    if (o.type == BB_OPT_INT || o.type == BB_OPT_FLOAT) {
        const double was = std::atof(value.c_str());
        const double to = std::clamp(was + dir * number_step(o), o.min, o.max);
        if (to == was) return false;
        apply_native(v, r, number_text(o, to));
        return true;
    }
    return false;
}

bool plugin_panel_native_set(int row, int value) {
    NativeView v;
    if (!native_view(v) || row < 0 || row >= static_cast<int>(v.rows.size())) return false;
    const plugin_menu::Row& r = v.rows[static_cast<std::size_t>(row)];
    const PluginOptionDesc& o = r.desc;
    if (o.type == BB_OPT_BOOL) {
        apply_native(v, r, value ? "true" : "false");
    } else if (o.type == BB_OPT_CHOICE) {
        const std::vector<std::string> choices = split(o.choices, '|');
        if (value < 0 || value >= static_cast<int>(choices.size())) return false;
        apply_native(v, r, choices[static_cast<std::size_t>(value)]);
    } else if (o.type == BB_OPT_INT || o.type == BB_OPT_FLOAT) {
        // The game's own slider gives the number; one of the host's, its position.
        const double to = game_slider(o) ? value : std::clamp(o.min + value * number_step(o), o.min, o.max);
        apply_native(v, r, number_text(o, to));
    } else {
        return false;
    }
    return true;
}

int plugin_panel_native_press(int row, std::string* question) {
    // The Plugins list: the menu picked, its first page on top of the list.
    {
        std::unique_lock<std::mutex> lk(g_mu);
        if (g_list) {
            lk.unlock();
            const std::vector<ListRow> rows = list_rows();
            if (row < 0 || row >= static_cast<int>(rows.size())) return 0;
            PluginPanelShow how;
            how.page = rows[static_cast<std::size_t>(row)].page;
            if (!set_panel(rows[static_cast<std::size_t>(row)].plugin, how)) return 0;
            std::lock_guard<std::mutex> again(g_mu);
            g_panel.native = true;
            g_panel.from_list = true;
            g_native.store(true);
            g_exclusive.store(false);
            return 2;
        }
    }
    NativeView v;
    if (!native_view(v) || row < 0) return 0;
    if (v.p.defaults && row == static_cast<int>(v.rows.size())) {
        if (question) *question = "Every row on this page back to its default?";
        return 3;
    }
    if (row >= static_cast<int>(v.rows.size())) return 0;
    const plugin_menu::Row& r = v.rows[static_cast<std::size_t>(row)];
    const PluginOptionDesc& o = r.desc;
    if (o.type == BB_OPT_TEXT) {
#if defined(BBHOST_HAVE_SDL3)
        host_text_entry_begin(value_now(v, r).c_str(), 128, false, o.label.c_str(), TextCharset::BasicLatin, true);
        std::lock_guard<std::mutex> lk(g_mu);
        g_native_typing = {true, v.p.owner, v.page_key, o.key, r.runtime};
#endif
        return 1;
    }
    if (o.type != BB_OPT_ACTION) return 0;
    if (o.flags & BB_OPT_CONFIRM) {
        if (question) *question = o.label + "?" + (o.help.empty() ? std::string() : "\n" + plugin_menu::one_line(o.help, 96));
        return 3;
    }
    plugins_run_action(v.p.owner, plugin_menu::event_key(v.page_key, r));
    if ((o.flags & BB_OPT_OPEN) && v.p.layout->page(o.def)) {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner == v.p.owner) {
            g_panel.nav.open(o.def, row);
            g_panel.cursor_to = 0;
        }
        return 2;
    }
    return 1;
}

bool plugin_panel_native_confirm(int row) {
    NativeView v;
    if (!native_view(v) || row < 0) return false;
    if (v.p.defaults && row == static_cast<int>(v.rows.size())) {
        for (const plugin_menu::Row& r : v.rows) {
            if (r.desc.type != BB_OPT_ACTION && r.desc.type != BB_OPT_INFO) apply_native(v, r, r.desc.def);
        }
        return false;
    }
    if (row >= static_cast<int>(v.rows.size())) return false;
    const plugin_menu::Row& r = v.rows[static_cast<std::size_t>(row)];
    plugins_run_action(v.p.owner, plugin_menu::event_key(v.page_key, r));
    if ((r.desc.flags & BB_OPT_OPEN) && v.p.layout->page(r.desc.def)) {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner == v.p.owner) {
            g_panel.nav.open(r.desc.def, row);
            g_panel.cursor_to = 0;
        }
        return true;
    }
    return false;
}

bool plugin_panel_native_turn(int dir) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_list || g_panel.owner.empty() || !g_panel.native || !g_panel.nav.turn(dir)) return false;
    g_last_page[g_panel.owner] = g_panel.nav.shown();
    g_panel.cursor_to = 0;
    return true;
}

bool plugin_panel_native_on_subpage() {
    std::lock_guard<std::mutex> lk(g_mu);
    return !g_list && !g_panel.owner.empty() && g_panel.native && (g_panel.nav.on_subpage() || g_panel.from_list);
}

bool plugin_panel_native_back() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_list || g_panel.owner.empty() || !g_panel.native) return false;
    if (g_panel.nav.on_subpage()) {
        g_panel.cursor_to = g_panel.nav.back();
        g_last_page[g_panel.owner] = g_panel.nav.shown();
        return true;
    }
    if (g_panel.from_list) {
        // Out of a menu the list opened: the list again (the plugin's menu closed by the player).
        g_closed[g_panel.owner] = true;
        host_log("plugin: %s's panel closed by the player (back to the Plugins list)", g_panel.owner.c_str());
        g_panel = Panel{};
        g_list = true;
        return true;
    }
    return false;
}

void plugin_panel_native_gone(bool closed, const char* by) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_panel.owner.empty() && g_panel.native) {
        if (closed) {
            g_closed[g_panel.owner] = true;
            host_log("plugin: %s's panel closed by %s", g_panel.owner.c_str(), by ? by : "the player");
        }
        g_panel.owner.clear();
        g_shown.store(false);
    } else if (g_list) {
        g_shown.store(false);
    }
    g_list = false;
    g_native.store(false);
    g_exclusive.store(false);
}

bool plugin_panel_native_corner_wanted() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_shown.load() && g_panel.native && g_panel.corner && !g_panel.owner.empty();
}

void plugin_panel_native_failed(const char* why) {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_list) {
            // The Plugins list has no page of the host's: nothing to show.
            g_list = false;
            g_native.store(false);
            g_shown.store(false);
            return;
        }
    }
    fall_back(why);
}

bool plugin_panel_native_menus_any() { return !list_rows().empty(); }

void plugin_panel_native_list() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_panel = Panel{};
    g_list = true;
    g_native.store(true);
    g_exclusive.store(false);
    g_shown.store(true);
    host_log("plugin: the Plugins list up in the game's dialog");
}

void plugin_panel_native_frame() {
#if defined(BBHOST_HAVE_SDL3)
    NativeTyping t;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_native_typing.on) return;
        t = g_native_typing;
    }
    std::string text;
    const int r = host_text_entry_poll(text);
    if (r == 0) return;
    host_text_entry_end();
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_native_typing.on = false;
        if (r == 1 && g_panel.owner == t.owner) {
            if (t.runtime) plugin_menu::set_value(g_runtime[t.owner], t.page, t.key, text);
        }
    }
    if (r == 1) {
        if (t.runtime) plugins_menu_option(t.owner, t.page + "/" + t.key, text);
        else plugins_set_option(t.owner, t.key, text);
        plugin_dialog::changed(false);
    }
#endif
}

bool plugin_panel_shown() { return g_shown.load(std::memory_order_relaxed); }

bool plugin_panel_overlay_shown() { return g_shown.load(std::memory_order_relaxed) && !g_native.load(std::memory_order_relaxed); }

bool plugin_panel_exclusive() { return g_shown.load(std::memory_order_relaxed) && g_exclusive.load(std::memory_order_relaxed); }

bool plugin_panel_native_centred() { return plugin_dialog::up() && !plugin_dialog::corner_up(); }

std::uint32_t plugin_panel_dialog_buttons() { return plugin_panel_shown() && !plugin_panel_overlay_shown() ? kPanelPadButtons : 0u; }

bool plugin_panel_key(int sc) {
    // SDL scancodes (USB usages): the arrows, Enter (and the keypad's), Page
    // Up, Page Down, Q, E, Escape and Backspace.
    switch (sc) {
    case 79: case 80: case 81: case 82:  // right, left, down, up
    case 40: case 88:                    // Return, keypad Enter
    case 75: case 78:                    // Page Up, Page Down
    case 20: case 8:                     // Q, E
    case 41: case 42:                    // Escape, Backspace
        return true;
    default:
        return false;
    }
}

bool plugin_panel_native_key(int sc) {
    switch (sc) {
    case 79: case 80: case 81: case 82:  // right, left, down, up
    case 40: case 88:                    // Return, keypad Enter
    case 41: case 42:                    // Escape, Backspace
        return true;
    default:
#if defined(BBHOST_HAVE_SDL3)
        return sc != 0 && (sc == host_binding_key(kBindMenuPageL) || sc == host_binding_key(kBindMenuPageR));
#else
        return false;
#endif
    }
}

#if defined(BBHOST_HAVE_SDL3)

namespace {

// A step left or right on the row with the focus: the d-pad or the arrows, held to repeat.
int step_pressed() {
    int d = 0;
    if (ImGui::IsKeyPressed(ImGuiKey_GamepadDpadLeft, true) || ImGui::IsKeyPressed(ImGuiKey_LeftArrow, true)) d -= 1;
    if (ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight, true) || ImGui::IsKeyPressed(ImGuiKey_RightArrow, true)) d += 1;
    return d;
}

// The value a number row shows while the pointer drags it: set when it lets go.
std::map<std::string, double> g_drag;

// The game's own menus' look - the System menu's pages, bbhost's PC pages
// among them (the options movie it makes from the game's): a dark plate
// between thin pale rules with an ornament at their middle, the page's name in
// muted gold, the rows in a warm off-white with the label set right against
// the middle and the value centred after it, a faint line under each row, the
// row with the cursor lit by a golden band and its value between two arrows,
// and the help under the rows in pale beige. Not recoloured by Menu hue and
// Menu vibrance, which are bbhost's menus' own (host/ingame_menu.cpp).
namespace look {
constexpr ImU32 kPlate = IM_COL32(7, 6, 5, 218);
constexpr ImU32 kPlateEdge = IM_COL32(0, 0, 0, 0);
constexpr ImU32 kRule = IM_COL32(184, 178, 164, 170);
constexpr ImU32 kTitle = IM_COL32(203, 172, 114, 255);
constexpr ImU32 kText = IM_COL32(228, 222, 209, 255);
constexpr ImU32 kTextLit = IM_COL32(255, 249, 230, 255);
constexpr ImU32 kTextDim = IM_COL32(150, 142, 127, 255);
constexpr ImU32 kHelp = IM_COL32(212, 202, 180, 255);
constexpr ImU32 kGuide = IM_COL32(170, 161, 142, 255);
constexpr ImU32 kGlow = IM_COL32(226, 172, 78, 120);
constexpr ImU32 kRowLine = IM_COL32(168, 160, 146, 64);
constexpr ImU32 kGold = IM_COL32(206, 172, 108, 255);
constexpr ImU32 kGoldLit = IM_COL32(236, 202, 136, 255);
constexpr ImU32 kGrab = IM_COL32(206, 172, 108, 150);     // a slider's mark: thin, under its number
constexpr ImU32 kGrabLit = IM_COL32(236, 202, 136, 210);
constexpr ImU32 kFrame = IM_COL32(255, 244, 220, 16);
constexpr ImU32 kFrameHot = IM_COL32(255, 244, 220, 28);
}  // namespace look

// A thin rule across [x0, x1] at y with the game's small ornament at its
// middle and its ends: a diamond, and short ticks.
void rule(ImDrawList* dl, float x0, float x1, float y, float s) {
    const float mid = (x0 + x1) * 0.5f, d = 4.0f * s;
    dl->AddLine(ImVec2(x0, y), ImVec2(mid - d * 2.2f, y), look::kRule, 1.0f);
    dl->AddLine(ImVec2(mid + d * 2.2f, y), ImVec2(x1, y), look::kRule, 1.0f);
    dl->AddQuad(ImVec2(mid - d, y), ImVec2(mid, y - d), ImVec2(mid + d, y), ImVec2(mid, y + d), look::kRule, 1.0f);
    dl->AddCircleFilled(ImVec2(mid, y), 1.3f * s, look::kRule);
    for (const float x : {x0, x1}) dl->AddLine(ImVec2(x, y - 2.5f * s), ImVec2(x, y + 2.5f * s), look::kRule, 1.0f);
}

// The arrows either side of the row with the cursor's value (the game's < >).
void value_arrows(ImDrawList* dl, ImVec2 a, ImVec2 b, float s, ImU32 col) {
    const float cy = (a.y + b.y) * 0.5f, h = 5.0f * s, w = 4.5f * s, in = 6.0f * s;
    dl->AddTriangleFilled(ImVec2(a.x + in, cy), ImVec2(a.x + in + w, cy - h), ImVec2(a.x + in + w, cy + h), col);
    dl->AddTriangleFilled(ImVec2(b.x - in, cy), ImVec2(b.x - in - w, cy - h), ImVec2(b.x - in - w, cy + h), col);
}

// The game's own menu art (host/game_menu_art.h), where the dump has it: the
// options dialog's pieces drawn as it draws them, at `k` of their pixels to a
// screen pixel. Each piece the dump lacks is drawn as above instead.
struct Art {
    const game_menu_art::Atlas* atlas = nullptr;
    ImTextureRef tex;
    float k = 1.0f;
    bool has(game_menu_art::Piece p) const { return atlas && atlas->piece[p].present; }
    float w(game_menu_art::Piece p) const { return static_cast<float>(atlas->piece[p].w) * k; }
    float h(game_menu_art::Piece p) const { return static_cast<float>(atlas->piece[p].h) * k; }
    // The piece's columns [px0, px1) of its own pixels (all of it by default)
    // over [a, b], mirrored across x and/or y.
    void image(ImDrawList* dl, game_menu_art::Piece p, ImVec2 a, ImVec2 b, ImU32 tint, bool flip_x = false, bool flip_y = false,
               float px0 = 0.0f, float px1 = -1.0f) const {
        const game_menu_art::Placed& q = atlas->piece[p];
        if (px1 < 0.0f) px1 = static_cast<float>(q.w);
        float u0 = q.u0 + (q.u1 - q.u0) * px0 / static_cast<float>(q.w), u1 = q.u0 + (q.u1 - q.u0) * px1 / static_cast<float>(q.w);
        float v0 = q.v0, v1 = q.v1;
        if (flip_x) std::swap(u0, u1);
        if (flip_y) std::swap(v0, v1);
        dl->AddImage(tex, a, b, ImVec2(u0, v0), ImVec2(u1, v1), tint);
    }
    // A five-part piece (the ornament, the band) across [x0, x1] from `top`:
    // its ends and middle at their size, the rule between them stretched.
    bool span(ImDrawList* dl, game_menu_art::Piece p, float x0, float x1, float top, ImU32 tint, bool flip_y) const {
        game_menu_art::Span sp{};
        const game_menu_art::Slices& s = game_menu_art::slices(p);
        if (!game_menu_art::stretch(s, x1 - x0, k, sp)) return false;
        for (int i = 0; i < 5; ++i) {
            if (sp.x1[i] <= sp.x0[i]) continue;
            image(dl, p, ImVec2(x0 + sp.x0[i], top), ImVec2(x0 + sp.x1[i], top + h(p)), tint, false, flip_y, s.x[i], s.x[i + 1]);
        }
        return true;
    }
    // A plain rule across [x0, x1] centred on y, its thin line at least a
    // pixel high.
    void rule(ImDrawList* dl, float x0, float x1, float y, ImU32 tint) const {
        const float hh = std::max(h(game_menu_art::kRule), 3.0f);
        image(dl, game_menu_art::kRule, ImVec2(x0, y - hh * 0.5f), ImVec2(x1, y + hh * 0.5f), tint);
    }
};

// The art at this frame's scale, or none: the atlas and its texture in the
// menus' context (registered with it the first time).
Art menu_art(float s) {
    Art a;
    a.atlas = game_menu_art::menu_art();
    if (!a.atlas) return a;
    ImTextureData* t = game_menu_art_texture();
    if (!t) {
        a.atlas = nullptr;
        return a;
    }
    a.tex = t->GetTexRef();
    // The options dialog draws its ornaments at about 0.4 of their pixels in
    // a 720-line picture; the panel is 560 wide at a scale of 1, 420 at 720.
    a.k = 0.53f * s;
    return a;
}

// The nav cursor onto the item just drawn, as the d-pad would put it there:
// SetKeyboardFocusHere would open a slider's text field instead.
void take_cursor(bool now) {
    if (!now) return;
    ImGui::SetFocusID(ImGui::GetItemID(), ImGui::GetCurrentWindow());
    ImGui::SetNavCursorVisible(true);
}

// A value shown as text across the value column, a button the pointer, Cross
// and Enter press: choices, switches and the page's Defaults.
bool value_button(const char* id, const std::string& text, ImVec2 size, bool lit) {
    ImGui::PushStyleColor(ImGuiCol_Text, lit ? look::kTextLit : look::kText);
    const bool pressed = ImGui::Button((text + "###" + id).c_str(), size);
    ImGui::PopStyleColor();
    return pressed;
}

// The rows' layers in the draw list's channels: the cursor's glow under the
// rows' tracks and lines, as the game's dialog lights a row behind them, and
// those under the widgets and the letters.
constexpr int kGlowLayer = 0, kLineLayer = 1, kWidgetLayer = 2;

struct RowIn {
    const PluginOptionDesc* o = nullptr;
    std::string value;    // its value now: the config's (a table row) or the menu's (a run-time row)
    bool grey = false;    // BB_OPT_WHEN not met
    std::string drag_id;  // where a slider keeps its value while the pointer drags it
};

struct RowOut {
    bool hot = false;      // the cursor or the pointer on it: its help shows
    bool focused = false;  // the cursor on it: lit
    bool changed = false;  // the player changed its value to `value`
    std::string value;
    bool pressed = false;  // a button pressed (an ACTION), or a TEXT row's box asked for
    float glow_x = -1.0f;  // a slider's: its value's glow centred on its track, as the game's are (not on the mark)
};

// One row in [a, b]: its label right-aligned before `mid`, its widget across
// the value column after it. With the game's art: the label's line fading out
// to the left, the value's line with its scrolled ends, a slider's track and
// its filled part (all under the widget, the draw list's first channel), and
// a choice's and a slider's arrows the dialog's own. What the player does to
// it comes back in RowOut; the caller keeps it (the config or the menu's
// run-time values).
RowOut row(const RowIn& in, ImVec2 a, ImVec2 b, float mid, float s, bool cursor, const Art& art) {
    RowOut out;
    const PluginOptionDesc& o = *in.o;
    ImGui::PushID(o.key.c_str());
    const std::string& value = in.value;
    const bool grey = in.grey;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float frame_h = ImGui::GetFrameHeight();
    const float gap = 14.0f * s;
    const ImVec2 val_a(mid + gap, a.y + (b.y - a.y - frame_h) * 0.5f);
    const ImVec2 val_size(b.x - val_a.x, frame_h);
    const ImU32 art_tint = grey ? IM_COL32(255, 255, 255, 110) : IM_COL32(255, 255, 255, 235);
    ImGui::BeginDisabled(grey);
    ImGui::SetCursorScreenPos(val_a);
    ImGui::SetNextItemWidth(val_size.x);
    bool arrows = false;      // the value is one of a list: arrows either side while it has the cursor
    bool slider = false;      // a number on a track: arrows either side of the track and the number, as the game's
    bool less = false, more = false;  // a slider's value is above its least, below its most: the arrow that way shows
    bool underline = false;   // its value's line under it (the game's art)
    const auto change = [&](const std::string& v) {
        out.changed = true;
        out.value = v;
        ask_sound(kMenuSoundMove);
    };
    switch (o.type) {
    case BB_OPT_BOOL: {
        const bool on = value == "true" || value == "1";
        bool to = on;
        if (value_button(o.key.c_str(), on ? "On" : "Off", val_size, false)) to = !on;
        take_cursor(cursor);
        if (ImGui::IsItemFocused() && !grey && step_pressed() != 0) to = !on;
        if (to != on) change(to ? "true" : "false");
        arrows = underline = true;
        break;
    }
    case BB_OPT_INT:
    case BB_OPT_FLOAT: {
        const std::string& id = in.drag_id;
        const auto it = g_drag.find(id);
        double v = it != g_drag.end() ? it->second : std::atof(value.c_str());
        bool changed = false;
        // The bar, and its number beside it on the right, as the game's
        // sliders have theirs (not under the bar's mark).
        const auto shown = [&](double x) {
            char t[32];
            if (o.type == BB_OPT_INT) std::snprintf(t, sizeof(t), "%d", static_cast<int>(std::lround(x)));
            else std::snprintf(t, sizeof(t), "%.2f", x);
            return std::string(t);
        };
        ImFont* nf = ImGui::GetFont();
        const float nfs = ImGui::GetFontSize();
        const float num_w = std::max(nf->CalcTextSizeA(nfs, FLT_MAX, 0.0f, shown(o.min).c_str()).x,
                                     nf->CalcTextSizeA(nfs, FLT_MAX, 0.0f, shown(o.max).c_str()).x) +
                            gap;
        const float bar_w = std::max(val_size.x - num_w, 8.0f * s);
        const bool track = art.has(game_menu_art::kSliderTrack) && art.has(game_menu_art::kSliderFill);
        if (track) {  // the game's track and filling, a thin mark of ours: none of the widget's own
            ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, IM_COL32(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, IM_COL32(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_SliderGrab, IM_COL32(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, IM_COL32(0, 0, 0, 0));
        }
        ImGui::SetNextItemWidth(bar_w);
        if (o.type == BB_OPT_INT) {
            int iv = static_cast<int>(std::lround(v));
            if (ImGui::SliderInt("##v", &iv, static_cast<int>(o.min), static_cast<int>(o.max), "")) g_drag[id] = iv;
            v = iv;
        } else {
            float fv = static_cast<float>(v);
            if (ImGui::SliderFloat("##v", &fv, static_cast<float>(o.min), static_cast<float>(o.max), "")) g_drag[id] = fv;
            v = fv;
        }
        const float f = o.max > o.min ? std::clamp(static_cast<float>((v - o.min) / (o.max - o.min)), 0.0f, 1.0f) : 0.0f;
        const float fx = val_a.x + 2.0f * s + (bar_w - 4.0f * s) * f;
        out.glow_x = val_a.x + bar_w * 0.5f;  // the track's middle, as the game lights its sliders
        if (track) {
            ImGui::PopStyleColor(5);
            const float th = std::max(frame_h * 0.42f, 4.0f), fh = th * 0.5f;
            const float cy = val_a.y + frame_h * 0.5f;
            dl->ChannelsSetCurrent(kLineLayer);
            art.image(dl, game_menu_art::kSliderTrack, ImVec2(val_a.x, cy - th * 0.5f), ImVec2(val_a.x + bar_w, cy + th * 0.5f), art_tint);
            if (fx > val_a.x + 2.0f * s)
                art.image(dl, game_menu_art::kSliderFill, ImVec2(val_a.x + 2.0f * s, cy - fh * 0.5f), ImVec2(fx, cy + fh * 0.5f),
                          art_tint, false, false, 0.0f, std::max(1.0f, art.atlas->piece[game_menu_art::kSliderFill].w * f));
            dl->ChannelsSetCurrent(kWidgetLayer);
            const bool lit = ImGui::IsItemFocused() || ImGui::IsItemActive();
            const ImU32 mark = grey ? look::kTextDim : lit ? look::kGoldLit : look::kGold;
            dl->AddRectFilled(ImVec2(fx - 1.5f * s, cy - th * 0.8f), ImVec2(fx + 1.5f * s, cy + th * 0.8f), mark);
        }
        {
            const std::string t = shown(v);
            const ImVec2 ts = nf->CalcTextSizeA(nfs, FLT_MAX, 0.0f, t.c_str());
            const ImU32 col = grey ? look::kTextDim : ImGui::IsItemFocused() ? look::kTextLit : look::kText;
            dl->AddText(nf, nfs, ImVec2(b.x - ts.x, val_a.y + (frame_h - ts.y) * 0.5f), col, t.c_str());
        }
        const double ends = (o.max - o.min) * 1e-6;  // a float's value read back from its text, at an end
        slider = true;
        less = v > o.min + ends;
        more = v < o.max - ends;
        take_cursor(cursor);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            changed = true;
            g_drag.erase(id);
        }
        if (!ImGui::IsItemActive() && ImGui::IsItemFocused() && !grey) {
            if (const int d = step_pressed()) {
                v = std::clamp(v + d * number_step(o), o.min, o.max);
                changed = true;
            }
        }
        if (changed) change(number_text(o, v));
        break;
    }
    case BB_OPT_CHOICE: {
        const std::vector<std::string> choices = split(o.choices, '|');
        int cur = 0;
        for (std::size_t i = 0; i < choices.size(); ++i) {
            if (choices[i] == value) cur = static_cast<int>(i);
        }
        const int n = static_cast<int>(choices.size());
        int to = cur;
        if (value_button(o.key.c_str(), n ? choices[static_cast<std::size_t>(cur)] : std::string(), val_size, false) && n)
            to = (cur + 1) % n;
        take_cursor(cursor);
        if (ImGui::IsItemFocused() && !grey && n) {
            if (const int d = step_pressed()) to = (cur + d + n) % n;
        }
        if (to != cur) change(choices[static_cast<std::size_t>(to)]);
        arrows = underline = true;
        break;
    }
    case BB_OPT_TEXT:
        // Its text; Cross, Enter or a click asks for the host's text box.
        if (value_button(o.key.c_str(), value, val_size, false)) out.pressed = true;
        take_cursor(cursor);
        underline = true;
        break;
    case BB_OPT_ACTION:
        ImGui::SetCursorScreenPos(ImVec2(a.x, val_a.y));
        if (value_button(o.key.c_str(), o.label, ImVec2(b.x - a.x, frame_h), false)) out.pressed = true;
        take_cursor(cursor);
        break;
    default: {
        // BB_OPT_INFO (and a type this host does not know): its value as
        // text, which the cursor passes over; the pointer still shows its help.
        ImFont* f = ImGui::GetFont();
        const float fs = ImGui::GetFontSize();
        const ImVec2 ts = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, value.c_str());
        dl->AddText(f, fs, ImVec2(val_a.x + (val_size.x - ts.x) * 0.5f, val_a.y + (frame_h - ts.y) * 0.5f), look::kText, value.c_str());
        ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
        ImGui::Dummy(val_size);
        ImGui::PopItemFlag();
        break;
    }
    }
    out.focused = ImGui::IsItemFocused();
    out.hot = out.focused || ImGui::IsItemHovered();
    // The value's line, its scrolled ends hanging under it (the art's left
    // end turned over, mirrored on the right), and the label's.
    if (underline && art.has(game_menu_art::kValueEnd) && art.has(game_menu_art::kRule)) {
        const float y = b.y - 4.0f * s;
        const float ew = art.w(game_menu_art::kValueEnd), eh = art.h(game_menu_art::kValueEnd);
        const float x0 = val_a.x + 2.0f * s, x1 = b.x - 2.0f * s;
        dl->ChannelsSetCurrent(kLineLayer);
        art.image(dl, game_menu_art::kValueEnd, ImVec2(x0, y - 1.0f), ImVec2(x0 + ew, y - 1.0f + eh), art_tint, false, true);
        art.image(dl, game_menu_art::kValueEnd, ImVec2(x1 - ew, y - 1.0f), ImVec2(x1, y - 1.0f + eh), art_tint, true, true);
        art.rule(dl, x0 + ew - 1.0f, x1 - ew + 1.0f, y, art_tint);
        dl->ChannelsSetCurrent(kWidgetLayer);
    }
    if (o.type != BB_OPT_ACTION && art.has(game_menu_art::kLabelLine)) {
        const float lh = std::max(art.h(game_menu_art::kLabelLine), 2.0f);
        dl->ChannelsSetCurrent(kLineLayer);
        art.image(dl, game_menu_art::kLabelLine, ImVec2(a.x, b.y - 2.0f * s - lh * 0.5f), ImVec2(mid - 3.0f * s, b.y - 2.0f * s + lh * 0.5f),
                  art_tint);
        dl->ChannelsSetCurrent(kWidgetLayer);
    }
    // The arrows, on the row with the cursor: a choice's at the ends of its
    // value; a slider's either side of its track and number, as the game's -
    // the left one in the middle of the gap between the column rule and the
    // track, the right one as far past the number - each gone at its end.
    if ((arrows || slider) && out.focused) {
        const float cy = val_a.y + frame_h * 0.5f, side = (val_a.x - mid) * 0.5f;  // a slider's arrows' middles: mid + side, b.x + side
        if (art.has(game_menu_art::kArrow)) {
            const float ah = std::round(ImGui::GetFontSize() * 1.05f), aw = ah * art.atlas->piece[game_menu_art::kArrow].w /
                                                                             static_cast<float>(art.atlas->piece[game_menu_art::kArrow].h);
            const float in_ = 4.0f * s;
            const ImU32 t = grey ? IM_COL32(255, 255, 255, 110) : IM_COL32(255, 255, 255, 255);
            float l0 = val_a.x + in_, l1 = val_a.x + in_ + aw, r0 = b.x - in_ - aw, r1 = b.x - in_;
            if (slider) {
                l0 = mid + side - aw * 0.5f;
                l1 = l0 + aw;
                r0 = b.x + side - aw * 0.5f;
                r1 = r0 + aw;
            }
            if (!slider || less) art.image(dl, game_menu_art::kArrow, ImVec2(l0, cy - ah * 0.5f), ImVec2(l1, cy + ah * 0.5f), t, true);
            if (!slider || more) art.image(dl, game_menu_art::kArrow, ImVec2(r0, cy - ah * 0.5f), ImVec2(r1, cy + ah * 0.5f), t);
        } else if (slider) {
            const float h = 5.0f * s, w = 4.5f * s, lm = mid + side, rm = b.x + side;
            const ImU32 col = grey ? look::kTextDim : look::kGold;
            if (less) dl->AddTriangleFilled(ImVec2(lm - w * 0.5f, cy), ImVec2(lm + w * 0.5f, cy - h), ImVec2(lm + w * 0.5f, cy + h), col);
            if (more) dl->AddTriangleFilled(ImVec2(rm + w * 0.5f, cy), ImVec2(rm - w * 0.5f, cy - h), ImVec2(rm - w * 0.5f, cy + h), col);
        } else {
            value_arrows(dl, val_a, val_a + val_size, s, grey ? look::kTextDim : look::kGold);
        }
    }
    // The label, set against the middle, as the game's are.
    if (o.type != BB_OPT_ACTION) {
        ImFont* f = ImGui::GetFont();
        const float fs = ImGui::GetFontSize();
        const ImVec2 ts = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, o.label.c_str());
        ImU32 col = out.focused ? look::kTextLit : look::kText;
        if (grey) col = look::kTextDim;
        dl->AddText(f, fs, ImVec2(mid - gap - ts.x, (a.y + b.y - ts.y) * 0.5f), col, o.label.c_str());
    }
    ImGui::EndDisabled();
    ImGui::PopID();
    return out;
}

// The help's and the guide's lines. The game sets its multi-line texts (its
// dialogs' help, its item descriptions) a line's height apart plus their
// fields' leading - 2 px on its 1080-line stage, in nearly all of them - and
// so does the panel, at its scale (900 lines at a scale of 1).
constexpr float kLeading = 2.0f * 900.0f / 1080.0f;

// How far the font's descenders (g j p q y, the brackets) reach under a line's
// box at `size`: the game's letters go a quarter of an em under the baseline,
// twice the descent the font gives, which is all a line's height keeps.
float descender_overhang(ImFont* f, float size) {
    ImFontBaked* b = f->GetFontBaked(size);
    float bottom = size;
    for (const char* c = "gjpqy()[]{},;"; *c; ++c) {
        if (const ImFontGlyph* g = b->FindGlyphNoFallback(static_cast<ImWchar>(*c))) bottom = std::max(bottom, g->Y1);
    }
    return bottom - size;
}

// `text` wrapped at `width` in `f` at `size`, line by line (ImGui's word wrap,
// an explicit new line kept): each line's [begin, end) without its trailing
// spaces.
std::vector<std::pair<const char*, const char*>> wrap_lines(ImFont* f, float size, const std::string& text, float width) {
    std::vector<std::pair<const char*, const char*>> lines;
    const char* at = text.c_str();
    const char* end = at + text.size();
    while (at < end) {
        const char* brk = f->CalcWordWrapPosition(size, at, end, width);
        if (const char* nl = static_cast<const char*>(std::memchr(at, '\n', static_cast<std::size_t>(brk - at)))) brk = nl + 1;
        if (brk == at) brk = at + 1;
        const char* stop = brk;
        while (stop > at && (stop[-1] == ' ' || stop[-1] == '\n')) --stop;
        lines.emplace_back(at, stop);
        at = brk;
        while (at < end && *at == ' ') ++at;
    }
    return lines;
}

// Text across [x0, x1] from `top`, wrapped at the width, a line every
// `advance`; each line centred (as the game's dialogs set their help) or
// from x0.
void text_lines(ImDrawList* dl, ImFont* f, float size, float x0, float x1, float top, float advance, ImU32 col, const std::string& text,
                bool centred) {
    float y = top;
    for (const auto& [a, b] : wrap_lines(f, size, text, x1 - x0)) {
        const float w = centred ? f->CalcTextSizeA(size, FLT_MAX, 0.0f, a, b).x : 0.0f;
        dl->AddText(f, size, ImVec2(centred ? (x0 + x1 - w) * 0.5f : x0, y), col, a, b);
        y += advance;
    }
}

// The controller's guide with the game's key guide icons for the buttons it
// names (L1, R1, L3, R3, Circle, Cross, Triangle, Square, d-pad), centred;
// false, and nothing drawn, when the art lacks one it names.
bool guide_with_icons(ImDrawList* dl, ImFont* f, float size, float x0, float x1, float y, ImU32 col, const std::string& text,
                      const Art& art) {
    using game_menu_art::Piece;
    static const std::pair<const char*, Piece> kButtons[] = {
        {"L1", game_menu_art::kIconL1},          {"R1", game_menu_art::kIconR1},          {"L2", game_menu_art::kIconL2},
        {"R2", game_menu_art::kIconR2},          {"L3", game_menu_art::kIconL3},          {"R3", game_menu_art::kIconR3},
        {"Circle", game_menu_art::kIconCircle},  {"Cross", game_menu_art::kIconCross},  {"Triangle", game_menu_art::kIconTriangle},
        {"Square", game_menu_art::kIconSquare},  {"d-pad", game_menu_art::kIconDpad},
    };
    struct Bit {
        std::string text;
        int icon = -1;
    };
    std::vector<Bit> bits;
    std::size_t i = 0;
    while (i < text.size()) {
        std::size_t j = i;
        if (text[i] == ' ') {
            while (j < text.size() && text[j] == ' ') ++j;
            bits.push_back({text.substr(i, j - i), -1});
        } else {
            while (j < text.size() && text[j] != ' ') ++j;
            std::string word = text.substr(i, j - i), after;
            if (!word.empty() && word.back() == ':') {
                after = ":";
                word.pop_back();
            }
            int icon = -1;
            for (const auto& b : kButtons)
                if (word == b.first) icon = b.second;
            if (icon >= 0) {
                if (!art.has(static_cast<Piece>(icon))) return false;
                bits.push_back({"", icon});
                if (!after.empty()) bits.push_back({after, -1});
            } else {
                bits.push_back({text.substr(i, j - i), -1});
            }
        }
        i = j;
    }
    const float icon = std::round(size * 1.35f), pad = std::round(size * 0.12f);
    float w = 0.0f;
    for (const Bit& b : bits) w += b.icon >= 0 ? icon + pad : f->CalcTextSizeA(size, FLT_MAX, 0.0f, b.text.c_str()).x;
    float x = (x0 + x1 - w) * 0.5f;
    const float th = f->CalcTextSizeA(size, FLT_MAX, 0.0f, "A").y;
    for (const Bit& b : bits) {
        if (b.icon >= 0) {
            const float iy = y + (th - icon) * 0.5f;
            art.image(dl, static_cast<Piece>(b.icon), ImVec2(x, iy), ImVec2(x + icon, iy + icon), IM_COL32(255, 255, 255, 255));
            x += icon + pad;
        } else {
            dl->AddText(f, size, ImVec2(x, y), col, b.text.c_str());
            x += f->CalcTextSizeA(size, FLT_MAX, 0.0f, b.text.c_str()).x;
        }
    }
    return true;
}

// The panel's sizes before the menus' scale: the corner panel's, or the game's
// options dialog's - on its 1080-line stage 915 x 738, its letters 24 px, its
// rows 57 px apart, its ornaments at 0.6 of their pixels - at the menus'
// scale (900 lines at a scale of 1).
struct Metrics {
    float text, title, small, where;  // the rows', the title's, the help's and the page count's letters
    float width, height;              // height 0: as the longest page needs (the corner)
    float pad_x, pad_y;
    float row_gap;                    // a row's height over its frame's
    float art_k;                      // the ornaments' pixels to a screen pixel
    int most_rows;                    // the rows a page shows before it scrolls (the corner)
};
constexpr float kStage = 900.0f / 1080.0f;
constexpr Metrics kCorner{17.0f, 22.0f, 15.0f, 15.0f, 560.0f, 0.0f, 26.0f, 12.0f, 6.0f, 0.53f, 12};
constexpr Metrics kDialog{24.0f * kStage, 24.0f * kStage, 24.0f * kStage, 18.0f * kStage, 915.0f * kStage, 738.0f * kStage,
                          40.0f * kStage, 14.0f * kStage, 26.0f * kStage, 0.6f * kStage, 0};

// The confirm a BB_OPT_CONFIRM button asks for, in the game's yes/no dialog's
// look: the presenter's thread only.
struct Confirm {
    bool on = false, appeared = false;
    std::string owner, key;  // the plugin and the key its on_action gets on Yes
    std::string question, help;
    std::string open;        // BB_OPT_OPEN too: the page Yes opens
    int row = 0;
} g_confirm;

// A TEXT row being typed in the host's text box (host/window.h): the
// presenter's thread only.
struct Typing {
    bool on = false;
    std::string owner, page, key;  // the row: its page and its own key
    bool runtime = false;
} g_typing;

}  // namespace

void plugin_panel_settle() {
    if (!g_typing.on && !g_confirm.on) return;
    std::string owner;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_shown.load()) owner = g_panel.owner;
    }
    if (g_typing.on && g_typing.owner != owner) {
        host_text_entry_end();
        g_typing.on = false;
    }
    if (g_confirm.on && g_confirm.owner != owner) g_confirm.on = false;
}

void plugin_panel_draw(float scale, bool pad) {
    if (!g_shown.load(std::memory_order_relaxed)) return;
    static const bool preview_pad = [] {  // BBHOST_PANEL_PREVIEW_PAD=1: the controller's guide in a preview
        const char* e = std::getenv("BBHOST_PANEL_PREVIEW_PAD");
        return e && e[0] == '1';
    }();
    pad = pad || preview_pad;
    Panel p;
    plugin_menu::Runtime rt;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner.empty() || !g_panel.layout) return;
        p = g_panel;
        g_panel.cursor_to = -1;
        const auto it = g_runtime.find(p.owner);
        if (it != g_runtime.end()) rt = it->second;
    }
    const plugin_menu::Layout& layout = *p.layout;
    // A TEXT row's box: its answer, once the player gives one; nothing else
    // reads the keys meanwhile.
    if (g_typing.on) {
        std::string text;
        const int r = host_text_entry_poll(text);
        if (r != 0) {
            host_text_entry_end();
            g_typing.on = false;
            if (r == 1 && g_typing.owner == p.owner) {
                if (g_typing.runtime) {
                    {
                        std::lock_guard<std::mutex> lk(g_mu);
                        plugin_menu::set_value(g_runtime[p.owner], g_typing.page, g_typing.key, text);
                    }
                    plugins_menu_option(p.owner, g_typing.page + "/" + g_typing.key, text);
                } else {
                    plugins_set_option(p.owner, g_typing.key, text);
                }
                ask_sound(kMenuSoundDecide);
            } else {
                ask_sound(kMenuSoundCancel);
            }
        }
    }
    const bool modal = g_typing.on || g_confirm.on;  // the box or the yes/no has the input
    const int n = static_cast<int>(p.nav.cycle.size());
    // The pages: L1 and R1, Page Up and Page Down, Q and E - a menu's own,
    // not on a page opened on top of one.
    int turn = 0;
    if (!modal && !p.nav.on_subpage()) {
        if (ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false) || ImGui::IsKeyPressed(ImGuiKey_PageUp, false) ||
            ImGui::IsKeyPressed(ImGuiKey_Q, false))
            turn = -1;
        if (ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false) || ImGui::IsKeyPressed(ImGuiKey_PageDown, false) ||
            ImGui::IsKeyPressed(ImGuiKey_E, false))
            turn = 1;
    }
    bool refocus = g_appeared.exchange(false);
    if (g_opened.exchange(false)) ask_sound(kMenuSoundOpen);
    if (turn && n > 1) {
        p.nav.turn(turn);
        refocus = true;
        ask_sound(kMenuSoundMove);
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner == p.owner) {
            g_panel.nav.turn(turn);
            g_last_page[p.owner] = g_panel.nav.shown();
        }
    }
    const std::string page_key = p.nav.shown();
    const plugin_menu::Page* page = layout.page(page_key);
    if (!page) return;
    const std::vector<plugin_menu::Row>& rows = plugin_menu::rows_of(*page, rt);
    // Circle, Escape and Backspace go back from a page opened on top, and
    // close the menu on one of its own - unless a row holds them (a slider
    // dragged) or the box or the yes/no has them.
    const bool back_pressed = !modal && !ImGui::IsAnyItemActive() &&
                              (ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false) || ImGui::IsKeyPressed(ImGuiKey_Escape, false) ||
                               ImGui::IsKeyPressed(ImGuiKey_Backspace, false));
    bool close = back_pressed && !p.nav.on_subpage();
    if (back_pressed && p.nav.on_subpage()) {
        // From the next frame the page under it, the cursor on the row that opened it.
        ask_sound(kMenuSoundCancel);
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner == p.owner) {
            g_panel.cursor_to = g_panel.nav.back();
            g_appeared.store(true);
        }
    }

    const float s = scale;
    const Metrics& m = p.corner ? kCorner : kDialog;
    Art art = menu_art(s);
    art.k = m.art_k * s;
    using game_menu_art::kOrnament;
    using game_menu_art::kBand;
    const bool ornate = art.has(kOrnament) && art.has(kBand);         // the dialog's own top and foot
    const bool inner_art = art.has(game_menu_art::kRule) && art.has(game_menu_art::kFiligree);
    const bool glow_art = art.has(game_menu_art::kGlow);
    ImGui::PushFont(launcher_ui::font_game(), m.text);
    const float text_size = ImGui::GetFontSize();
    const float title_size = text_size * (m.title / m.text), small_size = text_size * (m.small / m.text);
    const float where_size = text_size * (m.where / m.text);
    // One size for every page, so that it never moves: the corner panel the
    // rows of its longest page (up to a dozen: more scroll), the Defaults row
    // and a help area the longest help fits in; the dialog the game's size.
    const float width = m.width * s, pad_x = m.pad_x * s, pad_y = m.pad_y * s;
    const float inner = width - 2.0f * pad_x;
    const ImVec2 frame_pad(8.0f * s, 3.0f * s);
    const float frame_h = text_size + 2.0f * frame_pad.y;
    const float row_h = frame_h + m.row_gap * s;
    // The help and the guide in the font at the small size: their lines a
    // line's height and the game's leading apart.
    ImFont* const font = ImGui::GetFont();
    const float leading = kLeading * s, advance = small_size + leading;
    const auto lines_of = [&](const std::string& t) { return t.empty() ? 0 : static_cast<int>(wrap_lines(font, small_size, t, inner).size()); };
    int most_rows = 0, help_lines = 0;
    for (const plugin_menu::Page& pg : layout.pages) {
        const std::vector<plugin_menu::Row>& r = plugin_menu::rows_of(pg, rt);
        most_rows = std::max(most_rows, static_cast<int>(r.size()));
        help_lines = std::max(help_lines, lines_of(pg.help));
        for (const plugin_menu::Row& row : r) help_lines = std::max(help_lines, lines_of(row.desc.help));
    }
    if (m.most_rows) most_rows = std::min(most_rows, m.most_rows);
    // Room for two to five lines, and under the last for its descenders.
    help_lines = std::clamp(help_lines, 2, 5);
    const float help_h = static_cast<float>(help_lines) * advance - leading + descender_overhang(font, small_size);
    // The guide: the plugin's, or the host's for the menu up.
    std::string guide_pad = p.guide, guide_keys = p.guide_keys;
    if (guide_pad.empty() && guide_keys.empty()) {
        const bool pages = n > 1 && !p.nav.on_subpage();
        const char* back = p.nav.on_subpage() ? "back" : "close";
        guide_pad = std::string(pages ? "L1 R1: page    " : "") + "d-pad: choose and change    Cross: OK    Circle: " + back;
        guide_keys = std::string(pages ? "Q E: page    " : "") + "arrows: choose and change    Enter: OK    Esc: " + back;
    }
    const int guide_lines = std::max(lines_of(guide_pad), lines_of(guide_keys));
    const float guide_h = (guide_lines ? static_cast<float>(guide_lines) * advance - leading : 0.0f) + (art.atlas ? small_size * 0.3f : 0.0f);
    const float rule_gap = 12.0f * s;
    // The game's top and foot are its ornament over its band; the rules about
    // the title and the help carry the filigree, which wants more room.
    const float outer_h = ornate ? art.h(kOrnament) + art.h(kBand) + 2.0f * s : rule_gap;
    const float inner_gap = inner_art ? 18.0f * s : rule_gap;
    // Under the help's rule the filigree's curls hang 17 of its 42 rows: the help starts below them.
    const float help_drop = inner_art ? std::max(0.0f, 17.0f * art.k * 0.75f + 4.0f * s - inner_gap * 0.5f) : 0.0f;
    const float title_h = title_size * 1.15f + (n > 1 ? where_size * 1.1f : 0.0f);
    const float defaults_h = p.defaults ? row_h + 6.0f * s : 0.0f;
    const float fixed = pad_y + outer_h + title_h + inner_gap + inner_gap + help_drop + help_h + 10.0f * s + guide_h + outer_h + pad_y;
    float rows_h, height;
    if (m.height > 0.0f) {
        // The dialog: its own height, the rows what is left of it.
        rows_h = std::max(m.height * s - fixed - defaults_h, row_h);
        most_rows = std::max(1, static_cast<int>(rows_h / row_h));
        height = fixed + rows_h + defaults_h;
    } else {
        rows_h = static_cast<float>(most_rows) * row_h;
        height = fixed + rows_h + defaults_h;
    }
    const bool scrolls = static_cast<int>(rows.size()) > most_rows;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    if (p.corner) {
        const float margin = 24.0f * s;
        ImGui::SetNextWindowPos(vp->WorkPos + vp->WorkSize - ImVec2(margin, margin), ImGuiCond_Always, ImVec2(1.0f, 1.0f));
    } else {
        // The game's dialogs dim the screen behind them.
        ImGui::GetBackgroundDrawList()->AddRectFilled(vp->Pos, vp->Pos + vp->Size, IM_COL32(0, 0, 0, 150));
        ImGui::SetNextWindowPos(vp->WorkPos + vp->WorkSize * 0.5f, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    }
    ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Always);
    if (refocus && !g_confirm.on) ImGui::SetNextWindowFocus();
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground;
    // The widgets in the game's colours: no frame but a faint one on the
    // sliders, gold for what moves, nothing of ImGui's own cursor - the
    // golden band is the cursor.
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, frame_pad);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, 3.0f * s);
    ImGui::PushStyleColor(ImGuiCol_Text, look::kText);
    ImGui::PushStyleColor(ImGuiCol_TextDisabled, look::kTextDim);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, look::kFrame);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, look::kFrameHot);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, look::kFrameHot);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, look::kGrab);
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, look::kGrabLit);
    ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(0, 0, 0, 0));
    // With the game's art, no plate under a value either: its glow is the
    // cursor, as in the game's dialog, and a fainter one the pointer's.
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, art.atlas ? IM_COL32(0, 0, 0, 0) : look::kFrame);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, art.atlas ? IM_COL32(0, 0, 0, 0) : look::kFrameHot);
    ImGui::PushStyleColor(ImGuiCol_NavCursor, IM_COL32(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(0, 0, 0, 0));
    constexpr int kColours = 12, kVars = 6;
    int click = 0;  // a page arrow clicked: the page turns from the next frame
    int focus = -1;  // the row with the cursor, or the Defaults row (rows.size())
    struct Pressed {
        const plugin_menu::Row* row = nullptr;
        int index = 0;
    } pressed;
    const auto apply = [&](const plugin_menu::Row& r, const std::string& v) {
        if (r.runtime) {
            {
                std::lock_guard<std::mutex> lk(g_mu);
                plugin_menu::set_value(g_runtime[p.owner], page_key, r.desc.key, v);
            }
            plugins_menu_option(p.owner, plugin_menu::event_key(page_key, r), v);
        } else {
            plugins_set_option(p.owner, r.desc.key, v);
        }
    };
    // Greyed out (BB_OPT_WHEN) unless the row it names holds one of its
    // values: a run-time row of this page, else one of the table's.
    const auto grey_of = [&](const plugin_menu::Row& r) {
        const std::size_t eq = r.desc.when.find('=');
        if (eq == std::string::npos) return false;
        const std::string key = r.desc.when.substr(0, eq);
        const std::vector<std::string> want = split(r.desc.when.substr(eq + 1), '|');
        for (const plugin_menu::Row& other : rows) {
            if (other.runtime && other.desc.key == key)
                return std::find(want.begin(), want.end(), plugin_menu::value_of(rt, page_key, other)) == want.end();
        }
        return greyed(*p.entry, r.desc);
    };
    if (ImGui::Begin("##plugin_panel", nullptr, flags)) {
        // A centred menu is the game's modal dialog: a click outside it does
        // not take the cursor away - from the next frame it is back where it
        // was.
        if (!p.corner && !g_confirm.on && !refocus && !ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
            std::lock_guard<std::mutex> lk(g_mu);
            if (g_panel.owner == p.owner) {
                g_panel.cursor_to = g_panel.focused_row;
                g_appeared.store(true);
            }
        }
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 o = ImGui::GetWindowPos();
        // The column rule down the middle where the game draws its; without
        // the art, the labels end a little before it.
        const float x0 = o.x + pad_x, x1 = o.x + width - pad_x, mid = o.x + width * (inner_art ? 0.5f : 0.46f);
        const ImU32 art_tint = IM_COL32(255, 255, 255, 235);
        // The plate: the dialog's slate, or a dark one darker at its top and foot.
        if (art.has(game_menu_art::kSlate)) {
            dl->AddRectFilled(o, o + ImVec2(width, height), IM_COL32(0, 0, 0, 90));  // over a bright world too
            art.image(dl, game_menu_art::kSlate, o, o + ImVec2(width, height), IM_COL32(255, 255, 255, 250));
        } else {
            dl->AddRectFilled(o, o + ImVec2(width, height), look::kPlate);
            dl->AddRectFilledMultiColor(o, o + ImVec2(width, height * 0.18f), IM_COL32(0, 0, 0, 70), IM_COL32(0, 0, 0, 70),
                                        look::kPlateEdge, look::kPlateEdge);
            dl->AddRectFilledMultiColor(o + ImVec2(0.0f, height * 0.82f), o + ImVec2(width, height), look::kPlateEdge,
                                        look::kPlateEdge, IM_COL32(0, 0, 0, 70), IM_COL32(0, 0, 0, 70));
        }
        float y = o.y + pad_y;
        // Its top: the ornament (a crest over its rule), the band under it.
        if (ornate) {
            const float ox0 = x0 - 14.0f * s, ox1 = x1 + 14.0f * s;
            art.span(dl, kOrnament, ox0, ox1, y, art_tint, false);
            art.span(dl, kBand, ox0, ox1, y + art.h(kOrnament), art_tint, false);
            y += outer_h;
        } else {
            rule(dl, x0 - 8.0f * s, x1 + 8.0f * s, y, s);
            y += rule_gap;
        }
        // The page's name in gold between its turns (the pointer's: the
        // d-pad's cursor goes over the rows only - L1 and R1 turn the pages),
        // and where it is among them; a page opened on top has neither.
        {
            ImFont* f = ImGui::GetFont();
            const ImVec2 ts = f->CalcTextSizeA(title_size, FLT_MAX, 0.0f, page->title.c_str());
            dl->AddText(f, title_size, ImVec2((x0 + x1 - ts.x) * 0.5f, y), look::kTitle, page->title.c_str());
            if (n > 1 && !p.nav.on_subpage()) {
                const float arrow = title_size * 0.9f;
                ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
                ImGui::SetCursorScreenPos(ImVec2(x0, y + (title_size * 1.15f - arrow) * 0.5f));
                if (ImGui::InvisibleButton("##prev", ImVec2(arrow, arrow)) && !modal) click = -1;
                const bool prev_hot = ImGui::IsItemHovered();
                ImGui::SetCursorScreenPos(ImVec2(x1 - arrow, y + (title_size * 1.15f - arrow) * 0.5f));
                if (ImGui::InvisibleButton("##next", ImVec2(arrow, arrow)) && !modal) click = 1;
                const bool next_hot = ImGui::IsItemHovered();
                ImGui::PopItemFlag();
                const float cy = y + title_size * 0.58f;
                if (art.has(game_menu_art::kArrow)) {
                    const float ah = std::round(title_size * 0.85f), aw = ah * art.atlas->piece[game_menu_art::kArrow].w /
                                                                          static_cast<float>(art.atlas->piece[game_menu_art::kArrow].h);
                    art.image(dl, game_menu_art::kArrow, ImVec2(x0 + 2.0f * s, cy - ah * 0.5f), ImVec2(x0 + 2.0f * s + aw, cy + ah * 0.5f),
                              prev_hot ? look::kGoldLit : look::kGold, true);
                    art.image(dl, game_menu_art::kArrow, ImVec2(x1 - 2.0f * s - aw, cy - ah * 0.5f), ImVec2(x1 - 2.0f * s, cy + ah * 0.5f),
                              next_hot ? look::kGoldLit : look::kGold);
                } else {
                    const float h = 6.0f * s, w = 6.0f * s;
                    dl->AddTriangleFilled(ImVec2(x0 + 2.0f * s, cy), ImVec2(x0 + 2.0f * s + w, cy - h), ImVec2(x0 + 2.0f * s + w, cy + h),
                                          prev_hot ? look::kGoldLit : look::kGold);
                    dl->AddTriangleFilled(ImVec2(x1 - 2.0f * s, cy), ImVec2(x1 - 2.0f * s - w, cy - h), ImVec2(x1 - 2.0f * s - w, cy + h),
                                          next_hot ? look::kGoldLit : look::kGold);
                }
                char where[32];
                std::snprintf(where, sizeof(where), "%d / %d", p.nav.at + 1, n);
                const ImVec2 ws = f->CalcTextSizeA(where_size, FLT_MAX, 0.0f, where);
                dl->AddText(f, where_size, ImVec2((x0 + x1 - ws.x) * 0.5f, y + title_size * 1.15f), look::kTextDim, where);
            }
        }
        y += title_h;
        // The rule under the title: the filigree over the column rule, its
        // finial pointing down it.
        const float kf = art.k * 0.75f;
        const auto inner_rule = [&](float ly, bool finial_down) {
            if (!inner_art) {
                rule(dl, x0, x1, ly, s);
                return;
            }
            art.rule(dl, x0, x1, ly, art_tint);
            const game_menu_art::Placed& fq = art.atlas->piece[game_menu_art::kFiligree];
            const float fw = fq.w * kf, fh = fq.h * kf;
            // The filigree's own rule runs at its pixel row 25 of 42 (29..30 of the texture's 4..46).
            const float line_at = (finial_down ? fq.h - 25.5f : 25.5f) * kf;
            art.image(dl, game_menu_art::kFiligree, ImVec2(mid - fw * 0.5f, ly - line_at), ImVec2(mid + fw * 0.5f, ly - line_at + fh),
                      art_tint, false, finial_down);
        };
        const float rule_top = y + inner_gap * 0.5f;
        inner_rule(rule_top, true);
        y += inner_gap;
        // Its rows, the one with the cursor lit; the glow and the lines go
        // under the row's widget (the draw list's first channel). A page with
        // more than it shows scrolls with the cursor, in a region of its own.
        std::string help = page->help;
        // `strength` 1 for the cursor's row; less for the one under the
        // pointer (the art's glow only). A slider's value is lit at its mark.
        // An ACTION row's glow on its text, across the row (`value_x` its middle, `half_w` the glow's half).
        const auto light = [&](ImDrawList* d, float top, bool label_side, float strength, float value_x, float half_w = -1.0f) {
            d->ChannelsSetCurrent(kGlowLayer);
            if (glow_art) {
                // The game's glow: brightest at the column rule, fading to the
                // left behind the label; on the value, brightest at its middle.
                const float cy = top + row_h * 0.5f, gh = row_h * 2.4f;
                const ImU32 label_tint = IM_COL32(255, 255, 255, static_cast<int>(215.0f * strength));
                const ImU32 value_tint = IM_COL32(255, 255, 255, static_cast<int>(170.0f * strength));
                if (label_side)
                    art.image(d, game_menu_art::kGlow, ImVec2(x0 - pad_x * 0.6f, cy - gh * 0.5f), ImVec2(mid, cy + gh * 0.5f), label_tint);
                const float hw = half_w > 0.0f ? half_w : (x1 - mid - 14.0f * s) * 0.45f, vh = row_h * 2.0f;
                const float vx = value_x >= 0.0f ? value_x : (mid + 14.0f * s + x1) * 0.5f;
                art.image(d, game_menu_art::kGlow, ImVec2(vx - hw, cy - vh * 0.5f), ImVec2(vx, cy + vh * 0.5f), value_tint);
                art.image(d, game_menu_art::kGlow, ImVec2(vx, cy - vh * 0.5f), ImVec2(vx + hw, cy + vh * 0.5f), value_tint, true);
            } else if (strength >= 1.0f) {
                const float cx = (x0 + x1) * 0.5f;
                d->AddRectFilledMultiColor(ImVec2(x0 - pad_x * 0.5f, top), ImVec2(cx, top + row_h), IM_COL32(0, 0, 0, 0), look::kGlow,
                                           look::kGlow, IM_COL32(0, 0, 0, 0));
                d->AddRectFilledMultiColor(ImVec2(cx, top), ImVec2(x1 + pad_x * 0.5f, top + row_h), look::kGlow, IM_COL32(0, 0, 0, 0),
                                           IM_COL32(0, 0, 0, 0), look::kGlow);
            }
            d->ChannelsSetCurrent(kWidgetLayer);
        };
        const float rows_top = y;
        float rows_scroll = 0.0f;  // how far the rows' region is scrolled
        // (in the draw list's two channels: the caller splits it)
        const auto draw_rows = [&](ImDrawList* d, float top0) {
            bool first = true;
            for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
                const plugin_menu::Row& r = rows[static_cast<std::size_t>(i)];
                const float top = top0 + static_cast<float>(i) * row_h;
                RowIn in;
                in.o = &r.desc;
                in.value = r.runtime ? plugin_menu::value_of(rt, page_key, r) : plugins_option_value(*p.entry, r.desc);
                in.grey = grey_of(r);
                in.drag_id = p.owner + "." + plugin_menu::event_key(page_key, r);
                const bool cursor = (refocus && !modal) && (p.cursor_to >= 0 ? i == p.cursor_to : first);
                ImGui::BeginDisabled(modal);
                const RowOut ro = row(in, ImVec2(x0, top), ImVec2(x1, top + row_h), mid, s, cursor, art);
                ImGui::EndDisabled();
                // An action's text is across the whole row: its glow is too.
                const bool action = r.desc.type == BB_OPT_ACTION;
                const float gx = action ? (x0 + x1) * 0.5f : ro.glow_x, gw = action ? (x1 - x0) * 0.32f : -1.0f;
                if (ro.focused) {
                    light(d, top, !action, 1.0f, gx, gw);
                    focus = i;
                } else if (ro.hot) {
                    light(d, top, !action, 0.4f, gx, gw);
                }
                if (ro.hot) help = r.desc.help;
                if (!art.has(game_menu_art::kLabelLine))
                    d->AddLine(ImVec2(x0, top + row_h - 0.5f), ImVec2(x1, top + row_h - 0.5f), look::kRowLine, 1.0f);
                if (ro.changed && !modal) apply(r, ro.value);
                if (ro.pressed && !modal) pressed = {&r, i};
                if (r.desc.type != BB_OPT_INFO) first = false;
            }
        };
        if (scrolls) {
            // The rows in a region of their own, which the cursor scrolls
            // (ImGui brings the row with it into view); its own draw list.
            if (refocus) ImGui::SetNextWindowScroll(ImVec2(0.0f, p.cursor_to >= 0 ? -1.0f : 0.0f));
            ImGui::SetCursorScreenPos(ImVec2(o.x, rows_top));
            ImGui::BeginChild("##rows", ImVec2(width, rows_h), ImGuiChildFlags_NavFlattened,
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);
            ImDrawList* cd = ImGui::GetWindowDrawList();
            cd->ChannelsSplit(3);
            cd->ChannelsSetCurrent(kWidgetLayer);
            draw_rows(cd, rows_top - ImGui::GetScrollY());
            cd->ChannelsMerge();
            ImGui::SetCursorScreenPos(ImVec2(o.x, rows_top - ImGui::GetScrollY()));
            ImGui::Dummy(ImVec2(width, static_cast<float>(rows.size()) * row_h));
            const float scroll = ImGui::GetScrollY(), most = ImGui::GetScrollMaxY();
            rows_scroll = scroll;
            ImGui::EndChild();
            // A thin mark at the right of where the page is, as the game's lists have.
            const float tx = x1 + pad_x * 0.45f, total = static_cast<float>(rows.size()) * row_h;
            const float thumb = std::max(rows_h * rows_h / total, 8.0f * s), at = most > 0.0f ? scroll / most : 0.0f;
            dl->AddLine(ImVec2(tx, rows_top), ImVec2(tx, rows_top + rows_h), look::kRowLine, 1.0f);
            dl->AddRectFilled(ImVec2(tx - 1.5f * s, rows_top + (rows_h - thumb) * at), ImVec2(tx + 1.5f * s, rows_top + (rows_h - thumb) * at + thumb),
                              look::kGold);
        }
        // Defaults: the page's rows back to theirs, below them all, at the
        // same place on every page.
        dl->ChannelsSplit(3);
        dl->ChannelsSetCurrent(kWidgetLayer);
        if (!scrolls) draw_rows(dl, rows_top);
        if (p.defaults) {
            const float top = rows_top + rows_h + 6.0f * s;
            ImGui::SetCursorScreenPos(ImVec2(mid + 14.0f * s, top + (row_h - frame_h) * 0.5f));
            ImGui::BeginDisabled(modal);
            const bool defaults_pressed = value_button("defaults", "Defaults", ImVec2(x1 - mid - 14.0f * s, frame_h), false);
            ImGui::EndDisabled();
            const bool focused = ImGui::IsItemFocused(), hovered = ImGui::IsItemHovered();
            if (focused) {
                light(dl, top, false, 1.0f, -1.0f);
                focus = static_cast<int>(rows.size());
            } else if (hovered) {
                light(dl, top, false, 0.4f, -1.0f);
            }
            if (focused || hovered) help = "Every row on this page back to its default.";
            if (art.has(game_menu_art::kValueEnd) && art.has(game_menu_art::kRule)) {
                const float ly = top + row_h - 4.0f * s, ew = art.w(game_menu_art::kValueEnd), eh = art.h(game_menu_art::kValueEnd);
                const float vx0 = mid + 16.0f * s, vx1 = x1 - 2.0f * s;
                dl->ChannelsSetCurrent(kLineLayer);
                art.image(dl, game_menu_art::kValueEnd, ImVec2(vx0, ly - 1.0f), ImVec2(vx0 + ew, ly - 1.0f + eh), art_tint, false, true);
                art.image(dl, game_menu_art::kValueEnd, ImVec2(vx1 - ew, ly - 1.0f), ImVec2(vx1, ly - 1.0f + eh), art_tint, true, true);
                art.rule(dl, vx0 + ew - 1.0f, vx1 - ew + 1.0f, ly, art_tint);
                dl->ChannelsSetCurrent(kWidgetLayer);
            }
            if (defaults_pressed && !modal) {
                for (const plugin_menu::Row& r : rows) {
                    if (r.desc.type != BB_OPT_ACTION && r.desc.type != BB_OPT_INFO) apply(r, r.desc.def);
                }
                ask_sound(kMenuSoundDecide);
            }
        }
        dl->ChannelsMerge();
        // The cursor moved to another row of the same page: the game's sound
        // for it (a page turned has made its own).
        {
            static std::string last_page;
            static int last_focus = -1;
            if (focus >= 0 && last_focus >= 0 && focus != last_focus && last_page == page_key && !turn) ask_sound(kMenuSoundMove);
            if (focus >= 0) last_focus = focus;
            last_page = page_key;
        }
        y = rows_top + rows_h + defaults_h;
        // The rule over the help (its finial up the column rule), and the
        // column rule itself between the two.
        const float rule_foot = y + inner_gap * 0.5f;
        if (inner_art) {
            // In pieces: not behind an action's text, which is across its row.
            const game_menu_art::Placed& q = art.atlas->piece[game_menu_art::kRule];
            const float t = std::max(art.h(game_menu_art::kRule), 3.0f), ya = rule_top + 6.0f * s, yb = rule_foot - 6.0f * s;
            std::vector<std::pair<float, float>> cuts;
            for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
                if (rows[static_cast<std::size_t>(i)].desc.type != BB_OPT_ACTION) continue;
                const float top = rows_top - rows_scroll + static_cast<float>(i) * row_h;
                const float c0 = std::max(top + 2.0f * s, rows_top), c1 = std::min(top + row_h - 2.0f * s, rows_top + rows_h);
                if (c1 > c0) cuts.emplace_back(c0, c1);
            }
            float from = ya;
            cuts.emplace_back(yb, yb);
            for (const auto& [c0, c1] : cuts) {
                const float to = std::min(c0, yb);
                if (to > from)
                    dl->AddImageQuad(art.tex, ImVec2(mid - t * 0.5f, from), ImVec2(mid + t * 0.5f, from), ImVec2(mid + t * 0.5f, to),
                                     ImVec2(mid - t * 0.5f, to), ImVec2(q.u0, q.v0), ImVec2(q.u0, q.v1), ImVec2(q.u1, q.v1),
                                     ImVec2(q.u1, q.v0), IM_COL32(255, 255, 255, 150));
                from = std::max(from, c1);
            }
        }
        inner_rule(rule_foot, false);
        // The help clear of the filigree's curls under that rule (17 of its 42 rows, times its scale).
        y = std::max(y + inner_gap, inner_art ? rule_foot + 17.0f * kf + 4.0f * s : 0.0f);
        // The help of the row with the cursor (or the page's), then the controls.
        {
            ImGui::PushFont(nullptr, m.small);
            ImFont* f = ImGui::GetFont();
            // Its area holds its last line's descenders; the clip lets an
            // accented capital of its first line rise as far.
            ImGui::PushClipRect(ImVec2(x0, y - (help_h - static_cast<float>(help_lines) * advance + leading)), ImVec2(x1, y + help_h), true);
            text_lines(dl, f, small_size, x0, x1, y, advance, look::kHelp, help, art.atlas != nullptr);
            ImGui::PopClipRect();
            y += help_h + 10.0f * s;
            const std::string& guide = pad ? guide_pad : guide_keys;
            if (!guide.empty()) {
                if (!(art.atlas && pad && guide_with_icons(dl, f, small_size, x0, x1, y, look::kGuide, guide, art)))
                    text_lines(dl, f, small_size, x0, x1, y, advance, look::kGuide, guide, art.atlas != nullptr);
            }
            ImGui::PopFont();
        }
        y += guide_h;
        // Its foot: the band over the ornament, both turned over.
        if (ornate) {
            const float ox0 = x0 - 14.0f * s, ox1 = x1 + 14.0f * s;
            art.span(dl, kBand, ox0, ox1, y, art_tint, true);
            art.span(dl, kOrnament, ox0, ox1, y + art.h(kBand), art_tint, true);
        } else {
            y += rule_gap * 0.5f;
            rule(dl, x0 - 8.0f * s, x1 + 8.0f * s, y, s);
        }
    }
    ImGui::End();
    // A button pressed: a TEXT row asks for the host's box, a CONFIRM one for
    // the yes/no first, an OPEN one opens its page on top after on_action.
    if (pressed.row) {
        const plugin_menu::Row& r = *pressed.row;
        if (r.desc.type == BB_OPT_TEXT) {
            const std::string v = r.runtime ? plugin_menu::value_of(rt, page_key, r) : plugins_option_value(*p.entry, r.desc);
            host_text_entry_begin(v.c_str(), 128, false, r.desc.label.c_str(), TextCharset::BasicLatin, true);
            g_typing = {true, p.owner, page_key, r.desc.key, r.runtime};
            ask_sound(kMenuSoundDecide);
        } else if (r.desc.flags & BB_OPT_CONFIRM) {
            g_confirm = Confirm{};
            g_confirm.on = g_confirm.appeared = true;
            g_confirm.owner = p.owner;
            g_confirm.key = plugin_menu::event_key(page_key, r);
            g_confirm.question = r.desc.label + "?";
            g_confirm.help = r.desc.help;
            g_confirm.open = (r.desc.flags & BB_OPT_OPEN) ? r.desc.def : std::string();
            g_confirm.row = pressed.index;
            ask_sound(kMenuSoundDecide);
        } else {
            plugins_run_action(p.owner, plugin_menu::event_key(page_key, r));
            ask_sound(kMenuSoundDecide);
            if ((r.desc.flags & BB_OPT_OPEN) && layout.page(r.desc.def)) {
                std::lock_guard<std::mutex> lk(g_mu);
                if (g_panel.owner == p.owner) {
                    g_panel.nav.open(r.desc.def, pressed.index);
                    g_appeared.store(true);
                }
            }
        }
    }
    // The yes/no: the game's small dialog over the menu, No lit at first.
    if (g_confirm.on && g_confirm.owner == p.owner) {
        const float cw = std::min(width, 640.0f * s);
        const auto q_lines = wrap_lines(font, title_size, g_confirm.question, cw - 2.0f * pad_x);
        const int h_lines = g_confirm.help.empty() ? 0 : static_cast<int>(wrap_lines(font, small_size, g_confirm.help, cw - 2.0f * pad_x).size());
        const float ch = pad_y * 2.0f + outer_h * 2.0f + static_cast<float>(q_lines.size()) * (title_size + leading) +
                         (h_lines ? 8.0f * s + static_cast<float>(h_lines) * advance : 0.0f) + 16.0f * s + row_h + 8.0f * s;
        ImGui::SetNextWindowPos(vp->WorkPos + vp->WorkSize * 0.5f, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(cw, ch), ImGuiCond_Always);
        if (g_confirm.appeared) ImGui::SetNextWindowFocus();
        int answer = 0;  // 1 yes, 2 no
        if (ImGui::Begin("##plugin_confirm", nullptr, flags)) {
            ImDrawList* d = ImGui::GetWindowDrawList();
            const ImVec2 o = ImGui::GetWindowPos();
            const float cx0 = o.x + pad_x, cx1 = o.x + cw - pad_x;
            if (art.has(game_menu_art::kSlate)) {
                d->AddRectFilled(o, o + ImVec2(cw, ch), IM_COL32(0, 0, 0, 200));
                art.image(d, game_menu_art::kSlate, o, o + ImVec2(cw, ch), IM_COL32(255, 255, 255, 250));
            } else {
                d->AddRectFilled(o, o + ImVec2(cw, ch), IM_COL32(7, 6, 5, 240));
            }
            float cy = o.y + pad_y;
            if (ornate) {
                art.span(d, kOrnament, cx0 - 14.0f * s, cx1 + 14.0f * s, cy, IM_COL32(255, 255, 255, 235), false);
                art.span(d, kBand, cx0 - 14.0f * s, cx1 + 14.0f * s, cy + art.h(kOrnament), IM_COL32(255, 255, 255, 235), false);
            } else {
                rule(d, cx0, cx1, cy, s);
            }
            cy += outer_h;
            text_lines(d, font, title_size, cx0, cx1, cy, title_size + leading, look::kTextLit, g_confirm.question, true);
            cy += static_cast<float>(q_lines.size()) * (title_size + leading);
            if (h_lines) {
                cy += 8.0f * s;
                text_lines(d, font, small_size, cx0, cx1, cy, advance, look::kHelp, g_confirm.help, true);
                cy += static_cast<float>(h_lines) * advance;
            }
            cy += 16.0f * s;
            const float bw = (cx1 - cx0) * 0.3f, gapx = (cx1 - cx0) * 0.1f, bx = (cx0 + cx1) * 0.5f;
            d->ChannelsSplit(3);
            d->ChannelsSetCurrent(kWidgetLayer);
            const char* const names[2] = {"Yes", "No"};
            for (int i = 0; i < 2; ++i) {
                const float left = i == 0 ? bx - gapx * 0.5f - bw : bx + gapx * 0.5f;
                ImGui::SetCursorScreenPos(ImVec2(left, cy + (row_h - frame_h) * 0.5f));
                if (value_button(names[i], names[i], ImVec2(bw, frame_h), false)) answer = i == 0 ? 1 : 2;
                if (i == 1) {
                    ImGui::SetItemDefaultFocus();  // No lit at first, as the game's yes/no has it
                    take_cursor(g_confirm.appeared);
                }
                if (ImGui::IsItemFocused() && glow_art) {
                    d->ChannelsSetCurrent(kGlowLayer);
                    const float gy = cy + row_h * 0.5f, gh = row_h * 2.0f, hw = bw * 0.5f;
                    art.image(d, game_menu_art::kGlow, ImVec2(left + bw * 0.5f - hw, gy - gh * 0.5f), ImVec2(left + bw * 0.5f, gy + gh * 0.5f),
                              IM_COL32(255, 255, 255, 170));
                    art.image(d, game_menu_art::kGlow, ImVec2(left + bw * 0.5f, gy - gh * 0.5f), ImVec2(left + bw * 0.5f + hw, gy + gh * 0.5f),
                              IM_COL32(255, 255, 255, 170), true);
                    d->ChannelsSetCurrent(kWidgetLayer);
                }
            }
            d->ChannelsMerge();
            cy += row_h + 8.0f * s;
            if (ornate) {
                art.span(d, kBand, cx0 - 14.0f * s, cx1 + 14.0f * s, cy, IM_COL32(255, 255, 255, 235), true);
                art.span(d, kOrnament, cx0 - 14.0f * s, cx1 + 14.0f * s, cy + art.h(kBand), IM_COL32(255, 255, 255, 235), true);
            } else {
                rule(d, cx0, cx1, cy + outer_h * 0.5f, s);
            }
            if (!g_confirm.appeared && (ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false) || ImGui::IsKeyPressed(ImGuiKey_Escape, false) ||
                                        ImGui::IsKeyPressed(ImGuiKey_Backspace, false)))
                answer = 2;
        }
        ImGui::End();
        g_confirm.appeared = false;
        if (answer == 1) {
            ask_sound(kMenuSoundDecide);
            plugins_run_action(g_confirm.owner, g_confirm.key);
            std::lock_guard<std::mutex> lk(g_mu);
            if (g_panel.owner == p.owner) {
                if (!g_confirm.open.empty() && layout.page(g_confirm.open))
                    g_panel.nav.open(g_confirm.open, g_confirm.row);
                else
                    g_panel.cursor_to = g_confirm.row;  // the cursor back on the button
                g_appeared.store(true);
            }
            g_confirm.on = false;
        } else if (answer == 2) {
            ask_sound(kMenuSoundCancel);
            g_confirm.on = false;
            std::lock_guard<std::mutex> lk(g_mu);
            if (g_panel.owner == p.owner) {
                g_panel.cursor_to = g_confirm.row;
                g_appeared.store(true);
            }
        }
    }
    ImGui::PopStyleColor(kColours);
    ImGui::PopStyleVar(kVars);
    ImGui::PopFont();
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner == p.owner && focus >= 0 && focus < static_cast<int>(rows.size())) g_panel.focused_row = focus;
    }
    if (click && n > 1) {
        ask_sound(kMenuSoundMove);
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner == p.owner) {
            g_panel.nav.turn(click);
            g_last_page[p.owner] = g_panel.nav.shown();
            g_appeared.store(true);
        }
    }
    if (close) {
        ask_sound(kMenuSoundCancel);
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner == p.owner) {
            g_closed[p.owner] = true;
            g_panel.owner.clear();
            g_exclusive.store(false);
            g_shown.store(false);
            host_log("plugin: %s's panel closed by the player", p.owner.c_str());
        }
    }
}

#else

void plugin_panel_settle() {}

void plugin_panel_draw(float, bool) {}

#endif
