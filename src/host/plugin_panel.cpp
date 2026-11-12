#define IMGUI_DEFINE_MATH_OPERATORS
#include "host/plugin_panel.h"

#include "bbhost_plugin.h"
#include "host/plugins.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

#if defined(BBHOST_HAVE_SDL3)
#include "imgui.h"
#include "imgui_internal.h"
#endif

namespace {

struct Panel {
    std::string owner;                         // the plugin whose panel is up; empty: none
    std::shared_ptr<const PluginEntry> entry;  // its options, as they were when it was shown
    std::vector<std::string> pages;  // its pages' heading keys, in order
    int page = 0;
    std::string guide, guide_keys;
    bool defaults = false;
};
std::mutex g_mu;
Panel g_panel;                                     // under g_mu
std::map<std::string, bool> g_closed;              // under g_mu: closed by the player since its plugin last asked
std::map<std::string, std::string> g_last_page;    // under g_mu: the page each plugin's panel showed last
std::atomic<bool> g_shown{false};
std::atomic<bool> g_appeared{false};               // shown since the last frame drew it: give it the focus

}  // namespace

bool plugin_panel_show(const std::string& name, const PluginPanelShow& how) {
    PluginEntry entry;
    bool found = false;
    for (PluginEntry& e : plugins_catalog()) {
        if (e.name == name && !e.visitor) {
            entry = std::move(e);
            found = true;
            break;
        }
    }
    if (!found) return false;
    std::vector<std::string> pages;
    for (const PluginOptionDesc& o : entry.options) {
        if (o.type == BB_OPT_HEADING && !o.page.empty() && o.page == o.key) pages.push_back(o.key);
    }
    if (pages.empty()) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    std::string want = how.page.empty() ? g_last_page[name] : how.page;
    int page = 0;
    if (!want.empty()) {
        const auto it = std::find(pages.begin(), pages.end(), want);
        if (it == pages.end()) {
            if (!how.page.empty()) return false;  // a page it has not got
        } else {
            page = static_cast<int>(it - pages.begin());
        }
    }
    const bool was = g_shown.load() && g_panel.owner == name;
    g_panel.owner = name;
    g_panel.entry = std::make_shared<const PluginEntry>(std::move(entry));
    g_panel.pages = std::move(pages);
    g_panel.page = page;
    g_panel.guide = how.guide;
    g_panel.guide_keys = how.guide_keys;
    g_panel.defaults = how.defaults;
    g_last_page[name] = g_panel.pages[static_cast<std::size_t>(page)];
    if (!was) {
        g_appeared.store(true);
        host_log("plugin: %s's panel up at its page %s", name.c_str(), g_last_page[name].c_str());
    }
    g_shown.store(true);
    return true;
}

void plugin_panel_hide(const std::string& name) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_panel.owner != name) return;
    g_panel.owner.clear();
    if (g_shown.exchange(false)) host_log("plugin: %s's panel down", name.c_str());
}

void plugin_panel_state(const std::string& name, PluginPanelState& out) {
    std::lock_guard<std::mutex> lk(g_mu);
    out.shown = g_shown.load() && g_panel.owner == name;
    auto it = g_closed.find(name);
    out.closed = it != g_closed.end() && it->second;
    if (it != g_closed.end()) it->second = false;
    out.page = g_last_page[name];
}

bool plugin_panel_shown() { return g_shown.load(std::memory_order_relaxed); }

bool plugin_panel_key(int sc) {
    // SDL scancodes (USB usages): the arrows, Enter (and the keypad's), Page
    // Up, Page Down, Q, E and Escape.
    switch (sc) {
    case 79: case 80: case 81: case 82:  // right, left, down, up
    case 40: case 88:                    // Return, keypad Enter
    case 75: case 78:                    // Page Up, Page Down
    case 20: case 8:                     // Q, E
    case 41:                             // Escape
        return true;
    default:
        return false;
    }
}

#if defined(BBHOST_HAVE_SDL3)

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream in(s);
    while (std::getline(in, cur, sep)) out.push_back(cur);
    return out;
}

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

// A step left or right on the row with the focus: the d-pad or the arrows, held to repeat.
int step_pressed() {
    int d = 0;
    if (ImGui::IsKeyPressed(ImGuiKey_GamepadDpadLeft, true) || ImGui::IsKeyPressed(ImGuiKey_LeftArrow, true)) d -= 1;
    if (ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight, true) || ImGui::IsKeyPressed(ImGuiKey_RightArrow, true)) d += 1;
    return d;
}

// The value a number row shows while the pointer drags it: set when it lets go.
std::map<std::string, double> g_drag;

// One row: its label at the left, its widget at the right. Returns its help
// when it has the focus.
// The nav cursor onto the item just drawn, as the d-pad would put it there:
// SetKeyboardFocusHere would open a slider's text field instead.
void take_cursor(bool now) {
    if (!now) return;
    ImGui::SetFocusID(ImGui::GetItemID(), ImGui::GetCurrentWindow());
    ImGui::SetNavCursorVisible(true);
}

void row(const PluginEntry& p, const PluginOptionDesc& o, float label_w, float widget_w, std::string* focused_help, bool cursor) {
    ImGui::PushID(o.key.c_str());
    const std::string value = plugins_option_value(p, o);
    const bool grey = greyed(p, o);
    ImGui::BeginDisabled(grey);
    ImGui::AlignTextToFramePadding();
    if (o.type != BB_OPT_ACTION) {
        ImGui::TextUnformatted(o.label.c_str());
        ImGui::SameLine(label_w);
    }
    ImGui::SetNextItemWidth(widget_w);
    bool hot = false;  // the row's widget has the cursor or the pointer: its help shows
    switch (o.type) {
    case BB_OPT_BOOL: {
        bool on = value == "true" || value == "1";
        bool flip = ImGui::Checkbox("##v", &on);
        take_cursor(cursor);
        hot = ImGui::IsItemFocused() || ImGui::IsItemHovered();
        if (!flip && ImGui::IsItemFocused() && !grey && step_pressed() != 0) {
            on = !on;
            flip = true;
        }
        if (flip) plugins_set_option(p.name, o.key, on ? "true" : "false");
        ImGui::SameLine();
        ImGui::TextDisabled("%s", on ? "On" : "Off");
        break;
    }
    case BB_OPT_INT:
    case BB_OPT_FLOAT: {
        const std::string id = p.name + "." + o.key;
        const auto it = g_drag.find(id);
        double v = it != g_drag.end() ? it->second : std::atof(value.c_str());
        bool changed = false;
        if (o.type == BB_OPT_INT) {
            int iv = static_cast<int>(std::lround(v));
            if (ImGui::SliderInt("##v", &iv, static_cast<int>(o.min), static_cast<int>(o.max))) g_drag[id] = iv;
            v = iv;
        } else {
            float fv = static_cast<float>(v);
            if (ImGui::SliderFloat("##v", &fv, static_cast<float>(o.min), static_cast<float>(o.max), "%.2f")) g_drag[id] = fv;
            v = fv;
        }
        take_cursor(cursor);
        hot = ImGui::IsItemFocused() || ImGui::IsItemHovered();
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            changed = true;
            g_drag.erase(id);
        }
        if (!ImGui::IsItemActive() && ImGui::IsItemFocused() && !grey) {
            if (const int d = step_pressed()) {
                const double step = o.type == BB_OPT_INT ? 1.0 : (o.max - o.min) / 100.0;
                v = std::clamp(v + d * step, o.min, o.max);
                changed = true;
            }
        }
        if (changed) plugins_set_option(p.name, o.key, number_text(o, v));
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
        const std::string shown = "<  " + (n ? choices[static_cast<std::size_t>(cur)] : std::string()) + "  >";
        if (ImGui::Button(shown.c_str(), ImVec2(widget_w, 0)) && n) to = (cur + 1) % n;
        take_cursor(cursor);
        hot = ImGui::IsItemFocused() || ImGui::IsItemHovered();
        if (ImGui::IsItemFocused() && !grey && n) {
            if (const int d = step_pressed()) to = (cur + d + n) % n;
        }
        if (to != cur) plugins_set_option(p.name, o.key, choices[static_cast<std::size_t>(to)]);
        break;
    }
    case BB_OPT_ACTION:
        if (ImGui::Button(o.label.c_str(), ImVec2(label_w + widget_w, 0))) plugins_run_action(p.name, o.key);
        take_cursor(cursor);
        hot = ImGui::IsItemFocused() || ImGui::IsItemHovered();
        break;
    default:
        ImGui::TextDisabled("%s", value.c_str());
        break;
    }
    if (hot) *focused_help = o.help;
    ImGui::EndDisabled();
    ImGui::PopID();
}

}  // namespace

void plugin_panel_draw(float scale, bool pad) {
    if (!g_shown.load(std::memory_order_relaxed)) return;
    Panel p;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner.empty()) return;
        p = g_panel;
    }
    const int n = static_cast<int>(p.pages.size());
    // The pages: L1 and R1, Page Up and Page Down, Q and E.
    int turn = 0;
    if (ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false) || ImGui::IsKeyPressed(ImGuiKey_PageUp, false) ||
        ImGui::IsKeyPressed(ImGuiKey_Q, false))
        turn = -1;
    if (ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false) || ImGui::IsKeyPressed(ImGuiKey_PageDown, false) ||
        ImGui::IsKeyPressed(ImGuiKey_E, false))
        turn = 1;
    bool refocus = g_appeared.exchange(false);
    if (turn && n > 1) {
        p.page = (p.page + turn + n) % n;
        refocus = true;
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner == p.owner) {
            g_panel.page = p.page;
            g_last_page[p.owner] = p.pages[static_cast<std::size_t>(p.page)];
        }
    }
    const std::string& page_key = p.pages[static_cast<std::size_t>(p.page)];
    std::string title;
    std::string page_help;
    for (const PluginOptionDesc& o : p.entry->options) {
        if (o.type == BB_OPT_HEADING && o.key == page_key) {
            title = o.label;
            page_help = o.help;
        }
    }
    // Circle and Escape close it, unless a row holds them (a slider dragged).
    bool close = !ImGui::IsAnyItemActive() &&
                 (ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false) || ImGui::IsKeyPressed(ImGuiKey_Escape, false));

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float margin = 24.0f * scale;
    const float width = 470.0f * scale;
    ImGui::SetNextWindowPos(vp->WorkPos + vp->WorkSize - ImVec2(margin, margin), ImGuiCond_Always, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowSize(ImVec2(width, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.86f);
    if (refocus) ImGui::SetNextWindowFocus();
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                   ImGuiWindowFlags_NoMove;
    int click = 0;  // a page arrow clicked: the page turns from the next frame
    if (ImGui::Begin("##plugin_panel", nullptr, flags)) {
        // The page's title between its turns, and where it is among them. The
        // turns are the pointer's: the d-pad's cursor goes over the rows only
        // (L1 and R1 turn the pages), so it starts on the first.
        const float x0 = ImGui::GetCursorPosX(), full = ImGui::GetContentRegionAvail().x;
        ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
        if (ImGui::ArrowButton("##prev", ImGuiDir_Left)) click = -1;
        ImGui::SameLine();
        const float title_w = ImGui::CalcTextSize(title.c_str()).x;
        ImGui::SetCursorPosX(x0 + std::max(0.0f, (full - title_w) * 0.5f));
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(title.c_str());
        ImGui::SameLine();
        ImGui::SetCursorPosX(x0 + full - ImGui::GetFrameHeight());
        if (ImGui::ArrowButton("##next", ImGuiDir_Right)) click = 1;
        ImGui::PopItemFlag();
        if (n > 1) {
            char where[32];
            std::snprintf(where, sizeof(where), "%d / %d", p.page + 1, n);
            ImGui::SetCursorPosX(x0 + std::max(0.0f, (full - ImGui::CalcTextSize(where).x) * 0.5f));
            ImGui::TextDisabled("%s", where);
        }
        ImGui::Separator();
        // Its rows.
        const float label_w = width * 0.42f;
        const float widget_w = width - label_w - ImGui::GetStyle().WindowPadding.x * 2.0f;
        std::string help = page_help;
        bool first = true;
        for (const PluginOptionDesc& o : p.entry->options) {
            if (o.page != page_key || o.type == BB_OPT_HEADING) continue;
            row(*p.entry, o, label_w, widget_w, &help, first && refocus);
            first = false;
        }
        // Defaults: the page's rows back to theirs.
        if (p.defaults) {
            ImGui::Spacing();
            if (ImGui::Button("Defaults", ImVec2(label_w + widget_w, 0))) {
                for (const PluginOptionDesc& o : p.entry->options) {
                    if (o.page == page_key && o.type != BB_OPT_HEADING && o.type != BB_OPT_ACTION) plugins_set_option(p.entry->name, o.key, o.def);
                }
            }
            if (ImGui::IsItemFocused() || ImGui::IsItemHovered()) help = "Every row on this page back to its default.";
        }
        // The help of the row with the focus, and the controls.
        ImGui::Separator();
        if (!help.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextWrapped("%s", help.c_str());
            ImGui::PopTextWrapPos();
        }
        const std::string& guide = pad ? p.guide : p.guide_keys;
        if (!guide.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextDisabled("%s", guide.c_str());
            ImGui::PopTextWrapPos();
        }
    }
    ImGui::End();
    if (click && n > 1) {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner == p.owner) {
            g_panel.page = (p.page + click + n) % n;
            g_last_page[p.owner] = g_panel.pages[static_cast<std::size_t>(g_panel.page)];
            g_appeared.store(true);
        }
    }
    if (close) {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_panel.owner == p.owner) {
            g_closed[p.owner] = true;
            g_panel.owner.clear();
            g_shown.store(false);
            host_log("plugin: %s's panel closed by the player", p.owner.c_str());
        }
    }
}

#else

void plugin_panel_draw(float, bool) {}

#endif
