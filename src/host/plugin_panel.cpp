#define IMGUI_DEFINE_MATH_OPERATORS
#include "host/plugin_panel.h"

#include "bbhost_plugin.h"
#include "host/game_menu_art.h"
#include "host/plugins.h"
#include "log.h"

#include <algorithm>
#include <cfloat>
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
#include "host/launcher_ui.h"
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
    how.guide = g ? g : "";
    how.guide_keys = gk ? gk : "";
    how.defaults = true;
    if (plugin_panel_show(name, how)) {
        done = true;
        host_log("plugin: %s's panel shown for a preview (BBHOST_PANEL_PREVIEW)", name.c_str());
    }
}

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

// The panel's letters: the serif nearest the game's menus' (host/launcher_ui.h).
constexpr float kTextSize = 17.0f, kTitleSize = 22.0f, kSmallSize = 15.0f;  // before the scale

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

struct RowOut {
    bool hot = false;      // the cursor or the pointer on it: its help shows
    bool focused = false;  // the cursor on it: lit
};

// One row in [a, b]: its label right-aligned before `mid`, its widget across
// the value column after it. With the game's art: the label's line fading out
// to the left, the value's line with its scrolled ends, a slider's track and
// its filled part (all under the widget, the draw list's first channel), and
// the choice's arrows the dialog's own.
RowOut row(const PluginEntry& p, const PluginOptionDesc& o, ImVec2 a, ImVec2 b, float mid, float s, bool cursor, const Art& art) {
    RowOut out;
    ImGui::PushID(o.key.c_str());
    const std::string value = plugins_option_value(p, o);
    const bool grey = greyed(p, o);
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
    bool underline = false;   // its value's line under it (the game's art)
    switch (o.type) {
    case BB_OPT_BOOL: {
        const bool on = value == "true" || value == "1";
        bool to = on;
        if (value_button(o.key.c_str(), on ? "On" : "Off", val_size, false)) to = !on;
        take_cursor(cursor);
        if (ImGui::IsItemFocused() && !grey && step_pressed() != 0) to = !on;
        if (to != on) {
            plugins_set_option(p.name, o.key, to ? "true" : "false");
        }
        arrows = underline = true;
        break;
    }
    case BB_OPT_INT:
    case BB_OPT_FLOAT: {
        const std::string id = p.name + "." + o.key;
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
        if (track) {
            ImGui::PopStyleColor(5);
            const float th = std::max(frame_h * 0.42f, 4.0f), fh = th * 0.5f;
            const float cy = val_a.y + frame_h * 0.5f;
            const float f = o.max > o.min ? static_cast<float>((v - o.min) / (o.max - o.min)) : 0.0f;
            dl->ChannelsSetCurrent(0);
            art.image(dl, game_menu_art::kSliderTrack, ImVec2(val_a.x, cy - th * 0.5f), ImVec2(val_a.x + bar_w, cy + th * 0.5f), art_tint);
            const float fx = val_a.x + 2.0f * s + (bar_w - 4.0f * s) * std::clamp(f, 0.0f, 1.0f);
            if (fx > val_a.x + 2.0f * s)
                art.image(dl, game_menu_art::kSliderFill, ImVec2(val_a.x + 2.0f * s, cy - fh * 0.5f), ImVec2(fx, cy + fh * 0.5f),
                          art_tint, false, false, 0.0f, std::max(1.0f, art.atlas->piece[game_menu_art::kSliderFill].w * std::clamp(f, 0.0f, 1.0f)));
            dl->ChannelsSetCurrent(1);
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
        take_cursor(cursor);
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
        if (changed) {
            plugins_set_option(p.name, o.key, number_text(o, v));
        }
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
        if (to != cur) {
            plugins_set_option(p.name, o.key, choices[static_cast<std::size_t>(to)]);
        }
        arrows = underline = true;
        break;
    }
    case BB_OPT_ACTION:
        ImGui::SetCursorScreenPos(ImVec2(a.x, val_a.y));
        if (value_button(o.key.c_str(), o.label, ImVec2(b.x - a.x, frame_h), false)) {
            plugins_run_action(p.name, o.key);
        }
        take_cursor(cursor);
        break;
    default:
        ImGui::TextDisabled("%s", value.c_str());
        break;
    }
    out.focused = ImGui::IsItemFocused();
    out.hot = out.focused || ImGui::IsItemHovered();
    // The value's line, its scrolled ends hanging under it (the art's left
    // end turned over, mirrored on the right), and the label's.
    if (underline && art.has(game_menu_art::kValueEnd) && art.has(game_menu_art::kRule)) {
        const float y = b.y - 4.0f * s;
        const float ew = art.w(game_menu_art::kValueEnd), eh = art.h(game_menu_art::kValueEnd);
        const float x0 = val_a.x + 2.0f * s, x1 = b.x - 2.0f * s;
        dl->ChannelsSetCurrent(0);
        art.image(dl, game_menu_art::kValueEnd, ImVec2(x0, y - 1.0f), ImVec2(x0 + ew, y - 1.0f + eh), art_tint, false, true);
        art.image(dl, game_menu_art::kValueEnd, ImVec2(x1 - ew, y - 1.0f), ImVec2(x1, y - 1.0f + eh), art_tint, true, true);
        art.rule(dl, x0 + ew - 1.0f, x1 - ew + 1.0f, y, art_tint);
        dl->ChannelsSetCurrent(1);
    }
    if (o.type != BB_OPT_ACTION && art.has(game_menu_art::kLabelLine)) {
        const float lh = std::max(art.h(game_menu_art::kLabelLine), 2.0f);
        dl->ChannelsSetCurrent(0);
        art.image(dl, game_menu_art::kLabelLine, ImVec2(a.x, b.y - 2.0f * s - lh * 0.5f), ImVec2(mid - 3.0f * s, b.y - 2.0f * s + lh * 0.5f),
                  art_tint);
        dl->ChannelsSetCurrent(1);
    }
    if (arrows && out.focused) {
        if (art.has(game_menu_art::kArrow)) {
            const float ah = std::round(ImGui::GetFontSize() * 1.05f), aw = ah * art.atlas->piece[game_menu_art::kArrow].w /
                                                                             static_cast<float>(art.atlas->piece[game_menu_art::kArrow].h);
            const float cy = val_a.y + frame_h * 0.5f, in = 4.0f * s;
            const ImU32 t = grey ? IM_COL32(255, 255, 255, 110) : IM_COL32(255, 255, 255, 255);
            art.image(dl, game_menu_art::kArrow, ImVec2(val_a.x + in, cy - ah * 0.5f), ImVec2(val_a.x + in + aw, cy + ah * 0.5f), t, true);
            art.image(dl, game_menu_art::kArrow, ImVec2(b.x - in - aw, cy - ah * 0.5f), ImVec2(b.x - in, cy + ah * 0.5f), t);
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

// The height `text` takes wrapped to `width` in the current font at `size`.
float wrapped_height(const std::string& text, float size, float width) {
    if (text.empty()) return 0.0f;
    return ImGui::GetFont()->CalcTextSizeA(size, FLT_MAX, width, text.c_str()).y;
}

// Text across [x0, x1] from `top`, wrapped at the width and each line centred,
// as the game's dialogs set their help: the height it took.
float centred_text(ImDrawList* dl, ImFont* f, float size, float x0, float x1, float top, ImU32 col, const std::string& text) {
    const char* at = text.c_str();
    const char* end = at + text.size();
    float y = top;
    const float line = f->CalcTextSizeA(size, FLT_MAX, 0.0f, "A").y;
    while (at < end) {
        const char* brk = f->CalcWordWrapPosition(size, at, end, x1 - x0);
        if (brk == at) brk = at + 1;
        const char* stop = brk;
        while (stop > at && (stop[-1] == ' ' || stop[-1] == '\n')) --stop;
        const ImVec2 ts = f->CalcTextSizeA(size, FLT_MAX, 0.0f, at, stop);
        dl->AddText(f, size, ImVec2((x0 + x1 - ts.x) * 0.5f, y), col, at, stop);
        y += line;
        at = brk;
        while (at < end && (*at == ' ' || *at == '\n')) ++at;
    }
    return y - top;
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

}  // namespace

void plugin_panel_draw(float scale, bool pad) {
    if (!g_shown.load(std::memory_order_relaxed)) return;
    static const bool preview_pad = [] {  // BBHOST_PANEL_PREVIEW_PAD=1: the controller's guide in a preview
        const char* e = std::getenv("BBHOST_PANEL_PREVIEW_PAD");
        return e && e[0] == '1';
    }();
    pad = pad || preview_pad;
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

    const float s = scale;
    const Art art = menu_art(s);
    using game_menu_art::kOrnament;
    using game_menu_art::kBand;
    const bool ornate = art.has(kOrnament) && art.has(kBand);         // the dialog's own top and foot
    const bool inner_art = art.has(game_menu_art::kRule) && art.has(game_menu_art::kFiligree);
    ImGui::PushFont(launcher_ui::font_game(), kTextSize);
    const float text_size = ImGui::GetFontSize();
    const float title_size = text_size * (kTitleSize / kTextSize), small_size = text_size * (kSmallSize / kTextSize);
    // One size for every page, so that it never moves: the rows of its
    // longest page, the Defaults row, and a help area the longest help fits
    // in, whichever row it is.
    const float width = 560.0f * s, pad_x = 26.0f * s, pad_y = 12.0f * s;
    const float inner = width - 2.0f * pad_x;
    const ImVec2 frame_pad(8.0f * s, 3.0f * s);
    const float frame_h = text_size + 2.0f * frame_pad.y;
    const float row_h = frame_h + 6.0f * s;
    int most_rows = 0;
    float help_h = 0.0f;
    {
        std::map<std::string, int> rows;
        for (const PluginOptionDesc& o : p.entry->options) {
            if (o.page.empty()) continue;
            if (o.type == BB_OPT_HEADING) {
                help_h = std::max(help_h, wrapped_height(o.help, small_size, inner));
                continue;
            }
            most_rows = std::max(most_rows, ++rows[o.page]);
            help_h = std::max(help_h, wrapped_height(o.help, small_size, inner));
        }
    }
    const float line_h = small_size * 1.25f;
    help_h = std::clamp(help_h, 2.0f * line_h, 4.0f * line_h);
    const float guide_h = std::max(wrapped_height(p.guide, small_size, inner), wrapped_height(p.guide_keys, small_size, inner)) +
                          (art.atlas ? small_size * 0.3f : 0.0f);
    const float rule_gap = 12.0f * s;
    // The game's top and foot are its ornament over its band; the rules about
    // the title and the help carry the filigree, which wants more room.
    const float outer_h = ornate ? art.h(kOrnament) + art.h(kBand) + 2.0f * s : rule_gap;
    const float inner_gap = inner_art ? 18.0f * s : rule_gap;
    // Under the help's rule the filigree's curls hang 17 of its 42 rows: the help starts below them.
    const float help_drop = inner_art ? std::max(0.0f, 17.0f * art.k * 0.75f + 4.0f * s - inner_gap * 0.5f) : 0.0f;
    const float title_h = title_size * 1.15f + (n > 1 ? small_size * 1.1f : 0.0f);
    const float rows_h = static_cast<float>(most_rows) * row_h + (p.defaults ? row_h + 6.0f * s : 0.0f);
    const float height = pad_y + outer_h + title_h + inner_gap + rows_h + inner_gap + help_drop + help_h + 10.0f * s + guide_h + outer_h + pad_y;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float margin = 24.0f * s;
    ImGui::SetNextWindowPos(vp->WorkPos + vp->WorkSize - ImVec2(margin, margin), ImGuiCond_Always, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Always);
    if (refocus) ImGui::SetNextWindowFocus();
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
    constexpr int kColours = 11, kVars = 6;
    int click = 0;  // a page arrow clicked: the page turns from the next frame
    if (ImGui::Begin("##plugin_panel", nullptr, flags)) {
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
        // and where it is among them.
        {
            ImFont* f = ImGui::GetFont();
            const ImVec2 ts = f->CalcTextSizeA(title_size, FLT_MAX, 0.0f, title.c_str());
            dl->AddText(f, title_size, ImVec2((x0 + x1 - ts.x) * 0.5f, y), look::kTitle, title.c_str());
            const float arrow = title_size * 0.9f;
            ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
            ImGui::SetCursorScreenPos(ImVec2(x0, y + (title_size * 1.15f - arrow) * 0.5f));
            if (ImGui::InvisibleButton("##prev", ImVec2(arrow, arrow))) click = -1;
            const bool prev_hot = ImGui::IsItemHovered();
            ImGui::SetCursorScreenPos(ImVec2(x1 - arrow, y + (title_size * 1.15f - arrow) * 0.5f));
            if (ImGui::InvisibleButton("##next", ImVec2(arrow, arrow))) click = 1;
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
            if (n > 1) {
                char where[32];
                std::snprintf(where, sizeof(where), "%d / %d", p.page + 1, n);
                const ImVec2 ws = f->CalcTextSizeA(small_size, FLT_MAX, 0.0f, where);
                dl->AddText(f, small_size, ImVec2((x0 + x1 - ws.x) * 0.5f, y + title_size * 1.15f), look::kTextDim, where);
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
        // under the row's widget (the draw list's first channel).
        std::string help = page_help;
        dl->ChannelsSplit(2);
        dl->ChannelsSetCurrent(1);
        const bool glow_art = art.has(game_menu_art::kGlow);
        // `strength` 1 for the cursor's row; less for the one under the
        // pointer (the art's glow only).
        const auto light = [&](float top, bool label_side, float strength) {
            dl->ChannelsSetCurrent(0);
            if (glow_art) {
                // The game's glow: brightest at the column rule, fading to the
                // left behind the label; on the value, brightest at its middle.
                const float cy = top + row_h * 0.5f, gh = row_h * 2.4f;
                const ImU32 label_tint = IM_COL32(255, 255, 255, static_cast<int>(215.0f * strength));
                const ImU32 value_tint = IM_COL32(255, 255, 255, static_cast<int>(170.0f * strength));
                if (label_side)
                    art.image(dl, game_menu_art::kGlow, ImVec2(x0 - pad_x * 0.6f, cy - gh * 0.5f), ImVec2(mid, cy + gh * 0.5f), label_tint);
                const float vx = (mid + 14.0f * s + x1) * 0.5f, hw = (x1 - mid - 14.0f * s) * 0.45f, vh = row_h * 2.0f;
                art.image(dl, game_menu_art::kGlow, ImVec2(vx - hw, cy - vh * 0.5f), ImVec2(vx, cy + vh * 0.5f), value_tint);
                art.image(dl, game_menu_art::kGlow, ImVec2(vx, cy - vh * 0.5f), ImVec2(vx + hw, cy + vh * 0.5f), value_tint, true);
            } else if (strength >= 1.0f) {
                const float cx = (x0 + x1) * 0.5f;
                dl->AddRectFilledMultiColor(ImVec2(x0 - pad_x * 0.5f, top), ImVec2(cx, top + row_h), IM_COL32(0, 0, 0, 0), look::kGlow,
                                            look::kGlow, IM_COL32(0, 0, 0, 0));
                dl->AddRectFilledMultiColor(ImVec2(cx, top), ImVec2(x1 + pad_x * 0.5f, top + row_h), look::kGlow, IM_COL32(0, 0, 0, 0),
                                            IM_COL32(0, 0, 0, 0), look::kGlow);
            }
            dl->ChannelsSetCurrent(1);
        };
        bool first = true;
        int i = 0;
        for (const PluginOptionDesc& opt : p.entry->options) {
            if (opt.page != page_key || opt.type == BB_OPT_HEADING) continue;
            const float top = y + static_cast<float>(i) * row_h;
            const RowOut r = row(*p.entry, opt, ImVec2(x0, top), ImVec2(x1, top + row_h), mid, s, first && refocus, art);
            if (r.focused) {
                light(top, opt.type != BB_OPT_ACTION, 1.0f);
            } else if (r.hot) {
                light(top, opt.type != BB_OPT_ACTION, 0.4f);
            }
            if (r.hot) help = opt.help;
            if (!art.has(game_menu_art::kLabelLine))
                dl->AddLine(ImVec2(x0, top + row_h - 0.5f), ImVec2(x1, top + row_h - 0.5f), look::kRowLine, 1.0f);
            first = false;
            ++i;
        }
        // Defaults: the page's rows back to theirs, below them all, at the
        // same place on every page.
        if (p.defaults) {
            const float top = y + static_cast<float>(most_rows) * row_h + 6.0f * s;
            ImGui::SetCursorScreenPos(ImVec2(mid + 14.0f * s, top + (row_h - frame_h) * 0.5f));
            const bool pressed = value_button("defaults", "Defaults", ImVec2(x1 - mid - 14.0f * s, frame_h), false);
            const bool focused = ImGui::IsItemFocused(), hovered = ImGui::IsItemHovered();
            if (focused) {
                light(top, false, 1.0f);
            } else if (hovered) {
                light(top, false, 0.4f);
            }
            if (focused || hovered) help = "Every row on this page back to its default.";
            if (art.has(game_menu_art::kValueEnd) && art.has(game_menu_art::kRule)) {
                const float ly = top + row_h - 4.0f * s, ew = art.w(game_menu_art::kValueEnd), eh = art.h(game_menu_art::kValueEnd);
                const float vx0 = mid + 16.0f * s, vx1 = x1 - 2.0f * s;
                dl->ChannelsSetCurrent(0);
                art.image(dl, game_menu_art::kValueEnd, ImVec2(vx0, ly - 1.0f), ImVec2(vx0 + ew, ly - 1.0f + eh), art_tint, false, true);
                art.image(dl, game_menu_art::kValueEnd, ImVec2(vx1 - ew, ly - 1.0f), ImVec2(vx1, ly - 1.0f + eh), art_tint, true, true);
                art.rule(dl, vx0 + ew - 1.0f, vx1 - ew + 1.0f, ly, art_tint);
                dl->ChannelsSetCurrent(1);
            }
            if (pressed) {
                for (const PluginOptionDesc& opt : p.entry->options) {
                    if (opt.page == page_key && opt.type != BB_OPT_HEADING && opt.type != BB_OPT_ACTION)
                        plugins_set_option(p.entry->name, opt.key, opt.def);
                }
            }
        }
        dl->ChannelsMerge();
        y += rows_h;
        // The rule over the help (its finial up the column rule), and the
        // column rule itself between the two.
        const float rule_foot = y + inner_gap * 0.5f;
        if (inner_art) {
            const game_menu_art::Placed& q = art.atlas->piece[game_menu_art::kRule];
            const float t = std::max(art.h(game_menu_art::kRule), 3.0f), ya = rule_top + 6.0f * s, yb = rule_foot - 6.0f * s;
            dl->AddImageQuad(art.tex, ImVec2(mid - t * 0.5f, ya), ImVec2(mid + t * 0.5f, ya), ImVec2(mid + t * 0.5f, yb),
                             ImVec2(mid - t * 0.5f, yb), ImVec2(q.u0, q.v0), ImVec2(q.u0, q.v1), ImVec2(q.u1, q.v1), ImVec2(q.u1, q.v0),
                             IM_COL32(255, 255, 255, 150));
        }
        inner_rule(rule_foot, false);
        // The help clear of the filigree's curls under that rule (17 of its 42 rows, times its scale).
        y = std::max(y + inner_gap, inner_art ? rule_foot + 17.0f * kf + 4.0f * s : 0.0f);
        // The help of the row with the cursor (or the page's), then the controls.
        {
            ImGui::PushFont(nullptr, kSmallSize);
            ImFont* f = ImGui::GetFont();
            ImGui::PushClipRect(ImVec2(x0, y), ImVec2(x1, y + help_h), true);
            if (art.atlas)
                centred_text(dl, f, small_size, x0, x1, y, look::kHelp, help);
            else
                dl->AddText(f, small_size, ImVec2(x0, y), look::kHelp, help.c_str(), nullptr, inner);
            ImGui::PopClipRect();
            y += help_h + 10.0f * s;
            const std::string& guide = pad ? p.guide : p.guide_keys;
            if (!guide.empty()) {
                if (!(art.atlas && pad && guide_with_icons(dl, f, small_size, x0, x1, y, look::kGuide, guide, art))) {
                    if (art.atlas)
                        centred_text(dl, f, small_size, x0, x1, y, look::kGuide, guide);
                    else
                        dl->AddText(f, small_size, ImVec2(x0, y), look::kGuide, guide.c_str(), nullptr, inner);
                }
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
    ImGui::PopStyleColor(kColours);
    ImGui::PopStyleVar(kVars);
    ImGui::PopFont();
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
