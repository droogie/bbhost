#define IMGUI_DEFINE_MATH_OPERATORS
#include "host/ingame_menu.h"

#if defined(BBHOST_HAVE_SDL3)
#include "bbhost_version.h"
#include "core/config.h"
#include "engine/patch_manifest.h"
#include "hle/fs.h"
#include "host/gpu.h"
#include "host/launcher_ui.h"
#include "host/options.h"
#include "host/plugin_ui.h"
#include "host/settings.h"
#include "log.h"

#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace {

namespace ui = launcher_ui;

std::atomic<bool> g_open{false};   // F9's menu; F10's is host_options_open()
std::atomic<bool> g_ready{false};  // the Vulkan backend started: F9 and F10 open something

// One input event, translated on the pump thread and applied on the
// presenter's (ImGui's IO is not thread-safe).
struct Input {
    enum Kind { Move, Button, Wheel, Key, Analog, Text, Focus } kind = Move;
    float x = 0, y = 0;
    int button = 0;
    bool down = false;
    ImGuiKey key = ImGuiKey_None;
    std::string text;
};
std::mutex g_mu;
std::vector<Input> g_queue;  // under g_mu

ImGuiContext* g_ctx = nullptr;
bool g_vk_ready = false;
VkFormat g_format = VK_FORMAT_UNDEFINED;
std::uint32_t g_images = 0;
PluginUi g_ui;
std::chrono::steady_clock::time_point g_last;

ImGuiKey key_of(SDL_Keycode k) {
    switch (k) {
    case SDLK_TAB: return ImGuiKey_Tab;
    case SDLK_LEFT: return ImGuiKey_LeftArrow;
    case SDLK_RIGHT: return ImGuiKey_RightArrow;
    case SDLK_UP: return ImGuiKey_UpArrow;
    case SDLK_DOWN: return ImGuiKey_DownArrow;
    case SDLK_PAGEUP: return ImGuiKey_PageUp;
    case SDLK_PAGEDOWN: return ImGuiKey_PageDown;
    case SDLK_HOME: return ImGuiKey_Home;
    case SDLK_END: return ImGuiKey_End;
    case SDLK_DELETE: return ImGuiKey_Delete;
    case SDLK_BACKSPACE: return ImGuiKey_Backspace;
    case SDLK_RETURN: return ImGuiKey_Enter;
    case SDLK_KP_ENTER: return ImGuiKey_KeypadEnter;
    case SDLK_SPACE: return ImGuiKey_Space;
    case SDLK_ESCAPE: return ImGuiKey_Escape;
    case SDLK_LCTRL: return ImGuiKey_LeftCtrl;
    case SDLK_RCTRL: return ImGuiKey_RightCtrl;
    case SDLK_LSHIFT: return ImGuiKey_LeftShift;
    case SDLK_RSHIFT: return ImGuiKey_RightShift;
    case SDLK_A: return ImGuiKey_A;
    case SDLK_C: return ImGuiKey_C;
    case SDLK_V: return ImGuiKey_V;
    case SDLK_X: return ImGuiKey_X;
    default: return ImGuiKey_None;
    }
}

// A controller's buttons as ImGui's gamepad keys: the d-pad and A move and
// press, B backs out (and closes), L1 and R1 turn the pages, Start closes.
ImGuiKey pad_key_of(Uint8 b) {
    switch (b) {
    case SDL_GAMEPAD_BUTTON_SOUTH: return ImGuiKey_GamepadFaceDown;
    case SDL_GAMEPAD_BUTTON_EAST: return ImGuiKey_GamepadFaceRight;
    case SDL_GAMEPAD_BUTTON_WEST: return ImGuiKey_GamepadFaceLeft;
    case SDL_GAMEPAD_BUTTON_NORTH: return ImGuiKey_GamepadFaceUp;
    case SDL_GAMEPAD_BUTTON_DPAD_UP: return ImGuiKey_GamepadDpadUp;
    case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return ImGuiKey_GamepadDpadDown;
    case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return ImGuiKey_GamepadDpadLeft;
    case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return ImGuiKey_GamepadDpadRight;
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return ImGuiKey_GamepadL1;
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return ImGuiKey_GamepadR1;
    case SDL_GAMEPAD_BUTTON_START: return ImGuiKey_GamepadStart;
    case SDL_GAMEPAD_BUTTON_BACK: return ImGuiKey_GamepadBack;
    default: return ImGuiKey_None;
    }
}

void push(Input in) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_queue.size() < 512) g_queue.push_back(std::move(in));
}

void set_open(bool open) {
    if (g_open.exchange(open) == open) return;
    host_log("patch and plugin menu: %s", open ? "open" : "closed");
}

// The time either menu opened (it fades in), and which was up last frame.
double g_shown_at = -100.0;
bool g_was_up = false;

// BBHOST_MENU_TEST's requests for the presenter (ingame_menu_pump): a page
// for the open menu (-1 none) and patch boxes to click, by name.
std::atomic<int> g_test_page{-1};
std::mutex g_test_mu;
std::vector<std::pair<std::string, bool>> g_test_clicks;  // under g_test_mu

// ---- The frame both menus share ---------------------------------------------

struct Page {
    const char* name;
    ui::Icon icon;
    const char* blurb;
};

// The menu over the game: a glass panel, first in the middle of the screen,
// moved by its top bar and sized from its edges and corners (kept while the
// game runs); the pages down its left under the name, the page's heading at
// the top of the right and the page under it - scrolling when `scrolls`, else
// left to fit its own parts in the height it is given. `page` moves with a
// click, or with the pad's L1 and R1. True when its Close was pressed.
template <class Body>
bool shell(const char* id, const char* caption, const Page* pages, int n, int* page, double* page_at, bool scrolls, Body body) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float s = ui::scale();
    const ImVec2 ds = vp->Size;
    ImVec2 size(std::min(ds.x - 32 * s, std::max(ds.x * 0.74f, 980 * s)), std::min(ds.y - 32 * s, std::max(ds.y * 0.82f, 600 * s)));
    size.x = std::min(size.x, std::round(size.y * 1.85f));
    ImVec2 pos = vp->Pos + ImVec2(std::round((ds.x - size.x) * 0.5f), std::round((ds.y - size.y) * 0.5f));
    auto go = [&](int p) {
        if (*page == p) return;
        *page = p;
        *page_at = ImGui::GetTime();
    };
    if (!ImGui::IsAnyItemActive() && n > 1) {
        if (ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false)) go((*page + n - 1) % n);
        if (ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false)) go((*page + 1) % n);
    }
    ImGui::SetNextWindowPos(pos, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(size, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(std::min(ds.x, 760 * s), std::min(ds.y, 440 * s)), ds);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    // The corner's grip as the glass's own edge would show it.
    ImGui::PushStyleColor(ImGuiCol_ResizeGrip, ui::alpha(ui::kViolet, 40));
    ImGui::PushStyleColor(ImGuiCol_ResizeGripHovered, ui::alpha(ui::kViolet, 140));
    ImGui::PushStyleColor(ImGuiCol_ResizeGripActive, ui::alpha(ui::kViolet, 200));
    ImGui::Begin(id, nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
    pos = ImGui::GetWindowPos();
    size = ImGui::GetWindowSize();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float r = std::round(18 * s);
    // A soft shadow under it, then the glass.
    dl->PushClipRectFullScreen();
    for (int i = 6; i >= 1; --i) {
        const float g = static_cast<float>(i) * 5 * s;
        dl->AddRectFilled(pos - ImVec2(g, g) + ImVec2(0, 8 * s), pos + size + ImVec2(g, g) + ImVec2(0, 8 * s),
                          ImGui::GetColorU32(IM_COL32(0, 0, 4, 12)), r + g);
    }
    dl->PopClipRect();
    ui::glass(dl, pos, pos + size, r, IM_COL32(13, 11, 32, 232), ui::alpha(ui::kViolet, 100));

    // The top bar: the handle the window is dragged by.
    const float bar_h = std::round(30 * s);
    {
        ImGui::SetCursorScreenPos(pos);
        ImGui::InvisibleButton("##bar", ImVec2(size.x, bar_h));
        const bool hot = ImGui::IsItemHovered() || ImGui::IsItemActive();
        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
            const ImVec2 to = pos + ImGui::GetIO().MouseDelta;
            ImGui::SetWindowPos(ImVec2(std::clamp(to.x, vp->Pos.x - size.x + 80 * s, vp->Pos.x + ds.x - 80 * s),
                                       std::clamp(to.y, vp->Pos.y, vp->Pos.y + ds.y - bar_h)));
        }
        if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        dl->AddRectFilled(pos, pos + ImVec2(size.x, bar_h), ImGui::GetColorU32(IM_COL32(5, 4, 16, hot ? 200 : 160)), r,
                          ImDrawFlags_RoundCornersTop);
        dl->AddLine(pos + ImVec2(r, bar_h - 0.5f), pos + ImVec2(size.x - r, bar_h - 0.5f), ImGui::GetColorU32(ui::alpha(ui::kViolet, 64)));
        // A grip in the middle: two short rules.
        const float gw = std::round(36 * s), cx = pos.x + std::round(size.x * 0.5f), cy = pos.y + std::round(bar_h * 0.5f);
        const ImU32 gc = ImGui::GetColorU32(ui::alpha(ui::kViolet, hot ? 220 : 120));
        dl->AddLine(ImVec2(cx - gw * 0.5f, cy - 2.5f * s), ImVec2(cx + gw * 0.5f, cy - 2.5f * s), gc, 1.5f * s);
        dl->AddLine(ImVec2(cx - gw * 0.5f, cy + 2.5f * s), ImVec2(cx + gw * 0.5f, cy + 2.5f * s), gc, 1.5f * s);
    }
    pos.y += bar_h;
    size.y -= bar_h;

    // The sidebar: the name, the pages, the keys and Close.
    // At most a third of the panel: a large Menu text size leaves the page room.
    const float side_w = std::round(std::min(236 * s, size.x * 0.34f));
    dl->AddRectFilled(pos, pos + ImVec2(side_w, size.y), ImGui::GetColorU32(IM_COL32(7, 6, 20, 150)), r, ImDrawFlags_RoundCornersBottomLeft);
    dl->AddLine(pos + ImVec2(side_w - 0.5f, r), pos + ImVec2(side_w - 0.5f, size.y - r), ImGui::GetColorU32(ui::alpha(ui::kViolet, 64)));
    bool close = false;
    // Close at the foot, always in view, with the keys over it where the
    // name and the pages (last frame's height) leave room; those scroll when
    // they are taller than the rest (a large Menu text size).
    const float hint_h = std::round(ImGui::GetFontSize() * 0.78f + 8 * s);
    const float side_h = size.y - 34 * s, hints = 3 * (hint_h + 6 * s) + 6 * s + ImGui::GetStyle().ItemSpacing.y;
    float foot = ImGui::GetFrameHeight() + 4 * s;
    static float nav_h = 0.0f;
    const bool keys = side_h - foot - hints - 8 * s >= nav_h;
    if (keys) foot += hints;
    ImGui::SetCursorScreenPos(pos + ImVec2(16 * s, 18 * s));
    ImGui::BeginChild("##side", ImVec2(side_w - 32 * s, std::max(side_h - foot - 8 * s, 40 * s)), 0, ImGuiWindowFlags_NoBackground);
    const float w = ImGui::GetContentRegionAvail().x;
    {
        ui::logo("BBHOST", BBHOST_VERSION, w, true);
        const float cfs = 11.5f * s, ctr = 1.7f * s;
        ImGui::SetCursorPosX(std::max(0.0f, (w - ui::spaced_width(ui::font_strong(), cfs, caption, ctr)) * 0.5f));
        ui::caption(caption);
        ImGui::Dummy(ImVec2(0, 8 * s));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 3 * s));
        for (int i = 0; i < n; ++i) {
            if (ui::nav_item(pages[i].name, pages[i].icon, *page == i)) go(i);
        }
        ImGui::PopStyleVar();
        nav_h = ImGui::GetCursorPosY();
    }
    ImGui::EndChild();
    ImGui::SetCursorScreenPos(pos + ImVec2(16 * s, 18 * s + side_h - foot));
    ImGui::BeginChild("##foot", ImVec2(w, foot), 0, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (keys) {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 6 * s));
        ui::key_hint("F9", "Patches & plugins");
        ui::key_hint("F10", "Settings");
        ui::key_hint("Esc", "Close");
        ImGui::PopStyleVar();
        ImGui::Dummy(ImVec2(0, 4 * s));
    }
    close = ui::ghost_button("Close", ImVec2(w, 0));
    ImGui::EndChild();

    // The page: its heading, then itself, fading in.
    ImGui::SetCursorScreenPos(pos + ImVec2(side_w, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(30 * s, 24 * s));
    ImGui::BeginChild("##page", ImVec2(size.x - side_w, size.y), ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    auto heading = [&] {
        ImGui::PushFont(ui::font_strong(), 27.0f);
        ImGui::TextUnformatted(pages[*page].name);
        ImGui::PopFont();
        ImGui::PushStyleColor(ImGuiCol_Text, ui::kTextDim);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(pages[*page].blurb);
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
        ImGui::Dummy(ImVec2(0, 6 * s));
    };
    // The heading stays over the page, or scrolls with it where it would take
    // more than a quarter of the panel (a large Menu text size).
    const float head_h = 27.0f * s * 1.3f + ImGui::CalcTextSize(pages[*page].blurb, nullptr, false, ImGui::GetContentRegionAvail().x).y;
    const bool head_scrolls = scrolls && head_h > size.y * 0.25f;  // a page that scrolls itself keeps it over it
    if (!head_scrolls) heading();
    const float shown = static_cast<float>(ImGui::GetTime() - *page_at);
    const float fade = 1.0f - std::pow(1.0f - std::min(1.0f, shown / 0.22f), 3.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * std::max(0.02f, fade));
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (1.0f - fade) * 10 * s);
    ImGui::BeginChild("##body", ImVec2(0, 0), 0,
                      ImGuiWindowFlags_NoBackground | (scrolls ? 0 : ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse));
    if (head_scrolls) heading();
    body();
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::EndChild();
    ImGui::End();
    return close;
}

// The game behind the menu, dimmed: darker at the top and the bottom.
void veil(float k) {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const ImVec2 o = vp->Pos, size = vp->Size;
    const int a = static_cast<int>(k * 255.0f);
    const ImU32 clear = ui::alpha(ui::kInk, 0);
    dl->AddRectFilled(o, o + size, ui::alpha(ui::kInk, a * 120 / 255));
    dl->AddRectFilledMultiColor(o, o + ImVec2(size.x, size.y * 0.3f), ui::alpha(ui::kInk, a * 90 / 255), ui::alpha(ui::kInk, a * 90 / 255),
                                clear, clear);
    dl->AddRectFilledMultiColor(o + ImVec2(0, size.y * 0.6f), o + size, clear, clear, ui::alpha(ui::kInk, a * 110 / 255),
                                ui::alpha(ui::kInk, a * 110 / 255));
}

// ---- F9: the patches --------------------------------------------------------

// The patch list as the menu shows it: the files read when the menu opens
// (patch_manifests_list, in the list's order), each with its switch as saved.
struct PatchRow {
    PatchListing p;
    bool on = false;  // its switch: [patches] in bbhost.toml, or the setting it follows
};
struct PatchMenu {
    bool read = false;
    std::vector<PatchRow> rows;
    char find[64] = {};
    std::string note;  // what the last click did
    bool note_ok = true;
    double note_at = -100.0;
};
PatchMenu g_patches;

std::string capitalised(std::string s) {
    if (!s.empty()) s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    return s;
}

bool contains_nocase(const std::string& hay, const char* needle) {
    if (!*needle) return true;
    const std::string n = needle;
    return std::search(hay.begin(), hay.end(), n.begin(), n.end(),
                       [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b)); }) !=
           hay.end();
}

// A [patches] key bbhost.toml can hold bare (the setup window's rule).
bool bare_key(const std::string& k) {
    return !k.empty() && std::all_of(k.begin(), k.end(), [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_'; });
}

bool toml_bool(const char* key, bool fallback) {
    const std::string v = config_value(key);
    return v.empty() ? fallback : v == "true" || v == "1";
}

// The switch a patch follows: a setting for the four that follow one (the
// setup window's same four), [patches] <name> for the rest.
bool switch_of(const PatchListing& p) {
    const std::string& o = p.option;
    if (o == "startup.skip_intro") return host_opt_get("skip_logos");
    if (o == "startup.performance_mode") return host_opt_get("performance_mode");
    if (o == "debug.free_camera") return host_opt_get("debug_camera");
    if (o == "loading.quick_reentry") return toml_bool("loading.quick_reentry", config().quick_reentry);
    if (o == "streaming.all_post_processors") return toml_bool("streaming.all_post_processors", config().all_post_processors);
    return patch_manifest_on(p.name, p.enabled);
}

// Saves a switch, on the window's thread: the setting, or the key in the
// bbhost.toml settings are saved to.
void save_switch(const PatchRow& r) {
    host_options_post([o = r.p.option, name = r.p.name, on = r.on] {
        if (o == "startup.skip_intro") return host_opt_set("skip_logos", on);
        if (o == "startup.performance_mode") return host_opt_set("performance_mode", on);
        if (o == "debug.free_camera") return host_opt_set("debug_camera", on);
        std::string section = "patches", key = name;
        if (o == "loading.quick_reentry") section = "loading", key = "quick_reentry";
        if (o == "streaming.all_post_processors") section = "streaming", key = "all_post_processors";
        const char* v = on ? "true" : "false";
        if (config_set_values(config_user_file(), {{section, key, v}})) config_value_set(section + "." + key, v);
        else host_log("patch menu: could not write %s", config_user_file().c_str());
    });
}

void read_patches(PatchMenu& m) {
    m.rows.clear();
    const char* mods = hle_fs_mods_root();
    for (PatchListing& p : patch_manifests_list(mods ? mods : "")) {
        PatchRow r;
        r.on = switch_of(p);
        r.p = std::move(p);
        m.rows.push_back(std::move(r));
    }
    m.read = true;
}

// A click on a patch's box: its switch saved, and its bytes in or out now
// where they can be; the others of its group off first.
void set_patch(PatchMenu& m, PatchRow& r, bool on) {
    if (on && !r.p.group.empty()) {
        for (PatchRow& o : m.rows) {
            if (&o == &r || o.p.group != r.p.group || !o.on) continue;
            o.on = false;
            save_switch(o);
            patch_manifest_set_live(o.p.name, false, nullptr);
        }
    }
    r.on = on;
    save_switch(r);
    std::string why;
    const bool now = patch_manifest_set_live(r.p.name, on, &why);
    m.note = patch_display_name(r.p) + (now ? (on ? " is on." : " is off.") : (on ? " turns on" : " turns off") + std::string(" at the next start: ") + why + ".");
    m.note_ok = now;
    m.note_at = ImGui::GetTime();
}

void patches_page(PatchMenu& m) {
    if (!m.read) read_patches(m);
    {
        std::vector<std::pair<std::string, bool>> clicks;
        {
            std::lock_guard<std::mutex> lk(g_test_mu);
            clicks.swap(g_test_clicks);
        }
        for (const auto& [name, on] : clicks) {
            for (PatchRow& r : m.rows) {
                if (r.p.name != name) continue;
                set_patch(m, r, on);
                host_log("menu test: patch %s %s: %s", name.c_str(), on ? "on" : "off", m.note.c_str());
            }
        }
    }
    const float s = ui::scale();
    const ImGuiStyle& st = ImGui::GetStyle();
    // The search, and the counts beside it.
    int on = 0, waiting = 0;
    std::vector<PatchLiveState> live(m.rows.size());
    for (std::size_t i = 0; i < m.rows.size(); ++i) {
        live[i] = patch_manifest_live_state(m.rows[i].p.name);
        on += m.rows[i].on;
        waiting += live[i].known && live[i].applied != m.rows[i].on;
    }
    {
        const float find_w = std::round(260 * s);
        const ImVec2 fp = ImGui::GetCursorScreenPos();
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(30 * s, st.FramePadding.y));
        ImGui::SetNextItemWidth(find_w);
        ImGui::InputTextWithHint("##find", "Search", m.find, sizeof(m.find));
        ImGui::PopStyleVar();
        ui::draw_icon(ImGui::GetWindowDrawList(), ui::Icon::Search, fp + ImVec2(15 * s, ImGui::GetFrameHeight() * 0.5f), 14 * s, ui::kTextDim);
        ImGui::SameLine(0, 16 * s);
        char counts[96];
        std::snprintf(counts, sizeof(counts), "%d of %zu on", on, m.rows.size());
        ImGui::AlignTextToFramePadding();
        ImGui::PushStyleColor(ImGuiCol_Text, ui::kTextDim);
        ImGui::TextUnformatted(counts);
        ImGui::PopStyleColor();
        if (waiting) {
            ImGui::SameLine();
            char w[64];
            std::snprintf(w, sizeof(w), "%d at the next start", waiting);
            ui::pill(w, ui::kWarn, ui::alpha(ui::kWarn, 38));
            ui::tooltip_for_item(nullptr, "Switches that take effect when the game starts again.");
        }
    }
    // What the last click did, for a few seconds.
    if (!m.note.empty() && ImGui::GetTime() - m.note_at < 6.0) {
        ui::banner(m.note_ok ? ui::Icon::Check : ui::Icon::Info, m.note_ok ? ui::kGood : ui::kWarn, m.note);
    }
    ImGui::Dummy(ImVec2(0, 2 * s));
    std::vector<std::size_t> shown;
    for (std::size_t i = 0; i < m.rows.size(); ++i) {
        const PatchListing& p = m.rows[i].p;
        if (contains_nocase(p.name, m.find) || contains_nocase(p.title, m.find) || contains_nocase(p.description, m.find)) shown.push_back(i);
    }
    // One line each, in two columns; the bubble has the rest. The card fits
    // the rows, or the page's height with them scrolling inside it.
    const std::size_t per_col = (shown.size() + 1) / 2;
    const float row_h = std::round(ImGui::GetFontSize() + 2 * 3 * s + 10 * s);
    const float fit = 2 * 14 * s + std::max<float>(1.0f, static_cast<float>(per_col)) * row_h + 4 * s;
    ui::card_begin("##patchlist", ImVec2(ImGui::GetContentRegionAvail().x, std::min(fit, std::max(ImGui::GetContentRegionAvail().y, 160 * s))));
    if (m.rows.empty()) ImGui::TextDisabled("No patches found.");
    else if (shown.empty()) ImGui::TextDisabled("No patch matches.");
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4 * s, 3 * s));
    const float col_gap = 22 * s;
    const float col_w = (ImGui::GetContentRegionAvail().x - col_gap) * 0.5f;
    const ImVec2 o = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (std::size_t k = 0; k < shown.size(); ++k) {
        const std::size_t i = shown[k];
        PatchRow& r = m.rows[i];
        const PatchLiveState& lv = live[i];
        const float col = static_cast<float>(k / per_col), row = static_cast<float>(k % per_col);
        const ImVec2 a = o + ImVec2(col * (col_w + col_gap), row * row_h);
        const ImVec2 b = a + ImVec2(col_w, row_h - 4 * s);
        const bool hot = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(a - ImVec2(6 * s, 0), b);
        if (hot) dl->AddRectFilled(a - ImVec2(6 * s, 0), b, ImGui::GetColorU32(IM_COL32(255, 255, 255, 16)), 7 * s);
        const bool unknown_option = !r.p.option.empty() && !patch_manifest_option_known(r.p.option);
        const bool fixed = !r.p.error.empty() || !lv.known || unknown_option || (r.p.option.empty() && !bare_key(r.p.name));
        // At the right: why it cannot be switched, or that it changes when the game starts again.
        float right = b.x - 4 * s;
        const float fh = ImGui::GetFrameHeight();
        auto tag = [&](const char* text, ImU32 col) {
            const float tfs = std::round(ImGui::GetFontSize() * 0.74f);
            const ImVec2 ts = ui::font_strong()->CalcTextSizeA(tfs, FLT_MAX, 0.0f, text);
            const ImVec2 pa(right - ts.x - 14 * s, (a.y + b.y) * 0.5f - ts.y * 0.5f - 2 * s);
            const ImVec2 pb(right, (a.y + b.y) * 0.5f + ts.y * 0.5f + 2 * s);
            dl->AddRectFilled(pa, pb, ImGui::GetColorU32(ui::alpha(col, 46)), (pb.y - pa.y) * 0.5f);
            dl->AddText(ui::font_strong(), tfs, pa + ImVec2(7 * s, 2 * s), ImGui::GetColorU32(col), text);
            right = pa.x - 6 * s;
        };
        if (fixed) {
            ui::draw_icon(dl, ui::Icon::Warning, ImVec2(right - fh * 0.32f, (a.y + b.y) * 0.5f), fh * 0.6f, ImGui::GetColorU32(ui::kWarn));
            right -= fh * 0.7f + 6 * s;
        } else if (lv.applied != r.on) {
            tag(r.on ? "next start" : "until restart", ui::kWarn);
        }
        // Its name up to the marks, cut short in a narrow menu.
        const float name_w = right - a.x - fh - ImGui::GetStyle().ItemInnerSpacing.x;
        const std::string name = ui::fit_text(patch_display_name(r.p), name_w) + "###patch";
        ImGui::SetCursorScreenPos(ImVec2(a.x, a.y + (b.y - a.y - fh) * 0.5f));
        ImGui::PushID(static_cast<int>(i));
        bool box = r.on, off = false;
        ImGui::BeginDisabled(fixed);
        if (ImGui::Checkbox(name.c_str(), fixed ? &off : &box) && !fixed) set_patch(m, r, box);
        ImGui::EndDisabled();
        ImGui::PopID();
        if (hot) {
            std::string note;
            auto add = [&](const std::string& t) {
                if (t.empty()) return;
                if (!note.empty()) note += "\n";
                note += t;
            };
            if (!r.p.error.empty()) add("Not read: " + r.p.error + ".");
            else if (!lv.known) add("Not read for this game's eboot as it started.");
            else if (unknown_option) add("Follows " + r.p.option + ", which this bbhost does not know.");
            // The patch's name and what it does; a note only for one that cannot be switched.
            ui::tooltip(patch_display_name(r.p).c_str(), capitalised(r.p.description.empty() ? std::string("No description.") : r.p.description), note);
        }
    }
    ImGui::PopStyleVar();
    ImGui::SetCursorScreenPos(o + ImVec2(0, static_cast<float>(per_col) * row_h));
    ImGui::Dummy(ImVec2(1, 1));
    ui::card_end();
}

// ---- The menus ---------------------------------------------------------------

int g_mods_page = 0;
double g_mods_page_at = -100.0;
int g_options_page = 0;
double g_options_page_at = -100.0;

bool draw_mods() {
    static const Page kPages[] = {
        {"Patches", ui::Icon::Patches,
         "Changes to the game's code. Most turn on or off at once; the rest when the game starts again."},
        {"Plugins", ui::Icon::Plugins, "Each plugin's settings and actions. Those marked change at once; the rest at the next start."},
    };
    return shell("##mods", "PATCHES & PLUGINS", kPages, 2, &g_mods_page, &g_mods_page_at, g_mods_page == 0, [] {
        if (g_mods_page == 0) {
            patches_page(g_patches);
        } else {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(14, 12, 34, 190));
            plugin_ui_draw(g_ui, ui::scale(), 0.0f);  // the list and the details fill what is left, each scrolling
            ImGui::PopStyleColor();
        }
    });
}

bool draw_options() {
    static std::vector<Page> pages;
    if (pages.empty()) {
        // By name: Display, Input, Graphics, PC Enhancements, Startup, Account, Updates.
        static const struct {
            const char* name;
            ui::Icon icon;
        } kIcons[] = {{"Display", ui::Icon::Display},   {"Input", ui::Icon::Game},       {"Graphics", ui::Icon::Image},
                      {"PC Enhancements", ui::Icon::Enhance}, {"Startup", ui::Icon::Play}, {"Account", ui::Icon::User},
                      {"Updates", ui::Icon::Download}};
        for (int i = 0; i < host_options_page_count(); ++i) {
            Page p{host_options_page_name(i), ui::Icon::Settings, host_options_page_blurb(i)};
            for (const auto& k : kIcons) {
                if (std::strcmp(k.name, p.name) == 0) p.icon = k.icon;
            }
            pages.push_back(p);
        }
    }
    if (pages.empty()) return false;
    return shell("##options", "SETTINGS", pages.data(), static_cast<int>(pages.size()), &g_options_page, &g_options_page_at, true,
                 [] { host_options_draw_page(g_options_page); });
}

}  // namespace

bool ingame_menu_open() { return g_open.load(std::memory_order_relaxed) || host_options_open(); }

bool ingame_menu_event(const SDL_Event& e, float ratio) {
    if (e.type == SDL_EVENT_KEY_DOWN && (e.key.scancode == SDL_SCANCODE_F9 || e.key.scancode == SDL_SCANCODE_F10)) {
        if (e.key.repeat) return true;
        if (!g_ready.load(std::memory_order_acquire)) {
            static bool told = false;
            if (!told) host_log("menu: the in-game menus did not start; F9 and F10 do nothing");
            told = true;
            return true;
        }
        const bool was = ingame_menu_open();
        if (e.key.scancode == SDL_SCANCODE_F9) {
            const bool open = !g_open.load();
            if (open) host_options_set_open(false);
            set_open(open);
        } else {
            const bool open = !host_options_open();
            if (open) set_open(false);
            host_options_set_open(open);
        }
        if (!was && ingame_menu_open()) {
            // Where the pointer is now: no motion may come before the first click.
            Input focus;
            focus.kind = Input::Focus;
            push(focus);
            Input in;
            SDL_GetMouseState(&in.x, &in.y);
            in.x *= ratio;
            in.y *= ratio;
            push(in);
        }
        return true;
    }
    if (!ingame_menu_open()) return false;
    switch (e.type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        if (const ImGuiKey k = key_of(e.key.key); k != ImGuiKey_None) {
            Input in;
            in.kind = Input::Key;
            in.key = k;
            in.down = e.type == SDL_EVENT_KEY_DOWN;
            push(in);
        }
        return true;
    case SDL_EVENT_TEXT_INPUT:
        if (e.text.text) {
            Input in;
            in.kind = Input::Text;
            in.text = e.text.text;
            push(in);
        }
        return true;
    case SDL_EVENT_MOUSE_MOTION: {
        Input in;
        in.kind = Input::Move;
        in.x = e.motion.x * ratio;
        in.y = e.motion.y * ratio;
        push(in);
        return true;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        Input in;
        in.kind = Input::Button;
        in.x = e.button.x * ratio;
        in.y = e.button.y * ratio;
        in.button = e.button.button == SDL_BUTTON_LEFT ? 0 : e.button.button == SDL_BUTTON_RIGHT ? 1 : 2;
        in.down = e.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
        push(in);
        return true;
    }
    case SDL_EVENT_MOUSE_WHEEL: {
        Input in;
        in.kind = Input::Wheel;
        in.x = e.wheel.x;
        in.y = e.wheel.y;
        push(in);
        return true;
    }
    // A controller moves through the menu too. Not taken: the window still
    // notes the pad in use; the game reads no pad while a menu is open.
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
        if (const ImGuiKey k = pad_key_of(e.gbutton.button); k != ImGuiKey_None) {
            Input in;
            in.kind = Input::Key;
            in.key = k;
            in.down = e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
            push(in);
        }
        return false;
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
        if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTX || e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTY) {
            const float v = static_cast<float>(e.gaxis.value) / 32767.0f;
            const bool x = e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTX;
            for (int side = 0; side < 2; ++side) {
                Input in;
                in.kind = Input::Analog;
                in.key = x ? (side ? ImGuiKey_GamepadLStickRight : ImGuiKey_GamepadLStickLeft)
                           : (side ? ImGuiKey_GamepadLStickDown : ImGuiKey_GamepadLStickUp);
                const float a = side ? v : -v;
                in.x = a > 0.25f ? std::min(1.0f, (a - 0.25f) / 0.75f) : 0.0f;
                in.down = in.x > 0.0f;
                push(in);
            }
        }
        return false;
    default:
        return false;
    }
}

void ingame_menu_pump() {
    static const std::vector<std::pair<double, std::string>> steps = [] {
        std::vector<std::pair<double, std::string>> out;
        const char* e = std::getenv("BBHOST_MENU_TEST");
        std::string s = e ? e : "";
        for (std::size_t at = 0; at < s.size();) {
            std::size_t end = s.find(',', at);
            if (end == std::string::npos) end = s.size();
            const std::string item = s.substr(at, end - at);
            if (const std::size_t colon = item.find(':'); colon != std::string::npos)
                out.emplace_back(std::atof(item.substr(0, colon).c_str()), item.substr(colon + 1));
            at = end + 1;
        }
        return out;
    }();
    if (steps.empty()) return;
    static const auto start = std::chrono::steady_clock::now();
    static std::size_t next = 0;
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    while (next < steps.size() && now >= steps[next].first && g_ready.load(std::memory_order_acquire)) {
        const std::string& step = steps[next++].second;
        host_log("menu test: %.1f s: %s", now, step.c_str());
        if (step == "f9") {
            const bool open = !g_open.load();
            if (open) host_options_set_open(false);
            set_open(open);
        } else if (step == "f10") {
            const bool open = !host_options_open();
            if (open) set_open(false);
            host_options_set_open(open);
        } else if (step == "close") {
            set_open(false);
            host_options_set_open(false);
        } else if (step.rfind("page:", 0) == 0) {
            g_test_page.store(std::atoi(step.c_str() + 5));
        } else if (step.rfind("patch:", 0) == 0) {
            const std::size_t eq = step.find('=');
            if (eq == std::string::npos) continue;
            std::lock_guard<std::mutex> lk(g_test_mu);
            g_test_clicks.emplace_back(step.substr(6, eq - 6), step.substr(eq + 1) == "on");
        }
    }
}

bool ingame_menu_init(VkInstance instance, VkPhysicalDevice phys, VkDevice device, std::uint32_t family, VkQueue queue,
                      VkFormat format, std::uint32_t image_count) {
    if (g_vk_ready && format == g_format && image_count == g_images) return true;
    ingame_menu_shutdown();
    IMGUI_CHECKVERSION();
    g_ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(g_ctx);
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors | ImGuiBackendFlags_HasGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    ui::init_ingame(1.0f);
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_3;
    info.Instance = instance;
    info.PhysicalDevice = phys;
    info.Device = device;
    info.QueueFamily = family;
    info.Queue = queue;
    info.DescriptorPoolSize = 16;
    info.MinImageCount = image_count < 2 ? 2 : image_count;
    info.ImageCount = info.MinImageCount;
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    g_format = format;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &g_format;
    if (!ImGui_ImplVulkan_Init(&info)) {
        host_log("menu: the Vulkan backend did not start; F9 and F10 do nothing");
        ImGui::DestroyContext(g_ctx);
        g_ctx = nullptr;
        return false;
    }
    g_images = image_count;
    g_vk_ready = true;
    g_ready.store(true, std::memory_order_release);
    g_ui.in_game = true;
    g_last = std::chrono::steady_clock::now();
    return true;
}

void ingame_menu_shutdown() {
    if (!g_ctx) return;
    g_ready.store(false, std::memory_order_release);
    ImGui::SetCurrentContext(g_ctx);
    if (g_vk_ready) ImGui_ImplVulkan_Shutdown();
    ImGui::DestroyContext(g_ctx);
    g_ctx = nullptr;
    g_vk_ready = false;
}

namespace {

// ImGui's frame for the menu, built and rendered to draw data; false when
// neither menu is open (no ImGui frame at all, so a closed menu costs the
// presenter nothing but this check).
bool build_frame(VkExtent2D extent) {
    const bool mods = g_open.load(std::memory_order_relaxed), options = host_options_open();
    if (!g_vk_ready || (!mods && !options)) {
        g_ui.scanned = false;     // read the files again when it next opens
        g_patches.read = false;
        g_was_up = false;
        return false;
    }
    ImGui::SetCurrentContext(g_ctx);
    ImGuiIO& io = ImGui::GetIO();
    const auto now = std::chrono::steady_clock::now();
    io.DeltaTime = std::max(1e-4f, std::chrono::duration<float>(now - g_last).count());
    g_last = now;
    io.DisplaySize = ImVec2(static_cast<float>(extent.width), static_cast<float>(extent.height));
    // The setup window's sizes, for a 900-line screen: 1.2 at 1080p, 2.4 at 4K.
    // Menu text size (F10, the setup window's Settings) on top: 100% to 400%.
    ui::set_scale(std::max(0.6f, static_cast<float>(extent.height) / 900.0f) * host_opt_menu_text_scale());
    // The pointer over the menu: ImGui draws it last, on top. The host's own
    // (host_overlay_cursor) is drawn under the menu and stands down while it
    // is open; with "Draw the pointer" off the window system's shows instead.
    io.MouseDrawCursor = host_settings_ref().draw_cursor;
    std::vector<Input> q;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        q.swap(g_queue);
    }
    for (const Input& in : q) {
        switch (in.kind) {
        case Input::Move: io.AddMousePosEvent(in.x, in.y); break;
        case Input::Button:
            io.AddMousePosEvent(in.x, in.y);
            io.AddMouseButtonEvent(in.button, in.down);
            break;
        case Input::Wheel: io.AddMouseWheelEvent(in.x, in.y); break;
        case Input::Key:
            io.AddKeyEvent(in.key, in.down);
            // ImGui reads shortcuts (Ctrl+A, Ctrl+V) from its modifier keys.
            if (in.key == ImGuiKey_LeftCtrl || in.key == ImGuiKey_RightCtrl) io.AddKeyEvent(ImGuiMod_Ctrl, in.down);
            if (in.key == ImGuiKey_LeftShift || in.key == ImGuiKey_RightShift) io.AddKeyEvent(ImGuiMod_Shift, in.down);
            break;
        case Input::Analog: io.AddKeyAnalogEvent(in.key, in.down, in.x); break;
        case Input::Text: io.AddInputCharactersUTF8(in.text.c_str()); break;
        case Input::Focus: io.AddFocusEvent(true); break;
        }
    }
    if (const int p = g_test_page.exchange(-1); p >= 0) {
        int& page = mods ? g_mods_page : g_options_page;
        const int n = mods ? 2 : host_options_page_count();
        if (p < n) page = p;
    }
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    if (!g_was_up) {
        g_shown_at = ImGui::GetTime();
        g_mods_page_at = g_options_page_at = g_shown_at;
    }
    g_was_up = true;
    // Escape, the pad's B or Start close it - unless something in it takes
    // them first: an open list, a field being typed, a held slider.
    const bool busy = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId) || io.WantTextInput || ImGui::IsAnyItemActive();
    bool close = !busy && (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false) ||
                           ImGui::IsKeyPressed(ImGuiKey_GamepadStart, false));
    // It fades in.
    const float k = std::min(1.0f, static_cast<float>(ImGui::GetTime() - g_shown_at) / 0.16f);
    veil(k);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, std::max(0.05f, k));
    close = (mods ? draw_mods() : draw_options()) || close;
    ImGui::PopStyleVar();
    if (close) {
        if (mods) set_open(false);
        else host_options_close();
    }
    ImGui::Render();
    // Menu hue and Menu vibrance, on everything the menus draw.
    float hue = 0.0f, vibrance = 0.0f;
    host_opt_menu_colour(&hue, &vibrance);
    ui::set_colour(hue, vibrance);
    ui::recolour(ImGui::GetDrawData());
    // ImGui's own texture uploads - the font atlas when the menu first opens,
    // and again whenever 1.92 adds glyphs to it - submit on the renderer's
    // queue and then wait for that queue to go idle. Left to RenderDrawData
    // they ran on this thread with no lock, beside the submission thread's
    // vkQueueSubmit on the same queue, which Vulkan requires the caller to
    // keep apart. Done here under the queue's lock (as the overlay's atlas
    // is, host/overlay.cpp), RenderDrawData then finds nothing to upload. A
    // frame with no texture to change - nearly all of them - takes no lock.
    ImDrawData* dd = ImGui::GetDrawData();
    bool uploads = false;
    if (dd && dd->Textures) {
        for (ImTextureData* tex : *dd->Textures) uploads = uploads || tex->Status != ImTextureStatus_OK;
    }
    if (uploads) {
        host_gpu_queue_lock();
        for (ImTextureData* tex : *dd->Textures) {
            if (tex->Status != ImTextureStatus_OK) ImGui_ImplVulkan_UpdateTexture(tex);
        }
        host_gpu_queue_unlock();
    }
    return true;
}

}  // namespace

bool ingame_menu_record_in_pass(VkCommandBuffer cmd, VkExtent2D extent) {
    if (!build_frame(extent)) return false;
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    return true;
}

void ingame_menu_record(VkCommandBuffer cmd, VkImageView view, VkExtent2D extent) {
    if (!build_frame(extent)) return;
    VkRenderingAttachmentInfo colour{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colour.imageView = view;
    colour.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colour.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colour.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colour;
    vkCmdBeginRendering(cmd, &ri);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

#else  // a build without a window: no menu

bool ingame_menu_open() { return false; }
bool ingame_menu_event(const SDL_Event&, float) { return false; }
void ingame_menu_pump() {}
bool ingame_menu_init(VkInstance, VkPhysicalDevice, VkDevice, std::uint32_t, VkQueue, VkFormat, std::uint32_t) { return false; }
void ingame_menu_shutdown() {}
void ingame_menu_record(VkCommandBuffer, VkImageView, VkExtent2D) {}
bool ingame_menu_record_in_pass(VkCommandBuffer, VkExtent2D) { return false; }

#endif
