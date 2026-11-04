#include "host/controls_page.h"

#include "engine/debug_menu.h"
#include "host/bindings.h"
#include "host/launcher_ui.h"
#include "host/options.h"
#include "host/window.h"

#include "imgui.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace ui = launcher_ui;

namespace {

// A choice the page began (and not the game's Key Bindings screen), which
// ends with the menu.
bool g_page_capture = false;
// The bindings as last saved: a change made on the page is written at once,
// as the game's screen writes its own.
std::uint64_t g_saved_serial = 0;

struct Section {
    const char* name;
    int first, last;  // BindAction, inclusive
};
constexpr Section kSections[] = {
    {"MOVEMENT", kBindMoveF, kBindLockOn},
    {"COMBAT", kBindAttack, kBindUseItem},
    {"ITEMS AND GESTURES", kBindInteract, kBindEffects},
    {"CAMERA AND MENUS", kBindLookU, kBindBack},
    {"MENU PAGES AND OPTIONS", kBindMenuPageL, kBindMenuStatus},
    {"OTHER", kBindL3, kBindDebugMenu},
};

// What the controller column says for an action without a button of its own
// to move: the stick it is, a menu's fixed button, or nothing.
const char* pad_fixed(int a) {
    if (a >= kBindMoveF && a <= kBindMoveR) return "Left stick";
    if (a >= kBindLookU && a <= kBindLookR) return "Right stick";
    if (const int m = host_pad_menu_input(a); m >= 0) return host_pad_input_name(m);
    return nullptr;
}

// One cell: the binding as a button, "Press ..." while it is being chosen.
bool cell(const char* id, const char* text, bool choosing, float w, bool enabled) {
    ImGui::PushID(id);
    if (!enabled) ImGui::BeginDisabled();
    if (choosing) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    const bool clicked = ImGui::Button(choosing ? (std::strcmp(id, "##pad") == 0 ? "Press a button" : "Press a key") : text,
                                       ImVec2(w, 0.0f));
    if (choosing) ImGui::PopStyleColor();
    if (!enabled) ImGui::EndDisabled();
    ImGui::PopID();
    return clicked && enabled;
}

void save_if_changed() {
    if (host_bind_capturing() >= 0 || host_pad_capturing() >= 0) return;
    if (const std::uint64_t s = host_bindings_serial(); s != g_saved_serial) {
        g_saved_serial = s;
        host_options_save_now();
    }
}

}  // namespace

void controls_page_draw(bool setup_window) {
    if (!g_saved_serial) g_saved_serial = host_bindings_serial();
    const float sc = ui::scale();
    if (g_page_capture && host_bind_capturing() < 0 && host_pad_capturing() < 0) g_page_capture = false;
    save_if_changed();

    // The settings' Input page's rows first.
    for (int i = 0; i < host_options_page_count(); ++i) {
        if (std::strcmp(host_options_page_name(i), "Input") != 0) continue;
        if (setup_window) host_options_draw_setup_page(i);
        else host_options_draw_page(i);
    }
    ImGui::Spacing();

    // The bindings' bar: Reset all on the left, and beside it how to choose a
    // binding - or, while one is being chosen, which and how to back out.
    const int key_choosing = host_bind_capturing(), pad_choosing = host_pad_capturing();
    ui::caption("BINDINGS");
    if (ui::ghost_button("Reset all to defaults", ImVec2(0.0f, 0.0f))) ImGui::OpenPopup("##reset_all");
    ui::tooltip_for_item("Reset all to defaults", "Every key, mouse button and controller button back to its default "
                                                  "(bbhost.toml's [keys] for the keys and the mouse).");
    if (ImGui::BeginPopup("##reset_all")) {
        ImGui::TextUnformatted("Put every binding back to its default?");
        if (ui::primary_button("Reset all", ImVec2(130.0f * sc, 0.0f))) {
            host_bindings_reset_all();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ui::ghost_button("Cancel", ImVec2(130.0f * sc, 0.0f))) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    ImGui::SameLine(0.0f, 14.0f * sc);
    ImGui::AlignTextToFramePadding();
    ImGui::PushTextWrapPos(0.0f);
    if (key_choosing >= 0) {
        ImGui::Text("%s: press the new key or mouse button. Escape cancels.", host_binding_info(key_choosing).label);
    } else if (pad_choosing >= 0) {
        ImGui::Text("%s: press the new controller button. Escape cancels.", host_binding_info(pad_choosing).label);
    } else {
        ImGui::TextDisabled("Click a box, then press the new key, mouse button or controller button; Escape cancels. "
                            "Right-click clears a key or a mouse button, and the arrow at the end of a line puts that "
                            "action back to its default. The controller's buttons move in play only: in menus each is "
                            "itself, so Confirm and Back stay where they were.");
    }
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_PadOuterX | ImGuiTableFlags_SizingFixedFit;
    if (!ImGui::BeginTable("##controls", 5, flags)) return;
    // The three binding columns share what the action's name and the reset
    // leave, up to 170 each: a narrow menu or a large text size keeps every
    // column, the reset's too, on the page.
    const float reset_w = std::max(ui::reset_width(), ImGui::CalcTextSize("Reset").x);
    const float spare = ImGui::GetContentRegionAvail().x - 150.0f * sc - reset_w - ImGui::GetStyle().CellPadding.x * 10.0f;
    const float cw = std::clamp(spare / 3.0f, 72.0f * sc, 170.0f * sc);
    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Keyboard", ImGuiTableColumnFlags_WidthFixed, cw);
    ImGui::TableSetupColumn("Mouse", ImGuiTableColumnFlags_WidthFixed, cw);
    ImGui::TableSetupColumn("Controller", ImGuiTableColumnFlags_WidthFixed, cw);
    ImGui::TableSetupColumn("Reset", ImGuiTableColumnFlags_WidthFixed, reset_w);
    ImGui::TableHeadersRow();
    for (const Section& sec : kSections) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Spacing();
        ui::caption(sec.name);
        for (int a = sec.first; a <= sec.last; ++a) {
            if (a == kBindDebugMenu && !debug_menu_active()) continue;  // as the game's screen lists it
            const BindingInfo& info = host_binding_info(a);
            ImGui::PushID(a);
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(info.label);
            ui::tooltip_for_item(info.label, info.help);

            // The key and the mouse button: one choice takes either, as the
            // game's screen has it.
            const int k = host_binding_key(a), m = host_binding_mouse(a);
            const char* key = k ? host_key_name(k) : nullptr;
            const char* mouse = m ? host_mouse_name(m) : nullptr;
            ImGui::TableSetColumnIndex(1);
            if (cell("##key", key ? key : "-", key_choosing == a, cw, true)) {
                host_bind_capture_begin(a);
                g_page_capture = true;
            }
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right) && k) host_binding_clear_key(a);
            ImGui::TableSetColumnIndex(2);
            if (cell("##mouse", mouse ? mouse : "-", key_choosing == a, cw, true)) {
                host_bind_capture_begin(a);
                g_page_capture = true;
            }
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right) && m) host_binding_clear_mouse(a);

            // The controller: its button, or what it is fixed to.
            ImGui::TableSetColumnIndex(3);
            const int pad = host_pad_input_of(a);
            if (pad >= 0) {
                if (cell("##pad", host_pad_input_name(pad), pad_choosing == a, cw, true)) {
                    host_pad_capture_begin(a);
                    g_page_capture = true;
                }
                if (a == kBindGestures || a == kBindEffects)
                    ui::tooltip_for_item(info.label, "Gestures and Personal Effects are the touchpad's two sides: they move together.");
            } else {
                const char* fixed = pad_fixed(a);
                cell("##pad", fixed ? fixed : "-", false, cw, false);
                if (host_pad_menu_input(a) >= 0) ui::tooltip_for_item(info.label, "A menu's own button: it does not move.");
            }

            ImGui::TableSetColumnIndex(4);
            const bool at_default = host_binding_at_default(a);
            ImGui::BeginDisabled(at_default);
            if (ui::icon_button("##reset", ui::Icon::Refresh, 0.0f)) host_binding_reset(a);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled))
                ui::tooltip(info.label, at_default ? "Its default key, mouse button and controller button."
                                                   : "Back to its default key, mouse button and controller button.");
            ImGui::PopID();
        }
    }
    ImGui::EndTable();
}

bool controls_capture_event(const SDL_Event& e) {
    if (host_pad_capturing() >= 0) {
        if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_ESCAPE) {
            host_pad_capture_cancel();
            return true;
        }
        if (const int in = host_pad_input_of_event(e); in >= 0) {
            host_pad_capture_input(in);
            return true;
        }
        // Nothing of the controller reaches the menu while one is chosen.
        return e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN || e.type == SDL_EVENT_GAMEPAD_BUTTON_UP;
    }
    if (host_bind_capturing() < 0) return false;
    switch (e.type) {
        case SDL_EVENT_KEY_DOWN:
            if (!e.key.repeat) host_bind_capture_key(e.key.scancode);
            return true;
        case SDL_EVENT_KEY_UP:
            return true;
        case SDL_EVENT_MOUSE_BUTTON_DOWN: {
            const std::uint8_t b = e.button.button;
            const int n = b == SDL_BUTTON_LEFT ? 1 : b == SDL_BUTTON_RIGHT ? 2 : b == SDL_BUTTON_MIDDLE ? 3 : b == SDL_BUTTON_X1 ? 4
                        : b == SDL_BUTTON_X2 ? 5 : 0;
            if (n) host_bind_capture_mouse(n);
            return true;
        }
        case SDL_EVENT_MOUSE_WHEEL:
            host_bind_capture_mouse(e.wheel.y > 0.0f ? 6 : 7);
            return true;
        default:
            return false;
    }
}

void controls_menu_closed() {
    if (!g_page_capture) return;
    g_page_capture = false;
    if (host_pad_capturing() >= 0) host_pad_capture_cancel();
    if (host_bind_capturing() >= 0) host_bind_capture_key(SDL_SCANCODE_ESCAPE);
}
