// The setup window's look (host/launcher.cpp): Kyo's night - ink and indigo,
// violet, electric blue and ice, a touch of pink - over a moonlit night,
// and the widgets the window is made of. Dear ImGui on SDL3's 2D renderer,
// like the window itself; nothing here is the game's renderer.
#pragma once

#include "imgui.h"

#include <string>

struct SDL_Renderer;

namespace launcher_ui {

// The palette.
inline constexpr ImU32 kInk = IM_COL32(7, 6, 15, 255);
inline constexpr ImU32 kText = IM_COL32(232, 236, 255, 255);
inline constexpr ImU32 kTextDim = IM_COL32(156, 162, 206, 255);
inline constexpr ImU32 kTextFaint = IM_COL32(112, 116, 162, 255);
inline constexpr ImU32 kViolet = IM_COL32(124, 77, 255, 255);
inline constexpr ImU32 kBlue = IM_COL32(61, 139, 255, 255);
inline constexpr ImU32 kIce = IM_COL32(191, 216, 255, 255);
inline constexpr ImU32 kPink = IM_COL32(255, 138, 200, 255);
inline constexpr ImU32 kGood = IM_COL32(94, 226, 170, 255);
inline constexpr ImU32 kWarn = IM_COL32(255, 186, 92, 255);
inline constexpr ImU32 kBad = IM_COL32(255, 96, 128, 255);

// A colour with its alpha replaced (0-255).
ImU32 alpha(ImU32 col, int a);

enum class Icon {
    Game, Online, Enhance, Patches, Shaders, Plugins, Settings,
    Folder, File, Check, Warning, Info, Play, Moon, Search, Copy, User, Download, Refresh, SignIn,
    Display, Image,
};

// Once the ImGui context and its renderer backend exist: the fonts (the
// system's own: Segoe UI and Palatino on Windows, Noto or DejaVu elsewhere,
// ImGui's built-in one when none is there), the style, scaled, and the
// painting. shutdown() before the renderer goes.
void init(SDL_Renderer* ren, float scale);
void shutdown();
// The same look over the game (host/ingame_menu.cpp, the F9 and F10 menus):
// the fonts and the style in the current context, no painting - the game is
// the backdrop, and glass is tinted glass over it. set_scale() again when the
// window's size changes. The two never run at once: the setup window has
// closed before the game's window opens.
void init_ingame(float scale);
void set_scale(float scale);

float scale();
ImFont* font_strong();  // semibold, for headings
ImFont* font_title();   // the serif, for the name and the big buttons

// Each frame, first: the painting covering the window (drifting a little),
// its veils, darker toward the sidebar and the bottom, and the motes.
void backdrop(float sidebar_w);

// A panel of frosted glass: the painting blurred under it, tinted.
void glass(ImDrawList* dl, ImVec2 a, ImVec2 b, float rounding, ImU32 tint, ImU32 edge);
// A panel of lightly frosted glass: the painting blurred a little under it,
// tinted - the pages' cards.
void pane(ImDrawList* dl, ImVec2 a, ImVec2 b, float rounding, ImU32 tint, ImU32 edge);

void draw_icon(ImDrawList* dl, Icon icon, ImVec2 center, float size, ImU32 col);
// Text with its letters spread `tracking` pixels apart; its width.
void spaced_text(ImDrawList* dl, ImFont* font, float size, ImVec2 pos, ImU32 col, const char* text, float tracking);
float spaced_width(ImFont* font, float size, const char* text, float tracking);

// The name in the sidebar: a pale moon (left out when `compact`), the name in
// spaced capitals, the version under it, centred in `width` from the cursor.
void logo(const char* name, const char* version, float width, bool compact = false);

// A card: glass with a small spaced caption, its contents padded. A size of
// 0 on y fits the contents.
void card_begin(const char* id, ImVec2 size, const char* caption = nullptr);
// `text` cut to `width` in the current font, ending "..." when cut.
std::string fit_text(const std::string& text, float width);
// The current window's child windows (drawn with a clear ChildBg) given the
// cards' backing, under them: for panels that are not cards themselves.
void child_cards();
void card_end();
// The small spaced capitals that head a card or a group.
void caption(const char* text);
// A row's label: dim, on the frame's baseline, then the cursor at `x` (from
// the row's start) for the control.
void row_label(const char* label, float x);

// An on/off switch, sliding. True when flipped.
bool toggle(const char* id, bool* v);
// A row: label (and a dimmer line under it) on the left, the switch on the right.
bool toggle_row(const char* label, const char* sub, bool* v, const char* tip = nullptr);
// One of `n` choices side by side, the chosen one lit. True when changed.
// `tips`, when given, holds a bubble for each choice (null for none), shown
// while that choice is hovered; `note` goes under each.
bool segmented(const char* id, int* v, const char* const* labels, int n, float width, const char* const* tips = nullptr,
               const char* note = nullptr);
// A dropdown with a slim chevron, as wide as SetNextItemWidth says. True when changed.
// `tips`, when given, holds a bubble for each item (null for none), shown while
// the item is hovered in the open list.
bool combo(const char* id, int* v, const char* const* items, int n, const char* const* tips = nullptr);
// `text` cut from the left ("...") to fit `width`.
std::string fit_left(const std::string& text, float width);
// The accent button: blue into violet. Disabled when !enabled.
bool primary_button(const char* label, ImVec2 size, bool enabled = true);
// A square button holding an icon.
bool icon_button(const char* id, Icon icon, float size, const char* tip = nullptr);
// A quiet button: a faint fill that lights when hovered.
bool ghost_button(const char* label, ImVec2 size = ImVec2(0, 0));
// The sidebar's entries: a dot of colour `dot` (0 for none) at the right, or
// `badge`'s text in it.
bool nav_item(const char* label, Icon icon, bool selected, ImU32 dot = 0, const char* badge = nullptr);
// An account's disc: the name's first letter on blue into violet, or a
// figure when there is no one signed in.
void avatar(const std::string& name, float size);
// The big one: PLAY, glowing.
bool play_button(const char* label, ImVec2 size, bool enabled);
// A small rounded label in the flow.
void pill(const char* text, ImU32 fg, ImU32 bg);
// A key's cap (F9) and, beside it, what it does: a line of a sidebar's foot.
void key_hint(const char* key, const char* what);
// A status icon (a check in a disc, or a warning) on the frame's line.
void status_icon(bool ok, const char* tip = nullptr);
// A progress bar: the track, the filled part blue into violet, the text on it.
void progress(float frac, const char* overlay, float height = 0.0f);
// A bubble at the pointer: a title, the text, a dimmer note, each optional.
// tooltip_for_item() shows it while the last item is hovered.
void tooltip(const char* title, const std::string& body, const std::string& note = {});
void tooltip_for_item(const char* title, const std::string& body, const std::string& note = {});
// A banner across the content: an icon, the text, in the icon's colour.
void banner(Icon icon, ImU32 col, const std::string& text);

// How far (0-1) an animation keyed by `id` has gone toward `on`.
float ease(ImGuiID id, bool on, float speed = 10.0f);

// The window's motion on or off (startup.setup_animations). Off: the
// background, the moon and the Play button stand still, there are no motes,
// and ease() jumps to where it is going.
void set_animations(bool on);
bool animations();
// ImGui::GetTime() while the window moves; one fixed moment while it does not.
float anim_time();

}  // namespace launcher_ui
