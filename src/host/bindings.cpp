#include "host/bindings.h"

#include "core/config.h"
#include "engine/debug_menu.h"
#include "engine/menu_pointer.h"
#include "host/window.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>

#if defined(BBHOST_HAVE_SDL3)
#include <SDL3/SDL.h>
#endif

namespace {

// Bloodborne's pad, for the record, since every row below is one of these:
// R1 attacks, R2 is the strong attack, L1 transforms the weapon, L2 is the
// firearm (and the parry), Circle dodges and sprints, Triangle drinks a Blood
// Vial ("KG_R_U: Restore HP (Uses Blood Vial)" in the game's own help text),
// Square uses the quick item ("KG_R_L: Use item"), Cross interacts, the d-pad
// switches weapons and items, R3 locks on, the touchpad's two sides are
// Gestures and Personal Effects. That is the US disc's layout; the Japanese
// region swaps Circle and Cross, in play and in the menus alike, and the
// region comes from param.sfo (hle/system.cpp).
constexpr std::uint32_t kL3 = 0x2, kR3 = 0x4, kOptions = 0x8, kUp = 0x10, kRight = 0x20, kDown = 0x40, kLeft = 0x80,
                        kL2 = 0x100, kR2 = 0x200, kL1 = 0x400, kR1 = 0x800, kTriangle = 0x1000, kCircle = 0x2000,
                        kCross = 0x4000, kSquare = 0x8000, kTouchPad = 0x100000;

// The defaults are DS3's shape - WASD, Space to roll, E to interact, R for the
// healing item, F for the quick item, Q to lock on, Shift as the strong
// modifier, the mouse's buttons for the hands - put on Bloodborne's buttons.
const BindingInfo kInfo[kBindCount] = {
    {"move_forward", "Move Forward", "Left stick up.", 0, "W", 0},
    {"move_back", "Move Back", "Left stick down.", 0, "S", 0},
    {"move_left", "Move Left", "Left stick left.", 0, "A", 0},
    {"move_right", "Move Right", "Left stick right.", 0, "D", 0},
    // Held with the movement keys, as Dark Souls III's Left Alt: a walk.
    {"walk", "Walk", "Hold with the movement keys to walk.", 0, "Left Alt", 0},
    {"roll", "Dodge / Sprint", "Circle. Tap to dodge, hold to sprint.", kCircle, "Space", 0},
    {"lock_on", "Lock On", "R3. Lock on to a target, or reset the camera.", kR3, "Q", 0},

    {"attack", "Attack", "R1. With the strong attack modifier held, R2.", kR1, nullptr, 1},
    {"strong_attack", "Strong Attack", "R2. Hold to charge.", kR2, "V", 0},
    {"strong_modifier", "Strong Attack Modifier", "Held with Attack, attacks strong (R2).", 0, "Left Shift", 0},
    // The middle button is Camera Zoom Reset's, beside the wheel's zoom.
    {"transform", "Transform", "L1. Transform the trick weapon.", kL1, "C", 0},
    {"firearm", "Firearm / Parry", "L2. Shoot the left-hand weapon.", kL2, "X", 2},
    {"blood_vial", "Blood Vial", "Triangle. Restore HP with a Blood Vial.", kTriangle, "R", 0},
    {"use_item", "Use Item", "Square. Use the selected quick item.", kSquare, "F", 0},

    {"interact", "Interact", "Cross. Talk, examine, pick up, open.", kCross, "E", 0},
    {"switch_item", "Switch Item", "D-pad down. Cycle the quick items.", kDown, "Down", 0},
    {"switch_right", "Switch Right Weapon", "D-pad right.", kRight, "Right", 0},
    {"switch_left", "Switch Left Weapon", "D-pad left.", kLeft, "Left", 0},
    {"dpad_up", "D-pad Up", "D-pad up.", kUp, "Up", 0},
    // The touchpad is a button with a side: the game's key guide reads
    // "<KG_TP_L> : Gestures" and "<KG_TP_R> : Personal Effects". A press alone
    // cannot say which, so these place a finger as well (apply below).
    {"gestures", "Gestures", "The touchpad's left side.", kTouchPad, "G", 0},
    {"personal_effects", "Personal Effects", "The touchpad's right side.", kTouchPad, "T", 0},

    {"look_up", "Look Up", "Right stick up, for the camera without the mouse.", 0, "I", 0},
    {"look_down", "Look Down", "Right stick down.", 0, "K", 0},
    {"look_left", "Look Left", "Right stick left.", 0, "J", 0},
    {"look_right", "Look Right", "Right stick right.", 0, "L", 0},
    {"menu", "Menu", "Options. Open the menu.", kOptions, "Escape", 0},
    // Menus need an OK and a Back that are not the gameplay keys sharing their
    // pad bits: nobody reaches for Space to accept or E to back out.
    // Their buttons are the region's (engine/menu_pointer.h), set in apply.
    {"confirm", "Menu Confirm", "OK, in menus.", 0, "Return", 0},
    {"back", "Menu Back", "Return, in menus.", 0, "Backspace", 0},

    // What a menu reads from the shoulders and two face buttons - its pages,
    // its two shortcuts, its display and status switches - on keys of its own.
    // In play those buttons are the attacks, the firearm, Transform, the Blood
    // Vial and the quick item, whose keys stay out of the menus
    // (host_binding_in_menus): the storage box's Show Storage / Show Inventory
    // (L2 and R2) was on the firearm's X and Strong Attack's V, and a menu's
    // next page or tab (R1) on Attack's left mouse button, which in a menu is
    // the pointer's click. The game's own key guides name them
    // (engine/key_prompts.cpp).
    {"menu_page_left", "Menu Previous Page", "L1 in menus: the previous page, tab or category.", kL1, "PageUp", 0},
    {"menu_page_right", "Menu Next Page", "R1 in menus: the next page, tab or category.", kR1, "PageDown", 0},
    {"menu_shortcut_left", "Menu Shortcut (L2)", "L2 in menus: the storage box's Show Storage, the rankings' top.", kL2,
     "Home", 0},
    {"menu_shortcut_right", "Menu Shortcut (R2)", "R2 in menus: the storage box's Show Storage, your own grades.", kR2,
     "End", 0},
    {"menu_display", "Menu Toggle Display", "Square in menus: Toggle Display, Unequip, Remove, Delete.", kSquare, "F",
     0},
    {"menu_status", "Menu Toggle Status", "Triangle in menus: Toggle Status, Switch, Initialize.", kTriangle, "R", 0},

    {"l3", "Left Stick Press", "L3.", kL3, "Left Ctrl", 0},
    // The follow camera's distance, in the world: each press a step of the
    // Camera distance setting (10%), Reset the game's own. No pad press.
    {"zoom_in", "Camera Zoom In", "Brings the camera a step closer.", 0, nullptr, 6},
    {"zoom_out", "Camera Zoom Out", "Takes the camera a step further out.", 0, nullptr, 7},
    {"zoom_reset", "Camera Zoom Reset", "The camera back to the game's own distance.", 0, nullptr, 3},
    // The developers' debug menu: not a pad press at all. The patch opens it
    // on the touchpad's left side, which is also Gestures; this key toggles
    // the menu itself (engine/debug_menu.h), when the Debug Menu plugin is
    // on. The Key Bindings screen lists it only then.
    {"debug_menu", "Debug Menu", "Opens and closes the developers' debug menu (the Debug Menu plugin).", 0, "`", 0},
};

// The controller's sixteen buttons (host_pad_*), in PadInput's order: their
// names on the Controls page and in [pad], and the pad bit each one is.
const char* const kPadNames[kPadInputs] = {"Cross", "Circle", "Square", "Triangle", "L1", "R1", "L2", "R2",
                                           "L3", "R3", "Options", "Touchpad", "D-pad Up", "D-pad Down", "D-pad Left", "D-pad Right"};
const char* const kPadTokens[kPadInputs] = {"Cross", "Circle", "Square", "Triangle", "L1", "R1", "L2", "R2",
                                            "L3", "R3", "Options", "Touchpad", "Up", "Down", "Left", "Right"};
const std::uint32_t kPadBits[kPadInputs] = {kCross, kCircle, kSquare, kTriangle, kL1, kR1, kL2, kR2,
                                            kL3, kR3, kOptions, kTouchPad, kUp, kDown, kLeft, kRight};
// What each button gives in play: the button whose bit it presses, itself
// until rebound. Always a permutation (a rebind swaps two).
std::atomic<int> g_pad_gives[kPadInputs] = {kPadCross, kPadCircle, kPadSquare, kPadTriangle, kPadL1, kPadR1, kPadL2, kPadR2,
                                            kPadL3, kPadR3, kPadOptions, kPadTouchpad, kPadUp, kPadDown, kPadLeft, kPadRight};
std::atomic<int> g_pad_capture{-1};
std::atomic<std::int64_t> g_pad_capture_ms{0};

int pad_input_of_bit(std::uint32_t bit) {
    for (int i = 0; i < kPadInputs; ++i) {
        if (kPadBits[i] == bit) return i;
    }
    return -1;
}

// Earlier names a bbhost.toml may still use.
const struct {
    const char* old_name;
    int action;
} kAliases[] = {
    {"dpad_down", kBindSwitchItem}, {"dpad_left", kBindSwitchLeft}, {"dpad_right", kBindSwitchRight},
};

std::atomic<int> g_key[kBindCount];
std::atomic<int> g_mouse[kBindCount];

std::atomic<int> g_capture{-1};
// The Debug Menu key: held per device (keyboard, mouse), since each device's
// actions are read on their own, and a press queued for the pad read.
std::atomic<bool> g_debug_held[2], g_debug_toggle{false};

void note_debug_key(int device, bool down) {
    if (!down) {
        g_debug_held[device].store(false, std::memory_order_relaxed);
    } else if (!g_debug_held[device].exchange(true) && debug_menu_active()) {
        g_debug_toggle.store(true);
    }
}
// The zoom actions likewise: held per device, each press queued for the pad
// read as a step (Out +1, In -1) or a reset (host_bind_take_zoom).
std::atomic<bool> g_zoom_held[2][3];
std::atomic<int> g_zoom_steps{0};
std::atomic<bool> g_zoom_reset{false};

void note_zoom_keys(int device, const bool held[kBindCount]) {
    const int actions[3] = {kBindZoomIn, kBindZoomOut, kBindZoomReset};
    for (int k = 0; k < 3; ++k) {
        const bool down = held[actions[k]];
        if (g_zoom_held[device][k].exchange(down) || !down) continue;
        if (k == 2) {
            g_zoom_reset.store(true);
        } else {
            g_zoom_steps.fetch_add(k == 0 ? -1 : 1);
        }
    }
}
std::atomic<int> g_capture_done{-1};
std::atomic<bool> g_hold{false};  // after a capture: blocked until nothing is held
std::atomic<std::int64_t> g_hold_ms{0};  // when that hold began
std::atomic<std::int64_t> g_capture_ms{0};  // when the capture began

std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int scancode_from_name(const std::string& name) {
#if defined(BBHOST_HAVE_SDL3)
    return static_cast<int>(SDL_GetScancodeFromName(name.c_str()));
#else
    (void)name;
    return 0;
#endif
}

const char* scancode_name(int sc) {
#if defined(BBHOST_HAVE_SDL3)
    const char* n = SDL_GetScancodeName(static_cast<SDL_Scancode>(sc));
    return n && *n ? n : nullptr;
#else
    (void)sc;
    return nullptr;
#endif
}

// Mouse 1..5 are the buttons; 6 and 7 the wheel turned up and down, which the
// pad read makes a short press of each notch (hle/system.cpp).
constexpr int kMouseInputs = 7;
const char* const kMouseNames[kMouseInputs + 1] = {nullptr,  "Left Button", "Right Button", "Middle Button",
                                                   "Button 4", "Button 5", "Wheel Up", "Wheel Down"};
// The same where the name has to fit in a sentence.
const char* const kMouseShort[kMouseInputs + 1] = {nullptr, "L Mouse", "R Mouse", "M Mouse", "Mouse 4", "Mouse 5", "Wheel Up", "Wheel Down"};
// Their names in [keys]: Mouse1..Mouse5, WheelUp, WheelDown.
std::string mouse_token(int m) { return m == 6 ? "WheelUp" : m == 7 ? "WheelDown" : "Mouse" + std::to_string(m); }

std::string trim(std::string v) {
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t' || v.back() == '"')) v.pop_back();
    std::size_t i = 0;
    while (i < v.size() && (v[i] == ' ' || v[i] == '\t' || v[i] == '"')) ++i;
    return v.substr(i);
}

int action_named(const char* name) {
    for (int a = 0; a < kBindCount; ++a) {
        if (std::strcmp(name, kInfo[a].key) == 0) return a;
    }
    for (const auto& al : kAliases) {
        if (std::strcmp(name, al.old_name) == 0) return al.action;
    }
    return -1;
}

// Bumped by every change, for consumers that cache what they built from a
// binding (host_bindings_serial).
std::atomic<std::uint64_t> g_serial{1};

// "E", "Mouse2", "X, Mouse2", "none": the whole binding, as a key and a mouse
// input (0 for none).
void parse_binding(int a, const std::string& text, const char* where, int* key_out, int* mouse_out) {
    int key = 0, mouse = 0;
    std::size_t pos = 0;
    while (pos <= text.size()) {
        std::size_t end = text.find(',', pos);
        if (end == std::string::npos) end = text.size();
        const std::string tok = trim(text.substr(pos, end - pos));
        pos = end + 1;
        if (tok.empty() || tok == "none") continue;
        if (tok.size() == 6 && tok.rfind("Mouse", 0) == 0 && tok[5] >= '1' && tok[5] <= '5') {
            mouse = tok[5] - '0';
            continue;
        }
        if (tok == "WheelUp" || tok == "WheelDown") {
            mouse = tok == "WheelUp" ? 6 : 7;
            continue;
        }
        const int sc = scancode_from_name(tok);
        if (!sc) {
            host_log("keys: %s: %s = \"%s\" - \"%s\" is not an SDL key name", where, kInfo[a].key, text.c_str(),
                     tok.c_str());
            continue;
        }
        key = sc;
    }
    *key_out = key;
    *mouse_out = mouse;
}

void set_from_text(int a, const std::string& text, const char* where) {
    int key = 0, mouse = 0;
    parse_binding(a, text, where, &key, &mouse);
    g_key[a].store(key, std::memory_order_relaxed);
    g_mouse[a].store(mouse, std::memory_order_relaxed);
    g_serial.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

std::uint64_t host_bindings_serial() { return g_serial.load(std::memory_order_relaxed); }

const BindingInfo& host_binding_info(int action) { return kInfo[action < 0 || action >= kBindCount ? 0 : action]; }

int host_binding_in_menus(int action) {
    switch (action) {
        case kBindConfirm:
        case kBindBack:
        case kBindMenuPageL:
        case kBindMenuPageR:
        case kBindMenuShortL:
        case kBindMenuShortR:
        case kBindMenuDisplay:
        case kBindMenuStatus:
            return 1;
        case kBindAttack:
        case kBindStrong:
        case kBindStrongMod:
        case kBindTransform:
        case kBindFirearm:
        case kBindBloodVial:
        case kBindUseItem:
            return -1;
        default:
            return 0;
    }
}

void host_bindings_load_defaults() {
    for (int a = 0; a < kBindCount; ++a) {
        g_key[a].store(kInfo[a].def_key ? scancode_from_name(kInfo[a].def_key) : 0, std::memory_order_relaxed);
        g_mouse[a].store(kInfo[a].def_mouse, std::memory_order_relaxed);
    }
    for (const auto& [name, value] : config().keys) {
        const int a = action_named(name.c_str());
        if (a < 0) {
            host_log("keys: bbhost.toml names no action \"%s\"", name.c_str());
            continue;
        }
        set_from_text(a, value, "bbhost.toml");
    }
}

bool host_bindings_parse(const char* name, const char* value) {
    const int a = action_named(name);
    if (a < 0) return false;
    set_from_text(a, value, "options");
    return true;
}

static void pad_save(std::FILE* f);  // below

void host_bindings_save(std::FILE* f) {
    std::fprintf(f, "\n[keys]\n");
    for (int a = 0; a < kBindCount; ++a) {
        std::string v;
        if (const char* n = scancode_name(g_key[a].load(std::memory_order_relaxed))) v = n;
        if (const int m = g_mouse[a].load(std::memory_order_relaxed)) {
            v += v.empty() ? "" : ", ";
            v += mouse_token(m);
        }
        std::fprintf(f, "%s = \"%s\"\n", kInfo[a].key, v.empty() ? "none" : v.c_str());
    }
    pad_save(f);
}

int host_binding_key(int action) { return action >= 0 && action < kBindCount ? g_key[action].load() : 0; }
int host_binding_mouse(int action) { return action >= 0 && action < kBindCount ? g_mouse[action].load() : 0; }

void host_binding_describe(int action, char* out, std::size_t n) {
    if (!n) return;
    const char* k = scancode_name(host_binding_key(action));
    // A lone ` is a speck in the menu's font.
    if (k && std::strcmp(k, "`") == 0) k = "` (backtick)";
    const int m = host_binding_mouse(action);
    const char* ms = m >= 1 && m <= kMouseInputs ? kMouseNames[m] : nullptr;
    if (k && ms) {
        std::snprintf(out, n, "%s  /  %s", k, ms);
    } else {
        std::snprintf(out, n, "%s", k ? k : ms ? ms : "Unbound");
    }
}

void host_binding_prompt(int action, char* out, std::size_t n) {
    if (!n) return;
    out[0] = '\0';
    const char* k = scancode_name(host_binding_key(action));
    const int m = host_binding_mouse(action);
    // The key first: a player who bound both reads the key in a sentence more
    // easily than "Left Button", and the prompt has room for one.
    const char* what = k ? k : m >= 1 && m <= kMouseInputs ? kMouseShort[m] : nullptr;
    if (what) std::snprintf(out, n, "%s", what);
}

namespace {

// Two opposite directions held at once cancel, as Dark Souls III's keyboard
// does (its digital-to-analog converter sums each axis's keys). Pressing A
// while sprinting on W+D so runs straight on until D is let go - two turns of
// 45 degrees. Letting the newer key win instead (#51's first cut) turned the
// stick 90 degrees in one frame, past the character script's 90-degree
// sprint-turn threshold, and the character spun as if told to turn round.
constexpr int kOpposites[4][2] = {
    {kBindMoveF, kBindMoveB}, {kBindMoveL, kBindMoveR}, {kBindLookU, kBindLookD}, {kBindLookL, kBindLookR}};

// Walk and Dodge / Sprint held on a mouse button, for the keyboard's movement
// keys (the mouse is read on the pad read, the keyboard on the window's update).
std::atomic<bool> g_walk_on_mouse{false}, g_roll_on_mouse{false};

// How far the movement keys push the stick while Walk is held, of a full push.
// Measured in the Hunter's Dream: under 35% the character stands, from 40% to
// 80% it walks (1.65 m/s), at a full push it runs (3.7 m/s) - so the middle of
// the walk, with room either side.
constexpr float kWalkPush = 0.6f;

// The keyboard's movement keys - whichever keys Move Forward, Back, Left and
// Right are bound to - move the stick as a thumb does rather than in one step.
// The keyboard's only (device 0, the window's thread): the mouse's buttons push
// at once, and a pad's sticks are its own and never come through here.
//  - From standing the push ramps up to full over kMoveRampMs.
//  - While Dodge / Sprint is held, a change of direction turns the stick
//    round the edge at its push, by at most 90 degrees in kMoveTurnMs (about
//    45 between two of the game's frames at 30 fps, as much as D added to W
//    always turned at once), and right round by the front (W+A to S+D goes
//    through W+D). The character script stops a sprint whose stick swings 90
//    degrees or more between two frames: W+A and W+D swapped at once ended
//    it, and W+D meeting A spun the hunter round (f9227ee5).
//  - A straight reversal - a key to its opposite with the other axis idle, A
//    to D or W to S, typed with a gap (up to kMoveRestMs) or with the two
//    cancelling between - flips the stick at once, as a thumb flicks a stick
//    across: the game's quick turn, as with a pad and v0.2.21's push. The keys
//    decide it, not where the stick is, so a reversal that comes while the
//    stick is still turning (W, D, then A at once) flips as well.
//  - Without it the stick points where the keys do at once, as it always has,
//    so letting go - the dodge - goes the way the keys point, never part of
//    the way round.
//  - Letting go of every movement key puts the stick at rest at once, but the
//    push and the direction are kept for kMoveRestMs: a change of direction is
//    typed as one key let go and the next pressed a few milliseconds later,
//    and starting from rest there took the stick through the walk for several
//    frames (v0.2.22) - a sprint turning from W to D slowed. Longer than that,
//    the next key is a start from standing.
constexpr float kMoveRampMs = 66.0f;
constexpr float kMoveTurnMs = 70.0f;
constexpr float kMoveRestMs = 150.0f;
constexpr float kQuarterTurn = 1.5707963f, kHalfTurn = 3.1415927f, kFullTurn = 6.2831853f;
float g_move_push = 0.0f;   // 0 at rest, 1 the full push
float g_move_angle = 0.0f;  // where the stick points: 0 forward, positive to the right (radians)
int g_move_x = 0, g_move_y = 0;  // the keys' last direction, kept over a gap as the push is
std::chrono::steady_clock::time_point g_move_at{}, g_move_held_at{};

// The stick for the movement keys held (x, y each -1, 0 or 1) at `now`, in a
// sprint or not: how far it is pushed (0..1) and which way (g_move_angle).
float move_stick(int x, int y, bool sprint, std::chrono::steady_clock::time_point now) {
    const float ms = std::clamp(std::chrono::duration<float, std::milli>(now - g_move_at).count(), 0.0f, kMoveTurnMs);
    g_move_at = now;
    if (x == 0 && y == 0) {
        if (std::chrono::duration<float, std::milli>(now - g_move_held_at).count() > kMoveRestMs) g_move_push = 0.0f;
        return 0.0f;
    }
    const float want = std::atan2(static_cast<float>(x), static_cast<float>(-y));
    const bool flip = (x != 0 && y == 0 && g_move_y == 0 && x == -g_move_x) ||
                      (y != 0 && x == 0 && g_move_x == 0 && y == -g_move_y);
    g_move_x = x;
    g_move_y = y;
    if (g_move_push <= 0.0f || !sprint || flip) {
        g_move_angle = want;  // from standing, not sprinting, or a straight reversal: where the keys point
    } else {
        float turn = std::remainder(want - g_move_angle, kFullTurn);
        if (std::fabs(turn) > kHalfTurn - 0.01f) {
            // Right round, both axes (W+A to S+D): by the front, whichever side that is.
            turn = std::fabs(std::remainder(g_move_angle + kQuarterTurn, kFullTurn)) <=
                           std::fabs(std::remainder(g_move_angle - kQuarterTurn, kFullTurn))
                       ? kHalfTurn
                       : -kHalfTurn;
        }
        const float step = kQuarterTurn * ms / kMoveTurnMs;
        g_move_angle =
            std::fabs(turn) <= step ? want : std::remainder(g_move_angle + std::copysign(step, turn), kFullTurn);
    }
    g_move_push = std::min(1.0f, g_move_push + std::min(ms, kMoveRampMs) / kMoveRampMs);
    g_move_held_at = now;
    return g_move_push;
}

}  // namespace

void host_bindings_keys_held(const bool* keys, int nkeys, bool held[kBindCount]) {
    for (int a = 0; a < kBindCount; ++a) {
        const int sc = g_key[a].load(std::memory_order_relaxed);
        held[a] = sc > 0 && sc < nkeys && keys[sc];
    }
    for (const auto& pair : kOpposites) {
        if (held[pair[0]] && held[pair[1]]) held[pair[0]] = held[pair[1]] = false;
    }
    held[kBindWalk] = held[kBindWalk] || g_walk_on_mouse.load(std::memory_order_relaxed);
    note_debug_key(0, held[kBindDebugMenu]);
    note_zoom_keys(0, held);
}

void host_bindings_mouse_held(std::uint32_t buttons, bool held[kBindCount]) {
    for (int a = 0; a < kBindCount; ++a) {
        const int m = g_mouse[a].load(std::memory_order_relaxed);
        held[a] = m >= 1 && m <= kMouseInputs && (buttons & (1u << (m - 1)));
    }
    g_walk_on_mouse.store(held[kBindWalk], std::memory_order_relaxed);
    g_roll_on_mouse.store(held[kBindRoll], std::memory_order_relaxed);
    note_debug_key(1, held[kBindDebugMenu]);
    note_zoom_keys(1, held);
}

void host_bindings_apply(const bool held[kBindCount], bool strong, PadState& p, int device,
                         std::chrono::steady_clock::time_point now) {
    bool any = false;
    const std::uint32_t before = p.buttons;  // what a controller holds, which a menu keeps
    for (int a = 0; a < kBindCount; ++a) {
        // The Debug Menu key and the zoom set no pad bit: their presses were
        // queued when they were read, and the pad read acts on them on the
        // game's thread (host_bind_take_debug_toggle, host_bind_take_zoom).
        if (!held[a] || a == kBindDebugMenu || a == kBindZoomIn || a == kBindZoomOut || a == kBindZoomReset) continue;
        any = true;
        std::uint32_t bit = kInfo[a].button;
        // Shift is the strong-attack modifier, as in DS3: one key rather than
        // a second binding for every attack.
        if (a == kBindAttack && strong) bit = kR2;
        if (a == kBindConfirm) bit = menu_confirm_button();
        if (a == kBindBack) bit = menu_back_button();
        const int where = host_binding_in_menus(a);
        if (where > 0) {
            // A menu's alone: the pad read adds it while a menu has the input.
            p.menu_buttons |= bit;
            continue;
        }
        p.buttons |= bit;
        if (where < 0) p.play_buttons |= bit & ~before;
        if (bit == kR2) p.r2 = 255;  // the triggers are analogue as well as buttons
        if (bit == kL2) p.l2 = 255;
        switch (a) {
            case kBindLookU: p.ry = kStickLow; break;
            case kBindLookD: p.ry = kStickHigh; break;
            case kBindLookL: p.rx = kStickLow; break;
            case kBindLookR: p.rx = kStickHigh; break;
            case kBindGestures:
            case kBindEffects:
                // One quarter in from the edge, half way down, and a stable id
                // for as long as it is held: the game tracks fingers by id, so a
                // new one every frame would read as tapping.
                if (p.touch_count < 2) {
                    PadState::Touch& t = p.touch[p.touch_count++];
                    t.x = static_cast<std::uint16_t>(a == kBindEffects ? kPadTouchW * 3 / 4 : kPadTouchW / 4);
                    t.y = static_cast<std::uint16_t>(kPadTouchH / 2);
                    t.id = 200;
                    t.down = true;
                }
                break;
            default: break;
        }
    }
    // The movement keys: a full push each way, a diagonal the square's corner
    // (as Dark Souls III's keys give) - or, with Walk held, a gentle push the
    // game reads as a walk, a diagonal's two axes shortened so that its reach,
    // and so the walk, is the same. Opposite keys cancel (keys_held above).
    // The keyboard's stick ramps, and in a sprint turns, as a thumb moves it
    // (move_stick): between the keys' own directions it runs along the
    // square's edge (the walk's circle), never inside it.
    const int x = (held[kBindMoveR] ? 1 : 0) - (held[kBindMoveL] ? 1 : 0);
    const int y = (held[kBindMoveB] ? 1 : 0) - (held[kBindMoveF] ? 1 : 0);
    if (device == 0) {
        const float push = move_stick(x, y, held[kBindRoll] || g_roll_on_mouse.load(std::memory_order_relaxed), now);
        if (push > 0.0f) {
            const float sx = std::sin(g_move_angle), sy = -std::cos(g_move_angle);
            const float edge = held[kBindWalk] ? kWalkPush : 1.0f / std::max(std::fabs(sx), std::fabs(sy));
            const float r = static_cast<float>(kStickHigh - kStickRest) * push * edge;
            // An axis the keys do not push is left as a controller has it.
            const long ix = std::lround(sx * r), iy = std::lround(sy * r);
            if (ix != 0) p.lx = static_cast<std::uint8_t>(kStickRest + ix);
            if (iy != 0) p.ly = static_cast<std::uint8_t>(kStickRest + iy);
        }
    } else if (x != 0 || y != 0) {
        float r = static_cast<float>(kStickHigh - kStickRest);
        if (held[kBindWalk]) r *= kWalkPush / (x != 0 && y != 0 ? std::sqrt(2.0f) : 1.0f);
        const int reach = static_cast<int>(std::lround(r));
        if (x != 0) p.lx = static_cast<std::uint8_t>(kStickRest + x * reach);
        if (y != 0) p.ly = static_cast<std::uint8_t>(kStickRest + y * reach);
    }
    if (any) p.connected = true;
}

void host_bindings_menu_pad(PadState& p) {
    // The triggers are analogue as well as buttons.
    if (p.play_buttons & kL2) p.l2 = 0;
    if (p.play_buttons & kR2) p.r2 = 0;
    p.buttons = (p.buttons & ~p.play_buttons) | p.menu_buttons;
    if (p.menu_buttons & kL2) p.l2 = 255;
    if (p.menu_buttons & kR2) p.r2 = 255;
}

void host_bind_capture_begin(int action) {
    if (action < 0 || action >= kBindCount) return;
    g_capture_ms.store(now_ms(), std::memory_order_relaxed);
    g_capture.store(action, std::memory_order_relaxed);
    host_log("keys: rebinding %s - press a key or a mouse button, or turn the wheel (Escape cancels, Delete unbinds)",
             kInfo[action].key);
}

int host_bind_capturing() { return g_capture.load(std::memory_order_relaxed); }

namespace {
// Whether two actions can work at the same time, so must not share a key.
bool contexts_meet(int a, int b) { return host_binding_in_menus(a) * host_binding_in_menus(b) != -1; }

void finish(int a, const char* what) {
    g_serial.fetch_add(1, std::memory_order_relaxed);
    g_capture.store(-1, std::memory_order_relaxed);
    g_hold_ms.store(now_ms(), std::memory_order_relaxed);
    g_hold.store(true, std::memory_order_relaxed);
    g_capture_done.store(a, std::memory_order_relaxed);
    char d[96];
    host_binding_describe(a, d, sizeof(d));
    host_log("keys: %s %s, now %s", kInfo[a].key, what, d);
}
}  // namespace

bool host_bind_capture_key(int scancode) {
    const int a = g_capture.load(std::memory_order_relaxed);
    if (a < 0) return false;
#if defined(BBHOST_HAVE_SDL3)
    if (scancode == SDL_SCANCODE_ESCAPE) {
        finish(a, "unchanged");
        return true;
    }
    if (scancode == SDL_SCANCODE_DELETE) {
        g_key[a].store(0, std::memory_order_relaxed);
        g_mouse[a].store(0, std::memory_order_relaxed);
        finish(a, "unbound");
        return true;
    }
#endif
    // One key, one action: whatever had it gives it up, which is what a
    // player rebinding a key means - but a menu's action and play's may share
    // one, since they never work at the same time.
    for (int b = 0; b < kBindCount; ++b) {
        if (b != a && g_key[b].load(std::memory_order_relaxed) == scancode && contexts_meet(a, b)) {
            g_key[b].store(0, std::memory_order_relaxed);
            host_log("keys: %s gives up that key", kInfo[b].key);
        }
    }
    g_key[a].store(scancode, std::memory_order_relaxed);
    finish(a, "rebound");
    return true;
}

bool host_bind_capture_mouse(int button) {
    const int a = g_capture.load(std::memory_order_relaxed);
    if (a < 0 || button < 1 || button > kMouseInputs) return false;
    // A click that starts a capture is often a double click, and its second
    // press is not a choice: it bound the left button to whatever row was
    // clicked, taking it from Attack. For the first 300 ms a button press is
    // swallowed; a deliberate one comes after the row reads "Press a key".
    if (now_ms() - g_capture_ms.load(std::memory_order_relaxed) < 300) {
        host_log("keys: a mouse press %d ms into the capture taken as part of the click that started it",
                 static_cast<int>(now_ms() - g_capture_ms.load(std::memory_order_relaxed)));
        return true;
    }
    for (int b = 0; b < kBindCount; ++b) {
        if (b != a && g_mouse[b].load(std::memory_order_relaxed) == button && contexts_meet(a, b)) {
            g_mouse[b].store(0, std::memory_order_relaxed);
            host_log("keys: %s gives up that button", kInfo[b].key);
        }
    }
    g_mouse[a].store(button, std::memory_order_relaxed);
    finish(a, "rebound");
    return true;
}

int host_bind_capture_take_done() { return g_capture_done.exchange(-1); }

bool host_bind_take_debug_toggle() { return g_debug_toggle.exchange(false); }

int host_bind_take_zoom(bool* reset) {
    *reset = g_zoom_reset.exchange(false);
    return g_zoom_steps.exchange(0);
}

void host_bind_queue_zoom(int steps, bool reset) {
    g_zoom_steps.fetch_add(steps);
    if (reset) g_zoom_reset.store(true);
}

void host_bindings_migrate_zoom() {
    // Transform's old default, the middle button: Camera Zoom Reset's now.
    if (g_mouse[kBindTransform].load() == 3 && g_mouse[kBindZoomReset].load() == 3) {
        g_mouse[kBindTransform].store(0);
        host_log("keys: the middle button is Camera Zoom Reset's now; Transform keeps its key");
    }
    // A wheel the player bound to an action stays that action's.
    for (const int z : {static_cast<int>(kBindZoomIn), static_cast<int>(kBindZoomOut)}) {
        const int m = g_mouse[z].load();
        if (m < 1 || m > kMouseInputs) continue;
        for (int a = 0; a < kBindCount; ++a) {
            if (a == z || g_mouse[a].load() != m) continue;
            g_mouse[z].store(0);
            host_log("keys: %s is %s's: %s left without it", kMouseNames[m], kInfo[a].key, kInfo[z].key);
            break;
        }
    }
    g_serial.fetch_add(1, std::memory_order_relaxed);
}

void host_bindings_migrate_menu_keys() {
    // The menu pages and options are new: an options file from before them
    // carries none, so they start on their defaults - unless the player put a
    // key there on an action that works in menus too, which keeps it.
    for (int a = kBindMenuPageL; a <= kBindMenuStatus; ++a) {
        const int k = g_key[a].load();
        if (!k) continue;
        for (int b = 0; b < kBindCount; ++b) {
            if (b == a || g_key[b].load() != k || !contexts_meet(a, b)) continue;
            g_key[a].store(0);
            host_log("keys: %s is %s's: %s left without it", scancode_name(k) ? scancode_name(k) : "?", kInfo[b].key, kInfo[a].key);
            break;
        }
    }
    g_serial.fetch_add(1, std::memory_order_relaxed);
}

bool host_bind_capture_blocking() {
    return g_capture.load(std::memory_order_relaxed) >= 0 || g_hold.load(std::memory_order_relaxed);
}

namespace {
std::atomic<std::int64_t> g_clear_ms{0};
}  // namespace

void host_bind_note_clear_key() { g_clear_ms.store(now_ms(), std::memory_order_relaxed); }

bool host_bind_take_clear_key() {
    const std::int64_t at = g_clear_ms.exchange(0);
    return at && now_ms() - at < 250;
}

void host_binding_clear(int action) {
    if (action < 0 || action >= kBindCount) return;
    g_key[action].store(0, std::memory_order_relaxed);
    g_mouse[action].store(0, std::memory_order_relaxed);
    g_serial.fetch_add(1, std::memory_order_relaxed);
    host_log("keys: %s cleared", kInfo[action].key);
}

void host_bind_capture_note_held(bool any_down) {
    // Released, or a second and a half gone: a key the window system reports
    // held with no end must not keep the pad away from the game.
    if (g_hold.load(std::memory_order_relaxed) &&
        (!any_down || now_ms() - g_hold_ms.load(std::memory_order_relaxed) > 1500)) {
        g_hold.store(false, std::memory_order_relaxed);
    }
}

void host_binding_clear_key(int action) {
    if (action < 0 || action >= kBindCount) return;
    g_key[action].store(0, std::memory_order_relaxed);
    g_serial.fetch_add(1, std::memory_order_relaxed);
    host_log("keys: %s's key cleared", kInfo[action].key);
}

void host_binding_clear_mouse(int action) {
    if (action < 0 || action >= kBindCount) return;
    g_mouse[action].store(0, std::memory_order_relaxed);
    g_serial.fetch_add(1, std::memory_order_relaxed);
    host_log("keys: %s's mouse button cleared", kInfo[action].key);
}

namespace {
// An action's default key and mouse input: the table's, unless bbhost.toml's
// [keys] names it (host_bindings_load_defaults).
void default_of(int action, int* key, int* mouse) {
    *key = kInfo[action].def_key ? scancode_from_name(kInfo[action].def_key) : 0;
    *mouse = kInfo[action].def_mouse;
    for (const auto& [name, value] : config().keys) {
        if (action_named(name.c_str()) == action) parse_binding(action, value, "bbhost.toml", key, mouse);
    }
}
}  // namespace

bool host_binding_at_default(int action) {
    if (action < 0 || action >= kBindCount) return true;
    int key = 0, mouse = 0;
    default_of(action, &key, &mouse);
    const int own = pad_input_of_bit(kInfo[action].button);
    const int pad = host_pad_input_of(action);
    return g_key[action].load() == key && g_mouse[action].load() == mouse && (pad < 0 || pad == own);
}

const char* host_key_name(int scancode) {
    const char* k = scancode_name(scancode);
    return k && std::strcmp(k, "`") == 0 ? "` (backtick)" : k;  // a lone ` is a speck
}
const char* host_mouse_name(int mouse) { return mouse >= 1 && mouse <= kMouseInputs ? kMouseNames[mouse] : nullptr; }

void host_binding_reset(int action) {
    if (action < 0 || action >= kBindCount) return;
    int key = 0, mouse = 0;
    default_of(action, &key, &mouse);
    for (int b = 0; b < kBindCount; ++b) {
        if (b == action || !contexts_meet(action, b)) continue;
        if (key && g_key[b].load(std::memory_order_relaxed) == key) g_key[b].store(0, std::memory_order_relaxed);
        if (mouse && g_mouse[b].load(std::memory_order_relaxed) == mouse) g_mouse[b].store(0, std::memory_order_relaxed);
    }
    g_key[action].store(key, std::memory_order_relaxed);
    g_mouse[action].store(mouse, std::memory_order_relaxed);
    if (const int own = pad_input_of_bit(kInfo[action].button); own >= 0 && host_pad_input_of(action) >= 0) {
        // Its button back on itself: whichever button gives it now takes the
        // one its own button gave.
        for (int p = 0; p < kPadInputs; ++p) {
            if (g_pad_gives[p].load() != own) continue;
            g_pad_gives[p].store(g_pad_gives[own].load());
            g_pad_gives[own].store(own);
            break;
        }
    }
    g_serial.fetch_add(1, std::memory_order_relaxed);
    host_log("keys: %s back to its defaults", kInfo[action].key);
}

void host_bindings_reset_all() {
    host_bindings_load_defaults();
    for (int p = 0; p < kPadInputs; ++p) g_pad_gives[p].store(p);
    g_serial.fetch_add(1, std::memory_order_relaxed);
    host_log("keys: every binding back to its default");
}

const char* host_pad_input_name(int input) { return input >= 0 && input < kPadInputs ? kPadNames[input] : nullptr; }
std::uint32_t host_pad_input_bit(int input) { return input >= 0 && input < kPadInputs ? kPadBits[input] : 0; }
std::uint32_t host_pad_play_bit(int input) {
    return input >= 0 && input < kPadInputs ? kPadBits[g_pad_gives[input].load(std::memory_order_relaxed)] : 0;
}

std::uint32_t host_pad_menu_bit(int input) {
    if (input < 0 || input >= kPadInputs) return 0;
    // Every button itself, then the OK's and the Back's moved in turn to the
    // button that gives them in play, the one that had them taking that
    // button's own: two swaps, so still a permutation.
    int gives[kPadInputs];
    for (int p = 0; p < kPadInputs; ++p) gives[p] = p;
    for (const std::uint32_t bit : {menu_confirm_button(), menu_back_button()}) {
        const int f = pad_input_of_bit(bit);
        int from = -1, to = -1;
        for (int p = 0; p < kPadInputs; ++p) {
            if (g_pad_gives[p].load(std::memory_order_relaxed) == f) from = p;
            if (gives[p] == f) to = p;
        }
        if (from >= 0 && to >= 0) std::swap(gives[from], gives[to]);
    }
    return kPadBits[gives[input]];
}

namespace {
// The button whose press an action's is: its pad bit's, for an action that
// works in play and has one (Gestures and Personal Effects share the
// touchpad's).
int pad_function_of(int action) {
    if (action < 0 || action >= kBindCount || host_binding_in_menus(action) > 0) return -1;
    return pad_input_of_bit(kInfo[action].button);
}
}  // namespace

int host_pad_input_of(int action) {
    const int f = pad_function_of(action);
    if (f < 0) return -1;
    for (int p = 0; p < kPadInputs; ++p) {
        if (g_pad_gives[p].load(std::memory_order_relaxed) == f) return p;
    }
    return -1;
}

int host_pad_menu_input(int action) {
    if (action < 0 || action >= kBindCount || host_binding_in_menus(action) <= 0) return -1;
    std::uint32_t bit = kInfo[action].button;
    if (action == kBindConfirm) bit = menu_confirm_button();
    if (action == kBindBack) bit = menu_back_button();
    for (int p = 0; p < kPadInputs; ++p) {
        if (host_pad_menu_bit(p) == bit) return p;
    }
    return -1;
}

void host_pad_capture_begin(int action) {
    if (pad_function_of(action) < 0) return;
    g_capture.store(-1, std::memory_order_relaxed);  // one capture at a time
    g_pad_capture_ms.store(now_ms(), std::memory_order_relaxed);
    g_pad_capture.store(action, std::memory_order_relaxed);
    host_log("pad: rebinding %s - press a controller button", kInfo[action].key);
}

int host_pad_capturing() { return g_pad_capture.load(std::memory_order_relaxed); }

void host_pad_capture_cancel() {
    const int a = g_pad_capture.exchange(-1);
    if (a >= 0) host_log("pad: %s unchanged", kInfo[a].key);
}

bool host_pad_capture_input(int input) {
    const int a = g_pad_capture.load(std::memory_order_relaxed);
    if (a < 0 || input < 0 || input >= kPadInputs) return false;
    if (now_ms() - g_pad_capture_ms.load(std::memory_order_relaxed) < 300) return true;  // the press that began it
    const int f = pad_function_of(a);
    for (int p = 0; p < kPadInputs; ++p) {
        if (g_pad_gives[p].load() != f) continue;
        // The pressed button gives the action's press; the one that gave it
        // takes what the pressed button gave.
        g_pad_gives[p].store(g_pad_gives[input].load());
        g_pad_gives[input].store(f);
        break;
    }
    g_pad_capture.store(-1, std::memory_order_relaxed);
    g_serial.fetch_add(1, std::memory_order_relaxed);
    host_log("pad: %s on %s", kInfo[a].key, kPadNames[input]);
    return true;
}

bool host_pad_parse(const char* name, const char* value) {
    const int a = action_named(name);
    const int f = pad_function_of(a);
    if (f < 0) return false;
    const std::string v = trim(value ? value : "");
    for (int input = 0; input < kPadInputs; ++input) {
        if (v != kPadTokens[input]) continue;
        for (int p = 0; p < kPadInputs; ++p) {
            if (g_pad_gives[p].load() != f) continue;
            g_pad_gives[p].store(g_pad_gives[input].load());
            g_pad_gives[input].store(f);
            break;
        }
        g_serial.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    host_log("pad: %s = \"%s\" - not a controller button", name, v.c_str());
    return false;
}

// [pad]: the play actions with a button of their own, each on its button,
// Gestures standing for the touchpad.
static void pad_save(std::FILE* f) {
    std::fprintf(f, "\n[pad]\n");
    for (int a = 0; a < kBindCount; ++a) {
        if (a == kBindEffects) continue;
        const int p = host_pad_input_of(a);
        if (p >= 0) std::fprintf(f, "%s = \"%s\"\n", kInfo[a].key, kPadTokens[p]);
    }
}
