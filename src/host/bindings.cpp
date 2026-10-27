#include "host/bindings.h"

#include "core/config.h"
#include "engine/debug_menu.h"
#include "engine/menu_pointer.h"
#include "host/window.h"
#include "log.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>

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
    {"move_forward", "前に移動", "左スティックの上方向。", 0, "W", 0},
    {"move_back", "後ろに移動", "左スティックの下方向。", 0, "S", 0},
    {"move_left", "左に移動", "左スティックの左方向。", 0, "A", 0},
    {"move_right", "右に移動", "左スティックの右方向。", 0, "D", 0},
    // Held with the movement keys, as Dark Souls III's Left Alt: a walk.
    {"walk", "歩く", "移動キーと同時に押すと歩きます。", 0, "Left Alt", 0},
    {"roll", "Dodge / Sprint", "日本版では×ボタン。短く押すと回避、長押しでダッシュします。", kCircle, "Space", 0},
    {"lock_on", "ロックオン", "R3。敵をロックオン、またはカメラを正面に戻します。", kR3, "Q", 0},

    {"attack", "攻撃", "R1。強攻撃の修飾キーと同時に押すと R2。", kR1, nullptr, 1},
    {"strong_attack", "強攻撃", "R2。長押しで溜め攻撃。", kR2, "V", 0},
    {"strong_modifier", "強攻撃の修飾キー", "攻撃と同時に押すと強攻撃 (R2) になります。", 0, "Left Shift", 0},
    {"transform", "武器の変形", "L1。仕掛け武器を変形します。", kL1, "C", 3},
    {"firearm", "銃撃 / パリィ", "L2。左手武器を使用します。", kL2, "X", 2},
    {"blood_vial", "輸血液", "△ボタン。輸血液で HP を回復します。", kTriangle, "R", 0},
    {"use_item", "アイテム使用", "□ボタン。選択中のアイテムを使用します。", kSquare, "F", 0},

    {"interact", "Interact", "日本版では○ボタン。話す・調べる・拾う・開ける。", kCross, "E", 0},
    {"switch_item", "アイテム切替", "方向キーの下。使用アイテムを切り替えます。", kDown, "Down", 0},
    {"switch_right", "右手武器切替", "方向キーの右。", kRight, "Right", 0},
    {"switch_left", "左手武器切替", "方向キーの左。", kLeft, "Left", 0},
    {"dpad_up", "方向キーの上", "方向キーの上。", kUp, "Up", 0},
    // The touchpad is a button with a side: the game's key guide reads
    // "<KG_TP_L> : Gestures" and "<KG_TP_R> : Personal Effects". A press alone
    // cannot say which, so these place a finger as well (apply below).
    {"gestures", "ジェスチャー", "タッチパッドの左側。", kTouchPad, "G", 0},
    {"personal_effects", "携行品", "タッチパッドの右側。", kTouchPad, "T", 0},

    {"look_up", "上を見る", "右スティックの上。マウスを使わないカメラ操作。", 0, "I", 0},
    {"look_down", "下を見る", "右スティックの下。", 0, "K", 0},
    {"look_left", "左を見る", "右スティックの左。", 0, "J", 0},
    {"look_right", "右を見る", "右スティックの右。", 0, "L", 0},
    {"menu", "メニュー", "OPTIONS。メニューを開きます。", kOptions, "Escape", 0},
    // Menus need an OK and a Back that are not the gameplay keys sharing their
    // pad bits: nobody reaches for Space to accept or E to back out.
    // Their buttons are the region's (engine/menu_pointer.h), set in apply.
    {"confirm", "メニューの決定", "メニュー内で決定します。", 0, "Return", 0},
    {"back", "メニューで戻る", "メニュー内で戻ります。", 0, "Backspace", 0},

    {"l3", "左スティック押し込み", "L3.", kL3, "Left Ctrl", 0},
    // The developers' debug menu: not a pad press at all. The patch opens it
    // on the touchpad's left side, which is also Gestures; this key toggles
    // the menu itself (engine/debug_menu.h), when the Debug Menu plugin is
    // on. The Key Bindings screen lists it only then.
    {"debug_menu", "デバッグメニュー", "開発用デバッグメニューを開閉します（プラグインが必要）。", 0, "`", 0},
};

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

// "E", "Mouse2", "X, Mouse2", "none": the whole binding.
void set_from_text(int a, const std::string& text, const char* where) {
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
    g_key[a].store(key, std::memory_order_relaxed);
    g_mouse[a].store(mouse, std::memory_order_relaxed);
    g_serial.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

std::uint64_t host_bindings_serial() { return g_serial.load(std::memory_order_relaxed); }

const BindingInfo& host_binding_info(int action) { return kInfo[action < 0 || action >= kBindCount ? 0 : action]; }

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
        std::snprintf(out, n, "%s", k ? k : ms ? ms : "未割り当て");
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

// Walk held on a mouse button, for the keyboard's movement keys (the mouse is
// read on the pad read, the keyboard on the window's update).
std::atomic<bool> g_walk_on_mouse{false};

// How far the movement keys push the stick while Walk is held, of a full push.
// Measured in the Hunter's Dream: under 35% the character stands, from 40% to
// 80% it walks (1.65 m/s), at a full push it runs (3.7 m/s) - so the middle of
// the walk, with room either side.
constexpr float kWalkPush = 0.6f;

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
}

void host_bindings_mouse_held(std::uint32_t buttons, bool held[kBindCount]) {
    for (int a = 0; a < kBindCount; ++a) {
        const int m = g_mouse[a].load(std::memory_order_relaxed);
        held[a] = m >= 1 && m <= kMouseInputs && (buttons & (1u << (m - 1)));
    }
    g_walk_on_mouse.store(held[kBindWalk], std::memory_order_relaxed);
    note_debug_key(1, held[kBindDebugMenu]);
}

void host_bindings_apply(const bool held[kBindCount], bool strong, PadState& p) {
    bool any = false;
    for (int a = 0; a < kBindCount; ++a) {
        // The Debug Menu key sets no pad bit: its press was queued when it was
        // read, and the pad read toggles the menu on the game's thread
        // (host_bind_take_debug_toggle).
        if (!held[a] || a == kBindDebugMenu) continue;
        any = true;
        std::uint32_t bit = kInfo[a].button;
        // Shift is the strong-attack modifier, as in DS3: one key rather than
        // a second binding for every attack.
        if (a == kBindAttack && strong) bit = kR2;
        if (a == kBindConfirm) bit = menu_confirm_button();
        if (a == kBindBack) bit = menu_back_button();
        p.buttons |= bit;
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
    const int x = (held[kBindMoveR] ? 1 : 0) - (held[kBindMoveL] ? 1 : 0);
    const int y = (held[kBindMoveB] ? 1 : 0) - (held[kBindMoveF] ? 1 : 0);
    if (x != 0 || y != 0) {
        int reach = kStickHigh - kStickRest;
        if (held[kBindWalk]) {
            const float r = static_cast<float>(reach) * kWalkPush / (x != 0 && y != 0 ? std::sqrt(2.0f) : 1.0f);
            reach = static_cast<int>(std::lround(r));
        }
        if (x != 0) p.lx = static_cast<std::uint8_t>(kStickRest + x * reach);
        if (y != 0) p.ly = static_cast<std::uint8_t>(kStickRest + y * reach);
    }
    if (any) p.connected = true;
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
    // player rebinding a key means.
    for (int b = 0; b < kBindCount; ++b) {
        if (b != a && g_key[b].load(std::memory_order_relaxed) == scancode) {
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
        if (b != a && g_mouse[b].load(std::memory_order_relaxed) == button) {
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

const char* host_binding_label(const char* label) {
    if (!std::strcmp(label, "Dodge / Sprint")) return "回避 / ダッシュ";
    if (!std::strcmp(label, "Interact")) return "調べる";
    return label;
}
