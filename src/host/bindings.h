#pragma once

// The PC port's key bindings, the way Dark Souls III PC keeps them: every
// action has a keyboard key **and** a mouse button, either may be unbound, and
// a binding is changed by choosing the action and pressing what it should be.
//
// Actions are named by what they do in Bloodborne rather than by pad button,
// because that is what a player rebinds, and each one sets the pad bit (or
// stick direction) the game reads for it. The order below is the order the
// Key Bindings screen lists them in: pages of seven
// (engine/option_menu.cpp), the last one short.
//
// Sources, weakest first: the defaults here, `bbhost.toml`'s [keys], then the
// [keys] section of `bbhost-options.toml`, which is what the in-game screen
// writes. A value names the whole binding: "E", "Mouse2", "X, Mouse2", or
// "none". Mouse1..Mouse5 are the left, right and middle buttons and the two
// side buttons; WheelUp and WheelDown the wheel, each notch a short press.

#include <cstddef>
#include <cstdint>
#include <cstdio>

struct PadState;

enum BindAction : int {
    // Movement
    kBindMoveF, kBindMoveB, kBindMoveL, kBindMoveR, kBindWalk, kBindRoll, kBindLockOn,
    // Combat
    kBindAttack, kBindStrong, kBindStrongMod, kBindTransform, kBindFirearm, kBindBloodVial, kBindUseItem,
    // Items and gestures
    kBindInteract, kBindSwitchItem, kBindSwitchRight, kBindSwitchLeft, kBindDpadUp, kBindGestures, kBindEffects,
    // Camera and menus
    kBindLookU, kBindLookD, kBindLookL, kBindLookR, kBindMenu, kBindConfirm, kBindBack,
    // Other
    kBindL3, kBindDebugMenu,
    kBindCount,
};
constexpr int kBindPages = 5, kBindPerPage = 7;
static_assert(kBindPages * kBindPerPage >= kBindCount && (kBindPages - 1) * kBindPerPage < kBindCount,
              "every action on a page, and no page empty");

struct BindingInfo {
    const char* key;       // the name in [keys]
    const char* label;     // the row's caption
    const char* help;      // the row's line help
    std::uint32_t button;  // the pad bit it sets; 0 for stick directions and the modifier
    const char* def_key;   // SDL key name, or nullptr
    int def_mouse;         // 1..5, or 0
};
const BindingInfo& host_binding_info(int action);

// Called by host_options_load(): the defaults, then bbhost.toml's [keys].
void host_bindings_load_defaults();
// One `name = value` line from the options file's [keys]. False when the name
// is not an action.
bool host_bindings_parse(const char* name, const char* value);
// Writes the [keys] section.
void host_bindings_save(std::FILE* f);

// Bumped whenever any binding changes, so what a consumer built from them
// (engine/key_prompts.h caches rewritten message text) knows to build again.
std::uint64_t host_bindings_serial();

int host_binding_key(int action);    // an SDL scancode, 0 when unbound
int host_binding_mouse(int action);  // 1..5 the buttons, 6 and 7 the wheel up and down; 0 when unbound
// What the screen shows: "E", "X  /  Right Button", "Unbound". UTF-8.
void host_binding_describe(int action, char* out, std::size_t n);
// What a button prompt in the game's own text shows, which has to be short
// because it sits inside a sentence: the bound key, or the mouse button as
// "L Mouse", and nothing at all when the action has neither
// (engine/key_prompts.h). UTF-8.
void host_binding_prompt(int action, char* out, std::size_t n);

// Applies a set of held actions to the pad the game reads. `strong` is the
// modifier's state, which turns Attack into the strong attack whichever device
// is holding it.
void host_bindings_apply(const bool held[kBindCount], bool strong, PadState& p);
// Actions held on the keyboard, from SDL's key state.
void host_bindings_keys_held(const bool* keys, int nkeys, bool held[kBindCount]);
// Actions held on the mouse, from MouseState::buttons (bit n-1 is button n;
// bits 5 and 6 the wheel's notches, as the pad read holds them).
void host_bindings_mouse_held(std::uint32_t buttons, bool held[kBindCount]);

// Rebinding, DS3's way: the screen starts a capture for an action, the next
// key or mouse button pressed becomes its binding (Escape cancels, Delete
// unbinds it), and the game sees an idle pad from the start of the capture
// until that key is released again.
void host_bind_capture_begin(int action);
int host_bind_capturing();  // the action being captured, or -1
// From the event pump. True when the event was the capture's.
bool host_bind_capture_key(int scancode);
bool host_bind_capture_mouse(int button);  // 1..5, or 6 and 7 for the wheel up and down
// The action whose capture ended since the last call, or -1. Its binding, and
// any it took a key or button from, changed.
int host_bind_capture_take_done();
bool host_bind_capture_blocking();
// From the pad update: whether any key or mouse button is still down, which is
// what ends the hold after a capture.
void host_bind_capture_note_held(bool any_down);

// Clearing, without a capture: Delete on a selected action unbinds its key
// and its mouse button both, so an action that has two can be brought back to
// one by clearing it and binding the one it should keep. The event pump notes
// the key; the Key Bindings screen takes it if an action row is selected
// (a press nobody takes within a quarter second is dropped).
void host_bind_note_clear_key();
// The Debug Menu key was pressed since the last call (engine/debug_menu.h).
bool host_bind_take_debug_toggle();
bool host_bind_take_clear_key();
void host_binding_clear(int action);

const char* host_binding_label(const char* label);
