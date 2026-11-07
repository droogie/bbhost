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

#include <chrono>
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
    // Menu pages and options: the pad buttons a menu reads for its own pages
    // and commands, which in play are the shoulders' and two face buttons'
    // actions (host_binding_in_menus)
    kBindMenuPageL, kBindMenuPageR, kBindMenuShortL, kBindMenuShortR, kBindMenuDisplay, kBindMenuStatus,
    // Other
    kBindL3, kBindZoomIn, kBindZoomOut, kBindZoomReset, kBindDebugMenu,
    kBindCount,
};
constexpr int kBindPages = 6, kBindPerPage = 7;
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
// Where an action works, as Dark Souls III keeps a menu's key assignment apart
// from play's (SprjMenuViewerKeyAssign beside SprjKeyAssign): 1 in menus only
// (Menu Confirm and Back, the menu pages and options), -1 in play only (the
// actions whose pad buttons a menu reads as something else: the attacks, the
// firearm, Transform, the Blood Vial and the quick item), 0 both (moving,
// which a menu reads as its directions, Interact and Dodge, its confirm and
// back, the rest). Two actions may share a key only when they never work at
// the same time.
int host_binding_in_menus(int action);

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
// is holding it. A menu action's button goes to `menu_buttons`, which the pad
// read adds only while a menu has the input; a play-only action's button goes
// to `buttons` and is noted in `play_buttons` as well, which the pad read takes
// out again while a menu has it (host/window.h PadState). `device` is 0 for the keyboard, 1 for the mouse: the
// keyboard's movement keys ramp up to a full push over 66 ms from standing
// still (host/bindings.cpp), on `now`'s clock (tests/key_move_test.cpp runs
// one of its own).
void host_bindings_apply(const bool held[kBindCount], bool strong, PadState& p, int device,
                         std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());
// The pad as a menu reads it, while one has the input: the play-only actions'
// buttons the keyboard put there taken out, the menu's own keys put in.
void host_bindings_menu_pad(PadState& p);
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
// The camera zoom's presses since the last call: Zoom Out's less Zoom In's,
// each a step of the Camera distance setting, and whether Zoom Reset was
// pressed (engine/camera.h camera_zoom).
int host_bind_take_zoom(bool* reset);
// The same presses from a test request (BBHOST_TEST_REQUESTS "zoom in|out|reset").
void host_bind_queue_zoom(int steps, bool reset);
// An options file from before the zoom actions (version 3 or older): the
// middle button they take by default was Transform's, which gives it up; an
// action the player put on the wheel keeps it, and the zoom goes without.
void host_bindings_migrate_zoom();
// An options file from before the menu pages and options (version 4 or
// older): they start on their default keys, but a key the player gave an
// action that works in menus too stays that action's, and the menu's goes
// without.
void host_bindings_migrate_menu_keys();
bool host_bind_take_clear_key();
void host_binding_clear(int action);
// One side of a binding cleared (the Controls page: a right-click on its key
// or its mouse button), and an action's key, mouse button and controller
// button back to their defaults - another action giving up a key or button it
// took meanwhile, as a capture makes it.
void host_binding_clear_key(int action);
void host_binding_clear_mouse(int action);
void host_binding_reset(int action);
// Whether an action's key, mouse button and controller button are all its
// defaults (its reset would change nothing).
bool host_binding_at_default(int action);
// A key's name as the screens show it ("E", "Left Shift", "` (backtick)"),
// and a mouse input's ("Right Button", "Wheel Up"); nullptr for none.
const char* host_key_name(int scancode);
const char* host_mouse_name(int mouse);
// Every binding back to its default: the keys and mouse buttons bbhost.toml's
// [keys] gives, and the controller as it is.
void host_bindings_reset_all();

// The controller, rebindable as the keys are (the Controls page, F10): each
// of a pad's sixteen buttons - the face buttons, the shoulders and triggers,
// the sticks' presses, Options, the touchpad and the d-pad - gives the press
// of the action it is bound to. In play only: in a menu every button is
// itself, so Cross and Circle answer and go back as the region has them. Every
// action keeps a button: a rebind swaps, the action whose button was taken
// getting the one the rebound action had. Saved in the options file's [pad].
enum PadInput : int {
    kPadCross, kPadCircle, kPadSquare, kPadTriangle, kPadL1, kPadR1, kPadL2, kPadR2,
    kPadL3, kPadR3, kPadOptions, kPadTouchpad, kPadUp, kPadDown, kPadLeft, kPadRight,
    kPadInputs,
};
const char* host_pad_input_name(int input);   // "Cross", "R2", "D-pad Up"; nullptr past the end
std::uint32_t host_pad_input_bit(int input);  // the pad bit the button is, as a menu reads it
std::uint32_t host_pad_play_bit(int input);   // the pad bit it gives in play
// The button an action is on in play, or -1 for one with no pad button of its
// own to move: the movement and look directions, the modifiers, the zoom, and
// the menus' own (host_pad_menu_input).
int host_pad_input_of(int action);
// A menu-only action's button, which does not move; -1 for the rest.
int host_pad_menu_input(int action);
// Choosing: the next button pressed or trigger pulled after begin becomes the
// action's (a press in the first 300 ms is the one that started it, not a
// choice). Cancel ends it unchanged.
void host_pad_capture_begin(int action);
int host_pad_capturing();  // the action, or -1
void host_pad_capture_cancel();
// From the event pump. True when the press was the capture's.
bool host_pad_capture_input(int input);
// [pad] lines, `roll = "Cross"`: an action and the button it is on.
bool host_pad_parse(const char* name, const char* value);
