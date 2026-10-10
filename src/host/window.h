#pragma once

// Host shell: SDL3 window, event pump, input state, and a Vulkan presenter.
// Everything here is optional; when the shell is not started (headless) the
// HLE keeps working without a display.

#include <cstdint>
#include <string>

#include "host/text_entry.h"

// The DualShock 4 touchpad surface, in its own pixels. This is what
// scePadGetControllerInformation reports, so the fingers below are in the same
// space the game is told to expect.
constexpr int kPadTouchW = 1920, kPadTouchH = 943;

// A stick at rest, as the game reads it: 127 is its zero - what a centred
// controller reads through our axis conversion too. 128 is a step off it, and
// the game reads that step as a push whenever the stick's other axis is held:
// its analog shaping (DLUI's, which Dark Souls III shares) zeroes a small axis
// only when its pair is small as well, then pushes what is left outward. With
// 128 the keyboard's W, with no controller connected, read as forward and a
// little right, and every turn with A or D cut a sprint short.
constexpr std::uint8_t kStickRest = 127;
// A key's full push, the same either way from the rest: 0 and 254. 255 is a
// step past full on the right and down sides only (0 is 127 below the rest,
// 255 is 128 above it), and pressing or letting go of D while sprinting
// dropped the sprint for 8 frames where A did not.
constexpr std::uint8_t kStickLow = kStickRest - 127, kStickHigh = kStickRest + 127;

struct PadState {
    std::uint32_t buttons = 0;   // ScePadButtonDataOffset bits
    std::uint8_t lx = kStickRest, ly = kStickRest, rx = kStickRest, ry = kStickRest;
    std::uint8_t l2 = 0, r2 = 0;
    bool connected = false;
    std::uint64_t timestamp = 0;
    // The pad reports up to two fingers. `id` counts up as fingers arrive, the
    // way the PS4's does, so a game can follow one across frames.
    struct Touch {
        std::uint16_t x = 0, y = 0;
        std::uint8_t id = 0;
        bool down = false;
    };
    Touch touch[2];
    std::uint8_t touch_count = 0;
    // The keys that always work a menu, whatever the bindings say: the arrows,
    // Enter, Backspace and Escape as the d-pad, Circle, Cross and Options. The
    // pad read adds these only while a menu is open, so rebinding what an
    // arrow does in play can never leave the menus without a way to move.
    std::uint32_t menu_buttons = 0;
};

// Creates the window (and Vulkan presenter when possible). Returns false when
// no display or SDL is unavailable; the caller then runs headless.
bool host_window_start(int width, int height, const char* title);
// Pump events once. Returns false when the user asked to quit.
bool host_window_pump();
// Ends the run as closing the window does (the title menu's Quit Game). Any thread.
void host_window_request_quit();
void host_window_stop();
// An error the player has to read before bbhost ends, in a box of its own (not
// modal to the game's window, whose thread may have stopped pumping). Returns
// when it is closed; at once without SDL.
void host_window_error_box(const char* title, const char* text);
bool host_window_active();

// Snapshot of controller/keyboard state for scePad.
PadState host_pad_state();

// The pointer, for menus. Bloodborne has no mouse
// device, so this is a channel of its own rather than part of PadState: a pad
// cannot carry a position.
struct MouseState {
    // Position in the presented picture's pixels, after reconstruction. The
    // window/swapchain letterbox is removed. Both 0 before the first present.
    float x = 0.0f, y = 0.0f;
    bool in_window = false;
    bool moved = false;             // moved since the previous read
    std::uint32_t buttons = 0;      // held now: bit 0 left, 1 right, 2 middle
    // Edges since the previous read, *latched*: a click that both arrives and
    // ends between two reads is still reported, which polling `buttons` alone
    // would lose.
    std::uint32_t pressed = 0, released = 0;
    float wheel = 0.0f;             // notches since the previous read, up positive
    // Relative motion since the previous read, in window pixels. This is what
    // a camera wants and the absolute position is not: in camera mode the
    // pointer is locked, so x/y stop moving while dx/dy keep arriving. DS3
    // gets the same thing by SetCursorPos-ing back to the middle of the client
    // rect every frame (CSMouseMan); SDL's relative mode is that without the
    // round trip through the desktop.
    float dx = 0.0f, dy = 0.0f;
};
// Reads and clears the latched edges, the wheel and `moved`. One consumer.
MouseState host_mouse_state();
// The position alone, without consuming anything - for drawing the pointer,
// which must not eat the edges its real consumer needs. False when the pointer
// is outside the window or nothing has been presented yet.
bool host_mouse_position(float& x, float& y);
// The pointer in the **menu movie's stage** pixels, which is the space the
// menu's row geometry is in. This is the one place
// that conversion lives, so a resolution change edits it here rather than
// everywhere a hit test is written. False when there is no pointer.
bool host_mouse_stage_position(float& x, float& y);

// Camera mode: the pointer is locked to the window and hidden, and only
// relative motion is reported. Off in menus, on in play - the caller decides,
// because nothing in the host can yet tell one from the other.
void host_mouse_set_relative(bool on);
bool host_mouse_relative();

// The strong-attack modifier (Shift by default) - see the key table.
bool host_key_strong();
// Whether the host draws its own pointer (the game has no cursor asset), which
// also hides the window system's.
void host_mouse_set_visible(bool visible);

// A line of text over the game for `seconds` (plugins' show_message); a new
// one replaces the last. Any thread.
void host_message_show(const char* text, float seconds);
// Text entry for the game's IME dialog: while active the window collects
// typed text in a box over the game, and the box owns the input - the game
// sees a connected pad with nothing pressed and no pointer, as a PS4 game
// does while the system's keyboard is up. Enter (or a controller's confirm)
// accepts, Escape (or its back) cancels. Safe to call from any thread.
// hidden: a star per character (a password).
// label: what the box over the game asks for (the game's own dialog title,
// "Enter Chalice Glyph"); "Enter name" without one. Ctrl+V (or Shift+Insert)
// pastes the clipboard.
// charset: what the box takes (host/text_entry.h); typed characters outside
// it are left out.
// select_initial: the starting text is a suggestion that the first key typed
// replaces (host/text_entry.h).
void host_text_entry_begin(const char* initial_utf8, unsigned max_chars, bool hidden = false, const char* label = nullptr,
                           TextCharset charset = TextCharset::Any, bool select_initial = false);
// 0 = editing, 1 = accepted, 2 = cancelled (or no window); fills `text`.
int host_text_entry_poll(std::string& text_utf8);
void host_text_entry_end();
// Whether a text entry is open (begun and not yet ended), whoever opened it.
bool host_text_entry_open();
void host_pad_rumble(std::uint8_t small, std::uint8_t large);
// A gamepad is open. What the port shows for a button prompt follows it when
// the setting is on Auto (engine/key_prompts.h).
bool host_pad_present();
// The device the player last used (DS3's switch): the latest key, click,
// wheel or mouse movement makes it the keyboard and mouse, the latest pad
// button or stick or trigger past its dead zone a controller. The serial
// counts the switches, for anything that must redo what it showed.
enum class InputDevice : int { KeyboardMouse = 0, Pad = 1 };
InputDevice host_input_device();
std::uint64_t host_input_serial();
// The pointer's last window position, the window's size and the display-buffer
// position they map to, as one line for a log.
void host_mouse_describe(char* out, std::size_t n);

// Live display settings, driven by the options screen (host/options.h). Both
// are safe before the window exists: they record the wish and the presenter
// picks it up when it comes up. V-Sync off rebuilds the swapchain on the next
// present with whichever no-wait mode the surface supports.
void host_window_set_fullscreen(bool on);
void host_window_set_vsync(bool on);
// The presented image's size in pixels (the swapchain: the window, or the
// display in fullscreen); false before there is one (headless). Any thread.
bool host_window_pixels(int* w, int* h);
// The refresh rate of the display the window is on, in Hz, or 0
// when there is no window (headless) or the display does not say. The flip
// clock runs at this rate instead of assuming 59.94 Hz.
float host_window_refresh_hz();
// The steady-clock time, in nanoseconds, at which the presenting thread last
// finished a present, and a counter of finished presents. With V-Sync on the
// present returns on the display's own vblank, which is the phase the flip
// clock steers towards. `serial` is 0 until the first present.
std::uint64_t host_present_last_ns(std::uint64_t* serial);

// What the presenter is told of a flip besides its buffer (hle/video.cpp).
struct PresentFlip {
    std::uint64_t submit_need = ~0ull;  // host_gpu_work_needs() when the flip arrived; ~0: not known, submit
    std::uint64_t arrived_ns = 0;       // steady clock when the flip was queued (for the flip-to-present time); 0: not known
    bool on_arrival = false;            // shown when it arrived, before its vblank completes it (BBHOST_PRESENT_ON_ARRIVAL)
    std::uint64_t temporal_frame = 0;   // captured DLAA / FG frame index under GPU lock
};
// Present one frame: copies the render target the game displays (`display_va`
// from sceVideoOutRegisterBuffers) into the swapchain, or clears when there
// is none yet. Safe to call from any thread: it hands the frame to the
// presenting thread, which keeps only the newest.
void host_present(int buffer_index, std::uint64_t display_va, unsigned display_w, unsigned display_h, const PresentFlip& flip = {});
