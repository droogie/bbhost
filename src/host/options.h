#pragma once

// The PC port's own settings (F10): a menu over the game drawn with the F9
// menu's frame (host/ingame_menu.h), in the setup window's look.
//
// These settings configure *bbhost*, not the game: resolution, pacing, what
// the pointer does. The PS4 build has nowhere to put them and the dump is
// never written, so the host draws the menu and owns the file it saves to.
//
// The menu is modal: while it is open the game sees no pad and no pointer.

#include <cstdint>
#include <functional>
#include <string>

// Reads the saved settings and applies the ones that take effect at startup.
// Safe to call before the window exists.
void host_options_load();
// Re-applies every setting. Called once the window exists, since the display
// ones could only be recorded before that.
void host_options_apply();

bool host_options_open();
// The window's thread (F10, host/ingame_menu.cpp). Opening drains the latched
// pointer edges, so a click from before it opened reaches nothing. Closing saves.
void host_options_set_open(bool open);
// The same close from the presenter (the menu's Close, Escape): saved at the
// window's next pump.
void host_options_close();

// F10's settings as pages of the in-game menu (host/ingame_menu.cpp): drawn on
// the presenter's thread into the current ImGui window, in the setup window's
// look. A change is handed to the window's thread, which applies it at its
// next pump (host_options_frame); the file is written when the menu closes.
int host_options_page_count();
const char* host_options_page_name(int page);
const char* host_options_page_blurb(int page);
void host_options_draw_page(int page);
// Runs `fn` on the window's thread at its next pump: the menus' changes to
// settings and to bbhost.toml, which can touch the window or write files.
void host_options_post(std::function<void()> fn);

// One pump of the window's thread: runs what the menus posted, and answers
// the resolution's keep-or-revert question.
void host_options_frame(float display_w, float display_h);

// The live settings are one object, host/settings.h (host_settings()), with
// typed fields; the calls below are the front ends' way in, by name.
// Read one by the same name host_opt_set() takes. Unknown names read false.
bool host_opt_get(const char* key);
// The render resolution the player chose, or what BBHOST_RES says. False when
// neither parses. Read once by the loader - it is a patch on the image.
bool host_opt_resolution(int* w, int* h);
int host_opt_resolution_index();
void host_opt_set_resolution_index(int i);
// The game's frame rate (engine/frame_rate.h): 30, 60, 90, 120, 150, 180,
// 240, 360, 480, 960, or 0 for Uncapped (the display's refresh rate); the
// file's key is still frame_cap. Same setting, same file.
int host_opt_frame_cap();
void host_opt_set_frame_cap(int fps);

// Set one by name, from the rows the port adds to the game's own System menu
// (engine/option_menu.h). The two front ends drive the same settings and the
// same file, so a change made in game is the change the F10 screen shows and
// bbhost-options.toml keeps. Unknown names are ignored.
void host_opt_set(const char* key, bool on);
// A setting as the index of its value (mouse_sens: 0..10), and setting it;
// -1 for an unknown name. Saved at once, like host_opt_set.
int host_opt_index(const char* key);
// Menu text size (menu_text_size) as a factor: 1.0 at 100%, 4.0 at 400%.
float host_opt_menu_text_scale();
void host_opt_set_index(const char* key, int index);
// Writes the options file now: the key bindings changed (host/bindings.h).
void host_options_save_now();

// A Steam Deck - Valve's Jupiter or Galileo, from /sys/devices/virtual/dmi/id
// on Linux and the BIOS registry key on Windows - where the System menu has a
// Steam Deck section (engine/option_menu.cpp). BBHOST_STEAM_DECK=1 or 0 says
// so instead.
bool host_steam_deck();
// Its two configurations as one choice: 0 is 30 fps at the screen's own
// 1280x800, 1 is 60 fps rendering at 960x600 (upscaled by FSR 1) with Model
// detail Low, which holds the heaviest views at 60. Read from the frame cap:
// 60 or more, or none, is 1. Setting one
// sets the frame cap, the resolution and the model detail together - the
// resolution without the keep-or-revert question, both sizes being known to
// fit - and saves. The frame rate itself changes on the next start, as the
// frame cap's does.
int host_deck_profile();
int host_deck_default_profile();  // the same, read from the frame cap's default
void host_set_deck_profile(int profile);

// Changes whenever any setting does: something that applies a setting live
// reads it again only when this has moved.
std::uint64_t host_opt_serial();

// What the Shaders setting's HLSL and Lifted mean, in a player's words: the
// setup window's bubbles and F10's.
extern const char* const kHostShaderModeTips[2];

// What each setting ships with - the table's own values, whatever the files
// say - for the game's Defaults row (engine/option_menu.cpp). Read the way
// host_opt_get / host_opt_index / host_opt_frame_cap read the current value.
bool host_opt_default_get(const char* key);
int host_opt_default_index(const char* key);
int host_opt_default_frame_cap();
int host_opt_default_resolution_index();

// The F10 screen's ACCOUNT actions, for the setup window (host/launcher.cpp):
// 0 link with Discord (a code approved on the website, opened in the
// browser), 1 create an account held by this PC (name), 2 recover one (name,
// recovery code), 3 a website sign-in code, 4 sign out. Each runs on a thread
// of its own; the lines say how it went. A sign-in or sign-out asks for the
// options file, which holds the token, to be saved: host_account_take_save.
void host_account_action(int which, const std::string& name, const std::string& code);
bool host_account_busy();
std::string host_account_signed_in();  // "Signed in as NAME" or "Not signed in"
std::string host_account_outcome();    // the last action's outcome, "" before one
std::string host_account_detail();     // a code the player needs (link, recovery, website), or ""
bool host_account_take_save();         // true once after a sign-in or sign-out
