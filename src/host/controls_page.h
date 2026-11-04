#pragma once

// The menu's Controls page (F10): every action with its key, its mouse button
// and its controller button side by side, each changed by clicking it and
// pressing the new one (host/bindings.h). The game's own Key Bindings screen
// stays as it was; this is the same bindings, all on one page, the controller
// too.

union SDL_Event;

// Draws the page into the menu's content area (host/ingame_menu.cpp).
void controls_page_draw();
// While the page is choosing a binding, the event pump gives it every event
// first: the next key, mouse button, wheel notch or controller button is the
// choice and reaches nothing else. True when it took the event.
bool controls_capture_event(const SDL_Event& e);
// The menu closed: a choice the page began ends unchanged.
void controls_menu_closed();
