#pragma once

// The mouse in the game's menus, done the way Dark Souls III does it.
//
// Every navigable list in Bloodborne's menus - the title menu, the System
// list, the option sections, the pause grid, the inventory, the message box
// buttons - is one component class (vtable 0x5799fb0) whose update,
// `sub_1ecff30`, reads the pad and moves the cursor with `sub_1ecf6b0`. That
// update is DS3's `sub_140b23520` without the mouse. The port adds the mouse
// back at the same place, with DS3's rules:
//
//   - hover only when the pointer moved and no direction is held;
//   - hit-test each visible item's `Cursor` clip on its **live** display
//     object - world matrix, inverse, local bounds - so no geometry is
//     tabulated and every screen works without the port knowing it exists;
//   - on a hit, move the cursor with the game's own `sub_1ecf6b0`, then do
//     what the update does after a pad move: call the component's listener
//     and post the cursor-move sound;
//   - a click is a decide **only** when it lands on an item.

#include <cstdint>

struct ElfImage;

// Hooks the list update. The caller has already checked the eboot is 1.09.
void menu_pointer_install(ElfImage* image);

// A left click while the pointer owns the menus. It does nothing on its own:
// the next list update that finds an item under the pointer claims it.
void menu_pointer_click();

// A click that landed on an item, once, as the ScePad button it means: decide
// (Circle), or Left / Right when it hit one side of an option row's value.
// 0 when there is nothing to press.
std::uint32_t menu_pointer_take_press();

// The pad buttons that decide and return in the menus: Cross and Circle, or
// the other way round in the game's Japanese region - and in the developers'
// debug menu while it is open, which decides with Circle in every region.
// Read from the game, so the mouse, Enter and Backspace always press the ones
// it means.
std::uint32_t menu_confirm_button();
std::uint32_t menu_back_button();
// The region's own decide button, without the debug menu's override: what a
// button prompt names, since a prompt must not change while a menu is open
// (engine/key_prompts.h).
std::uint32_t menu_region_confirm_button();

// Counts list updates that had input - a menu with focus.
std::uint64_t menu_pointer_tick();

// True while the title/front-end flow is active and the game world has not started.
bool menu_pointer_front_end();

// Is the player in a menu? True while some list has input focus. DS3's
// CSMouseMan switches the same way - a pointer while UI owns the input,
// mouse-look otherwise - and the camera and the pointer both key off this.
bool menu_pointer_in_menu();
