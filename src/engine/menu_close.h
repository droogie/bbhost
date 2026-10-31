#pragma once

// A menu window that closes answers at once, not after its fade-out.
//
// Every window of the game's menus that closes on a choice - the title's, the
// pause grid, the inventory and its lists, an item's command menu - shares
// one Close (sub_1ed6820, vtable slot 8 of each; addresses are Binary
// Ninja's). It plays the movie's "FadeOut" (or "Disabled_FadeOut") and queues
// two tasks on the window: wait while the root clip is still playing that
// label (sub_1ed7580, a functor nothing but Close makes - Scaleform's play
// state and current label), then write the answer at +0x1c0 that the
// window's owner is waiting for. The windows' common update (sub_1ed5b10)
// reads no input while a task is current (+0x10) or queued (+0x40). So a
// choice that closes a window takes effect only once the fade has played -
// 6 frames of a 60 fps movie, 100 ms, twice that in the 30 fps ones - and a
// press in that time is lost: Circle twice, 100 ms apart, closed the inventory
// and not the pause grid behind it.
//
// The wait is answered "not playing" from the start, so the answer lands on
// the frame of the press, the window it returns to takes input on the next
// one, and that second Circle closes the pause grid. The closing window goes
// when its owner lets it go, so its fade-out is not seen; the window behind
// it, or the next one, fades in as before and takes input while it does.
// The message, Yes/No and quantity boxes (vtable 0x579cc10: "Cannot discard
// currently equipped items", quitting to the title, Hunter's Mark, buying,
// selling, fortifying) end without Close, but their OK (sub_1feaad0, slot 11,
// which the item command menu and three other dialogs call too) does nothing
// until the box has been up its minimum (+0xde0, 0.5 s; the time up, +0x1f10,
// is what their update adds the frame's seconds to). An OK sooner was lost,
// and the press had to be made again. The time up is raised to the minimum
// when OK is pressed, so OK takes at once; Circle never waited.
// The loading screen's window (vtable 0x579c460) keeps
// the wait: it closes on the world's first frame, and its fade-out hides the
// frames the world takes to settle - cut, the player's model showed white and
// the scene ungraded for a moment after every load.
//
// BBHOST_MENU_FADE_WAIT=1 keeps the game's waits, both. BBHOST_MENU_LOG=1
// logs each Close and the frame its answer lands, and each box's OK.

struct ElfImage;

void menu_close_install(ElfImage* image);
