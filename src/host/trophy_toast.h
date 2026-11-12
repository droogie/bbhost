#pragma once

// Trophy popups, drawn on the host overlay (host/overlay.h).

// Queue a popup for a trophy id. Safe from any thread: the game's trophy
// worker calls it from sceNpTrophyUnlockTrophy.
void host_trophy_toast_push(int trophy_id);

// Draw the popup being shown, if any. Call it from the window pump, after
// host_overlay_reset() and before host_overlay_commit(). display_w and
// display_h are the overlay's coordinate space (the game's display buffer).
void host_trophy_toast_draw(float display_w, float display_h);
