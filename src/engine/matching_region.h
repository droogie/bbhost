#pragma once

// Network > Matching Regions starts at Worldwide.
//
// The option is a byte of the game's options block, +0xf: 1 Worldwide
// (message 0x9d80), 0 Local (0x9d81) - the Network section's handler,
// sub_1f21e80, builds the row from it with sub_1f2a100 (addresses are Binary
// Ninja's). The block's constructor, sub_208af20, writes 0 there
// (0x208af63, `mov byte [rbx+0xf], 0`); every fresh block and the menu's
// Defaults come from it. It writes 1 here, so a new profile and Defaults
// start at Worldwide; the player can still pick Local in the menu. Only on
// the 1.09 eboot, and only where the bytes are the expected ones.
// BBHOST_MATCHING_REGION=game keeps the game's 0.
//
// Not the settings global's byte at 0x5962878+0xd10: that one is the
// user-generated media restriction - non-zero leaves Player Name Display
// only Online ID and shows "Blocked by parental control".

struct ElfImage;

void matching_region_install(ElfImage* image);
