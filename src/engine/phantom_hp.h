#pragma once

// Full Phantom HP: a hunter summoned into another hunter's world keeps their
// full max HP there. For as long as a hunter plays in another's world the game
// keeps a SpEffectParam row on them whose maxHpRate, 0.7, takes their max HP
// to 70%: 9006 a cooperator's, 9026 an invader's, and 9036, 9046 and 9056 a
// covenant's natural enemy's (the rows the game names for a Vileblood, a
// hunter of Vilebloods and a hunter of hunters).
//
// world.full_phantom_hp (BBHOST_FULL_PHANTOM_HP=0|1, on by default): those
// five rows' maxHpRate is 1.0 in memory, written once the 1.09 eboot's table
// has loaded and again whenever the game loads it anew (engine/params.cpp
// calls in). A row that is not at the game's 0.7 (a mod's) is left as it is.
// Off, the game's 70%.
//
// It is each visitor's own setting. A visitor's max HP is made by its own game
// from its own rows, and the other games in the session - the host's among
// them - show the HP and max HP that game sends: a host with it off sees a
// visitor with it on at full HP, and the other way round. Hits take what they
// always do.

// SpEffectParam was loaded, or loaded anew (engine/params.cpp, main thread).
void phantom_hp_params_loaded();
