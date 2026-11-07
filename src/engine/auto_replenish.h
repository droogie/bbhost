#pragma once

// Auto-Replenish Items: twenty more consumables refill from storage the
// way Blood Vials and Quicksilver Bullets do. EquipParamGoods' isAutoReplenish
// is set on those two alone in the game; with world.auto_replenish
// (BBHOST_AUTO_REPLENISH=0|1, on by default) it is set, in memory, on 1100
// Antidote, 1101 Sedative, 1110 Beast Blood Pellet, 1120 Blue Elixir, 1200
// Molotov Cocktail, 1201 Delayed Molotov, 1210 Poison Knife, 1220 Oil Urn, 1230
// Pebble, 1240 Throwing Knife, 1250 Rope Molotov Cocktail, 1251 Delayed Rope
// Molotov, 1260 Pungent Blood Cocktail, 1270 Numbing Mist, 1300 Fire Paper,
// 1320 Bolt Paper, 1330 Bone Marrow Ash, 1400 Bold Hunter's Mark, 1410 Shining
// Coins and 2030 Lead Elixir - written once the 1.09 eboot's table has loaded
// and again whenever the game loads it anew (engine/params.cpp calls in). Off,
// the game's two.
//
// The game refills these from the storage box up to what the hunter can carry
// (the goods' maxNum; less when storage holds less) in one place, the map
// load's step that also tops up the vials (sub_18dace0 from sub_1d39470): after
// a death in the hunter's own world and on every arrival in the Hunter's
// Dream - not on travel between the world's lamps, nor on joining or leaving
// another player's world.

// EquipParamGoods was loaded, or loaded anew (engine/params.cpp, main thread).
void auto_replenish_params_loaded();
