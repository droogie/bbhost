#pragma once

// The port's menu assets, made from the player's own dump.
//
// The PC sections of the System menu (engine/option_menu.h) need files the
// shipped game does not have: menu/optionsetting.gfx with the port's section
// sprites in it, msg/<language>/menu.msgbnd.dcx for every language the dump
// has with the port's message ids (the game reads the folder its region and
// language pick - one language only left the others showing "(110006)" and
// crashing in the PC sections), and menu/title.gfx / title_dlc.gfx with a
// sixth command line for the title's Quit Game.
// Both are edits of the dump's files - nothing in them is authored here but the
// strings (tools/pc_option_messages.tsv, built in) - so no package can carry
// them, and every install has to make its own. This does, at start: into
// <data>/bbhost/menu-assets, again only when the generator, the strings or the
// dump's files change (a stamp beside them says which they were made from).
// The FS serves that directory behind the player's own overlay (paths.mods),
// so a hand-made file there still wins.
//
// This is the only generator: there is no script and no manual step, on any
// platform. Its first version was checked byte-identical to the Python tools
// it replaced (2fc0d74).

#include <string>

// System > PC Settings' slots (engine/option_menu.cpp's g_toggles and its
// Texture Quality and Frame Rate lists): the movie's PCSetting section has
// this many rows, the Defaults label below them (build_option_movie). On
// Windows one more than the donor's six, as PC Graphics has, for Stay
// Borderless, which has no row elsewhere.
#if defined(_WIN32)
constexpr int kPcSettingSlots = 7;
#else
constexpr int kPcSettingSlots = 6;
#endif

// Makes the assets if they are missing or stale and registers the directory
// with the FS. False, with the reason logged, when the dump lacks a source.
bool menu_assets_ensure();
