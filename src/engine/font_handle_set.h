#pragma once
// Scaleform's font-handle set: the font handles a movie's FontManager made
// (CreatedFonts), a HashSet of FontHandle* keyed by font name and flags, which
// engine/graphics_patch.cpp's font_set_remove_hook gives a removal by the
// handle's own pointer (why: there).
//
// The table, as the game's code walks it (HashSet RemoveAlt 0x27d6a80, its
// lookup 0x27d6980, its add 0x27d6250): {count, mask, entries[mask + 1]}, an
// entry 32 bytes - the next entry of its chain (-2 an empty entry, -1 the
// chain's end), the chain's natural slot (hash & mask: the chain's head sits
// there), the FontHandle*, a byte.

#include <cstdint>

// Removes the entry that holds `handle` itself, as the game's removal unlinks
// one: a chain's head is replaced by its next entry (that entry's slot
// emptied), any other entry is unlinked from the one before it; the count goes
// down by one. True when removed. False, the table untouched, when no entry
// holds the handle, or the table or the handle's chain is not as described.
bool font_set_remove_handle(std::uint8_t* table, std::uint64_t handle);
