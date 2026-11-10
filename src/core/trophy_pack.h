#pragma once

// The trophy names of the player's own game, read from its trophy00.trp.
//
// The pack's names, grades and descriptions are XML encrypted with AES-128-CBC.
// The key is the game's NP communication ID (NPWRxxxxx_00) encrypted with a
// 16-byte trophy key that is not in the game files, so the player supplies it
// (BBHOST_TROPHY_KEY, or <data>/trophy_key.txt) and nothing is bundled.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct TrophyInfo {
    int id = -1;
    char grade = 'B';  // P platinum, G gold, S silver, B bronze
    bool hidden = false;
    std::string name;    // UTF-8
    std::string detail;  // UTF-8; may hold newlines
    std::uint64_t icon_pos = 0, icon_size = 0;  // its TROPnnn.PNG in the pack; 0 for none
};

// SceNpTrophyGrade: 1 platinum, 2 gold, 3 silver, 4 bronze.
inline int trophy_grade_code(char grade) {
    switch (grade) {
        case 'P': return 1;
        case 'G': return 2;
        case 'S': return 3;
        default: return 4;
    }
}

// Reads and decrypts one trophy pack. `key_hex` is the 32 hex digits of the
// trophy key. On failure returns false and says why in *err; `out` is only
// filled on success.
bool trophy_pack_parse(const std::string& trp_path, const std::string& np_comm_id, const std::string& key_hex,
                       std::vector<TrophyInfo>* out, std::string* err);

// The NP communication ID in an npbind.dat (the first NPWRddddd_dd in it), or
// an empty string.
std::string trophy_pack_comm_id(const std::string& npbind_path);

// Loads the pack for the game: finds the ID in npbind_path, parses trp_path and
// keeps the result for trophy_pack_find. Safe to call again.
bool trophy_pack_load(const std::string& trp_path, const std::string& npbind_path, const std::string& key_hex,
                      std::string* err);

// Installs the function that loads the game's pack. The first trophy_pack_find
// calls it (once), so the pack is read when something first asks for a name,
// not before the game's files are there. Call this once at start-up.
void trophy_pack_set_loader(void (*loader)());

// A copy of one trophy's info; false when no pack is loaded or the id is not in it.
// Safe from any thread. `out` may be null (just to trigger the load).
bool trophy_pack_find(int id, TrophyInfo* out);

std::size_t trophy_pack_count();

// A trophy's icon, the PNG file as the pack holds it (TROPnnn.PNG, which is
// not encrypted). False when no pack is loaded or it has none for the id.
bool trophy_pack_icon(int id, std::vector<std::uint8_t>* png);

// A PNG as 8-bit RGBA, straight alpha. Colour types grey, grey+alpha, RGB, RGB+alpha
// and palette, at any bit depth the format allows; not interlaced. On failure
// returns false and says why in *err.
bool png_decode_rgba(const std::uint8_t* data, std::size_t size, int* width, int* height, std::vector<std::uint8_t>* rgba,
                     std::string* err);

// RGBA resized to dw x dh by averaging the source pixels each new one covers
// (a box filter, done on premultiplied colour so transparent edges stay clean).
std::vector<std::uint8_t> rgba_resize(const std::vector<std::uint8_t>& src, int sw, int sh, int dw, int dh);
