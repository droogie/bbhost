#pragma once

// The letters of the game's own menus, for ImGui (a plugin's panel,
// host/plugin_panel.cpp): the font the menus draw with is a Scaleform font
// library in the player's dump - menu/<language>/font.gfx, one DefineFont3
// ("TT-ReiMin Y10 B" in engus's, 7,652 glyphs as quadratic outlines) - and its
// glyphs are rasterised here as ImGui asks for them, at the size it asks for
// (an ImFontLoader). Nothing of the font is shipped: it is read from the dump
// at start, as engine/menu_assets.cpp reads the menus it edits.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace game_font {

struct Glyph {
    std::uint32_t shape = 0, shape_end = 0;  // its SHAPE: offsets into Font::data
    float advance = 0;                       // in font units
};

struct Font {
    std::vector<std::uint8_t> data;  // the DefineFont tag's body
    std::string name;
    float units_per_em = 20480.0f;   // DefineFont3: 1024 an em, in twentieths
    float ascent = 0, descent = 0;   // font units (both positive)
    std::vector<std::uint16_t> codes;  // the glyphs' code points, as stored
    std::vector<std::pair<std::uint32_t, int>> by_code;  // (code, glyph), sorted
    std::vector<Glyph> glyphs;
    int index_of(std::uint32_t code) const;  // -1: none
};

// The first DefineFont3 (or DefineFont2) in a GFX or SWF file's bytes (CFX and
// CWS zlib-compressed ones too): false, with the reason, when there is none or
// it is not whole.
bool parse(const std::vector<std::uint8_t>& file, Font& out, std::string* why = nullptr);

// A glyph's outline: closed contours of points (x, y pairs) in font units, y
// down from the baseline, the curves cut into lines no further than
// `tolerance` units from them.
using Contour = std::vector<float>;
bool outline(const Font& font, int glyph, std::vector<Contour>& out, float tolerance);

// The contours' coverage (0-255) in a w x h bitmap, a point (x, y) of them at
// (x * scale + ox, y * scale + oy): the non-zero rule, 16 samples a pixel.
void rasterise(const std::vector<Contour>& contours, float scale, float ox, float oy, int w, int h, std::uint8_t* out);

// The font of the game's menus, read once from the dump (the first of the
// languages' font.gfx found): nullptr when there is none.
const Font* menu_font();

}  // namespace game_font

struct ImFont;
// The game's menu font as an ImFont in the current ImGui context (its own
// ImFontLoader); nullptr when the dump has none. The atlas lends it the font.
ImFont* game_font_add_to_imgui();
