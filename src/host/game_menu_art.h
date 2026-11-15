#pragma once

// The game's own menu art for a plugin's panel (host/plugin_panel.cpp): pieces
// of the textures the game's options dialog draws with (menu/optionsetting.gfx
// names them: MENU_Common_00003 and MENU_Common_00001 of menu/common.tpf.dcx,
// and its key guide's KG_* icons) and the slate its other menus lie on
// (MENU_Common_00002), read from the player's dump the first time a panel is
// drawn, as host/game_font.h reads the menus' letters. Nothing of them is
// shipped.
//
// The formats, as read from the files: DCX "DFLT" is one zlib stream after the
// DCA header; a PS4 TPF lists its textures in 0x24-byte entries (data offset,
// size, format, ..., width, height, name offset in UTF-16); a texture is BC
// blocks in 8x8-block tiles, Morton order inside a tile, the tiles row-major
// and both dimensions padded to 8 blocks. The art is BC7 (format 102).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace game_menu_art {

// ---- the formats

// A DCX "DFLT" file's contents (the zlib stream after "DCA\0" and its size).
bool inflate_dcx(const std::vector<std::uint8_t>& file, std::vector<std::uint8_t>& out);

struct TpfTexture {
    std::string name;
    int format = 0;  // 102 BC7, 0 and 1 BC1, 3 BC2, 5 BC3
    int width = 0, height = 0;
    std::size_t offset = 0, size = 0;  // its blocks in the TPF
};
// A PS4 TPF's textures (platform byte 4); false when it is not one or an entry
// lies outside it.
bool parse_tpf(const std::vector<std::uint8_t>& tpf, std::vector<TpfTexture>& out);

// The byte offset of block (bx, by) in a PS4-tiled texture blocks_w blocks wide.
std::size_t tiled_block_offset(int bx, int by, int blocks_w, int bytes_per_block);

// One BC7 block as 4x4 RGBA pixels, row-major (64 bytes). A reserved mode
// (a first byte of 0) is transparent black, as the format has it.
void decode_bc7(const std::uint8_t block[16], std::uint8_t out[64]);

// The pixels [x0, x1) x [y0, y1) of a PS4-tiled BC7 texture as RGBA, row-major;
// only the blocks under them are decoded. False when the region or the data
// does not fit.
bool decode_region(const std::uint8_t* data, std::size_t size, int width, int height, int x0, int y0, int x1, int y1,
                   std::vector<std::uint8_t>& out);

// RGBA pixels w x h at half their size (each 2x2 block averaged by its
// alpha); a last odd row or column is left out.
std::vector<std::uint8_t> halve(const std::vector<std::uint8_t>& rgba, int w, int h);

// Art drawn on black and added to the picture (the game's ornaments and glows)
// as straight alpha: the brightest channel the alpha, the colour divided by it,
// so that blending it over the picture adds the same light.
void additive_to_alpha(std::uint8_t* rgba, std::size_t pixels);

// ---- the pieces

enum Piece : int {
    kOrnament,    // the dialog's top and foot: a rule with scrolled ends and a crest at its middle
    kBand,        // the shaded band beside that rule
    kFiligree,    // the small filigree where the inner rules meet the column rule
    kRule,        // a plain stretch of rule, drawn as long as it is needed
    kLabelLine,   // the line under a row's label, fading out to the left
    kValueEnd,    // the scrolled end of a value's underline (the left one; mirrored for the right)
    kGlow,        // the cursor's golden glow, brightest at its right end
    kArrow,       // a choice's arrow, pointing right
    kSlate,       // the menus' slate (MENU_Common_00002's, at half its size)
    kSliderTrack, // a slider's track
    kSliderFill,  // a slider's filled part
    kIconCircle,  // the key guide's icons, by the button they show
    kIconCross,
    kIconTriangle,
    kIconSquare,
    kIconL1,
    kIconR1,
    kIconL2,
    kIconR2,
    kIconL3,
    kIconR3,
    kIconDpad,
    kIconDpadUpDown,
    kIconDpadLeftRight,
    kPieceCount
};

struct Source {
    const char* texture;   // in menu/common.tpf.dcx
    int x0, y0, x1, y1;    // its pixels there
    bool additive;         // drawn added on black: turned to straight alpha
    int shrink = 1;        // 2: kept at half its size (the slate, whose grain survives it)
};
const Source& source(Piece p);

// Where the 5-part pieces may stretch (kOrnament: the rule between its ends
// and its crest): the piece's own x of the ends of its five parts - an end,
// a stretch, the middle, a stretch, an end - from 0 to its width.
struct Slices {
    float x[6];
};
const Slices& slices(Piece p);

// A 5-part piece drawn across a span `width` wide at `scale` (its pixels to
// the screen's): the ends and the middle keep their size, the two stretches
// share what is left, equally. part[i] is the screen x span of part i from
// the span's left; false when the span is narrower than the fixed parts.
struct Span {
    float x0[5], x1[5];
};
bool stretch(const Slices& s, float width, float scale, Span& out);

// The atlas: every piece the dump had, packed into one RGBA texture.
struct Placed {
    bool present = false;
    int w = 0, h = 0;              // the piece's own size in pixels
    float u0 = 0, v0 = 0, u1 = 0, v1 = 0;  // its texels in the atlas
};
struct Atlas {
    int width = 0, height = 0;
    std::vector<std::uint8_t> rgba;
    Placed piece[kPieceCount];
};

// The pieces packed in shelves (sizes in pixels; a size of 0 x 0 is a piece
// left out), 2 px apart with each piece's edge repeated into the gap so that
// filtering never reaches a neighbour, in a width x height atlas; false when
// they do not all fit.
bool pack(const std::vector<std::vector<std::uint8_t>>& pixels, const std::vector<int>& widths,
          const std::vector<int>& heights, int width, int height, Atlas& out);

// The pieces from a TPF's contents (common.tpf, inflated): the ones it has.
bool build(const std::vector<std::uint8_t>& tpf, Atlas& out, std::string* why = nullptr);

// The art, read once from the dump (menu/common.tpf.dcx): nullptr when the
// dump has none, or with BBHOST_MENU_ART=0 (a test of the panel without it).
// A call while a preload reads it waits for it.
const Atlas* menu_art();
// Reads it on a thread of its own (the read, the inflating and the decoding
// take a fifth of a second): a panel's first frame then has it at hand.
void preload();

}  // namespace game_menu_art

struct ImTextureData;
// The atlas as a texture of the current ImGui context, registered with it the
// first time it is asked for there; nullptr when the dump has no art.
ImTextureData* game_menu_art_texture();
