#pragma once

// Button prompts drawn for the controller in the player's hands (#206).
//
// The prompts are textures in menu/common.tpf.dcx: twenty KG_* icons
// (engine/key_prompts.h) and the HUD's Circle, Cross and L3 in
// MENU_Common_00091. For an Xbox pad, a Steam Deck or a Steam Controller the
// port writes a copy of the dump's file with that controller's glyphs, BC7
// block for block, and mounts it as a generated FS layer. The glyphs are
// Kenney's Input Prompts (CC0), encoded by tools/make_prompt_glyphs.py into
// prompt_glyphs_data.inc; nothing of the game's is in the repository.
//
// "Controller prompts" (host/options.cpp): Auto follows the pad connected at
// start, Xbox always, Default the game's own. Read once, at start;
// BBHOST_PROMPT_GLYPHS=0 keeps the game's own.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class PromptStyle : int { PlayStation = 0, Xbox, SteamDeck, SteamController };
const char* prompt_style_name(PromptStyle s);  // "PlayStation", "Xbox", ...

// What the setting's Auto picks: the first connected pad's family
// (host_pad_family, host/window.h, as an int), and without a pad a Steam
// Deck's own controls on a Deck, else the game's own.
PromptStyle prompt_style_auto(int pad_family, bool steam_deck);
// The setting (0 Auto, 1 Xbox, 2 Default: the PS4's) resolved.
PromptStyle prompt_style_choose(int setting, int pad_family, bool steam_deck);

// ---- The texture file, for tests: no FS, no settings.

using GlyphBytes = std::vector<std::uint8_t>;

// One BC7 texture with a single mip level in a PS4 TPF: where its blocks
// start in the TPF and its size in pixels.
struct TpfTexture {
    std::string name;
    std::size_t offset = 0;
    int width = 0, height = 0;
};
// The TPF's single-level BC7 textures; false, with why, when it is not a
// PS4 TPF with UTF-16 names or its entries run past its end.
bool tpf_bc7_textures(const GlyphBytes& tpf, std::vector<TpfTexture>* out, std::string* why);

// Byte offset of 4x4 block (bx, by) in a PS4 BC texture `blocks_wide` blocks
// across: 8x8-block tiles in rows, each tile's blocks in Morton order.
std::size_t ps4_bc_block_offset(unsigned bx, unsigned by, unsigned blocks_wide);

// The style's glyphs written into an uncompressed TPF in place: each KG_*
// texture (32x32), whole or only its symbol's blocks over the edge of one of
// the game's own buttons, and the HUD atlas's strip. False, with why, when one
// of them is missing or not the size the 1.09 file has (a prompt mod already,
// say), the TPF then unchanged. PlayStation changes nothing and is true.
bool prompt_glyphs_write_tpf(GlyphBytes& tpf, PromptStyle style, std::string* why);

// The same for the DCX-compressed file as it is on disc (DFLT); the result is
// compressed the same way. Empty, with why, on failure.
GlyphBytes prompt_glyphs_apply_dcx(const GlyphBytes& dcx, PromptStyle style, std::string* why);

// ---- At start

// Chooses the style and, unless it is PlayStation, makes (or reuses) the
// patched copy under <data>/bbhost/prompt-glyphs/<style> and mounts it as a
// generated FS layer. After the window has opened the pads already plugged in
// and before the game runs: main.cpp calls it right before _start.
void prompt_glyphs_install();
