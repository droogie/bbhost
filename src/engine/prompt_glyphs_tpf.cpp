// The texture side of engine/prompt_glyphs.h: finding the prompt textures in
// a PS4 TPF and writing the generated BC7 blocks into them. Kept apart from
// the start-up half so a test links it with zlib alone.
#include "engine/prompt_glyphs.h"

#include <zlib.h>

#include <cstring>

namespace {

#include "engine/prompt_glyphs_data.inc"

// The KG_* textures in the order tools/make_prompt_glyphs.py encodes them.
const char* const kPromptSlots[] = {"KG_Cancel", "KG_OK",  "KG_R_L", "KG_R_U", "KG_L1",   "KG_L2",    "KG_L3",
                                    "KG_R1",     "KG_R2",  "KG_R3",  "KG_LS",  "KG_RS",   "KG_L_U",   "KG_L_D",
                                    "KG_L_LR",   "KG_L_UD", "KG_L_UDLR", "KG_Start", "KG_TP_L", "KG_TP_R"};
constexpr int kSlotCount = sizeof(kPromptSlots) / sizeof(kPromptSlots[0]);
constexpr int kIconSide = 32;
constexpr std::size_t kIconBytes = (kIconSide / 4) * (kIconSide / 4) * 16;

// The HUD's atlas and the strip of it that holds its Circle, Cross and L3
// glyphs, in 4x4 blocks: (165, 30) to (189, 38) of a 1024x512 texture.
const char kHudAtlas[] = "MENU_Common_00091";
constexpr int kHudAtlasW = 1024, kHudAtlasH = 512;
constexpr unsigned kHudStripX = 165, kHudStripY = 30, kHudStripW = 24, kHudStripH = 8;

static_assert(sizeof(kGlyphs_xbox) == kSlotCount * kIconBytes, "the Xbox glyphs");
static_assert(sizeof(kGlyphs_steamdeck) == kSlotCount * kIconBytes, "the Steam Deck glyphs");
static_assert(sizeof(kGlyphs_steamcontroller) == kSlotCount * kIconBytes, "the Steam Controller glyphs");
static_assert(sizeof(kHudStrip_xbox) == kHudStripW * kHudStripH * 16, "the Xbox HUD strip");
static_assert(sizeof(kGlyphBase_xbox) == kSlotCount && sizeof(kGlyphMask_xbox) == kSlotCount * 8, "the Xbox masks");

// Each glyph's block mask (bit y * 8 + x) and the slot whose original texture
// it goes over, so a symbol keeps the game's own button edge; the HUD strip
// writes only the block columns it changes.
struct Pack {
    const std::uint8_t* glyphs;
    const std::uint8_t* strip;
    const std::uint64_t* masks;
    const std::uint8_t* bases;
    std::uint32_t strip_columns;
};
bool pack_of(PromptStyle s, Pack* p) {
    switch (s) {
        case PromptStyle::Xbox: *p = {kGlyphs_xbox, kHudStrip_xbox, kGlyphMask_xbox, kGlyphBase_xbox, kHudStripColumns_xbox}; return true;
        case PromptStyle::SteamDeck:
            *p = {kGlyphs_steamdeck, kHudStrip_steamdeck, kGlyphMask_steamdeck, kGlyphBase_steamdeck,
                  kHudStripColumns_steamdeck};
            return true;
        case PromptStyle::SteamController:
            *p = {kGlyphs_steamcontroller, kHudStrip_steamcontroller, kGlyphMask_steamcontroller,
                  kGlyphBase_steamcontroller, kHudStripColumns_steamcontroller};
            return true;
        default: return false;
    }
}

std::uint32_t rd32(const GlyphBytes& b, std::size_t i) {
    return b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (static_cast<std::uint32_t>(b[i + 3]) << 24);
}
std::uint16_t rd16(const GlyphBytes& b, std::size_t i) { return static_cast<std::uint16_t>(b[i] | (b[i + 1] << 8)); }
std::uint32_t rdbe32(const GlyphBytes& b, std::size_t i) {
    return (static_cast<std::uint32_t>(b[i]) << 24) | (b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3];
}
void wrbe32(GlyphBytes& b, std::size_t i, std::uint32_t v) {
    b[i] = static_cast<std::uint8_t>(v >> 24);
    b[i + 1] = static_cast<std::uint8_t>(v >> 16);
    b[i + 2] = static_cast<std::uint8_t>(v >> 8);
    b[i + 3] = static_cast<std::uint8_t>(v);
}

// Blocks in row order written into a tiled texture at block (x, y), only the
// columns set in `columns`.
void put_blocks(GlyphBytes& tpf, const TpfTexture& t, const std::uint8_t* blocks, unsigned x, unsigned y, unsigned w, unsigned h,
                std::uint32_t columns) {
    const unsigned wide = static_cast<unsigned>(t.width) / 4;
    for (unsigned r = 0; r < h; ++r)
        for (unsigned c = 0; c < w; ++c)
            if (columns >> c & 1)
                std::memcpy(tpf.data() + t.offset + ps4_bc_block_offset(x + c, y + r, wide), blocks + (r * w + c) * 16, 16);
}

std::size_t find4(const GlyphBytes& b, const char* tag) {
    for (std::size_t i = 0; i + 4 <= b.size(); ++i)
        if (std::memcmp(b.data() + i, tag, 4) == 0) return i;
    return std::string::npos;
}

}  // namespace

const char* prompt_style_name(PromptStyle s) {
    switch (s) {
        case PromptStyle::Xbox: return "Xbox";
        case PromptStyle::SteamDeck: return "Steam Deck";
        case PromptStyle::SteamController: return "Steam Controller";
        default: return "PlayStation";
    }
}

std::size_t ps4_bc_block_offset(unsigned bx, unsigned by, unsigned blocks_wide) {
    const unsigned tiles_wide = (blocks_wide + 7) / 8;
    const unsigned tile = (by / 8) * tiles_wide + bx / 8;
    unsigned morton = 0;
    for (unsigned bit = 0; bit < 3; ++bit) {
        morton |= ((bx >> bit) & 1u) << (2 * bit);
        morton |= ((by >> bit) & 1u) << (2 * bit + 1);
    }
    return (static_cast<std::size_t>(tile) * 64 + morton) * 16;
}

bool tpf_bc7_textures(const GlyphBytes& tpf, std::vector<TpfTexture>* out, std::string* why) {
    out->clear();
    // Header: "TPF\0", data size, count, then platform (4: PS4), a flag and the
    // name encoding (1: UTF-16). Each PS4 entry is 0x24 bytes - offset, size,
    // format, type, mip count, flags, width, height, two words, name offset,
    // a float block's presence, DXGI format - and a float block follows it
    // when present (a word, then its length and that many bytes).
    if (tpf.size() < 0x10 || std::memcmp(tpf.data(), "TPF\0", 4) != 0) {
        *why = "not a TPF texture file";
        return false;
    }
    const std::uint32_t count = rd32(tpf, 8);
    if (tpf[0xc] != 4 || tpf[0xe] != 1) {
        *why = "not a PS4 TPF with UTF-16 names";
        return false;
    }
    constexpr std::int32_t kBc7 = 98;  // DXGI_FORMAT_BC7_UNORM
    std::size_t at = 0x10;
    for (std::uint32_t i = 0; i < count; ++i) {
        if (at + 0x24 > tpf.size()) {
            *why = "the texture entries run past the file";
            return false;
        }
        const std::uint32_t offset = rd32(tpf, at), size = rd32(tpf, at + 4);
        const std::uint8_t mips = tpf[at + 10];
        const int width = static_cast<std::int16_t>(rd16(tpf, at + 12)), height = static_cast<std::int16_t>(rd16(tpf, at + 14));
        const std::uint32_t name_at = rd32(tpf, at + 24);
        const bool floats = rd32(tpf, at + 28) != 0;
        const std::int32_t dxgi = static_cast<std::int32_t>(rd32(tpf, at + 32));
        at += 0x24;
        if (floats) {
            if (at + 8 > tpf.size()) {
                *why = "a texture's float block runs past the file";
                return false;
            }
            at += 8 + rd32(tpf, at + 4);
        }
        std::string name;
        for (std::size_t n = name_at; n + 1 < tpf.size(); n += 2) {
            const std::uint16_t c = rd16(tpf, n);
            if (!c) break;
            name.push_back(c < 0x80 ? static_cast<char>(c) : '?');
        }
        // BC7 is a byte a pixel: one level's size is width * height.
        if (dxgi == kBc7 && mips == 1 && width > 0 && height > 0 && size == static_cast<std::uint32_t>(width * height) &&
            offset + static_cast<std::size_t>(size) <= tpf.size())
            out->push_back({name, offset, width, height});
    }
    return true;
}

bool prompt_glyphs_write_tpf(GlyphBytes& tpf, PromptStyle style, std::string* why) {
    Pack pack;
    if (!pack_of(style, &pack)) return true;
    std::vector<TpfTexture> textures;
    if (!tpf_bc7_textures(tpf, &textures, why)) return false;
    const auto find = [&](const char* name, int w, int h) -> const TpfTexture* {
        for (const TpfTexture& t : textures)
            if (t.name == name && t.width == w && t.height == h) return &t;
        return nullptr;
    };
    // Every target first, so a file that lacks one is left as it came.
    const TpfTexture* slots[kSlotCount];
    for (int i = 0; i < kSlotCount; ++i) {
        if (!(slots[i] = find(kPromptSlots[i], kIconSide, kIconSide))) {
            *why = std::string("no 32x32 BC7 ") + kPromptSlots[i] + " (not the 1.09 file, or a prompt mod's)";
            return false;
        }
    }
    const TpfTexture* atlas = find(kHudAtlas, kHudAtlasW, kHudAtlasH);
    if (!atlas) {
        *why = std::string("no 1024x512 BC7 ") + kHudAtlas;
        return false;
    }
    // The bases as the file has them, before any slot is written. A 32x32 is
    // one 8x8-block tile, so a texture's bytes copy to another's whole.
    std::vector<std::uint8_t> original(kSlotCount * kIconBytes);
    for (int i = 0; i < kSlotCount; ++i) std::memcpy(original.data() + i * kIconBytes, tpf.data() + slots[i]->offset, kIconBytes);
    constexpr unsigned kWide = kIconSide / 4;
    for (int i = 0; i < kSlotCount; ++i) {
        std::uint8_t* to = tpf.data() + slots[i]->offset;
        std::memcpy(to, original.data() + pack.bases[i] * kIconBytes, kIconBytes);
        const std::uint8_t* glyph = pack.glyphs + i * kIconBytes;
        for (unsigned b = 0; b < 64; ++b)
            if (pack.masks[i] >> b & 1)
                std::memcpy(to + ps4_bc_block_offset(b % kWide, b / kWide, kWide), glyph + b * 16, 16);
    }
    put_blocks(tpf, *atlas, pack.strip, kHudStripX, kHudStripY, kHudStripW, kHudStripH, pack.strip_columns);
    return true;
}

GlyphBytes prompt_glyphs_apply_dcx(const GlyphBytes& dcx, PromptStyle style, std::string* why) {
    // DCX as the disc has it: a header of DCX, DCS (the two sizes, big-endian),
    // DCP ("DFLT") and DCA blocks, then one zlib stream.
    if (dcx.size() < 0x4c || std::memcmp(dcx.data(), "DCX\0", 4) != 0) {
        *why = "not a DCX file";
        return {};
    }
    const std::size_t dcs = find4(dcx, "DCS\0"), dcp = find4(dcx, "DCP\0"), dca = find4(dcx, "DCA\0");
    if (dcs == std::string::npos || dcp == std::string::npos || dca == std::string::npos || dca + 8 > dcx.size() ||
        std::memcmp(dcx.data() + dcp + 4, "DFLT", 4) != 0) {
        *why = "not a DFLT DCX file";
        return {};
    }
    const std::size_t head = dca + rdbe32(dcx, dca + 4);
    const std::uint32_t raw_size = rdbe32(dcx, dcs + 4);
    if (head > dcx.size() || raw_size == 0 || raw_size > (256u << 20)) {
        *why = "the DCX header's sizes are not sane";
        return {};
    }
    GlyphBytes tpf(raw_size);
    uLongf got = raw_size;
    if (uncompress(tpf.data(), &got, dcx.data() + head, static_cast<uLong>(dcx.size() - head)) != Z_OK || got != raw_size) {
        *why = "the DCX does not inflate to its stated size";
        return {};
    }
    if (!prompt_glyphs_write_tpf(tpf, style, why)) return {};
    uLongf clen = compressBound(static_cast<uLong>(tpf.size()));
    GlyphBytes out(dcx.begin(), dcx.begin() + static_cast<std::ptrdiff_t>(head));
    out.resize(head + clen);
    if (compress2(out.data() + head, &clen, tpf.data(), static_cast<uLong>(tpf.size()), 9) != Z_OK) {
        *why = "compress2";
        return {};
    }
    out.resize(head + clen);
    wrbe32(out, dcs + 8, static_cast<std::uint32_t>(clen));
    return out;
}

PromptStyle prompt_style_auto(int pad_family, bool steam_deck) {
    // host_pad_family's values (host/window.h PadFamily).
    switch (pad_family) {
        case 1: return PromptStyle::PlayStation;
        case 2: return PromptStyle::Xbox;
        case 3: return PromptStyle::SteamDeck;
        case 4: return PromptStyle::SteamController;
        case 5: return PromptStyle::Xbox;  // another pad: most are laid out as XInput's
        default: return steam_deck ? PromptStyle::SteamDeck : PromptStyle::PlayStation;
    }
}

PromptStyle prompt_style_choose(int setting, int pad_family, bool steam_deck) {
    switch (setting) {
        case 1: return PromptStyle::Xbox;
        case 2: return PromptStyle::PlayStation;
        default: return prompt_style_auto(pad_family, steam_deck);
    }
}
