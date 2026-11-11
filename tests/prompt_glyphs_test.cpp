// The controller prompts' glyphs (engine/prompt_glyphs.h): the PS4 block
// order, the TPF reader, the write into the prompt textures and the HUD
// atlas's strip (and nowhere else), the DCX round trip, a file without the
// prompt textures left alone, and what Auto picks for each controller. The
// TPF is made here, shaped as the 1.09 menu/common.tpf is - no game file.
// With BBHOST_APP0 set, the dump's own file is patched as well.
#include "engine/prompt_glyphs.h"

#include <zlib.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c)                                                     \
    do {                                                             \
        if (!(c)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                \
        }                                                            \
    } while (0)

const char* const kSlots[] = {"KG_Cancel", "KG_OK", "KG_R_L",  "KG_R_U",  "KG_L1",   "KG_L2",    "KG_L3",
                              "KG_R1",     "KG_R2", "KG_R3",   "KG_LS",   "KG_RS",   "KG_L_U",   "KG_L_D",
                              "KG_L_LR",   "KG_L_UD", "KG_L_UDLR", "KG_Start", "KG_TP_L", "KG_TP_R"};

void put32(GlyphBytes& b, std::size_t i, std::uint32_t v) {
    for (int k = 0; k < 4; ++k) b[i + k] = static_cast<std::uint8_t>(v >> (8 * k));
}
void put16(GlyphBytes& b, std::size_t i, std::uint32_t v) {
    b[i] = static_cast<std::uint8_t>(v);
    b[i + 1] = static_cast<std::uint8_t>(v >> 8);
}

struct Tex {
    std::string name;
    int w, h;
    bool floats;
};

// A PS4 TPF of BC7 textures, each filled with 0xEE, names after the data.
GlyphBytes make_tpf(const std::vector<Tex>& texs) {
    std::size_t head = 0x10;
    for (const Tex& t : texs) head += 0x24 + (t.floats ? 8 + 4 : 0);
    GlyphBytes b(head);
    std::memcpy(b.data(), "TPF\0", 4);
    put32(b, 8, static_cast<std::uint32_t>(texs.size()));
    b[0xc] = 4;  // PS4
    b[0xd] = 3;
    b[0xe] = 1;  // UTF-16 names
    std::size_t at = 0x10;
    std::vector<std::size_t> entry;
    for (const Tex& t : texs) {
        entry.push_back(at);
        at += 0x24 + (t.floats ? 12 : 0);
    }
    for (std::size_t i = 0; i < texs.size(); ++i) {
        const Tex& t = texs[i];
        const std::size_t e = entry[i], data = b.size(), size = static_cast<std::size_t>(t.w) * t.h;
        b.resize(data + size, 0xee);
        put32(b, e, static_cast<std::uint32_t>(data));
        put32(b, e + 4, static_cast<std::uint32_t>(size));
        b[e + 10] = 1;  // one mip level
        put16(b, e + 12, static_cast<std::uint32_t>(t.w));
        put16(b, e + 14, static_cast<std::uint32_t>(t.h));
        put32(b, e + 28, t.floats ? 1 : 0);
        put32(b, e + 32, 98);  // BC7
        if (t.floats) put32(b, e + 0x24 + 4, 4);
    }
    for (std::size_t i = 0; i < texs.size(); ++i) {
        put32(b, entry[i] + 24, static_cast<std::uint32_t>(b.size()));
        for (char c : texs[i].name) {
            b.push_back(static_cast<std::uint8_t>(c));
            b.push_back(0);
        }
        b.push_back(0);
        b.push_back(0);
    }
    put32(b, 4, static_cast<std::uint32_t>(b.size()));
    return b;
}

std::vector<Tex> common_like() {
    std::vector<Tex> t{{"MENU_Common_00000", 256, 256, false}};
    for (const char* s : kSlots) t.push_back({s, 32, 32, false});
    t.push_back({"MENU_Common_00091", 1024, 512, true});  // a float block in front of the rest
    t.push_back({"MENU_Common_00092", 64, 64, false});
    return t;
}

GlyphBytes make_dcx(const GlyphBytes& raw) {
    uLongf clen = compressBound(static_cast<uLong>(raw.size()));
    GlyphBytes z(clen);
    compress2(z.data(), &clen, raw.data(), static_cast<uLong>(raw.size()), 6);
    z.resize(clen);
    GlyphBytes d(0x4c, 0);
    auto be = [&](std::size_t i, std::uint32_t v) {
        for (int k = 0; k < 4; ++k) d[i + k] = static_cast<std::uint8_t>(v >> (24 - 8 * k));
    };
    std::memcpy(d.data(), "DCX\0", 4);
    std::memcpy(d.data() + 0x18, "DCS\0", 4);
    be(0x1c, static_cast<std::uint32_t>(raw.size()));
    be(0x20, static_cast<std::uint32_t>(clen));
    std::memcpy(d.data() + 0x24, "DCP\0", 4);
    std::memcpy(d.data() + 0x28, "DFLT", 4);
    std::memcpy(d.data() + 0x44, "DCA\0", 4);
    be(0x48, 8);
    d.insert(d.end(), z.begin(), z.end());
    return d;
}

const TpfTexture* find(const std::vector<TpfTexture>& v, const char* name) {
    for (const TpfTexture& t : v)
        if (t.name == name) return &t;
    return nullptr;
}

void block_order() {
    // Inside one 8x8 tile: Morton order, x in the low bit.
    CHECK(ps4_bc_block_offset(0, 0, 8) == 0);
    CHECK(ps4_bc_block_offset(1, 0, 8) == 16);
    CHECK(ps4_bc_block_offset(0, 1, 8) == 32);
    CHECK(ps4_bc_block_offset(7, 7, 8) == 63 * 16);
    // Tiles in rows: a 1024-wide texture is 32 tiles across.
    CHECK(ps4_bc_block_offset(8, 0, 256) == 64 * 16);
    CHECK(ps4_bc_block_offset(0, 8, 256) == 32 * 64 * 16);
    // Every block of a 24x8 strip lands on its own 16 bytes.
    std::set<std::size_t> seen;
    for (unsigned y = 30; y < 38; ++y)
        for (unsigned x = 165; x < 189; ++x) seen.insert(ps4_bc_block_offset(x, y, 256));
    CHECK(seen.size() == 24 * 8);
}

void write_and_round_trip() {
    const GlyphBytes tpf = make_tpf(common_like());
    std::vector<TpfTexture> texs;
    std::string why;
    CHECK(tpf_bc7_textures(tpf, &texs, &why));
    CHECK(texs.size() == common_like().size());
    const TpfTexture* atlas = find(texs, "MENU_Common_00091");
    CHECK(atlas && atlas->width == 1024 && atlas->height == 512);

    GlyphBytes x = tpf;
    CHECK(prompt_glyphs_write_tpf(x, PromptStyle::Xbox, &why));
    CHECK(x.size() == tpf.size());
    // Every prompt texture written to; the strip's blocks in the atlas
    // and nothing else of it; the other textures and the header untouched.
    std::size_t changed = 0;
    for (std::size_t i = 0; i < x.size(); ++i) changed += x[i] != tpf[i];
    for (const char* s : kSlots) {
        const TpfTexture* t = find(texs, s);
        std::size_t same = 0;
        for (std::size_t i = 0; i < 1024; ++i) same += x[t->offset + i] == 0xee;
        CHECK(same < 1024);
    }
    std::vector<bool> strip(512 * 1024 / 16 * 16, false);
    for (unsigned y = 30; y < 38; ++y)
        for (unsigned xx = 165; xx < 189; ++xx) strip[ps4_bc_block_offset(xx, y, 256) / 16] = true;
    bool outside = false;
    for (std::size_t blk = 0; blk < 1024 * 512 / 16; ++blk)
        if (!strip[blk])
            for (int k = 0; k < 16; ++k) outside |= x[atlas->offset + blk * 16 + k] != 0xee;
    CHECK(!outside);
    const TpfTexture* other = find(texs, "MENU_Common_00000");
    bool other_changed = false;
    for (int i = 0; i < other->width * other->height; ++i) other_changed |= x[other->offset + i] != tpf[other->offset + i];
    CHECK(!other_changed);
    CHECK(std::memcmp(x.data(), tpf.data(), 0x10) == 0);
    CHECK(changed > 0);

    // An Xbox disc is only its symbol over the game's Square: a corner block
    // (outside the symbol) is KG_R_L's as the file had it, before KG_R_L's own
    // symbol went in; a key cap keeps its own corner, the d-pad is whole.
    {
        GlyphBytes y = tpf;
        const TpfTexture *sq = find(texs, "KG_R_L"), *start = find(texs, "KG_Start"), *l1 = find(texs, "KG_L1"),
                         *pad = find(texs, "KG_L_U");
        for (int k = 0; k < 16; ++k) y[sq->offset + k] = static_cast<std::uint8_t>(0x50 + k);
        CHECK(prompt_glyphs_write_tpf(y, PromptStyle::Xbox, &why));
        CHECK(std::memcmp(y.data() + start->offset, y.data() + sq->offset, 16) == 0);
        CHECK(y[start->offset] == 0x50 && y[start->offset + 15] == 0x5f);
        bool l1_corner = true;
        for (int k = 0; k < 16; ++k) l1_corner &= y[l1->offset + k] == 0xee;
        CHECK(l1_corner);
        std::size_t pad_same = 0;
        for (std::size_t i = 0; i < 1024; ++i) pad_same += y[pad->offset + i] == 0xee;
        CHECK(pad_same < 64);
    }

    // Each style its own glyphs; PlayStation nothing.
    GlyphBytes d = tpf, p = tpf;
    CHECK(prompt_glyphs_write_tpf(d, PromptStyle::SteamDeck, &why));
    CHECK(d != x);
    CHECK(prompt_glyphs_write_tpf(p, PromptStyle::PlayStation, &why));
    CHECK(p == tpf);

    // The DCX: inflated, written and deflated with its header kept.
    const GlyphBytes dcx = make_dcx(tpf);
    const GlyphBytes out = prompt_glyphs_apply_dcx(dcx, PromptStyle::Xbox, &why);
    CHECK(!out.empty());
    if (!out.empty()) {
        CHECK(std::memcmp(out.data(), dcx.data(), 0x20) == 0);  // magic and the raw size
        GlyphBytes back(tpf.size());
        uLongf got = static_cast<uLongf>(back.size());
        CHECK(uncompress(back.data(), &got, out.data() + 0x4c, static_cast<uLong>(out.size() - 0x4c)) == Z_OK);
        CHECK(back == x);
    }
}

void refuses_other_files() {
    std::string why;
    // A prompt texture of another size (a prompt mod's): nothing written.
    std::vector<Tex> modded = common_like();
    modded[3].w = modded[3].h = 64;
    GlyphBytes m = make_tpf(modded);
    const GlyphBytes before = m;
    CHECK(!prompt_glyphs_write_tpf(m, PromptStyle::Xbox, &why));
    CHECK(m == before);
    CHECK(why.find("KG_R_L") != std::string::npos);
    // No atlas.
    std::vector<Tex> no_atlas = common_like();
    no_atlas.erase(no_atlas.end() - 2);
    GlyphBytes n = make_tpf(no_atlas);
    CHECK(!prompt_glyphs_write_tpf(n, PromptStyle::Xbox, &why));
    // Not a TPF, not a DCX.
    GlyphBytes junk(256, 0x41);
    CHECK(!prompt_glyphs_write_tpf(junk, PromptStyle::Xbox, &why));
    CHECK(prompt_glyphs_apply_dcx(junk, PromptStyle::Xbox, &why).empty());
}

void auto_choice() {
    // host/window.h PadFamily: None, PlayStation, Xbox, Steam Deck, Steam Controller, other.
    CHECK(prompt_style_auto(0, false) == PromptStyle::PlayStation);
    CHECK(prompt_style_auto(0, true) == PromptStyle::SteamDeck);
    CHECK(prompt_style_auto(1, true) == PromptStyle::PlayStation);
    CHECK(prompt_style_auto(2, false) == PromptStyle::Xbox);
    CHECK(prompt_style_auto(3, false) == PromptStyle::SteamDeck);
    CHECK(prompt_style_auto(4, false) == PromptStyle::SteamController);
    CHECK(prompt_style_auto(5, false) == PromptStyle::Xbox);
    // The setting: Auto, Xbox, Default (PS4). A chosen one wins over the pad.
    CHECK(prompt_style_choose(0, 2, false) == PromptStyle::Xbox);
    CHECK(prompt_style_choose(0, 3, false) == PromptStyle::SteamDeck);
    CHECK(prompt_style_choose(1, 1, true) == PromptStyle::Xbox);
    CHECK(prompt_style_choose(1, 0, false) == PromptStyle::Xbox);
    CHECK(prompt_style_choose(2, 2, false) == PromptStyle::PlayStation);
    CHECK(prompt_style_choose(2, 3, true) == PromptStyle::PlayStation);
    CHECK(prompt_style_choose(7, 2, false) == PromptStyle::Xbox);  // out of range: Auto
}

// The dump's own file, when there is one: every style goes in.
int dump_file() {
    const char* app0 = std::getenv("BBHOST_APP0");
    if (!app0 || !*app0) return 0;
    std::ifstream f(std::string(app0) + "/dvdroot_ps4/menu/common.tpf.dcx", std::ios::binary);
    if (!f) return 0;
    const GlyphBytes dcx((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (PromptStyle s : {PromptStyle::Xbox, PromptStyle::SteamDeck, PromptStyle::SteamController}) {
        std::string why;
        const GlyphBytes out = prompt_glyphs_apply_dcx(dcx, s, &why);
        if (out.empty()) std::printf("dump: %s: %s\n", prompt_style_name(s), why.c_str());
        CHECK(!out.empty());
    }
    return 1;
}

}  // namespace

int main() {
    block_order();
    write_and_round_trip();
    refuses_other_files();
    auto_choice();
    const int dump = dump_file();
    if (g_fail) return 1;
    std::printf("prompt_glyphs_test: ok%s\n", dump ? " (and the dump's common.tpf.dcx)" : "");
    return 0;
}
