// host/game_font.h on the dump's own menu font (menu/engus/font.gfx under
// BBHOST_APP0): the DefineFont3 found and read whole, its letters' outlines
// closed and inside their advance, and the rasteriser's coverage where a
// letter's ink is - an O's ring filled and its middle empty (the non-zero rule
// keeps a hole a hole), an l a narrow upright bar. Skipped (77) without the dump.
#include "host/game_font.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

int main() {
    const char* app0 = std::getenv("BBHOST_APP0");
    std::vector<std::uint8_t> bytes;
    if (app0 && *app0) {
        for (const char* rel : {"/dvdroot_ps4/menu/engus/font.gfx", "/menu/engus/font.gfx"}) {
            std::ifstream f(std::string(app0) + rel, std::ios::binary);
            if (!f) continue;
            bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            break;
        }
    }
    if (bytes.empty()) {
        std::printf("game_font_test skipped: no menu font (BBHOST_APP0)\n");
        return 77;
    }
    // A file that is not one, and one cut short.
    {
        game_font::Font none;
        std::string why;
        CHECK(!game_font::parse(std::vector<std::uint8_t>(64, 0), none, &why) && !why.empty());
        std::vector<std::uint8_t> cut(bytes.begin(), bytes.begin() + 4096);
        CHECK(!game_font::parse(cut, none, &why));
    }
    game_font::Font font;
    std::string why;
    CHECK(game_font::parse(bytes, font, &why));
    if (g_failures) {
        std::fprintf(stderr, "game_font_test: %s\n", why.c_str());
        return 1;
    }
    std::printf("font \"%s\": %zu glyphs, ascent %g, descent %g\n", font.name.c_str(), font.glyphs.size(), font.ascent, font.descent);
    CHECK(font.glyphs.size() > 1000);
    CHECK(font.ascent > 0 && font.descent > 0 && font.ascent > font.descent);
    // Every printable ASCII letter is there, its outline closed and drawn
    // inside the line, its advance positive.
    for (char c = '!'; c <= '~'; ++c) {
        const int g = font.index_of(static_cast<unsigned char>(c));
        CHECK(g >= 0);
        if (g < 0) continue;
        std::vector<game_font::Contour> cs;
        CHECK(game_font::outline(font, g, cs, 4.0f) && !cs.empty());
        CHECK(font.glyphs[static_cast<std::size_t>(g)].advance > 0);
        // y down from the baseline: the tallest letters near the ascent, the
        // descenders (g, y: 5,220 units) well past the font's own descent (2,460).
        for (const game_font::Contour& k : cs) {
            for (std::size_t i = 0; i + 1 < k.size(); i += 2) {
                CHECK(k[i + 1] > -font.ascent * 1.2f && k[i + 1] < font.ascent * 0.45f);
            }
        }
    }
    CHECK(font.index_of(' ') >= 0);
    CHECK(font.index_of(0x10ffff) < 0);
    // 'O' at 64 pixels a line: ink on its ring, none at its middle.
    const auto draw = [&](char c, int size, std::vector<std::uint8_t>& px, int& w, int& h) {
        const float k = static_cast<float>(size) / (font.ascent + font.descent);
        std::vector<game_font::Contour> cs;
        game_font::outline(font, font.index_of(static_cast<unsigned char>(c)), cs, 0.2f / k);
        w = size;
        h = size;
        px.assign(static_cast<std::size_t>(w * h), 0);
        game_font::rasterise(cs, k, 0.0f, font.ascent * k, w, h, px.data());
    };
    std::vector<std::uint8_t> px;
    int w = 0, h = 0;
    draw('O', 64, px, w, h);
    long ink = 0;
    for (std::uint8_t v : px) ink += v;
    CHECK(ink > 255 * 100);
    // Its middle: the box of the ink, its centre empty.
    int x0 = w, x1 = -1, y0 = h, y1 = -1;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (px[static_cast<std::size_t>(y * w + x)] > 128) {
                x0 = std::min(x0, x);
                x1 = std::max(x1, x);
                y0 = std::min(y0, y);
                y1 = std::max(y1, y);
            }
        }
    }
    CHECK(x1 > x0 + 10 && y1 > y0 + 10);
    if (x1 > x0 && y1 > y0) {
        CHECK(px[static_cast<std::size_t>(((y0 + y1) / 2) * w + (x0 + x1) / 2)] < 32);
        CHECK(px[static_cast<std::size_t>(((y0 + y1) / 2) * w + x0 + 1)] > 128 ||
              px[static_cast<std::size_t>(((y0 + y1) / 2) * w + x0)] > 128);
    }
    // 'l': taller than wide.
    draw('l', 64, px, w, h);
    x0 = w, x1 = -1, y0 = h, y1 = -1;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (px[static_cast<std::size_t>(y * w + x)] > 128) {
                x0 = std::min(x0, x);
                x1 = std::max(x1, x);
                y0 = std::min(y0, y);
                y1 = std::max(y1, y);
            }
        }
    }
    CHECK(y1 - y0 > 3 * (x1 - x0));
    if (g_failures) {
        std::fprintf(stderr, "game_font_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("game_font_test: ok\n");
    return 0;
}
