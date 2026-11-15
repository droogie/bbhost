#include "host/game_font.h"

#if !defined(BBHOST_GAME_FONT_NO_FS)
#include "hle/fs.h"
#include "log.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>

namespace game_font {

namespace {

// Bits, most significant first, as SWF packs them.
struct Bits {
    const std::uint8_t* p;
    std::size_t size;  // bytes
    std::size_t bit = 0;
    bool bad = false;
    std::uint32_t ub(int n) {
        std::uint32_t v = 0;
        for (int i = 0; i < n; ++i, ++bit) {
            if ((bit >> 3) >= size) {
                bad = true;
                return 0;
            }
            v = (v << 1) | ((p[bit >> 3] >> (7 - (bit & 7))) & 1u);
        }
        return v;
    }
    std::int32_t sb(int n) {
        if (n <= 0) return 0;
        std::uint32_t v = ub(n);
        if (n < 32 && (v & (1u << (n - 1)))) v |= ~0u << n;
        return static_cast<std::int32_t>(v);
    }
    void align() { bit = (bit + 7) & ~std::size_t{7}; }
};

std::uint16_t u16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }
std::uint32_t u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | static_cast<std::uint32_t>(p[1]) << 8 | static_cast<std::uint32_t>(p[2]) << 16 |
           static_cast<std::uint32_t>(p[3]) << 24;
}

bool fail(std::string* why, const char* what) {
    if (why) *why = what;
    return false;
}

// A DefineFont2 or DefineFont3 tag's body into `out`.
bool parse_font(const std::uint8_t* b, std::size_t n, bool font3, Font& out, std::string* why) {
    if (n < 6) return fail(why, "a font tag too short");
    const std::uint8_t flags = b[2];
    const bool has_layout = flags & 0x80, wide_offsets = flags & 0x08, wide_codes = flags & 0x04;
    std::size_t at = 4;
    const std::size_t name_len = b[at++];
    if (at + name_len + 2 > n) return fail(why, "a font's name past its tag");
    out.name.assign(reinterpret_cast<const char*>(b + at), name_len);
    while (!out.name.empty() && out.name.back() == '\0') out.name.pop_back();
    at += name_len;
    const std::size_t count = u16(b + at);
    at += 2;
    const std::size_t table = at, step = wide_offsets ? 4 : 2;
    if (count == 0 || table + (count + 1) * step > n) return fail(why, "a font's offsets past its tag");
    std::vector<std::uint32_t> offs(count + 1);
    for (std::size_t i = 0; i <= count; ++i) offs[i] = wide_offsets ? u32(b + table + i * 4) : u16(b + table + i * 2);
    const std::size_t codes_at = table + offs[count];
    const std::size_t code_size = wide_codes ? 2 : 1;
    if (codes_at + count * code_size > n) return fail(why, "a font's codes past its tag");
    out.data.assign(b, b + n);
    out.glyphs.assign(count, Glyph{});
    out.codes.assign(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t s = static_cast<std::uint32_t>(table + offs[i]);
        const std::uint32_t e = static_cast<std::uint32_t>(i + 1 < count ? table + offs[i + 1] : codes_at);
        if (s > e || e > n) return fail(why, "a glyph's shape past its tag");
        out.glyphs[i].shape = s;
        out.glyphs[i].shape_end = e;
        out.codes[i] = wide_codes ? u16(b + codes_at + i * 2) : b[codes_at + i];
    }
    out.by_code.clear();
    for (std::size_t i = 0; i < count; ++i) out.by_code.emplace_back(out.codes[i], static_cast<int>(i));
    std::sort(out.by_code.begin(), out.by_code.end());
    out.units_per_em = font3 ? 20480.0f : 1024.0f;
    if (has_layout) {
        std::size_t l = codes_at + count * code_size;
        if (l + 6 + count * 2 > n) return fail(why, "a font's layout past its tag");
        out.ascent = static_cast<float>(u16(b + l));
        out.descent = static_cast<float>(u16(b + l + 2));
        l += 6;
        for (std::size_t i = 0; i < count; ++i) out.glyphs[i].advance = static_cast<float>(static_cast<std::int16_t>(u16(b + l + i * 2)));
    } else {
        out.ascent = out.units_per_em * 0.88f;
        out.descent = out.units_per_em * 0.12f;
        for (Glyph& g : out.glyphs) g.advance = out.units_per_em * 0.5f;
    }
    return true;
}

}  // namespace

int Font::index_of(std::uint32_t code) const {
    if (by_code.size() != codes.size()) {
        for (std::size_t i = 0; i < codes.size(); ++i) {
            if (codes[i] == code) return static_cast<int>(i);
        }
        return -1;
    }
    const auto it = std::lower_bound(by_code.begin(), by_code.end(), std::make_pair(code, 0),
                                     [](const std::pair<std::uint32_t, int>& a, const std::pair<std::uint32_t, int>& b) { return a.first < b.first; });
    return it != by_code.end() && it->first == code ? it->second : -1;
}

bool parse(const std::vector<std::uint8_t>& file, Font& out, std::string* why) {
    if (file.size() < 9) return fail(why, "too short");
    const char s0 = static_cast<char>(file[0]);
    if (!((s0 == 'G' || s0 == 'F') && file[1] == 'F' && file[2] == 'X') && !(s0 == 'F' && file[1] == 'W' && file[2] == 'S')) {
        if ((s0 == 'C' && file[1] == 'F' && file[2] == 'X') || (s0 == 'C' && file[1] == 'W' && file[2] == 'S'))
            return fail(why, "compressed (the dump's are not)");
        return fail(why, "not a GFX or SWF file");
    }
    const std::uint8_t* b = file.data() + 8;
    const std::size_t n = file.size() - 8;
    // The stage's RECT, the frame rate and count.
    Bits r{b, n};
    const int nbits = static_cast<int>(r.ub(5));
    r.bit += static_cast<std::size_t>(nbits) * 4;
    r.align();
    std::size_t at = (r.bit >> 3) + 4;
    while (at + 2 <= n) {
        const std::uint16_t head = u16(b + at);
        const int code = head >> 6;
        std::size_t len = head & 0x3f;
        at += 2;
        if (len == 0x3f) {
            if (at + 4 > n) break;
            len = u32(b + at);
            at += 4;
        }
        if (at + len > n) return fail(why, "a tag past the file");
        if (code == 75 || code == 48) return parse_font(b + at, len, code == 75, out, why);
        if (code == 0) break;
        at += len;
    }
    return fail(why, "no font in it");
}

bool outline(const Font& font, int glyph, std::vector<Contour>& out, float tolerance) {
    out.clear();
    if (glyph < 0 || static_cast<std::size_t>(glyph) >= font.glyphs.size()) return false;
    const Glyph& g = font.glyphs[static_cast<std::size_t>(glyph)];
    Bits r{font.data.data() + g.shape, g.shape_end - g.shape};
    const int fill_bits = static_cast<int>(r.ub(4)), line_bits = static_cast<int>(r.ub(4));
    float x = 0, y = 0;
    Contour cur;
    const auto close = [&]() {
        if (cur.size() >= 6) out.push_back(cur);
        cur.clear();
    };
    for (int guard = 0; guard < 100000 && !r.bad; ++guard) {
        if (r.ub(1) == 0) {
            // A style change, or the end.
            const std::uint32_t f = r.ub(5);
            if (f == 0) break;
            if (f & 1) {  // move to
                close();
                const int mb = static_cast<int>(r.ub(5));
                x = static_cast<float>(r.sb(mb));
                y = static_cast<float>(r.sb(mb));
                cur.push_back(x);
                cur.push_back(y);
            }
            if (f & 2) r.ub(fill_bits);
            if (f & 4) r.ub(fill_bits);
            if (f & 8) r.ub(line_bits);
            if (f & 16) return false;  // new styles: not in a glyph
            continue;
        }
        if (cur.empty()) {
            cur.push_back(x);
            cur.push_back(y);
        }
        if (r.ub(1) == 1) {  // straight
            const int nb = static_cast<int>(r.ub(4)) + 2;
            float dx = 0, dy = 0;
            if (r.ub(1)) {
                dx = static_cast<float>(r.sb(nb));
                dy = static_cast<float>(r.sb(nb));
            } else if (r.ub(1)) {
                dy = static_cast<float>(r.sb(nb));
            } else {
                dx = static_cast<float>(r.sb(nb));
            }
            x += dx;
            y += dy;
            cur.push_back(x);
            cur.push_back(y);
        } else {  // a quadratic curve
            const int nb = static_cast<int>(r.ub(4)) + 2;
            const float cx = x + static_cast<float>(r.sb(nb)), cy = y + static_cast<float>(r.sb(nb));
            const float ax = cx + static_cast<float>(r.sb(nb)), ay = cy + static_cast<float>(r.sb(nb));
            const float ddx = x - 2 * cx + ax, ddy = y - 2 * cy + ay;
            const float dd = std::sqrt(ddx * ddx + ddy * ddy);
            const int steps = std::clamp(static_cast<int>(std::ceil(std::sqrt(dd / std::max(tolerance, 1e-3f) * 0.25f))), 1, 32);
            for (int i = 1; i <= steps; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(steps), u = 1.0f - t;
                cur.push_back(u * u * x + 2 * u * t * cx + t * t * ax);
                cur.push_back(u * u * y + 2 * u * t * cy + t * t * ay);
            }
            x = ax;
            y = ay;
        }
    }
    close();
    return !r.bad;
}

void rasterise(const std::vector<Contour>& contours, float scale, float ox, float oy, int w, int h, std::uint8_t* out) {
    std::fill(out, out + static_cast<std::size_t>(w) * static_cast<std::size_t>(h), std::uint8_t{0});
    if (w <= 0 || h <= 0) return;
    struct Edge {
        float x0, y0, x1, y1;
        int dir;
    };
    std::vector<Edge> edges;
    for (const Contour& c : contours) {
        const std::size_t pts = c.size() / 2;
        for (std::size_t i = 0; i < pts; ++i) {
            const std::size_t j = (i + 1) % pts;
            const float x0 = c[2 * i] * scale + ox, y0 = c[2 * i + 1] * scale + oy;
            const float x1 = c[2 * j] * scale + ox, y1 = c[2 * j + 1] * scale + oy;
            if (y0 == y1) continue;
            edges.push_back(y0 < y1 ? Edge{x0, y0, x1, y1, 1} : Edge{x1, y1, x0, y0, -1});
        }
    }
    constexpr int kSub = 4;  // 4 x 4 samples a pixel
    std::vector<int> count(static_cast<std::size_t>(w));
    std::vector<std::pair<float, int>> xs;
    for (int py = 0; py < h; ++py) {
        std::fill(count.begin(), count.end(), 0);
        for (int sy = 0; sy < kSub; ++sy) {
            const float y = static_cast<float>(py) + (static_cast<float>(sy) + 0.5f) / kSub;
            xs.clear();
            for (const Edge& e : edges) {
                if (y < e.y0 || y >= e.y1) continue;
                xs.emplace_back(e.x0 + (y - e.y0) * (e.x1 - e.x0) / (e.y1 - e.y0), e.dir);
            }
            std::sort(xs.begin(), xs.end());
            int wind = 0;
            for (std::size_t k = 0; k + 1 < xs.size(); ++k) {
                wind += xs[k].second;
                if (wind == 0) continue;
                // Samples at x = px + (sx + 0.5) / kSub inside [a, b).
                const float a = xs[k].first, b = xs[k + 1].first;
                const int first = std::max(0, static_cast<int>(std::ceil(a * kSub - 0.5f)));
                const int last = std::min(w * kSub - 1, static_cast<int>(std::ceil(b * kSub - 0.5f)) - 1);
                for (int s = first; s <= last; ++s) ++count[static_cast<std::size_t>(s / kSub)];
            }
        }
        for (int px = 0; px < w; ++px)
            out[static_cast<std::size_t>(py) * static_cast<std::size_t>(w) + static_cast<std::size_t>(px)] =
                static_cast<std::uint8_t>(std::min(255, count[static_cast<std::size_t>(px)] * 255 / (kSub * kSub)));
    }
}

#if !defined(BBHOST_GAME_FONT_NO_FS)
const Font* menu_font() {
    static std::once_flag once;
    static Font font;
    static bool ok = false;
    std::call_once(once, [] {
        static const char* const kLanguages[] = {"engus", "enggb", "frafr", "deude", "itait", "spaes", "jpnjp"};
        for (const char* lang : kLanguages) {
            const std::string path = hle_fs_map_path_base((std::string("/app0/dvdroot_ps4/menu/") + lang + "/font.gfx").c_str());
            std::ifstream f(path, std::ios::binary);
            if (!f) continue;
            std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            std::string why;
            if (parse(bytes, font, &why)) {
                ok = true;
                host_log("menu: the game's menu font from menu/%s/font.gfx: %zu glyphs, ascent %g, descent %g", lang,
                         font.glyphs.size(), static_cast<double>(font.ascent), static_cast<double>(font.descent));
                return;
            }
            host_log("menu: menu/%s/font.gfx is not a font this reads (%s)", lang, why.c_str());
        }
        host_log("menu: no menu font in the dump; a plugin's panel uses the system's serif");
    });
    return ok ? &font : nullptr;
}
#else
const Font* menu_font() { return nullptr; }  // (the test reads the file itself)
#endif

}  // namespace game_font

#if defined(BBHOST_HAVE_SDL3)

#include "imgui.h"
#include "imgui_internal.h"

namespace {

// ImGui's sizes are a line's height, ascent and descent (stb_truetype's
// ScaleForPixelHeight): a font unit at `size` pixels.
float unit_px(const game_font::Font& f, float size) { return size / (f.ascent + f.descent); }

bool src_init(ImFontAtlas*, ImFontConfig* src) {
    src->FontLoaderData = src->FontData;  // the game_font::Font, which outlives the atlas
    return true;
}

void src_destroy(ImFontAtlas*, ImFontConfig* src) { src->FontLoaderData = nullptr; }

bool src_contains(ImFontAtlas*, ImFontConfig* src, ImWchar c) {
    return static_cast<const game_font::Font*>(src->FontLoaderData)->index_of(c) >= 0;
}

bool baked_init(ImFontAtlas*, ImFontConfig* src, ImFontBaked* baked, void*) {
    const auto& f = *static_cast<const game_font::Font*>(src->FontLoaderData);
    if (!src->MergeMode) {
        const float k = unit_px(f, baked->Size);
        baked->Ascent = ImCeil(f.ascent * k);
        baked->Descent = ImFloor(-f.descent * k);
    }
    return true;
}

bool load_glyph(ImFontAtlas* atlas, ImFontConfig* src, ImFontBaked* baked, void*, ImWchar codepoint, ImFontGlyph* out_glyph,
                float* out_advance_x) {
    const auto& f = *static_cast<const game_font::Font*>(src->FontLoaderData);
    const int gi = f.index_of(codepoint);
    if (gi < 0) return false;
    const float k = unit_px(f, baked->Size);
    const float advance = f.glyphs[static_cast<std::size_t>(gi)].advance * k;
    if (out_advance_x) {
        *out_advance_x = advance;
        return true;
    }
    out_glyph->Codepoint = codepoint;
    out_glyph->AdvanceX = advance;
    const float density = src->RasterizerDensity * baked->RasterizerDensity, kr = k * density;
    std::vector<game_font::Contour> contours;
    if (!game_font::outline(f, gi, contours, 0.2f / kr) || contours.empty()) return true;  // a space
    float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
    for (const game_font::Contour& c : contours) {
        for (std::size_t i = 0; i + 1 < c.size(); i += 2) {
            x0 = std::min(x0, c[i]);
            x1 = std::max(x1, c[i]);
            y0 = std::min(y0, c[i + 1]);
            y1 = std::max(y1, c[i + 1]);
        }
    }
    const int bx0 = static_cast<int>(std::floor(x0 * kr)), by0 = static_cast<int>(std::floor(y0 * kr));
    const int w = static_cast<int>(std::ceil(x1 * kr)) - bx0 + 1, h = static_cast<int>(std::ceil(y1 * kr)) - by0 + 1;
    if (w <= 0 || h <= 0 || w > 1024 || h > 1024) return true;
    const ImFontAtlasRectId id = ImFontAtlasPackAddRect(atlas, w, h);
    if (id == ImFontAtlasRectId_Invalid) return false;
    ImTextureRect* r = ImFontAtlasPackGetRect(atlas, id);
    std::vector<std::uint8_t> px(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    game_font::rasterise(contours, kr, static_cast<float>(-bx0), static_cast<float>(-by0), w, h, px.data());
    const float recip = 1.0f / density, top = IM_ROUND(baked->Ascent);
    out_glyph->X0 = static_cast<float>(bx0) * recip;
    out_glyph->Y0 = static_cast<float>(by0) * recip + top;
    out_glyph->X1 = static_cast<float>(bx0 + static_cast<int>(r->w)) * recip;
    out_glyph->Y1 = static_cast<float>(by0 + static_cast<int>(r->h)) * recip + top;
    out_glyph->Visible = true;
    out_glyph->PackId = id;
    ImFontAtlasBakedSetFontGlyphBitmap(atlas, baked, src, out_glyph, r, px.data(), ImTextureFormat_Alpha8, w);
    return true;
}

}  // namespace

ImFont* game_font_add_to_imgui() {
    const game_font::Font* f = game_font::menu_font();
    if (!f) return nullptr;
    static ImFontLoader loader;
    loader.Name = "the game's menu font";
    loader.FontSrcInit = &src_init;
    loader.FontSrcDestroy = &src_destroy;
    loader.FontSrcContainsGlyph = &src_contains;
    loader.FontBakedInit = &baked_init;
    loader.FontBakedLoadGlyph = &load_glyph;
    ImFontConfig cfg;
    cfg.FontData = const_cast<game_font::Font*>(f);
    cfg.FontDataSize = static_cast<int>(sizeof(*f));
    cfg.FontDataOwnedByAtlas = false;
    cfg.FontLoader = &loader;
    std::snprintf(cfg.Name, sizeof(cfg.Name), "%s", "the game's menu font");
    return ImGui::GetIO().Fonts->AddFont(&cfg);
}

#else

ImFont* game_font_add_to_imgui() { return nullptr; }

#endif
