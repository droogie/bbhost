// The sizes of render targets (host/rt_extent.h), without the game or a GPU.
// The sizes learned from scissors: the 2560x1440 glare - YEBIS's other glare
// pyramid learning 523x441 at an address where its glare then draws a 640x360
// level with a scissor of its whole pitch - and back again; the
// 1920x1080 case a larger scissor fixed (392 learned in a 512 pitch, then
// 480); a scissor the padded size on first sight; a view's size found by a
// draw with no view; a scissor the padded size over a smaller size learned for
// the same geometry, which stays and is flagged for the census. Then the rule
// as a property over a grid of resolutions, three pitch alignments and two
// row alignments, YEBIS-shaped targets (the glare's levels, the other
// pyramid's) sharing one address in every order: each target drawn with its
// own size as its scissor is that size wherever its padding lets a scissor be
// believed, its padded size where not, whatever was drawn there before. The
// registry's part: a YEBIS texture read from its object as YEBIS's creator
// lays it out, a destroyed one read as no texture, which registered texture
// sizes which draw, and target_image's order - a registered size over a
// learned one, and a texture gone leaves nothing behind. The same cases by
// the address alone (BBHOST_RT_LEARN_BY_GEOMETRY=0, the rule before) must
// fail, and eight deliberate mistakes must each be caught.
#define RT_EXTENT_MUTANTS
#include "host/rt_extent.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

int g_rt_extent_mutant = 0;

namespace {

int g_failed = 0;
bool g_quiet = false;  // a run that is meant to fail says nothing of each check
void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failed;
        if (!g_quiet && !g_rt_extent_mutant) std::printf("FAIL: %s\n", what);
    }
}

using rt_extent::Answer;
using rt_extent::Table;

// Two of the glare's addresses at 2560x1440 (0x1695b0000 the bright pass's
// target, 0x162180000 the first blur's).
constexpr std::uint64_t kBright = 0x1695b0000ull, kBlur = 0x162180000ull;

void all_cases(bool by_geometry) {
    check(rt_extent::plausible(392, 512) && rt_extent::plausible(480, 512) && rt_extent::plausible(523, 640) &&
              rt_extent::plausible(1080, 1088) && rt_extent::plausible(441, 448),
          "sizes their padding comes from are plausible");
    check(!rt_extent::plausible(0, 64) && !rt_extent::plausible(700, 640) && !rt_extent::plausible(300, 360) &&
              !rt_extent::plausible(100, 640) && !rt_extent::plausible(360, 360),
          "sizes it cannot come from are not");

    // The bug: one address, the pyramid's level (pitch 640, 448 rows) and the glare's (pitch 640, 384 rows, as at Mensis).
    {
        Table t(by_geometry);
        Answer a = t.draw(kBright, 640, 448, 523, 441);
        check(a.w == 523 && a.h == 441 && a.changed, "the pyramid's level learns 523x441 from its scissor");
        a = t.draw(kBright, 640, 384, 640, 360);
        check(a.w == 640 && a.h == 360, "the glare's 640x360 level at that address, its scissor its whole pitch, is 640x360");
        check(!a.padded_over_w && !a.padded_over_h, "nothing was learned for the glare's geometry, so its scissor is over nothing");
        a = t.draw(kBright, 640, 384, 0, 0);
        check(a.w == 640 && a.h == 360, "a clear of the glare's level (no scissor) is 640x360");
        a = t.draw(kBright, 640, 448, 0, 0);
        check(a.w == 523 && a.h == 441 && !a.changed, "back in the pyramid's geometry, a clear finds 523x441 still known");
        a = t.draw(kBright, 640, 448, 523, 441);
        check(a.w == 523 && a.h == 441 && !a.changed, "and the pyramid's draw too, learning nothing new");
        a = t.draw(kBright, 640, 384, 640, 360);
        check(a.w == 640 && a.h == 360 && !a.changed, "and the glare's level after it is 640x360 again");
    }

    // The 1920x1080 case: 392 learned in a 512 pitch, then the level's own scissor 480 in the same geometry.
    {
        Table t(by_geometry);
        Answer a = t.draw(kBlur, 512, 384, 392, 331);
        check(a.w == 392 && a.h == 331 && a.changed, "392x331 learned in a 512 pitch");
        a = t.draw(kBlur, 512, 384, 480, 270);
        check(a.w == 480 && a.changed, "a larger plausible scissor (480) replaces it: a pass cannot draw past its target");
        check(a.h == 331, "a smaller one (270 rows) does not shrink it");
        a = t.draw(kBlur, 512, 384, 392, 331);
        check(a.w == 480 && a.h == 331 && !a.changed, "nor does the smaller pass that drew first, coming again");
    }

    // A scissor the padded size on first sight: the padded size, nothing stored.
    {
        Table t(by_geometry);
        Answer a = t.draw(0x170000000ull, 640, 360, 640, 360);
        check(a.w == 640 && a.h == 360 && !a.changed && t.size() == 0, "a scissor the padded size on first sight: the padded size, nothing stored");
        a = t.draw(0x170000000ull, 640, 360, 600, 300);
        check(a.w == 600 && a.changed, "a smaller plausible scissor after it still teaches the width");
        check(a.h == 360, "and a height the padding cannot come from (300 of 360 rows) is not believed");
    }

    // A view's texture says the size: a draw with no view of the same geometry takes it, one of another does not.
    {
        Table t(by_geometry);
        check(t.view(kBright, 640, 448, 523, 441), "a view's texture says 523x441");
        check(!t.view(kBright, 640, 448, 523, 441), "saying it again changes nothing");
        Answer a = t.draw(kBright, 640, 448, 640, 448);
        check(a.w == 523 && a.h == 441, "a draw with no view in that geometry makes it 523x441");
        a = t.draw(kBright, 640, 384, 640, 360);
        check(a.w == 640 && a.h == 360, "a draw with no view in another geometry at the address does not");
        t.view(0x180000000ull, 640, 360, 100, 50);
        a = t.draw(0x180000000ull, 640, 360, 640, 360);
        check(a.w == 640 && a.h == 360, "a view's size the padding cannot come from (100x50 of 640x360) does not size a draw with no view");
    }

    // What is left: one geometry, a scissor the padded size over a smaller learned size. It stays, flagged.
    {
        Table t(by_geometry);
        t.draw(kBright, 640, 448, 523, 441);
        Answer a = t.draw(kBright, 640, 448, 640, 448);
        check(a.w == 523 && a.h == 441 && !a.changed, "a scissor the padded size does not replace a size learned in its geometry");
        check(a.padded_over_w && a.padded_over_h && a.known_w == 523 && a.known_h == 441,
              "and is flagged for the census, with what was known");
        a = t.draw(kBright, 640, 448, 523, 441);
        check(!a.padded_over_w && !a.padded_over_h, "the target's own scissor is not flagged");
    }
}

// --- the rule as a property over resolutions and reuse orders ----------------

std::uint32_t align(std::uint32_t v, std::uint32_t a) { return (v + a - 1) / a * a; }
struct Target {
    std::uint32_t w, h, pitch, padded_h;
};
// What a target drawn with its own size as its scissor must be: that size where
// its padding lets the scissor be believed, its padded size where not (a
// scissor the padded size teaches nothing).
std::uint32_t expected(std::uint32_t v, std::uint32_t padded) { return v == padded || rt_extent::plausible(v, padded) ? v : padded; }

constexpr std::uint32_t kGrid[][2] = {{1280, 720},  {1280, 800},  {1366, 768},  {1600, 900},  {1920, 1080}, {2560, 1080},
                                      {2560, 1440}, {3440, 1440}, {3840, 2160}, {960, 540},   {1024, 768},  {1680, 1050},
                                      {1920, 1200}, {2048, 1152}, {3200, 1800}, {5120, 2160}, {5120, 2880}};
int g_cases = 0, g_shared_geometry = 0;  // orders checked; target sets two of whose targets share a geometry (left out)

// The answers the rule gives against expected(), over every order: the number that differ.
int property(bool by_geometry, bool count_cases) {
    int bad = 0, said = 0;
    if (count_cases) g_cases = g_shared_geometry = 0;
    for (const auto& r : kGrid) {
        for (std::uint32_t aw : {64u, 128u, 256u}) {
            for (std::uint32_t ah : {8u, 64u}) {
                // The glare's first three levels and the other pyramid's first two, YEBIS-shaped.
                std::vector<Target> t;
                for (std::uint32_t k : {4u, 8u, 16u}) {
                    const std::uint32_t w = (r[0] + k - 1) / k, h = (r[1] + k - 1) / k;
                    t.push_back({w, h, align(w, aw), align(h, ah)});
                }
                for (std::uint32_t k : {1u, 2u}) {
                    const std::uint32_t w = (r[0] * 523 + 1280) / 2560 / k, h = (r[1] * 441 + 720) / 1440 / k;
                    t.push_back({w, h, align(w, aw), align(h, ah)});
                }
                // Every pair and triple of them on one address, in every order, the order twice (A B A B).
                const int n = static_cast<int>(t.size());
                for (int mask = 0; mask < (1 << n); ++mask) {
                    const int count = __builtin_popcount(mask);
                    if (count < 2 || count > 3) continue;
                    std::vector<int> set;
                    for (int i = 0; i < n; ++i)
                        if (mask & (1 << i)) set.push_back(i);
                    bool shared = false;
                    for (int a : set)
                        for (int b : set)
                            if (a < b && t[a].pitch == t[b].pitch && t[a].padded_h == t[b].padded_h &&
                                (t[a].w != t[b].w || t[a].h != t[b].h))
                                shared = true;
                    if (shared) {  // two targets the registers cannot tell apart: the case left (padded_over)
                        if (count_cases) ++g_shared_geometry;
                        continue;
                    }
                    std::sort(set.begin(), set.end());
                    do {
                        if (count_cases) ++g_cases;
                        Table table(by_geometry);
                        for (int pass = 0; pass < 2; ++pass) {
                            for (int i : set) {
                                const Target& x = t[i];
                                const std::uint32_t ew = expected(x.w, x.pitch), eh = expected(x.h, x.padded_h);
                                const Answer own = table.draw(kBright, x.pitch, x.padded_h, x.w, x.h);
                                const Answer clear = table.draw(kBright, x.pitch, x.padded_h, 0, 0);
                                const Answer whole = table.draw(kBright, x.pitch, x.padded_h, x.pitch, x.padded_h);
                                for (const Answer& a : {own, clear, whole}) {
                                    if (a.w == ew && a.h == eh) continue;
                                    ++bad;
                                    if (!g_quiet && !g_rt_extent_mutant && said++ < 6)
                                        std::printf("FAIL: %ux%u: a %ux%u target (padded %ux%u) is %ux%u\n", r[0], r[1], x.w, x.h, x.pitch,
                                                    x.padded_h, a.w, a.h);
                                }
                            }
                        }
                    } while (std::next_permutation(set.begin(), set.end()));
                }
            }
        }
    }
    return bad;
}

void property_cases(bool by_geometry) {
    const int bad = property(by_geometry, false);
    check(bad == 0, "every YEBIS-shaped target on a shared address is its own size, in every order, at every resolution");
}

// --- the registry's YEBIS textures -----------------------------------------------

constexpr std::uint64_t kVtable = 0x570bc00 + 0x10000000ull, kBaseVtable = 0x570bc80 + 0x10000000ull;

// A YEBIS 2D texture's T# (+0xc4) as its creator leaves it (0x15df840): on
// `memory`, the description's size, the surface's pitch.
void yebis_tsharp(std::uint32_t (&t)[8], std::uint64_t memory, std::uint32_t width, std::uint32_t height, std::uint32_t pitch) {
    for (std::uint32_t& x : t) x = 0;
    t[0] = static_cast<std::uint32_t>(memory >> 8);
    t[1] = 0x1c600000;  // (format bits, as a T# of the glare's has them)
    t[2] = (width - 1) | (height - 1) << 14;
    t[3] = 0x94e003ac;
    t[4] = (pitch - 1) << 13;
}

void registry_cases(bool by_geometry) {
    // As YEBIS's creator lays a texture out; the glare's T# from a capture: 640x360, pitch 640.
    const std::uint32_t captured[8] = {0x01621800, 0x1c600000, 0x0059c27f, 0x94e003ac, 0x004fe000, 0, 0, 0};
    rt_extent::YebisTexture t = rt_extent::yebis_texture(captured);
    check(t.memory == kBlur && t.w == 640 && t.h == 360 && t.pitch == 640, "the glare's captured T#: 640x360 on its memory, pitch 640");
    std::uint32_t w[8];
    yebis_tsharp(w, kBlur, 523, 441, 640);
    t = rt_extent::yebis_texture(w);
    check(t.w == 523 && t.h == 441 && t.pitch == 640, "the other pyramid's level: 523x441 in a pitch of 640");
    // Live, destroyed, another's.
    check(rt_extent::yebis_live(kVtable, kVtable, t, kBlur), "the live texture is read as live");
    rt_extent::YebisTexture freed = t;
    freed.memory = 0;  // the destructor's free zeroes the T#'s base (0x2e4b230)
    check(!rt_extent::yebis_live(kBaseVtable, kVtable, freed, kBlur), "a destroyed texture is not");
    check(!rt_extent::yebis_live(kVtable, kVtable, freed, kBlur), "nor one whose T# no longer has its memory, its vtable as it was");
    check(!rt_extent::yebis_live(kVtable, kVtable, t, kBright), "nor another texture's object at the address");

    // Which registered texture sizes which draw.
    const rt_extent::Registered glare{640, 360, true, 640}, pyramid{523, 441, true, 640}, gx{523, 441, false, 0};
    check(rt_extent::registered_fits(glare, 640, 384) && rt_extent::registered_fits(pyramid, 640, 448),
          "a YEBIS texture sizes a draw into its own surface");
    check(!rt_extent::registered_fits(pyramid, 640, 384), "the pyramid's texture does not size the glare's draw at its address");
    check(!rt_extent::registered_fits(glare, 640, 448), "nor the glare's texture the pyramid's draw: 448 rows are not 360 rows padded");
    check(!rt_extent::registered_fits(glare, 768, 384), "nor a draw into a surface of another pitch");
    const rt_extent::Registered level2{320, 180, true, 384};  // a smaller texture made on the memory of a live 640x360 one
    check(!rt_extent::registered_fits(level2, 640, 360), "a smaller YEBIS texture on the memory sizes no draw into the larger one's target");
    check(rt_extent::registered_fits(rt_extent::Registered{1920, 1080, true, 1920}, 1920, 1088) &&
              rt_extent::registered_fits(rt_extent::Registered{480, 270, true, 512}, 512, 272) &&
              rt_extent::registered_fits(rt_extent::Registered{236, 199, true, 256}, 256, 256),
          "rows padded to 8, 64 or 256 are the surface's");
    check(rt_extent::registered_fits(gx, 640, 448) && !rt_extent::registered_fits(gx, 512, 448), "a GX texture sizes what it fits");
    check(!rt_extent::registered_fits(rt_extent::Registered{}, 640, 360), "no texture sizes nothing");

    // target_image's order: the registry's texture first; one it does not size, or none, the learned sizes - and
    // a YEBIS texture's size is not kept for after it.
    const auto size = [](Table& table, const rt_extent::Registered* r, std::uint32_t pitch, std::uint32_t padded_h, std::uint32_t right,
                         std::uint32_t bottom) {
        if (r && rt_extent::registered_fits(*r, pitch, padded_h)) {
            if (!r->yebis) table.view(kBright, pitch, padded_h, r->w, r->h);
            return std::make_pair(r->w, r->h);
        }
        const Answer a = table.draw(kBright, pitch, padded_h, right, bottom);
        return std::make_pair(a.w, a.h);
    };
    Table table(by_geometry);
    table.draw(kBright, 640, 360, 523, 360);  // a size learned in the glare's geometry, a target of 523 drawn there
    check(size(table, &glare, 640, 360, 640, 360) == std::make_pair(640u, 360u), "a registered size wins over a learned one");
    check(size(table, &glare, 640, 360, 0, 0) == std::make_pair(640u, 360u), "for a clear too");
    // The texture destroyed (the registry has nothing there): the next target drawn there is sized from what was
    // drawn, never from the dead texture's size.
    check(size(table, nullptr, 640, 360, 523, 360) == std::make_pair(523u, 360u),
          "the texture gone, a 523-wide target drawn there is 523 wide, not the dead texture's 640");
}

// The rule before, as the owner's log shows it: the glare's level 523x360.
void the_rule_before() {
    Table t(false);
    t.draw(kBright, 640, 448, 523, 441);
    const Answer a = t.draw(kBright, 640, 384, 640, 360);
    check(a.w == 523 && a.h == 360, "by the address alone the glare's level is 523x360, as before");
}

}  // namespace

int main() {
    all_cases(true);
    property_cases(true);
    registry_cases(true);
    the_rule_before();
    std::printf("property: %d orders of YEBIS-shaped targets on one address over %zu resolutions (%d sets with two targets of one "
                "geometry left out)\n",
                (property(true, true), g_cases), sizeof(kGrid) / sizeof(kGrid[0]), g_shared_geometry);
    int failed = g_failed;
    g_failed = 0;
    g_quiet = true;
    const int before = property(false, false);
    all_cases(false);
    property_cases(false);
    g_quiet = false;
    std::printf("the rule by the address alone: %d answers of the property's wrong\n", before);
    if (g_failed == 0) {
        std::printf("FAIL: the rule by the address alone (BBHOST_RT_LEARN_BY_GEOMETRY=0) passes the cases\n");
        ++failed;
    } else {
        std::printf("the rule by the address alone caught: %d checks failed\n", g_failed);
    }
    static const char* const kMistakes[8] = {"the padded height left out of the key",
                                             "a scissor the padded size replaces a smaller learned size",
                                             "a scissor the padded size learned on first sight",
                                             "a learned size used where its padding cannot come from",
                                             "a larger scissor never replaces a learned size",
                                             "a scissor the padded size over a learned size not flagged",
                                             "a YEBIS texture sizes a draw into another surface",
                                             "a destroyed YEBIS texture read as live"};
    for (int m = 1; m <= 8; ++m) {
        g_failed = 0;
        g_rt_extent_mutant = m;
        all_cases(true);
        property_cases(true);
        registry_cases(true);
        g_rt_extent_mutant = 0;
        if (g_failed == 0) {
            std::printf("FAIL: mistake %d (%s) not caught\n", m, kMistakes[m - 1]);
            ++failed;
        } else {
            std::printf("mistake %d (%s) caught: %d checks failed\n", m, kMistakes[m - 1], g_failed);
        }
    }
    std::printf(failed ? "rt_extent_test: %d failed\n" : "rt_extent_test: ok\n", failed);
    return failed ? 1 : 0;
}
