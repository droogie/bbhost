// The learned sizes of render targets (host/rt_extent.h), without the game or
// a GPU: the 2560x1440 glare - YEBIS's cutscenes' glare pyramid learning
// 523x441 at an address where its glare then draws a 640x360 level with a
// scissor of its whole pitch - and back again; the 1920x1080 case a larger
// scissor fixed (392 learned in a 512 pitch, then 480); a scissor the padded
// size on first sight; a view's size found by a draw with no view; and a
// scissor the padded size over a smaller size learned for the same geometry,
// which stays and is flagged for the census. The same cases by the address
// alone (BBHOST_RT_LEARN_BY_GEOMETRY=0, the rule before) must fail, and six
// deliberate mistakes in the rule must each be caught.
#define RT_EXTENT_MUTANTS
#include "host/rt_extent.h"

#include <cstdint>
#include <cstdio>

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

    // The bug: one address, the pyramid's level (pitch 640, 448 rows) and the glare's (pitch 640, 360 rows).
    {
        Table t(by_geometry);
        Answer a = t.draw(kBright, 640, 448, 523, 441);
        check(a.w == 523 && a.h == 441 && a.changed, "the pyramid's level learns 523x441 from its scissor");
        a = t.draw(kBright, 640, 360, 640, 360);
        check(a.w == 640 && a.h == 360, "the glare's 640x360 level at that address, its scissor its whole pitch, is 640x360");
        check(!a.padded_over_w && !a.padded_over_h, "nothing was learned for the glare's geometry, so its scissor is over nothing");
        a = t.draw(kBright, 640, 360, 0, 0);
        check(a.w == 640 && a.h == 360, "a clear of the glare's level (no scissor) is 640x360");
        a = t.draw(kBright, 640, 448, 0, 0);
        check(a.w == 523 && a.h == 441 && !a.changed, "back in the pyramid's geometry, a clear finds 523x441 still known");
        a = t.draw(kBright, 640, 448, 523, 441);
        check(a.w == 523 && a.h == 441 && !a.changed, "and the pyramid's draw too, learning nothing new");
        a = t.draw(kBright, 640, 360, 640, 360);
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
        a = t.draw(kBright, 640, 360, 640, 360);
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

// The rule before, as the owner's log shows it: the glare's level 523x360.
void the_rule_before() {
    Table t(false);
    t.draw(kBright, 640, 448, 523, 441);
    const Answer a = t.draw(kBright, 640, 360, 640, 360);
    check(a.w == 523 && a.h == 360, "by the address alone the glare's level is 523x360, as before");
}

}  // namespace

int main() {
    all_cases(true);
    the_rule_before();
    int failed = g_failed;
    g_failed = 0;
    g_quiet = true;
    all_cases(false);
    g_quiet = false;
    if (g_failed == 0) {
        std::printf("FAIL: the rule by the address alone (BBHOST_RT_LEARN_BY_GEOMETRY=0) passes the cases\n");
        ++failed;
    } else {
        std::printf("the rule by the address alone caught: %d checks failed\n", g_failed);
    }
    static const char* const kMistakes[6] = {"the padded height left out of the key",
                                             "a scissor the padded size replaces a smaller learned size",
                                             "a scissor the padded size learned on first sight",
                                             "a learned size used where its padding cannot come from",
                                             "a larger scissor never replaces a learned size",
                                             "a scissor the padded size over a learned size not flagged"};
    for (int m = 1; m <= 6; ++m) {
        g_failed = 0;
        g_rt_extent_mutant = m;
        all_cases(true);
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
