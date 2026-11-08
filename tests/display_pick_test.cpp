// Which display the display buffers are made for (core/display_pick.h): the one with the most pixels, its own
// width and height; on a tie the one shaped like the starting size, then the primary - checked on desktops worked
// out by hand, in every order their displays can be listed (Windows' EnumDisplayDevices and SDL each list them in
// their own), and the checks shown to catch wrong choices (mutants), the widest and the tallest of all among them.
#include "core/display_pick.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

namespace {

struct Display {
    int w, h;
    bool primary;
};
struct Size {
    int w, h;
};
using PickFn = Size (*)(const std::vector<Display>&, int, int);

int g_fail = 0;
bool g_quiet = false;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            if (!g_quiet) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

Size largest(const std::vector<Display>& ds, int like_w, int like_h) {
    LargestDisplay pick{like_w, like_h};
    for (const Display& d : ds) pick.take(d.w, d.h, d.primary);
    return {pick.w, pick.h};
}

bool listed_before(const Display& a, const Display& b) { return std::tie(a.w, a.h, a.primary) < std::tie(b.w, b.h, b.primary); }

// The same choice whatever order the displays are listed in.
void expect(PickFn pick, std::vector<Display> ds, int like_w, int like_h, int want_w, int want_h, int line) {
    std::sort(ds.begin(), ds.end(), listed_before);
    do {
        const Size got = pick(ds, like_w, like_h);
        if (got.w != want_w || got.h != want_h) {
            if (!g_quiet) std::printf("FAIL line %d: %dx%d, not %dx%d\n", line, got.w, got.h, want_w, want_h);
            ++g_fail;
            return;
        }
    } while (std::next_permutation(ds.begin(), ds.end(), listed_before));
}
#define EXPECT(pick, ds, like_w, like_h, want_w, want_h) expect(pick, ds, like_w, like_h, want_w, want_h, __LINE__)

int run(PickFn pick) {
    const int before = g_fail;
    // One display: it, whatever the starting size.
    EXPECT(pick, (std::vector<Display>{{1920, 1080, true}}), 1920, 1080, 1920, 1080);
    EXPECT(pick, (std::vector<Display>{{1920, 1080, true}}), 0, 0, 1920, 1080);
    EXPECT(pick, (std::vector<Display>{{1920, 1080, true}}), 1080, 1920, 1920, 1080);
    EXPECT(pick, (std::vector<Display>{{3840, 2160, true}}), 1280, 720, 3840, 2160);
    // A landscape 2560x1440 beside a portrait 1440x2560 (the report: the two largest sides made 2560x2560): the
    // one shaped like the starting size, else the primary - never the turned one's height with the other's width.
    const std::vector<Display> turned{{2560, 1440, true}, {1440, 2560, false}};
    EXPECT(pick, turned, 2560, 1440, 2560, 1440);
    EXPECT(pick, turned, 1280, 720, 2560, 1440);
    EXPECT(pick, turned, 1440, 2560, 1440, 2560);  // started on the turned one: its shape, though not the primary
    EXPECT(pick, turned, 0, 0, 2560, 1440);        // no starting size: the primary
    const std::vector<Display> turned_primary{{2560, 1440, false}, {1440, 2560, true}};
    EXPECT(pick, turned_primary, 0, 0, 1440, 2560);
    EXPECT(pick, turned_primary, 1920, 1080, 2560, 1440);  // the starting size's shape before the primary
    EXPECT(pick, (std::vector<Display>{{1920, 1080, false}, {1080, 1920, true}}), 1600, 900, 1920, 1080);
    // More pixels before shape and primary.
    EXPECT(pick, (std::vector<Display>{{1920, 1080, true}, {3840, 2160, false}}), 1920, 1080, 3840, 2160);
    EXPECT(pick, (std::vector<Display>{{1080, 1920, true}, {2560, 1440, false}}), 1080, 1920, 2560, 1440);
    EXPECT(pick, (std::vector<Display>{{7680, 4320, false}, {3840, 2160, true}}), 3840, 2160, 7680, 4320);
    // An ultrawide beside a taller, smaller 16:10: the ultrawide (2.76 against 2.30 million pixels), not 2560x1200.
    EXPECT(pick, (std::vector<Display>{{2560, 1080, true}, {1920, 1200, false}}), 1920, 1200, 2560, 1080);
    // Three: the primary the smallest, two of a size, turned to each other.
    const std::vector<Display> three{{2560, 1440, false}, {1440, 2560, false}, {1920, 1080, true}};
    EXPECT(pick, three, 1920, 1080, 2560, 1440);
    EXPECT(pick, three, 1080, 1920, 1440, 2560);
    // A size of 0 or less is no display.
    EXPECT(pick, (std::vector<Display>{{0, 1080, true}}), 1920, 1080, 0, 0);
    EXPECT(pick, (std::vector<Display>{{-1, -1, true}, {1280, 800, false}}), 1280, 800, 1280, 800);
    EXPECT(pick, (std::vector<Display>{{1920, 0, true}, {1280, 720, false}}), 1920, 1080, 1280, 720);
    EXPECT(pick, (std::vector<Display>{}), 1920, 1080, 0, 0);
    return g_fail - before;
}

// The most pixels, a tie decided by the weights given to the starting size's shape and to the primary.
Size weighed(const std::vector<Display>& ds, int like_w, int like_h, int shape, int primary) {
    Size s{0, 0};
    std::uint64_t best = 0;
    int best_rank = -1;
    for (const Display& d : ds) {
        if (d.w <= 0 || d.h <= 0) continue;
        const std::uint64_t px = static_cast<std::uint64_t>(d.w) * static_cast<std::uint64_t>(d.h);
        const bool like = like_w > 0 && like_h > 0 && (d.w >= d.h) == (like_w >= like_h);
        const int rank = (like ? shape : 0) + (d.primary ? primary : 0);
        if (px > best || (px == best && rank > best_rank)) {
            best = px;
            best_rank = rank;
            s = {d.w, d.h};
        }
    }
    return s;
}

// The widest and the tallest of all (the choice this replaced).
Size m_widest_tallest(const std::vector<Display>& ds, int, int) {
    Size s{0, 0};
    for (const Display& d : ds)
        if (d.w > 0 && d.h > 0) s = {std::max(s.w, d.w), std::max(s.h, d.h)};
    return s;
}
Size m_first_listed(const std::vector<Display>& ds, int w, int h) { return weighed(ds, w, h, 0, 0); }
Size m_primary_first(const std::vector<Display>& ds, int w, int h) { return weighed(ds, w, h, 1, 2); }
Size m_shape_ignored(const std::vector<Display>& ds, int w, int h) { return weighed(ds, w, h, 0, 1); }
// The widest display, the first listed of those.
Size m_widest(const std::vector<Display>& ds, int, int) {
    Size s{0, 0};
    for (const Display& d : ds)
        if (d.w > 0 && d.h > 0 && d.w > s.w) s = {d.w, d.h};
    return s;
}

}  // namespace

int main() {
    const int fails = run(&largest);
    const struct {
        const char* name;
        PickFn fn;
    } mutants[] = {{"the widest and the tallest of all", m_widest_tallest},
                   {"the first listed on a tie", m_first_listed},
                   {"the primary before the starting size's shape", m_primary_first},
                   {"the starting size's shape ignored", m_shape_ignored},
                   {"the widest", m_widest}};
    int caught = 0;
    g_quiet = true;
    for (const auto& m : mutants) {
        if (run(m.fn) > 0) {
            ++caught;
        } else {
            std::printf("FAIL mutant not caught: %s\n", m.name);
        }
    }
    g_quiet = false;
    const int total = static_cast<int>(sizeof(mutants) / sizeof(mutants[0]));
    if (fails || caught != total) {
        std::printf("display_pick_test: %d failure(s), %d of %d mutants caught\n", fails, caught, total);
        return 1;
    }
    std::printf("ok - display pick, %d mutants caught\n", caught);
    return 0;
}
