// The vertex clamp's lookup (host/vertex_clamp.h) runs exactly where its result
// is used - the buffer shadow, vertex staging, the race census - and nowhere
// else: every combination of the four switches, then the devices as the
// defaults set them up (buffer_shadow_on, vertex_stage_on; the census off).
// Then five deliberate mistakes, each of which must be caught.
#define VERTEX_CLAMP_MUTANTS
#include "host/vertex_clamp.h"

#include <cstdio>

int g_vertex_clamp_mutant = 0;

namespace {

int g_failed = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failed;
        if (!g_vertex_clamp_mutant) std::printf("FAIL: %s\n", what);
    }
}

void all_cases() {
    using vertex_clamp::lookup_wanted;
    for (int m = 0; m < 16; ++m) {
        const bool clamp = m & 1, shadow = m & 2, stage = m & 4, race = m & 8;
        check(lookup_wanted(clamp, shadow, stage, race) == (clamp && (shadow || stage || race)),
              "the clamp on and the shadow, staging or the census taking the extent, in every combination");
    }
    // A discrete GPU: staging on, the shadow too unless memory is tight - looked up, as before.
    check(lookup_wanted(true, true, true, false), "a discrete GPU with the buffer shadow looks up");
    check(lookup_wanted(true, false, true, false), "a discrete GPU with tight memory (staging, no shadow) looks up");
    // An integrated GPU stages nothing; with 8 GiB or more of device-local memory it has the shadow.
    check(lookup_wanted(true, true, false, false), "an integrated GPU with room for the mirrors looks up");
    check(!lookup_wanted(true, false, false, false), "an integrated GPU without room (a Steam Deck, a 680M) does not");
    check(lookup_wanted(true, false, false, true), "it does for the race census");
    check(!lookup_wanted(false, true, true, true), "BBHOST_VERTEX_CLAMP=0: nowhere");
}

}  // namespace

int main() {
    all_cases();
    int failed = g_failed;
    static const char* const kMistakes[5] = {"the lookup everywhere, as before", "the buffer shadow left out", "staging left out",
                                             "the race census left out", "BBHOST_VERTEX_CLAMP=0 ignored"};
    for (int m = 1; m <= 5; ++m) {
        g_failed = 0;
        g_vertex_clamp_mutant = m;
        all_cases();
        g_vertex_clamp_mutant = 0;
        if (g_failed == 0) {
            std::printf("FAIL: mistake %d (%s) not caught\n", m, kMistakes[m - 1]);
            ++failed;
        } else {
            std::printf("mistake %d (%s) caught: %d checks failed\n", m, kMistakes[m - 1], g_failed);
        }
    }
    std::printf(failed ? "vertex_clamp_test: %d failed\n" : "vertex_clamp_test: ok\n", failed);
    return failed ? 1 : 0;
}
