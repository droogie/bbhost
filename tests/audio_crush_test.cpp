// Old-hardware sound (host/audio_crush.h) on hand-made buffers: off leaves
// the samples alone; the hold repeats each frame freq / rate times and keeps
// its phase across buffers (one buffer split in two gives the same samples);
// the bits truncate S16 and F32 to their steps; the low-pass smooths a step.
#include "host/audio_crush.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
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
    // Off.
    audio_crush_set(0, 0, 0.0f);
    CHECK(!audio_crush_on());

    // S16 stereo at 48000 held to 12000: each frame four times; 8 bits.
    {
        audio_crush_set(12000, 8, 0.0f);
        CHECK(audio_crush_on());
        std::vector<std::int16_t> in(2 * 64), whole(in.size()), split(in.size());
        for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<std::int16_t>(static_cast<int>(i) * 397 - 20000);
        AudioCrushState a, b;
        audio_crush_apply(a, in.data(), whole.data(), 64, 2, false, 48000);
        audio_crush_apply(b, in.data(), split.data(), 23, 2, false, 48000);
        audio_crush_apply(b, in.data() + 46, split.data() + 46, 41, 2, false, 48000);
        CHECK(whole == split);
        for (unsigned f = 0; f < 64; ++f) {
            for (unsigned c = 0; c < 2; ++c) {
                const std::int16_t v = whole[2 * f + c];
                CHECK((v & 0xff) == 0);                                // 8 bits: the low 8 dropped
                CHECK(v == (in[2 * (f / 4 * 4) + c] & ~0xff));         // the run's first frame, truncated
            }
        }
    }

    // F32 mono, 4 bits: steps of 1/8, toward minus infinity; in place.
    {
        audio_crush_set(0, 4, 0.0f);
        std::vector<float> s = {0.0f, 0.1f, -0.1f, 0.99f, -1.0f, 1.5f};
        AudioCrushState st;
        audio_crush_apply(st, s.data(), s.data(), 6, 1, true, 48000);
        CHECK(s[0] == 0.0f && s[1] == 0.0f && s[2] == -0.125f && s[3] == 0.875f && s[4] == -1.0f && s[5] == 0.875f);
    }

    // The low-pass alone: a step rises smoothly, never past it.
    {
        audio_crush_set(0, 0, 1000.0f);
        std::vector<float> s(64, 1.0f);
        AudioCrushState st;
        audio_crush_apply(st, s.data(), s.data(), 64, 1, true, 48000);
        CHECK(s[0] > 0.0f && s[0] < 0.2f);
        for (std::size_t i = 1; i < s.size(); ++i) CHECK(s[i] > s[i - 1] && s[i] < 1.0f);
    }

    audio_crush_set(0, 0, 0.0f);
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("audio crush: all checks passed\n");
    return 0;
}
