// Old-hardware sound (host/audio_crush.h) on hand-made buffers: off leaves
// the samples alone; the bits round S16 and F32 to their nearest step, so
// what is quieter than half a step is silence (no static, no DC offset); a
// hold keeps its state across buffers (one buffer split in two gives the same
// samples) and its output filter takes out the steps' images above the rate
// (the screech) while the tone itself passes; the low-pass smooths a step.
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

// The amplitude of one frequency in a mono buffer (Goertzel).
static double amplitude(const std::vector<float>& x, double hz, double fs) {
    const double w = 2.0 * 3.141592653589793 * hz / fs, c = 2.0 * std::cos(w);
    double s1 = 0.0, s2 = 0.0;
    for (float v : x) {
        const double s = v + c * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    return 2.0 * std::sqrt(s1 * s1 + s2 * s2 - c * s1 * s2) / static_cast<double>(x.size());
}

int main() {
    // Off.
    audio_crush_set(0, 0, 0.0f);
    CHECK(!audio_crush_on());

    // S16, 8 bits alone: the nearest multiple of 256, quiet samples silence.
    {
        audio_crush_set(0, 8, 0.0f);
        CHECK(audio_crush_on());
        const std::vector<std::int16_t> in = {0, 100, -100, 127, -129, 200, 1000, -1000, 32767, -32768};
        std::vector<std::int16_t> out(in.size());
        AudioCrushState st;
        audio_crush_apply(st, in.data(), out.data(), static_cast<unsigned>(in.size()), 1, false, 48000);
        const std::vector<std::int16_t> want = {0, 0, 0, 0, -256, 256, 1024, -1024, 32512, -32768};
        CHECK(out == want);
    }

    // F32 mono, 4 bits: steps of 1/8, rounded; in place.
    {
        audio_crush_set(0, 4, 0.0f);
        std::vector<float> s = {0.0f, 0.05f, -0.05f, 0.1f, -0.1f, 0.99f, -1.0f, 1.5f};
        AudioCrushState st;
        audio_crush_apply(st, s.data(), s.data(), static_cast<unsigned>(s.size()), 1, true, 48000);
        CHECK(s[0] == 0.0f && s[1] == 0.0f && s[2] == 0.0f && s[3] == 0.125f && s[4] == -0.125f && s[5] == 0.875f &&
              s[6] == -1.0f && s[7] == 0.875f);
    }

    // Near-silence at 8 bits, held: exactly silent, no static.
    {
        audio_crush_set(11025, 8, 0.0f);
        std::vector<float> s(2 * 4800);
        for (std::size_t i = 0; i < s.size(); ++i) s[i] = 0.003f * std::sin(0.37f * static_cast<float>(i));
        AudioCrushState st;
        audio_crush_apply(st, s.data(), s.data(), 4800, 2, true, 48000);
        for (float v : s) CHECK(v == 0.0f);
    }

    // A held 1 kHz tone (F32 stereo at 48000, held to 11025, no bits): one
    // buffer or the same split in two alike; the tone through, its images
    // around the rate (10025, 12025 Hz) and twice it (21050, 23050 Hz) down.
    {
        audio_crush_set(11025, 0, 0.0f);
        const unsigned frames = 48000;
        std::vector<float> in(2 * frames), whole(in.size()), split(in.size());
        for (unsigned f = 0; f < frames; ++f) {
            in[2 * f] = in[2 * f + 1] = 0.5f * static_cast<float>(std::sin(2.0 * 3.141592653589793 * 1000.0 * f / 48000.0));
        }
        AudioCrushState a, b;
        audio_crush_apply(a, in.data(), whole.data(), frames, 2, true, 48000);
        audio_crush_apply(b, in.data(), split.data(), 9001, 2, true, 48000);
        audio_crush_apply(b, in.data() + 2 * 9001, split.data() + 2 * 9001, frames - 9001, 2, true, 48000);
        CHECK(whole == split);
        std::vector<float> left(frames);
        for (unsigned f = 0; f < frames; ++f) left[f] = whole[2 * f];
        const double tone = amplitude(left, 1000.0, 48000.0);
        CHECK(tone > 0.4);
        for (double image : {10025.0, 12025.0, 21050.0, 23050.0}) {
            const double db = 20.0 * std::log10(amplitude(left, image, 48000.0) / tone);
            std::printf("  image at %.0f Hz: %.1f dB\n", image, db);
            CHECK(db < -30.0);
        }
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
