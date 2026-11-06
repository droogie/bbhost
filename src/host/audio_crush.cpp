#include "host/audio_crush.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace {

std::atomic<int> g_rate{0}, g_bits{0};
std::atomic<float> g_lowpass{0.0f};
std::atomic<bool> g_on{false};

}  // namespace

void audio_crush_set(int rate_hz, int bits, float lowpass_hz) {
    rate_hz = std::max(rate_hz, 0);
    bits = std::max(bits, 0);
    lowpass_hz = std::isfinite(lowpass_hz) ? std::max(lowpass_hz, 0.0f) : 0.0f;
    g_rate.store(rate_hz, std::memory_order_relaxed);
    g_bits.store(bits, std::memory_order_relaxed);
    g_lowpass.store(lowpass_hz, std::memory_order_relaxed);
    g_on.store(rate_hz || bits || lowpass_hz > 0.0f, std::memory_order_release);
}

bool audio_crush_on() { return g_on.load(std::memory_order_relaxed); }

void audio_crush_apply(AudioCrushState& s, const void* in, void* out, unsigned frames, unsigned channels, bool is_float,
                       unsigned freq) {
    if (!channels || channels > kAudioCrushChannels || !freq) return;
    const int rate = g_rate.load(std::memory_order_relaxed);
    const int bits = g_bits.load(std::memory_order_relaxed);
    const float cutoff = g_lowpass.load(std::memory_order_relaxed);
    // The low-pass: y += a (x - y), a = 1 - e^(-2 pi fc / fs); a = 1 passes x.
    const bool lp = cutoff > 0.0f && cutoff < 0.5f * static_cast<float>(freq);
    const float a = lp ? 1.0f - std::exp(-6.2831853f * cutoff / static_cast<float>(freq)) : 1.0f;
    // The hold: a new frame each time the phase reaches 1, rate / freq a frame.
    const double step = rate > 0 && static_cast<unsigned>(rate) < freq ? static_cast<double>(rate) / freq : 1.0;
    const int max_bits = is_float ? 23 : 15;
    const int keep = bits > 0 && bits <= max_bits ? bits : 0;
    const float levels = keep ? std::ldexp(1.0f, keep - 1) : 0.0f;  // steps of 1 / levels
    // The output filter after a hold: a Butterworth low-pass at 0.45 of the
    // rate held to (RBJ's biquad), taking out the steps' images - the held
    // staircase's energy around the rate and its multiples, a screech.
    const bool post = step < 1.0;
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    if (post) {
        const double w = 6.283185307179586 * 0.45 * rate / freq, c = std::cos(w), al = std::sin(w) / (2.0 * 0.7071067811865476);
        const double a0 = 1.0 + al;
        b0 = static_cast<float>((1.0 - c) / 2.0 / a0);
        b1 = static_cast<float>((1.0 - c) / a0);
        b2 = b0;
        a1 = static_cast<float>(-2.0 * c / a0);
        a2 = static_cast<float>((1.0 - al) / a0);
    }

    const float* fin = static_cast<const float*>(in);
    const std::int16_t* s16in = static_cast<const std::int16_t*>(in);
    float* fout = static_cast<float*>(out);
    std::int16_t* sout = static_cast<std::int16_t*>(out);
    for (unsigned f = 0; f < frames; ++f) {
        const unsigned base = f * channels;
        for (unsigned c = 0; c < channels; ++c) {
            const float x = is_float ? fin[base + c] : static_cast<float>(s16in[base + c]) * (1.0f / 32768.0f);
            s.lp[c] = lp ? s.lp[c] + a * (x - s.lp[c]) : x;
        }
        if (s.phase >= 1.0) {
            s.phase -= std::floor(s.phase);
            for (unsigned c = 0; c < channels; ++c) s.hold[c] = s.lp[c];
        }
        s.phase += step;
        for (unsigned c = 0; c < channels; ++c) {
            float y = s.hold[c];
            // Rounded to the nearest step: what is quieter than half a step is
            // silence, where truncating it flickered between 0 and a step below
            // (static, and a DC offset of half a step).
            if (keep) y = std::nearbyint(std::clamp(y, -1.0f, 1.0f - 1.0f / levels) * levels) / levels;
            if (post) {
                float o = b0 * y + b1 * s.x1[c] + b2 * s.x2[c] - a1 * s.y1[c] - a2 * s.y2[c];
                if (std::fabs(o) < 1e-15f) o = 0.0f;  // a decayed tail is silence, not slow denormals
                s.x2[c] = s.x1[c];
                s.x1[c] = y;
                s.y2[c] = s.y1[c];
                s.y1[c] = o;
                y = o;
            }
            if (is_float) {
                fout[base + c] = y;
            } else {
                sout[base + c] = static_cast<std::int16_t>(std::clamp(std::lround(y * 32768.0f), -32768L, 32767L));
            }
        }
    }
}
