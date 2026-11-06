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
    const float levels = keep ? std::ldexp(1.0f, keep - 1) : 0.0f;          // F32: steps of 1 / levels
    const std::int32_t mask = keep ? ~((std::int32_t{1} << (16 - std::min(keep, 15))) - 1) : ~0;  // S16

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
            const float y = s.hold[c];
            if (is_float) {
                // Truncated toward minus infinity, as dropping the low bits of an integer does.
                fout[base + c] = keep ? std::floor(std::clamp(y, -1.0f, 1.0f - 1.0f / levels) * levels) / levels : y;
            } else {
                const long v = std::clamp(std::lround(y * 32768.0f), -32768L, 32767L);
                sout[base + c] = static_cast<std::int16_t>(static_cast<std::int32_t>(v) & mask);
            }
        }
    }
}
