#pragma once

// Old-hardware sound (plugin API 21's audio_crush): the game's final mix, as
// host_audio_write hands it to the device, made to sound like a PS1-era sound
// chip's - a one-pole low-pass for the muffled top end, sample-and-hold down
// to a lower rate, the samples rounded to fewer bits, and the held steps
// smoothed again by an output filter as a DAC's was.

constexpr unsigned kAudioCrushChannels = 8;  // outputs with more are left as they are

// One output's filter state, kept across its buffers so their boundaries don't
// click: the low-pass's last outputs, the held frame and the hold's phase, and
// the output filter's last two inputs and outputs.
struct AudioCrushState {
    float lp[kAudioCrushChannels] = {};
    float hold[kAudioCrushChannels] = {};
    double phase = 1.0;  // >= 1: the next frame is taken
    float x1[kAudioCrushChannels] = {}, x2[kAudioCrushChannels] = {};
    float y1[kAudioCrushChannels] = {}, y2[kAudioCrushChannels] = {};
};

// What every output does from its next buffer. rate_hz: the rate held to (0,
// or the output's own or above, off); bits: 1..15 for S16 outputs, 1..23 for
// F32 (0 or above, off), rounded to the nearest step, no dither (silence stays
// silent); lowpass_hz: the cutoff (0, or half the output's rate or above,
// off). A hold also smooths its steps after the bits, at 0.45 of rate_hz (a
// second-order low-pass, as a DAC's output filter took out the steps' images
// above the rate - the screech). Any thread.
void audio_crush_set(int rate_hz, int bits, float lowpass_hz);
// Whether any of the three is set: one relaxed load, so off costs nothing.
bool audio_crush_on();
// frames * channels interleaved samples from in to out (they may be the same),
// S16 or F32 at freq Hz.
void audio_crush_apply(AudioCrushState& s, const void* in, void* out, unsigned frames, unsigned channels, bool is_float,
                       unsigned freq);
