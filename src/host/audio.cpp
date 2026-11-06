#include "host/audio.h"
#include "host/audio_crush.h"
#include "core/portable.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(BBHOST_HAVE_SDL3)
#include <SDL3/SDL.h>
#endif

namespace {

#if defined(BBHOST_HAVE_SDL3)
// A write uses its output's stream outside g_mu while it waits for the queue
// to drain and hands SDL the data. `writers` counts those uses; close and
// shutdown set `closing` (a waiting write stops early and drops its data) and
// destroy the stream only once no write holds it. Destroying it under a write
// in progress was a use-after-free inside SDL (core 2044885: the guest's audio
// thread in SDL_GetAudioStreamQueued / SDL_PutAudioStreamData).
struct Out {
    SDL_AudioStream* stream = nullptr;
    unsigned frame_bytes = 0;
    unsigned write_bytes = 0;
    unsigned freq = 0, channels = 0;
    bool is_float = false;
    int writers = 0;  // guarded by g_mu
    std::atomic<bool> closing{false};
    // What reaches the device, per window of a few seconds (written by the
    // one thread that writes this output): writes, underruns (the stream's
    // queue was empty when the next buffer came - the device ran dry), the
    // longest time between writes, the loudest sample and how many were
    // over full scale (clipped by the device conversion).
    std::chrono::steady_clock::time_point window_start{}, last_write{};
    std::uint64_t writes = 0, underruns = 0, total_writes = 0, over = 0, nans = 0;
    double longest_gap_ms = 0.0;
    float peak = 0.0f;
    std::FILE* dump = nullptr;  // BBHOST_AUDIO_DUMP: raw samples as written
    // The device asked for more than the stream held (SDL's get callback):
    // the rest of that period was silence - an audible dropout. Counted from
    // the first write on; `starved_bytes` is how much was missing.
    std::atomic<std::uint64_t> starved{0}, starved_bytes{0};
    std::atomic<bool> started{false};
    // The pacer (host_audio_write): when the guest's next call may return, the
    // queue's smoothed level, and the most the device took in one pull.
    std::chrono::steady_clock::time_point due{};
    double queued_avg = 0.0;
    std::atomic<int> max_pull{0};
    // Old-hardware sound (audio_crush.h): its state across buffers and the
    // buffer it writes into (the guest's own is left alone).
    AudioCrushState crush;
    std::vector<unsigned char> crushed;
};

void SDLCALL on_device_get(void* userdata, SDL_AudioStream*, int additional_amount, int total_amount) {
    Out* o = static_cast<Out*>(userdata);
    if (total_amount > o->max_pull.load(std::memory_order_relaxed)) o->max_pull.store(total_amount, std::memory_order_relaxed);
    if (additional_amount > 0 && o->started.load(std::memory_order_relaxed)) {
        o->starved.fetch_add(1, std::memory_order_relaxed);
        o->starved_bytes.fetch_add(static_cast<std::uint64_t>(additional_amount), std::memory_order_relaxed);
    }
}
std::mutex g_mu;
std::condition_variable g_writers_done;
std::unordered_map<int, Out> g_outs;  // element references survive rehashing
int g_next = 1;
bool g_audio_init = false;
bool g_audio_failed = false;
bool g_audio_shutdown = false;  // host_audio_shutdown ran: no new outputs

bool ensure_audio() {
    if (g_audio_init) {
        return true;
    }
    if (g_audio_failed) {
        return false;
    }
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        host_log("SDL audio unavailable: %s (mixer runs without output)", SDL_GetError());
        g_audio_failed = true;
        return false;
    }
    g_audio_init = true;
    return true;
}
#endif

}  // namespace

int host_audio_open(unsigned freq, unsigned channels, bool is_float, unsigned frames_per_write) {
#if defined(BBHOST_HAVE_SDL3)
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_audio_shutdown || !ensure_audio()) {
        return 0;
    }
    SDL_AudioSpec spec{};
    spec.format = is_float ? SDL_AUDIO_F32 : SDL_AUDIO_S16;
    spec.channels = static_cast<int>(channels);
    spec.freq = static_cast<int>(freq);
    SDL_AudioStream* st = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
    if (!st) {
        host_log("SDL_OpenAudioDeviceStream failed: %s", SDL_GetError());
        return 0;
    }
    int h = g_next++;
    Out& o = g_outs[h];
    o.stream = st;
    SDL_SetAudioStreamGetCallback(st, on_device_get, &o);
    SDL_ResumeAudioStreamDevice(st);
    o.frame_bytes = channels * (is_float ? 4u : 2u);
    o.write_bytes = o.frame_bytes * frames_per_write;
    o.freq = freq;
    o.channels = channels;
    o.is_float = is_float;
    // BBHOST_AUDIO_DUMP=<dir>: every output's samples, exactly as the device
    // gets them (the game's, after audio_crush), to
    // <dir>/audio-<handle>-<Hz>-<ch>ch-<f32|s16>.raw.
    if (const char* dir = std::getenv("BBHOST_AUDIO_DUMP")) {
        const std::string path = std::string(dir) + "/audio-" + std::to_string(h) + "-" + std::to_string(freq) + "-" +
                                 std::to_string(channels) + "ch-" + (is_float ? "f32" : "s16") + ".raw";
        o.dump = std::fopen(path.c_str(), "wb");
        host_log("audio: dumping output %d to %s%s", h, path.c_str(), o.dump ? "" : " (cannot open)");
    }
    host_log("audio: %u Hz %u ch %s, %u frames per write", freq, channels, is_float ? "f32" : "s16", frames_per_write);
    return h;
#else
    (void)freq;
    (void)channels;
    (void)is_float;
    (void)frames_per_write;
    return 0;
#endif
}

void host_audio_write(int handle, const void* data, unsigned frames) {
#if defined(BBHOST_HAVE_SDL3)
    Out* o = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_outs.find(handle);
        if (it == g_outs.end() || it->second.closing.load()) {
            return;
        }
        o = &it->second;
        ++o->writers;
    }
    const unsigned bytes = frames * o->frame_bytes;
    if (data && audio_crush_on()) {
        o->crushed.resize(bytes);
        audio_crush_apply(o->crush, data, o->crushed.data(), frames, o->channels, o->is_float, o->freq);
        data = o->crushed.data();
    }
    const auto start = std::chrono::steady_clock::now();
    {
        // The window's statistics (see Out). An underrun is counted from the
        // tenth write on: the first ones fill an empty queue by design.
        if (o->total_writes == 0) o->window_start = start;
        if (o->total_writes >= 10 && SDL_GetAudioStreamQueued(o->stream) == 0) ++o->underruns;
        if (o->total_writes) {
            o->longest_gap_ms = std::max(o->longest_gap_ms, std::chrono::duration<double, std::milli>(start - o->last_write).count());
        }
        o->last_write = start;
        ++o->writes;
        ++o->total_writes;
        if (o->total_writes == 10) o->started = true;
        if (o->is_float && data) {
            const float* f = static_cast<const float*>(data);
            for (unsigned i = 0, n = frames * o->channels; i < n; ++i) {
                const float v = std::fabs(f[i]);
                if (std::isnan(v)) {
                    ++o->nans;
                    continue;
                }
                o->peak = std::max(o->peak, v);
                if (v > 1.0f) ++o->over;
            }
        }
        if (o->dump && data) std::fwrite(data, 1, bytes, o->dump);
        const double window_s = std::chrono::duration<double>(start - o->window_start).count();
        static const bool always = [] { const char* e = std::getenv("BBHOST_AUDIO_STATS"); return e && *e == '1'; }();
        if (window_s >= 5.0) {
            const std::uint64_t starved = o->starved.exchange(0), starved_bytes = o->starved_bytes.exchange(0);
            if (always || o->underruns || starved || o->over || o->nans) {
                host_log("audio: output %d: %.1f writes/s (real time %.1f), %llu underrun(s), device short %llu time(s) "
                         "(%.1f ms of silence), longest gap %.1f ms, peak %.3f, %llu sample(s) over full scale, %llu NaN",
                         handle, static_cast<double>(o->writes) / window_s,
                         static_cast<double>(o->freq) * o->frame_bytes / (o->write_bytes ? o->write_bytes : 1),
                         static_cast<unsigned long long>(o->underruns), static_cast<unsigned long long>(starved),
                         1000.0 * static_cast<double>(starved_bytes) / (static_cast<double>(o->frame_bytes) * o->freq),
                         o->longest_gap_ms, static_cast<double>(o->peak), static_cast<unsigned long long>(o->over),
                         static_cast<unsigned long long>(o->nans));
            }
            o->window_start = start;
            o->writes = o->underruns = o->over = o->nans = 0;
            o->longest_gap_ms = 0.0;
            o->peak = 0.0f;
        }
    }
    // The guest is paced like the console's audio hardware: one call a buffer
    // period on a steady clock, whatever the host device does. The console's
    // sceAudioOutOutput returns as its previous buffer starts to play - every
    // 5.3 ms for 256 frames - and FMOD's output thread lives by that, taking
    // each block from its mixer's ring as the call returns. This loop used to
    // pace by the host queue instead (wait while more than three blocks were
    // queued), which handed the device's period to the guest: a Steam Deck's
    // PipeWire pulls 1024 frames at a time, so four calls returned at once
    // every 21 ms, the output thread ran ahead of the mixer, and two block
    // boundaries in five were stale ring data - a click each, static with a
    // 187 Hz buzz (the same dump on this desktop: 5 in 18,297). Now the data
    // goes in at once and the call returns on the next period; the queue's
    // target is the device's largest pull plus four blocks, and the period
    // stretches or shrinks by up to 2% to hold it there, so the guest follows
    // the device's clock. Short of half the target (the start, or after the
    // guest stalled through a load) it refills without waiting. Four blocks,
    // not two: on the Deck the guest's own call came up to 32 ms late every
    // few seconds (its mixer thread on busy cores), and with 32 ms queued
    // each of those was 5 ms of silence.
    if (!o->closing.load()) {
        SDL_PutAudioStreamData(o->stream, data, static_cast<int>(bytes));
        const int queued = SDL_GetAudioStreamQueued(o->stream);
        const double cap = 0.4 * static_cast<double>(o->freq) * o->frame_bytes;  // 400 ms
        const double target = std::min(cap, static_cast<double>(std::max<int>(3 * static_cast<int>(o->write_bytes),
                                                                              o->max_pull.load(std::memory_order_relaxed) +
                                                                                  4 * static_cast<int>(o->write_bytes))));
        o->queued_avg += (static_cast<double>(queued) - o->queued_avg) / 16.0;
        const auto now = std::chrono::steady_clock::now();
        if (queued < target / 2) {
            o->due = now;
        } else {
            const double period_us = 1e6 * frames / static_cast<double>(o->freq ? o->freq : 48000);
            const double adj = std::clamp((o->queued_avg - target) / target * 0.02, -0.02, 0.02);
            if (o->due.time_since_epoch().count() == 0 || now - o->due > std::chrono::milliseconds(50)) o->due = now;
            o->due += std::chrono::microseconds(static_cast<long long>(period_us * (1.0 + adj)));
            if (o->due > now) host_sleep_until(o->due);
        }
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (--o->writers == 0) {
        g_writers_done.notify_all();
    }
#else
    (void)handle;
    (void)data;
    (void)frames;
#endif
}

void host_audio_set_gain(int handle, float gain) {
#if defined(BBHOST_HAVE_SDL3)
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_outs.find(handle);
    if (it != g_outs.end() && !it->second.closing.load()) {
        SDL_SetAudioStreamGain(it->second.stream, gain);
    }
#else
    (void)handle;
    (void)gain;
#endif
}

void host_audio_close(int handle) {
#if defined(BBHOST_HAVE_SDL3)
    std::unique_lock<std::mutex> lock(g_mu);
    auto it = g_outs.find(handle);
    if (it == g_outs.end() || it->second.closing.exchange(true)) {
        return;  // unknown, or another close or the shutdown owns it
    }
    // Looked up again after every wake: host_audio_shutdown may have destroyed it.
    g_writers_done.wait(lock, [&] {
        auto found = g_outs.find(handle);
        return found == g_outs.end() || found->second.writers == 0;
    });
    auto found = g_outs.find(handle);
    if (found != g_outs.end()) {
        SDL_DestroyAudioStream(found->second.stream);
        if (found->second.dump) std::fclose(found->second.dump);
        g_outs.erase(found);
    }
#else
    (void)handle;
#endif
}

void host_audio_pause_all() {
#if defined(BBHOST_HAVE_SDL3)
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto& [handle, o] : g_outs) {
        if (o.stream) SDL_PauseAudioStreamDevice(o.stream);
    }
#endif
}

void host_audio_shutdown() {
#if defined(BBHOST_HAVE_SDL3)
    std::unique_lock<std::mutex> lock(g_mu);
    g_audio_shutdown = true;
    for (auto& [handle, o] : g_outs) {
        o.closing = true;
    }
    g_writers_done.wait(lock, [] {
        for (const auto& [handle, o] : g_outs) {
            if (o.writers) return false;
        }
        return true;
    });
    for (auto& [handle, o] : g_outs) {
        SDL_DestroyAudioStream(o.stream);
        if (o.dump) std::fclose(o.dump);
    }
    g_outs.clear();
    g_writers_done.notify_all();  // a close waiting on one of these finds it gone
    if (g_audio_init) {
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        g_audio_init = false;
    }
#endif
}
