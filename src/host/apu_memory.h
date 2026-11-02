#pragma once
// What an integrated GPU can give back of the memory it holds in play beyond
// what it draws with (gpu.cpp's memory keeper, textures.cpp's eviction and
// drain). On an integrated GPU no video memory reserve applies (its video
// memory is the same RAM), so the caches the reserve trims on a card with
// little video memory - textures made ahead of their first bind, textures no
// bind has looked up for a while, the image heap's free room - and the upload
// staging kept from a load on all stay for the session: on a Radeon 8060S at
// a save's world, 832 MiB of staging and 1.3-1.6 GiB of caches. Kept apart
// from Vulkan so a test can drive it.
//
// Three parts, each with its switch: the environment first (any device, for
// tests and comparisons), then the config, then the default - on for an
// integrated GPU under Windows (kWindowsApuDefaults), off elsewhere. Two
// sizes follow the same defaults: the upload staging's floor and the buffer
// shadow's mirror windows.
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace apu_memory {

// The defaults on an integrated GPU under Windows: all three parts. On a
// Radeon 680M (AMD's Windows driver, 960x540, 60 fps) each cost no frame:
// - a 64 MiB staging chunk's vkFreeMemory took 0.4-3.7 ms (mean 1.1; 22-29 ms
//   on NVIDIA's Linux driver, which is why a discrete GPU keeps its staging);
// - the frames in the frees' half minute, the load times and the textures a
//   bind made stayed within the default's own spread;
// - Task Manager's shared GPU memory at the world went 5,880 -> 5,452 MiB
//   (staging back) and 5,880 -> 5,628-5,724 (the budget and the lean heap).
// Elsewhere every part is the environment's or the config's to turn on.
constexpr bool kWindowsApuDefaults = true;

// An integrated GPU under Windows (`integrated`: 1 yes, 0 no, -1 not chosen yet).
inline bool windows_apu(int integrated, bool windows_host) { return integrated == 1 && windows_host; }
inline bool apu_defaults(int integrated, bool windows_host) { return kWindowsApuDefaults && windows_apu(integrated, windows_host); }

// A switch: the environment (`env`: 0 off, anything else on), then the config
// (`config`: true on, anything else off), then `fallback`.
inline bool flag(const char* env, const char* config, bool fallback) {
    if (env && *env) return env[0] != '0';
    if (config && *config) return std::strcmp(config, "true") == 0;
    return fallback;
}

// Part 1: upload staging over the floor goes back in play, a 64 MiB chunk
// every two seconds once the world has settled (staging_give_now)
// (BBHOST_STAGING_RETURN, gpu.staging_return). On by default on an integrated
// GPU under Windows and on a graphics card with small video memory
// (`small_vram` 1, which imports by window too: its shared GPU memory is the
// staging and the windows), off elsewhere: on an RTX 4070 each chunk's free
// stalled the frame it fell in (29-47 ms against 14-17), so the memory policy
// (host/memory_policy.h) passes `small_vram` 1 only off NVIDIA's driver.
inline bool staging_return(const char* env, const char* config, int integrated, bool windows_host, int small_vram = 0) {
    return flag(env, config, apu_defaults(integrated, windows_host) || (integrated == 0 && small_vram == 1));
}

// The world has settled once no loading screen is up and `settle_ms` have
// passed since the latest load ended (BBHOST_CACHE_SETTLE_S). A load that
// ends - `was_loading` to not `loading` at `now_ms` - restarts the count, so
// a death or a lamp soon after an arrival starts it again.
inline std::uint64_t latest_load_end(std::uint64_t latest_end_ms, bool was_loading, bool loading, std::uint64_t now_ms) {
    return was_loading && !loading ? now_ms : latest_end_ms;
}
inline bool settled(std::uint64_t latest_end_ms, std::uint64_t now_ms, bool loading, std::uint64_t settle_ms) {
    return latest_end_ms && !loading && now_ms >= latest_end_ms && now_ms - latest_end_ms >= settle_ms;
}

// Whether part 1 gives a chunk of staging back now: on, over the floor, the
// world settled, and `pace_ms` since the last chunk unless video memory runs
// short. Counted from the latest load's end, not from when staging went over
// the floor: on the 680M a death within 30 s of an arrival had the frees fall
// 7-10 s into the next arrival, while the area streams in and staging may
// have to grow again - an allocation on the upload path.
inline bool staging_give_now(bool on, std::uint64_t free_bytes, std::uint64_t floor, bool settled_now, bool reserve_short,
                             std::uint64_t now_ms, std::uint64_t last_given_ms, std::uint64_t pace_ms) {
    return on && free_bytes > floor && settled_now && (reserve_short || now_ms - last_given_ms >= pace_ms);
}

// The upload staging's floor in MiB: made at the start, and what part 1 gives
// staging back down to (BBHOST_STAGING_FLOOR_MB, a number of MiB on any
// device). 128 on an integrated GPU under Windows, 256 elsewhere: on a Radeon
// 680M at a lamp, at 1080p, 128 held 128 MiB less of the shared GPU memory at
// the same frame rate and load times, with two to four more 64 MiB chunks
// given back a run (each free 1.4 ms at most).
constexpr std::uint64_t kStagingFloorMiB = 256, kWindowsApuStagingFloorMiB = 128;

inline std::uint64_t staging_floor_mib(const char* env, int integrated, bool windows_host) {
    if (env && *env) return std::strtoull(env, nullptr, 10);
    return apu_defaults(integrated, windows_host) ? kWindowsApuStagingFloorMiB : kStagingFloorMiB;
}

// The buffer shadow's mirror windows in MiB (buffer_shadow.cpp window_pages):
// 64 on a discrete GPU, 4 on an integrated one, 1 on an integrated GPU under
// Windows; BBHOST_SHADOW_WINDOW_MB, a power of two from 1 to 512, on any
// device (anything else keeps the default). On the Radeon 680M at a lamp,
// 1 MiB windows held 199-223 MiB less than 4 MiB ones at 1080p and at
// 960x540, at the same frame rate, with about 4 M more ranges a run read
// from host memory across a window's edge. The command processor's time a
// flip stayed the same at 60 fps (960x540); at 1080p, where the GPU held the
// frame rate to 27 fps, it was 0.4 ms longer in its draw phase with none of
// its counted waits changing, while the same reads at 60 fps cost nothing -
// a cost only while the GPU is the limit, not work per window.
constexpr std::uint64_t kShadowWindowMiB = 64, kIntegratedShadowWindowMiB = 4, kWindowsApuShadowWindowMiB = 1;

inline std::uint64_t shadow_window_mib(const char* env, int integrated, bool windows_host) {
    std::uint64_t mb = integrated != 1                            ? kShadowWindowMiB
                       : apu_defaults(integrated, windows_host) ? kWindowsApuShadowWindowMiB
                                                                : kIntegratedShadowWindowMiB;
    if (env && *env) {
        const std::uint64_t v = std::strtoull(env, nullptr, 10);
        if (v >= 1 && v <= 512 && (v & (v - 1)) == 0) mb = v;
    }
    return mb;
}

// Part 2: the caches' budget, in MiB of the device-local memory we hold; 0
// is none. Over it, textures are made at their first bind rather than ahead,
// those no bind has looked up for 20 s go, the textures' blocks drain at the
// lean thresholds and empty blocks go back at once (BBHOST_CACHE_BUDGET_MB,
// gpu.cache_budget_mb: a number of MiB, on any device).
constexpr std::uint64_t kWindowsApuBudgetMiB = 2048;

inline std::uint64_t cache_budget_mib(const char* env, const char* config, int integrated, bool windows_host) {
    if (env && *env) return std::strtoull(env, nullptr, 10);
    if (config && *config) return std::strtoull(config, nullptr, 10);
    return apu_defaults(integrated, windows_host) ? kWindowsApuBudgetMiB : 0;
}

// Part 3: a lean image heap - no spare block kept in play, and the textures'
// blocks drained at the lean thresholds (BBHOST_HEAP_LEAN, gpu.heap_lean).
inline bool heap_lean(const char* env, const char* config, int integrated, bool windows_host) {
    return flag(env, config, apu_defaults(integrated, windows_host));
}

// Whether the caches yield: the device-local bytes we hold, and `more` about
// to be made, over the budget (none when the budget is 0).
inline bool over_budget(std::uint64_t budget, std::uint64_t local, std::uint64_t more) {
    return budget && local + more > budget;
}

// How far over the budget the device-local bytes we hold are; 0 under it or
// with no budget.
inline std::uint64_t excess(std::uint64_t budget, std::uint64_t local) { return budget && local > budget ? local - budget : 0; }

// The textures' blocks' drain, once a second in play: it empties the block
// with the least in use when the textures' blocks hold `holes_at` of free
// room between them and that block no more than `most_used`, taking at most
// `hot_per_turn` of what a bind looked up in the last 2 s. The lean
// thresholds are a card's whose mirrors were refused room: half a block of
// room, up to 192 MiB in use, 32 MiB a turn.
struct DrainLimits {
    std::uint64_t holes_at, most_used, hot_per_turn;
};

inline DrainLimits drain_limits(bool lean) {
    constexpr std::uint64_t kBlock = 256ull << 20;
    if (lean) return {kBlock / 2, 192ull << 20, 32ull << 20};
    return {kBlock * 3 / 2, 128ull << 20, 16ull << 20};
}

// The image heap's spare blocks in play: BBHOST_IMAGE_HEAP_SPARE (`env`, a
// number) when set, else none with tight memory, a card with little video
// memory or a lean heap, else one.
inline std::uint64_t spare_play_blocks(const char* env, bool few, bool lean) {
    if (env && *env) return std::strtoull(env, nullptr, 10);
    return few || lean ? 0 : 1;
}

}  // namespace apu_memory
