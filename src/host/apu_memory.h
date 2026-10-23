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
// tests and comparisons), then the config, then the default - off everywhere
// for now (kWindowsApuDefaults).
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
// every two seconds once it has been over the floor for 30 s
// (BBHOST_STAGING_RETURN, gpu.staging_return). Off by default: on an RTX 4070
// each chunk's free stalled the frame it fell in (29-47 ms against 14-17).
inline bool staging_return(const char* env, const char* config, int integrated, bool windows_host) {
    return flag(env, config, apu_defaults(integrated, windows_host));
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
