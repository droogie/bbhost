#pragma once
// How the game's GPU-visible direct memory reaches the GPU (gpu.cpp,
// imports_whole_decide): imported whole when its mappings are made, or a
// window at a time as something first needs it. Kept apart from Vulkan so a
// test can drive it.
#include <cstring>

namespace import_mode {

enum class Mode { kWhole, kWindows, kUndecided };

// In order: BBHOST_IMPORT_WINDOWS (`env`: 1 windows, anything else whole
// spans), then gpu.import_windows in the config (`config`: true windows,
// anything else whole spans), then the device: windows on an integrated GPU
// under Windows, whose driver pins the whole spans resident and counts them
// as the process's shared GPU memory (a Radeon 680M: 8.0 -> 6.2 GB at the
// same frame rate), and whole spans on a discrete GPU and on Linux, where an
// integrated GPU imports through udmabuf dma-bufs with tight memory - not
// measured with window imports yet (the Steam Deck kit sets them itself).
// `integrated` is -1 until the device is chosen: undecided then.
inline Mode decide(const char* env, const char* config, int integrated, bool windows_host) {
    if (env && *env) return env[0] == '1' ? Mode::kWindows : Mode::kWhole;
    if (config && *config) return std::strcmp(config, "true") == 0 ? Mode::kWindows : Mode::kWhole;
    if (integrated < 0) return Mode::kUndecided;
    return integrated == 1 && windows_host ? Mode::kWindows : Mode::kWhole;
}

}  // namespace import_mode
