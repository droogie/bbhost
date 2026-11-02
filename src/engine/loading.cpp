#include "engine/loading.h"

#include "engine/addr.h"
#include "engine/frame_rate.h"
#include "hle/modules.h"
#include "host/gpu.h"
#include "log.h"

#include <cstdlib>

#include <atomic>
#include <chrono>

namespace {

constexpr std::uint64_t kNowLoadingRequested = 0x596286b;  // u8
constexpr std::uint64_t kNowLoadingHelper = 0x593e8b8;     // CSNowLoadingHelper*
// SprjRemo* (written in SprjRemoStep::STEP_Init); its manager at +8 has a
// running flag in +0x168 bit 0 and a count of queued cutscenes at +0xf8
// (Kyo's offsets): a cutscene's events are timed for the frame rate, so a
// load it follows keeps the cap.
constexpr std::uint64_t kSprjRemo = 0x5940058;
std::atomic<bool> g_fast{false};
const bool g_fast_enabled = [] {
    const char* e = std::getenv("BBHOST_FAST_LOADING");
    return !(e && e[0] == '0');
}();

std::uint64_t g_slide = 0;
std::atomic<bool> g_loading{false};
std::atomic<bool> g_in_world{false};
std::chrono::steady_clock::time_point g_start;

template <typename T>
T at(std::uint64_t va) {
    return *reinterpret_cast<const volatile T*>(static_cast<std::uintptr_t>(va));
}

std::uint64_t guest(std::uint64_t bn) { return g_slide + (bn - kPreferredGuestSlide); }

}  // namespace

void loading_bind(std::uint64_t slide) {
    g_slide = slide;
    g_start = std::chrono::steady_clock::now();
}

bool loading_screen_up() { return g_loading.load(std::memory_order_relaxed); }
bool loading_in_world() { return g_in_world.load(std::memory_order_relaxed); }
bool loading_fast_now() { return g_fast.load(std::memory_order_relaxed); }

void loading_tick(std::uint64_t flip) {
    if (!g_slide) return;
    static bool seen_world = false;
    static std::chrono::steady_clock::time_point began;
    static std::uint64_t began_flip = 0, began_unshown = 0;
    const bool requested = at<std::uint8_t>(guest(kNowLoadingRequested)) != 0;
    const std::uint64_t helper = at<std::uint64_t>(guest(kNowLoadingHelper));
    const bool in_game = helper && (at<std::uint8_t>(helper + 0x50) || at<std::uint8_t>(helper + 0x51));
    const bool loading = requested && !in_game;
    g_in_world.store(in_game, std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    const bool was = g_loading.exchange(loading, std::memory_order_relaxed);
    if (loading != was) host_gpu_set_loading(loading);  // the memory keeper readies the coming area's memory
    bool cutscene = false;
    if (const std::uint64_t remo = at<std::uint64_t>(guest(kSprjRemo))) {
        if (const std::uint64_t man = at<std::uint64_t>(remo + 8)) {
            cutscene = (at<std::uint32_t>(man + 0x168) & 1) != 0 || at<std::uint64_t>(man + 0xf8) != 0;
        }
    }
    const bool fast = g_fast_enabled && loading && !cutscene && frame_rate_game_fps() != 30;  // every rate above 30
    if (g_fast.exchange(fast, std::memory_order_relaxed) != fast) hle_video_set_loading_uncapped(fast);
    if (loading && !was) {
        began = now;
        began_flip = flip;
        began_unshown = hle_video_loading_unshown();
        if (seen_world) host_log("loading: begins, flip %llu", static_cast<unsigned long long>(flip));  // a death or a warp
    } else if (!loading && was) {
        host_log("loading: %.2f s (flips %llu-%llu, %llu of them not shown)", std::chrono::duration<double>(now - began).count(),
                 static_cast<unsigned long long>(began_flip), static_cast<unsigned long long>(flip),
                 static_cast<unsigned long long>(hle_video_loading_unshown() - began_unshown));
    }
    if (in_game && !seen_world) {
        seen_world = true;
        host_gpu_world_reached();  // the start's shader compiles go back to a quarter of the threads
        host_log("world: the first in-game frame, flip %llu, %.1f s after start", static_cast<unsigned long long>(flip),
                 std::chrono::duration<double>(now - g_start).count());
    }
}
