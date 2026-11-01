#include "guest_abi.h"
#include "engine/sound_wake.h"

#include "core/elf.h"
#include "core/host_clock.h"
#include "core/memory.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace {

// The pass's sleep in the sound thread's loop (sub_2c00e30): the call to the
// usleep wrapper, after mov edi, [rbx+0x8d4] (MainThreadUpdateTime); sub edi,
// eax (the pass's milliseconds); jbe past the call.
constexpr std::uint64_t kSleepCall = 0x2c0137d;
constexpr std::uint64_t kUsleep = 0x248bac0;  // sceKernelUsleep for an argument above 0
constexpr std::uint64_t kSleepSetup = 0x2c01373;
constexpr std::uint8_t kSleepSetupBytes[] = {0x8b, 0xbb, 0xd4, 0x08, 0x00, 0x00, 0x29, 0xc7, 0x76, 0x05};
// The longest a wait lasts whatever the game asks: its 33333 us, with room.
constexpr std::int32_t kLongestWaitUs = 100000;

enum Mode : int { kOff, kCensus, kWake };
std::atomic<int> g_mode{kOff};

std::mutex g_mu;
std::condition_variable g_cv;
std::uint64_t g_frames = 0;  // frames begun (sound_wake_frame), under g_mu

// The sound thread's own: the frames begun as its last pass began, and as its
// last pass ended; when its pass began and its last ended.
std::uint64_t g_seen = 0, g_ended_at = 0, g_pass_t0 = 0, g_end_t = 0;

// The census since the last report: passes by the frames begun between their
// ends and the last pass's (0, 1, 2, 3 or more), the passes' time (a woken
// pass's, from its wake to its end), the time between their ends, the waits
// the game's own sleep ended.
std::atomic<std::uint64_t> g_passes{0}, g_by_frames[4], g_pass_ns{0}, g_pass_max_ns{0}, g_period_ns{0}, g_timeouts{0};

std::uint64_t slide_of(ElfImage* image) { return image->mem.slide - kPreferredGuestSlide; }

void max_into(std::atomic<std::uint64_t>& a, std::uint64_t v) {
    std::uint64_t m = a.load(std::memory_order_relaxed);
    while (v > m && !a.compare_exchange_weak(m, v, std::memory_order_relaxed)) {
    }
}

// The sleep's call. rdi: the sleep the game asks, its period less the pass's
// milliseconds. Returns 1 when ours stood for the call (the wake), 0 to go on
// to the game's sleep (the census alone).
GUEST_ABI std::int64_t sleep_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t now = host_clock_monotonic_ns();
    const int mode = g_mode.load(std::memory_order_relaxed);
    const auto asked = static_cast<std::int32_t>(static_cast<std::uint32_t>(saved[5]));
    std::unique_lock<std::mutex> lock(g_mu);
    // (The first pass's end only starts the count: every frame before it would
    // be its.)
    if (g_end_t) {
        g_passes.fetch_add(1, std::memory_order_relaxed);
        g_by_frames[std::min<std::uint64_t>(g_frames - g_ended_at, 3)].fetch_add(1, std::memory_order_relaxed);
        g_period_ns.fetch_add(now - g_end_t, std::memory_order_relaxed);
    }
    g_ended_at = g_frames;
    g_end_t = now;
    if (mode != kWake) return 0;
    if (g_pass_t0) {
        g_pass_ns.fetch_add(now - g_pass_t0, std::memory_order_relaxed);
        max_into(g_pass_max_ns, now - g_pass_t0);
    }
    // No frame begun since this pass began: wait for one, or for the game's
    // own sleep. One begun meanwhile: the next pass goes at once.
    if (g_frames == g_seen) {
        const auto us = std::clamp<std::int32_t>(asked, 0, kLongestWaitUs);
        if (!g_cv.wait_for(lock, std::chrono::microseconds(us), [] { return g_frames != g_seen; }))
            g_timeouts.fetch_add(1, std::memory_order_relaxed);
    }
    g_seen = g_frames;
    lock.unlock();
    g_pass_t0 = host_clock_monotonic_ns();
    return 1;
}

bool place(ElfImage* image, int mode) {
    if (g_mode.load(std::memory_order_relaxed) != kOff) {
        g_mode.store(std::max(g_mode.load(std::memory_order_relaxed), mode), std::memory_order_relaxed);
        return true;
    }
    const std::uint64_t s = slide_of(image);
    const auto* at = static_cast<const std::uint8_t*>(guest_ptr(image->mem, s + kSleepSetup));
    if (std::memcmp(at, kSleepSetupBytes, sizeof kSleepSetupBytes) != 0) {
        host_log("sound: 0x%llx is not the sound thread's sleep; its passes keep their own period",
                 static_cast<unsigned long long>(kSleepSetup));
        return false;
    }
    g_mode.store(mode, std::memory_order_relaxed);
    if (!engine_call_site_hook(image, s + kSleepCall, s + kUsleep, reinterpret_cast<void*>(&sleep_hook))) {
        g_mode.store(kOff, std::memory_order_relaxed);
        host_log("sound: the sound thread's sleep (0x%llx) could not be hooked; its passes keep their own period",
                 static_cast<unsigned long long>(kSleepCall));
        return false;
    }
    return true;
}

}  // namespace

bool sound_wake_install(ElfImage* image) {
    if (!place(image, kWake)) return false;
    host_log("sound: the sound thread is woken once a frame (the frame clock): a pass for each frame's requests at any frame "
             "rate, the game's own sleep the longest it waits (engine/sound_wake.h)");
    return true;
}

void sound_wake_census_install(ElfImage* image) {
    if (g_mode.load(std::memory_order_relaxed) != kOff) return;
    if (place(image, kCensus))
        host_log("sound: BBHOST_SOUND_CENSUS=1: the sound thread's passes counted by the frames begun between them (its sleep "
                 "left as it is)");
}

void sound_wake_frame() {
    if (g_mode.load(std::memory_order_relaxed) == kOff) return;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_frames;
    }
    g_cv.notify_one();
}

std::string sound_wake_report() {
    const int mode = g_mode.load(std::memory_order_relaxed);
    if (mode == kOff) return {};
    const std::uint64_t passes = g_passes.exchange(0, std::memory_order_relaxed);
    std::uint64_t by[4];
    for (int i = 0; i < 4; ++i) by[i] = g_by_frames[i].exchange(0, std::memory_order_relaxed);
    const std::uint64_t pass_ns = g_pass_ns.exchange(0, std::memory_order_relaxed), max_ns = g_pass_max_ns.exchange(0, std::memory_order_relaxed);
    const std::uint64_t period_ns = g_period_ns.exchange(0, std::memory_order_relaxed);
    const std::uint64_t timeouts = g_timeouts.exchange(0, std::memory_order_relaxed);
    // The monotonic clock in milliseconds, as the AT9 decoder's census stamps
    // its starts (hle/ajm.cpp, BBHOST_AJM_STARTS), to cut a window by.
    const auto now_ms = static_cast<unsigned long long>(host_clock_monotonic_ns() / 1000000);
    char buf[360];
    if (mode == kWake) {
        std::snprintf(buf, sizeof(buf),
                      "sound: t=%llu %llu passes woken by the frame; frames begun between their ends 0/1/2/3+: %llu/%llu/%llu/%llu; "
                      "a pass %.3f ms (most %.3f), %.2f ms apart; %llu waits ended by the game's own sleep",
                      now_ms, static_cast<unsigned long long>(passes), static_cast<unsigned long long>(by[0]), static_cast<unsigned long long>(by[1]),
                      static_cast<unsigned long long>(by[2]), static_cast<unsigned long long>(by[3]),
                      passes ? static_cast<double>(pass_ns) / 1e6 / static_cast<double>(passes) : 0.0, static_cast<double>(max_ns) / 1e6,
                      passes ? static_cast<double>(period_ns) / 1e6 / static_cast<double>(passes) : 0.0,
                      static_cast<unsigned long long>(timeouts));
    } else {
        std::snprintf(buf, sizeof(buf),
                      "sound: t=%llu %llu passes on their own sleep (census); frames begun between their ends 0/1/2/3+: "
                      "%llu/%llu/%llu/%llu; %.2f ms apart",
                      now_ms, static_cast<unsigned long long>(passes), static_cast<unsigned long long>(by[0]), static_cast<unsigned long long>(by[1]),
                      static_cast<unsigned long long>(by[2]), static_cast<unsigned long long>(by[3]),
                      passes ? static_cast<double>(period_ns) / 1e6 / static_cast<double>(passes) : 0.0);
    }
    return buf;
}
