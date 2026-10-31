#include "guest_abi.h"
#include "engine/menu_close.h"

#include "core/elf.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "hle/modules.h"
#include "log.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

// The close's wait condition: (functor {window, label}) -> the window's root
// clip is playing `label`. The task waits while it is true.
constexpr std::uint64_t kStillFading = 0x1ed7580;
constexpr std::uint8_t kStillFadingPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x70};
// The loading screen's window class: it closes (answer 2) on the flip of the
// first in-game frame. The functor's window is at +8.
constexpr unsigned long long kLoadingScreenVtable = 0x579c460;
// BBHOST_MENU_LOG: Close (window, answer) and the task that writes the answer
// (functor {window, answer}: window+0x1c0 = answer).
constexpr std::uint64_t kClose = 0x1ed6820;
constexpr std::uint8_t kClosePrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                           0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xc8, 0x02, 0x00, 0x00};
// The message and Yes/No boxes' OK (sub_1feaad0, vtable slot 11): nothing
// until the time the box has been up (+0x1f10, seconds, which their update
// adds to) reaches its minimum (+0xde0).
constexpr std::uint64_t kBoxConfirm = 0x1feaad0;
constexpr std::uint8_t kBoxConfirmPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                0x53, 0x48, 0x81, 0xec, 0x88, 0x01, 0x00, 0x00};
constexpr std::uint64_t kAnswer = 0x1ed77b0;
constexpr std::uint8_t kAnswerPrologue[] = {0x48, 0x8b, 0x47, 0x08, 0x48, 0x8b, 0x4f, 0x10,
                                            0x48, 0x89, 0x88, 0xc0, 0x01, 0x00, 0x00};

std::uint64_t g_slide = 0;

std::uint64_t rd64(std::uint64_t a) {
    std::uint64_t v = 0;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(a)), sizeof(v));
    return v;
}

// A window's class, as its vtable's Binary Ninja address.
unsigned long long vtable_of(std::uint64_t window) {
    return static_cast<unsigned long long>(rd64(window) - g_slide + kPreferredGuestSlide);
}

GUEST_ABI std::int64_t still_fading_hook(std::uint64_t, std::uint64_t* saved) {
    // The loading screen's window keeps the game's wait: it closes on the
    // world's first frame, and its fade-out covers the ~50 frames the world
    // needs to settle (the player's model drawn white, no colour grading).
    if (vtable_of(rd64(saved[5] + 8)) == kLoadingScreenVtable) return 0;
    saved[-1] = 0;  // not playing: the wait is over
    return 1;
}

bool g_log = false;

GUEST_ABI std::int64_t box_confirm_hook(std::uint64_t, const std::uint64_t* saved) {
    float* const shown = reinterpret_cast<float*>(static_cast<std::uintptr_t>(saved[5] + 0x1f10));
    const float least = *reinterpret_cast<const float*>(static_cast<std::uintptr_t>(saved[5] + 0xde0));
    if (g_log)
        host_log("menus: box %llx (vtable %llx) OK after %.3f s of its %.3f at flip %llu", static_cast<unsigned long long>(saved[5]),
                 vtable_of(saved[5]), static_cast<double>(*shown), static_cast<double>(least),
                 static_cast<unsigned long long>(hle_video_flip_count()));
    if (*shown < least) *shown = least;  // up long enough: the OK goes through
    return 0;
}

GUEST_ABI std::int64_t close_hook(std::uint64_t, const std::uint64_t* saved) {
    host_log("menus: window %llx (vtable %llx) closes with %lld at flip %llu", static_cast<unsigned long long>(saved[5]),
             vtable_of(saved[5]), static_cast<long long>(saved[4]), static_cast<unsigned long long>(hle_video_flip_count()));
    return 0;
}

GUEST_ABI std::int64_t answer_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t window = rd64(saved[5] + 8);
    host_log("menus: window %llx (vtable %llx) answers %lld at flip %llu", static_cast<unsigned long long>(window),
             vtable_of(window), static_cast<long long>(rd64(saved[5] + 0x10)),
             static_cast<unsigned long long>(hle_video_flip_count()));
    return 0;
}

bool env_on(const char* name) {
    const char* e = std::getenv(name);
    return e && e[0] == '1';
}

}  // namespace

void menu_close_install(ElfImage* image) {
    if (!image || image->sha256 != kEboot109Sha256) return;
    g_slide = image->mem.slide;
    const auto at = [](std::uint64_t a) { return g_slide + (a - kPreferredGuestSlide); };
    g_log = env_on("BBHOST_MENU_LOG");
    if (env_on("BBHOST_MENU_FADE_WAIT")) {
        host_log("menus: a closing window answers after its fade-out and a box takes OK after its minimum time, as the game has "
                 "it (BBHOST_MENU_FADE_WAIT=1)");
    } else {
        if (engine_prologue_hook(image, at(kStillFading), kStillFadingPrologue, sizeof(kStillFadingPrologue),
                                 reinterpret_cast<void*>(&still_fading_hook))) {
            host_log("menus: a closing window answers at once, not after its fade-out (BBHOST_MENU_FADE_WAIT=1 waits)");
        } else {
            host_log("menus: the close's fade-out wait (0x%llx) is not the expected code; closing windows wait for it",
                     static_cast<unsigned long long>(kStillFading));
        }
        if (engine_prologue_hook(image, at(kBoxConfirm), kBoxConfirmPrologue, sizeof(kBoxConfirmPrologue),
                                 reinterpret_cast<void*>(&box_confirm_hook))) {
            host_log("menus: a message or Yes/No box takes OK as soon as it is up (BBHOST_MENU_FADE_WAIT=1 waits)");
        } else {
            host_log("menus: the boxes' OK (0x%llx) is not the expected code; a box takes OK after its minimum time",
                     static_cast<unsigned long long>(kBoxConfirm));
        }
    }
    if (g_log) {
        engine_prologue_hook(image, at(kClose), kClosePrologue, sizeof(kClosePrologue), reinterpret_cast<void*>(&close_hook));
        engine_prologue_hook(image, at(kAnswer), kAnswerPrologue, sizeof(kAnswerPrologue), reinterpret_cast<void*>(&answer_hook));
    }
}
