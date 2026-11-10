#include "engine/mouse_camera.h"

// What DS3 does, read from its executable (the 1.15 in-game dump; the Steam
// build keeps these functions encrypted on disk):
//
// - DLUID::MouseDevice polls DirectInput (GetDeviceState into DIMOUSESTATE2)
//   and converts lX/lY to float unscaled: the camera works in raw counts.
// - The game's pad object reads them as two logical axes, signs them by the
//   PC key config's mouse invert bytes (-1 when not inverted), and scales
//   them by ((sensitivity * 0.1) * 1.5 + 0.5) * 0.2, the sensitivity a float
//   0..10 defaulting to 5 (CSPcKeyConfig +0x710, +0x715, +0x716). They are 0
//   while the mouse is a pointer (a menu up).
// - PadManipulator keeps them beside the stick, not in it (+0xb0 against the
//   stick's +0x80).
// - ChrExFollowCam's update, after the stick has turned the camera:
//       if (stick moved this frame) skip;             // the stick has priority
//       pitch -= mouse.pitch * (pi / 180);
//       yaw   -= mouse.yaw   * (pi / 180);
//       clamp pitch to the stick path's limits; camera input present
//   and that is all - the angle is per count, whatever the frame time.
// - Locked on, a frame whose movement is over 50 counts switches the target
//   in that direction, then waits 0.5 s unless the movement drops under 25
//   counts first (the lock-on manager, sub_14085eac0 in the dump).
//
// Bloodborne's ChrExFollowCam::Update (sub_183ac60) is the same function
// without the mouse. Its free camera turns by the stick in the block at
// 0x183cd75 - angles += dt * stick * speed, both wrapped, pitch clamped to
// [+0x1f0, +0x1ec] - and stores them at 0x183ce54; the next block, 0x183ce67,
// restarts the camera's return to centre when the frame had input
// ([rbp-0x3c4]). DS3's step goes between the two. Lock-on and the scripted
// camera turns reach 0x183ce67 by other paths and never through the store, so
// the store is also how the step knows the free camera ran.
//
// Where the step runs (engine/mouse_camera_step.h holds it):
// - Our source of the update in place (the decomp list, decomp/camera/
//   follow_camera.cpp): ours takes the counts first thing and turns after the
//   stick, where DS3 does. No hooks.
// - The game's update in place: hooks on its prologue, the free camera's
//   store and the fast turn's two exits, as below.
// - A compare run (BBHOST_DECOMP_COMPARE=1): the stand-in runs the game's
//   update and ours on the same input, so the game's gets the store hooks and
//   the stand-in takes the counts for both (mouse_camera_compare_begin).
//
// Auto-rotation: as the character moves, the game turns the camera by itself
// - it reads the yaw back from where the lagging camera is, the pitch return's
// weight ramps down, and the new camera turns away from walls. Those are three
// of kHoldSites, the community patch's four stores; its fourth, the auto
// turn's step, only R3's recentre reaches, and the mouse leaves it be. While
// the keyboard and mouse were used last and the mouse turns the camera, the
// three are held (unless the setting says otherwise): ours skips them, and
// where the game's update runs they are swapped for nops - the patch's bytes -
// and back when a controller is picked up.

#include "core/elf.h"
#include "core/memory.h"
#include "core/thunk.h"
#include "decomp/decomp.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "engine/mouse_camera_step.h"
#include "engine/mouse_camera_stubs.h"
#include "guest_abi.h"
#include "log.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

using ull = unsigned long long;
using Clock = std::chrono::steady_clock;

// NS_SPRJ::ChrExFollowCam::Update, the player's camera in play: push rbp;
// mov rbp, rsp; push r15..rbx; sub rsp, 0x3d8 (20 bytes).
constexpr std::uint64_t kUpdate = 0x183ac60;
constexpr std::uint8_t kUpdatePrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                            0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xd8, 0x03, 0x00, 0x00};
// Its free camera's store: vmovss [r13+0x140], xmm0; vextractps [r13+0x144],
// xmm1, 1 (19 bytes), reached only from the stick step before it.
constexpr std::uint64_t kTurnStore = 0x183ce54;
constexpr std::uint8_t kTurnStoreBytes[] = {0xc4, 0xc1, 0x7a, 0x11, 0x85, 0x40, 0x01, 0x00, 0x00, 0xc4,
                                            0xc3, 0x79, 0x17, 0x8d, 0x44, 0x01, 0x00, 0x00, 0x01};
constexpr std::uint64_t kTurnResume = 0x183ce67;
// The fast turn's two exits: vmovss [r13+0x140], xmm0; jmp 0x183ce67.
constexpr std::uint64_t kRampExits[] = {0x183f9bc, 0x183fabf};
constexpr std::uint8_t kRampExitBytes[2][14] = {
    {0xc4, 0xc1, 0x7a, 0x11, 0x85, 0x40, 0x01, 0x00, 0x00, 0xe9, 0x9d, 0xd4, 0xff, 0xff},
    {0xc4, 0xc1, 0x7a, 0x11, 0x85, 0x40, 0x01, 0x00, 0x00, 0xe9, 0x9a, 0xd3, 0xff, 0xff}};
constexpr std::size_t kPitch = 0x140, kYaw = 0x144;

// The lock-on switch, in counts per frame at the 60 fps DS3 runs at; here as
// rates, so Bloodborne's 30 fps asks for the same movement.
constexpr float kFlickSwitch = 50.0f * 60.0f, kFlickRearm = 25.0f * 60.0f;
constexpr float kFlickCooldown = 0.5f;  // seconds, as DS3 sets it after a switch
// How long the stick shows a flick: two frames at 30 fps, enough for the game
// to see a push and a release.
constexpr auto kFlickHold = std::chrono::milliseconds(80);

enum class Mode { None, Ours, Hooks, Compare };
Mode g_mode = Mode::None;
const char* mode_name() {
    return g_mode == Mode::Ours ? "ours" : g_mode == Mode::Compare ? "a compare run" : "hooks";
}

const bool g_log = [] {
    const char* e = std::getenv("BBHOST_MOUSE_CAMERA_LOG");
    return e && e[0] == '1';
}();

// The lock-on flick, decided on the pad read.
std::mutex g_mu;
Clock::time_point g_last_read;
float g_cooldown = 0.0f;
float g_flick_x = 0.0f, g_flick_y = 0.0f;
Clock::time_point g_flick_until;
std::atomic<std::uint64_t> g_flicks{0};
std::atomic<int> g_hold_logged{-1};

// The hold sites in the game's update, where it runs: their code, whether the
// nops are in, and whether a patch had already made them nops (then they stay).
std::uint8_t* g_hold_code[4] = {};
GuestMemory* g_hold_mem = nullptr;
bool g_hold_nopped = false, g_hold_fixed = false;

void write_code(std::uint8_t* at, const std::uint8_t* bytes, std::size_t n) {
    const std::uint64_t va = reinterpret_cast<std::uint64_t>(at);
    const std::uint64_t lo = va & ~0xfffull, hi = (va + n + 0xfff) & ~0xfffull;
    if (!guest_protect_rwx(g_hold_mem, lo, hi - lo)) return;
    std::memcpy(at, bytes, n);
    guest_protect_rx(g_hold_mem, lo, hi - lo);
}

// The game's update holding auto-rotation when ours would: the four stores
// as nops, or as the game's. On the camera's thread, before the update runs.
void sync_hold(bool hold) {
    if (!g_hold_mem || g_hold_fixed || hold == g_hold_nopped) return;
    for (int i = 0; i < 4; ++i) {
        if (g_hold_code[i] && (mouse_camera::kHeldWithMouse >> i & 1))
            write_code(g_hold_code[i], hold ? mouse_camera::kNop9 : mouse_camera::kHoldSites[i].game, 9);
    }
    g_hold_nopped = hold;
}

// Every camera update of the game's, first (the prologue hook): this update's
// counts, and the game's code holding auto-rotation or not.
GUEST_ABI std::int64_t update_hook(std::uint64_t, const std::uint64_t*) {
    mouse_camera::begin_update();
    sync_hold(mouse_camera::held());
    // When the free camera stops (lock-on, a scripted turn) and comes back.
    static bool was_turning = false;
    static int switches = 0;
    const bool turned_before = mouse_camera::g_frame.free_before;
    if (mouse_camera::g_updates.load(std::memory_order_relaxed) > 2 && turned_before != was_turning &&
        (g_log || switches < 8)) {
        ++switches;
        host_log("mouse camera: the free camera %s", turned_before ? "is back" : "stopped (locked on, or the game turns it)");
    }
    was_turning = turned_before;
    return 0;
}

// The free camera of the game's update, after the stick has turned it: DS3's
// mouse step.
GUEST_ABI void free_camera_hook(std::uint8_t* cam, std::uint8_t* input, float stick_pitch, float stick_yaw) {
    float was_pitch = 0.0f, was_yaw = 0.0f;
    std::memcpy(&was_pitch, cam + kPitch, 4);
    std::memcpy(&was_yaw, cam + kYaw, 4);
    if (!mouse_camera::turn_camera(cam, input, stick_pitch, stick_yaw)) return;
    const std::uint64_t n = mouse_camera::g_turns.load(std::memory_order_relaxed);
    if (n == 1 || (g_log && n < 400)) {
        float pitch = 0.0f, yaw = 0.0f;
        std::memcpy(&pitch, cam + kPitch, 4);
        std::memcpy(&yaw, cam + kYaw, 4);
        host_log("mouse camera: %+.0f,%+.0f counts: pitch %.4f -> %.4f, yaw %.4f -> %.4f", mouse_camera::g_frame.dx,
                 mouse_camera::g_frame.dy, was_pitch, pitch, was_yaw, yaw);
    }
}

// A jmp [rip+0] to `dest` over the first bytes of `at`, the rest int3.
bool redirect(ElfImage* image, std::uint64_t va, std::size_t n, const void* dest) {
    auto* code = static_cast<std::uint8_t*>(guest_ptr(image->mem, va));
    const std::uint64_t lo = va & ~0xfffull, hi = (va + n + 0xfff) & ~0xfffull;
    if (!code || !guest_protect_rwx(&image->mem, lo, hi - lo)) return false;
    const std::uint64_t to = reinterpret_cast<std::uint64_t>(dest);
    code[0] = 0xff;
    code[1] = 0x25;  // jmp [rip+0]
    std::memset(code + 2, 0, 4);
    std::memcpy(code + 6, &to, 8);
    std::memset(code + 14, 0xcc, n - 14);
    guest_protect_rx(&image->mem, lo, hi - lo);
    return true;
}

}  // namespace

float mouse_camera_degrees_per_count(float sensitivity) { return mouse_camera::degrees_per_count(sensitivity); }

int mouse_camera_setting_from_ds3(int setting) { return mouse_camera::setting_from_ds3(setting); }

bool mouse_camera_installed() { return g_mode != Mode::None; }

void mouse_camera_install(ElfImage* image) {
    if (!image || image->sha256 != kEboot109Sha256) return;
    const auto at = [&](std::uint64_t bn) { return image->mem.slide + (bn - kPreferredGuestSlide); };
    const int placed = decomp_placed("NS_SPRJ::ChrExFollowCam::Update");
    if (placed == 1) {
        g_mode = Mode::Ours;
        host_log("mouse camera: by an angle a count, as Dark Souls III turns it - %.3f degrees at sensitivity 5, after the "
                 "stick and only without it; a flick switches the lock-on target - in our follow camera",
                 mouse_camera_degrees_per_count(5));
        return;
    }
    // The game's update runs, alone or beside ours: its free camera's store.
    auto* store = static_cast<std::uint8_t*>(guest_ptr(image->mem, at(kTurnStore)));
    if (!store || std::memcmp(store, kTurnStoreBytes, sizeof(kTurnStoreBytes)) != 0) {
        host_log("mouse camera: refused, 0x%llx is not the follow camera's store; the mouse turns it through the stick",
                 static_cast<ull>(kTurnStore));
        return;
    }
    // Its prologue takes the counts, unless ours is a compare run's stand-in
    // there, which takes them for both.
    if (placed == 0 && !engine_prologue_hook(image, at(kUpdate), kUpdatePrologue, sizeof(kUpdatePrologue),
                                             reinterpret_cast<void*>(&update_hook))) {
        host_log("mouse camera: refused, 0x%llx is not the follow camera's update; the mouse turns it through the stick",
                 static_cast<ull>(kUpdate));
        return;
    }
    g_mouse_camera_host = thunk_wrap(reinterpret_cast<void*>(&free_camera_hook));
    g_mouse_camera_resume = at(kTurnResume);
    if (!redirect(image, at(kTurnStore), sizeof(kTurnStoreBytes), reinterpret_cast<const void*>(&bb_mouse_camera_stub))) {
        host_log("mouse camera: cannot write the follow camera's code; the mouse turns it through the stick");
        return;
    }
    // The fast turn's exits, so that "the free camera did not run" is lock-on
    // or the game turning the camera, never the stick held over.
    for (int i = 0; i < 2; ++i) {
        auto* exit = static_cast<std::uint8_t*>(guest_ptr(image->mem, at(kRampExits[i])));
        if (!exit || std::memcmp(exit, kRampExitBytes[i], 14) != 0 ||
            !redirect(image, at(kRampExits[i]), 14, reinterpret_cast<const void*>(&bb_mouse_camera_ramp_stub))) {
            host_log("mouse camera: 0x%llx is not the fast turn's exit; a lock-on flick may also show while the stick is held "
                     "over", static_cast<ull>(kRampExits[i]));
        }
    }
    // The hold sites, to swap. A patch that made them nops already keeps them.
    g_hold_mem = &image->mem;
    for (int i = 0; i < 4; ++i) {
        auto* code = static_cast<std::uint8_t*>(guest_ptr(image->mem, at(mouse_camera::kHoldSites[i].bn)));
        if (code && std::memcmp(code, mouse_camera::kHoldSites[i].game, 9) == 0) {
            g_hold_code[i] = code;
        } else if (code && std::memcmp(code, mouse_camera::kNop9, 9) == 0) {
            g_hold_fixed = true;
        }
    }
    if (g_hold_fixed) host_log("mouse camera: a patch holds the camera's auto-rotation already, with any device");
    g_mode = placed == 2 ? Mode::Compare : Mode::Hooks;
    host_log("mouse camera: by an angle a count, as Dark Souls III turns it - %.3f degrees at sensitivity 5, after the "
             "stick and only without it; a flick switches the lock-on target - by hooks on the game's follow camera%s",
             mouse_camera_degrees_per_count(5), placed == 2 ? " (beside ours, compared)" : "");
}

void mouse_camera_publish(float sensitivity, bool invert_x, bool invert_y, bool hold_auto_rotation) {
    const float per_count = mouse_camera_degrees_per_count(sensitivity);
    std::uint32_t bits;
    std::memcpy(&bits, &per_count, 4);
    mouse_camera::g_degrees_per_count.store(bits, std::memory_order_relaxed);
    mouse_camera::g_flags.store((invert_x ? mouse_camera::kInvertX : 0u) | (invert_y ? mouse_camera::kInvertY : 0u) |
                                    (hold_auto_rotation ? mouse_camera::kHoldAutoRotation : 0u),
                                std::memory_order_relaxed);
    // On a change: the device switches with every key or stick the player
    // reaches for, so only the first few.
    static int changes = 0;
    if (g_hold_logged.exchange(hold_auto_rotation ? 1 : 0) != (hold_auto_rotation ? 1 : 0) && (g_log || changes < 6)) {
        ++changes;
        host_log("mouse camera: auto-rotation %s", hold_auto_rotation ? "held (the keyboard and mouse were used last)"
                                                                      : "the game's own");
    }
}

bool mouse_camera_flick(float& x, float& y) {
    std::lock_guard<std::mutex> lock(g_mu);
    const Clock::time_point now = Clock::now();
    float dt = g_last_read != Clock::time_point{} ? std::chrono::duration<float>(now - g_last_read).count() : 0.0f;
    if (dt > 0.25f) dt = 0.25f;  // a load or a pause is not a fast mouse
    g_last_read = now;
    // The counts of the camera updates since the last read that came after
    // one without the free camera - lock-on, or the game turning it.
    float dx = 0.0f, dy = 0.0f;
    mouse_camera::unpack(mouse_camera::g_flick_counts.exchange(0, std::memory_order_acq_rel), dx, dy);
    if (mouse_camera::g_flick_updates.exchange(0, std::memory_order_acq_rel) != 0 && dt > 0.0f) {
        const float m = std::sqrt(dx * dx + dy * dy);
        const float rate = m / dt;
        if (rate < kFlickRearm) {
            g_cooldown = 0.0f;
        } else {
            g_cooldown = g_cooldown - dt < 0.0f ? 0.0f : g_cooldown - dt;
        }
        if (g_cooldown <= 0.0f && rate > kFlickSwitch) {
            g_flick_x = dx / m;
            g_flick_y = dy / m;
            g_flick_until = now + kFlickHold;
            g_cooldown = kFlickCooldown;
            const std::uint64_t n = g_flicks.fetch_add(1, std::memory_order_relaxed);
            if (g_log || n < 4) {
                host_log("mouse camera: a flick of %.0f counts (%.0f a second) switches the lock-on target, toward %+.2f,%+.2f",
                         m, rate, g_flick_x, g_flick_y);
            }
        }
    }
    if (now >= g_flick_until) return false;
    x = g_flick_x;
    y = g_flick_y;
    return true;
}

void mouse_camera_compare_begin() {
    mouse_camera::begin_update();
    sync_hold(mouse_camera::held());
}

void mouse_camera_report() {
    if (g_mode == Mode::None) return;
    host_log("mouse camera (%s): %llu camera updates, %llu turned by the mouse, %llu where the stick had the camera, %llu "
             "lock-on flicks; auto-rotation held in %llu",
             mode_name(), static_cast<ull>(mouse_camera::g_updates.load()), static_cast<ull>(mouse_camera::g_turns.load()),
             static_cast<ull>(mouse_camera::g_stick_frames.load()), static_cast<ull>(g_flicks.load()),
             static_cast<ull>(mouse_camera::g_held.load()));
}
