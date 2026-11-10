#pragma once

// The mouse's part of the follow camera's step (engine/mouse_camera.h), shared
// by both ways the step runs: our source of NS_SPRJ::ChrExFollowCam::Update
// (decomp/camera/follow_camera.cpp), a leaf on the game's thread as the game
// left it, and the hooks on the game's own update where ours is not in
// place. Everything here is lock-free and calls nothing of the host's, so a
// leaf can use it: the window adds the counts, the pad read publishes the
// settings and the input device, the camera takes the counts at the start
// of its update and turns by them after the stick.

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

// The free camera ran in this camera update: the stick's step or its fast
// turn. The hooks' asm stubs set it too (engine/mouse_camera.cpp).
extern "C" std::uint8_t bb_mouse_camera_free_now;

namespace mouse_camera {

inline std::uint64_t pack(float x, float y) {
    std::uint32_t a, b;
    std::memcpy(&a, &x, 4);
    std::memcpy(&b, &y, 4);
    return static_cast<std::uint64_t>(a) | static_cast<std::uint64_t>(b) << 32;
}
inline void unpack(std::uint64_t v, float& x, float& y) {
    const std::uint32_t a = static_cast<std::uint32_t>(v), b = static_cast<std::uint32_t>(v >> 32);
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
}
// x and y added to a packed pair.
inline void add(std::atomic<std::uint64_t>& to, float x, float y) {
    std::uint64_t was = to.load(std::memory_order_relaxed);
    for (;;) {
        float ox, oy;
        unpack(was, ox, oy);
        if (to.compare_exchange_weak(was, pack(ox + x, oy + y), std::memory_order_acq_rel, std::memory_order_relaxed))
            return;
    }
}

// The degrees a count at a sensitivity of 0..10 (the 0-100 setting's tenth):
// 0.005 at 0 to 0.4 at 10, each step of the setting about 4.5% further than
// the one below it - each whole number about half again as far - and 0.045 at
// the default 5. DS3's own scale (ds3_degrees_per_count) ran from 0.1 to 0.4:
// its slowest turned the camera 80 to 160 degrees for an inch of a mouse at
// 800 to 1600 counts an inch, and its two ends were only four times apart.
inline constexpr float kSlowestDegrees = 0.005f, kFastestDegrees = 0.4f;
inline float degrees_per_count(float sensitivity) {
    const float s = sensitivity < 0.0f ? 0.0f : sensitivity > 10.0f ? 10.0f : sensitivity;
    return kSlowestDegrees * std::pow(kFastestDegrees / kSlowestDegrees, s * 0.1f);
}
// The sensitivity (0..10) that turns `degrees` a count.
inline float sensitivity_for_degrees(float degrees) {
    if (!(degrees > kSlowestDegrees)) return 0.0f;
    const float s = 10.0f * std::log(degrees / kSlowestDegrees) / std::log(kFastestDegrees / kSlowestDegrees);
    return s < 10.0f ? s : 10.0f;
}
// DS3's degrees a count at its sensitivity of 0..10, 0.2 * (s * 0.15 + 0.5) as
// its pad object computes it: the scale an options file before version 8 saved.
inline float ds3_degrees_per_count(float sensitivity) {
    const float s = sensitivity < 0.0f ? 0.0f : sensitivity > 10.0f ? 10.0f : sensitivity;
    return ((s * 0.1f) * 1.5f + 0.5f) * 0.2f;
}
// A 0-100 setting saved on DS3's scale, on today's: the same degrees a count.
inline int setting_from_ds3(int setting) {
    const float s = sensitivity_for_degrees(ds3_degrees_per_count(static_cast<float>(setting) * 0.1f));
    const long i = std::lround(s * 10.0f);
    return i < 0 ? 0 : i > 100 ? 100 : static_cast<int>(i);
}

// The window's relative motion while the mouse turns the camera, since the
// camera last took it: SDL's relative motion in the device's own counts, raw
// and unaccelerated like the DirectInput counts DS3 reads. Motion from before
// camera mode began is not in it (host/window.cpp).
inline std::atomic<std::uint64_t> g_counts{0};

// What the pad read publishes on every read: the degrees a count at the
// sensitivity set (float bits), and these flags.
enum : std::uint32_t {
    kInvertX = 1,
    kInvertY = 2,
    // The camera's own turns are held: the last input was the keyboard and
    // mouse, the mouse turns the camera, and the auto-rotation setting is off.
    kHoldAutoRotation = 4,
};
inline std::atomic<std::uint32_t> g_degrees_per_count{0};
inline std::atomic<std::uint32_t> g_flags{0};

// The camera's update, as the camera's own thread sees it: the counts taken
// at its start, and whether the free camera ran in the update before.
struct Frame {
    float dx = 0.0f, dy = 0.0f;
    bool free_before = false;
    bool hold = false;  // auto-rotation held in this update (kHoldAutoRotation as it began)
};
inline Frame g_frame;

// Counts taken in updates after one without the free camera - lock-on, or the
// game turning the camera - and how many such updates: the lock-on flick's,
// which the pad read decides (mouse_camera_flick).
inline std::atomic<std::uint64_t> g_flick_counts{0};
inline std::atomic<std::uint32_t> g_flick_updates{0};

// The exit report's.
inline std::atomic<std::uint64_t> g_updates{0}, g_turns{0}, g_stick_frames{0}, g_held{0};

// The camera's own turns as the character moves: the four stores the
// community patch "Disable Camera Auto Rotation via Movement" (Imedved, Kyo)
// makes no-ops, each vmovss [r13+N], xmm - 9 bytes - in the game's update.
// Holding auto-rotation skips them: ours does not store, and where the game's
// update runs they are swapped for nine nops while it is held.
struct HoldSite {
    std::uint64_t bn;
    std::uint8_t game[9];
};
inline constexpr HoldSite kHoldSites[4] = {
    {0x183c6e8, {0xc4, 0xc1, 0x7a, 0x11, 0x85, 0x44, 0x01, 0x00, 0x00}},  // the yaw read back from where the camera is
    {0x183c870, {0xc4, 0xc1, 0x7a, 0x11, 0xad, 0x44, 0x01, 0x00, 0x00}},  // the auto turn's step of the yaw
    {0x183c984, {0xc4, 0xc1, 0x7a, 0x11, 0x8d, 0x30, 0x01, 0x00, 0x00}},  // the pitch return's weight ramping down
    {0x183dde6, {0xc4, 0xc1, 0x7a, 0x11, 0x85, 0x94, 0x02, 0x00, 0x00}},  // the turn away from a wall
};
inline constexpr std::uint8_t kNop9[9] = {0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
// The sites the mouse holds: all but the auto turn's step. An auto turn has
// one cause - the camera's turn-behind request (+0x262), which only the lock-on
// handler makes, when the lock-on button finds no target (sub_1c8e900 at
// 0x1c90d59, through the camera manager's +0x1141) - so holding that step
// holds no turn of the camera's own, only R3's recentre. The patch's own nops
// at it are still mirrored where a player applied them.
inline constexpr unsigned kHeldWithMouse = 0b1101;

// Every camera update, first: this update's counts.
inline void begin_update() {
    g_updates.fetch_add(1, std::memory_order_relaxed);
    float dx, dy;
    unpack(g_counts.exchange(0, std::memory_order_acq_rel), dx, dy);
    g_frame.dx = dx;
    g_frame.dy = dy;
    g_frame.hold = (g_flags.load(std::memory_order_relaxed) & kHoldAutoRotation) != 0;
    if (g_frame.hold) g_held.fetch_add(1, std::memory_order_relaxed);
    g_frame.free_before = __atomic_load_n(&bb_mouse_camera_free_now, __ATOMIC_RELAXED) != 0;
    __atomic_store_n(&bb_mouse_camera_free_now, 0, __ATOMIC_RELAXED);
    if (!g_frame.free_before) {
        if (dx != 0.0f || dy != 0.0f) add(g_flick_counts, dx, dy);
        g_flick_updates.fetch_add(1, std::memory_order_relaxed);
    }
}

// The free camera ran in this update.
inline void free_camera_ran() { __atomic_store_n(&bb_mouse_camera_free_now, 1, __ATOMIC_RELAXED); }

// Whether this update holds auto-rotation - fixed when it began, whatever the
// pad read publishes meanwhile.
inline bool held() { return g_frame.hold; }

// DS3's step, after the stick has turned the free camera: when the stick has
// not, the pitch and yaw turn by this update's counts - an angle a count, no
// frame time, no dead zone, no cap - the pitch held within the stick step's
// limits and the yaw kept within a turn either way. Whether it turned (the
// update then had camera input, as a stick that turned it does).
inline bool turn(float stick_pitch, float stick_yaw, float& pitch, float& yaw, float pitch_min, float pitch_max) {
    const float dx = g_frame.dx, dy = g_frame.dy;
    if (stick_pitch != 0.0f || stick_yaw != 0.0f) {
        if (dx != 0.0f || dy != 0.0f) g_stick_frames.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (dx == 0.0f && dy == 0.0f) return false;
    constexpr float kRadiansPerDegree = 0.0174532924f;  // DS3's constant, the float of pi/180
    constexpr float kPi = 3.14159274f, kTwoPi = 6.28318548f;
    const std::uint32_t flags = g_flags.load(std::memory_order_relaxed);
    const std::uint32_t bits = g_degrees_per_count.load(std::memory_order_relaxed);
    float per_count;
    std::memcpy(&per_count, &bits, 4);
    // DS3's signs: a count is negated unless the axis is inverted, and the
    // camera subtracts.
    const float mouse_pitch = ((flags & kInvertY) ? 1.0f : -1.0f) * dy * per_count;
    const float mouse_yaw = ((flags & kInvertX) ? 1.0f : -1.0f) * dx * per_count;
    pitch -= mouse_pitch * kRadiansPerDegree;
    yaw -= mouse_yaw * kRadiansPerDegree;
    pitch = pitch_min > pitch ? pitch_min : (pitch_max < pitch ? pitch_max : pitch);
    // The game reads the yaw back from where the camera is each update, which
    // bounds it; while auto-rotation is held it does not, so the mouse bounds
    // it here. A turn a step: no mouse turns the camera 64 times in an update.
    for (int i = 0; i < 64 && yaw > kPi; ++i) yaw -= kTwoPi;
    for (int i = 0; i < 64 && yaw < -kPi; ++i) yaw += kTwoPi;
    g_turns.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// The step on the game's camera object, at the free camera's store (the hook
// there, engine/mouse_camera.cpp): pitch +0x140, yaw +0x144, its limits
// +0x1ec (the highest) and +0x1f0 (the lowest), and the update's input byte
// set when the mouse turned it.
inline bool turn_camera(std::uint8_t* cam, std::uint8_t* input, float stick_pitch, float stick_yaw) {
    free_camera_ran();
    float pitch, yaw, lowest, highest;
    std::memcpy(&pitch, cam + 0x140, 4);
    std::memcpy(&yaw, cam + 0x144, 4);
    std::memcpy(&highest, cam + 0x1ec, 4);
    std::memcpy(&lowest, cam + 0x1f0, 4);
    if (!turn(stick_pitch, stick_yaw, pitch, yaw, lowest, highest)) return false;
    std::memcpy(cam + 0x140, &pitch, 4);
    std::memcpy(cam + 0x144, &yaw, 4);
    *input = 1;
    return true;
}

}  // namespace mouse_camera
