// Isometric: a Diablo-like camera - high above the hunter, looking down at a
// fixed angle, turned left and right by the right stick (or the mouse) and
// zoomed by the wheel, never pulled in by walls - with the omni-roll built in.
//
//   [plugins]
//   iso = true
//   [iso]
//   enabled = true      # the isometric view
//   pitch = 55.0        # degrees below the horizon
//   distance = 14.0     # metres from the hunter (the wheel zooms from here)
//   fov = 35.0          # vertical field of view, degrees
//   height = 1.0        # the point looked at, metres above the hunter's feet
//   follow = 0.08       # seconds the view takes to catch up with the hunter (0: at once)
//   omni_roll = true    # 4-way omni quicksteps (Modern Movement's, without its quick item)
//
// Two cameras. The game keeps its own - ChrFollowCam (the camera manager's
// +0x60) orbits the hunter, and the manager's blend of it (and of the lock-on
// camera) is what the controls move by and the aim reads. The scene's camera
// is a copy of that, made each frame and drawn from (plugin API 20's on_view).
// So the game's camera is kept level, at the yaw the view turns to (on_camera),
// and only the drawn copy looks down (on_view): the hunter walks where the stick
// points on the screen and aims straight ahead, not at the floor, and the drawn
// camera, placed by us at the hunter, is never pushed in by the map.
//
// ChrFollowCam's fields, from its debug page (0x183f000.., the labels' own
// offsets) and its update (sub_183ac60):
//   +0x140 CamRotX (pitch, rad: positive looks down; RotRange MaxX 70 / MinX -40 deg)
//   +0x144 CamRotY (yaw, rad), +0x150 TargetRotX
//   +0x15c IsHandControl - 1 while the stick (or mouse) turns it: 0x183b8cc sets it
//          when the input is not zero, not on a reset, and not while locked on
//          (+0x260) unless the lock camera adjustment (+0x276) is on
//   +0x1ec/+0x1f0 RotRange MaxX/MinX, +0x1f4/+0x1f8 RotRangeAtLock MaxX/MinX (deg)
//   +0x240 LockRotChaseRate - how fast the lock-on turns the camera to the target
//   +0x274 UpdateTargetRotXByLock, +0x275 UpdateTargetRotXByFrontLock,
//   +0x276 the lock camera adjustment ("ロックカメラ調整が有効か")
// The debug menu edits these same fields (its page holds pointers into the
// camera, not copies); the per-area LockCamParam rows feed some of them again,
// which is why they are written every frame rather than once.
//
// The view's yaw is ours. The follow camera's own yaw drifts by itself - the
// auto-rotation behind a running hunter, the lock-on turning it to the target -
// and only its changes made by hand (IsHandControl, not locked on) turn the
// view. The game's camera is then turned to the view's yaw and levelled, so the
// stick's "up" is the screen's up whatever the follow camera did.
#include "bbhost_plugin.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN,
    "iso",
    "Isometric",
    "0.1.0",
    "Kyo",
    "A Diablo-like camera: high above the hunter at a fixed angle, turned by the right stick or the mouse, zoomed by the "
    "wheel, never pulled in by walls; the aim stays level. With the omni-roll built in.",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, "view", "The view", "The right stick or the mouse turns it left and right; the wheel zooms.", nullptr},
    {BB_OPT_BOOL | BB_OPT_LIVE, "enabled", "Isometric camera",
     "The view from high above the hunter. Off: the game's own camera.", "true"},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "pitch", "Angle", "Degrees the view looks down below the horizon.", "55.0", 25, 89},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "distance", "Distance", "Metres from the hunter; the wheel zooms in and out from it.", "14.0",
     3, 40},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "fov", "Field of view", "The view's vertical field of view, in degrees.", "35.0", 15, 75},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "height", "Look-at height", "The point the view centres on, metres above the hunter's feet.",
     "1.0", 0, 2.5},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "follow", "Follow lag",
     "Seconds the view takes to catch up with the hunter (0: fixed to them).", "0.08", 0, 0.5},
    {BB_OPT_HEADING, "play", "Gameplay", nullptr, nullptr},
    {BB_OPT_BOOL | BB_OPT_LIVE, "omni_roll", "Omni-roll",
     "Quicksteps and rolls in all four directions, as Modern Movement's (without its D-pad quick item).", "true"},
    {0}};

namespace {

const BbHostApi* api = nullptr;

struct Site {
    std::uint64_t address;
    std::uint32_t size;
    const char* expect;
    const char* bytes;
};
const Site kOmniRoll[] = {
#include "omni_roll.inc"
};
// Below this the sites are hooks in the game's own code; the rest are caves
// nothing reaches without them.
constexpr std::uint64_t kCaves = 0x4000000;

struct Settings {
    bool enabled = true;
    float pitch = 55.0f, distance = 14.0f, fov = 35.0f, height = 1.0f, follow = 0.08f;
    bool omni_roll = true;
} s;

constexpr float kPi = 3.14159265358979f;
constexpr float kZoomStep = 1.15f, kZoomNear = 3.0f, kZoomFar = 40.0f;

// The follow camera's fields (see the top).
constexpr std::size_t kRotX = 0x140, kRotY = 0x144, kTargetRotX = 0x150, kHandControl = 0x15c, kLocked = 0x260;
constexpr std::size_t kRotRange = 0x1ec;  // 4 floats: MaxX, MinX, AtLock MaxX, AtLock MinX
constexpr std::size_t kLockRotChaseRate = 0x240, kLockFlags = 0x274;  // 3 bytes

struct Camera {
    std::uint8_t* follow = nullptr;  // the ChrFollowCam we wrote to
    float saved_range[4]{};
    float saved_chase = 0;
    std::uint8_t saved_flags[3]{};
    bool have_view = false;
    float yaw = 0;           // the view's yaw (world: forward = (sin, 0, cos))
    float last_game = 0;     // the game camera's yaw as it made it last frame
    float zoom = 1.0f;
    float focus[3]{};
    bool have_focus = false;
    float right_sign = 1.0f;  // the game's right is +/-(cos, 0, -sin)
    bool view_ready = false;  // on_camera ran this frame: on_view may draw
    std::chrono::steady_clock::time_point last;
} cam;

bool omni_applied = false;
bool omni_owned[sizeof(kOmniRoll) / sizeof(kOmniRoll[0])]{};  // written by us, not found already in place

bool truthy(const char* v) { return v && (std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0); }

float number(const char* key, float fallback, float lo, float hi) {
    const char* v = api->config(key);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const float f = std::strtof(v, &end);
    if (end == v || !std::isfinite(f)) return fallback;
    return f < lo ? lo : f > hi ? hi : f;
}

void read_options() {
    const char* e = api->config("iso.enabled");
    s.enabled = !e || !*e || truthy(e);
    const char* o = api->config("iso.omni_roll");
    s.omni_roll = !o || !*o || truthy(o);
    s.pitch = number("iso.pitch", 55.0f, 25, 89);
    s.distance = number("iso.distance", 14.0f, 3, 40);
    s.fov = number("iso.fov", 35.0f, 15, 75);
    s.height = number("iso.height", 1.0f, 0, 2.5f);
    s.follow = number("iso.follow", 0.08f, 0, 0.5f);
}

float wrap(float a) {
    while (a > kPi) a -= 2 * kPi;
    while (a < -kPi) a += 2 * kPi;
    return a;
}

// The omni-roll: every site on, or the hooks back as the game had them (the
// caves stay - nothing jumps to them). A site already as it writes it belongs
// to the patch list's Modern Movement, which stays in charge of it.
void omni_roll(bool on) {
    if (on == omni_applied) return;
    int written = 0, already = 0, refused = 0;
    for (std::size_t i = 0; i < sizeof(kOmniRoll) / sizeof(kOmniRoll[0]); ++i) {
        const Site& t = kOmniRoll[i];
        if (on) {
            std::uint8_t now[4096];
            const bool in_place = t.size <= sizeof(now) && api->read(t.address, now, t.size) == 0 &&
                                  std::memcmp(now, t.bytes, t.size) == 0;
            if (in_place) {
                ++already;
                omni_owned[i] = false;
            } else if (api->patch(t.address, t.expect, t.bytes, t.size) == 0) {
                ++written;
                omni_owned[i] = true;
            } else {
                ++refused;
                omni_owned[i] = false;
            }
        } else if (omni_owned[i] && t.address < kCaves) {
            if (api->patch(t.address, t.bytes, t.expect, t.size) == 0) ++written;
            omni_owned[i] = false;
        }
    }
    omni_applied = on;
    if (on)
        api->log("iso: omni-roll on (%d sites written, %d already in place, %d refused)", written, already, refused);
    else
        api->log("iso: omni-roll off (%d hooks given back)", written);
}

std::uint8_t* follow_cam(BbCamera* c) {
    auto* manager = reinterpret_cast<std::uint8_t*>(c) - 0x10;
    std::uint64_t p = 0;
    std::memcpy(&p, manager + 0x60, 8);
    return reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(p));
}

template <typename T>
T field(const std::uint8_t* base, std::size_t at) {
    T v;
    std::memcpy(&v, base + at, sizeof(T));
    return v;
}
template <typename T>
void set_field(std::uint8_t* base, std::size_t at, T v) {
    std::memcpy(base + at, &v, sizeof(T));
}

// The follow camera's limits back as they were (the switch off, or another camera).
void give_back() {
    if (!cam.follow) return;
    std::memcpy(cam.follow + kRotRange, cam.saved_range, sizeof(cam.saved_range));
    set_field(cam.follow, kLockRotChaseRate, cam.saved_chase);
    std::memcpy(cam.follow + kLockFlags, cam.saved_flags, sizeof(cam.saved_flags));
    cam.follow = nullptr;
}

void take(std::uint8_t* f) {
    if (cam.follow == f) return;
    give_back();
    float range[4];
    std::memcpy(range, f + kRotRange, sizeof(range));
    // A follow camera: its RotRange in degrees, MaxX above MinX.
    if (!(range[0] > 0 && range[0] <= 90 && range[1] < range[0] && range[1] >= -90)) {
        static int complaints = 0;
        if (complaints++ < 4) api->log("iso: the camera manager's +0x60 is not a follow camera (RotRange %g/%g); left alone",
                                       static_cast<double>(range[0]), static_cast<double>(range[1]));
        return;
    }
    cam.follow = f;
    std::memcpy(cam.saved_range, range, sizeof(range));
    cam.saved_chase = field<float>(f, kLockRotChaseRate);
    std::memcpy(cam.saved_flags, f + kLockFlags, sizeof(cam.saved_flags));
    api->log("iso: follow camera %p (RotRange %g/%g, at lock %g/%g, lock chase %g)", static_cast<void*>(f),
             static_cast<double>(range[0]), static_cast<double>(range[1]), static_cast<double>(range[2]),
             static_cast<double>(range[3]), static_cast<double>(cam.saved_chase));
}

void release() {
    give_back();
    cam.have_view = false;
    cam.have_focus = false;
    cam.view_ready = false;
    api->input_take(nullptr, 0, 0, 0, 0);
}

// The game's camera: its yaw followed by hand only, then turned to the view's
// yaw and levelled - what the controls and the aim read.
void on_camera(BbCamera* c, void*) {
    cam.view_ready = false;
    if (!s.enabled || (api->loading_screen && api->loading_screen())) {
        if (cam.follow || cam.have_view) release();
        return;
    }
    std::uint8_t* f = follow_cam(c);
    if (!f) return;
    take(f);
    if (cam.follow != f) return;

    const float game = std::atan2(c->forward[0], c->forward[2]);
    const float level_len = std::sqrt(c->forward[0] * c->forward[0] + c->forward[2] * c->forward[2]);
    if (!std::isfinite(game) || level_len < 1e-3f) return;
    const bool hand = field<std::uint8_t>(f, kHandControl) != 0, locked = field<std::uint8_t>(f, kLocked) != 0;
    if (!cam.have_view) {
        cam.yaw = game;
        cam.zoom = 1.0f;
        cam.have_view = true;
        cam.last = std::chrono::steady_clock::now();
        api->input_take(nullptr, 0, 0, 1, 0);  // the wheel zooms the view, not the game's camera
        api->log("iso: on (angle %.0f, distance %.1f m, fov %.0f)", static_cast<double>(s.pitch),
                 static_cast<double>(s.distance), static_cast<double>(s.fov));
    } else if (hand && !locked) {
        cam.yaw = wrap(cam.yaw + wrap(game - cam.last_game));
    }
    cam.last_game = game;
    // ISO_LOG=1: twice a second, what the follow camera did and the view's yaw.
    static const bool trace = std::getenv("ISO_LOG") != nullptr;
    static int frames = 0;
    if (trace && frames++ % 30 == 0)
        api->log("iso: game yaw %.3f view yaw %.3f hand %d locked %d rotX %.3f rotY %.3f fwd.y %.3f", static_cast<double>(game),
                 static_cast<double>(cam.yaw), hand ? 1 : 0, locked ? 1 : 0, static_cast<double>(field<float>(f, kRotX)),
                 static_cast<double>(field<float>(f, kRotY)), static_cast<double>(c->forward[1]));
    // The game's right: which way round its basis is.
    const float rx = std::cos(game), rz = -std::sin(game);
    cam.right_sign = c->right[0] * rx + c->right[2] * rz < 0 ? -1.0f : 1.0f;

    // Level and at the view's yaw.
    const float sy = std::sin(cam.yaw), cy = std::cos(cam.yaw);
    const float fwd[3] = {sy, 0, cy}, right[3] = {cam.right_sign * cy, 0, -cam.right_sign * sy};
    for (int i = 0; i < 3; ++i) {
        c->forward[i] = fwd[i];
        c->right[i] = right[i];
        c->up[i] = i == 1 ? 1.0f : 0.0f;
    }
    // The follow camera kept level too (its pitch is the aim's when it is not
    // the matrix's), the stick's up and down given nothing to turn, and the
    // lock-on left to pick targets without turning the camera to them.
    set_field(f, kRotX, 0.0f);
    set_field(f, kTargetRotX, 0.0f);
    const float range[4] = {0, 0, 0, 0};
    std::memcpy(f + kRotRange, range, sizeof(range));
    set_field(f, kLockRotChaseRate, 0.0f);
    std::memset(f + kLockFlags, 0, 3);

    // The wheel's notches: each 15% nearer or further.
    BbInput in;
    if (api->input(&in) == 0 && in.playing && in.wheel != 0) {
        const float base = s.distance > 0.1f ? s.distance : 0.1f;
        float z = cam.zoom * std::pow(kZoomStep, -in.wheel);
        if (z * base < kZoomNear) z = kZoomNear / base;
        if (z * base > kZoomFar) z = kZoomFar / base;
        cam.zoom = z;
    }
    cam.view_ready = true;
}

// The drawn camera: from above, looking down at the hunter.
void on_view(BbCamera* c, void*) {
    if (!cam.view_ready) return;
    cam.view_ready = false;
    float feet[3], facing = 0;
    if (api->player_position(feet, &facing) != 0) return;
    const auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - cam.last).count();
    cam.last = now;
    if (!(dt > 0) || dt > 0.25f) dt = 0.25f;
    const float target[3] = {feet[0], feet[1] + s.height, feet[2]};
    float moved = 0;
    for (int i = 0; i < 3; ++i) moved += (target[i] - cam.focus[i]) * (target[i] - cam.focus[i]);
    // A warp or a load: no gliding across the map.
    if (!cam.have_focus || moved > 25.0f * 25.0f || s.follow <= 0) {
        std::memcpy(cam.focus, target, sizeof(target));
        cam.have_focus = true;
    } else {
        const float k = 1.0f - std::exp(-dt / s.follow);
        for (int i = 0; i < 3; ++i) cam.focus[i] += (target[i] - cam.focus[i]) * k;
    }
    const float p = s.pitch * kPi / 180.0f, sp = std::sin(p), cp = std::cos(p);
    const float sy = std::sin(cam.yaw), cy = std::cos(cam.yaw);
    const float fwd[3] = {sy * cp, -sp, cy * cp};
    const float up[3] = {sy * sp, cp, cy * sp};
    const float right[3] = {cam.right_sign * cy, 0, -cam.right_sign * sy};
    const float dist = s.distance * cam.zoom;
    for (int i = 0; i < 3; ++i) {
        c->forward[i] = fwd[i];
        c->up[i] = up[i];
        c->right[i] = right[i];
        c->position[i] = cam.focus[i] - fwd[i] * dist;
    }
    c->fov_y = s.fov * kPi / 180.0f;
}

void on_option(const char* key, const char* value, void*) {
    read_options();
    if (std::strcmp(key, "omni_roll") == 0) omni_roll(s.omni_roll);
    if (std::strcmp(key, "enabled") == 0 && !s.enabled) release();
    api->log("iso: %s = %s", key, value ? value : "?");
}

}  // namespace

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* host) {
    api = host;
    if (host->version < 20 || host->size < offsetof(BbHostApi, on_view) + sizeof(void*)) {
        host->log("iso: needs plugin API 20 (on_view); this host has %u", host->version);
        return 1;
    }
    if (!host->eboot_is_109()) {
        host->log("iso: not the 1.09 eboot; off");
        return 1;
    }
    read_options();
    if (host->on_camera(&on_camera, nullptr) != 0 || host->on_view(&on_view, nullptr) != 0) {
        host->log("iso: the host has no camera hooks; off");
        return 1;
    }
    host->on_option("iso", &on_option, nullptr);
    omni_roll(s.omni_roll);
    host->log("iso: Isometric %s (angle %.0f, distance %.1f m, fov %.0f), omni-roll %s", s.enabled ? "on" : "off",
              static_cast<double>(s.pitch), static_cast<double>(s.distance), static_cast<double>(s.fov),
              s.omni_roll ? "on" : "off");
    return 0;
}
