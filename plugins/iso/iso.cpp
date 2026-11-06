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
// +0x60) orbits the hunter; the manager's blend of it and the cameras it owns
// (the monocular's aim camera, the lock-on's) is what the controls move by and
// the aim reads. The scene's camera is a copy of that, made each frame and drawn
// from (plugin API 20's on_view): only the copy is turned to look down from
// above, so the aim stays level and the drawn camera, placed by us at the
// hunter, is never pushed in by the map.
//
// The game's camera is made to suit the view once, as the game builds it: the
// defaults its constructors write are patched (nothing is written each frame).
// Each value lives in two places - ChrFollowCam has two constructors
// (0x183a3c0.. and 0x183a7b0.., the second the one in play: its values are the
// debug page's), and LOCK_TGT_MAN's lock-on rules are set by its constructor
// (0x1c8dc71..) and again by its reset (0x1c8de03..); the debug pages point at
// the fields both fill. ChrFollowCam (offsets from its debug page, 0x183f000..):
//   +0x198/+0x19c ChrTransChaseRate(ForNormal), +0x1ac/+0x1b0 TargetChaseRateXZ
//     (ForNormal), +0x1b8 TargetChaseRateY - 1: the camera keeps up with the
//     hunter at once. Lagging, it trailed a running hunter and looked back at
//     them, its yaw drifting from CamRotY and the stick's directions with it.
//   +0x1bc/+0x1c0 TargetChaseRateMinAng/MaxAng - 0: no dead zone before the
//     point looked at follows the hunter; +0x170 the pivot's sideways shift - 0.
//   +0x1ec..+0x1f8 RotRange MaxX/MinX and at lock (radians; the page shows
//     degrees) - 0: it never tilts, so the aim is level.
//   +0x240 LockRotChaseRate - 0, +0x274/+0x275 UpdateTargetRotXByLock/FrontLock -
//     off: the lock-on picks targets without turning the camera.
//   The auto-rotation's four stores (disable-camera-auto-rotation's sites).
// LOCK_TGT_MAN (offsets in its object):
//   +0x2914 the invisible-lock timeout (2 s), +0x2924 ChrLockRangeMaxRadius
//   (22 m), +0x294c/+0x2954 CamFrontNear/FarRangeLockHalfAng (60/30 degrees),
//   +0x2958 CamFrontFarRangeLockRadius (22 m): seen from high above, the cones
//   and the line of sight broke the lock early.
//
// The view turns with CamRotY (+0x144), which only the stick or the mouse now
// moves. When the game's camera leaves the follow camera - the monocular's
// over-the-shoulder aim - the view glides to it, and back when it is put away.
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
    "0.2.1",
    "Kyo",
    "A Diablo-like camera: high above the hunter at a fixed angle, turned by the right stick or the mouse, zoomed by the "
    "wheel, never pulled in by walls; the aim stays level. With the omni-roll built in.",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, "view", "The view", "The right stick or the mouse turns it left and right; the wheel zooms.", nullptr},
    {BB_OPT_BOOL, "enabled", "Isometric camera",
     "The view from high above the hunter, and the game's camera made to suit it. Off: the game's own camera. Applies at "
     "the next start.",
     "true"},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "pitch", "Angle", "Degrees the view looks down below the horizon.", "55.0", 25, 89},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "distance", "Distance", "Metres from the hunter; the wheel zooms in and out from it.", "14.0",
     3, 40},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "fov", "Field of view", "The view's vertical field of view, in degrees.", "35.0", 15, 75},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "flatten", "Flatten (telephoto)",
     "Less perspective: the view narrowed to this field of view (degrees) and moved back to frame the hunter as the Field "
     "of view at the Distance does - near orthographic. 0: off. The game's fog grows with the distance, so small values "
     "look hazy.",
     "0", 0, 15},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "height", "Look-at height", "The point the view centres on, metres above the hunter's feet.",
     "1.0", 0, 2.5},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "follow", "Follow lag",
     "Seconds the view takes to catch up with the hunter (0: fixed to them).", "0.08", 0, 0.5},
    {BB_OPT_HEADING, "lock", "Lock-on", "Applies at the next start.", nullptr},
    {BB_OPT_FLOAT, "lock_range", "Lock-on range", "Metres a target can be locked at and held (the game's 22).", "30.0", 10,
     60},
    {BB_OPT_FLOAT, "lock_angle", "Lock-on angle",
     "Half the cone, in degrees, around where the hunter's camera looks that a target can be locked in (the game's 60 near, 30 "
     "far); 180 is all round.",
     "180.0", 30, 180},
    {BB_OPT_FLOAT, "lock_hidden", "Hidden target timeout",
     "Seconds a locked target may be out of sight before the lock lets go (the game's 2).", "8.0", 1, 30},
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
    float pitch = 55.0f, distance = 14.0f, fov = 35.0f, height = 1.0f, follow = 0.08f, flatten = 0;
    float lock_range = 30.0f, lock_angle = 180.0f, lock_hidden = 8.0f;
    bool omni_roll = true;
} s;

constexpr float kPi = 3.14159265358979f;
constexpr float kZoomStep = 1.15f, kZoomNear = 3.0f, kZoomFar = 40.0f;
// The game's camera off the follow camera by this much (metres, radians of
// field of view) is another camera's - the monocular's: the view hands over.
constexpr float kAwayPos = 0.3f, kAwayFov = 0.05f, kHandOver = 0.2f;  // seconds

// A constant a constructor writes: the instruction at `at`, its value `skip`
// bytes in, `size` bytes (4 a float, 1 a flag), from `was` to `want`.
struct Default {
    std::uint64_t at;
    std::uint8_t skip, size;
    float was;
    float want;  // overwritten from the settings for the lock-on's
};
enum : int { kLockTimeout = 1, kLockRange, kLockAngle };
struct Tagged {
    Default d;
    int from_setting;  // 0: fixed
};
// mov dword [rbx+off], imm32 = c7 83 off32 imm32 (value 6 in); movabs rax, imm64
// = 48 b8 imm64 (the first field 2 in, the second 6); mov byte [rbx+off], imm8
// = c6 83 off32 imm8 (6 in).
Tagged kDefaults[] = {
    // ChrFollowCam, both constructors: chase rates, RotRange, lock chase.
    {{0x183a4b3, 6, 4, 0.1f, 1.0f}, 0},       {{0x183a4bd, 6, 4, 0.1f, 1.0f}, 0},
    {{0x183a4e5, 6, 4, 0.1f, 1.0f}, 0},       {{0x183a4ef, 6, 4, 0.1f, 1.0f}, 0},
    {{0x183a503, 6, 4, 0.03f, 1.0f}, 0},      {{0x183a585, 6, 4, 1.22173f, 0.0f}, 0},
    {{0x183a58f, 6, 4, -0.698132f, 0.0f}, 0}, {{0x183a599, 6, 4, 0.872665f, 0.0f}, 0},
    {{0x183a5a3, 6, 4, -0.698132f, 0.0f}, 0}, {{0x183a657, 6, 4, 0.1f, 0.0f}, 0},
    {{0x183a6a6, 6, 1, 1, 0}, 0},             {{0x183a6ad, 6, 1, 1, 0}, 0},
    {{0x183a8b1, 6, 4, 0.1f, 1.0f}, 0},       {{0x183a8bb, 6, 4, 0.1f, 1.0f}, 0},
    {{0x183a8e3, 6, 4, 0.1f, 1.0f}, 0},       {{0x183a8ed, 6, 4, 0.1f, 1.0f}, 0},
    {{0x183a901, 6, 4, 0.03f, 1.0f}, 0},      {{0x183a983, 6, 4, 1.22173f, 0.0f}, 0},
    {{0x183a98d, 6, 4, -0.698132f, 0.0f}, 0}, {{0x183a997, 6, 4, 0.698132f, 0.0f}, 0},
    {{0x183a9a1, 6, 4, -0.698132f, 0.0f}, 0}, {{0x183aa55, 6, 4, 0.3f, 0.0f}, 0},
    // TargetChaseRateMinAng/MaxAng (+0x1bc/+0x1c0, 10 and 20 degrees): the
    // point looked at did not follow the hunter until they were 10 degrees off
    // it, so the camera (and the stick's directions) stayed up to 10 degrees off
    // CamRotY; Max is left a hair above Min. FulcrumDistRateByLrMoveMaxDist
    // (+0x170, the base constructor's only): the pivot moved with sideways steps.
    {{0x183a50d, 6, 4, 0.174533f, 0.0f}, 0},  {{0x183a517, 6, 4, 0.349066f, 0.01f}, 0},
    {{0x183a90b, 6, 4, 0.174533f, 0.0f}, 0},  {{0x183a915, 6, 4, 0.349066f, 0.01f}, 0},
    {{0x183a44f, 6, 4, 0.3f, 0.0f}, 0},
    // LOCK_TGT_MAN: its constructor, then its reset.
    {{0x1c8dc8f, 6, 4, 2.0f, 0}, kLockTimeout},
    {{0x1c8dcb7, 6, 4, 22.0f, 0}, kLockRange},
    {{0x1c8dd1b, 6, 4, 1.047198f, 0}, kLockAngle},
    {{0x1c8dd2f, 6, 4, 0.523599f, 0}, kLockAngle},
    {{0x1c8dd39, 6, 4, 22.0f, 0}, kLockRange},
    {{0x1c8de14, 6, 4, 2.0f, 0}, kLockTimeout},
    {{0x1c8de25, 2, 4, 22.0f, 0}, kLockRange},
    {{0x1c8de69, 6, 4, 1.047198f, 0}, kLockAngle},
    {{0x1c8de7a, 6, 4, 0.523599f, 0}, kLockAngle},
    {{0x1c8de8b, 6, 4, 22.0f, 0}, kLockRange},
};
// The auto-rotation's stores (disable-camera-auto-rotation.toml), 9 bytes each.
const std::uint64_t kAutoRotation[] = {0x183c6e8, 0x183c984, 0x183dde6, 0x183c870};

// The follow camera's fields read each frame.
constexpr std::size_t kRotY = 0x144;

struct View {
    float zoom = 1.0f;
    float focus[3]{};
    bool have_focus = false;
    float away = 0;        // 0 the iso view, 1 the game's camera (the monocular), eased between
    float away_want = 0;   // what on_camera saw this frame
    bool ready = false;    // on_camera ran this frame: on_view may draw
    float yaw = 0, right_sign = 1.0f;
    std::chrono::steady_clock::time_point last;
    bool started = false;
} view;

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
    s.flatten = number("iso.flatten", 0, 0, 15);
    s.height = number("iso.height", 1.0f, 0, 2.5f);
    s.follow = number("iso.follow", 0.08f, 0, 0.5f);
    s.lock_range = number("iso.lock_range", 30.0f, 10, 60);
    s.lock_angle = number("iso.lock_angle", 180.0f, 30, 180);
    s.lock_hidden = number("iso.lock_hidden", 8.0f, 1, 30);
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

// The game's camera and lock-on as the view needs them, patched into the
// constructors that build them (see the top): before the game runs, once.
void camera_defaults() {
    int written = 0, refused = 0;
    for (Tagged& t : kDefaults) {
        Default d = t.d;
        if (t.from_setting == kLockTimeout) d.want = s.lock_hidden;
        else if (t.from_setting == kLockRange) d.want = s.lock_range;
        else if (t.from_setting == kLockAngle) d.want = s.lock_angle * kPi / 180.0f;
        // The instruction checked by its opcode and the value it holds by
        // what the game ships (within a rounding), then the value replaced.
        std::uint8_t insn[10], want[4];
        const bool movabs = d.skip == 2 || (d.skip == 6 && d.size == 4 && t.d.at >= 0x1c8de00 && t.d.at < 0x1c8de8b);
        const std::uint8_t op[2] = {static_cast<std::uint8_t>(movabs ? 0x48 : d.size == 1 ? 0xc6 : 0xc7),
                                    static_cast<std::uint8_t>(movabs ? 0xb8 : 0x83)};
        float now = 0;
        bool ok = api->read(d.at, insn, sizeof(insn)) == 0 && insn[0] == op[0] && insn[1] == op[1];
        if (ok && d.size == 4) {
            std::memcpy(&now, insn + d.skip, 4);
            ok = std::fabs(now - d.was) <= 1e-4f * (std::fabs(d.was) > 1 ? std::fabs(d.was) : 1.0f) || now == d.want;
            std::memcpy(want, &d.want, 4);
        } else if (ok) {
            ok = insn[d.skip] == static_cast<std::uint8_t>(d.was) || insn[d.skip] == static_cast<std::uint8_t>(d.want);
            want[0] = static_cast<std::uint8_t>(d.want);
        }
        if (ok && api->patch(d.at + d.skip, insn + d.skip, want, d.size) == 0) {
            ++written;
        } else {
            ++refused;
            api->log("iso: 0x%llx is not the constant it should be; left alone", static_cast<unsigned long long>(d.at));
        }
    }
    const std::uint8_t store_len = 9;
    std::uint8_t nops[store_len];
    std::memset(nops, 0x90, sizeof(nops));
    for (const std::uint64_t at : kAutoRotation) {
        std::uint8_t now[store_len];
        // vmovss [r13+off], xmm: c4 c1 7a 11 ..; what is there is checked by its first four bytes.
        if (api->read(at, now, store_len) == 0 && (std::memcmp(now, nops, store_len) == 0 ||
                                                   (now[0] == 0xc4 && now[1] == 0xc1 && now[2] == 0x7a && now[3] == 0x11)) &&
            api->patch(at, now, nops, store_len) == 0)
            ++written;
        else
            ++refused;
    }
    api->log("iso: the game's camera and lock-on set for the view (%d constants and stores written, %d refused); lock-on "
             "%.0f m, %.0f degrees, %.0f s hidden",
             written, refused, static_cast<double>(s.lock_range), static_cast<double>(s.lock_angle),
             static_cast<double>(s.lock_hidden));
}

std::uint8_t* follow_cam(BbCamera* c) {
    auto* manager = reinterpret_cast<std::uint8_t*>(c) - 0x10;
    std::uint64_t p = 0;
    std::memcpy(&p, manager + 0x60, 8);
    return reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(p));
}

float field(const std::uint8_t* base, std::size_t at) {
    float v;
    std::memcpy(&v, base + at, sizeof(v));
    return v;
}

// The game's camera, read only: the view's yaw (CamRotY), and how far the
// manager's camera is from the follow camera's (another camera taking over).
void on_camera(BbCamera* c, void*) {
    view.ready = false;
    if (api->loading_screen && api->loading_screen()) {
        view.have_focus = false;
        return;
    }
    const std::uint8_t* f = follow_cam(c);
    if (!f) return;
    view.yaw = field(f, kRotY);
    if (!std::isfinite(view.yaw)) return;
    const auto* own = reinterpret_cast<const BbCamera*>(f + 0x10);
    float d2 = 0;
    for (int i = 0; i < 3; ++i) d2 += (c->position[i] - own->position[i]) * (c->position[i] - own->position[i]);
    const float dfov = std::fabs(c->fov_y - own->fov_y);
    float away = std::sqrt(d2) / kAwayPos;
    if (dfov / kAwayFov > away) away = dfov / kAwayFov;
    view.away_want = away > 1.0f ? 1.0f : away;
    // The game's right: which way round its basis is.
    const float rx = std::cos(view.yaw), rz = -std::sin(view.yaw);
    view.right_sign = c->right[0] * rx + c->right[2] * rz < 0 ? -1.0f : 1.0f;
    if (!view.started) {
        view.started = true;
        view.last = std::chrono::steady_clock::now();
        api->input_take(nullptr, 0, 0, 1, 0);  // the wheel zooms the view, not the game's camera
        api->log("iso: on (angle %.0f, distance %.1f m, fov %.0f)", static_cast<double>(s.pitch),
                 static_cast<double>(s.distance), static_cast<double>(s.fov));
    }
    // The wheel's notches: each 15% nearer or further.
    BbInput in;
    if (api->input(&in) == 0 && in.playing && in.wheel != 0) {
        const float base = s.distance > 0.1f ? s.distance : 0.1f;
        float z = view.zoom * std::pow(kZoomStep, -in.wheel);
        if (z * base < kZoomNear) z = kZoomNear / base;
        if (z * base > kZoomFar) z = kZoomFar / base;
        view.zoom = z;
    }
    // ISO_LOG=1: twice a second, the game's camera and the hunter.
    static const bool trace = std::getenv("ISO_LOG") != nullptr;
    static int frames = 0;
    if (trace && frames++ % 30 == 0) {
        float feet[3] = {}, facing = 0;
        api->player_position(feet, &facing);
        api->log("iso: rotY %.3f matrix yaw %.3f fwd.y %.3f away %.2f (%.2f m, fov %.3f/%.3f) facing %.3f at %.2f %.2f %.2f",
                 static_cast<double>(view.yaw), static_cast<double>(std::atan2(c->forward[0], c->forward[2])),
                 static_cast<double>(c->forward[1]), static_cast<double>(view.away_want), static_cast<double>(std::sqrt(d2)),
                 static_cast<double>(c->fov_y), static_cast<double>(own->fov_y), static_cast<double>(facing),
                 static_cast<double>(feet[0]), static_cast<double>(feet[1]), static_cast<double>(feet[2]));
    }
    view.ready = true;
}

void normalise(float v[3]) {
    const float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 1e-6f)
        for (int i = 0; i < 3; ++i) v[i] /= l;
}

// The drawn camera: from above, looking down at the hunter - or, while another
// camera has the game (the monocular), on the way to it or that camera itself.
void on_view(BbCamera* c, void*) {
    if (!view.ready) return;
    view.ready = false;
    const auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - view.last).count();
    view.last = now;
    if (!(dt > 0) || dt > 0.25f) dt = 0.25f;
    // Toward the other camera and back over kHandOver.
    const float step = dt / kHandOver;
    if (view.away < view.away_want) view.away = view.away + step < view.away_want ? view.away + step : view.away_want;
    else view.away = view.away - step > view.away_want ? view.away - step : view.away_want;
    if (view.away >= 1.0f) {
        view.have_focus = false;  // the hunter may have moved while it was away: no glide back from there
        return;                   // the game's own camera, as copied
    }
    float feet[3], facing = 0;
    if (api->player_position(feet, &facing) != 0) return;
    const float target[3] = {feet[0], feet[1] + s.height, feet[2]};
    float moved = 0;
    for (int i = 0; i < 3; ++i) moved += (target[i] - view.focus[i]) * (target[i] - view.focus[i]);
    // A warp or a load: no gliding across the map.
    if (!view.have_focus || moved > 25.0f * 25.0f || s.follow <= 0) {
        std::memcpy(view.focus, target, sizeof(target));
        view.have_focus = true;
    } else {
        const float k = 1.0f - std::exp(-dt / s.follow);
        for (int i = 0; i < 3; ++i) view.focus[i] += (target[i] - view.focus[i]) * k;
    }
    const float p = s.pitch * kPi / 180.0f, sp = std::sin(p), cp = std::cos(p);
    const float sy = std::sin(view.yaw), cy = std::cos(view.yaw);
    float fwd[3] = {sy * cp, -sp, cy * cp};
    float up[3] = {sy * sp, cp, cy * sp};
    float fov = s.fov * kPi / 180.0f;
    float dist = s.distance * view.zoom;
    // Flatten: a narrower view from further back, framed the same at the
    // hunter. The game's limits (its 38-48 degree clamp) are the follow
    // camera's; the drawn camera takes any field of view. A true orthographic
    // projection is not possible this way: the deferred passes rebuild
    // positions from depth as a perspective view does.
    if (s.flatten > 0 && s.flatten < s.fov) {
        const float narrow = s.flatten * kPi / 180.0f;
        dist *= std::tan(fov / 2) / std::tan(narrow / 2);
        fov = narrow;
    }
    float pos[3];
    for (int i = 0; i < 3; ++i) pos[i] = view.focus[i] - fwd[i] * dist;
    // Part way to the game's camera: eased, the basis made square again.
    if (view.away > 0) {
        const float t = view.away * view.away * (3 - 2 * view.away);
        for (int i = 0; i < 3; ++i) {
            pos[i] += (c->position[i] - pos[i]) * t;
            fwd[i] += (c->forward[i] - fwd[i]) * t;
            up[i] += (c->up[i] - up[i]) * t;
        }
        fov += (c->fov_y - fov) * t;
        normalise(fwd);
        const float d = up[0] * fwd[0] + up[1] * fwd[1] + up[2] * fwd[2];
        for (int i = 0; i < 3; ++i) up[i] -= fwd[i] * d;
        normalise(up);
    }
    // right = up x forward, turned the way the game's own basis is.
    float right[3] = {up[1] * fwd[2] - up[2] * fwd[1], up[2] * fwd[0] - up[0] * fwd[2], up[0] * fwd[1] - up[1] * fwd[0]};
    normalise(right);
    const float along = right[0] * std::cos(view.yaw) - right[2] * std::sin(view.yaw);
    const float flip = (along < 0 ? -1.0f : 1.0f) * view.right_sign;
    for (int i = 0; i < 3; ++i) {
        c->forward[i] = fwd[i];
        c->up[i] = up[i];
        c->right[i] = right[i] * flip;
        c->position[i] = pos[i];
    }
    c->fov_y = fov;
}

void on_option(const char* key, const char* value, void*) {
    read_options();
    if (std::strcmp(key, "omni_roll") == 0) omni_roll(s.omni_roll);
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
    host->on_option("iso", &on_option, nullptr);
    omni_roll(s.omni_roll);
    if (s.enabled) {
        if (host->on_camera(&on_camera, nullptr) != 0 || host->on_view(&on_view, nullptr) != 0) {
            host->log("iso: the host has no camera hooks; the view is off");
        } else {
            camera_defaults();
        }
    }
    host->log("iso: Isometric %s (angle %.0f, distance %.1f m, fov %.0f), omni-roll %s", s.enabled ? "on" : "off",
              static_cast<double>(s.pitch), static_cast<double>(s.distance), static_cast<double>(s.fov),
              s.omni_roll ? "on" : "off");
    return 0;
}
