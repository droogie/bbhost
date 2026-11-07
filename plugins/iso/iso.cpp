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
//   flatten = 0         # a narrower view from further back (telephoto), degrees; 0 off
//   cutaway = true      # roofs and walls between the view and the hunter faded out
//   cutaway_size = 2.0  # the cone's tip, metres across, just above the hunter
//   cutaway_spread = 0.6  # metres wider for each metre toward the view
//
// Always on with the view: the omni-roll (4-way quicksteps, Modern Movement's
// without its quick item), the lock-on at 60 m all round with 30 s out of
// sight, and the sound heard from the hunter rather than the camera.
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
//   +0x1bc/+0x1c0 TargetChaseRateMinAng/MaxAng - 0: no dead zone before the
//     point looked at follows the hunter (it stayed up to 10 degrees off
//     CamRotY, and the stick's directions with it); +0x170 the pivot's sideways
//     shift - 0. The chase rates (+0x198.., the camera trailing the hunter) are
//     the game's own.
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

#include <atomic>
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
    "0.4.4",
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
    {BB_OPT_BOOL | BB_OPT_LIVE, "cutaway", "Cutaway",
     "Roofs, ceilings and walls between the view and the hunter faded out: a cone above the hunter, widening toward the "
     "view. Players, enemies and objects (and what they carry) are never cut. Each zoom step builds the map's shaders "
     "for it once (a short pause the first time).",
     "true"},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "cutaway_size", "Cutaway size", "Metres across the cone's tip, just above the hunter.", "2.0",
     0.5, 6},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "cutaway_spread", "Cutaway spread",
     "How fast the cone widens toward the view: metres wider for each metre nearer it.", "0.6", 0, 1.5},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "follow", "Follow lag",
     "Seconds the view takes to catch up with the hunter (0: fixed to them).", "0.08", 0, 0.5},
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
    bool cutaway = true;
    float cutaway_size = 2.0f, cutaway_spread = 0.6f;
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
// The lock-on from high above (Kyo): 60 m, all round, 30 s out of sight.
constexpr float kLockRangeM = 60.0f, kLockAngleDeg = 180.0f, kLockHiddenS = 30.0f;
struct Tagged {
    Default d;
    int from_setting;  // 0: fixed
};
// mov dword [rbx+off], imm32 = c7 83 off32 imm32 (value 6 in); movabs rax, imm64
// = 48 b8 imm64 (the first field 2 in, the second 6); mov byte [rbx+off], imm8
// = c6 83 off32 imm8 (6 in).
Tagged kDefaults[] = {
    // ChrFollowCam, both constructors: RotRange, lock chase. Its chase rates
    // stay the game's own (Kyo: the camera's chase as it was).
    {{0x183a585, 6, 4, 1.22173f, 0.0f}, 0},
    {{0x183a58f, 6, 4, -0.698132f, 0.0f}, 0}, {{0x183a599, 6, 4, 0.872665f, 0.0f}, 0},
    {{0x183a5a3, 6, 4, -0.698132f, 0.0f}, 0}, {{0x183a657, 6, 4, 0.1f, 0.0f}, 0},
    {{0x183a6a6, 6, 1, 1, 0}, 0},             {{0x183a6ad, 6, 1, 1, 0}, 0},
    {{0x183a983, 6, 4, 1.22173f, 0.0f}, 0},
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
// The auto-rotation's stores (disable-camera-auto-rotation.toml), 9 bytes each,
// and two more: as the stick (or mouse) starts turning it, the follow camera
// takes its CamRotX and CamRotY again from where it stands against the point
// it looks at (0x183cac7..: the angles of +0xd0 - +0x100, stored at 0x183cb56
// and 0x183cb97). Trailing a moving hunter, that turned it toward the trail -
// against the hunter's way - on every frame of a turn; without them a turn is
// the stick's alone.
const std::uint64_t kAutoRotation[] = {0x183c6e8, 0x183c984, 0x183dde6, 0x183c870, 0x183cb56, 0x183cb97};

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
    s.pitch = number("iso.pitch", 55.0f, 25, 89);
    s.distance = number("iso.distance", 14.0f, 3, 40);
    s.fov = number("iso.fov", 35.0f, 15, 75);
    s.flatten = number("iso.flatten", 0, 0, 15);
    s.height = number("iso.height", 1.0f, 0, 2.5f);
    s.follow = number("iso.follow", 0.08f, 0, 0.5f);
    const char* cu = api->config("iso.cutaway");
    s.cutaway = !cu || !*cu || truthy(cu);
    s.cutaway_size = number("iso.cutaway_size", 2.0f, 0.5f, 6);
    s.cutaway_spread = number("iso.cutaway_spread", 0.6f, 0, 1.5f);
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
        if (t.from_setting == kLockTimeout) d.want = kLockHiddenS;
        else if (t.from_setting == kLockRange) d.want = kLockRangeM;
        else if (t.from_setting == kLockAngle) d.want = kLockAngleDeg * kPi / 180.0f;
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
             written, refused, static_cast<double>(kLockRangeM), static_cast<double>(kLockAngleDeg),
             static_cast<double>(kLockHiddenS));
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

// The sound's listener at the hunter. The game's sound manager keeps its
// listeners (count at +0x280, 0x70 bytes each from +0xc0: a matrix with the
// place in its last row, then the velocity) and each frame, in its update
// (0x2c14280: this, dt), hands every listener's place to FMOD
// (System::set3DListenerAttributes) and to each of its sound emitters, whose
// own update decides from it what is in reach and how loud it plays. The game
// puts the drawn camera there - 14 m and more up in the air here, so the
// places' sounds were faint or never started. The hook moves the places to
// the hunter's head as the update starts; forward and up stay the camera's,
// so left and right are still the screen's. The place is taken on the game's
// thread (on_view) and read here on the sound's: a read that meets a write
// halfway hears from a frame's movement away, once.
volatile float g_ear[3] = {0, 0, 0};
std::atomic<bool> g_ear_on{false};  // while the view is ours
constexpr float kEarHeight = 1.5f;  // the hunter's head, metres over the feet
constexpr std::uint64_t kSoundUpdate = 0x2c14280;

int sound_update_hook(BbHookCtx* c, void*) {
    auto* const manager = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(c->arg[0]));
    if (!g_ear_on.load(std::memory_order_acquire) || !manager) return 0;
    const std::uint32_t count = *reinterpret_cast<const std::uint32_t*>(manager + 0x280);
    static const bool trace = std::getenv("ISO_LOG") != nullptr;
    static std::atomic<int> calls{0};
    const bool log = trace && calls.fetch_add(1, std::memory_order_relaxed) % 300 == 0;
    for (std::uint32_t i = 0; i < count && i < 4; ++i) {
        float* const place = reinterpret_cast<float*>(manager + 0xc0 + i * 0x70 + 0x30);
        if (log)
            api->log("iso: listener %u %.2f %.2f %.2f (the game's %.2f %.2f %.2f)", i, static_cast<double>(g_ear[0]),
                     static_cast<double>(g_ear[1]), static_cast<double>(g_ear[2]), static_cast<double>(place[0]),
                     static_cast<double>(place[1]), static_cast<double>(place[2]));
        place[0] = g_ear[0];
        place[1] = g_ear[1];
        place[2] = g_ear[2];
    }
    return 0;
}

// The cutaway (plugin API 23): a cone over the hunter, its tip `size` across
// a little above the point looked at, widening `spread` metres a metre toward
// the camera, faded at its edges - roofs, ceilings and near walls, never the
// characters, the objects or their parts (the host tells them by the
// G-buffer's stencil). Sent when its numbers change - each new set builds the
// map's pipelines again - and off while another camera has the view.
constexpr float kCutHeight = 0.6f;   // metres over the point looked at before anything is cut
constexpr float kCutMargin = 0.5f;   // metres in front of it
constexpr float kCutFeather = 1.5f;  // metres it fades in over
void cutaway(float dist) {
    static BbCutaway sent{};
    static bool sent_any = false;
    BbCutaway want{};
    if (s.cutaway && view.away <= 0.0f && dist > 0) {
        want.distance = dist;
        want.pitch = s.pitch * kPi / 180.0f;
        want.radius = s.cutaway_size / 2;
        want.widen = s.cutaway_spread;
        want.height = kCutHeight;
        want.margin = kCutMargin;
        want.feather = kCutFeather;
    }
    if (sent_any && std::fabs(want.distance - sent.distance) < 0.125f && want.radius == sent.radius && want.widen == sent.widen &&
        want.pitch == sent.pitch)
        return;
    sent = want;
    sent_any = true;
    if (api->version >= 23 && api->size >= offsetof(BbHostApi, view_cutaway2) + sizeof(void*))
        api->view_cutaway2(want.radius > 0 ? &want : nullptr);
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
    if (view.away > 0.0f) cutaway(0);  // the other camera's view: nothing cut
    if (view.away > 0.0f) g_ear_on.store(false, std::memory_order_release);  // another camera: its own place
    if (view.away >= 1.0f) {
        view.have_focus = false;  // the hunter may have moved while it was away: no glide back from there
        return;                   // the game's own camera, as copied
    }
    float feet[3], facing = 0;
    if (api->player_position(feet, &facing) != 0) return;
    const float target[3] = {feet[0], feet[1] + s.height, feet[2]};
    g_ear[0] = feet[0];
    g_ear[1] = feet[1] + kEarHeight;
    g_ear[2] = feet[2];
    g_ear_on.store(true, std::memory_order_release);
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
    cutaway(dist);
}

void on_option(const char* key, const char* value, void*) {
    read_options();
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
    omni_roll(true);  // always: part of the mode (Kyo)
    if (s.enabled) {
        static const std::uint8_t kSoundUpdatePrologue[17] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                                               0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x68};
        if (host->hook2(kSoundUpdate, kSoundUpdatePrologue, sizeof(kSoundUpdatePrologue), &sound_update_hook, nullptr) != 0)
            host->log("iso: the sound manager's update is not where it should be; the sound stays at the camera");
        if (host->on_camera(&on_camera, nullptr) != 0 || host->on_view(&on_view, nullptr) != 0) {
            host->log("iso: the host has no camera hooks; the view is off");
        } else {
            camera_defaults();
        }
    }
    host->log("iso: Isometric %s (angle %.0f, distance %.1f m, fov %.0f), omni-roll %s", s.enabled ? "on" : "off",
              static_cast<double>(s.pitch), static_cast<double>(s.distance), static_cast<double>(s.fov),
              "on");
    return 0;
}
