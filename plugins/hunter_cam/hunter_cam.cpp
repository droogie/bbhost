// Hunter Cam: the hunter seen from closer in. Over the shoulder - a close
// third-person camera behind one shoulder, as Silent Hill 2's remake frames its
// hero, swapped on the fly with a key and zoomed with the wheel - or first
// person, where only the hunter's weapons are drawn; a mode that keeps the
// hunter facing where the camera looks at full speed; and the dodge as a roll
// or a sidestep.
//
//   [plugins]
//   hunter_cam = true
//   [hunter_cam]
//   camera = true            # the over-the-shoulder camera
//   shoulder = "right"       # right | left: the shoulder it starts over
//   swap_key = "Tab"         # swaps shoulders while playing (None: no key)
//   distance = 1.7           # metres behind the shoulder (the wheel zooms from it)
//   offset = 0.55            # metres to the side
//   height = 0.15            # metres above the game's look-at point
//   fov = 0                  # vertical field of view, degrees (0: the game's)
//   swap_time = 0.25         # seconds a swap takes
//   first_person = false     # the view from the hunter's eyes
//   first_person_key = "Z"   # toggles first person while playing
//   eye_height = 1.55        # metres above the feet, in first person
//   first_person_fov = 0     # first person's field of view, degrees (0: the game's)
//   face_camera = false      # always face where the camera looks
//   dodge = 1                # 0 the roll, 1 a sidestep in the stick's direction
//
// The camera. The game keeps its own follow camera (ChrFollowCam, the camera
// manager's +0x60), which orbits the hunter, pulls in at walls and turns with
// the stick; the manager's blend of it and the cameras it owns (the
// monocular's aim, the lock-on's) is the player's camera, which on_camera
// hands us each frame before anything is drawn from it. Only its place (and
// its field of view) is changed - never where it looks - so the stick, the
// lock-on and the aim behave as the game has them. Over the shoulder, the
// place is the follow camera's focus raised by the Height, moved sideways by
// the Side offset and back along the view by the Distance (times the wheel's
// zoom); the game's sphere cast with the follow camera's own filter (0x25,
// sub_1c090e0) tests the way out to the shoulder and back, so a wall pulls the
// camera in (at once; it eases back out). In first person the place is the
// hunter's eyes: their feet raised by the Eye height, a hand's width forward.
//
// The zoom. The wheel and the middle button (the host's Camera Zoom keys,
// plugin API 26) zoom the shoulder camera 15% a notch and reset it. Once a
// plugin takes those keys the host's own zoom stops, so the plugin takes them
// only when its camera first runs, and from then on, wherever its camera is
// not in charge (the plugin off, the camera option off), zooms the game's
// camera itself as the host did: 20% a notch, half to six times out, along the
// view from the point it looks at.
//
// First person. The hunter's equipment is drawn as parts: PlayerIns +0x4d0
// keeps 24 model inses from +0x28, each with a display entity whose +0x210
// bit 2 says it is drawn (the Quake plugin's viewmodel); a model ins's file name
// is at +0x148 -> +0x18 (WP_ the weapons, one slot a hand: 7 and 11; HD_, BD_,
// AM_, LG_ the armour, FC_, HR_ and FG_ the face). In first person every
// part but the weapons - the slots whose model is a WP_ file - is hidden that
// way, frame by frame (a change of equipment makes new ones), and shown again
// as first person ends, the plugin goes off or the world loads. The hunter
// faces where the camera looks there, as with Always face the camera.
//
// Facing the camera. The monocular puts the hunter in its precision-shooting
// state: ChrIns +0x1e2 bit 2, which the HKS env query 0x18 (精密射撃中か) reads
// first (0x1e13c68); with it the game turns the hunter to where the camera
// looks, steps them by the stick as a strafe and hands the player's camera to
// the monocular's aim camera (first person, a crosshair). With Always face the
// camera on (or in first person) the plugin sets that bit each frame and keeps
// the view itself, from the aim camera's turning. The game's script, though,
// holds the precision-shooting hunter to a walk, so the plugin answers the
// script's env queries for the player itself (a hook on the dispatcher's
// entry, 0x1e13760, which returns the answer in eax): precision shooting
// (0x18) no, locked on (0x12) yes - the lock-on's strafing locomotion, which
// runs and sprints. While SpEffect 2000 is on (stateInfo 15: the monocular
// itself in use) the game's answers and the monocular's view stand.
//
// The dodge. The script picks the dodge as Act_Rolling decides it:
// env(0, Arm_Rolling) yes, the stamina (0x1d), the height fallen, then
// "locked on or precision shooting" a step in the stick's direction, else the
// roll. At Roll the plugin answers both no inside that decision - found by its
// order: the action request for Arm_Rolling (17) followed by the stamina query,
// which the script asks only when the request was yes - so the hunter rolls,
// locked on or not. At Sidestep the omni-roll's sites are written (the
// Isometric plugin's, from Kyo's Modern Movement Lite: four-way quicksteps
// without a lock) unless something already wrote them, which keeps them: the
// patch list's Modern Movement Lite works beside the plugin at either setting,
// and only what the plugin wrote is taken back.
#include "bbhost_plugin.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define KEY_CHOICES "Tab|Z|B|H|M|N|Mouse 4|Mouse 5|None"

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN | BB_PLUGIN_LIVE_SWITCH,
    "hunter_cam",
    "Hunter Cam",
    "1.1.0",
    "Kyo",
    "The hunter from closer in: an over-the-shoulder camera (right or left, swapped with a key, zoomed with the wheel) "
    "or first person with only the weapons drawn; the hunter can always face where the camera looks, and the dodge can "
    "be a roll or a sidestep.",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, "h_view", "Over the shoulder",
     "The camera keeps the game's turning, lock-on and aim; only where it stands moves. The wheel zooms it, the middle "
     "button resets the zoom.",
     nullptr, 0, 0, nullptr},
    {BB_OPT_BOOL | BB_OPT_LIVE, "camera", "Over-the-shoulder camera",
     "The view from close behind one of the hunter's shoulders. Off: the game's own camera.", "true", 0, 0, nullptr},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "shoulder", "Shoulder", "The shoulder the camera looks over.", "right", 0, 0, "right|left"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "swap_key", "Swap shoulder key",
     "Swaps shoulders while playing (keyboard and mouse). Pick a key nothing else is bound to.", "Tab", 0, 0, KEY_CHOICES},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "distance", "Distance", "Metres behind the shoulder; the wheel zooms from here.", "1.7", 0.8,
     4.0, nullptr},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "offset", "Side offset", "Metres to the side of the hunter's middle.", "0.55", 0.0, 1.2,
     nullptr},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "height", "Height", "Metres above the point the game's camera looks at.", "0.15", -0.5, 1.0,
     nullptr},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "fov", "Field of view", "Vertical field of view in degrees; 0 keeps the game's.", "0", 0, 90,
     nullptr},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "swap_time", "Swap time", "Seconds the camera takes to move to the other shoulder.", "0.25",
     0, 1, nullptr},
    {BB_OPT_HEADING, "h_first", "First person", "The view from the hunter's eyes: only their weapons are drawn.", nullptr, 0,
     0, nullptr},
    {BB_OPT_BOOL | BB_OPT_LIVE, "first_person", "First person",
     "Starts in first person. The hunter always faces where they look there.", "false", 0, 0, nullptr},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "first_person_key", "First person key",
     "Switches between first person and the camera above while playing (keyboard and mouse).", "Z", 0, 0, KEY_CHOICES},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "eye_height", "Eye height", "Metres above the hunter's feet.", "1.55", 1.2, 1.9, nullptr},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "first_person_fov", "First-person field of view",
     "Vertical field of view in degrees; 0 keeps the game's.", "0", 0, 100, nullptr},
    {BB_OPT_HEADING, "h_movement", "Movement", nullptr, nullptr, 0, 0, nullptr},
    {BB_OPT_BOOL | BB_OPT_LIVE, "face_camera", "Always face the camera",
     "The hunter always faces where the camera looks and strafes by the stick, at full speed (they run and sprint as "
     "when locked on), with the monocular's crosshair up.",
     "false", 0, 0, nullptr},
    {BB_OPT_INT | BB_OPT_LIVE, "dodge", "Dodge (0 Roll, 1 Sidestep)",
     "0: always the roll, locked on too. 1: a quick sidestep in the stick's direction, as when locked on, without a "
     "lock. The Modern Movement Lite patch keeps working beside it.",
     "1", 0, 1, nullptr},
    {0, nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr}};

namespace {

const BbHostApi* api = nullptr;

constexpr float kPi = 3.14159265358979f;

template <typename T>
T get(const std::uint8_t* p, std::size_t off) {
    T v;
    std::memcpy(&v, p + off, sizeof(v));
    return v;
}

bool has_field(std::size_t offset) { return api->size >= offset + sizeof(void*); }

// ---- settings ----------------------------------------------------------------

enum class Key { Tab, Z, B, H, M, N, Mouse4, Mouse5, None };

struct Settings {
    bool on = false;  // the plugin's own switch (BB_PLUGIN_LIVE_SWITCH)
    bool camera = true;
    float shoulder = 1.0f;  // +1 right, -1 left
    Key swap_key = Key::Tab, first_key = Key::Z;
    float distance = 1.7f, offset = 0.55f, height = 0.15f, fov = 0.0f, swap_time = 0.25f;
    bool first_person = false;
    float eye_height = 1.55f, first_fov = 0.0f;
    bool face_camera = false;
    int dodge = 1;
} s;

bool truthy(const char* v) { return v && (std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0); }

bool flag(const char* key, bool fallback) {
    const char* v = api->config(key);
    return !v || !*v ? fallback : truthy(v);
}

float number(const char* key, float fallback, float lo, float hi) {
    const char* v = api->config(key);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const float f = std::strtof(v, &end);
    if (end == v || !std::isfinite(f)) return fallback;
    return f < lo ? lo : f > hi ? hi : f;
}

Key key_named(const char* v, Key fallback) {
    if (!v || !*v) return fallback;
    static const struct {
        const char* name;
        Key key;
    } kKeys[] = {{"Tab", Key::Tab}, {"Z", Key::Z}, {"B", Key::B}, {"H", Key::H}, {"M", Key::M},
                 {"N", Key::N}, {"Mouse 4", Key::Mouse4}, {"Mouse 5", Key::Mouse5}, {"None", Key::None}};
    for (const auto& k : kKeys)
        if (std::strcmp(v, k.name) == 0) return k.key;
    return fallback;
}

// The key's SDL scancode (USB HID usage), 0 for a mouse button or none.
std::uint16_t scancode(Key k) {
    switch (k) {
    case Key::Tab: return 43;
    case Key::Z: return 29;
    case Key::B: return 5;
    case Key::H: return 11;
    case Key::M: return 16;
    case Key::N: return 17;
    default: return 0;
    }
}

// BbInput's button bit for a mouse button, 0 for a key.
std::uint32_t mouse_bit(Key k) { return k == Key::Mouse4 ? 8u : k == Key::Mouse5 ? 16u : 0u; }

void read_options() {
    s.on = truthy(api->config("plugins.hunter_cam"));
    s.camera = flag("hunter_cam.camera", true);
    const char* side = api->config("hunter_cam.shoulder");
    s.shoulder = side && std::strcmp(side, "left") == 0 ? -1.0f : 1.0f;
    s.swap_key = key_named(api->config("hunter_cam.swap_key"), Key::Tab);
    s.first_key = key_named(api->config("hunter_cam.first_person_key"), Key::Z);
    s.distance = number("hunter_cam.distance", 1.7f, 0.8f, 4.0f);
    s.offset = number("hunter_cam.offset", 0.55f, 0.0f, 1.2f);
    s.height = number("hunter_cam.height", 0.15f, -0.5f, 1.0f);
    s.fov = number("hunter_cam.fov", 0.0f, 0.0f, 90.0f);
    s.swap_time = number("hunter_cam.swap_time", 0.25f, 0.0f, 1.0f);
    s.first_person = flag("hunter_cam.first_person", false);
    s.eye_height = number("hunter_cam.eye_height", 1.55f, 1.2f, 1.9f);
    s.first_fov = number("hunter_cam.first_person_fov", 0.0f, 0.0f, 100.0f);
    s.face_camera = flag("hunter_cam.face_camera", false);
    s.dodge = static_cast<int>(number("hunter_cam.dodge", 1.0f, 0.0f, 1.0f) + 0.5f);
}

// ---- the game's things -------------------------------------------------------

constexpr std::uint64_t kWorldChrMan = 0x593e878;    // WorldChrMan*: +0x60 the main player's ChrIns
constexpr std::uint64_t kPhysicsSystem = 0x593d700;  // +0x28 the collision world the camera casts in
constexpr std::uint64_t kSphereCast = 0x1c090e0;     // (world, filter, from, delta, hit, normal, radius, fraction*) -> hit
constexpr std::uint32_t kCameraFilter = 0x25;        // the follow camera's own
constexpr std::int32_t kMonocular = 2000;            // SpEffect: the monocular in use (stateInfo 15)

// A global of the image's, by its Binary Ninja address (read() moves it by the
// image's slide).
std::uint8_t* ptr_at(std::uint64_t bn) {
    std::uint64_t v = 0;
    return api->read(bn, &v, 8) == 0 ? reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(v)) : nullptr;
}

std::uint8_t* player() {
    std::uint8_t* man = ptr_at(kWorldChrMan);
    return man ? get<std::uint8_t*>(man, 0x60) : nullptr;
}

std::uint8_t* collision_world() {
    std::uint8_t* sys = ptr_at(kPhysicsSystem);
    return sys ? get<std::uint8_t*>(sys, 0x28) : nullptr;
}

// ---- vectors -----------------------------------------------------------------

struct V3 {
    float x, y, z;
};
V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(V3 a, float k) { return {a.x * k, a.y * k, a.z * k}; }
float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float length(V3 a) { return std::sqrt(dot(a, a)); }
V3 unit(V3 a) {
    const float l = length(a);
    return l > 1e-6f ? a * (1.0f / l) : V3{0, 0, 0};
}
V3 v3(const float* f) { return {f[0], f[1], f[2]}; }
void put(float* f, V3 v) {
    f[0] = v.x;
    f[1] = v.y;
    f[2] = v.z;
}

// The game's sphere cast: a sphere of `radius` swept from `from` along
// `delta`; on a hit, the share of delta it got before it touched.
bool sphere_cast(std::uint8_t* world, V3 from, V3 delta, float radius, float& fraction) {
    alignas(16) float f[4] = {from.x, from.y, from.z, 1.0f};
    alignas(16) float d[4] = {delta.x, delta.y, delta.z, 0.0f};
    alignas(16) float hit[4] = {};
    alignas(16) float n[4] = {};
    alignas(16) float frac[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    BbCallRegs r{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(world);
    r.arg[1] = kCameraFilter;
    r.arg[2] = reinterpret_cast<std::uint64_t>(f);
    r.arg[3] = reinterpret_cast<std::uint64_t>(d);
    r.arg[4] = reinterpret_cast<std::uint64_t>(hit);
    r.arg[5] = reinterpret_cast<std::uint64_t>(n);
    r.xmm[0] = bb_f32(radius);
    r.stack[0] = reinterpret_cast<std::uint64_t>(frac);
    if (api->call_guest(kSphereCast, &r) != 0 || (r.ret & 0xff) == 0) return false;
    fraction = frac[0] < 0 ? 0 : frac[0] > 1 ? 1 : frac[0];
    return true;
}

// ---- the omni-roll (Sidestep) ------------------------------------------------

struct Site {
    std::uint64_t address;
    std::uint32_t size;
    const char* expect;
    const char* bytes;
};
const Site kOmniRoll[] = {
#include "../iso/omni_roll.inc"
};
constexpr std::size_t kSites = sizeof(kOmniRoll) / sizeof(kOmniRoll[0]);
// Below this the sites are hooks in the game's own code; the rest are caves
// nothing reaches without them.
constexpr std::uint64_t kCaves = 0x4000000;

bool omni_applied = false;
bool omni_owned[kSites]{};  // written by us, not found already in place

void omni_roll(bool on) {
    if (on == omni_applied) return;
    int written = 0, already = 0, refused = 0;
    for (std::size_t i = 0; i < kSites; ++i) {
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
        api->log("hunter_cam: sidestep on (%d sites written, %d already in place, %d refused)", written, already, refused);
    else
        api->log("hunter_cam: sidestep's own sites given back (%d)", written);
}

// ---- facing the camera: the precision-shooting bit ---------------------------

constexpr std::size_t kAimFlags = 0x1e2;  // ChrIns: bit 2 precision shooting
constexpr std::uint8_t kAimBit = 4;
bool g_aim_ours = false;                       // the bit is set because we set it
std::uint8_t* g_aim_chr = nullptr;             // on this character
std::atomic<std::uint8_t*> g_player{nullptr};  // the player's ChrIns, for the env hook
std::atomic<bool> g_answer_aim{false};         // the env hook answers lock-on yes, precision no
std::atomic<bool> g_force_roll{false};         // the env hook makes the dodge a roll
std::atomic<bool> g_monocular{false};          // SpEffect 2000 on: the monocular in use
bool g_first = false;                          // in first person now (the key or the option)

bool aim_wanted() { return s.on && (s.face_camera || g_first); }

// Once a frame, on the main thread: the bit set while it should be, cleared
// once it should not (only when we set it).
void face_camera_frame() {
    const bool loading = api->loading_screen && api->loading_screen();
    const bool want = aim_wanted() && !loading;
    std::uint8_t* chr = player();
    if (chr != g_aim_chr) g_aim_ours = false;  // a load: the character it was on is gone
    g_aim_chr = chr;
    g_player.store(loading ? nullptr : chr);
    g_monocular.store(chr && api->sp_effect_has(nullptr, kMonocular) != 0);
    if (chr) {
        if (want) {
            chr[kAimFlags] |= kAimBit;
            g_aim_ours = true;
        } else if (g_aim_ours) {
            chr[kAimFlags] &= static_cast<std::uint8_t>(~kAimBit);
            g_aim_ours = false;
        }
    }
    g_answer_aim.store(g_aim_ours && !g_monocular.load());
    g_force_roll.store(s.on && s.dodge == 0);
}

// The aim is ours and the monocular is not in use: our camera keeps the view.
bool aim_is_ours() { return g_aim_ours && !g_monocular.load(); }

// ---- the script's env queries for the player ---------------------------------

constexpr std::uint64_t kEnv = 0x1e13760;  // (this: ChrIns**, id, Lua state) -> eax
constexpr std::uint8_t kEnvPrologue[17] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                           0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x48};
constexpr std::uint64_t kLuaHasArg = 0xd38b90, kLuaInt = 0xd38b40;  // (state, index)
constexpr std::uint32_t kEnvAction = 0x00, kEnvLock = 0x12, kEnvAim = 0x18, kEnvStamina = 0x1d;
constexpr std::int32_t kArmRolling = 17;

// The dodge's decision, found by its order: 1 after env(0, Arm_Rolling), 2
// once the stamina query that only a yes leads to followed it.
int g_dodge = 0, g_dodge_calls = 0;
bool g_env_hooked = false;

int env_hook(BbHookCtx* c, void*) {
    const bool roll = g_force_roll.load(std::memory_order_relaxed);
    const bool aim = g_answer_aim.load(std::memory_order_relaxed);
    if (!roll && !aim) return 0;
    std::uint8_t* const* self = reinterpret_cast<std::uint8_t* const*>(static_cast<std::uintptr_t>(c->arg[0]));
    std::uint8_t* const me = g_player.load(std::memory_order_relaxed);
    if (!self || !me || *self != me) return 0;
    const auto id = static_cast<std::uint32_t>(c->arg[1]);
    if (id == kEnvAction) {
        g_dodge = 0;
        if (roll) {
            BbCallRegs r{};
            r.arg[0] = c->arg[2];
            r.arg[1] = 2;
            if (api->call_guest(kLuaHasArg, &r) == 0 && (r.ret & 0xff)) {
                BbCallRegs v{};
                v.arg[0] = c->arg[2];
                v.arg[1] = 2;
                if (api->call_guest(kLuaInt, &v) == 0 && static_cast<std::int32_t>(v.ret) == kArmRolling) g_dodge = 1;
            }
        }
        return 0;
    }
    if (g_dodge == 1) {
        g_dodge = id == kEnvStamina ? 2 : 0;
        g_dodge_calls = 0;
    } else if (g_dodge == 2 && ++g_dodge_calls > 8) {
        g_dodge = 0;
    }
    if (id != kEnvLock && id != kEnvAim) return 0;
    if (g_dodge == 2) {  // the roll, locked on or not
        if (id == kEnvAim) g_dodge = 0;
        c->ret = 0;
        return 1;
    }
    if (aim) {  // the lock-on's locomotion - it runs - not precision shooting's walk
        c->ret = id == kEnvLock ? 1 : 0;
        return 1;
    }
    return 0;
}

// ---- first person: the body's parts hidden, the weapons kept -----------------

constexpr std::size_t kPartsHolder = 0x4d0;  // PlayerIns: 24 model inses from +0x28
constexpr int kPartSlots = 24;
constexpr std::uint32_t kPartDrawn = 0x4;  // the display entity's +0x210

struct Parts {
    std::uint8_t* hidden[kPartSlots] = {};  // display entities hidden here, shown again on release
    std::uint8_t* seen[kPartSlots] = {};    // the model ins last looked at in each slot
    std::uint8_t* chr = nullptr;
    std::uint32_t weapons = 0;  // a bit a slot: the ones holding a WP_ model
} parts;

// A UTF-16 name at a runtime address: whether it starts with `prefix`.
bool name_starts(std::uint64_t at, const char* prefix) {
    std::uint16_t w[8];
    if (at < 0x10000 || api->read(at, w, sizeof(w)) != 0) return false;
    for (int i = 0; prefix[i]; ++i)
        if (w[i] != static_cast<std::uint8_t>(prefix[i])) return false;
    return true;
}

// Whether a part's model is a weapon's: its file name (model ins +0x148 ->
// +0x18 -> UTF-16: WP_A_1000, HD_F_8000, FG_A_7003 - FG the face's parts)
// starts with WP_.
bool part_is_weapon(std::uint8_t* mi) {
    std::uint64_t res = 0, name = 0;
    return api->read(reinterpret_cast<std::uint64_t>(mi) + 0x148, &res, 8) == 0 && res >= 0x10000 &&
           api->read(res + 0x18, &name, 8) == 0 && name_starts(name, "WP_");
}

std::uint8_t* part_mi(std::uint8_t* holder, int slot) {
    return get<std::uint8_t*>(holder, 0x28 + 8 * static_cast<std::size_t>(slot));
}
std::uint8_t* part_disp(std::uint8_t* holder, int slot) {
    std::uint8_t* mi = part_mi(holder, slot);
    return mi ? get<std::uint8_t*>(mi, 0x18) : nullptr;
}

void set_drawn(std::uint8_t* disp, bool drawn) {
    std::uint32_t f = get<std::uint32_t>(disp, 0x210);
    f = drawn ? f | kPartDrawn : f & ~kPartDrawn;
    std::memcpy(disp + 0x210, &f, 4);
}

void parts_show() {
    std::uint8_t* holder = parts.chr ? get<std::uint8_t*>(parts.chr, kPartsHolder) : nullptr;
    for (int k = 0; k < kPartSlots; ++k) {
        if (holder && parts.hidden[k] && part_disp(holder, k) == parts.hidden[k]) set_drawn(parts.hidden[k], true);
        parts.hidden[k] = nullptr;
    }
}

// Every part but the weapons hidden (each frame: new equipment, new parts).
void parts_hide(std::uint8_t* chr) {
    if (chr != parts.chr) {  // another character: the old one's parts are gone
        for (int k = 0; k < kPartSlots; ++k) parts.hidden[k] = parts.seen[k] = nullptr;
        parts.chr = chr;
        parts.weapons = 0;
    }
    std::uint8_t* holder = chr ? get<std::uint8_t*>(chr, kPartsHolder) : nullptr;
    if (!holder) return;
    bool changed = false;
    for (int k = 0; k < kPartSlots; ++k) {
        std::uint8_t* mi = part_mi(holder, k);
        if (mi != parts.seen[k]) {  // a new model in the slot: what it is, again
            parts.seen[k] = mi;
            changed = true;
            if (mi && part_is_weapon(mi)) parts.weapons |= 1u << k;
            else parts.weapons &= ~(1u << k);
        }
        std::uint8_t* disp = mi ? get<std::uint8_t*>(mi, 0x18) : nullptr;
        if (!disp) continue;
        const bool drawn = (get<std::uint32_t>(disp, 0x210) & kPartDrawn) != 0;
        if (parts.weapons >> k & 1) {
            if (parts.hidden[k] == disp && !drawn) set_drawn(disp, true);
            parts.hidden[k] = nullptr;
        } else if (drawn) {
            set_drawn(disp, false);
            parts.hidden[k] = disp;
        }
    }
    if (changed) api->log("hunter_cam: first person keeps the weapon parts 0x%06x drawn", parts.weapons);
}

// ---- the camera --------------------------------------------------------------

struct View {
    float side = 1.0f;        // the eased shoulder: +1 right, -1 left
    float side_want = 1.0f;   // where the key or the option put it
    float side_from = 1.0f;   // the swap's start
    float swap_t = 1.0f;      // 0..1 through the swap
    float reach = -1.0f;      // the eased distance behind the shoulder (-1: none yet)
    float away = 0.0f;        // 0 ours, 1 another camera's (the monocular's)
    float eye = 1.45f;        // the follow camera's look-at height over the feet
    float fov = 0.0f;         // its field of view (0: none seen yet)
    V3 focus{0, 0, 0};        // from the hunter: the look-at point (or the eyes) chased after them
    bool have_focus = false;
    bool keys_down[2] = {};   // the swap key, the first-person key
    float zoom = 1.0f;        // the wheel's, on the shoulder camera's distance
    float game_zoom = 1.0f;   // the wheel's on the game's camera, once we took the keys
    float game_reach = -1.0f; // its eased distance
    bool zoom_taken = false;
    std::chrono::steady_clock::time_point last;
    bool started = false;
} view;

// The game's camera off its follow camera by this much is another camera's.
constexpr float kAwayPos = 0.3f, kAwayFov = 0.05f, kHandOver = 0.2f;  // m, rad, s
constexpr float kRadius = 0.2f;       // the camera's sphere against walls
constexpr float kEaseOut = 0.25f;     // seconds to ease back out after a wall
constexpr float kMinReach = 0.15f;    // metres it keeps behind the shoulder
constexpr float kChase = 0.06f;       // seconds the point from the hunter takes to catch up
constexpr float kEyeForward = 0.12f;  // the eyes, metres ahead of the feet's middle
constexpr float kZoomStep = 1.15f, kZoomNear = 0.6f, kZoomFar = 6.0f;             // the shoulder camera's
constexpr float kGameZoomStep = 1.2f, kGameZoomNear = 0.5f, kGameZoomFar = 6.0f;  // the host's own

std::uint8_t* follow_cam(BbCamera* c) {
    auto* manager = reinterpret_cast<std::uint8_t*>(c) - 0x10;
    return get<std::uint8_t*>(manager, 0x60);
}

void swap_to(float side) {
    if (side == view.side_want) return;
    view.side_from = view.side;
    view.side_want = side;
    view.swap_t = 0.0f;
}

void banner(const char* text) {
    if (has_field(offsetof(BbHostApi, show_text))) api->show_text(text, 1.0f, 0.5f, 0.85f, 1.0f, BB_TEXT_CENTRE);
}

// The keys' presses, once per press while playing (one read of the input a frame).
void read_keys() {
    BbInput in;
    const bool ok = api->input(&in) == 0 && in.playing;
    const Key keys[2] = {s.swap_key, s.first_key};
    for (int i = 0; i < 2; ++i) {
        bool down = false, pressed = false;
        if (ok && keys[i] != Key::None) {
            if (const std::uint16_t sc = scancode(keys[i])) {
                down = in.key[sc] != 0;
                pressed = down && !view.keys_down[i];
            } else if (const std::uint32_t bit = mouse_bit(keys[i])) {
                down = (in.buttons & bit) != 0;
                pressed = (in.pressed & bit) != 0;
            }
        }
        view.keys_down[i] = down;
        if (!pressed) continue;
        if (i == 0 && !g_first) {
            swap_to(-view.side_want);
            banner(view.side_want > 0 ? "Right shoulder" : "Left shoulder");
        } else if (i == 1) {
            g_first = !g_first;
            view.reach = -1.0f;
            view.have_focus = false;
            banner(g_first ? "First person" : (s.camera ? "Over the shoulder" : "Game camera"));
        }
    }
}

bool zoom_keys() { return api->version >= 26 && has_field(offsetof(BbHostApi, camera_zoom)); }

// The wheel's notches and the middle button since the last frame, once we
// hold the keys: on the shoulder camera's zoom while it is in charge, else on
// the game camera's as the host had it.
void read_zoom(bool ours) {
    if (!zoom_keys()) return;
    if (!view.zoom_taken) {
        if (!ours) return;  // the host keeps them until our camera first runs
        view.zoom_taken = true;
        api->log("hunter_cam: the wheel zooms the shoulder camera (and the game's camera where ours is off)");
    }
    int steps = 0, reset = 0;
    api->camera_zoom(&steps, &reset);
    float& z = ours ? view.zoom : view.game_zoom;
    if (reset) z = 1.0f;
    if (!steps) return;
    if (ours) {
        const float base = s.distance > 0.1f ? s.distance : 0.1f;
        z *= std::pow(kZoomStep, static_cast<float>(steps));
        if (z * base < kZoomNear) z = kZoomNear / base;
        if (z * base > kZoomFar) z = kZoomFar / base;
    } else {
        z *= std::pow(kGameZoomStep, static_cast<float>(steps));
        if (z < kGameZoomNear) z = kGameZoomNear;
        if (z > kGameZoomFar) z = kGameZoomFar;
    }
}

// HUNTER_CAM_LOG=1: every 8th frame, the camera, the hunter and the aim.
void trace(const BbCamera* c, V3 pos) {
    static const bool logging = std::getenv("HUNTER_CAM_LOG") != nullptr;
    static int frames = 0;
    if (!logging || frames++ % 8 != 0) return;
    const V3 fwd = unit(v3(c->forward));
    float feet[3] = {}, facing = 0;
    api->player_position(feet, &facing);
    // The character's yaw faces (-sin, 0, -cos): half a turn from the view's.
    const float view_yaw = std::atan2(fwd.x, fwd.z);
    float off = facing + kPi - view_yaw;
    while (off > kPi) off -= 2 * kPi;
    while (off < -kPi) off += 2 * kPi;
    api->log("hunter_cam: %s side %.2f reach %.2f zoom %.2f/%.2f away %.2f cam %.2f %.2f %.2f fov %.3f near %.3f view yaw "
             "%.3f facing off %.3f aim %d feet %.2f %.2f %.2f",
             g_first ? "fp" : "ots", static_cast<double>(view.side), static_cast<double>(view.reach),
             static_cast<double>(view.zoom), static_cast<double>(view.game_zoom), static_cast<double>(view.away),
             static_cast<double>(pos.x), static_cast<double>(pos.y), static_cast<double>(pos.z), static_cast<double>(c->fov_y),
             static_cast<double>(c->near_z), static_cast<double>(view_yaw), static_cast<double>(off),
             (g_aim_chr ? (g_aim_chr[kAimFlags] & kAimBit) : 0) | (g_monocular.load() ? 16 : 0), static_cast<double>(feet[0]),
             static_cast<double>(feet[1]), static_cast<double>(feet[2]));
}

// The game's own camera, zoomed by the wheel as the host zoomed it: along the
// view from the point it looks at, a wall pulling it in.
void game_camera_zoom(BbCamera* c, const std::uint8_t* f, float dt) {
    if (view.game_zoom == 1.0f && view.game_reach < 0) return;
    const V3 focus = v3(reinterpret_cast<const float*>(f + 0xd0));
    const V3 pos = v3(c->position);
    const float dist = length(pos - focus);
    if (dist < 0.05f) return;
    const V3 back = (pos - focus) * (1.0f / dist);
    float want = dist * view.game_zoom;
    if (std::uint8_t* world = collision_world()) {
        float frac = 1.0f;
        if (sphere_cast(world, focus, back * want, kRadius, frac)) want *= frac;
    }
    if (view.game_reach < 0 || want < view.game_reach) view.game_reach = want;
    else view.game_reach += (want - view.game_reach) * (1.0f - std::exp(-dt / kEaseOut));
    if (view.game_zoom == 1.0f && std::fabs(view.game_reach - dist) < 0.01f) {
        view.game_reach = -1.0f;
        return;
    }
    put(c->position, focus + back * view.game_reach);
}

void on_camera(BbCamera* c, void*) {
    const std::uint8_t* f = follow_cam(c);
    if (!f) return;
    const auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - view.last).count();
    view.last = now;
    if (!(dt >= 0) || dt > 0.25f) dt = 0.25f;
    const bool loading = api->loading_screen && api->loading_screen();
    if (!s.on || loading) {
        view.started = false;
        view.reach = -1.0f;
        view.have_focus = false;
        if (!loading) {
            read_zoom(false);
            game_camera_zoom(c, f, dt);
        }
        return;
    }
    if (!view.started) {
        view.started = true;
        view.side = view.side_want = view.side_from = s.shoulder;
        view.swap_t = 1.0f;
        api->log("hunter_cam: camera on (%s, %s shoulder, %.2f m back, %.2f m aside, %.2f m up)",
                 g_first ? "first person" : s.camera ? "over the shoulder" : "game camera", s.shoulder > 0 ? "right" : "left",
                 static_cast<double>(s.distance), static_cast<double>(s.offset), static_cast<double>(s.height));
    }
    read_keys();
    const bool first = g_first;
    if (!first && !s.camera) {  // the game's camera (first person off, the shoulder camera off)
        view.reach = -1.0f;
        view.have_focus = false;
        read_zoom(false);
        game_camera_zoom(c, f, dt);
        return;
    }
    view.game_reach = -1.0f;
    read_zoom(!first);

    // Another camera (the monocular's) has the player's camera: hand over -
    // but not to the aim camera our facing brings, whose turning we keep.
    const auto* own = reinterpret_cast<const BbCamera*>(f + 0x10);
    const bool aim = aim_is_ours();
    float away = 0.0f;
    if (!aim) {
        away = length(v3(c->position) - v3(own->position)) / kAwayPos;
        const float dfov = std::fabs(c->fov_y - own->fov_y) / kAwayFov;
        if (dfov > away) away = dfov;
        if (away > 1.0f) away = 1.0f;
    }
    const float step = kHandOver > 0 ? dt / kHandOver : 1.0f;
    view.away = view.away < away ? (view.away + step < away ? view.away + step : away)
                                 : (view.away - step > away ? view.away - step : away);
    if (view.away >= 1.0f) {
        view.reach = -1.0f;
        view.have_focus = false;
        trace(c, v3(c->position));
        return;
    }

    const V3 fwd = unit(v3(c->forward));
    V3 right = v3(c->right);
    right.y = 0;  // the shoulder is beside the hunter, whatever the pitch
    right = unit(right);
    if (length(fwd) < 0.5f || length(right) < 0.5f) return;

    // The hunter's feet, and the point from them: the look-at point in the aim
    // (whose camera sits in the hunter's head) and the eyes in first person.
    float feet[3] = {};
    const bool have_feet = api->player_position(feet, nullptr) == 0;
    V3 focus = v3(reinterpret_cast<const float*>(f + 0xd0));
    if (!aim && !first) {
        if (have_feet && view.away <= 0.0f) {
            const float eye = focus.y - feet[1];
            if (eye > 0.5f && eye < 2.5f) view.eye = eye;
            if (c->fov_y > 0.1f && c->fov_y < 2.5f) view.fov = c->fov_y;
        }
        view.focus = focus;
        view.have_focus = true;
    } else if (have_feet) {
        V3 target{feet[0], feet[1] + (first ? s.eye_height : view.eye), feet[2]};
        if (first) target = target + unit(V3{fwd.x, 0, fwd.z}) * kEyeForward;
        if (!view.have_focus || length(target - view.focus) > 5.0f) {
            view.focus = target;
            view.have_focus = true;
        } else {
            view.focus = view.focus + (target - view.focus) * (1.0f - std::exp(-dt / kChase));
        }
        focus = view.focus;
    }

    V3 pos;
    if (first) {
        pos = focus;
        if (c->near_z > 0.05f) c->near_z = 0.05f;  // the weapons in hand stay in view
    } else {
        // The shoulder, eased across on a swap.
        if (view.swap_t < 1.0f) {
            view.swap_t = s.swap_time > 0 ? view.swap_t + dt / s.swap_time : 1.0f;
            if (view.swap_t > 1.0f) view.swap_t = 1.0f;
            const float t = view.swap_t * view.swap_t * (3 - 2 * view.swap_t);
            view.side = view.side_from + (view.side_want - view.side_from) * t;
        } else {
            view.side = view.side_want;
        }
        const V3 pivot = focus + V3{0, s.height, 0};
        const V3 aside = right * (s.offset * view.side);
        V3 shoulder = pivot + aside;
        float want = s.distance * view.zoom;
        // Walls: out to the shoulder, then back from it.
        if (std::uint8_t* world = collision_world()) {
            float frac = 1.0f;
            if (length(aside) > 1e-3f && sphere_cast(world, pivot, aside, kRadius, frac)) shoulder = pivot + aside * frac;
            if (sphere_cast(world, shoulder, fwd * -want, kRadius, frac)) want *= frac;
        }
        if (want < kMinReach) want = kMinReach;
        // In at once, out eased.
        if (view.reach < 0 || want < view.reach) view.reach = want;
        else view.reach += (want - view.reach) * (1.0f - std::exp(-dt / kEaseOut));
        pos = shoulder - fwd * view.reach;
    }

    // Part way to the other camera.
    if (view.away > 0) {
        const float t = view.away * view.away * (3 - 2 * view.away);
        pos = pos + (v3(c->position) - pos) * t;
    }
    put(c->position, pos);
    const float deg = first ? s.first_fov : s.fov;
    if (deg > 0) {
        const float ours = deg * kPi / 180.0f;
        c->fov_y = ours + (c->fov_y - ours) * view.away;
    } else if (aim && view.fov > 0) {
        c->fov_y = view.fov;  // the aim camera zooms in; ours keeps the follow camera's
    }
    trace(c, pos);
}

// ---- the switches ------------------------------------------------------------

void apply() {
    omni_roll(s.on && s.dodge == 1);
    face_camera_frame();  // at once: the host may not call on_frame while the plugin is off
    if (!(s.on && g_first)) parts_show();
    const Key keys[2] = {s.swap_key, s.first_key};
    for (const Key k : keys) {
        if (const std::uint16_t sc = scancode(k); sc && api->key_bound && api->key_bound(sc))
            api->log("hunter_cam: a key of ours is also bound to a game action; it presses both");
    }
}

void on_frame(void*) {
    face_camera_frame();
    std::uint8_t* chr = player();
    const bool loading = api->loading_screen && api->loading_screen();
    if (s.on && g_first && chr && !loading) parts_hide(chr);
    else parts_show();
}

void on_option(const char* key, const char* value, void*) {
    const float was = s.shoulder;
    const bool was_first = s.first_person;
    read_options();
    if (key && std::strcmp(key, "shoulder") == 0 && s.shoulder != was) swap_to(s.shoulder);
    if (key && std::strcmp(key, "first_person") == 0 && s.first_person != was_first) g_first = s.first_person;
    apply();
    api->log("hunter_cam: %s = %s", key && *key ? key : "(plugin)", value ? value : "?");
}

}  // namespace

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* host) {
    api = host;
    if (host->version < 13 || !has_field(offsetof(BbHostApi, key_bound))) {
        host->log("hunter_cam: needs plugin API 13 (on_camera); this host has %u", host->version);
        return 1;
    }
    if (!host->eboot_is_109()) {
        host->log("hunter_cam: not the 1.09 eboot; off");
        return 1;
    }
    read_options();
    g_first = s.first_person;
    host->on_option("hunter_cam", &on_option, nullptr);
    if (host->on_camera(&on_camera, nullptr) != 0) host->log("hunter_cam: the host has no camera hook; the camera stays the game's");
    host->on_frame(&on_frame, nullptr);
    g_env_hooked = host->hook2(kEnv, kEnvPrologue, sizeof(kEnvPrologue), &env_hook, nullptr) == 0;
    if (!g_env_hooked)
        host->log("hunter_cam: the script's env queries are not where they should be: facing the camera walks, Roll "
                  "stays the game's");
    apply();
    host->log("hunter_cam: Hunter Cam %s (%s, %s shoulder, face the camera %s, dodge %s)", s.on ? "on" : "off",
              g_first ? "first person" : s.camera ? "over the shoulder" : "game camera", s.shoulder > 0 ? "right" : "left",
              s.face_camera ? "on" : "off", s.dodge ? "sidestep" : "roll");
    return 0;
}
