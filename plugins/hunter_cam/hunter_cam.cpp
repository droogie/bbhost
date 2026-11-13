// Hunter Cam: the hunter seen from closer in. Over the shoulder - a close
// third-person camera behind one shoulder, as Silent Hill 2's remake frames its
// hero, swapped on the fly with a key and zoomed with the wheel - or first
// person, where only the hunter's arms and weapons are drawn and the mouse
// turns the view; a mode that keeps the hunter facing where the camera looks
// at full speed; the right stick's speed; and the dodge as a roll or a sidestep.
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
//   field_of_view = 0        # % wider than the game's, as F10's Field of view
//   swap_time = 0.25         # seconds a swap takes
//   first_person = false     # the view from the hunter's eyes
//   first_person_key = "Z"   # toggles first person while playing
//   head_bob = false         # the eyes ride on the hunter's head in first person
//   eye_height = 1.55        # metres above the feet, in first person (head_bob off)
//   first_person_field_of_view = 0  # % wider, in first person (0: field_of_view's)
//   look_limit = 80          # degrees up or down in first person (the game's aim: 35 up, 40 down)
//   stick_speed = 1.0        # the right stick's turning, times the game's
//   mouse_sensitivity = 50   # the mouse in the aim (first person, facing the camera), F10's scale
//   mouse_invert_y = false   # and its up and down reversed
//   face_camera = false      # always face where the camera looks
//   crosshair = true         # the monocular's sight while facing the camera
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
// hunter's eyes: their feet raised by the Eye height, a hand's width forward -
// or, with Head bob, the eyes' dummy poly on the head's bone, as posed.
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
// part but the weapons and the arms (the gloves, bare or armoured) - the slots
// whose model is a WP_ or AM_ file - is hidden that way, frame by frame (a change of equipment makes new ones), and shown again
// as first person ends, the plugin goes off or the world loads. The hunter
// faces where the camera looks there, as with Always face the camera, and the
// aim camera that brings may turn 80 degrees up and down (Look up/down limit)
// where the game holds it to 35 and 40. Its field of view is the follow
// camera's, which only updates while it has the view: F10's Field of view
// reaches it through LockCamParam's camFovY, read directly.
//
// The aim. With the precision-shooting bit set the game also hides X's prompt
// and press (the action button's and the NPC talk's checks of it), refuses
// items their effect, and puts a sight at the screen's middle; while the bit
// is the plugin's those checks are passed, and the sight's only with the Crosshair on. The mouse turns the
// aim camera itself (the host's mouse turns only the follow camera).
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
// and only what the plugin wrote is taken back. Aiming (facing the camera, first
// person), a roll could only go straight ahead - the game holds the aiming
// hunter's facing, and reads the aim before the script runs - so a roll asked
// for there is put off a frame (the stamina query answered 0), the aim dropped
// for the roll, and the script, asking again, rolls the hunter the stick's way
// as unaimed; the view is held still meanwhile and the aim given back its angles.
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
    "1.3.2",
    "Kyo",
    "The hunter from closer in: an over-the-shoulder camera (right or left, swapped with a key, zoomed with the wheel) "
    "or first person with only the arms and weapons drawn; the hunter can always face where the camera looks, the right "
    "stick's speed is yours, and the dodge can be a roll or a sidestep.",
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
    {BB_OPT_INT | BB_OPT_LIVE, "field_of_view", "Field of view (+%)",
     "How much wider the camera sees, as F10's Field of view widens it: 0 is the game's (with F10's own on top); +50 "
     "about 97 degrees across.",
     "0", 0, 100, nullptr},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "swap_time", "Swap time", "Seconds the camera takes to move to the other shoulder.", "0.25",
     0, 1, nullptr},
    {BB_OPT_HEADING, "h_first", "First person", "The view from the hunter's eyes: only their weapons are drawn.", nullptr, 0,
     0, nullptr},
    {BB_OPT_BOOL | BB_OPT_LIVE, "first_person", "First person",
     "Starts in first person. The hunter always faces where they look there.", "false", 0, 0, nullptr},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "first_person_key", "First person key",
     "Switches between first person and the camera above while playing (keyboard and mouse).", "Z", 0, 0, KEY_CHOICES},
    {BB_OPT_BOOL | BB_OPT_LIVE, "head_bob", "Head bob",
     "The view rides on the hunter's head: it bobs, sways and dips as the head does when they run, roll and swing. Off: "
     "a steady view at the Eye height.",
     "false", 0, 0, nullptr},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "eye_height", "Eye height", "Metres above the hunter's feet (Head bob off).", "1.55", 1.2,
     1.9, nullptr},
    {BB_OPT_INT | BB_OPT_LIVE, "first_person_field_of_view", "First-person field of view (+%)",
     "How much wider the view from the eyes is, as F10's Field of view widens it: 0 takes the Field of view above.", "0",
     0, 100, nullptr},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "look_limit", "Look up/down limit",
     "Degrees the view turns up or down from level in first person.", "80", 45, 89, nullptr},
    {BB_OPT_HEADING, "h_controls", "Turning", nullptr, nullptr, 0, 0, nullptr},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "stick_speed", "Right stick speed",
     "How fast the right stick turns the camera, times the game's own (its Camera speed option still applies).", "1.0",
     0.25, 4.0, nullptr},
    {BB_OPT_INT | BB_OPT_LIVE, "mouse_sensitivity", "Mouse sensitivity (facing the camera)",
     "How far the mouse turns the view in first person and with Always face the camera, on F10's Mouse sensitivity "
     "scale (50 its default); elsewhere F10's own applies.",
     "50", 0, 100, nullptr},
    {BB_OPT_BOOL | BB_OPT_LIVE, "mouse_invert_y", "Reverse mouse Y-axis (facing the camera)",
     "Reverses the mouse's up and down there, as F10's Reverse mouse Y-axis does elsewhere.", "false", 0, 0, nullptr},
    {BB_OPT_HEADING, "h_movement", "Movement", nullptr, nullptr, 0, 0, nullptr},
    {BB_OPT_BOOL | BB_OPT_LIVE, "face_camera", "Always face the camera",
     "The hunter always faces where the camera looks and strafes by the stick, at full speed (they run and sprint as "
     "when locked on).",
     "false", 0, 0, nullptr},
    {BB_OPT_BOOL | BB_OPT_LIVE, "crosshair", "Crosshair",
     "The monocular's sight at the middle of the screen while the hunter faces the camera (and in first person).", "true",
     0, 0, nullptr},
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
    float distance = 1.7f, offset = 0.55f, height = 0.15f, swap_time = 0.25f;
    int fov = 0;  // +% wider
    bool first_person = false, head_bob = false;
    float eye_height = 1.55f, look_limit = 80.0f;
    int first_fov = 0;  // +% wider
    float stick_speed = 1.0f;
    int mouse_sensitivity = 50;
    bool mouse_invert_y = false;
    bool face_camera = false, crosshair = true;
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
    s.fov = static_cast<int>(number("hunter_cam.field_of_view", 0.0f, 0.0f, 100.0f) + 0.5f);
    s.swap_time = number("hunter_cam.swap_time", 0.25f, 0.0f, 1.0f);
    s.first_person = flag("hunter_cam.first_person", false);
    s.eye_height = number("hunter_cam.eye_height", 1.55f, 1.2f, 1.9f);
    s.head_bob = flag("hunter_cam.head_bob", false);
    s.first_fov = static_cast<int>(number("hunter_cam.first_person_field_of_view", 0.0f, 0.0f, 100.0f) + 0.5f);
    s.look_limit = number("hunter_cam.look_limit", 80.0f, 45.0f, 89.0f);
    s.stick_speed = number("hunter_cam.stick_speed", 1.0f, 0.25f, 4.0f);
    s.mouse_sensitivity = static_cast<int>(number("hunter_cam.mouse_sensitivity", 50.0f, 0.0f, 100.0f) + 0.5f);
    s.mouse_invert_y = flag("hunter_cam.mouse_invert_y", false);
    s.face_camera = flag("hunter_cam.face_camera", false);
    s.crosshair = flag("hunter_cam.crosshair", true);
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

// ---- the game's own checks of the precision-shooting bit ---------------------
//
// Each is `test <r>h, 4` on ChrIns +0x1e1's dword (the bit +0x1e2 & 4); its
// immediate made 0, the check passes as if the hunter were not aiming.

struct CodeSite {
    std::uint64_t at;
    std::uint8_t game[3];
};
// What X does: the action button's two checks (SprjActionButtonMan, 0x20d91f0's
// query and its update: no prompt, no press while aiming) and the NPC talk's
// three (SprjWorldTalkMan's).
const CodeSite kInteract[] = {{0x20d9323, {0xf6, 0xc6, 0x04}},
                              {0x20d992a, {0xf6, 0xc5, 0x04}},
                              {0x1d3d1f1, {0xf6, 0xc4, 0x04}},
                              {0x1d3d3c5, {0xf6, 0xc4, 0x04}},
                              {0x243e787, {0xf6, 0xc4, 0x04}}};
// The sight: the HUD puts the lock cursor at the screen's middle (960, 540)
// while the bit is set (0x1ffe4e1) - else while the monocular's stateInfo 15
// is on, which still shows it when the monocular itself is in use.
const CodeSite kSight[] = {{0x1ffe4e1, {0xf6, 0xc5, 0x04}}};
// Items: the item's use (sub_197f200, reached once the animation gets to the
// moment it takes effect) is refused while aiming - the bit, masked at
// 0x197f9b4 (and rcx, 0x400), joins the other "not now" reasons - so a Blood
// Vial is drunk, a paper rubbed on, a tool raised, and nothing heals, buffs or
// is spent. Its mask made 0, items work as they do unaimed.
const CodeSite kItems[] = {{0x197f9b6, {0xe1, 0x00, 0x04}}};

struct SiteSet {
    const CodeSite* sites;
    std::size_t n;
    const char* what;
    bool on = false;
    bool refused = false;
};
SiteSet g_interact{kInteract, sizeof(kInteract) / sizeof(kInteract[0]), "X's action button and talk"};
SiteSet g_sight{kSight, sizeof(kSight) / sizeof(kSight[0]), "the crosshair"};
SiteSet g_items{kItems, sizeof(kItems) / sizeof(kItems[0]), "items"};
// Until when (steady clock, ns) a roll runs unaimed. Aiming, the game holds
// the hunter facing where the camera looks - it reads that before the script
// runs - so a roll would only ever go straight ahead. The script's roll is
// put off a frame instead (env_hook), the aim dropped, and the roll it then
// starts turns to the stick as unaimed rolls do.
std::atomic<long long> g_roll_until{0};
constexpr long long kRollTurnNs = 800000000;  // 0.8 s: the roll, then the aim again


long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The checks passed (on) or the game's again (off).
void set_sites(SiteSet& g, bool on) {
    if (on == g.on || g.refused) return;
    std::size_t done = 0;
    for (; done < g.n; ++done) {
        const CodeSite& t = g.sites[done];
        const std::uint8_t ours[3] = {t.game[0], t.game[1], 0};
        if (api->patch(t.at, on ? t.game : ours, on ? ours : t.game, 3) != 0) break;
    }
    if (done < g.n) {  // not the code we know: put back what was changed and leave it be
        for (std::size_t k = 0; k < done; ++k) {
            const CodeSite& t = g.sites[k];
            const std::uint8_t ours[3] = {t.game[0], t.game[1], 0};
            api->patch(t.at, on ? ours : t.game, on ? t.game : ours, 3);
        }
        g.refused = true;
        api->log("hunter_cam: 0x%llx is not the code expected; %s stays the game's", static_cast<unsigned long long>(g.sites[done].at),
                 g.what);
        return;
    }
    g.on = on;
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

// Plugin API 29: one of the game's cutscenes asked for or playing. It places
// the camera and the hunter itself: the camera, the facing (the aim bit) and
// the body go back to the game for its length, as at a load.
bool in_cutscene() {
    return api && api->version >= 29 && api->size >= offsetof(BbHostApi, cutscene) + sizeof(void*) && api->cutscene &&
           api->cutscene() != 0;
}
bool aim_wanted() { return s.on && (s.face_camera || g_first) && !in_cutscene(); }
bool rolling() { return now_ns() < g_roll_until.load(); }

// Once a frame, on the main thread: the bit set while it should be, cleared
// once it should not (only when we set it).
void face_camera_frame() {
    const bool loading = api->loading_screen && api->loading_screen();
    const bool want = aim_wanted() && !loading && !rolling();
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
    const bool ours = g_aim_ours && !g_monocular.load();
    g_answer_aim.store(ours);
    g_force_roll.store(s.on && s.dodge == 0);
    // Aiming as ours, X still opens doors, picks up and talks; the sight only
    // if wanted. The monocular itself keeps the game's checks.
    set_sites(g_interact, ours);
    set_sites(g_items, ours);
    set_sites(g_sight, ours && !s.crosshair);
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
        if (g_dodge == 2 && aim && (me[kAimFlags] & kAimBit)) {
            // A roll asked for while aiming: not this frame (no stamina, as
            // far as the script knows) - the aim drops, and the request, still
            // there next frame, rolls unaimed.
            g_dodge = 0;
            g_roll_until.store(now_ns() + kRollTurnNs);
            me[kAimFlags] &= static_cast<std::uint8_t>(~kAimBit);
            c->ret = 0;
            return 1;
        }
    } else if (g_dodge == 2 && ++g_dodge_calls > 8) {
        g_dodge = 0;
    }
    if (id != kEnvLock && id != kEnvAim) return 0;
    if (g_dodge == 2) {  // the roll, locked on or not
        if (id == kEnvAim) g_dodge = 0;
        c->ret = 0;
        return 1;
    }
    if (aim && !rolling()) {  // the lock-on's locomotion - it runs - not precision shooting's walk
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
    std::uint32_t weapons = 0;  // a bit a slot: the ones kept drawn (a WP_ or AM_ model)
} parts;

// A UTF-16 name at a runtime address: whether it starts with `prefix`.
bool name_starts(std::uint64_t at, const char* prefix) {
    std::uint16_t w[8];
    if (at < 0x10000 || api->read(at, w, sizeof(w)) != 0) return false;
    for (int i = 0; prefix[i]; ++i)
        if (w[i] != static_cast<std::uint8_t>(prefix[i])) return false;
    return true;
}

// Whether a part stays drawn in first person: its file name (model ins +0x148
// -> +0x18 -> UTF-16: WP_A_1000, AM_M_8000, HD_F_8000, FG_A_7003 - FG the
// face's parts) starts with WP_ (a weapon) or AM_ (the arms, bare or armoured).
bool part_kept(std::uint8_t* mi) {
    std::uint64_t res = 0, name = 0;
    return api->read(reinterpret_cast<std::uint64_t>(mi) + 0x148, &res, 8) == 0 && res >= 0x10000 &&
           api->read(res + 0x18, &name, 8) == 0 && (name_starts(name, "WP_") || name_starts(name, "AM_"));
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
            if (mi && part_kept(mi)) parts.weapons |= 1u << k;
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
    if (changed) api->log("hunter_cam: first person keeps the weapon and arm parts 0x%06x drawn", parts.weapons);
}

// ---- the head: a dummy poly of the body's model ------------------------------
//
// ChrIns's vtable +0x6b0 (chr, &dummy) gives the model a dummy poly is on (the
// body's for one under 1000); the model's +0x50 its table - +4 the count, +8
// the entries (0x50 bytes each), +0x10 (id, first entry) pairs sorted by id -
// and +0x58 (model, out, &{entry, 1}) an entry's world matrix as the skeleton
// stands now, its position in the last row (what SprjSfx's dummy-poly
// attachment does, 0x2135bb0). Dummy poly 7 is at the eyes: 1.54 m over the
// feet, on the head's bone.

constexpr std::int32_t kEyesDummy = 7;

std::uint64_t bn_of_rt(std::uint64_t rt) { return rt - api->guest_addr(0x400000) + 0x400000; }

bool head_position(std::uint8_t* chr, V3& out) {
    if (!chr) return false;
    const auto* vt = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(get<std::uint64_t>(chr, 0)));
    if (!vt) return false;
    static std::int32_t dummy;
    dummy = kEyesDummy;
    BbCallRegs r{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(chr);
    r.arg[1] = reinterpret_cast<std::uint64_t>(&dummy);
    if (api->call_guest(bn_of_rt(vt[0x6b0 / 8]), &r) != 0 || r.ret < 0x10000) return false;
    auto* model = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(r.ret));
    const auto* mvt = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(get<std::uint64_t>(model, 0)));
    r = BbCallRegs{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(model);
    if (!mvt || api->call_guest(bn_of_rt(mvt[0x50 / 8]), &r) != 0 || r.ret < 0x10000) return false;
    const auto* table = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(r.ret));
    const std::int32_t count = get<std::int32_t>(table, 4);
    const std::uint8_t* entries = get<std::uint8_t*>(table, 8);
    const std::uint8_t* pairs = get<std::uint8_t*>(table, 0x10);
    if (!entries || !pairs || count <= 0 || count > 4096) return false;
    int lo = 0, hi = count - 1, found = -1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        const std::int32_t id = get<std::int32_t>(pairs, 8 * static_cast<std::size_t>(mid));
        if (id == kEyesDummy) {
            found = mid;
            break;
        }
        if (id < kEyesDummy) lo = mid + 1;
        else hi = mid - 1;
    }
    if (found < 0) return false;
    const std::int32_t first = get<std::int32_t>(pairs, 8 * static_cast<std::size_t>(found) + 4);
    alignas(16) static float m[16];
    alignas(16) static std::uint64_t span[2];
    std::memset(m, 0, sizeof m);
    span[0] = reinterpret_cast<std::uint64_t>(entries + 0x50 * static_cast<std::size_t>(first));
    span[1] = 1;
    r = BbCallRegs{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(model);
    r.arg[1] = reinterpret_cast<std::uint64_t>(m);
    r.arg[2] = reinterpret_cast<std::uint64_t>(span);
    if (api->call_guest(bn_of_rt(mvt[0x58 / 8]), &r) != 0 || m[15] != 1.0f) return false;
    out = V3{m[12], m[13], m[14]};
    return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z);
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
    float row_fov = 0.0f;     // its LockCamParam row's while it last ran (0: not yet)
    V3 focus{0, 0, 0};        // from the hunter: the look-at point (or the eyes) chased after them
    bool have_focus = false;
    bool keys_down[2] = {};   // the swap key, the first-person key
    float zoom = 1.0f;        // the wheel's, on the shoulder camera's distance
    float game_zoom = 1.0f;   // the wheel's on the game's camera, once we took the keys
    float game_reach = -1.0f; // its eased distance
    bool zoom_taken = false;
    float mouse_dx = 0.0f, mouse_dy = 0.0f;  // this frame's mouse counts while playing
    float basis[3][4] = {};   // the last frame's view (right, up, forward)
    float held[3][4] = {};    // held through a roll's unaimed moments
    bool have_basis = false, roll_hold = false;
    float held_angles[2] = {};  // the aim's pitch and yaw as the roll began
    int restore = 0;            // frames left putting them back
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
constexpr float kHeadForward = 0.04f; // with Head bob: metres ahead of the eyes' dummy poly, past the face
constexpr float kZoomStep = 1.15f, kZoomNear = 0.6f, kZoomFar = 6.0f;             // the shoulder camera's
constexpr float kGameZoomStep = 1.2f, kGameZoomNear = 0.5f, kGameZoomFar = 6.0f;  // the host's own

// LockCamParam row 0's camFovY in radians, 0 if unread: F10's Field of view
// scales every row's by one factor (engine/camera.cpp), so how it has changed
// says how the follow camera's would have.
float lock_cam_fov(const std::uint8_t*) {
    static float last = 0;
    static int age = 0;
    if (age++ % 15 != 0) return last;  // twice a second is enough for a setting
    char v[32] = {};
    last = 0;
    if (api->param_get("LockCamParam", 0, "camFovY", v, sizeof v) == 0) {
        const float deg = std::strtof(v, nullptr);
        if (deg > 0 && deg < 170) last = deg * kPi / 180.0f;
    }
    return last;
}

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

// The keys' presses, once per press while playing (one read of the input a frame).
void read_keys() {
    BbInput in;
    const bool ok = api->input(&in) == 0 && in.playing;
    view.mouse_dx = ok ? in.dx : 0.0f;
    view.mouse_dy = ok ? in.dy : 0.0f;
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
        } else if (i == 1) {
            g_first = !g_first;
            view.reach = -1.0f;
            view.have_focus = false;
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

// ---- turning: the stick's speed, the mouse in the aim, first person's pitch ---
//
// The follow camera (ChrExFollowCam) turns at speeds it eases between its
// param's minimum and maximum by the Camera speed option (+0x20c..+0x218, the
// fast turn's +0x224..+0x230); the aim camera (ChrAimCam, the manager's +0x68,
// sub_1833860) the same way from +0x104..+0x110, its pitch (+0x148) and yaw
// (+0x14c) turned by them and the pitch held within +0xf4..+0xf0 degrees. The
// plugin scales the speeds by the Right stick speed, sets the aim's pitch
// limits in first person, and turns the aim by the mouse as the host turns the
// follow camera; what it wrote is put back when it is no longer wanted, and a
// value it did not write is the game's new one.

constexpr std::size_t kAimCam = 0x68;  // the camera manager's ChrAimCam
constexpr std::size_t kAimPitch = 0x148, kAimYaw = 0x14c, kAimPitchMax = 0xf0, kAimPitchMin = 0xf4;
constexpr float kTwoPi = 2 * kPi;

struct Held {
    std::size_t off;
    int n;
    std::uint8_t* cam = nullptr;
    float game[4]{}, ours[4]{};
    bool held = false;
};
Held g_follow_speed{0x20c, 4}, g_follow_fast{0x224, 4}, g_aim_speed{0x104, 4}, g_aim_limits{kAimPitchMax, 2};

// The game's values back, if they are still ours and the camera is the same.
void release(Held& v, std::uint8_t* cam) {
    if (v.held && cam && cam == v.cam) {
        float now[4];
        std::memcpy(now, cam + v.off, 4 * static_cast<std::size_t>(v.n));
        if (std::memcmp(now, v.ours, 4 * static_cast<std::size_t>(v.n)) == 0)
            std::memcpy(cam + v.off, v.game, 4 * static_cast<std::size_t>(v.n));
    }
    v.held = false;
    v.cam = cam;
}

// Ours in place of the game's: f(the game's value, its index).
template <typename F>
void hold(Held& v, std::uint8_t* cam, F f) {
    if (!cam) {
        v.held = false;
        v.cam = nullptr;
        return;
    }
    const std::size_t bytes = 4 * static_cast<std::size_t>(v.n);
    float now[4];
    std::memcpy(now, cam + v.off, bytes);
    if (cam != v.cam || !v.held || std::memcmp(now, v.ours, bytes) != 0) std::memcpy(v.game, now, bytes);
    v.cam = cam;
    v.held = true;
    for (int i = 0; i < v.n; ++i) v.ours[i] = f(v.game[i], i);
    std::memcpy(cam + v.off, v.ours, bytes);
}

// The host's Mouse sensitivity scale (engine/mouse_camera_step.h): 0.005 to 0.4
// degrees a count over 0..100, 0.045 at 50.
float degrees_per_count(int setting) {
    const float t = static_cast<float>(setting < 0 ? 0 : setting > 100 ? 100 : setting) * 0.01f;
    return 0.005f * std::pow(0.4f / 0.005f, t);
}

// The mouse turns the aim, after the stick (the next update draws it): as the
// host turns the follow camera, down looks down unless reversed.
void mouse_turn(std::uint8_t* aim) {
    if (!aim || (view.mouse_dx == 0.0f && view.mouse_dy == 0.0f)) return;
    const float k = degrees_per_count(s.mouse_sensitivity) * kPi / 180.0f;
    float pitch = get<float>(aim, kAimPitch) + (s.mouse_invert_y ? -1.0f : 1.0f) * view.mouse_dy * k;
    float yaw = get<float>(aim, kAimYaw) + view.mouse_dx * k;
    const float hi = get<float>(aim, kAimPitchMax) * kPi / 180.0f, lo = get<float>(aim, kAimPitchMin) * kPi / 180.0f;
    if (lo < hi) pitch = pitch < lo ? lo : pitch > hi ? hi : pitch;
    for (int i = 0; i < 64 && yaw > kPi; ++i) yaw -= kTwoPi;
    for (int i = 0; i < 64 && yaw < -kPi; ++i) yaw += kTwoPi;
    std::memcpy(aim + kAimPitch, &pitch, 4);
    std::memcpy(aim + kAimYaw, &yaw, 4);
}

// Once a frame while the plugin is on: the speeds, the limits and the mouse.
void turning(std::uint8_t* follow, std::uint8_t* aim) {
    const float k = s.stick_speed;
    if (k != 1.0f) {
        const auto scale = [k](float game, int) { return game * k; };
        hold(g_follow_speed, follow, scale);
        hold(g_follow_fast, follow, scale);
        hold(g_aim_speed, aim, scale);
    } else {
        release(g_follow_speed, follow);
        release(g_follow_fast, follow);
        release(g_aim_speed, aim);
    }
    if (g_first && aim) {  // the game's: 35 degrees up and 40 down
        const float deg = s.look_limit;
        hold(g_aim_limits, aim, [deg](float, int i) { return i == 0 ? deg : -deg; });
    } else {
        release(g_aim_limits, aim);
    }
    if (aim_is_ours()) mouse_turn(aim);
}

void turning_off(std::uint8_t* follow, std::uint8_t* aim) {
    release(g_follow_speed, follow);
    release(g_follow_fast, follow);
    release(g_aim_speed, aim);
    release(g_aim_limits, aim);
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
    const std::uint8_t* aim = get<std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(c) - 0x10, kAimCam);
    if (aim)
        api->log("hunter_cam: aim pitch %.3f yaw %.3f limits %.4f %.4f speeds %.3f %.3f %.3f %.3f fwd %.3f %.3f %.3f",
                 static_cast<double>(get<float>(aim, kAimPitch)), static_cast<double>(get<float>(aim, kAimYaw)),
                 static_cast<double>(get<float>(aim, kAimPitchMax)), static_cast<double>(get<float>(aim, kAimPitchMin)),
                 static_cast<double>(get<float>(aim, 0x104)), static_cast<double>(get<float>(aim, 0x108)),
                 static_cast<double>(get<float>(aim, 0x10c)), static_cast<double>(get<float>(aim, 0x110)),
                 static_cast<double>(fwd.x), static_cast<double>(fwd.y), static_cast<double>(fwd.z));
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
    std::uint8_t* const follow = const_cast<std::uint8_t*>(f);
    std::uint8_t* const aim_cam = get<std::uint8_t*>(reinterpret_cast<std::uint8_t*>(c) - 0x10, kAimCam);
    const bool cutscene = in_cutscene();
    if (!s.on || loading || cutscene) {
        turning_off(follow, aim_cam);
        view.started = false;
        view.reach = -1.0f;
        view.have_focus = false;
        if (!loading && !cutscene) {
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
    turning(follow, aim_cam);
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
    // (Through a roll the aim is down and the game blends to its follow
    // camera, which takes the aim's angles: still ours, so the view stays put.)
    const bool aim = aim_is_ours() || (aim_wanted() && rolling() && !g_monocular.load());
    if (aim && !aim_is_ours()) {
        // The view held as it was, and the follow camera kept on the aim's
        // angles, which the aim takes back from it after the roll.
        if (!view.roll_hold && view.have_basis) {
            std::memcpy(view.held, view.basis, sizeof view.held);
            if (aim_cam) std::memcpy(view.held_angles, aim_cam + kAimPitch, 8);
        }
        view.roll_hold = view.have_basis;
        if (view.roll_hold) {
            std::memcpy(c->right, view.held[0], 16);
            std::memcpy(c->up, view.held[1], 16);
            std::memcpy(c->forward, view.held[2], 16);
        }
        if (aim_cam) {
            std::memcpy(follow + 0x140, aim_cam + kAimPitch, 4);
            std::memcpy(follow + 0x144, aim_cam + kAimYaw, 4);
        }
    } else if (view.roll_hold || view.restore > 0) {
        // The aim back: it takes its angles from the follow camera, which
        // drifted (a reset to +0x154/+0x158 that its next update applies) -
        // the ones it had instead, and the view as held until it has them.
        if (view.roll_hold) view.restore = 3;
        view.roll_hold = false;
        --view.restore;
        if (aim_cam && aim_is_ours()) {
            std::memcpy(aim_cam + kAimPitch, view.held_angles, 8);
            std::memcpy(aim_cam + 0x154, view.held_angles, 8);
        }
        std::memcpy(c->right, view.held[0], 16);
        std::memcpy(c->up, view.held[1], 16);
        std::memcpy(c->forward, view.held[2], 16);
    }
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
    } else if (V3 head; first && s.head_bob && head_position(player(), head)) {
        // The eyes go where the head goes, each frame as it is posed.
        view.focus = head + unit(V3{fwd.x, 0, fwd.z}) * kHeadForward;
        view.have_focus = true;
        focus = view.focus;
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
    // The field of view: the follow camera's (the aim camera zooms in; ours
    // keeps the follow camera's), widened as F10's widens it.
    float base = c->fov_y;
    const float row = lock_cam_fov(f);
    if (aim) {
        // The follow camera eases to its LockCamParam row's camFovY (which
        // F10's Field of view scales) only in its own update, which does not
        // run while the aim camera has the view: its last, scaled as the row
        // has changed since.
        if (own->fov_y > 0.1f && own->fov_y < 2.5f) base = own->fov_y;
        else if (view.fov > 0) base = view.fov;
        if (row > 0 && view.row_fov <= 0) view.row_fov = row;  // never seen it run: take it as current
        if (row > 0) base *= row / view.row_fov;
    } else if (row > 0) {
        view.row_fov = row;
    }
    const int pct = first && s.first_fov > 0 ? s.first_fov : s.fov;
    float ours_fov = base * (1.0f + 0.01f * static_cast<float>(pct));
    if (ours_fov > 2.6f) ours_fov = 2.6f;
    c->fov_y = ours_fov + (c->fov_y - ours_fov) * view.away;
    std::memcpy(view.basis[0], c->right, 16);
    std::memcpy(view.basis[1], c->up, 16);
    std::memcpy(view.basis[2], c->forward, 16);
    view.have_basis = true;
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

    if (s.on && g_first && chr && !loading && !in_cutscene()) parts_hide(chr);
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
