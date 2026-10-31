// Quake: Bloodborne in first person, moving and seeing as Quake 3 does, with
// a choice of physics: Quake 3's own, CPMA, Quake 1 + CPMA, QuakeWorld and
// DeFRaG's.
//
//   [plugins]
//   quake = true
//   [quake]
//   enabled = true             # the setup window's and the in-game plugin menu switch it
//   physics = "Quake 3 (VQ3)"  # or CPMA, Quake 1 + CPMA, DeFRaG (VQ3), DeFRaG (CPM), QuakeWorld
//   auto_hop = true            # holding jump hops again on landing
//   gravity = 800              # g_gravity
//   knockback = 1000           # g_knockback: rocket and grenade jumps, plasma climbing
//   sensitivity = 5.0          # Quake 3's: 0.022 degrees a mouse count, times this
//   fov = 90                   # Quake 3's cg_fov: across a 4:3 picture
//   draw_gun = true            # the gun at the lower right, with gun_bob, gun_fov and gun_x/y/z
//   plasma = "A Call Beyond"   # or Blacksky Eye: what the plasma gun fires
//
// Every setting is live: the launcher's Plugins page and F9's change it while
// playing.
//
// Where it sits. The game's state update runs the camera manager's blend
// (the follow camera's update, then the blend into the manager's own camera)
// and then copies that camera into the scene's; between the two the host
// calls on_camera (plugin API 13). The character's own update has run by
// then, so that one callback is the frame here: the movement, the player put
// where it left them, and the camera put at the eye with Quake 3's view on
// top (cg_view.c: the bob, the roll and pitch from the velocity, the landing
// dip, the step and crouch smoothing, the damage kick).
//
// The movement is bg_pmove.c's, run on the frame in steps of at most 1/125 s
// (Quake 3 at com_maxfps 125), with its presets (VQ3, CPM, QuakeWorld and a
// Quake 1 + CPMA hybrid), and DeFRaG's: VQ3's or
// CPM's in steps of exactly 8 ms with the velocity rounded after each, as
// Quake 3 rounds it (the higher 125 fps jump). Its traces are the game's own sphere cast
// (0x1c090e0, the follow camera's) with the characters' filter, so the body
// meets the map the hunter walks on (its lo-hit: the invisible walls along
// ledges, the stairs smoothed into ramps, which it climbs up to 60 degrees):
// three spheres, at the feet, the waist and the head, standing as Quake 3's
// box does on its feet alone. The filter meets no characters, so the
// creatures and other hunters near the player are
// met as Quake 3 meets other players: upright cylinders of their NpcParam hit
// radius and height, slid along and stood on. The game is told nothing of the movement: W, A, S,
// D, Space, Left Ctrl, the right mouse button and the mouse's motion are taken
// from it (input_take), the player is placed where the movement put them each
// frame - as the game's own teleport does it (sub_1d39c80: the physics
// module's position and its copy, its warp byte, the character proxy's
// transform) - and the game's gravity is held off the player (ChrIns+0x1e1
// bit 6, the debug menu's "No Gravity"). Of the body only the gun is drawn
// (the viewmodel, below); with the gun off, nothing (ChrIns+0x1e2 bit 4,
// DRAW_ENABLE). Bloodborne's fall damage never comes: the game's own fall is
// never started.
//
// Units: a Quake unit is 1/32 m, which makes the hunter's 1.75 m Quake 3's
// 56-unit player. Speeds are in units a second, as Quake's are: 320 is 10 m/s,
// a jump of 270 rises 1.42 m, and gravity is Quake's 800 (25 m/s^2).
//
// The guns: the left mouse button fires the one that is up from Quake 3's
// muzzle along the view, at Quake 3's rate, through the game's own bullet
// emitters (the shot, below), with the gun's own muzzle flash and sound from
// its firing animation; the wheel and the free number keys switch them, the
// new one named small in the bottom left corner. Every level hits as +10 and
// scales with the hunter's stats; the rocket and the grenade fly by Quake 3's
// numbers (the grenade bouncing until its 2.5 s fuse), the shotgun and the
// machine gun fire their guns' own bullets, the plasma gun one of A Call
// Beyond's stars or Blacksky Eye's shard, and the blasts throw the player as
// Quake 3's splash does. The gun that is up is drawn in the left hand's place
// and put at the lower right of the view with Quake 3's weapon bob, kick, and
// drop and raise.
#include "bbhost/sdk.hpp"


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN | BB_PLUGIN_GAMEPLAY,
    "quake",
    "Quake",
    "0.3.16",
    "Kyo",
    "First person with Quake 3's movement and view: strafe jumping, bunny hopping, crouching, and the bob, roll and "
    "landing dip of Quake 3's camera.",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, nullptr, "Quake",
     "W A S D move, Space or the right mouse button jumps, Left Ctrl crouches, the left mouse button fires, the wheel or "
     "1-5 (where unbound) pick the weapon."},
    {BB_OPT_BOOL | BB_OPT_LIVE, "enabled", "First person", "Off gives the hunter back to the game.", "true"},
    {BB_OPT_HEADING, nullptr, "Movement"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "physics", "Physics",
     "Whose movement. Quake 3 (VQ3): strafe jump with W and A or D. CPMA: Challenge ProMode's air control, braking "
     "and double jump. Quake 1 + CPMA: A or D alone air strafes as QuakeWorld, W with A or D strafe jumps as Quake 3, "
     "W alone steers as CPMA, a quick second jump goes higher. DeFRaG: Quake 3's or CPMA's at DeFRaG's fixed 125 fps, "
     "the speed rounded each step as Quake 3 does - its higher jumps and overbounces. QuakeWorld: air strafing with A "
     "or D alone, low ground friction.",
     "Quake 3 (VQ3)", 0, 0, "Quake 3 (VQ3)|CPMA|Quake 1 + CPMA|DeFRaG (VQ3)|DeFRaG (CPM)|QuakeWorld"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "auto_hop", "Hold jump to hop",
     "Holding jump jumps again the moment you land, with no ground friction in between: bunny hopping without timing "
     "each press. Off is Quake: let go and press again for each hop.",
     "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "crouch", "Crouch", "Left Ctrl crouches: a quarter of the speed, a lower body.", "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "sounds", "Jump and land sounds",
     "The hunter's own push-off on a jump and thud on a landing - the floor's and the attire's sounds, as a roll and a "
     "fall make them - where Quake 3 plays jump1 and land1.",
     "true"},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "gravity", "Gravity", "Quake's g_gravity, units a second squared (800 is Quake's own).",
     "800", 100.0, 1600.0},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "knockback", "Knockback",
     "Quake 3's g_knockback: how hard your own explosions throw you - rocket and grenade jumps, plasma climbing (1000 is "
     "Quake 3's own, 0 none).",
     "1000", 0.0, 3000.0},
    {BB_OPT_HEADING, nullptr, "View"},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "sensitivity", "Mouse sensitivity",
     "Quake's sensitivity: 0.022 degrees a mouse count (m_yaw), times this. The same number turns as far as it does in "
     "Quake 3 at the same mouse DPI.",
     "5.0", 0.1, 30.0},
    {BB_OPT_BOOL | BB_OPT_LIVE, "invert", "Invert mouse", nullptr, "false"},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "fov", "Field of view",
     "Quake 3's cg_fov: degrees across a 4:3 picture; a wider screen sees more at the sides.", "90", 50.0, 140.0},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "bob", "View bob",
     "Quake 3's bob, roll and landing dip, times this (1 is Quake 3's own, 0 none).", "1.0", 0.0, 2.0},
    {BB_OPT_HEADING, nullptr, "Crosshair"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "crosshair", "Crosshair",
     "A dot in a ring at the middle of the view, as Quake 3's cg_drawCrosshair: drawn over the world and under the "
     "game's own HUD and menus.",
     "true"},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "crosshair_size", "Crosshair size",
     "Quake 3's cg_crosshairSize: across, in its 640x480 screen's pixels (24 is Quake 3's own).", "24", 8.0, 64.0},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "crosshair_transparency", "Crosshair transparency",
     "Percent: 0 is solid, 100 gone (50 by default).", "50", 0.0, 100.0},
    {BB_OPT_HEADING, nullptr, "Gun"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "draw_gun", "Draw the gun",
     "The gun in your hand at the lower right of the view, as Quake 3's cg_drawGun.", "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "muzzle_flash", "Muzzle flash",
     "Each gun's own flash and smoke at the muzzle as it fires (some are bright). Off by default.", "false"},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "gun_bob", "Gun bob",
     "How much the gun sways as you run, drifts when still, dips when you land and kicks when it fires, times this (1 "
     "is Quake 3's own, 0 holds it still).",
     "1.0", 0.0, 2.0},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "gun_fov", "Gun field of view",
     "The field of view the gun is drawn with, as a viewmodel fov: lower draws it bigger and closer in, higher smaller "
     "and further out; the world keeps its own. Degrees across a 4:3 picture, as the field of view.",
     "90", 40.0, 140.0},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "gun_x", "Gun forward", "Quake 3's cg_gun_x: units the gun sits forward of where it is held.",
     "0", -40.0, 40.0},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "gun_y", "Gun left", "Quake 3's cg_gun_y: units to the left.", "0", -40.0, 40.0},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "gun_z", "Gun up", "Quake 3's cg_gun_z: units up.", "0", -40.0, 40.0},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "shotgun_spread", "Shotgun spread",
     "Quake 3's DEFAULT_SHOTGUN_SPREAD: each pellet lands at a random point of a square this many units either way at "
     "8192 in front (700 is Quake 3's own, about 5 degrees; 2000 is about the Blunderbuss's own width).",
     "700", 100.0, 2000.0},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "plasma", "Plasma",
     "What the plasma gun fires: one of A Call Beyond's small stars, flying straight and seeking nothing, or Blacksky "
     "Eye's shard.",
     "A Call Beyond", 0, 0, "A Call Beyond|Blacksky Eye"},
    {BB_OPT_END}};

namespace {

bb::Plugin g;

// ---- vectors, the game's axes (y up) ---------------------------------------

struct V3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
};
V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
float length(V3 a) { return std::sqrt(dot(a, a)); }
float normalize(V3& a) {
    const float l = length(a);
    if (l > 0.0f) a = a * (1.0f / l);
    return l;
}

constexpr float kPi = 3.14159265f;
constexpr float kDeg = kPi / 180.0f;

// ---- units and Quake 3's constants ------------------------------------------

constexpr float kUnit = 1.0f / 32.0f;  // metres a Quake unit
constexpr float kGravity = 800.0f;  // DEFAULT_GRAVITY: the missiles' (the player's is g_gravity)
constexpr float kStepSize = 18.0f;
// Quake 3 stands on nothing steeper than 45.6 degrees (MIN_WALK_NORMAL 0.7);
// the map the characters walk on (its lo-hit) smooths every staircase into one
// ramp, 50 to 56 degrees up the streets of Yharnam, which the hunter climbs.
// So the body stands on what the hunter stands on: up to 60 degrees.
constexpr float kMinWalkNormal = 0.5f;
constexpr float kOverclip = 1.001f;
constexpr float kDuckScale = 0.25f;
constexpr float kStandHeight = 56.0f, kCrouchHeight = 40.0f;  // feet to the top of the head
constexpr int kStandEye = 50, kCrouchEye = 36;                // feet to the eye: Quake 3's view heights over its origin
constexpr float kRadius = 0.3f;                               // the body's, metres (Quake 3's box is 0.47 m across)
constexpr float kSkin = 0.01f;                                // what a move stops short of a surface by, metres
constexpr double kMaxStep = 1.0 / 125.0;                      // the longest physics step
constexpr double kMaxFrame = 0.05;                            // a hitch is not a run-up
constexpr double kDoubleJumpSeconds = 0.4;                    // CPM's window
constexpr float kAxisSnap = 0.3f;                             // an axis held under this counts as untouched

// cg_view.c's and cg_weapons.c's.
constexpr double kDuckTime = 100.0, kStepTime = 200.0, kLandDeflectTime = 150.0, kLandReturnTime = 300.0;
constexpr double kDamageDeflectTime = 100.0, kDamageReturnTime = 400.0;
constexpr float kMaxStepChange = 32.0f;
constexpr float kBobUp = 0.005f, kBobPitch = 0.002f, kBobRoll = 0.002f, kRunPitch = 0.002f, kRunRoll = 0.005f;

// ---- the settings ------------------------------------------------------------

enum class Physics { Vq3, Cpm, Qw, Hybrid };

// The presets: bg_pmove.c, bg_promode.c's CPM, QuakeWorld's pmove.c, and a
// Hybrid of QuakeWorld's ground and strafe with Quake 3's and CPM's air.
struct Tuning {
    float max_speed, accel, air_accel, friction, stop_speed, jump, air_stop_accel, strafe_accel, air_wish_cap, air_control,
        double_jump;
};
constexpr Tuning kVq3 = {320, 10, 1, 6, 100, 270, 1, 1, 400, 0, 0};
constexpr Tuning kCpm = {320, 15, 1, 8, 100, 270, 2.5f, 70, 30, 150, 100};
constexpr Tuning kQw = {320, 10, 10, 4, 100, 270, 1, 1, 30, 0, 0};
constexpr Tuning kHybrid = {320, 10, 1, 4, 100, 270, 2.5f, 10, 30, 150, 100};

struct Settings {
    bool enabled = true;
    Physics physics = Physics::Vq3;
    Tuning t = kVq3;
    bool defrag = false;  // DeFRaG: steps of exactly 8 ms and the velocity rounded after each, as Quake 3 does at 125 fps
    bool auto_hop = true;
    bool crouch = true;
    bool sounds = true;
    float gravity = 800.0f;
    float knockback = 1000.0f;
    float sensitivity = 5.0f;
    bool invert = false;
    float fov = 90.0f;
    float bob = 1.0f;
    bool crosshair = true;
    float crosshair_size = 24.0f, crosshair_alpha = 0.5f;
    bool draw_gun = true;
    bool muzzle_flash = false;
    float gun_bob = 1.0f;
    float gun_fov = 90.0f;
    float gun_x = 0.0f, gun_y = 0.0f, gun_z = 0.0f;
    float shotgun_spread = 700.0f;  // DEFAULT_SHOTGUN_SPREAD, units either way at 8192
    int plasma = 0;  // kPlasmaLooks: A Call Beyond's star, Blacksky Eye's shard
} s;

void read_options() {
    s.enabled = g.config_bool("quake.enabled", true);
    if (const char* e = std::getenv("QUAKE_OFF"); e && e[0] == '1') s.enabled = false;  // a test: the plugin passive
    // The choice's words, and the short names an older config holds.
    const std::string p = g.config("quake.physics", "Quake 3 (VQ3)");
    s.defrag = p.rfind("DeFRaG", 0) == 0 || p == "defrag";
    s.physics = p == "CPMA" || p == "DeFRaG (CPM)" || p == "cpm"   ? Physics::Cpm
                : p == "Quake 1 + CPMA" || p == "hybrid"          ? Physics::Hybrid
                : p == "QuakeWorld" || p == "qw"                  ? Physics::Qw
                                                                  : Physics::Vq3;
    s.t = s.physics == Physics::Vq3 ? kVq3 : s.physics == Physics::Cpm ? kCpm : s.physics == Physics::Qw ? kQw : kHybrid;
    s.auto_hop = g.config_bool("quake.auto_hop", true);
    s.crouch = g.config_bool("quake.crouch", true);
    s.sounds = g.config_bool("quake.sounds", true);
    s.gravity = std::clamp(static_cast<float>(g.config_number("quake.gravity", 800.0)), 100.0f, 1600.0f);
    s.knockback = std::clamp(static_cast<float>(g.config_number("quake.knockback", 1000.0)), 0.0f, 3000.0f);
    s.sensitivity = std::clamp(static_cast<float>(g.config_number("quake.sensitivity", 5.0)), 0.01f, 100.0f);
    s.invert = g.config_bool("quake.invert", false);
    s.fov = std::clamp(static_cast<float>(g.config_number("quake.fov", 90.0)), 50.0f, 140.0f);
    s.bob = std::clamp(static_cast<float>(g.config_number("quake.bob", 1.0)), 0.0f, 2.0f);
    s.crosshair = g.config_bool("quake.crosshair", true);
    s.crosshair_size = std::clamp(static_cast<float>(g.config_number("quake.crosshair_size", 24.0)), 8.0f, 64.0f);
    s.crosshair_alpha = 1.0f - std::clamp(static_cast<float>(g.config_number("quake.crosshair_transparency", 50.0)), 0.0f, 100.0f) / 100.0f;
    s.draw_gun = g.config_bool("quake.draw_gun", true);
    s.muzzle_flash = g.config_bool("quake.muzzle_flash", false);
    s.gun_bob = std::clamp(static_cast<float>(g.config_number("quake.gun_bob", 1.0)), 0.0f, 2.0f);
    s.gun_fov = std::clamp(static_cast<float>(g.config_number("quake.gun_fov", 90.0)), 40.0f, 140.0f);
    s.gun_x = std::clamp(static_cast<float>(g.config_number("quake.gun_x", 0.0)), -40.0f, 40.0f);
    s.gun_y = std::clamp(static_cast<float>(g.config_number("quake.gun_y", 0.0)), -40.0f, 40.0f);
    s.gun_z = std::clamp(static_cast<float>(g.config_number("quake.gun_z", 0.0)), -40.0f, 40.0f);
    s.plasma = g.config("quake.plasma", "A Call Beyond") == "Blacksky Eye" ? 1 : 0;
    s.shotgun_spread = std::clamp(static_cast<float>(g.config_number("quake.shotgun_spread", 700.0)), 100.0f, 2000.0f);
}

const char* physics_name() {
    switch (s.physics) {
    case Physics::Vq3: return s.defrag ? "DeFRaG (VQ3)" : "Quake 3 (VQ3)";
    case Physics::Cpm: return s.defrag ? "DeFRaG (CPM)" : "CPMA";
    case Physics::Qw: return "QuakeWorld";
    default: return "Quake 1 + CPMA";
    }
}

// BBHOST-style switches for testing (docs/plugins.md, "Quake"):
// QUAKE_LOG=1 logs the movement four times a second; QUAKE_TEST=<script>
// drives it without a player (below); QUAKE_PLACE=adopt lets the game move the
// player between frames and moves on from where it left them.
const bool g_log = [] {
    const char* e = std::getenv("QUAKE_LOG");
    return e && e[0] == '1';
}();
const bool g_adopt = [] {
    const char* e = std::getenv("QUAKE_PLACE");
    return e && std::strcmp(e, "adopt") == 0;
}();
const bool g_probe = [] {
    const char* e = std::getenv("QUAKE_PROBE");
    return e && e[0] == '1';
}();
// QUAKE_SOUNDS=1 logs every sound the game starts (its name and where).
const bool g_sounds = [] {
    const char* e = std::getenv("QUAKE_SOUNDS");
    return e && e[0] == '1';
}();

// ---- the game ------------------------------------------------------------------

constexpr std::uint64_t kWorldChrMan = 0x593e878;  // WorldChrMan*: +0x60 the main player's ChrIns
constexpr std::uint64_t kPhysicsSystem = 0x593d700;  // +0x28 the collision world the camera casts in
constexpr std::uint64_t kSphereCast = 0x1c090e0;     // (world, filter, from, delta, hit, normal, radius, fraction*) -> hit
// The casts' collision filters (FROM's Havok group filter info: the layer in
// the low five bits, 0x20 keeping the caster's own bodies out of it). The
// follow camera's, 0x25, sees the map's solid geometry and nothing only
// characters meet. The body casts the characters' own, 0x29 (layer 9): the
// one the game's teleport checks a character's new place with (sub_1d39c80's
// cast at 0x1d3a0aa) and the characters' other casts pass (0x1c07bd0) - so it
// meets what the hunter meets and nothing else: the invisible walls along
// ledges and the lamp posts, the floors and stairs the hunter walks - and
// passes what the hunter passes. As Quake 3's playerclip stops players and not
// missiles, the shots and blasts keep the camera's.
constexpr std::uint32_t kCastFilter = 0x25;  // the follow camera's: the shots, the blasts' line of sight
constexpr std::uint32_t kBodyFilter = 0x29;  // the characters': the player's body
std::uint32_t g_body_filter = kBodyFilter;   // QUAKE_FILTER=<hex> (a test) casts another

template <typename T>
T& at(std::uint8_t* p, std::size_t off) {
    return *reinterpret_cast<T*>(p + off);
}

// A global of the image's, by its Binary Ninja address (read() moves it by the
// image's slide).
std::uint8_t* ptr_at(std::uint64_t bn) {
    std::uint64_t v = 0;
    return g.api->read(bn, &v, 8) == 0 ? reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(v)) : nullptr;
}

std::uint8_t* player() {
    std::uint8_t* man = ptr_at(kWorldChrMan);
    return man ? at<std::uint8_t*>(man, 0x60) : nullptr;
}

// ChrIns -> modules (+0x3b0) -> the physics module (+0x68): rotation +0x1d0
// (pitch, yaw, roll), position +0x1e0.
std::uint8_t* physics_of(std::uint8_t* chr) {
    std::uint8_t* modules = chr ? at<std::uint8_t*>(chr, 0x3b0) : nullptr;
    return modules ? at<std::uint8_t*>(modules, 0x68) : nullptr;
}

// The ladder's state (the debug display's "Ladder state", sub_1d23e00): a
// byte of the character's action module, ChrIns -> ChrCtrl +0x58 -> +0x38 -> +0x60.
int ladder_state(std::uint8_t* chr) {
    std::uint8_t* ctrl = chr ? at<std::uint8_t*>(chr, 0x58) : nullptr;
    std::uint8_t* act = ctrl ? at<std::uint8_t*>(ctrl, 0x38) : nullptr;
    return act ? at<std::int8_t>(act, 0x60) : -128;
}

std::uint8_t* collision_world() {
    std::uint8_t* sys = ptr_at(kPhysicsSystem);
    return sys ? at<std::uint8_t*>(sys, 0x28) : nullptr;
}

V3 read_position(std::uint8_t* phys) {
    const float* p = &at<float>(phys, 0x1e0);
    return {p[0], p[1], p[2]};
}

// The player where the movement left them, as the game's teleport puts a
// character (sub_1d39c80): the position and its copy, the byte that makes the
// next physics step place the character proxy there instead of moving it, and
// the proxy's own transform (w 1).
void place(std::uint8_t* phys, V3 p) {
    float* a = &at<float>(phys, 0x1e0);
    float* b = &at<float>(phys, 0x1f0);
    a[0] = b[0] = p.x;
    a[1] = b[1] = p.y;
    a[2] = b[2] = p.z;
    if (g_adopt) return;
    at<std::uint8_t>(phys, 0x210) = 1;
    at<std::uint8_t>(phys, 0x32a) = 1;
    if (std::uint8_t* holder = at<std::uint8_t*>(phys, 0x218)) {
        if (std::uint8_t* body = at<std::uint8_t*>(holder, 0x38)) {
            float* t = &at<float>(body, 0x70);
            t[0] = p.x;
            t[1] = p.y;
            t[2] = p.z;
            t[3] = 1.0f;
        }
    }
    at<std::uint8_t>(phys, 0x211) = 0;
}

// The character faces where the view looks: the game's yaw faces (-sin, 0,
// -cos), the follow camera's input direction as sub_18368b0 takes it from the
// view (atan2 of the negated forward).
void face(std::uint8_t* phys, float yaw) {
    float a = yaw + kPi;
    while (a > kPi) a -= 2.0f * kPi;
    while (a < -kPi) a += 2.0f * kPi;
    at<float>(phys, 0x1d4) = a;
}

// The game's own switches on the player while Quake has it: no gravity
// (ChrIns+0x1e1 bit 6, the debug menu's "No Gravity") and no drawing (+0x1e2
// bit 4). Both are the player's ChrIns alone - the world's gravity, which
// props, breakables, ragdolls and every other character fall by, is never
// touched - and gravity goes back the moment the player dies, so the death
// falls as the game has it.
//
// The Restore Debug Dash patch hooks the character's movement step (a jump at
// 0x1cbd711 over its vmovss [rbp-0x7c], xmm0) and, with the dash off, clears
// that same no-gravity bit on the player every step: the game's gravity then
// holds the hunter to the floor under every jump. While Quake has the player
// the step's own instruction goes back in place of the jump - the dash is
// Left Ctrl, Quake's crouch, anyway - and the hook returns with the player.
constexpr std::uint64_t kDashSite = 0x1cbd711;
constexpr std::uint8_t kDashOwn[5] = {0xc5, 0xfa, 0x11, 0x45, 0x84};
constexpr std::uint8_t kDashHook[5] = {0xe9, 0xce, 0xe1, 0x07, 0x03};
bool g_dash_lifted = false;

void lift_dash(bool lift) {
    if (lift == g_dash_lifted) return;
    if (lift) {
        if (g.api->patch(kDashSite, kDashHook, kDashOwn, sizeof kDashOwn) != 0) return;  // no dash patch
        g.log("quake: the debug dash's hook set aside while Quake has the player (it clears the no-gravity bit)");
    } else if (g.api->patch(kDashSite, kDashOwn, kDashHook, sizeof kDashHook) != 0) {
        g.log("quake: the debug dash's hook could not be put back");
    }
    g_dash_lifted = lift;
}

void hold_body(std::uint8_t* chr, bool weightless, bool hidden) {
    if (!chr) return;
    lift_dash(weightless);
    std::uint8_t& f1 = at<std::uint8_t>(chr, 0x1e1);
    std::uint8_t& f2 = at<std::uint8_t>(chr, 0x1e2);
    if (weightless) f1 |= 0x40;
    else f1 &= static_cast<std::uint8_t>(~0x40);
    if (hidden) f2 &= static_cast<std::uint8_t>(~0x10);
    else f2 |= 0x10;
}

// ---- traces -------------------------------------------------------------------

std::uint64_t g_casts = 0;

// The game's sphere cast: a sphere of `radius` swept from `from` along `delta`
// (metres); on a hit, the fraction of delta it got and the surface's normal.
bool sphere_cast(std::uint8_t* world, V3 from, V3 delta, float radius, float& fraction, V3& normal, std::uint32_t filter = kCastFilter) {
    alignas(16) float f[4] = {from.x, from.y, from.z, 1.0f};
    alignas(16) float d[4] = {delta.x, delta.y, delta.z, 0.0f};
    alignas(16) float hit[4] = {};
    alignas(16) float n[4] = {};
    alignas(16) float frac[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    BbCallRegs r{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(world);
    r.arg[1] = filter;
    r.arg[2] = reinterpret_cast<std::uint64_t>(f);
    r.arg[3] = reinterpret_cast<std::uint64_t>(d);
    r.arg[4] = reinterpret_cast<std::uint64_t>(hit);
    r.arg[5] = reinterpret_cast<std::uint64_t>(n);
    r.xmm[0] = bb_f32(radius);
    r.stack[0] = reinterpret_cast<std::uint64_t>(frac);
    ++g_casts;
    if (g.api->call_guest(kSphereCast, &r) != 0 || (r.ret & 0xff) == 0) return false;
    fraction = std::clamp(frac[0], 0.0f, 1.0f);
    normal = {n[0], n[1], n[2]};  // out of the surface, toward the sphere
    normalize(normal);
    return true;
}

// The body's cast, with the hunter's filter. `sphere` (0 the feet, 1 the
// waist, 2 the head; -1 none) has the hit judged as that part of the body
// meets it (body_trace).
bool body_cast(std::uint8_t* world, V3 from, V3 delta, float radius, float& fraction, V3& normal, int sphere = -1) {
    const float len = length(delta);
    const V3 dir = len > 1e-6f ? delta * (1.0f / len) : V3{};
    float f = 1.0f;
    V3 n;
    if (!sphere_cast(world, from, delta, radius, f, n, g_body_filter)) return false;
    // Quake 3's body is a box, and only its bottom stands on anything: what
    // the waist's and the head's spheres meet under their middles - the top
    // edge of a chest-high wall, a lip beside the body - is the box's side
    // against it, not a slope to ride up and over; and falling past it, the
    // side slides by - where the body comes down is the feet's to find.
    if (sphere > 0 && n.y > 0.0f) {
        V3 flat{n.x, 0.0f, n.z};
        if (dir.y < -0.5f || normalize(flat) < 1e-3f) return false;
        n = flat;
    }
    // A sphere already touching a surface - the floor under the feet, a wall
    // slid along - is stopped by it only when the move goes into it: the cast
    // reports the touch as a hit at the start whichever way the sphere goes.
    if (sphere >= 0 && f <= 0.0f && dot(n, dir) > -0.1f) return false;
    fraction = f;
    normal = n;
    return true;
}

struct Trace {
    float fraction = 1.0f;  // of the move made
    V3 end;                 // where the feet got to
    V3 normal{0.0f, 1.0f, 0.0f};
    bool hit = false;
    bool startsolid = false;  // stuck where it began: no part of the move made
};

// The body, feet at `from`, `height` units tall, moved to `to`: three spheres,
// at the feet, the waist and the head, the first of them to touch stopping it
// kSkin short of the surface. A sphere already touching a surface - the floor
// under the feet, a wall slid along - is stopped by it only when the move goes
// into it: the cast reports the touch as a hit at the start whichever way the
// sphere goes.
Trace body_trace(std::uint8_t* world, V3 from, V3 to, float height) {
    Trace t;
    t.end = to;
    const V3 delta = to - from;
    const float len = length(delta);
    if (len < 1e-6f || !world) return t;
    const V3 dir = delta * (1.0f / len);
    const float h = height * kUnit;
    const float centres[3] = {kRadius, std::max(kRadius, h * 0.5f), std::max(kRadius, h - kRadius)};
    static int logged = 0;
    for (int k = 0; k < 3; ++k) {
        float frac = 1.0f;
        V3 n;
        const bool hit = body_cast(world, from + V3{0.0f, centres[k], 0.0f}, delta, kRadius, frac, n, k);
        if (g_log && hit && frac <= 0.0f && logged < 24 && dir.y > -0.9f) {
            ++logged;
            g.log("quake: sphere %d stuck at the start of a %.3f %.3f %.3f move: normal %.2f %.2f %.2f", k,
                  static_cast<double>(delta.x), static_cast<double>(delta.y), static_cast<double>(delta.z),
                  static_cast<double>(n.x), static_cast<double>(n.y), static_cast<double>(n.z));
        }
        if (hit && frac < t.fraction) {
            t.fraction = frac;
            t.normal = n;
            t.hit = true;
        }
    }
    if (!t.hit) return t;
    // Quake 3's box meets a stair's edge with its flat side and steps up it
    // (PM_StepSlideMove); a sphere meets it on its curve, with a normal that
    // would turn the run into a hop. Below a walkable slope, a horizontal
    // move is stopped as by a riser.
    if (std::fabs(dir.y) < 0.5f && t.normal.y > 0.0f && t.normal.y < kMinWalkNormal) {
        V3 flat{t.normal.x, 0.0f, t.normal.z};
        if (normalize(flat) > 1e-3f) t.normal = flat;
    }
    const float made = std::max(0.0f, t.fraction * len - kSkin);
    t.fraction = made / len;
    t.end = from + delta * t.fraction;
    return t;
}

// ---- bg_pmove.c -----------------------------------------------------------------

struct Cmd {
    float forward = 0.0f, side = 0.0f;  // -1..1
    bool jump = false, crouch = false;
    bool fire = false;
};

// Quake 3's player state and the frame's pml, in the game's axes: the feet in
// metres, the velocity in units a second.
struct Pm {
    V3 origin, velocity;
    bool on_ground = false;  // groundEntityNum != ENTITYNUM_NONE
    bool ground_plane = false, walking = false;
    V3 ground_normal{0.0f, 1.0f, 0.0f};
    bool ducked = false;
    bool jump_held = true;  // PMF_JUMP_HELD: a jump held through a load is not a press
    double last_jump = -1e9;
    // The last place it stood, and how long it has been in the air: a body
    // that a missed cast let into the map's solid falls forever, and goes back.
    V3 last_ground;
    double air_time = 0.0;
    float bob_cycle = 0.0f;  // bobCycle, 0..256
    int eye = kStandEye;     // the view height over the feet
    // Events for the view (cgame's): a landing's dip, a step up's height.
    float land_change = 0.0f;
    bool landed = false;
    float takeoff_y = 0.0f, peak_y = 0.0f;  // QUAKE_LOG: how high each time in the air rose
    float stepped = 0.0f;
    // Events for the sounds: a jump, and how hard the last landing was (0
    // none, 1 a footstep's, 2 EV_FALL_SHORT, 3 EV_FALL_MEDIUM, 4 EV_FALL_FAR).
    bool jumped = false;
    int land = 0;
    bool footstep = false;  // EV_FOOTSTEP: the bob cycle passed a half while running
    // PMF_TIME_KNOCKBACK's pm_time: thrown by a blast, the ground neither
    // brakes nor steers (no friction, the air's acceleration) for this long.
    double knock_time = 0.0;
} pm;

double g_now = 0.0;  // seconds, the movement's clock

float body_height() { return pm.ducked ? kCrouchHeight : kStandHeight; }

void clip_velocity(V3& v, V3 n, float overbounce) {
    float backoff = dot(v, n);
    backoff = backoff < 0.0f ? backoff * overbounce : backoff / overbounce;
    v = v - n * backoff;
}

// PM_Friction, on the ground: the horizontal speed loses
// max(speed, stopspeed) * friction a second.
void friction(float dt) {
    V3 v = pm.velocity;
    if (pm.walking) v.y = 0.0f;
    const float speed = length(v);
    if (speed < 1.0f) {
        pm.velocity.x = pm.velocity.z = 0.0f;
        return;
    }
    float drop = 0.0f;
    // Getting knocked back, no friction.
    if (pm.walking && pm.knock_time <= 0.0) drop = std::max(speed, s.t.stop_speed) * s.t.friction * dt;
    const float k = std::max(0.0f, speed - drop) / speed;
    pm.velocity = pm.velocity * k;
}

// PM_Accelerate: only the speed along the wish direction is capped, which is
// why turning into a strafe keeps adding speed.
void accelerate(V3 wishdir, float wishspeed, float accel, float dt) {
    const float add = wishspeed - dot(pm.velocity, wishdir);
    if (add <= 0.0f) return;
    pm.velocity = pm.velocity + wishdir * std::min(add, accel * dt * wishspeed);
}

// QuakeWorld's air accelerate: the speed along the wish direction is capped
// at the wish cap (30), the step taken at the full wish speed.
void qw_accelerate(V3 wishdir, float wishspeed, float accel, float cap, float dt) {
    const float add = std::min(wishspeed, cap) - dot(pm.velocity, wishdir);
    if (add <= 0.0f) return;
    pm.velocity = pm.velocity + wishdir * std::min(add, accel * wishspeed * dt);
}

// CPM's air control: the velocity turned toward the wish direction, its
// speed kept, harder the nearer it already points there.
void air_steer(V3 wishdir, float wishspeed, float dt) {
    if (wishspeed == 0.0f) return;
    V3 h{pm.velocity.x, 0.0f, pm.velocity.z};
    const float speed = normalize(h);
    if (speed < 1e-4f) return;
    const float d = dot(h, wishdir);
    if (d > 0.0f) {
        const float k = 32.0f * s.t.air_control * d * d * dt;
        h = h * speed + wishdir * k;
        normalize(h);
    }
    pm.velocity.x = h.x * speed;
    pm.velocity.z = h.z * speed;
}

bool alone(float axis, float other) { return std::fabs(axis) >= kAxisSnap && std::fabs(other) < kAxisSnap; }

void air_accelerate(V3 wishdir, float wishspeed, const Cmd& cmd, float dt) {
    switch (s.physics) {
    case Physics::Cpm: {
        const float wishspeed2 = wishspeed;
        float accel = dot(pm.velocity, wishdir) < 0.0f ? s.t.air_stop_accel : s.t.air_accel;
        if (alone(cmd.side, cmd.forward)) {
            wishspeed = std::min(wishspeed, s.t.air_wish_cap);
            accel = s.t.strafe_accel;
        }
        accelerate(wishdir, wishspeed, accel, dt);
        if (s.t.air_control > 0.0f && alone(cmd.forward, cmd.side)) air_steer(wishdir, wishspeed2, dt);
        break;
    }
    case Physics::Qw:
        qw_accelerate(wishdir, wishspeed, s.t.air_accel, s.t.air_wish_cap, dt);
        break;
    case Physics::Hybrid: {
        if (alone(cmd.side, cmd.forward)) {
            qw_accelerate(wishdir, wishspeed, s.t.strafe_accel, s.t.air_wish_cap, dt);
            break;
        }
        const float wishspeed2 = wishspeed;
        const float accel = dot(pm.velocity, wishdir) < 0.0f ? s.t.air_stop_accel : s.t.air_accel;
        accelerate(wishdir, wishspeed, accel, dt);
        if (s.t.air_control > 0.0f && alone(cmd.forward, cmd.side)) air_steer(wishdir, wishspeed2, dt);
        break;
    }
    default:
        accelerate(wishdir, wishspeed, s.t.air_accel, dt);
        break;
    }
}

// PM_SlideMove: the move, clipped against what it meets, up to four planes.
bool slide_move(std::uint8_t* world, bool gravity, float dt) {
    constexpr int kMaxPlanes = 5;
    V3 planes[kMaxPlanes];
    int numplanes = 0;
    V3 primal = pm.velocity, end_velocity = pm.velocity;
    if (gravity) {
        end_velocity.y -= s.gravity * dt;
        pm.velocity.y = (pm.velocity.y + end_velocity.y) * 0.5f;
        primal.y = end_velocity.y;
        if (pm.ground_plane) clip_velocity(pm.velocity, pm.ground_normal, kOverclip);
    }
    float time_left = dt;
    if (pm.ground_plane) planes[numplanes++] = pm.ground_normal;
    V3 v = pm.velocity;
    normalize(v);
    planes[numplanes++] = v;
    int bump = 0;
    for (; bump < 4; ++bump) {
        const V3 to = pm.origin + pm.velocity * (time_left * kUnit);
        const Trace tr = body_trace(world, pm.origin, to, body_height());
        if (tr.startsolid) {
            pm.velocity.y = 0.0f;  // trapped: no falling builds up, sideways is still allowed
            return true;
        }
        if (tr.fraction > 0.0f) pm.origin = tr.end;
        if (!tr.hit) break;
        time_left -= time_left * tr.fraction;
        if (numplanes >= kMaxPlanes) {
            pm.velocity = {};
            return true;
        }
        // The same plane again: nudge the velocity out along it.
        int i = 0;
        for (; i < numplanes; ++i) {
            if (dot(tr.normal, planes[i]) > 0.99f) {
                pm.velocity = pm.velocity + tr.normal;
                break;
            }
        }
        if (i < numplanes) continue;
        planes[numplanes++] = tr.normal;
        for (i = 0; i < numplanes; ++i) {
            if (dot(pm.velocity, planes[i]) >= 0.1f) continue;
            V3 clip = pm.velocity, end_clip = end_velocity;
            clip_velocity(clip, planes[i], kOverclip);
            clip_velocity(end_clip, planes[i], kOverclip);
            bool stop = false;
            for (int j = 0; j < numplanes && !stop; ++j) {
                if (j == i || dot(clip, planes[j]) >= 0.1f) continue;
                clip_velocity(clip, planes[j], kOverclip);
                clip_velocity(end_clip, planes[j], kOverclip);
                if (dot(clip, planes[i]) >= 0.0f) continue;
                // Along the crease of the two.
                V3 dir = cross(planes[i], planes[j]);
                normalize(dir);
                clip = dir * dot(dir, pm.velocity);
                end_clip = dir * dot(dir, end_velocity);
                for (int k = 0; k < numplanes; ++k) {
                    if (k == i || k == j || dot(clip, planes[k]) >= 0.1f) continue;
                    stop = true;  // a third plane: stop dead
                    break;
                }
            }
            if (stop) {
                pm.velocity = {};
                return true;
            }
            pm.velocity = clip;
            end_velocity = end_clip;
            break;
        }
    }
    if (gravity) pm.velocity = end_velocity;
    return bump != 0;
}

// PM_StepSlideMove: a blocked move tried again a step higher, then pushed
// back down; a step up shows in the view as cgame's EV_STEP smoothing.
void step_slide_move(std::uint8_t* world, bool gravity, float dt) {
    const V3 start_o = pm.origin, start_v = pm.velocity;
    if (!slide_move(world, gravity, dt)) return;
    const V3 slid_o = pm.origin, slid_v = pm.velocity;
    const float h = body_height();
    Trace down = body_trace(world, start_o, start_o - V3{0.0f, kStepSize * kUnit, 0.0f}, h);
    // Never step up with velocity still going up.
    if (pm.velocity.y > 0.0f && (!down.hit || down.normal.y < kMinWalkNormal)) return;
    const Trace up = body_trace(world, start_o, start_o + V3{0.0f, kStepSize * kUnit, 0.0f}, h);
    if (up.startsolid) return;
    const float step = up.end.y - start_o.y;
    pm.origin = up.end;
    pm.velocity = start_v;
    slide_move(world, gravity, dt);
    down = body_trace(world, pm.origin, pm.origin - V3{0.0f, step, 0.0f}, h);
    // A step that comes down on something too steep to stand on is no step:
    // Quake 3's box cannot rest against a wall's face, a sphere wedged on it
    // can - and from there the next step is over a chest-high invisible wall.
    if (down.hit && down.normal.y < kMinWalkNormal) {
        pm.origin = slid_o;
        pm.velocity = slid_v;
        return;
    }
    if (!down.startsolid) pm.origin = down.end;
    // Clipped against what it came down on - a stair's top, not the curve of
    // the sphere over its edge, which would throw the run upward.
    if (down.hit) {
        if (down.normal.y >= 0.95f) clip_velocity(pm.velocity, down.normal, kOverclip);
        else pm.velocity.y = std::min(pm.velocity.y, 0.0f);
    }
    const float delta = (pm.origin.y - start_o.y) / kUnit;
    if (delta > 2.0f) pm.stepped += delta < 7.0f ? 4.0f : delta < 11.0f ? 8.0f : delta < 15.0f ? 12.0f : 16.0f;
}

// PM_CrashLand: how hard the landing was, as the view's dip.
void crash_land(float previous_vy) {
    // The velocity at the moment of landing; delta as Quake 3 squares it.
    const float delta = previous_vy * previous_vy * 0.0001f * (pm.ducked ? 2.0f : 1.0f);
    if (delta < 1.0f) return;
    float change = 0.0f;
    if (delta > 60.0f) change = -24.0f, pm.land = 4;       // EV_FALL_FAR
    else if (delta > 40.0f) change = -16.0f, pm.land = 3;  // EV_FALL_MEDIUM
    else if (delta > 7.0f) change = -8.0f, pm.land = 2;    // EV_FALL_SHORT
    else pm.land = std::max(pm.land, 1);                   // EV_FOOTSTEP
    if (change != 0.0f) {
        pm.land_change = change;
        pm.landed = true;
    }
    pm.bob_cycle = 0.0f;
}

// PM_GroundTrace: on the ground when a short trace down meets a surface that
// is walkable.
void ground_trace(std::uint8_t* world, float previous_vy) {
    const Trace tr = body_trace(world, pm.origin, pm.origin - V3{0.0f, 0.25f * kUnit + kSkin, 0.0f}, body_height());
    if (!tr.hit) {
        pm.on_ground = pm.ground_plane = pm.walking = false;
        return;
    }
    // What the body stands on, as a box's flat bottom would find it: straight
    // under its middle, not where the sphere's curve touches an edge.
    V3 normal = tr.normal;
    float frac = 1.0f;
    V3 under;
    if (body_cast(world, pm.origin + V3{0.0f, kRadius, 0.0f}, V3{0.0f, -(kRadius + 0.1f), 0.0f}, 0.02f, frac, under) &&
        under.y >= kMinWalkNormal) {
        normal = under;
    }
    pm.ground_normal = normal;
    // Thrown off the ground.
    if (pm.velocity.y > 0.0f && dot(pm.velocity, normal) > 10.0f) {
        pm.on_ground = pm.ground_plane = pm.walking = false;
        return;
    }
    if (normal.y < kMinWalkNormal) {
        pm.on_ground = false;
        pm.ground_plane = true;
        pm.walking = false;
        return;
    }
    pm.ground_plane = pm.walking = true;
    if (!pm.on_ground) crash_land(previous_vy);
    pm.on_ground = true;
}

// PM_CheckDuck: down while the key is held, up again once there is room.
void check_duck(std::uint8_t* world, const Cmd& cmd) {
    if (cmd.crouch && s.crouch) {
        pm.ducked = true;
    } else if (pm.ducked) {
        const float top = kCrouchHeight * kUnit - kRadius;
        float frac = 1.0f;
        V3 n;
        const bool blocked = body_cast(world, pm.origin + V3{0.0f, top, 0.0f},
                                         V3{0.0f, (kStandHeight - kCrouchHeight) * kUnit + kSkin, 0.0f}, kRadius, frac, n);
        if (!blocked) pm.ducked = false;
    }
    pm.eye = pm.ducked ? kCrouchEye : kStandEye;
}

// PM_CheckJump, with CPM's double jump: a jump within 0.4 s of the last goes
// higher.
bool check_jump(const Cmd& cmd) {
    if (!cmd.jump) return false;
    if (pm.jump_held && !s.auto_hop) return false;
    pm.ground_plane = pm.walking = pm.on_ground = false;
    pm.jump_held = true;
    float v = s.t.jump;
    if (s.t.double_jump > 0.0f && g_now - pm.last_jump < kDoubleJumpSeconds) v += s.t.double_jump;
    pm.last_jump = g_now;
    pm.velocity.y = v;
    pm.jumped = true;  // EV_JUMP
    return true;
}

// PM_Footsteps: the bob cycle, which only the ground advances.
void footsteps(const Cmd& cmd, float dt) {
    const float xyspeed = std::sqrt(pm.velocity.x * pm.velocity.x + pm.velocity.z * pm.velocity.z);
    if (!pm.on_ground) return;
    if (cmd.forward == 0.0f && cmd.side == 0.0f) {
        if (xyspeed < 5.0f) pm.bob_cycle = 0.0f;
        return;
    }
    const float bobmove = pm.ducked ? 0.5f : 0.4f;
    const int old = static_cast<int>(pm.bob_cycle);
    pm.bob_cycle = std::fmod(pm.bob_cycle + bobmove * dt * 1000.0f, 256.0f);
    // A step each time the cycle crosses a half; crouched, no sound.
    if (((old + 64) ^ (static_cast<int>(pm.bob_cycle) + 64)) & 128) {
        if (!pm.ducked) pm.footstep = true;
    }
}

// One step of Quake 3's PmoveSingle, the yaw the view had.
void pmove_step(std::uint8_t* world, const Cmd& cmd, float yaw, float dt) {
    if (!cmd.jump) pm.jump_held = false;
    const float previous_vy = pm.velocity.y;
    check_duck(world, cmd);
    ground_trace(world, previous_vy);
    if (pm.knock_time > 0.0) pm.knock_time = std::max(0.0, pm.knock_time - dt);  // PM_DropTimers

    const V3 forward{std::sin(yaw), 0.0f, std::cos(yaw)}, right{std::cos(yaw), 0.0f, -std::sin(yaw)};
    auto wish = [&](V3 f, V3 r, float& speed) {
        V3 w = f * cmd.forward + r * cmd.side;
        const float l = normalize(w);
        // PM_CmdScale's speed, by the larger of the two axes: a full
        // deflection either way is the full speed, so a diagonal is no faster.
        speed = l > 1e-4f ? s.t.max_speed * std::min(1.0f, std::max(std::fabs(cmd.forward), std::fabs(cmd.side))) : 0.0f;
        return w;
    };

    if (pm.walking && check_jump(cmd)) {
        // PM_AirMove from here: no friction on the way through, which is the
        // whole of bunny hopping.
    }
    if (pm.walking) {
        friction(dt);
        // The moves projected onto the ground.
        V3 f = forward, r = right;
        clip_velocity(f, pm.ground_normal, kOverclip);
        clip_velocity(r, pm.ground_normal, kOverclip);
        normalize(f);
        normalize(r);
        float wishspeed = 0.0f;
        const V3 wishdir = wish(f, r, wishspeed);
        if (pm.ducked) wishspeed = std::min(wishspeed, s.t.max_speed * kDuckScale);
        // Hit by a blast, the player is not in full control for a moment:
        // the air's acceleration, and gravity, on the ground.
        const bool knocked = pm.knock_time > 0.0;
        accelerate(wishdir, wishspeed, knocked ? s.t.air_accel : s.t.accel, dt);
        if (knocked) pm.velocity.y -= s.gravity * dt;
        const float vel = length(pm.velocity);
        clip_velocity(pm.velocity, pm.ground_normal, kOverclip);
        // Up or down a slope keeps its speed.
        normalize(pm.velocity);
        pm.velocity = pm.velocity * vel;
        if (pm.velocity.x != 0.0f || pm.velocity.z != 0.0f) step_slide_move(world, false, dt);
    } else {
        float wishspeed = 0.0f;
        const V3 wishdir = wish(forward, right, wishspeed);
        air_accelerate(wishdir, wishspeed, cmd, dt);
        // A ground too steep to stand on is slid along.
        if (pm.ground_plane) clip_velocity(pm.velocity, pm.ground_normal, kOverclip);
        step_slide_move(world, true, dt);
    }
    ground_trace(world, previous_vy);
    footsteps(cmd, dt);
    if (g_log) {
        if (!pm.on_ground) {
            pm.peak_y = std::max(pm.peak_y, pm.origin.y);
        } else if (pm.peak_y > pm.takeoff_y + 0.05f) {
            g.log("quake: in the air up to %.0f units over where it left the ground", static_cast<double>((pm.peak_y - pm.takeoff_y) / kUnit));
            pm.peak_y = pm.takeoff_y = pm.origin.y;
        } else {
            pm.peak_y = pm.takeoff_y = pm.origin.y;
        }
    }
    if (pm.on_ground) {
        pm.last_ground = pm.origin;
        pm.air_time = 0.0;
    } else if ((pm.air_time += dt) > 4.0 && pm.velocity.y < -2000.0f) {
        g.log("quake: four seconds falling at %.0f units a second with nothing under - back to %.2f %.2f %.2f",
              static_cast<double>(-pm.velocity.y), static_cast<double>(pm.last_ground.x),
              static_cast<double>(pm.last_ground.y), static_cast<double>(pm.last_ground.z));
        pm.origin = pm.last_ground;
        pm.velocity = {};
        pm.air_time = 0.0;
    }
}

// ---- the view: cg_view.c ----------------------------------------------------------

struct View {
    float yaw = 0.0f, pitch = 0.0f;  // radians; yaw 0 looks along +z and grows to the right, pitch up
    bool primed = false;
    double ms = 0.0;  // cg.time
    float land_change = 0.0f;
    double land_time = -1e9;
    float step_change = 0.0f;
    double step_time = -1e9;
    float duck_change = 0.0f;
    double duck_time = -1e9;
    int eye = kStandEye;
    double damage_time = -1e9;
    float dmg_pitch = 0.0f, dmg_roll = 0.0f;
    std::int64_t hp = -1;
} view;

// CG_DamageFeedback with no direction (Bloodborne's hits do not say where
// they came from): the view kicks up by 5 to 10 degrees, more the lower the
// health.
void damage_feedback(std::int64_t damage, std::int64_t hp, std::int64_t max_hp) {
    if (max_hp <= 0 || damage <= 0) return;
    const float points = 100.0f * static_cast<float>(damage) / static_cast<float>(max_hp);
    const float health = 100.0f * static_cast<float>(hp) / static_cast<float>(max_hp);
    const float scale = health < 40.0f ? 1.0f : 40.0f / health;
    view.dmg_pitch = -std::clamp(points * scale, 5.0f, 10.0f);
    view.dmg_roll = 0.0f;
    view.damage_time = view.ms;
}

// Where the eye is and how the view turns: CG_OffsetFirstPersonView, in
// Quake 3's angles (degrees; pitch down and roll to the right positive) and
// units over the feet.
void place_view(BbCamera& cam, bool dead, V3 origin, float& out_yaw, float& out_pitch, float& out_roll) {
    const float ms = static_cast<float>(view.ms);
    float pitch = 0.0f, roll = 0.0f, up = 0.0f;
    const float b = s.bob;
    const float xyspeed = std::sqrt(pm.velocity.x * pm.velocity.x + pm.velocity.z * pm.velocity.z);
    const int cycle = static_cast<int>(pm.bob_cycle) & 255;
    const bool odd = (cycle & 128) != 0;
    const float fracsin = std::fabs(std::sin(static_cast<float>(cycle & 127) / 127.0f * kPi));

    if (dead) {
        // Quake 3's dead view: on the floor, rolled over.
        roll = 40.0f;
        pitch = -15.0f;
        up = 8.0f;
    } else {
        // The damage kick.
        float ratio = static_cast<float>(view.ms - view.damage_time);
        if (ratio < kDamageDeflectTime) {
            ratio /= static_cast<float>(kDamageDeflectTime);
            pitch += ratio * view.dmg_pitch;
            roll += ratio * view.dmg_roll;
        } else {
            ratio = 1.0f - (ratio - static_cast<float>(kDamageDeflectTime)) / static_cast<float>(kDamageReturnTime);
            if (ratio > 0.0f) {
                pitch += ratio * view.dmg_pitch;
                roll += ratio * view.dmg_roll;
            }
        }
        // The velocity: forward pitches the view, sideways rolls it.
        const float cy = std::cos(view.yaw), sy = std::sin(view.yaw), cp = std::cos(view.pitch), sp = std::sin(view.pitch);
        const V3 forward{sy * cp, sp, cy * cp}, left{-cy, 0.0f, sy};
        pitch += dot(pm.velocity, forward) * kRunPitch * b;
        roll -= dot(pm.velocity, left) * kRunRoll * b;
        // The bob, visible even at low speeds; crouched it is three times.
        const float speed = std::max(xyspeed, 200.0f);
        float delta = fracsin * kBobPitch * speed * b;
        if (pm.ducked) delta *= 3.0f;
        pitch += delta;
        delta = fracsin * kBobRoll * speed * b;
        if (pm.ducked) delta *= 3.0f;
        if (odd) delta = -delta;
        roll += delta;

        up = static_cast<float>(view.eye);
        // The crouch, eased over DUCK_TIME.
        float t = static_cast<float>(view.ms - view.duck_time);
        if (t < kDuckTime) up -= view.duck_change * (static_cast<float>(kDuckTime) - t) / static_cast<float>(kDuckTime);
        // The bob up and down.
        up += std::min(6.0f, fracsin * xyspeed * kBobUp) * b;
        // The landing dip.
        t = static_cast<float>(view.ms - view.land_time);
        if (t < kLandDeflectTime) {
            up += view.land_change * (t / static_cast<float>(kLandDeflectTime)) * b;
        } else if (t < kLandDeflectTime + kLandReturnTime) {
            up += view.land_change * (1.0f - (t - static_cast<float>(kLandDeflectTime)) / static_cast<float>(kLandReturnTime)) * b;
        }
        // A step up, eased out.
        t = static_cast<float>(view.ms - view.step_time);
        if (t < kStepTime) up -= view.step_change * (static_cast<float>(kStepTime) - t) / static_cast<float>(kStepTime);
    }
    (void)ms;

    const float yaw = view.yaw;
    const float vpitch = std::clamp(view.pitch - pitch * kDeg, -89.0f * kDeg, 89.0f * kDeg);
    const float r = roll * kDeg;
    const float cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(vpitch), sp = std::sin(vpitch);
    const V3 forward{sy * cp, sp, cy * cp};
    const V3 right0{cy, 0.0f, -sy};
    const V3 up0 = cross(forward, right0);
    // Rolled to the right: the top of the view leans toward its right side.
    const V3 rup = up0 * std::cos(r) + right0 * std::sin(r);
    const V3 rright = right0 * std::cos(r) - up0 * std::sin(r);
    const V3 eye = origin + V3{0.0f, up * kUnit, 0.0f};
    const auto put = [](float* row, V3 v, float w) {
        row[0] = v.x;
        row[1] = v.y;
        row[2] = v.z;
        row[3] = w;
    };
    put(cam.right, rright, 0.0f);
    put(cam.up, rup, 0.0f);
    put(cam.forward, forward, 0.0f);
    put(cam.position, eye, 1.0f);
    // cg_fov is the angle across a 4:3 picture; the camera's is vertical.
    cam.fov_y = 2.0f * std::atan(std::tan(s.fov * 0.5f * kDeg) * 0.75f);
    out_yaw = yaw;
    out_pitch = vpitch;
    out_roll = r;
}

// ---- the guns: always +10, and Quake 3's projectiles -------------------------------
//
// Quake's weapons are five of Bloodborne's firearms (build/HANDOFF-quake.md):
// the rocket launcher is the Church Cannon, the grenade launcher the Cannon,
// the plasma gun the Piercing Rifle, the shotgun the Hunter Blunderbuss and the
// machine gun the Gatling Gun. A firearm's level is its EquipParamWeapon row,
// the base id + 100 a level: every level of these five takes the +10 row's
// attack and scaling, so a gun of any level hits as a +10 does and still
// scales with the hunter's stats - the game works the damage out as ever.
//
// Their bullets (BulletParam, by way of BehaviorParam_PC 1<variation>800 and
// 810): the rocket and the grenade fly as Quake 3's do, from g_missile.c - a
// rocket at 900 units a second, straight, for 15 s before it bursts (the
// Cannon's shell gave out after 0.5 s and 10 m), splash 120; a grenade at 700,
// bouncing until its 2.5 s fuse (grenades_frame), splash 150. The shotgun and
// the machine gun fire the Blunderbuss's and the Gatling Gun's own bullets,
// and the plasma gun one of A Call Beyond's small stars, flying straight and
// seeking nothing, or Blacksky Eye's shard (the plasma setting) in the
// Piercing Rifle's place. Damage stays the game's. Off, every row is as the
// game loaded it.
constexpr std::uint32_t kGuns[] = {15000000, 35000000, 36000000, 6000000, 33000000};
// The shotgun as if it carried Kyo's one-circle blood gem: Blood ATK UP +28.7%
// and Add blood ATK +7.3 (SpEffects 22519 and 13119 - blood is the game's
// neutral attack, which the weapon rows have no field for). Approximated on
// the row: its physical attack x1.287 + 7.3 and its bloodtinge scaling
// (correctMagic, 100 on the Blunderbuss) x1.287. The gem's third effect, WPN
// durability DOWN -65.3, does nothing to a gun and is left out.
constexpr std::uint32_t kShotgunGun = 6000000;
constexpr float kGemRate = 1.287f, kGemAdd = 7.3f;
constexpr float kGrenadeSpeed = 700.0f;  // units a second
constexpr float kRocket = 900.0f * kUnit, kGrenade = kGrenadeSpeed * kUnit;
struct BulletRule {
    std::uint32_t id;
    float speed;    // m/s, kept from start to end; 0 leaves the row's
    float gravity;  // m/s^2 throughout; < 0 leaves the row's
    float life;     // s; 0 leaves the row's
    float radius;   // the hit sphere, m; 0 leaves the row's (a burst's splash)
};
constexpr BulletRule kBulletRules[] = {
    {700, kRocket, 0.0f, 15.0f, 0.0f},          {710, kRocket, 0.0f, 15.0f, 0.0f},  // rocket
    {701, 0.0f, -1.0f, 0.0f, 120.0f * kUnit},   {711, 0.0f, -1.0f, 0.0f, 120.0f * kUnit},
    {720, kGrenade, 0.0f, 2.5f, 0.0f},          {730, kGrenade, 800.0f * kUnit, 2.5f, 0.0f},  // grenade (720: flown here)
    {721, kGrenade * 0.65f, 800.0f * kUnit, 2.5f, 0.0f},
    {722, 0.0f, -1.0f, 0.0f, 150.0f * kUnit},   {731, 0.0f, -1.0f, 0.0f, 150.0f * kUnit},
};
// The machine gun's and the shotgun's own bullets (the Gatling Gun's 740 and
// 750, the Blunderbuss's pellets 601 and 611) all but hitscan, as Quake 3's
// are: 150 m/s out to 8192 units.
constexpr std::uint32_t kHitscan[] = {740, 750, 601, 611};
constexpr float kHitscanSpeed = 150.0f, kHitscanRange = 8192.0f * kUnit;
constexpr float kPlasma = 1000.0f * kUnit;  // half Quake 3's 2000 (Kyo: slower), 10 s of it
// The plasma gun's shot (the Piercing Rifle's 540 and 550 rows, its L1 and
// charged shots, take the source's row): A Call Beyond's star 381 (the big
// star 380 bursts into 24 of them, each homing at 120 degrees a second under
// gravity 30; this one does neither), its burst 382; or Blacksky Eye's 460,
// straight already, its flash 461.
struct PlasmaLook {
    std::int32_t bullet, burst;
};
constexpr PlasmaLook kPlasmaLooks[] = {{381, 382}, {460, 461}};

std::vector<std::pair<std::uint32_t, bb::EQUIP_PARAM_WEAPON_ST>> g_saved_weapons;
std::vector<std::pair<std::uint32_t, bb::BULLET_PARAM_ST>> g_saved_bullets;
std::vector<std::pair<std::uint32_t, bb::ATK_PARAM_ST>> g_saved_atks;
bool g_guns_quake = false;
const void* g_guns_table = nullptr;  // the first row written: a new table (a reload) is written again

void guns_restore() {
    for (auto& [id, row] : g_saved_weapons)
        if (auto* w = g.param<bb::params::EquipParamWeapon>(id)) *w = row;
    for (auto& [id, row] : g_saved_bullets)
        if (auto* b = g.param<bb::params::Bullet>(id)) *b = row;
    for (auto& [id, row] : g_saved_atks)
        if (auto* a = g.param<bb::params::AtkParam_Pc>(id)) *a = row;
    g_saved_weapons.clear();
    g_saved_bullets.clear();
    g_saved_atks.clear();
    g_guns_quake = false;
}

void guns_apply() {
    auto* probe = g.param<bb::params::EquipParamWeapon>(kGuns[0] + 1000);
    if (!probe || !g.param<bb::params::Bullet>(700)) return;  // the tables load during the boot
    if (g_guns_quake && probe == g_guns_table) return;
    if (probe != g_guns_table) {
        // Not the rows saved from: those went with the table.
        g_saved_weapons.clear();
        g_saved_bullets.clear();
        g_saved_atks.clear();
    }
    int weapons = 0, bullets = 0;
    for (const std::uint32_t base : kGuns) {
        const auto* top_row = g.param<bb::params::EquipParamWeapon>(base + 1000);
        if (!top_row) continue;
        const bb::EQUIP_PARAM_WEAPON_ST top_copy = *top_row;
        const auto* top = &top_copy;
        // Up to +10 itself: the shotgun's gem goes on the row the shot names too.
        for (std::uint32_t level = 0; level <= (base == kShotgunGun ? 10u : 9u); ++level) {
            auto* w = g.param<bb::params::EquipParamWeapon>(base + level * 100);
            if (!w) continue;
            g_saved_weapons.emplace_back(base + level * 100, *w);
            w->attackBasePhysics = top->attackBasePhysics;
            w->attackBaseMagic = top->attackBaseMagic;
            w->attackBaseFire = top->attackBaseFire;
            w->attackBaseThunder = top->attackBaseThunder;
            w->attackBaseStamina = top->attackBaseStamina;
            w->attackBaseParry = top->attackBaseParry;
            w->attackBaseRepel = top->attackBaseRepel;
            w->saWeaponDamage = top->saWeaponDamage;
            w->correctStrength = top->correctStrength;
            w->correctAgility = top->correctAgility;
            w->correctMagic = top->correctMagic;
            w->correctFaith = top->correctFaith;
            w->correctType = top->correctType;
            if (base == kShotgunGun) {
                w->attackBasePhysics = static_cast<std::uint16_t>(std::lround(top->attackBasePhysics * kGemRate + kGemAdd));
                w->correctStrength = top->correctStrength * kGemRate;
                w->correctAgility = top->correctAgility * kGemRate;
                w->correctMagic = top->correctMagic * kGemRate;
                w->correctFaith = top->correctFaith * kGemRate;
            }
            ++weapons;
        }
    }
    for (const BulletRule& r : kBulletRules) {
        auto* b = g.param<bb::params::Bullet>(r.id);
        if (!b) continue;
        g_saved_bullets.emplace_back(r.id, *b);
        if (r.speed > 0.0f) {
            b->initVellocity = r.speed;
            b->maxVellocity = r.gravity > 0.0f ? 100.0f : r.speed;  // a falling grenade gains speed
            b->minVellocity = r.gravity > 0.0f ? 0.0f : r.speed;
            b->accelInRange = b->accelOutRange = 0.0f;
            b->accelTime = 0.0f;
            b->homingAngle = 0;
        }
        if (r.gravity >= 0.0f) b->gravityInRange = b->gravityOutRange = r.gravity;
        if (r.life > 0.0f) {
            b->life = r.life;
            if (r.speed > 0.0f) b->dist = r.speed * r.life;  // all of it in range
        }
        if (r.radius > 0.0f) b->hitRadius = r.radius;
        ++bullets;
    }
    // The grenade launcher's shell: the game's is a short throw (0.4 s) that
    // turns into a falling shell (721) bursting (722) on the next thing it
    // meets. Quake 3's bounces until its 2.5 s fuse, so the shell passes
    // through the map, the plugin flies it (grenades_frame), and its end -
    // the fuse, or a character it hits - is the burst itself.
    if (auto* b = g.param<bb::params::Bullet>(720)) {
        b->isPenetrateMap = 1;
        b->HitBulletID = 722;
        b->isInheritSpeedToChild = 0;
        b->maxVellocity = 100.0f;
        b->minVellocity = 0.0f;
    }
    // The grenade's burst shows its own explosion: 720 handed its effect on to
    // its child (the falling shell 721 it turned into), which here is the
    // burst.
    if (auto* b = g.param<bb::params::Bullet>(720)) b->isInheritEffectToChild = 0;
    // The plasma gun's shots, the look's row in their place.
    if (const auto* src = g.param<bb::params::Bullet>(static_cast<std::uint32_t>(kPlasmaLooks[s.plasma].bullet))) {
        for (const std::uint32_t id : {540u, 550u}) {
            auto* b = g.param<bb::params::Bullet>(id);
            if (!b) continue;
            g_saved_bullets.emplace_back(id, *b);
            *b = *src;
            b->homingAngle = 0;
            b->gravityInRange = b->gravityOutRange = 0.0f;
            b->launchConditionType = 0;
            b->initVellocity = b->maxVellocity = b->minVellocity = kPlasma;
            b->accelInRange = b->accelOutRange = 0.0f;
            b->accelTime = 0.0f;
            b->life = 10.0f;
            b->dist = kPlasma * 10.0f;
            ++bullets;
        }
    }
    for (const std::uint32_t id : kHitscan) {
        auto* b = g.param<bb::params::Bullet>(id);
        if (!b) continue;
        g_saved_bullets.emplace_back(id, *b);
        b->initVellocity = b->maxVellocity = b->minVellocity = kHitscanSpeed;
        b->accelInRange = b->accelOutRange = 0.0f;
        b->accelTime = 0.0f;
        b->dist = kHitscanRange;
        b->life = kHitscanRange / kHitscanSpeed;
        ++bullets;
    }
    // The grenade's burst breaks what the rocket's does: its attacks (621 and
    // 631) have no object power (atkObj 0), the rocket's (601) 80 - crates,
    // barrels and boxes stood through it.
    if (const auto* rocket = g.param<bb::params::AtkParam_Pc>(601)) {
        for (const std::uint32_t id : {621u, 631u}) {
            auto* a = g.param<bb::params::AtkParam_Pc>(id);
            if (!a) continue;
            g_saved_atks.emplace_back(id, *a);
            a->atkObj = rocket->atkObj;
        }
    }
    g_guns_table = probe;
    g_guns_quake = true;
    g.log("quake: the guns at +10 whatever their level (%d rows), %d bullets flying as Quake 3's", weapons, bullets);
    if (const auto* sg = g.param<bb::params::EquipParamWeapon>(kShotgunGun + 1000))
        g.log("quake: the shotgun's blood gem: physical %u, bloodtinge scaling %.1f", sg->attackBasePhysics,
              static_cast<double>(sg->correctMagic));
}

// ---- input -------------------------------------------------------------------------

constexpr std::uint16_t kKeyA = 4, kKeyD = 7, kKeyS = 22, kKeyW = 26, kKeySpace = 44, kKeyLCtrl = 224;
constexpr std::uint16_t kTaken[] = {kKeyW, kKeyA, kKeyS, kKeyD, kKeySpace, kKeyLCtrl};
constexpr std::uint32_t kMouseLeft = 1, kMouseRight = 2;

// QUAKE_TEST=<what>@<from>-<to>,...: the movement driven by a script, in
// seconds since the player was first in the world - fwd, back, left, right,
// jump, crouch, yaw:<degrees a second>, pitch:<degrees a second>, and dump@<t>,
// which asks the host's test requests (BBHOST_TEST_REQUESTS) for the frame.
struct TestStep {
    std::string what;
    float arg = 0.0f;
    double from = 0.0, to = 0.0;
    bool done = false;
};
std::vector<TestStep> g_test;
double g_test_start = -1.0;

void parse_test() {
    const char* e = std::getenv("QUAKE_TEST");
    if (!e) return;
    std::string spec = e;
    std::size_t i = 0;
    while (i < spec.size()) {
        std::size_t j = spec.find(',', i);
        if (j == std::string::npos) j = spec.size();
        const std::string item = spec.substr(i, j - i);
        i = j + 1;
        const std::size_t at_pos = item.find('@');
        if (at_pos == std::string::npos) continue;
        TestStep t;
        t.what = item.substr(0, at_pos);
        if (const std::size_t c = t.what.find(':'); c != std::string::npos) {
            t.arg = std::strtof(t.what.c_str() + c + 1, nullptr);
            t.what.resize(c);
        }
        const std::string span = item.substr(at_pos + 1);
        const std::size_t dash = span.find('-');
        t.from = std::strtod(span.c_str(), nullptr);
        t.to = dash == std::string::npos ? t.from + 0.1 : std::strtod(span.c_str() + dash + 1, nullptr);
        g_test.push_back(t);
    }
    g.log("quake: QUAKE_TEST with %zu steps", g_test.size());
}

struct Frame {
    Cmd cmd;
    float dx = 0.0f, dy = 0.0f;  // the mouse's counts
    float wheel = 0.0f;          // notches, up positive
    int pick = -1;               // a number key pressed: the weapon it picks
    bool playing = false;
};

// ---- the weapons: which one is up ----------------------------------------------------
//
// Quake 3's order (its 2 to 8, the gauntlet and the guns this has not left
// out): 1 the machine gun, 2 the shotgun, 3 the grenade launcher, 4 the rocket
// launcher, 5 the plasma gun. The wheel goes through them as Quake 3 binds it
// (down weapnext, up weapprev); the number keys pick one, each only where the
// player's bindings leave the key free. A change takes Quake 3's time: the gun
// goes down for 200 ms and comes up for 250 (WEAPON_DROPPING, WEAPON_RAISING).
struct QuakeWeapon {
    const char* name;
    const char* gun;          // the Bloodborne gun it is made of
    std::uint32_t weapon;     // its EquipParamWeapon base row; + 1000 is +10
    std::uint32_t behavior;   // its BehaviorParam_PC L1 shot: 1e8 + variation * 1000 + 800
    std::uint32_t bullet;     // the shot's BulletParam row
    double refire_ms;         // Quake 3's weapon time after a shot
    float spread;             // Quake 3's spread, units across at 8192 (0: straight)
};
constexpr QuakeWeapon kWeapons[] = {
    {"Machine Gun", "Gatling Gun", 33000000, 103300800, 740, 100.0, 200.0f},
    {"Shotgun", "Hunter Blunderbuss", 6000000, 100600800, 600, 1000.0, 0.0f},  // the pellets' fan is the bullet's
    // Each gun's model and damage with the other's shot: the Cannon fires the
    // Church Cannon's shell (720, flown as the grenade) and the Church Cannon
    // the Cannon's (700, the rocket).
    {"Grenade Launcher", "Cannon", 15000000, 103500800, 720, 800.0, 0.0f},
    {"Rocket Launcher", "Church Cannon", 35000000, 101500800, 700, 800.0, 0.0f},
    {"Plasma Gun", "Piercing Rifle", 36000000, 103600800, 540, 100.0, 0.0f},
};
constexpr int kWeaponCount = sizeof(kWeapons) / sizeof(kWeapons[0]);
constexpr std::uint16_t kKeyDigit1 = 30;  // SDL scancodes 30..38 are 1..9
constexpr double kDropMs = 200.0, kRaiseMs = 250.0;

struct Weapons {
    int current = 3;         // the rocket launcher, as a Quake player would reach for first
    int pending = -1;        // the one coming up
    double change_ms = -1e9; // when the change began
    bool digit_free[kWeaponCount] = {};
} arms;

// The number keys the bindings leave free, asked once the host can say.
void find_free_digits() {
    for (int i = 0; i < kWeaponCount; ++i) {
        const std::uint16_t sc = static_cast<std::uint16_t>(kKeyDigit1 + i);
        arms.digit_free[i] = g.api->key_bound && g.api->key_bound(sc) == 0;
    }
}

// The gun is down or coming up: no shot until it is up (Quake 3's
// weaponstate).
bool weapon_ready(double ms) { return arms.pending < 0 && ms - arms.change_ms >= kDropMs + kRaiseMs; }

void select_weapon(int index, double ms) {
    index = ((index % kWeaponCount) + kWeaponCount) % kWeaponCount;
    if (index == (arms.pending >= 0 ? arms.pending : arms.current)) return;
    arms.pending = index;
    arms.change_ms = ms;
    char line[96];
    std::snprintf(line, sizeof(line), "%s - %s", kWeapons[index].name, kWeapons[index].gun);
    // Small, in the bottom left corner, where Bloodborne's HUD leaves room.
    g.text_at(line, 1.5f, 0.02f, 0.96f, 0.7f, BB_TEXT_LEFT | BB_TEXT_BOTTOM);
}

// ---- the shot -------------------------------------------------------------------------
//
// A gun's shot is a bullet emitter the bullet manager (*0x593e8d0) makes from
// an emit request: sub_1e03b50(manager, request), which takes an emitter off
// its free list and fills it from the 0x440-byte request; the manager's next
// update fires the bullets (and an impact's child, through sub_1e03e50). The
// request below is one of the game's own - Kyo's Cannon shot, dumped by
// QUAKE_PROBE (build/HANDOFF-quake.md) - with what the request builder
// (sub_1dedc20) never writes there zeroed (its caller's stack leftovers), the
// two objects it makes put back for this run's image (+0x130, and the one
// sub_2c51220 builds at +0x140), and the builder's own tail from +0x380. Each
// shot then names the gun, its +10 row (the attack the damage is worked out
// from, with the hunter's stats), the player as its owner, the muzzle in front
// of the eye as where it starts and a point down the view as what it flies at.
constexpr std::uint64_t kBulletManager = 0x593e8d0;
constexpr std::uint64_t kEmit = 0x1e03b50;
constexpr std::uint64_t kGameStateMan = 0x5956678;
constexpr std::uint32_t kShotTemplate[0x380 / 4] = {
    0x060cc780, 0x000002bc, 0x00000003, 0x10044000, 0x00010001, 0xffffffff, 0xffffffff, 0x00000003,  // +000
    0x00000000, 0x00000000, 0xffffffff, 0x00000000, 0x3d088800, 0x00000002, 0x00000001, 0xffffffff,  // +020
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,  // +040
    0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,  // +060
    0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,  // +080
    0xfffffffe, 0x00e4e1c0, 0xfffffffe, 0x00e4e1c0, 0x2a2a0000, 0x3f8048d1, 0x3f800000, 0x00000000,  // +0a0
    0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x00000000,  // +0c0
    0xffffffff, 0xc230b492, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000,  // +0e0
    0xc3397186, 0xc1d93009, 0x429a020c, 0x00000000, 0x00000004, 0x00000004, 0x01807d92, 0x00000003,  // +100
    0x00000000, 0x40000000, 0x40000000, 0x40000000, 0x0531cac0, 0x00000003, 0x00000000, 0x00000003,  // +120
    0x052dbe70, 0x00000003, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,  // +140
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,  // +160
    0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000000, 0xffffffff, 0x00000000, 0xffffffff,  // +180
    0xffffffff, 0x00000000, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000,  // +1a0
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,  // +1c0
    0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,  // +1e0
    0x3f800000, 0x7f7fff00, 0x00000000, 0x3f000007, 0xffffffff, 0xffffffff, 0x3f800000, 0x3f800000,  // +200
    0x2a2a2a2a, 0x2a2a2a2a, 0x01471f18, 0x00000006, 0xae6ff060, 0x00000004, 0x1256ead8, 0x00000001,  // +220
    0xae6ff010, 0x00000004, 0x018c8711, 0x00000003, 0xae6fee2c, 0x00000004, 0x02bd5250, 0x00000003,  // +240
    0xae6ff3f8, 0x00000004, 0xae6ff0b0, 0x00000004, 0xae6ff3f8, 0x00000004, 0x006aa6ac, 0x00000003,  // +260
    0xc33ff655, 0xc1dd7d9d, 0x428ae135, 0x3f800000, 0x428ae135, 0x428ae135, 0x428ae135, 0x428ae135,  // +280
    0xc1dd7d9d, 0xc1dd7d9d, 0xc1dd7d9d, 0xc1dd7d9d, 0x3f271f47, 0x00000000, 0xbf41ec52, 0x428ae135,  // +2a0
    0xc3405516, 0xc1dd7d9d, 0x428a0550, 0x3f800000, 0x00000000, 0x00000000, 0x3edbe501, 0x80000000,  // +2c0
    0xae6ff000, 0x00000000, 0x00000000, 0xbf800000, 0x00000000, 0xc1dd7d00, 0x00000000, 0x3f800000,  // +2e0
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,  // +300
    0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000,  // +320
    0x00000000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000,  // +340
    0x00000000, 0x00000000, 0xffffffff, 0x00000011, 0x00000001, 0x00000000, 0x00000000, 0x3f000000,  // +360
};

alignas(16) std::uint8_t g_request[0x440];

// The request for one shot of `w` from `muzzle` at `aim`.
bool build_request(const QuakeWeapon& w, std::uint32_t owner, V3 muzzle, V3 aim, V3 eye, float game_yaw) {
    std::uint8_t* r = g_request;
    std::memset(r, 0, sizeof(g_request));
    std::memcpy(r, kShotTemplate, sizeof(kShotTemplate));
    // The builder never writes these: whatever its caller's stack held.
    std::memset(r + 0x118, 0, 8);
    std::memset(r + 0x138, 0, 8);
    std::memset(r + 0x220, 0, 0x60);
    std::memset(r + 0x2e0, 0, 4);
    // Its two objects, for this run's image.
    const std::uint64_t vt130 = g.api->guest_addr(0x571cac0), vt140 = g.api->guest_addr(0x56dbe70);
    const std::uint64_t empty = g.api->guest_addr(0x5721db0);
    if (!vt130 || !vt140 || !empty) return false;
    std::memcpy(r + 0x130, &vt130, 8);
    std::memcpy(r + 0x140, &vt140, 8);
    std::memset(r + 0x148, 0, 0x18);
    // Its tail, as sub_1dedc20 writes it.
    const auto put = [&](std::size_t off, auto v) { std::memcpy(r + off, &v, sizeof(v)); };
    put(0x368, -1);
    put(0x36c, 0x11);
    put(0x370, 1);
    put(0x380, 0.1f);
    put(0x384, 10.0f);
    put(0x38c, 0.785398185f);
    put(0x390, 0.785398185f);
    put(0x3a8, -1.0f);
    for (const std::size_t base : {std::size_t{0x3b0}, std::size_t{0x3e0}}) {
        for (std::size_t row = 0; row < 3; ++row) put(base + row * 20, 1.0f);
    }
    put(0x410, -1);
    put(0x418, empty);
    if (std::uint8_t* gsm = ptr_at(kGameStateMan)) std::memcpy(r + 0x420, gsm + 0xa68, 16);
    // The shot.
    put(0x000, w.behavior);
    put(0x004, w.bullet);
    // No dummy poly (the game's shot names the hand's, 3): the bullet starts
    // at the launch transform instead - the two 3x4s at +0x3b0 and +0x3e0,
    // columns right, up, back along the flight and the point - at the muzzle,
    // flying at the aim. Its orientation stays the 4x4's at +0x320 (identity),
    // so a bullet's own turn is the gun's spread alone (flights_frame).
    put(0x008, -1);
    {
        V3 f = aim - muzzle;
        normalize(f);
        V3 rt = cross(V3{0.0f, 1.0f, 0.0f}, f);
        if (normalize(rt) < 1e-4f) rt = V3{1.0f, 0.0f, 0.0f};
        const V3 up = cross(f, rt);
        const V3 cols[4] = {rt, up, f * -1.0f, muzzle};
        for (const std::size_t base : {std::size_t{0x3b0}, std::size_t{0x3e0}}) {
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 4; ++col) {
                    const V3& c = cols[col];
                    put(base + 16 * row + 4 * col, row == 0 ? c.x : row == 1 ? c.y : c.z);
                }
            }
        }
    }
    put(0x00c, owner);
    put(0x0a4, w.weapon + 1000);
    put(0x0ac, w.weapon + 1000);
    put(0x100, aim.x);
    put(0x104, aim.y);
    put(0x108, aim.z);
    put(0x280, muzzle.x);
    put(0x284, muzzle.y);
    put(0x288, muzzle.z);
    put(0x28c, 1.0f);
    for (std::size_t k = 0; k < 4; ++k) {
        put(0x290 + 4 * k, muzzle.z);
        put(0x2a0 + 4 * k, muzzle.y);
    }
    put(0x2b0, -std::sin(game_yaw));
    put(0x2b8, std::cos(game_yaw));
    put(0x2bc, muzzle.z);
    put(0x2c0, eye.x);
    put(0x2c4, eye.y);
    put(0x2c8, eye.z);
    put(0x2cc, 1.0f);
    put(0x2f4, muzzle.y);
    return true;
}

bb::Rng g_spread_rng(0x9e3779b9u);
double g_next_shot_ms = 0.0;
std::uint64_t g_shots = 0;
double g_fire_ms = -1e9;  // the last shot, for the gun's kick

// A shot in the air until its bullets are found (flights_frame): where it
// leaves and the way it goes.
struct Shot {
    std::int32_t bullet;  // the BulletParam row it is made of
    V3 muzzle, dir, right, up;
    double ms;
};
std::vector<Shot> g_fired;

// ---- Quake 3's splash: rocket and grenade jumps, plasma climbing ---------------------
//
// g_combat.c's G_RadiusDamage as it pushes the player: a blast within its
// splash radius of the player's box does damage * (1 - distance / radius)
// points, and G_Damage throws by those points (at most 200) times
// g_knockback / 200 (the mass) units a second, along the line from the blast
// to the player's origin raised 24 units; then for points * 2 ms (50 to 200)
// the ground neither brakes nor steers (PMF_TIME_KNOCKBACK). Your own blasts
// throw you as hard as anyone's - Quake 3 halves the self damage only after
// the knockback is worked out, "so rocket jumping works" - and here they do no
// damage to you at all. Rocket 100 points over 120 units, grenade 100 over
// 150, plasma 15 over 20: a rocket under the feet adds about 475 units a
// second, each plasma ball into a wall beside you up to 75.
constexpr float kQ3HalfWidth = 15.0f;  // units: Quake 3's player box is 30 across
constexpr double kPrestep = 0.05;      // MISSILE_PRESTEP_TIME: a missile starts as if fired 50 ms before

// Whether a blast at `from` reaches `to` (CanDamage's trace): a thin cast,
// started just off the surface the blast is on.
bool reaches(std::uint8_t* world, V3 from, V3 to) {
    V3 d = to - from;
    const float len = normalize(d);
    if (len < 0.05f) return true;
    float frac = 1.0f;
    V3 n;
    return !sphere_cast(world, from + d * 0.04f, d * (len - 0.04f), 0.01f, frac, n);
}

void blast(std::uint8_t* world, V3 at, float damage, float radius, const char* what) {
    if (s.knockback <= 0.0f) return;
    const float hw = kQ3HalfWidth * kUnit;
    const V3 lo{pm.origin.x - hw, pm.origin.y, pm.origin.z - hw};
    const V3 hi{pm.origin.x + hw, pm.origin.y + body_height() * kUnit, pm.origin.z + hw};
    const auto out = [](float p, float a, float b) { return p < a ? a - p : p > b ? p - b : 0.0f; };
    const float dist = length(V3{out(at.x, lo.x, hi.x), out(at.y, lo.y, hi.y), out(at.z, lo.z, hi.z)}) / kUnit;
    if (dist >= radius) return;
    // CanDamage: the box's middle, or one of four corners around it, in sight.
    const V3 mid = (lo + hi) * 0.5f;
    bool seen = reaches(world, at, mid);
    for (int c = 0; c < 4 && !seen; ++c) seen = reaches(world, at, mid + V3{(c & 1) ? hw : -hw, 0.0f, (c & 2) ? hw : -hw});
    if (!seen) return;
    const int knockback = std::min(200, static_cast<int>(damage * (1.0f - dist / radius)));
    if (knockback <= 0) return;
    V3 dir = pm.origin + V3{0.0f, 48.0f * kUnit, 0.0f} - at;  // the origin, 24 over the feet, raised 24
    normalize(dir);
    pm.velocity = pm.velocity + dir * (s.knockback * static_cast<float>(knockback) / 200.0f);
    if (pm.knock_time <= 0.0) pm.knock_time = std::clamp(knockback * 2, 50, 200) / 1000.0;
    if (g_log)
        g.log("quake: %s blast %.0f units away: knockback %d, velocity now %.0f %.0f %.0f", what, static_cast<double>(dist), knockback,
              static_cast<double>(pm.velocity.x), static_cast<double>(pm.velocity.y), static_cast<double>(pm.velocity.z));
}

// ---- the guns' own flashes and sounds ----------------------------------------
//
// What each gun's firing animation adds to its shot, from the player's TAE
// (chr/c0000.anibnd.dcx: a150 the Gatling Gun, a102 the Blunderbuss, a152 the
// Church Cannon, a118 the Cannon, a153 the Piercing Rifle): the muzzle effect
// of a type-96 event and the shot's sound of a type-128 one. The effect is
// made as a character's own are (sub_1ca6410's way, as tools/fx_player makes
// them): a SprjSfx request for the player, bound to a dummy poly of the model
// PlayerIns vfunc +0x6b0 names for the id (the bind takes the number the
// model itself has: 215 of 20215) - 20000 and up the left hand's weapon, here
// the Quake gun's own muzzle (the barrel's end: 215, the Cannon's
// 206, the Gatling Gun's 100 as the game has it), since the body the game puts
// the Cannons' flash on (dummy poly 3) is not drawn - and ended after its
// event's length (the request's flags +0x10 bit 0, as sub_1cac8b0 ends a
// character's). The sound is the TAE's own call (sub_17e0290: a handle out,
// the point, the type - 1 a 'c' sound of the character's bank, 5 an 's' -
// and the id), at the muzzle.
struct GunFx {
    std::int32_t effect, dummy;  // the muzzle effect and its dummy poly
    double hold_ms;              // its event's length
    std::int32_t sound_type, sound;
};
constexpr GunFx kGunFx[kWeaponCount] = {
    {6602, 20100, 1167.0, 5, 6603},  // the Gatling Gun's barrels turning, their spin-up (spin_frame below)
    {6302, 20215, 1000.0, 1, 700},
    {6202, 20206, 2000.0, 1, 730},  // the Cannon's
    {6702, 20215, 2000.0, 1, 730},  // the Church Cannon's
    {6802, 20215, 1000.0, 1, 960},
};
constexpr std::uint64_t kSprjSfx = 0x5940370;    // SprjSfx*, the effects
constexpr std::uint64_t kSfxParams = 0x57bf090;  // its requests' parameters (+0x30 a character's)
constexpr std::uint64_t kSfxOffset = 0x593e970;  // a request's offset as the game starts one: 32 bytes for its +0x30
constexpr std::uint64_t kSfxRequest = 0x2378160, kSfxBind = 0x2372fd0, kSoundAt = 0x17e0290;
constexpr std::uint64_t kSoundKinds = 0x4b348a0;  // 12 bytes by ChrIns+0x78 (14 the rest): +3 makes a 'c' sound a 'g'

bool guest_ptr(std::uint64_t v) { return v > 0x100000000ull && v < 0x800000000000ull; }
std::uint64_t bn_of(std::uint64_t runtime) { return runtime - g.api->guest_addr(0x400000) + 0x400000; }

struct Flash {
    std::uint64_t request;
    double end_ms;
};
std::vector<Flash> g_flashes;

// The effect `id` on the player's dummy poly `dummy`: its request (ours to
// end), 0 for none made.
std::uint64_t flash_on(std::uint8_t* chr, std::int32_t dummy, std::int32_t id) {
    if (!s.muzzle_flash) return 0;
    std::uint8_t* sfx = ptr_at(kSprjSfx);
    std::uint8_t* params = ptr_at(kSfxParams);
    if (!chr || !sfx || !params) return 0;
    static std::int32_t dmy, eid;
    static std::uint64_t request;
    const auto* vtable = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(at<std::uint64_t>(chr, 0)));
    dmy = dummy;
    BbCallRegs r{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(chr);
    r.arg[1] = reinterpret_cast<std::uint64_t>(&dmy);
    if (g.api->call_guest(bn_of(vtable[0x6b0 / 8]), &r) != 0 || !guest_ptr(r.ret)) return 0;
    const std::uint64_t model = r.ret;
    eid = id;
    request = 0;
    r = BbCallRegs{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(&request);
    r.arg[1] = reinterpret_cast<std::uint64_t>(sfx);
    r.arg[2] = reinterpret_cast<std::uint64_t>(chr);
    r.arg[3] = reinterpret_cast<std::uint64_t>(&eid);
    r.arg[4] = reinterpret_cast<std::uint64_t>(params + 0x30);
    if (g.api->call_guest(kSfxRequest, &r) != 0 || !guest_ptr(request)) return 0;
    auto* req = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(request));
    std::uint8_t offset[32];
    if (g.api->read(kSfxOffset, offset, sizeof offset) == 0) std::memcpy(req + 0x30, offset, sizeof offset);
    // On a weapon's or an armour's model the dummy poly is the model's own
    // number (215 of 20215); bound by the whole id, the request retires
    // before it makes anything.
    dmy = dummy >= 10000 ? dummy % 1000 : dummy;
    r = BbCallRegs{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(&request);
    r.arg[1] = reinterpret_cast<std::uint64_t>(&dmy);
    r.arg[2] = model;
    g.api->call_guest(kSfxBind, &r);
    return request;
}

// Ended, and our reference dropped: SprjSfx's list keeps its own, and the last
// one out releases it (its vtable's first slot, as DLReferenceCountObject's
// Unref does).
void flash_end(std::uint64_t request) {
    auto* req = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(request));
    at<std::uint8_t>(req, 0x10) |= 1;
    std::int32_t& refs = at<std::int32_t>(req, 0x8);
    if (refs > 1) {
        --refs;
        return;
    }
    if (refs != 1) return;
    refs = 0;
    const auto* vtable = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(at<std::uint64_t>(req, 0)));
    BbCallRegs r{};
    r.arg[0] = request;
    g.api->call_guest(bn_of(vtable[0]), &r);
}

void flash(std::uint8_t* chr, std::int32_t dummy, std::int32_t id, double hold_ms, double ms) {
    if (const std::uint64_t req = flash_on(chr, dummy, id)) g_flashes.push_back({req, ms + hold_ms});
}

// Each frame: the flashes whose events are over ended (all of them for a
// gun put away or the player let go).
void flashes_frame(double ms, bool all) {
    for (std::size_t i = 0; i < g_flashes.size();) {
        if (all || ms >= g_flashes[i].end_ms) {
            flash_end(g_flashes[i].request);
            g_flashes.erase(g_flashes.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
}

// The TAE's sound at a point: a type-1 sound ('c', the character's own bank)
// becomes a 'g' for the kinds of character the TAE makes them so.
void sound_at(std::uint8_t* chr, V3 where, std::int32_t type, std::int32_t id) {
    if (!chr || type < 0 || id <= 0) return;
    // The TAE's floor (8) and attire (9) sounds are the character's own 'c'
    // sounds: the event names the hundred and the character's material module
    // (ChrIns+0x3b0 -> +0xa0) the rest - the floor under it at +0x10, the
    // armour's at +0x12 (sub_1e249d0) - c000003001 a landing on stone,
    // c000000158 the attire's rustle in armour 58.
    if (type == 8 || type == 9) {
        std::uint8_t* modules = at<std::uint8_t*>(chr, 0x3b0);
        std::uint8_t* mat = modules ? at<std::uint8_t*>(modules, 0xa0) : nullptr;
        if (!mat) return;
        id = id / 100 * 100 + (type == 8 ? at<std::uint8_t>(mat, 0x10) : at<std::uint16_t>(mat, 0x12));
        type = 1;
    }
    if (type == 1) {
        std::uint8_t ghost = 0;
        g.api->read(kSoundKinds + 12 * std::min(at<std::uint32_t>(chr, 0x78), 14u) + 3, &ghost, 1);
        if (ghost) type = 10;
    }
    alignas(16) static float point[4];
    static std::uint64_t handle[2];
    point[0] = where.x;
    point[1] = where.y;
    point[2] = where.z;
    point[3] = 1.0f;
    BbCallRegs r{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(handle);
    r.arg[1] = reinterpret_cast<std::uint64_t>(point);
    r.arg[2] = static_cast<std::uint32_t>(type);
    r.arg[3] = static_cast<std::uint32_t>(id);
    g.api->call_guest(kSoundAt, &r);
}

// A sound by its name, at a point: what sub_17e0290 does once it has spelled
// the name - the name object (sub_2bcb330, 0x110 bytes), sub_2bd3a80(handle
// out, name, point, -1, 0, 1, 0), the handle let go (sub_2be2c50) - for the
// names its twelve kinds cannot spell, as the hit sounds' 'z'. Nothing while
// the game's sound is held (the byte at 0x593d6f0).
void named_sound(const char* name, V3 where) {
    std::uint8_t held = 0;
    if (g.api->read(0x593d6f0, &held, 1) != 0 || held) return;
    alignas(16) static std::uint8_t object[0x110];
    alignas(16) static char text[16];
    alignas(16) static float point[4];
    static std::uint64_t handle[2];
    std::snprintf(text, sizeof text, "%s", name);
    point[0] = where.x;
    point[1] = where.y;
    point[2] = where.z;
    point[3] = 1.0f;
    BbCallRegs r{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(object);
    r.arg[1] = reinterpret_cast<std::uint64_t>(text);
    if (g.api->call_guest(0x2bcb330, &r) != 0) return;
    r = BbCallRegs{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(handle);
    r.arg[1] = reinterpret_cast<std::uint64_t>(object);
    r.arg[2] = reinterpret_cast<std::uint64_t>(point);
    r.arg[3] = 0xffffffffu;
    r.arg[4] = 0;
    r.arg[5] = 1;
    r.stack[0] = 0;
    if (g.api->call_guest(0x2bd3a80, &r) != 0) return;
    r = BbCallRegs{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(handle);
    g.api->call_guest(0x2be2c50, &r);
}

// Quake 3's jump1 and land1, in the hunter's own sounds (player TAE a00):
// a jump is a roll's push-off (x2001) with the attire's swish (b400); a landing
// a fall's - a short one (a8200) x3001 and b100, a hard one (a8202) x6000 on
// top; a drop of a step or two just a footstep (x1001), as is each of
// PM_Footsteps' steps while running. At the feet.
void body_sounds(std::uint8_t* chr) {
    const int land = pm.land;
    const bool jumped = pm.jumped, footstep = pm.footstep;
    pm.land = 0;
    pm.jumped = pm.footstep = false;
    if (!s.sounds || !chr) return;
    const V3 feet = pm.origin;
    if (land == 1 || (footstep && land == 0)) sound_at(chr, feet, 8, 1001);
    if (land >= 2) {
        sound_at(chr, feet, 8, 3001);
        sound_at(chr, feet, 9, 100);
    }
    if (land >= 3) sound_at(chr, feet, 8, 6000);
    if (jumped) {
        sound_at(chr, feet, 8, 2001);
        sound_at(chr, feet, 9, 400);
    }
}

// The Gatling Gun's barrels, as its held animations turn them: a burst's
// first shot spins them up (s6603) with their flash (6602), the flash goes
// on as they keep turning (6603, its next animation's), and a burst's end
// winds them down - the smoke and sound of 6604.
struct Spin {
    bool on = false;
    double last_ms = -1e9, loop_ms = -1e9;
    double start_ms = -1e9, stop_ms = -1e9;  // the burst's, for the barrels' animations
    std::uint64_t fx = 0;
    V3 muzzle;
} spin;

void spin_shot(std::uint8_t* chr, V3 muzzle, double ms) {
    if (!spin.on) {
        sound_at(chr, muzzle, kGunFx[0].sound_type, kGunFx[0].sound);
        spin.fx = flash_on(chr, kGunFx[0].dummy, kGunFx[0].effect);
        spin.on = true;
        spin.loop_ms = ms;
        spin.start_ms = ms;
    }
    spin.last_ms = ms;
    spin.muzzle = muzzle;
}

void spin_frame(std::uint8_t* chr, double ms, bool stop) {
    if (!spin.on) return;
    if (stop || ms - spin.last_ms > kWeapons[0].refire_ms + 150.0) {
        if (spin.fx) flash_end(spin.fx);
        spin.fx = 0;
        spin.on = false;
        spin.stop_ms = ms;
        if (chr) {
            flash(chr, kGunFx[0].dummy, 6604, 1000.0, ms);
            sound_at(chr, spin.muzzle, 5, 6604);
        }
        return;
    }
    if (ms - spin.loop_ms >= 1100.0) {
        if (spin.fx) flash_end(spin.fx);
        spin.fx = flash_on(chr, kGunFx[0].dummy, 6603);
        spin.loop_ms = ms;
    }
}

// One shot of the gun that is up, from the eye along the view: Quake 3's
// muzzle 14 units ahead, and its spread (random within spread/8192 either way).
void fire(std::uint8_t* chr, V3 eye, float yaw, float pitch, float game_yaw) {
    const QuakeWeapon& w = kWeapons[arms.current];
    const float cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch);
    const V3 forward{sy * cp, sp, cy * cp}, right{cy, 0.0f, -sy};
    const V3 up = cross(forward, right);
    V3 dir = forward;
    if (w.spread > 0.0f) {
        const float r = g_spread_rng.unit() * 2.0f * kPi;
        const float a = (g_spread_rng.unit() * 2.0f - 1.0f) * w.spread / 8192.0f;
        const float b = (g_spread_rng.unit() * 2.0f - 1.0f) * w.spread / 8192.0f;
        dir = forward + right * (std::cos(r) * a) + up * (std::sin(r) * b);
        normalize(dir);
    }
    const V3 muzzle = eye + forward * (14.0f * kUnit);
    // The game's own aim is level: it turns a new bullet up or down toward the
    // request's aim point itself, and flights_frame turns the bullet's own turn
    // into the view's - with the view's pitch in both, a shot 30 degrees up
    // flew 60 up, and one at the feet went over the shoulder.
    const V3 level{sy, 0.0f, cy};
    const V3 aim = muzzle + level * 100.0f;
    std::uint8_t* manager = ptr_at(kBulletManager);
    if (!manager || !build_request(w, at<std::uint32_t>(chr, 0x8), muzzle, aim, eye, game_yaw)) return;
    BbCallRegs regs{};
    regs.arg[0] = reinterpret_cast<std::uint64_t>(manager);
    regs.arg[1] = reinterpret_cast<std::uint64_t>(g_request);
    const int ok = g.api->call_guest(kEmit, &regs);
    ++g_shots;
    g_fire_ms = view.ms;
    g_fired.push_back({static_cast<std::int32_t>(w.bullet), muzzle, dir, right, up, view.ms});
    // The gun's own flash and sound, as its firing animation makes them.
    if (arms.current == 0) {
        spin_shot(chr, muzzle, view.ms);
    } else {
        const GunFx& fx = kGunFx[arms.current];
        flash(chr, fx.dummy, fx.effect, fx.hold_ms, view.ms);
        sound_at(chr, muzzle, fx.sound_type, fx.sound);
    }
    if (g_log || g_shots <= 3)
        g.log("quake: %s shot %llu: emitter %d (%s)", w.name, static_cast<unsigned long long>(g_shots),
              static_cast<int>(regs.ret), ok == 0 ? "made" : "the call failed");
}

// ---- the bullets in flight ---------------------------------------------------------
//
// The game starts a shot at the left hand's gun - hidden, at the hip - and
// sends it level along the hunter's facing, whatever the request says. So
// each new bullet of ours (the bullet manager's active list, *0x593e8d0 +0x8,
// next +0x868; the instance's owner +0x2fc, BulletParam row +0x2f4, position
// +0x10 and the last frame's +0x20, orientation quaternion +0x30, velocity
// +0x40 and +0x50) is taken, the frame it appears, to where a Quake 3 missile
// would be by then and turned along the view at its own speed - the shotgun's
// fifteen pellets each to a random point of Quake 3's pattern (700 units
// either way at 8192, the shotgun_spread option). And a rocket's or grenade's burst (its HitBulletID bullet) is
// where the blast that throws the player is.
constexpr std::uint64_t kBulletNext = 0x868;
constexpr std::size_t kBulletId = 0x2f4, kBulletOwner = 0x2fc;
struct Seen {
    std::uint64_t at;
    std::int32_t id;
};
std::vector<Seen> g_seen;  // bullets of ours already dealt with, while they last

bool rd(std::uint64_t a, void* out, std::size_t n) { return g.api->read(a, out, n) == 0; }

// The orientation that turns +z to `d`: the yaw about y after the pitch
// about x (the game's quaternion: x, y, z, w).
void face_quat(float* q, V3 d) {
    const float yaw = std::atan2(d.x, d.z), pitch = std::asin(std::clamp(d.y, -1.0f, 1.0f));
    const float cy = std::cos(yaw * 0.5f), sy = std::sin(yaw * 0.5f), cp = std::cos(pitch * 0.5f), sp = -std::sin(pitch * 0.5f);
    q[0] = cy * sp;
    q[1] = cp * sy;
    q[2] = -sy * sp;
    q[3] = cy * cp;
}

// Where +z goes under the game's quaternion (x, y, z, w): a bullet's
// direction in the frame it was turned in.
V3 quat_z(const float* q) {
    return {2.0f * (q[0] * q[2] + q[3] * q[1]), 2.0f * (q[1] * q[2] - q[3] * q[0]), 1.0f - 2.0f * (q[0] * q[0] + q[1] * q[1])};
}

void aim_bullet(std::uint8_t* world, std::uint64_t b, const Shot& shot, V3 dir, double ms, bool own_turn = true) {
    auto* f = reinterpret_cast<float*>(static_cast<std::uintptr_t>(b));
    const float speed = std::sqrt(f[16] * f[16] + f[17] * f[17] + f[18] * f[18]);  // +0x40
    // The bullet's own turn - none for most - from the shot's straight ahead
    // (the request's identity orientation) to the view's. A shotgun pellet's
    // is its emitter's, in the world's frame: it has none of its own here.
    if (own_turn) {
        const V3 local = quat_z(f + 12);  // +0x30
        V3 up = cross(dir, shot.right);
        normalize(up);
        V3 turned = shot.right * local.x + up * local.y + dir * local.z;
        if (normalize(turned) > 1e-3f) dir = turned;
    }
    // Where a Quake 3 missile is by now: its prestep and the time since the
    // shot, short of a wall in the way.
    // A wall or floor in the way stops it short by its own radius: a bullet
    // that starts inside a surface flies through it (a shot at the feet).
    float travel = speed * static_cast<float>(kPrestep + std::max(0.0, ms - shot.ms) / 1000.0);
    float frac = 1.0f;
    V3 n;
    std::int32_t id = 0;
    rd(b + kBulletId, &id, 4);
    const auto* row = id > 0 ? g.param<bb::params::Bullet>(static_cast<std::uint32_t>(id)) : nullptr;
    const float radius = row ? std::clamp(row->hitRadius, 0.0f, 1.0f) : 0.0f;
    if (world && travel > 0.0f && sphere_cast(world, shot.muzzle, dir * travel, 0.05f, frac, n))
        travel = std::max(0.0f, travel * frac - 0.1f - radius);
    const V3 p = shot.muzzle + dir * travel;
    for (const int off : {4, 8}) {  // +0x10 and +0x20
        f[off] = p.x;
        f[off + 1] = p.y;
        f[off + 2] = p.z;
    }
    face_quat(f + 12, dir);  // +0x30
    for (const int off : {16, 20}) {  // +0x40 and +0x50
        f[off] = dir.x * speed;
        f[off + 1] = dir.y * speed;
        f[off + 2] = dir.z * speed;
    }
}

// Quake 3's grenade in flight (g_missile.c): TR_GRAVITY at 800, and
// G_BounceMissile's EF_BOUNCE_HALF off whatever it meets - the velocity
// mirrored in the surface and kept at 0.65, the grenade come to rest once it
// meets a floor (normal over 0.2 up) slower than 40 units a second. The game's
// bullet, which passes through the map, is put where it is each frame; its
// life is the fuse, and a character it touches sets it off.
struct Grenade {
    std::uint64_t at;
    V3 pos;  // metres
    V3 vel;  // units a second
    bool resting = false;
    double ms;
    int bounces = 0;
};
std::vector<Grenade> g_grenades;
constexpr float kGrenadeRadius = 0.05f;

void grenade_step(std::uint8_t* world, Grenade& gr, float dt) {
    if (gr.resting) return;
    const V3 v0 = gr.vel;
    gr.vel.y -= kGravity * dt;
    const V3 delta = (v0 + gr.vel) * (0.5f * dt * kUnit);
    float frac = 1.0f;
    V3 n;
    if (!world || !sphere_cast(world, gr.pos, delta, kGrenadeRadius, frac, n) || dot(n, delta) >= 0.0f) {
        gr.pos = gr.pos + delta;
        return;
    }
    gr.pos = gr.pos + delta * frac;
    const V3 v = v0 + (gr.vel - v0) * frac;  // the velocity at the hit
    // Its knock on what it hits, as a blunt blow on stone sounds (the hit
    // sounds' z006001: a blow, 01 the street's stone) - Quake 3's
    // hgrenb1a/hgrenb2a - once it hits hard enough to be heard; a grenade
    // settling makes none.
    if (-dot(v, n) > 30.0f) named_sound("z006001001", gr.pos);
    gr.vel = (v - n * (2.0f * dot(v, n))) * 0.65f;
    ++gr.bounces;
    if (n.y > 0.2f && length(gr.vel) < 40.0f) {
        gr.vel = {};
        gr.resting = true;
        return;
    }
    gr.pos = gr.pos + n * kUnit;  // Quake 3 lifts it one unit off the surface
}

void grenades_frame(std::uint8_t* world, double ms) {
    for (Grenade& gr : g_grenades) {
        double dt = std::min((ms - gr.ms) / 1000.0, kMaxFrame);
        gr.ms = ms;
        const V3 before = gr.pos;
        while (dt > 1e-6) {
            const double h = std::min(dt, kMaxStep);
            grenade_step(world, gr, static_cast<float>(h));
            dt -= h;
        }
        auto* f = reinterpret_cast<float*>(static_cast<std::uintptr_t>(gr.at));
        const V3 last = gr.resting ? gr.pos : before;
        for (int k = 0; k < 3; ++k) {
            (&f[4])[k] = (&gr.pos.x)[k];   // +0x10
            (&f[8])[k] = (&last.x)[k];     // +0x20
            (&f[16])[k] = (&gr.vel.x)[k] * kUnit;  // +0x40
            (&f[20])[k] = (&gr.vel.x)[k] * kUnit;  // +0x50
        }
        if (!gr.resting) {
            V3 d = gr.vel;
            if (normalize(d) > 1e-3f) face_quat(f + 12, d);
        }
    }
}

void flights_frame(std::uint8_t* world, std::uint32_t owner, double ms) {
    std::uint8_t* manager = ptr_at(kBulletManager);
    std::uint64_t b = 0;
    if (!manager || !rd(reinterpret_cast<std::uint64_t>(manager) + 8, &b, 8)) return;
    std::vector<Seen> alive;
    for (int n = 0; b && n < 512; ++n) {
        std::uint32_t who = 0;
        std::int32_t id = 0;
        float pos[4] = {};
        std::uint64_t next = 0;
        if (!rd(b + kBulletOwner, &who, 4) || !rd(b + kBulletId, &id, 4) || !rd(b + 0x10, pos, sizeof pos) ||
            !rd(b + kBulletNext, &next, 8))
            break;
        if (who == owner) {
            alive.push_back({b, id});
            const bool known = std::any_of(g_seen.begin(), g_seen.end(), [&](const Seen& k) { return k.at == b && k.id == id; });
            if (!known) {
                const V3 at{pos[0], pos[1], pos[2]};
                // A burst: the blast.
                if (id == 701 || id == 711) blast(world, at, 100.0f, 120.0f, "rocket");
                else if (id == 722 || id == 731) blast(world, at, 100.0f, 150.0f, "grenade");
                else if (id == kPlasmaLooks[s.plasma].burst) blast(world, at, 15.0f, 20.0f, "plasma");
                // The shotgun's pellets (601, three from each of the
                // Blunderbuss's five emitters, its fixed 5 x 3 grid): each to
                // a random point of Quake 3's pattern (CG_ShotgunPattern:
                // crandom() * spread right and up at 8192), about the newest
                // shotgun shot's view.
                if (id == 601 || id == 611) {
                    for (auto it = g_fired.rbegin(); it != g_fired.rend(); ++it) {
                        if (it->bullet != 600) continue;
                        const float r = (g_spread_rng.unit() * 2.0f - 1.0f) * s.shotgun_spread;
                        const float u = (g_spread_rng.unit() * 2.0f - 1.0f) * s.shotgun_spread;
                        V3 dir = it->dir * 8192.0f + it->right * r + it->up * u;
                        normalize(dir);
                        aim_bullet(world, b, *it, dir, ms, false);
                        if (g_log)
                            g.log("quake: pellet %d at %+.2f %+.2f degrees", id, static_cast<double>(std::atan2(r, 8192.0f) / kDeg),
                                  static_cast<double>(std::atan2(u, 8192.0f) / kDeg));
                        break;
                    }
                }
                // A shot of ours, newest first.
                for (auto it = g_fired.rbegin(); it != g_fired.rend(); ++it) {
                    if (it->bullet == id) {
                        V3 dir = it->dir;
                        // The Blunderbuss's five emitters go straight along the
                        // view (the game fans them -12 to 12 degrees): the
                        // spread is their pellets' (601, above).
                        if (id == 720) {  // the grenade launcher aims a little up (forward[2] += 0.2)
                            dir.y += 0.2f;
                            normalize(dir);
                        }
                        aim_bullet(world, b, *it, dir, ms, id != 600);
                        if (id == 720) {
                            const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(b));
                            g_grenades.push_back({b, V3{f[4], f[5], f[6]}, dir * kGrenadeSpeed, false, ms});
                        }
                        if (id != 600) it->bullet = -1;  // one bullet a shot; the Blunderbuss's is five emitters
                        break;
                    }
                }
            }
        }
        b = next;
    }
    // A grenade whose bullet is gone has gone off.
    g_grenades.erase(std::remove_if(g_grenades.begin(), g_grenades.end(),
                                    [&](const Grenade& gr) {
                                        const bool live = std::any_of(alive.begin(), alive.end(),
                                                                      [&](const Seen& k) { return k.at == gr.at && k.id == 720; });
                                        if (!live && g_log)
                                            g.log("quake: grenade off after %d bounces at %.2f %.2f %.2f%s", gr.bounces,
                                                  static_cast<double>(gr.pos.x), static_cast<double>(gr.pos.y),
                                                  static_cast<double>(gr.pos.z), gr.resting ? ", at rest" : "");
                                        return !live;
                                    }),
                     g_grenades.end());
    grenades_frame(world, ms);
    g_seen = std::move(alive);
    g_fired.erase(std::remove_if(g_fired.begin(), g_fired.end(), [&](const Shot& sh) { return ms - sh.ms > 400.0; }), g_fired.end());
}

// Each frame: the change's drop, then the new gun raised.
void weapons_frame(const Frame& in, double ms) {
    const int from = arms.pending >= 0 ? arms.pending : arms.current;
    if (in.pick >= 0) select_weapon(in.pick, ms);
    else if (in.wheel <= -1.0f) select_weapon(from + 1, ms);
    else if (in.wheel >= 1.0f) select_weapon(from - 1, ms);
    if (arms.pending >= 0 && ms - arms.change_ms >= kDropMs) {
        arms.current = arms.pending;
        arms.pending = -1;
        g.log("quake: weapon %d, %s (%s)", arms.current + 1, kWeapons[arms.current].name, kWeapons[arms.current].gun);
    }
}

// A frame dump through the host's test requests (BBHOST_TEST_REQUESTS),
// named by the test's second.
void request_dump(double at) {
    const char* dir = std::getenv("BBHOST_TEST_REQUESTS");
    if (!dir) return;
    char name[600], body[1300];
    std::snprintf(name, sizeof(name), "%s/quake-%07.2f", dir, at);
    std::snprintf(body, sizeof(body), "dump %s/../quake-%05.2f.ppm\n", dir, at);
    const std::string tmp = std::string(name) + ".tmp", req = std::string(name) + ".req";
    if (FILE* fp = std::fopen(tmp.c_str(), "wb")) {
        std::fputs(body, fp);
        std::fclose(fp);
        std::rename(tmp.c_str(), req.c_str());
    }
    g.log("quake: test dump at %.2f s", at);
}

Frame read_input() {
    Frame f;
    BbInput in{};
    if (g.api->input(&in) == 0 && in.playing) {
        f.playing = true;
        f.cmd.forward = static_cast<float>(in.key[kKeyW]) - static_cast<float>(in.key[kKeyS]);
        f.cmd.side = static_cast<float>(in.key[kKeyD]) - static_cast<float>(in.key[kKeyA]);
        f.cmd.jump = in.key[kKeySpace] || (in.buttons & kMouseRight) || (in.pressed & kMouseRight);
        f.cmd.crouch = in.key[kKeyLCtrl] != 0;
        f.cmd.fire = (in.buttons & kMouseLeft) || (in.pressed & kMouseLeft);
        f.dx = in.dx;
        f.dy = in.dy;
        f.wheel = in.wheel;
        static bool was[kWeaponCount] = {};
        for (int i = 0; i < kWeaponCount; ++i) {
            const bool down = arms.digit_free[i] && in.key[kKeyDigit1 + i];
            if (down && !was[i]) f.pick = i;
            was[i] = down;
        }
    }
    if (!g_test.empty()) {
        if (g_test_start < 0.0) g_test_start = g_now;
        const double t = g_now - g_test_start;
        f.playing = true;
        for (TestStep& step : g_test) {
            if (step.what == "dump" && !step.done && t >= step.from) {
                step.done = true;
                request_dump(step.from);
                continue;
            }
            if (t < step.from || t >= step.to) continue;
            if (step.what == "fwd") f.cmd.forward = 1.0f;
            else if (step.what == "back") f.cmd.forward = -1.0f;
            else if (step.what == "left") f.cmd.side = -1.0f;
            else if (step.what == "right") f.cmd.side = 1.0f;
            else if (step.what == "jump") f.cmd.jump = true;
            else if (step.what == "crouch") f.cmd.crouch = true;
            else if (step.what == "fire") f.cmd.fire = true;
            else if (step.what == "yaw") view.yaw += step.arg * kDeg / 60.0f;
            else if (step.what == "face") view.yaw = step.arg * kDeg;
            else if (step.what == "pitch") view.pitch += step.arg * kDeg / 60.0f;
            else if (step.what == "weapnext" && !step.done) f.wheel = -1.0f, step.done = true;
            else if (step.what == "weapprev" && !step.done) f.wheel = 1.0f, step.done = true;
            else if (step.what == "weapon" && !step.done) f.pick = static_cast<int>(step.arg) - 1, step.done = true;
        }
    }
    return f;
}

// ---- the gun in view: Quake 3's viewmodel -------------------------------------------
//
// The hunter's equipment is drawn as parts: the PlayerIns's parts holder
// (+0x4d0) keeps 24 model inses from +0x28, slot 7 the left hand's weapon,
// each with a display entity whose +0x210 bit 2 says it is drawn. In first
// person every part but the left hand's weapon is hidden that way (the
// ChrIns's own draw switch would hide the gun with the body). The weapon in
// the left hand is made the Quake gun that is up: its EquipParamWeapon id is
// written into the hunter's left weapon (GameDataMan +0x8 -> +0x240 equipment,
// ids +0x64), and the game loads it as it does any change of weapon - a moment
// the first time - and the hunter's own weapon goes back whenever Quake lets
// go (and, should the game end first, at the next start: the plugin's folder
// keeps it). The gun is then put where Quake 3's CG_AddViewWeapon puts its
// own. A model's world matrix is worked out each frame by the scene graph's
// pull, 0x2166e10(node, out, link): the parent's matrix (the hand's dummy
// polygon) by the link's index, times the link's local matrix, into the
// model's holder (+0x20 -> +0x60); a hook there writes ours for the gun's.
constexpr std::size_t kPartsHolder = 0x4d0;
constexpr int kGunSlot = 7, kPartSlots = 24;
constexpr std::uint32_t kPartDrawn = 0x4;
constexpr std::uint64_t kPull = 0x2166e10;
constexpr std::uint64_t kGameDataMan = 0x593b130;

struct Viewmodel {
    bool hooked = false;
    std::uint64_t out = 0;          // the gun holder's matrix, which the pull writes ours into (0: nothing)
    float m[2][16] = {};            // the matrix, double-buffered for the pull (on whichever thread it runs)
    int cur = 0;
    std::uint8_t* hidden[kPartSlots] = {};  // display entities hidden here, shown again on release
    std::int32_t original = 0;      // the hunter's own left weapon while a Quake gun stands in (0: none)
    double kick_ms = -1e9;          // the last shot
} vm;
std::uint64_t g_pulls = 0;

int vm_pull(BbHookCtx* c, void*) {
    ++g_pulls;
    const std::uint64_t out = vm.out;
    if (!out || c->arg[1] != out) return 0;
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(out)), vm.m[vm.cur], sizeof vm.m[0]);
    c->ret = 1;
    return 1;
}

std::int32_t* left_weapon_id() {
    std::uint8_t* man = ptr_at(kGameDataMan);
    std::uint8_t* record = man ? at<std::uint8_t*>(man, 0x8) : nullptr;
    return record ? &at<std::int32_t>(record, 0x240 + 0x64) : nullptr;
}

bool is_quake_gun(std::int32_t id) {
    for (const QuakeWeapon& w : kWeapons)
        if (id >= static_cast<std::int32_t>(w.weapon) && id <= static_cast<std::int32_t>(w.weapon) + 1000) return true;
    return false;
}

std::string original_file() {
    const char* dir = g.api->plugin_dir ? g.api->plugin_dir("quake") : nullptr;
    return dir ? std::string(dir) + "/left_weapon.txt" : std::string();
}

// The hunter's left weapon given back (or found from the last session's file).
void vm_unequip() {
    std::int32_t* id = left_weapon_id();
    std::int32_t original = vm.original;
    const std::string file = original_file();
    if (!original && !file.empty()) {
        if (FILE* f = std::fopen(file.c_str(), "rb")) {
            if (std::fscanf(f, "%d", &original) != 1) original = 0;
            std::fclose(f);
        }
    }
    if (id && original && is_quake_gun(*id)) {
        g.log("quake: the hunter's left weapon back: %d (was %d)", original, *id);
        *id = original;
    }
    if (id && (vm.original || original)) {
        vm.original = 0;
        if (!file.empty()) std::remove(file.c_str());
    }
}

// The Quake gun `weapon` in the left hand (its +10 row). A weapon the
// player equips there meanwhile (the game's menu) becomes the one given back.
void vm_equip(int weapon) {
    std::int32_t* id = left_weapon_id();
    if (!id) return;
    const auto want = static_cast<std::int32_t>(kWeapons[weapon].weapon + 1000);
    if (*id == want) return;
    const std::string file = original_file();
    if (!is_quake_gun(*id)) {
        if (vm.original != *id) g.log("quake: the hunter's left weapon %d stands aside for the Quake guns", *id);
        vm.original = *id;
        if (FILE* f = file.empty() ? nullptr : std::fopen(file.c_str(), "wb")) {
            std::fprintf(f, "%d\n", vm.original);
            std::fclose(f);
        }
    } else if (!vm.original && !file.empty()) {
        // A Quake gun already there: a session that ended with it in hand;
        // the hunter's own is in the plugin's folder.
        if (FILE* f = std::fopen(file.c_str(), "rb")) {
            if (std::fscanf(f, "%d", &vm.original) != 1) vm.original = 0;
            std::fclose(f);
        }
    }
    *id = want;
}

std::uint8_t* part_disp(std::uint8_t* parts, int slot) {
    std::uint8_t* mi = at<std::uint8_t*>(parts, 0x28 + 8 * static_cast<std::size_t>(slot));
    return mi ? at<std::uint8_t*>(mi, 0x18) : nullptr;
}

// Every part hidden but the gun (or the gun too).
void vm_hide(std::uint8_t* chr, bool gun) {
    std::uint8_t* parts = at<std::uint8_t*>(chr, kPartsHolder);
    if (!parts) return;
    for (int k = 0; k < kPartSlots; ++k) {
        std::uint8_t* disp = part_disp(parts, k);
        if (!disp) continue;
        std::uint32_t& flags = at<std::uint32_t>(disp, 0x210);
        if (k == kGunSlot && gun) {
            if (vm.hidden[k] == disp) flags |= kPartDrawn;
            vm.hidden[k] = nullptr;
        } else if (flags & kPartDrawn) {
            flags &= ~kPartDrawn;
            vm.hidden[k] = disp;
        }
    }
}

void vm_show(std::uint8_t* chr) {
    std::uint8_t* parts = chr ? at<std::uint8_t*>(chr, kPartsHolder) : nullptr;
    for (int k = 0; k < kPartSlots; ++k) {
        if (parts && vm.hidden[k] && part_disp(parts, k) == vm.hidden[k]) at<std::uint32_t>(vm.hidden[k], 0x210) |= kPartDrawn;
        vm.hidden[k] = nullptr;
    }
}

// Where each gun sits: the grip's place in the view, in Quake units forward,
// right and down of the eye (Quake 3's tag_weapon in its hand model), and its
// kick - how far back and how many degrees up a shot throws it, and for how
// long (Quake 3's TORSO_ATTACK frames on the hand model: 6 at 15 fps, the
// machine gun's and plasma gun's started again each shot).
struct GunPose {
    float forward, right, down;
    float kick_back, kick_pitch, kick_ms;
};
constexpr GunPose kGunPoses[kWeaponCount] = {
    {7.0f, 8.0f, 9.0f, 1.5f, 1.0f, 100.0f},   // machine gun: the Gatling Gun
    {3.0f, 6.0f, 7.0f, 5.0f, 6.0f, 400.0f},   // shotgun: the Hunter Blunderbuss (5 units back, Kyo's)
    {8.0f, 9.0f, 4.0f, 5.0f, 3.0f, 400.0f},   // grenade launcher: the Cannon
    {8.0f, 7.0f, 9.0f, 4.0f, 4.0f, 400.0f},   // rocket launcher: the Church Cannon
    {9.0f, 6.0f, 8.0f, 1.5f, 1.0f, 100.0f},   // plasma gun: the Piercing Rifle
};

// cg_weapons.c's CG_CalculateWeaponPosition and CG_AddViewWeapon, from the
// view as drawn: the gun turned by the bob (roll and yaw swinging with the
// step, pitch with the speed), dropped on landing, drifting while idle; its
// grip at its own place along those turned axes, the view's cg_gun_x/y/z and
// a lower place at a wide fov; a shot's kick and a change's drop and raise on
// top. Then the matrix the pull hands the gun: rows x, y and z of the model
// (its barrel along -z) and its place.
void vm_place(const BbCamera& cam, float yaw, float pitch, float roll) {
    const float b = s.gun_bob;
    const float xyspeed = std::sqrt(pm.velocity.x * pm.velocity.x + pm.velocity.z * pm.velocity.z);
    const int cycle = static_cast<int>(pm.bob_cycle) & 255;
    const float bobfracsin = std::fabs(std::sin(static_cast<float>(cycle & 127) / 127.0f * kPi));
    const float scale = (cycle & 128) ? -xyspeed : xyspeed;
    // Degrees, Quake 3's signs (pitch down, yaw left, roll right positive).
    float a_pitch = xyspeed * bobfracsin * 0.005f * b;
    float a_yaw = scale * bobfracsin * 0.01f * b;
    float a_roll = scale * bobfracsin * 0.005f * b;
    const float idle = (xyspeed + 40.0f) * std::sin(static_cast<float>(view.ms) * 0.001f) * 0.01f * b;
    a_pitch += idle;
    a_yaw += idle;
    a_roll += idle;
    // The landing drop: a quarter of the view's dip.
    float drop = 0.0f;
    const double since_land = view.ms - view.land_time;
    if (since_land < kLandDeflectTime) drop = view.land_change * 0.25f * static_cast<float>(since_land / kLandDeflectTime);
    else if (since_land < kLandDeflectTime + kLandReturnTime)
        drop = view.land_change * 0.25f * static_cast<float>((kLandDeflectTime + kLandReturnTime - since_land) / kLandReturnTime);
    drop *= b;
    // A shot's kick: out in a quarter of its time, back in the rest.
    const int shown = arms.current;
    const GunPose& pose = kGunPoses[shown];
    float back = 0.0f;
    const double since_shot = view.ms - vm.kick_ms;
    if (since_shot >= 0.0 && since_shot < pose.kick_ms) {
        const float u = static_cast<float>(since_shot / pose.kick_ms);
        const float k = u < 0.25f ? u / 0.25f : 1.0f - (u - 0.25f) / 0.75f;
        back = pose.kick_back * k * b;
        a_pitch -= pose.kick_pitch * k * b;
    }
    // A change: down for WEAPON_DROPPING's 200 ms, up for WEAPON_RAISING's 250.
    float lower = 0.0f;
    const double since_change = view.ms - arms.change_ms;
    if (arms.pending >= 0) lower = static_cast<float>(std::min(1.0, since_change / kDropMs));
    else if (since_change < kDropMs + kRaiseMs) lower = 1.0f - static_cast<float>(std::max(0.0, since_change - kDropMs) / kRaiseMs);
    a_pitch += lower * 30.0f;
    // The turned axes: Quake 3's angles onto ours (yaw to the right, pitch up).
    const float gy = yaw - a_yaw * kDeg, gp = pitch - a_pitch * kDeg, gr = roll + a_roll * kDeg;
    const float cy = std::cos(gy), sy = std::sin(gy), cp = std::cos(gp), sp = std::sin(gp);
    const V3 f{sy * cp, sp, cy * cp};
    const V3 r0{cy, 0.0f, -sy};
    const V3 u0 = cross(f, r0);
    const V3 u = u0 * std::cos(gr) + r0 * std::sin(gr);
    const V3 r = r0 * std::cos(gr) - u0 * std::sin(gr);
    // The view's own axes for cg_gun_x/y/z.
    const V3 vr{cam.right[0], cam.right[1], cam.right[2]}, vu{cam.up[0], cam.up[1], cam.up[2]},
        vf{cam.forward[0], cam.forward[1], cam.forward[2]};
    const V3 eye{cam.position[0], cam.position[1], cam.position[2]};
    V3 p = eye + V3{0.0f, drop * kUnit, 0.0f};
    p = p + vf * (s.gun_x * kUnit) - vr * (s.gun_y * kUnit) + vu * (s.gun_z * kUnit);
    p = p + f * ((pose.forward - back) * kUnit) + r * (pose.right * kUnit) - u * ((pose.down + lower * 20.0f) * kUnit);
    // The gun's own field of view: drawn by the world's camera, it is spread
    // across and up the view by tan(view/2) / tan(gun/2), which puts every
    // point of it where a camera of the gun's fov would - the gun as a
    // viewmodel fov draws it, its depth (and so what it hides) unchanged.
    const float half_gun = std::atan(std::tan(s.gun_fov * 0.5f * kDeg) * 0.75f);
    const float spread = std::tan(cam.fov_y * 0.5f) / std::tan(half_gun);
    const auto fov = [&](V3 v) { return vr * (dot(v, vr) * spread) + vu * (dot(v, vu) * spread) + vf * dot(v, vf); };
    const V3 rows[4] = {fov(r * -1.0f), fov(u), fov(f * -1.0f), eye + fov(p - eye)};
    const int next = vm.cur ^ 1;
    float* m = vm.m[next];
    for (int k = 0; k < 4; ++k) {
        m[4 * k] = rows[k].x;
        m[4 * k + 1] = rows[k].y;
        m[4 * k + 2] = rows[k].z;
        m[4 * k + 3] = k == 3 ? 1.0f : 0.0f;
    }
    vm.cur = next;
}

// Each frame Quake has the player: the gun up in the left hand, the rest of
// the hunter hidden, and the gun's holder handed to the pull.
void vm_frame(std::uint8_t* chr, bool alive) {
    if (!vm.hooked) {
        vm.hooked = true;  // once the world is up: hooking at the boot moved the guest's heap window
        static const std::uint8_t prologue[16] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                  0x53, 0x48, 0x83, 0xec, 0x48, 0x48, 0x89, 0xd3};
        const int rc = g.api->hook2(kPull, prologue, sizeof prologue, &vm_pull, nullptr);
        g.log("quake: the gun's place in view %s", rc == 0 ? "hooked" : "not hooked (another eboot?)");
        if (rc != 0) vm.out = 0;
    }
    if (!s.draw_gun || !alive) {
        vm.out = 0;
        vm_hide(chr, false);
        return;
    }
    vm_equip(arms.pending >= 0 ? arms.pending : arms.current);
    vm_hide(chr, true);
    std::uint8_t* parts = at<std::uint8_t*>(chr, kPartsHolder);
    std::uint8_t* mi = parts ? at<std::uint8_t*>(parts, 0x28 + 8 * kGunSlot) : nullptr;
    std::uint8_t* holder = mi ? at<std::uint8_t*>(mi, 0x20) : nullptr;
    vm.out = holder ? reinterpret_cast<std::uint64_t>(holder) + 0x60 : 0;
}

void vm_release(std::uint8_t* chr) {
    vm.out = 0;
    vm_show(chr);
    vm_unequip();
}

// ---- the frame ---------------------------------------------------------------------

using Clock = std::chrono::steady_clock;
Clock::time_point g_last;
bool g_have_last = false;
bool g_active = false;       // Quake has the player
std::uint8_t* g_held = nullptr;  // the ChrIns whose switches are held
V3 g_written;                // where the player was last put
bool g_written_valid = false;
std::uint64_t g_frames = 0;
double g_next_log = 0.0;
// DeFRaG: the time not yet stepped (pmove_fixed's leftover), and the origin
// before the last step - the view is drawn between the two.
double g_carry = 0.0;
V3 g_prev_origin;
double g_off_dump = 0.0;  // QUAKE_OFF_AFTER: a frame dump due then

bool g_on_ladder = false;
// Off the rungs, still in the game's hands until its climb-off stops moving
// the hunter: when that began (-1 none), when they last moved, where they were.
double g_ladder_exit_ms = -1.0, g_ladder_still_ms = 0.0;
V3 g_ladder_last;
constexpr float kLadderStill = 0.002f;     // metres a frame that count as standing still
constexpr double kLadderSettleMs = 150.0;  // that long still and the climb-off is over
constexpr double kLadderExitMs = 3000.0;   // and never longer than this

// The barrels turn by the gun model's own animations (parts wp_a_1750's
// anibnd), which the game plays as the hunter's firing animations play (a150
// 52900 the spin-up, 52910 firing, 52920 the wind-down): each frame every part
// of the hunter is handed the hunter's animation and its time
// (sub_20a6e30(model ins, id, time, dt, default)), and a part with that
// animation plays it. Quake plays none of the hunter's, so the drawn gun (the
// left hand's part, slot 7) is handed the burst's instead.
constexpr std::uint64_t kModelAnim = 0x20a6e30;
constexpr double kSpinLoopMs = 933.3, kSpinDownMs = 1000.0;  // the firing and wind-down animations' lengths

int gun_spin(BbHookCtx* c, void*) {
    if (!g_active || arms.current != 0 || arms.pending >= 0) return 0;
    std::uint8_t* chr = player();
    std::uint8_t* parts = chr ? at<std::uint8_t*>(chr, kPartsHolder) : nullptr;
    if (!parts || c->arg[0] != reinterpret_cast<std::uint64_t>(at<std::uint8_t*>(parts, 0x28 + 7 * 8))) return 0;
    const double ms = view.ms;
    std::int32_t id;
    double t;
    // QUAKE_GUNANIM=<id>:<ms> (a test): the gun held in that animation at that time.
    static const char* hold = std::getenv("QUAKE_GUNANIM");
    if (hold) {
        id = std::atoi(hold);
        t = std::atof(std::strchr(hold, ':') ? std::strchr(hold, ':') + 1 : "0");
    } else if (spin.on) {
        // Firing turns them at once: the loop from the first shot, no spin-up.
        id = 150052910, t = std::fmod(ms - spin.start_ms, kSpinLoopMs);
    } else if (ms - spin.stop_ms < kSpinDownMs) {
        id = 150052920, t = ms - spin.stop_ms;
    } else {
        return 0;
    }
    c->arg[1] = static_cast<std::uint32_t>(id);
    c->xmm[0] = bb_f32(static_cast<float>(t / 1000.0));
    return 0;
}


// `moving` false (on a ladder) leaves the movement keys to the game.
void take_input(bool on, bool moving = true) {
    if (!on) {
        g.api->input_take(nullptr, 0, 0, 0, 0);
        return;
    }
    find_free_digits();
    std::uint16_t keys[sizeof(kTaken) / sizeof(kTaken[0]) + kWeaponCount];
    std::size_t n = 0;
    if (moving)
        for (const std::uint16_t k : kTaken) keys[n++] = k;
    for (int i = 0; i < kWeaponCount; ++i)
        if (arms.digit_free[i]) keys[n++] = static_cast<std::uint16_t>(kKeyDigit1 + i);
    // The wheel picks the weapon here: the game's camera does not zoom by it.
    g.api->input_take(keys, n, kMouseLeft | kMouseRight, 1, 1);
}

void release() {
    g_on_ladder = false;
    g_ladder_exit_ms = -1.0;
    if (g_held) hold_body(g_held, false, false);
    lift_dash(false);
    vm_release(g_held ? g_held : player());
    g_held = nullptr;
    if (g_active) take_input(false);
    g_active = false;
    g_written_valid = false;
    g_fired.clear();
    // The guns' flashes ended while the player they are on is still in the
    // world; after a load or a new player the game has done with them.
    if (!g.loading_screen() && player()) {
        spin_frame(nullptr, view.ms, true);
        flashes_frame(view.ms, true);
    }
    g_flashes.clear();
    spin = Spin{};
}

// The player found again (a load, a warp, a death, the switch turned on):
// the movement starts from where they are, at rest.
void begin(std::uint8_t* phys, const BbCamera& cam) {
    pm = Pm{};
    pm.origin = pm.last_ground = read_position(phys);
    // QUAKE_START=x,y,z (a test): the first time in, the player is put there.
    static bool started = false;
    if (const char* e = std::getenv("QUAKE_START"); e && !started) {
        V3 p;
        if (std::sscanf(e, "%f,%f,%f", &p.x, &p.y, &p.z) == 3) {
            pm.origin = pm.last_ground = p;
            place(phys, p);
        }
    }
    started = true;
    g_prev_origin = pm.origin;
    g_carry = 0.0;
    view.land_time = view.step_time = view.duck_time = view.damage_time = -1e9;
    view.eye = kStandEye;
    view.hp = -1;
    if (!view.primed) {
        // The view starts where the game's camera looked.
        view.yaw = std::atan2(cam.forward[0], cam.forward[2]);
        view.pitch = std::asin(std::clamp(cam.forward[1], -1.0f, 1.0f));
        view.primed = true;
        const float game_yaw = at<float>(phys, 0x1d4);
        g.log("quake: in at %.2f %.2f %.2f, the camera's yaw %.3f, the character's %.3f (the view's + pi: %.3f)",
              static_cast<double>(pm.origin.x), static_cast<double>(pm.origin.y), static_cast<double>(pm.origin.z),
              static_cast<double>(view.yaw), static_cast<double>(game_yaw),
              static_cast<double>(std::remainder(view.yaw + kPi, 2.0f * kPi)));
    }
    g_written = pm.origin;
    g_written_valid = true;
    if (const char* e = std::getenv("QUAKE_FILTER")) g_body_filter = static_cast<std::uint32_t>(std::strtoul(e, nullptr, 16));
}

void probe_equipment();
int probe_emit(BbHookCtx* c, void*);

// QUAKE_SOUNDS: sub_2bd3a80(handle out, name, point, ...), where the TAE's
// sounds and the effects' own go (FMOD's event by name).
int probe_sound(BbHookCtx* c, void*) {
    const auto* name = reinterpret_cast<const char*>(static_cast<std::uintptr_t>(c->arg[1]));
    const auto* p = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(c->arg[2]));
    if (p) g.log("quake: sound %.40s at %.2f %.2f %.2f", name ? name : "?", static_cast<double>(p[0]), static_cast<double>(p[1]),
                 static_cast<double>(p[2]));
    else g.log("quake: sound %.40s", name ? name : "?");
    return 0;
}

// Quake 3's SnapVector at the end of PmoveSingle: the velocity to whole
// units a second (rounded to nearest, as its x86 build does) - what makes a
// 125 fps jump reach higher than the arithmetic says.
void snap_velocity() {
    pm.velocity.x = std::nearbyint(pm.velocity.x);
    pm.velocity.y = std::nearbyint(pm.velocity.y);
    pm.velocity.z = std::nearbyint(pm.velocity.z);
}

// ---- the crosshair -----------------------------------------------------------
//
// RendMan's draw lists take the game's debug shapes (the follow camera's
// keep-out spheres among them) and are played after the scene - over its
// effects and post-processing, under the HUD and the menus. The current list's
// shape object (+0x20 the lists, their [+0x20] the current one at +0x10 + 8 *
// that, its +0x40) holds the look the next shape takes: colours at +0x20 and
// +0x30, modes at +0x18, +0x1c and +0x70, each change flagged at +0x8. Its
// lines are 0x135d8d0(lists, from, to), its spheres 0x135d810(lists, 3x4).
constexpr std::uint64_t kRendMan = 0x5940298;
constexpr std::uint64_t kDrawSphere = 0x135d810, kDrawLine = 0x135d8d0;
constexpr float kCrosshairAhead = 0.25f;  // metres in front of the eye: inside the body, nothing nearer

struct alignas(16) V4 {
    float x, y, z, w;
};

std::uint8_t* debug_lists() {
    std::uint8_t* rendman = ptr_at(kRendMan);
    return rendman ? at<std::uint8_t*>(rendman, 0x20) : nullptr;
}

std::uint8_t* debug_shape(std::uint8_t* lists) {
    std::uint8_t* list = at<std::uint8_t*>(lists, 0x10 + 8 * static_cast<std::int64_t>(at<std::int32_t>(lists, 0x20)));
    return list ? at<std::uint8_t*>(list, 0x40) : nullptr;
}

void debug_look(std::uint8_t* lists, V4 colour) {
    std::uint8_t* shape = debug_shape(lists);
    if (!shape) return;
    BbCallRegs r{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(shape);
    r.arg[1] = 0xffffffffu;
    g.api->call_guest(bn_of(at<std::uint64_t>(at<std::uint8_t*>(shape, 0), 0x28)), &r);  // begun
    shape = debug_shape(lists);
    if (!shape) return;
    auto set_i = [&](std::size_t off, std::int32_t v, std::uint8_t flag) {
        if (at<std::int32_t>(shape, off) == v) return;
        at<std::int32_t>(shape, off) = v;
        at<std::uint8_t>(shape, 0x8) |= flag;
    };
    set_i(0x70, 2, 0x80);
    set_i(0x1c, 1, 0x8);
    set_i(0x18, 2, 0x4);
    for (const auto [off, flag] : {std::pair<std::size_t, std::uint8_t>{0x20, 0x10}, {0x30, 0x20}}) {
        V4& c = at<V4>(shape, off);
        if (c.x == colour.x && c.y == colour.y && c.z == colour.z && c.w == colour.w) continue;
        c = colour;
        at<std::uint8_t>(shape, 0x8) |= flag;
    }
}

void debug_line(std::uint8_t* lists, V3 a, V3 b) {
    const V4 from{a.x, a.y, a.z, 1.0f}, to{b.x, b.y, b.z, 1.0f};
    BbCallRegs r{};
    r.arg[0] = reinterpret_cast<std::uint64_t>(lists);
    r.arg[1] = reinterpret_cast<std::uint64_t>(&from);
    r.arg[2] = reinterpret_cast<std::uint64_t>(&to);
    g.api->call_guest(kDrawLine, &r);
}

// A flat disc: a sphere of radius r across `right` and `up`, none along the view.
void debug_disc(std::uint8_t* lists, V3 c, V3 right, V3 up, V3 forward, float r) {
    const V3 x = right * r, y = up * r, z = forward * 1e-4f;
    const V4 rows[3] = {{x.x, y.x, z.x, c.x}, {x.y, y.y, z.y, c.y}, {x.z, y.z, z.z, c.z}};
    BbCallRegs regs{};
    regs.arg[0] = reinterpret_cast<std::uint64_t>(lists);
    regs.arg[1] = reinterpret_cast<std::uint64_t>(rows);
    g.api->call_guest(kDrawSphere, &regs);
}

// Quake 3's crosshair at the middle of the view: a dot in a ring, its size in
// the 640x480 screen's pixels (cg_crosshairSize), white.
void draw_crosshair(const BbCamera& cam) {
    if (!s.crosshair || s.crosshair_alpha <= 0.0f) return;
    std::uint8_t* lists = debug_lists();
    if (!lists) return;
    const V3 eye{cam.position[0], cam.position[1], cam.position[2]};
    const V3 right{cam.right[0], cam.right[1], cam.right[2]}, up{cam.up[0], cam.up[1], cam.up[2]};
    const V3 forward{cam.forward[0], cam.forward[1], cam.forward[2]};
    const V3 c = eye + forward * kCrosshairAhead;
    // Metres at that distance a 480-line screen's pixel spans.
    const float pixel = kCrosshairAhead * std::tan(cam.fov_y * 0.5f) / 240.0f;
    const float ring = s.crosshair_size * 0.5f * pixel * 0.75f, dot_r = s.crosshair_size * pixel * 0.03f;
    const float a = s.crosshair_alpha;
    // The ring: a halo of fainter circles round a solid one.
    constexpr int kSegments = 32;
    const float widths[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    for (const float w : widths) {
        debug_look(lists, {1.0f, 1.0f, 1.0f, a * (w == 0.0f ? 1.0f : std::fabs(w) < 1.0f ? 0.6f : 0.25f)});
        const float rr = ring + w * pixel * 0.5f;
        V3 prev = c + right * rr;
        for (int i = 1; i <= kSegments; ++i) {
            const float t = 2.0f * kPi * static_cast<float>(i) / kSegments;
            const V3 next = c + right * (rr * std::cos(t)) + up * (rr * std::sin(t));
            debug_line(lists, prev, next);
            prev = next;
        }
    }
    debug_look(lists, {1.0f, 1.0f, 1.0f, a});
    debug_disc(lists, c, right, up, forward, dot_r);
}

void camera(BbCamera& cam) {
    const Clock::time_point now = Clock::now();
    double dt = g_have_last ? std::chrono::duration<double>(now - g_last).count() : 0.0;
    g_last = now;
    g_have_last = true;
    dt = std::min(dt, kMaxFrame);
    g_now += dt;
    view.ms = g_now * 1000.0;

    if (g_off_dump > 0.0 && g_now >= g_off_dump) {
        g_off_dump = 0.0;
        request_dump(99.0);  // QUAKE_OFF_AFTER's: the hunter as the game has them again
    }
    std::uint8_t* chr = player();
    std::uint8_t* phys = physics_of(chr);
    std::uint8_t* world = collision_world();
    const std::int64_t hp = g.stat("hp", -1), max_hp = g.stat("max_hp", -1);
    static std::uint64_t calls = 0;
    if (g_log && (calls++ % 600) == 0 && !g_active) {
        g.log("quake: camera call %llu - enabled %d, player %p, physics %p, world %p, loading %d, hp %lld/%lld",
              static_cast<unsigned long long>(calls), s.enabled ? 1 : 0, static_cast<void*>(chr), static_cast<void*>(phys),
              static_cast<void*>(world), g.loading_screen() ? 1 : 0, static_cast<long long>(hp), static_cast<long long>(max_hp));
    }
    if (g_probe && chr && !g.loading_screen()) {
        static bool hooked = false;
        if (!hooked) {
            hooked = true;
            probe_equipment();
            static const std::uint8_t prologue[17] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                                      0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x48};
            g.log("quake: probe on the bullet emitters: %d %d",
                  g.api->hook2(0x1e03b50, prologue, sizeof prologue, &probe_emit, reinterpret_cast<void*>(1)),
                  g.api->hook2(0x1e03e50, prologue, sizeof prologue, &probe_emit, reinterpret_cast<void*>(2)));
        }
    }
    // QUAKE_POSLOG=1 (a test): where the player is, four times a second, the
    // game's walk too (QUAKE_OFF_AFTER) - to set a run beside the game's own.
    if (std::getenv("QUAKE_POSLOG") && phys && !g.loading_screen()) {
        static double next = 0.0;
        if (g_now >= next) {
            next = g_now + 0.25;
            const V3 p = read_position(phys);
            g.log("quake: pos %.2f %.2f %.2f (t %.2f)", static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z), g_now);
        }
    }
    // The Gatling Gun's barrels turning (gun_spin below), from the first frame in the world on.
    if (chr && !g.loading_screen()) {
        static bool hooked = false;
        if (!hooked) {
            hooked = true;
            static const std::uint8_t prologue[17] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                                      0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x18};
            if (g.api->hook2(kModelAnim, prologue, sizeof prologue, &gun_spin, nullptr) != 0)
                g.log("quake: the Gatling Gun's barrels will not turn (no hook on the model animation)");
        }
    }
    if (g_sounds && chr && !g.loading_screen()) {
        static bool hooked = false;
        if (!hooked) {
            hooked = true;
            static const std::uint8_t prologue[15] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                      0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x30};
            g.log("quake: QUAKE_SOUNDS: %d", g.api->hook2(0x2bd3a80, prologue, sizeof prologue, &probe_sound, nullptr));
        }
    }
    if (!s.enabled || !chr || !phys || !world || g.loading_screen() || max_hp <= 0) {
        if (g_active || g_held) release();
        // A Quake gun left in the hunter's hand by a session that ended early.
        static bool checked = false;
        if (!checked && chr && !g.loading_screen()) {
            checked = true;
            vm_unequip();
        }
        return;
    }
    if (g_held && g_held != chr) release();
    if (!g_active) {
        g_active = true;
        take_input(true);
        begin(phys, cam);
    }
    g_held = chr;
    const bool dead = hp == 0;
    // On a ladder (its state 0 and up; -1 off it) the climb is the game's:
    // its gravity, its keys, its placing - the view stays the eye, and Quake
    // takes the player back at the top or the bottom. The state ends as the
    // hunter lets go of the rungs, before the climb over the top has carried
    // them onto the ledge: the game keeps the player while its animation
    // still moves them, and Quake takes them back once they stand still.
    const int rungs = ladder_state(chr);
    static int rungs_was = -1;
    if (rungs != rungs_was) {
        if (g_log) g.log("quake: ladder state %d", rungs);
        rungs_was = rungs;
    }
    if (rungs >= 0) {
        g_ladder_exit_ms = -1.0;
    } else if (g_on_ladder && g_ladder_exit_ms < 0.0) {
        g_ladder_exit_ms = g_ladder_still_ms = view.ms;
    }
    const V3 now_at = read_position(phys);
    if (g_ladder_exit_ms >= 0.0 && length(now_at - g_ladder_last) > kLadderStill) g_ladder_still_ms = view.ms;
    g_ladder_last = now_at;
    const bool climbing_off = g_ladder_exit_ms >= 0.0 && view.ms - g_ladder_still_ms < kLadderSettleMs &&
                              view.ms - g_ladder_exit_ms < kLadderExitMs;
    if (g_ladder_exit_ms >= 0.0 && !climbing_off) g_ladder_exit_ms = -1.0;
    const bool ladder = rungs >= 0 || climbing_off;
    if (ladder != g_on_ladder) {
        g_on_ladder = ladder;
        take_input(true, !ladder);
        g.log("quake: %s a ladder", ladder ? "on" : "off");
    }
    // The body hidden part by part, the gun kept (vm_frame); with no gun
    // drawn, the whole character by its draw switch.
    hold_body(chr, !dead && !ladder, !s.draw_gun);
    ++g_frames;

    // QUAKE_OFF_AFTER=<s> (a test): the plugin lets go of the player after
    // that long in the world, as the switch turned off would.
    if (const char* e = std::getenv("QUAKE_OFF_AFTER"); e && g_test_start >= 0.0 && g_now - g_test_start > std::atof(e)) {
        s.enabled = false;
        g.log("quake: QUAKE_OFF_AFTER - the player is the game's again");
        release();
        g_off_dump = g_now + 2.0;
        return;
    }
    const Frame in = read_input();
    weapons_frame(in, view.ms);
    // A gun on its way down takes its flashes with it.
    if (arms.pending >= 0) {
        spin_frame(chr, view.ms, true);
        flashes_frame(view.ms, true);
    }
    const float per_count = 0.022f * s.sensitivity * kDeg;
    view.yaw += in.dx * per_count;
    view.pitch += (s.invert ? in.dy : -in.dy) * per_count;
    view.pitch = std::clamp(view.pitch, -88.0f * kDeg, 88.0f * kDeg);
    view.yaw = std::remainder(view.yaw, 2.0f * kPi);

    // Someone else moved the player: a warp, a load's placement, the game's
    // own step between frames (QUAKE_PLACE=adopt), a ladder.
    const V3 game = read_position(phys);
    if (g_written_valid) {
        const V3 moved = game - g_written;
        if (length(moved) > 2.0f) {
            begin(phys, cam);
        } else if (g_adopt || length(moved) > 0.25f) {
            pm.origin = game;
            g_prev_origin = game;
        }
    }

    // The shots in the air: ours turned to the view, and the blasts that
    // throw the player - before the movement, which they push.
    const std::uint32_t handle = at<std::uint32_t>(chr, 0x8);
    flights_frame(world, handle, view.ms);

    V3 draw_origin = pm.origin;
    if (ladder) {
        pm.origin = g_prev_origin = draw_origin = g_written = game;
        pm.velocity = {};
        g_written_valid = true;
    } else if (!dead && dt > 0.0) {
        Cmd cmd = in.cmd;
        if (!in.playing) cmd = Cmd{};
        const int eye_was = pm.eye;
        if (s.defrag) {
            // pmove_fixed: steps of exactly 8 ms, the velocity snapped after
            // each, the rest carried to the next frame.
            g_carry += dt;
            int n = 0;
            while (g_carry >= kMaxStep - 1e-9 && n < 16) {
                g_prev_origin = pm.origin;
                pmove_step(world, cmd, view.yaw, static_cast<float>(kMaxStep));
                snap_velocity();
                g_carry -= kMaxStep;
                ++n;
            }
            const float a = static_cast<float>(std::clamp(g_carry / kMaxStep, 0.0, 1.0));
            draw_origin = g_prev_origin + (pm.origin - g_prev_origin) * a;
        } else {
            const int steps = std::max(1, static_cast<int>(std::ceil(dt / kMaxStep)));
            const float h = static_cast<float>(dt / steps);
            for (int i = 0; i < steps; ++i) pmove_step(world, cmd, view.yaw, h);
            g_prev_origin = draw_origin = pm.origin;
        }
        if (pm.eye != eye_was) {
            view.duck_change = static_cast<float>(pm.eye - eye_was);
            view.duck_time = view.ms;
        }
        view.eye = pm.eye;
        if (pm.landed) {
            view.land_change = pm.land_change;
            view.land_time = view.ms;
            pm.landed = false;
        }
        body_sounds(chr);
        if (pm.stepped > 0.0f) {
            // CG_EntityEvent's EV_STEP: on top of a step not yet eased out.
            const double since = view.ms - view.step_time;
            const float old = since < kStepTime ? view.step_change * static_cast<float>((kStepTime - since) / kStepTime) : 0.0f;
            view.step_change = std::min(old + pm.stepped, kMaxStepChange);
            view.step_time = view.ms;
            pm.stepped = 0.0f;
        }
        place(phys, pm.origin);
        face(phys, view.yaw);
        g_written = pm.origin;
        g_written_valid = true;
        // The gun: held fire shoots at Quake 3's rate once the gun is up.
        if (cmd.fire && weapon_ready(view.ms) && view.ms >= g_next_shot_ms) {
            const V3 eye = pm.origin + V3{0.0f, static_cast<float>(pm.eye) * kUnit, 0.0f};
            fire(chr, eye, view.yaw, view.pitch, at<float>(phys, 0x1d4));
            vm.kick_ms = view.ms;
            g_next_shot_ms = view.ms + kWeapons[arms.current].refire_ms;
        }
    }
    flashes_frame(view.ms, false);
    spin_frame(chr, view.ms, dead);
    if (view.hp > 0 && hp >= 0 && hp < view.hp) damage_feedback(view.hp - hp, hp, max_hp);
    view.hp = hp;
    float yaw = 0.0f, pitch = 0.0f, roll = 0.0f;
    place_view(cam, dead, draw_origin, yaw, pitch, roll);
    vm_frame(chr, !dead);
    if (vm.out) vm_place(cam, yaw, pitch, roll);
    if (!dead) draw_crosshair(cam);

    if (g_log && g_now >= g_next_log) {
        g_next_log = g_now + 0.25;
        const float xy = std::sqrt(pm.velocity.x * pm.velocity.x + pm.velocity.z * pm.velocity.z);
        g.log("quake: t %.2f at %.2f %.2f %.2f (game %.2f %.2f %.2f) speed %.0f up %.0f %s%s yaw %.1f pitch %.1f casts %llu "
              "frames %llu hp %lld pulls %llu ladder %d",
              g_now, static_cast<double>(pm.origin.x), static_cast<double>(pm.origin.y), static_cast<double>(pm.origin.z),
              static_cast<double>(game.x), static_cast<double>(game.y), static_cast<double>(game.z), static_cast<double>(xy),
              static_cast<double>(pm.velocity.y), pm.on_ground ? "ground" : "air", pm.ducked ? " ducked" : "",
              static_cast<double>(view.yaw / kDeg), static_cast<double>(view.pitch / kDeg),
              static_cast<unsigned long long>(g_casts), static_cast<unsigned long long>(g_frames), static_cast<long long>(hp),
              static_cast<unsigned long long>(g_pulls), ladder_state(chr));
    }
}

// QUAKE_PROBE=1: every bullet emitter the game makes - sub_1e03b50(the bullet
// manager, the emit request), which takes it off the free list at +0x30 and
// fills it from the request - logged with the request's first 0x380 bytes and
// the player's handle, to read the request's layout off a real shot.

// The local player's equipment ids (GameDataMan 0x593b130 -> the local
// record +0x8 -> its equipment +0x240 -> the 16 item ids +0x64).
void probe_equipment() {
    std::uint8_t* man = ptr_at(0x593b130);
    std::uint8_t* record = man ? at<std::uint8_t*>(man, 0x8) : nullptr;
    if (!record) return;
    char line[400];
    int n = std::snprintf(line, sizeof(line), "quake: probe equipment ids:");
    for (int i = 0; i < 16; ++i)
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %d", at<std::int32_t>(record, 0x240 + 0x64 + 4 * i));
    g.log("%s", line);
    n = std::snprintf(line, sizeof(line), "quake: probe equipment refs:");
    for (int i = 0; i < 16; ++i)
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %08x", at<std::uint32_t>(record, 0x240 + 0x24 + 4 * i));
    g.log("%s", line);
}

int probe_emit(BbHookCtx* c, void* which) {
    // The player's own shots - a request that holds the player's handle - up
    // to ten, and the first three of anyone else's (an enemy's gun firing
    // first must not use them up).
    static int mine = 0, others = 0;
    const auto* req = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(c->arg[1]));
    std::uint8_t* chr = player();
    const std::uint32_t handle = chr ? at<std::uint32_t>(chr, 0x8) : 0u;
    int at_offset = -1;
    for (int k = 0; req && handle && k + 4 <= 0x380 && at_offset < 0; k += 4) {
        std::uint32_t v;
        std::memcpy(&v, req + k, 4);
        if (v == handle) at_offset = k;
    }
    if (at_offset >= 0 ? mine >= 10 : others >= 3) return 0;
    const int logged = at_offset >= 0 ? ++mine : ++others;
    g.log("quake: probe emit, %s %d (allocator %d), the player's handle %08x %s +0x%x: manager %llx request %llx",
          at_offset >= 0 ? "the player's" : "another's", logged, static_cast<int>(reinterpret_cast<std::uintptr_t>(which)),
          handle, at_offset >= 0 ? "at" : "not in it", at_offset >= 0 ? at_offset : 0,
          static_cast<unsigned long long>(c->arg[0]), static_cast<unsigned long long>(c->arg[1]));
    if (at_offset >= 0) {
        // Where the player stands and looks, to find the position and the
        // direction among the request's floats.
        std::uint8_t* phys = physics_of(chr);
        if (phys) {
            const V3 p = read_position(phys);
            g.log("quake: probe the player at %.3f %.3f %.3f facing yaw %.3f", static_cast<double>(p.x),
                  static_cast<double>(p.y), static_cast<double>(p.z), static_cast<double>(at<float>(phys, 0x1d4)));
        }
    }
    for (int row = 0; row < 0x380 && req; row += 32) {
        char line[200];
        int n = std::snprintf(line, sizeof(line), "quake: req +%03x:", row);
        for (int k = 0; k < 32; k += 4) {
            std::uint32_t v;
            std::memcpy(&v, req + row + k, 4);
            n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %08x", v);
        }
        g.log("%s", line);
    }
    return 0;
}

}  // namespace

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    if (!g.attach(api, 13)) return 1;
    if (!api->eboot_is_109()) return 1;
    read_options();
    parse_test();
    if (!g.on_camera(&camera)) {
        g.log("quake: the host has no camera hook; not loaded");
        return 1;
    }
    g.api->on_option("quake", [](const char* key, const char* value, void*) {
        const int plasma = s.plasma;
        read_options();
        // Another plasma: the rows written again from the game's (every_frame).
        if (s.plasma != plasma && g_guns_quake) guns_restore();
        g.log("quake: %s changed to %s (%s physics)", key, value ? value : "?", physics_name());
    }, nullptr);
    g.on_world_load([](std::uint32_t) { g_written_valid = false; });
    // The guns' rows, once the tables are there and whenever the switch moves.
    g.every_frame([] {
        static std::uint64_t frame = 0;
        if (frame++ % 30 != 0) return;
        if (s.enabled) guns_apply();
        else if (g_guns_quake) guns_restore();
    });
    g.log("quake: %s, %s physics, hold jump to hop %s, sensitivity %.1f, fov %.0f, gun fov %.0f at %.1f %.1f %.1f, plasma %s",
          s.enabled ? "on" : "off", physics_name(), s.auto_hop ? "on" : "off", static_cast<double>(s.sensitivity),
          static_cast<double>(s.fov), static_cast<double>(s.gun_fov), static_cast<double>(s.gun_x), static_cast<double>(s.gun_y),
          static_cast<double>(s.gun_z), s.plasma ? "Blacksky Eye" : "A Call Beyond");
    return 0;
}
