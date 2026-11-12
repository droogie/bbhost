// Over the Shoulder: a close third-person camera behind one of the hunter's
// shoulders, as Silent Hill 2's remake frames its hero - the hunter off to one
// side of the screen, the way ahead in the middle - with a key that swaps
// shoulders on the fly, a mode that keeps the hunter facing where the camera
// looks (the monocular's walk), and the dodge as a roll or a sidestep.
//
//   [plugins]
//   enable = ["shoulder"]
//   [shoulder]
//   camera = true          # the over-the-shoulder camera
//   shoulder = "right"     # right | left: the shoulder it starts over
//   swap_key = "Tab"       # swaps shoulders while playing (None: no key)
//   distance = 1.7         # metres behind the shoulder
//   offset = 0.55          # metres to the side
//   height = 0.15          # metres above the game's look-at point
//   fov = 0                # vertical field of view, degrees (0: the game's)
//   swap_time = 0.25       # seconds a swap takes
//   face_camera = false    # always face where the camera looks
//   dodge = 0              # 0 the roll, 1 a sidestep in the stick's direction
//
// The camera. The game keeps its own follow camera (ChrFollowCam, the camera
// manager's +0x60), which orbits the hunter, pulls in at walls and turns with
// the stick; the manager's blend of it and the cameras it owns (the
// monocular's aim, the lock-on's) is the player's camera, which on_camera
// hands us each frame before anything is drawn from it. Only its place is
// moved - never where it looks - so the stick, the lock-on and the aim behave
// as the game has them, the hunter's movement keeps following the camera's
// facing, and the view is drawn from the shoulder. The place is the follow
// camera's focus (the point it looks at, chased after the hunter) raised by
// the Height, moved sideways by the Side offset and back along the view by the
// Distance. The game's sphere cast with the follow camera's own filter (0x25,
// sub_1c090e0) tests the way out from the focus to the shoulder and then back
// to the place, so a wall pulls the camera in rather than letting it look
// through. Pulled in at once, it eases back out. When another camera has the
// player's camera - the monocular's - the shoulder gives way to it, as
// Isometric's view does.
//
// Facing the camera. The monocular puts the hunter in its precision-shooting
// state: ChrIns +0x1e2 bit 2, which the HKS env query 0x18 (精密射撃中か)
// reads first (0x1e13c68) - the state's other source, a SpEffect of
// stateInfo 15 (2000, the monocular's), counts in the SpEffect list's state
// counters (+0x1c8 +0x22) - and with it the game turns the hunter to where the
// camera looks, walks them by the stick as a strafe (measured: 1.6 m to the
// side with the facing held to the view's), makes the dodge a step and hands
// the player's camera to the monocular's aim camera, a first-person view
// with a crosshair. With Always face the camera on, the plugin sets that bit
// each frame while the hunter is in the world and clears it once the option or
// the plugin goes off; the shoulder camera then keeps the view, placed from
// the hunter (their feet and the follow camera's look-at height) and turned
// as the aim camera turns, at the follow camera's field of view, crosshair
// and all. While SpEffect 2000 is on - the monocular itself in use - the
// monocular's view has the screen.
//
// The dodge. Sidestep is the omni-roll the Isometric plugin carries (four-way
// quicksteps in the stick's direction, as the lock-on's, without a lock):
// the same sites (../iso/omni_roll.inc, from Kyo's Modern Movement Lite).
// While the plugin is on the slider decides: at Sidestep every site is the
// omni-roll's, at Roll its hooks are the game's own code again - even where
// the patch list's Modern Movement Lite or the Isometric plugin wrote them -
// and with the plugin off each site is put back as it was found. With Always
// face the camera on, the game's precision shooting makes every dodge a step
// whatever the slider says.
#include "bbhost_plugin.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN | BB_PLUGIN_LIVE_SWITCH,
    "shoulder",
    "Over the Shoulder",
    "1.0.0",
    "Kyo",
    "A close third-person camera over the hunter's right or left shoulder, swapped with a key; the hunter can always "
    "face where the camera looks, and the dodge can be a roll or a sidestep.",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, "h_camera", "Camera", "The camera keeps the game's turning, lock-on and aim; only where it stands moves.",
     nullptr},
    {BB_OPT_BOOL | BB_OPT_LIVE, "camera", "Over-the-shoulder camera",
     "The view from close behind one of the hunter's shoulders. Off: the game's own camera.", "true"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "shoulder", "Shoulder", "The shoulder the camera looks over.", "right", 0, 0, "right|left"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "swap_key", "Swap shoulder key",
     "Swaps shoulders while playing (keyboard and mouse). Pick a key nothing else is bound to.", "Tab", 0, 0,
     "Tab|Z|B|H|M|Mouse 4|Mouse 5|None"},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "distance", "Distance", "Metres behind the shoulder.", "1.7", 0.8, 4.0},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "offset", "Side offset", "Metres to the side of the hunter's middle.", "0.55", 0.0, 1.2},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "height", "Height", "Metres above the point the game's camera looks at.", "0.15", -0.5,
     1.0},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "fov", "Field of view", "Vertical field of view in degrees; 0 keeps the game's.", "0", 0,
     90},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "swap_time", "Swap time", "Seconds the camera takes to move to the other shoulder.", "0.25",
     0, 1},
    {BB_OPT_HEADING, "h_movement", "Movement", nullptr, nullptr},
    {BB_OPT_BOOL | BB_OPT_LIVE, "face_camera", "Always face the camera",
     "The hunter always faces where the camera looks and walks sideways and back by the stick, as with the monocular "
     "up (a walk: no running or sprinting), with its crosshair. The dodge is then always a step that way.",
     "false"},
    {BB_OPT_INT | BB_OPT_LIVE, "dodge", "Dodge (0 Roll, 1 Sidestep)",
     "0: the game's roll. 1: a quick sidestep in the stick's direction, as when locked on, without a lock. While the "
     "plugin is on this wins over the Modern Movement Lite patch.",
     "0", 0, 1},
    {0}};

namespace {

const BbHostApi* api = nullptr;

constexpr float kPi = 3.14159265358979f;

// ---- settings ----------------------------------------------------------------

enum class Key { Tab, Z, B, H, M, Mouse4, Mouse5, None };

struct Settings {
    bool on = false;  // the plugin's own switch (BB_PLUGIN_LIVE_SWITCH)
    bool camera = true;
    float shoulder = 1.0f;  // +1 right, -1 left
    Key swap_key = Key::Tab;
    float distance = 1.7f, offset = 0.55f, height = 0.15f, fov = 0.0f, swap_time = 0.25f;
    bool face_camera = false;
    int dodge = 0;
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

Key key_named(const char* v) {
    if (!v || !*v) return Key::Tab;
    static const struct {
        const char* name;
        Key key;
    } kKeys[] = {{"Tab", Key::Tab},       {"Z", Key::Z},           {"B", Key::B},       {"H", Key::H},
                 {"M", Key::M},           {"Mouse 4", Key::Mouse4}, {"Mouse 5", Key::Mouse5}, {"None", Key::None}};
    for (const auto& k : kKeys)
        if (std::strcmp(v, k.name) == 0) return k.key;
    return Key::Tab;
}

// The key's SDL scancode (USB HID usage), 0 for a mouse button or none.
std::uint16_t scancode(Key k) {
    switch (k) {
    case Key::Tab: return 43;
    case Key::Z: return 29;
    case Key::B: return 5;
    case Key::H: return 11;
    case Key::M: return 16;
    default: return 0;
    }
}

// BbInput's button bit for a mouse button, 0 for a key.
std::uint32_t mouse_bit(Key k) { return k == Key::Mouse4 ? 8u : k == Key::Mouse5 ? 16u : 0u; }

void read_options() {
    const char* p = api->config("plugins.shoulder");
    s.on = truthy(p);
    s.camera = flag("shoulder.camera", true);
    const char* side = api->config("shoulder.shoulder");
    s.shoulder = side && std::strcmp(side, "left") == 0 ? -1.0f : 1.0f;
    s.swap_key = key_named(api->config("shoulder.swap_key"));
    s.distance = number("shoulder.distance", 1.7f, 0.8f, 4.0f);
    s.offset = number("shoulder.offset", 0.55f, 0.0f, 1.2f);
    s.height = number("shoulder.height", 0.15f, -0.5f, 1.0f);
    s.fov = number("shoulder.fov", 0.0f, 0.0f, 90.0f);
    s.swap_time = number("shoulder.swap_time", 0.25f, 0.0f, 1.0f);
    s.face_camera = flag("shoulder.face_camera", false);
    s.dodge = static_cast<int>(number("shoulder.dodge", 0.0f, 0.0f, 1.0f) + 0.5f);
}

// ---- the game's things -------------------------------------------------------

constexpr std::uint64_t kWorldChrMan = 0x593e878;   // WorldChrMan*: +0x60 the main player's ChrIns
constexpr std::uint64_t kPhysicsSystem = 0x593d700;  // +0x28 the collision world the camera casts in
constexpr std::uint64_t kSphereCast = 0x1c090e0;     // (world, filter, from, delta, hit, normal, radius, fraction*) -> hit
constexpr std::uint32_t kCameraFilter = 0x25;        // the follow camera's own
constexpr std::int32_t kAimSpEffect = 2000;          // the monocular in use (stateInfo 15)

template <typename T>
T get(const std::uint8_t* p, std::size_t off) {
    T v;
    std::memcpy(&v, p + off, sizeof(v));
    return v;
}

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

// What we did to a site: nothing, wrote the omni-roll's bytes, or put the
// game's own back over another's omni-roll (the patch list's Modern Movement
// Lite, the Isometric plugin) so the slider's Roll holds.
enum class Mine : std::uint8_t { None, Wrote, Unwrote };
Mine omni_mine[kSites]{};
enum class Omni { AsFound, Sidestep, Roll };
Omni omni_now = Omni::AsFound;

bool site_is(const Site& t, const char* bytes) {
    std::uint8_t now[4096];
    return t.size <= sizeof(now) && api->read(t.address, now, t.size) == 0 && std::memcmp(now, bytes, t.size) == 0;
}

// Sidestep: every site the omni-roll's. Roll: the hooks the game's own. As
// found: what we changed put back - the plugin off.
void omni_roll(Omni want) {
    if (want == omni_now) return;
    int changed = 0, kept = 0, refused = 0;
    for (std::size_t i = 0; i < kSites; ++i) {
        const Site& t = kOmniRoll[i];
        const bool hook = t.address < kCaves;
        bool ok = true;
        if (want == Omni::Sidestep) {
            if (omni_mine[i] == Mine::Unwrote) {
                ok = api->patch(t.address, t.expect, t.bytes, t.size) == 0;
                if (ok) omni_mine[i] = Mine::None, ++changed;
            } else if (site_is(t, t.bytes)) {
                ++kept;
            } else {
                ok = api->patch(t.address, t.expect, t.bytes, t.size) == 0;
                if (ok) omni_mine[i] = Mine::Wrote, ++changed;
            }
        } else if (want == Omni::Roll) {
            if (!hook) continue;  // a cave nothing reaches without its hook
            if (site_is(t, t.bytes)) {
                ok = api->patch(t.address, t.bytes, t.expect, t.size) == 0;
                if (ok) omni_mine[i] = omni_mine[i] == Mine::Wrote ? Mine::None : Mine::Unwrote, ++changed;
            }
        } else {
            if (omni_mine[i] == Mine::Wrote && hook) {
                ok = api->patch(t.address, t.bytes, t.expect, t.size) == 0;
                if (ok) ++changed;
            } else if (omni_mine[i] == Mine::Unwrote) {
                ok = api->patch(t.address, t.expect, t.bytes, t.size) == 0;
                if (ok) ++changed;
            }
            if (ok) omni_mine[i] = Mine::None;
        }
        if (!ok) ++refused;
    }
    omni_now = want;
    api->log("shoulder: dodge %s (%d sites changed, %d already so, %d refused)",
             want == Omni::Sidestep ? "sidestep" : want == Omni::Roll ? "roll" : "as found", changed, kept, refused);
}

// ---- facing the camera (the monocular's precision shooting) ------------------

constexpr std::size_t kAimFlags = 0x1e2;  // ChrIns: bit 2 precision shooting
constexpr std::uint8_t kAimBit = 4;
bool g_aim_ours = false;            // the bit is set because we set it
std::uint8_t* g_aim_chr = nullptr;  // on this character

// Once a frame, on the main thread: the bit set while it should be, cleared
// once it should not (only when we set it).
void face_camera_frame() {
    const bool want = s.on && s.face_camera && !(api->loading_screen && api->loading_screen());
    std::uint8_t* chr = player();
    if (chr != g_aim_chr) g_aim_ours = false;  // a load: the character it was on is gone
    g_aim_chr = chr;
    if (!chr) return;
    if (want) {
        chr[kAimFlags] |= kAimBit;
        g_aim_ours = true;
    } else if (g_aim_ours) {
        chr[kAimFlags] &= static_cast<std::uint8_t>(~kAimBit);
        g_aim_ours = false;
    }
}

// The aim is ours and the monocular is not in use: the shoulder keeps the view.
bool aim_is_ours() { return g_aim_ours && api->sp_effect_has(nullptr, kAimSpEffect) == 0; }

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
    V3 focus{0, 0, 0};        // in the aim, the look-at point chased after the hunter
    bool have_focus = false;
    bool key_down = false;
    std::chrono::steady_clock::time_point last;
    bool started = false;
} view;

// The game's camera off its follow camera by this much is another camera's.
constexpr float kAwayPos = 0.3f, kAwayFov = 0.05f, kHandOver = 0.2f;  // m, rad, s
constexpr float kRadius = 0.2f;     // the camera's sphere against walls
constexpr float kEaseOut = 0.25f;   // seconds to ease back out after a wall
constexpr float kMinReach = 0.15f;  // metres it keeps behind the shoulder
constexpr float kChase = 0.06f;     // seconds the aim's look-at point takes to catch up

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

// The swap key's press (keyboard or mouse), once per press while playing.
void read_swap_key() {
    if (s.swap_key == Key::None) return;
    BbInput in;
    if (api->input(&in) != 0 || !in.playing) {
        view.key_down = false;
        return;
    }
    bool down = false, pressed = false;
    if (const std::uint16_t sc = scancode(s.swap_key)) {
        down = in.key[sc] != 0;
        pressed = down && !view.key_down;
    } else if (const std::uint32_t bit = mouse_bit(s.swap_key)) {
        down = (in.buttons & bit) != 0;
        pressed = (in.pressed & bit) != 0;
    }
    view.key_down = down;
    if (pressed) {
        swap_to(-view.side_want);
        if (api->size >= offsetof(BbHostApi, show_text) + sizeof(void*))
            api->show_text(view.side_want > 0 ? "Right shoulder" : "Left shoulder", 1.0f, 0.5f, 0.85f, 1.0f, BB_TEXT_CENTRE);
    }
}

// SHOULDER_LOG=1: every 8th frame, the camera, the hunter and the aim.
void trace(const BbCamera* c, V3 pos) {
    const V3 fwd = unit(v3(c->forward));
    static const bool logging = std::getenv("SHOULDER_LOG") != nullptr;
    static int frames = 0;
    if (logging && frames++ % 8 == 0) {
        float feet[3] = {}, facing = 0;
        api->player_position(feet, &facing);
        // The character's yaw faces (-sin, 0, -cos): half a turn from the view's.
        const float view_yaw = std::atan2(fwd.x, fwd.z);
        float off = facing + kPi - view_yaw;
        while (off > kPi) off -= 2 * kPi;
        while (off < -kPi) off += 2 * kPi;
        api->log("shoulder: side %.2f reach %.2f/%.2f away %.2f cam %.2f %.2f %.2f view yaw %.3f facing off %.3f aim %d "
                 "feet %.2f %.2f %.2f",
                 static_cast<double>(view.side), static_cast<double>(view.reach), static_cast<double>(s.distance),
                 static_cast<double>(view.away), static_cast<double>(pos.x), static_cast<double>(pos.y),
                 static_cast<double>(pos.z), static_cast<double>(view_yaw), static_cast<double>(off),
                 (player() ? (player()[kAimFlags] & kAimBit) : 0) | (api->sp_effect_has(nullptr, kAimSpEffect) << 4),
                 static_cast<double>(feet[0]), static_cast<double>(feet[1]),
                 static_cast<double>(feet[2]));
    }
}

void on_camera(BbCamera* c, void*) {
    if (!s.on || !s.camera) {
        view.started = false;
        view.reach = -1.0f;
        return;
    }
    if (api->loading_screen && api->loading_screen()) {
        view.reach = -1.0f;
        return;
    }
    const std::uint8_t* f = follow_cam(c);
    if (!f) return;
    const auto now = std::chrono::steady_clock::now();
    float dt = view.started ? std::chrono::duration<float>(now - view.last).count() : 0.0f;
    view.last = now;
    if (!(dt >= 0) || dt > 0.25f) dt = 0.25f;
    if (!view.started) {
        view.started = true;
        view.side = view.side_want = view.side_from = s.shoulder;
        view.swap_t = 1.0f;
        api->log("shoulder: camera on (%s shoulder, %.2f m back, %.2f m aside, %.2f m up)", s.shoulder > 0 ? "right" : "left",
                 static_cast<double>(s.distance), static_cast<double>(s.offset), static_cast<double>(s.height));
    }
    read_swap_key();

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

    // The shoulder, eased across on a swap.
    if (view.swap_t < 1.0f) {
        view.swap_t = s.swap_time > 0 ? view.swap_t + dt / s.swap_time : 1.0f;
        if (view.swap_t > 1.0f) view.swap_t = 1.0f;
        const float t = view.swap_t * view.swap_t * (3 - 2 * view.swap_t);
        view.side = view.side_from + (view.side_want - view.side_from) * t;
    } else {
        view.side = view.side_want;
    }

    const V3 fwd = unit(v3(c->forward));
    V3 right = v3(c->right);
    right.y = 0;  // the shoulder is beside the hunter, whatever the pitch
    right = unit(right);
    if (length(fwd) < 0.5f || length(right) < 0.5f) return;
    // The point looked at: the follow camera's, or in the aim (whose camera
    // sits in the hunter's head) the hunter's feet raised to the follow
    // camera's look-at height, chased.
    float feet[3] = {};
    const bool have_feet = api->player_position(feet, nullptr) == 0;
    V3 focus = v3(reinterpret_cast<const float*>(f + 0xd0));
    if (!aim) {
        if (have_feet && view.away <= 0.0f) {
            const float eye = focus.y - feet[1];
            if (eye > 0.5f && eye < 2.5f) view.eye = eye;
            if (c->fov_y > 0.1f && c->fov_y < 2.5f) view.fov = c->fov_y;
        }
        view.focus = focus;
        view.have_focus = true;
    } else if (have_feet) {
        const V3 target{feet[0], feet[1] + view.eye, feet[2]};
        if (!view.have_focus || length(target - view.focus) > 5.0f) {
            view.focus = target;
            view.have_focus = true;
        } else {
            view.focus = view.focus + (target - view.focus) * (1.0f - std::exp(-dt / kChase));
        }
        focus = view.focus;
    }
    const V3 pivot = focus + V3{0, s.height, 0};
    const V3 aside = right * (s.offset * view.side);
    V3 shoulder = pivot + aside;
    float want = s.distance;

    // Walls: out to the shoulder, then back from it.
    if (std::uint8_t* world = collision_world()) {
        float frac = 1.0f;
        if (length(aside) > 1e-3f && sphere_cast(world, pivot, aside, kRadius, frac)) shoulder = pivot + aside * frac;
        if (sphere_cast(world, shoulder, fwd * -want, kRadius, frac)) want *= frac;
    }
    if (want < kMinReach) want = kMinReach;
    // In at once, out eased.
    if (view.reach < 0 || want < view.reach) {
        view.reach = want;
    } else {
        const float k = 1.0f - std::exp(-dt / kEaseOut);
        view.reach += (want - view.reach) * k;
    }
    V3 pos = shoulder - fwd * view.reach;

    // Part way to the other camera.
    if (view.away > 0) {
        const float t = view.away * view.away * (3 - 2 * view.away);
        pos = pos + (v3(c->position) - pos) * t;
    }
    c->position[0] = pos.x;
    c->position[1] = pos.y;
    c->position[2] = pos.z;
    if (s.fov > 0) {
        const float ours = s.fov * kPi / 180.0f;
        c->fov_y = ours + (c->fov_y - ours) * view.away;
    } else if (aim && view.fov > 0) {
        c->fov_y = view.fov;
    }
    trace(c, pos);
}

// ---- the switches ------------------------------------------------------------

void apply() {
    omni_roll(!s.on ? Omni::AsFound : s.dodge == 1 ? Omni::Sidestep : Omni::Roll);
    face_camera_frame();  // at once: the host may not call on_frame while the plugin is off
    if (s.swap_key != Key::None) {
        if (const std::uint16_t sc = scancode(s.swap_key); sc && api->key_bound && api->key_bound(sc))
            api->log("shoulder: the swap key is also bound to a game action; it presses both");
    }
}

void on_frame(void*) { face_camera_frame(); }

void on_option(const char* key, const char* value, void*) {
    const float was = s.shoulder;
    read_options();
    if (key && std::strcmp(key, "shoulder") == 0 && s.shoulder != was) swap_to(s.shoulder);
    apply();
    api->log("shoulder: %s = %s", key && *key ? key : "(plugin)", value ? value : "?");
}

}  // namespace

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* host) {
    api = host;
    if (host->version < 13 || host->size < offsetof(BbHostApi, key_bound) + sizeof(void*)) {
        host->log("shoulder: needs plugin API 13 (on_camera); this host has %u", host->version);
        return 1;
    }
    if (!host->eboot_is_109()) {
        host->log("shoulder: not the 1.09 eboot; off");
        return 1;
    }
    read_options();
    host->on_option("shoulder", &on_option, nullptr);
    if (host->on_camera(&on_camera, nullptr) != 0) host->log("shoulder: the host has no camera hook; the camera stays the game's");
    host->on_frame(&on_frame, nullptr);
    apply();
    host->log("shoulder: Over the Shoulder %s (camera %s, %s shoulder, face the camera %s, dodge %s)", s.on ? "on" : "off",
              s.camera ? "on" : "off", s.shoulder > 0 ? "right" : "left", s.face_camera ? "on" : "off",
              s.dodge ? "sidestep" : "roll");
    return 0;
}
