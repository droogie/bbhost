// Photo mode: a free camera over the world, for pictures. Based on Nox's
// bbphoto, rebuilt on bbhost's plugin API: nothing in it writes the game's code
// or reads an address of its own.
//
// Hold Cross and press L3, or press P, to start it; the same again (or Escape)
// ends it. Three modes:
//   Frozen  the world held still (world_pause) while the camera flies; Cross
//           (N on the keyboard) lets it run two more updates - a frame advance.
//   Static  the camera stays where it is and the hunter plays on: the
//           controller and the keys are the game's again, but for photo mode's
//           own.
//   Live    the camera flies while the world runs; the hunter stands, since
//           the camera has the controls (pad_take, input_take).
// The left stick (WASD) moves the camera the way it looks, the right stick (the
// mouse) turns it, L2 and R2 (Left Ctrl and Space) take it down and up, L1 and
// R1 (the right and left mouse buttons) make it slower and faster. The view is
// the scene's camera (on_view): the game's own - the one the follow camera
// keeps and the controls move by - is untouched, so ending photo mode puts the
// game's view back as it was. The game's HUD, menus and prompts are left out
// of the picture while it runs (hud_hide).
//
// Only with nobody else in the world (world_state): it does not start while
// another player is in it or on the way, nor in another player's world, and it
// ends as soon as one is coming - when matchmaking finds one (the game hosts its
// session from then on, before they join the room) or when one joins the room -
// so the world runs before their game needs it. It also ends at a loading
// screen, a cutscene or one of the game's menus.
#include "bbhost_plugin.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

namespace {

constexpr const char* kName = "photo_mode";
constexpr float kPi = 3.14159265358979f;

const BbHostApi* api = nullptr;

enum Mode : int { kFrozen = 0, kStatic = 1, kLive = 2 };

// SDL scancodes (the USB HID usages BbInput's keys are indexed by).
constexpr std::uint16_t kKeyA = 4, kKeyD = 7, kKeyN = 17, kKeyP = 19, kKeyS = 22, kKeyW = 26;
constexpr std::uint16_t kKeyEscape = 41, kKeyTab = 43, kKeySpace = 44;
constexpr std::uint16_t kKeyLCtrl = 224, kKeyLShift = 225, kKeyLAlt = 226, kKeyRShift = 229, kKeyRAlt = 230;
// The mouse's buttons in BbInput.buttons.
constexpr std::uint32_t kMouseLeft = 1, kMouseRight = 2;

// What the camera does with the controls.
constexpr float kSpeeds[] = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f};  // metres a second
constexpr int kSpeedCount = static_cast<int>(sizeof(kSpeeds) / sizeof(kSpeeds[0]));
constexpr int kSpeedStart = 3;
constexpr float kTurnRate = 2.2f;        // radians a second at a full stick, at a 60 degree field of view
constexpr float kMouseTurn = 0.0035f;    // radians a count, as Dark Souls III's mouse (at 60 degrees)
constexpr float kPitchLimit = 1.53f;     // short of straight up and down
constexpr std::uint32_t kStepUpdates = 2;
// The world up this long before photo mode starts: the loading screen's page
// fades out over the world's first frames, and a held world would hold it there.
constexpr float kWorldSettle = 2.0f;

struct Camera {
    float pos[3] = {0, 0, 0};
    float yaw = 0, pitch = 0, roll = 0;  // radians: yaw about the world's up, pitch below the horizon positive
    float fov = 43.0f * kPi / 180.0f;    // vertical, radians
    float right_sign = 1.0f;             // the game's basis: right = up x forward, or its opposite
    float up_sign = 1.0f;
};

struct State {
    bool on = false;
    int mode = kFrozen;
    bool seed = false;          // the next on_view takes the game's view as the camera's start
    bool seeded = false;
    Camera cam;
    float start_pos[3] = {0, 0, 0};  // where the camera started, for the log
    int speed = kSpeedStart;
    bool cross_armed = false;   // Cross let go since the start (it was held to start)
    std::chrono::steady_clock::time_point last;
    bool have_last = false;
    float look_x = 0, look_y = 0;  // the mouse's counts since the last frame
    bool hud = false;              // the game's HUD shown in photo mode
    bool world_up = false;         // a world in and no loading screen, since world_since
    std::chrono::steady_clock::time_point world_since;
} st;

void normalise(float* v) {
    const float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 1e-6f) {
        for (int i = 0; i < 3; ++i) v[i] /= l;
    }
}

void cross(const float* a, const float* b, float* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

float dot(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// The camera's basis from its angles, turned the way the game's own is.
void basis(const Camera& c, float* right, float* up, float* fwd) {
    const float sy = std::sin(c.yaw), cy = std::cos(c.yaw), sp = std::sin(c.pitch), cp = std::cos(c.pitch);
    fwd[0] = sy * cp;
    fwd[1] = -sp;
    fwd[2] = cy * cp;
    float u[3] = {sy * sp, cp, cy * sp};
    float r[3];
    cross(u, fwd, r);
    normalise(r);
    for (int i = 0; i < 3; ++i) {
        r[i] *= c.right_sign;
        u[i] *= c.up_sign;
    }
    // The roll: right and up turned about forward.
    const float cr = std::cos(c.roll), sr = std::sin(c.roll);
    for (int i = 0; i < 3; ++i) {
        right[i] = r[i] * cr + u[i] * sr;
        up[i] = u[i] * cr - r[i] * sr;
    }
}

// The camera's start: the game's view as it is drawn this frame.
void seed_from(const BbCamera* view) {
    Camera& c = st.cam;
    float f[3] = {view->forward[0], view->forward[1], view->forward[2]};
    normalise(f);
    c.yaw = std::atan2(f[0], f[2]);
    c.pitch = std::asin(std::clamp(-f[1], -1.0f, 1.0f));
    c.pitch = std::clamp(c.pitch, -kPitchLimit, kPitchLimit);
    c.roll = 0;
    for (int i = 0; i < 3; ++i) c.pos[i] = st.start_pos[i] = view->position[i];
    c.fov = view->fov_y > 0.05f && view->fov_y < 3.0f ? view->fov_y : c.fov;
    // Which way the game's right and up point against ours.
    c.right_sign = c.up_sign = 1.0f;
    float r[3], u[3], fw[3];
    basis(c, r, u, fw);
    c.right_sign = dot(r, view->right) < 0 ? -1.0f : 1.0f;
    c.up_sign = dot(u, view->up) < 0 ? -1.0f : 1.0f;
    st.seeded = true;
}

void on_view(BbCamera* view, void*) {
    if (!st.on) return;
    if (st.seed) {
        st.seed = false;
        seed_from(view);
        api->log("photo mode: the camera starts at %.2f %.2f %.2f, field of view %.1f degrees", static_cast<double>(st.cam.pos[0]),
                 static_cast<double>(st.cam.pos[1]), static_cast<double>(st.cam.pos[2]), static_cast<double>(st.cam.fov * 180.0f / kPi));
    }
    if (!st.seeded) return;
    float r[3], u[3], f[3];
    basis(st.cam, r, u, f);
    for (int i = 0; i < 3; ++i) {
        view->right[i] = r[i];
        view->up[i] = u[i];
        view->forward[i] = f[i];
        view->position[i] = st.cam.pos[i];
    }
    view->right[3] = view->up[3] = view->forward[3] = 0.0f;
    view->position[3] = 1.0f;
    view->fov_y = st.cam.fov;
}

// What photo mode takes from the game in each mode.
void take_controls() {
    if (!st.on) {
        api->pad_take(0, 0);
        api->input_take(nullptr, 0, 0, 0, 0);
        return;
    }
    // Photo mode's own keys always (Escape ends it, so the game's menu does not
    // open over the picture); the camera's too while it flies, when Shift and
    // Alt (the strong attack and the walk) do nothing either.
    static const std::uint16_t own[] = {kKeyP, kKeyEscape, kKeyTab};
    static const std::uint16_t flying[] = {kKeyP,     kKeyEscape, kKeyTab,    kKeyN,      kKeyW,    kKeyA,   kKeyS, kKeyD,
                                           kKeySpace, kKeyLCtrl,  kKeyLShift, kKeyRShift, kKeyLAlt, kKeyRAlt};
    if (st.mode == kStatic) {
        // The hunter plays: the controller is the game's but for Options (the
        // game's menu would open over the picture) and the touchpad's.
        api->pad_take(BB_PAD_OPTIONS | BB_PAD_TOUCHPAD, 0);
        api->input_take(own, sizeof(own) / sizeof(own[0]), 0, 0, 0);
    } else {
        api->pad_take(0xffffffffu, BB_PAD_LEFT_STICK | BB_PAD_RIGHT_STICK | BB_PAD_TRIGGERS);
        api->input_take(flying, sizeof(flying) / sizeof(flying[0]), kMouseLeft | kMouseRight, 1, 1);
    }
}

void apply_mode() {
    api->world_pause(kName, st.on && st.mode == kFrozen ? 1 : 0);
    take_controls();
}

const char* mode_name(int m) { return m == kFrozen ? "Frozen" : m == kStatic ? "Static" : "Live"; }

void set_mode(int m) {
    if (m == st.mode) return;
    st.mode = m;
    if (st.on) {
        apply_mode();
        api->log("photo mode: %s", mode_name(m));
    }
}

void start() {
    st.on = true;
    st.seed = true;
    st.seeded = false;
    st.mode = kFrozen;  // it always starts with the world held
    st.cross_armed = false;
    st.have_last = false;
    st.look_x = st.look_y = 0;
    apply_mode();
    api->hud_hide(kName, st.hud ? 0 : 1);
    api->log("photo mode: on (%s)", mode_name(st.mode));
}

void stop(const char* why) {
    if (!st.on) return;
    st.on = false;
    float d2 = 0;
    for (int i = 0; i < 3; ++i) d2 += (st.cam.pos[i] - st.start_pos[i]) * (st.cam.pos[i] - st.start_pos[i]);
    const bool seeded = st.seeded;
    st.seeded = false;
    api->world_pause(kName, 0);
    take_controls();
    api->hud_hide(kName, 0);
    if (seeded)
        api->log("photo mode: off (%s); the camera ended %.2f m from where it started", why, static_cast<double>(std::sqrt(d2)));
    else
        api->log("photo mode: off (%s)", why);
}

// Why photo mode cannot run now; nullptr when it can.
const char* refused(const BbWorldState& w) {
    if (!(w.flags & BB_WORLD_IN) || (w.flags & BB_WORLD_LOADING)) return "no world";
    if (!st.on && std::chrono::duration<float>(std::chrono::steady_clock::now() - st.world_since).count() < kWorldSettle)
        return "the world is still coming in";
    if (w.flags & BB_WORLD_CUTSCENE) return "a cutscene";
    if (w.flags & BB_WORLD_MENU) return "a menu is open";
    if (w.flags & BB_WORLD_VISITING) return "in another player's world";
    if (w.others) return "another player is in the world";
    // A player on the way: in the world's room already, or matched to come -
    // the game hosts its session from the match on, before they join the room.
    if (w.arriving || w.session == BB_SESSION_HOSTING || w.session == BB_SESSION_HOST) return "a player is on the way";
    if (w.session == BB_SESSION_LEAVING) return "the session is closing";
    return nullptr;
}

// What the player is told when photo mode will not start, for refused()'s reasons.
const char* refusal_message(const char* why) {
    if (!std::strcmp(why, "a menu is open")) return "Photo mode: close the menu first";
    if (!std::strcmp(why, "a cutscene")) return "Photo mode: not during a cutscene";
    if (!std::strcmp(why, "no world") || !std::strcmp(why, "the world is still coming in"))
        return "Photo mode: not while the world loads";
    return "Photo mode needs the world to yourself";
}

float axis_dead(float v) { return std::fabs(v) < 0.08f ? 0.0f : v; }

void fly(const BbPad& pad, const BbInput& in, bool keys_ok, float dt) {
    Camera& c = st.cam;
    // Speed: L1 and R1, or the mouse's buttons.
    std::uint32_t clicks = keys_ok ? in.pressed : 0;
    if ((pad.pressed & BB_PAD_R1) || (clicks & kMouseLeft)) st.speed = std::min(st.speed + 1, kSpeedCount - 1);
    if ((pad.pressed & BB_PAD_L1) || (clicks & kMouseRight)) st.speed = std::max(st.speed - 1, 0);
    const float speed = kSpeeds[st.speed];
    // Turning: the right stick, and the mouse's counts, slower when zoomed in.
    const float zoom = c.fov / (60.0f * kPi / 180.0f);
    const float rx = axis_dead(pad.rx), ry = axis_dead(pad.ry);
    float yaw = rx * kTurnRate * zoom * dt, pitch = ry * kTurnRate * zoom * dt;
    yaw += st.look_x * kMouseTurn * zoom;
    pitch += st.look_y * kMouseTurn * zoom;
    st.look_x = st.look_y = 0;
    c.yaw += yaw * c.right_sign;
    c.pitch = std::clamp(c.pitch + pitch, -kPitchLimit, kPitchLimit);
    // Moving: the left stick and WASD along where the camera looks, L2 and R2
    // (Left Ctrl and Space) along the world's up.
    float mx = axis_dead(pad.lx), mz = -axis_dead(pad.ly), my = pad.r2 - pad.l2;
    if (keys_ok) {
        mx += (in.key[kKeyD] ? 1.0f : 0.0f) - (in.key[kKeyA] ? 1.0f : 0.0f);
        mz += (in.key[kKeyW] ? 1.0f : 0.0f) - (in.key[kKeyS] ? 1.0f : 0.0f);
        my += (in.key[kKeySpace] ? 1.0f : 0.0f) - (in.key[kKeyLCtrl] ? 1.0f : 0.0f);
    }
    mx = std::clamp(mx, -1.0f, 1.0f);
    mz = std::clamp(mz, -1.0f, 1.0f);
    my = std::clamp(my, -1.0f, 1.0f);
    float r[3], u[3], f[3];
    Camera level = c;
    level.roll = 0;
    basis(level, r, u, f);
    for (int i = 0; i < 3; ++i) c.pos[i] += (f[i] * mz + r[i] * mx) * speed * dt;
    c.pos[1] += my * speed * dt;
}

void on_frame(void*) {
    BbWorldState w{};
    w.size = sizeof(w);
    if (api->world_state(&w) != 0) return;
    const bool world_up = (w.flags & BB_WORLD_IN) && !(w.flags & BB_WORLD_LOADING);
    if (world_up && !st.world_up) st.world_since = std::chrono::steady_clock::now();
    st.world_up = world_up;
    BbPad pad{};
    pad.size = sizeof(pad);
    api->pad(&pad);
    BbInput in{};
    const bool have_keys = api->input(&in) == 0;
    // The keyboard's presses: on the frame a key goes down.
    static std::uint8_t prev_keys[512];
    auto key_down = [&](std::uint16_t k) { return have_keys && in.key[k] && !prev_keys[k]; };
    const bool p_down = key_down(kKeyP), escape_down = key_down(kKeyEscape), n_down = key_down(kKeyN);
    if (have_keys) std::memcpy(prev_keys, in.key, sizeof(prev_keys));

    // Start and end: hold Cross and press L3, or P; Escape ends it too.
    const bool chord = (pad.buttons & BB_PAD_CROSS) && (pad.pressed & BB_PAD_L3);
    const bool toggle = chord || p_down;
    if (!st.on) {
        if (toggle) {
            if (const char* why = refused(w)) {
                api->log("photo mode: not started - %s", why);
                api->show_message(refusal_message(why), 3.0f);
            } else {
                start();
            }
        }
    } else if (const char* why = refused(w)) {
        stop(why);
        const bool player = !std::strcmp(why, "a player is on the way") || !std::strcmp(why, "another player is in the world");
        api->show_message(player ? "Photo mode ended: another player is coming" : "Photo mode ended", 2.0f);
    } else if (toggle || escape_down) {
        stop("asked");
    }
    if (!st.on) {
        st.have_last = false;
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    float dt = st.have_last ? std::chrono::duration<float>(now - st.last).count() : 0.0f;
    st.last = now;
    st.have_last = true;
    dt = std::clamp(dt, 0.0f, 0.25f);  // a hitch moves the camera no further than a quarter second's way

    // The frame advance: Cross (once let go since the start) or N.
    if (!(pad.buttons & BB_PAD_CROSS)) st.cross_armed = true;
    const bool step = (st.cross_armed && (pad.pressed & BB_PAD_CROSS) && !chord) || n_down;
    if (step && st.mode == kFrozen) api->world_step(kName, kStepUpdates);

    const bool keys_ok = have_keys && in.playing;
    if (keys_ok) {
        st.look_x += in.dx;
        st.look_y += in.dy;
    }
    if (st.mode != kStatic && st.seeded) fly(pad, in, keys_ok, dt);
}

void on_option(const char* key, const char* value, void*) {
    if (!key || !value) return;
    if (std::strcmp(key, "mode") == 0) {
        set_mode(std::strcmp(value, "Static") == 0 ? kStatic : std::strcmp(value, "Live") == 0 ? kLive : kFrozen);
    } else if (std::strcmp(key, "hud") == 0) {
        st.hud = std::strcmp(value, "true") == 0;
        if (st.on) api->hud_hide(kName, st.hud ? 0 : 1);
    }
}

void read_options() {
    st.hud = std::strcmp(api->config("photo_mode.hud"), "true") == 0;
}

}  // namespace

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_CHOICE | BB_OPT_LIVE, "mode", "Mode",
     "Frozen holds the world still while the camera flies; Static keeps the camera where it is while you play on; Live "
     "flies the camera while the world runs. Photo mode always starts Frozen.",
     "Frozen", 0, 0, "Frozen|Static|Live"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "hud", "HUD", "Show the game's HUD, menus and prompts in photo mode. Off leaves them out of the picture.",
     "false"},
    {0}};

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    0,
    "photo_mode",
    "Photo mode",
    "1.0.0",
    "Nox-",
    "A free camera for pictures: the world held still or running, the camera flown by the controller or the keyboard "
    "and mouse, the HUD out of the way. Hold Cross and press L3, or press P. Only with nobody else in your world.",
};

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* host) {
    if (host->version < 29 || host->size < offsetof(BbHostApi, hud_hide) + sizeof(void*)) {
        host->log("photo mode: needs plugin API 29; this host has %u", host->version);
        return 1;
    }
    api = host;
    if (!api->eboot_is_109()) {
        api->log("photo mode: not the 1.09 eboot");
        return 1;
    }
    read_options();
    api->on_option(kName, &on_option, nullptr);
    if (api->on_view(&on_view, nullptr) != 0 || api->on_frame(&on_frame, nullptr) != 0) {
        api->log("photo mode: the host would not call it each frame");
        return 1;
    }
    if (api->key_bound(kKeyP)) api->log("photo mode: P is also one of the game's keys; it starts photo mode as well");
    api->log("photo mode: ready - hold Cross and press L3, or press P");
    return 0;
}
