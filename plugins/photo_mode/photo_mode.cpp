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
// Its panel (panel_show) opens with it: six pages of its options - the
// camera, the picture, depth of field, chromatic aberration, the fog and
// motion blur - whose look goes over the area's own while photo mode runs
// (post_look) and nowhere else. R3 or Tab shows it again once Circle or
// Escape has hidden it; the camera flies either way.
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
constexpr std::uint16_t kKeyA = 4, kKeyD = 7, kKeyE = 8, kKeyN = 17, kKeyP = 19, kKeyQ = 20, kKeyS = 22, kKeyW = 26;
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
// stays and fades out over the world's first frames - some 75 of them, two
// and a half seconds at 30 - and a held world would hold it there.
constexpr float kWorldSettle = 3.0f;

// The options (bb_plugin_options): what the panel's rows hold.
struct Settings {
    int mode = 0;            // Mode, as Mode
    int fov_offset = 0;      // degrees more or less than the game's view
    int roll = 0;            // degrees
    bool hud = false;
    int vignette = 0;        // Area's own, None, Light, Medium, Strong, Heavy
    int saturation = 100, brightness = 100, exposure = 100, gamma = 100, contrast = 100;  // % of the area's
    int dof = 0;             // Area's own, On, Off
    int dof_focus = 30;      // (it / 10) squared metres
    float dof_aperture = 5.6f;
    int dof_near = 0, dof_far = 1;
    int ca = 0;              // Area's own, On, Off
    int ca_lateral[2] = {6, 0}, ca_uniform[2] = {0, 0};  // thousandths
    int fog_distance = 100, fog_density = 100, fog_contrast = 100;  // % of the area's
    int fog_start = 0, fog_height = 0;                              // metres added
    int blur = 0;            // Area's own, On, Off
    int blur_time = 9, blur_length = 19;  // hundredths
    int blur_scale = 40;                  // tenths
    int blur_samples = 4;
} set;

struct Camera {
    float pos[3] = {0, 0, 0};
    float yaw = 0, pitch = 0, roll = 0;  // radians: yaw about the world's up, pitch below the horizon positive
    float fov = 43.0f * kPi / 180.0f;    // vertical, radians: the game's at the start, then the Field of view row's
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
    float game_fov = 43.0f * kPi / 180.0f;  // the game's view's field of view when the camera started
    bool panel = false;            // its panel up (as last asked or told)
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
    st.game_fov = view->fov_y > 0.05f && view->fov_y < 3.0f ? view->fov_y : st.game_fov;
    c.fov = std::clamp(st.game_fov + set.fov_offset * kPi / 180.0f, 10.0f * kPi / 180.0f, 120.0f * kPi / 180.0f);
    // Which way the game's right and up point against ours.
    c.right_sign = c.up_sign = 1.0f;
    float r[3], u[3], fw[3];
    basis(c, r, u, fw);
    c.right_sign = dot(r, view->right) < 0 ? -1.0f : 1.0f;
    c.up_sign = dot(u, view->up) < 0 ? -1.0f : 1.0f;
    c.roll = set.roll * kPi / 180.0f;  // the Roll row's, now the signs are the game's
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
    static const std::uint16_t flying[] = {kKeyP,     kKeyEscape, kKeyTab,    kKeyN,      kKeyW,    kKeyA,    kKeyS, kKeyD, kKeyQ,
                                           kKeyE,     kKeySpace,  kKeyLCtrl,  kKeyLShift, kKeyRShift, kKeyLAlt, kKeyRAlt};
    if (st.mode == kStatic) {
        // The hunter plays: the controller is the game's but for Options (the
        // game's menu would open over the picture), the touchpad's and R3,
        // which shows the panel.
        api->pad_take(BB_PAD_OPTIONS | BB_PAD_TOUCHPAD | BB_PAD_R3, 0);
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

// The panel (panel_show): its guide for the controller and for the keyboard.
constexpr const char* kGuide = "L1 R1: page    d-pad: choose and change    Circle: hide    R3: show";
constexpr const char* kGuideKeys = "Q E: page    arrows: choose and change    Esc: hide    Tab: show";

void show_panel(bool up) {
    if (up) {
        BbPanel p{};
        p.size = sizeof(p);
        p.flags = BB_PANEL_DEFAULTS;
        p.page = nullptr;  // the page it showed last
        p.guide = kGuide;
        p.guide_keys = kGuideKeys;
        st.panel = api->panel_show(kName, &p) == 0;
    } else {
        api->panel_show(kName, nullptr);
        st.panel = false;
    }
}

// The look the rows make (post_look): only what differs from the area's own.
constexpr float kVignettePower[] = {0.0f, 0.0f, 0.3f, 0.6f, 1.0f, 1.5f};  // by the Vignette row, Area's own first
void post_look() {
    BbLook l{};
    l.size = sizeof(l);
    if (set.saturation != 100 || set.brightness != 100 || set.exposure != 100 || set.gamma != 100 || set.contrast != 100) {
        l.set |= BB_LOOK_TONE;
        l.exposure = set.exposure / 100.0f;
        l.gamma = set.gamma / 100.0f;
        l.saturation = set.saturation / 100.0f;
        l.brightness = set.brightness / 100.0f;
        l.contrast = set.contrast / 100.0f;
    }
    if (set.vignette > 0) {
        l.set |= BB_LOOK_VIGNETTE;
        l.vignette = kVignettePower[std::clamp(set.vignette, 0, 5)];
    }
    if (set.dof > 0) {
        l.set |= BB_LOOK_DOF;
        l.dof = set.dof == 1;
        const float d = set.dof_focus / 10.0f;
        l.dof_focus = d * d;
        l.dof_aperture = set.dof_aperture;
        l.dof_near_levels = set.dof_near;
        l.dof_far_levels = set.dof_far;
    }
    if (set.ca > 0) {
        l.set |= BB_LOOK_ABERRATION;
        l.aberration = set.ca == 1;
        for (int i = 0; i < 2; ++i) {
            l.lateral[i] = set.ca_lateral[i] / 1000.0f;
            l.uniform[i] = set.ca_uniform[i] / 1000.0f;
        }
    }
    if (set.blur > 0) {
        l.set |= BB_LOOK_MOTION_BLUR;
        l.motion_blur = set.blur == 1;
        l.blur_time = set.blur_time / 100.0f;
        l.blur_length = set.blur_length / 100.0f;
        l.blur_scale = set.blur_scale / 10.0f;
        l.blur_samples = set.blur_samples;
    }
    if (set.fog_distance != 100 || set.fog_density != 100 || set.fog_contrast != 100 || set.fog_start != 0 || set.fog_height != 0) {
        l.set |= BB_LOOK_FOG;
        l.fog_distance = set.fog_distance / 100.0f;
        l.fog_density = set.fog_density / 100.0f;
        l.fog_contrast = set.fog_contrast / 100.0f;
        l.fog_distance_offset = static_cast<float>(set.fog_start);
        l.fog_height_offset = static_cast<float>(set.fog_height);
    }
    std::uint32_t available = 0;
    api->post_look(kName, l.set ? &l : nullptr, &available);
    static std::uint32_t told = 0;
    const std::uint32_t missing = l.set & ~available & (BB_LOOK_DOF | BB_LOOK_MOTION_BLUR);
    if (missing && missing != told)
        api->log("photo mode: %s cannot show now (Performance mode leaves it out)",
                 missing == BB_LOOK_DOF ? "depth of field" : missing == BB_LOOK_MOTION_BLUR ? "motion blur" : "depth of field and motion blur");
    told = missing;
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
    api->option_set(kName, "mode", "Frozen");  // the panel's row says so too
    api->hud_hide(kName, set.hud ? 0 : 1);
    post_look();
    show_panel(true);
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
    api->post_look(kName, nullptr, nullptr);
    show_panel(false);
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
    const bool p_down = key_down(kKeyP), escape_down = key_down(kKeyEscape), n_down = key_down(kKeyN), key_down_tab = key_down(kKeyTab);
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

    // The panel: up or hidden (Circle and Escape hide it - the host's), and
    // R3 or Tab to show or hide it.
    BbPanelState ps{};
    ps.size = sizeof(ps);
    if (api->panel_state(kName, &ps) == 0) st.panel = ps.shown != 0;
    if ((pad.pressed & BB_PAD_R3) || key_down_tab) show_panel(!st.panel);

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

// A row's value into the settings; false for a key that is not one.
bool take_option(const char* key, const char* value) {
    std::string v = value ? value : "";
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
    const auto num = [&](int lo, int hi, int* out) { *out = std::clamp(std::atoi(v.c_str()), lo, hi); };
    const auto choice = [&](const char* const* names, int n, int* out) {
        for (int i = 0; i < n; ++i) {
            if (v == names[i]) *out = i;
        }
    };
    static const char* const kModes[] = {"Frozen", "Static", "Live"};
    static const char* const kVignettes[] = {"Area's own", "None", "Light", "Medium", "Strong", "Heavy"};
    static const char* const kSwitch[] = {"Area's own", "On", "Off"};
    const std::string k = key;
    if (k == "mode") choice(kModes, 3, &set.mode);
    else if (k == "fov") num(-40, 60, &set.fov_offset);
    else if (k == "roll") num(-90, 90, &set.roll);
    else if (k == "hud") set.hud = v == "true";
    else if (k == "vignette") choice(kVignettes, 6, &set.vignette);
    else if (k == "saturation") num(0, 200, &set.saturation);
    else if (k == "brightness") num(20, 250, &set.brightness);
    else if (k == "exposure") num(10, 250, &set.exposure);
    else if (k == "gamma") num(50, 150, &set.gamma);
    else if (k == "contrast") num(80, 120, &set.contrast);
    else if (k == "dof") choice(kSwitch, 3, &set.dof);
    else if (k == "dof_focus") num(5, 200, &set.dof_focus);
    else if (k == "dof_aperture") set.dof_aperture = std::clamp(static_cast<float>(std::atof(v.c_str())), 0.7f, 32.0f);
    else if (k == "dof_near") num(0, 7, &set.dof_near);
    else if (k == "dof_far") num(0, 7, &set.dof_far);
    else if (k == "ca") choice(kSwitch, 3, &set.ca);
    else if (k == "ca_lateral_x") num(-100, 100, &set.ca_lateral[0]);
    else if (k == "ca_lateral_y") num(-100, 100, &set.ca_lateral[1]);
    else if (k == "ca_uniform_x") num(-100, 100, &set.ca_uniform[0]);
    else if (k == "ca_uniform_y") num(-100, 100, &set.ca_uniform[1]);
    else if (k == "fog_distance") num(10, 200, &set.fog_distance);
    else if (k == "fog_density") num(0, 200, &set.fog_density);
    else if (k == "fog_contrast") num(0, 200, &set.fog_contrast);
    else if (k == "fog_start") num(-100, 100, &set.fog_start);
    else if (k == "fog_height") num(-200, 200, &set.fog_height);
    else if (k == "blur") choice(kSwitch, 3, &set.blur);
    else if (k == "blur_time") num(0, 100, &set.blur_time);
    else if (k == "blur_length") num(0, 100, &set.blur_length);
    else if (k == "blur_scale") num(0, 100, &set.blur_scale);
    else if (k == "blur_samples") num(1, 16, &set.blur_samples);
    else return false;
    return true;
}

void on_option(const char* key, const char* value, void*) {
    if (!key || !value || !take_option(key, value)) return;
    if (std::strcmp(key, "mode") == 0) {
        set_mode(set.mode);
    } else if (std::strcmp(key, "hud") == 0) {
        if (st.on) api->hud_hide(kName, set.hud ? 0 : 1);
    } else if (std::strcmp(key, "fov") == 0) {
        st.cam.fov = std::clamp(st.game_fov + set.fov_offset * kPi / 180.0f, 10.0f * kPi / 180.0f, 120.0f * kPi / 180.0f);
    } else if (std::strcmp(key, "roll") == 0) {
        st.cam.roll = set.roll * kPi / 180.0f;
    } else if (st.on) {
        post_look();
    }
}

}  // namespace

// The menus show the first heading; the panel's six pages are BB_OPT_PANEL
// headings and the rows under them. Every row is live, and every row says what
// it does.
#define PANEL(key, label, help) {BB_OPT_HEADING | BB_OPT_PANEL, key, label, help, "", 0, 0, nullptr}
#define LIVE_INT(key, label, help, def, lo, hi) {BB_OPT_INT | BB_OPT_LIVE, key, label, help, def, lo, hi, nullptr}
#define LIVE_INT_WHEN(key, label, help, def, lo, hi, when) {BB_OPT_INT | BB_OPT_LIVE | BB_OPT_WHEN, key, label, help, def, lo, hi, when}
extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, "about", "Photo mode",
     "Hold Cross and press L3, or press P, to start photo mode and to end it (Escape ends it too). Its panel opens with it: "
     "L1 and R1 (Q and E) turn its pages, the d-pad (the arrows) chooses and changes a row, Circle (Escape) hides it and R3 "
     "(Tab) shows it again. Only with nobody else in your world.",
     "", 0, 0, nullptr},

    PANEL("camera", "Camera", "How photo mode's camera sees, and what the world does meanwhile."),
    {BB_OPT_CHOICE | BB_OPT_LIVE, "mode", "Mode",
     "Frozen holds the world still while the camera flies (Cross or N lets it run two more updates). Static keeps the "
     "camera where it is while you play on. Live flies the camera while the world runs. Photo mode always starts Frozen.",
     "Frozen", 0, 0, "Frozen|Static|Live"},
    LIVE_INT("fov", "Field of view", "Degrees more or less than the game's own view: 0 keeps it, less zooms in.", "0", -40, 60),
    LIVE_INT("roll", "Roll", "Tilts the camera, in degrees: 0 keeps it level.", "0", -90, 90),
    {BB_OPT_BOOL | BB_OPT_LIVE, "hud", "HUD", "Shows the game's HUD, menus and prompts in photo mode. Off leaves them out of the picture.",
     "false", 0, 0, nullptr},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "vignette", "Vignette",
     "Darkens the picture's corners. Area's own keeps the area's (most have none).", "Area's own", 0, 0,
     "Area's own|None|Light|Medium|Strong|Heavy"},

    PANEL("picture", "Picture", "The picture's tone, in % of the area's own: 100 keeps the area's."),
    LIVE_INT("saturation", "Saturation", "% of the area's own colour: 0 is black and white, 200 twice as vivid.", "100", 0, 200),
    LIVE_INT("brightness", "Brightness", "% of the area's own brightness, after the tone map.", "100", 20, 250),
    LIVE_INT("exposure", "Exposure", "% of the area's own exposure, before the tone map: the light the picture is made from.", "100", 10, 250),
    LIVE_INT("gamma", "Gamma", "% of the area's own gamma: more lifts the shadows and pales the colours.", "100", 50, 150),
    LIVE_INT("contrast", "Contrast",
             "% of the area's own contrast. Steep: a few percent goes a long way, and 110 crushes most shadows to black.", "100", 80, 120),

    PANEL("dof", "Depth of field",
          "The blur away from the focus. Performance mode leaves it out, and with it these rows do nothing."),
    {BB_OPT_CHOICE | BB_OPT_LIVE, "dof", "Depth of field",
     "Area's own keeps the area's depth of field. On uses the rows below; Off leaves it out. Performance mode leaves it out "
     "whatever this says.",
     "Area's own", 0, 0, "Area's own|On|Off"},
    LIVE_INT_WHEN("dof_focus", "Focus", "Where it is sharp: 10 is 1 m, 30 is 9 m, 100 is 100 m (the number over ten, squared).",
                  "30", 5, 200, "dof=On"),
    {BB_OPT_FLOAT | BB_OPT_LIVE | BB_OPT_WHEN, "dof_aperture", "Aperture",
     "The lens's f-number: the smaller, the more blurred away from the focus. The areas use 5.6.", "5.6", 0.7, 32, "dof=On"},
    LIVE_INT_WHEN("dof_near", "Blur in front", "How far the blur in front of the focus spreads: 0 to 7.", "0", 0, 7, "dof=On"),
    LIVE_INT_WHEN("dof_far", "Blur behind", "How far the blur behind the focus spreads: 0 to 7.", "1", 0, 7, "dof=On"),

    PANEL("ca", "Chromatic aberration", "Colour fringes towards the picture's edges, as a lens makes them."),
    {BB_OPT_CHOICE | BB_OPT_LIVE, "ca", "Chromatic aberration",
     "Area's own keeps the area's (most have none). On uses the rows below; Off leaves it out.", "Area's own", 0, 0,
     "Area's own|On|Off"},
    LIVE_INT_WHEN("ca_lateral_x", "Lateral, across", "Fringes growing towards the edges, across the picture, in thousandths (an area's is 6).",
                  "6", -100, 100, "ca=On"),
    LIVE_INT_WHEN("ca_lateral_y", "Lateral, up and down", "The same up and down, in thousandths.", "0", -100, 100, "ca=On"),
    LIVE_INT_WHEN("ca_uniform_x", "Uniform, across", "Fringes the same all over, across the picture, in thousandths.", "0", -100, 100, "ca=On"),
    LIVE_INT_WHEN("ca_uniform_y", "Uniform, up and down", "The same up and down, in thousandths.", "0", -100, 100, "ca=On"),

    PANEL("fog", "Fog", "The area's fog and haze: 100% and 0 m keep the area's."),
    LIVE_INT("fog_distance", "Distance", "% of the area's distance to the fog: less brings it closer.", "100", 10, 200),
    LIVE_INT("fog_density", "Density", "% of the area's fog: 0 clears it, 200 doubles it.", "100", 0, 200),
    LIVE_INT("fog_contrast", "Contrast", "% of the area's fog contrast: more darkens what the fog covers.", "100", 0, 200),
    LIVE_INT("fog_start", "Start", "Metres added to where the fog begins: more pushes it away.", "0", -100, 100),
    LIVE_INT("fog_height", "Height fog", "Metres added to the height fog's height: less raises the fog over the ground.", "0", -200, 200),

    PANEL("blur", "Motion blur",
          "The blur of what moves, as the camera sees it. Performance mode leaves it out, and with it these rows do nothing."),
    {BB_OPT_CHOICE | BB_OPT_LIVE, "blur", "Motion blur",
     "Area's own keeps the area's. On uses the rows below; Off leaves it out. Performance mode leaves it out whatever this says.",
     "Area's own", 0, 0, "Area's own|On|Off"},
    LIVE_INT_WHEN("blur_time", "Time", "How long a moment the blur covers, in hundredths (an area's is about 9).", "9", 0, 100, "blur=On"),
    LIVE_INT_WHEN("blur_length", "Length", "The longest a streak gets, in hundredths (an area's is about 19).", "19", 0, 100, "blur=On"),
    LIVE_INT_WHEN("blur_scale", "Camera blur", "How much the camera's own turning blurs, in tenths (an area's is up to 40).", "40", 0, 100, "blur=On"),
    LIVE_INT_WHEN("blur_samples", "Samples", "Steps along each streak: more is smoother and slower.", "4", 1, 16, "blur=On"),
    {0, nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr}};
#undef PANEL
#undef LIVE_INT
#undef LIVE_INT_WHEN

namespace {

// The rows as the player's config holds them, or their defaults.
void read_options() {
    for (const BbOption* o = bb_plugin_options; o->type != BB_OPT_END; ++o) {
        if ((o->type & 0xffu) == BB_OPT_HEADING || !o->key) continue;
        const std::string key = std::string(kName) + "." + o->key;
        const char* v = api->config(key.c_str());
        take_option(o->key, v && v[0] ? v : o->def);
    }
}

}  // namespace

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
    if (host->version < 29 || host->size < offsetof(BbHostApi, option_set) + sizeof(void*)) {
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
