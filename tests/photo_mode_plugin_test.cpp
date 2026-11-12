// plugins/photo_mode on a host of our own: no game, no GPU. The plugin is
// loaded as the host loads plugins and driven frame by frame - the world's
// state, the controller and the keyboard as the test sets them - while every
// plugin API 29 call it makes is recorded. Checked: on a host older than API 29
// it refuses before asking anything; it starts on P and on Cross held with L3
// pressed, Frozen, with the world held, the controls taken and the HUD out of
// the picture; the camera starts at the game's view and flies the way it looks;
// Cross (once let go) and N advance a held world by two updates; Static gives
// the controls back but Options and the touchpad, Live flies over a running
// world; it ends on Escape, P and the chord, and at a loading screen, a
// cutscene or a menu, giving everything back; it does not start, and it ends,
// with another player in the world or on the way, or in another's world; the
// HUD option shows the HUD. Its options table is whole.
//
//   photo_mode_plugin_test <photo_mode.so>
#include "bbhost_plugin.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

namespace {

constexpr std::uint16_t kP = 19, kN = 17, kW = 26, kEscape = 41;

struct Host {
    std::map<std::string, std::string> config;
    std::vector<std::string> log, messages;
    void (*frame_fn)(void*) = nullptr;
    void* frame_user = nullptr;
    void (*view_fn)(BbCamera*, void*) = nullptr;
    void* view_user = nullptr;
    void (*option_fn)(const char*, const char*, void*) = nullptr;
    void* option_user = nullptr;
    // What the plugin asked for last.
    int paused = 0, hud_hidden = 0, steps = 0;
    std::uint32_t take_buttons = 0, take_axes = 0;
    std::vector<std::uint16_t> taken_keys;
    std::uint32_t taken_mouse = 0;
    int taken_motion = 0;
    // What the host reports.
    BbWorldState world{};
    BbPad pad{};
    BbInput input{};
    int calls = 0;  // every call that asks for something (pause, take, hide, step)
} H;

void fake_log(const char* fmt, ...) {
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    H.log.push_back(b);
}
const char* fake_config(const char* key) {
    static std::string v;
    const auto it = H.config.find(key);
    v = it == H.config.end() ? "" : it->second;
    return v.c_str();
}
int fake_eboot_is_109() { return 1; }
int fake_on_frame(void (*fn)(void*), void* user) {
    H.frame_fn = fn;
    H.frame_user = user;
    return 0;
}
int fake_on_view(void (*fn)(BbCamera*, void*), void* user) {
    H.view_fn = fn;
    H.view_user = user;
    return 0;
}
int fake_on_option(const char* name, void (*fn)(const char*, const char*, void*), void* user) {
    CHECK(std::strcmp(name, "photo_mode") == 0);
    H.option_fn = fn;
    H.option_user = user;
    return 0;
}
int fake_key_bound(std::uint16_t) { return 0; }
int fake_show_message(const char* text, float) {
    H.messages.push_back(text);
    return 0;
}
int fake_world_pause(const char* name, int paused) {
    CHECK(std::strcmp(name, "photo_mode") == 0);
    H.paused = paused;
    ++H.calls;
    return 0;
}
int fake_world_step(const char* name, std::uint32_t updates) {
    CHECK(std::strcmp(name, "photo_mode") == 0);
    H.steps += static_cast<int>(updates);
    ++H.calls;
    return H.paused ? 0 : 1;
}
int fake_pad(BbPad* out) {
    CHECK(out->size == sizeof(BbPad));
    *out = H.pad;
    out->size = sizeof(BbPad);
    H.pad.pressed = 0;  // the frame's presses, read once a frame
    return 0;
}
int fake_pad_take(std::uint32_t buttons, std::uint32_t axes) {
    H.take_buttons = buttons;
    H.take_axes = axes;
    ++H.calls;
    return 0;
}
int fake_input(BbInput* out) {
    *out = H.input;
    H.input.pressed = 0;
    H.input.dx = H.input.dy = 0;
    return 0;
}
int fake_input_take(const std::uint16_t* sc, std::size_t n, std::uint32_t buttons, int, int motion) {
    H.taken_keys.assign(sc, sc + n);
    H.taken_mouse = buttons;
    H.taken_motion = motion;
    ++H.calls;
    return 0;
}
int fake_world_state(BbWorldState* out) {
    CHECK(out->size == sizeof(BbWorldState));
    *out = H.world;
    out->size = sizeof(BbWorldState);
    return 0;
}
int fake_hud_hide(const char* name, int hidden) {
    CHECK(std::strcmp(name, "photo_mode") == 0);
    H.hud_hidden = hidden;
    ++H.calls;
    return 0;
}
int fake_player_position(float xyz[3], float* yaw) {
    if (xyz) xyz[0] = xyz[1] = xyz[2] = 0;
    if (yaw) *yaw = 0;
    return 0;
}

struct Plugin {
#if defined(_WIN32)
    HMODULE handle = nullptr;
    void* sym(const char* name) { return reinterpret_cast<void*>(GetProcAddress(handle, name)); }
#else
    void* handle = nullptr;
    void* sym(const char* name) { return dlsym(handle, name); }
#endif
    int (*image)(const BbHostApi*) = nullptr;
    const BbOption* options = nullptr;
    const BbPluginInfo* info = nullptr;
};

Plugin load(const char* path) {
    Plugin p;
#if defined(_WIN32)
    p.handle = LoadLibraryA(path);
#else
    p.handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
    if (!p.handle) return p;
    p.image = reinterpret_cast<int (*)(const BbHostApi*)>(p.sym("bb_plugin_image"));
    p.options = static_cast<const BbOption*>(p.sym("bb_plugin_options"));
    p.info = static_cast<const BbPluginInfo*>(p.sym("bb_plugin_info"));
    return p;
}

BbHostApi make_api(std::uint32_t version) {
    BbHostApi api;
    std::memset(&api, 0, sizeof api);
    api.version = version;
    api.size = sizeof(BbHostApi);
    api.log = &fake_log;
    api.config = &fake_config;
    api.eboot_is_109 = &fake_eboot_is_109;
    api.on_frame = &fake_on_frame;
    api.on_view = &fake_on_view;
    api.on_option = &fake_on_option;
    api.key_bound = &fake_key_bound;
    api.show_message = &fake_show_message;
    api.input = &fake_input;
    api.input_take = &fake_input_take;
    api.player_position = &fake_player_position;
    if (version >= 29) {
        api.world_pause = &fake_world_pause;
        api.world_step = &fake_world_step;
        api.pad = &fake_pad;
        api.pad_take = &fake_pad_take;
        api.world_state = &fake_world_state;
        api.hud_hide = &fake_hud_hide;
    } else {
        api.size = static_cast<std::uint32_t>(offsetof(BbHostApi, on_camera_target) + sizeof(void*));
    }
    return api;
}

void frame() {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    H.frame_fn(H.frame_user);
}
// A key held for one frame, then let go for the next.
void tap_key(std::uint16_t k) {
    H.input.key[k] = 1;
    frame();
    H.input.key[k] = 0;
    frame();
}
void tap_pad(std::uint32_t b) {
    H.pad.buttons |= b;
    H.pad.pressed |= b;
    frame();
    H.pad.buttons &= ~b;
    frame();
}
// The world in for longer than the plugin's settle (the loading screen's page
// fades out over its first frames).
void settle() {
    frame();
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    frame();
}
// The game's view, as on_view hands it: looking along +z from (1, 2, 3).
BbCamera game_view() {
    BbCamera c{};
    c.right[0] = 1;   // the game's right for a camera looking along +z, up +y
    c.up[1] = 1;
    c.forward[2] = 1;
    c.position[0] = 1;
    c.position[1] = 2;
    c.position[2] = 3;
    c.position[3] = 1;
    c.fov_y = 0.75f;
    c.aspect = 16.0f / 9.0f;
    c.near_z = 0.1f;
    c.far_z = 1000.0f;
    return c;
}
bool logged(const char* text) {
    for (const std::string& l : H.log) {
        if (l.find(text) != std::string::npos) return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: photo_mode_plugin_test <photo_mode.so>\n");
        return 2;
    }
    Plugin p = load(argv[1]);
    CHECK(p.handle && p.image && p.info && p.options);
    if (!p.handle || !p.image) return 1;

    // Its info and options: not a gameplay plugin (no session rules), not opt-in.
    CHECK(std::strcmp(p.info->name, "photo_mode") == 0);
    CHECK((p.info->flags & BB_PLUGIN_GAMEPLAY) == 0);
    CHECK((p.info->flags & BB_PLUGIN_OPT_IN) == 0);
    {
        std::map<std::string, int> keys;
        for (const BbOption* o = p.options; o->type; ++o) {
            if ((o->type & 0xff) == BB_OPT_HEADING || (o->type & 0xff) == BB_OPT_ACTION) continue;
            CHECK(o->key && o->label && o->help && o->help[0]);  // every row says what it does
            CHECK(keys[o->key]++ == 0);
            if ((o->type & 0xff) == BB_OPT_CHOICE) CHECK(std::strstr(o->choices, o->def) != nullptr);
        }
    }

    // An API 28 host: refused before asking for anything.
    {
        const BbHostApi old = make_api(28);
        CHECK(p.image(&old) != 0);
        CHECK(H.frame_fn == nullptr && H.calls == 0);
    }

    const BbHostApi api = make_api(29);
    CHECK(p.image(&api) == 0);
    CHECK(H.frame_fn && H.view_fn && H.option_fn);
    H.world.flags = BB_WORLD_IN;
    H.pad.connected = 1;
    H.input.playing = 1;
    for (int i = 0; i < 5; ++i) frame();
    CHECK(H.calls == 0);  // asks for nothing until started
    // Not in the world's first two seconds.
    tap_key(kP);
    CHECK(H.paused == 0 && logged("the world is still coming in"));
    settle();

    // P starts it: Frozen, the world held, the controller and the camera's keys taken, the HUD out.
    tap_key(kP);
    CHECK(H.paused == 1);
    CHECK(H.hud_hidden == 1);
    CHECK((H.take_buttons & (BB_PAD_CROSS | BB_PAD_L1 | BB_PAD_R1 | BB_PAD_OPTIONS)) == (BB_PAD_CROSS | BB_PAD_L1 | BB_PAD_R1 | BB_PAD_OPTIONS));
    CHECK(H.take_axes == (BB_PAD_LEFT_STICK | BB_PAD_RIGHT_STICK | BB_PAD_TRIGGERS));
    CHECK(H.taken_motion == 1);
    {
        bool w = false;
        for (std::uint16_t k : H.taken_keys) w = w || k == kW;
        CHECK(w);
    }
    // The camera starts at the game's view and flies the way it looks.
    BbCamera view = game_view();
    H.view_fn(&view, H.view_user);
    CHECK(std::fabs(view.position[0] - 1) < 1e-4f && std::fabs(view.position[2] - 3) < 1e-4f);
    CHECK(std::fabs(view.forward[2] - 1) < 1e-4f && std::fabs(view.right[0] - 1) < 1e-4f && std::fabs(view.up[1] - 1) < 1e-4f);
    CHECK(std::fabs(view.fov_y - 0.75f) < 1e-4f);
    H.pad.ly = -1.0f;  // the stick up: forward
    for (int i = 0; i < 10; ++i) frame();
    H.pad.ly = 0.0f;
    view = game_view();
    H.view_fn(&view, H.view_user);
    CHECK(view.position[2] > 3.01f);
    CHECK(std::fabs(view.position[0] - 1) < 1e-3f && std::fabs(view.position[1] - 2) < 1e-3f);
    H.pad.lx = 1.0f;  // right: along the game's right
    for (int i = 0; i < 10; ++i) frame();
    H.pad.lx = 0.0f;
    view = game_view();
    H.view_fn(&view, H.view_user);
    CHECK(view.position[0] > 1.01f);

    // The frame advance: Cross (it was not held to start) and N, two updates each.
    tap_pad(BB_PAD_CROSS);
    CHECK(H.steps == 2);
    tap_key(kN);
    CHECK(H.steps == 4);

    // Static: the world runs, the controller is the game's but Options and the touchpad; Live flies over it.
    H.option_fn("mode", "Static", H.option_user);
    CHECK(H.paused == 0);
    CHECK(H.take_buttons == (BB_PAD_OPTIONS | BB_PAD_TOUCHPAD) && H.take_axes == 0);
    CHECK(H.taken_motion == 0);
    view = game_view();
    H.view_fn(&view, H.view_user);
    const float static_z = view.position[2];
    H.pad.ly = -1.0f;
    for (int i = 0; i < 6; ++i) frame();
    H.pad.ly = 0.0f;
    view = game_view();
    H.view_fn(&view, H.view_user);
    CHECK(std::fabs(view.position[2] - static_z) < 1e-5f);  // the camera stays
    H.option_fn("mode", "Live", H.option_user);
    CHECK(H.paused == 0 && H.take_axes == (BB_PAD_LEFT_STICK | BB_PAD_RIGHT_STICK | BB_PAD_TRIGGERS));
    const int steps_before = H.steps;
    tap_key(kN);
    CHECK(H.steps == steps_before);  // no frame advance over a running world

    // Escape ends it: everything given back.
    tap_key(kEscape);
    CHECK(H.paused == 0 && H.hud_hidden == 0 && H.take_buttons == 0 && H.take_axes == 0 && H.taken_keys.empty() && H.taken_motion == 0);
    view = game_view();
    H.view_fn(&view, H.view_user);
    CHECK(std::fabs(view.position[2] - 3) < 1e-5f);  // the game's view again

    // The chord: Cross held, L3 pressed - it starts Frozen whatever the mode was.
    H.pad.buttons = BB_PAD_CROSS;
    frame();
    H.pad.pressed = BB_PAD_L3;
    H.pad.buttons = BB_PAD_CROSS | BB_PAD_L3;
    frame();
    H.pad.buttons = 0;
    frame();
    CHECK(H.paused == 1);
    // ... Cross was held to start: no frame advance until it has been let go.
    CHECK(H.steps == steps_before);

    // Another player on the way: it ends at once, and does not start again.
    H.world.arriving = 1;
    frame();
    CHECK(H.paused == 0 && H.hud_hidden == 0 && H.take_buttons == 0);
    CHECK(logged("a player is on the way"));
    tap_key(kP);
    CHECK(H.paused == 0);
    H.world.arriving = 0;
    H.world.others = 1;
    tap_key(kP);
    CHECK(H.paused == 0);
    CHECK(logged("another player is in the world"));
    CHECK(!H.messages.empty());
    H.world.others = 0;
    // Matched to a player: the game hosts its session before they join the room.
    H.world.flags = BB_WORLD_IN | BB_WORLD_OPEN;
    for (const int role : {BB_SESSION_HOSTING, BB_SESSION_HOST, BB_SESSION_LEAVING}) {
        H.world.session = role;
        tap_key(kP);
        CHECK(H.paused == 0);
    }
    H.world.flags = BB_WORLD_IN;
    H.world.session = BB_SESSION_NONE;
    tap_key(kP);
    CHECK(H.paused == 1);  // nobody coming: it starts
    H.world.session = BB_SESSION_HOSTING;
    H.world.flags = BB_WORLD_IN | BB_WORLD_OPEN;
    frame();
    CHECK(H.paused == 0 && H.hud_hidden == 0 && H.take_buttons == 0);  // a match ends it
    H.world.session = BB_SESSION_NONE;
    H.world.flags = BB_WORLD_IN | BB_WORLD_VISITING;
    tap_key(kP);
    CHECK(H.paused == 0);
    H.world.flags = BB_WORLD_IN | BB_WORLD_MENU;
    tap_key(kP);
    CHECK(H.paused == 0);
    H.world.flags = BB_WORLD_IN | BB_WORLD_CUTSCENE;
    tap_key(kP);
    CHECK(H.paused == 0);
    H.world.flags = BB_WORLD_LOADING;
    tap_key(kP);
    CHECK(H.paused == 0);

    // A loading screen, a menu or a cutscene ends it.
    for (const std::uint32_t flags : {BB_WORLD_LOADING, BB_WORLD_IN | BB_WORLD_MENU, BB_WORLD_IN | BB_WORLD_CUTSCENE}) {
        const bool from_load = H.world.flags & BB_WORLD_LOADING;
        H.world.flags = BB_WORLD_IN;
        if (from_load) settle();
        tap_key(kP);
        CHECK(H.paused == 1);
        H.world.flags = flags;
        frame();
        CHECK(H.paused == 0 && H.hud_hidden == 0 && H.take_buttons == 0);
    }

    // The HUD option shows the HUD in photo mode, live.
    H.world.flags = BB_WORLD_IN;
    tap_key(kP);
    CHECK(H.hud_hidden == 1);
    H.option_fn("hud", "true", H.option_user);
    CHECK(H.hud_hidden == 0);
    tap_key(kP);
    CHECK(H.paused == 0);
    H.config["photo_mode.hud"] = "true";

    if (g_failures) {
        std::fprintf(stderr, "photo_mode_plugin_test: %d failure(s)\n", g_failures);
        for (const std::string& l : H.log) std::fprintf(stderr, "  log: %s\n", l.c_str());
        return 1;
    }
    std::printf("photo_mode_plugin_test: ok\n");
    return 0;
}
