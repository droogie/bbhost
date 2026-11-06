// plugins/ps1 on a host of our own: no game, no GPU. The plugin is loaded as
// the host loads plugins, started with a config, and its options changed as
// the in-game menu changes them; every plugin API 18 call it makes is
// recorded. Checked: the defaults ask for the whole look at bb_plugin_init -
// the 3D at the Resolution brought up nearest, 8-bit colour with the colour
// grading nearest, unfiltered textures, a 240-line snap - before the
// image phase (the render size and the targets' formats are decided between
// the two); each option says what it should, at the start and live; on a host
// older than API 18 it refuses before calling anything; on another eboot it
// takes the look back as it unloads; its options table is
// whole (unique keys, defaults among their choices). Old-hardware sound
// (API 21's audio_crush): off by default, each choice's rate, bits and
// low-pass, Custom's own from the config and live, and nothing asked of an
// API 20 host, where the look works as before. The live switch (API 21):
// loaded off it asks for nothing, then everything as it is turned on in game
// and back as it goes off; PS1 look off leaves the picture alone and the
// sound old; Custom's rows grey out unless Custom, the look's under its
// heading unless PS1 look.
//
//   ps1_plugin_test <ps1.so>
#include "bbhost_plugin.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
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

struct Host {
    std::map<std::string, std::string> config = {{"plugins.ps1", "true"}};
    std::vector<std::string> calls;  // "snap 240", "filter 1", "scale 1 0", "colour 8", "grading 1"
    std::vector<std::string> log;
    void (*option_fn)(const char*, const char*, void*) = nullptr;
    void* option_user = nullptr;
    bool is_109 = true;
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
int fake_eboot_is_109() { return H.is_109 ? 1 : 0; }
int fake_on_option(const char* name, void (*fn)(const char*, const char*, void*), void* user) {
    CHECK(std::strcmp(name, "ps1") == 0);
    H.option_fn = fn;
    H.option_user = user;
    return 0;
}
void fake_scene_snap(std::uint32_t lines) { H.calls.push_back("snap " + std::to_string(lines)); }
void fake_scene_filter(int filter) { H.calls.push_back("filter " + std::to_string(filter)); }
void fake_scene_scale(int scale, std::uint32_t lines) { H.calls.push_back("scale " + std::to_string(scale) + " " + std::to_string(lines)); }
void fake_scene_colour_bits(int bits) { H.calls.push_back("colour " + std::to_string(bits)); }
void fake_grading_filter(int filter) { H.calls.push_back("grading " + std::to_string(filter)); }
void fake_audio_crush(int rate, int bits, float lowpass) {
    H.calls.push_back("crush " + std::to_string(rate) + " " + std::to_string(bits) + " " + std::to_string(static_cast<int>(lowpass + 0.5f)));
}

struct Plugin {
#if defined(_WIN32)
    HMODULE handle = nullptr;
    void* sym(const char* name) { return reinterpret_cast<void*>(GetProcAddress(handle, name)); }
#else
    void* handle = nullptr;
    void* sym(const char* name) { return dlsym(handle, name); }
#endif
    int (*init)(const BbHostApi*) = nullptr;
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
    p.init = reinterpret_cast<int (*)(const BbHostApi*)>(p.sym("bb_plugin_init"));
    p.image = reinterpret_cast<int (*)(const BbHostApi*)>(p.sym("bb_plugin_image"));
    p.options = static_cast<const BbOption*>(p.sym("bb_plugin_options"));
    p.info = static_cast<const BbPluginInfo*>(p.sym("bb_plugin_info"));
    return p;
}

BbHostApi make_api(std::uint32_t version) {
    BbHostApi api;
    std::memset(&api, 0, sizeof api);
    api.version = version;
    api.size = version >= 21   ? sizeof api
               : version >= 18 ? static_cast<std::uint32_t>(offsetof(BbHostApi, audio_crush))
                               : static_cast<std::uint32_t>(offsetof(BbHostApi, scene_snap));
    api.log = fake_log;
    api.config = fake_config;
    api.eboot_is_109 = fake_eboot_is_109;
    api.on_option = fake_on_option;
    if (version >= 18) {
        api.scene_snap = fake_scene_snap;
        api.scene_filter = fake_scene_filter;
        api.scene_scale = fake_scene_scale;
        api.scene_colour_bits = fake_scene_colour_bits;
        api.grading_filter = fake_grading_filter;
    }
    if (version >= 21) api.audio_crush = fake_audio_crush;
    return api;
}

bool has(const std::vector<std::string>& calls, const std::string& c) {
    for (const std::string& x : calls)
        if (x == c) return true;
    return false;
}

void dump(const char* what) {
    std::fprintf(stderr, "--- %s: calls\n", what);
    for (const std::string& c : H.calls) std::fprintf(stderr, "  %s\n", c.c_str());
    for (const std::string& l : H.log) std::fprintf(stderr, "  log: %s\n", l.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ps1_plugin_test <ps1 plugin>\n");
        return 2;
    }
    Plugin p = load(argv[1]);
    CHECK(p.handle && p.init && p.image && p.options && p.info);
    if (!p.handle || !p.init || !p.image || !p.options || !p.info) return 1;

    // What it says about itself: opt-in, not a gameplay plugin (the picture alone).
    CHECK(std::strcmp(p.info->name, "ps1") == 0);
    CHECK((p.info->flags & BB_PLUGIN_OPT_IN) != 0);
    CHECK((p.info->flags & BB_PLUGIN_GAMEPLAY) == 0);
    CHECK((p.info->flags & BB_PLUGIN_LIVE_SWITCH) != 0);

    // Its options: unique keys; the toggles on; each choice's default among its choices.
    {
        std::set<std::string> keys;
        std::map<std::string, std::string> defaults, when;
        for (const BbOption* o = p.options; o->type != BB_OPT_END; ++o) {
            CHECK(o->key && *o->key);
            CHECK(keys.insert(o->key).second);
            const std::uint32_t type = o->type & 0xff;
            if (type == BB_OPT_CHOICE) {
                CHECK(o->choices && o->def);
                const std::string list = std::string("|") + o->choices + "|";
                CHECK(list.find(std::string("|") + o->def + "|") != std::string::npos);
            }
            if (o->def) defaults[o->key] = o->def;
            if (o->type & BB_OPT_WHEN) {
                CHECK(type != BB_OPT_CHOICE && o->choices);
                when[o->key] = o->choices ? o->choices : "";
            }
        }
        CHECK(when["sound_rate"] == "sound=Custom");
        CHECK(when["sound_bits"] == "sound=Custom");
        CHECK(when["look"] == "look_3d=true");
        CHECK(defaults["look_3d"] == "true");
        CHECK(defaults["unfiltered"] == "true");
        CHECK(defaults["colour_8bit"] == "true");
        CHECK(defaults["snap"] == "240p");
        CHECK(defaults["resolution"] == "Native");
        CHECK(defaults["sound"] == "Off");
    }

    // An older host: refused before any call.
    {
        H = Host{};
        BbHostApi api = make_api(17);
        CHECK(p.init(&api) != 0);
        CHECK(H.calls.empty());
    }

    // The defaults: the whole look asked for at init, nothing more at image.
    {
        H = Host{};
        BbHostApi api = make_api(BB_PLUGIN_API_VERSION);
        CHECK(p.init(&api) == 0);
        CHECK(has(H.calls, "scale 1 0"));
        CHECK(has(H.calls, "colour 8"));
        CHECK(has(H.calls, "grading 1"));
        CHECK(has(H.calls, "filter 1"));
        CHECK(has(H.calls, "snap 240"));
        CHECK(has(H.calls, "crush 0 0 0"));  // the game's own sound
        const std::size_t at_init = H.calls.size();
        CHECK(p.image(&api) == 0);
        CHECK(H.calls.size() == at_init);
        CHECK(H.option_fn != nullptr);
        if (g_failures) dump("defaults");

        // Live changes, as the menus make them.
        if (H.option_fn) {
            H.calls.clear();
            H.option_fn("snap", "Native", H.option_user);
            CHECK(has(H.calls, "snap " + std::to_string(BB_SNAP_PIXELS)));
            H.option_fn("snap", "Off", H.option_user);
            CHECK(has(H.calls, "snap 0"));
            H.option_fn("unfiltered", "false", H.option_user);
            CHECK(has(H.calls, "filter 0"));
            H.option_fn("colour_8bit", "false", H.option_user);
            CHECK(has(H.calls, "colour 0"));
            CHECK(has(H.calls, "grading 0"));  // the grading smooth again
            H.option_fn("resolution", "240p", H.option_user);
            CHECK(has(H.calls, "scale 1 240"));
            H.option_fn("resolution", "Native", H.option_user);
            CHECK(has(H.calls, "scale 1 0"));

            // Old-hardware sound.
            H.calls.clear();
            H.option_fn("sound", "PS1", H.option_user);
            CHECK(has(H.calls, "crush 22050 12 10000"));
            H.option_fn("sound", "Crunchy", H.option_user);
            CHECK(has(H.calls, "crush 11025 8 0"));
            H.option_fn("sound", "Custom", H.option_user);
            CHECK(has(H.calls, "crush 22050 8 9923"));  // Custom's defaults, low-passed at 0.45 of the rate
            H.option_fn("sound_rate", "16000.0", H.option_user);
            CHECK(has(H.calls, "crush 16000 8 7200"));
            H.option_fn("sound_bits", "6.0", H.option_user);
            CHECK(has(H.calls, "crush 16000 6 7200"));
            H.option_fn("sound_bits", "16.0", H.option_user);
            CHECK(has(H.calls, "crush 16000 0 7200"));  // 16 bits: no truncation
            H.option_fn("sound", "Off", H.option_user);
            CHECK(H.calls.back() == "crush 0 0 0");
            H.calls.clear();
            H.option_fn("sound_rate", "8000.0", H.option_user);  // Custom's rate while Off: still off
            CHECK(H.calls.size() == 1 && H.calls.back() == "crush 0 0 0");
            if (g_failures) dump("live");
        }
    }

    // A config of its own at the start.
    {
        H = Host{};
        H.config = {{"plugins.ps1", "true"},      {"ps1.snap", "Off"},          {"ps1.unfiltered", "false"}, {"ps1.colour_8bit", "false"},
                    {"ps1.resolution", "360p"},   {"ps1.sound", "Custom"},     {"ps1.sound_rate", "8000.0"},
                    {"ps1.sound_bits", "4.0"}};
        BbHostApi api = make_api(BB_PLUGIN_API_VERSION);
        CHECK(p.init(&api) == 0);
        CHECK(has(H.calls, "snap 0"));
        CHECK(has(H.calls, "filter 0"));
        CHECK(has(H.calls, "colour 0"));
        CHECK(has(H.calls, "grading 0"));
        CHECK(has(H.calls, "scale 1 360"));
        CHECK(has(H.calls, "crush 8000 4 3600"));
        if (g_failures) dump("config");
    }

    // Another eboot: off at the image phase.
    {
        H = Host{};
        H.is_109 = false;
        BbHostApi api = make_api(BB_PLUGIN_API_VERSION);
        CHECK(p.init(&api) == 0);
        H.calls.clear();
        CHECK(p.image(&api) != 0);
        // Unloaded, nothing could change the look: it is taken back.
        CHECK(has(H.calls, "snap 0"));
        CHECK(has(H.calls, "filter 0"));
        CHECK(has(H.calls, "colour 0"));
        CHECK(has(H.calls, "grading 0"));
        CHECK(has(H.calls, "scale 0 0"));
        CHECK(has(H.calls, "crush 0 0 0"));
        if (g_failures) dump("other eboot");
    }

    // Loaded off (the live switch): nothing until it is turned on in game.
    {
        H = Host{};
        H.config = {{"ps1.sound", "Crunchy"}};
        BbHostApi api = make_api(BB_PLUGIN_API_VERSION);
        CHECK(p.init(&api) == 0);
        CHECK(H.calls.empty());
        CHECK(p.image(&api) == 0);
        CHECK(H.calls.empty());
        CHECK(H.option_fn != nullptr);
        if (H.option_fn) {
            H.option_fn("snap", "120p", H.option_user);  // kept for later, not applied
            CHECK(!has(H.calls, "snap 120"));
            H.calls.clear();
            H.config["plugins.ps1"] = "true";
            H.option_fn("", "true", H.option_user);
            CHECK(has(H.calls, "scale 1 0"));
            CHECK(has(H.calls, "colour 8"));
            CHECK(has(H.calls, "filter 1"));
            CHECK(has(H.calls, "snap 240"));
            CHECK(has(H.calls, "crush 11025 8 0"));
            H.calls.clear();
            H.config["plugins.ps1"] = "false";
            H.option_fn("", "false", H.option_user);
            CHECK(has(H.calls, "scale 0 0"));
            CHECK(has(H.calls, "colour 0"));
            CHECK(has(H.calls, "grading 0"));
            CHECK(has(H.calls, "filter 0"));
            CHECK(has(H.calls, "snap 0"));
            CHECK(has(H.calls, "crush 0 0 0"));
            H.calls.clear();
            H.option_fn("sound", "PS1", H.option_user);  // off: the sound stays the game's
            CHECK(!has(H.calls, "crush 22050 12 10000"));
        }
        if (g_failures) dump("switch");
    }

    // PS1 look off: the game's picture, the sound still old; on again live.
    {
        H = Host{};
        H.config = {{"plugins.ps1", "true"}, {"ps1.look_3d", "false"}, {"ps1.sound", "Crunchy"}};
        BbHostApi api = make_api(BB_PLUGIN_API_VERSION);
        CHECK(p.init(&api) == 0);
        CHECK(!has(H.calls, "scale 1 0") && !has(H.calls, "snap 240") && !has(H.calls, "colour 8"));
        CHECK(has(H.calls, "crush 11025 8 0"));
        CHECK(p.image(&api) == 0);
        if (H.option_fn) {
            H.calls.clear();
            H.option_fn("resolution", "240p", H.option_user);  // the look off: kept, not applied
            CHECK(H.calls.empty());
            H.config["ps1.look_3d"] = "true";
            H.config["ps1.resolution"] = "240p";
            H.option_fn("look_3d", "true", H.option_user);
            CHECK(has(H.calls, "scale 1 240"));
            CHECK(has(H.calls, "snap 240"));
            H.calls.clear();
            H.config["ps1.look_3d"] = "false";
            H.option_fn("look_3d", "false", H.option_user);
            CHECK(has(H.calls, "scale 0 0") && has(H.calls, "snap 0"));
            CHECK(!has(H.calls, "crush 0 0 0"));  // the sound left as it is
        }
        if (g_failures) dump("look off");
    }

    // An API 20 host: the look as before, the sound left alone.
    {
        H = Host{};
        H.config = {{"ps1.sound", "PS1"}};
        BbHostApi api = make_api(20);
        CHECK(p.init(&api) == 0);
        CHECK(has(H.calls, "snap 240"));
        CHECK(p.image(&api) == 0);
        if (H.option_fn) H.option_fn("sound", "Crunchy", H.option_user);
        for (const std::string& c : H.calls) CHECK(c.rfind("crush", 0) != 0);
        if (g_failures) dump("API 20");
    }

    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("ps1 plugin: all checks passed\n");
    return 0;
}
