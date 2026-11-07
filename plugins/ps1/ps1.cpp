// Bloodborne PS1: the 3D drawn the way the first PlayStation drew it - its vertices
// snapped to the screen's grid, its textures unfiltered, its colour in 8 bits
// a channel, and the 3D at a Resolution of its own, brought up to the window
// pixel for pixel. The HUD and the menus stay as they are, at the window's
// size.
//
//   [plugins]
//   ps1 = true
//   [ps1]
//   look_3d = true         # the PS1 look below (false: the game's own picture, the sound alone)
//   snap = "240p"          # the grid the vertices land on: Off, 480p/360p/240p/120p
//   colour_8bit = true     # the 3D's colour targets RGBA8 (and the colour grading unsmoothed)
//   dither = true          # the picture in 15-bit colour through the PS1's 4x4 dither, before the HUD
//   unfiltered = true      # the 3D's textures sampled nearest: no bilinear, trilinear or anisotropic filtering
//   resolution = "Native"  # the 3D's size: Native (the Resolution setting, at most the window's), or 480p/360p/240p
//   sound = "Off"          # old-hardware sound: Off, PS1, Crunchy or Custom
//   sound_rate = 22050.0   # Custom's rate (Hz)
//   sound_bits = 8.0       # Custom's bits a sample
//
// Everything here is the host's (plugin API 18, and 21 for the sound): this
// file only says what the options ask for, at the start and again as they
// change in the game.
//
// Turned on and off in the game (API 21's BB_PLUGIN_LIVE_SWITCH): the host
// loads it off too, and it asks for nothing until the player's switch comes
// on - then everything at once, and taken back when it goes off.
//
// What is the 3D. The host's camera draws: the draws with the scene's depth
// that are not full-screen rect lists - the world, its characters, the sky,
// the effects and the transparency, the lights' volumes - as the upscaler's
// jitter takes them. The shadow maps (another depth), the post-processing (rect
// lists, YEBIS's chain) and the HUD and menus (no depth) keep the game's own.
//
// Vertex snap (scene_snap). The PS1's geometry engine wrote each vertex's
// place on the screen in whole pixels, so a polygon's corners jumped from
// pixel to pixel as it moved: the wobble of its 3D. The host moves each camera
// draw's vertex, as its vertex stage writes it, to the nearest corner of a
// grid `lines` cells high (as many across as the 3D's shape gives: 240p is
// 427 x 240 at 16:9, about 570 x 240 at 21:9) - 240 is the PS1's own screen.
// No "Native" (the 3D's own pixels): at a high Resolution its wobble is too
// small to see, the same as Off; a saved Native (or 0.1.0's "3D pixels")
// reads as Off.
//
// Unfiltered textures (scene_filter). The PS1 had no bilinear filtering: each
// pixel took its texel whole. The camera draws' samplers become nearest, the
// mip nearest too, anisotropy off; depth comparisons (the shadows' samples)
// keep theirs.
//
// 8-bit colour (scene_colour_bits). The scene's colour targets - the lit
// scene, the light accumulation, the reflections - are made RGBA8, as Kyo's
// "rgba8f color space" patch makes them, where the game has RGBA16F and the
// Performance patch R11G11B10F: the lighting bands and clips at white. The
// targets are made again when it changes.
//
// The colour grading. Each area's grading tables (map/mXX/mXX_cgrading.tpf:
// m24_00_CGrading0001 ...) are 16x256 textures, a 16x16x16 table each; every
// frame the game copies the one its draw parameters pick (or a blend of two)
// into one 16x16x16 table with a compute shader, and the post-processing's
// grading pass (ColorGrading.ppo) samples that smoothly between its entries.
// It is not the 3D's and stays smooth - unless 8-bit colour is on: then it is
// sampled nearest as well (grading_filter), and the grading steps from entry
// to entry, 16 shades a channel.
//
// Dithering (scene_dither, API 25). The PS1's GPU drew in 15-bit colour, 5
// bits a channel, and dithered its shading down to it with a 4x4 ordered
// pattern: the fine cross-hatch over its gradients. The host rounds the
// finished picture - the 3D and its post effects, before the HUD - to 5 bits
// a channel through the same matrix, a cell of it a pixel of the 3D (at 240p
// as coarse as the PS1's). On by default; on a host before API 25 it does
// nothing.
//
// The 3D's size (scene_scale). While the plugin is on the 3D is drawn at the
// Resolution and the game, its HUD and menus at the window's size, and the 3D
// is brought up to the window texel for texel, with no smoothing: no upscaler
// (FSR or DLSS stand aside while it is on) and no anti-aliasing (the game's
// FXAA, after that copy, would smooth the pixels again). The game and its HUD
// are always the window's size: a Resolution as large as the window or larger
// draws the 3D at the window's size too (not above it, filtered down); 480,
// 360p or 240p draw it that many lines high whatever the Resolution, as wide as the
// window's shape gives.
//
// Old-hardware sound (audio_crush, API 21). Everything the game plays, as its
// final mix reaches the device: low-passed (the PS1's SPU and its Gaussian
// interpolation lost the top end), held down to a lower rate and rounded to
// fewer bits, no dither, the host smoothing the held steps as a DAC's output
// filter did. PS1: 22050 Hz, 12 bits, low-passed at 10 kHz - muffled, a little
// grainy. Crunchy: 11025 Hz, 8 bits and no low-pass going in, so what is above
// half the rate folds back down - a cheap sampler's. Custom: its own rate and bits, low-passed
// just under the rate's Nyquist. Off by default; on a host before API 21 the
// option does nothing and the look works as before.
#include "bbhost_plugin.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN | BB_PLUGIN_LIVE_SWITCH,
    "ps1",
    "Bloodborne PS1",
    "0.4.0",
    "Kyo",
    "Vertex snapping like the PS1, optional 8-bit colour space and a decoupled render: the 3D at its own resolution, the HUD and menus "
    "native. Optionally the sound of an old console's sound chip. Turns on and off in the game.",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_BOOL | BB_OPT_LIVE, "look_3d", "PS1 look",
     "The 3D drawn as the PS1 drew it, with the settings below. Off: the game's own picture (the sound can still be old).", "true"},
    {BB_OPT_HEADING | BB_OPT_WHEN, "look", "The 3D only", "The HUD and the menus stay as they are, at the window's size.", nullptr, 0, 0,
     "look_3d=true"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "snap", "Vertex snap",
     "The grid the 3D's corners land on, as the PS1's landed on its pixels: they jump from cell to cell as they move. 240p "
     "is the PS1's own screen. The grid follows the screen's shape (240p is 427 cells across at 16:9).",
     "240p", 0, 0, "Off|480p|360p|240p|120p"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "unfiltered", "Unfiltered textures",
     "The 3D's textures sampled nearest - no bilinear, trilinear or anisotropic filtering - as the PS1 had none.", "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "colour_8bit", "8-bit colour",
     "The 3D's colour in 8 bits a channel, as the rgba8f patch makes it: the lighting bands and clips at white. The colour "
     "grading steps between its entries too, where it is smooth without it.",
     "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "dither", "Dithering",
     "The picture in the PS1's 15-bit colour, its gradients dithered with its 4x4 pattern - the fine cross-hatch over its "
     "shading - in cells of the 3D's pixels. The HUD and the menus are not dithered.",
     "true"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "resolution", "Resolution",
     "The 3D's resolution, brought up to the window pixel for pixel; the HUD and menus stay at the window's. Native: the "
     "Resolution setting, at most the window's. 480p, 360p and 240p: that many lines high, as wide as the window's shape gives "
     "(240p is 424 x 240 at 16:9, about 570 x 240 at 21:9), whatever the window's size.",
     "Native", 0, 0, "Native|480p|360p|240p"},
    {BB_OPT_HEADING, "audio", "Sound", "Everything the game plays.", nullptr},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "sound", "Old-hardware sound",
     "The game's sound as an old console's sound chip played it. PS1: 22050 Hz, 12 bits, the top end muffled above 10 kHz. "
     "Crunchy: 11025 Hz and 8 bits, unfiltered going in, so it aliases like a cheap sampler. Custom: the rate and bits below.",
     "Off", 0, 0, "Off|PS1|Crunchy|Custom"},
    {BB_OPT_FLOAT | BB_OPT_LIVE | BB_OPT_WHEN, "sound_rate", "Custom rate (Hz)",
     "Custom's sample rate: each sample held until the next at this rate, muffled just below half of it.", "22050.0", 2000.0, 48000.0,
     "sound=Custom"},
    {BB_OPT_FLOAT | BB_OPT_LIVE | BB_OPT_WHEN, "sound_bits", "Custom bits", "Custom's bits a sample: 16 is the game's own, 8 a crunchy sampler's.",
     "8.0", 1.0, 16.0, "sound=Custom"},
    {0}};

namespace {

const BbHostApi* api = nullptr;

bool truthy(const char* v) { return v && (std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0); }

// An option's text, or its default when the config does not say.
const char* option(const char* key, const char* fallback) {
    const char* v = api->config(key);
    return v && *v ? v : fallback;
}

// "240p" (or 0.1.0's "240 lines") -> 240; "Off", "Native" (0.1.0's "Resolution") -> 0.
std::uint32_t lines_of(const char* v) {
    const long n = v ? std::strtol(v, nullptr, 10) : 0;
    return n > 0 ? static_cast<std::uint32_t>(n) : 0;
}

// The snap's grid: "240p" -> 240; Off, and 0.2.0's Native and 0.1.0's
// "3D pixels" (the 3D's own pixels, no longer offered) -> 0, off.
std::uint32_t snap_of(const char* v) { return lines_of(v); }

// The plugin's switch: on a host before API 21 it is loaded only when on; on
// one after, BB_PLUGIN_LIVE_SWITCH loads it off too, to be turned on in game.
bool g_on = true;

void set_snap(const char* v) { api->scene_snap(snap_of(v)); }
void set_unfiltered(bool on) { api->scene_filter(on ? BB_FILTER_NEAREST : BB_FILTER_AS_IS); }
void set_colour_8bit(bool on) {
    api->scene_colour_bits(on ? 8 : 0);
    // The colour grading: smooth unless the colour is 8-bit.
    api->grading_filter(on ? BB_FILTER_NEAREST : BB_FILTER_AS_IS);
}
void set_resolution(const char* v) { api->scene_scale(BB_SCALE_NEAREST, lines_of(v)); }

bool has_dither(const BbHostApi* host) {
    return host->version >= 25 && host->size >= offsetof(BbHostApi, scene_dither) + sizeof(void*) && host->scene_dither;
}
void set_dither(bool on) {
    if (has_dither(api)) api->scene_dither(on ? 5 : 0);
}

// The PS1 look: on with the plugin and its "PS1 look".
bool look_on() { return g_on && truthy(option("ps1.look_3d", "true")); }

// The whole look as the options say, or (off) the game's own picture. The
// host ignores a setting that does not change, so asking again is free.
void apply_look() {
    if (!look_on()) {
        api->scene_snap(0);
        api->scene_filter(BB_FILTER_AS_IS);
        api->scene_colour_bits(0);
        api->grading_filter(BB_FILTER_AS_IS);
        api->scene_scale(BB_SCALE_AS_IS, 0);
        set_dither(false);
        return;
    }
    // config()'s text lasts until the next call: each value copied first.
    const std::string resolution = option("ps1.resolution", "Native"), snap = option("ps1.snap", "240p");
    const bool colour_8bit = truthy(option("ps1.colour_8bit", "true")), unfiltered = truthy(option("ps1.unfiltered", "true"));
    set_resolution(resolution.c_str());
    set_colour_8bit(colour_8bit);
    set_unfiltered(unfiltered);
    set_snap(snap.c_str());
    set_dither(truthy(option("ps1.dither", "true")));
}

// Old-hardware sound: the choice and Custom's rate and bits, kept so a change
// to one asks for all three again.
struct Sound {
    std::string choice = "Off";
    double rate = 22050.0, bits = 8.0;
} g_sound;

bool has_audio_crush(const BbHostApi* host) {
    return host->version >= 21 && host->size >= offsetof(BbHostApi, audio_crush) + sizeof(void*) && host->audio_crush;
}

void apply_sound() {
    if (!has_audio_crush(api)) return;
    const std::string& c = g_on ? g_sound.choice : std::string();
    if (c == "PS1") {
        api->audio_crush(22050, 12, 10000.0f);
    } else if (c == "Crunchy") {
        api->audio_crush(11025, 8, 0.0f);
    } else if (c == "Custom") {
        const int rate = static_cast<int>(g_sound.rate + 0.5), bits = static_cast<int>(g_sound.bits + 0.5);
        // 16 bits is the game's own: no truncation.
        api->audio_crush(rate > 0 ? rate : 0, bits > 0 && bits < 16 ? bits : 0, rate > 0 ? 0.45f * static_cast<float>(rate) : 0.0f);
    } else {
        api->audio_crush(0, 0, 0.0f);
    }
}

double number(const char* v, double fallback) {
    char* end = nullptr;
    const double n = v && *v ? std::strtod(v, &end) : 0.0;
    return v && *v && end != v ? n : fallback;
}

void read_sound() {
    // config()'s text lasts until the next call: each value copied first.
    g_sound.choice = option("ps1.sound", "Off");
    g_sound.rate = number(api->config("ps1.sound_rate"), 22050.0);
    g_sound.bits = number(api->config("ps1.sound_bits"), 8.0);
}

void on_option(const char* key, const char* value, void*) {
    if (!*key) {
        // The plugin turned on or off in game (BB_PLUGIN_LIVE_SWITCH).
        g_on = truthy(value);
        apply_look();
        apply_sound();
    } else if (std::strcmp(key, "sound") == 0) {
        g_sound.choice = value ? value : "Off";
        apply_sound();
    } else if (std::strcmp(key, "sound_rate") == 0) {
        g_sound.rate = number(value, g_sound.rate);
        apply_sound();
    } else if (std::strcmp(key, "sound_bits") == 0) {
        g_sound.bits = number(value, g_sound.bits);
        apply_sound();
    } else if (std::strcmp(key, "look_3d") == 0) {
        apply_look();  // the config already holds the new value
    } else if (!look_on()) {
        // The look's own settings: kept in the config for when it is on.
    } else if (std::strcmp(key, "snap") == 0) set_snap(value);
    else if (std::strcmp(key, "unfiltered") == 0) set_unfiltered(truthy(value));
    else if (std::strcmp(key, "colour_8bit") == 0) set_colour_8bit(truthy(value));
    else if (std::strcmp(key, "dither") == 0) set_dither(truthy(value));
    else if (std::strcmp(key, "resolution") == 0) set_resolution(value);
}

}  // namespace

// The 3D's size and its targets' format are decided as the game's code is
// patched, before bb_plugin_image: both are asked for here.
extern "C" BB_PLUGIN_EXPORT int bb_plugin_init(const BbHostApi* host) {
    api = host;
    if (host->version < 18 || host->size < offsetof(BbHostApi, grading_filter) + sizeof(void*)) {
        host->log("ps1: needs plugin API 18 (the 3D's look); this host has %u", host->version);
        return 1;
    }
    // Loaded off (BB_PLUGIN_LIVE_SWITCH): nothing asked for until it is turned on.
    g_on = host->version < 21 || truthy(host->config("plugins.ps1"));
    read_sound();
    if (!g_on) return 0;
    if (look_on()) apply_look();
    apply_sound();
    return 0;
}

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* host) {
    api = host;
    if (!host->eboot_is_109()) {
        // What bb_plugin_init asked for is taken back: unloaded, nothing could change it.
        host->scene_snap(0);
        host->scene_filter(BB_FILTER_AS_IS);
        host->scene_colour_bits(0);
        host->grading_filter(BB_FILTER_AS_IS);
        host->scene_scale(BB_SCALE_AS_IS, 0);
        if (has_dither(host)) host->scene_dither(0);
        if (has_audio_crush(host)) host->audio_crush(0, 0, 0.0f);
        host->log("ps1: not the 1.09 eboot; off");
        return 1;
    }
    host->on_option("ps1", &on_option, nullptr);
    // config()'s text lasts until the next call: each value copied first.
    if (!g_on) {
        host->log("ps1: Bloodborne PS1 loaded off; turned on in game (F9) it starts at once");
        return 0;
    }
    const std::string snap = option("ps1.snap", "240p"), resolution = option("ps1.resolution", "Native");
    const bool unfiltered = truthy(option("ps1.unfiltered", "true")), colour_8bit = truthy(option("ps1.colour_8bit", "true"));
    const bool look = look_on();
    if (look) {
        const bool dither = truthy(option("ps1.dither", "true"));
        host->log("ps1: Bloodborne PS1 on: vertex snap %s, textures %s, colour %s, %s, the 3D at %s, sound %s", snap.c_str(),
                  unfiltered ? "unfiltered" : "as the game filters them", colour_8bit ? "8-bit" : "as the game makes it",
                  !dither ? "no dither" : has_dither(host) ? "dithered to 15-bit" : "no dither (needs plugin API 25)", resolution.c_str(),
                  has_audio_crush(host) ? g_sound.choice.c_str() : "as it is (needs plugin API 21)");
    } else {
        host->log("ps1: Bloodborne PS1 on: the game's own picture, sound %s",
                  has_audio_crush(host) ? g_sound.choice.c_str() : "as it is (needs plugin API 21)");
    }
    return 0;
}
