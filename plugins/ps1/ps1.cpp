// Bloodborne PS1: the 3D drawn the way the first PlayStation drew it - its vertices
// snapped to the screen's grid, its textures unfiltered, its colour in 8 bits
// a channel, and the 3D at a Resolution of its own, brought up to the window
// pixel for pixel. The HUD and the menus stay as they are, at the window's
// size.
//
//   [plugins]
//   ps1 = true
//   [ps1]
//   snap = "240p"          # the grid the vertices land on: Off, Native (the 3D's own pixels), 480p/360p/240p/120p
//   colour_8bit = true     # the 3D's colour targets RGBA8 (and the colour grading unsmoothed)
//   unfiltered = true      # the 3D's textures sampled nearest: no bilinear, trilinear or anisotropic filtering
//   resolution = "Native"  # the 3D's size: Native (the Resolution setting, at most the window's), or 480p/360p/240p
//
// Everything here is the host's (plugin API 18): this file only says what the
// options ask for, at the start and again as they change in the game.
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
// 427 x 240 at 16:9, about 570 x 240 at 21:9) - 240 is the PS1's own screen; Native, the 3D's
// own pixels (no wobble you would see at a high Resolution).
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
// The 3D's size (scene_scale). While the plugin is on the 3D is drawn at the
// Resolution and the game, its HUD and menus at the window's size, and the 3D
// is brought up to the window texel for texel, with no smoothing: no upscaler
// (FSR or DLSS stand aside while it is on) and no anti-aliasing (the game's
// FXAA, after that copy, would smooth the pixels again). The game and its HUD
// are always the window's size: a Resolution as large as the window or larger
// draws the 3D at the window's size too (not above it, filtered down); 480,
// 360p or 240p draw it that many lines high whatever the Resolution, as wide as the
// window's shape gives.
#include "bbhost_plugin.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN,
    "ps1",
    "Bloodborne PS1",
    "0.2.0",
    "Kyo",
    "Vertex snapping like the PS1, optional 8-bit colour space and a decoupled render: the 3D at its own resolution, the HUD and menus "
    "native.",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, "look", "The 3D only", "The HUD and the menus stay as they are, at the window's size.", nullptr},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "snap", "Vertex snap",
     "The grid the 3D's corners land on, as the PS1's landed on its pixels: they jump from cell to cell as they move. 240p "
     "is the PS1's own screen; Native, the 3D's own pixels. The grid follows the screen's shape (240p is 427 cells across at 16:9).",
     "240p", 0, 0, "Off|Native|480p|360p|240p|120p"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "unfiltered", "Unfiltered textures",
     "The 3D's textures sampled nearest - no bilinear, trilinear or anisotropic filtering - as the PS1 had none.", "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "colour_8bit", "8-bit colour",
     "The 3D's colour in 8 bits a channel, as the rgba8f patch makes it: the lighting bands and clips at white. The colour "
     "grading steps between its entries too, where it is smooth without it.",
     "true"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "resolution", "Resolution",
     "The 3D's resolution, brought up to the window pixel for pixel; the HUD and menus stay at the window's. Native: the "
     "Resolution setting, at most the window's. 480p, 360p and 240p: that many lines high, as wide as the window's shape gives "
     "(240p is 424 x 240 at 16:9, about 570 x 240 at 21:9), whatever the window's size.",
     "Native", 0, 0, "Native|480p|360p|240p"},
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

// The snap's grid: Native (0.1.0's "3D pixels") the 3D's own pixels.
std::uint32_t snap_of(const char* v) {
    if (v && (std::strcmp(v, "Native") == 0 || std::strcmp(v, "3D pixels") == 0)) return BB_SNAP_PIXELS;
    return lines_of(v);
}

void set_snap(const char* v) { api->scene_snap(snap_of(v)); }
void set_unfiltered(bool on) { api->scene_filter(on ? BB_FILTER_NEAREST : BB_FILTER_AS_IS); }
void set_colour_8bit(bool on) {
    api->scene_colour_bits(on ? 8 : 0);
    // The colour grading: smooth unless the colour is 8-bit.
    api->grading_filter(on ? BB_FILTER_NEAREST : BB_FILTER_AS_IS);
}
void set_resolution(const char* v) { api->scene_scale(BB_SCALE_NEAREST, lines_of(v)); }

void on_option(const char* key, const char* value, void*) {
    if (std::strcmp(key, "snap") == 0) set_snap(value);
    else if (std::strcmp(key, "unfiltered") == 0) set_unfiltered(truthy(value));
    else if (std::strcmp(key, "colour_8bit") == 0) set_colour_8bit(truthy(value));
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
    set_resolution(option("ps1.resolution", "Native"));
    set_colour_8bit(truthy(option("ps1.colour_8bit", "true")));
    set_unfiltered(truthy(option("ps1.unfiltered", "true")));
    set_snap(option("ps1.snap", "240p"));
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
        host->log("ps1: not the 1.09 eboot; off");
        return 1;
    }
    host->on_option("ps1", &on_option, nullptr);
    // config()'s text lasts until the next call: each value copied first.
    const std::string snap = option("ps1.snap", "240p"), resolution = option("ps1.resolution", "Native");
    const bool unfiltered = truthy(option("ps1.unfiltered", "true")), colour_8bit = truthy(option("ps1.colour_8bit", "true"));
    host->log("ps1: Bloodborne PS1 on: vertex snap %s, textures %s, colour %s, the 3D at %s", snap.c_str(),
              unfiltered ? "unfiltered" : "as the game filters them", colour_8bit ? "8-bit" : "as the game makes it", resolution.c_str());
    return 0;
}
