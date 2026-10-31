// Lo-Hit: the map's lo-hit collision drawn in place of the map, under its effects.
//
//   [plugins]
//   lohit = true
//   [lohit]
//   pass = "3e/25"      # the scene pass it is drawn ahead of: category/sub-category (BBHOST_SCENE_CATEGORIES=1)
//   whole_map = false   # every loaded part's lo-hit, not only the part stood on and its neighbours
//
// What it draws. The world's hit manager (WorldHitMan) has a debug draw of
// the parts the player last stood on (its two slots at +0x2de8/+0x2df0): each
// frame sub_1c8a150 flags those parts for it (+0x130 bit 0x10) from the WORLD
// HIT MAN page's "show lo-hit debug" (0x596e8f0) and draws them into RendMan's
// first draw list. The plugin turns that read into `mov al, 1` (0x1c8a170), and
// takes the map's own geometry out of the frame (GROUP MASK > Map, 0x5527b88,
// read where the frame's draw groups are built, 0x1972125): the lo-hit is shown
// in its place.
//
// The parts connected to it. Each hit part (0x138 bytes: WorldHitMan +0x18,
// 0x20 a block, its +0x18 the groups, 0xb8 each, their +0x80 the parts) holds
// two 256-bit group masks, +0x74 and +0x94, which the residency pick
// (sub_1c86b50) tests against the player's. A hook on
// sub_1c8a150's entry flags for the draw (+0x130 bit 0x10) every part whose
// +0x94 mask shares a group with the one the player stands on (the
// first slot), and takes the flag off those it flagged the frame before: the
// part stood on and its neighbours, one step out. A part whose lo-hit is not
// loaded is not drawn (sub_1c832e0 needs +0xd8 = 6). With `whole_map` every
// part is flagged: all the lo-hit the map has loaded.
//
// Where it draws. The draw step (sub_22d82d0) plays RendMan's draw lists after
// the scene (sub_135d780 at 0x22d85e7, the lo-hit's list first), into the
// display buffers, over the effects and the post-processing - the effects
// vanish behind the collision. The host gives this plugin a command context in
// the scene just ahead of the first chunk of `pass` (on_scene_category, plugin
// API 15), and the list is played there with the parameters the draw step
// would give it; the draw step's own play of that list is skipped when it was
// played early. A frame with nothing at or after that pass plays it where the
// game does. The early play sets a token the draw step's next play of the list
// takes: a frame count would not do, the draw step running across the frame's
// start.
//
// Its colours. The late play draws into the display buffers after the
// tone-mapping: its colours, (0.18, 0, 0.3) and (0.18, 0.15, 0) (rodata
// 0x4d00840 / 0x4d00850, read by its draw sub_1c06490 alone), come out as
// they are (46, 0, 76). In the scene they go through the tone-mapping with
// the rest and came out 2.7x brighter (124, 11, 163), so the plugin writes
// colours there that come out as the late ones do (kEarlyColour; [lohit]
// early_colour = "r,g,b,r,g,b" replaces them). The colours are the
// launcher's, dark: the walls its deep violet (46, 20, 108), the floors its
// blue at half (32, 56, 128), in place of the game's purple and brown; the
// game's alpha, 0.5, kept.
//
// The post-effects (Giga Performance Patch's) are the host's Performance mode
// now (the setup window's main page, F10 > STARTUP).
#include "bbhost_plugin.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN,
    "lohit",
    "Lo-Hit",
    "0.3.9",
    "Kyo",
    "The map's lo-hit collision drawn in place of the map, with the effects showing over it.",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_TEXT | BB_OPT_LIVE, "pass", "Scene pass",
     "The scene's draw pass the lo-hit goes ahead of, as category/sub-category in hex (3e/25: the effects).", "3e/25"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "whole_map", "Whole map",
     "The lo-hit of every part of the map that is loaded. Off: the part you stand on and the parts connected to it.",
     "false"},
    {0}};

namespace {

const BbHostApi* api = nullptr;

constexpr std::uint64_t kRendMan = 0x5940298;    // RendMan: +0x10 the scene camera's owner, +0x20 the first draw list
constexpr std::uint64_t kGx = 0x59406c8;         // GX: +0x90 the output entry (+0x80/+0x84 its width and height)
constexpr std::uint64_t kPlayList = 0x135d780;   // (draw list, parameters): plays the list's other buffer
constexpr std::uint64_t kCameraInto = 0x2592d20; // (parameters, camera): its view and projection into them

using PlayFn = BB_GUEST_ABI void (*)(std::uint64_t list, void* params);
using CameraFn = BB_GUEST_ABI void (*)(void* params, std::uint64_t camera);

bool g_early = true;  // drawn in the scene; false when the draw lists' play cannot be hooked
int g_pass = 0xfa5;   // 3e/25: the effects (their items leave it with GROUP MASK Sfx off)
std::uint64_t g_rendman_slot = 0, g_gx_slot = 0;
PlayFn g_play = nullptr;
CameraFn g_camera_into = nullptr;

std::atomic<bool> g_played_early{false};  // set by the early play, taken by the draw step's next play
thread_local bool t_ours = false;          // our own play passing through the hook

template <class T> T at(std::uint64_t a) {
    T v;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(a)), sizeof v);
    return v;
}

std::uint64_t first_list() {
    const std::uint64_t rendman = g_rendman_slot ? at<std::uint64_t>(g_rendman_slot) : 0;
    return rendman ? at<std::uint64_t>(rendman + 0x20) : 0;
}

// The draw step's parameters for its draw lists (sub_22d82d0, 0x22d8435-0x22d85dc):
// the command context and its device, three identities, the output's size,
// 0.96 at +0x140, 1 at +0x218, then the scene camera's view and projection.
struct alignas(16) Params {
    std::uint8_t b[0x220];
};

void fill_params(Params& p, std::uint64_t cc) {
    std::memset(&p, 0, sizeof p);
    std::memcpy(p.b, &cc, 8);
    const std::uint64_t device = at<std::uint64_t>(cc + 8);
    std::memcpy(p.b + 8, &device, 8);
    static const float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (std::size_t off : {0x10, 0x90, 0xd0}) std::memcpy(p.b + off, id, sizeof id);
    const std::uint64_t gx = at<std::uint64_t>(g_gx_slot);
    const std::uint64_t out = gx ? at<std::uint64_t>(gx + 0x90) : 0;
    if (out) {
        std::memcpy(p.b + 0x110, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(out + 0x80)), 8);
    }
    const std::uint32_t near_scale = 0x3f75be0b;
    std::memcpy(p.b + 0x140, &near_scale, 4);
    p.b[0x218] = 1;
    const std::uint64_t rendman = at<std::uint64_t>(g_rendman_slot);
    const std::uint64_t owner = rendman ? at<std::uint64_t>(rendman + 0x10) : 0;
    const std::uint64_t camera = owner ? at<std::uint64_t>(owner + 0x18) : 0;
    if (camera) g_camera_into(&p, camera);
}

// In the scene, ahead of the pass's first chunk (a GX worker).
void play_early(std::uint64_t cc, void*) {
    const std::uint64_t list = first_list();
    if (!list || !cc) return;
    Params p;
    fill_params(p, cc);
    t_ours = true;
    g_play(list, &p);
    t_ours = false;
    g_played_early.store(true, std::memory_order_release);
}

// sub_135d780's entry: the draw step's own play of the first list, skipped
// when it was played early since its last play.
int on_play(BbHookCtx* ctx, void*) {
    if (t_ours || ctx->arg[0] != first_list()) return 0;
    return g_played_early.exchange(false, std::memory_order_acq_rel) ? 1 : 0;
}

// ---- The connected parts ----

constexpr std::uint64_t kHitDraw = 0x1c8a150;  // WorldHitMan's debug draw (WorldHitMan, ...), once a frame
constexpr std::uint32_t kPartSize = 0x138, kFlags = 0x130, kDrawLo = 0x10;
constexpr std::uint32_t kMaskOffset = 0x94;  // the mask that says which parts connect (+0x74: none at Central Yharnam)

std::atomic<bool> g_whole_map{false};  // every part flagged, not only the neighbours
std::uint64_t g_flagged[4096];
std::size_t g_n_flagged = 0;
int g_diag = -1;  // LOHIT_DIAG=1: the three masks' counts, every 120 frames
unsigned g_diag_calls = 0;

bool overlaps(std::uint64_t a, std::uint64_t b) {
    for (int k = 0; k < 4; ++k)
        if (at<std::uint64_t>(a + 8 * k) & at<std::uint64_t>(b + 8 * k)) return true;
    return false;
}

template <class F> void each_part(std::uint64_t whm, F&& f) {
    const std::int32_t blocks = at<std::int32_t>(whm + 0x14);
    const std::uint64_t block = at<std::uint64_t>(whm + 0x18);
    for (std::int32_t b = 0; b < blocks && block; ++b) {
        const std::int32_t groups = at<std::int32_t>(block + 0x20 * b + 0x10);
        const std::uint64_t group = at<std::uint64_t>(block + 0x20 * b + 0x18);
        for (std::int32_t g = 0; g < groups && group; ++g) {
            const std::int32_t parts = at<std::int32_t>(group + 0xb8 * g + 0x78);
            const std::uint64_t part = at<std::uint64_t>(group + 0xb8 * g + 0x80);
            for (std::int32_t k = 0; k < parts && part; ++k) f(part + static_cast<std::uint64_t>(kPartSize) * k);
        }
    }
}

void set_draw(std::uint64_t part, bool on) {
    std::uint8_t* flags = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(part + kFlags));
    *flags = on ? (*flags | kDrawLo) : (*flags & ~kDrawLo);
}

// sub_1c8a150's entry (the main thread): last frame's neighbours unflagged,
// this frame's flagged (with `whole_map`, every part).
int on_hit_draw(BbHookCtx* ctx, void*) {
    const std::uint64_t whm = ctx->arg[0];
    const std::uint64_t standing = whm ? at<std::uint64_t>(whm + 0x2de8) : 0;
    for (std::size_t i = 0; i < g_n_flagged; ++i) set_draw(g_flagged[i], false);
    g_n_flagged = 0;
    if (!standing) return 0;
    if (g_diag > 0 && ++g_diag_calls % 120 == 0) {
        int n74 = 0, n94 = 0, cross = 0, total = 0, loaded = 0;
        each_part(whm, [&](std::uint64_t part) {
            ++total;
            loaded += at<std::uint8_t>(part + 0xd8) == 6;
            n74 += overlaps(part + 0x74, standing + 0x74);
            n94 += overlaps(part + 0x94, standing + 0x94);
            cross += overlaps(part + 0x74, standing + 0x94) || overlaps(part + 0x94, standing + 0x74);
        });
        api->log("lohit diag: %d parts, %d with lo-hit loaded; sharing a group with the one stood on: +0x74 %d, +0x94 %d, "
                 "across %d", total, loaded, n74, n94, cross);
    }
    const bool whole = g_whole_map.load(std::memory_order_relaxed);
    each_part(whm, [&](std::uint64_t part) {
        if (part == standing || g_n_flagged == sizeof g_flagged / sizeof g_flagged[0]) return;
        if (!whole && !overlaps(part + kMaskOffset, standing + kMaskOffset)) return;
        if (at<std::uint8_t>(part + kFlags) & kDrawLo) return;  // flagged by the game (its other slot)
        set_draw(part, true);
        g_flagged[g_n_flagged++] = part;
    });
    return 0;
}

// ---- The sites ----

struct Site {
    const char* what;
    std::uint64_t at;
    std::uint8_t n;
    bool data;
    std::uint8_t game[8], ours[8];
};

// The lo-hit shown in place of the map.
const Site kLoHit[] = {
    // sub_1c8a150: mov al, [WORLD HIT MAN's show lo-hit debug] -> mov al, 1
    {"the lo-hit's switch", 0x1c8a170, 6, false, {0x8a, 0x05, 0x7a, 0x47, 0xce, 0x03}, {0xb0, 0x01, 0x0f, 0x1f, 0x40, 0x00}},
    // GROUP MASK > Map 1 -> 0: the map's geometry left out of the frame's draw groups
    {"the map's draw group", 0x5527b88, 1, true, {0x01}, {0x00}},
};

// Each site to ours; one already ours (a patch manifest wrote the same) counts as done.
template <std::size_t N> int apply(const Site (&sites)[N]) {
    int done = 0;
    for (const Site& s : sites) {
        std::uint8_t now[8];
        if (api->read(s.at, now, s.n) != 0) continue;
        if (std::memcmp(now, s.ours, s.n) == 0) {
            ++done;
            continue;
        }
        if (std::memcmp(now, s.game, s.n) != 0) {
            char hex[32] = {0};
            for (int k = 0; k < s.n; ++k) std::snprintf(hex + 3 * k, 4, "%02x ", now[k]);
            api->log("lohit: %s at %#llx holds %sneither the game's nor ours; left alone", s.what,
                     static_cast<unsigned long long>(s.at), hex);
            continue;
        }
        const int r = s.data ? api->write(s.at, s.ours, s.n) : api->patch(s.at, s.game, s.ours, s.n);
        if (r == 0) ++done;
        else api->log("lohit: %s at %#llx not written (%d)", s.what, static_cast<unsigned long long>(s.at), r);
    }
    return done;
}

// ---- The colours ----

constexpr std::uint64_t kColours = 0x4d00840;  // two vec4s, read by the lo-hit draw sub_1c06490 alone
constexpr float kGameColour[8] = {0.18f, 0.0f, 0.3f, 0.5f, 0.18f, 0.15f, 0.0f, 0.5f};
constexpr float kAlpha = 0.5f;  // the game's
// The launcher's deep violet and its blue at half (launcher_ui.cpp kRamp): walls, then floors.
constexpr float kLateColour[8] = {0.180f, 0.078f, 0.424f, kAlpha, 0.125f, 0.220f, 0.502f, kAlpha};
// Measured at Central Yharnam's lamp: the scene's curve there is about 1.88 x^1.3
// (linear light) before the display's gamma, so (0.0405, 0, 0.0817) comes out
// at 44-46, 3, 76 as the late play's (0.18, 0, 0.3) does at 46, 0, 76. Black
// keeps a lift of 3-4. Brighter colours come out brighter than that curve
// says: the launcher's, measured at the same place and corrected once.
float kEarlyColour[8] = {0.0258f, 0.0033f, 0.0778f, kAlpha, 0.0126f, 0.0287f, 0.1657f, kAlpha};

void read_early_colour(const char* v) {
    if (!v || !*v) return;
    float c[6];
    int n = 0;
    const char* at = v;
    while (n < 6 && *at) {
        char* end = nullptr;
        c[n] = std::strtof(at, &end);
        if (end == at) break;
        ++n;
        at = *end == ',' ? end + 1 : end;
    }
    if (n != 6) {
        api->log("lohit: early_colour '%s' is not six numbers; kept", v);
        return;
    }
    const float out[8] = {c[0], c[1], c[2], kAlpha, c[3], c[4], c[5], kAlpha};
    std::memcpy(kEarlyColour, out, sizeof out);
}

void write_colours() {
    const int r = api->patch(kColours, kGameColour, g_early ? kEarlyColour : kLateColour, sizeof kGameColour);
    if (r != 0) api->log("lohit: 0x4d00840 is not the lo-hit's colours (%d); kept as they are", r);
}

// "3e/05" (category/sub-category, hex) -> 0xf85; -1 when it is not one.
int parse_pass(const char* v) {
    if (!v || !*v) return -1;
    char* end = nullptr;
    const long cat = std::strtol(v, &end, 16);
    if (end == v || *end != '/' || cat < 0 || cat > 0x3f) return -1;
    const char* sub_at = end + 1;
    const long sub = std::strtol(sub_at, &end, 16);
    if (end == sub_at || *end || sub < 0 || sub > 0x3f) return -1;
    return static_cast<int>(cat << 6 | sub);
}

void set_pass(const char* v) {
    const int p = parse_pass(v);
    if (p < 0) {
        api->log("lohit: pass '%s' is not category/sub-category in hex (3e/00); keeping %02x/%02x", v ? v : "", g_pass >> 6,
                 g_pass & 0x3f);
    } else {
        g_pass = p;
    }
    if (g_early) api->on_scene_category(g_pass, &play_early, nullptr);
}

bool truthy(const char* v) {
    return v && (std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0 || std::strcmp(v, "on") == 0);
}

void on_option(const char* key, const char* value, void*) {
    if (std::strcmp(key, "pass") == 0) set_pass(value);
    else if (std::strcmp(key, "whole_map") == 0) g_whole_map.store(truthy(value), std::memory_order_relaxed);
}

}  // namespace

extern "C" BB_PLUGIN_EXPORT int bb_plugin_init(const BbHostApi* host) {
    api = host;
    if (host->version < 15 || host->size < offsetof(BbHostApi, on_scene_category) + sizeof(void*)) {
        host->log("lohit: needs plugin API 15 (on_scene_category); this host has %u", host->version);
        return 1;
    }
    return 0;
}

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* host) {
    api = host;
    if (!host->eboot_is_109()) {
        host->log("lohit: not the 1.09 eboot; off");
        return 1;
    }
    g_rendman_slot = host->guest_addr(kRendMan);
    g_gx_slot = host->guest_addr(kGx);
    g_play = reinterpret_cast<PlayFn>(static_cast<std::uintptr_t>(host->guest_addr(kPlayList)));
    g_camera_into = reinterpret_cast<CameraFn>(static_cast<std::uintptr_t>(host->guest_addr(kCameraInto)));
    if (!g_rendman_slot || !g_gx_slot || !g_play || !g_camera_into) return 1;
    // push rbp; mov rbp, rsp; push r15; push r14; push rbx; push rax; mov r14, rsi; mov rbx, rdi
    static const std::uint8_t play_entry[16] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                0x53, 0x50, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb};
    if (host->hook2(kPlayList, play_entry, sizeof play_entry, &on_play, nullptr) != 0) {
        host->log("lohit: 0x135d780 is not the draw lists' play; the lo-hit is drawn where the game draws it");
        g_early = false;
    }
    host->on_option("lohit", &on_option, nullptr);
    const int shown = apply(kLoHit);
    if (const char* d = std::getenv("LOHIT_DIAG")) g_diag = std::atoi(d);
    g_whole_map.store(truthy(host->config("lohit.whole_map")), std::memory_order_relaxed);
    // push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp, 0x48
    static const std::uint8_t hit_draw_entry[17] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                                    0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x48};
    if (host->hook2(kHitDraw, hit_draw_entry, sizeof hit_draw_entry, &on_hit_draw, nullptr) != 0)
        host->log("lohit: 0x1c8a150 is not WorldHitMan's debug draw; the connected parts are not drawn");
    if (g_early) read_early_colour(host->config("lohit.early_colour"));
    write_colours();
    if (g_early) {
        const char* c = host->config("lohit.pass");
        set_pass(c && *c ? c : "3e/25");
    }
    host->log("lohit: the lo-hit shown in place of the map (%d of 2 sites), %s (ahead of the scene's %02x/%02x), %s", shown,
              g_early ? "under the effects" : "over everything", g_pass >> 6, g_pass & 0x3f,
              g_whole_map.load() ? "the whole map" : "the part stood on and its neighbours");
    return 0;
}
