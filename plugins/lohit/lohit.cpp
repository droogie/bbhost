// Lo-Hit: the map's lo-hit collision drawn in place of the map, under its effects.
//
//   [plugins]
//   lohit = true
//   [lohit]
//   pass = "3e/25"      # the scene pass it is drawn ahead of: category/sub-category (BBHOST_SCENE_CATEGORIES=1)
//   whole_map = true    # every loaded part's lo-hit, not only the part stood on and its neighbours
//   show_hi_hit = false # the hi-hit (what characters and objects collide with) in place of the lo-hit
//   wall_r = 46  wall_g = 20  wall_b = 108    # the walls' colour, as it comes out on the screen (0-255)
//   floor_r = 32 floor_g = 56 floor_b = 128   # the floors'
//   occluder_pass = "3e/04"  # where its depth-only play goes, for occlusion (ahead of the scene's objects)
//   ambient = 30        # the light no lamp reaches, in thousandths (the map's comes from its lightmaps)
//   sun = 0             # older hosts (unlit): faces away from a fixed high sun darkened, 0-100
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
// colours there that come out as the late ones do. The colours are options,
// as they come out on the screen (0-255): by default the launcher's, dark -
// the walls its deep violet (46, 20, 108), the floors its blue at half (32,
// 56, 128) - in place of the game's purple and brown, at alpha 1 (the game's
// 0.5 drew the same: its faces are opaque already). A changed colour builds
// the meshes again (their shade is in their vertices).
//
// Hi-hit. Each part also holds its hi-hit (+0x110, loaded at +0x100 = 6),
// drawn by sub_1c06380 into the same list for the flag bit 0x20, in its own
// colours (rodata 0x4d00820 / 0x4d00830, (0.045, 0, 0.3) and (0.045, 0.15,
// 0)), when the page's "show hi-hit debug" (0x596e8f1, read at 0x1c8a176) is
// on. With `show_hi_hit` that read is the one made 1 instead of the lo-hit's, the
// parts are flagged with 0x20, and its colours are the lo-hit's. Its meshes go
// through the same draw (sub_13dc900), so they are built once as well.
//
// Occlusion. The game culls with occlusion queries: each object's draw
// (0x2650690) is wrapped in a query (GX Begin 0x2569ec0, End 0x256a070) and
// drawn the next frame predicated on it. With the map's geometry left out,
// nothing is in the depth buffer when the objects draw (3e/04 on), so nothing
// behind a wall is culled, and the lo-hit drawn at 3e/25 is too late to be an
// occluder. So (plugin API 17; always, LOHIT_OCCLUSION=0 for a test without
// it) the list is played a second time
// ahead of `occluder_pass` with no colour written (BB_DRAW_DEPTH_ONLY), and
// the coloured play at `pass` tests less-or-equal against that depth
// (BB_DRAW_DEPTH_EQUAL). Drawn with its colour that early, it goes into the
// G-buffer and comes out lit, near black.
//
// Lit (host API 19; always, where the host has it). Drawn at 3e/25 the lo-hit is flat colour over
// the lit scene: the lighting, the fog in it included, has run by then
// (every pass from 3e/13 to 3e/25 gave the same frame). Lit, the coloured
// play goes ahead of 3e/0c, where the G-buffer is still bound, styled
// BB_DRAW_GBUFFER: the host draws it with its own pixel shader into the
// G-buffer's targets - the colours as the albedo (linear: the game's lighting
// and tone-mapping make the screen colour), each triangle's world normal in
// its vertices' alpha (normal_batch) - and marks the stencil as the map's
// draws do, which the lights test. The game then lights it as the map: the
// sun, the lamps, the shadows, the fog. The map's lightmaps give it its light
// where no lamp reaches; the lo-hit has none, so `ambient` stands in.
//
// The post-effects (Giga Performance Patch's) are the host's Performance mode
// now (the setup window's main page, F10 > Game).
#include "bbhost_plugin.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <unordered_map>
#include <vector>

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN,
    "lohit",
    "Lo-Hit",
    "0.5.2",
    "Kyo",
    "The map's lo-hit collision drawn in place of the map, with the effects showing over it.",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_TEXT | BB_OPT_LIVE, "pass", "Scene pass",
     "The scene's draw pass the lo-hit goes ahead of, as category/sub-category in hex (3e/25: the effects).", "3e/25"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "whole_map", "Whole map",
     "The lo-hit of every part of the map that is loaded. Off: the part you stand on and the parts connected to it.",
     "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "show_hi_hit", "Hi-hit",
     "The hi-hit - what characters and objects collide with, in its full detail - in place of the lo-hit.", "false"},
    {BB_OPT_INT | BB_OPT_LIVE, "ambient", "Ambient light",
     "The light where no lamp or sun reaches (the map's own comes from its lightmaps). 0: black there.", "30",
     0, 100},
    {BB_OPT_HEADING, "colours", "Colours", "As they come out on the screen, 0-255. Lo-hit and hi-hit alike.", nullptr},
    {BB_OPT_INT | BB_OPT_LIVE, "wall_r", "Walls: red", nullptr, "46", 0, 255},
    {BB_OPT_INT | BB_OPT_LIVE, "wall_g", "Walls: green", nullptr, "20", 0, 255},
    {BB_OPT_INT | BB_OPT_LIVE, "wall_b", "Walls: blue", nullptr, "108", 0, 255},
    {BB_OPT_INT | BB_OPT_LIVE, "floor_r", "Floors: red", nullptr, "32", 0, 255},
    {BB_OPT_INT | BB_OPT_LIVE, "floor_g", "Floors: green", nullptr, "56", 0, 255},
    {BB_OPT_INT | BB_OPT_LIVE, "floor_b", "Floors: blue", nullptr, "128", 0, 255},
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
bool g_cache_on = true;    // LOHIT_CACHE=0: the game's own build every frame
bool g_buffers_on = true;  // LOHIT_BUFFERS=0: the copies played to sub_13673c0 a batch at a time
int g_pass = 0xfa5;   // 3e/25: the effects (their items leave it with GROUP MASK Sfx off)
std::uint64_t g_rendman_slot = 0, g_gx_slot = 0;
PlayFn g_play = nullptr;
CameraFn g_camera_into = nullptr;

// Game frames (on_frame, the main thread), and the frame each play last ran
// in (+1: 0 never). The draw step's own play of the list is skipped while the
// early play has run in the last two frames. It was a flag the early play set
// and the draw step took, but the two run on different threads in no fixed
// order: some frames the draw step came first and drew the list flat over
// the lit one, some the flag was taken a frame late and neither drew it (the
// "lit by the game" flicker: lit, flat, gone). By frame, not by call: a
// worker may record a frame's passes while the main thread is a frame on.
std::atomic<std::uint64_t> g_frame{0};
std::atomic<std::uint64_t> g_early_frame{0};
void frame_tick(void*) { g_frame.fetch_add(1, std::memory_order_relaxed); }
void stamp(std::atomic<std::uint64_t>& at) { at.store(g_frame.load(std::memory_order_relaxed) + 1, std::memory_order_release); }
bool recent(const std::atomic<std::uint64_t>& at) {
    const std::uint64_t f = at.load(std::memory_order_acquire);
    return f && g_frame.load(std::memory_order_relaxed) + 1 - f <= 2;
}
thread_local bool t_ours = false;          // our own play passing through the hook
int g_diag = -1;  // LOHIT_DIAG=1: the masks' and the meshes' counts, every 120 frames

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

bool g_can_occlude = false;               // plugin API 17: the second segment and the depth styles
std::atomic<bool> g_occlusion{true};      // the depth-only play ahead of g_occluder_pass
int g_occluder_pass = 0xf84;              // 3e/04: ahead of the scene's objects
// The frame the depth-only play last ran in. The coloured play lands on that
// depth (BB_DRAW_DEPTH_EQUAL) when it is on and has played in the last two
// frames (recent):
// the two plays run on GX workers that build their passes side by side, so
// this frame's depth may be recorded after the coloured play - the GPU still
// runs 3e/04 ahead of 3e/0c. It was a flag the depth play set and the
// coloured play took: whenever the coloured play's worker came first the
// flag was not set, and that frame's lo-hit went without its depth test and
// without the G-buffer style - drawn flat, unlit (the "lit by the game"
// flicker, frame to frame).
std::atomic<std::uint64_t> g_depth_frame{0};
bool depth_play_on() { return g_can_occlude && g_occlusion.load(std::memory_order_relaxed) && g_cache_on && g_buffers_on; }
// Lit (host API 19): the coloured play into the G-buffer at kLitPass, where
// the game lights it (BB_DRAW_GBUFFER; 0 when not lit).
constexpr int kLitPass = 0xf8c;  // 3e/0c: the G-buffer still bound, ahead of the lighting
bool g_can_light = false;        // host API 19 and the depth-only play
std::atomic<bool> g_lit{false};
std::atomic<int> g_gbuffer_style{0};

// In the scene, ahead of the pass's first chunk (a GX worker).
void play_early(std::uint64_t cc, void*) {
    const std::uint64_t list = first_list();
    if (!list || !cc) return;
    Params p;
    fill_params(p, cc);
    const bool on_depth = depth_play_on() && recent(g_depth_frame);
    // The G-buffer style whenever lit, the depth test with it.
    const int style = (on_depth ? BB_DRAW_DEPTH_EQUAL : 0) | g_gbuffer_style.load(std::memory_order_relaxed);
    if (style) api->draw_style(style);
    t_ours = true;
    g_play(list, &p);
    t_ours = false;
    if (style) api->draw_style(BB_DRAW_AS_IS);
    stamp(g_early_frame);
}

// Ahead of the scene's objects: the list's depth alone, for the occlusion tests.
void play_depth(std::uint64_t cc, void*) {
    // The coloured play lands on this depth only when both draw the same
    // vertices: the meshes' buffers (built every frame, it comes out black).
    if (!depth_play_on()) return;
    const std::uint64_t list = first_list();
    if (!list || !cc) return;
    Params p;
    fill_params(p, cc);
    api->draw_style(BB_DRAW_DEPTH_ONLY);
    t_ours = true;
    g_play(list, &p);
    t_ours = false;
    api->draw_style(BB_DRAW_AS_IS);
    stamp(g_depth_frame);
}

// sub_135d780's entry: the draw step's own play of the first list, skipped
// while it is played early (in the last two frames).
int on_play(BbHookCtx* ctx, void*) {
    if (t_ours || ctx->arg[0] != first_list()) return 0;
    return recent(g_early_frame) ? 1 : 0;
}

// ---- The meshes, built once ----
//
// The list's lo-hit command (sub_13dc3f0) draws a part's mesh with
// sub_13dc900 (renderer, mesh): every frame it walks the mesh's Havok shapes
// into display geometry (sub_2e6d1f0, allocating), and its drawer (sub_13dcb00,
// vtable 0x56f5cd0) computes each triangle's face normal and shade
// (sub_13dd010) into three 0x30-byte vertices (position, colour), handing
// every 1024 triangles to sub_13673c0 (renderer, vertices, triangles, 0x30
// bytes of the drawer). That sets the camera's constants and draws the batch
// twice with sub_2590be0, which copies the vertices into GX's upload ring
// each time. For the whole map (some 250,000 triangles at Central Yharnam)
// that is 14-20 ms a frame on one GX worker and 70 MiB of copies, with the
// GPU 30% busy.
//
// The plugin runs that body once per mesh with sub_13673c0's batches copied,
// puts them in one immutable vertex buffer of the mesh's own (GX's
// CreateBuffer, 0x2565c10), and on the frames after hands sub_13673c0 the
// whole mesh at once, with sub_2590be0's two draws binding that buffer where
// they would have copied: two draws a mesh, nothing built or copied.
//
// Its shade. The drawer blends each triangle between the two colours by
// |normal . the camera's forward| (sub_13dd010, the drawer's view rows at
// +0xa0..+0xc8): faces toward the camera the first colour, faces edge-on the
// second, recomputed as the camera turns. Built once, a mesh would keep the
// shade of the frame it was built in, each mesh another frame's. So the
// plugin shades the copies itself, from the world: a wall the first colour
// (violet), a floor the second (blue), walls a little darker the more they
// face across kShadeLight, so their corners still show. Triangles whose
// colour is not between the two (another material's) keep it.

constexpr std::uint64_t kMeshDraw = 0x13dc900;   // (renderer, mesh)
constexpr std::uint64_t kBatchDraw = 0x13673c0;  // (renderer, vertices, triangles, drawer +0x30)
constexpr std::uint64_t kDrawerMake = 0x13dcb00, kDrawerFree = 0x13dcc40;  // (drawer, [renderer +0x30], renderer); (drawer)
constexpr std::uint64_t kShapesMake = 0x2e6d1f0, kShapesDraw = 0x2e6d9d0, kShapesFree = 0x2e6d890;  // (shapes, mesh); (shapes, drawer); (shapes)
constexpr std::uint64_t kDrawUp = 0x2590be0;         // (helper, topology, vertices, data, stride): a batch's draw from a copy
constexpr std::uint64_t kImmediateDraw = 0x25696c0;  // gx_immediate_draw(context, context + 0xb730, vertices, first)
constexpr std::uint64_t kCreateBuffer = 0x2565c10;   // gx_create_buffer(device, &holder, desc, initial data, 0, 0, placement)
constexpr std::size_t kMeshKey = 0x90;  // the mesh record (its table's stride): changed, the mesh is built again
constexpr std::size_t kVertex = 0x30;

using BatchFn = BB_GUEST_ABI void (*)(std::uint64_t renderer, const void* verts, std::uint64_t tris, const void* xf);
using MakeDrawerFn = BB_GUEST_ABI void (*)(void* drawer, std::uint64_t r30, std::uint64_t renderer);
using ShapesFn = BB_GUEST_ABI void (*)(void* shapes, std::uint64_t mesh);
using ShapesDrawFn = BB_GUEST_ABI void (*)(void* shapes, void* drawer);
using ObjFn = BB_GUEST_ABI void (*)(void* obj);
using DrawFn = BB_GUEST_ABI void (*)(std::uint64_t ctx, std::uint64_t state, std::uint64_t count, std::uint64_t first);
using CreateFn = BB_GUEST_ABI std::uint32_t (*)(std::uint64_t device, std::uint64_t* out, const void* desc, const void* init,
                                                std::uint64_t, std::uint64_t, std::uint64_t placement);
using FreeFn = BB_GUEST_ABI void (*)(std::uint64_t allocator, std::uint64_t object);

BatchFn g_batch_draw = nullptr;
MakeDrawerFn g_drawer_make = nullptr;
ObjFn g_drawer_free = nullptr, g_shapes_free = nullptr;
ShapesFn g_shapes_make = nullptr;
ShapesDrawFn g_shapes_draw = nullptr;
DrawFn g_immediate_draw = nullptr;
CreateFn g_create_buffer = nullptr;

template <class T> void put(std::uint64_t a, T v) {
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(a)), &v, sizeof v);
}

struct Batch {
    std::uint32_t tris;
    alignas(16) std::uint8_t xf[0x30];  // read with vmovdqa
    std::vector<std::uint8_t> verts;
};
struct Mesh {
    std::uint8_t key[kMeshKey];
    bool built = false;
    std::uint32_t gen = 0;  // g_mesh_gen it was built for
    std::vector<Batch> batches;  // without a buffer of its own
    std::uint64_t buffer = 0;    // the GX buffer's holder
    std::uint32_t tris = 0;
    alignas(16) std::uint8_t xf[0x30];
    std::uint32_t used = 0;  // g_seconds at its last play
    int busy = 0;            // GX workers building or playing it (the sweep leaves it)
};

std::mutex g_mesh_mu;
std::unordered_map<std::uint64_t, Mesh> g_meshes;
std::vector<std::pair<std::uint64_t, std::uint32_t>> g_dead_buffers;  // holders to release, with the second they went
std::atomic<std::uint32_t> g_seconds{0};
thread_local Mesh* t_capture = nullptr;
thread_local bool t_mesh_ours = false;  // a mesh drawn by us (built or played)
thread_local std::uint64_t t_bind = 0;  // our whole-mesh play: the buffer sub_2590be0 binds
thread_local int t_pass = 0;            // sub_2590be0's calls in one sub_13673c0: 0 the faces, 1 the edges
std::atomic<std::uint32_t> g_mesh_gen{0};  // bumped when the colours change: every mesh built again
std::atomic<std::uint64_t> g_built{0}, g_played{0}, g_tris_played{0}, g_draws{0};

std::uint32_t seconds_now() {
    using namespace std::chrono;
    return static_cast<std::uint32_t>(duration_cast<seconds>(steady_clock::now().time_since_epoch()).count());
}

// The game's release of a GX object (as at 0x257a0d5): the last reference
// runs its in-place destructor (a buffer's frees its resource, 0x256d690) and
// gives it back to its allocator.
void release_buffer(std::uint64_t holder) {
    auto* refs = reinterpret_cast<std::atomic<std::int32_t>*>(static_cast<std::uintptr_t>(holder + 0x10));
    if (refs->fetch_sub(1) != 1) return;
    const std::uint64_t allocator = at<std::uint64_t>(holder + 8);
    reinterpret_cast<ObjFn>(static_cast<std::uintptr_t>(at<std::uint64_t>(at<std::uint64_t>(holder))))(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(holder)));
    reinterpret_cast<FreeFn>(static_cast<std::uintptr_t>(at<std::uint64_t>(at<std::uint64_t>(allocator) + 0x70)))(allocator, holder);
}

// The batches into one immutable vertex buffer (D3D11's description: bytes,
// usage 1, bind 1 vertex; the initial data's first field its pointer).
std::uint64_t make_buffer(const std::vector<Batch>& batches, std::uint32_t tris) {
    const std::uint64_t gx = g_gx_slot ? at<std::uint64_t>(g_gx_slot) : 0;
    const std::uint64_t device = gx ? at<std::uint64_t>(gx) : 0;
    if (!device || !tris) return 0;
    std::vector<std::uint8_t> all;
    all.reserve(std::size_t(tris) * 3 * kVertex);
    for (const Batch& b : batches) all.insert(all.end(), b.verts.begin(), b.verts.end());
    const std::uint32_t desc[6] = {static_cast<std::uint32_t>(all.size()), 1, 1, 0, 0, 0};
    const std::uint64_t init[2] = {reinterpret_cast<std::uint64_t>(all.data()), 0};
    std::uint64_t holder = 0;
    if (g_create_buffer(device, &holder, desc, init, 0, 0, 0) != 0) return 0;
    return holder;
}

const float* g_shade_colours = nullptr;  // the two colours written for the game (write_colours)
std::atomic<int> g_sun{0};               // the sun option, 0-100: how much faces away from it darken

struct V3 {
    float x, y, z;
};
V3 sub3(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

// The batch's triangles shaded from the world (see above): vertex = position
// (vec4) at 0, colour (vec4) at 0x10.
void shade_batch(Batch& b) {
    const float* c = g_shade_colours;
    if (!c) return;
    const float d[3] = {c[0] - c[4], c[1] - c[5], c[2] - c[6]};
    const float dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if (dd <= 0.0f) return;
    constexpr float kLx = 0.6f, kLz = 0.8f;  // kShadeLight: across the map, level
    constexpr V3 kSun = {0.3f, 0.86f, 0.41f};  // high, to one side (unit)
    const float sun = static_cast<float>(g_sun.load(std::memory_order_relaxed)) / 100.0f;
    for (std::uint32_t t = 0; t < b.tris; ++t) {
        std::uint8_t* tri = b.verts.data() + std::size_t(t) * 3 * kVertex;
        float col[4];
        std::memcpy(col, tri + 0x10, sizeof col);
        // On the line between the two colours: where (0 the second, 1 the first).
        const float r[3] = {col[0] - c[4], col[1] - c[5], col[2] - c[6]};
        const float along = (r[0] * d[0] + r[1] * d[1] + r[2] * d[2]) / dd;
        const float off[3] = {r[0] - along * d[0], r[1] - along * d[1], r[2] - along * d[2]};
        if (along < -0.01f || along > 1.01f || off[0] * off[0] + off[1] * off[1] + off[2] * off[2] > 1e-6f * (dd + 1e-6f)) continue;
        V3 p[3];
        for (int k = 0; k < 3; ++k) std::memcpy(&p[k], tri + k * kVertex, sizeof(V3));
        const V3 e1 = sub3(p[1], p[0]), e2 = sub3(p[2], p[0]);
        V3 n = {e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x};
        const float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        if (len <= 0.0f) continue;
        n = {n.x / len, n.y / len, n.z / len};
        const float wall = std::sqrt(std::max(0.0f, 1.0f - n.y * n.y));  // 1 upright, 0 level
        const float across = wall > 1e-4f ? std::fabs(n.x * kLx + n.z * kLz) / wall : 0.0f;
        const float k = wall * (1.0f - 0.3f * across);
        // The sun: faces turned to kSun lit, faces turned away down to a quarter.
        const float lit = std::max(0.0f, n.x * kSun.x + n.y * kSun.y + n.z * kSun.z);
        const float f = 1.0f - sun + sun * (0.25f + 0.75f * lit);
        const float out[4] = {f * (c[4] + k * d[0]), f * (c[5] + k * d[1]), f * (c[6] + k * d[2]), col[3]};
        for (int v = 0; v < 3; ++v) std::memcpy(tri + v * kVertex + 0x10, out, sizeof out);
    }
}

// Lit (g_lit): each triangle's world normal in its
// vertices' alpha, as the host's G-buffer shader reads it (gbuffer_flat.frag):
// -(1 + u + 2048 v), u and v its octahedral coordinates in 11 bits.
void normal_batch(Batch& b) {
    for (std::uint32_t t = 0; t < b.tris; ++t) {
        std::uint8_t* tri = b.verts.data() + std::size_t(t) * 3 * kVertex;
        V3 p[3];
        for (int k = 0; k < 3; ++k) std::memcpy(&p[k], tri + k * kVertex, sizeof(V3));
        const V3 e1 = sub3(p[1], p[0]), e2 = sub3(p[2], p[0]);
        V3 n = {e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x};
        const float l1 = std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z);
        if (l1 <= 0.0f) continue;
        float u = n.x / l1, v = n.y / l1;
        if (n.z < 0.0f) {
            const float ou = u;
            u = (1.0f - std::fabs(v)) * (ou >= 0.0f ? 1.0f : -1.0f);
            v = (1.0f - std::fabs(ou)) * (v >= 0.0f ? 1.0f : -1.0f);
        }
        const auto q = [](float x) { return static_cast<std::uint32_t>(std::lround((std::clamp(x, -1.0f, 1.0f) * 0.5f + 0.5f) * 2047.0f)); };
        const float code = -(1.0f + static_cast<float>(q(u) + 2048u * q(v)));
        for (int k = 0; k < 3; ++k) std::memcpy(tri + k * kVertex + 0x1c, &code, sizeof code);
    }
}

int on_batch_draw(BbHookCtx* ctx, void*) {
    t_pass = 0;
    Mesh* m = t_capture;
    if (!m) return 0;
    const std::uint32_t tris = static_cast<std::uint32_t>(ctx->arg[2]);
    if (!tris || !ctx->arg[1]) return 0;
    Batch b;
    b.tris = tris;
    std::memcpy(b.xf, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(ctx->arg[3])), sizeof b.xf);
    const auto* v = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(ctx->arg[1]));
    b.verts.assign(v, v + std::size_t(tris) * 3 * kVertex);
    m->batches.push_back(std::move(b));
    return 0;
}

// sub_2590be0's entry, in a mesh drawn by us: the edges' draw (sub_13673c0's
// second, costing the GPU more than the faces) left out; in a whole-mesh play, its steps with the
// mesh's buffer bound where the ring's copy would be (0x2590c6f-0x2590cfa).
int on_draw_up(BbHookCtx* ctx, void*) {
    if (!t_mesh_ours) return 0;
    if (t_pass++ & 1) return 1;
    const std::uint64_t buffer = t_bind;
    if (!buffer) return 0;
    const std::uint32_t topology = static_cast<std::uint32_t>(ctx->arg[1]), count = static_cast<std::uint32_t>(ctx->arg[2]);
    const std::uint32_t stride = static_cast<std::uint32_t>(ctx->arg[4]);
    const std::uint64_t dc = at<std::uint64_t>(at<std::uint64_t>(ctx->arg[0] + 8));
    const std::uint64_t gctx = at<std::uint64_t>(dc + 8);
    put<std::uint32_t>(dc + 0x268, topology);
    put<std::uint32_t>(gctx + 0xb738, topology);
    put<std::uint64_t>(dc + 0x280, buffer);
    put<std::uint32_t>(dc + 0x380, stride);
    put<std::uint32_t>(dc + 0x400, 0);
    put<std::uint8_t>(gctx + 0x25f0, at<std::uint8_t>(gctx + 0x25f0) | 1);
    put<std::uint64_t>(gctx + 0xb750, buffer);
    put<std::uint32_t>(gctx + 0xb850, stride);
    put<std::uint32_t>(gctx + 0xb8d0, 0);
    std::memset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dc + 0x1670)), 0, 0xb8);
    put<std::uint64_t>(dc + 0x1680, 0xe);
    put<std::uint64_t>(dc + 0x1688, 8);
    g_immediate_draw(gctx, gctx + 0xb730, count, 0);
    g_draws.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

// sub_13dc900's entry (the GX worker playing the list).
int on_mesh_draw(BbHookCtx* ctx, void*) {
    if (!g_cache_on) return 0;
    const std::uint64_t renderer = ctx->arg[0], mesh = ctx->arg[1];
    if (!renderer || !mesh) return 0;
    const auto* key = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(mesh));
    const std::uint32_t now = g_seconds.load(std::memory_order_relaxed);
    Mesh* m;
    {
        std::lock_guard<std::mutex> lock(g_mesh_mu);
        Mesh& slot = g_meshes[mesh];
        slot.used = now;
        ++slot.busy;
        if (slot.built && slot.gen == g_mesh_gen.load(std::memory_order_relaxed) && std::memcmp(slot.key, key, kMeshKey) == 0) {
            m = &slot;
        } else {
            std::memcpy(slot.key, key, kMeshKey);
            slot.batches.clear();
            if (slot.buffer) g_dead_buffers.emplace_back(slot.buffer, now);
            slot.buffer = 0;
            slot.built = false;
            m = nullptr;
            t_capture = &slot;
        }
    }
    if (!m) {
        // sub_13dc900's body, its batches copied as they go (sub_13644a0's
        // begin is the batch draw's first step too; sub_13644e0 is `mov al, 1`).
        Mesh& slot = *t_capture;
        alignas(16) std::uint8_t drawer[0x100], shapes[0x40];
        t_mesh_ours = true;
        g_drawer_make(drawer, at<std::uint64_t>(renderer + 0x30), renderer);
        g_shapes_make(shapes, mesh);
        g_shapes_draw(shapes, drawer);
        g_shapes_free(shapes);
        g_drawer_free(drawer);
        t_mesh_ours = false;
        t_capture = nullptr;
        // One buffer when every batch has the same drawer constants (one drawer: always).
        std::uint32_t tris = 0;
        bool same = true;
        for (const Batch& b : slot.batches) {
            tris += b.tris;
            same = same && std::memcmp(b.xf, slot.batches.front().xf, sizeof b.xf) == 0;
        }
        slot.gen = g_mesh_gen.load(std::memory_order_relaxed);
        for (Batch& b : slot.batches) shade_batch(b);
        if (g_gbuffer_style.load(std::memory_order_relaxed))
            for (Batch& b : slot.batches) normal_batch(b);
        if (g_buffers_on && same && tris) {
            if (const std::uint64_t buffer = make_buffer(slot.batches, tris)) {
                std::lock_guard<std::mutex> lock(g_mesh_mu);
                slot.buffer = buffer;
                slot.tris = tris;
                std::memcpy(slot.xf, slot.batches.front().xf, sizeof slot.xf);
                std::vector<Batch>().swap(slot.batches);
            }
        }
        {
            std::lock_guard<std::mutex> lock(g_mesh_mu);
            slot.built = true;
            --slot.busy;
        }
        g_built.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
    std::uint64_t tris = 0;
    t_mesh_ours = true;
    if (m->buffer) {
        // sub_13673c0 sets the camera's constants and draws twice; both draws bind the buffer.
        static const std::uint8_t kNoVertices[16] = {};
        t_bind = m->buffer;
        g_batch_draw(renderer, kNoVertices, m->tris, m->xf);
        t_bind = 0;
        tris = m->tris;
    } else {
        for (const Batch& b : m->batches) {
            g_batch_draw(renderer, b.verts.data(), b.tris, b.xf);
            tris += b.tris;
        }
    }
    t_mesh_ours = false;
    {
        std::lock_guard<std::mutex> lock(g_mesh_mu);
        --m->busy;
    }
    g_played.fetch_add(1, std::memory_order_relaxed);
    g_tris_played.fetch_add(tris, std::memory_order_relaxed);
    return 1;
}

// Once a second (the main thread): meshes not played for 10 s go (a part
// unloaded), buffers let go of 2 s before are released; with LOHIT_DIAG, the counts.
void sweep_meshes() {
    const std::uint32_t now = seconds_now();
    if (now == g_seconds.load(std::memory_order_relaxed)) return;
    g_seconds.store(now, std::memory_order_relaxed);
    std::size_t n = 0, buffers = 0, bytes = 0, gone = 0;
    std::vector<std::uint64_t> release;
    {
        std::lock_guard<std::mutex> lock(g_mesh_mu);
        for (auto it = g_meshes.begin(); it != g_meshes.end();) {
            if (!it->second.busy && now - it->second.used > 10) {
                if (it->second.buffer) g_dead_buffers.emplace_back(it->second.buffer, now);
                it = g_meshes.erase(it);
                ++gone;
                continue;
            }
            ++n;
            if (it->second.buffer) {
                ++buffers;
                bytes += std::size_t(it->second.tris) * 3 * kVertex;
            }
            for (const Batch& b : it->second.batches) bytes += b.verts.size();
            ++it;
        }
        for (auto it = g_dead_buffers.begin(); it != g_dead_buffers.end();) {
            if (now - it->second < 2) {
                ++it;
                continue;
            }
            release.push_back(it->first);
            it = g_dead_buffers.erase(it);
        }
    }
    for (std::uint64_t b : release) release_buffer(b);
    if (g_diag > 0)
        api->log("lohit diag: %zu meshes kept (%zu in buffers, %zu KiB), %zu dropped, %zu buffers released; in the last second "
                 "%llu built, %llu played (%llu triangles), %llu draws from buffers",
                 n, buffers, bytes >> 10, gone, release.size(), static_cast<unsigned long long>(g_built.exchange(0)),
                 static_cast<unsigned long long>(g_played.exchange(0)),
                 static_cast<unsigned long long>(g_tris_played.exchange(0)),
                 static_cast<unsigned long long>(g_draws.exchange(0)));
}

// ---- The connected parts ----

constexpr std::uint64_t kHitDraw = 0x1c8a150;  // WorldHitMan's debug draw (WorldHitMan, ...), once a frame
constexpr std::uint32_t kPartSize = 0x138, kFlags = 0x130, kDrawLo = 0x10, kDrawHi = 0x20;
constexpr std::uint32_t kMaskOffset = 0x94;  // the mask that says which parts connect (+0x74: none at Central Yharnam)

std::atomic<bool> g_whole_map{true};  // every part flagged, not only the neighbours
std::atomic<bool> g_hi{false};        // the hi-hit's flag, not the lo-hit's
std::uint8_t g_flag_bit = kDrawLo;    // the flag g_flagged were given
std::uint64_t g_flagged[4096];
std::size_t g_n_flagged = 0;
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
    *flags = on ? (*flags | g_flag_bit) : (*flags & ~g_flag_bit);
}

// sub_1c8a150's entry (the main thread): last frame's neighbours unflagged,
// this frame's flagged (with `whole_map`, every part).
int on_hit_draw(BbHookCtx* ctx, void*) {
    const std::uint64_t whm = ctx->arg[0];
    const std::uint64_t standing = whm ? at<std::uint64_t>(whm + 0x2de8) : 0;
    sweep_meshes();
    for (std::size_t i = 0; i < g_n_flagged; ++i) set_draw(g_flagged[i], false);
    g_n_flagged = 0;
    g_flag_bit = g_hi.load(std::memory_order_relaxed) ? kDrawHi : kDrawLo;
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
        if (at<std::uint8_t>(part + kFlags) & g_flag_bit) return;  // flagged by the game (its other slot)
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

// sub_1c8a150: mov sil, [show hi-hit debug] -> mov sil, 1; and the lo-hit's
// switch back to the game's while the hi-hit is shown.
const Site kHiSwitch = {"the hi-hit's switch", 0x1c8a176, 7, false, {0x40, 0x8a, 0x35, 0x74, 0x47, 0xce, 0x03},
                        {0x40, 0xb6, 0x01, 0x0f, 0x1f, 0x40, 0x00}};

// sub_1c86b50 (the residency pick) asks each part for its hit (+0x11a): 4, the
// lo-hit and the hi-hit, for the part stood on and those its +0x74 mask names;
// 2, the lo-hit alone, for those connected by +0x94. With the hi-hit shown the
// connected ones ask for 4 as well, so theirs is loaded to be drawn. Whether
// the area is loaded (its return) still asks only for their lo-hit.
const Site kHiResident = {"the connected parts' hit", 0x1c86d46, 1, false, {0x02}, {0x04}};

void show_hi(bool hi) {
    if (hi == g_hi.load(std::memory_order_relaxed)) return;
    const Site& lo = kLoHit[0];
    if (api->patch(kHiResident.at, hi ? kHiResident.game : kHiResident.ours, hi ? kHiResident.ours : kHiResident.game, 1) != 0)
        api->log("lohit: 0x1c86d46 is not the residency pick's 2; the connected parts' hi-hit is not loaded");
    if (api->patch(lo.at, hi ? lo.ours : lo.game, hi ? lo.game : lo.ours, lo.n) != 0 ||
        api->patch(kHiSwitch.at, hi ? kHiSwitch.game : kHiSwitch.ours, hi ? kHiSwitch.ours : kHiSwitch.game, kHiSwitch.n) != 0)
        api->log("lohit: the switches at 0x1c8a170/0x1c8a176 are not as expected; the %s is not shown", hi ? "hi-hit" : "lo-hit");
    g_hi.store(hi, std::memory_order_relaxed);
}

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
constexpr std::uint64_t kHiColours = 0x4d00820;  // the hi-hit's, read by sub_1c06380 alone
constexpr float kHiGameColour[8] = {0.045f, 0.0f, 0.3f, 0.5f, 0.045f, 0.15f, 0.0f, 0.5f};
constexpr float kAlpha = 1.0f;  // opaque (the game's 0.5): nothing seen through it, an occluder
// The colours for the scene (the early play): the launcher's were measured
// at Central Yharnam's lamp and corrected once (kEarlyColour); any other
// comes out near 0.72 x its linear value (a fit to those six channels: the
// scene's tone-mapping, between the play and the screen).
constexpr float kEarlyColour[8] = {0.0258f, 0.0033f, 0.0778f, kAlpha, 0.0126f, 0.0287f, 0.1657f, kAlpha};
constexpr int kDefaultRgb[6] = {46, 20, 108, 32, 56, 128};
int g_rgb[6] = {46, 20, 108, 32, 56, 128};  // walls r g b, floors r g b (0-255, on the screen)
float g_colours[8];                         // what is written: walls, floors
bool g_colours_written = false;

float srgb_linear(int v) {
    const float c = static_cast<float>(v) / 255.0f;
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

void make_colours() {
    for (int k = 0; k < 6; ++k) {
        const int i = k < 3 ? k : k + 1;
        if (!g_early) g_colours[i] = static_cast<float>(g_rgb[k]) / 255.0f;
        else if (g_gbuffer_style.load(std::memory_order_relaxed)) g_colours[i] = srgb_linear(g_rgb[k]);  // an albedo, lit by the game
        else if (g_rgb[k] == kDefaultRgb[k]) g_colours[i] = kEarlyColour[i];
        else g_colours[i] = 0.72f * srgb_linear(g_rgb[k]);
    }
    g_colours[3] = g_colours[7] = kAlpha;
}

// The lo-hit's and the hi-hit's colours (read on the main thread, where
// sub_1c06490 and sub_1c06380 record them): the first time patched over the
// game's, after that written; the meshes built again.
void write_colours() {
    make_colours();
    g_shade_colours = g_colours;
    if (!g_colours_written) {
        const int r = api->patch(kColours, kGameColour, g_colours, sizeof kGameColour);
        if (r != 0) api->log("lohit: 0x4d00840 is not the lo-hit's colours (%d); kept as they are", r);
        if (api->patch(kHiColours, kHiGameColour, g_colours, sizeof kHiGameColour) != 0)
            api->log("lohit: 0x4d00820 is not the hi-hit's colours; kept as they are");
        g_colours_written = r == 0;
        return;
    }
    if (api->write(kColours, g_colours, sizeof g_colours) != 0 || api->write(kHiColours, g_colours, sizeof g_colours) != 0)
        api->log("lohit: the colours could not be written");
    g_mesh_gen.fetch_add(1, std::memory_order_relaxed);
}

int rgb_index(const char* key) {
    static const char* const kKeys[6] = {"wall_r", "wall_g", "wall_b", "floor_r", "floor_g", "floor_b"};
    for (int k = 0; k < 6; ++k)
        if (std::strcmp(key, kKeys[k]) == 0) return k;
    return -1;
}

bool read_rgb(int k, const char* v) {
    if (!v || !*v) return false;
    const int n = static_cast<int>(std::strtod(v, nullptr));
    const int c = n < 0 ? 0 : n > 255 ? 255 : n;
    if (c == g_rgb[k]) return false;
    g_rgb[k] = c;
    return true;
}

bool read_sun(const char* v) {
    if (!v || !*v) return false;
    const int n = static_cast<int>(std::strtod(v, nullptr));
    const int s = n < 0 ? 0 : n > 100 ? 100 : n;
    return g_sun.exchange(s, std::memory_order_relaxed) != s;
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
    if (g_early) api->on_scene_category(g_lit.load() ? kLitPass : g_pass, &play_early, nullptr);
}

// Lit by the game or not: the coloured play's pass and style, and the meshes
// built again (an albedo and the normals in place of the screen colours).
void set_lit(bool on) {
    on = on && g_can_light;
    if (g_lit.exchange(on) == on) return;
    g_gbuffer_style.store(on ? BB_DRAW_GBUFFER : 0, std::memory_order_relaxed);
    api->on_scene_category(on ? kLitPass : g_pass, &play_early, nullptr);
    write_colours();
}

void set_ambient(const char* v) {
    if (!g_can_light || !v || !*v) return;
    const double n = std::strtod(v, nullptr);
    api->gbuffer_ambient(static_cast<float>((n < 0 ? 0 : n > 100 ? 100 : n) / 1000.0));
}

bool truthy(const char* v) {
    return v && (std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0 || std::strcmp(v, "on") == 0);
}

void on_option(const char* key, const char* value, void*) {
    if (std::strcmp(key, "pass") == 0) set_pass(value);
    else if (std::strcmp(key, "whole_map") == 0) g_whole_map.store(truthy(value), std::memory_order_relaxed);
    else if (std::strcmp(key, "show_hi_hit") == 0) show_hi(truthy(value));
    else if (const int k = rgb_index(key); k >= 0) {
        if (read_rgb(k, value)) write_colours();
    }
    else if (std::strcmp(key, "ambient") == 0) set_ambient(value);
    else if (std::strcmp(key, "sun") == 0) {
        if (read_sun(value)) g_mesh_gen.fetch_add(1, std::memory_order_relaxed);
    }
    else if (std::strcmp(key, "occluder_pass") == 0 && g_can_occlude) {
        const int p = parse_pass(value);
        if (p >= 0) g_occluder_pass = p;
        api->on_scene_category(g_occluder_pass, &play_depth, nullptr);
    }
}

// An option's value, or its default when the config does not say.
bool option_on(const char* key, bool fallback) {
    const char* v = api->config(key);
    return v && *v ? truthy(v) : fallback;
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
    g_whole_map.store(option_on("lohit.whole_map", true), std::memory_order_relaxed);
    {
        static const char* const kKeys[6] = {"lohit.wall_r", "lohit.wall_g", "lohit.wall_b",
                                             "lohit.floor_r", "lohit.floor_g", "lohit.floor_b"};
        for (int k = 0; k < 6; ++k) read_rgb(k, host->config(kKeys[k]));
        read_sun(host->config("lohit.sun"));
    }
    // push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp, 0x48
    static const std::uint8_t hit_draw_entry[17] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                                    0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x48};
    if (const char* c = std::getenv("LOHIT_CACHE")) g_cache_on = c[0] != '0';
    if (const char* c = std::getenv("LOHIT_BUFFERS")) g_buffers_on = c[0] != '0';
    g_seconds.store(seconds_now(), std::memory_order_relaxed);
    g_immediate_draw = reinterpret_cast<DrawFn>(static_cast<std::uintptr_t>(host->guest_addr(kImmediateDraw)));
    g_create_buffer = reinterpret_cast<CreateFn>(static_cast<std::uintptr_t>(host->guest_addr(kCreateBuffer)));
    g_batch_draw = reinterpret_cast<BatchFn>(static_cast<std::uintptr_t>(host->guest_addr(kBatchDraw)));
    g_drawer_make = reinterpret_cast<MakeDrawerFn>(static_cast<std::uintptr_t>(host->guest_addr(kDrawerMake)));
    g_drawer_free = reinterpret_cast<ObjFn>(static_cast<std::uintptr_t>(host->guest_addr(kDrawerFree)));
    g_shapes_make = reinterpret_cast<ShapesFn>(static_cast<std::uintptr_t>(host->guest_addr(kShapesMake)));
    g_shapes_draw = reinterpret_cast<ShapesDrawFn>(static_cast<std::uintptr_t>(host->guest_addr(kShapesDraw)));
    g_shapes_free = reinterpret_cast<ObjFn>(static_cast<std::uintptr_t>(host->guest_addr(kShapesFree)));
    // push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp, 0xc8
    static const std::uint8_t batch_entry[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                                 0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xc8, 0x00, 0x00, 0x00};
    // push rbp; mov rbp, rsp; push r15; push r14; push r12; push rbx; sub rsp, 0x110
    static const std::uint8_t mesh_entry[18] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                                0x54, 0x53, 0x48, 0x81, 0xec, 0x10, 0x01, 0x00, 0x00};
    if (g_cache_on && (!g_batch_draw || !g_drawer_make || !g_drawer_free || !g_shapes_make || !g_shapes_draw ||
                       !g_shapes_free ||
                       host->hook2(kBatchDraw, batch_entry, sizeof batch_entry, &on_batch_draw, nullptr) != 0 ||
                       host->hook2(kMeshDraw, mesh_entry, sizeof mesh_entry, &on_mesh_draw, nullptr) != 0)) {
        host->log("lohit: 0x13dc900 / 0x13673c0 are not the mesh draws; each frame builds the lo-hit again");
        g_cache_on = false;
    }
    // push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp, 0x38
    static const std::uint8_t draw_up_entry[17] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                                   0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x38};
    if (g_cache_on && g_buffers_on &&
        (!g_immediate_draw || !g_create_buffer ||
         host->hook2(kDrawUp, draw_up_entry, sizeof draw_up_entry, &on_draw_up, nullptr) != 0)) {
        host->log("lohit: 0x2590be0 is not the batches' draw; the meshes are played a batch at a time");
        g_buffers_on = false;
    }
    if (host->hook2(kHitDraw, hit_draw_entry, sizeof hit_draw_entry, &on_hit_draw, nullptr) != 0)
        host->log("lohit: 0x1c8a150 is not WorldHitMan's debug draw; the connected parts are not drawn");
    write_colours();
    if (option_on("lohit.show_hi_hit", false)) show_hi(true);
    if (g_early) {
        if (host->on_frame) host->on_frame(&frame_tick, nullptr);
        const char* c = host->config("lohit.pass");
        set_pass(c && *c ? c : "3e/25");
        g_can_occlude = host->version >= 17 && host->size >= offsetof(BbHostApi, draw_style) + sizeof(void*);
        if (const char* o = std::getenv("LOHIT_OCCLUSION")) g_occlusion.store(o[0] != '0', std::memory_order_relaxed);
        if (g_can_occlude) {
            const int op = parse_pass(host->config("lohit.occluder_pass"));
            if (op >= 0) g_occluder_pass = op;
            host->on_scene_category(g_occluder_pass, &play_depth, nullptr);
        }
        g_can_light = g_can_occlude && g_occlusion.load() && host->version >= 19 &&
                      host->size >= offsetof(BbHostApi, gbuffer_ambient) + sizeof(void*);
        if (g_can_light) {
            const char* a = host->config("lohit.ambient");
            set_ambient(a && *a ? a : "30");
            set_lit(true);
        }
    }
    host->log("lohit: the lo-hit shown in place of the map (%d of 2 sites), %s (ahead of the scene's %02x/%02x), %s%s; "
              "meshes %s, colours %d,%d,%d / %d,%d,%d",
              shown, g_early ? "under the effects" : "over everything", g_pass >> 6, g_pass & 0x3f,
              g_whole_map.load() ? "the whole map" : "the part stood on and its neighbours", g_hi.load() ? " (the hi-hit)" : "",
              !g_cache_on ? "built every frame" : g_buffers_on ? "built once into buffers" : "built once", g_rgb[0], g_rgb[1],
              g_rgb[2], g_rgb[3], g_rgb[4], g_rgb[5]);
    host->log("lohit: occlusion %s", !g_can_occlude ? "not available (needs plugin API 17 and the early play)"
                                     : g_occlusion.load() ? "on: its depth ahead of the scene's objects" : "off");
    host->log("lohit: lit by the game %s", !g_can_light ? "not available (needs plugin API 19 and occlusion)"
                                           : g_lit.load() ? "on: drawn into the G-buffer ahead of 3e/0c" : "off");
    return 0;
}
