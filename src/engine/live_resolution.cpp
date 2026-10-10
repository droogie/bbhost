#include "guest_abi.h"
#include "engine/live_resolution.h"

#include "core/elf.h"
#include "core/portable.h"
#include "core/memory.h"
#include "core/thunk.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "engine/gx_resources.h"
#include "engine/option_menu.h"
#include "hle/modules.h"
#include "host/gpu.h"
#include "host/options.h"
#include "host/settings.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

// Addresses are Binary Ninja's (0x400000 base).
//
// Dark Souls III PC changes resolution through its GX device: ResizeBuffers,
// then every GXSystemMessageListener's resize slot, of which GXSceneContext
// releases and rebuilds its size-dependent targets and the Sprj listener
// sets the screen-size globals. Bloodborne keeps the listeners - the device
// (0x59406c8) holds them at +0x5f8, slot +0x18 is resize (w, h), and
// sub_2599c90 broadcasts it, though nothing in the shipped game calls it -
// and five answer:
//
//   0x219e850  the Sprj listener: res words and UI scale (sub_2417700)
//   0x269c890  the graphics manager's (gm+8 -> sub_269c200): its own
//              targets, the scene context's release and rebuild, then YEBIS
//              shut down and started again
//   0x26c6680  a scene context
//   0x2675680, 0x27030b0  half-size targets of their own
//
// What the PS4 build lacks is the PC side, which is here:
//   - a quiet point (present_frame_end_hook);
//   - display buffers that fit every size (gx_init_call_hook): the console
//     allocates them once at the render size and registers them with
//     VideoOut, so they are made at the largest size and the window shows
//     the top-left w x h (hle_video_set_picture);
//   - YEBIS's restart, which frees its textures into a heap it has already
//     given back (dead-region frees, below);
//   - Scaleform's viewports, which each movie took from the res words when
//     it was built (swf_viewports).

namespace {

constexpr std::uint64_t kGxDeviceSlot = 0x59406c8;
constexpr std::uint64_t kGraphicsManagerSlot = 0x5940dd8;
constexpr std::uint64_t kResWidth = 0x55289f8, kResHeight = 0x55289fc;
constexpr std::uint64_t kWindowSize = 0x59404d8;  // SprjInitStuff's window width, height
constexpr std::uint64_t kResizeBroadcast = 0x2599c90;      // GX device: every listener's resize (w, h)
constexpr std::uint64_t kFrameEnd = 0x2598fd0;             // GX device: frame end
constexpr std::uint64_t kPresentFrameEndCall = 0x2197853;  // the present step's `call kFrameEnd`
constexpr std::uint64_t kGxInit = 0x2594580;               // GX init: device, display buffers, output entry
constexpr std::uint64_t kGxInitCall = 0x2196bee;           // sub_21969f0's `call kGxInit`
constexpr std::uint64_t kGxInitWidthRead = 0x2196a6b, kGxInitHeightRead = 0x2196a7a;
constexpr std::uint64_t kYebisShutdown = 0x25d2830;
constexpr std::uint64_t kListenerYebisShutdownCall = 0x269c82a;  // in the graphics manager's listener
constexpr std::uint64_t kGxFree = 0x15e6b00, kGxFreeRecord = 0x15e69e0;
constexpr std::uint64_t kSwfPlayerCtor = 0x2358360, kSwfPlayerDtor = 0x2358670, kSwfPlayerVtable = 0x5756a00;

// The largest entries of the Resolution setting, wide and tall: 5120x2160
// (21:9) and 3840x2160 (16:9) - the display buffers fit both.
constexpr std::uint32_t kMaxW = 5120, kMaxH = 2160;

std::uint64_t g_slide = 0;
bool g_installed = false;
const std::uint64_t* g_device_slot = nullptr;
const std::uint64_t* g_manager_slot = nullptr;
const std::uint32_t* g_res_words = nullptr;
std::uint32_t g_display[2] = {0, 0};  // the display buffers' size

std::atomic<std::uint64_t> g_request{0};  // w << 32 | h, 0 for none
std::atomic<bool> g_request_confirm{false};  // ask to keep it once it is shown
// A change asked to be kept comes from a menu: the pick list it was chosen
// in closes over the next few frames, and a resize in the middle of that
// left it open (taking the keys meant for the question). It waits this long.
std::atomic<std::int64_t> g_request_not_before{0};  // steady_clock ns
constexpr std::int64_t kMenuSettleNs = 1500'000'000;

// ---- Keep or go back -------------------------------------------------------------
//
// Dark Souls III asks with its generic yes/no dialog (message 920000) and a
// ten-second countdown. Bloodborne has the same dialog: the request object
// at MenuMan (0x5962878) +0xd08 - message id, the two buttons' ids in group
// 204 (3 YES, 4 NO), +0x10 byte 0 for group-78 text - opened by sub_1abebf0
// with mode 2 (two buttons), the way the quit-game question is
// (sub_1b58370). The dialog sits on the open menu's stack (the PC Graphics
// screen) or, with none open, as its own; the answer arrives through
// MenuMan's handler (message 0x3eb) at +0x16c, -1 until then. It is opened
// and read from the present step, on the main thread, after the change.
constexpr std::uint64_t kMenuManSlot = 0x5962878, kOpenGenDialog = 0x1abebf0;
constexpr std::uint32_t kKeepMessage = 920000, kYesButton = 3, kNoButton = 4;
enum class Confirm { None, ToOpen, Open, OpenInMenu };
Confirm g_confirm = Confirm::None;  // main thread
std::chrono::steady_clock::time_point g_confirm_until;
bool g_confirm_standalone = false;  // opened with no menu up (main thread)
bool g_confirm_closing = false;     // the ten seconds ran out over PC Graphics
std::atomic<int> g_answer{0};  // 1 keep, 2 go back; taken by the host
std::atomic<std::uint64_t> g_current{0};  // w << 32 | h once the output entry is set
std::atomic<std::uint64_t> g_before_asked{0};  // the size before the last change that asks to be kept
std::atomic<std::uint64_t> g_applied{0};
// BBHOST_RESIZE_TEST=<flip>:<w>x<h>[,...]: changes at those flips.
std::vector<std::pair<std::uint64_t, std::uint64_t>> g_tests;  // flip, size; main thread only
std::size_t g_next_test = 0;

std::uint64_t guest(std::uint64_t bn) { return g_slide + (bn - kPreferredGuestSlide); }
void* guest_fn(std::uint64_t bn) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest(bn))); }
template <typename T>
T& at_(std::uint64_t a) { return *reinterpret_cast<T*>(static_cast<std::uintptr_t>(a)); }
std::uint64_t pack(std::uint32_t w, std::uint32_t h) { return static_cast<std::uint64_t>(w) << 32 | h; }

// ---- YEBIS ----------------------------------------------------------------------
//
// YEBIS's shutdown (sub_25d2830) gives back its private heap first - the
// region at +0x2648 of its glue object (yebis+0xaf0), +0x2714 bytes, carved
// from the GX heap at its start (sub_25d2dd0) - and only then releases its
// objects, whose textures live in that region: each texture's free
// (sub_2e4b230 -> the GX free wrappers sub_15e6b00 / sub_15e69e0 -> YEBIS's
// free callback, which asks which heap owns the address) lands in the outer
// heap with an address that is no block of it, and the heap panics
// ("Requested block not marked as allocated"). The path never ran on the
// console. The region went back whole, so while the shutdown runs a free
// inside it is dropped here.
std::atomic<std::uint64_t> g_dead_lo{0}, g_dead_hi{0};
std::atomic<int> g_dead_frees{0};
bool in_dead_region(std::uint64_t a) {
    const std::uint64_t hi = g_dead_hi.load(std::memory_order_relaxed);
    return hi && a >= g_dead_lo.load(std::memory_order_relaxed) && a < hi;
}
GUEST_ABI std::int64_t gx_free_hook(std::uint64_t, const std::uint64_t* saved) {
    if (!in_dead_region(saved[5])) return 0;  // rdi: the address
    g_dead_frees.fetch_add(1, std::memory_order_relaxed);
    return 1;
}
GUEST_ABI std::int64_t gx_free_record_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t rec = saved[5];  // rdi: {address, size, ..., +0x10 address}
    if (!rec || !in_dead_region(at_<std::uint64_t>(rec + 0x10))) return 0;
    at_<std::uint64_t>(rec) = 0;  // as the function leaves the record
    at_<std::uint64_t>(rec + 0x10) = 0;
    at_<std::uint32_t>(rec + 8) = 0;
    g_dead_frees.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

// The graphics manager's listener (sub_269c200) shuts YEBIS down at
// 0x269c82a (`call sub_25d2830`, rdi = the glue object) after rebuilding the
// scene context, and starts it again itself when the shutdown cleared its
// started word (+0x3160). The call goes through here, which notes the region
// for the frees above around it.
GUEST_ABI std::int64_t yebis_shutdown_call_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t glue = saved[5];
    const std::uint64_t lo = at_<std::uint64_t>(glue + 0x2648);
    const std::int32_t size = at_<std::int32_t>(glue + 0x2714);
    g_dead_lo.store(lo);
    g_dead_hi.store(lo && size > 0 ? lo + static_cast<std::uint64_t>(size) : 0);
    hle_call_guest6(guest_fn(kYebisShutdown), static_cast<std::int64_t>(glue), 0, 0, 0, 0, 0);
    g_dead_hi.store(0);
    g_dead_lo.store(0);
    return 1;
}

// ---- Scaleform -------------------------------------------------------------------
//
// A movie keeps the viewport its player gave it: the SwfPlayer constructor
// (sub_2358360, vtable 0x5756a00) passes its movie (+0x18) the res words
// through the movie's SetViewport (vtable +0x68), a GFx viewport - buffer
// w, h; left, top; width, height; five zero words; scale 1.0, aspect 1.0.
// The constructor is hooked to keep the players, and a change sets each live
// one's viewport again. Players come and go with menus; the list is pruned
// of entries whose vtable is gone.
std::mutex g_swf_mu;
std::vector<std::uint64_t> g_swf_players;  // under g_swf_mu
GUEST_ABI std::int64_t swf_player_ctor_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t p = saved[5];
    std::lock_guard<std::mutex> lk(g_swf_mu);
    if (g_swf_players.size() >= 256) {
        const std::uint64_t vt = guest(kSwfPlayerVtable);
        g_swf_players.erase(std::remove_if(g_swf_players.begin(), g_swf_players.end(),
                                           [&](std::uint64_t q) { return at_<std::uint64_t>(q) != vt; }),
                            g_swf_players.end());
    }
    if (std::find(g_swf_players.begin(), g_swf_players.end(), p) == g_swf_players.end()) g_swf_players.push_back(p);
    return 0;
}

// The destructor (sub_2358670; vtable slot 1) takes a player off the list: a
// freed player's memory can be reused while still showing the vtable, and
// its "movie" then is some other object.
GUEST_ABI std::int64_t swf_player_dtor_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t p = saved[5];
    std::lock_guard<std::mutex> lk(g_swf_mu);
    g_swf_players.erase(std::remove(g_swf_players.begin(), g_swf_players.end(), p), g_swf_players.end());
    return 0;
}

struct GfxViewport {
    std::int32_t buf_w, buf_h, left, top, width, height, zero[5];
    float scale, aspect;
};
static_assert(offsetof(GfxViewport, scale) == 0x2c && sizeof(GfxViewport) == 0x34, "the GFx viewport");

int swf_viewports(std::uint32_t w, std::uint32_t h) {
    const std::uint64_t vt = guest(kSwfPlayerVtable);
    std::vector<std::uint64_t> players;
    {
        std::lock_guard<std::mutex> lk(g_swf_mu);
        players = g_swf_players;
    }
    const auto iw = static_cast<std::int32_t>(w), ih = static_cast<std::int32_t>(h);
    GfxViewport vp{iw, ih, 0, 0, iw, ih, {0, 0, 0, 0, 0}, 1.0f, 1.0f};
    int set = 0;
    for (std::uint64_t p : players) {
        // Constructed to the end (+0x50 set last) and not destroyed since.
        if (!p || at_<std::uint64_t>(p) != vt || !at_<std::uint8_t>(p + 0x50)) continue;
        const std::uint64_t movie = at_<std::uint64_t>(p + 0x18);
        if (!movie) continue;
        const std::uint64_t fn = at_<std::uint64_t>(at_<std::uint64_t>(movie) + 0x68);
        hle_call_guest6(reinterpret_cast<void*>(static_cast<std::uintptr_t>(fn)), static_cast<std::int64_t>(movie),
                        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(&vp)), 0, 0, 0, 0);
        ++set;
    }
    return set;
}

// ---- Scene contexts made on demand ---------------------------------------------
//
// GXSceneContext (constructor sub_26c1050, vtable 0x57b8450, its listener
// subobject at +8) builds its targets in Initialize (sub_26c39f0) from the
// size it is given. The graphics manager's own context is rebuilt by its
// listener; a context registered with the device (its +8 in the list) by its
// own resize slot, sub_26c6540. Others - sub_26f8c70 and sub_26f9330 make
// them at the output entry's size when the game asks - are not, and drawn at
// a larger size later they overrun what they allocated.
constexpr std::uint64_t kSceneCtxCtor = 0x26c1050, kSceneCtxDtor = 0x26c2e10, kSceneCtxVtable = 0x57b8450, kSceneCtxResize = 0x26c6540;
std::mutex g_ctx_mu;
std::vector<std::uint64_t> g_ctxs;  // under g_ctx_mu
bool g_ctx_untracked = false;       // the destructor is not hooked: leave them be
std::atomic<int> g_ctx_gone{0};     // taken off by the destructor since the last change
GUEST_ABI std::int64_t scene_ctx_ctor_hook(std::uint64_t, const std::uint64_t* saved) {
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    const std::uint64_t c = saved[5];
    if (std::find(g_ctxs.begin(), g_ctxs.end(), c) == g_ctxs.end()) g_ctxs.push_back(c);
    return 0;
}
// The destructor (sub_26c2e10, vtable slot 0; slot 1 frees after it) takes a
// context off the list. It stores its own vtable first and has no base to
// put back afterwards, so a destroyed context still shows it while its
// members show their bases' - and rebuilt, the first of those called past
// its base's slots jumped to 0 (the tone map's LUT at +0x21e0, slot +0x90).
GUEST_ABI std::int64_t scene_ctx_dtor_hook(std::uint64_t, const std::uint64_t* saved) {
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    const auto it = std::remove(g_ctxs.begin(), g_ctxs.end(), saved[5]);
    if (it != g_ctxs.end()) g_ctx_gone.fetch_add(1, std::memory_order_relaxed);
    g_ctxs.erase(it, g_ctxs.end());
    return 0;
}
// The unregistered contexts that are the render size or a half or quarter
// of it go to the same fraction of the new size, the way their resize slot
// (sub_26c6540) would do it: release (sub_26c5a60(ctx, 1)) when it holds
// targets, then Reinitialize (sub_26c58e0) with its own stored description
// (+0x18, 0x40 bytes) at the new size.
constexpr std::uint64_t kSceneCtxRelease = 0x26c5a60, kSceneCtxReinit = 0x26c58e0;
// Only contexts that were there before the broadcast: a registered parent
// remakes its children at the new size during it, and a child of half the
// new size can look like a quarter of the old one.
std::vector<std::uint64_t> live_scene_ctxs() {
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    g_ctxs.erase(std::remove_if(g_ctxs.begin(), g_ctxs.end(), [](std::uint64_t c) { return at_<std::uint64_t>(c) != guest(kSceneCtxVtable); }),
                 g_ctxs.end());
    return g_ctxs;
}
int resize_scene_ctxs(std::uint64_t dev, std::uint64_t gm, const std::vector<std::uint64_t>& before, std::uint32_t ow, std::uint32_t oh,
                      std::uint32_t w, std::uint32_t h) {
    if (g_ctx_untracked) return 0;
    const std::uint64_t list = at_<std::uint64_t>(dev + 0x5f8);
    std::vector<std::uint64_t> ctxs;
    for (std::uint64_t c : live_scene_ctxs())
        if (std::find(before.begin(), before.end(), c) != before.end()) ctxs.push_back(c);
    int done = 0;
    for (std::uint64_t c : ctxs) {
        if (c == gm + 0x60 || c == gm + 0x2c60) continue;  // the manager's: its listener
        bool reg = false;
        for (std::uint64_t it = list ? at_<std::uint64_t>(list + 8) : 0, end = list ? at_<std::uint64_t>(list + 0x10) : 0; it < end; it += 8)
            reg |= at_<std::uint64_t>(it) == c + 8;
        if (reg) continue;  // its own resize slot runs in the broadcast
        const std::uint32_t cw = at_<std::uint32_t>(c + 0x18), ch = at_<std::uint32_t>(c + 0x1c);
        int shift = -1;
        for (int k = 0; k <= 2 && shift < 0; ++k)
            if (cw == (ow >> k) && ch == (oh >> k)) shift = k;
        if (shift < 0) continue;  // a fixed size of its own
        if (at_<std::uint8_t>(c + 0x58) && (at_<std::uint8_t>(c + 0x50) || at_<std::uint8_t>(c + 0x52) || at_<std::uint8_t>(c + 0x53)))
            hle_call_guest6(guest_fn(kSceneCtxRelease), static_cast<std::int64_t>(c), 1, 0, 0, 0, 0);
        at_<std::uint8_t>(c + 0x58) = 1;
        alignas(16) std::uint8_t desc[0x40];
        std::memcpy(desc, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(c + 0x18)), sizeof(desc));
        const std::uint32_t nw = w >> shift, nh = h >> shift;
        std::memcpy(desc, &nw, 4);
        std::memcpy(desc + 4, &nh, 4);
        hle_call_guest6(guest_fn(kSceneCtxReinit), static_cast<std::int64_t>(c), static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(desc)),
                        static_cast<std::int64_t>(gm), 1, 0, 0);
        ++done;
    }
    return done;
}

// ---- Another shape than 16:9 ------------------------------------------------------
//
// The renders follow the res words at any shape; three things assume 16:9.
//
// The cameras. Each kind's constructor stores a 16:9 aspect beside its field
// of view (+0x50 fov, +0x54 aspect, +0x58 near, +0x5c far; the main one at
// 0x183a35d, a dozen others), and the camera manager's per-frame blend
// (sub_18368b0, from both game-state updates) copies them from its two
// sub-cameras (+0x60, +0x68) into its own, which the scene and the floating
// plates project with. At 21:9 that is a 16:9 picture stretched sideways. So
// the blend's prologue gives its sub-cameras and itself the screen's aspect
// wherever they hold 16:9 - or the aspect last given. The field of view is
// vertical (LockCamParam's camFovY), so the height seen stays the game's:
// wider screens see more across (Hor+), and a 16:10 one (the Steam Deck's
// 1280x800) a little less - what the community resolution patches do by
// rewriting the main constructor's immediate (1.6 in their 16:10 one).
//
// The floating plates (co-op players' names, enemy gauges, the lock-on
// marker). Scaleform shows the 1920x1080 stage whole and centred (its
// show-all mode: the HUD stays 16:9 in the middle, between bars of world at
// the sides of a wider screen or above and below on a taller one), and the
// plates are placed in that stage from the camera's clip space: stage x =
// (ndc * 0.5 + 0.5) * 1920, y = (ndc * -0.5 + 0.5) * 1080 (the scale at
// 0x4cf9a00 and two copies, the size pinned stage reads). A plate at the
// screen's edge belongs at x -m or 1920+m, m = 960 * (rx - 1) for rx the
// aspect over 16:9 - or, taller, at y -n or 1080+n, n = 540 * (ry - 1) - so
// the scale becomes (0.5 * rx, -0.5 * ry), and the plates' "is it on screen"
// tests reach past the stage by as much on each side (their >= 0 branches
// skipped, as the community patches skip them).
constexpr float kAspect169 = 16.0f / 9.0f;
constexpr std::uint64_t kCameraBlend = 0x18368b0;
constexpr std::uint64_t kPlateScale[] = {0x4cf9a00, 0x4cf9a60, 0x4cf9ad0};  // (0.5, -0.5, 0.5, 0): players, enemies, lock-on
constexpr std::uint64_t kAllyPlateRight = 0x4d26ea4, kAllyPlateBottom = 0x4d26ea8;  // sub_1a42380's 1920.0f, 1080.0f
// The pinned stage reads that are bounds (`mov eax/ecx, 1920` or `1080`, the value at +1).
constexpr std::uint64_t kStageRightBounds[] = {0x1ffd491, 0x1a54df6, 0x1ab3372};
constexpr std::uint64_t kStageBottomBounds[] = {0x1ffd4d1, 0x1a54e1c, 0x1ab3398};
struct LowerBound {
    std::uint64_t at;
    std::uint8_t game[3], open[3];
    std::uint8_t n;
};
constexpr LowerBound kLeftBounds[] = {
    {0x1ffd473, {0x77, 0x30}, {0x66, 0x90}, 2},              // sub_1ffb2a0: 0 > x -> off screen
    {0x1a54dd9, {0x77, 0x2f}, {0x66, 0x90}, 2},              // sub_1a54d80 ("F20_open_eneny_HP")
    {0x1ab3370, {0x77, 0x5a}, {0x66, 0x90}, 2},              // sub_1ab3320 ("F20_LockCursor")
    {0x1a42520, {0x0f, 0x96, 0xc1}, {0xb1, 0x01, 0x90}, 3},  // sub_1a42380: setbe cl (0 <= x) -> mov cl, 1
};
constexpr LowerBound kTopBounds[] = {
    {0x1ffd4cf, {0x77, 0x14}, {0x66, 0x90}, 2},  // sub_1ffb2a0: 0 > y -> off screen
    {0x1a54e1a, {0x77, 0x14}, {0x66, 0x90}, 2},  // sub_1a54d80
    {0x1ab3396, {0x77, 0x34}, {0x66, 0x90}, 2},  // sub_1ab3320
    {0x1a4253c, {0x77, 0x20}, {0x66, 0x90}, 2},  // sub_1a42380
};
ElfImage* g_image = nullptr;
bool g_aspect_ok = false;               // every site above was as expected
std::atomic<float> g_aspect{kAspect169};  // the screen's, w / h
std::atomic<float> g_aspect_given{kAspect169};  // what the cameras were last given

// The cameras: a sub-camera or the manager holding 16:9, or what they were
// given before a change, get the screen's aspect. Before the blend copies.
GUEST_ABI std::int64_t camera_blend_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t mgr = saved[5];  // rdi
    const float want = g_aspect.load(std::memory_order_relaxed), was = g_aspect_given.load(std::memory_order_relaxed);
    if (!mgr) return 0;
    auto fix = [&](std::uint64_t cam) {
        if (!cam) return;
        float& a = at_<float>(cam + 0x54);
        if (a != want && (std::fabs(a - kAspect169) < 1e-4f || a == was)) a = want;
    };
    fix(at_<std::uint64_t>(mgr + 0x60));
    fix(at_<std::uint64_t>(mgr + 0x68));
    fix(mgr);
    return 0;
}

bool code_write(std::uint64_t bn, const void* bytes, std::size_t n) {
    const std::uint64_t at = guest(bn), lo = at & ~0xfffull, hi = (at + n + 0xfff) & ~0xfffull;
    if (!guest_protect_rwx(&g_image->mem, lo, hi - lo)) return false;
    std::memcpy(guest_ptr(g_image->mem, at), bytes, n);
    guest_protect_rx(&g_image->mem, lo, hi - lo);  // the segment's own: text and read-only data share it
    return true;
}

// Checked once, at load: the plate constants, the ally plates' bounds and
// the lower-bound branches as the game has them.
bool aspect_sites_ok() {
    for (std::uint64_t a : kPlateScale) {
        const auto* f = static_cast<const float*>(guest_ptr(g_image->mem, guest(a)));
        if (f[0] != 0.5f || f[1] != -0.5f) return false;
    }
    if (*static_cast<const float*>(guest_ptr(g_image->mem, guest(kAllyPlateRight))) != 1920.0f ||
        *static_cast<const float*>(guest_ptr(g_image->mem, guest(kAllyPlateBottom))) != 1080.0f)
        return false;
    for (const LowerBound& b : kLeftBounds)
        if (std::memcmp(guest_ptr(g_image->mem, guest(b.at)), b.game, b.n) != 0) return false;
    for (const LowerBound& b : kTopBounds)
        if (std::memcmp(guest_ptr(g_image->mem, guest(b.at)), b.game, b.n) != 0) return false;
    return true;
}

// A pinned read's value (graphics_patch.cpp, ui_stage_pin: mov eax/ecx, imm32).
bool write_pinned(std::uint64_t at, std::uint32_t value) {
    const auto* p = static_cast<const std::uint8_t*>(guest_ptr(g_image->mem, guest(at)));
    return (p[0] == 0xb8 || p[0] == 0xb9) && code_write(at + 1, &value, 4);
}

// The screen is w x h: the cameras' aspect, the plates' scale and bounds. At
// the GX init (after the load-time pins) and with each change, on the main
// thread between frames - the plates' code runs there too.
void apply_aspect(std::uint32_t w, std::uint32_t h) {
    if (!g_aspect_ok || !w || !h) return;
    const float aspect = static_cast<float>(w) / static_cast<float>(h);
    const float rx = std::max(1.0f, aspect / kAspect169), ry = std::max(1.0f, kAspect169 / aspect);
    const bool wide = rx > 1.0005f, tall = ry > 1.0005f;
    g_aspect_given.store(g_aspect.load());
    g_aspect.store(wide || tall ? aspect : kAspect169);
    const float scale[2] = {0.5f * (wide ? rx : 1.0f), -0.5f * (tall ? ry : 1.0f)};
    const auto right = static_cast<std::uint32_t>(1920 + (wide ? std::ceil(960.0f * (rx - 1.0f)) : 0.0f));
    const auto bottom = static_cast<std::uint32_t>(1080 + (tall ? std::ceil(540.0f * (ry - 1.0f)) : 0.0f));
    const float right_f = static_cast<float>(right), bottom_f = static_cast<float>(bottom);
    for (std::uint64_t a : kPlateScale) code_write(a, scale, sizeof(scale));
    code_write(kAllyPlateRight, &right_f, 4);
    code_write(kAllyPlateBottom, &bottom_f, 4);
    int bounds = 0;
    for (std::uint64_t a : kStageRightBounds) bounds += write_pinned(a, right);
    for (std::uint64_t a : kStageBottomBounds) bounds += write_pinned(a, bottom);
    for (const LowerBound& b : kLeftBounds) code_write(b.at, wide ? b.open : b.game, b.n);
    for (const LowerBound& b : kTopBounds) code_write(b.at, tall ? b.open : b.game, b.n);
    host_log("resolution: %ux%u is %.3f:1 - cameras %s, plates' scale (%.3f, %.3f), on screen x %d..%u, y %d..%u (%d of %zu bounds)", w, h,
             static_cast<double>(aspect), wide ? "widened" : tall ? "narrowed (the height kept)" : "16:9", static_cast<double>(scale[0]),
             static_cast<double>(scale[1]), -static_cast<int>(right - 1920), right, -static_cast<int>(bottom - 1080), bottom, bounds,
             sizeof(kStageRightBounds) / sizeof(kStageRightBounds[0]) + sizeof(kStageBottomBounds) / sizeof(kStageBottomBounds[0]));
}

// ---- The change ------------------------------------------------------------------

bool device_ready(std::uint64_t* dev, std::uint64_t* gm) {
    *dev = g_device_slot ? *g_device_slot : 0;
    *gm = g_manager_slot ? *g_manager_slot : 0;
    return *dev && *gm && at_<std::uint64_t>(*dev + 0x90);
}

// A scene context whose targets did not fit. Initialize makes a new object
// for its +0x1ba0 pass at +0x1bc0 (the release freed the old one) and marks
// the pass on (+0x6c = 3), but the pass takes the new object (its +0x18,
// the context's +0x1bb8) only once its two full-size targets are made
// (sub_12adba0) - so where they did not fit, the next frame calls into the
// freed one. The registry's count of creations that failed is the general
// sign (a size short by a little fails elsewhere first: a null texture); this
// one also catches a run with the registry off (grow_gfx_heap, below, has
// what each size needs).
std::uint64_t unfit_scene_ctx() {
    for (std::uint64_t c : live_scene_ctxs())
        if (at_<std::uint32_t>(c + 0x6c) == 3 && at_<std::uint64_t>(c + 0x1bb8) != at_<std::uint64_t>(c + 0x1bc0)) return c;
    return 0;
}

// What the render targets' heap has left (grow_gfx_heap, below): the GX
// context (the device's +0) keeps its memory allocator at +0xdc20, a
// CSGraphicsPrivateAllocator (vtable 0x57aec90) over GFX_GraphicsPrivate
// (+0x8) and GFX_GraphicsPrivateB (+0x10); a heap's slot +0x30 is its free
// bytes and +0x38 its largest free block. 0 when it is not that allocator.
constexpr std::uint64_t kGraphicsPrivateAllocatorVtable = 0x57aec90;
std::uint64_t target_heap_free_mib(std::uint64_t dev, std::uint64_t* largest_mib) {
    *largest_mib = 0;
    const std::uint64_t ctx = at_<std::uint64_t>(dev), a = ctx ? at_<std::uint64_t>(ctx + 0xdc20) : 0;
    if (!a || at_<std::uint64_t>(a) != guest(kGraphicsPrivateAllocatorVtable)) return 0;
    const std::uint64_t heap = at_<std::uint64_t>(a + 8);
    if (!heap) return 0;
    auto slot = [&](int off) {
        return static_cast<std::uint64_t>(
                   hle_call_guest6(reinterpret_cast<void*>(static_cast<std::uintptr_t>(at_<std::uint64_t>(at_<std::uint64_t>(heap) + off))),
                                   static_cast<std::int64_t>(heap), 0, 0, 0, 0, 0)) >>
               20;
    };
    *largest_mib = slot(0x38);
    return slot(0x30);
}

// ---- The display buffers' viewport and scissor ----------------------------------
//
// Binding a render target makes its whole rectangle the viewport and the
// scissor, worked out from its GX resource - a 2D texture's (+0x38 = 3)
// width and height at +0x44/+0x46 - in six places: the viewport
// (sub_25a1e90, called by the binds; sub_25a1a00 for the target at the
// context's +0x1ec0) and the scissor (sub_25a1c50, sub_25a2580, sub_25a2650
// and the state flush sub_259fcf0, which sets one when none is). The frame's
// passes set their own from the output entry; the debug draws keep these -
// CHR DBG's hit capsules, FRPG-NET's nodes and their labels - and the display
// buffers' resource (the output entry's +0x58) says the buffers' 5120x2160,
// so they were drawn 5120/w across and 2160/h down. The resource's size has
// to stay: a scene context compares it with its own (sub_26c39f0) and, told
// they matched, rendered straight into the display buffers, and the world
// came out fogged over. So each place's two loads jump to a stub that gives
// that one resource the picture's size and every other its own, as display
// buffers of w x h would.
struct RectSite {
    std::uint64_t at;
    const std::uint8_t* bytes;  // the game's, from the two loads to the store
    std::uint8_t n;
    std::uint8_t res, w, h;  // registers: the resource, width, height (rax 0, rcx 1, rdx 2)
};
constexpr std::uint8_t kViewportBytes[38] = {
    0x0f, 0xb7, 0x50, 0x44, 0x0f, 0xb7, 0x40, 0x46,  // movzx edx, word [rax+0x44]; movzx eax, word [rax+0x46]
    0xd3, 0xea, 0xc5, 0xf8, 0x57, 0xc0,              // shr edx, cl; vxorps xmm0, xmm0, xmm0
    0xc4, 0xe1, 0xfa, 0x2a, 0xc2, 0xd3, 0xe8,        // vcvtsi2ss xmm0, xmm0, rdx; shr eax, cl
    0xc4, 0xe1, 0xfa, 0x2a, 0xc8, 0xc5, 0xf8, 0x14,  // vcvtsi2ss xmm1, xmm0, rax; vunpcklps xmm0, xmm0, xmm1
    0xc1, 0xc5, 0xf0, 0x57, 0xc9, 0xc5, 0xf0, 0x16,  // vxorps xmm1, xmm1, xmm1; vmovlhps xmm0, xmm1, xmm0
    0xc0,
};
constexpr std::uint8_t kViewport2Bytes[38] = {
    0x0f, 0xb7, 0x42, 0x44, 0x0f, 0xb7, 0x52, 0x46,  // movzx eax, word [rdx+0x44]; movzx edx, word [rdx+0x46]
    0xd3, 0xe8, 0xc5, 0xf8, 0x57, 0xc0,              // shr eax, cl; vxorps xmm0, xmm0, xmm0
    0xc4, 0xe1, 0xfa, 0x2a, 0xc0, 0xd3, 0xea,        // vcvtsi2ss xmm0, xmm0, rax; shr edx, cl
    0xc4, 0xe1, 0xfa, 0x2a, 0xca, 0xc5, 0xf8, 0x14,  // vcvtsi2ss xmm1, xmm0, rdx; vunpcklps xmm0, xmm0, xmm1
    0xc1, 0xc5, 0xf0, 0x57, 0xc9, 0xc5, 0xf0, 0x16,  // vxorps xmm1, xmm1, xmm1; vmovlhps xmm0, xmm1, xmm0
    0xc0,
};
constexpr std::uint8_t kScissorBytes[25] = {
    0x0f, 0xb7, 0x48, 0x44, 0x0f, 0xb7, 0x40, 0x46,  // movzx ecx, word [rax+0x44]; movzx eax, word [rax+0x46]
    0x48, 0xc1, 0xe0, 0x20, 0x48, 0x09, 0xc8,        // shl rax, 32; or rax, rcx
    0xc4, 0xe1, 0xf9, 0x6e, 0xc0,                    // vmovq xmm0, rax
    0xc5, 0xf9, 0x73, 0xf8, 0x08,                    // vpslldq xmm0, xmm0, 8: (0, 0, w, h)
};
constexpr RectSite kRectSites[] = {
    {0x25a1ed5, kViewportBytes, sizeof(kViewportBytes), 0, 2, 0},
    {0x25a1a33, kViewport2Bytes, sizeof(kViewport2Bytes), 2, 0, 2},
    {0x25a1c82, kScissorBytes, sizeof(kScissorBytes), 0, 1, 0},
    {0x25a25d2, kScissorBytes, sizeof(kScissorBytes), 0, 1, 0},
    {0x25a269a, kScissorBytes, sizeof(kScissorBytes), 0, 1, 0},
    {0x259ffe9, kScissorBytes, sizeof(kScissorBytes), 0, 1, 0},
};
std::atomic<std::uint64_t>* g_rect_resource = nullptr;  // the stubs': the display buffers' resource
std::atomic<std::uint64_t>* g_rect_size = nullptr;      // the stubs': h << 32 | w

// All or none: a site not as expected leaves every one as the game has it.
bool install_display_rect_stubs() {
    for (const RectSite& s : kRectSites)
        if (std::memcmp(guest_ptr(g_image->mem, guest(s.at)), s.bytes, s.n) != 0) return false;
    auto* page = static_cast<std::uint8_t*>(host_page_alloc(0x1000, true));
    if (!page) return false;
    const auto addr = [](const std::uint8_t* p) { return reinterpret_cast<std::uint64_t>(p); };
    const auto rip32 = [&](std::uint8_t* at, const std::uint8_t* end, const std::uint8_t* target) {
        const auto d = static_cast<std::int32_t>(static_cast<std::int64_t>(addr(target)) - static_cast<std::int64_t>(addr(end)));
        std::memcpy(at, &d, 4);
    };
    std::uint8_t* const res_slot = page;
    std::uint8_t* const size_slot = page + 8;
    std::uint8_t* c = page + 0x40;
    static const std::uint8_t jmp_abs[6] = {0xff, 0x25, 0x00, 0x00, 0x00, 0x00};  // jmp [rip+0]
    struct Jump {
        std::uint64_t at, stub;
        std::size_t n;
    };
    std::vector<Jump> jumps;
    for (const RectSite& s : kRectSites) {
        std::uint8_t* const stub = c;
        c[0] = 0x48, c[1] = 0x3b, c[2] = static_cast<std::uint8_t>(0x05 | s.res << 3);  // cmp res, [rip -> the resource]
        rip32(c + 3, c + 7, res_slot);
        c[7] = 0x75, c[8] = 0x0e;                                         // jne: the game's loads
        c[9] = 0x8b, c[10] = static_cast<std::uint8_t>(0x05 | s.w << 3);  // mov w, [rip -> width]
        rip32(c + 11, c + 15, size_slot);
        c[15] = 0x8b, c[16] = static_cast<std::uint8_t>(0x05 | s.h << 3);  // mov h, [rip -> height]
        rip32(c + 17, c + 21, size_slot + 4);
        c[21] = 0xeb, c[22] = 0x08;           // jmp past the game's loads
        std::memcpy(c + 23, s.bytes, s.n);    // the loads, then the rest as the game has it
        c += 23 + s.n;
        std::memcpy(c, jmp_abs, 6);
        const std::uint64_t back = guest(s.at) + s.n;
        std::memcpy(c + 6, &back, 8);
        c += 14;
        c += (16 - addr(c) % 16) % 16;
        jumps.push_back({s.at, addr(stub), s.n});
    }
    g_rect_resource = reinterpret_cast<std::atomic<std::uint64_t>*>(res_slot);
    g_rect_size = reinterpret_cast<std::atomic<std::uint64_t>*>(size_slot);
    for (const Jump& j : jumps) {
        std::uint8_t site[64];
        std::memset(site, 0xcc, j.n);
        std::memcpy(site, jmp_abs, 6);
        std::memcpy(site + 6, &j.stub, 8);
        if (!code_write(j.at, site, j.n)) return false;  // the protection change failed: so would every other
    }
    return true;
}

// The picture's size for the stubs, and the resource it is for (once, at the GX init).
void set_display_rect(std::uint64_t out, std::uint32_t w, std::uint32_t h) {
    if (!g_rect_size) return;
    g_rect_size->store(static_cast<std::uint64_t>(h) << 32 | w);
    if (g_rect_resource->load() || !out) return;
    const std::uint64_t res = at_<std::uint64_t>(out + 0x58);
    if (res && at_<std::uint8_t>(res + 0x38) == 3) g_rect_resource->store(res);
}

// ---- The scene's own targets unless both sides match ----------------------------
//
// GXSceneContext::Initialize (sub_26c39f0) takes the output entry's
// DefRenderTarget and DefDepthStencil (or the targets its description names)
// for its own when they are the size it builds for - but its test passes when
// either side matches: the width (`je` at 0x26c3af3) or else the height
// (0x26c3b04) for the colour target, the same pair at 0x26c3b44 / 0x26c3b56
// for the depth. On the console the display buffers are the render size and
// both sides match. Here they are the largest size a live change can reach,
// so a render size that shares one side with them - 3840x2160's height in
// 5120x2160 buffers, 5120x1440's width, a 16:9 size on an ultrawide screen's
// height - had the graphics manager's scene render its picture into the
// display buffer itself. YEBIS then took the display buffer for its back
// buffer and made its whole chain at the display buffers' size (5120x2160,
// 2560x1080, 3136x882 ... for a 3840x2160 picture), and the picture came out
// black with a grey band from x = w * w / 5120. Here both sides must match:
// the first `je` becomes a `jne` to the not-matching path, so a width that
// differs goes there and the height decides only when the width matched.
// BBHOST_SCENE_BOTH_SIDES=0 leaves the game's test.
struct BothSidesSite {
    std::uint64_t at;
    std::uint8_t game[2], both[2];
};
constexpr BothSidesSite kBothSidesSites[] = {
    {0x26c3af3, {0x74, 0x36}, {0x75, 0x11}},  // colour: je 0x26c3b2b (keep) -> jne 0x26c3b06 (make its own)
    {0x26c3b44, {0x74, 0x33}, {0x75, 0x12}},  // depth: je 0x26c3b79 (keep) -> jne 0x26c3b58 (make its own)
};

bool g_both_sides = false;  // install_scene_both_sides() went in

// Where the graphics manager's scene renders its picture at w x h, by the test
// Initialize makes (the patched one or the game's), for the log.
const char* scene_target_path(std::uint32_t w, std::uint32_t h) {
    const bool wide = g_display[0] == w, tall = g_display[1] == h;
    return (g_both_sides ? wide && tall : wide || tall) ? "straight into the display buffers" : "into a target of its own";
}

// All or none, as the rectangle stubs.
bool install_scene_both_sides() {
    if (const char* e = std::getenv("BBHOST_SCENE_BOTH_SIDES"); e && e[0] == '0') {
        host_log("resolution: scene contexts keep the game's either-side size test (BBHOST_SCENE_BOTH_SIDES=0)");
        return false;
    }
    for (const BothSidesSite& s : kBothSidesSites)
        if (std::memcmp(guest_ptr(g_image->mem, guest(s.at)), s.game, sizeof(s.game)) != 0) {
            host_log("resolution: the scene context's size test is not as expected; it keeps the game's either-side test");
            return false;
        }
    for (const BothSidesSite& s : kBothSidesSites)
        if (!code_write(s.at, s.both, sizeof(s.both))) return false;
    host_log("resolution: a scene context renders into the display buffers only when both their sides match its size");
    return true;
}

// ---- DefDepthStencil at the render size -----------------------------------------
//
// The GX init makes DefDepthStencil (the output entry's +0x70) at the size
// the pinned reads give, beside the display buffers. Unlike them it is an
// ordinary depth texture nothing outside the game sees - and at that size no
// scene context took it: a context renders into the output entry's targets
// when either side matches its own (sub_26c39f0), and otherwise makes its
// own. So the scene's depth went to a target of its own, and the debug draws,
// which bind DefRenderTarget with DefDepthStencil (sub_25b4190) as the
// console's scene does, tested against nothing: the hit capsules and the
// navmesh showed through walls and over the character. DS3 makes its
// DefDepthStencil again with every change (GXDevice::Resize); here it is made
// at the render size after the GX init and before each change's broadcast,
// as a new object like the init's (0x90 bytes, vtable 0x57b6860, sub_259f130
// with format 0x14, one mip, the entry's samples at +0x2c, 0x100) put in the
// entry's place. The contexts and their target sets (sub_2666420 keeps the
// object and its views) hold the old one until the broadcast rebuilds them,
// and the frame's first bind of it is the next present step's, so the
// entry's reference to it goes a few frames later.
constexpr std::uint64_t kGxAllocatorSlot = 0x5940680;
constexpr std::uint64_t kDepthStencilVtable = 0x57b6860, kDepthStencilInit = 0x259f130;
constexpr std::uint64_t kDepthStencilDefaults = 0x4d20760;  // the 16 bytes the init puts at +0x20
constexpr std::uint64_t kDefDepthStencilName = 0x4dc2e02;   // u"DefDepthStencil"
struct OldDepth {
    std::uint64_t obj, flip;
};
std::vector<OldDepth> g_old_depth;  // main thread: the GX init, the change, the frame end

void call_slot(std::uint64_t obj, int slot, std::int64_t a = 0, std::int64_t b = 0, std::int64_t c = 0) {
    hle_call_guest6(reinterpret_cast<void*>(static_cast<std::uintptr_t>(at_<std::uint64_t>(at_<std::uint64_t>(obj) + slot))),
                    static_cast<std::int64_t>(obj), a, b, c, 0, 0);
}
void release_ref(std::uint64_t obj) {
    std::int32_t& refs = at_<std::int32_t>(obj + 8);
    if (refs-- <= 1) call_slot(obj, 0);  // delete
}
void release_old_depth(std::uint64_t flip) {
    std::size_t kept = 0;
    for (const OldDepth& o : g_old_depth) {
        if (flip >= o.flip + 3)
            release_ref(o.obj);
        else
            g_old_depth[kept++] = o;
    }
    g_old_depth.resize(kept);
}

bool remake_default_depth(std::uint64_t out, std::uint32_t w, std::uint32_t h) {
    const std::uint64_t old = at_<std::uint64_t>(out + 0x70);
    if (!old || at_<std::uint64_t>(old) != guest(kDepthStencilVtable)) return false;
    if (const std::uint64_t res = at_<std::uint64_t>(old + 0x40); res && at_<std::uint16_t>(res + 0x44) == w && at_<std::uint16_t>(res + 0x46) == h)
        return true;
    const std::uint64_t alloc = at_<std::uint64_t>(guest(kGxAllocatorSlot));
    if (!alloc) return false;
    const auto obj = static_cast<std::uint64_t>(hle_call_guest6(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(at_<std::uint64_t>(at_<std::uint64_t>(alloc) + 0x58))), static_cast<std::int64_t>(alloc),
        0x90, 0x10, 0, 0, 0));
    if (!obj) return false;
    std::memset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(obj)), 0, 0x90);
    at_<std::uint64_t>(obj) = guest(kDepthStencilVtable);
    at_<float>(obj + 0x10) = 1.0f;
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(obj + 0x20)), guest_ptr(g_image->mem, guest(kDepthStencilDefaults)), 16);
    at_<std::uint64_t>(obj + 0x78) = ~0ull;
    at_<std::uint32_t>(obj + 0x80) = ~0u;
    struct {
        std::uint32_t w, h, format, mips;
        std::uint64_t samples;
        std::uint32_t flags, zero;
        std::uint64_t zero2;
    } desc = {w, h, 0x14, 1, at_<std::uint64_t>(out + 0x2c), 0x100, 0, 0};
    const auto made = hle_call_guest6(guest_fn(kDepthStencilInit), static_cast<std::int64_t>(obj),
                                      static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(&desc)), 0, 0, 0, 0);
    if (!(made & 0xff)) {
        call_slot(obj, 0);
        return false;
    }
    at_<std::int32_t>(obj + 8) += 1;
    call_slot(obj, 0x80, static_cast<std::int64_t>(guest(kDefDepthStencilName)));
    at_<std::uint64_t>(out + 0x70) = obj;
    g_old_depth.push_back({old, hle_video_flip_count()});
    return true;
}

// The output entry's render size and viewport (x, y, w, h floats at +0x88;
// GX init's aspect fit inside the display buffer), which the scene contexts
// rebuild from, DefDepthStencil and the rectangle binding the display buffers
// sets; then the game's broadcast and the contexts it does not reach.
int broadcast_size(std::uint64_t dev, std::uint64_t gm, std::uint32_t ow, std::uint32_t oh, std::uint32_t w, std::uint32_t h) {
    const std::uint64_t out = at_<std::uint64_t>(dev + 0x90);
    at_<std::uint32_t>(out + 0x80) = w;
    at_<std::uint32_t>(out + 0x84) = h;
    at_<float>(out + 0x88) = 0.0f;
    at_<float>(out + 0x8c) = 0.0f;
    at_<float>(out + 0x90) = static_cast<float>(w);
    at_<float>(out + 0x94) = static_cast<float>(h);
    set_display_rect(out, w, h);
    if (!remake_default_depth(out, w, h)) host_log("resolution: DefDepthStencil was not made at %ux%u; the scene keeps a depth of its own", w, h);
    const std::vector<std::uint64_t> before = live_scene_ctxs();
    hle_call_guest6(guest_fn(kResizeBroadcast), static_cast<std::int64_t>(dev), w, h, 0, 0, 0);
    return resize_scene_ctxs(dev, gm, before, ow, oh, w, h);
}

// False when nothing changed, or the size did not fit and the one before it
// was put back.
bool resize(std::uint32_t w, std::uint32_t h) {
    std::uint64_t dev = 0, gm = 0;
    if (!device_ready(&dev, &gm)) return false;
    // The graphics manager's listener (its subobject at +8, whose resize slot
    // is `sub rdi, 8; jmp sub_269c200`) does the post-effect's part; without
    // it in the list the change would leave YEBIS at the old size.
    const std::uint64_t list = at_<std::uint64_t>(dev + 0x5f8);
    bool found = false;
    for (std::uint64_t it = list ? at_<std::uint64_t>(list + 8) : 0, end = list ? at_<std::uint64_t>(list + 0x10) : 0; it < end; it += 8)
        found |= at_<std::uint64_t>(it) == gm + 8;
    if (!found) {
        host_log("resolution: refused - the graphics manager is not among the GX device's listeners");
        return false;
    }
    const std::uint64_t out = at_<std::uint64_t>(dev + 0x90);
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint32_t was_w = at_<std::uint32_t>(out + 0x80), was_h = at_<std::uint32_t>(out + 0x84);
    g_dead_frees.store(0);
    const std::uint64_t failed = gx_resources_create_failures();
    int ctxs = broadcast_size(dev, gm, was_w, was_h, w, h);
    bool fitted = true;
    const std::uint64_t unfit = unfit_scene_ctx();
    if (unfit || gx_resources_create_failures() != failed) {
        host_log("resolution: %ux%u did not fit the render-target heap (%llu resources not made%s); back to %ux%u", w, h,
                 (unsigned long long)(gx_resources_create_failures() - failed), unfit ? ", a scene context's pass among them" : "", was_w, was_h);
        ctxs += broadcast_size(dev, gm, w, h, was_w, was_h);
        w = was_w;
        h = was_h;
        fitted = false;
    }
    // The window size SprjInitStuff derives from the res words at the start
    // (0x59404d8/dc), which sub_212c630 divides the res words by; the Sprj
    // listener sets only the res words and the UI scale. Booted at this size
    // the two are equal, so they are made equal again.
    at_<std::uint32_t>(guest(kWindowSize)) = w;
    at_<std::uint32_t>(guest(kWindowSize + 4)) = h;
    const int movies = swf_viewports(w, h);
    apply_aspect(w, h);
    hle_video_set_picture(w, h);
    g_current.store(pack(w, h));
    g_applied.fetch_add(1);
    std::uint64_t largest = 0;
    const std::uint64_t free_mib = target_heap_free_mib(dev, &largest);
    host_log("resolution: %ux%u -> %ux%u in %lld ms, the scene %s (%d other scene contexts, %d destroyed since the last change; "
             "%d Scaleform viewports, %d YEBIS frees dropped; render-target heap %llu MiB free, largest block %llu)",
             was_w, was_h, w, h,
             static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()),
             scene_target_path(w, h), ctxs, g_ctx_gone.exchange(0), movies, g_dead_frees.load(), (unsigned long long)free_mib, (unsigned long long)largest);
    return fitted && (w != was_w || h != was_h);
}

void confirm_tick() {
    if (g_confirm == Confirm::None) return;
    const std::uint64_t mm = at_<std::uint64_t>(guest(kMenuManSlot));
    if (!mm) return;
    if (g_confirm == Confirm::OpenInMenu) {
        if (const int a = option_menu_confirm_take()) {
            host_log("resolution: answered over PC Graphics (%s)", a == 1 ? "keep" : "go back");
            g_answer.store(a);
            g_confirm = Confirm::None;
        } else if (std::chrono::steady_clock::now() >= g_confirm_until) {
            // Ten seconds: the question is answered NO for the player, and
            // the answer arrives like any other; if even that does not come,
            // the change goes back regardless three seconds later.
            if (!g_confirm_closing) {
                host_log("resolution: no answer in 10 s");
                option_menu_confirm_close();
                g_confirm_closing = true;
                g_confirm_until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            } else {
                g_answer.store(2);
                g_confirm = Confirm::None;
            }
        }
        return;
    }
    if (g_confirm == Confirm::ToOpen && option_menu_confirm_possible()) {
        g_confirm_closing = false;
        // Asked from PC Graphics: the question goes on that screen's own
        // stack, modal over it (engine/option_menu.h).
        option_menu_confirm_ask(kKeepMessage);
        g_confirm = Confirm::OpenInMenu;
        g_confirm_until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        host_log("resolution: asking to keep it");
        return;
    }
    if (g_confirm == Confirm::ToOpen) {
        if (at_<std::uint32_t>(mm + 0x78)) return;  // another dialog is up; ask after it
        const std::uint64_t r = at_<std::uint64_t>(mm + 0xd08);
        if (!r) return;
        at_<std::uint32_t>(r + 0x00) = kKeepMessage;
        at_<std::uint32_t>(r + 0x04) = kYesButton;
        at_<std::uint32_t>(r + 0x08) = kNoButton;
        at_<std::uint8_t>(r + 0x10) = 0;  // the text from group 78
        at_<std::uint32_t>(r + 0x14) = 0;
        at_<std::int32_t>(r + 0x18) = -1;
        at_<std::uint32_t>(r + 0x1c) = 0;
        // As the quit-game question sets them before its dialog (sub_1b58370):
        // the menu that asked stays up behind it.
        at_<std::uint32_t>(mm + 0x184) = 1;
        at_<std::uint32_t>(mm + 0x258) = 1;
        at_<std::uint32_t>(mm + 0x254) = 0;
        at_<std::int32_t>(mm + 0x16c) = -1;
        at_<std::int32_t>(mm + 0x170) = -1;
        at_<std::int32_t>(mm + 0x1cc) = -1;
        hle_call_guest6(guest_fn(kOpenGenDialog), static_cast<std::int64_t>(r), 2, 0, 0, 0, 0);
        g_confirm_standalone = false;  // known once +0x78 is seen set, a frame later
        g_confirm = Confirm::Open;
        g_confirm_until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        host_log("resolution: asking to keep it");
        return;
    }
    // On a menu's stack the answer comes through MenuMan's handler to +0x16c;
    // standing alone, the dialog clears +0x78 when it closes and leaves the
    // button at +0x1cc - 1 YES, 2 NO, 0 when Circle closed it.
    const std::int32_t menu_answer = at_<std::int32_t>(mm + 0x16c);
    const bool up = at_<std::uint32_t>(mm + 0x78) != 0;
    const bool standalone_closed = g_confirm_standalone && !up;
    if (up) g_confirm_standalone = true;
    if (menu_answer != -1 || standalone_closed) {
        const std::int32_t v = menu_answer != -1 ? menu_answer : at_<std::int32_t>(mm + 0x1cc);
        const bool keep = menu_answer != -1 ? v == 0 : v == 1;
        host_log("resolution: answered %d (%s; +0x16c %d, +0x1cc %d)", v, keep ? "keep" : "go back", menu_answer,
                 at_<std::int32_t>(mm + 0x1cc));
        g_answer.store(keep ? 1 : 2);
        g_confirm = Confirm::None;
        return;
    }
    if (std::chrono::steady_clock::now() >= g_confirm_until) {
        host_log("resolution: no answer in 10 s");
        if (g_confirm_standalone) {
            at_<std::int32_t>(mm + 0x1cc) = -1;  // as lua_cli_CloseGenDialog closes it
            at_<std::uint32_t>(mm + 0x78) = 0;
        }
        g_answer.store(2);
        g_confirm = Confirm::None;
    }
}

// The quiet point. The present step (sub_21976e0, main thread) queues the
// frame's command context (sub_2594340) and calls the GX device's frame end,
// sub_2598fd0 (0x2197853). In the device's queued mode (+0x84) that swaps
// the double-buffered queues and enqueues the frame's play; the GX thread
// then plays it - the YEBIS composition reads the frame's scene textures -
// while the main loop goes on and the workers record the next frame. So the
// change goes right after that call returns, before the main loop moves on:
// the call is sent through a stub whose host side makes it itself, then
// waits for the frame it ended to be shown (two flips: the queue runs a
// frame behind), for the command processor and the host GPU to finish, and
// resizes. Nothing records meanwhile (the main thread is here) and nothing
// recorded is still to be played.
GUEST_ABI std::int64_t present_frame_end_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t dev = saved[5];  // rdi
    const std::uint64_t f0 = hle_video_flip_count();
    if (g_next_test < g_tests.size() && f0 >= g_tests[g_next_test].first) g_request.store(g_tests[g_next_test++].second);
    const bool settled = std::chrono::steady_clock::now().time_since_epoch().count() >= g_request_not_before.load();
    const std::uint64_t req = settled ? g_request.exchange(0) : 0;
    hle_call_guest6(guest_fn(kFrameEnd), static_cast<std::int64_t>(dev), 0, 0, 0, 0, 0);
    if (!g_old_depth.empty()) release_old_depth(f0);
    confirm_tick();
    if (!req) return 1;  // the frame end ran; back to the caller
    const auto w = static_cast<std::uint32_t>(req >> 32), h = static_cast<std::uint32_t>(req);
    if (req == g_current.load()) {
        g_request_confirm.store(false);
        return 1;
    }
    const auto start = std::chrono::steady_clock::now();
    while (hle_video_flip_count() < f0 + 2 && std::chrono::steady_clock::now() - start < std::chrono::milliseconds(500))
        host_sleep_us(200);
    hle_gnm_cp_drain();  // the command processor has run everything submitted
    host_gpu_flush();    // and the host GPU has finished it
    const bool confirm = g_request_confirm.exchange(false);
    const std::uint64_t before = g_current.load();
    if (resize(w, h)) {
        if (confirm) {
            g_before_asked.store(before);
            g_confirm = Confirm::ToOpen;
        }
    } else if (confirm) {
        g_answer.store(2);  // it went back by itself: the menus as for a no
    }
    return 1;
}

// The display buffers and the output entry come from the GX init
// (sub_2594580, called from sub_21969f0 at 0x2196bee), whose descriptor's
// display size (+0x30/+0x34) sub_21969f0 reads from the res words
// (0x2196a6b, 0x2196a7a). The output entry's render size (+0x80/+0x84) is
// that size aspect-fitted, DefDepthStencil is made at it, and the scene
// context later builds its targets from it. Here the two reads give the
// largest size, and right after the init the render size goes back to the
// res words - so the buffers fit any size and everything else starts at the
// chosen one.
GUEST_ABI std::int64_t gx_init_call_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::int64_t ok = hle_call_guest6(guest_fn(kGxInit), static_cast<std::int64_t>(saved[5]), 0, 0, 0, 0, 0);
    std::uint64_t dev = 0, gm = 0;
    device_ready(&dev, &gm);
    const std::uint64_t out = dev ? at_<std::uint64_t>(dev + 0x90) : 0;
    if (out) {
        const std::uint32_t w = g_res_words[0], h = g_res_words[1];
        host_log("resolution: display buffers %ux%u, rendering %ux%u; the scene renders %s", at_<std::uint32_t>(out + 0x80),
                 at_<std::uint32_t>(out + 0x84), w, h, scene_target_path(w, h));
        at_<std::uint32_t>(out + 0x80) = w;
        at_<std::uint32_t>(out + 0x84) = h;
        at_<float>(out + 0x88) = 0.0f;
        at_<float>(out + 0x8c) = 0.0f;
        at_<float>(out + 0x90) = static_cast<float>(w);
        at_<float>(out + 0x94) = static_cast<float>(h);
        set_display_rect(out, w, h);
        if (g_rect_size && !g_rect_resource->load())
            host_log("resolution: the display buffers' resource is not a 2D texture; debug draws keep the buffers' size");
        if (!remake_default_depth(out, w, h)) host_log("resolution: DefDepthStencil was not made at %ux%u; the scene keeps a depth of its own", w, h);
        hle_video_set_picture(w, h);
        g_current.store(pack(w, h));
        apply_aspect(w, h);
    }
    (void)ok;  // the caller ignores it too
    return 1;
}

bool pin_display_read(ElfImage* image, std::uint64_t site, std::uint64_t word, std::uint32_t value) {
    auto* p = static_cast<std::uint8_t*>(guest_ptr(image->mem, guest(site)));
    // lea rax, [rip+word]; mov eax, [rax]  ->  mov eax, imm32; nop4
    const std::int32_t disp = static_cast<std::int32_t>(static_cast<std::int64_t>(word) - static_cast<std::int64_t>(site + 7));
    std::uint8_t want[9] = {0x48, 0x8d, 0x05, 0, 0, 0, 0, 0x8b, 0x00};
    std::memcpy(want + 3, &disp, 4);
    if (std::memcmp(p, want, 9) != 0) return false;
    const std::uint64_t lo = guest(site) & ~0xfffull, hi = (guest(site) + 9 + 0xfff) & ~0xfffull;
    if (!guest_protect_rwx(&image->mem, lo, hi - lo)) return false;
    std::uint8_t code[9] = {0xb8, 0, 0, 0, 0, 0x0f, 0x1f, 0x40, 0x00};
    std::memcpy(code + 1, &value, 4);
    std::memcpy(p, code, 9);
    guest_protect_rx(&image->mem, lo, hi - lo);
    return true;
}

// Render targets come from SprjMemory's GFX_GraphicsPrivate heap (size
// table 0x4b36d40, row 1 - the mode the game runs in - entry 3: 0x8ef00000,
// 2287 MiB; CSGraphicsPrivateAllocator sends 1 MiB and up there and the rest
// to GFX_GraphicsPrivateB, entry 4, 886 MiB, each falling back to the
// other). That is the stock 1920x1080's budget, and streaming fills it as an
// area loads - so targets made larger later have nowhere to come from; the
// allocator returns null and a pass writes through it (a null read in the
// light pass, a heap overrun, a jump into data) or keeps a freed pointer
// (unfit_scene_ctx). Measured with the allocator's own counters (vtable
// 0x57aec90; +0x30 free, +0x38 largest block) in Central Yharnam: each size
// costs about 270 MiB per million pixels past 1080p (1680 for 3840x2160), and
// the stock game leaves ~750 MiB free at 1080p. So the heap grows by that
// cost and a margin for the largest size the run can reach.
//
// The heaps are carved from the 6 GiB of direct memory in table order, with
// little to spare: +1 GiB fits, but +2 GiB left no room for
// GFX_GraphicsPrivateB (it was not made, and its allocations went to the
// grown heap - nothing gained) and +3 GiB failed at the start. So past 1 GiB
// the direct memory grows by the same amount (hle_kernel_set_dmem_size). The
// memory is the PC's but pinned for the GPU, so a
// run is not grown for sizes it will not reach: a run with live changes
// covers 3440x1440's pixels (+1088 MiB, about v0.2.5's +1 GiB) or its
// starting size's when larger, and a change to more pixels than that applies
// when the game next starts (live_resolution_can). A GPU with little memory
// (host_gpu_memory_tight: a Steam Deck) covers 1080p's, the stock heap: the
// heap is imported for the GPU whole, and the gigabyte more did not fit.
constexpr std::uint64_t kPx1080 = 1920ull * 1080ull, kPxLiveDefault = 3440ull * 1440ull;
constexpr std::uint64_t kFitsInDmemMiB = 1024;  // growth the stock 6 GiB of direct memory holds
std::uint64_t g_live_max_px = ~0ull;  // the most pixels a live change may have

std::uint64_t gfx_heap_more_mib(std::uint64_t px) {
    if (px <= kPx1080) return 0;
    const double mp = static_cast<double>(px - kPx1080) / 1e6;
    const auto mib = static_cast<std::uint64_t>(std::ceil((270.0 * mp + 256.0) / 64.0)) * 64;
    return std::max(mib, kFitsInDmemMiB);
}

void grow_gfx_heap(ElfImage* image, std::uint64_t px) {
    constexpr std::uint64_t kGfxPrivate = 0x4b36d40 + 1 * 0x80 + 3 * 8;
    constexpr std::uint64_t kGfxPrivateStock = 0x8ef00000ull;
    const char* e = std::getenv("BBHOST_GFX_HEAP_MORE_MIB");
    const bool given = e && *e;
    std::uint64_t more = given ? std::strtoull(e, nullptr, 10) : gfx_heap_more_mib(px);
    if (!more) {
        if (!given) g_live_max_px = px;
        return;
    }
    auto* sz = static_cast<std::uint64_t*>(guest_ptr(image->mem, guest(kGfxPrivate)));
    if (*sz != kGfxPrivateStock) {
        host_log("resolution: refused, GFX_GraphicsPrivate's size is 0x%llx, not 0x%llx", (unsigned long long)*sz,
                 (unsigned long long)kGfxPrivateStock);
        g_live_max_px = kPx1080;
        return;
    }
    if (!guest_protect_rw(&image->mem, guest(kGfxPrivate) & ~0xfffull, 0x1000)) {
        g_live_max_px = kPx1080;
        return;
    }
    const std::uint64_t dmem_more = more > kFitsInDmemMiB ? more - kFitsInDmemMiB : 0;
    if (dmem_more && !hle_kernel_set_dmem_size(hle_kernel_dmem_size() + (dmem_more << 20))) {
        more = kFitsInDmemMiB;  // the direct memory exists already: what fits in it
        px = kPx1080 + static_cast<std::uint64_t>((static_cast<double>(more) - 256.0) / 270.0 * 1e6);
    }
    *sz = kGfxPrivateStock + (more << 20);
    g_live_max_px = given ? ~0ull : px;
    host_log("resolution: GFX_GraphicsPrivate %llu -> %llu MiB, direct memory %llu MiB - live changes up to %.1f million pixels",
             (unsigned long long)(kGfxPrivateStock >> 20), (unsigned long long)(*sz >> 20), (unsigned long long)(hle_kernel_dmem_size() >> 20),
             given ? 99.9 : static_cast<double>(px) / 1e6);
}

}  // namespace

bool live_resolution_install(ElfImage* image) {
    g_slide = image->mem.slide;
    g_image = image;
    int w = 0, h = 0;
    if (!host_opt_resolution(&w, &h)) w = 1920, h = 1080;
    const bool live = [] {
        const char* e = std::getenv("BBHOST_LIVE_RESOLUTION");
        return !(e && e[0] == '0');
    }();
    const std::uint64_t start_px = static_cast<std::uint64_t>(w) * static_cast<std::uint32_t>(h);
    const HostSettings& display = host_startup_settings();
    const std::uint32_t output_w = static_cast<std::uint32_t>(display.output_width);
    const std::uint32_t output_h = static_cast<std::uint32_t>(display.output_height);
    const std::uint64_t output_px = static_cast<std::uint64_t>(output_w) * output_h;
    const bool tight = live && host_gpu_memory_tight();
    // How large a live change may go: the largest display's size (a picture
    // larger than every screen only scales down), or the starting size when
    // that is larger. The display buffers and the render-target heap are made
    // for it at the start and cost memory pinned for the GPU for the whole
    // run whether a change ever comes or not - made at 5120x2160 they were
    // 4 x 42.5 MiB of display buffers, a 5120x2160 depth target, whole-image
    // clears of them every frame and +1,088 MiB of heap on a 1080p screen.
    // A larger entry applies at the next start (live_resolution_can).
    // BBHOST_LIVE_MAX=<w>x<h> sets the size, =max the Resolution setting's
    // largest entries (5120x2160), as before.
    std::uint32_t max_w = kMaxW, max_h = kMaxH;
    const char* max_why = "the Resolution setting's largest entries";
    int desk_w = 0, desk_h = 0;
    if (const char* e = std::getenv("BBHOST_LIVE_MAX"); e && *e) {
        unsigned ew = 0, eh = 0;
        if (std::strcmp(e, "max") != 0 && std::sscanf(e, "%ux%u", &ew, &eh) == 2 && ew && eh) {
            max_w = std::min<std::uint32_t>(ew, kMaxW);
            max_h = std::min<std::uint32_t>(eh, kMaxH);
            max_why = "BBHOST_LIVE_MAX";
        }
    } else if (host_desktop_max_size(&desk_w, &desk_h)) {
        max_w = std::min<std::uint32_t>(static_cast<std::uint32_t>(desk_w), kMaxW);
        max_h = std::min<std::uint32_t>(static_cast<std::uint32_t>(desk_h), kMaxH);
        max_why = "the largest display";
    }
    if (tight) {
        // Only sizes within 1080p's pixels: the heap is imported whole.
        max_w = std::min<std::uint32_t>(max_w, 1920);
        max_h = std::min<std::uint32_t>(max_h, 1080);
    }
    const std::uint64_t live_px = std::max(std::max(start_px, output_px), std::min(static_cast<std::uint64_t>(max_w) * max_h, kPxLiveDefault));
    grow_gfx_heap(image, !live ? start_px : live_px);
    if (tight)
        host_log("resolution: live changes up to %.1f million pixels, the GPU's memory is tight (larger sizes apply at the next start)",
                 static_cast<double>(g_live_max_px) / 1e6);
    if (!live) {
        host_log("resolution: changes wait for the next run (BBHOST_LIVE_RESOLUTION=0)");
        return false;
    }
    g_device_slot = static_cast<const std::uint64_t*>(guest_ptr(image->mem, guest(kGxDeviceSlot)));
    g_manager_slot = static_cast<const std::uint64_t*>(guest_ptr(image->mem, guest(kGraphicsManagerSlot)));
    g_res_words = static_cast<const std::uint32_t*>(guest_ptr(image->mem, guest(kResWidth)));
    // The widest and tallest entry a live change can reach (above), or the
    // starting size when larger.
    g_display[0] = std::max(std::max(max_w, static_cast<std::uint32_t>(w)), output_w);
    g_display[1] = std::max(std::max(max_h, static_cast<std::uint32_t>(h)), output_h);
    host_log("resolution: live changes up to %ux%u (%s); larger sizes apply at the next start", g_display[0], g_display[1], max_why);

    // Every edit is checked before any is made, so a mismatch leaves the
    // image as it was. The call sites and prologues are checked by the hooks
    // themselves; those go in last, the frame-end one after the rest.
    static const std::uint8_t free_pro[16] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x48, 0x83, 0xec, 0x78, 0x48, 0x89, 0xfb};
    static const std::uint8_t free_rec_pro[18] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x70, 0x49, 0x89, 0xfe};
    static const std::uint8_t swf_ctor_pro[15] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x70};
    const auto bytes_at = [&](std::uint64_t bn, const std::uint8_t* b, std::size_t n) {
        return std::memcmp(guest_ptr(image->mem, guest(bn)), b, n) == 0;
    };
    if (!bytes_at(kGxFree, free_pro, sizeof(free_pro)) || !bytes_at(kGxFreeRecord, free_rec_pro, sizeof(free_rec_pro)) ||
        !bytes_at(kSwfPlayerCtor, swf_ctor_pro, sizeof(swf_ctor_pro))) {
        host_log("resolution: refused, the GX free or SwfPlayer prologues are not as expected; changes wait for the next run");
        return false;
    }
    // The GX init hook first: on its own it only puts the render size back
    // to the res words, which is what the init gives without the pins.
    if (!engine_call_site_hook(image, guest(kGxInitCall), guest(kGxInit), reinterpret_cast<void*>(&gx_init_call_hook))) {
        host_log("resolution: the GX init hook did not go in; changes wait for the next run");
        return false;
    }
    if (!pin_display_read(image, kGxInitWidthRead, kResWidth, g_display[0]) ||
        !pin_display_read(image, kGxInitHeightRead, kResHeight, g_display[1])) {
        host_log("resolution: refused, the GX init's display size reads are not as expected; changes wait for the next run");
        return false;
    }
    if (!install_display_rect_stubs())
        host_log("resolution: the render-target viewport code is not as expected; debug draws keep the display buffers' size");
    g_both_sides = install_scene_both_sides();
    if (!engine_prologue_hook(image, guest(kGxFree), free_pro, sizeof(free_pro), reinterpret_cast<void*>(&gx_free_hook)) ||
        !engine_prologue_hook(image, guest(kGxFreeRecord), free_rec_pro, sizeof(free_rec_pro), reinterpret_cast<void*>(&gx_free_record_hook)) ||
        !engine_prologue_hook(image, guest(kSwfPlayerCtor), swf_ctor_pro, sizeof(swf_ctor_pro), reinterpret_cast<void*>(&swf_player_ctor_hook)) ||
        !engine_call_site_hook(image, guest(kListenerYebisShutdownCall), guest(kYebisShutdown), reinterpret_cast<void*>(&yebis_shutdown_call_hook)) ||
        !engine_call_site_hook(image, guest(kPresentFrameEndCall), guest(kFrameEnd), reinterpret_cast<void*>(&present_frame_end_hook))) {
        host_log("resolution: a hook did not go in; the display buffers are %ux%u but changes wait for the next run", g_display[0], g_display[1]);
        return false;
    }
    {
        // push rbp; mov rbp, rsp; push r15; push r14; push rbx; push rax; mov r14, rdi; lea rax, [rip -> vtable]
        static const std::uint8_t pro[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xfe,
                                             0x48, 0x8d, 0x05, 0x7c, 0xe3, 0x3f, 0x03};
        std::uint8_t exec[23] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xfe, 0x48, 0xb8};
        const std::uint64_t vt = guest(kSwfPlayerVtable);
        std::memcpy(exec + 15, &vt, 8);
        if (!engine_prologue_hook_exec(image, guest(kSwfPlayerDtor), pro, sizeof(pro), exec, sizeof(exec), reinterpret_cast<void*>(&swf_player_dtor_hook)))
            host_log("resolution: the SwfPlayer destructor hook did not go in");
    }
    {
        // push rbp; mov rbp, rsp; push r14; push rbx; mov r14, rsi; mov rbx, rdi; lea rax, [rip -> vtable]
        static const std::uint8_t pro[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb,
                                             0x48, 0x8d, 0x05, 0xec, 0x73, 0x0f, 0x03};
        std::uint8_t exec[23] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0xb8};
        const std::uint64_t vt = guest(kSceneCtxVtable);
        std::memcpy(exec + 15, &vt, 8);
        if (!engine_prologue_hook_exec(image, guest(kSceneCtxCtor), pro, sizeof(pro), exec, sizeof(exec), reinterpret_cast<void*>(&scene_ctx_ctor_hook)))
            host_log("resolution: the scene context constructor hook did not go in");
        // push rbp; mov rbp, rsp; push r14; push rbx; mov rbx, rdi; lea rax, [rip -> vtable]
        static const std::uint8_t dtor_pro[17] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89,
                                                  0xfb, 0x48, 0x8d, 0x05, 0x2f, 0x56, 0x0f, 0x03};
        std::uint8_t dtor_exec[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89, 0xfb, 0x48, 0xb8};
        std::memcpy(dtor_exec + 12, &vt, 8);
        if (!engine_prologue_hook_exec(image, guest(kSceneCtxDtor), dtor_pro, sizeof(dtor_pro), dtor_exec, sizeof(dtor_exec),
                                       reinterpret_cast<void*>(&scene_ctx_dtor_hook))) {
            // Without it a context destroyed since could be rebuilt; only
            // the ones the broadcast reaches are.
            g_ctx_untracked = true;
            host_log("resolution: the scene context destructor hook did not go in; contexts made on demand keep their size");
        }
    }
    {
        // Another shape than 16:9 (apply_aspect): the cameras' blend, and the plates'
        // constants and branches as the game has them.
        static const std::uint8_t blend_pro[17] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                                   0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x48};
        g_aspect_ok = aspect_sites_ok() &&
                      engine_prologue_hook(image, guest(kCameraBlend), blend_pro, sizeof(blend_pro), reinterpret_cast<void*>(&camera_blend_hook));
        if (!g_aspect_ok) host_log("resolution: the camera blend or the plates' code is not as expected; sizes other than 16:9 stretch");
    }
    for (const char* e = std::getenv("BBHOST_RESIZE_TEST"); e && *e;) {
        unsigned long long flip = 0;
        unsigned tw = 0, th = 0;
        if (std::sscanf(e, "%llu:%ux%u", &flip, &tw, &th) != 3 || !tw || !th) break;
        g_tests.emplace_back(flip, pack(tw, th));
        host_log("resolution: test change to %ux%u at flip %llu", tw, th, flip);
        e = std::strchr(e, ',');
        if (e) ++e;
    }
    g_installed = true;
    host_log("resolution: live changes on, display buffers %ux%u", g_display[0], g_display[1]);
    return true;
}

bool live_resolution_available() {
    std::uint64_t dev = 0, gm = 0;
    return g_installed && device_ready(&dev, &gm);
}

bool live_resolution_can(unsigned w, unsigned h) {
    return g_installed && w && h && w <= g_display[0] && h <= g_display[1] && static_cast<std::uint64_t>(w) * h <= g_live_max_px;
}

bool live_resolution_request(unsigned w, unsigned h, bool confirm) {
    if (!live_resolution_can(w, h)) {
        if (g_installed && static_cast<std::uint64_t>(w) * h > g_live_max_px)
            host_log("resolution: %ux%u needs more render-target memory than this run was given at its start; it applies when the game next starts", w, h);
        return false;
    }
    g_request_confirm.store(confirm);
    g_request_not_before.store(confirm ? std::chrono::steady_clock::now().time_since_epoch().count() + kMenuSettleNs : 0);
    g_request.store(pack(w, h));
    return true;
}

int live_resolution_take_answer() { return g_answer.exchange(0); }

void live_resolution_go_back() {
    const std::uint64_t b = g_before_asked.exchange(0);
    if (b && b != g_current.load()) {
        g_request_confirm.store(false);
        g_request_not_before.store(0);
        g_request.store(b);
    }
}

std::uint64_t live_resolution_current(unsigned* w, unsigned* h) {
    const std::uint64_t c = g_current.load();
    if (w) *w = static_cast<unsigned>(c >> 32);
    if (h) *h = static_cast<unsigned>(c & 0xffffffffu);
    return g_applied.load();
}
