// The follow camera's step (decomp/camera/follow_camera.cpp) against the game's own
// code (tests/decomp/eboot_kit.h): generated cameras run frame after frame through
// the game's ChrExFollowCam::Update and through ours - each its own copy,
// fed the same characters, sticks, flags, worlds and dt - and after every
// frame the whole 0x350-byte object, the GameStateMan overrides it consumes,
// every sphere cast's question and the debug draw are compared. The casts
// and the draw are stubs both versions reach (the game's code either way);
// the sine, arctangent and tangent the step calls through the game's import
// stubs are bound to the host's, for both. Run under the console's
// floating-point mode.
//
// Two of ours' additions run against the game's code too. The mouse: on half
// the cameras the mouse camera's own stubs (engine/mouse_camera_stubs.cpp) go
// into the game's code - its free camera's store and the fast turn's exits,
// as the hooks place them - and both versions turn by the same counts. The
// hold of auto-rotation: the community patch's nops go into the game's code
// at all four of its stores or at one, and ours, reading them, must hold the
// same; with the three the mouse holds, ours also runs on the game's own
// bytes holding them by its flag alone.
#include "decomp/decomp.h"
#include "engine/camera_target.h"
#include "engine/frame_rate_step.h"
#include "engine/mouse_camera.h"
#include "engine/mouse_camera_step.h"
#include "engine/mouse_camera_stubs.h"

#include "eboot_kit.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <sys/mman.h>

void follow_cam_update(std::uint8_t* cam, std::uint8_t* chr, void* world, float dt);
bool follow_camera_body_ok(const std::uint8_t* entry);

// The compare stand-in's, which the test does not run (engine/mouse_camera.cpp in the game).
void mouse_camera_compare_begin() { mouse_camera::begin_update(); }

namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using s32 = std::int32_t;
using u64 = std::uint64_t;

constexpr u64 kUpdate = 0x183ac60;
constexpr std::size_t kCam = 0x350;

eboot_kit::Rng g_rng(0xca3e7a);

// ---- the game's libm, through its import stubs -------------------------------

float host_fsin(u32 q, u32, float x) { return (q & 1) ? std::cos(x) : std::sin(x); }
float host_atan2f(float y, float x) { return std::atan2(y, x); }
float host_tanf(float x) { return std::tan(x); }

// ---- the sphere casts and the debug draw, as stubs both versions call -------
//
// A cast is a pure function of what it is asked (and the frame's salt): hit
// or not, the fraction and the hit point - so the game's step and ours see
// the same answers exactly when they ask the same questions.

u64 g_salt = 0;
u64 mix(u64 h, u64 v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

// A float's bits, every NaN the same: which NaN an operation passes on
// depends on its operand order, which a compiler may swap.
u32 bits(float v) { return v != v ? 0x7fc00000u : std::bit_cast<u32>(v); }

struct Cast {
    u32 filter, from[3], delta[3], radius;
    bool hit_wanted, normal_wanted;
    bool operator==(const Cast&) const = default;
};
std::vector<Cast> g_casts;

bool cast_stub(void*, u32 filter, const float* from, const float* delta, float* hit, float* normal, float radius, float* frac) {
    Cast c{filter, {}, {}, bits(radius), hit != nullptr, normal != nullptr};
    for (int k = 0; k < 3; ++k) {
        c.from[k] = bits(from[k]);
        c.delta[k] = bits(delta[k]);
    }
    g_casts.push_back(c);
    u64 h = mix(g_salt, filter);
    for (int k = 0; k < 3; ++k) h = mix(mix(h, bits(from[k])), bits(delta[k]));  // xyz: a sphere cast reads no w
    h = mix(h, bits(radius));
    const bool hits = (h & 3) != 0;
    const float t = static_cast<float>((h >> 8) & 0xffff) / 65536.0f;
    if (frac) *frac = t;
    if (hits && hit) {
        for (int k = 0; k < 3; ++k) hit[k] = from[k] + delta[k] * t;
        hit[3] = 1.0f;
    }
    if (hits && normal) {
        normal[0] = 0.0f;
        normal[1] = 1.0f;
        normal[2] = 0.0f;
        normal[3] = 0.0f;
    }
    return hits;
}

std::vector<float> g_drawn;  // the debug draw's matrices, in order
int g_draw_begins = 0;
void draw_begin(void*, int) { ++g_draw_begins; }
void draw_submit(void*, const float* mat) {
    for (int k = 0; k < 12; ++k) g_drawn.push_back(mat[k]);
}

// ---- memory the game's code reads with aligned loads -------------------------

std::vector<void*> g_blocks;
u8* block(std::size_t n) {
    void* p = std::aligned_alloc(64, (n + 63) & ~std::size_t{63});
    std::memset(p, 0, n);
    g_blocks.push_back(p);
    return static_cast<u8*>(p);
}

template <class T>
void put(u8* p, std::size_t at, T v) {
    std::memcpy(p + at, &v, sizeof v);
}
void put_slot(u64 bn, const void* v) { std::memcpy(eboot_kit::at(bn), &v, 8); }

float value(float lo, float hi) { return g_rng.value(lo, hi); }
void vec4(u8* p, std::size_t at, float lo, float hi, float w) {
    put(p, at + 0, value(lo, hi));
    put(p, at + 4, value(lo, hi));
    put(p, at + 8, value(lo, hi));
    put(p, at + 12, w);
}

// ---- the world the step reads -------------------------------------------------

u8* g_gsm;         // GameStateMan: +0x14/+0x18 the camera-param overrides
u8* g_repo;        // SoloParamRepository: LockCamParam in slot 33
u8* g_lockcam;     // the param file
std::vector<s32> g_ids;

void make_params() {
    g_repo = block(0xa00);
    u8* cap = block(0x100);
    u8* inner = block(0x100);
    const int rows = 12;
    const std::size_t data = 0x40 + rows * 0x18, row_at = 0x400;
    u8* raw = block(0x10 + 0x800 + rows * 8);
    g_lockcam = raw + 0x10;
    put<u32>(raw, 0, static_cast<u32>(row_at + rows * 0x40));  // the file's size: the id index follows
    put<std::uint16_t>(g_lockcam, 0xa, rows);
    g_lockcam[0x2d] = 4;  // u64 row offsets
    g_lockcam[0x2e] = 2;
    (void)data;
    g_ids.clear();
    u8* index = g_lockcam + ((row_at + rows * 0x40 + 15) & ~std::size_t{15});
    for (int i = 0; i < rows; ++i) {
        const s32 id = i == 0 ? 0 : 1000 + i * 100;  // id 0: the row a missing id falls back to
        g_ids.push_back(id);
        put<u64>(g_lockcam, 0x40 + i * 0x18 + 8, row_at + i * 0x40);
        u8* row = g_lockcam + row_at + i * 0x40;
        put(row, 0x0, value(1.5f, 6.0f));    // camDistTarget
        put(row, 0x4, value(-60.0f, 0.0f));  // rotRangeMinX, degrees
        put(row, 0x8, value(0.0f, 1.0f));    // lockRotXShiftRatio
        put(row, 0xc, value(0.5f, 2.0f));    // chrOrgOffset_Y
        put(row, 0x14, value(30.0f, 60.0f)); // camFovY, degrees
        put<s32>(index, i * 8, id);
        put<s32>(index, i * 8 + 4, i);
    }
    put<u32>(g_repo, 0x9b8, 1);
    put(g_repo, 0x9c0, cap);
    put(cap, 0x70, inner);
    put(inner, 0x70, g_lockcam);
    put_slot(0x5940340, g_repo);
}

// The followed character: ChrCtrl (+0x58) with the pivot character (+8) and
// the pad objects (+0x80, +0x280: their +0x40 is the camera stick), and the
// transform module (*(chr+0x3b0)+0x68: +0x1d0 rotation, +0x1e0 position,
// +0x326 yaw-only).
struct Chr {
    u8 *chr, *ctrl, *pivot_mod, *pad, *pad2, *mod;
};

u8* g_local_chr = nullptr;
u8* g_follow_target = nullptr;
void* spectator_target(void* chr, void*) {
    return chr == g_local_chr ? g_follow_target : nullptr;
}

// Model the two native spectator hooks without redirecting the local pad.
void spectator_reference(u8* chr, u8* ctrl, const Chr& local, const u8* target) {
    std::memcpy(chr, local.chr, 0x400);
    std::memcpy(ctrl, local.ctrl, 0x300);
    put(chr, 0x58, ctrl);
    if (!target) return;
    const u8* owner;
    const u8* pivot = nullptr;
    const u8* model;
    std::memcpy(&owner, target + 0x58, sizeof owner);
    if (owner) std::memcpy(&pivot, owner + 8, sizeof pivot);
    std::memcpy(&model, target + 0x3b0, sizeof model);
    if (pivot) put(ctrl, 8, pivot);
    if (model) put(chr, 0x3b0, model);
}

Chr make_chr() {
    Chr c{};
    c.chr = block(0x400);
    c.ctrl = block(0x300);
    u8* pivot = block(0x400);
    u8* pivot_box = block(0x80);
    c.pivot_mod = block(0x400);
    c.pad = block(0x80);
    c.pad2 = block(0x80);
    u8* box = block(0x80);
    c.mod = block(0x400);
    put(c.chr, 0x58, c.ctrl);
    put(c.ctrl, 0x8, pivot);
    put(pivot, 0x3b0, pivot_box);
    put(pivot_box, 0x68, c.pivot_mod);
    put(c.ctrl, 0x80, c.pad);
    put(c.chr, 0x3b0, box);
    put(box, 0x68, c.mod);
    return c;
}

void vary_chr(Chr& c) {
    vec4(c.pivot_mod, 0x1e0, -50.0f, 50.0f, 1.0f);
    vec4(c.mod, 0x1d0, -3.2f, 3.2f, 0.0f);
    vec4(c.mod, 0x1e0, -50.0f, 50.0f, 1.0f);
    c.mod[0x326] = g_rng.chance(0.5);
    put<u64>(c.ctrl, 0x280, g_rng.chance(0.3) ? reinterpret_cast<u64>(c.pad2) : 0);
    for (u8* pad : {c.pad, c.pad2}) {
        const bool still = g_rng.chance(0.3);
        put(pad, 0x40, still ? 0.0f : value(-1.0f, 1.0f));
        put(pad, 0x44, still ? 0.0f : value(-1.0f, 1.0f));
        put(pad, 0x48, value(-1.0f, 1.0f));
        put(pad, 0x4c, value(-1.0f, 1.0f));
    }
}

// The keep-out spheres: a list at [0x593e858]+8, each entry a centre (+0),
// a radius byte (+0x60), flags (+0x61), the next (+0x68).
u8* g_keepout_root;
void make_keepout(const u8* cam) {
    g_keepout_root = block(0x40);
    u8* prev = nullptr;
    const int n = g_rng.range(0, 3);
    for (int i = 0; i < n; ++i) {
        u8* e = block(0x80);
        float centre[4];
        std::memcpy(centre, cam + 0x40, 16);
        for (int k = 0; k < 3; ++k) centre[k] += value(-2.0f, 2.0f);
        centre[3] = 1.0f;
        std::memcpy(e, centre, 16);
        e[0x60] = static_cast<u8>(g_rng.range(0, 4));
        e[0x61] = static_cast<u8>(g_rng.u32());
        if (prev) put(prev, 0x68, e);
        else put(g_keepout_root, 8, e);
        prev = e;
    }
    put_slot(0x593e858, g_keepout_root);
}

// A camera: values around the constructor's, flags and ids by chance.
void make_camera(u8* cam) {
    for (std::size_t at = 0; at < kCam; at += 4) put(cam, at, value(-2.0f, 2.0f));
    vec4(cam, 0x10, -1.0f, 1.0f, 0.0f);
    vec4(cam, 0x20, -1.0f, 1.0f, 0.0f);
    vec4(cam, 0x30, -1.0f, 1.0f, 0.0f);
    vec4(cam, 0x40, -50.0f, 50.0f, 1.0f);
    for (std::size_t at = 0x60; at <= 0x100; at += 0x10) vec4(cam, at, -50.0f, 50.0f, at >= 0x90 ? 1.0f : 0.0f);
    vec4(cam, 0x110, -0.5f, 0.5f, 0.0f);
    put(cam, 0x124, g_rng.chance(0.6) ? 0.0f : value(-1.0f, 1.0f));
    put(cam, 0x50, value(0.5f, 1.0f));
    put(cam, 0x140, value(-1.2f, 1.2f));
    put(cam, 0x144, value(-3.2f, 3.2f));
    for (std::size_t at : {0x170u, 0x174u, 0x178u, 0x17cu, 0x18cu, 0x190u, 0x194u, 0x198u, 0x19cu, 0x1a0u, 0x1a4u, 0x1a8u, 0x1acu,
                           0x1b0u, 0x1b4u, 0x1b8u, 0x1c4u, 0x1c8u, 0x1ccu, 0x234u, 0x240u, 0x23cu, 0x288u, 0x330u})
        put(cam, at, value(0.0f, 1.0f));
    for (std::size_t at : {0x180u, 0x184u, 0x188u}) put(cam, at, value(1.0f, 6.0f));
    put(cam, 0x1ec, value(0.5f, 1.4f));
    put(cam, 0x1f0, value(-1.4f, -0.2f));
    for (std::size_t at : {0x1d8u, 0x1dcu, 0x1e0u, 0x1e4u, 0x1e8u, 0x264u, 0x268u}) put(cam, at, value(0.05f, 1.0f));
    for (std::size_t at = 0x204; at <= 0x230; at += 4) put(cam, at, value(0.5f, 6.0f));
    put(cam, 0x238, value(0.5f, 1.5f));
    vec4(cam, 0x250, -50.0f, 50.0f, 1.0f);
    vec4(cam, 0x160, -0.5f, 2.0f, 0.0f);
    for (std::size_t at : {0x13cu, 0x13du, 0x15cu, 0x260u, 0x261u, 0x262u, 0x263u, 0x28cu}) cam[at] = g_rng.chance(0.15);
    cam[0x274] = g_rng.chance(0.7);
    cam[0x275] = g_rng.chance(0.7);
    cam[0x276] = g_rng.chance(0.3);
    cam[0x2d0] = g_rng.chance(0.8);
    cam[0x2d1] = g_rng.chance(0.2);  // the keep-out spheres' debug draw
    cam[0x334] = g_rng.chance(0.9);
    cam[0x344] = g_rng.chance(0.3);
    const auto id = [] { return g_rng.chance(0.6) ? -1 : g_rng.chance(0.85) ? g_ids[g_rng.range(0, 11)] : g_rng.chance(0.5) ? 0 : 77; };
    put<s32>(cam, 0x320, g_rng.chance(0.8) ? -1 : id());
    put<s32>(cam, 0x324, id());
    put<s32>(cam, 0x328, id());
    put<s32>(cam, 0x32c, g_ids[g_rng.range(0, 11)]);
}

void vary_frame(u8* a, u8* b) {
    // What other code writes between frames: the one-shot requests, the
    // lock-on target, the overrides.
    for (std::size_t at : {0x13cu, 0x260u, 0x261u, 0x262u}) {
        const u8 v = g_rng.chance(0.05);
        a[at] = b[at] = v;
    }
    float t[4] = {value(-50.0f, 50.0f), value(-50.0f, 50.0f), value(-50.0f, 50.0f), 1.0f};
    std::memcpy(a + 0x250, t, 16);
    std::memcpy(b + 0x250, t, 16);
    put<s32>(g_gsm, 0x14, g_rng.chance(0.9) ? -1 : g_ids[g_rng.range(0, 11)]);
    put<s32>(g_gsm, 0x18, g_rng.chance(0.9) ? -1 : g_ids[g_rng.range(0, 11)]);
    eboot_kit::at(0x5527a94)[0] = g_rng.chance(0.5);
}

// ---- the mouse, and the hold ---------------------------------------------------

// What the store stub calls in the game's code: the hook's step.
void free_camera_host(u8* cam, u8* input, float stick_pitch, float stick_yaw) {
    mouse_camera::turn_camera(cam, input, stick_pitch, stick_yaw);
}

constexpr u64 kTurnStore = 0x183ce54;
constexpr u64 kRampExits[2] = {0x183f9bc, 0x183fabf};
u8 g_store_game[19], g_ramp_game[2][14];

void jump_to(u64 bn, std::size_t n, const void* dest) {
    u8* code = eboot_kit::at(bn);
    const u64 to = reinterpret_cast<u64>(dest);
    code[0] = 0xff;
    code[1] = 0x25;
    std::memset(code + 2, 0, 4);
    std::memcpy(code + 6, &to, 8);
    std::memset(code + 14, 0xcc, n - 14);
}

// The mouse camera's stubs in the game's code, or its own bytes back.
void place_mouse_stubs(bool on) {
    if (on) {
        jump_to(kTurnStore, sizeof g_store_game, reinterpret_cast<const void*>(&bb_mouse_camera_stub));
        for (const u64 at : kRampExits) jump_to(at, 14, reinterpret_cast<const void*>(&bb_mouse_camera_ramp_stub));
    } else {
        std::memcpy(eboot_kit::at(kTurnStore), g_store_game, sizeof g_store_game);
        for (int i = 0; i < 2; ++i) std::memcpy(eboot_kit::at(kRampExits[i]), g_ramp_game[i], 14);
    }
}

// The community patch's nops at the hold sites in `mask`, the game's bytes at the rest.
void place_hold_nops(unsigned mask) {
    for (int i = 0; i < 4; ++i) {
        const auto& site = mouse_camera::kHoldSites[i];
        std::memcpy(eboot_kit::at(site.bn), (mask >> i & 1) ? mouse_camera::kNop9 : site.game, sizeof site.game);
    }
}

// A frame's mouse: still, a few counts, a sweep, or a fraction (SDL's
// relative motion is a float).
float counts() {
    if (g_rng.chance(0.4)) return 0.0f;
    if (g_rng.chance(0.1)) return static_cast<float>(g_rng.range(-900, 900));
    if (g_rng.chance(0.2)) return g_rng.uniform(-3.0f, 3.0f);
    return static_cast<float>(g_rng.range(-20, 20));
}

bool same(const u8* a, const u8* b, std::size_t n, std::size_t* where) {
    for (std::size_t k = 0; k < n; k += 4) {
        float x, y;
        std::memcpy(&x, a + k, 4);
        std::memcpy(&y, b + k, 4);
        if (eboot_kit::same_float(x, y)) continue;
        *where = k;
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* test = "decomp_follow_camera";
    eboot_kit::load(test);
    static const u8 kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xd8, 0x03, 0x00, 0x00};
    if (std::memcmp(eboot_kit::at(kUpdate), kEntry, sizeof kEntry) != 0) {
        std::printf("%s: the update is not where the decomp expects it\n", test);
        return 1;
    }
    // The body check: the game's body passes, with the run-time patches too,
    // and a changed byte does not.
    {
        std::vector<u8> body(eboot_kit::at(kUpdate), eboot_kit::at(0x183facd));
        bool ok = follow_camera_body_ok(body.data());
        const u8 fov[4] = {0x48, 0x89, 0xf0, 0x90};
        std::memcpy(&body[0x183af4e - kUpdate], fov, 4);
        body[0x183dc97 - kUpdate] = 0xd6;
        body[0x183dc98 - kUpdate] = 0x05;
        body[0x183dcfe - kUpdate] = 0x00;
        ok = ok && follow_camera_body_ok(body.data());
        // The community patch's nops at the hold sites pass; anything else there does not.
        for (const auto& site : mouse_camera::kHoldSites) std::memcpy(&body[site.bn - kUpdate], mouse_camera::kNop9, 9);
        ok = ok && follow_camera_body_ok(body.data());
        body[mouse_camera::kHoldSites[2].bn - kUpdate] = 0xcc;
        ok = ok && !follow_camera_body_ok(body.data());
        for (const auto& site : mouse_camera::kHoldSites) std::memcpy(&body[site.bn - kUpdate], site.game, 9);
        body[0x183ce54 - kUpdate] ^= 0xff;
        ok = ok && !follow_camera_body_ok(body.data());
        body[0x183ce54 - kUpdate] ^= 0xff;
        // A changed constant ours writes as a literal fails it too; the two
        // 60 fps constants, which ours reads, do not.
        for (u64 page : {0x4cf2000ull, 0x4d25000ull}) mprotect(eboot_kit::at(page), 0x1000, PROT_READ | PROT_WRITE);
        for (const u64 at : {0x4cf2534ull, 0x4d25e88ull, 0x4d25e68ull, 0x4d25e94ull}) {
            eboot_kit::at(at)[1] ^= 1;
            const bool read_at_run_time = at == 0x4d25e68 || at == 0x4d25e94;
            ok = ok && follow_camera_body_ok(body.data()) == read_at_run_time;
            eboot_kit::at(at)[1] ^= 1;
        }
        if (!ok) {
            std::printf("%s: the body check is wrong\n", test);
            return 1;
        }
    }
    for (const auto& [name, host] : {std::pair<const char*, const void*>{"_FSin", reinterpret_cast<const void*>(&host_fsin)},
                                     {"atan2f", reinterpret_cast<const void*>(&host_atan2f)},
                                     {"tanf", reinterpret_cast<const void*>(&host_tanf)}}) {
        if (!eboot_kit::bind(name, host)) {
            std::printf("%s: the eboot has no %s import\n", test, name);
            return 1;
        }
    }
    g_gsm = block(0x100);
    put_slot(0x5956678, g_gsm);
    const float axis[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    std::memcpy(eboot_kit::at(0x597be80), axis, 16);
    make_params();
    eboot_kit::stub(0x1c090e0, reinterpret_cast<const void*>(&cast_stub));
    eboot_kit::stub(0x135d810, reinterpret_cast<const void*>(&draw_submit));
    // RendMan (slot 0x5940298) +0x20 -> the draw lists: [+0x20] the current
    // one, its object at +0x10 + 8 * that, +0x40 the shape object, whose
    // vtable +0x28 begins a shape.
    u8* rendman = block(0x40);
    u8* lists = block(0x80);
    u8* list = block(0x80);
    u8* shape = block(0x100);
    static void* shape_vtable[8] = {};
    shape_vtable[0x28 / 8] = reinterpret_cast<void*>(&draw_begin);
    put(rendman, 0x20, lists);
    put<s32>(lists, 0x20, 0);
    put(lists, 0x10, list);
    put(list, 0x40, shape);
    put(shape, 0, reinterpret_cast<void*>(shape_vtable));
    put_slot(0x5940298, rendman);
    u8 shape_before[0x100], shape_a[0x100];
    int dummy_world = 0;

    const auto game = eboot_kit::fn<void(GUEST_ABI*)(u8*, u8*, void*, float)>(kUpdate);
    const int cameras = argc > 1 ? std::atoi(argv[1]) : 2000, frames = 60;
    g_mouse_camera_host = reinterpret_cast<void*>(&free_camera_host);
    g_mouse_camera_resume = 0x183ce67;
    std::memcpy(g_store_game, eboot_kit::at(kTurnStore), sizeof g_store_game);
    for (int i = 0; i < 2; ++i) std::memcpy(g_ramp_game[i], eboot_kit::at(kRampExits[i]), 14);
    if (!camera_target::register_callback(spectator_target, nullptr) ||
        camera_target::register_callback(spectator_target, nullptr)) return 1;
    int mouse_frames = 0, held_frames = 0, original_frames = 0, spectator_frames = 0;
    u8* a = block(kCam);
    u8* b = block(kCam);
    u8 gsm_a[0x20], gsm_b[0x20];
    int bad = 0, steps = 0;
    u8* code = eboot_kit::at(0x183a000);
    mprotect(code, 0x6000, PROT_READ | PROT_WRITE | PROT_EXEC);
    for (u64 page : {0x4d25000ull}) mprotect(eboot_kit::at(page), 0x1000, PROT_READ | PROT_WRITE);
    for (int c = 0; c < cameras; ++c) {
        g_rng.specials = c % 4 == 3;
        // The run-time patches: fov-uncap (on by default) and the 60 fps
        // sites and constants, each by chance.
        static const u8 kFovGame[4] = {0x48, 0x0f, 0x46, 0xc6}, kFovPatched[4] = {0x48, 0x89, 0xf0, 0x90};
        std::memcpy(eboot_kit::at(0x183af4e), (c % 3) ? kFovPatched : kFovGame, 4);
        const bool sixty = (c % 5) >= 3;
        eboot_kit::at(0x183dc97)[0] = sixty ? 0xd6 : 0x57;
        eboot_kit::at(0x183dc97)[1] = sixty ? 0x05 : 0x06;
        eboot_kit::at(0x183dcfe)[0] = sixty ? 0x00 : 0x01;
        const u32 ease = sixty ? 0x3d523176u : 0x3dcccccdu, rate = sixty ? 0x3c89b0c3u : 0x3d088889u;  // engine/frame_rate.cpp at 60
        std::memcpy(eboot_kit::at(0x4d25e68), &ease, 4);
        std::memcpy(eboot_kit::at(0x4d25e94), &rate, 4);
        // The Camera tuning Original at a frame of 1/30 s (engine/frame_rate.cpp)
        // on every other camera with the game's sites: the converted path,
        // with the game's own values and a frame of the rate they are for, so
        // every weight it converts is the game's and the step must be too.
        const bool original = !sixty && (c / 5) % 2 == 1;
        frame_rate_step::g_frame_seconds.store(original ? 1.0f / 30.0f : 0.0f);
        frame_rate_step::g_camera_tuned_fps.store(original ? 30 : 0);
        if (original) original_frames += frames;
        // The mouse on odd cameras, at a sensitivity and with axes of its own.
        const bool mouse = c % 2 == 1;
        place_mouse_stubs(mouse);
        const float per_count = mouse_camera::degrees_per_count(g_rng.range(0, 10));
        u32 per_count_bits;
        std::memcpy(&per_count_bits, &per_count, 4);
        mouse_camera::g_degrees_per_count.store(per_count_bits);
        const u32 axes = (g_rng.chance(0.3) ? mouse_camera::kInvertX : 0u) | (g_rng.chance(0.3) ? mouse_camera::kInvertY : 0u);
        // The hold: none on most cameras, all four sites, or one of them; on
        // every other all-four camera the three the mouse holds instead, and
        // ours holds them by its flag alone.
        const int hold_mode = c % 7;
        const bool by_flag = hold_mode == 2 && (c / 7) % 2 == 1;
        const unsigned nops = by_flag ? mouse_camera::kHeldWithMouse
                              : hold_mode == 2 ? 0xfu
                              : hold_mode >= 3 ? 1u << (hold_mode - 3)
                                               : 0u;
        place_hold_nops(nops);
        Chr chr = make_chr();
        Chr remote = make_chr();
        u8* reference_chr = block(0x400);
        u8* reference_ctrl = block(0x300);
        g_local_chr = chr.chr;
        make_camera(a);
        std::memcpy(b, a, kCam);
        make_keepout(a);
        for (int f = 0; f < frames; ++f) {
            vary_chr(chr);
            vary_chr(remote);
            const u8* remote_model;
            const u8* remote_pivot;
            std::memcpy(&remote_model, remote.chr + 0x3b0, sizeof remote_model);
            std::memcpy(&remote_pivot, remote.ctrl + 8, sizeof remote_pivot);
            g_follow_target = c % 3 != 0 && f % 5 != 0 ? remote.chr : nullptr;
            // A retiring target can lose either attachment independently.
            if (f % 7 == 0) put<u64>(remote.chr, 0x58, 0);
            if (f % 11 == 0) put<u64>(remote.ctrl, 8, 0);
            if (f % 13 == 0) put<u64>(remote.chr, 0x3b0, 0);
            spectator_reference(reference_chr, reference_ctrl, chr, g_follow_target);
            vary_frame(a, b);
            const float dt = g_rng.chance(0.9) ? value(0.01f, 0.05f) : value(-0.1f, 1.0f);
            void* world = g_rng.chance(0.5) ? &dummy_world : nullptr;
            u8* followed = g_rng.chance(0.02) ? nullptr : chr.chr;  // no character: only the param ease runs
            g_salt = g_rng.next();
            u8 gsm[0x20];
            std::memcpy(gsm, g_gsm, sizeof gsm);
            std::memcpy(shape_before, shape, sizeof shape_before);
            g_drawn.clear();
            g_draw_begins = 0;
            g_casts.clear();
            // This frame's counts, for the game's update (taken first, as the
            // hook on its prologue takes them) and then for ours.
            const float dx = mouse ? counts() : 0.0f, dy = mouse ? counts() : 0.0f;
            const u8 free_before = bb_mouse_camera_free_now;
            mouse_camera::g_flags.store(axes);
            mouse_camera::g_counts.store(mouse_camera::pack(dx, dy));
            mouse_camera::begin_update();
            const u64 turns_before = mouse_camera::g_turns.load();
            game(a, followed ? reference_chr : nullptr, world, dt);
            const u8 free_a = bb_mouse_camera_free_now;
            if (mouse_camera::g_turns.load() != turns_before) ++mouse_frames;
            const std::vector<Cast> casts_a = g_casts;
            g_casts.clear();
            std::memcpy(gsm_a, g_gsm, sizeof gsm_a);
            std::memcpy(shape_a, shape, sizeof shape_a);
            const std::vector<float> drawn_a = g_drawn;
            const int begins_a = g_draw_begins;
            std::memcpy(g_gsm, gsm, sizeof gsm);
            std::memcpy(shape, shape_before, sizeof shape_before);
            g_drawn.clear();
            g_draw_begins = 0;
            bb_mouse_camera_free_now = free_before;
            mouse_camera::g_flags.store(axes | (by_flag ? mouse_camera::kHoldAutoRotation : 0u));
            mouse_camera::g_counts.store(mouse_camera::pack(dx, dy));
            if (by_flag) place_hold_nops(0);
            follow_cam_update(b, followed, world, dt);
            if (followed && g_follow_target) ++spectator_frames;
            put(remote.chr, 0x58, remote.ctrl);
            put(remote.ctrl, 8, remote_pivot);
            put(remote.chr, 0x3b0, remote_model);
            if (by_flag) place_hold_nops(nops);
            if (nops) ++held_frames;
            // Whether the free camera ran: noted by the stubs in the game's code, by ours itself.
            const bool free_same =
                !mouse || (mouse_camera::g_frame.free_before == (free_before != 0) && bb_mouse_camera_free_now == free_a);
            mouse_camera::g_flick_counts.store(0);
            mouse_camera::g_flick_updates.store(0);
            std::memcpy(gsm_b, g_gsm, sizeof gsm_b);
            ++steps;
            std::size_t at = 0;
            std::size_t where = 0;
            const bool draws_same = begins_a == g_draw_begins && drawn_a.size() == g_drawn.size() &&
                                    same(reinterpret_cast<const u8*>(drawn_a.data()), reinterpret_cast<const u8*>(g_drawn.data()),
                                         drawn_a.size() * sizeof(float), &where) &&
                                    same(shape_a, shape, sizeof shape_a, &where);
            const bool casts_same = casts_a == g_casts;
            const bool camera_same = same(a, b, kCam, &at) && std::memcmp(gsm_a, gsm_b, sizeof gsm_a) == 0;
            if (!draws_same || !casts_same || !camera_same || !free_same) {
                if (++bad <= 12) {
                    if (!draws_same) std::printf("camera %d frame %d: the debug draw differs\n", c, f);
                    if (!free_same) std::printf("camera %d frame %d: whether the free camera ran differs\n", c, f);
                    for (std::size_t k = 0; !casts_same && k < std::max(casts_a.size(), g_casts.size()); ++k) {
                        if (k < casts_a.size() && k < g_casts.size() && casts_a[k] == g_casts[k]) continue;
                        const auto show = [](const char* who, const std::vector<Cast>& v, std::size_t k) {
                            if (k >= v.size()) return (void)std::printf("  %s: no cast %zu\n", who, k);
                            const Cast& x = v[k];
                            std::printf("  %s cast %zu: filter %x from %08x %08x %08x delta %08x %08x %08x radius %08x hit %d normal %d\n", who,
                                        k, x.filter, x.from[0], x.from[1], x.from[2], x.delta[0], x.delta[1], x.delta[2], x.radius,
                                        x.hit_wanted, x.normal_wanted);
                        };
                        std::printf("camera %d frame %d: the casts differ at %zu of %zu/%zu\n", c, f, k, casts_a.size(), g_casts.size());
                        show("game", casts_a, k);
                        show("ours", g_casts, k);
                        break;
                    }
                    if (!camera_same) {
                        u32 x = 0, y = 0;
                        std::memcpy(&x, a + at, 4);
                        std::memcpy(&y, b + at, 4);
                        std::printf("camera %d frame %d: +0x%zx differs: the game's %08x (%g), ours %08x (%g)\n", c, f, at, x,
                                    std::bit_cast<float>(x), y, std::bit_cast<float>(y));
                    }
                }
                if (!camera_same) std::memcpy(b, a, kCam);  // go on from the game's
            }
        }
    }
    place_mouse_stubs(false);
    place_hold_nops(0);
    frame_rate_step::g_frame_seconds.store(0.0f);
    frame_rate_step::g_camera_tuned_fps.store(0);
    std::printf("%s: %d cameras, %d frames, %d differ; %d frames turned by the mouse, %d with auto-rotation held, %d with "
                "the Original camera tuning at a 1/30 s frame; %d spectator frames\n",
                test, cameras, steps, bad, mouse_frames, held_frames, original_frames, spectator_frames);
    return bad || !mouse_frames || !held_frames || !original_frames || !spectator_frames ? 1 : 0;
}
