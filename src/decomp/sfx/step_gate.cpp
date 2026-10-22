// The effects' frames between whole steps (docs/decomp.md, the effects
// track's first milestone): an effect instance's update, sub_2c4ff30 at Binary
// Ninja 0x2c4ff30, the FFX light module's, sub_131e660 at 0x131e660, and the
// wobble motions' updates, sub_2e89d10, sub_2e8e430 and sub_2e896f0.
//
// Above 30 fps the effects step in whole 1/30 s (or 1/60 s) steps
// (engine/sfx_schedule.h) and the frames between run the effect manager's
// whole update with a step of 0: every instance follows its parent and makes
// its render items each frame, and nothing moves in time. But the game's
// effects count part of their lives in updates, so such a frame still moved
// them on (a census, 2026-10-19: lavapipe under BBHOST_SFX_SIM_FPS):
// a node's state edges (0x2c4df80's switch: 2->3, 6->7 or 9, 7->9 come once
// an update whatever the step), its emissions in state 1 (a count at +0xe0,
// the next at +0xe4, each one more draw from the effect RNG), the definition
// changes its conditions pick (sub_2c81730), the controllers' conditions
// (sub_2c820f0), and the light module's flicker phase (+0x30, which adds the
// node's age times a rate on every call) and its per-call blend (+0x110).
// Ebrietas' 1251721 then never ended for about one attack in six.
//
// On a frame the gate is told is between whole steps (frame_rate.cpp,
// decomp_sfx_gate_frame) - and that is not the game's own still frame,
// SprjSfx+0x8c, whose zero the game counts at 30 too - the two do only what
// a step of 0 has to: an active instance's nodes run their containers (the
// world matrix that follows the parent, the modules' render items) and
// nothing else, its controllers wait for the next whole step, and a light
// module runs as the game's with its phase rate and its blend held for the
// call (+0x80 as 0, +0x38 as 1, both put back) - every other effect of the
// call exactly the game's. Every other frame, and a creation's pre-roll
// (sub_2c500e0's updates, with its stand-in render context, vtable
// 0x56dbd80), is the game's own code through the trampoline.
#include "decomp/sfx/step_gate.h"

#include "decomp/sfx/node_update.h"

#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "log.h"

using namespace decomp;

#ifdef DECOMP_SFX_STEP_GATE_MUTANTS
extern int g_sfx_gate_mutant;
#define MUTANT(n) (g_sfx_gate_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kSprjSfx = 0x5940370;            // SprjSfx's singleton pointer
constexpr u64 kNodeNext = 0x2c4f0e0;           // a node tree's next node (child +0x70, sibling +0x68, parent +0x60)
constexpr u64 kNodeRelease = 0x2c4ef00;        // a dead node's two clean-ups before its slot is freed
constexpr u64 kNodeClear = 0x2c4df20;
constexpr u64 kPoolFree = 0x2fa1c20;           // (pool, p, size, alignment)
constexpr u64 kNodePool = 0x5ac8290;           // the pool the nodes come from (a pointer)
constexpr u64 kContainerUpdate = 0x2c4c8c0;    // (node + 0x80, &step, definition, render context)
constexpr u64 kControllerUpdate = 0x2c820f0;   // (controller, instance, step)
constexpr u64 kPrerollContext = 0x56dbd80;     // sub_2c500e0's stand-in render context's vtable
constexpr u64 kGxRecordUpdate = 0x2cbcf50;     // a GPU particle system's record taking an item: (record, item)
constexpr u64 kGxType2713 = 0x56eb0d0;         // GPU particle type 0x2713's vtable: its parameters are curves read at the elapsed time
constexpr u64 kGxTypeRibbon = 0x56e3620;       // FFX-format GX type 0x28's vtable: a ribbon (update sub_2ec1e00)
constexpr std::size_t kRibbonDefinition = 0x12c;  // what sub_2ec1e00 reads of its definition (type +8): up to +0x128's dword
constexpr u64 kWobbleA = 0x2e89d10;            // motion 0x570e6d0's update (and the end of 0x570e690's): count +0x6c, period +0x68
constexpr u64 kWobbleABase = 0x2e866f0;        //   its base motion, which moves by the step
constexpr u64 kWobbleB = 0x2e8e430;            // motion 0x570e8d0's update: count +0x7c, period +0x78
constexpr u64 kWobbleBBase = 0x2e8bfd0;        //   its base motion
constexpr u64 kWobbleFollow = 0x2e896f0;       // motion 0x570e690's update: the lerp toward the parent, then kWobbleA
constexpr u64 kLightFades = 0x12e5500;         // the effect manager's fading GX lights (+0x118): (list, step)
constexpr u64 kWindSourcesStep = 0x1314660;    // the wind sources' registry (*0x58b0bb0) stepped: (registry, step)
constexpr u64 kWindReceivers = 0x27174e0;      // the wind system's (*0x5940e00) receivers' update: (system, step)

using InstanceUpdate = GUEST_ABI u8 (*)(u64 instance, u64 context, float step);
using LightUpdate = GUEST_ABI void (*)(u64 module, u64 params, u64 context);
using NodeFn = GUEST_ABI void (*)(u64 node);
using PoolFree = GUEST_ABI void (*)(u64 pool, u64 p, u64 size, u64 alignment);
using NextFn = GUEST_ABI u64 (*)(u64 node);
using ContainerFn = GUEST_ABI void (*)(u64 container, float* step, u64 definition, u64 context);
using ControllerFn = GUEST_ABI u64 (*)(u64 controller, u64 instance, float step);
using MotionFn = GUEST_ABI void (*)(u64 motion, u64 node, const float* step, u64 local);
using GxRecordFn = GUEST_ABI void (*)(u64 record, u64 item);
using LightFadesFn = GUEST_ABI void (*)(u64 list, float step);
using WindStepFn = GUEST_ABI void (*)(u64 registry, float step);
using WindReceiversFn = GUEST_ABI void (*)(u64 system, float step);

void* g_game_instance = nullptr;
void* g_game_light = nullptr;
void* g_game_wobble_a = nullptr;
void* g_game_wobble_b = nullptr;
void* g_game_wobble_follow = nullptr;
void* g_game_gx_record = nullptr;
void* g_game_light_fades = nullptr;
void* g_game_wind_step = nullptr;
void* g_game_wind_receivers = nullptr;

std::atomic<u32> g_between{0};
std::atomic<u32> g_step_between{0};  // a frame between whole effect steps, the gate on or off (decomp_sfx_step_frame)
// The frame's own time and the carry before and after it (decomp_sfx_frame_time),
// for the smooth steps' frame domain, and whether those run (decomp_sfx_smooth).
std::atomic<float> g_frame_step{0.0f}, g_frame_from{0.0f}, g_frame_into{0.0f};
std::atomic<u32> g_smooth{0};
std::atomic<u64> g_smooth_updates{0}, g_smooth_kept{0};
std::atomic<u64> g_lights_smooth{0}, g_lights_counted{0}, g_fades_smooth{0};
std::atomic<u64> g_frames_between{0}, g_instances_held{0}, g_lights_held{0}, g_motions_held{0}, g_still_frames{0}, g_records_held{0};
std::atomic<u64> g_ribbons_between{0}, g_ribbons_whole{0}, g_ribbons_unmarked{0};
std::atomic<u64> g_wind_steps_smooth{0}, g_wind_steps_other{0}, g_records_whole{0}, g_motions_smooth{0};
std::atomic<u64> g_warmups_held{0};
std::atomic<u32> g_hold_warmup{1};  // BBHOST_SFX_WIND_WARMUP_HOLD=0: the wind system's warm-up counts every frame, as before

// The ribbons (their GX types) of the instances the smooth steps run this frame, marked as smooth_update takes an
// instance, for the record update: it runs later from the render context's queue, and a pre-roll's (sub_2c500e0's
// updates, through the game's node update) or a held instance's look the same to it. A small open table of words
// (the frame's serial << 48 | the type), a slot of another frame free to take; wiped when the serial's 16 bits wrap.
constexpr std::size_t kRibbonSlots = 1024;
std::atomic<u64> g_ribbon_marks[kRibbonSlots] = {};
std::atomic<u32> g_frame_serial{1};
constexpr u64 kMarkType = 0xffffffffffffull;
inline std::size_t ribbon_slot(u64 type) { return static_cast<std::size_t>(((type >> 3) * 0x9E3779B97F4A7C15ull) >> 54) & (kRibbonSlots - 1); }
inline u64 ribbon_frame() { return static_cast<u64>(g_frame_serial.load(std::memory_order_relaxed) & 0xffffu) << 48; }
bool ribbon_mark(u64 type) {
    const u64 now = ribbon_frame(), want = now | (type & kMarkType);
    std::size_t h = ribbon_slot(type);
    for (int k = 0; k < 16; ++k, h = (h + 1) & (kRibbonSlots - 1)) {
        u64 w = g_ribbon_marks[h].load(std::memory_order_relaxed);
        while ((w & ~kMarkType) != now)
            if (g_ribbon_marks[h].compare_exchange_weak(w, want, std::memory_order_release, std::memory_order_relaxed)) return true;
        if (w == want) return true;
    }
    return false;
}
bool ribbon_marked(u64 type) {
    const u64 now = ribbon_frame(), want = now | (type & kMarkType);
    std::size_t h = ribbon_slot(type);
    for (int k = 0; k < 16; ++k, h = (h + 1) & (kRibbonSlots - 1)) {
        const u64 w = g_ribbon_marks[h].load(std::memory_order_acquire);
        if (w == want) return true;
        if ((w & ~kMarkType) != now) return false;
    }
    return false;
}

// The game's vaddss x, x, y: the record's time plus an item's step, as sub_2cbcf50 adds them; and vsubss, its
// elapsed time.
inline float add_ss(float x, float y) {
    __asm__("addss %1, %0" : "+x"(x) : "x"(y));
    return x;
}
inline float sub_ss(float x, float y) {
    __asm__("subss %1, %0" : "+x"(x) : "x"(y));
    return x;
}
// The next float up (or down) from a finite x, by its bits (no libm: nextafter may set errno, the host's TLS).
inline float ulp_toward(float x, bool up) {
    u32 b = std::bit_cast<u32>(x);
    if ((b & 0x7fffffffu) == 0) return std::bit_cast<float>(up ? 1u : 0x80000001u);
    const bool positive = !(b & 0x80000000u);
    b = positive == up ? b + 1 : b - 1;
    return std::bit_cast<float>(b);
}

// A frame bbhost put between whole steps that is not the game's own still frame.
inline bool held() {
    if (!g_between.load(std::memory_order_relaxed)) return false;
    const u64 sfx = load<u64>(game_address(kSprjSfx));
    if (sfx && load<u32>(sfx + 0x8c) && !MUTANT(1)) {
        g_still_frames.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

// A frame between whole effect steps - the gate on or off, 30 or 60 steps a
// second - that is not the game's own still frame.
inline bool between_steps() {
    if (!g_step_between.load(std::memory_order_relaxed)) return false;
    const u64 sfx = load<u64>(game_address(kSprjSfx));
    return !(sfx && load<u32>(sfx + 0x8c) && !MUTANT(18));
}

// A node on such a frame: what 0x2c4df80 does with its state machine, its
// definitions and its set-up left out - the container updates it makes and
// the time adds (of a step of 0) in their places. A dead node (+0x78 bit 2)
// is kept for the next whole step's update to free: a container update marks
// a node dead in state 3 once its last pending render items are out and its
// +0xa0 module, if any, is done (0x2c4ccf1), and the node update frees it
// only at its next entry - at 30 the next update, 1/30 s
// later; freed on the frame after, its instance and request retired 1/30 -
// 1/fps early (Ebrietas's 1251701 and 1251721, 2026-10-19: 17 ms at 60, 25
// at 120). An inactive one (bit 0 clear) is left alone; a staged one (bit 3)
// runs its container once, a plain one twice (before its definition's time
// add at +0xe0, and at the end), each with a copy of the step and the render
// context.
inline u8 node_held(u64 node, const float* step, u64 definition, u64 context) {
    const u32 flags = load<u32>(node + 0x78);
    if (flags & 4) return MUTANT(8) ? 0 : 1;
    if (!(flags & 1)) return 1;
    const auto container = game_function<ContainerFn>(kContainerUpdate);
    if (!(flags & 8) && !MUTANT(2)) {
        float s = *step;
        container(node + 0x80, &s, definition, context);
        store<float>(node + 0xe0, load<float>(node + 0xe0) + *step);
    }
    float s = *step;
    container(node + 0x80, &s, definition, MUTANT(3) ? 0 : context);
    store<float>(node + 0x7c, load<float>(node + 0x7c) + *step);
    return 1;
}

// --- the smooth steps (effect steps smooth) -----------------------------------------
//
// An instance whose every class is on the converted list runs the frame domain:
// on a frame with a whole step its controllers with that step and its nodes
// through sfx_node::update with the counted parts given the step and the
// container calls the frame's; on a frame between, only the container calls,
// the frame's step, the nodes' times put as far past their whole step as the
// frame is. The classes: each node's motion (+0x90; a composite's two at +8
// and +0x10), its modules (+0x98, +0xa0, +0xa8) and the targets of its pending
// binders (the container's list +0x30: next +0x18, target +0x20) - and a GX
// binder's type instance (*(target +0x10) +8). An instance with one class off
// the list stays on the held frames above, the class counted.

constexpr u64 kManagerOf = 0x70;  // SprjSfx +0x70: the effect manager (its time scale +0x5d8, its own still flag +0x5ce)

// The converted classes' vtables, by Binary Ninja address: the classes whose
// updates do with their step only what time does, each read in Binary Ninja -
// the motions, the modules, the binders, the StandardParticle and Spark
// simulations, the FFX-format GX types and the wind sources 0x283c and 0x283d
// also held to the game's own code by tests/decomp/sfx_classes_test.cpp (one
// whole step against the frame domain's frames) - and the FFX light, whose one
// counted part sfx_light_update runs once a whole step
// (tests/decomp/sfx_step_gate_test.cpp). Off the list, and so on the held
// frames as with 30: the module 0x5760f20. The game never makes GPU particle type
// 0x2712 (0x56eb030): its registered create (0x12e7da0) returns null, so a
// record of it never gets a type and never updates. The wobbles are on it: their
// counts and re-rolls held between whole steps by sfx_wobble_a, sfx_wobble_b and
// sfx_wobble_follow, their base motions (0x570e5d0's, 0x570e7d0's) and the
// follows by the frame's own time - 0x570e890's follow is the game's own, its
// call of sub_2e8e430 at its end reaching sfx_wobble_b. The ribbon, FFX-format
// GX type 0x28, and the GPU particle type 0x2713 (its parameters curves read at
// the elapsed time) are on it for sfx_gx_record_update, which inserts the
// ribbon's points once a whole step at the whole step's time and gives 0x2713's
// curves the whole step's elapsed time (tests/decomp/sfx_step_gate_test.cpp).
constexpr u64 kConverted[] = {
    0x570e650,  // motion: velocity along itself by step x speed curve x +0x24, position by step x velocity, facing the move
    0x570e610,  // motion: the same with gravity (data_5ad1150 x a curve at the node's time, in the parent's frame) by step
    0x570e5d0,  // motion: the same with the effects' wind world's force by step, and its drift added to the move
    0x570e6d0,  // motion: 0x570e5d0's, then a count and a re-roll every +0x68 calls - held between whole steps (sfx_wobble_a)
    0x570e690,  // motion: a share of the parent's move, then 0x570e6d0's - its count held between whole steps (sfx_wobble_follow)
    0x570e850,  // motion: the velocity plus the speed curve x a direction, the position by step x that, facing the move
    0x570e810,  // motion: the same with gravity (a curve at the node's time) by step, in the parent's frame
    0x570e7d0,  // motion: the same with the wind world's force by step and its drift added to the move
    0x570e8d0,  // motion: 0x570e7d0's, then a count and a re-roll of the direction every +0x78 calls - held between whole steps (sfx_wobble_b)
    0x570e890,  // motion: a share of the parent's move (the direction turned with it), then 0x570e8d0's (its call of sub_2e8e430 ours)
    0x570e790,  // motion: turned by step x its rates (or three curves)
    0x570e710,  // motion: a transform; never reads its step
    0x570e910,  // motion: its position a curve's value at the node's time; never reads its step
    0x56eb2e0,  // module: the children follow; never reads the parameters the step is in
    0x5756e50,  // module (Sprj): its model advanced by step x +0xc4
    0x56e3090,  // binder: CPU particles; time += step, ended by it, the item's count born at it
    0x56e3100,  // binder: a GPU particle system's record; prev = cur, cur += step, its type handed the difference
    0x56eb1d0,  // GPU particle type 0x2710 StandardParticle: simulation time, emission timers by the elapsed time
    0x56eb120,  // GPU particle type 0x2718 Spark: the same
    0x56eb220,  // GPU particle type 0x2711 WanderingVision: the elapsed time handed to its simulation
    0x56ea9b0,  // GPU particle type 0x283e: a volume from the record's matrix; no time
    0x56eaa90,  // GPU particle type 0x27d8: no time
    0x56eb290,  // GPU particle type 0x2968: constant bounds
    0x56ebba0,  // module: the FFX light; its flicker phase counted once a whole step at the step's time (sfx_light_update)
    0x56e35d0,  // GX type 0x6b (FFX format): only its bounds, from its properties' largest values; no time
    0x56e3530,  // GX type 0x14 / 0x3b (FFX format): +0x10 += the elapsed time x a rate read at the scaled time, its bounds
    0x56e3580,  // GX type 0x2b (FFX format): +0x20, or +0x18 and +0x1c, += the elapsed time x rates at the scaled time
    0x56ebe20,  // GX type 3 / 0x3d (FFX format): +0x18 and +0x1c += the elapsed time x rates; a root definition's value
    0x56eadf0,  // GX type 0x283c: a wind source; its fade += the elapsed time, its velocity the move over it (0 for none)
    0x56eac40,  // GX type 0x283d: a wind source without the velocity; its fade += the elapsed time
    0x56e3620,  // GX type 0x28 (FFX format): a ribbon; its inserts once a whole step at its time (sfx_gx_record_update)
    0x56eb0d0,  // GPU particle type 0x2713: its curves read at the whole step's elapsed time (sfx_gx_record_update)
};
const u64* g_converted = kConverted;
std::size_t g_converted_count = sizeof kConverted / sizeof kConverted[0];

// The converted classes' vtables as the game has them, in a small open table that conv_build fills when the smooth
// steps start (and the tests when they change the list): one probe or two a class where the list's scan took up to
// one compare a converted class, for every motion, module and binder of every smooth instance, every frame. A list
// it was not built for is scanned as before.
constexpr std::size_t kConvSlots = 128;
struct ConvTable {
    const u64* list = nullptr;
    std::size_t count = 0;
    u64 slot[kConvSlots] = {};
};
ConvTable g_conv;
inline std::size_t conv_slot(u64 vt) { return static_cast<std::size_t>(((vt >> 3) * 0x9E3779B97F4A7C15ull) >> 57) & (kConvSlots - 1); }
void conv_build() {
    ConvTable t;
    t.list = g_converted;
    t.count = g_converted_count;
    for (std::size_t i = 0; i < g_converted_count && i < kConvSlots / 2; ++i) {
        const u64 vt = game_address(g_converted[i]);
        std::size_t h = conv_slot(vt);
        while (t.slot[h] && t.slot[h] != vt) h = (h + 1) & (kConvSlots - 1);
        t.slot[h] = vt;
    }
    if (g_converted_count > kConvSlots / 2) t.list = nullptr;  // too many for the table: scanned
    g_conv = t;
}
inline bool converted_vt(u64 vt) {
    if (!vt) return false;
    if (g_conv.list == g_converted && g_conv.count == g_converted_count && g_conv.list) {
        for (std::size_t h = conv_slot(vt);; h = (h + 1) & (kConvSlots - 1)) {
            const u64 seen = g_conv.slot[h];
            if (seen == vt) return true;
            if (!seen) return false;
        }
    }
    for (std::size_t i = 0; i < g_converted_count; ++i)
        if (vt == game_address(g_converted[i])) return true;
    return false;
}
inline bool converted_class(u64 object) { return !object || converted_vt(load<u64>(object)); }

// A node tree's next node, as sub_2c4f0e0 (kNodeNext) walks it: its first child (+0x70), else the next sibling
// (+0x68) of it or of the nearest parent (+0x60) that has one, else none.
inline u64 node_next(u64 node) {
    if (const u64 child = load<u64>(node + 0x70)) return child;
    for (; node; node = load<u64>(node + 0x60))
        if (const u64 sibling = load<u64>(node + 0x68)) return sibling;
    return 0;
}

// The classes that kept an instance on the held frames: their vtables and counts, a small open table
// (a slot taken once, never given back; past it the overflow count).
constexpr std::size_t kKeptSlots = 32;
std::atomic<u64> g_kept_class[kKeptSlots] = {}, g_kept_count[kKeptSlots] = {};
std::atomic<u64> g_kept_overflow{0};
void count_kept(u64 vtable) {
    for (std::size_t i = 0; i < kKeptSlots; ++i) {
        u64 seen = g_kept_class[i].load(std::memory_order_relaxed);
        if (!seen && g_kept_class[i].compare_exchange_strong(seen, vtable, std::memory_order_relaxed)) seen = vtable;
        if (seen == vtable) {
            g_kept_count[i].fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    g_kept_overflow.fetch_add(1, std::memory_order_relaxed);
}

// A motion is on the list, or a composite (0x570e750: its two parts at +8 and
// +0x10, run one after the other) whose parts are - never by itself, and four
// deep at most; else its class, in `kept`.
constexpr u64 kCompositeMotion = 0x570e750;
bool motion_converted(u64 motion, int depth, u64& kept) {
    if (!motion) return true;
    const u64 vt = load<u64>(motion);
    if (vt == game_address(kCompositeMotion)) {
        if (depth < 4)
            return motion_converted(load<u64>(motion + 8), depth + 1, kept) && motion_converted(load<u64>(motion + 0x10), depth + 1, kept);
    } else if (converted_class(motion)) {
        return true;
    }
    kept = vt;
    return false;
}

// The first class of the instance's off the list (its vtable), or 0 when every one is on it. With `marks`, the same
// walk collects the instance's ribbons and GPU particle types 0x2713 for ribbon_mark (up to kMarks; past that
// `*nmarks` is kMarks + 1 and mark_ribbons walks them again).
constexpr int kMarks = 16;
u64 first_unconverted(u64 instance, u64* marks = nullptr, int* nmarks = nullptr, bool* unbound = nullptr) {
    const u64 binder_vt = game_address(0x56e3100), ribbon_vt = game_address(kGxTypeRibbon), vt2713 = game_address(kGxType2713);
    for (u64 node = load<u64>(instance + 0x78); node; node = node_next(node)) {
        u64 kept = 0;
        if (!motion_converted(load<u64>(node + 0x90), 0, kept)) return kept;
        for (const u64 slot : {0x98ull, 0xa0ull, 0xa8ull}) {
            const u64 m = load<u64>(node + slot);
            if (!converted_class(m)) return load<u64>(m);
        }
        for (u64 item = load<u64>(node + 0x80 + 0x30); item; item = load<u64>(item + 0x18)) {
            const u64 target = load<u64>(item + 0x20);
            if (!converted_class(target)) return load<u64>(target);
            // a GX binder's type instance (the GPU particle systems' kinds)
            if (target && load<u64>(target) == binder_vt) {
                const u64 holder = load<u64>(target + 0x10);
                const u64 type = holder ? load<u64>(holder + 8) : 0;
                if (!type && unbound) *unbound = true;  // the GX side makes its type when it binds the record
                if (!converted_class(type)) return load<u64>(type);
                if (marks && type && (load<u64>(type) == ribbon_vt || load<u64>(type) == vt2713)) {
                    if (*nmarks < kMarks) marks[*nmarks] = type;
                    if (*nmarks <= kMarks) ++*nmarks;
                }
            }
        }
    }
    return 0;
}

// The instance's ribbons and GPU particle types 0x2713 marked for this frame (ribbon_mark); false when one finds no slot.
bool mark_ribbons(u64 instance) {
    for (u64 node = load<u64>(instance + 0x78); node; node = node_next(node))
        for (u64 item = load<u64>(node + 0x80 + 0x30); item; item = load<u64>(item + 0x18)) {
            const u64 target = load<u64>(item + 0x20);
            if (!target || load<u64>(target) != game_address(0x56e3100)) continue;
            const u64 holder = load<u64>(target + 0x10);
            const u64 type = holder ? load<u64>(holder + 8) : 0;
            if (type && (load<u64>(type) == game_address(kGxTypeRibbon) || load<u64>(type) == game_address(kGxType2713)) && !ribbon_mark(type))
                return false;
        }
    return true;
}

// The instances' classifications (classify), kept for the frames between whole steps. An instance's node tree, the
// classes its nodes use and its pending binders change only in what the effects count, on whole steps - the node
// state machine sets nodes up, binds binders and frees the dead ones then - except a GX binder's type, which the GX
// side makes when it first binds the record. So a classification made since the last whole step stands on a frame
// between them while the instance still has its definition (+0x38), its root node (+0x78) and that node's first child
// (+0x70), and every GX binder it met had its type; at most kSlotMarks ribbons and 0x2713 types to mark are kept with
// it (more: classified every frame). A frame with a whole step always classifies anew. One 64-byte slot an instance,
// by its address (a collision is classified again), behind a sequence word: a writer takes it odd by
// compare-and-swap or leaves it, readers check it even and unchanged around their reads. A released instance's slot
// is cleared at its last update (+0x88 bit 2), so an instance made later at the same address starts afresh.
constexpr std::size_t kClassSlots = 2048;
constexpr int kSlotMarks = 2;
struct alignas(64) ClassSlot {
    std::atomic<u32> seq{0};
    std::atomic<u32> serial{0};
    std::atomic<u64> inst{0}, def{0}, root{0}, child{0}, kept{0};
    std::atomic<u64> marks[kSlotMarks] = {};
};
ClassSlot g_class_slots[kClassSlots];
std::atomic<u32> g_step_serial{1};  // frames with a whole step so far
std::atomic<u64> g_class_made{0}, g_class_reused{0};
inline ClassSlot& class_slot(u64 instance) {
    return g_class_slots[static_cast<std::size_t>(((instance >> 4) * 0x9E3779B97F4A7C15ull) >> 53) & (kClassSlots - 1)];
}

// The instance's first class off the list (0: none) and, when there is none, its ribbons and 0x2713 types to mark
// (`*nmarks` past kMarks: too many to collect), as first_unconverted finds them - kept for the frames to the next
// whole step.
u64 classify(u64 instance, u64* marks, int* nmarks) {
    const u64 def = load<u64>(instance + 0x38), root = load<u64>(instance + 0x78), child = root ? load<u64>(root + 0x70) : 0;
    const u32 serial = g_step_serial.load(std::memory_order_relaxed);
    ClassSlot& slot = class_slot(instance);
    *nmarks = 0;
    if (g_between.load(std::memory_order_relaxed)) {
        const u32 s1 = slot.seq.load(std::memory_order_acquire);
        if (!(s1 & 1) && slot.inst.load(std::memory_order_relaxed) == instance &&
            (slot.serial.load(std::memory_order_relaxed) == serial || MUTANT(51)) &&
            ((slot.def.load(std::memory_order_relaxed) == def && slot.root.load(std::memory_order_relaxed) == root &&
              slot.child.load(std::memory_order_relaxed) == child) ||
             MUTANT(52))) {
            const u64 kept = slot.kept.load(std::memory_order_relaxed);
            u64 kept_marks[kSlotMarks];
            for (int i = 0; i < kSlotMarks; ++i) kept_marks[i] = slot.marks[i].load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (slot.seq.load(std::memory_order_relaxed) == s1) {
                g_class_reused.fetch_add(1, std::memory_order_relaxed);
                for (int i = 0; i < kSlotMarks && kept_marks[i]; ++i) marks[(*nmarks)++] = kept_marks[i];
                return kept;
            }
        }
    }
    bool unbound = false;
    const u64 kept = first_unconverted(instance, marks, nmarks, &unbound);
    g_class_made.fetch_add(1, std::memory_order_relaxed);
    if ((!unbound || MUTANT(53)) && *nmarks <= kSlotMarks) {
        u32 s = slot.seq.load(std::memory_order_relaxed);
        if (!(s & 1) && slot.seq.compare_exchange_strong(s, s + 1, std::memory_order_relaxed)) {
            std::atomic_thread_fence(std::memory_order_release);
            slot.inst.store(instance, std::memory_order_relaxed);
            slot.serial.store(serial, std::memory_order_relaxed);
            slot.def.store(def, std::memory_order_relaxed);
            slot.root.store(root, std::memory_order_relaxed);
            slot.child.store(child, std::memory_order_relaxed);
            slot.kept.store(kept, std::memory_order_relaxed);
            for (int i = 0; i < kSlotMarks; ++i) slot.marks[i].store(!kept && i < *nmarks ? marks[i] : 0, std::memory_order_relaxed);
            slot.seq.store(s + 2, std::memory_order_release);
        }
    }
    return kept;
}

// A released instance's classification let go (its address may hold another instance next).
void forget_class(u64 instance) {
    ClassSlot& slot = class_slot(instance);
    u32 s = slot.seq.load(std::memory_order_relaxed);
    if (slot.inst.load(std::memory_order_relaxed) != instance || (s & 1)) return;
    if (!slot.seq.compare_exchange_strong(s, s + 1, std::memory_order_relaxed)) return;
    std::atomic_thread_fence(std::memory_order_release);
    if (slot.inst.load(std::memory_order_relaxed) == instance) slot.inst.store(0, std::memory_order_relaxed);
    slot.seq.store(s + 2, std::memory_order_release);
}

u8 smooth_update(u64 instance, u64 context, float step) {
    g_smooth_updates.fetch_add(1, std::memory_order_relaxed);
    const bool tick = !g_between.load(std::memory_order_relaxed) || MUTANT(20);
    const float scale = load<float>(instance + 8);
    // The manager's time scale and still flag, as it hands the instances their step (0x26ff340).
    const u64 manager = load<u64>(load<u64>(game_address(kSprjSfx)) + kManagerOf);
    const float time_scale = load<float>(manager + 0x5d8);
    const bool still = load<u8>(manager + 0x5ce) != 0;
    // The frame's step as the manager would hand it (none while it is still), then the instance's scale
    // on it, the game's order: at 30 the whole step's bits.
    sfx_node::Steps st;
    st.tick = tick;
    st.smooth = true;
    st.counted = step * scale;
    st.frame = (still ? 0.0f : g_frame_step.load(std::memory_order_relaxed) * time_scale) * scale;
    if (!still) {
        st.from = g_frame_from.load(std::memory_order_relaxed) * time_scale * (MUTANT(21) ? 1.0f : scale);
        st.into = g_frame_into.load(std::memory_order_relaxed) * time_scale * scale;
    }
    if (tick) {
        for (u64 c = load<u64>(instance + 0x80); c; c = load<u64>(c + 0x20)) game_function<ControllerFn>(kControllerUpdate)(c, instance, st.counted);
    }
    const u64 definition = load<u64>(instance + 0x38);
    if (!definition) return 1;
    for (u64 node = load<u64>(instance + 0x78); node;) {
        const u64 next = node_next(node);
        if (!sfx_node::update(node, st, definition, context)) {
            game_function<NodeFn>(kNodeRelease)(node);
            game_function<NodeFn>(kNodeClear)(node);
            game_function<PoolFree>(kPoolFree)(load<u64>(game_address(kNodePool)), node, 0x100, 0x10);
        }
        node = next;
    }
    return 1;
}

}  // namespace

// sub_2c4ff30(instance, render context, step): a released instance (+0x88
// bit 2) loses its node tree; an active one (bit 0) scales the step by +0x8,
// steps its controllers (+0x80, next +0x20) and then its nodes (+0x78) with
// its definition (+0x38), freeing the dead. On a held frame the controllers
// are skipped and each node runs node_held(), which frees none.
DECOMP_LEAF u8 sfx_instance_update(u64 instance, u64 context, float step) {
    const u32 flags = load<u32>(instance + 0x88);
    if (g_smooth.load(std::memory_order_relaxed) && (flags & 4) && !MUTANT(54)) forget_class(instance);
    if (g_smooth.load(std::memory_order_relaxed) && !(flags & 4) && (flags & 1)) {
        const u64 sfx = load<u64>(game_address(kSprjSfx));
        if (!(sfx && load<u32>(sfx + 0x8c))) {  // the game's own still frame stays the game's
            u64 marks[kMarks];
            int nmarks = 0;
            if (const u64 kept = classify(instance, marks, &nmarks)) {
                g_smooth_kept.fetch_add(1, std::memory_order_relaxed);
                count_kept(kept);
            } else if (nmarks > kMarks ? !mark_ribbons(instance) : !std::all_of(marks, marks + nmarks, [](u64 t) { return ribbon_mark(t); })) {
                g_smooth_kept.fetch_add(1, std::memory_order_relaxed);
                count_kept(game_address(kGxTypeRibbon));  // no room to mark a ribbon (or a 0x2713) this frame: held, as at 30
            } else {
                return smooth_update(instance, context, step);
            }
        }
    }
    if (!held() || (flags & 4) || !(flags & 1)) return reinterpret_cast<InstanceUpdate>(g_game_instance)(instance, context, step);
    g_instances_held.fetch_add(1, std::memory_order_relaxed);
    const float s = step * load<float>(instance + 8);
    if (MUTANT(4)) {  // the controllers stepped anyway
        for (u64 c = load<u64>(instance + 0x80); c; c = load<u64>(c + 0x20)) game_function<ControllerFn>(kControllerUpdate)(c, instance, s);
    }
    const u64 definition = load<u64>(instance + 0x38);
    if (!definition) return 1;
    for (u64 node = load<u64>(instance + 0x78); node;) {
        const u64 next = node_next(node);
        float copy = s;
        if (!node_held(node, &copy, definition, context)) {
            game_function<NodeFn>(kNodeRelease)(node);
            game_function<NodeFn>(kNodeClear)(node);
            game_function<PoolFree>(kPoolFree)(load<u64>(game_address(kNodePool)), node, 0x100, 0x10);
        }
        node = next;
    }
    return 1;
}

// sub_131e660(module, params, render context): the FFX light module's update
// - its GX light's colour, radius and place from its curves at the node's
// age, its flicker from noise at the phase +0x30, which first adds the age
// times the rate +0x80 (when +0x7c > 0), and +0x110, which mode +0x38
// (2) blends toward a curve each call and (0) sets, (1) keeps, and its
// colours +0xf0 / +0x100 and radii +0x114 / +0x118, which mode +0x3c does
// the same with (a new definition sets the modes, sub_131f120). On a held
// frame (not a pre-roll's) the call runs with +0x80 as 0 and +0x38 as 1, so
// the phase and +0x110 stay; both are put back, and the phase's bits too.
//
// The smooth steps' frame domain (a light in the node whose container call
// sfx_node::smooth_call names): on a frame with a whole step the game's call
// first, at the node's time of the step - its phase add, blends and their
// stores exactly as on the update the game makes at 30 - and then on every
// frame the call at the frame's time with the rate as 0 and a blending mode
// (2) as keeping (1), all put back: the GX light's colour, radius, place and
// visibility at the frame's own time, the flicker from the phase of the last
// whole step, 30 noise steps a second as on the PS4.
inline void light_frame_call(LightUpdate game, u64 module, u64 params, u64 context) {
    const u32 rate = load<u32>(module + 0x80), phase = load<u32>(module + 0x30);
    const u32 mode = load<u32>(module + 0x38), colour_mode = load<u32>(module + 0x3c);
    if (!MUTANT(24)) store<float>(module + 0x80, 0.0f);
    if (mode == 2 && !MUTANT(25)) store<u32>(module + 0x38, 1u);
    if (colour_mode == 2 && !MUTANT(25)) store<u32>(module + 0x3c, 1u);
    game(module, params, context);
    store<u32>(module + 0x80, rate);
    store<u32>(module + 0x38, mode);
    store<u32>(module + 0x3c, colour_mode);
    if (!MUTANT(24)) store<u32>(module + 0x30, phase);  // the add of the age times 0 kept a -0 or an infinite age's NaN from it
}

DECOMP_LEAF void sfx_light_update(u64 module, u64 params, u64 context) {
    const auto game = reinterpret_cast<LightUpdate>(g_game_light);
    const bool preroll = context && load<u64>(context) == game_address(kPrerollContext);
    const u64 node = load<u64>(module + 8);
    if (!preroll && node) {
        const sfx_node::SmoothCall call = sfx_node::smooth_call();
        if (call.node == node) {
            g_lights_smooth.fetch_add(1, std::memory_order_relaxed);
            if (call.tick && !MUTANT(22)) {
                g_lights_counted.fetch_add(1, std::memory_order_relaxed);
                const u32 shifted = load<u32>(node + 0x7c);
                if (!MUTANT(23)) store<u32>(node + 0x7c, call.time);
                game(module, params, context);
                if (load<u32>(node + 0x7c) == call.time && !MUTANT(27)) store<u32>(node + 0x7c, shifted);
            }
            if (!MUTANT(26)) light_frame_call(game, module, params, context);
            return;
        }
    }
    if (!held() || preroll) {
        game(module, params, context);
        return;
    }
    g_lights_held.fetch_add(1, std::memory_order_relaxed);
    const u32 rate = load<u32>(module + 0x80), mode = load<u32>(module + 0x38), phase = load<u32>(module + 0x30);
    if (!MUTANT(5)) store<float>(module + 0x80, 0.0f);
    if (!MUTANT(6)) store<u32>(module + 0x38, 1u);
    game(module, params, context);
    store<u32>(module + 0x80, rate);
    if (!MUTANT(7)) store<u32>(module + 0x38, mode);
    if (!MUTANT(5)) store<u32>(module + 0x30, phase);  // the add of the age times 0 kept a -0 or an infinite age's NaN from it
}

// sub_12e5500(the effect manager's fade list +0x118, step): the GX lights of
// FFX lights that ended, each fading out over its time (+0x20) - its colours
// times the time left (+0x24) over it, the time left less the step, the
// lights let go once none is left - stepped by the manager's update with the
// effects' step times its time scale (+0x5d8; 0 when it is still, +0x5ce),
// so above 30 a fade came in whole 1/30 s steps. The smooth steps give it the
// frame's own time instead, on the game's scale: the same seconds of fade, a
// change each frame. The game's own still frame (SprjSfx+0x8c) is left to it.
DECOMP_LEAF void sfx_light_fades(u64 list, float step) {
    const auto game = reinterpret_cast<LightFadesFn>(g_game_light_fades);
    float s = step;
    if (g_smooth.load(std::memory_order_relaxed) && !MUTANT(28)) {
        const u64 sfx = load<u64>(game_address(kSprjSfx));
        if (!(sfx && load<u32>(sfx + 0x8c))) {
            const u64 manager = list - 0x118;
            g_fades_smooth.fetch_add(1, std::memory_order_relaxed);
            s = load<u8>(manager + 0x5ce) ? 0.0f : g_frame_step.load(std::memory_order_relaxed) * load<float>(manager + 0x5d8);
        }
    }
    game(list, s);
}

// sub_1314660(the wind sources' registry *0x58b0bb0, step): every registered
// wind source - the map's wind areas (WindIns, sub_210ec40) and the effects'
// wind types 0x283c / 0x283d alike, class 0x56ec140 - stepped through its
// vtable +0x58, sub_2d552d0: its last step (+0x38), its age (+0x1f4 += step)
// and its gust phase (+0x1b4 += step x its rate +0x1a8 x 10), nothing counted
// (an entry's default object, 0x56ebf70, steps nothing). The effect manager's
// update (0x26ff3ee) hands it the effects' step times its time scale (+0x5d8),
// so above 30 the gusts moved in whole 1/30 s, held between. The smooth steps
// give it the frame's own time on the same scale instead: the same seconds of
// gusts, a change each frame. Only the manager's call (its step: the whole
// step's, or none, on its scale) is taken; another caller's, and the game's own
// still frame (SprjSfx+0x8c), are the game's.
DECOMP_LEAF void sfx_wind_sources_step(u64 registry, float step) {
    const auto game = reinterpret_cast<WindStepFn>(g_game_wind_step);
    float s = step;
    if (g_smooth.load(std::memory_order_relaxed) && !MUTANT(38)) {
        const u64 sfx = load<u64>(game_address(kSprjSfx));
        const u64 manager = sfx ? load<u64>(sfx + kManagerOf) : 0;
        if (manager && !load<u32>(sfx + 0x8c)) {
            const float time_scale = load<float>(manager + 0x5d8);
            const float whole = g_step_between.load(std::memory_order_relaxed) ? 0.0f : 1.0f / 30.0f;
            if (std::bit_cast<u32>(whole * time_scale) == std::bit_cast<u32>(step)) {
                g_wind_steps_smooth.fetch_add(1, std::memory_order_relaxed);
                s = g_frame_step.load(std::memory_order_relaxed) * (MUTANT(39) ? 1.0f : time_scale);
            } else {
                g_wind_steps_other.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    game(registry, s);
}

// The wind system's receivers' update (sub_27174e0, read in Binary Ninja; the
// effect manager's update calls it at 0x26ff41a with its step). After a reset
// (sub_27136a0, sub_2717100: +0x4c = 1, +0x4d = 1, +0x50 = 60) it counts its
// calls - +0x50 down by one a call while +0x4d is set, then +0x4d cleared - and
// clears the reset's one-shot +0x4c; while either is set the grass and trees'
// sway (sub_2713870) settles (each receiver's call count +0xc30, its first
// call's damping of +0x6a0). The receivers' callbacks it hands the step to
// (+0x180, sub_132c730) are never set - their set-up (sub_132c230) zeroes them
// and nothing writes them - and the rest of the update (each receiver's
// overlapping wind sources, sub_132c750; its own time +0x58) does not read the
// step. So the count is the update's one counted part: 60 calls, 2 s at 30, and
// 1 s at 60 or 0.5 s at 120 with the frames between whole steps counting too.
// On such a frame - the gate on or off, not the game's own still frame - the
// count (+0x4d, +0x50) waits for the whole step: the rest of the update runs as
// the game's, +0x4c cleared by the first call as the game clears it.
DECOMP_LEAF void sfx_wind_receivers(u64 system, float step) {
    const auto game = reinterpret_cast<WindReceiversFn>(g_game_wind_receivers);
    if (!g_hold_warmup.load(std::memory_order_relaxed) || MUTANT(49) || !between_steps()) {
        game(system, step);
        return;
    }
    g_warmups_held.fetch_add(1, std::memory_order_relaxed);
    const u8 on = load<u8>(system + 0x4d);
    const u32 left = load<u32>(system + 0x50);
    game(system, step);
    if (MUTANT(50)) return;
    store<u8>(system + 0x4d, on);
    store<u32>(system + 0x50, left);
}

// The wobble motions. Motion classes 0x570e6d0 and 0x570e690 (whose update,
// sub_2e896f0, ends in the first's, sub_2e89d10) and 0x570e8d0 (sub_2e8e430)
// run their base motion and then count their calls - +0x6c against the
// period +0x68, or +0x7c against +0x78 - and every period turn their direction
// +0x10 by two angles from the FX RNG, scaled by a curve at the node's time;
// sub_2e896f0 first lerps the motion toward its parent's matrix by a curve
// factor, once a call. They run inside the container calls node_held still
// makes, so on the frames between whole steps they counted every frame: at
// 120 the wobble re-rolled and homed four times as often as at 30 (a census,
// 2026-10-19: 0x570e690's +0x6c changed on 725,823 of 726,170 in-between
// calls). On a held frame each runs only its base motion (sub_2e866f0 or
// sub_2e8bfd0, which moves by the step, 0 there): no count, no re-roll, no
// lerp, so they come 30 times a second at any rate, as at 30.
DECOMP_LEAF void sfx_wobble_a(u64 motion, u64 node, const float* step, u64 local) {
    if (!held() || MUTANT(10)) {
        reinterpret_cast<MotionFn>(g_game_wobble_a)(motion, node, step, local);
        return;
    }
    g_motions_held.fetch_add(1, std::memory_order_relaxed);
    if (!MUTANT(12)) game_function<MotionFn>(kWobbleABase)(motion, node, step, local);
    if (MUTANT(9)) store<u32>(motion + 0x6c, load<u32>(motion + 0x6c) + 1);  // counted anyway
}

DECOMP_LEAF void sfx_wobble_b(u64 motion, u64 node, const float* step, u64 local) {
    if (!held() || MUTANT(44)) {
        reinterpret_cast<MotionFn>(g_game_wobble_b)(motion, node, step, local);
        return;
    }
    g_motions_held.fetch_add(1, std::memory_order_relaxed);
    game_function<MotionFn>(kWobbleBBase)(motion, node, step, local);
    if (MUTANT(45)) store<u32>(motion + 0x7c, load<u32>(motion + 0x7c) + 1);  // counted anyway
}

// With the smooth steps, a smooth instance's container call (sfx_node::smooth_call
// names its node) between whole steps runs sub_2e896f0 itself: its follow takes
// a share (a curve at the node's time, or +0xb0) of the parent's move since the
// last call (the parent's rows it keeps at +0x70..+0x9f) - a share of each
// frame's move adds up to that share of the whole step's - and its base motion
// moves by the frame's step; its count and re-roll wait for the whole step, the
// period +0x68 held at its largest for the call and the count +0x6c put back
// (the wobble's own update, sub_2e89d10, is ours and holds them anyway).
DECOMP_LEAF void sfx_wobble_follow(u64 motion, u64 node, const float* step, u64 local) {
    const auto game = reinterpret_cast<MotionFn>(g_game_wobble_follow);
    if (!held() || MUTANT(11)) {
        game(motion, node, step, local);
        return;
    }
    if (sfx_node::smooth_call().node == node && !MUTANT(42)) {
        g_motions_smooth.fetch_add(1, std::memory_order_relaxed);
        const u32 count = load<u32>(motion + 0x6c), period = load<u32>(motion + 0x68);
        store<u32>(motion + 0x68, 0xffffffffu);
        game(motion, node, step, local);
        store<u32>(motion + 0x68, period);
        if (!MUTANT(43)) store<u32>(motion + 0x6c, count);
        return;
    }
    g_motions_held.fetch_add(1, std::memory_order_relaxed);
    game_function<MotionFn>(kWobbleABase)(motion, node, step, local);
}

// sub_2cbcf50(record, item): a GPU particle system's record takes an item of
// its node's container call - its matrix rows, the previous time +0x24 = the
// time +0x28, the time += the item's step +0x60 - and hands its type (+8) the
// record +0x10 for the type's update (vtable +0x10), which reads the elapsed
// time, +0x28 - +0x24. Type 0x2713 (vtable 0x56eb0d0) reads its definition's
// five keyframe curves at that elapsed time - the last item's step, not an
// age - into its two sizes +0x1c / +0x20 and its three colours +0x50 / +0x60 /
// +0x70, and its draw (vtable +0x18) reads the same curves at it again. At 30
// that is 1/30 s on every frame; on a frame between whole effect steps its
// items have a step of 0, so the curves were read at 0 there and at the whole
// step on the others, and wherever a curve is not flat over that range the
// values took turns every frame - with the gate on or off, at 30 or 60 steps
// a second. On a frame between whole steps the record of a 0x2713 whose
// elapsed time is not 0 keeps its previous time and its type the five values:
// the game's call runs as it is - the matrix, the bounds, every other field -
// and both are put back, so the type's update and its draw read the last whole
// step's elapsed time and values, as on every frame at 30.
// The ribbon, FFX-format GX type 0x28 (0x56e3620; update sub_2ec1e00, read in
// Binary Ninja), under the smooth steps. Its type keeps a history of points:
// times +0x88, places +0x90 / +0x98 / +0xa0, lengths +0xa8 and two optional
// arrays, +0x7c of them kept, room for +0x78. Each update, at its record's time
// (+0x28) alone - it never reads the elapsed time:
//   - points whose time plus the definition's life (+0x8c) has passed go, and
//     the oldest go to make room for the update's new ones;
//   - the points from the last kept one to the emitter (the record's matrix)
//     are made again: a spline between the last two inserted places and a line
//     from the last to the emitter, plus an extrapolated one - drawn, not kept
//     (+0x80 of them);
//   - when more than the definition's interval (+0x88) has passed since the
//     last insert (+0x1c), the spline's points are kept, the place history
//     shifts, and one draw of the FX RNG sets the next point's scalar (+0x4c):
//     the insert, counted - at most one an update;
//   - with smoothing (+0x98 above 0), every kept point moves that share toward
//     the next one - counted too, once an update.
// So at 30 a ribbon inserts at most 30 times a second, and its spacing, its
// draws and its smoothing follow its updates; with the frame's own time on
// every frame it would insert and smooth up to once a frame (four times at
// 120, overflowing room the set-up sized for 60 inserts a second). The smooth
// steps' frame domain therefore runs it:
//   - between whole steps, the game's call with a copy of its definition in
//     the type's +8 whose interval is +inf and smoothing 0, put back after: no
//     insert, draw or smoothing, and the drawn points made again to the
//     emitter at the frame's own time - the ribbon's head moves every frame;
//   - on a frame with a whole step, the game's call as it is, at the whole
//     step's time and place: a smooth instance's item carries the frame's
//     step, which ends `into` (the frame's carry past the step, scaled) after
//     it, so the call gets the step less that, and the emitter's matrix that
//     far between the last frame's (the record's) and this one's - the
//     inserts, draws and smoothing of 30, at 30's times - then a second, as
//     between whole steps, by `into` with this frame's matrix, so the drawn
//     points end at the emitter at the frame's own time; the record's times
//     are left as the game's one call leaves them.
// A held instance's item between whole steps carries no step, and on a frame
// with one the whole step's (as does a frame of 1/30 s itself): the game's.
// The record update runs after the instance's (the render context queues its
// items), so what the frame is comes from the frame's own numbers.
namespace {

inline void ribbon_held_call(GxRecordFn game, u64 record, u64 item, u64 type) {
    const u64 def = load<u64>(type + 8);
    alignas(16) u8 copy[0x130];
    std::memcpy(copy, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(def)), kRibbonDefinition);
    const u64 at = reinterpret_cast<u64>(copy);
    if (!MUTANT(29)) store<u32>(at + 0x88, 0x7f800000u);  // +inf: no insert
    if (!MUTANT(29) && !MUTANT(30)) store<u32>(at + 0x98, 0u);  // no smoothing
    store<u64>(type + 8, at);
    game(record, item);
    if (load<u64>(type + 8) == at && !MUTANT(34)) store<u64>(type + 8, def);
}

// Whether a ribbon's record update is a smooth instance's, and how: 0 the game's, 1 between whole steps, 2 a frame with
// a whole step whose item ends `into` past it.
inline int ribbon_mode(u64 record, u64 item, u64 type, float& into) {
    if (!g_smooth.load(std::memory_order_relaxed)) return 0;
    if (!ribbon_marked(type) && !MUTANT(37)) {  // not a smooth instance's this frame: a pre-roll's, a held one's
        g_ribbons_unmarked.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const u64 sfx = load<u64>(game_address(kSprjSfx));
    if (sfx && load<u32>(sfx + 0x8c)) return 0;  // the game's own still frame
    const float step = load<float>(item + 0x60);
    if (!(step > 0.0f)) return 0;  // a held instance between whole steps, or the manager still
    if (g_step_between.load(std::memory_order_relaxed)) return 1;
    // A frame with a whole step: a smooth instance's step is the frame's (the node update's (frame x time scale) x
    // scale); a held one's the whole step's. A frame of 1/30 s, where the two are one, is the game's.
    const float frame = g_frame_step.load(std::memory_order_relaxed);
    if (!(frame < 1.0f / 30.0f)) return 0;
    const u64 node = load<u64>(record + 0x10);
    const u64 instance = node ? load<u64>(node + 0x58) : 0;
    const u64 manager = sfx ? load<u64>(sfx + kManagerOf) : 0;
    if (!instance || !manager || load<u8>(manager + 0x5ce)) return 0;
    const float time_scale = load<float>(manager + 0x5d8), scale = load<float>(instance + 8);
    if (std::bit_cast<u32>(frame * time_scale * scale) != std::bit_cast<u32>(step)) return 0;
    into = g_frame_into.load(std::memory_order_relaxed) * time_scale * scale;
    return 2;
}

}  // namespace

DECOMP_LEAF void sfx_gx_record_update(u64 record, u64 item) {
    const auto game = reinterpret_cast<GxRecordFn>(g_game_gx_record);
    const u64 type = load<u64>(record + 8);
    if (type && load<u64>(type) == game_address(kGxTypeRibbon)) {
        float into = 0.0f;
        const int mode = ribbon_mode(record, item, type, into);
        if (mode == 1) {
            g_ribbons_between.fetch_add(1, std::memory_order_relaxed);
            ribbon_held_call(game, record, item, type);
            return;
        }
        if (mode == 2) {
            g_ribbons_whole.fetch_add(1, std::memory_order_relaxed);
            const float step = load<float>(item + 0x60);
            if (!(into > 0.0f) || !(into < step) || MUTANT(31)) {
                game(record, item);  // the frame ends on the whole step (or a mutant's: the inserts at the frame's time)
                return;
            }
            const u32 before = load<u32>(record + 0x28);
            // the whole step's time, and its place: the matrix rows (the place in their w lanes) as far between the last
            // frame's, which the record holds, and this frame's as the step is into the frame
            const float share = (step - into) / step;
            float rows[12];
            std::memcpy(rows, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(item)), sizeof rows);
            if (!MUTANT(35))
                for (int i = 0; i < 12; ++i) {
                    const float last = load<float>(record + 0x50 + 4 * static_cast<u64>(i));
                    store<float>(item + 4 * static_cast<u64>(i), last + (rows[i] - last) * share);
                }
            store<float>(item + 0x60, step - into);
            game(record, item);
            std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(item)), rows, sizeof rows);
            if (!MUTANT(32)) {
                store<float>(item + 0x60, into);  // and on to the frame's time and place: the drawn points, nothing counted
                ribbon_held_call(game, record, item, type);
            }
            if (!MUTANT(33)) {
                store<u32>(record + 0x24, before);
                store<float>(record + 0x28, add_ss(std::bit_cast<float>(before), step));
            }
            store<float>(item + 0x60, step);
            return;
        }
    }
    if (type && load<u64>(type) == game_address(kGxType2713) && g_smooth.load(std::memory_order_relaxed) && ribbon_marked(type)) {
        // A smooth instance's GPU particle type 0x2713 (marked by its instance's update this frame): its update reads its
        // five curves at the record's elapsed time, as does its draw - the whole step's at 30. The game's call with the
        // whole step (1/30 on the manager's and the instance's scales) for the type's update, then the record's times as
        // the frame's step leaves them, the previous one where the draw reads the same elapsed time again.
        const u64 sfx = load<u64>(game_address(kSprjSfx));
        const u64 manager = sfx ? load<u64>(sfx + kManagerOf) : 0;
        const u64 node = load<u64>(record + 0x10);
        const u64 instance = node ? load<u64>(node + 0x58) : 0;
        const float step = load<float>(item + 0x60);
        if (manager && instance && !load<u32>(sfx + 0x8c) && step > 0.0f) {
            g_records_whole.fetch_add(1, std::memory_order_relaxed);
            const float whole = 1.0f / 30.0f * load<float>(manager + 0x5d8) * load<float>(instance + 8);
            const float before = load<float>(record + 0x28);
            if (!MUTANT(40)) store<float>(item + 0x60, whole);
            game(record, item);
            store<float>(item + 0x60, step);
            if (MUTANT(40)) return;
            const float elapsed = sub_ss(load<float>(record + 0x28), before), cur = add_ss(before, step);
            float prev = sub_ss(cur, elapsed);
            for (int k = 0; k < 8 && sub_ss(cur, prev) != elapsed; ++k) prev = ulp_toward(prev, sub_ss(cur, prev) > elapsed);
            store<float>(record + 0x24, MUTANT(41) ? before : prev);
            store<float>(record + 0x28, cur);
            return;
        }
    }
    const bool is_2713 = type && (load<u64>(type) == game_address(kGxType2713) || MUTANT(15));
    if (!is_2713 || (load<float>(item + 0x60) != 0.0f && !MUTANT(16)) || (load<u32>(record + 0x24) == load<u32>(record + 0x28) && !MUTANT(17)) ||
        !between_steps()) {
        game(record, item);
        return;
    }
    g_records_held.fetch_add(1, std::memory_order_relaxed);
    const u32 previous = load<u32>(record + 0x24);
    u64 sizes = load<u64>(type + 0x1c);
    u8 colours[0x30];
    std::memcpy(colours, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(type + 0x50)), sizeof colours);
    game(record, item);
    if (!MUTANT(13)) store<u32>(record + 0x24, previous);
    if (!MUTANT(14)) {
        store<u64>(type + 0x1c, sizes);
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(type + 0x50)), colours, sizeof colours);
    }
}

void decomp_sfx_gate_frame(bool between) {
    g_between.store(between ? 1u : 0u, std::memory_order_relaxed);
    if (!between) g_step_serial.fetch_add(1, std::memory_order_relaxed);
    if (between) g_frames_between.fetch_add(1, std::memory_order_relaxed);
}

void decomp_sfx_step_frame(bool between) { g_step_between.store(between ? 1u : 0u, std::memory_order_relaxed); }

bool decomp_sfx_between_steps() { return g_step_between.load(std::memory_order_relaxed) != 0; }

void decomp_sfx_frame_time(float frame, float from, float into) {
    // a new frame for the ribbons' marks: the serial's 16 bits wrapping, the table wiped (between frames: no effect
    // update runs now) and 0 skipped
    u32 serial = g_frame_serial.load(std::memory_order_relaxed) + 1;
    if (!(serial & 0xffffu)) {
        for (auto& m : g_ribbon_marks) m.store(0, std::memory_order_relaxed);
        ++serial;
    }
    g_frame_serial.store(serial, std::memory_order_relaxed);
    g_frame_step.store(frame, std::memory_order_relaxed);
    g_frame_from.store(from, std::memory_order_relaxed);
    g_frame_into.store(into, std::memory_order_relaxed);
}

void decomp_sfx_smooth(bool on) {
    if (on) conv_build();
    g_smooth.store(on ? 1u : 0u, std::memory_order_relaxed);
}

bool decomp_sfx_gate_placed() { return decomp_placed("sub_2c4ff30") == 1 && decomp_placed("sub_131e660") == 1; }

namespace {

// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; sub rsp, imm8 / imm32.
constexpr std::uint8_t kInstanceEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x18};
constexpr std::uint8_t kLightEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x58, 0x01, 0x00, 0x00};
// The three motions' (sub rsp, 0x78 for the two counting ones; 0x58 for the follow).
constexpr std::uint8_t kWobbleEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x78};
constexpr std::uint8_t kFollowEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x58};
// The record update's (sub rsp, 0x68).
constexpr std::uint8_t kGxRecordEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x68};
// The light fades' (sub rsp, 0x18).
constexpr std::uint8_t kLightFadesEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x18};
// The wind sources' step: push rbp; mov rbp, rsp; push r14, rbx; sub rsp, 0x10; vmovss [rbp-0x14], xmm0.
constexpr std::uint8_t kWindStepEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x83, 0xec, 0x10, 0xc5, 0xfa, 0x11, 0x45, 0xec};
// The wind receivers' update: push rbp; mov rbp, rsp; push r15..rbx; sub rsp, 0x78; vmovss [rbp-0x8c], xmm0.
constexpr std::uint8_t kWindReceiversEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53,
                                                0x48, 0x83, 0xec, 0x78, 0xc5, 0xfa, 0x11, 0x85, 0x74, 0xff, 0xff, 0xff};

void report() {
    if (!decomp_sfx_gate_placed()) return;
    host_log("decomp: the effects' frames between whole steps: %llu frames; %llu instance updates, %llu light updates and %llu wobble "
             "motion updates held, %llu GPU type 0x2713 items kept at their whole step's time; %llu calls on the game's own still frame left to it",
             static_cast<ull>(g_frames_between.load()), static_cast<ull>(g_instances_held.load()), static_cast<ull>(g_lights_held.load()),
             static_cast<ull>(g_motions_held.load()), static_cast<ull>(g_records_held.load()), static_cast<ull>(g_still_frames.load()));
    host_log("decomp:   the wind system's warm-up count held for the whole step on %llu frames (BBHOST_SFX_WIND_WARMUP_HOLD=%u)",
             static_cast<ull>(g_warmups_held.load()), g_hold_warmup.load());
    if (!g_smooth.load()) return;
    host_log("decomp: the smooth effect steps: %llu instance updates took them, %llu stayed on the held frames for a class off the converted "
             "list (%zu classes on it); their classes looked at %llu times, %llu more taken from the frame's whole step",
             static_cast<ull>(g_smooth_updates.load()), static_cast<ull>(g_smooth_kept.load()), g_converted_count,
             static_cast<ull>(g_class_made.load()), static_cast<ull>(g_class_reused.load()));
    host_log("decomp:   FFX light updates by the frame's time: %llu, %llu of them after the whole step's own; light fades by the frame's time: %llu",
             static_cast<ull>(g_lights_smooth.load()), static_cast<ull>(g_lights_counted.load()), static_cast<ull>(g_fades_smooth.load()));
    host_log("decomp:   ribbon updates: %llu between whole steps (the head by the frame's time, nothing inserted), %llu on frames with one, "
             "%llu not a smooth instance's (a pre-roll's or a held one's: the game's)",
             static_cast<ull>(g_ribbons_between.load()), static_cast<ull>(g_ribbons_whole.load()), static_cast<ull>(g_ribbons_unmarked.load()));
    host_log("decomp:   wind source steps by the frame's time: %llu (%llu another caller's, the game's); GPU particle type 0x2713 records at "
             "the whole step's elapsed time: %llu; following wobble motions between whole steps by the frame's time: %llu",
             static_cast<ull>(g_wind_steps_smooth.load()), static_cast<ull>(g_wind_steps_other.load()), static_cast<ull>(g_records_whole.load()),
             static_cast<ull>(g_motions_smooth.load()));
    for (std::size_t i = 0; i < kKeptSlots; ++i) {
        const u64 vt = g_kept_class[i].load();
        if (!vt) break;
        host_log("decomp:   kept by class 0x%llx: %llu instance updates", static_cast<ull>(vt - g_decomp_slide + 0x400000),
                 static_cast<ull>(g_kept_count[i].load()));
    }
    if (const u64 more = g_kept_overflow.load()) host_log("decomp:   kept by classes past the table's %zu: %llu", kKeptSlots, static_cast<ull>(more));
}

}  // namespace

void decomp_sfx_step_gate_add() {
    const char* area = "sfx";
    if (const char* e = std::getenv("BBHOST_SFX_WIND_WARMUP_HOLD"); e && (e[0] == '0' || e[0] == '1')) g_hold_warmup.store(e[0] == '1' ? 1u : 0u);
    DecompFunction instance{"sub_2c4ff30", area, 0x2c4ff30, kInstanceEntry, sizeof(kInstanceEntry),
                            reinterpret_cast<void*>(&sfx_instance_update), DecompKind::Leaf, &g_game_instance};
    instance.report = &report;
    decomp_add(instance);
    decomp_add({"sub_131e660", area, 0x131e660, kLightEntry, sizeof(kLightEntry), reinterpret_cast<void*>(&sfx_light_update), DecompKind::Leaf,
                &g_game_light});
    decomp_add({"sub_2e89d10", area, kWobbleA, kWobbleEntry, sizeof(kWobbleEntry), reinterpret_cast<void*>(&sfx_wobble_a), DecompKind::Leaf,
                &g_game_wobble_a});
    decomp_add({"sub_2e8e430", area, kWobbleB, kWobbleEntry, sizeof(kWobbleEntry), reinterpret_cast<void*>(&sfx_wobble_b), DecompKind::Leaf,
                &g_game_wobble_b});
    decomp_add({"sub_2e896f0", area, kWobbleFollow, kFollowEntry, sizeof(kFollowEntry), reinterpret_cast<void*>(&sfx_wobble_follow),
                DecompKind::Leaf, &g_game_wobble_follow});
    decomp_add({"sub_2cbcf50", area, kGxRecordUpdate, kGxRecordEntry, sizeof(kGxRecordEntry), reinterpret_cast<void*>(&sfx_gx_record_update),
                DecompKind::Leaf, &g_game_gx_record});
    decomp_add({"sub_12e5500", area, kLightFades, kLightFadesEntry, sizeof(kLightFadesEntry), reinterpret_cast<void*>(&sfx_light_fades),
                DecompKind::Leaf, &g_game_light_fades});
    decomp_add({"sub_1314660", area, kWindSourcesStep, kWindStepEntry, sizeof(kWindStepEntry), reinterpret_cast<void*>(&sfx_wind_sources_step),
                DecompKind::Leaf, &g_game_wind_step});
    decomp_add({"sub_27174e0", area, kWindReceivers, kWindReceiversEntry, sizeof(kWindReceiversEntry),
                reinterpret_cast<void*>(&sfx_wind_receivers), DecompKind::Leaf, &g_game_wind_receivers});
}
