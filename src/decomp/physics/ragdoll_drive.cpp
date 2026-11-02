// The ragdoll keyframe controller's drive, by the frame's step (docs/decomp.md):
// sub_ae4600, the per-frame virtual (vtable slot 5) of the 0xe0-byte rigid-body
// controller that keeps a ragdoll's bodies on its animation. It hands Havok's
// keyframe hierarchy step (0xae9b90, hknpRagdollKeyFrameHierarchyUtility's)
// the bodies (the struct at +0x50: the ragdoll, its bodies' map, the step at
// +0x60, the bones' weights at +0x70), the keyframe (the struct at +0x80: the
// pose at +0xb0), the controller's palette (+0x40: its ControlData -
// hierarchyGain 0.17, velocityGain 0.6, positionGain 0.05 ...) and each body's
// palette index (+0xd0). The step turns the pose's change into the velocities
// the bodies are given; the physics world then moves them by its own step,
// the frame's.
//
// The step is the game's 1/30 for good: the character's ragdoll (the 0x1f0-byte
// object sub_1c0c2b0 builds, from sub_19124e0 or sub_1c14ba0) writes it into its
// controller once, at 0x1c0ca97, and drives the controller from its update
// (0x1c0e9d6, 0x1c113c8) every frame. Above 30 the frame-rate code writes the
// rate chosen at start into that immediate, so a frame of any other length -
// a rate the PC does not hold, a variable refresh rate, a dip - asks the
// bodies to cover a pose's change in the wrong time: at a 1/960 step with
// frames of 1/150 the drive's gains act 6.4 times. With the frame clock on
// (BBHOST_FRAME_CLOCK=1) the frame-rate code leaves that 1/30 as the game has
// it, and ours hands the keyframe step the frame clock's step
// (engine/frame_clock.h) for a controller whose step is still that 1/30, for
// the call only: the field keeps the game's value, and at 30 the clock's step
// is the game's 1/30 - the game's own call. A step anyone else wrote is that
// writer's and goes on as it is: Havok Behavior's rigid-body ragdoll
// controller (an hkbRagdollController, vtable 0x5697820, made by its physics
// interface's factory, sub_e79880) holds one of these (vtable 0x5697860, whose
// slot 5 is this drive too) and writes its caller's step into its +0x60
// before every drive (its own slot 5, sub_e7cfb0). Of the class's methods only
// this one reads +0x60.
//
// The three tests are the game's (each "not above 0", so a NaN goes on), and
// the two stores before the call: +0xb0 from +0x20 (the pose) and +0x70 from
// +0x30 (the bones' weights), from the controller's base (sub_aea9d0). The
// keyframe step reads the step once (0xaea08a) and writes nothing of the
// bodies' struct; nothing reads the drive's result.
//
// A leaf: entered straight from the game's code, it touches only the game's
// memory and its own globals. It is on the decomp list only with the frame
// clock on; BBHOST_RAGDOLL_STEP=0 leaves it off too (the frame-rate code then
// writes the rate's step into the constructor, as with the clock off).
// BBHOST_DECOMP_COMPARE=1 runs the game's version with the step ours hands,
// and checks ours' reading of the tests and the stores against what it does.
#include "decomp/decomp.h"
#include "decomp/guest.h"
#include "engine/frame_clock.h"

#include "core/thunk.h"
#include "log.h"

#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>

using namespace decomp;

#ifdef DECOMP_PHYSICS_RAGDOLL_DRIVE_MUTANTS
extern int g_ragdoll_drive_mutant;
#define MUTANT(n) (g_ragdoll_drive_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kDrive = 0xae4600;
constexpr u64 kKeyFrameStep = 0xae9b90;
constexpr u64 kBehaviorController = 0x5697860;  // Havok Behavior's rigid-body controller, which hands its own step
// The controller's fields.
constexpr u64 kWeight = 0x18, kPose = 0x20, kBoneWeights = 0x30, kPalette = 0x40, kBodies = 0x50, kStep = 0x60,
              kBodiesWeights = 0x70, kKeyFrame = 0x80, kKeyFramePose = 0xb0, kPaletteIndices = 0xd0;
constexpr u64 kRagdollBodies = 0x18;  // the ragdoll's (at +0x50) bodies: none, no drive

using KeyFrameStep = GUEST_ABI u64 (*)(u64 key_frame, u64 bodies, u64 palette, u64 indices);

void* g_game_drive = nullptr;
// Drives handed the frame's step, the game's 1/30 at a frame of 1/30, and a
// step someone else wrote; and the drives of Havok Behavior's controller.
std::atomic<u64> g_framed{0}, g_thirtieth{0}, g_handed{0}, g_behavior{0};

u32 bits(float f) { return std::bit_cast<u32>(f); }

// The step the keyframe step is handed for a controller whose field holds
// `step`: the frame clock's for the constructor's 1/30, else the field.
float drive_step(float step) {
    if (bits(step) != frame_clock::kThirtiethBits && !MUTANT(1)) return step;
    return MUTANT(2) ? step * frame_clock::frames30() : frame_clock::dt();
}

// The game's tests: whether the drive goes on to the keyframe step.
bool drives(u64 self) {
    if (MUTANT(3) ? load<float>(self + kWeight) < 0.0f : load<float>(self + kWeight) <= 0.0f) return false;
    if (load<u64>(load<u64>(self + kBodies) + kRagdollBodies) == 0 && !MUTANT(7)) return false;
    if (MUTANT(4) ? !(load<float>(self + kStep) > 0.0f) : load<float>(self + kStep) <= 0.0f) return false;
    return true;
}

}  // namespace

DECOMP_LEAF void ragdoll_drive(u64 self) {
    if (!drives(self)) return;
    store<u64>(self + kKeyFramePose, load<u64>(self + (MUTANT(5) ? kBoneWeights : kPose)));
    store<u64>(self + kBodiesWeights, load<u64>(self + kBoneWeights));
    const auto key_frame_step = game_function<KeyFrameStep>(kKeyFrameStep);
    const u64 palette = load<u64>(self + kPalette), indices = load<u64>(self + kPaletteIndices);
    const float step = load<float>(self + kStep), frame = drive_step(step);
    if (load<u64>(self) == game_address(kBehaviorController)) g_behavior.fetch_add(1, std::memory_order_relaxed);
    if (bits(frame) == bits(step)) {
        (bits(step) == frame_clock::kThirtiethBits ? g_thirtieth : g_handed).fetch_add(1, std::memory_order_relaxed);
        key_frame_step(self + kKeyFrame, self + kBodies, palette, indices);
        return;
    }
    g_framed.fetch_add(1, std::memory_order_relaxed);
    store<float>(self + kStep, frame);
    key_frame_step(self + kKeyFrame, self + kBodies, palette, indices);
    if (!MUTANT(6)) store<float>(self + kStep, step);
}

namespace {

// BBHOST_DECOMP_COMPARE=1: the game's drive, handed the step ours hands, does
// the work; ours' reading of it has to agree - whether it goes on to the
// keyframe step (its two stores show it: they are cleared first, and put back
// when it stops short) and what those stores write - and the field is the
// game's again after. `game` runs the game's drive on the controller.
template <class Game>
bool compare_one(u64 self, Game game) {
    const bool ours_drives = drives(self);
    const u64 pose = load<u64>(self + kPose), weights = load<u64>(self + kBoneWeights);
    const u64 had_pose = load<u64>(self + kKeyFramePose), had_weights = load<u64>(self + kBodiesWeights);
    const float step = load<float>(self + kStep), frame = ours_drives ? drive_step(step) : step;
    store<u64>(self + kKeyFramePose, ~pose);
    store<u64>(self + kBodiesWeights, ~weights);
    if (bits(frame) != bits(step)) store<float>(self + kStep, frame);
    game(self);
    if (bits(frame) != bits(step)) store<float>(self + kStep, step);
    const bool game_drove = load<u64>(self + kKeyFramePose) == pose && load<u64>(self + kBodiesWeights) == weights;
    const bool game_stopped = load<u64>(self + kKeyFramePose) == ~pose && load<u64>(self + kBodiesWeights) == ~weights;
    if (game_stopped) {
        store<u64>(self + kKeyFramePose, had_pose);
        store<u64>(self + kBodiesWeights, had_weights);
    }
    return ours_drives ? game_drove : game_stopped;
}

DecompCompare g_counts;
std::atomic<int> g_compare_logs{0};

GUEST_ABI void compare_drive(u64 self) {
    const float weight = load<float>(self + kWeight), step = load<float>(self + kStep);
    const bool same = compare_one(self, [](u64 s) { hle_call_guest6(g_game_drive, static_cast<std::int64_t>(s), 0, 0, 0, 0, 0); });
    g_counts.calls.fetch_add(1, std::memory_order_relaxed);
    if (same) return;
    g_counts.differ.fetch_add(1, std::memory_order_relaxed);
    if (g_compare_logs.fetch_add(1) < 16) {
        host_log("decomp: the ragdoll drive of 0x%llx differs from the game's: ours %s where the game's does not (weight %g, step %g)",
                 static_cast<ull>(self), drives(self) ? "drives" : "stops", static_cast<double>(weight), static_cast<double>(step));
    }
}

void report() {
    const int placed = decomp_placed("sub_ae4600");
    if (placed == 0) return;
    host_log("decomp: the ragdoll drive: %llu drives by the frame's step, %llu at a frame of 1/30, %llu by a step another wrote; "
             "%llu of them Havok Behavior's controller's",
             static_cast<ull>(g_framed.load()), static_cast<ull>(g_thirtieth.load()), static_cast<ull>(g_handed.load()),
             static_cast<ull>(g_behavior.load()));
}

// vxorps xmm0, xmm0, xmm0; vucomiss xmm0, [rdi+0x18].
constexpr u8 kEntry[] = {0xc5, 0xf8, 0x57, 0xc0, 0xc5, 0xf8, 0x2e, 0x47, 0x18};
// The whole function ours mirrors, to its ret (0xae4600..0xae4652).
constexpr u8 kBody[] = {
    0xc5, 0xf8, 0x57, 0xc0, 0xc5, 0xf8, 0x2e, 0x47, 0x18, 0x73, 0x47, 0x48, 0x8b, 0x47, 0x50, 0x48, 0x83, 0x78, 0x18, 0x00, 0x74,
    0x3c, 0xc5, 0xf8, 0x57, 0xc0, 0xc5, 0xf8, 0x2e, 0x47, 0x60, 0x73, 0x31, 0x48, 0x8b, 0x4f, 0x20, 0x48, 0x89, 0xf8, 0x48, 0x83,
    0xe8, 0x80, 0x48, 0x89, 0x8f, 0xb0, 0x00, 0x00, 0x00, 0x48, 0x8b, 0x4f, 0x30, 0x48, 0x89, 0x4f, 0x70, 0x48, 0x8b, 0x57, 0x40,
    0x48, 0x8b, 0x8f, 0xd0, 0x00, 0x00, 0x00, 0x48, 0x8d, 0x77, 0x50, 0x48, 0x89, 0xc7, 0xe9, 0x3e, 0x55, 0x00, 0x00, 0xc3,
};
static_assert(sizeof kBody == 0x53, "0xae4600 to its ret");

}  // namespace

// The body check, for decomp_add and the test.
bool ragdoll_drive_body_ok(const std::uint8_t* entry) {
    for (std::size_t i = 0; i < sizeof kBody; ++i)
        if (entry[i] != kBody[i]) return false;
    return true;
}

void decomp_physics_ragdoll_drive_add() {
    // Only with the frame clock on (engine/frame_rate.cpp's kHandedOver keeps
    // the constructor's 1/30 for it then), and not with BBHOST_RAGDOLL_STEP=0.
    const char* e = std::getenv("BBHOST_RAGDOLL_STEP");
    if (!frame_clock::on() || (e && e[0] == '0')) return;
    DecompFunction fn{"sub_ae4600", "physics", kDrive, kEntry, sizeof(kEntry), reinterpret_cast<void*>(&ragdoll_drive), DecompKind::Leaf,
                      &g_game_drive};
    fn.compare = reinterpret_cast<void*>(&compare_drive);
    fn.counts = &g_counts;
    fn.report = &report;
    fn.body_ok = &ragdoll_drive_body_ok;
    decomp_add(fn);
}
