#include "host/reflex.h"
#include "guest_abi.h"
#include "engine/frame_rate.h"
#include "engine/frame_rate_step.h"
#include "engine/frame_tick.h"
#include "engine/np_test.h"
#include "engine/event_flags.h"
#include "engine/player_data.h"
#include "engine/lua_events.h"
#include "engine/world_chr.h"
#include "engine/params.h"
#include "engine/sprj_flipper.h"
#include "engine/loading.h"
#include "engine/change_appearance.h"
#include "engine/rebirth.h"

#include "core/elf.h"
#include "core/memory.h"
#include "core/host_clock.h"
#include "core/portable.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "hle/hle.h"
#include "hle/modules.h"
#include "host/frame_stats.h"
#include "host/plugins.h"
#include "host/options.h"
#include "host/settings.h"
#include "host/window.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vector>

namespace {

std::atomic<float> g_time_scale{1.0f};

constexpr std::uint64_t kFrameManager = 0x2434770;
constexpr std::uint8_t kFrameManagerPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                                  0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x38};
constexpr std::uint32_t kModeSixty = 2;

// The game's pace: 30 (its own), a fixed rate (60, 90, 120: the game's mode 2
// with a 1/fps target), or uncapped (the same with the target the display's
// refresh rate or 1/240 s). Kyo's FPS++ lists for 1.09 (60 and 90 FPS++ 1.0,
// and his 1.3 lists with yogesh1239's high-FPS fixes) are the source of most
// of what follows; every value and code site is checked against the eboot's
// bytes before anything is written, and each is done natively - values for
// the rate, loads and in-place rewrites, no new code in the image.
//
// The game's task step is a fixed 1/30 a frame (sub_2418d20's step-info,
// 0x2418e36); the 60 FPS++ cave (0x2418e3d, sixty_fps_sites.inc) replaces it
// with the measured frame time clamped to [target, 1/30] and keeps that in a
// slot (0x575a908) that the steps the game builds for itself read instead of
// their 1/30 (the list's own loads, and kFrameTimeSites: the physics and AI
// worlds, the ragdoll/clip drive, grass). What is not a step is converted:
//
// The game's fixed steps as values: every site the "60 FPS++" list gives a
// float constant, as the value it had and the value it gets at 60. 83 are
// 1/30 s steps made 1/60, 30 are 30 a second made 60; the rest are per-frame
// weights and decays retuned to the shorter frame (0.1 -> 0.05, 0.95 ->
// 0.9747...). Some are immediates in instructions, some constants in .rodata;
// each is checked for its old value before anything is written, and one
// mismatch leaves the game at 30. The list's code rewrites are kSixtySites
// below. Above 60 the ones in kRated follow the rate - the .rodata ones written
// from each frame's time - and the rest keep their 60 value, as Kyo's 90 and
// 120 lists hold them; that rule was his ("the known-good 120 FPS++ block
// holds 1/60 at these same addresses while running correctly at ~120 fps"),
// and the physics world's step (kFrameTimeSites) was one it got wrong, so a
// site leaves it only once its system is read.
//
// Not every 1/30 in the game is a step. A duration the data gives in frames
// (an event's "wait N frames", a light's fade frames, param fields) becomes
// seconds as `vcvtsi2ss; vmulss [1/30]`, and what holds it counts down by
// the frame's real seconds (the event wait: sub_1bc8630 makes it, sub_1bb5830
// subtracts the step) - so 1/30 is right at any frame rate. An earlier
// version of the list pointed 22 of those loads at the measured frame time
// and set 15 of their constants to 1/60 (halving the durations at 60), and
// two more conversions (0x4caf1c0, 0x4d27284) to 1/15 (doubling them): one
// "wait N frames" command ran four times as long by one path as by the
// other. Kyo's later list leaves them all at 1/30, and so does this table -
// with three more his 1.3 list (from yogesh1239's timing measurements) puts
// back: 0x4c88688 (a count of 1/30 s frames in sub_11ca140's rate sums) and
// the two shortest-duration floors max(t, 1/30) at 0x12f06f1 and 0x12f6992,
// which at 1/60 let the durations they guard run at half length.
// Three per-frame amounts the lists leave at 1/30 are made per-frame here
// instead: the debug camera's step (0x1972a80), the target bank's refresh
// accumulator (0x4d2577c, read through the frame's time below anyway), and
// the follow camera's chase-rate ease (0x4d25e94: 1 - (1 - 1/30)^(30 dt)).
// The physics world's expected minimum step (hkpWorldCinfo, 0x0f383ca) is a
// tolerance hint at 1/fps; 0x4d29170's step is read by one load the 60 FPS++
// sites point at the frame's time.
// Not here: the effects' rate. The effect manager's constructor (sub_26fb190)
// stores 30.0 at its +0 (the immediate at 0x26fb39f) and the GPU particle
// update multiplies elapsed seconds by 0x4cad7fc (30.0): both turn effect
// data, written in 30 fps frames, into seconds - an emitter's interval is its
// frames / 30 counted down by elapsed time - so they are the data's rate, not
// the game's. The FPS++ lists set both to 60, which halved every frame-given
// interval and lifetime of the sfx emitters and ran GPU particles at twice
// their speed; Dark Souls III on PC, at 60, keeps the same constructor's 30.
// The FX particle system steps by the frame's time and reads neither.
struct Constant {
    std::uint32_t at;
    std::uint32_t was, now;  // the eboot's value, and the one at 60
};
constexpr Constant kConstants[] = {
    {0x0a3fc3f, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x0f383ca, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x12b5458, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x12b55b6, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x13734f3, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x13ef51c, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x13ef6ec, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x16efe5c, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x183a4c3, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a4cd, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a4d7, 0x3e99999a, 0x3d4ccccd},  // 0.3 -> 0.05
    {0x183a4e1, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a4f5, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a4ff, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a509, 0x3cf5c28f, 0x3c75c28f},  // 0.03 -> 0.015
    {0x183a527, 0x3dcccccd, 0x3e4ccccd},  // 0.1 -> 0.2
    {0x183a531, 0x3e99999a, 0x3e4ccccd},  // 0.3 -> 0.2
    {0x183a53b, 0x3e99999a, 0x3e19999a},  // 0.3 -> 0.15
    {0x183a649, 0x3f860a92, 0x3f32b8c2},  // 1.0472 -> 0.698132
    {0x183a653, 0x3d75c28f, 0x3d23d70a},  // 0.06 -> 0.04
    {0x183a65d, 0x3dcccccd, 0x3e19999a},  // 0.1 -> 0.15
    {0x183a8c1, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a8cb, 0x3e800000, 0x3d4ccccd},  // 0.25 -> 0.05
    {0x183a8d5, 0x3e99999a, 0x3d4ccccd},  // 0.3 -> 0.05
    {0x183a8df, 0x3e99999a, 0x3d4ccccd},  // 0.3 -> 0.05
    {0x183a8f3, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a8fd, 0x3e800000, 0x3d4ccccd},  // 0.25 -> 0.05
    {0x183a907, 0x3cf5c28f, 0x3c75c28f},  // 0.03 -> 0.015
    {0x183a925, 0x3ecccccd, 0x3e4ccccd},  // 0.4 -> 0.2
    {0x183a92f, 0x3ecccccd, 0x3e4ccccd},  // 0.4 -> 0.2
    {0x183a939, 0x3e99999a, 0x3e19999a},  // 0.3 -> 0.15
    {0x183aa47, 0x3f860a92, 0x3f32b8c2},  // 1.0472 -> 0.698132
    {0x183aa51, 0x3d75c28f, 0x3d23d70a},  // 0.06 -> 0.04
    {0x183aa5b, 0x3e99999a, 0x3e19999a},  // 0.3 -> 0.15
    {0x1972a80, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ab67ef, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x1bf9b74, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1c0ca9a, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1daa65e, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea4003, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea40a0, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea4146, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea49ce, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea4bda, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea5248, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea5476, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea552d, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea57c0, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea5ac5, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea6015, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea6045, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea6275, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea62a5, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea632f, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1eb540b, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1eb54dc, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1eb90aa, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1eb9117, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ebb111, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ebf0d3, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec3f69, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec3f9a, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec3fd8, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec4009, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec404e, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec40a4, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec40e0, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec4136, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec4167, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec41a5, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec41dc, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1f032f2, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1f03762, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1f6aa32, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1f96d9a, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x200d8e2, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x200e142, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x200e2d4, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x200fa52, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x20119d2, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2012652, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x20127cc, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2012932, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2012ac4, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2012c32, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x203acb7, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x222bcd7, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x23dd026, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x23dede6, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x2418e39, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2418f38, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x243460c, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x243485f, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2434883, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x24348b8, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x25df519, 0x3c888889, 0x3c088889},  // 0.0166667 -> 0.00833333
    {0x4befdcc, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4c9f358, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4cad65c, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4cad660, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4cad664, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4ccc3f4, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4cf6420, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x4cf6424, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x4cf6428, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x4cf642c, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x4d2577c, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d25e68, 0x3dcccccd, 0x3d523176},  // 0.1 -> 0.0513167 (1 - 0.9^(1/2))
    {0x4d25e94, 0x3d088889, 0x3c89b0c3},  // 0.0333333 -> 0.0168070
    {0x4d265fc, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d26618, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d27554, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d275f4, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d27c4c, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d28e98, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d29170, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d29b80, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d29e74, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
};

// The constants above that follow the rate, and how; the rest of the table
// keeps its 60 value at every rate above 30. `each_frame`: .rodata the game
// reads where it uses it, written again every frame from the frame's seconds
// at every converted rate (the same clamped frame time the FPS++ cave hands
// the game's task step: each_frame below), so it follows a frame that ran
// long as the task step does. The others are code immediates
// or tied to a frame count (0x4cf6420, a tenth of 0x192c405's interval a
// frame) and take the rate's value (uncapped: its reference rate, 120).
enum class Kind : std::uint8_t {
    Step,    // a frame's seconds: dt
    Amount,  // an amount a frame, the game's own at 30 a second: was * 30 dt
    Decay,   // a smoothing weight a frame: 1 - (1 - was)^(30 dt)
    Keep,    // a share kept a frame: was^(30 dt), the same share kept a second
    Rate,    // a count a second the game wrote as 30 (frames in a second): was / (30 dt)
};
struct Rated {
    std::uint32_t at;
    Kind kind;
    bool each_frame;
};
constexpr Rated kRated[] = {
    // The physics world's expected minimum step (hkpWorldCinfo, sub_f38300):
    // a tolerance hint, 1/fps (Kyo's 90 and 120: 1/90, 1/120).
    {0x0f383ca, Kind::Step, false},
    // The debug camera's step: a code immediate.
    {0x1972a80, Kind::Step, false},
    // The ragdoll keyframe controller's drive step (sub_1c0c2b0, an
    // hkaRagdollRigidBodyController's +0x60, checked > 0 at 0xae461a): the
    // physics world's step, which now follows the frame.
    {0x1c0ca9a, Kind::Step, false},
    // YEBIS's elapsed time once more: the post-process context's +4 (set to
    // 1/60 here in its initialisation, 0x25df516, and never again) is stored
    // into CPostEffect+0x6ac every frame (0x25d7e56) after the write of
    // 0x4d2979c's value (0x25d6af0, kDtLoads), so it is the one the eye
    // adaptation and afterimage decays read; half a frame, at the rate's value.
    {0x25df519, Kind::Amount, false},
    // A tenth of the change over 0x192c3ff's 10-frame interval, a frame: it
    // and the interval (kCounts) go together, 1/20 and 20 at 60, 1/30 and 30
    // at 90, 1/40 and 40 at 120.
    {0x4cf6420, Kind::Amount, false},
    {0x4cf6424, Kind::Amount, false},
    {0x4cf6428, Kind::Amount, false},
    {0x4cf642c, Kind::Amount, false},
    // The follow camera's distance ease, 0.1 a frame at 30 (0x183af8c); Kyo's
    // first 60 list held 0.025, half the 60 value (his 1.3: scaled by dt).
    {0x4d25e68, Kind::Decay, true},
    // The follow camera's chase-rate ease (0x183e2b3: value += (target - value) * w).
    {0x4d25e94, Kind::Decay, true},
    // The per-frame values that were here (the sprint's wall test, remote
    // characters' locomotion, the hitbox velocity, the targeting ring, the
    // bone impulse, the UI steps, the ghosts' fade, YEBIS's frame time,
    // momentum, the sprint ramps and the slope slide) are read through
    // kDtLoads' slots instead; 0x4d2577c and 0x4d29170, whose loads the 60
    // FPS++ sites point at the frame-time slot, need nothing.
};
constexpr bool rated_in_constants() {
    for (const Rated& r : kRated) {
        bool found = false;
        for (const Constant& c : kConstants) found = found || c.at == r.at;
        if (!found) return false;
    }
    return true;
}
static_assert(rated_in_constants(), "every rated value is one of kConstants (its old value is checked there)");

// The list's other value changes. Counts that follow the rate: a countdown
// of 10 frames (0x192c3ff's movl into +0xf8), the default of 5 an int setting
// is read with (0x1929675's mov edx, before sub_24eba00), a worker's
// usleep(33000) in a polling loop (0x25629f0), and the sound system's update
// time. That thread (MagicOrchestra over FMOD, its loop 0x2c00e30) updates
// every MainThreadUpdateTime us, read from the text config at 0x4b33b40 into
// +0x8d4 (sub_2bf67f0, strtol base 10; the engine's own default is 16666).
// Bloodborne ships 33333: at 60 the game posts two frames of sound requests
// per update, and some attack sounds played half as often as at 30
// (2026-09-30). The digits, in place: one update a frame at every rate, and
// at the uncapped mode's 240 (04166).
enum class Count : std::uint8_t {
    Frames,  // was * fps / 30
    Micros,  // was * 30 / fps
    Digits,  // Micros, as five ASCII digits
};
struct Counted {
    std::uint32_t at;
    std::uint8_t n;
    std::uint32_t was;
    Count kind;
};
constexpr Counted kCounts[] = {
    {0x192c405, 4, 10, Count::Frames},
    {0x1929676, 4, 5, Count::Frames},
    {0x25629f1, 4, 33000, Count::Micros},
    {0x4b33c51, 5, 33333, Count::Digits},
};

// And the same at every rate: the frame-time manager's catch-up allowance
// zeroed (its movabs rcx, 0x1e0000000n into +0x268/+0x26c) - inside the
// function ours replaces, where it only matters with BBHOST_FRAME_SOURCE=0.
// (A load of the ragdoll/clip drive's step was moved here to a converted
// constant until 2026-10-10; kFrameTimeSites reads the frame's time there.)
struct Bytes {
    std::uint32_t at;
    std::uint8_t n;
    std::uint8_t was[10], now[10];
};
constexpr Bytes kBytes[] = {
    {0x2434863, 10, {0x48, 0xb9, 0x01, 0, 0, 0, 0x1e, 0, 0, 0}, {0x48, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0}},
    {0x2434887, 10, {0x48, 0xb9, 0, 0, 0, 0, 0x1e, 0, 0, 0}, {0x48, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0}},
};

// The rest of that list (Kyo's "60 FPS++" for 1.09), generated with its old
// bytes from the eboot by tools/fps_sites.py; the tables above are left out
// of it. What it does:
//  - the measured frame time, clamped to [target, 1/30] by a cave over a panic
//    path (0x2418e3d, the flipper's +0x264 against +0x18), stored in a free
//    slot (0x575a908) that the loads of shared 1/30 steps the list names read, and
//    multiplied in where 0x21bc181 had a fixed step - those systems step by
//    the frame that ran rather than by 1/30 (the list's author credits the
//    cloth and jumping fixes at 60 to these). The target is ours (1/60, 1/90,
//    or the uncapped limit), so this is what makes 90 and uncapped real time;
//  - two runs of parameter reads (0x183ff29, 0x1840799, beside the camera's
//    field-of-view clamp) replaced by stores of the list's 60 fps values, and
//    two edits in the camera's update (0x183dc97, 0x183dcfe) - left out with
//    the Camera tuning Original (kKyoCameraSites);
//  - a few constants and table bytes (0x47df586, 0x3abc2xx), the cloth's
//    substep count forced to 1 (0xb6e67a: Kyo's "the part files were authored
//    at 30 fps with 2 substeps: a rate of 60"), the motion blur's sample
//    check (0xfbc40f).
// Left out, each with its reason in the generator: the frame-time manager's
// own edits (ours replaces that function), a frame-mode store, the mode-name
// pointer table, the timed wait turned into a return, and the sound thread's
// sleep cap (a cave over a live function) - that thread's update time is in
// kCounts instead. BBHOST_SIXTY_SITES=0 leaves them all out (and so 90 and
// uncapped, which need them).
struct Site {
    std::uint32_t at;
    const char* was;
    const char* now;
};
constexpr Site kSixtySites[] = {
#include "engine/sixty_fps_sites.inc"
};

// Steps the lists left fixed, made to read the frame's time - the clamped
// measured frame time the 60 FPS++ cave keeps in its slot (0x575a908) every
// frame, what the game's task step is at every converted rate. Applied with
// the 60 FPS++ sites (the slot is theirs) at every rate. Each found by Kyo's
// 1.3 lists and yogesh1239's Timing and AI/Camera fixes (min(dt, 1/30) in
// caves at 0x5490000) except the physics world, and done here as loads and
// in-place rewrites, no new code: Binary Ninja addresses, the eboot's bytes,
// ours.
constexpr Site kFrameTimeSites[] = {
    // NS_SPRJ::SprjTargetBankManager::Update's refresh accumulator:
    // vaddss xmm0, xmm0, [0x4d2577c -> the slot].
    {0x016a4416, "c5fa58055e136803", "c5fa5805ea640b04"},
    // The ragdoll/clip drive (sub_1915ec0, from ChrIns::vfunc_34) hands its
    // three players a step of 1/30 s a call (about 220 calls a second near
    // enemies): yogesh1239 measured those clocks at 2.0 and 4.0 s a second at
    // 60 and 120. Path B (to 0x1c11650) and path C (to 0x1c0df90):
    // vmovss xmm0, [0x4d26364 -> the slot].
    {0x01915fab, "c5fa1005b1034103", "c5fa10055549e403"},
    {0x01916030, "c5fa10052c034103", "c5fa1005d048e403"},
    // Path A (to 0x1c11040) uses the same 1/30 first to turn a frame count
    // into seconds - which must stay 1/30 - and then as the step. Before:
    // vcvtsi2ss xmm1,xmm0,ecx; vmovss xmm0,[1/30]; vmulss xmm1,xmm1,xmm0;
    // mov rdi,[rbx+0x20]; mov eax,[rax+0x1e1]; test al,4; sete al; movzx
    // ecx,al; mov rsi,r14; mov rdx,r15; add rsp,8. After: mov rdi,[rbx+0x20];
    // vcvtsi2ss xmm1,xmm0,ecx; vmulss xmm1,xmm1,[1/30]; vmovss xmm0,[slot];
    // xor ecx,ecx; test byte [rax+0x1e1],4; sete cl; mov rsi,r14;
    // mov rdx,r15; pop rax (the prologue's alignment push); nop.
    {0x01915f65, "c5fa2ac9c5fa1005f3034103c5f259c8488b7b208b80e1010000a8040f94c00fb6c84c89f64c89fa4883c408",
     "488b7b20c5fa2ac9c5f2590def034103c5fa10058b49e40331c9f680e1010000040f94c14c89f64c89fa5890"},
    // SprjHkAiManager's Havok AI world step (sub_222bc10): the step-info
    // {0x56efd40, 1/30} it passes to 0x21cdad0 (hkaiWorld's character and
    // behaviour step). The lists run it at 1/60 at every rate - enemy steering
    // 1.5 times as fast at 90. Before: lea rcx,[0x56efd30]; add rcx,0x10;
    // mov [rbp-0x1c0],rcx; mov dword [rbp-0x1b8],1/30. After: lea
    // rcx,[0x56efd40]; mov [rbp-0x1c0],rcx; mov ecx,[slot];
    // mov [rbp-0x1b8],ecx; nop2 (rcx is not an argument of the call).
    {0x0222bcbf, "488d0d6a404c034883c11048898d40feffffc78548feffff8988083d",
     "488d0d7a404c0348898d40feffff8b0d35ec5203898d48feffff6690"},
    // The Havok physics world's step (sub_1bf99d0: hkpWorld's multithreaded
    // step through sub_f5a2e0 and sub_f5b100, a step-info whose dt is 1/30):
    // every list held 1/60 at every rate (Lance McDonald's uncapped list alone
    // reads the frame time, capped at 1/60), so rigid bodies, ragdolls and
    // breakables ran 1.5 times as fast at 90 and twice at 120. Before:
    // mov qword [rbp-0x40],0 (dead: the next store fills it); mov dword
    // [rbp-0x48],1/30. After: mov eax,[slot] (rax is loaded next);
    // mov [rbp-0x48],eax; nop6.
    {0x01bf9b69, "48c745c000000000c745b88988083d", "8b05990db6038945b8660f1f440000"},
    // Grass sway (sub_2713870): with the GX device's +0x100 at 1 it steps a
    // fixed 1/60 a frame (the 60 FPS++ sites already point its other path at
    // the slot): vmovss xmm4, [0x4d29e70 -> the slot].
    {0x02713d8d, "c5fa1025db606102", "c5fa1025736b0403"},
};

// The play time (GameDataMan+0x94, milliseconds: the save list's clock and
// the play log's): NS_SPRJ::MoveMapStep::InGameUpdate adds the frame's step
// to it in whole milliseconds, truncated - (int)(dt * 1000) - so a 16.7 ms
// frame adds 16 and the clock runs 4% slow at 60 and 120 (1% at the game's
// own 30 and at 90, 14% at 144 Hz uncapped). At a converted rate it adds a
// count of ours instead: the same step (the 60 FPS++ cave's
// min(max(measured, target), 1/30)) in milliseconds with the remainder
// carried, written by our SprjFlipper::Update every frame into the image's
// tail (core/elf.h). Before: vmulss xmm0,xmm1,[1000.0]; vcvttss2si
// rcx,xmm0. After: mov ecx,[count]; nop7 - xmm1 (the step) is what the code
// after reads, and xmm0 is written before it is read again.
// (NS_SPRJ::GameDataMan::AccumulatePlayTime, 0x18eb6a0, is the same code out
// of line; nothing calls it.)
constexpr std::uint32_t kPlayTimeAdd = 0x1d419d6;
constexpr char kPlayTimeAddWas[] = "c5f259051267fe02c4e1fa2cc8";
static_assert(sizeof(kPlayTimeAddWas) - 1 == 2 * 13, "the add and its replacement are 13 bytes");

// The task step itself: the 60 FPS++ cave (0x2418e3d, as kSixtySites writes
// it) computes it as movss xmm0,[rbx+0x264]; maxss xmm0,[rbx+0x18]; minss
// xmm0,[1/30] - the flipper's measured frame time clamped to [target, 1/30]
// - and stores it into the step-info and the slot. With our
// SprjFlipper::Update the cave loads ours instead (movss xmm0,[step]; nop8;
// nop5), written every frame in the image's tail just before the cave runs:
// the same clamp, times a plugin's time scale (frame_rate_set_time_scale),
// at most 1/30. Above 30 everything steps by that step, so slow motion
// scales the step rather than stretching the frames.
constexpr std::uint32_t kStepLoad = 0x2418e3d;
constexpr char kStepLoadWas[] = "f30f108364020000f30f5f4318f30f5d0512d59002";
static_assert(sizeof(kStepLoadWas) - 1 == 2 * 21, "the cave's three instructions are 21 bytes");

// The Camera tuning option (host/options.cpp camera_tuning;
// BBHOST_CAMERA_TUNING=fps++|original), read when the game starts. FPS++
// keeps Kyo's retune of the follow camera from his 60 FPS++ list, as bbhost
// always has: his parameter stores (0x183ff29, 0x1840799, kSixtySites), the
// constructor's defaults (kConstants 0x183a4c3..0x183aa5b, the halved chase
// rates and a 40 degree direct pitch angle) and two edits in the camera's
// update (0x183dc97, the jump past the chase-rate pick; 0x183dcfe, the wall
// escape when neither probe hits), with its weights converted from a 60 fps
// frame (frame_rate_step::g_camera_tuned_fps). Original leaves all of those
// as the game has them and converts the game's own weights from a 30 fps
// frame - the camera of the game at 30 at every rate. At 30 nothing here is
// written: the camera is the game's either way (Kyo's 30 FPS++ list, a
// separate retune with stale values, was never applied).
constexpr std::uint32_t kKyoCameraSites[] = {0x183dc97, 0x183dcfe, 0x183ff29, 0x1840799};
constexpr std::uint32_t kKyoCameraFirst = 0x183a4c3, kKyoCameraLast = 0x183aa5b;  // the constructor's immediates
bool kyo_camera_site(std::uint32_t at) {
    return std::find(std::begin(kKyoCameraSites), std::end(kKyoCameraSites), at) != std::end(kKyoCameraSites);
}
bool kyo_camera_constant(std::uint32_t at) { return at >= kKyoCameraFirst && at <= kKyoCameraLast; }

// Redirected loads: a constant the game reads where it means "a frame" - a
// weight or a share kept a frame, an amount a frame, a step, a count a
// second - is read from a slot of ours instead, in the image's tail
// (core/elf.h), which each frame's seconds rewrite. The instruction's rip
// displacement is pointed at the slot; the constant itself stays the game's.
// So a constant other code reads with another meaning (a duration, a unit)
// keeps it, no page of .rodata is written while the game runs, and any rate -
// one that moves every frame too - converts exactly. The table,
// engine/frame_rate_dt_loads.inc, is generated by tools/dt_loads.py from
// tools/dt_loads.txt and the eboot: each instruction's bytes and its
// displacement's target are checked first, and one mismatch leaves the whole
// table out. Kinds as kRated's; an Amount may be negative (a timer counted
// down a fixed amount a call).
struct DtLoad {
    std::uint32_t insn;      // the instruction (Binary Ninja address)
    std::uint8_t len, disp;  // its length and where its 4-byte rip displacement is
    std::uint32_t constant;  // the value it reads (.rodata or .data)
    Kind kind;               // how that value follows the frame's time
    std::uint8_t lanes;      // 1 (a dword), 2 (a qword), 4 (an xmmword)
    const char* bytes;       // the instruction as the eboot has it
};
//
// What it holds (each with its reader in the .inc):
//  - kept shares a frame: the movement controller's momentum (0.95 a frame on
//    the body's X/Z, chr_motion_physics_update; 0.95^(30 dt) keeps a jump's
//    reach at any rate), the sprint wall test's ramps (x0.8 blocked, x1.2
//    free), CSSlopeCtrl's slope fall damping (0.65, a .data tunable);
//  - amounts a frame: the slope fall speed (1.0, .data), the replay ghosts'
//    fade (+-0.1), YEBIS's frame time (half a 30 fps frame: the eye
//    adaptation's), a state-table fade (+-0.05 at 0x1d3251e/0x1d3259b), and
//    SprjWorldAiManager::TickLua's timer (-1/30 a call before a full GC of
//    the AI Lua state: every 0.25 s at 60, 0.125 s at 120 until now);
//  - rates (a distance this frame times 30 is a speed: 1/dt): the sprint's
//    wall test (blocked under 1 unit a second; at 60 a sprint from rest
//    started blocked), remote characters' locomotion (NetAIManipulator's 2.5,
//    NetworkManipulator's 0.01/2/6: at 60 a remote player never showed the
//    dash), the attack hitbox velocity (two 4-lane vectors), a physics-driven
//    bone's impulse (close the gap in one physics step), the targeting ring's
//    slot (x30) and overflow (/30);
//  - steps: the equipment screen's character preview, the shop's tab
//    animation.
constexpr DtLoad kDtLoads[] = {
#include "engine/frame_rate_dt_loads.inc"
};
constexpr bool dt_loads_apart() {
    for (const DtLoad& d : kDtLoads) {
        for (const Constant& c : kConstants)
            if (c.at >= d.constant && c.at < d.constant + 4u * d.lanes) return false;
    }
    return true;
}
static_assert(dt_loads_apart(), "a redirected constant is not also rewritten in place (kConstants)");

// The cloth above 60 (tools/fps_sites.py, from Kyo's "90 FPS++"). With one
// substep a frame (0xb6e67a above) the hcl solver runs at the frame rate,
// against the 60 the part files were authored for (30 fps, 2 substeps), and
// the six constraint operators' stiffness takes up the ratio: each gets a
// cave in a dead hkMonitorStream timer block (its guard JAE made a JMP) that
// multiplies the stiffness it is called with (lane 0, which the operator
// broadcasts) by the scalars at 0x4d3bd00 - written over the tail of an
// assert message's file name ("DLLightMutex.cpp"), read only when a mutex
// assert fires. The scalar is (60 dt)^2, at most 1, every frame: Kyo's 1.3
// package (4/9 at 90, 1/4 at 120). His first package multiplied by 60 dt and
// also changed the damping exponent (0xb6fbd4) and the integrator's dt * dt
// (0xb73f84); 1.3 leaves both as the game has them, and so does this table.
// His first 90 list also fixed the character turn's step at 0.0125
// (0x1cbdb42) - turns 12.5% fast at 90 - which 1.3 drops: the turn steps by
// the frame's time, as at every other rate.
constexpr Site kClothSites[] = {
#include "engine/ninety_fps_sites.inc"
};

// No redirected load is an instruction another table rewrites (the 60 FPS++
// list points some loads at the frame-time slot; those constants are dead).
constexpr std::size_t hex_len(const char* h) {
    std::size_t n = 0;
    while (h[n]) ++n;
    return n / 2;
}
template <std::size_t N>
constexpr bool dt_loads_clear_of(const Site (&sites)[N]) {
    for (const DtLoad& d : kDtLoads) {
        for (const Site& site : sites) {
            if (d.insn < site.at + hex_len(site.was) && site.at < d.insn + d.len) return false;
        }
    }
    return true;
}
static_assert(dt_loads_clear_of(kSixtySites) && dt_loads_clear_of(kFrameTimeSites) && dt_loads_clear_of(kClothSites),
              "a redirected load is not rewritten by another table");
constexpr std::uint32_t kClothScalars = 0x4d3bd00;  // 4 x the stiffness scale
constexpr std::uint8_t kClothScalarsWas[16] = {'D', 'L', 'L', 'i', 'g', 'h', 't', 'M', 'u', 't', 'e', 'x', '.', 'c', 'p', 'p'};

// The uncapped mode: the game's frame limit and the reference rate for what
// cannot follow the frame time (code immediates, frame counts) - Kyo's
// uncapped list takes 120's values too.
constexpr int kUncappedMaxFps = 240;
constexpr int kUncappedReferenceFps = 120;

// The rates the Frame rate option offers besides 30 and uncapped
// (host/options.cpp, engine/option_menu.cpp, host/launcher.cpp list the
// same): each runs the game's mode 2 with a 1/fps target.
constexpr int kFrameRates[] = {60, 90, 120};

// A frame rate as one of the modes: 30, a converted rate, or 0 (uncapped).
// Another number takes the nearest mode below it.
int mode_of(int fps) {
    if (fps <= 0) return 0;
    int mode = 30;
    for (const int r : kFrameRates) {
        if (fps >= r) mode = r;
    }
    return mode;
}

std::string mode_name(int fps) { return fps == 0 ? std::string("uncapped") : std::to_string(fps); }

std::vector<std::uint8_t> hex_bytes(const char* h) {
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; h[i] && h[i + 1]; i += 2) {
        auto nib = [](char c) { return static_cast<std::uint8_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10); };
        out.push_back(static_cast<std::uint8_t>(nib(h[i]) << 4 | nib(h[i + 1])));
    }
    return out;
}

float bits_float(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}
std::uint32_t float_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// A rated constant at a frame rate (fractional for the uncapped mode's frame
// time). At 60 the table's own value is written instead (the same within a
// rounding of Decay's).
float rated_value(Kind kind, float was, double fps) {
    switch (kind) {
        case Kind::Step: return static_cast<float>(1.0 / fps);
        case Kind::Amount: return static_cast<float>(static_cast<double>(was) * 30.0 / fps);
        case Kind::Decay: return static_cast<float>(1.0 - std::pow(1.0 - static_cast<double>(was), 30.0 / fps));
        case Kind::Keep: return static_cast<float>(std::pow(static_cast<double>(was), 30.0 / fps));
        case Kind::Rate: return static_cast<float>(static_cast<double>(was) * fps / 30.0);
    }
    return was;
}

const Rated* rated(std::uint32_t at) {
    for (const Rated& r : kRated)
        if (r.at == at) return &r;
    return nullptr;
}

// A count's bytes at a frame rate.
void count_bytes(const Counted& c, int fps, std::uint8_t* out) {
    const std::uint64_t v = c.kind == Count::Frames ? static_cast<std::uint64_t>(c.was) * static_cast<std::uint64_t>(fps) / 30
                                                    : static_cast<std::uint64_t>(c.was) * 30 / static_cast<std::uint64_t>(fps);
    if (c.kind == Count::Digits) {
        char d[16];
        std::snprintf(d, sizeof(d), "%05u", static_cast<unsigned>(std::min<std::uint64_t>(v, 99999)));
        std::memcpy(out, d, c.n);
        return;
    }
    const auto u = static_cast<std::uint32_t>(v);
    std::memcpy(out, &u, c.n);
}
void count_was(const Counted& c, std::uint8_t* out) {
    if (c.kind == Count::Digits) {
        char d[16];
        std::snprintf(d, sizeof(d), "%05u", static_cast<unsigned>(c.was));
        std::memcpy(out, d, c.n);
        return;
    }
    std::memcpy(out, &c.was, c.n);
}

// The game's pace: 30, a converted rate (60, 90, 120), or 0 for uncapped.
int g_game_fps = 30;
std::size_t g_sites_applied = 0, g_frame_time_applied = 0, g_cloth_applied = 0;
std::atomic<std::uint64_t> g_frames{0};
// The every-frame values (kRated's each_frame) at every converted rate: where
// each goes and what it started from; above 60 the cloth's stiffness scale,
// over a frame time smoothed across ~10 frames so a hitch does not jolt it.
struct EachFrame {
    std::uint64_t va;
    float was;
    Kind kind;
};
std::vector<EachFrame> g_each;
std::uint64_t g_cloth_va = 0;
float g_cloth_dt = 1.0f / 60.0f;
// The play time's count (kPlayTimeAdd) and the part of a millisecond carried.
std::uint64_t g_play_ms_va = 0;
double g_play_ms_carry = 0.0;
// The task step the cave loads (kStepLoad); while it is ours a plugin's time
// scale scales the step, not the frames' pace.
std::uint64_t g_step_va = 0;
std::atomic<bool> g_scale_by_step{false};
// The redirected loads' slots (kDtLoads): one per constant, kind and width,
// 16 bytes each in the image's tail, rewritten every frame from the constant
// as it is then (a .data tunable the debug menu edits keeps working).
struct DtSlot {
    std::uint64_t va, src;
    Kind kind;
    std::uint8_t lanes;
};
std::vector<DtSlot> g_dt_slots;
std::size_t g_dt_loads_applied = 0;

template <typename T>
void put(std::uint64_t a, T v) {
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(a)), &v, sizeof(v));
}
template <typename T>
T get(std::uint64_t a) {
    T v;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(a)), sizeof(v));
    return v;
}

// --- SprjFlipper::Update as source -----------------------------------------
//
// What the eboot's 0x2434770 does, in order, read from its decompile
// (Binary Ninja) with sprj_flipper.h's field names: pick the mode (the
// secondary one for one frame when requested), write the frame-advance
// policy and the target interval for it, take the one-shot overrides, sample
// the clock, shorten this frame's interval when the last frames ran late
// (down to a third), sleep to within 5 ms of the deadline and spin the rest,
// record the frame in the 32-entry history and the 16-sample FPS average,
// decide whether the next frame is "behind", and ask the window for 60 Hz
// once when told to. sub_13e3980, which chose the foreground history count
// and the sleeping wait, returns 1 in this build, so only that path exists.
//
// Ours differs in one thing: which mode runs is the host's choice (the
// Frame rate option), not the game's fields, and above 30 the game's own
// one-shot resets to 30 (the multiplayer route, the "reset to 30" request)
// do not apply. At 60 it is the game's own mode 2; at 90 and uncapped the
// same with our target (1/90, or the uncapped limit). The clock and the sleep
// are the host's; the arithmetic is the eboot's, float for float, so compare
// mode can demand equality.

struct FlipClock {
    // The clock as the eboot reads it: gettimeofday in microseconds. In a
    // replay (compare mode) it returns the release time the guest reached,
    // and the sleep does nothing.
    virtual std::int64_t now() = 0;
    virtual void sleep_us(std::int64_t us) = 0;
    // How close to the deadline the wait sleeps before it spins: the eboot's
    // 5 ms, or less for a host sleep that keeps time better (RealClock).
    virtual float spin_margin_s() { return 0.00499999989f; }
    // The least time left that is worth a sleep: the eboot's 1 ms.
    virtual float min_sleep_s() { return 0.00100000005f; }
    // Between two reads of the clock while spinning.
    virtual void spin_pause() {}
    // The wait is over (accounting only).
    virtual void wait_done() {}
    virtual ~FlipClock() = default;
};

// The limiter's margin, learnt from its own sleeps (CPU-04). The eboot
// sleeps to 5 ms before the deadline and spins the rest on gettimeofday: a
// fifth of the main thread's CPU at 60 fps went there (the 2026-10-04
// profile), taken from its SMT sibling - and on an APU (the Radeon 8060S),
// from the GPU's share of the package's power. This is the cost Kyo found in
// KyoPS4x's traces as "the game calls gettimeofday constantly" (Game:Main
// ~30% in posix_gettimeofday at a cap). A fixed 2 ms kept most of it back on
// Linux, where 1 ms cost flip-interval p95 a millisecond on a busy machine;
// how late a sleep wakes is the host's, though: Windows' high-resolution
// waitable timer is usually a few hundred microseconds late, a loaded Linux
// box more. So every sleep's lateness is kept (the last 128), and the margin
// is the second-latest of them plus 200 us, within 0.3-2 ms; it starts at
// 2 ms and moves after 32 sleeps. BBHOST_LIMITER_MARGIN_US=<n> fixes it
// (2000 is the old behaviour).
struct LimiterTuning {
    std::int64_t fixed_us = -1;  // BBHOST_LIMITER_MARGIN_US, -1 for learnt
    std::int64_t late_us[128] = {};
    std::uint32_t samples = 0;
    std::int64_t margin_us = 2000;
    std::int64_t late_p99_us = 0;  // the second-latest of the window
    // Since the last report (frame_rate_limiter_window): the main thread
    // only writes these, the report thread only reads and resets.
    std::atomic<std::uint64_t> waits{0}, sleeps{0}, spin_us{0}, sleep_late_us{0};
    std::atomic<std::int64_t> shown_margin_us{2000}, shown_late_p99_us{0};
};
LimiterTuning g_limiter;
const bool g_limiter_env = [] {
    if (const char* e = std::getenv("BBHOST_LIMITER_MARGIN_US"); e && *e && std::strcmp(e, "auto") != 0) {
        g_limiter.fixed_us = std::clamp<std::int64_t>(std::strtoll(e, nullptr, 10), 0, 5000);
        g_limiter.margin_us = g_limiter.fixed_us;
        g_limiter.shown_margin_us.store(g_limiter.fixed_us);
    }
    return true;
}();

void limiter_note_sleep(std::int64_t asked_us, std::int64_t took_us) {
    LimiterTuning& t = g_limiter;
    const std::int64_t late = std::max<std::int64_t>(0, took_us - asked_us);
    t.sleeps.fetch_add(1, std::memory_order_relaxed);
    t.sleep_late_us.fetch_add(static_cast<std::uint64_t>(late), std::memory_order_relaxed);
    t.late_us[t.samples % 128] = late;
    ++t.samples;
    if (t.fixed_us >= 0 || t.samples < 32 || (t.samples % 16) != 0) return;
    const std::uint32_t n = std::min<std::uint32_t>(t.samples, 128);
    std::int64_t first = 0, second = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::int64_t v = t.late_us[i];
        if (v > first) {
            second = first;
            first = v;
        } else if (v > second) {
            second = v;
        }
    }
    t.late_p99_us = second;
    t.margin_us = std::clamp<std::int64_t>(second + 200, 300, 2000);
    t.shown_margin_us.store(t.margin_us, std::memory_order_relaxed);
    t.shown_late_p99_us.store(second, std::memory_order_relaxed);
}

struct RealClock final : FlipClock {
    // The guest's gettimeofday clock (hle/libc.cpp), read the same way.
    std::int64_t now() override { return host_clock_realtime_us(); }
    void sleep_us(std::int64_t us) override {
        const std::int64_t t0 = now();
        host_sleep_us(static_cast<std::uint64_t>(us));
        const std::int64_t t1 = now();
        limiter_note_sleep(us, t1 - t0);
        slept_until_ = t1;
        spun_from_ = 0;
    }
    float spin_margin_s() override { return static_cast<float>(g_limiter.margin_us) * 1e-6f; }
    // A sleep is worth taking down to the margin itself, not only above the
    // eboot's 1 ms: with a margin under 1 ms the remainder would be spun.
    float min_sleep_s() override { return spin_margin_s(); }
    // The spin pauses between reads.
    void spin_pause() override {
        if (!spun_from_) spun_from_ = slept_until_ ? slept_until_ : now();
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#endif
    }
    void wait_done() override {
        g_limiter.waits.fetch_add(1, std::memory_order_relaxed);
        if (spun_from_) g_limiter.spin_us.fetch_add(static_cast<std::uint64_t>(std::max<std::int64_t>(0, now() - spun_from_)), std::memory_order_relaxed);
        slept_until_ = 0;
        spun_from_ = 0;
    }

private:
    // This frame's wait (one RealClock lives for one update): when its last
    // sleep ended, and when its spin began.
    std::int64_t slept_until_ = 0;
    std::int64_t spun_from_ = 0;
};

struct ReplayClock final : FlipClock {
    std::int64_t at;
    explicit ReplayClock(std::int64_t t) : at(t) {}
    std::int64_t now() override { return at; }
    void sleep_us(std::int64_t) override {}
};

// The unsigned-to-float conversion the compiler emitted for the frame's
// microseconds (a negative count halves and doubles); the count is never
// negative in practice, so this is float(int64).
inline float us_to_float(std::int64_t us) {
    if (us >= 0) return static_cast<float>(us);
    const auto u = static_cast<std::uint64_t>(us);
    const float h = static_cast<float>((u >> 1) | (u & 1));
    return h + h;
}

inline std::int64_t seconds_to_us(float s) {
    const double d = static_cast<double>(s) * 1000000.0;  // the eboot converts to double first
    return d >= 9223372036854775808.0 ? static_cast<std::int64_t>(d - 9223372036854775808.0) ^ INT64_MIN
                                      : static_cast<std::int64_t>(d);
}

// The window object at data_5940e00: told once to run at 60 Hz when the
// flipper is asked to (+0x4c, +0x4d flags and +0x50 = 60).
constexpr std::uint64_t kWindowObjectSlot = 0x5940e00;
std::uint64_t g_slide = 0;
// This frame runs with no target (a loading screen, engine/loading.h); set by
// the hook for its own update only, never for compare mode's replay.
bool g_fast_target = false;
// BBHOST_BENCH_UNCAPPED=1 (with BBHOST_UNCAP=1, which completes flips at
// once): every frame runs with no target - a measure of how fast the machine
// can run the game's frames, not a way to play it: at 60 and 90 the steps
// Kyo's lists fix at 1/60 then go by faster than real time. Stand still
// to measure. The uncapped mode is the way to play it fast.
const bool g_bench_uncapped = [] {
    const char* e = std::getenv("BBHOST_BENCH_UNCAPPED");
    return e && e[0] == '1';
}();

constexpr float kThirtieth = 0.0333333351f, kSixtieth = 0.0166666675f;  // 0x3d088889, 0x3c888889

// `target`: 0 runs the game's own mode (30); above 0 the converted pace, the
// game's mode 2 with this target - kSixtieth at 60, so 60 is the eboot's own
// mode 2 exactly.
void sprj_flipper_update(SprjFlipper* f, FlipClock& clock, float target) {
    const bool converted = target > 0.0f;
    f->secondary_mode_active = f->secondary_mode_requested;
    f->secondary_mode_requested = 0;
    std::uint32_t mode = f->secondary_mode_active ? f->secondary_flip_mode_raw : f->primary_flip_mode_raw;
    if (converted) mode = static_cast<std::uint32_t>(SprjFlipMode::Fps60WithoutSkip);
    if (mode > 4) mode = 0;
    switch (mode) {
        case 0: f->frame_advance_count = 2; f->allow_frame_skip = 1; f->target_frame_seconds = kThirtieth; f->foreground_history_count = 1; f->background_history_count = 30; break;
        case 1: f->frame_advance_count = 2; f->allow_frame_skip = 1; f->target_frame_seconds = kThirtieth; f->foreground_history_count = 0; f->background_history_count = 30; break;
        case 2: f->frame_advance_count = 1; f->allow_frame_skip = 1; f->target_frame_seconds = kSixtieth; f->foreground_history_count = 0; f->background_history_count = 30; break;
        case 3: f->frame_advance_count = 1; f->allow_frame_skip = 0; f->target_frame_seconds = kThirtieth; f->foreground_history_count = 1; f->background_history_count = 30; break;
        default: f->frame_advance_count = 1; f->allow_frame_skip = 0; f->target_frame_seconds = kThirtieth; f->foreground_history_count = 0; f->background_history_count = 30; break;
    }
    if (converted) {
        f->target_frame_seconds = target;
        // The 60 FPS patch list zeroes both history counts (its two movabs
        // edits); the same here.
        f->foreground_history_count = 0;
        f->background_history_count = 0;
        // A loading screen (engine/loading.h): no target, so no wait; the
        // 60 FPS++ frame-time cave (0x2418e3d) clamps the measured frame time
        // to [target, 1/30], so game time stays real time.
        if (g_fast_target) f->target_frame_seconds = 0.0f;
    }
    bool no_wait = false;
    if (!converted && f->reset_to_30_fps_pending) {
        f->frame_advance_count = 1;
        f->allow_frame_skip = 0;
        f->target_frame_seconds = kThirtieth;
        f->foreground_history_count = 0;
        f->background_history_count = 0;
        f->reset_to_30_fps_pending = 0;
        no_wait = true;
    }
    if (!converted) {
        if (f->override_foreground_history_count >= 0) f->foreground_history_count = static_cast<std::uint32_t>(f->override_foreground_history_count);
        if (f->override_background_history_count >= 0) f->background_history_count = static_cast<std::uint32_t>(f->override_background_history_count);
        if (f->reset_frame_history) {
            f->foreground_history_count = 0;
            f->background_history_count = 0;
            f->reset_frame_history = 0;
        }
        if (f->force_frame_advance) {  // one 64-bit store over both counts; the flag is not cleared here
            f->foreground_history_count = 0;
            f->background_history_count = 0;
        }
    } else {
        f->override_foreground_history_count = -1;
        f->override_background_history_count = -1;
        f->reset_frame_history = 0;
        f->force_frame_advance = 0;
        f->reset_to_30_fps_pending = 0;
    }
    f->previous_frame_microseconds = f->current_frame_microseconds;
    std::int64_t now = clock.now();
    f->current_frame_microseconds = static_cast<std::uint64_t>(now);
    // A plugin's time scale (frame_rate_set_time_scale): at 30 the game
    // advances a fixed step a frame, so a longer frame is slower time and a
    // shorter one faster, as far as the machine keeps up. At a converted rate
    // with the step ours (each_frame) the step scales and the pace stays.
    const float pace_scale = converted && g_scale_by_step.load(std::memory_order_relaxed)
                                 ? 1.0f
                                 : g_time_scale.load(std::memory_order_relaxed);
    std::int64_t interval_us = seconds_to_us(f->target_frame_seconds / pace_scale);

    // The last frames that ran behind, up to the foreground count of them:
    // how much time they took together.
    const std::uint32_t allowance = f->foreground_history_count;
    std::uint32_t walked = 0;
    std::uint64_t behind_us = 0;
    for (int back = 0; walked < allowance; --back) {
        const SprjFlipperFrameHistoryEntry& e = f->frame_history[(f->frame_history_index + static_cast<std::uint32_t>(back)) & 0x1f];
        behind_us += e.elapsed_microseconds;
        ++walked;
        if (e.delayed_or_behind == 0) break;
    }
    if (f->frame_behind) {
        const std::uint64_t budget = static_cast<std::uint64_t>(walked + 1) * static_cast<std::uint64_t>(interval_us);
        if (behind_us + static_cast<std::uint64_t>(interval_us) > budget) {
            const std::uint64_t third = static_cast<std::uint64_t>(interval_us) / 3;
            if (budget <= behind_us) {
                interval_us = static_cast<std::int64_t>(third);
            } else {
                std::uint64_t rem = budget - behind_us;
                if (static_cast<std::uint64_t>(interval_us) < rem) rem = static_cast<std::uint64_t>(interval_us);
                if (third > rem) rem = third;
                interval_us = static_cast<std::int64_t>(rem);
            }
        }
    }
    if (!(no_wait || f->force_no_sleep)) {
        const float target_s = us_to_float(interval_us) / 1000000.0f;
        const std::int64_t prev = static_cast<std::int64_t>(f->previous_frame_microseconds);
        float remaining = target_s - us_to_float(now - prev) / 1000000.0f;
        const float margin = clock.spin_margin_s();
        const float min_sleep = clock.min_sleep_s();
        while (!(0.0f > remaining)) {
            const float left = std::min(target_s, remaining);
            if (left <= 0.0f) break;
            if (!(left <= min_sleep) && !(left <= margin)) {
                const auto us = static_cast<std::int32_t>((left - margin) * 1000.0f * 1000.0f);
                if (us > 0) clock.sleep_us(us);
            } else {
                clock.spin_pause();
            }
            now = clock.now();
            f->current_frame_microseconds = static_cast<std::uint64_t>(now);
            remaining = target_s + us_to_float(now - prev) / -1000000.0f;
        }
        clock.wait_done();
    }
    if (f->force_no_sleep) f->frame_behind = 1;
    const std::uint32_t idx = (f->frame_history_index + 1) & 0x1f;
    f->frame_history_index = idx;
    const std::int64_t frame_us = static_cast<std::int64_t>(f->current_frame_microseconds - f->previous_frame_microseconds);
    f->frame_history[idx].elapsed_microseconds = static_cast<std::uint64_t>(frame_us);
    f->frame_history[idx].delayed_or_behind = f->frame_behind;
    const float frame_s = us_to_float(frame_us) / 1000000.0f;
    f->measured_frame_seconds = frame_s;
    // On time, unless this frame and the late ones before it overran their
    // budget (plus 1% of the target), in which case only when the allowance
    // is used up.
    bool exhausted = allowance == 0;
    if (f->frame_behind) exhausted = walked >= allowance;
    const std::int64_t tolerance_us = seconds_to_us(f->target_frame_seconds * 0.00999999978f);
    const std::int64_t target_us = seconds_to_us(f->target_frame_seconds);
    const std::uint64_t budget = static_cast<std::uint64_t>(walked + 1) * static_cast<std::uint64_t>(target_us) + static_cast<std::uint64_t>(tolerance_us);
    bool on_time = true;
    if (static_cast<std::uint64_t>(frame_us) + behind_us >= budget) on_time = exhausted;
    if (f->force_no_sleep) on_time = false;
    f->frame_behind = on_time ? 0 : 1;
    if (!on_time) f->frame_advance_count = 0;
    float sum = 0.0f;
    for (int k = 0; k < 15; ++k) {
        const float v = f->fps_frame_time_history[k + 1];
        f->fps_frame_time_history[k] = v;
        sum = sum + v;
    }
    f->fps_frame_time_history[15] = frame_s;
    sum = sum + frame_s;
    f->calculated_fps = sum <= 0.00100000005f ? 0.0f : 16.0f / sum;
    f->force_no_sleep = 0;
    if (g_slide) {
        const auto window = get<std::uint64_t>(g_slide + (kWindowObjectSlot - kPreferredGuestSlide));
        if (window && f->request_window_60_hz) {
            put<std::uint8_t>(window + 0x4c, 1);
            put<std::uint8_t>(window + 0x4d, 1);
            put<std::uint32_t>(window + 0x50, 60);
            f->request_window_60_hz = 0;
        }
    }
}

// --- the entry ----------------------------------------------------------------

enum class Source { Guest, Ours, Compare };
Source g_source = Source::Ours;

// Compare mode: the guest's update runs; its caller's return address is
// swapped for frame_ret (asm below), which calls frame_after as the update
// returns. There ours replays the frame from the snapshot taken at entry,
// with the release time the guest reached, and every field the update
// writes is compared before any other code can touch the object.
struct Pending {
    SprjFlipper* flipper;
    std::uint64_t ret;  // the caller's return address
    SprjFlipper before;
};
thread_local Pending t_pending;
std::atomic<std::uint64_t> g_compared{0}, g_mismatched{0};

void compare_fields(const SprjFlipper& ours, const SprjFlipper& guest) {
    static int logged = 0;
    const auto miss = [&](const char* what, double a, double b) {
        g_mismatched.fetch_add(1, std::memory_order_relaxed);
        if (logged++ < 20) {
            const SprjFlipper& b0 = t_pending.before;
            host_log("frame source: compare mismatch %s: ours %g guest %g (before: mode %u/%u active %u fg %u bg %u ovr %d/%d flags %u %u %u %u reset30 %u)",
                     what, a, b, b0.primary_flip_mode_raw, b0.secondary_flip_mode_raw, b0.secondary_mode_active,
                     b0.foreground_history_count, b0.background_history_count, b0.override_foreground_history_count,
                     b0.override_background_history_count, b0.frame_behind, b0.reset_frame_history, b0.force_frame_advance,
                     b0.force_no_sleep, b0.reset_to_30_fps_pending);
        }
    };
    const auto eq = [&](const char* what, auto a, auto b) {
        if (!(a == b)) miss(what, static_cast<double>(a), static_cast<double>(b));
    };
    eq("frame_advance_count", ours.frame_advance_count, guest.frame_advance_count);
    eq("allow_frame_skip", ours.allow_frame_skip, guest.allow_frame_skip);
    eq("target_frame_seconds", ours.target_frame_seconds, guest.target_frame_seconds);
    eq("previous_frame_microseconds", ours.previous_frame_microseconds, guest.previous_frame_microseconds);
    eq("current_frame_microseconds", ours.current_frame_microseconds, guest.current_frame_microseconds);
    eq("frame_history_index", ours.frame_history_index, guest.frame_history_index);
    const std::uint32_t i = guest.frame_history_index & 0x1f;
    eq("history.elapsed", ours.frame_history[i].elapsed_microseconds, guest.frame_history[i].elapsed_microseconds);
    eq("history.behind", ours.frame_history[i].delayed_or_behind, guest.frame_history[i].delayed_or_behind);
    eq("measured_frame_seconds", ours.measured_frame_seconds, guest.measured_frame_seconds);
    eq("foreground_history_count", ours.foreground_history_count, guest.foreground_history_count);
    eq("background_history_count", ours.background_history_count, guest.background_history_count);
    eq("frame_behind", ours.frame_behind, guest.frame_behind);
    eq("reset_frame_history", ours.reset_frame_history, guest.reset_frame_history);
    eq("force_no_sleep", ours.force_no_sleep, guest.force_no_sleep);
    eq("secondary_mode_active", ours.secondary_mode_active, guest.secondary_mode_active);
    eq("secondary_mode_requested", ours.secondary_mode_requested, guest.secondary_mode_requested);
    for (int k = 0; k < 16; ++k) eq("fps_history", ours.fps_frame_time_history[k], guest.fps_frame_time_history[k]);
    eq("calculated_fps", ours.calculated_fps, guest.calculated_fps);
    eq("reset_to_30_fps_pending", ours.reset_to_30_fps_pending, guest.reset_to_30_fps_pending);
    eq("request_window_60_hz", ours.request_window_60_hz, guest.request_window_60_hz);
}

}  // namespace
extern "C" void frame_ret();
extern "C" GUEST_ABI std::uint64_t frame_after();
namespace {

// BBHOST_FRAME_STATS: the main loop's work in the frame that is ending. The
// manager's +0x28 still holds the time (gettimeofday, us) it released the last
// frame at, until the update about to run replaces it.
void note_main_frame_work(std::uint64_t m) {
    if (!frame_stats_enabled()) return;
    // The same clock the guest's gettimeofday reads (hle/libc.cpp), on both
    // platforms.
    const std::int64_t now = host_clock_realtime_us();
    const std::int64_t last = get<std::int64_t>(m + 0x28);
    if (last > 0 && now >= last && now - last < 10000000) frame_stats_main_frame(static_cast<std::uint64_t>(now - last), hle_video_flip_count());
}

// The uncapped mode's frame limit: BBHOST_UNCAPPED_FPS=N (30..240) if set;
// else, with V-Sync on, the display's refresh rate (a PC game's V-Sync: no
// frame the display cannot show), and with it off 240. Asked every frame; the
// refresh rate is cached by the window (once a second) and V-Sync read again
// only when a setting changed.
float uncapped_target() {
    static const int forced = [] {
        const char* e = std::getenv("BBHOST_UNCAPPED_FPS");
        const int v = e && *e ? std::atoi(e) : 0;
        return v <= 0 ? 0 : std::clamp(v, 30, kUncappedMaxFps);
    }();
    float limit = static_cast<float>(kUncappedMaxFps);
    if (forced) {
        limit = static_cast<float>(forced);
    } else {
        static std::uint64_t serial = 0;
        static bool vsync = true;
        if (const std::uint64_t s = host_opt_serial(); s != serial) {
            serial = s;
            vsync = host_settings().vsync;
        }
        if (vsync) {
            const float hz = host_window_refresh_hz();  // 0 headless, or when the display does not say
            if (hz >= 30.0f) limit = std::min(limit, hz);
        }
    }
    static float said = -1.0f;
    if (limit != said) {
        said = limit;
        host_log("frame rate: uncapped, at most %.2f fps (%s)", static_cast<double>(limit),
                 forced                                          ? "BBHOST_UNCAPPED_FPS"
                 : limit < static_cast<float>(kUncappedMaxFps) ? "V-Sync on: the display's refresh rate"
                                                                 : "the mode's limit");
    }
    return 1.0f / limit;
}

// BBHOST_TEST_FPS_SWING (a test switch, converted rates only): the target
// moves while the game runs, so one process can be held at two rates or at a
// rate that keeps changing - what a variable refresh display, a busy scene
// or a debugger does to the frame time. The same heap at both rates is what
// a rate-invariance differential compares (tmp tools: rateinv).
//   square:<lo>:<hi>:<s>   <lo> fps for <s> seconds, then <hi> for <s>, ...
//   random:<lo>:<hi>:<s>   a new rate in [lo, hi] every <s> seconds
//   jitter:<lo>:<hi>       a new rate in [lo, hi] every frame
//   sine:<lo>:<hi>:<s>     from lo to hi and back over <s> seconds
// The code immediates and counts keep the mode's values (uncapped: 120's).
struct FpsSwing {
    enum Kind { Off, Square, Random, Jitter, Sine } kind = Off;
    float lo = 0, hi = 0, period = 1;
};
const FpsSwing g_swing = [] {
    FpsSwing s;
    const char* e = std::getenv("BBHOST_TEST_FPS_SWING");
    if (!e || !*e) return s;
    char kind[16] = {};
    float lo = 0, hi = 0, period = 1;
    const int n = std::sscanf(e, "%15[a-z]:%f:%f:%f", kind, &lo, &hi, &period);
    if (n < 3 || !(lo >= 10.0f) || !(hi >= lo) || !(hi <= 1000.0f) || !(period > 0.0f)) {
        host_log("frame rate: BBHOST_TEST_FPS_SWING=%s not understood (kind:lo:hi[:seconds]); ignored", e);
        return s;
    }
    s.kind = !std::strcmp(kind, "square")   ? FpsSwing::Square
             : !std::strcmp(kind, "random") ? FpsSwing::Random
             : !std::strcmp(kind, "jitter") ? FpsSwing::Jitter
             : !std::strcmp(kind, "sine")   ? FpsSwing::Sine
                                            : FpsSwing::Off;
    s.lo = lo;
    s.hi = hi;
    s.period = period;
    if (s.kind == FpsSwing::Off) host_log("frame rate: BBHOST_TEST_FPS_SWING kind %s unknown; ignored", kind);
    else host_log("frame rate: BBHOST_TEST_FPS_SWING %s %.1f..%.1f fps, %.2f s (a test switch)", kind, static_cast<double>(lo),
                  static_cast<double>(hi), static_cast<double>(period));
    return s;
}();

// The swing's target for this frame, or `base` when it is off.
float swing_target(float base) {
    if (g_swing.kind == FpsSwing::Off) return base;
    static const auto t0 = std::chrono::steady_clock::now();
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    static std::uint64_t rng = 0x9e3779b97f4a7c15ull;
    const auto next_unit = [] {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return static_cast<double>(rng >> 11) * (1.0 / 9007199254740992.0);
    };
    double fps = g_swing.lo;
    switch (g_swing.kind) {
        case FpsSwing::Square: fps = (static_cast<std::int64_t>(t / g_swing.period) & 1) ? g_swing.hi : g_swing.lo; break;
        case FpsSwing::Random: {
            static std::int64_t slot = -1;
            static double held = g_swing.lo;
            const auto now_slot = static_cast<std::int64_t>(t / g_swing.period);
            if (now_slot != slot) {
                slot = now_slot;
                held = g_swing.lo + (g_swing.hi - g_swing.lo) * next_unit();
            }
            fps = held;
            break;
        }
        case FpsSwing::Jitter: fps = g_swing.lo + (g_swing.hi - g_swing.lo) * next_unit(); break;
        case FpsSwing::Sine:
            fps = g_swing.lo + (g_swing.hi - g_swing.lo) * 0.5 * (1.0 - std::cos(2.0 * 3.14159265358979 * t / g_swing.period));
            break;
        case FpsSwing::Off: break;
    }
    return static_cast<float>(1.0 / fps);
}

// The converted pace's target for this frame; 0 at 30.
float pace_target() {
    if (g_game_fps == 0) return swing_target(uncapped_target());
    if (g_game_fps == 60) return swing_target(kSixtieth);  // the eboot's own mode 2
    if (g_game_fps > 60) return swing_target(1.0f / static_cast<float>(g_game_fps));
    return 0.0f;
}

// After the update, at every converted rate, for the frame about to run: the
// task step (kStepLoad) - the measured frame time clamped to [target, 1/30]
// as the 60 FPS++ cave has it (a loading screen has no target: the measured
// time), times a plugin's time scale, at most 1/30 - the play time's
// milliseconds from it, and the values that follow the frame's seconds, from
// dt: the same with the uncapped limit as the floor while a loading screen
// has no target. Above 60 the cloth's stiffness scale: the square of its
// solver rate (one substep a frame) against the 60 its part files were
// authored at, (60 dt)^2, at most 1 (below 60 the cloth keeps the 60
// tuning), over a smoothed frame time. Each store is one aligned value; a
// reader on a worker sees the last frame's value or this one's.
void each_frame(const SprjFlipper* f) {
    const float scale = g_scale_by_step.load(std::memory_order_relaxed) ? g_time_scale.load(std::memory_order_relaxed) : 1.0f;
    const auto scaled = [scale](float s) { return scale == 1.0f ? s : std::min(s * scale, kThirtieth); };
    const float step = scaled(std::min(std::max(f->measured_frame_seconds, f->target_frame_seconds), kThirtieth));
    if (step > 0.0f) {  // also NaN: the cave keeps the last step
        frame_tick_frame(step);  // the fixed ticks' count for this frame (engine/frame_tick.h)
        if (g_step_va) put<float>(g_step_va, step);
        if (g_play_ms_va) {
            // The step in whole milliseconds, the rest carried to the next frame.
            g_play_ms_carry += static_cast<double>(step) * 1000.0;
            const auto ms = static_cast<std::int32_t>(g_play_ms_carry);
            g_play_ms_carry -= ms;
            put<std::int32_t>(g_play_ms_va, ms);
        }
    }
    const float floor_s = std::max(f->target_frame_seconds, 1.0f / static_cast<float>(kUncappedMaxFps));
    const float dt = scaled(std::min(std::max(f->measured_frame_seconds, floor_s), kThirtieth));
    if (!(dt > 0.0f)) return;  // also NaN
    frame_rate_step::g_frame_seconds.store(dt, std::memory_order_relaxed);
    const double fps = 1.0 / static_cast<double>(dt);
    for (const EachFrame& e : g_each) put<float>(e.va, rated_value(e.kind, e.was, fps));
    for (const DtSlot& d : g_dt_slots) {
        for (std::uint8_t k = 0; k < d.lanes; ++k) put<float>(d.va + 4u * k, rated_value(d.kind, get<float>(d.src + 4u * k), fps));
    }
    g_cloth_dt += (dt - g_cloth_dt) * 0.1f;
    const float cloth = std::min(1.0f, 60.0f * g_cloth_dt);
    if (g_cloth_va) {
        for (int k = 0; k < 4; ++k) put<float>(g_cloth_va + 4 * static_cast<std::uint64_t>(k), cloth * cloth);
    }
    // BBHOST_FRAME_RATE_LOG=1 (or the older BBHOST_UNCAPPED_LOG=1): what this
    // wrote, once a second.
    static const bool log_each = [] {
        const char* e = std::getenv("BBHOST_FRAME_RATE_LOG");
        const char* u = std::getenv("BBHOST_UNCAPPED_LOG");
        return (e && e[0] == '1') || (u && u[0] == '1');
    }();
    if (log_each) {
        static auto last = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        if (now - last >= std::chrono::seconds(1)) {
            last = now;
            host_log("frame rate: frame %.2f ms (%.0f fps, %.0f measured over 16), momentum kept %.5f, sprint test x%.1f, "
                     "cloth stiffness %.3f",
                     static_cast<double>(dt) * 1000.0, fps, static_cast<double>(f->calculated_fps),
                     static_cast<double>(rated_value(Kind::Keep, 0.95f, fps)),
                     static_cast<double>(rated_value(Kind::Rate, 30.0f, fps)),
                     static_cast<double>(g_cloth_va ? cloth * cloth : 1.0f));
        }
    }
}

// rdi: the frame-time manager.
GUEST_ABI std::int64_t frame_manager_hook(std::uint64_t, const std::uint64_t* saved) {
    auto* f = reinterpret_cast<SprjFlipper*>(static_cast<std::uintptr_t>(saved[5]));
    const bool converted = g_game_fps != 30;
    if (g_source == Source::Ours) {
        note_main_frame_work(saved[5]);
        RealClock clock;
        g_fast_target = converted && (g_bench_uncapped || loading_fast_now());
        sprj_flipper_update(f, clock, pace_target());
        g_fast_target = false;
        // Finish both pacing waits before sampling input or running the frame's
        // host callbacks. Their work belongs to this simulation ID as well.
        host::reflex::on_frame_begin();
    }
    np_test_tick();  // the online test hook wants the main thread, once a frame
    params_tick();   // the param repository walk (engine/params.h)
    lua_events_tick();  // named Lua events queued from any thread (engine/lua_events.h)
    event_flags_tick(); // the one-time check of our flag reader against the game's (engine/event_flags.h)
    player_data_tick(); // the loaded character's numbers, once (engine/player_data.h)
    plugins_frame();    // the plugins' on_frame callbacks (include/bbhost_plugin.h, version 2)
    change_appearance_tick();  // the mirror's editor in character creation's frame (engine/change_appearance.h)
    rebirth_tick();     // the altar's rebirth: the host's half of its talk script (engine/rebirth.h)
    world_chr_tick();   // a test warp a plugin queued this frame (BBHOST_TEST_WARP, engine/world_chr.h)
    loading_tick(hle_video_flip_count());  // the loading screens' lengths and the first in-game frame (engine/loading.h)
    if (g_source == Source::Ours) {
        if (converted) each_frame(f);
        if (g_frames.fetch_add(1, std::memory_order_relaxed) == 0) {
            host_log("frame source: SprjFlipper::Update is ours (0x%llx, %s); the guest's no longer runs",
                     static_cast<unsigned long long>(saved[5]), converted ? mode_name(g_game_fps).c_str() : "the game's own mode");
        }
        return 1;  // done: the stub returns to the caller
    }
    if (g_source == Source::Compare) {
        // The snapshot is taken after the hook's own writes below (at 60 it
        // puts the guest in mode 2 and clears the one-shots), so both sides
        // see the same inputs; frame_after replays it as the guest returns.
        t_pending.flipper = f;
        t_pending.ret = saved[6];
        const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&frame_ret);
    }
    if (g_game_fps != 60) {
        if (g_source == Source::Compare) t_pending.before = *f;
        return 0;
    }
    const std::uint64_t m = saved[5];
    note_main_frame_work(m);
    if (g_frames.fetch_add(1, std::memory_order_relaxed) == 0) {
        host_log("frame rate: the frame-time manager is 0x%llx, mode %u (one-shot %u), interval %g s",
                 static_cast<unsigned long long>(m), get<std::uint32_t>(m + 8), get<std::uint32_t>(m + 0xc), get<float>(m + 0x18));
    }
    put<std::uint32_t>(m + 8, kModeSixty);
    put<std::uint32_t>(m + 0xc, kModeSixty);
    put<std::int32_t>(m + 0x2bc, -1);
    put<std::int32_t>(m + 0x2c0, -1);
    put<std::uint8_t>(m + 0x271, 0);
    put<std::uint8_t>(m + 0x272, 0);
    put<std::uint8_t>(m + 0x2c4, 0);
    if (g_source == Source::Compare) t_pending.before = *f;
    return 0;
}

// The Frame rate option as the game started with it. BBHOST_GAME_FPS=30, 60,
// 90 or 0 (also "uncapped": atoi reads 0) overrides it.
int wanted_fps() {
    if (const char* e = std::getenv("BBHOST_GAME_FPS"); e && *e) return mode_of(std::atoi(e));
    return mode_of(host_settings().frame_cap);
}

// The follow camera's tuning above 30 fps: the game's own (Original) unless
// BBHOST_CAMERA_TUNING=fps++ (or kyo, 0) asks for Kyo's 60 FPS++ retune. The
// menus no longer offer it; a saved camera_tuning is ignored.
bool camera_original() {
    if (const char* e = std::getenv("BBHOST_CAMERA_TUNING"); e && *e) {
        const std::string v(e);
        if (v == "original" || v == "orig" || v == "1") return true;
        if (v == "fps++" || v == "kyo" || v == "0") return false;
        host_log("frame rate: BBHOST_CAMERA_TUNING=%s is neither fps++ nor original; Original runs", e);
    }
    return true;
}

// Whether every site of a group holds the list's old bytes; the first that
// does not is named.
template <std::size_t N>
bool sites_hold(ElfImage* image, const Site (&sites)[N], const char* group) {
    for (const Site& site : sites) {
        const std::vector<std::uint8_t> was = hex_bytes(site.was);
        const std::uint64_t at = image->mem.slide + (site.at - kPreferredGuestSlide);
        if (std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), was.data(), was.size()) != 0) {
            host_log("frame rate: 0x%x does not hold %s's old bytes (%zu bytes); its %zu code rewrites refused", site.at, group,
                     was.size(), N);
            return false;
        }
    }
    return true;
}

// Checks every value and site, picks the mode they allow, then writes. 30
// (nothing written) if a value table does not hold the eboot's values; 60
// without the code rewrites if those do not hold theirs; 60 if 90's or
// uncapped's do not. Never part of a group.
int apply_rate(ElfImage* image, int want) {
    const auto va = [&](std::uint32_t bn) { return image->mem.slide + (bn - kPreferredGuestSlide); };
    for (const Constant& c : kConstants) {
        if (get<std::uint32_t>(va(c.at)) != c.was) {
            host_log("frame rate: refused, 0x%x holds 0x%08x, not 0x%08x", c.at, get<std::uint32_t>(va(c.at)), c.was);
            return 30;
        }
    }
    for (const Counted& c : kCounts) {
        std::uint8_t was[8];
        count_was(c, was);
        if (std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va(c.at))), was, c.n) != 0) {
            host_log("frame rate: refused, 0x%x does not hold %u", c.at, c.was);
            return 30;
        }
    }
    for (const Bytes& b : kBytes) {
        if (std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va(b.at))), b.was, b.n) != 0) {
            host_log("frame rate: refused, 0x%x does not hold the expected %u bytes", b.at, b.n);
            return 30;
        }
    }
    // The code rewrites, by group: all of a group or none of it. The
    // frame-time sites read the slot the 60 FPS++ cave keeps, so they go with
    // the 60 FPS++ sites; the cloth's caves are for rates above 60.
    static const bool sites_on = [] {
        const char* e = std::getenv("BBHOST_SIXTY_SITES");
        return !(e && e[0] == '0');
    }();
    if (!sites_on) host_log("frame rate: BBHOST_SIXTY_SITES=0, the 60 FPS++ code rewrites left out");
    const bool sixty_ok = sites_on && sites_hold(image, kSixtySites, "the 60 FPS++ list") &&
                          sites_hold(image, kFrameTimeSites, "the frame-time sites");
    int mode = want;
    if (mode != 60) {
        const bool cloth_ok = sixty_ok && sites_hold(image, kClothSites, "the cloth caves") &&
                              std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va(kClothScalars))),
                                          kClothScalarsWas, sizeof(kClothScalarsWas)) == 0;
        if (!cloth_ok) {
            host_log("frame rate: %s needs the 60 FPS++, frame-time and cloth code rewrites, which were refused; the game "
                     "runs at 60",
                     mode_name(mode).c_str());
            mode = 60;
        }
    }

    // The 1.09 image's executable segment ends at 0x54d96dc (its first PT_LOAD);
    // a site past it is in the read-write data segment, and its page goes back
    // to read-write - the frame-time slot (0x575a908) is written every frame.
    constexpr std::uint32_t kCodeEnd = 0x54d96dc;
    const auto write = [&](std::uint32_t bn, const void* src, std::size_t n) {
        const std::uint64_t at = va(bn), lo = at & ~0xfffull, hi = (at + n + 0xfff) & ~0xfffull;
        guest_protect_rwx(&image->mem, lo, hi - lo);
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(at)), src, n);
        if (bn < kCodeEnd) guest_protect_rx(&image->mem, lo, hi - lo);
        else guest_protect_rw(&image->mem, lo, hi - lo);
    };
    // The Camera tuning: Original leaves Kyo's camera sites and constants out.
    const bool original_camera = camera_original();
    const auto write_sites = [&](const auto& sites) {
        std::size_t n = 0;
        for (const Site& site : sites) {
            if (original_camera && kyo_camera_site(site.at)) continue;
            const std::vector<std::uint8_t> now = hex_bytes(site.now);
            write(site.at, now.data(), now.size());
            ++n;
        }
        return n;
    };
    // The reference rate's values for uncapped's code immediates and frame
    // counts; the sound thread keeps up with the fastest frame.
    const int fps = mode == 0 ? kUncappedReferenceFps : mode;
    // A rated constant's rule gives the table's own value at 60.
    std::size_t rated_fixed = 0, camera_kept = 0;
    for (const Constant& c : kConstants) {
        if (original_camera && kyo_camera_constant(c.at)) {
            ++camera_kept;
            continue;
        }
        std::uint32_t now = c.now;
        if (const Rated* r = rated(c.at); r && mode != 60) {
            now = float_bits(rated_value(r->kind, bits_float(c.was), fps));
            rated_fixed += r->each_frame ? 0 : 1;
        }
        write(c.at, &now, 4);
    }
    for (const Counted& c : kCounts) {
        std::uint8_t now[8];
        count_bytes(c, c.kind == Count::Digits && mode == 0 ? kUncappedMaxFps : fps, now);
        write(c.at, now, c.n);
    }
    for (const Bytes& b : kBytes) write(b.at, b.now, b.n);
    // After the constants: two frame-time sites rewrite instructions whose
    // immediates kConstants converts (0x1bf9b74, 0x222bcd7, for a run
    // without the code rewrites).
    if (sixty_ok) {
        g_sites_applied = write_sites(kSixtySites);
        g_frame_time_applied = write_sites(kFrameTimeSites);
        // The task step and the play time's count, ours: written by our
        // SprjFlipper::Update every frame, in 16 bytes of the image's tail
        // (with the guest's, BBHOST_FRAME_SOURCE=0 or 2 at 60, the cave and
        // the game's add stay).
        const std::uint64_t block = image->tail_va + image->tail_used + kPreferredGuestSlide;  // as a BN address
        const std::uint64_t step_bn = block, play_bn = block + 8;
        const std::int64_t step_disp = static_cast<std::int64_t>(step_bn) - static_cast<std::int64_t>(kStepLoad + 8);
        const std::int64_t play_disp = static_cast<std::int64_t>(play_bn) - static_cast<std::int64_t>(kPlayTimeAdd + 6);
        const auto holds = [&](std::uint32_t at, const char* hex) {
            const std::vector<std::uint8_t> was = hex_bytes(hex);
            return std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va(at))), was.data(), was.size()) == 0;
        };
        const auto rel32 = [](std::int64_t d) { return d >= INT32_MIN && d <= INT32_MAX; };
        if (g_source != Source::Ours) {
            host_log("frame rate: the guest's SprjFlipper::Update runs; the task step and the play time stay the game's");
        } else if (image->tail_used + 16 > image->tail_size || !rel32(step_disp) || !rel32(play_disp)) {
            host_log("frame rate: no room in the image's tail; the task step and the play time stay the game's");
        } else {
            image->tail_used += 16;
            const auto le = [](std::int64_t d, std::uint8_t* out) {
                const auto u = static_cast<std::uint32_t>(static_cast<std::int32_t>(d));
                for (int k = 0; k < 4; ++k) out[k] = static_cast<std::uint8_t>(u >> (8 * k));
            };
            if (holds(kStepLoad, kStepLoadWas)) {
                g_step_va = va(static_cast<std::uint32_t>(step_bn));
                put<float>(g_step_va, 1.0f / static_cast<float>(fps));
                std::uint8_t now[21] = {0xf3, 0x0f, 0x10, 0x05, 0, 0, 0, 0,               // movss xmm0, [step]
                                        0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,  // nop8
                                        0x0f, 0x1f, 0x44, 0x00, 0x00};                   // nop5
                le(step_disp, now + 4);
                write(kStepLoad, now, sizeof(now));
                g_scale_by_step.store(true, std::memory_order_relaxed);
            } else {
                host_log("frame rate: 0x%x does not hold the 60 FPS++ cave's step; the cave keeps computing it", kStepLoad);
            }
            if (holds(kPlayTimeAdd, kPlayTimeAddWas)) {
                g_play_ms_va = va(static_cast<std::uint32_t>(play_bn));
                put<std::int32_t>(g_play_ms_va, static_cast<std::int32_t>(1000 / fps));
                std::uint8_t now[13] = {0x8b, 0x0d, 0, 0, 0, 0,                    // mov ecx, [count]
                                        0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00};  // nop7
                le(play_disp, now + 2);
                write(kPlayTimeAdd, now, sizeof(now));
            } else {
                host_log("frame rate: 0x%x does not hold the play time's add; the game's stays", kPlayTimeAdd);
            }
        }
        // The redirected loads (kDtLoads): every instruction checked, then
        // the slots placed and the displacements pointed at them.
        bool dt_ok = sizeof(kDtLoads) > 0;
        for (const DtLoad& d : kDtLoads) {
            const std::vector<std::uint8_t> was = hex_bytes(d.bytes);
            const auto* p = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(va(d.insn)));
            std::int32_t disp;
            std::memcpy(&disp, p + d.disp, 4);
            if (was.size() != d.len || std::memcmp(p, was.data(), was.size()) != 0 ||
                static_cast<std::int64_t>(d.insn) + d.len + disp != static_cast<std::int64_t>(d.constant)) {
                host_log("frame rate: 0x%x is not the load of 0x%x the redirected loads expect; the %zu are left out", d.insn,
                         d.constant, sizeof(kDtLoads) / sizeof(kDtLoads[0]));
                dt_ok = false;
                break;
            }
        }
        if (dt_ok) {
            std::vector<std::pair<std::uint64_t, std::uint64_t>> keys;  // (constant | kind | lanes, slot BN address)
            const auto key_of = [](const DtLoad& d) {
                return static_cast<std::uint64_t>(d.constant) << 16 | static_cast<std::uint64_t>(d.kind) << 8 | d.lanes;
            };
            for (const DtLoad& d : kDtLoads) {
                if (std::none_of(keys.begin(), keys.end(), [&](const auto& k) { return k.first == key_of(d); })) keys.push_back({key_of(d), 0});
            }
            const std::uint64_t start = (image->tail_used + 15) & ~std::uint64_t(15);
            if (start + 16 * keys.size() > image->tail_size) {
                host_log("frame rate: no room in the image's tail for %zu redirected loads' slots; left out", keys.size());
            } else {
                image->tail_used = start + 16 * keys.size();
                for (std::size_t i = 0; i < keys.size(); ++i) {
                    keys[i].second = image->tail_va + start + 16 * i + kPreferredGuestSlide;  // as a BN address
                    DtSlot slot{};
                    slot.va = va(static_cast<std::uint32_t>(keys[i].second));
                    const auto constant = static_cast<std::uint32_t>(keys[i].first >> 16);
                    slot.kind = static_cast<Kind>((keys[i].first >> 8) & 0xff);
                    slot.lanes = static_cast<std::uint8_t>(keys[i].first & 0xff);
                    slot.src = va(constant);
                    for (std::uint8_t k = 0; k < slot.lanes; ++k)
                        put<float>(slot.va + 4u * k, rated_value(slot.kind, get<float>(slot.src + 4u * k), fps));
                    g_dt_slots.push_back(slot);
                }
                for (const DtLoad& d : kDtLoads) {
                    const auto it = std::find_if(keys.begin(), keys.end(), [&](const auto& k) { return k.first == key_of(d); });
                    const std::int64_t nd = static_cast<std::int64_t>(it->second) - (static_cast<std::int64_t>(d.insn) + d.len);
                    if (nd < INT32_MIN || nd > INT32_MAX) continue;  // the tail is within reach of all of the code
                    const auto d32 = static_cast<std::int32_t>(nd);
                    write(d.insn + d.disp, &d32, 4);
                    ++g_dt_loads_applied;
                }
            }
        }
    }
    // The camera's weights hold Kyo's 60 fps values (kConstants' defaults,
    // and his stores with the sites) or, with Original, the game's 30 fps
    // ones: the decomp camera converts them from a frame of that rate to each
    // frame's length (decomp/camera/follow_camera.cpp), from the frame time
    // each_frame keeps, which needs no site.
    frame_rate_step::g_camera_tuned_fps.store(original_camera ? 30 : 60, std::memory_order_relaxed);
    if (original_camera) {
        host_log("frame rate: the follow camera's tuning is Original: the game's own values, converted from its 30 fps "
                 "frames to each frame's length (Kyo's %zu camera sites and the constructor's %zu defaults left as the "
                 "game has them)",
                 sizeof(kKyoCameraSites) / sizeof(kKyoCameraSites[0]), camera_kept);
    } else {
        host_log("frame rate: the follow camera's tuning is FPS++: Kyo's 60 fps retune from his 60 FPS++ list, converted "
                 "from 60 fps frames to each frame's length (Camera tuning: Original for the game's own)");
    }
    if (mode != 60) {
        g_cloth_applied = write_sites(kClothSites);
        g_cloth_va = va(kClothScalars);
        const float scale = std::min(1.0f, 60.0f / static_cast<float>(fps));
        const float scalars[4] = {scale * scale, scale * scale, scale * scale, scale * scale};
        write(kClothScalars, scalars, sizeof(scalars));
        g_cloth_dt = 1.0f / static_cast<float>(fps);
    }
    // The every-frame values: their pages stay writable for good (they are
    // .rodata in the executable segment, RWX in case a page also holds code),
    // whatever another writer does to them later - live resolution writes
    // the plate sizes beside the sprint test's value and puts that page back
    // to read-only (core/memory.h).
    std::vector<std::uint64_t> pages;
    for (const Rated& r : kRated) {
        if (!r.each_frame) continue;
        const Constant* c = std::find_if(std::begin(kConstants), std::end(kConstants), [&](const Constant& k) { return k.at == r.at; });
        g_each.push_back({va(r.at), bits_float(c->was), r.kind});
        pages.push_back(va(r.at) & ~0xfffull);
    }
    if (g_cloth_va) pages.push_back(g_cloth_va & ~0xfffull);
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    for (const std::uint64_t p : pages) {
        if (p < va(kCodeEnd)) guest_keep_writable(&image->mem, p);  // .data pages are writable already
    }
    host_log("frame rate: %zu values%s follow the frame time every frame (%zu pages of .rodata kept writable), and %zu loads "
             "read %zu slots of ours that do; %zu other constants follow the rate (code immediates and counts at %d's values), "
             "the rest at their 60 values",
             g_each.size(), g_cloth_va ? " and the cloth's stiffness scale" : "", pages.size(), g_dt_loads_applied, g_dt_slots.size(),
             rated_fixed, fps);
    return mode;
}

}  // namespace

// As the guest's update returns (compare mode): ours on the snapshot, the
// fields compared, then on to the caller.
extern "C" GUEST_ABI std::uint64_t frame_after() {
    Pending& p = t_pending;
    SprjFlipper replay = p.before;
    ReplayClock clock(static_cast<std::int64_t>(p.flipper->current_frame_microseconds));
    const std::uint64_t slide = g_slide;
    g_slide = 0;  // the window request is not replayed (it would be written twice)
    sprj_flipper_update(&replay, clock, g_game_fps == 60 ? kSixtieth : 0.0f);  // compare runs at 30 and 60 only
    g_slide = slide;
    replay.request_window_60_hz = p.flipper->request_window_60_hz;
    compare_fields(replay, *p.flipper);
    const std::uint64_t n = g_compared.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n % 1800 == 0 || n == 60) {
        host_log("frame source: compare %llu frames, %llu field mismatches", static_cast<unsigned long long>(n),
                 static_cast<unsigned long long>(g_mismatched.load()));
    }
    return p.ret;
}
extern "C" void* g_frame_after_thunk = nullptr;
#if defined(_WIN32)
#define FR_TYPE_(n)
#else
#define FR_TYPE_(n) ".type " #n ", @function\n"
#endif
asm(".text\n.globl frame_ret\n" FR_TYPE_(frame_ret) "frame_ret:\n"
    "push %rax\n push %rdx\n call *g_frame_after_thunk(%rip)\n"
    "mov %rax, %r11\n pop %rdx\n pop %rax\n jmp *%r11\n");

int frame_rate_game_fps() { return g_game_fps; }

void frame_rate_install(ElfImage* image) {
    const std::uint64_t at = image->mem.slide + (kFrameManager - kPreferredGuestSlide);
    g_slide = image->mem.slide;
    loading_bind(image->mem.slide);
    // BBHOST_FRAME_SOURCE: 1 (default) SprjFlipper::Update is ours; 0 the
    // guest's runs with the mode hook; 2 compare, the guest's runs and ours
    // replays each frame beside it.
    if (const char* e = std::getenv("BBHOST_FRAME_SOURCE"); e && e[0]) {
        g_source = e[0] == '0' ? Source::Guest : e[0] == '2' ? Source::Compare : Source::Ours;
    }
    const int want = wanted_fps();
    // 90 and uncapped set a target the game's own modes do not have: ours
    // runs them (compare and the guest's run at 30 and 60).
    if ((want > 60 || want == 0) && g_source != Source::Ours) {
        host_log("frame source: BBHOST_FRAME_SOURCE runs the guest's update at 30 and 60 only; ours runs %s", mode_name(want).c_str());
        g_source = Source::Ours;
    }
    if (g_source == Source::Compare) g_frame_after_thunk = hle_wrap_fn(reinterpret_cast<void*>(&frame_after));
    if (want == 30) {
        host_log("frame rate: the game's own 30; the follow camera is the game's own (the Camera tuning, %s, applies above "
                 "30)",
                 camera_original() ? "Original" : "FPS++");
        // Ours runs the game's own mode at 30 (or the guest's does with
        // BBHOST_FRAME_SOURCE=0, when only the online test tick needs the hook).
        {  // ours runs the game's own mode at 30 (the param walk and the online tick ride on the same hook)
            if (!engine_prologue_hook(image, at, kFrameManagerPrologue, sizeof(kFrameManagerPrologue),
                                      reinterpret_cast<void*>(&frame_manager_hook))) {
                host_log("frame source: refused, 0x%llx is not the frame-time manager's prologue; the guest's runs",
                         static_cast<unsigned long long>(kFrameManager));
                g_source = Source::Guest;
            }
        }
        hle_video_set_game_pace(30);
        return;
    }
    if (!engine_prologue_hook(image, at, kFrameManagerPrologue, sizeof(kFrameManagerPrologue),
                              reinterpret_cast<void*>(&frame_manager_hook))) {
        host_log("frame rate: refused, 0x%llx is not the frame-time manager's prologue; the game stays at 30",
                 static_cast<unsigned long long>(kFrameManager));
        hle_video_set_game_pace(30);
        return;
    }
    const int got = apply_rate(image, want);
    if (got == 30) {
        host_log("frame rate: the game's steps could not be converted; staying at 30");
        hle_video_set_game_pace(30);
        return;  // the hook keeps mode 2 only while g_game_fps says 60
    }
    g_game_fps = got;
    hle_video_set_game_pace(got);  // above 60 and uncapped: flips complete at once (hle/video.cpp)
    frame_tick_install(image);
    const std::size_t kCountsN = sizeof(kCounts) / sizeof(kCounts[0]), kBytesN = sizeof(kBytes) / sizeof(kBytes[0]);
    const std::size_t constants = sizeof(kConstants) / sizeof(kConstants[0]);
    char target[96];
    if (got == 0) {
        std::snprintf(target, sizeof(target), "the step following the measured frame time, clamped to 1/%d..1/30 s", kUncappedMaxFps);
    } else {
        std::snprintf(target, sizeof(target), "a 1/%d s target", got);
    }
    host_log("frame rate: %s (the frame-time manager's mode 2, %s; %zu constants and %zu other values converted, %zu 60 FPS++, "
             "%zu frame-time and %zu cloth code sites%s%s)",
             mode_name(got).c_str(), target, constants, kCountsN + kBytesN, g_sites_applied, g_frame_time_applied, g_cloth_applied,
             g_step_va ? ", the task step ours" : "", g_play_ms_va ? ", the play time's add" : "");
    if (got != want)
        host_log("frame rate: %s was asked for; %s runs (the reason is above)", mode_name(want).c_str(), mode_name(got).c_str());
}

void frame_rate_set_time_scale(float scale) {
    if (!(scale >= 0.1f)) scale = 0.1f;  // also NaN
    if (scale > 4.0f) scale = 4.0f;
    g_time_scale.store(scale, std::memory_order_relaxed);
    // An eased change sets it every frame: logged in steps of a tenth.
    static float logged = 1.0f;
    if (std::fabs(scale - logged) >= 0.1f || (scale == 1.0f && logged != 1.0f)) {
        logged = scale;
        host_log("frame rate: time scale %.2f", static_cast<double>(scale));
    }
}

std::string frame_rate_limiter_window() {
    LimiterTuning& t = g_limiter;
    const std::uint64_t waits = t.waits.exchange(0, std::memory_order_relaxed);
    const std::uint64_t sleeps = t.sleeps.exchange(0, std::memory_order_relaxed);
    const std::uint64_t spin = t.spin_us.exchange(0, std::memory_order_relaxed);
    const std::uint64_t late = t.sleep_late_us.exchange(0, std::memory_order_relaxed);
    if (!waits && !sleeps) return "limiter idle";
    char buf[220];
    std::snprintf(buf, sizeof(buf), "limiter margin %lld us (%s), sleeps %llu late avg %.0f us p99 %lld us, spin %.2f ms a frame over %llu",
                  static_cast<long long>(t.shown_margin_us.load(std::memory_order_relaxed)),
                  t.fixed_us >= 0 ? "BBHOST_LIMITER_MARGIN_US" : "learnt", static_cast<unsigned long long>(sleeps),
                  sleeps ? static_cast<double>(late) / static_cast<double>(sleeps) : 0.0,
                  static_cast<long long>(t.shown_late_p99_us.load(std::memory_order_relaxed)),
                  waits ? static_cast<double>(spin) / 1000.0 / static_cast<double>(waits) : 0.0,
                  static_cast<unsigned long long>(waits));
    return buf;
}
