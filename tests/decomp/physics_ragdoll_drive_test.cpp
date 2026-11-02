// The ragdoll keyframe controller's drive (decomp/physics/ragdoll_drive.cpp)
// beside the game's own, sub_ae4600, in the loaded eboot - the keyframe
// hierarchy step both hand on to (0xae9b90) a recorder here, which keeps its
// four arguments, the step it finds in the bodies' struct and the two fields
// the drive stores before the call:
//   - with the frame clock at the game's 1/30 (as at 30): on generated
//     controllers - the weight, the ragdoll's bodies and the step each above 0,
//     0, -0, below, NaN, a denormal, an infinity; the constructor's 1/30 often;
//     the controllers' own vtable, Havok Behavior's and garbage - ours stops
//     where the game's stops, and where it goes on hands the recorder what the
//     game's hands, bit for bit, and leaves every byte of the controller as the
//     game's leaves it;
//   - at frames of other lengths (60, 90, 120, 144, 165, 240, 360 and 1000 fps,
//     and any between 1/1000 and 1/30 s): a controller whose step is the
//     constructor's 1/30 hands the keyframe step the clock's step, bit for bit,
//     and holds its 1/30 again after; one whose step is anything else - the
//     step Havok Behavior's controller hands it - is the game's throughout;
//   - the compare stand-in's reading (BBHOST_DECOMP_COMPARE=1) agrees with the
//     game's drive on the same controllers, and leaves them as the game's drive
//     handed ours' step leaves them;
// and each of the source's mutants caught, the function's bytes the ones ours
// mirrors. Skips without the 1.09 eboot.
#define DECOMP_PHYSICS_RAGDOLL_DRIVE_MUTANTS
int g_ragdoll_drive_mutant = 0;
#include "decomp/physics/ragdoll_drive.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

eboot_kit::Rng g_rng(0xae4600);

// What the keyframe step was handed.
struct Seen {
    int calls = 0;
    u64 key_frame = 0, bodies = 0, palette = 0, indices = 0;
    u32 step = 0;
    u64 pose = 0, weights = 0;  // the keyframe's +0x30 and the bodies' +0x20 at the call
    bool operator==(const Seen&) const = default;
};
Seen g_seen;

GUEST_ABI u64 key_frame_recorder(u64 key_frame, u64 bodies, u64 palette, u64 indices) {
    ++g_seen.calls;
    g_seen.key_frame = key_frame;
    g_seen.bodies = bodies;
    g_seen.palette = palette;
    g_seen.indices = indices;
    g_seen.step = load<u32>(bodies + 0x10);
    g_seen.pose = load<u64>(key_frame + 0x30);
    g_seen.weights = load<u64>(bodies + 0x20);
    return 0x5eed;
}

// The controller (0xe0 bytes, with room around it) and the ragdoll its +0x50
// points at - one of each, at fixed addresses, so the game's run and ours see
// the same pointers.
struct alignas(16) Memory {
    u8 controller[0x100];
    u8 ragdoll[0x40];
};
Memory g_mem, g_start, g_after_game;
u64 self() { return reinterpret_cast<u64>(g_mem.controller); }

using Drive = GUEST_ABI void (*)(u64);
Drive g_game = nullptr;

int g_failed = 0;
bool g_quiet = false;
void fail(int c, const char* what) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: case %d: %s\n", c, what);
}

float special(float lo, float hi) {
    eboot_kit::Rng& r = g_rng;
    r.specials = true;
    const float v = r.value(lo, hi);
    r.specials = false;
    return v;
}

// A generated controller: garbage everywhere the drive does not read, its
// fields often what the game builds.
void generate() {
    for (u8& b : g_start.controller) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_start.ragdoll) b = static_cast<u8>(g_rng.u32());
    const auto put = [](u64 o, auto v) { std::memcpy(g_start.controller + o, &v, sizeof v); };
    const u64 vtable = g_rng.chance(0.6) ? 0x5661640 : g_rng.chance(0.5) ? kBehaviorController : g_rng.next();
    put(0, vtable);
    put(kWeight, g_rng.chance(0.6) ? 1.0f : special(-2.0f, 2.0f));
    put(kBodies, reinterpret_cast<u64>(g_mem.ragdoll));
    const u64 bodies = g_rng.chance(0.15) ? 0 : g_rng.next() | 1;
    std::memcpy(g_start.ragdoll + kRagdollBodies, &bodies, 8);
    float step = frame_clock::kThirtieth;
    if (g_rng.chance(0.4)) step = g_rng.chance(0.5) ? 1.0f / 60.0f : special(-1.0f, 1.0f);
    put(kStep, step);
}

// The game's drive and ours from the same controller: what each handed the
// keyframe step, and the controller each left.
void both(Seen& game, Seen& ours) {
    g_mem = g_start;
    g_seen = Seen{};
    g_game(self());
    game = g_seen;
    g_after_game = g_mem;
    g_mem = g_start;
    g_seen = Seen{};
    ragdoll_drive(self());
    ours = g_seen;
}

constexpr float kSteps[] = {1.0f / 60.0f,  1.0f / 90.0f,  1.0f / 120.0f, 1.0f / 144.0f,
                            1.0f / 165.0f, 1.0f / 240.0f, 1.0f / 360.0f, 1.0f / 1000.0f};
constexpr int kStepCount = sizeof kSteps / sizeof kSteps[0];

// One case: the clock at `dt`.
void one(int c, float dt) {
    frame_clock::publish(dt);
    generate();
    Seen game, ours;
    both(game, ours);
    const bool framed = game.calls == 1 && game.step == frame_clock::kThirtiethBits && bits(dt) != frame_clock::kThirtiethBits;
    Seen want = game;
    if (framed) want.step = bits(dt);
    if (!(ours == want)) {
        fail(c, ours.calls != game.calls ? "stops where the game's goes on, or the other way"
                : framed                 ? "the frame's step, or another argument, at a converted frame"
                                         : "what the keyframe step is handed");
    }
    if (std::memcmp(&g_mem, &g_after_game, sizeof g_mem) != 0) fail(c, "the controller after the drive");
}

long run(int cases) {
    long n = 0;
    for (int c = 0; c < cases; ++c, ++n) {
        // At 30 (the clock's own 1/30), at the rates' frames, at any between.
        const int k = g_rng.range(-2, kStepCount);
        const float dt = k < 0 ? frame_clock::kThirtieth : k < kStepCount ? kSteps[k] : g_rng.uniform(0.001f, frame_clock::kThirtieth);
        one(c, dt);
    }
    frame_clock::g_dt.store(frame_clock::kThirtieth);
    frame_clock::g_frames30.store(1.0f);
    return n;
}

// The compare stand-in's reading against the game's drive, which it runs here
// directly: it must agree on every controller, and leave the controller as the
// game's drive handed ours' step leaves it.
long compare(int cases) {
    long n = 0;
    for (int c = 0; c < cases; ++c, ++n) {
        const int k = g_rng.range(-2, kStepCount - 1);
        const float dt = k < 0 ? frame_clock::kThirtieth : kSteps[k];
        frame_clock::publish(dt);
        generate();
        // ours, for what the game's drive is to be handed
        Seen ours;
        g_mem = g_start;
        g_seen = Seen{};
        ragdoll_drive(self());
        ours = g_seen;
        const Memory after_ours = g_mem;
        g_mem = g_start;
        g_seen = Seen{};
        if (!compare_one(self(), [](u64 s) { g_game(s); })) fail(c, "the compare stand-in disagrees with the game's drive");
        if (!(g_seen == ours)) fail(c, "the compare stand-in did not hand the game's drive ours' step");
        if (std::memcmp(&g_mem, &after_ours, sizeof g_mem) != 0) fail(c, "the compare stand-in left the controller otherwise");
    }
    frame_clock::g_dt.store(frame_clock::kThirtieth);
    frame_clock::g_frames30.store(1.0f);
    return n;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_ragdoll_drive_mutant = mutant;
    g_quiet = true;
    run(3000);
    g_quiet = false;
    g_ragdoll_drive_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("physics_ragdoll_drive_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 200000;
    const u8* at = eboot_kit::file_at("physics_ragdoll_drive_test", kDrive);
    if (!at || std::memcmp(at, kEntry, sizeof kEntry) != 0) fail(-1, "the entry's instructions");
    if (!at || !ragdoll_drive_body_ok(at)) fail(-1, "the function's bytes are not the ones ours mirrors");
    eboot_kit::stub(kKeyFrameStep, reinterpret_cast<const void*>(&key_frame_recorder));
    g_game = eboot_kit::fn<Drive>(kDrive);

    const long total = run(cases);
    const u64 framed = g_framed.load(), thirtieth = g_thirtieth.load(), handed = g_handed.load();
    if (!framed || !thirtieth || !handed) fail(-1, "a kind of drive never ran (framed, at 1/30, handed)");
    const long compared = compare(cases / 10);
    for (const int mutant : {1, 2, 3, 4, 5, 6, 7}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("physics_ragdoll_drive_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("physics_ragdoll_drive_test: ok - %ld cases (%llu drives by the frame's step, %llu at 1/30, %llu by a step handed "
                "on), %ld through the compare stand-in, 7 mutants caught\n",
                total, static_cast<ull>(framed), static_cast<ull>(thirtieth), static_cast<ull>(handed), compared);
    return 0;
}
