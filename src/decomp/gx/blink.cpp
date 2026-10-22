// The GX lights' own blink - the flicker of the torches, braziers and candles -
// at 30's pace at any frame rate.
//
// A GX light keeps a blink (point lights, vtable 0x57b7670, at +0x130; spot
// lights, 0x57b76d0, at +0x1ac; 0x57b7610's at +0x150 - the map's BTL lights
// and the effects' light modules alike), which sub_12a25f0 steps once a frame
// with the task step, from the light's update (its vtable +0x18, which the
// light bank's update, sub_263ea40, runs in the GX world's update, sub_21972e0,
// after the world's clock and before the effects' update):
//   +0x14, +0x18  the shortest and the longest cycle (both 0: no blink, the
//                 factor 1, the time left as it is)
//   +0x1c, +0x20  the factor's low and high
//   +0x24         the cycle's length; +0x28 the time in it
//   +0x2c         the factor, which the light's colour is scaled by
//   +0x4..+0x10   the light's own xorshift128 state: nothing else draws from it
// Once the time is past the length, a new cycle begins: a length drawn between
// the two from the light's generator, the time 0. The factor is a triangle over
// the cycle, low + (high - low) |1 - 2 time / length| - high at its ends, low at
// its middle - and then the time moves on by the step.
// A cycle ends on the first update past its length, and the time past it is
// dropped: half a frame on average, 1/60 s at 30 but 1/240 s at 120. So above
// 30 the cycles are shorter and the lights blink faster (4.95 cycles a game
// second at 30 in the Grand Cathedral's boss room, 5.24 at 60, 5.31 at 120).
//
// Ours begins a cycle only on a frame that takes a whole effect step - the
// effects' schedule (engine/sfx_schedule.h, decomp_sfx_between_steps): every
// frame at 30, one in four at 120. On a frame between whole steps, a light past
// its cycle's length holds the factor its next cycle begins with (its cycle's
// end value, the high), its time moving on, and draws nothing: the factor comes
// from the game's own update run on a copy, and the light's generator is left
// as it is, so the next whole step begins the cycle with the length the copy
// drew. The cycles then last what they last at 30 - a whole number of 1/30 s,
// the time past the length up to the next whole step held at the high - and
// each light draws one length a cycle as at 30. Within a cycle, and on every
// frame that takes a whole step, the update is the game's. At 30 every frame
// takes a whole step: the game's, as wherever bbhost does not schedule the
// effects' steps (effect steps frame); with effect steps 60 the cycles begin on
// whole 1/60 s, 60's pace.
//
// BBHOST_BLINK_HOLD=0: the game's update on every frame.
#include "decomp/decomp.h"
#include "decomp/guest.h"
#include "decomp/sfx/step_gate.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>

#include "log.h"

using namespace decomp;

#ifdef DECOMP_GX_BLINK_MUTANTS
extern int g_blink_mutant;
#define MUTANT(n) (g_blink_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kBlinkUpdate = 0x12a25f0;
constexpr u64 kShortest = 0x14, kLongest = 0x18, kHigh = 0x20, kLength = 0x24, kTime = 0x28, kFactor = 0x2c;
constexpr u64 kBlinkBytes = 0x30;  // what the update reads and writes lies in +0 .. +0x30

using BlinkUpdate = GUEST_ABI void (*)(u64 blink, float step);

void* g_game_blink = nullptr;
bool g_hold_on = true;
std::atomic<u64> g_begun{0}, g_held{0};

}  // namespace

DECOMP_LEAF void blink_update(u64 blink, float step) {
    const auto game = reinterpret_cast<BlinkUpdate>(g_game_blink);
    if (!g_hold_on || MUTANT(1)) return game(blink, step);
    // no blink: the game's sets the factor 1
    if (load<float>(blink + kShortest) == 0.0f && load<float>(blink + kLongest) == 0.0f && !MUTANT(2)) return game(blink, step);
    const float length = load<float>(blink + kLength), time = load<float>(blink + kTime);
    // the game's test (a NaN time or length is not past)
    if (!(MUTANT(3) ? !(time <= length) : time > length)) return game(blink, step);
    if (!decomp_sfx_between_steps() && !MUTANT(4)) {
        g_begun.fetch_add(1, std::memory_order_relaxed);
        return game(blink, step);  // a whole step: the game's new cycle
    }
    // Between whole steps, past the cycle's end: its next cycle's first factor,
    // the game's update on a copy; the time moves on.
    if (MUTANT(7)) {
        game(blink, 0.0f);  // the light's own generator drawn
    } else if (!MUTANT(5)) {
        alignas(16) u8 next[kBlinkBytes];
        const u64 at = reinterpret_cast<u64>(next);
        for (u64 o = 0; o < kBlinkBytes; o += 8) store<u64>(at + o, load<u64>(blink + o));
        game(at, 0.0f);
        store<float>(blink + kFactor, MUTANT(6) ? load<float>(blink + kHigh) : load<float>(at + kFactor));
    }
    store<float>(blink + kTime, time + step);
    g_held.fetch_add(1, std::memory_order_relaxed);
}

void decomp_gx_blink_report() {
    if (decomp_placed("sub_12a25f0") != 1) return;
    const u64 begun = g_begun.load(), held = g_held.load();
    host_log("decomp: the GX lights' blink: %llu cycles begun on whole effect steps, %llu updates past a cycle's end held between "
             "them (%.2f a cycle)%s",
             static_cast<ull>(begun), static_cast<ull>(held), begun ? static_cast<double>(held) / static_cast<double>(begun) : 0.0,
             g_hold_on ? "" : " (BBHOST_BLINK_HOLD=0: every update the game's)");
}

namespace {
// vmovss xmm2, [rdi+0x14]; vxorps xmm1, xmm1, xmm1; vucomiss xmm2, xmm1.
constexpr std::uint8_t kBlinkEntry[] = {0xc5, 0xfa, 0x10, 0x57, 0x14, 0xc5, 0xf0, 0x57, 0xc9, 0xc5, 0xf8, 0x2e, 0xd1};
}  // namespace

void decomp_gx_blink_add() {
    const char* e = std::getenv("BBHOST_BLINK_HOLD");
    g_hold_on = !(e && e[0] == '0');
    DecompFunction blink{"sub_12a25f0", "gx", kBlinkUpdate, kBlinkEntry, sizeof(kBlinkEntry), reinterpret_cast<void*>(&blink_update),
                         DecompKind::Leaf, &g_game_blink};
    blink.report = &decomp_gx_blink_report;
    decomp_add(blink);
}
