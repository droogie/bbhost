// The replay ghosts by game time (docs/decomp.md): ReplayManipulator's update,
// sub_192c390, the slot 9 of its vtable (0x57707d0) - the manipulator that
// moves a recorded character, a bloodstain's or a ghost's, along the
// recording another player's game took.
//
// A recording is 0x7c-byte samples taken each 1/3 s of game time: the
// recorder (sub_1d09b40) adds every step it is handed and takes a sample once
// they reach 1/3 s (0x3eaaaaab) - the position packed in 18-bit fields of
// 1/50, the yaw in 7 bits, the sample's events after. The update plays them
// back in frames. Its countdown (+0xf8) goes down one an update with a step
// above 0 and, reaching 0, comes back to 10 (0x192c3ff) for the next sample
// (+0xfc counts them): its position (+0xb0) and yaw (+0xc0), and a tenth of
// the change to each from where the character is (+0xd0 and +0xe0, by
// 0x4cf6420's four 0.1s). Every update then moves the character by the
// position's tenth (the move, vtable +0x130, ReplayManipulator's being the
// store into +0x70) - by the whole change instead past 10 units, by nothing
// within 1/1000 - and turns it by the yaw's (+0x20), so a sample's change is
// covered in the 10 frames before the next: 1/3 s at 30. Its elapsed time
// (+0x104) counts the steps, and the fade-out a second before the end
// (+0x108) reads that.
//
// At a converted rate the frame-rate tables make that 10 x N/30 frames and a
// tenth x 30/N, the recording's pace while the frames take 1/N s and no
// other length. By default at a converted rate (and with the frame clock on,
// BBHOST_FRAME_CLOCK=1, at every rate) both keep the game's
// (engine/frame_rate.cpp, kHandedOver and kCountsHandedOver) and ours plays
// the recording by game time at any frame length:
//   - a frame in which no 30 fps frame of game time begins (frame_clock's
//     ticks30: 0 or 1 a frame) holds the countdown - the game's update sees
//     2 there, which its decrement leaves above 0, and the countdown is put
//     back after - so the countdown counts 30 fps frames of game time and a
//     sample is taken in the frame where the game at 30 takes it, every 1/3
//     s of game time, as the recorder took them. A new manipulator's first
//     update takes the first sample at once, as the game's does;
//   - the move and the turn the update leaves are its tenths times the frame
//     in 30 fps frames, so a sample's change is covered in game time, not in
//     frames. The jump past 10 units stays whole.
// At 30 every frame begins one 30 fps frame and is one: the game's update,
// untouched.
//
// A leaf: entered straight from the game's code, ours runs the game's update
// through the trampoline around its own work, touching only the
// manipulator's fields. It is on the decomp list by default at a converted
// rate, where the frame-time manager publishes the clock's 30 fps frames for
// it with the clock off too (frame_clock::publishing), and at every rate with
// the clock on; at the game's own 30 with the clock off the game's update
// stays. BBHOST_REPLAY_STEP=0 leaves it off (and the tables' values at start).
#include "decomp/decomp.h"
#include "decomp/guest.h"
#include "engine/frame_clock.h"

#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

using namespace decomp;

#ifdef DECOMP_CHR_REPLAY_MANIPULATOR_MUTANTS
extern int g_replay_manipulator_mutant;
#define MUTANT(n) (g_replay_manipulator_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kUpdate = 0x192c390, kUpdateEnd = 0x192c930;  // to its call of __stack_chk_fail
constexpr u64 kBodyHash = 0x65b4ccfbbc05261aull;            // FNV-1a of the bytes ours runs around
constexpr u64 kVtable = 0x57707d0;                          // ReplayManipulator's
// The manipulator's fields.
constexpr u64 kTurn = 0x20, kMove = 0x70, kTurnTenth = 0xe0, kMoveTenth = 0xd0, kRecording = 0xf0, kCountdown = 0xf8,
              kSample = 0xfc;
constexpr u64 kSampleCount = 0x90;  // the recording's

using Update = GUEST_ABI u64 (*)(u64 self, u64 chr, float step);

void* g_game_update = nullptr;
// Updates; frames that held the countdown; samples taken; moves and turns
// scaled to the frame.
std::atomic<u64> g_updates{0}, g_held{0}, g_samples{0}, g_moves{0}, g_turns{0};

bool same16(u64 a, u64 b) { return load<u64>(a) == load<u64>(b) && load<u64>(a + 8) == load<u64>(b + 8); }

// The four lanes at `from`, times `frames`, into `to`.
void scaled4(u64 to, u64 from, float frames) {
    for (u64 k = 0; k < 16; k += 4) store<float>(to + k, load<float>(from + k) * frames);
}

}  // namespace

DECOMP_LEAF u64 replay_manipulator_update(u64 self, u64 chr, float step) {
    const auto game = reinterpret_cast<Update>(g_game_update);
    if (load<u64>(self) != game_address(kVtable) && !MUTANT(1)) return game(self, chr, step);
    g_updates.fetch_add(1, std::memory_order_relaxed);
    // The game's own test for a recording to play: without one it ends the
    // replay and moves nothing.
    const u64 recording = load<u64>(self + kRecording);
    const bool plays = recording != 0 && load<s32>(recording + kSampleCount) > 0;
    // The countdown at 0 or below is a new manipulator's (the factory zeroes
    // it, sub_1cc0280): the game's first update takes the first sample at
    // once, before its first move, and so does ours.
    const s32 countdown = load<s32>(self + kCountdown);
    const bool hold = frame_clock::ticks30() == 0 && countdown > 0 && !MUTANT(2);
    const s32 sample = load<s32>(self + kSample);
    if (hold) {
        store<s32>(self + kCountdown, MUTANT(3) ? 1 : 2);
        g_held.fetch_add(1, std::memory_order_relaxed);
    }
    const u64 result = game(self, chr, step);
    if (hold && !MUTANT(4)) store<s32>(self + kCountdown, countdown);
    if (load<s32>(self + kSample) != sample) g_samples.fetch_add(1, std::memory_order_relaxed);
    const float frames = frame_clock::frames30();
    if (!plays || (frames == 1.0f && !MUTANT(5))) return result;
    // The move it left is its tenth (or nothing, the tenth zeroed within
    // 1/1000) - not the jump past 10 units, which zeroes the tenth after.
    if (same16(self + kMove, self + kMoveTenth) && !MUTANT(6)) {
        scaled4(self + kMove, self + kMoveTenth, frames);
        g_moves.fetch_add(1, std::memory_order_relaxed);
    }
    if (!MUTANT(7)) {
        scaled4(self + kTurn, self + kTurnTenth, frames);
        g_turns.fetch_add(1, std::memory_order_relaxed);
    }
    return result;
}

namespace {

u64 fnv1a(u64 h, u8 b) { return (h ^ b) * 0x100000001b3ull; }

void report() {
    if (decomp_placed("sub_192c390") == 0) return;
    host_log("decomp: the replay ghosts: %llu updates, %llu holding the sample countdown (frames that begin no 30 fps frame), %llu samples "
             "taken; %llu moves and %llu turns scaled to the frame",
             static_cast<ull>(g_updates.load()), static_cast<ull>(g_held.load()), static_cast<ull>(g_samples.load()),
             static_cast<ull>(g_moves.load()), static_cast<ull>(g_turns.load()));
}

// push rbp; mov rbp, rsp; push r15.
constexpr u8 kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};

}  // namespace

// The body check, for decomp_add and the test: the update ours runs around
// is the one it was written against, whole.
bool replay_manipulator_body_ok(const std::uint8_t* entry) {
    u64 h = 0xcbf29ce484222325ull;
    for (u64 a = kUpdate; a < kUpdateEnd; ++a) h = fnv1a(h, entry[a - kUpdate]);
    return h == kBodyHash;
}

void decomp_chr_replay_manipulator_add() {
    // Where the clock is published - a converted rate by default, every rate
    // with the clock on (engine/frame_rate.cpp keeps the game's countdown and
    // tenths for it then) - and not with BBHOST_REPLAY_STEP=0.
    const char* e = std::getenv("BBHOST_REPLAY_STEP");
    if ((!frame_clock::on() && !frame_clock::publishing()) || (e && e[0] == '0')) return;
    DecompFunction fn{"sub_192c390", "chr", kUpdate, kEntry, sizeof(kEntry), reinterpret_cast<void*>(&replay_manipulator_update),
                      DecompKind::Leaf, &g_game_update};
    fn.report = &report;
    fn.body_ok = &replay_manipulator_body_ok;
    decomp_add(fn);
}
