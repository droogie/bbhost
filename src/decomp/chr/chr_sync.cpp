// The local player's character sync by game time (docs/decomp.md):
// sub_1929470, PadManipulator's +0x50 virtual (its vtable 0x5770430) - the
// packet the local player's character state goes to the other players in.
//
// Each update counts its countdown (+0xc8) down one, hands its packet
// (+0xc0) to the character's network module (sub_1e1e270, with "a send
// frame" when the countdown was below 2 - the module's queued state goes in
// then), and when the countdown was 1 or less fills the packet - the
// position and yaw, the HP share, the action state (sub_1e73390,
// sub_1e72220) - and reloads the countdown from the int setting
// ChrSync.FrameSkipCount, with 5 when nothing sets it (0x1929675's mov edx,
// 5 before sub_24eba00): a packet each 5 updates, 6 a second at 30.
//
// At a converted rate the frame-rate tables make the default 5 x N/30
// updates, which sends 6 a second while the frames take 1/N s and no other
// length, and not at all where the setting is given. By default at a
// converted rate (and with the frame clock on, BBHOST_FRAME_CLOCK=1, at every
// rate) the default keeps the game's 5 (engine/frame_rate.cpp,
// kCountsHandedOver) and ours counts the countdown in
// 30 fps frames of game time: a frame in which none begins (frame_clock's
// ticks30, 0 or 1 a frame) holds it - the game's update sees 2 there, which
// is no send frame and which its decrement leaves at 1, and the countdown is
// put back after - so a packet goes each FrameSkipCount 30 fps frames, in the
// frame where the game at 30 sends it, 6 a second as at 30, whatever the
// setting says and at any frame length. The rest of the update runs every
// frame, as the game's does at 30 (the module marks the packet each update
// its state asks it to). At 30 every frame begins one 30 fps frame: the
// game's update, untouched.
//
// A leaf: entered straight from the game's code, ours runs the game's update
// through the trampoline around its own work, touching only the countdown.
// It is on the decomp list by default at a converted rate, where the
// frame-time manager publishes the clock's 30 fps frames for it with the clock
// off too (frame_clock::publishing), and at every rate with the clock on; at
// the game's own 30 with the clock off the game's sync stays.
// BBHOST_CHR_SYNC_STEP=0 leaves it off (and the tables' count at start).
#include "decomp/decomp.h"
#include "decomp/guest.h"
#include "engine/frame_clock.h"

#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>

using namespace decomp;

#ifdef DECOMP_CHR_SYNC_MUTANTS
extern int g_chr_sync_mutant;
#define MUTANT(n) (g_chr_sync_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kSync = 0x1929470, kSyncEnd = 0x19296c2;  // to its call of __stack_chk_fail
constexpr u64 kBodyHash = 0x7e546384af3e3fc8ull;         // FNV-1a of the bytes ours runs around
constexpr u64 kCountdown = 0xc8;

using Sync = GUEST_ABI u64 (*)(u64 self, u64 chr);

void* g_game_sync = nullptr;
// Updates; frames that held the countdown; updates that reloaded it (a
// packet filled).
std::atomic<u64> g_updates{0}, g_held{0}, g_packets{0};

}  // namespace

DECOMP_LEAF u64 chr_sync_update(u64 self, u64 chr) {
    const auto game = reinterpret_cast<Sync>(g_game_sync);
    g_updates.fetch_add(1, std::memory_order_relaxed);
    const s32 countdown = load<s32>(self + kCountdown);
    if (frame_clock::ticks30() != 0 || MUTANT(1)) {
        const u64 result = game(self, chr);
        if (countdown <= 1) g_packets.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    store<s32>(self + kCountdown, MUTANT(2) ? 1 : 2);
    g_held.fetch_add(1, std::memory_order_relaxed);
    const u64 result = game(self, chr);
    if (!MUTANT(3)) store<s32>(self + kCountdown, countdown);
    return result;
}

namespace {

u64 fnv1a(u64 h, u8 b) { return (h ^ b) * 0x100000001b3ull; }

void report() {
    if (decomp_placed("sub_1929470") == 0) return;
    host_log("decomp: the character sync: %llu updates, %llu holding the countdown (frames that begin no 30 fps frame), %llu packets filled",
             static_cast<ull>(g_updates.load()), static_cast<ull>(g_held.load()), static_cast<ull>(g_packets.load()));
}

// push rbp; mov rbp, rsp; push r15.
constexpr u8 kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};

}  // namespace

// The body check, for decomp_add and the test: the update ours runs around
// is the one it was written against, whole (the default's immediate as the
// game has it, which the frame-rate code leaves for ours).
bool chr_sync_body_ok(const std::uint8_t* entry) {
    u64 h = 0xcbf29ce484222325ull;
    for (u64 a = kSync; a < kSyncEnd; ++a) h = fnv1a(h, entry[a - kSync]);
    return h == kBodyHash;
}

void decomp_chr_sync_add() {
    // Where the clock is published - a converted rate by default, every rate
    // with the clock on (engine/frame_rate.cpp keeps the game's default count
    // for it then) - and not with BBHOST_CHR_SYNC_STEP=0.
    const char* e = std::getenv("BBHOST_CHR_SYNC_STEP");
    if ((!frame_clock::on() && !frame_clock::publishing()) || (e && e[0] == '0')) return;
    DecompFunction fn{"sub_1929470", "chr", kSync, kEntry, sizeof(kEntry), reinterpret_cast<void*>(&chr_sync_update),
                      DecompKind::Leaf, &g_game_sync};
    fn.report = &report;
    fn.body_ok = &chr_sync_body_ok;
    decomp_add(fn);
}
