// Dantelion's spin-then-sleep mutex, DLAdaptiveMutex::Lock (docs/decomp.md):
// Binary Ninja 0x24368c0. Every job queue of the engine's work pools takes one
// to pop a job (NS_SPRJ::CSEzWork::ProcessOne), and so do the heaps, the GX
// layer and the character code.
//
// The lock word is the u64 at +0x18: bits 0-39 the owner (the low 40 bits of
// scePthreadSelf; 0 free, 0xffffffffff handed to a woken sleeper), bits 40-63
// how many threads sleep on the semaphore at +0x10. The s32 at +0x8 is how
// long a contender spins, in time-stamp ticks, before it counts itself in and
// sleeps. The game's unlock (sub_2436a40) frees the word, or, with a sleeper
// counted, hands the lock over (owner 0xffffffffff, one fewer counted) and
// signals the semaphore once; the woken thread owns the lock.
//
// The game's spin is a `lock cmpxchg` of the word with its own value and an
// rdtsc on every turn, no pause. Six character threads popping jobs from one
// queue keep that cache line moving between cores, and the holder's unlock
// has to win it back each time: in the world at the Central Yharnam lamp the
// CSChrThread workers spent 60% of their time in this spin, and the main loop
// waits ~1.4 ms a frame for their jobs (WorldChrMan::BuildCharacterUpdateLists).
// Ours makes the same transitions in the same order - take the word when its
// owner is free or ourselves, count in and sleep once the game's spin budget
// has passed - but spins on a plain load with `pause` in between, and swaps
// only to take the lock or to count itself in. A swap the game made with the
// word's own value changed nothing; leaving it out changes only how fast the
// line moves.
//
// A leaf: entered from the game's code on its thread, it calls only the
// game's own import stubs (scePthreadSelf, sceKernelWaitSema) and DL_PANIC,
// as the game's version does. There is no compare run - a lock is checked by
// what it excludes: tests/decomp/adaptive_mutex_test.cpp runs threads taking
// the game's lock and ours on the same word, with the game's unlock.
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <atomic>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

using namespace decomp;

namespace {

constexpr u64 kLock = 0x24368c0;          // DLAdaptiveMutex::Lock
constexpr u64 kPthreadSelf = 0x2fbfe78;   // the import stub the game's Lock calls
constexpr u64 kWaitSema = 0x2fbec78;      // sceKernelWaitSema's stub
constexpr u64 kPanic = 0x24b55b0;         // DL_PANIC
constexpr u64 kPanicFile = 0x4d3b017;     // the message the game's Lock panics with
constexpr u64 kPanicText = 0x4d3b0cb;

// push rbp; mov rbp, rsp; push r15; push r14 (8 bytes, whole instructions; the
// pushes of r12 and rbx and the call to scePthreadSelf follow).
constexpr std::uint8_t kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56};

constexpr u64 kOwnerMask = 0xffffffffffull;
constexpr u64 kSleepersMask = 0xffffff0000000000ull;
constexpr u64 kOneSleeper = 0x10000000000ull;

using SelfFn = GUEST_ABI u64 (*)();
using WaitSemaFn = GUEST_ABI s32 (*)(u64 sema, s32 need, void* timeout);
using PanicFn = GUEST_ABI void (*)(u64 file, s32 line, u64 text, ...);  // the game clears al: variadic

inline u64 ticks() {
#if defined(__x86_64__) || defined(_M_X64)
    return __rdtsc();
#else
    return 0;
#endif
}

inline void relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

DECOMP_LEAF s32 adaptive_mutex_lock(u64 m) {
    const u64 self = game_function<SelfFn>(kPthreadSelf)();
    const u64 start = ticks();
    auto* word = reinterpret_cast<std::atomic<u64>*>(static_cast<std::uintptr_t>(m + 0x18));
    u64 now = ticks();
    u64 until = start + static_cast<u64>(static_cast<s64>(load<s32>(m + 0x8)));  // movsxd, then a 64-bit add
    u64 old = word->load(std::memory_order_relaxed);
    for (;;) {
        const u64 owner = old & kOwnerMask;
        if (owner == 0 || owner == self) {
            // Free (or ours already): take it, the sleepers' count kept.
            const u64 want = (old & kSleepersMask) | (self & kOwnerMask);
            if (word->compare_exchange_weak(old, want, std::memory_order_acquire, std::memory_order_relaxed)) return 0;
            continue;  // `old` is the word now; the same moment's budget, as the game's retry
        }
        if (now < until) {
            // Held, and still within the spin: watch the word, not swap it.
            relax();
            now = ticks();
            until = start + static_cast<u64>(static_cast<s64>(load<s32>(m + 0x8)));
            old = word->load(std::memory_order_relaxed);
            continue;
        }
        // The spin is over: count ourselves in, then sleep until an unlock
        // hands us the lock.
        if (!word->compare_exchange_weak(old, old + kOneSleeper, std::memory_order_acquire, std::memory_order_relaxed)) continue;
        if (game_function<WaitSemaFn>(kWaitSema)(load<u64>(m + 0x10), 1, nullptr) != 0)
            game_function<PanicFn>(kPanic)(game_address(kPanicFile), 0xa9, game_address(kPanicText));
        return 0;
    }
}

}  // namespace

void decomp_adaptive_mutex_add() {
    DecompFunction fn{"DLAdaptiveMutex::Lock", "sync", kLock, kEntry, sizeof(kEntry), reinterpret_cast<void*>(&adaptive_mutex_lock),
                      DecompKind::Leaf};
    decomp_add(fn);
}
