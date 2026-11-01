// The step of GX's job barrier that runs a job on the main thread
// (docs/decomp.md): sub_25b9980 at Binary Ninja 0x25b9980, whose one caller is
// the barrier itself (sub_25b8710), which the main thread waits in after every
// parallel-for of GX's jobs - the render queue's above all.
//
// GX's job system (the barrier's object at +0) is participant 0, the main
// thread, followed by the GX workers, 0x530 bytes each: a deque of jobs at
// +0x18 (its count at +0x38), the participant's index at +0xa0, the system
// (participants at [+8], their count at [+8] + 8) at +8. The game's step takes
// its own deque's front (sub_1247b60) or, with its deque empty, sweeps the
// others from a random one - the time-stamp counter plus its index, modulo
// their count - stealing half of the first that has jobs into its own deque
// (sub_1247d20) and taking the front; then it runs the job (its vtable +0,
// with the index). The barrier calls it while its own deque has jobs, then
// spins until every worker has gone idle.
//
// So the main thread steals only when it comes in with an empty deque, and
// once the jobs in its own deque are done it never steals again: whatever is
// still queued on a worker - the parallel-for gives the chunks of rank 5 and
// under to participant i mod n - runs there while the main thread spins. The
// render queue's census at Old Yharnam (RTX 4070, uncapped, 10-26): the main
// thread done 1.4 ms into a 2.9 ms parallel-for while the workers started
// chunks up to 2.4 ms in; at Central Yharnam done at 1.35 ms of 2.6, at
// Cathedral Ward at 1.14 of 1.77. On v0.2.23 the main loop spins 2-4 ms a
// flip in the barrier, five workers busy in 50-57% of those ticks.
//
// Ours (BBHOST_GX_BARRIER_STEAL; on unless 0): the game's step, the same
// calls in the same order, and then, when its own deque is empty, the same
// sweep once more without taking: a deque it stole into keeps the barrier
// calling, so the main thread takes its share of what is left, as the
// workers' own loop (sub_25b9760) does. Off, ours is the game's step.
// Measured on an RTX 4070 at 1080p, uncapped, a b b a: the barrier's spin
// 2.8-4.4 -> 1.2-1.7 ms a frame, the frame rate +4.6% at Old Yharnam (where
// the command processor then limits), +6.7% at the Nightmare of Mensis, +6.1%
// at Central Yharnam, +5.9% at Cathedral Ward; the frames the same.
//
// A leaf: entered from the barrier on the main thread, it calls only the
// game's own functions (the deque's pop and steal, sceKernelReadTsc's stub,
// the job) and counts in globals of its own.
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>

using namespace decomp;

#ifdef DECOMP_GX_JOB_BARRIER_MUTANTS
extern int g_jb_mutant;
#define MUTANT(n) (g_jb_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr int kMutants = 5;  // the test's deliberate faults (tests/decomp/gx_job_barrier_test.cpp)

constexpr u64 kRunOne = 0x25b9980;   // the barrier's step (ours)
constexpr u64 kPop = 0x1247b60;      // (deque, &job): false when the deque is empty
constexpr u64 kSteal = 0x1247d20;    // (victim's deque, thief's deque): true when it moved jobs
constexpr u64 kReadTsc = 0x2fc0de8;  // sceKernelReadTsc's stub

// push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12; push rbx;
// sub rsp, 0x18 (17 bytes, whole instructions; the stack guard's
// rip-relative load follows).
constexpr std::uint8_t kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                   0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x18};

constexpr u64 kParticipantSize = 0x530;

using PopFn = GUEST_ABI bool (*)(u64 deque, u64* job);
using StealFn = GUEST_ABI bool (*)(u64 victim, u64 thief);
using TscFn = GUEST_ABI u64 (*)();
using RunFn = GUEST_ABI void (*)(u64 job, u32 index);

bool g_steal = true;
std::atomic<std::uint64_t> g_jobs{0}, g_sweeps_after{0}, g_stolen_after{0};

// The game's sweep: from (index + the time-stamp counter) mod the count, each
// other participant in turn, until a steal moves jobs or the own deque is not
// empty. True when the own deque got jobs.
bool sweep(u64 self, u64 own) {
    const u64 tsc = game_function<TscFn>(kReadTsc)();
    u64 system = load<u64>(self + 8);
    u32 n = load<u32>(system + 8);
    if (n == 0) return false;  // (the game's divides first)
    const u64 from = MUTANT(1) ? tsc : static_cast<u64>(load<u32>(self + 0xa0)) + tsc;
    const u32 start = MUTANT(5) ? static_cast<u32>(from) % n : static_cast<u32>(from % n);
    for (u32 i = 0; i < n;) {
        const u32 at = (start + i) % n;
        const u64 p = load<u64>(system) + static_cast<u64>(at) * kParticipantSize;
        if (p != self || MUTANT(2)) {
            if (game_function<StealFn>(kSteal)(p + 0x18, own) || load<u64>(self + 0x38) != 0) return true;
            system = load<u64>(self + 8);
        }
        ++i;
        n = load<u32>(system + 8);
    }
    return false;
}

DECOMP_LEAF void gx_job_run_one(u64 self) {
    const u64 own = self + 0x18;
    u64 job = 0;
    if (!game_function<PopFn>(kPop)(own, &job) && sweep(self, own)) game_function<PopFn>(kPop)(own, &job);
    if (job) reinterpret_cast<RunFn>(static_cast<std::uintptr_t>(load<u64>(load<u64>(job))))(job, MUTANT(4) ? 0 : load<u32>(self + 0xa0));
    if (!g_steal) return;
    if (job) g_jobs.fetch_add(1, std::memory_order_relaxed);
    if (load<u64>(self + 0x38) != 0 && !MUTANT(3)) return;
    g_sweeps_after.fetch_add(1, std::memory_order_relaxed);
    if (sweep(self, own)) g_stolen_after.fetch_add(1, std::memory_order_relaxed);
}

void report() {
    if (!g_steal) return;
    host_log("decomp: GX's job barrier ran %llu jobs on the main thread; with its own deque empty it swept the workers' %llu "
             "times and took jobs %llu times (BBHOST_GX_BARRIER_STEAL)",
             static_cast<ull>(g_jobs.load()), static_cast<ull>(g_sweeps_after.load()), static_cast<ull>(g_stolen_after.load()));
}

}  // namespace

void decomp_gx_job_barrier_add() {
    const char* e = std::getenv("BBHOST_GX_BARRIER_STEAL");
    g_steal = !(e && e[0] == '0');
    DecompFunction fn{"sub_25b9980", "gx", kRunOne, kEntry, sizeof(kEntry), reinterpret_cast<void*>(&gx_job_run_one),
                      DecompKind::Leaf};
    fn.report = &report;
    decomp_add(fn);
}
