// The GX layer's blocks for draw resource tables, taken back (docs/decomp.md):
// every GX draw's commit copies its T#, S# and V# tables into memory from
// sub_2aaddc0, which carves them out of 64 KiB blocks; a block's status word
// turns 4 once the GPU has finished the work that used it, and sub_2aaa860 -
// at Binary Ninja 0x2aaa860 - walks the used blocks and frees those. The
// allocator calls it when no block has room left, and if nothing comes free
// it returns 0: the commit then points the stage's table at address 0, and
// every texture the draw reads is missing. The glitch hunt found this behind
// the black frames (the gamma pass's lookup table is read through such a
// table - host/glitch.cpp): the pool had run dry, with
// blocks still waiting on labels the GPU had not written yet.
//
// The allocator runs that walk lazily, whenever its pool looks empty - about
// once a frame - and it usually frees blocks. On the console the GPU runs
// close behind the game and the labels have landed by then; here the command
// processor and the GPU run further behind (labels of four or eight bytes land
// at the end of a command buffer, and the command processor keeps recording
// into one while the game's submissions queue up), and at a heavy moment - a
// loading screen, a burst of combat effects - a walk can find every block in
// use still waiting. So ours does the game's walk and, only when it freed
// nothing, lets the GPU catch up on what is recorded and walks again, and if
// that is not enough, lets the command processor walk what it holds first.
// The catch-up goes a submission at a time first (decomp_gx_gpu_step): the
// GPU keeps the rest of what it was handed to run while the walk finds what
// came free (BBHOST_GX_RECLAIM_STEP=0: everything recorded at once, as before).
// From anywhere else (the flip's own reclaim) it is the game's walk alone.
// The wait is bounded: a few milliseconds where there was a draw with its
// textures and constants missing.
#include "decomp/gx/block_reclaim.h"

#include "decomp/decomp.h"
#include "decomp/guest.h"

#include "core/portable.h"
#include "core/thunk.h"
#include "hle/modules.h"
#include "host/gpu.h"
#include "log.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace decomp;

namespace {

// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; and rsp, -32
// (20 bytes; the rip-relative stack-guard load follows).
constexpr std::uint8_t kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                   0x41, 0x54, 0x53, 0x48, 0x81, 0xe4, 0xe0, 0xff, 0xff, 0xff};
// The return addresses of the calls made when a pool is empty: sub_2aaddc0's
// (the draw resource tables) and sub_2ad0d30's (another allocator over the
// same blocks, the same retry).
constexpr std::uint64_t kFromAllocator = 0x2aade27, kFromAllocator2 = 0x2ad0d97;
// How long the command processor may take to walk what it has been given.
constexpr auto kCpBudget = std::chrono::milliseconds(100);

// BBHOST_GX_RECLAIM_WAIT=0: count the empty pools and do the game's walk
// only, as before (a measurement's baseline).
const bool g_wait = [] {
    const char* e = std::getenv("BBHOST_GX_RECLAIM_WAIT");
    return !(e && e[0] == '0');
}();

// BBHOST_GX_RENAME_WAIT=0 (rename_wait.cpp, the discard rename's wait off):
// the caller's return address read through the guest's map list, as before.
const bool g_host_read = [] {
    const char* e = std::getenv("BBHOST_GX_RENAME_WAIT");
    return !(e && e[0] == '0');
}();
// BBHOST_GX_RECLAIM_STEP=0: no step at a time - the catch-up waits for
// everything recorded at once (a flush), as before.
const bool g_steps = [] {
    const char* e = std::getenv("BBHOST_GX_RECLAIM_STEP");
    return !(e && e[0] == '0');
}();
// The step-at-a-time catch-up's bounds: steps, and time before the flush.
constexpr int kMaxSteps = 64;
constexpr auto kStepBudget = std::chrono::milliseconds(200);

void* g_game = nullptr;
std::atomic<std::uint64_t> g_calls{0}, g_stuck{0}, g_freed_by_step{0}, g_steps_taken{0}, g_freed_by_gpu{0}, g_freed_by_cp{0}, g_still{0};
// The walks from the discard rename's empty path (kFromAllocator2), on this
// thread and in all: the rename's wait (rename_wait.cpp) tells by them that a
// rename failed for want of room, not for its table.
thread_local std::uint64_t t_rename_walks = 0;
std::atomic<std::uint64_t> g_rename_walks{0};
std::atomic<std::uint64_t> g_waited_us{0}, g_longest_us{0};

// The pool's count of blocks in use: the walk takes one off for each block
// it frees, so an unchanged count after it is a walk that found nothing done.
std::uint64_t in_use(std::uint64_t pool) {
    return *reinterpret_cast<const volatile std::uint64_t*>(static_cast<std::uintptr_t>(pool + 0xc8));
}

// sub_2aaa860(pool, command buffer): the pool's used blocks, walked. From the
// allocators' empty-pool path this is the pool's ordinary, lazy reclaim -
// about once a frame - and usually frees blocks. Only a walk that frees
// nothing means every block in use is waiting on a label the GPU has not
// written: then the GPU catches up on what is recorded (its labels land) and
// the walk runs again, and if that is not enough, the command processor
// walks what it still holds first. Without that the allocation failed and
// the draw's tables pointed at address 0.
GUEST_ABI void reclaim(std::uint64_t pool, std::uint64_t cb, std::uint64_t, std::uint64_t rsp, std::uint64_t) {
    // rsp comes from the frame-capturing thunk; were that skipped (its page
    // full), it would be whatever the caller left in rcx - read nothing then.
    // It is the calling thread's own stack: a GX thread's, which the host
    // made, lies outside the guest's map list, and asking that list refused
    // every call made on one - the waits below never ran for them. So the
    // word is read as a host read that cannot fault (BBHOST_GX_RENAME_WAIT=0:
    // through the map list as before).
    std::uint64_t from = 0;
    if (g_host_read) {
        if (rsp && !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(rsp)), &from, sizeof(from))) from = 0;
    } else if (rsp && hle_kernel_va_mapped(rsp, 8)) {
        from = *reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(rsp));
    }
    const bool empty_pool = from == game_address(kFromAllocator) || from == game_address(kFromAllocator2);
    if (from == game_address(kFromAllocator2)) {
        ++t_rename_walks;
        g_rename_walks.fetch_add(1, std::memory_order_relaxed);
    }
    const std::uint64_t before = empty_pool ? in_use(pool) : 0;
    hle_call_guest<std::int64_t>(g_game, pool, cb);
    if (!empty_pool) return;
    g_calls.fetch_add(1, std::memory_order_relaxed);
    if (in_use(pool) < before) return;
    const std::uint64_t n = g_stuck.fetch_add(1, std::memory_order_relaxed);
    if (!g_wait) return;
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t backlog = hle_gnm_submitted_total() - hle_gnm_retired_total();
    // 0. The GPU a submission further at a time, the walk after each: the
    //    oldest blocks' labels land first, and the GPU keeps the rest of what
    //    it was handed to run meanwhile.
    int stage = 0, steps = 0;
    bool freed = false;
    while (g_steps && !freed && steps < kMaxSteps && std::chrono::steady_clock::now() - t0 < kStepBudget && decomp_gx_gpu_step()) {
        ++steps;
        hle_call_guest<std::int64_t>(g_game, pool, cb);
        freed = in_use(pool) < before;
    }
    g_steps_taken.fetch_add(static_cast<std::uint64_t>(steps), std::memory_order_relaxed);
    if (!freed) {
        // 1. What is recorded: submitted and run, its labels written.
        stage = 1;
        host_gpu_flush();
        hle_call_guest<std::int64_t>(g_game, pool, cb);
        freed = in_use(pool) < before;
    }
    if (!freed) {
        // 2. What the command processor still holds, walked first.
        stage = 2;
        const std::uint64_t target = hle_gnm_submitted_total();
        while (hle_gnm_retired_total() < target && std::chrono::steady_clock::now() - t0 < kCpBudget) {
            host_sleep_us(200);
        }
        host_gpu_flush();
        hle_call_guest<std::int64_t>(g_game, pool, cb);
        freed = in_use(pool) < before;
    }
    (!freed ? g_still : stage == 0 ? g_freed_by_step : stage == 1 ? g_freed_by_gpu : g_freed_by_cp).fetch_add(1, std::memory_order_relaxed);
    const auto us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    g_waited_us.fetch_add(us, std::memory_order_relaxed);
    std::uint64_t longest = g_longest_us.load(std::memory_order_relaxed);
    while (us > longest && !g_longest_us.compare_exchange_weak(longest, us, std::memory_order_relaxed)) {
    }
    // The first 40, and the long ones (100 ms and more, 40 of them).
    static std::atomic<int> long_said{0};
    if (n < 40 || (us >= 100000 && long_said.fetch_add(1, std::memory_order_relaxed) < 40)) {
        char how[96];
        std::snprintf(how, sizeof(how), "the GPU %d submission%s further freed some", steps, steps == 1 ? "" : "s");
        host_log("decomp: the GX resource-table blocks were all waiting on the GPU at flip %llu (%llu jobs queued); %s after "
                 "%llu us",
                 static_cast<ull>(hle_video_flip_count()), static_cast<ull>(backlog),
                 !freed ? "still none free" : stage == 0 ? how : stage == 1 ? "the GPU's catching up freed some"
                                                                 : "the command processor's and the GPU's catching up freed some",
                 static_cast<ull>(us));
    }
}

void report() {
    if (!g_calls.load()) return;
    host_log("decomp: the GX resource-table blocks: %llu walks from an empty pool, %llu of them freeing nothing%s; %llu freed "
             "a submission at a time (%llu steps), %llu once the GPU caught up, %llu once the command processor and the GPU had, "
             "%llu still not (the allocation failed); %llu ms waited in all, the longest %llu ms",
             static_cast<ull>(g_calls.load()), static_cast<ull>(g_stuck.load()),
             g_wait ? "" : " (BBHOST_GX_RECLAIM_WAIT=0: each an allocation that failed)", static_cast<ull>(g_freed_by_step.load()),
             static_cast<ull>(g_steps_taken.load()), static_cast<ull>(g_freed_by_gpu.load()), static_cast<ull>(g_freed_by_cp.load()),
             static_cast<ull>(g_still.load()), static_cast<ull>(g_waited_us.load() / 1000), static_cast<ull>(g_longest_us.load() / 1000));
}

}  // namespace

std::uint64_t decomp_gx_rename_reclaims_here() { return t_rename_walks; }

bool decomp_gx_gpu_step() {
    const std::uint64_t done = host_gpu_submissions_completed();
    if (done + 1 >= host_gpu_work_needs()) return false;  // none handed over still running: what is left takes a flush
    return host_gpu_wait_completed(done + 1, 50'000'000ull);
}
std::uint64_t decomp_gx_rename_reclaims() { return g_rename_walks.load(std::memory_order_relaxed); }

void decomp_gx_block_reclaim_add() {
    DecompFunction fn{"sub_2aaa860", "gx", 0x2aaa860, kEntry, sizeof(kEntry), reinterpret_cast<void*>(&reclaim),
                      DecompKind::HostedFrame};
    fn.original = &g_game;
    fn.report = &report;
    decomp_add(fn);
}
