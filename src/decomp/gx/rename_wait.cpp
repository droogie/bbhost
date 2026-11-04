// GX's discard rename, with a wait when it finds no room (docs/decomp.md):
// sub_2ad0d30 - Binary Ninja 0x2ad0d30 - hands a discard Map of a dynamic
// buffer (Map type 3, and two other callers) a renamed copy from the
// context's allocator over GX's upload heap: the CPU writes the frame's bytes
// there, the context's draws read them, and GX copies them back to the
// buffer's own memory on the GPU. With no room for the copy it walks the
// pool's done blocks (sub_2aaa860, ours in block_reclaim.cpp) and tries once
// more; failing that it returns 9, Map hands out nothing, and the game skips
// the frame's write: every draw of the frame then reads the buffer's own
// memory - the last copy-back's bytes, or none - in place of the frame's.
//
// The heap's blocks come free only once the GPU has run the work that used
// them (their labels). On the console the GPU runs close behind the game; here
// the command processor and the GPU can run a frame or two further behind -
// GPU-bound at a high frame rate, ~99% busy - and at a heavy moment (a camera
// turn, an arrival, an effects ring's 4 MiB copy) the walk finds too little
// done. Each failed Map lost a frame's constants: the scene's light and fog
// passes read stale constants for a frame (the warm light gone, the haze
// gone), and the effects' vertex ring its vertices (a sprite blown up).
//
// So ours runs the game's rename and, only when it failed for want of room
// (it walked the pool from its empty path: block_reclaim.cpp counts those
// walks), lets the GPU run one more of the submissions it was handed - its
// labels land, the oldest blocks come free - and runs the rename again, a
// submission at a time; then, if that is not enough, it lets the GPU catch up
// on everything recorded, and then the command processor walk what it still
// holds first, running the rename after each. Waiting for everything at once
// (a flush) at every one drained the GPU and left it idle: at 3840x2160
// uncapped, where the base lost ~100,000 discard Maps in two places' run,
// that cost 18% of the burning Dream's frame rate. A failure for the rename table's own chunk is the game's as before (it
// keeps the copy it took, so a second run would take another). The wait is
// bounded, as a driver's would be on a discard it has no room for: a frame's
// time where there was a frame lit with another frame's constants.
//
// The reclaim's own wait (block_reclaim.cpp) never ran for a call made on a
// GX thread: it read the caller's return address only through the guest's
// map list, which holds none of the host-made thread stacks. It reads it as
// a host read now, and its wait runs where it was meant to.
//
// BBHOST_GX_RENAME_WAIT=0: count the failures and keep the game's result, and
// the reclaim's return address read as before - today's path, the
// measurement's baseline.
//
// Hosted: it calls the game's function and the host's waits.
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
#include <cstdlib>

using namespace decomp;

namespace {

// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; sub rsp, 0x38
constexpr std::uint8_t kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                   0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x38};
// What it returns (in eax) when there was no room (or no table chunk).
constexpr std::uint32_t kNoRoom = 9;
// How long the command processor may take to walk what it has been given.
constexpr auto kCpBudget = std::chrono::milliseconds(100);
// The step-at-a-time catch-up's bounds (block_reclaim.cpp's): steps, and time before the flush.
constexpr int kMaxSteps = 64;
constexpr auto kStepBudget = std::chrono::milliseconds(200);

const bool g_wait = [] {
    const char* e = std::getenv("BBHOST_GX_RENAME_WAIT");
    return !(e && e[0] == '0');
}();

void* g_game = nullptr;
std::atomic<std::uint64_t> g_failed{0}, g_for_room{0}, g_by_step{0}, g_steps_taken{0}, g_by_gpu{0}, g_by_cp{0}, g_still{0}, g_table{0};
std::atomic<std::uint64_t> g_waited_us{0}, g_longest_us{0};

std::uint32_t run_game(std::uint64_t table, std::uint64_t ctx, std::uint64_t out, std::uint64_t resource) {
    return static_cast<std::uint32_t>(hle_call_guest<std::uint64_t>(g_game, table, ctx, out, resource));
}

// sub_2ad0d30(rename table, context + 0x10, mapped address out, resource):
// 0, or 9 with nothing handed out.
GUEST_ABI std::uint64_t discard_rename(std::uint64_t table, std::uint64_t ctx, std::uint64_t out, std::uint64_t resource) {
    const std::uint64_t walks = decomp_gx_rename_reclaims_here();
    std::uint32_t r = run_game(table, ctx, out, resource);
    if (r != kNoRoom) return r;
    g_failed.fetch_add(1, std::memory_order_relaxed);
    if (decomp_gx_rename_reclaims_here() == walks) {
        g_table.fetch_add(1, std::memory_order_relaxed);  // the table's chunk, not the copy's room: the game's as before
        return r;
    }
    const std::uint64_t n = g_for_room.fetch_add(1, std::memory_order_relaxed);
    if (!g_wait) return r;
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t backlog = hle_gnm_submitted_total() - hle_gnm_retired_total();
    // 0. The GPU a submission further at a time, the rename again after each
    //    (it walks and takes its copy): the oldest blocks come free first, and
    //    the GPU keeps the rest of what it was handed to run meanwhile.
    int stage = 0, steps = 0;
    while (r == kNoRoom && steps < kMaxSteps && std::chrono::steady_clock::now() - t0 < kStepBudget && decomp_gx_gpu_step()) {
        ++steps;
        r = run_game(table, ctx, out, resource);
    }
    g_steps_taken.fetch_add(static_cast<std::uint64_t>(steps), std::memory_order_relaxed);
    if (r == kNoRoom) {
        // 1. What is recorded: submitted and run, its labels written; the rename again.
        stage = 1;
        host_gpu_flush();
        r = run_game(table, ctx, out, resource);
    }
    if (r == kNoRoom) {
        // 2. What the command processor still holds, walked first.
        stage = 2;
        const std::uint64_t target = hle_gnm_submitted_total();
        while (hle_gnm_retired_total() < target && std::chrono::steady_clock::now() - t0 < kCpBudget) host_sleep_us(200);
        host_gpu_flush();
        r = run_game(table, ctx, out, resource);
    }
    (r == kNoRoom ? g_still : stage == 0 ? g_by_step : stage == 1 ? g_by_gpu : g_by_cp).fetch_add(1, std::memory_order_relaxed);
    const auto us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    g_waited_us.fetch_add(us, std::memory_order_relaxed);
    std::uint64_t longest = g_longest_us.load(std::memory_order_relaxed);
    while (us > longest && !g_longest_us.compare_exchange_weak(longest, us, std::memory_order_relaxed)) {
    }
    static std::atomic<int> long_said{0};
    if (n < 40 || (us >= 100000 && long_said.fetch_add(1, std::memory_order_relaxed) < 40)) {
        std::uint32_t bytes = 0;
        if (resource && hle_kernel_va_mapped(resource + 0x14, 4)) bytes = *reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(resource + 0x14));
        host_log("decomp: a discard Map found no room for its %u-byte copy at flip %llu (%llu jobs queued); %s after %llu us", bytes,
                 static_cast<unsigned long long>(hle_video_flip_count()), static_cast<unsigned long long>(backlog),
                 r == kNoRoom ? "still none (the game's failure)"
                 : stage == 0 ? (steps == 1 ? "it took one once the GPU ran a submission further" : "it took one once the GPU ran some submissions further")
                 : stage == 1 ? "it took one once the GPU caught up"
                              : "it took one once the command processor and the GPU had",
                 static_cast<unsigned long long>(us));
    }
    return r;
}

void report() {
    if (!g_failed.load()) return;
    if (!g_wait) {
        host_log("decomp: GX's discard renames: %llu found no room (BBHOST_GX_RENAME_WAIT=0: each a Map that failed, as before)",
                 static_cast<unsigned long long>(g_failed.load()));
        return;
    }
    host_log("decomp: GX's discard renames: %llu found no room or table chunk - %llu for want of room, %llu for the table "
             "(the game's); %llu took a copy a submission at a time (%llu steps), %llu once the GPU caught up, %llu once the command "
             "processor and the GPU had, %llu still not; %llu ms waited in all, the longest %llu ms",
             static_cast<unsigned long long>(g_failed.load()), static_cast<unsigned long long>(g_for_room.load()),
             static_cast<unsigned long long>(g_table.load()), static_cast<unsigned long long>(g_by_step.load()),
             static_cast<unsigned long long>(g_steps_taken.load()), static_cast<unsigned long long>(g_by_gpu.load()),
             static_cast<unsigned long long>(g_by_cp.load()), static_cast<unsigned long long>(g_still.load()),
             static_cast<unsigned long long>(g_waited_us.load() / 1000), static_cast<unsigned long long>(g_longest_us.load() / 1000));
}

}  // namespace

void decomp_gx_rename_wait_add() {
    DecompFunction fn{"sub_2ad0d30", "gx", 0x2ad0d30, kEntry, sizeof(kEntry), reinterpret_cast<void*>(&discard_rename), DecompKind::Hosted};
    fn.original = &g_game;
    fn.report = &report;
    decomp_add(fn);
}
