// The work pools' job pop (decomp/sync/ez_work.cpp) beside the game's own
// (tests/decomp/eboot_kit.h). A pool, its job list, jobs, groups and
// allocators are built in the test's memory with vtables of recorders - the
// queue lock's take and release, the list lock's, the event's signal, the
// allocators' free, a job's run and destructor, a group's destroy - and the
// game functions it calls (the group's completion, the finished list's push,
// the job's allocator, DL_PANIC) are recorders too. On generated pools -
// empty, one to five jobs, a count that disagrees with an empty list, jobs of
// every kind (0, 1, 2 and a bad one), with no group or one at any countdown
// and reference count, a completion that takes the job's group away - ours
// must make the game's calls with the game's arguments in the game's order,
// leave the memory as the game's leaves it, and return what it returns. (The
// pool's allocator here is a recorder, so ours never puts its node cache in;
// tests/decomp/ez_node_cache_test.cpp runs the cache.) Six deliberate
// mistakes must be caught, and the entry holds the instructions written
// against. Skips without the 1.09 eboot.
#define DECOMP_EZ_WORK_MUTANTS
int g_ez_mutant = 0;
#include "decomp/sync/ez_work.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0xe2);

struct Call {
    int who;
    u64 a[3];
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_rec;

enum Who {
    kLock, kUnlock, kListLock, kListUnlock, kEventSignal, kPoolFree, kJobFree, kRun, kDestroy, kGroupDestroy,
    kSignalGroupId, kPushDoneId, kAllocatorOfId, kPanicId,
};

alignas(64) u8 g_mem[0x4000];
u64 base() { return reinterpret_cast<u64>(g_mem); }
// The layout: the pool, its objects' vtables, nodes, jobs, groups.
constexpr u64 kPool = 0x0, kPoolAlloc = 0x200, kJobAlloc = 0x240, kVtables = 0x300, kAnchor = 0x800, kNodes = 0x840, kJobs = 0xa00,
              kGroups = 0x1000;
constexpr u64 kVtLock = kVtables + 0x0, kVtListLock = kVtables + 0x80, kVtEvent = kVtables + 0x100, kVtPoolAlloc = kVtables + 0x180,
              kVtJobAlloc = kVtables + 0x200, kVtJob = kVtables + 0x280, kVtGroup = kVtables + 0x300;
bool g_signal_takes_group = false;  // the group's completion clears the job's group pointer
u64 g_signal_job = 0;

GUEST_ABI u64 rec_lock(u64 self, u64 t) {
    g_rec.push_back({self == base() + kPool + 0x30 ? kLock : kListLock, {self, t & 0xffffffffull, 0}});
    return 0;
}
GUEST_ABI u64 rec_unlock(u64 self) {
    g_rec.push_back({self == base() + kPool + 0x30 ? kUnlock : kListUnlock, {self, 0, 0}});
    return 0;
}
GUEST_ABI u64 rec_event(u64 self, u64 v) {
    g_rec.push_back({kEventSignal, {self, v & 0xffffffffull, 0}});
    return 0;
}
GUEST_ABI u64 rec_pool_free(u64 self, u64 p) {
    g_rec.push_back({kPoolFree, {self, p, 0}});
    return 0;
}
GUEST_ABI u64 rec_job_free(u64 self, u64 p) {
    g_rec.push_back({kJobFree, {self, p, 0}});
    return 0;
}
GUEST_ABI u64 rec_run(u64 self) {
    g_rec.push_back({kRun, {self, 0, 0}});
    return 0;
}
GUEST_ABI u64 rec_destroy(u64 self) {
    g_rec.push_back({kDestroy, {self, 0, 0}});
    return 0;
}
GUEST_ABI u64 rec_group_destroy(u64 self) {
    g_rec.push_back({kGroupDestroy, {self, 0, 0}});
    return 0;
}
GUEST_ABI u64 rec_signal_group(u64 at, u64 n) {
    g_rec.push_back({kSignalGroupId, {at, n & 0xffffffffull, 0}});
    if (g_signal_takes_group && g_signal_job) store<u64>(g_signal_job + 8, 0);
    return 0;
}
GUEST_ABI u64 rec_push_done(u64 list, u64 pjob) {
    g_rec.push_back({kPushDoneId, {list, load<u64>(pjob), 0}});  // the job the pointer points at, not the stack address
    return 0;
}
GUEST_ABI u64 rec_allocator_of(u64 job) {
    g_rec.push_back({kAllocatorOfId, {job, 0, 0}});
    return base() + kJobAlloc;
}
GUEST_ABI void rec_panic(u64 file, u64 line, u64 text) { g_rec.push_back({kPanicId, {file, line & 0xffffffffull, text}}); }

int g_failed = 0;
bool g_quiet = false;
void fail(int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: case %d: %s\n", trial, why);
}

u64 garbage() { return (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32(); }

void build_case() {
    for (u8& b : g_mem) b = static_cast<u8>(g_rng.u32());
    const u64 b = base();
    // vtables
    store<u64>(b + kVtLock + 0x18, reinterpret_cast<u64>(&rec_lock));
    store<u64>(b + kVtLock + 0x28, reinterpret_cast<u64>(&rec_unlock));
    store<u64>(b + kVtListLock + 0x18, reinterpret_cast<u64>(&rec_lock));
    store<u64>(b + kVtListLock + 0x28, reinterpret_cast<u64>(&rec_unlock));
    store<u64>(b + kVtEvent + 0x28, reinterpret_cast<u64>(&rec_event));
    store<u64>(b + kVtPoolAlloc + 0x70, reinterpret_cast<u64>(&rec_pool_free));
    store<u64>(b + kVtJobAlloc + 0x70, reinterpret_cast<u64>(&rec_job_free));
    store<u64>(b + kVtJob + 0x0, reinterpret_cast<u64>(&rec_destroy));
    store<u64>(b + kVtJob + 0x10, reinterpret_cast<u64>(&rec_run));
    store<u64>(b + kVtGroup + 0x0, reinterpret_cast<u64>(&rec_group_destroy));
    // the pool's objects
    store<u64>(b + kPool + 0x30, b + kVtLock);
    store<u64>(b + kPool + 0x70, b + kVtListLock);
    store<u64>(b + kPool + 0x90, b + kVtEvent);
    store<u64>(b + kPool + 0x28, b + kPoolAlloc);
    store<u64>(b + kPoolAlloc, b + kVtPoolAlloc);
    store<u64>(b + kJobAlloc, b + kVtJobAlloc);
    // jobs and groups (up to 6 each); a job's kind 0/1/2 or a bad one, its group none or one
    for (int j = 0; j < 6; ++j) {
        const u64 job = b + kJobs + 0x40ull * static_cast<u64>(j);
        store<u64>(job, b + kVtJob);
        const int kr = g_rng.range(0, 9);
        store<u32>(job + 0x10, kr < 3 ? 0u : kr < 6 ? 1u : kr < 8 ? 2u : static_cast<u32>(g_rng.range(3, 100)));
        if (g_rng.chance(0.75)) {
            const u64 group = b + kGroups + 0x40ull * static_cast<u64>(j);
            store<u64>(job + 8, group);
            store<u64>(group, b + kVtGroup);
            const s32 counts[] = {1, 2, 3, 0, -1, static_cast<s32>(g_rng.u32())};
            store<s32>(group + 0x30, counts[g_rng.range(0, 5)]);
            store<s32>(group + 0x8, counts[g_rng.range(0, 5)]);
        } else {
            store<u64>(job + 8, 0);
        }
    }
    // the list: an anchor and n nodes, circular; the count agrees, or not
    const u64 anchor = b + kAnchor;
    const int n = g_rng.chance(0.15) ? 0 : g_rng.range(1, 5);
    u64 prev = anchor;
    for (int k = 0; k < n; ++k) {
        const u64 node = b + kNodes + 0x20ull * static_cast<u64>(k);
        const u64 job = g_rng.chance(0.08) ? 0 : b + kJobs + 0x40ull * static_cast<u64>(g_rng.range(0, 5));
        store<u64>(node + 0x10, job);
        store<u64>(node + 8, prev);
        store<u64>(prev, node);
        prev = node;
    }
    store<u64>(prev, anchor);
    store<u64>(anchor + 8, prev);
    // the anchor's own +0x10: what the game reads as the job when the list is empty
    store<u64>(anchor + 0x10, g_rng.chance(0.5) ? 0 : b + kJobs + 0x40ull * static_cast<u64>(g_rng.range(0, 5)));
    store<u64>(b + kPool + 0x18, anchor);
    u64 count = static_cast<u64>(n);
    if (g_rng.chance(0.1)) count = static_cast<u64>(g_rng.range(1, 3));  // disagrees (n may be 0)
    if (g_rng.chance(0.05)) count = 0;
    store<u64>(b + kPool + 0x20, count);
    g_signal_takes_group = g_rng.chance(0.2);
}

long run(int cases) {
    using F = GUEST_ABI u64 (*)(u64);
    static u8 start[sizeof g_mem], after_game[sizeof g_mem];
    long n = 0;
    for (int t = 0; t < cases; ++t) {
        build_case();
        std::memcpy(start, g_mem, sizeof g_mem);
        // the job whose group a completion may take: the one the pop will find
        {
            const u64 anchor = load<u64>(base() + kPool + 0x18);
            g_signal_job = load<u64>(load<u64>(anchor) + 0x10);
        }
        g_rec.clear();
        const u64 game_ret = eboot_kit::fn<F>(kProcessOne)(base() + kPool);
        const std::vector<Call> game = g_rec;
        std::memcpy(after_game, g_mem, sizeof g_mem);

        std::memcpy(g_mem, start, sizeof g_mem);
        g_rec.clear();
        const u64 our_ret = reinterpret_cast<F>(&ez_work_process_one)(base() + kPool);
        ++n;
        if ((our_ret & 0xffffffffull) != (game_ret & 0xffffffffull)) fail(t, "the return value");
        if (g_rec != game) fail(t, "the calls");
        if (std::memcmp(g_mem, after_game, sizeof g_mem) != 0) fail(t, "the memory");
    }
    return n;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_ez_mutant = mutant;
    g_quiet = true;
    run(3000);
    g_quiet = false;
    g_ez_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("ez_work_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 20000;
    eboot_kit::stub(kSignalGroup, reinterpret_cast<const void*>(&rec_signal_group));
    eboot_kit::stub(kPushDone, reinterpret_cast<const void*>(&rec_push_done));
    eboot_kit::stub(kAllocatorOf, reinterpret_cast<const void*>(&rec_allocator_of));
    eboot_kit::stub(kPanic, reinterpret_cast<const void*>(&rec_panic));
    const u8* at = eboot_kit::file_at("ez_work_test", kProcessOne);
    if (!at || std::memcmp(at, kEntry, sizeof kEntry) != 0) fail(-1, "the entry's instructions");
    const long total = run(cases);
    for (int mutant = 1; mutant <= 6; ++mutant) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("ez_work_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("ez_work_test: ok - %ld cases, 6 mutants caught\n", total);
    return 0;
}
