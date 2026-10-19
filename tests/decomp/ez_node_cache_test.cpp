// The work pools' node cache (decomp/sync/ez_work.cpp) through the game's own
// code (tests/decomp/eboot_kit.h): a pool built in the test's memory as
// CSEzWork's constructor leaves one - its job list, finished list and
// workers' list, each with a sentinel from the general allocator - and that
// allocator a recorder heap (the class's vtable, 0x57aefa0, its alloc and free
// slots aimed at the recorder). Two arms run the same generated sequence:
// pushes through the game's Enqueue (0x24320b0) and its unwoken twin
// (0x24322e0), both CSEzWorkList::Append; pops; the game's DeleteCompleted
// (0x2431be0); direct allocations the cache must pass on; then the game's
// ~CSEzWork (0x2431cf0). Arm A pops with the game's ProcessOne alone, so no
// cache; arm B with ours, which puts the cache in, and the game's at random
// after it, which then frees into the cache. Jobs of every kind (0 freed by
// the pop, 1 to the finished list, 2 the owner's), runs of up to 1,500 to
// fill the cache past its cap, and pools whose list allocates from another
// heap of the same class, which must keep it. The arms must run, destroy and
// free the same jobs in the same order, signal the same events and hold the
// same lists after every step; every block of the heap must come back exactly
// once, to the allocator it came from; the cache must stay under its cap, be
// called only with the pool's lock held (or from the destructor) and be free
// again when the pool is gone. Eight deliberate mistakes must be caught.
// Skips without the 1.09 eboot.
#define DECOMP_EZ_WORK_MUTANTS
int g_ez_mutant = 0;
bool g_queue_held = false, g_in_destructor = false, g_direct = false;
int g_unlocked_calls = 0;
#define EZ_NODE_CACHE_CHECK(self)                                         \
    do {                                                                  \
        if (!g_queue_held && !g_in_destructor && !g_direct) ++g_unlocked_calls; \
    } while (0)
#include "decomp/sync/ez_work.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace {

constexpr u64 kEnqueue = 0x24320b0;
constexpr u64 kPushQuiet = 0x24322e0;
constexpr u64 kDeleteCompleted = 0x2431be0;
constexpr u64 kDestructor = 0x2431cf0;

eboot_kit::Rng g_rng(0x5ca7);

// What both arms must agree on.
struct Event {
    int what;
    u64 a, b;
    bool operator==(const Event&) const = default;
};
enum { kRun, kJobDtor, kJobFree, kSignal, kClose, kPanicked, kQueue, kFinished, kSema };
std::vector<Event> g_events;

// The heap behind the general allocator (and a second one of the same class).
struct Heap {
    std::map<u64, std::pair<u64, u64>> live;  // block -> (size, the allocator object it came from)
    std::vector<u64> spare;                   // freed blocks, reused last-in first-out as a heap does
    u64 allocs = 0, last = 0;
    int bad_frees = 0;
};
Heap g_heap;
alignas(64) u8 g_arena[0x40 * 8192];
u64 g_arena_used = 0;

alignas(16) u8 g_objects[0x100];
u64 general() { return reinterpret_cast<u64>(g_objects); }        // the general allocator object
u64 other() { return reinterpret_cast<u64>(g_objects) + 0x40; }   // another heap's, the same class
u64 jobs_from() { return reinterpret_cast<u64>(g_objects) + 0x80; }  // the jobs' allocator

alignas(64) u8 g_pool_mem[0x200];
u64 pool() { return reinterpret_cast<u64>(g_pool_mem); }
alignas(64) u8 g_vtables[0x400];
u64 vt(int i) { return reinterpret_cast<u64>(g_vtables) + 0x80ull * static_cast<u64>(i); }
constexpr int kVtQueueLock = 0, kVtListLock = 1, kVtEvent = 2, kVtJobs = 3, kVtJob = 4;

constexpr int kJobsMax = 2048;
alignas(64) u8 g_jobs[0x20 * kJobsMax];
u64 job_at(int i) { return reinterpret_cast<u64>(g_jobs) + 0x20ull * static_cast<u64>(i); }

GUEST_ABI u64 rec_heap_alloc(u64 self, u64 size, u64 align) {
    (void)align;
    u64 p;
    if (!g_heap.spare.empty()) {
        p = g_heap.spare.back();
        g_heap.spare.pop_back();
    } else {
        p = reinterpret_cast<u64>(g_arena) + g_arena_used;
        g_arena_used += 0x40;
        if (g_arena_used > sizeof g_arena) std::abort();
    }
    std::memset(reinterpret_cast<void*>(p), 0xa5, 0x40);
    g_heap.live[p] = {size, self};
    ++g_heap.allocs;
    g_heap.last = p;
    return p;
}
GUEST_ABI u64 rec_heap_free(u64 self, u64 p) {
    if (p == 0) return 0;
    auto it = g_heap.live.find(p);
    if (it == g_heap.live.end() || it->second.second != self) {
        ++g_heap.bad_frees;  // not allocated, freed twice, or to another allocator than its own
        return 0;
    }
    g_heap.live.erase(it);
    g_heap.spare.push_back(p);
    return 0;
}
GUEST_ABI u64 rec_lock(u64 self, u64) {
    if (self == pool() + 0x30) g_queue_held = true;
    return 0;
}
GUEST_ABI u64 rec_unlock(u64 self) {
    if (self == pool() + 0x30) g_queue_held = false;
    return 0;
}
GUEST_ABI u64 rec_event_signal(u64, u64 v) {
    g_events.push_back({kSignal, v & 0xffffffffull, 0});
    return 0;
}
GUEST_ABI u64 rec_event_close(u64 self) {
    g_events.push_back({kClose, self - pool(), 0});
    return 0;
}
GUEST_ABI u64 rec_job_run(u64 self) {
    g_events.push_back({kRun, self, 0});
    return 0;
}
GUEST_ABI u64 rec_job_dtor(u64 self) {
    g_events.push_back({kJobDtor, self, 0});
    return 0;
}
GUEST_ABI u64 rec_job_free(u64, u64 p) {
    g_events.push_back({kJobFree, p, 0});
    return 0;
}
GUEST_ABI u64 rec_signal_group(u64, u64) { return 0; }
GUEST_ABI u64 rec_allocator_of(u64) { return jobs_from(); }
GUEST_ABI void rec_panic(u64, u64 line, u64) { g_events.push_back({kPanicked, line & 0xffffffffull, 0}); }
GUEST_ABI int rec_delete_sema(u64 handle) {
    g_events.push_back({kSema, handle, 0});
    return 0;
}

int g_failed = 0;
const bool g_trace = std::getenv("EZ_TEST_TRACE") != nullptr;
bool g_quiet = false;
void fail(int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: case %d: %s\n", trial, why);
}

using Alloc = GUEST_ABI u64 (*)(u64, u64, u64);
using Free = GUEST_ABI u64 (*)(u64, u64);
u64 alloc_through(u64 object, u64 size, u64 align) {
    return reinterpret_cast<Alloc>(static_cast<std::uintptr_t>(load<u64>(load<u64>(object) + 0x58)))(object, size, align);
}
u64 free_through(u64 object, u64 p) {
    return reinterpret_cast<Free>(static_cast<std::uintptr_t>(load<u64>(load<u64>(object) + 0x70)))(object, p);
}

// A list's sentinel, as the constructor makes it: from its allocator directly.
u64 sentinel(u64 from) {
    const u64 s = alloc_through(from, 0x18, 8);
    store<u64>(s, s);
    store<u64>(s + 8, s);
    return s;
}

// The list's jobs in order (bounded: a broken list is a failure, not a hang).
bool walk(u64 head, std::vector<u64>* out) {
    out->clear();
    u64 n = load<u64>(head);
    for (int k = 0; n != head; ++k) {
        if (k > kJobsMax + 8) return false;
        out->push_back(load<u64>(n + 0x10));
        n = load<u64>(n);
    }
    return true;
}

void build_pool(bool other_heap) {
    std::memset(g_pool_mem, 0, sizeof g_pool_mem);
    const u64 p = pool();
    store<u64>(p + 0x18, sentinel(other_heap ? other() : general()));
    store<u64>(p + 0x20, 0);
    store<u64>(p + 0x28, other_heap ? other() : general());
    store<u64>(p + 0x30, vt(kVtQueueLock));
    store<u64>(p + 0x40, 0x5e3a0001);  // the queue lock's semaphore
    store<u64>(p + 0x58, sentinel(general()));
    store<u64>(p + 0x60, 0);
    store<u64>(p + 0x68, general());
    store<u64>(p + 0x70, vt(kVtListLock));
    store<u64>(p + 0x80, 0x5e3a0002);  // the list lock's
    store<u64>(p + 0x90, vt(kVtEvent));
    store<u64>(p + 0xd0, sentinel(general()));
    store<u64>(p + 0xd8, 0);
    store<u64>(p + 0xe0, general());
}

struct Arm {
    std::vector<Event> events;
    u64 heap_allocs = 0;
    bool ok = true;
};

// One case's sequence, generated from `seed`, on one arm.
Arm run_arm(int trial, u64 seed, bool ours, bool other_heap) {
    eboot_kit::Rng rng(seed);
    g_events.clear();
    g_heap = Heap{};
    g_arena_used = 0;
    build_pool(other_heap);
    std::memset(g_jobs, 0, sizeof g_jobs);
    Arm arm;
    int next_job = 0;
    const bool big = rng.chance(0.12);
    const int steps = big ? 2 : rng.range(1, 60);
    using F1 = GUEST_ABI u64 (*)(u64);
    using F2 = GUEST_ABI u64 (*)(u64, u64);
    const auto snapshot = [&] {
        std::vector<u64> q, f;
        if (!walk(load<u64>(pool() + 0x18), &q) || !walk(load<u64>(pool() + 0x58), &f)) {
            arm.ok = false;
            return;
        }
        for (u64 j : q) g_events.push_back({kQueue, j, 0});
        for (u64 j : f) g_events.push_back({kFinished, j, 0});
        g_events.push_back({kQueue, load<u64>(pool() + 0x20), load<u64>(pool() + 0x60)});
        for (const NodeCache& c : g_caches)
            if (c.count > kCap) {
                fail(trial, "a cache past its cap");
                arm.ok = false;
            }
    };
    const auto push_one = [&] {
        if (next_job >= kJobsMax) return;
        const u64 j = job_at(next_job++);
        store<u64>(j, vt(kVtJob));
        store<u64>(j + 8, 0);
        const int k = rng.range(0, 9);
        store<u32>(j + 0x10, big ? 2u : k < 4 ? 0u : k < 7 ? 1u : 2u);
        if (rng.chance(0.5))
            eboot_kit::fn<F2>(kEnqueue)(pool(), j);
        else
            eboot_kit::fn<F2>(kPushQuiet)(pool(), j);
    };
    const auto pop_one = [&] {
        if (ours && rng.chance(0.7))
            reinterpret_cast<F1>(&ez_work_process_one)(pool());
        else
            eboot_kit::fn<F1>(kProcessOne)(pool());
        if (!ours) (void)rng.chance(0.7);  // the same draws in both arms
    };
    for (int s = 0; s < steps && arm.ok; ++s) {
        // a big case: 1,100-1,500 pushes, then every pop and some on an empty list
        const int what = big ? (s == 0 ? 0 : 4) : rng.range(0, 9);
        if (g_trace) std::fprintf(stderr, "case %d arm %d step %d what %d\n", trial, ours ? 1 : 0, s, what);
        if (what <= 3) {
            const int n = big ? rng.range(1100, 1500) : rng.range(1, 6);
            for (int i = 0; i < n; ++i) push_one();
        } else if (what <= 7) {
            const int n = big ? 1600 : rng.range(1, 7);
            for (int i = 0; i < n; ++i) pop_one();
        } else if (what == 8) {
            eboot_kit::fn<F1>(kDeleteCompleted)(pool());
        } else {
            // what the cache must pass on: other sizes, a stricter alignment
            g_direct = true;
            const u64 slot = load<u64>(pool() + 0x28);
            const u64 sizes[][2] = {{0x30, 8}, {0x18, 16}, {0x10, 8}};
            for (const auto& sz : sizes) {
                const u64 before = g_heap.allocs;
                const u64 b = alloc_through(slot, sz[0], sz[1]);
                const auto it = g_heap.live.find(b);
                if (g_heap.allocs != before + 1 || b != g_heap.last || it == g_heap.live.end() || it->second.first != sz[0]) {
                    fail(trial, "a block that is not a node not passed on to the heap");
                    arm.ok = false;
                }
                free_through(it == g_heap.live.end() ? general() : it->second.second, b);
            }
            g_direct = false;
        }
        snapshot();
    }
    if (!arm.ok) {
        fail(trial, ours ? "a list broken (ours)" : "a list broken (the game's)");
        return arm;  // no destructor over a broken list: it would not end
    }
    if (ours && other_heap && load<u64>(pool() + 0x28) != other()) {
        fail(trial, "a cache put in a list that allocates from another heap");
        arm.ok = false;
    }
    // ~CSEzWork: the pool goes, and with it the cache
    if (g_trace) std::fprintf(stderr, "case %d arm %d destructor\n", trial, ours ? 1 : 0);
    g_in_destructor = true;
    eboot_kit::fn<F1>(kDestructor)(pool());
    g_in_destructor = false;
    arm.events = g_events;
    arm.heap_allocs = g_heap.allocs;
    if (!g_heap.live.empty()) {
        fail(trial, ours ? "blocks not back in the heap after the pool's end (ours)" : "blocks not back in the heap (the game's)");
        arm.ok = false;
    }
    if (g_heap.bad_frees) {
        fail(trial, "a block freed twice, unallocated or to another allocator");
        arm.ok = false;
    }
    for (const NodeCache& c : g_caches)
        if (c.state.load() != kFree) {
            fail(trial, "a cache not free after its pool's end");
            arm.ok = false;
            break;
        }
    return arm;
}

void reset_caches() {
    for (NodeCache& c : g_caches) {
        c.state.store(kFree);
        c.top = 0;
        c.count = 0;
    }
}

long run(int cases) {
    long n = 0;
    for (int t = 0; t < cases; ++t) {
        const u64 seed = g_rng.next();
        const bool other_heap = g_rng.chance(0.1);
        reset_caches();
        const Arm game = run_arm(t, seed, false, other_heap);
        reset_caches();
        const int unlocked_before = g_unlocked_calls;
        const Arm mine = run_arm(t, seed, true, other_heap);
        ++n;
        if (!game.ok || !mine.ok) continue;
        if (mine.events != game.events) fail(t, "the jobs' runs, frees, signals or the lists differ");
        if (mine.heap_allocs > game.heap_allocs) fail(t, "more blocks from the heap with the cache than without");
        if (g_unlocked_calls != unlocked_before) fail(t, "the cache called without the pool's lock");
    }
    return n;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_ez_mutant = mutant;
    g_quiet = true;
    run(400);
    g_quiet = false;
    g_ez_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("ez_node_cache_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 3000;
    eboot_kit::stub(kSignalGroup, reinterpret_cast<const void*>(&rec_signal_group));
    eboot_kit::stub(kAllocatorOf, reinterpret_cast<const void*>(&rec_allocator_of));
    eboot_kit::stub(kPanic, reinterpret_cast<const void*>(&rec_panic));
    if (!eboot_kit::bind("sceKernelDeleteSema", reinterpret_cast<const void*>(&rec_delete_sema))) fail(-1, "no sceKernelDeleteSema to bind");
    // The general allocator's class: its alloc and free slots aimed at the recorder heap.
    store<u64>(game_address(kAllocatorVtable) + 0x58, reinterpret_cast<u64>(&rec_heap_alloc));
    store<u64>(game_address(kAllocatorVtable) + 0x70, reinterpret_cast<u64>(&rec_heap_free));
    store<u64>(general(), game_address(kAllocatorVtable));
    store<u64>(general() + 8, 0x4ea90001);  // its heap: never reached
    store<u64>(other(), game_address(kAllocatorVtable));
    store<u64>(other() + 8, 0x4ea90002);
    store<u64>(game_address(kGeneralAllocator), general());
    // the jobs' allocator, the locks, the event, a job
    store<u64>(jobs_from(), vt(kVtJobs));
    store<u64>(vt(kVtJobs) + 0x70, reinterpret_cast<u64>(&rec_job_free));
    for (int v : {kVtQueueLock, kVtListLock}) {
        store<u64>(vt(v) + 0x18, reinterpret_cast<u64>(&rec_lock));
        store<u64>(vt(v) + 0x28, reinterpret_cast<u64>(&rec_unlock));
    }
    store<u64>(vt(kVtEvent) + 0x28, reinterpret_cast<u64>(&rec_event_signal));
    store<u64>(vt(kVtEvent) + 0x30, reinterpret_cast<u64>(&rec_event_close));
    store<u64>(vt(kVtJob) + 0x0, reinterpret_cast<u64>(&rec_job_dtor));
    store<u64>(vt(kVtJob) + 0x10, reinterpret_cast<u64>(&rec_job_run));

    const long total = run(cases);
    // the cache was used: put in, nodes taken back, kept, passed on past its cap, given back at the end
    u64 hits = 0, kept = 0, passed = 0, ends = 0;
    for (const NodeCache& c : g_caches) {
        hits += c.hits;
        kept += c.kept;
        passed += c.passed;
        ends += c.ends;
    }
    if (!hits || !kept || !passed || !ends) fail(-1, "the cache was not exercised (no takes, keeps, passes or ends)");
    const int mutants[] = {7, 8, 9, 10, 11, 12, 13, 14};
    for (int m : mutants) {
        if (!caught(m)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", m);
        }
    }
    if (g_failed) {
        std::printf("ez_node_cache_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("ez_node_cache_test: ok - %ld cases (%llu nodes taken from a cache, %llu kept, %llu passed on, %llu pools ended), "
                "8 mutants caught\n",
                total, static_cast<ull>(hits), static_cast<ull>(kept), static_cast<ull>(passed), static_cast<ull>(ends));
    return 0;
}
