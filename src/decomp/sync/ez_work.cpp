// The engine's work pools' job pop as source (docs/decomp.md):
// NS_SPRJ::CSEzWork::ProcessOne, Binary Ninja 0x2432370. Every worker of a
// pool - the six CSChrThread workers among them - calls it in a loop
// (CSEzWorkWorker::Run): take the pool's queue lock (a DLAdaptiveMutex, at
// +0x30), unlink the first node of the job list (+0x18, its count at +0x20),
// free the node through the list's allocator (+0x28), let the lock go, run
// the job, then settle it - the group it belongs to (+0x8: a countdown at
// +0x30 whose last tick signals the group, a reference count at +0x8) and, by
// its kind (+0x10), nothing more (2), the finished list (1: pushed at +0x50
// under the lock at +0x70, then the event at +0x90 signalled with 0x43) or its
// own destruction and free (0). Ours makes the game's calls in the game's
// order.
//
// The node cache. Every push of a job allocates its list node (0x18 bytes,
// CSEzWorkList::Append 0x2432780) and every pop frees one, both under the
// queue lock and both through the engine's general allocator (the object
// data_5940420 points at), whose heap takes a lock of its own: at Central
// Yharnam the character pool pushes ~375 jobs a frame, the heap's free sat
// inside the queue lock's hold, and the six workers convoyed on that lock -
// 46-47% of its takes found it held, 3-3.5 ms a frame of waiting, ~60% of
// their time. Freeing the node after the unlock instead only moved the convoy
// onto the heap's lock, which then met all of the pool's churn. So each pool
// gets a cache of its own nodes: the first time ProcessOne sees a pool whose
// job list still allocates from the general allocator, it puts a stand-in in
// the list's allocator slot - a copy of that allocator object (its class's
// vtable, 0x57aefa0, and its heap) whose aligned alloc (+0x58) and free
// (+0x70) are ours. A free keeps the node, up to 1,024 a pool; an alloc of a
// node takes one back; anything else, and a cache that is empty or full, is
// passed to the general allocator. Every alloc and free on a pool's job list
// runs under that pool's queue lock (Enqueue 0x24320b0, its unwoken twin
// 0x24322e0, the character jobs' submitters, ProcessOne) or in its destructor,
// so the cache needs no lock of its own. The list's sentinel was allocated by
// the constructor from the general allocator directly, and ~CSEzWork
// (0x2431cf0) frees it last, through the slot: that free is the pool's end,
// and the cache gives every node it holds back first. BBHOST_EZ_NODE_CACHE=0
// leaves the pools as they are (ProcessOne still ours); BBHOST_DECOMP_OFF with
// ProcessOne's name keeps the game's pop as well.
//
// Leaves: ProcessOne is entered from the game's code on its thread, and the
// cache's alloc and free are called by the game's code through the stand-in's
// vtable, also on its thread - none of them goes through the FS thunk, so
// none touches the host's thread-locals, takes a lock or allocates, and the
// file is built without the stack protector (no canary check).
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

using namespace decomp;

#ifdef DECOMP_EZ_WORK_MUTANTS
extern int g_ez_mutant;
#define MUTANT(n) (g_ez_mutant == (n))
#else
#define MUTANT(n) false
#endif

// A test's check on every call of the cache (that its pool's lock is held).
#ifndef EZ_NODE_CACHE_CHECK
#define EZ_NODE_CACHE_CHECK(self) ((void)0)
#endif

namespace {

constexpr u64 kProcessOne = 0x2432370;   // NS_SPRJ::CSEzWork::ProcessOne
constexpr u64 kSignalGroup = 0x247e780;  // the group's completion: (group + 0x10, 1)
constexpr u64 kPushDone = 0x2432780;     // CSEzWorkList::Append: the finished list's push, (pool + 0x50, &job)
constexpr u64 kAllocatorOf = 0x247b720;  // the allocator a job came from
constexpr u64 kPanic = 0x24b55b0;        // DL_PANIC(file, line, text, ...)
constexpr u64 kUnrefText = 0x4d40331;    // "DLReferenceCountObject: Invalid Unref() call. ..."
constexpr u64 kPoolFile = 0x4d3afb6;     // CSEzWork.cpp's path
constexpr u64 kPoolText = 0x4d3aff1;
constexpr s32 kPoolLine = 0x168;

// The engine's general allocator: the pointer every CSEzWork's lists take at
// construction (and some 770 other functions), and its class's vtable - 28
// slots, +0x58 the aligned alloc (calls_object_allocator_mb, 0x23c5f30) and
// +0x70 the free (0x23c5fc0), each taking the heap's lock; every method reads
// only the object's +0x8, its heap.
constexpr u64 kGeneralAllocator = 0x5940420;
constexpr u64 kAllocatorVtable = 0x57aefa0;
constexpr int kVtableSlots = 28;
constexpr u64 kAllocSlot = 0x58, kFreeSlot = 0x70;

constexpr u64 kNodeSize = 0x18;  // a std::list<T*> node: next, prev, value
constexpr u32 kCap = 1024;       // nodes a pool's cache keeps (24 KB)
constexpr int kCaches = 32;      // pools with a cache

// push rbp; mov rbp, rsp; push r15; push r14 (8 bytes, whole instructions; the
// pushes of r13, r12 and rbx and the stack guard's load follow).
constexpr std::uint8_t kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56};

using LockFn = GUEST_ABI u64 (*)(u64 self, u64 timeout);
using UnlockFn = GUEST_ABI u64 (*)(u64 self);
using AllocFn = GUEST_ABI u64 (*)(u64 self, u64 size, u64 align);
using FreeFn = GUEST_ABI u64 (*)(u64 self, u64 p);
using OneFn = GUEST_ABI u64 (*)(u64 self);
using SignalFn = GUEST_ABI u64 (*)(u64 self, u64 value);
using PanicFn = GUEST_ABI void (*)(u64 file, s32 line, u64 text, ...);  // the game clears al: variadic

// The virtual method at `offset` of an object's vtable.
template <class F>
inline F method(u64 object, u64 offset) {
    return reinterpret_cast<F>(static_cast<std::uintptr_t>(load<u64>(load<u64>(object) + offset)));
}

// The game's `lock xadd dword [p], -1`: the old value.
inline s32 count_down(u64 p) { return reinterpret_cast<std::atomic<s32>*>(static_cast<std::uintptr_t>(p))->fetch_sub(1); }

const bool g_cache_on = [] {
    const char* e = std::getenv("BBHOST_EZ_NODE_CACHE");
    return !(e && *e == '0');
}();

// A pool's stand-in allocator. What the game sees of it is an allocator
// object - the vtable pointer, then the general allocator's heap - and the
// rest is the cache's.
struct alignas(64) NodeCache {
    u64 vtable;    // -> slots
    u64 heap;      // the general allocator's +0x8, which the methods not replaced read
    u64 original;  // the general allocator object it stands in for
    u64 sentinel;  // the pool's job list's sentinel: its free is the pool's end
    u64 top;       // the nodes kept, linked through their first word
    u32 count;
    std::atomic<u32> state;  // kFree, kClaimed, kInUse
    // what the exit report says (written under the pool's lock; kept across a slot's reuse)
    u64 hits, misses, kept, passed, ends;
    u64 slots[kVtableSlots];
};
enum : u32 { kFree, kClaimed, kInUse };
NodeCache g_caches[kCaches];
std::atomic<u32> g_cache_full{0};  // pools that found every cache taken

DECOMP_LEAF u64 node_cache_alloc(u64 self, u64 size, u64 align);
DECOMP_LEAF u64 node_cache_free(u64 self, u64 p);

inline NodeCache& cache_of(u64 self) { return *reinterpret_cast<NodeCache*>(static_cast<std::uintptr_t>(self)); }

// The pool's list allocator from now on: a cache of its own, if one is free.
// Under the pool's queue lock.
inline void install(u64 pool, u64 general) {
    for (NodeCache& c : g_caches) {
        u32 expect = kFree;
        if (!c.state.compare_exchange_strong(expect, kClaimed, std::memory_order_acquire)) continue;
        const u64 vt = load<u64>(general);
        for (int i = 0; i < kVtableSlots; ++i) c.slots[i] = load<u64>(vt + 8ull * static_cast<u64>(i));
        c.slots[kAllocSlot / 8] = reinterpret_cast<u64>(&node_cache_alloc);
        c.slots[kFreeSlot / 8] = reinterpret_cast<u64>(&node_cache_free);
        c.vtable = reinterpret_cast<u64>(&c.slots[0]);
        c.heap = load<u64>(general + 8);
        c.original = general;
        c.sentinel = load<u64>(pool + 0x18);
        c.top = 0;
        c.count = 0;
        c.state.store(kInUse, std::memory_order_release);
        store<u64>(pool + 0x28, MUTANT(13) ? general : reinterpret_cast<u64>(&c));
        return;
    }
    g_cache_full.fetch_add(1, std::memory_order_relaxed);
}

// Each pop looks: a pool whose job list still allocates from the general
// allocator gets its cache. Under the pool's queue lock.
inline void maybe_install(u64 pool) {
    if (!g_cache_on) return;
    const u64 general = load<u64>(game_address(kGeneralAllocator));
    const u64 slot = load<u64>(pool + 0x28);
    if (general == 0 || (slot != general && !MUTANT(12))) return;
    if (load<u64>(slot) != game_address(kAllocatorVtable)) return;  // not the class it was written against
    install(pool, slot);
}

DECOMP_LEAF u64 node_cache_alloc(u64 self, u64 size, u64 align) {
    EZ_NODE_CACHE_CHECK(self);
    NodeCache& c = cache_of(self);
    if ((size == kNodeSize || MUTANT(11)) && align <= 8 && c.top) {
        const u64 node = c.top;
        if (!MUTANT(7)) c.top = load<u64>(node);
        --c.count;
        ++c.hits;
        return node;
    }
    ++c.misses;
    return method<AllocFn>(c.original, kAllocSlot)(c.original, size, align);
}

DECOMP_LEAF u64 node_cache_free(u64 self, u64 p) {
    EZ_NODE_CACHE_CHECK(self);
    NodeCache& c = cache_of(self);
    const u64 original = c.original;
    if (p != 0 && p == c.sentinel && !MUTANT(10)) {
        // ~CSEzWork's last free: the pool is gone, and so is its cache.
        if (!MUTANT(9)) {
            // exactly the nodes counted in: a broken link cannot run on
            for (u32 n = c.count < kCap ? c.count : kCap; n != 0 && c.top; --n) {
                const u64 node = c.top;
                c.top = load<u64>(node);
                method<FreeFn>(original, kFreeSlot)(original, node);
            }
            c.top = 0;
            c.count = 0;
        }
        ++c.ends;
        c.sentinel = 0;
        if (!MUTANT(14)) c.state.store(kFree, std::memory_order_release);
        return method<FreeFn>(original, kFreeSlot)(original, p);
    }
    if (p != 0 && (c.count < kCap || MUTANT(8))) {
        store<u64>(p, c.top);
        c.top = p;
        ++c.count;
        ++c.kept;
        return 0;
    }
    ++c.passed;
    return method<FreeFn>(original, kFreeSlot)(original, p);
}

DECOMP_LEAF u32 ez_work_process_one(u64 pool) {
    const u64 lock = pool + 0x30;
    method<LockFn>(lock, 0x18)(lock, 0xffffffffu);
    u64 job = 0;
    if (load<u64>(pool + 0x20) != 0) {
        const u64 anchor = load<u64>(pool + 0x18);
        const u64 first = load<u64>(anchor);
        job = load<u64>(first + 0x10);  // read before the empty-list check, as the game's
        if (first != anchor && !MUTANT(1)) {
            // prev->next = next, then next->prev = prev, each read again.
            store<u64>(load<u64>(first + 8), load<u64>(first));
            store<u64>(load<u64>(first) + 8, load<u64>(first + 8));
            const u64 allocator = load<u64>(pool + 0x28);
            method<FreeFn>(allocator, 0x70)(allocator, MUTANT(3) ? job : first);
            store<u64>(pool + 0x20, load<u64>(pool + 0x20) - (MUTANT(2) ? 0 : 1));
        }
    }
    maybe_install(pool);
    method<UnlockFn>(lock, 0x28)(lock);
    if (!job) return 0;

    method<OneFn>(job, 0x10)(job);  // run it
    if (const u64 group = load<u64>(job + 8)) {
        if (count_down(group + 0x30) <= (MUTANT(4) ? 0 : 1)) game_function<SignalFn>(kSignalGroup)(group + 0x10, 1);
        if (const u64 again = load<u64>(job + 8)) {
            const s32 refs = count_down(again + 8);
            if (refs == 1) {
                method<OneFn>(again, 0)(again);  // the last reference: destroyed
            } else if (refs <= 0) {
                game_function<PanicFn>(kPanic)(0, 0x3e, game_address(kUnrefText));
            }
            if (!MUTANT(5)) store<u64>(job + 8, 0);
        }
    }
    const u32 kind = load<u32>(job + 0x10);
    if (kind == 2) return 1;
    if (kind == 1) {
        u64 finished = job;  // the push takes a pointer to it
        const u64 list_lock = pool + 0x70;
        method<LockFn>(list_lock, 0x18)(list_lock, 0xffffffffu);
        game_function<SignalFn>(kPushDone)(pool + 0x50, reinterpret_cast<u64>(&finished));
        method<UnlockFn>(list_lock, 0x28)(list_lock);
        const u64 event = pool + 0x90;
        method<SignalFn>(event, 0x28)(event, MUTANT(6) ? 0x42 : 0x43);
        return 1;
    }
    if (kind == 0) {
        const u64 from = game_function<OneFn>(kAllocatorOf)(job);
        method<OneFn>(job, 0)(job);  // its destructor
        method<FreeFn>(from, 0x70)(from, job);
        return 1;
    }
    game_function<PanicFn>(kPanic)(game_address(kPoolFile), kPoolLine, game_address(kPoolText));
    return 1;
}

// Hosted, at exit.
void ez_work_report() {
    u64 pools = 0, live = 0, hits = 0, misses = 0, kept = 0, passed = 0, ends = 0, held = 0;
    for (const NodeCache& c : g_caches) {
        if (c.state.load(std::memory_order_acquire) == kInUse) {
            ++live;
            held += c.count;
        }
        if (c.hits || c.misses || c.kept || c.passed || c.state.load(std::memory_order_relaxed) == kInUse) ++pools;
        hits += c.hits;
        misses += c.misses;
        kept += c.kept;
        passed += c.passed;
        ends += c.ends;
    }
    if (!g_cache_on) {
        host_log("decomp: CSEzWork's node cache off (BBHOST_EZ_NODE_CACHE=0)");
        return;
    }
    host_log("decomp: CSEzWork's node cache - %llu pools (%llu live, holding %llu nodes; %llu ended); allocs %llu from a cache, %llu "
             "passed on; frees %llu kept, %llu passed on; %u pools found every cache taken",
             static_cast<ull>(pools), static_cast<ull>(live), static_cast<ull>(held), static_cast<ull>(ends), static_cast<ull>(hits),
             static_cast<ull>(misses), static_cast<ull>(kept), static_cast<ull>(passed), g_cache_full.load(std::memory_order_relaxed));
}

}  // namespace

void decomp_ez_work_add() {
    DecompFunction fn{"NS_SPRJ::CSEzWork::ProcessOne", "sync", kProcessOne, kEntry, sizeof(kEntry),
                      reinterpret_cast<void*>(&ez_work_process_one), DecompKind::Leaf};
    fn.report = &ez_work_report;
    decomp_add(fn);
}
