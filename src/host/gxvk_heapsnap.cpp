// GX on Vulkan, stage 1d.1: GX's upload heap's constant buffers snapshotted
// with no lookups (host/gxvk.h).
//
// A constant buffer the game writes on the CPU is copied as its draw records
// (render.cpp resolve_stage_buffers), after four lookups of its range: a copy
// version over it, a twin holding it, a GPU write still to land on it, its
// mapping. Over GX's upload heap - the renamed copies of every constant
// buffer GX updates, its Map discards and its ring - the answers are known,
// and a binding there takes the snapshot with them: no copy version (the heap
// is a copy-back's source, never its destination), no twin (no GX buffer's
// own memory lies in it), no GPU write (the GPU writes nothing there), mapped
// (the device's region, for the session). The copy, its bytes, its ring and
// its sharing within a recording are today's.
//
// The answers hold while nothing writes the heap but the CPU and nothing
// unmaps it, which is watched at the writers (heap_written): a GPU write
// noted over the heap, a copy version made into it, a twin made over it, its
// memory unmapped. In the command processor's order each of those comes
// before any binding that could see it, so the first one turns the known
// answers off for the rest of the run before it matters: said once, counted,
// and every binding of the heap takes the lookups again.
//
// BBHOST_GXVK_HEAP_SNAP unset or 1 (the default) takes the answers as known;
// =compare keeps today's path and checks each binding's lookups against them;
// =0 is today's path, and nothing here runs.
#include "host/gxvk.h"

#include "log.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// tests/gxvk_heapsnap_test.cpp builds this file with mutants: small wrong
// turns it switches on one at a time and must catch.
#ifdef GXVK_HEAPSNAP_MUTANTS
extern int g_heapsnap_mutant;
#define MUTANT(n) (g_heapsnap_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace gxvk {
namespace {

using ull = unsigned long long;

#ifdef GXVK_HEAPSNAP_TESTING
HeapSnap g_mode = HeapSnap::kOff;  // tests/gxvk_heapsnap_test.cpp sets it (heap_snap_test_reset)
#else
const HeapSnap g_mode = [] {
    const char* e = std::getenv("BBHOST_GXVK_HEAP_SNAP");
    if (!e || !*e) return HeapSnap::kOn;  // the default
    if (std::strcmp(e, "compare") == 0) return HeapSnap::kCompare;
    return e[0] == '0' ? HeapSnap::kOff : HeapSnap::kOn;
}();
#endif

// The snapshot's own limit (resolve_stage_buffers copies up to 1 MiB).
constexpr std::uint64_t kMaxBytes = 1u << 20;

std::atomic<std::uint64_t> g_lo{0}, g_hi{0};  // the heap's bounds: g_lo published last
std::atomic<bool> g_claimed{false};            // the first heap only, as the arena's
std::atomic<bool> g_tripped{false};
std::atomic<int> g_said{0};

// Counted by the command processor under the renderer's lock: added without a
// locked instruction (gpu_internal.h's bump), which on the draw path waits
// for every earlier store to drain.
void bump(std::atomic<std::uint64_t>& c, std::uint64_t n = 1) { c.store(c.load(std::memory_order_relaxed) + n, std::memory_order_relaxed); }
std::atomic<std::uint64_t> g_fast{0}, g_fast_bytes{0}, g_fell[kHeapSnapWhys] = {};
// From any thread (the kernel's unmaps), and rare: locked adds.
std::atomic<std::uint64_t> g_guards[kHeapGuards] = {}, g_diffs[kHeapDiffs] = {};
std::atomic<int> g_diffs_said{0};

const char* const kWhyName[kHeapSnapWhys] = {"the shader stores to it", "empty or over 1 MiB", "it runs past the heap's end",
                                             "its draw has CPU-written windows", "snapshots off", "a check needs them",
                                             "the copies' ring not made yet", "a guard tripped"};
const char* const kGuardName[kHeapGuards] = {"GPU writes into the heap", "copy versions into it", "twins over it", "its memory unmapped"};
const char* const kGuardOne[kHeapGuards] = {"a GPU write into the heap", "a copy version into the heap", "a twin over the heap",
                                            "the heap's memory unmapped"};
const char* const kDiffName[kHeapDiffs] = {"a copy version", "a twin", "a GPU write still to land", "unmapped"};

}  // namespace

HeapSnap heap_snap_mode() { return g_mode; }

int heap_snap_class(const HeapSnapCase& c) {
    if (!c.lo || (c.base < c.lo && !MUTANT(5)) || c.base >= c.hi) return kHeapSnapNotHeap;
    if (c.tripped && !MUTANT(3)) return kHeapSnapTripped;
    if (c.stored && !MUTANT(2)) return kHeapSnapStored;
    if (!c.bytes || (MUTANT(4) ? c.bytes >= kMaxBytes : c.bytes > kMaxBytes)) return kHeapSnapSize;
    if (MUTANT(1) ? c.bytes > c.hi - c.base + 1 : c.bytes > c.hi - c.base) return kHeapSnapPastEnd;
    if (c.windows && !MUTANT(9)) return kHeapSnapWindows;
    if (!c.snapshots) return kHeapSnapOff;
    if (c.check) return kHeapSnapCheck;
    if (!c.ring && !MUTANT(10)) return kHeapSnapNoRing;
    return kHeapSnapFast;
}

void heap_snap_region(std::uint64_t lo, std::uint64_t hi) {
    if (g_mode == HeapSnap::kOff || !lo || hi <= lo || g_claimed.exchange(true, std::memory_order_relaxed)) return;
    g_hi.store(hi, std::memory_order_relaxed);
    g_lo.store(lo, std::memory_order_release);  // a reader that sees it sees g_hi too
    host_log("gxvk heap snapshots: GX's upload heap is 0x%llx, %llu MiB: its constant buffers %s (BBHOST_GXVK_HEAP_SNAP=0: today's path)",
             static_cast<ull>(lo), static_cast<ull>((hi - lo) >> 20),
             g_mode == HeapSnap::kCompare ? "bound as today, their lookups checked against the answers known for the heap"
                                          : "snapshotted with the lookups' answers known");
}

bool heap_snap_binding(std::uint64_t base, std::uint64_t bytes, bool stored, bool windows, bool snapshots, bool check, bool ring) {
    HeapSnapCase c;
    c.lo = g_lo.load(std::memory_order_acquire);
    if (!c.lo) return false;
    c.hi = g_hi.load(std::memory_order_relaxed);
    if (base < c.lo || base >= c.hi) return false;  // not the heap's: nothing to count
    c.base = base;
    c.bytes = bytes;
    c.stored = stored;
    c.windows = windows;
    c.snapshots = snapshots;
    c.check = check;
    c.ring = ring;
    c.tripped = g_tripped.load(std::memory_order_acquire);
    const int why = heap_snap_class(c);
    if (why == kHeapSnapNotHeap) return false;
    if (why != kHeapSnapFast) {
        bump(g_fell[why]);
        return false;
    }
    bump(g_fast);
    bump(g_fast_bytes, bytes);
    return true;
}

void heap_written(HeapGuard what, std::uint64_t va, std::uint64_t bytes) {
    if (g_mode == HeapSnap::kOff || !bytes) return;
    const std::uint64_t lo = g_lo.load(std::memory_order_acquire);
    if (lo) {
        const std::uint64_t hi = g_hi.load(std::memory_order_relaxed), end = va + bytes;
        const bool meets = MUTANT(6) ? va >= lo && va < hi : MUTANT(7) ? va < hi && end > lo + 1 : va < hi && end > lo;
        if (!meets) return;
    } else if (!MUTANT(8)) {
        return;  // no heap yet: nothing it could hold
    }
    g_guards[what].fetch_add(1, std::memory_order_relaxed);
    if (MUTANT(11)) return;  // (the trip forgotten)
    if (!g_tripped.exchange(true, std::memory_order_acq_rel) && g_said.fetch_add(1, std::memory_order_relaxed) < 1)
        host_log("gxvk heap snapshots: %s, 0x%llx, %llu bytes - the heap's constant buffers take the lookups again from here on",
                 kGuardOne[what], static_cast<ull>(va), static_cast<ull>(bytes));
}

void heap_snap_disagree(HeapSnapDiff what, std::uint64_t va, std::uint64_t bytes) {
    g_diffs[what].fetch_add(1, std::memory_order_relaxed);
    if (g_diffs_said.fetch_add(1, std::memory_order_relaxed) < 8)
        host_log("gxvk heap snapshots: compare: the lookups found %s over 0x%llx, %llu bytes, where the heap's answer is none",
                 kDiffName[what], static_cast<ull>(va), static_cast<ull>(bytes));
}

#ifdef GXVK_HEAPSNAP_TESTING
void heap_snap_test_reset(HeapSnap mode, std::uint64_t lo, std::uint64_t hi) {
    g_mode = mode;
    g_lo.store(0);
    g_hi.store(0);
    g_claimed.store(false);
    g_tripped.store(false);
    g_said.store(0);
    g_fast.store(0);
    g_fast_bytes.store(0);
    for (auto& a : g_fell) a.store(0);
    for (auto& a : g_guards) a.store(0);
    for (auto& a : g_diffs) a.store(0);
    g_diffs_said.store(0);
    if (lo) heap_snap_region(lo, hi);
}
HeapSnapTestCounts heap_snap_test_counts() {
    HeapSnapTestCounts c;
    c.fast = g_fast.load();
    for (int w = 0; w < kHeapSnapWhys; ++w) c.fell[w] = g_fell[w].load();
    for (int w = 0; w < kHeapGuards; ++w) c.guards[w] = g_guards[w].load();
    for (int w = 0; w < kHeapDiffs; ++w) c.diffs[w] = g_diffs[w].load();
    c.tripped = g_tripped.load();
    return c;
}
#endif

void heap_snap_report() {
    if (g_mode == HeapSnap::kOff) return;
    const auto v = [](const std::atomic<std::uint64_t>& a) { return static_cast<ull>(a.load(std::memory_order_relaxed)); };
    if (!g_lo.load()) {
        host_log("gxvk heap snapshots: GX's upload heap was never made known: every constant buffer took today's path");
        return;
    }
    char fell[640] = "";
    std::size_t at = 0;
    for (int w = 0; w < kHeapSnapWhys && at < sizeof fell; ++w)
        at += static_cast<std::size_t>(std::snprintf(fell + at, sizeof fell - at, "%s%s %llu", w ? ", " : "", kWhyName[w], v(g_fell[w])));
    const bool cmp = g_mode == HeapSnap::kCompare;
    host_log("gxvk heap snapshots: %s: %llu (%llu MiB); the lookups because %s",
             cmp ? "compared, bound as today: the heap's constant-buffer bindings that would take the answers known"
                 : "the heap's constant-buffer bindings snapshotted with the lookups' answers known",
             v(g_fast), v(g_fast_bytes) >> 20, fell);
    char guards[400] = "";
    at = 0;
    for (int w = 0; w < kHeapGuards && at < sizeof guards; ++w)
        at += static_cast<std::size_t>(std::snprintf(guards + at, sizeof guards - at, "%s%s %llu", w ? ", " : "", kGuardName[w], v(g_guards[w])));
    host_log("gxvk heap snapshots: the guards: %s%s", guards,
             g_tripped.load() ? " - tripped: the lookups from the first on" : "");
    if (cmp) {
        char diffs[300] = "";
        at = 0;
        std::uint64_t n = 0;
        for (int w = 0; w < kHeapDiffs && at < sizeof diffs; ++w) {
            n += g_diffs[w].load();
            at += static_cast<std::size_t>(std::snprintf(diffs + at, sizeof diffs - at, "%s%s %llu", w ? ", " : "", kDiffName[w], v(g_diffs[w])));
        }
        host_log("gxvk heap snapshots: compare: the lookups disagreed with the known answers %llu times: %s", static_cast<ull>(n), diffs);
    }
}

}  // namespace gxvk
