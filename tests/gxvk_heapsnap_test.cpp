// GX on Vulkan's stage 1d.1 (host/gxvk_heapsnap.cpp): the choice that lets a
// constant-buffer binding in GX's upload heap take its snapshot with the four
// lookups' answers known, and the guards at the writers that take that back.
//
// The choice is checked against a model of its rule on generated bindings
// around heaps of several sizes and places, and on every edge by hand: the
// heap's first and last byte, a range ending at its end and one byte past it,
// the snapshot's 1 MiB limit and an empty range, and each reason a binding
// takes the lookups, in the order the exit report counts them. The bindings'
// counts must add up, and a binding outside the heap must count nothing.
// The guards must trip on any range that meets the heap - inside it, over
// either edge, a byte in - and on nothing else, nor before the heap is known;
// a trip must hold for the rest of the run, so that every binding of the heap
// takes the lookups after it. With the switch off nothing may trip or count;
// with compare the choice is the same and only the binding differs (render.cpp).
// Then eleven deliberate mistakes, each of which must be caught.
// Built with GXVK_HEAPSNAP_TESTING and GXVK_HEAPSNAP_MUTANTS (CMakeLists.txt).
#include "host/gxvk.h"

#include <cstdint>
#include <cstdio>
#include <random>

int g_heapsnap_mutant = 0;

namespace {

using u64 = std::uint64_t;
using gxvk::HeapSnap;
using gxvk::HeapSnapCase;

int g_failed = 0;
bool g_quiet = false;
void fail(const char* what, long step) {
    ++g_failed;
    if (!g_quiet && g_failed <= 20) std::printf("FAIL: step %ld: %s\n", step, what);
}

// The rule, written again: outside the heap nothing; then, in this order, a
// trip, a store, an empty range or one over the snapshot's 1 MiB, a range
// past the heap's end, CPU-written windows, snapshots off, a check that needs
// the lookups, no ring yet; else the answers are known.
int model(const HeapSnapCase& c) {
    if (!c.lo || c.base < c.lo || c.base >= c.hi) return gxvk::kHeapSnapNotHeap;
    if (c.tripped) return gxvk::kHeapSnapTripped;
    if (c.stored) return gxvk::kHeapSnapStored;
    if (c.bytes == 0 || c.bytes > (u64{1} << 20)) return gxvk::kHeapSnapSize;
    if (c.base + c.bytes > c.hi) return gxvk::kHeapSnapPastEnd;
    if (c.windows) return gxvk::kHeapSnapWindows;
    if (!c.snapshots) return gxvk::kHeapSnapOff;
    if (c.check) return gxvk::kHeapSnapCheck;
    if (!c.ring) return gxvk::kHeapSnapNoRing;
    return gxvk::kHeapSnapFast;
}

void expect_class(const HeapSnapCase& c, int want, long step, const char* what) {
    const int got = gxvk::heap_snap_class(c);
    if (got != want || got != model(c)) fail(what, step);
}

// The edges, by hand, for one heap.
void edges(u64 lo, u64 hi, long& step) {
    HeapSnapCase c;
    c.lo = lo;
    c.hi = hi;
    const auto at = [&](u64 base, u64 bytes) {
        HeapSnapCase k = c;
        k.base = base;
        k.bytes = bytes;
        return k;
    };
    expect_class(at(lo, 256), gxvk::kHeapSnapFast, ++step, "the heap's first byte: the answers known");
    expect_class(at(lo - 16, 256), gxvk::kHeapSnapNotHeap, ++step, "below the heap: not the heap's");
    expect_class(at(lo - 1, 256), gxvk::kHeapSnapNotHeap, ++step, "a byte below the heap: not the heap's");
    expect_class(at(hi - 16, 16), gxvk::kHeapSnapFast, ++step, "a range ending at the heap's end: the answers known");
    expect_class(at(hi - 16, 17), gxvk::kHeapSnapPastEnd, ++step, "a range one byte past the heap's end: the lookups");
    expect_class(at(hi - 1, 1), gxvk::kHeapSnapFast, ++step, "the heap's last byte: the answers known");
    expect_class(at(hi, 16), gxvk::kHeapSnapNotHeap, ++step, "the heap's end: not the heap's");
    expect_class(at(lo, u64{1} << 20), gxvk::kHeapSnapFast, ++step, "exactly 1 MiB: the snapshot copies it");
    expect_class(at(lo, (u64{1} << 20) + 4), gxvk::kHeapSnapSize, ++step, "over 1 MiB: the lookups");
    expect_class(at(lo, 0), gxvk::kHeapSnapSize, ++step, "an empty range: the lookups");
    HeapSnapCase k = at(lo + 4096, 512);
    k.stored = true;
    expect_class(k, gxvk::kHeapSnapStored, ++step, "a range the shader stores to: the lookups");
    k = at(lo + 4096, 512);
    k.windows = true;
    expect_class(k, gxvk::kHeapSnapWindows, ++step, "a draw with CPU-written windows: the lookups");
    k = at(lo + 4096, 512);
    k.snapshots = false;
    expect_class(k, gxvk::kHeapSnapOff, ++step, "snapshots off: the lookups");
    k = at(lo + 4096, 512);
    k.check = true;
    expect_class(k, gxvk::kHeapSnapCheck, ++step, "a check that needs the lookups: the lookups");
    k = at(lo + 4096, 512);
    k.ring = false;
    expect_class(k, gxvk::kHeapSnapNoRing, ++step, "no ring yet: the lookups");
    k = at(lo + 4096, 512);
    k.tripped = true;
    k.stored = true;
    expect_class(k, gxvk::kHeapSnapTripped, ++step, "a trip comes first");
    k = at(lo + 4096, 512);
    k.lo = 0;
    expect_class(k, gxvk::kHeapSnapNotHeap, ++step, "no heap known: not the heap's");
}

// Generated bindings around one heap, against the model.
void generated(u64 lo, u64 hi, std::mt19937_64& rng, long& step) {
    std::uniform_int_distribution<int> coin(0, 9);
    for (int n = 0; n < 20000; ++n) {
        HeapSnapCase c;
        c.lo = lo;
        c.hi = hi;
        const int where = coin(rng);
        const u64 span = hi - lo;
        c.base = where == 0   ? lo - 1 - rng() % 4096
                 : where == 1 ? hi + rng() % 4096
                 : where == 2 ? hi - 1 - rng() % 2048
                 : where == 3 ? lo + rng() % 64
                              : lo + (rng() % span);
        c.base &= ~u64{3};
        const int size = coin(rng);
        c.bytes = size == 0 ? 0 : size == 1 ? (u64{1} << 20) + (rng() % 3) * 4 : size == 2 ? hi - c.base + (rng() % 2) * 4 : 16 + (rng() % 4096) * 4;
        c.stored = coin(rng) == 0;
        c.windows = coin(rng) == 0;
        c.snapshots = coin(rng) != 0;
        c.check = coin(rng) == 0;
        c.ring = coin(rng) != 0;
        c.tripped = coin(rng) == 0;
        if (gxvk::heap_snap_class(c) != model(c)) {
            fail("a generated binding: the choice is not the rule's", ++step);
            return;
        }
    }
}

// The bindings' counts, the guards and the trip, in one mode.
void counted(HeapSnap mode, long& step) {
    const u64 lo = 0x143400040, hi = lo + (160u << 20);
    gxvk::heap_snap_test_reset(mode, lo, hi);
    const bool on = mode != HeapSnap::kOff;
    // Before anything: bindings in, out, and refused.
    const bool in = gxvk::heap_snap_binding(lo + 0x1000, 256, false, false, true, false, true);
    if (in != on) fail("a binding in the heap: the answers known with the switch on, nothing with it off", ++step);
    if (gxvk::heap_snap_binding(lo - 0x1000, 256, false, false, true, false, true)) fail("a binding below the heap took the answers", ++step);
    if (gxvk::heap_snap_binding(hi, 256, false, false, true, false, true)) fail("a binding at the heap's end took the answers", ++step);
    if (gxvk::heap_snap_binding(lo + 0x2000, 256, true, false, true, false, true)) fail("a stored-to binding took the answers", ++step);
    if (gxvk::heap_snap_binding(hi - 16, 32, false, false, true, false, true)) fail("a binding past the heap's end took the answers", ++step);
    auto k = gxvk::heap_snap_test_counts();
    u64 fell = 0;
    for (u64 f : k.fell) fell += f;
    if (k.fast != (on ? 1u : 0u) || fell != (on ? 2u : 0u)) fail("the bindings' counts: one with the answers, two refused, none outside", ++step);
    if (on && (k.fell[gxvk::kHeapSnapStored] != 1 || k.fell[gxvk::kHeapSnapPastEnd] != 1)) fail("each refusal counted by its reason", ++step);
    // Guards that must not trip: below, ending at the start, at the end, after.
    gxvk::heap_written(gxvk::kHeapGuardGpuWrite, lo - 0x2000, 0x1000);
    gxvk::heap_written(gxvk::kHeapGuardGpuWrite, lo - 0x40, 0x40);
    gxvk::heap_written(gxvk::kHeapGuardCopyVersion, hi, 0x40);
    gxvk::heap_written(gxvk::kHeapGuardTwin, hi + 0x1000, 0x1000);
    gxvk::heap_written(gxvk::kHeapGuardUnmap, lo + 0x1000, 0);
    k = gxvk::heap_snap_test_counts();
    if (k.tripped || k.guards[0] + k.guards[1] + k.guards[2] + k.guards[3]) fail("a guard tripped on a range that does not meet the heap", ++step);
    if (on && !gxvk::heap_snap_binding(lo + 0x3000, 64, false, false, true, false, true))
        fail("the answers lost without a trip", ++step);
    if (!on) {
        gxvk::heap_written(gxvk::kHeapGuardGpuWrite, lo + 0x1000, 64);
        k = gxvk::heap_snap_test_counts();
        if (k.tripped || k.guards[0]) fail("the switch off: a guard counted or tripped", ++step);
        return;
    }
    // A write one byte into the heap trips it, and the trip holds.
    gxvk::heap_written(gxvk::kHeapGuardGpuWrite, lo - 0x10, 0x11);
    k = gxvk::heap_snap_test_counts();
    if (!k.tripped || k.guards[gxvk::kHeapGuardGpuWrite] != 1) fail("a GPU write one byte into the heap: tripped, and counted", ++step);
    if (gxvk::heap_snap_binding(lo + 0x4000, 64, false, false, true, false, true)) fail("a binding took the answers after a trip", ++step);
    k = gxvk::heap_snap_test_counts();
    if (k.fell[gxvk::kHeapSnapTripped] != 1) fail("a binding after a trip counted as such", ++step);
    gxvk::heap_written(gxvk::kHeapGuardTwin, hi - 1, 1);
    gxvk::heap_written(gxvk::kHeapGuardCopyVersion, lo - 0x100, 0x2000);
    gxvk::heap_written(gxvk::kHeapGuardUnmap, lo, hi - lo);
    k = gxvk::heap_snap_test_counts();
    if (!k.tripped || k.guards[gxvk::kHeapGuardTwin] != 1 || k.guards[gxvk::kHeapGuardCopyVersion] != 1 || k.guards[gxvk::kHeapGuardUnmap] != 1)
        fail("the heap's last byte, a range over its start and the whole heap: each counted, and the trip held", ++step);
    // Each kind trips by itself.
    for (int what = 0; what < gxvk::kHeapGuards; ++what) {
        gxvk::heap_snap_test_reset(mode, lo, hi);
        gxvk::heap_written(static_cast<gxvk::HeapGuard>(what), lo + 0x8000 + static_cast<u64>(what) * 64, 64);
        k = gxvk::heap_snap_test_counts();
        if (!k.tripped || k.guards[what] != 1) fail("a guard kind did not trip on a range inside the heap", ++step);
        if (gxvk::heap_snap_binding(lo + 0x100, 64, false, false, true, false, true)) fail("the answers taken after a guard of its kind", ++step);
    }
    // Before the heap is known nothing trips, and a binding counts nothing.
    gxvk::heap_snap_test_reset(mode, 0, 0);
    gxvk::heap_written(gxvk::kHeapGuardGpuWrite, lo + 0x1000, 64);
    if (gxvk::heap_snap_binding(lo + 0x1000, 64, false, false, true, false, true)) fail("the answers taken before the heap was known", ++step);
    k = gxvk::heap_snap_test_counts();
    if (k.tripped || k.guards[0] || k.fast) fail("a guard or a binding counted before the heap was known", ++step);
    // The heap made known later: the answers from then on.
    gxvk::heap_snap_region(lo, hi);
    if (!gxvk::heap_snap_binding(lo + 0x1000, 64, false, false, true, false, true)) fail("the answers not taken once the heap was known", ++step);
}

long run_all(std::mt19937_64& rng) {
    long step = 0;
    const u64 heaps[][2] = {{0x143400040, 0x143400040 + (160u << 20)}, {0x10000, 0x10000 + (4u << 20)}, {0xf000000000, 0xf000000000 + (1u << 22)}};
    for (const auto& h : heaps) {
        edges(h[0], h[1], step);
        generated(h[0], h[1], rng, step);
    }
    counted(HeapSnap::kOff, step);
    counted(HeapSnap::kOn, step);
    counted(HeapSnap::kCompare, step);
    return step;
}

}  // namespace

int main() {
    std::mt19937_64 rng(0x1d1);
    const long steps = run_all(rng);
    if (g_failed) {
        std::printf("gxvk_heapsnap_test: %d failures in %ld checks\n", g_failed, steps);
        return 1;
    }
    std::printf("gxvk_heapsnap_test: %ld checks and 3 x 20000 generated bindings passed\n", steps);
    // The mistakes: each must fail some check.
    const char* const kMutants[] = {"",
                                    "a range one byte past the heap's end taken",
                                    "a stored-to range taken",
                                    "a trip ignored by the choice",
                                    "exactly 1 MiB refused",
                                    "a range below the heap taken",
                                    "a guard missing a range that starts before the heap and runs into it",
                                    "a guard missing a range that ends one byte into the heap",
                                    "a guard tripping before the heap is known",
                                    "CPU-written windows ignored",
                                    "no ring ignored",
                                    "a guard that counts but does not trip"};
    int caught = 0;
    const int mutants = static_cast<int>(sizeof kMutants / sizeof kMutants[0]) - 1;
    g_quiet = true;
    for (int m = 1; m <= mutants; ++m) {
        g_heapsnap_mutant = m;
        g_failed = 0;
        std::mt19937_64 r(0x1d1);
        run_all(r);
        if (g_failed) {
            ++caught;
            std::printf("  mutant %d (%s): caught (%d checks failed)\n", m, kMutants[m], g_failed);
        } else {
            std::printf("  mutant %d (%s): NOT caught\n", m, kMutants[m]);
        }
    }
    g_heapsnap_mutant = 0;
    std::printf("gxvk_heapsnap_test: %d of %d mutants caught\n", caught, mutants);
    return caught == mutants ? 0 : 1;
}
