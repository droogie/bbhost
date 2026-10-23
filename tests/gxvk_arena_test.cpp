// GX on Vulkan's arena (host/gxvk_arena.cpp): the release proof against a
// model of GX's upload heap.
//
// The model is the block pool as GX runs it: ranges are handed out first-fit
// from a free list, the command processor binds parts of the ranges out in
// submissions numbered as they are recorded, a range goes back with a
// release that lands once a later submission has completed, and only landed
// ranges are handed out again - unless the run puts one back early on
// purpose, which is the reuse the proof exists to catch. Each hand-out is
// told to the arena as the pool's node, and each binding as the command
// processor binds it. The arena must find the first early reuse (a range
// handed out again over a binding whose submission has not completed) at the
// hand-out that makes it and nothing before, in hand-outs that cover a
// released range whole, in part, or several at once - and find nothing when
// the counts raised after the labels land trail the one raised before (the
// game reads a release label as soon as it is in memory) - and then stop: no
// records kept, no hand-out recorded, every binding of the heap left to
// today's path. It must refuse in-place binding to a range that runs past its
// hand-out; and with every binding sampled (compare) it must find bytes
// changed under a queued draw, and only those - a range handed out again is
// rewritten as it should be. The same proof holds copy-backs' sources (stage
// 1d.2a, arena_hold) with the arena's bindings off: a hold stamps its record
// as a binding does, one past its hand-out or with no hand-out over it is
// refused, and a hand-out over a held source whose copy has not run waits for
// the GPU (the recording under way submitted and everything run) instead of
// failing the proof - each such hand-out once, the sixteenth within a minute
// stopping the proof.
// Then thirteen deliberate mistakes, each of which must be caught.
// Built with GXVK_ARENA_TESTING and GXVK_ARENA_MUTANTS (CMakeLists.txt), as its arena is.
#include "host/gxvk.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>

int g_arena_mutant = 0;

// What the arena asks of the rest of bbhost.
std::uint64_t g_completed = 0;  // submissions completed: those numbered below it
// How far the counts raised once a submission's labels are in memory, or once
// its slot retires, trail the landing's count, raised before the labels land.
std::uint64_t g_lag = 0;
std::uint64_t trailing() { return g_completed > g_lag ? g_completed - g_lag : 0; }
// The submission being recorded; host_gpu_flush submits it, and every submission runs.
std::uint64_t g_serial = 0;
long g_flush_calls = 0;
void host_gpu_flush() {
    ++g_flush_calls;
    g_completed = ++g_serial;
}
std::uint64_t host_gpu_completed_submits_seen() { return trailing(); }
namespace label_landing {
std::uint64_t landing() { return g_completed; }
std::uint64_t landed() { return trailing(); }
}  // namespace label_landing
// Copy-backs from the heap (stage 1d.2a): its mode, which keeps the records for the holds with the bindings off.
gxvk::HeapCopy g_copy_mode = gxvk::HeapCopy::kOff;
namespace gxvk {
bool census() { return false; }
HeapCopy heap_copy_mode() { return g_copy_mode; }
void heap_snap_region(std::uint64_t, std::uint64_t) {}  // stage 1d.1's (tests/gxvk_heapsnap_test.cpp)
}  // namespace gxvk

namespace {

using u64 = std::uint64_t;
using gxvk::Arena;

int g_failed = 0;
bool g_quiet = false;
void fail(const char* what, long step) {
    ++g_failed;
    if (!g_quiet && g_failed <= 20) std::printf("FAIL: step %ld: %s\n", step, what);
}

// The heap: real bytes, so the arena can hash what a binding reads.
constexpr u64 kHeapBytes = 4u << 20;
alignas(64) std::uint8_t g_heap[kHeapBytes + 64];
const u64 kPool = 0x7a7a0000;

// A hand-out as the pool returns it: its node, whose +8 is its start, and
// +0x18 the node after it in address order, whose +8 is its end.
struct Node {
    alignas(8) std::uint8_t b[0x30] = {};
};
std::vector<Node*> g_nodes;  // kept until the end, as the pool's nodes are
u64 node_for(u64 start, u64 end) {
    Node* a = new Node;
    Node* z = new Node;
    g_nodes.push_back(a);
    g_nodes.push_back(z);
    const u64 za = reinterpret_cast<u64>(z);
    std::memcpy(a->b + 8, &start, 8);
    std::memcpy(a->b + 0x18, &za, 8);
    std::memcpy(z->b + 8, &end, 8);
    return reinterpret_cast<u64>(a);
}

struct Out {
    u64 start, end;
    u64 last_bound = 0;  // the newest submission + 1 that bound part of it; 0: none
    bool released = false;
    u64 lands_at = 0;  // released: lands once this many submissions have completed
};

struct Result {
    long reuses_early = 0;  // the model's own count: hand-outs' overlaps of a range a running submission bound
    long handouts = 0, bindings = 0, in_place = 0, straddles = 0, bytes_changed = 0;
    long first_model = -1, first_arena = -1;  // the steps of the model's first early reuse and of the arena's first find
    long handouts_early = 0;                  // hand-outs over at least one range a running submission bound
    bool failed = false;                      // the arena's proof has failed (in place off)
    long in_place_after = 0;                  // bindings bound in place after that
};

// One run. `early`: the chance a hand-out takes a released range before its
// release has landed (and so, perhaps, over a binding still running). `holds`:
// the draws' reads are copy-backs' sources held (arena_hold, stage 1d.2a), with
// the arena's bindings off - the records kept for the holds alone.
Result run(Arena mode, long steps, double early, double rewrite, std::uint64_t seed, bool holds = false) {
    g_copy_mode = holds ? gxvk::HeapCopy::kOn : gxvk::HeapCopy::kOff;
    gxvk::arena_test_reset(mode);
    g_completed = 0;
    std::mt19937_64 rng(seed);
    const auto chance = [&](double p) { return std::uniform_real_distribution<double>(0, 1)(rng) < p; };
    const auto range = [&](u64 lo, u64 hi) { return std::uniform_int_distribution<u64>(lo, hi)(rng); };
    const u64 base = (reinterpret_cast<u64>(g_heap) + 15) & ~u64{15};
    gxvk::arena_region(kPool, base, kHeapBytes);
    std::map<u64, Out> out;  // by start: every hand-out the pool has not taken back
    u64& serial = g_serial;  // the submission being recorded (host_gpu_flush moves it on)
    serial = 0;
    g_flush_calls = 0;
    Result r;
    const auto free_at = [&](u64 start, u64 end, bool take_early) {
        // Whether [start, end) is free to hand out: nothing out overlaps it but
        // released ranges (landed - or, taking early, not yet).
        for (auto it = out.begin(); it != out.end(); ++it) {
            const Out& o = it->second;
            if (o.end <= start || o.start >= end) continue;
            if (!o.released) return false;
            if (o.lands_at > g_completed && !take_early) return false;
        }
        return true;
    };
    for (long step = 0; step < steps; ++step) {
        const double what = std::uniform_real_distribution<double>(0, 1)(rng);
        if (what < 0.25) {
            // A hand-out, 16-byte aligned, of 16 B - 192 KiB, somewhere free.
            const u64 bytes = range(1, 12288) * 16;
            const bool take_early = chance(early);
            u64 start = 0;
            bool found = false;
            for (int tries = 0; tries < 24 && !found; ++tries) {
                start = base + range(0, (kHeapBytes - bytes) / 16) * 16;
                found = free_at(start, start + bytes, take_early);
            }
            if (!found) continue;
            const u64 end = start + bytes;
            // Each range it goes over that a running submission bound: the model's count.
            bool early_here = false;
            for (auto it = out.begin(); it != out.end();) {
                Out& o = it->second;
                if (o.end <= start || o.start >= end) {
                    ++it;
                    continue;
                }
                if (o.last_bound > g_completed) {
                    ++r.reuses_early;
                    early_here = true;
                    if (r.first_model < 0) r.first_model = step;
                }
                it = out.erase(it);
            }
            if (early_here) ++r.handouts_early;
            out[start] = {start, end};
            gxvk::arena_handout(kPool, node_for(start, end));
            ++r.handouts;
            if (r.first_arena < 0 && gxvk::arena_test_counts().violations_order) r.first_arena = step;
            // The new owner writes its bytes.
            std::memset(reinterpret_cast<void*>(start), static_cast<int>(range(0, 255)), static_cast<std::size_t>(bytes));
        } else if (what < 0.75) {
            // A binding of part of a range out and not released, in this submission;
            // now and then one running past its range's end.
            if (out.empty()) continue;
            auto it = out.begin();
            std::advance(it, static_cast<long>(range(0, out.size() - 1)));
            Out& o = it->second;
            if (o.released) continue;
            const u64 len = o.end - o.start;
            const u64 at = o.start + range(0, (len - 1) / 16) * 16;
            u64 bytes = range(1, std::min<u64>(o.end - at, 4096));
            const bool past = chance(0.03);
            if (past) bytes = o.end - at + 16;
            if (at + bytes > base + kHeapBytes) continue;
            if (!r.failed) r.failed = !gxvk::arena_test_counts().in_place;
            const bool placed = holds ? gxvk::arena_hold(at, bytes, serial) : gxvk::arena_bind(at, bytes, serial, gxvk::kUseConstant, false);
            ++r.bindings;
            if (r.failed && placed) ++r.in_place_after;
            if (past) {
                ++r.straddles;
                if (placed) fail(holds ? "a copy-back's source past its hand-out's end held" : "a binding past its hand-out's end bound in place", step);
            } else {
                o.last_bound = std::max(o.last_bound, serial + 1);
                if (placed) ++r.in_place;
            }
            // A CPU write into bytes a queued draw reads: the bytes check's to find (compare samples every binding).
            if (!past && chance(rewrite)) {
                std::memset(reinterpret_cast<void*>(at), 0x5c ^ static_cast<int>(step & 0xff), 1);
                ++r.bytes_changed;
            }
        } else if (what < 0.85) {
            // The submission is handed over; the next is recorded.
            ++serial;
        } else if (what < 0.95) {
            // Submissions complete, in order, never past the one being recorded.
            if (g_completed < serial) g_completed += range(1, serial - g_completed);
        } else {
            // A release: lands once a submission after the last binding has completed.
            if (out.empty()) continue;
            auto it = out.begin();
            std::advance(it, static_cast<long>(range(0, out.size() - 1)));
            Out& o = it->second;
            if (o.released) continue;
            o.released = true;
            o.lands_at = std::max(o.last_bound, serial + 1);
        }
    }
    // Everything completes; the samples still waiting are checked at the report.
    g_completed = serial + 1;
    return r;
}

// The clean runs and the early-reuse runs, as one check. Returns the failures.
int check_all(bool say) {
    const int before = g_failed;
    // Clean: no early reuse, no rewrite - the arena must find nothing.
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        const Result r = run(Arena::kInPlace, 60000, 0.0, 0.0, seed);
        const auto c = gxvk::arena_test_counts();
        if (r.reuses_early) fail("the model reused early with early 0", 0);
        if (c.violations_order || c.violations_bytes) fail("a violation found in a clean run", 0);
        if (!c.in_place) fail("in place stopped in a clean run", 0);
        if (r.in_place == 0) fail("nothing bound in place in a clean run", 0);
        if (say)
            std::printf("  clean run %llu: %ld hand-outs, %ld bindings (%ld in place, %ld past their range refused), no violation found\n",
                        static_cast<unsigned long long>(seed), r.handouts, r.bindings, r.in_place, r.straddles);
    }
    // Clean, with the counts raised after the labels land trailing by up to three submissions: the
    // game reuses a range as soon as its release label is in memory, so nothing is early.
    for (std::uint64_t seed = 4; seed <= 5; ++seed) {
        g_lag = 3;
        const Result r = run(Arena::kInPlace, 60000, 0.0, 0.0, seed);
        g_lag = 0;
        const auto c = gxvk::arena_test_counts();
        if (c.violations_order || c.violations_bytes) fail("a violation found where only the counts after the labels trailed", 0);
        if (!c.in_place) fail("in place stopped where only the counts after the labels trailed", 0);
        if (say)
            std::printf("  clean run %llu, the counts after the labels three submissions behind: %ld hand-outs, %ld bindings (%ld in place), "
                        "no violation found\n",
                        static_cast<unsigned long long>(seed), r.handouts, r.bindings, r.in_place);
    }
    // Early reuse: the first found at the hand-out that made it, and nothing before; then the arena stops.
    long made = 0, stopped = 0;
    for (std::uint64_t seed = 11; seed <= 16; ++seed) {
        const Result r = run(Arena::kInPlace, 40000, 0.3, 0.0, seed);
        const auto c = gxvk::arena_test_counts();
        made += r.reuses_early;
        if (r.first_arena != r.first_model) fail("the first early reuse not found at the hand-out that made it", r.first_model);
        if (c.violations_order > 1) fail("early reuses checked after the proof had failed", 0);
        if (c.violations_bytes) fail("bytes found changed where nothing rewrote them", 0);
        if (!r.reuses_early) continue;
        if (c.in_place) fail("in place still on after an early reuse", 0);
        if (c.live) fail("records kept after the proof failed", 0);
        if (!c.unrecorded) fail("hand-outs recorded after the proof failed", 0);
        if (!c.proof_off) fail("no binding of the heap left to today's path after the proof failed", 0);
        if (r.in_place_after) fail("a binding bound in place after the proof failed", 0);
        ++stopped;
    }
    if (say)
        std::printf("  early reuse: the model handed out over %ld ranges a running submission bound; the arena found each run's first as it was "
                    "made and stopped there (%ld runs)\n",
                    made, stopped);
    // Bytes rewritten under queued draws, every binding sampled (compare).
    long rewrites = 0, changed = 0;
    for (std::uint64_t seed = 21; seed <= 24; ++seed) {
        const Result r = run(Arena::kCompare, 30000, 0.0, 0.002, seed);
        gxvk::arena_report();  // checks the samples still waiting
        const auto c = gxvk::arena_test_counts();
        rewrites += r.bytes_changed;
        changed += static_cast<long>(c.violations_bytes);
        if (r.bytes_changed && !c.violations_bytes) fail("bytes rewritten under a queued draw not found", 0);
        if (c.violations_order) fail("an early reuse found where the model made none (compare)", 0);
        if (r.in_place) fail("compare bound in place", 0);
        if (c.violations_bytes && c.live) fail("records kept after changed bytes failed the proof (compare)", 0);
    }
    // No rewrite, compare: a range handed out again is rewritten by its new owner, which is no violation.
    for (std::uint64_t seed = 31; seed <= 33; ++seed) {
        run(Arena::kCompare, 30000, 0.0, 0.0, seed);
        gxvk::arena_report();
        const auto c = gxvk::arena_test_counts();
        if (c.violations_bytes) fail("bytes found changed where only new owners wrote", 0);
        if (c.samples_checked == 0) fail("no sample checked", 0);
    }
    if (say) std::printf("  bytes: %ld rewrites under queued draws made, %ld found; none found where ranges were only handed out again\n", rewrites,
                         changed);
    // Copy-backs' sources held (stage 1d.2a), with the arena's bindings off: the records are kept for the holds, a hold
    // stamps its record as a binding does, so an early reuse over a held source is found at the hand-out that made it;
    // a hold past its hand-out is refused, and after the proof failed none is held.
    long held = 0, hold_made = 0;
    for (std::uint64_t seed = 51; seed <= 53; ++seed) {
        const Result r = run(Arena::kOff, 40000, 0.0, 0.0, seed, true);
        const auto c = gxvk::arena_test_counts();
        held += static_cast<long>(c.holds);
        if (c.violations_order || c.violations_bytes) fail("a violation found in a clean run of holds", 0);
        if (!c.holds || r.in_place != static_cast<long>(c.holds)) fail("holds not counted as held in a clean run", 0);
        if (c.hold_fell != static_cast<std::uint64_t>(r.straddles)) fail("holds refused other than those past their hand-out", 0);
    }
    // Early reuse over held sources: each such hand-out waits once - the recording under way submitted, everything
    // run - so no violation is found; the sixteenth wait stops the proof (none held after it, no records kept). A
    // short run with few early hand-outs waits for each and goes on.
    long waited = 0;
    for (std::uint64_t seed = 61; seed <= 64; ++seed) {
        const bool few = seed >= 63;
        const Result r = run(Arena::kOff, 40000, few ? 0.3 : 1.0, 0.0, seed, true);
        if (say) std::printf("    holds run %llu: %ld early hand-outs, %llu waits\n", static_cast<unsigned long long>(seed), r.handouts_early,
                             static_cast<unsigned long long>(gxvk::arena_test_counts().hold_waits));
        const auto c = gxvk::arena_test_counts();
        hold_made += r.reuses_early;
        waited += static_cast<long>(c.hold_waits);
        if (c.violations_order) fail("an early reuse over a held source failed the proof instead of waiting", r.first_model);
        if (static_cast<long>(c.hold_waits) != g_flush_calls) fail("a hand-out's wait without the GPU's catching up", 0);
        const long want = std::min<long>(r.handouts_early, 16);
        if (static_cast<long>(c.hold_waits) != want) fail("hand-outs over held sources not waiting once each, up to sixteen", r.first_model);
        if (r.handouts_early >= 16) {
            if (c.in_place) fail("the proof not stopped after sixteen waits", 0);
            if (r.in_place_after) fail("a copy-back's source held after the proof stopped", 0);
            if (c.live) fail("records kept after the proof stopped (holds)", 0);
        } else if (!c.in_place) {
            fail("the proof stopped with fewer than sixteen waits", 0);
        }
        if (!r.handouts_early) fail("a run made no early hand-out over a held source (the test's own)", 0);
    }
    // The limit: one range handed out, a source held in it in the recording under way, the range handed out again
    // before that has run - twenty times, well within a minute. The first sixteen wait and find nothing; the sixteenth
    // stops the proof, and after it no hand-out is recorded and no source held.
    {
        g_copy_mode = gxvk::HeapCopy::kOn;
        gxvk::arena_test_reset(Arena::kOff);
        g_completed = g_serial = 0;
        g_flush_calls = 0;
        const u64 base = (reinterpret_cast<u64>(g_heap) + 15) & ~u64{15};
        gxvk::arena_region(kPool, base, kHeapBytes);
        long held_after = 0;
        for (int i = 0; i < 20; ++i) {
            const u64 s = base + 65536 * static_cast<u64>(i % 4);
            gxvk::arena_handout(kPool, node_for(s, s + 4096));
            const bool on = gxvk::arena_test_counts().in_place;
            if (gxvk::arena_hold(s + 64, 64, g_serial) && !on) ++held_after;
            gxvk::arena_handout(kPool, node_for(s, s + 4096));
        }
        const auto c = gxvk::arena_test_counts();
        if (c.violations_order) fail("a hand-out over a held source failed the proof instead of waiting (the limit's run)", 0);
        if (c.hold_waits != 16) fail("the waits not stopped at sixteen", 0);
        if (c.in_place || c.live || held_after) fail("the proof not stopped at the sixteenth wait, or a source held after it", 0);
        g_copy_mode = gxvk::HeapCopy::kOff;
    }
    {
        g_copy_mode = gxvk::HeapCopy::kOn;
        gxvk::arena_test_reset(Arena::kOff);
        const u64 base = (reinterpret_cast<u64>(g_heap) + 15) & ~u64{15};
        gxvk::arena_region(kPool, base, kHeapBytes);
        gxvk::arena_handout(kPool, node_for(base + 4096, base + 8192));
        const bool none = gxvk::arena_hold(base + 64, 32, 0);
        const bool inside = gxvk::arena_hold(base + 4096 + 64, 64, 0);
        const auto c = gxvk::arena_test_counts();
        if (none || !inside || c.holds != 1 || c.hold_fell != 1) fail("a source no hand-out holds held, or one inside a hand-out not", 0);
        g_copy_mode = gxvk::HeapCopy::kOff;
    }
    if (say) std::printf("  copy-backs' sources: %ld held over three clean runs, none refused but past their hand-out; the model handed out "
                         "over %ld held sources early, and those hand-outs waited for the GPU (%ld waits, sixteen at most a run) with no "
                         "violation\n", held, hold_made, waited);
    // Bindings outside the heap are not the arena's: nothing bound, nothing counted (an address above the heap once
    // wrapped the bounds test round and counted as a fallback); one running past the heap's end is refused.
    {
        run(Arena::kInPlace, 2000, 0.0, 0.0, 41);
        const u64 base = (reinterpret_cast<u64>(g_heap) + 15) & ~u64{15};
        const auto c0 = gxvk::arena_test_counts();
        const bool above = gxvk::arena_bind(base + kHeapBytes + 4096, 64, 1, gxvk::kUseConstant, false);
        const bool below = gxvk::arena_bind(base - 4096, 64, 1, gxvk::kUseConstant, false);
        const auto c1 = gxvk::arena_test_counts();
        if (above || below || c1.fell != c0.fell) fail("a binding outside the heap bound or counted", 0);
        const bool past = gxvk::arena_bind(base + kHeapBytes - 32, 64, 1, gxvk::kUseConstant, false);
        const auto c2 = gxvk::arena_test_counts();
        if (past || c2.fell != c1.fell + 1) fail("a binding over the heap's end not refused and counted", 0);
        if (say) std::printf("  bindings outside the heap: not the arena's; one over its end refused\n");
    }
    return g_failed - before;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (check_all(true)) {
        std::printf("gxvk_arena_test: %d failures\n", g_failed);
        return 1;
    }
    static const char* const kMistake[] = {"", "a binding's stamp one submission short", "a hand-out over part of a record not checked",
                                           "a binding past its hand-out bound in place", "a sample checked after its range was handed out again",
                                           "the heap's bounds test wrapping round above it", "the arena going on with its records after the proof failed",
                                           "the landing's count taken once the labels are in memory, not before",
                                           "a copy-back's source past its hand-out held", "a copy-back's source held without its stamp",
                                           "a copy-back's source no hand-out holds held", "a hand-out over a held source not waiting",
                                           "the count of done submissions not read again after the wait", "the waits not limited"};
    int missed = 0;
    for (int m = 1; m <= 13; ++m) {
        g_arena_mutant = m;
        g_quiet = true;
        const int bad = check_all(false);
        g_quiet = false;
        g_arena_mutant = 0;
        g_failed -= bad;
        if (bad) {
            std::printf("  mistake %d (%s) caught: %d checks failed\n", m, kMistake[m], bad);
        } else {
            std::printf("FAIL: mistake %d (%s) not caught\n", m, kMistake[m]);
            ++missed;
        }
    }
    for (Node* n : g_nodes) delete n;
    if (missed) {
        std::printf("gxvk_arena_test: %d mistakes not caught\n", missed);
        return 1;
    }
    std::printf("gxvk_arena_test: ok\n");
    return 0;
}
