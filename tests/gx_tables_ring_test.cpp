// GX's tables ring (host/gx_tables_ring.h) without the game or a GPU.
//
// The command processor's guard: placements of every size for recordings that
// run late - a GPU a whole lap or more behind, recordings the size of the ring
// - and run parts that cannot wait. No placement is handed bytes an earlier
// placement of a recording that has not run holds; a part gets none where the
// command processor would wrap or wait, and nothing moves for it.
//
// The pool's arena: pool threads place tables and tag their bytes, a command
// processor thread splices their commits into recordings (late ones refused,
// as kListMaxAge has it; some splices held up between their first look and
// the stamp, as a descheduled command processor would be) and a GPU thread
// runs the recordings in order, checking every spliced commit's tag is still
// in its bytes when its recording runs - no take handed them out before.
#include "host/gx_tables_ring.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace {

std::atomic<int> g_failures{0};
void check(bool ok, const char* what, unsigned long long a, unsigned long long b) {
    if (ok) return;
    if (++g_failures <= 12) std::printf("FAIL: %s (%llu, %llu)\n", what, a, b);
}

void guard_test() {
    using gx_ring::Guard;
    std::mt19937_64 rng(77);
    struct Placed {
        std::uint64_t at, bytes, serial;
    };
    for (int pass = 0; pass < 2; ++pass) {
        Guard g;
        if (pass == 1) g.end = gx_ring::kRingBytes - (16ull << 20);  // (the pool's arena carved off)
        std::uint64_t open = 0, completed = 0, waits = 0, parts_stopped = 0, placed = 0;
        std::deque<Placed> live;                    // placements of recordings not run yet, in order
        std::multimap<std::uint64_t, Placed> by_at;  // the same by where they are
        constexpr std::uint64_t kLongest = 16ull << 20;
        const auto ran = [&](std::uint64_t s) { return completed > s; };
        bool in_part = false;
        const auto wait = [&](std::uint64_t s) {
            check(!in_part, "a run part's take never waits", s, open);
            check(s < open, "a wait for a recording not submitted yet", s, open);
            completed = std::max(completed, s + 1);
            ++waits;
        };
        for (int step = 0; step < 240000; ++step) {
            const int phase = (step / 20000) % 4;  // small tables; chunks; recordings of tens of MiB; a GPU far behind
            const std::uint64_t bytes = phase == 2 && rng() % 8 == 0 ? (1 + rng() % 16) << 20
                                        : phase == 1                ? 64 << 10
                                                                    : 16 * (1 + rng() % 1024);
            const std::uint64_t align = std::uint64_t{8} << (rng() % 6);
            const bool part = rng() % 5 == 0;
            Guard before = g;
            in_part = part;
            std::uint64_t at = g.take(bytes, align, open, part, ran, wait);
            in_part = false;
            if (part && at != Guard::kNone) check(g.lap == before.lap && g.waits == before.waits, "a run part's take never wraps", step, g.lap);
            if (part && at == Guard::kNone) {
                ++parts_stopped;
                check(std::memcmp(&before, &g, sizeof(g)) == 0, "a part's refusal moved nothing", step, 0);
                // ...and the command processor would have wrapped or waited there.
                Guard copy = before;
                std::uint64_t w = 0;
                copy.take(bytes, align, open, false, ran, [&](std::uint64_t) { ++w; });
                check(w > 0 || copy.lap != before.lap || bytes > before.end, "a part refused only at a wrap or a wait", step, bytes);
                at = g.take(bytes, align, open, false, ran, wait);  // (committed in order: the command processor's own take)
            }
            check(at != Guard::kNone && at + bytes <= g.end, "a placement inside the part", at, bytes);
            if (at == Guard::kNone) continue;
            ++placed;
            for (auto it = by_at.lower_bound(at + bytes); it != by_at.begin();) {
                --it;
                const Placed& p = it->second;
                if (p.at + kLongest <= at) break;
                if (p.serial < completed || p.serial == open) continue;  // (run, or the same recording's: never overlapping)
                check(!(p.at < at + bytes && at < p.at + p.bytes), "bytes a recording not run yet holds handed out again", at, p.serial);
            }
            live.push_back({at, bytes, open});
            by_at.emplace(at, live.back());
            // A recording a few placements long, submitted; the GPU runs them late - in the last phase not at all
            // until a placement waits for one.
            if (rng() % (phase == 2 ? 4 : 64) == 0) ++open;
            if (phase != 3 && open > 3 && rng() % 8 == 0) completed = std::max(completed, open - 1 - rng() % 3);
            while (!live.empty() && live.front().serial < completed) {
                const auto [lo, hi] = by_at.equal_range(live.front().at);
                for (auto it = lo; it != hi; ++it) {
                    if (it->second.serial == live.front().serial && it->second.bytes == live.front().bytes) {
                        by_at.erase(it);
                        break;
                    }
                }
                live.pop_front();
            }
        }
        check(waits > 0 && parts_stopped > 0 && g.laps > 20, "the test met waits, part refusals and laps", waits, g.laps);
        std::printf("gx_tables_ring_test: guard (end %llu MiB): %llu placements, %llu laps, %llu waits, %llu part refusals\n",
                    static_cast<unsigned long long>(g.end >> 20), static_cast<unsigned long long>(placed),
                    static_cast<unsigned long long>(g.laps), static_cast<unsigned long long>(waits),
                    static_cast<unsigned long long>(parts_stopped));
    }
}

void arena_test() {
    using gx_ring::Arena;
    constexpr std::uint64_t kMaxAge = 4;
    Arena a;
    // The arena's bytes, a tag a word (atomics: the threads' writes and the checks race by design when a guard fails).
    std::vector<std::atomic<std::uint32_t>> mem(Arena::kBytes / 4);
    std::atomic<std::uint64_t> flip{1}, completed{0};
    std::atomic<bool> stop{false};
    struct Commit {
        std::uint32_t seg;
        std::uint64_t lap, at, bytes, flip;
        std::uint32_t tag;
    };
    std::mutex qmu;
    std::deque<Commit> queue;
    std::atomic<std::uint32_t> next_tag{1};
    std::atomic<std::uint64_t> placed{0}, refused{0};
    const auto pool = [&](int seed) {
        std::mt19937_64 r(seed);
        Arena::Cursor c;
        while (!stop.load(std::memory_order_relaxed)) {
            const std::uint64_t bytes = 4 * (1 + r() % 4096);
            const std::uint64_t at = a.place(c, bytes, 16, flip.load(), completed.load(), kMaxAge);
            if (at == Arena::kNone) {
                refused.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::yield();
                continue;
            }
            const std::uint32_t tag = next_tag.fetch_add(1);
            for (std::uint64_t w = at / 4; w < (at + bytes) / 4; ++w) mem[w].store(tag, std::memory_order_relaxed);
            // Its segment is its own while it places there: nobody took it (its lap would have moved).
            check(a.lap_is(c.seg, c.lap), "an owned segment's lap kept while its owner places there", c.seg, c.lap);
            placed.fetch_add(1, std::memory_order_relaxed);
            // (The third thread's commits are never spliced - refused at their splices - so its segment's stamp stays
            // old: only its owning it keeps it from a take while it rests.)
            if (seed % 3 != 2) {
                std::lock_guard<std::mutex> lk(qmu);
                queue.push_back({c.seg, c.lap, at, bytes, flip.load(), tag});
                if (queue.size() > 4096) queue.pop_front();  // (never spliced: dropped)
            }
            // Resting now and then, its segment kept (the pool at a frame cap) - long enough for it to look old.
            if (r() % 256 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };
    const auto intact = [&](const Commit& m) {
        for (std::uint64_t w = m.at / 4; w < (m.at + m.bytes) / 4; ++w) {
            if (mem[w].load(std::memory_order_relaxed) != m.tag) return false;
        }
        return true;
    };
    // The recordings: the open one's serial, each one's spliced commits; the GPU runs them in order, late.
    std::mutex rmu;
    std::uint64_t open = 0;
    std::deque<std::vector<Commit>> recordings(1);
    std::atomic<std::uint64_t> spliced{0}, old{0}, moved{0}, checked{0};
    std::thread cp([&] {
        std::mt19937_64 r(5);
        while (!stop.load(std::memory_order_relaxed)) {
            Commit m;
            {
                std::unique_lock<std::mutex> lk(qmu);
                if (queue.empty()) {
                    lk.unlock();
                    std::this_thread::yield();
                    continue;
                }
                m = queue.front();
                queue.pop_front();
            }
            if (m.flip + kMaxAge < flip.load()) {  // (too old to splice: drawn in order)
                old.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (!a.lap_is(m.seg, m.lap)) {
                moved.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (r() % 8 == 0) std::this_thread::sleep_for(std::chrono::microseconds(150));  // (held up between its looks)
            std::lock_guard<std::mutex> lk(rmu);
            if (!a.splice(m.seg, m.lap, open)) {
                moved.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            check(intact(m), "a spliced commit's tables intact at its splice", m.seg, m.tag);
            recordings.back().push_back(m);
            spliced.fetch_add(1, std::memory_order_relaxed);
            if (recordings.back().size() >= 16) {  // submitted: a new recording opens
                ++open;
                recordings.emplace_back();
            }
        }
    });
    std::thread gpu([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::microseconds(300));  // (late)
            std::lock_guard<std::mutex> lk(rmu);
            if (recordings.size() < 3) continue;
            for (const Commit& m : recordings.front()) {
                check(intact(m), "a spliced commit's tables intact when its recording runs", m.seg, m.tag);
                checked.fetch_add(1, std::memory_order_relaxed);
            }
            recordings.pop_front();
            completed.fetch_add(1);
        }
    });
    std::thread flipper([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            flip.fetch_add(1);
        }
    });
    std::vector<std::thread> pools;
    for (int t = 0; t < 3; ++t) pools.emplace_back(pool, 300 + t);  // (seeds 300, 301, 302: the last one's never spliced)
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    stop.store(true);
    for (std::thread& t : pools) t.join();
    cp.join();
    gpu.join();
    flipper.join();
    check(spliced.load() > 1000 && checked.load() > 1000 && a.takes.load() > 16, "the arena test spliced, ran and took segments again",
          spliced.load(), a.takes.load());
    std::printf("gx_tables_ring_test: arena: %llu placed (%llu refused: no segment free), %llu segments taken, %llu spliced (%llu too old, "
                "%llu taken again first), %llu checked as their recordings ran\n",
                static_cast<unsigned long long>(placed.load()), static_cast<unsigned long long>(refused.load()),
                static_cast<unsigned long long>(a.takes.load()), static_cast<unsigned long long>(spliced.load()),
                static_cast<unsigned long long>(old.load()), static_cast<unsigned long long>(moved.load()),
                static_cast<unsigned long long>(checked.load()));
}

// The arena's rules one at a time: an owned segment is no other thread's; a segment let go is taken again only once old
// enough and once its last splice's recording has run; a take moves the lap, and a splice after it says so.
void arena_rules_test() {
    using gx_ring::Arena;
    Arena a;
    Arena::Cursor c0, c1;
    std::uint64_t at0 = a.place(c0, 1024, 16, 10, 0, 4), at1 = a.place(c1, 1024, 16, 10, 0, 4);
    check(at0 != Arena::kNone && at1 != Arena::kNone && c0.seg != c1.seg, "two threads, two segments", c0.seg, c1.seg);
    // Thread 0 fills its segment and moves on; its old segment is not taken while young or while its splice's
    // recording has not run.
    const std::uint32_t first = c0.seg;
    const std::uint64_t lap0 = c0.lap;
    check(a.splice(first, lap0, 7), "a splice at the commit's lap", first, lap0);
    a.place(c0, Arena::kSegment, 16, 11, 0, 4);  // (a full segment's worth: a new one)
    check(c0.seg != first, "a full segment left for a new one", c0.seg, first);
    for (std::uint32_t k = 0; k < Arena::kSegments * 2; ++k) {
        Arena::Cursor t;
        const std::uint64_t at = a.place(t, Arena::kSegment, 16, 13, 100, 4);  // (young: flip 13 is within 4 + 2 of 11)
        check(at == Arena::kNone || t.seg != first, "a young segment not taken", t.seg, first);
    }
    Arena b;
    Arena::Cursor d0;
    b.place(d0, 64, 16, 10, 0, 4);
    const std::uint32_t s0 = d0.seg;
    check(b.splice(s0, d0.lap, 5), "stamped with recording 5", s0, 5);
    b.place(d0, Arena::kSegment, 16, 10, 0, 4);  // (let go)
    bool taken = false;
    for (int k = 0; k < 64 && !taken; ++k) {
        Arena::Cursor t;
        if (b.place(t, Arena::kSegment, 16, 100, 5, 4) != Arena::kNone) taken |= t.seg == s0;  // (recording 5 not run: completed 5)
    }
    check(!taken, "not taken before its splice's recording has run", s0, 5);
    Arena e;
    Arena::Cursor f0;
    e.place(f0, 64, 16, 10, 0, 4);
    const std::uint32_t s1 = f0.seg;
    const std::uint64_t l1 = f0.lap;
    e.place(f0, Arena::kSegment, 16, 10, 0, 4);  // (let go, never spliced)
    for (int k = 0; k < 64; ++k) {
        Arena::Cursor t;
        e.place(t, Arena::kSegment, 16, 100, 1, 4);
    }
    check(!e.lap_is(s1, l1) && !e.splice(s1, l1, 3), "a splice after a take says the lap moved", s1, l1);
    check(e.place(f0, Arena::kSegment + 1, 16, 100, 1, 4) == Arena::kNone, "longer than a segment: none", 0, 0);
}

}  // namespace

int main() {
    guard_test();
    arena_rules_test();
    arena_test();
    std::printf("gx_tables_ring_test: %d failures\n", g_failures.load());
    return g_failures.load() ? 1 : 0;
}
