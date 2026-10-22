// The copy watch (host/gxvk_watch.h) against a plain list of its ranges:
// random adds and removes up to every slot taken, random destinations - most
// of them meeting a watched range or touching one's ends, the rest anywhere in
// a 64 GiB space, so granules 1 GiB apart share their counters - and ranges
// wider than the table. Its answers must be the list's: the published ranges
// meet a destination exactly when a range does, the granules never say no
// where a range meets it, a full watch takes no more, and once every range is
// gone no granule holds a count. Then five deliberate mistakes, each of which
// must be caught: the granules asked only at the destination's start, a range
// counted one granule short, a removal that leaves its counts, a range ending
// where the destination starts taken as meeting it, and the search's range
// asked for its own end instead of the highest end so far. Last, readers ask
// while a writer adds and removes ranges, as GX's threads do while the table
// changes: a range watched throughout must always be found, and an address no
// range ever covers never.
#define GXVK_WATCH_MUTANTS
#include "host/gxvk_watch.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

int g_watch_mutant = 0;

namespace {

struct Range {
    std::uint64_t lo, hi;
    int slot;
};

struct Result {
    std::uint64_t queries = 0, wrong_hit = 0, missed = 0, false_near = 0, far_queries = 0, wrong_full = 0, left_counts = 0;
    std::uint64_t far_plain = 0, far_plain_near = 0;  // destinations anywhere while no range wider than the table is watched
};

constexpr std::uint64_t GiB = 1ull << 30, MiB = 1ull << 20;

Result run(unsigned seed) {
    using Watch = gxvk::CopyWatch<>;  // as the table has it
    std::mt19937_64 rng(seed);
    auto rand = [&](std::uint64_t n) { return n ? rng() % n : 0; };
    Result r;
    for (int round = 0; round < 40; ++round) {
        auto* w = new Watch;
        std::vector<Range> live;
        const auto meets = [&](std::uint64_t va, std::uint64_t bytes) {
            for (const Range& x : live)
                if (va < x.hi && va + bytes > x.lo) return true;
            return false;
        };
        for (int step = 0; step < 4000; ++step) {
            const std::uint64_t op = rand(100);
            if (op < 30) {
                // A buffer's range: 16 bytes to 8 MiB, now and then wider than the table's 1 GiB.
                const std::uint64_t lo = 16 + (rand(64 * GiB) & ~15ull);
                const std::uint64_t bytes = rand(50) == 0 ? GiB + rand(2 * GiB) : 16 + (rand(rand(2) ? 64 * 1024 : 8 * MiB) & ~15ull);
                const int slot = w->add(lo, lo + bytes);
                if ((slot < 0) != (live.size() == 64)) ++r.wrong_full;
                if (slot >= 0) live.push_back({lo, lo + bytes, slot});
            } else if (op < 50 && !live.empty()) {
                const std::size_t k = rand(live.size());
                w->remove(live[k].slot);
                live.erase(live.begin() + static_cast<std::ptrdiff_t>(k));
            } else {
                std::uint64_t va, bytes = 4 + (rand(rand(4) ? 4096 : 4 * MiB) & ~3ull);
                if (!live.empty() && rand(3)) {
                    // Near a watched range: inside it, across either end, or just outside.
                    const Range& x = live[rand(live.size())];
                    switch (rand(4)) {
                        case 0: va = x.lo + rand(x.hi - x.lo); break;
                        case 1: va = x.lo > bytes ? x.lo - bytes + rand(2 * bytes) : x.lo; break;
                        case 2: va = x.hi - rand(std::min<std::uint64_t>(bytes, x.hi - x.lo)); break;
                        default: va = rand(2) ? x.hi : (x.lo > bytes ? x.lo - bytes : x.lo); break;
                    }
                } else {
                    va = rand(64 * GiB);
                    ++r.far_queries;
                    bool wide = false;
                    for (const Range& x : live) wide = wide || x.hi - x.lo >= GiB;
                    if (!wide) {
                        ++r.far_plain;
                        if (w->near(va, bytes) && !meets(va, bytes)) ++r.far_plain_near;
                    }
                }
                ++r.queries;
                const bool m = meets(va, bytes), hit = w->hit(va, bytes), near = w->near(va, bytes);
                if (hit != m) ++r.wrong_hit;
                if (m && !near) ++r.missed;
                if (w->written(va, bytes) != m) ++r.wrong_hit;
                if (near && !m) ++r.false_near;
            }
        }
        for (const Range& x : live) w->remove(x.slot);
        live.clear();
        for (int q = 0; q < 2000; ++q) {
            const std::uint64_t va = rand(64 * GiB), bytes = 4 + rand(4 * MiB);
            if (w->near(va, bytes) || w->written(va, bytes)) ++r.left_counts;
        }
        if (!w->empty()) ++r.left_counts;
        delete w;
    }
    return r;
}

bool clean(const Result& r) { return !r.wrong_hit && !r.missed && !r.wrong_full && !r.left_counts; }

}  // namespace

int main() {
    int failed = 0;
    const Result r = run(1);
    std::printf("copy watch: %llu destinations asked (%llu anywhere): a slot's answer wrong %llu, granules missing a range %llu, "
                "a full watch wrong %llu, counts left %llu; granules near without a range %llu (%.2f%% of all); of %llu destinations "
                "anywhere while no range wider than the table was watched, %.2f%% near without a range\n",
                static_cast<unsigned long long>(r.queries), static_cast<unsigned long long>(r.far_queries),
                static_cast<unsigned long long>(r.wrong_hit), static_cast<unsigned long long>(r.missed),
                static_cast<unsigned long long>(r.wrong_full), static_cast<unsigned long long>(r.left_counts),
                static_cast<unsigned long long>(r.false_near), r.queries ? 100.0 * static_cast<double>(r.false_near) / static_cast<double>(r.queries) : 0.0,
                static_cast<unsigned long long>(r.far_plain), r.far_plain ? 100.0 * static_cast<double>(r.far_plain_near) / static_cast<double>(r.far_plain) : 0.0);
    if (!clean(r)) {
        std::printf("FAIL: the watch disagrees with its ranges\n");
        ++failed;
    }
    static const char* const kMistakes[5] = {"the granules asked only at the destination's start", "a range counted one granule short",
                                             "a removal that leaves its counts", "a range ending at the destination's start taken as meeting it",
                                             "the search's range asked for its own end, not the highest end so far"};
    for (int m = 1; m <= 5; ++m) {
        g_watch_mutant = m;
        const Result x = run(2);
        g_watch_mutant = 0;
        if (clean(x)) {
            std::printf("FAIL: mistake %d (%s) not caught\n", m, kMistakes[m - 1]);
            ++failed;
        } else {
            std::printf("mistake %d (%s) caught: slot answers wrong %llu, granules missing %llu, counts left %llu\n", m, kMistakes[m - 1],
                        static_cast<unsigned long long>(x.wrong_hit), static_cast<unsigned long long>(x.missed),
                        static_cast<unsigned long long>(x.left_counts));
        }
    }
    // Readers against a writer: one range watched throughout, others added and
    // removed around it as fast as the writer can, readers asking all the while.
    {
        static gxvk::CopyWatch<> w;
        std::mutex mu;
        constexpr std::uint64_t kHeld = 0x200000000ull, kHeldEnd = kHeld + 0x60;
        {
            std::lock_guard<std::mutex> lk(mu);
            w.add(kHeld, kHeldEnd);
        }
        std::atomic<bool> stop{false};
        std::atomic<std::uint64_t> asked{0}, lost{0}, phantom{0};
        std::thread writer([&] {
            std::mt19937_64 rng(7);
            std::vector<int> slots;
            for (int i = 0; i < 400000; ++i) {
                std::lock_guard<std::mutex> lk(mu);
                if (slots.size() < 40 && (rng() & 1)) {
                    const std::uint64_t lo = 0x100000000ull + (rng() % 0x80000000ull & ~15ull);
                    const std::uint64_t a = lo < kHeld && lo + 0x10000 > kHeld ? lo - 0x20000 : lo;
                    const int slot = w.add(a, a + 16 + (rng() % 0x8000 & ~15ull));
                    if (slot >= 0) slots.push_back(slot);
                } else if (!slots.empty()) {
                    const std::size_t k = rng() % slots.size();
                    w.remove(slots[k]);
                    slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(k));
                }
            }
            stop = true;
        });
        std::vector<std::thread> readers;
        for (int t = 0; t < 3; ++t) {
            readers.emplace_back([&] {
                std::uint64_t n = 0, l = 0, ph = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    l += !w.hit(kHeld + 0x20, 8) || !w.written(kHeld + 0x20, 8);
                    ph += w.hit(0x900000000ull, 64);  // no range ever starts above 0x180000000 + 0x8000
                    ++n;
                }
                asked += n;
                lost += l;
                phantom += ph;
            });
        }
        writer.join();
        for (std::thread& t : readers) t.join();
        std::printf("readers against a writer: %llu asked, the held range lost %llu times, an unwatched address found %llu times\n",
                    static_cast<unsigned long long>(asked.load()), static_cast<unsigned long long>(lost.load()),
                    static_cast<unsigned long long>(phantom.load()));
        if (lost || phantom || !asked) {
            std::printf("FAIL: a reader saw the ranges mid-change\n");
            ++failed;
        }
    }
    std::printf(failed ? "gxvk_watch_test: %d failed\n" : "gxvk_watch_test: ok\n", failed);
    return failed ? 1 : 0;
}
