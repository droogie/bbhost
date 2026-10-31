// The page table's view of the window imports (host/import_table.h): a
// window whose entries name an older import keeps them until the queue sees
// the import that replaced it; a window whose entries named nothing points at
// the new one at once; an import goes when neither view names it.
#include "host/import_table.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

struct Imp {
    std::uint64_t made_serial = 0;
    int id = 0;
};

constexpr std::uint32_t kN = 64;

struct Chunk {
    Imp* window[kN] = {};
    Imp* table[kN] = {};
    std::vector<std::pair<std::uint32_t, Imp*>> writes;  // (window, import) written
    std::vector<Imp*> retired;
    // Windows where an older import covers less of the window than its
    // replacement (a span that grew), and the queue's progress a wait gives.
    std::set<std::uint32_t> short_cover;
    std::uint64_t after_wait = 0;
    int waits = 0;

    import_table::Done install(std::uint32_t a, std::uint32_t b, Imp* c, std::uint64_t completed, bool defer = true) {
        return import_table::install(
            window, table, kN, a, b, c, completed, defer, [&](std::uint32_t w) { writes.push_back({w, c}); },
            [&](std::uint32_t w, Imp*) { return !short_cover.count(w); },
            [&] {
                ++waits;
                return after_wait ? after_wait : c->made_serial;
            },
            [&](Imp* o) { retired.push_back(o); });
    }
    import_table::Done publish(std::uint64_t completed) {
        return import_table::publish(
            window, table, kN, completed, [&](std::uint32_t w, Imp* c) { writes.push_back({w, c}); }, [&](Imp* o) { retired.push_back(o); });
    }
};

void test_seen() {
    Imp c{10, 1};
    CHECK(!import_table::seen(c, 9));  // submission 9 (made before it) has not run
    CHECK(import_table::seen(c, 10));  // submissions 0-9 have run
    CHECK(import_table::seen(c, 11));
    // The sink, or the import itself: written whatever runs.
    CHECK(import_table::on_cover<Imp>(nullptr, c, 0, true, true) == import_table::Entry::kWrite);
    CHECK(import_table::on_cover(&c, c, 0, true, true) == import_table::Entry::kWrite);
    Imp old{2, 2};
    CHECK(import_table::on_cover(&old, c, 9, true, true) == import_table::Entry::kKeep);
    CHECK(import_table::on_cover(&old, c, 9, true, false) == import_table::Entry::kWait);  // it covers less: wait, then name c
    CHECK(import_table::on_cover(&old, c, 10, true, true) == import_table::Entry::kWrite);
    CHECK(import_table::on_cover(&old, c, 10, true, false) == import_table::Entry::kWrite);
    CHECK(import_table::on_cover(&old, c, 9, false, true) == import_table::Entry::kWrite);  // the switch off: at once, as before
}

void test_fresh_in_flight() {
    // Windows nothing had imported: written at once even with the queue busy.
    Chunk k;
    Imp n{5, 1};
    const import_table::Done d = k.install(3, 4, &n, 0);
    CHECK(d.written == 2 && d.fresh == 2 && d.kept == 0 && d.retired == 0);
    CHECK(k.window[3] == &n && k.window[4] == &n && k.table[3] == &n && k.table[4] == &n);
    CHECK(k.writes.size() == 2 && k.retired.empty() && k.waits == 0);
    CHECK(import_table::lagging(k.window, k.table, kN) == 0);
}

void test_merge_waits_for_the_queue() {
    Chunk k;
    Imp o{1, 1}, n{7, 2};
    k.install(4, 4, &o, 1);  // the queue idle: o's entries at once
    k.writes.clear();
    // n replaces o over windows 3-5 while submissions 1-6 (made before n) may run.
    const import_table::Done d = k.install(3, 5, &n, 4);
    CHECK(d.written == 2 && d.fresh == 2 && d.kept == 1 && d.retired == 0);
    CHECK(k.window[4] == &n && k.table[4] == &o);  // bindings take n, walks still o
    CHECK(k.table[3] == &n && k.table[5] == &n);   // new pages: at once
    CHECK(k.waits == 0);
    CHECK(k.retired.empty());  // o's pages are still the walks'
    CHECK(import_table::named(k.window, k.table, kN, &o));
    CHECK(import_table::lagging(k.window, k.table, kN) == 1);
    // Not yet: submission 6 has not run.
    k.writes.clear();
    import_table::Done p = k.publish(6);
    CHECK(p.written == 0 && p.retired == 0 && k.table[4] == &o);
    // Submissions 0-6 have run: the entries move to n and o goes, once.
    p = k.publish(7);
    CHECK(p.written == 1 && p.retired == 1);
    CHECK(k.table[4] == &n && k.writes.size() == 1 && k.writes[0].first == 4 && k.writes[0].second == &n);
    CHECK(k.retired.size() == 1 && k.retired[0] == &o);
    CHECK(!import_table::named(k.window, k.table, kN, &o));
    CHECK(import_table::lagging(k.window, k.table, kN) == 0);
    // Again: nothing more.
    p = k.publish(100);
    CHECK(p.written == 0 && p.retired == 0 && k.retired.size() == 1);
}

void test_merge_idle_queue() {
    Chunk k;
    Imp o{1, 1}, n{3, 2};
    k.install(2, 2, &o, 1);
    const import_table::Done d = k.install(2, 3, &n, 3);  // every submission before n has run
    CHECK(d.written == 2 && d.kept == 0 && d.retired == 1);
    CHECK(k.table[2] == &n && k.retired.size() == 1 && k.retired[0] == &o);
}

void test_switch_off() {
    Chunk k;
    Imp o{1, 1}, n{9, 2};
    k.install(2, 2, &o, 1);
    const import_table::Done d = k.install(2, 3, &n, 2, false);  // BBHOST_IMPORT_TABLE_DEFER=0: as before
    CHECK(d.written == 2 && d.kept == 0 && d.retired == 1 && k.table[2] == &n);
}

void test_chain_before_seen() {
    // o -> n1 (window 10 kept on o, window 11 new: n1 at once) -> n2 before
    // the queue saw either: both windows keep what their entries name (o, n1)
    // until the queue sees n2, then both go.
    Chunk k;
    Imp o{1, 1}, n1{5, 2}, n2{8, 3};
    k.install(10, 10, &o, 1);
    k.install(10, 11, &n1, 2);
    CHECK(k.table[10] == &o && k.table[11] == &n1 && k.retired.empty());
    const import_table::Done d = k.install(9, 11, &n2, 3);
    // Window 11's entries named n1 (fresh at its install): kept until n2 is seen too.
    CHECK(d.kept == 2 && k.table[10] == &o && k.table[11] == &n1 && k.table[9] == &n2);
    CHECK(k.retired.empty());  // n1 is still window 11's walks'
    import_table::Done p = k.publish(7);
    CHECK(p.written == 0);
    p = k.publish(8);
    CHECK(p.written == 2 && p.retired == 2);
    CHECK(k.table[10] == &n2 && k.table[11] == &n2);
    std::set<Imp*> r(k.retired.begin(), k.retired.end());
    CHECK(r.size() == 2 && r.count(&o) && r.count(&n1));
}

void test_two_older_imports() {
    // A wider range over two imports: both kept, both retired once at publish.
    Chunk k;
    Imp a{1, 1}, b{1, 2}, n{4, 3};
    k.install(20, 21, &a, 1);
    k.install(22, 23, &b, 1);
    const import_table::Done d = k.install(20, 23, &n, 2);
    CHECK(d.kept == 4 && d.written == 0 && d.retired == 0);
    const import_table::Done p = k.publish(4);
    CHECK(p.written == 4 && p.retired == 2 && k.retired.size() == 2);
}

void test_short_cover() {
    // The older import covers less of window 7 than its replacement (a span
    // that grew): its new pages are wanted now, so the caller waits for the
    // queue to see n, and window 7 names n at once; window 8, which o covers
    // whole, does not wait again (the queue sees n by then).
    Chunk k;
    Imp o{1, 1}, n{6, 2};
    k.install(7, 8, &o, 1);
    k.writes.clear();
    k.short_cover.insert(7);
    const import_table::Done d = k.install(7, 8, &n, 2);
    CHECK(d.waited && k.waits == 1);
    CHECK(d.written == 2 && d.kept == 0 && d.retired == 1);
    CHECK(k.table[7] == &n && k.table[8] == &n && k.retired.size() == 1 && k.retired[0] == &o);
    CHECK(import_table::lagging(k.window, k.table, kN) == 0);
    // A window whose older import covers it whole does not wait.
    Chunk j;
    Imp p{1, 1}, q{6, 2};
    j.install(3, 3, &p, 1);
    const import_table::Done e = j.install(3, 3, &q, 2);
    CHECK(!e.waited && j.waits == 0 && e.kept == 1);
}

void test_bounds() {
    Chunk k;
    Imp n{1, 1};
    CHECK(k.install(5, 4, &n, 0).written == 0);   // an empty range
    CHECK(k.install(0, kN, &n, 0).written == 0);  // past the windows
    CHECK(k.window[0] == nullptr);
}

// Random installs, publishes and queue progress against the rule's
// invariants: entries that named an import only move to one the queue sees
// (or one made over nothing), nothing retired is still named, an import is
// retired once, and once the queue has run everything the two views agree.
void test_random() {
    std::mt19937 rng(20261031);
    for (int round = 0; round < 200; ++round) {
        Chunk k;
        std::vector<Imp*> all;
        std::set<Imp*> retired;
        std::uint64_t flushes = 0, completed = 0;
        int next_id = 1;
        for (int step = 0; step < 200; ++step) {
            const int op = static_cast<int>(rng() % 4);
            if (op == 0) {
                ++flushes;  // a submission
            } else if (op == 1 && completed < flushes) {
                completed += 1 + rng() % (flushes - completed);  // some of them run
            } else if (op == 2) {
                const std::uint32_t a = rng() % kN, len = 1 + rng() % 4;
                const std::uint32_t b = std::min(kN - 1, a + len - 1);
                // A merge widens over every import in [a, b], as dmem_import does.
                std::uint32_t lo = a, hi = b;
                for (bool grew = true; grew;) {
                    grew = false;
                    for (std::uint32_t w = lo; w <= hi; ++w) {
                        Imp* o = k.window[w];
                        if (!o) continue;
                        for (std::uint32_t v = 0; v < kN; ++v) {
                            if (k.window[v] != o) continue;
                            if (v < lo) lo = v, grew = true;
                            if (v > hi) hi = v, grew = true;
                        }
                    }
                }
                Imp* c = new Imp{flushes, next_id++};
                all.push_back(c);
                Imp* before[kN];
                for (std::uint32_t w = 0; w < kN; ++w) before[w] = k.table[w];
                const std::size_t r0 = k.retired.size();
                // Now and then an older import covers less of a window than c.
                k.short_cover.clear();
                if (rng() % 8 == 0) k.short_cover.insert(lo + rng() % (hi - lo + 1));
                const int waits = k.waits;
                k.install(lo, hi, c, completed);
                if (k.waits != waits) completed = std::max(completed, c->made_serial);  // the wait ran the queue up to c
                for (std::uint32_t w = 0; w < kN; ++w) {
                    if (k.table[w] == before[w]) continue;
                    CHECK(k.table[w] == c);
                    CHECK(before[w] == nullptr || import_table::seen(*c, completed));
                }
                for (std::size_t i = r0; i < k.retired.size(); ++i) {
                    CHECK(!retired.count(k.retired[i]));
                    retired.insert(k.retired[i]);
                }
            } else {
                Imp* before[kN];
                for (std::uint32_t w = 0; w < kN; ++w) before[w] = k.table[w];
                const std::size_t r0 = k.retired.size();
                k.publish(completed);
                for (std::uint32_t w = 0; w < kN; ++w) {
                    if (k.table[w] == before[w]) continue;
                    CHECK(k.table[w] == k.window[w] && import_table::seen(*k.table[w], completed));
                }
                for (std::size_t i = r0; i < k.retired.size(); ++i) {
                    CHECK(!retired.count(k.retired[i]));
                    retired.insert(k.retired[i]);
                }
            }
            for (Imp* r : retired) CHECK(!import_table::named(k.window, k.table, kN, r));
            for (std::uint32_t w = 0; w < kN; ++w) CHECK(!k.table[w] || k.window[w]);
        }
        // The queue runs everything: the views agree, and every import but the
        // ones still named has gone.
        completed = flushes + 1;
        k.publish(completed);
        CHECK(import_table::lagging(k.window, k.table, kN) == 0);
        for (std::uint32_t w = 0; w < kN; ++w) CHECK(k.table[w] == k.window[w]);
        for (Imp* c : all) {
            const bool is_named = import_table::named(k.window, k.table, kN, c);
            const bool gone = std::find(k.retired.begin(), k.retired.end(), c) != k.retired.end();
            CHECK(is_named != gone);
            delete c;
        }
    }
}

}  // namespace

int main() {
    test_seen();
    test_fresh_in_flight();
    test_merge_waits_for_the_queue();
    test_merge_idle_queue();
    test_switch_off();
    test_chain_before_seen();
    test_two_older_imports();
    test_short_cover();
    test_bounds();
    test_random();
    if (g_fail) {
        std::printf("import_table_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("import_table_test: ok\n");
    return 0;
}
