// Copies left unmade (host/copy_elide.h): random sequences of copies, sampled
// reads, other reads, partial and whole writes and retires over a handful of
// images, run twice - once with every copy made when it is asked for (the
// model), once through the ledger, whose copies are made only when it says
// so and whose sampled reads bind what it redirects them to - and every read
// of every sequence must see the same contents in both. Then the ledger's
// own cases one by one (what it drops, what it makes, the chain and the
// limit), and four mutants of the renderer's side of the rule - a source
// write that skips the ledger, a partial write of a destination taken for a
// whole one, a readback that skips it, a sampled read bound without asking -
// each of which the random runs must catch. The random runs go again with
// the ledger's history on (worth_deferring: some copies made where asked).
#include "host/copy_elide.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int g_failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

using copy_elide::Ledger;
using copy_elide::Pending;

enum Op { kCopy, kSample, kReadOther, kWrite, kOverwrite, kRetire, kOps };
struct Event {
    Op op;
    int a, b;  // images (a: the copy's source / the image touched, b: the copy's destination)
};

constexpr int kImages = 5;
// Image i's base: spaced like render targets, never 0 (0 is "no redirect").
std::uint64_t base_of(int i) { return 0x150000000ull + static_cast<std::uint64_t>(i) * 0x1000000ull; }
int image_of(std::uint64_t base) { return static_cast<int>((base - 0x150000000ull) / 0x1000000ull); }

std::vector<Event> random_events(std::mt19937& rng, int n) {
    std::vector<Event> out;
    std::uniform_int_distribution<int> img(0, kImages - 1), op(0, 99);
    for (int k = 0; k < n; ++k) {
        const int r = op(rng);
        Event e{};
        e.a = img(rng);
        e.b = img(rng);
        // The game's mix: copies and sampled reads most, then writes.
        e.op = r < 22 ? kCopy : r < 60 ? kSample : r < 66 ? kReadOther : r < 84 ? kWrite : r < 94 ? kOverwrite : kRetire;
        if (e.op == kCopy && e.a == e.b) e.b = (e.a + 1) % kImages;
        out.push_back(e);
    }
    return out;
}

// A write in part: what the image holds after it depends on what it held.
std::uint64_t mix(std::uint64_t v, std::uint64_t t) { return (v ^ (t * 0x9e3779b97f4a7c15ull)) * 0xff51afd7ed558ccdull; }

// What a run saw: one entry a read (sampled or other).
std::vector<std::uint64_t> model(const std::vector<Event>& ev) {
    std::uint64_t v[kImages];
    for (int i = 0; i < kImages; ++i) v[i] = 1000 + i;
    std::vector<std::uint64_t> seen;
    std::uint64_t t = 0;
    for (const Event& e : ev) {
        ++t;
        switch (e.op) {
            case kCopy: v[e.b] = v[e.a]; break;
            case kSample:
            case kReadOther: seen.push_back(v[e.a]); break;
            case kWrite: v[e.a] = mix(v[e.a], t); break;
            case kOverwrite:
            case kRetire: v[e.a] = 1000000 + t; break;
            default: break;
        }
    }
    return seen;
}

// The renderer's side, with its mutants.
enum Mutant { kNone, kSkipSrcWrite, kPartialAsWhole, kSkipReadOther, kSkipRedirect };

struct Stats {
    std::uint64_t made = 0, dropped = 0, redirected = 0, deferred = 0, asked = 0;
};

// `history`: the renderer's phase 1c - a copy the ledger says is not worth leaving unmade (worth_deferring) is made
// where it is asked, as record_target_copy_locked makes it: the source read as it is, the destination overwritten whole.
std::vector<std::uint64_t> elided(const std::vector<Event>& ev, Mutant m, Stats* st = nullptr, std::size_t most = 16, bool history = false) {
    std::uint64_t phys[kImages];
    for (int i = 0; i < kImages; ++i) phys[i] = 1000 + i;
    Ledger ledger(most);
    std::vector<std::uint64_t> seen;
    Stats s;
    const auto make = [&](const Pending& p, copy_elide::End) {
        phys[image_of(p.dst)] = phys[image_of(p.src)];
        ++s.made;
    };
    const auto drop = [&](const Pending&) { ++s.dropped; };
    std::uint64_t t = 0;
    for (const Event& e : ev) {
        ++t;
        const std::uint64_t a = base_of(e.a);
        switch (e.op) {
            case kCopy:
                if (history && !ledger.worth_deferring(a, base_of(e.b))) {
                    ledger.touch(a, copy_elide::kReadOther, make, drop);
                    ledger.touch(base_of(e.b), copy_elide::kOverwrite, make, drop);
                    phys[e.b] = phys[e.a];
                    ++s.asked;
                    break;
                }
                ledger.defer(a, base_of(e.b), 0, make, drop);
                ++s.deferred;
                break;
            case kSample: {
                const std::uint64_t r = m == kSkipRedirect ? 0 : ledger.redirect(a);
                if (r) ++s.redirected;
                seen.push_back(phys[image_of(r ? r : a)]);
                break;
            }
            case kReadOther:
                if (m != kSkipReadOther) ledger.touch(a, copy_elide::kReadOther, make, drop);
                seen.push_back(phys[e.a]);
                break;
            case kWrite:
                if (m == kPartialAsWhole) {
                    ledger.touch(a, copy_elide::kOverwrite, make, drop);
                } else if (!(m == kSkipSrcWrite && ledger.is_src(a))) {
                    ledger.touch(a, copy_elide::kWrite, make, drop);
                }
                phys[e.a] = mix(phys[e.a], t);
                break;
            case kOverwrite:
                ledger.touch(a, copy_elide::kOverwrite, make, drop);
                phys[e.a] = 1000000 + t;
                break;
            case kRetire:
                ledger.touch(a, copy_elide::kRetire, make, drop);
                phys[e.a] = 1000000 + t;
                break;
            default: break;
        }
        // The invariant itself: a pending copy's source holds what its destination would.
        for (const Pending& p : ledger.pending()) {
            CHECK(!ledger.is_dst(p.src));  // no chains
        }
    }
    if (st) *st = s;
    return seen;
}

void random_runs() {
    std::mt19937 rng(20261030);
    Stats total;
    int caught[5] = {};
    const int runs = 4000;
    for (int r = 0; r < runs; ++r) {
        const std::vector<Event> ev = random_events(rng, 60 + r % 80);
        const std::vector<std::uint64_t> want = model(ev);
        Stats s;
        const std::vector<std::uint64_t> got = elided(ev, kNone, &s, 1 + r % 6);
        CHECK(got == want);
        Stats sh;
        CHECK(elided(ev, kNone, &sh, 1 + r % 6, true) == want);
        total.asked += sh.asked;
        total.made += s.made;
        total.dropped += s.dropped;
        total.redirected += s.redirected;
        total.deferred += s.deferred;
        for (int m = kSkipSrcWrite; m <= kSkipRedirect; ++m) {
            if (elided(ev, static_cast<Mutant>(m)) != want) ++caught[m];
        }
    }
    // Every copy is either made or dropped or still pending at the end.
    CHECK(total.made + total.dropped <= total.deferred);
    CHECK(total.dropped > 0 && total.redirected > 0 && total.made > 0 && total.asked > 0);
    std::printf("random: %d runs, %llu copies left unmade, %llu made later, %llu never made, %llu reads redirected; with the history, "
                "%llu made where asked\n",
                runs, static_cast<unsigned long long>(total.deferred), static_cast<unsigned long long>(total.made),
                static_cast<unsigned long long>(total.dropped), static_cast<unsigned long long>(total.redirected),
                static_cast<unsigned long long>(total.asked));
    const char* const names[5] = {"", "a source write that skips the ledger", "a partial write taken for a whole one",
                                  "a readback that skips the ledger", "a sampled read bound without asking"};
    for (int m = kSkipSrcWrite; m <= kSkipRedirect; ++m) {
        std::printf("mutant %d (%s): caught in %d of %d runs\n", m, names[m], caught[m], runs);
        CHECK(caught[m] > 0);
    }
}

void cases() {
    std::uint64_t made = 0, dropped = 0;
    std::vector<int> ends;
    const auto make = [&](const Pending&, copy_elide::End e) {
        ++made;
        ends.push_back(e);
    };
    const auto drop = [&](const Pending&) { ++dropped; };
    const std::uint64_t S = base_of(0), D = base_of(1), E = base_of(2);
    {
        // Read while the source is unchanged, then the destination copied into again: never made.
        Ledger l;
        l.defer(S, D, 0, make, drop);
        CHECK(l.redirect(D) == S);
        CHECK(l.redirect(S) == 0);
        l.defer(S, D, 0, make, drop);
        CHECK(made == 0 && dropped == 1);
        l.touch(D, copy_elide::kOverwrite, make, drop);
        CHECK(made == 0 && dropped == 2 && !l.live());
    }
    {
        // The source drawn into: made first; read the source another way: nothing.
        made = dropped = 0;
        ends.clear();
        Ledger l;
        l.defer(S, D, 0, make, drop);
        l.touch(S, copy_elide::kReadOther, make, drop);
        CHECK(made == 0 && l.live());
        l.touch(S, copy_elide::kWrite, make, drop);
        CHECK(made == 1 && !l.live() && ends.back() == copy_elide::kEndSrcWrite);
    }
    {
        // Two copies of one source: both made at its write, in order.
        made = dropped = 0;
        ends.clear();
        Ledger l;
        l.defer(S, D, 0, make, drop);
        l.defer(S, E, 0, make, drop);
        CHECK(l.redirect(D) == S && l.redirect(E) == S);
        l.touch(S, copy_elide::kOverwrite, make, drop);
        CHECK(made == 2 && !l.live());
    }
    {
        // A chain: a copy out of a pending destination makes that one first.
        made = dropped = 0;
        ends.clear();
        Ledger l;
        l.defer(S, D, 0, make, drop);
        l.defer(D, E, 0, make, drop);
        CHECK(made == 1 && ends.back() == copy_elide::kEndSrcPending);
        CHECK(l.redirect(E) == D && !l.is_dst(D));
    }
    {
        // A destination written in part is made first; retired, forgotten.
        made = dropped = 0;
        ends.clear();
        Ledger l;
        l.defer(S, D, 0, make, drop);
        l.touch(D, copy_elide::kWrite, make, drop);
        CHECK(made == 1 && ends.back() == copy_elide::kEndDstWrite);
        l.defer(S, D, 0, make, drop);
        l.touch(D, copy_elide::kRetire, make, drop);
        CHECK(made == 1 && dropped == 1);
        l.defer(S, D, 0, make, drop);
        l.touch(D, copy_elide::kReadOther, make, drop);
        CHECK(made == 2 && ends.back() == copy_elide::kEndDstRead);
        l.defer(S, D, 0, make, drop);
        l.touch(S, copy_elide::kRetire, make, drop);
        CHECK(made == 3 && ends.back() == copy_elide::kEndSrcRetire);
    }
    {
        // The limit makes the oldest; a flush makes the rest.
        made = dropped = 0;
        ends.clear();
        Ledger l(2);
        l.defer(S, D, 0, make, drop);
        l.defer(S, E, 0, make, drop);
        l.defer(S, base_of(3), 0, make, drop);
        CHECK(made == 1 && ends.back() == copy_elide::kEndLimit && l.size() == 2);
        CHECK(!l.is_dst(D) && l.is_dst(E));
        l.flush(make);
        CHECK(made == 3 && !l.live());
    }
    {
        // Phase 1c: a pair made after all twice in a row is made where asked, one in a window left unmade again - 2
        // after two made, doubling with each made again up to kRetry; one exchanged or dropped starts over; other
        // pairs are not affected.
        made = dropped = 0;
        ends.clear();
        Ledger l;
        CHECK(l.worth_deferring(S, D));
        l.defer(S, D, 0, make, drop);
        l.touch(S, copy_elide::kWrite, make, drop);
        CHECK(made == 1 && l.worth_deferring(S, D));
        l.defer(S, D, 0, make, drop);
        l.touch(S, copy_elide::kWrite, make, drop);
        CHECK(made == 2);
        // Each window: the copies asked for until one is left unmade again, which is made after all once more.
        for (std::uint32_t want : {2u, 4u, 8u, 16u, 32u, 64u, 64u}) {
            std::uint32_t n = 1;
            while (!l.worth_deferring(S, D) && n < 1000) ++n;
            CHECK(n == want);
            l.defer(S, D, 0, make, drop);
            l.touch(S, copy_elide::kWrite, make, drop);
        }
        CHECK(l.worth_deferring(S, E) && l.worth_deferring(D, S));
        while (!l.worth_deferring(S, D)) {
        }
        l.defer(S, D, 0, make, drop);
        CHECK(l.take(S));  // exchanged
        CHECK(l.worth_deferring(S, D) && l.worth_deferring(S, D));
        // made twice again, then dropped: starts over
        for (int k = 0; k < 2; ++k) {
            l.defer(S, D, 0, make, drop);
            l.touch(D, copy_elide::kReadOther, make, drop);
        }
        CHECK(!l.worth_deferring(S, D));
        l.defer(S, D, 0, make, drop);
        l.touch(D, copy_elide::kOverwrite, make, drop);
        CHECK(dropped == 1 && l.worth_deferring(S, D));
        // a flush counts as made
        l.defer(S, D, 0, make, drop);
        l.flush(make);
        l.defer(S, D, 0, make, drop);
        l.flush(make);
        CHECK(!l.worth_deferring(S, D));
    }
}

}  // namespace

int main() {
    cases();
    random_runs();
    if (g_failures) {
        std::fprintf(stderr, "copy_elide_test: %d failures\n", g_failures);
        return 1;
    }
    std::printf("copy_elide_test: ok\n");
    return 0;
}
