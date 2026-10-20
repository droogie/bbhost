// The twins' miss filter (host/twin_filter.h) against the map walks it stands
// in for, without the GPU: a model of host/gpu.cpp's twin state - candidates
// made, armed by a counting dispatch, twins made on a lookup inside an armed
// one, let go at an edge or by a page-table walk, dropped with their resource
// through the gone queue, CPU-mapped ones dropped - driven by random
// sequences, with every lookup answered twice, the old way (the drains, then
// the live map, then the armed ranges, as twin_locate_locked, twin_covers_locked,
// twin_holds_live_locked and the candidate tests walk them) and the new way
// (the filter first, when no drop is pending), on two copies of the state:
// the answers and the states after must be the same. Four mutants of the new
// way (a stale filter, an off-by-one in its test, the pending drops ignored,
// the armed ranges left out) must each be caught.
//
//   twin_filter_test           the random sequences and the mutants
//   twin_filter_test --bench   ns a lookup on the miss path, the maps against the filter
#include "host/twin_filter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

using u64 = std::uint64_t;

int g_failed = 0;
// What the shipped way's lookups went through, so a pass is not an empty one.
struct Coverage {
    unsigned long long fast = 0, pending = 0, served = 0, slow = 0, made = 0, edges = 0, live_seen = 0, armed_seen = 0;
} g_cov;
void check_that(bool ok, const char* what, int line) {
    if (ok) return;
    if (++g_failed <= 40) std::printf("FAIL (line %d): %s\n", line, what);
}
#define CHECK(x) check_that((x), #x, __LINE__)

struct Cand {
    u64 bytes = 0, gen = 0;
    bool let_go = false, armed = false;
    bool operator==(const Cand& o) const { return bytes == o.bytes && gen == o.gen && let_go == o.let_go && armed == o.armed; }
};
struct Live {
    u64 bytes = 0, gen = 0;
};

// The new way's variants: the filter as shipped, and the mutants.
enum class Way { Old, New, MutStale, MutOffByOne, MutIgnorePending, MutNoArmed };

// A mutant filter: the shipped one but for the one change its way names.
struct MutantFilter {
    std::vector<std::pair<u64, u64>> spans;  // merged
    u64 shape = ~0ull;
    bool built = false;
    template <class L, class A>
    void refresh(const L& live, const A& armed, Way way) {
        const u64 s = live.shape() + armed.shape();
        if (built && (s == shape || way == Way::MutStale)) return;  // MutStale: built once, never again
        std::vector<std::pair<u64, u64>> v;
        for (const auto& [m, t] : live) v.push_back({m, m + std::max<u64>(t.bytes, 1)});
        if (way != Way::MutNoArmed) {
            for (const auto& [m, e] : armed) v.push_back({m, std::max<u64>(e, m + 1)});
        }
        std::sort(v.begin(), v.end());
        spans.clear();
        for (const auto& sp : v) {
            if (!spans.empty() && sp.first <= spans.back().second) {
                spans.back().second = std::max(spans.back().second, sp.second);
            } else {
                spans.push_back(sp);
            }
        }
        shape = s;
        built = true;
    }
    bool meets(u64 va, u64 bytes, Way way) const {
        for (const auto& [s, e] : spans) {
            if (way == Way::MutOffByOne ? (va + 1 < e && s < va + bytes) : (va < e && s < va + bytes)) return true;
        }
        return false;
    }
};

// One copy of the renderer's twin state, as host/gpu.cpp keeps it at level 3.
struct State {
    std::map<u64, Cand> cand;
    twins::ShapedMap<Live> live;
    twins::ShapedMap<u64> armed;
    std::vector<std::pair<u64, u64>> gone_q;
    u64 gone_n = 0, gone_seen = 0;
    std::set<u64> cpu_mapped;
    u64 cpu_n = 0, cpu_seen = 0;
    // The new way's filter (the shipped one, or a mutant).
    twins::MissFilter filter;
    MutantFilter mutant;
    u64 served = 0;  // the counts a lookup leaves behind
    u64 refused = 0;

    bool same(const State& o) const {
        if (cand != o.cand || live.size() != o.live.size() || armed.size() != o.armed.size()) return false;
        for (auto a = live.begin(), b = o.live.begin(); a != live.end(); ++a, ++b) {
            if (a->first != b->first || a->second.bytes != b->second.bytes || a->second.gen != b->second.gen) return false;
        }
        for (auto a = armed.begin(), b = o.armed.begin(); a != armed.end(); ++a, ++b) {
            if (a->first != b->first || a->second != b->second) return false;
        }
        return gone_q == o.gone_q && gone_n == o.gone_n && gone_seen == o.gone_seen && cpu_mapped == o.cpu_mapped && cpu_n == o.cpu_n &&
               cpu_seen == o.cpu_seen && served == o.served && refused == o.refused;
    }

    // ---- as host/gpu.cpp does them -------------------------------------------------
    void armed_forget(u64 memory) {  // twin_armed_forget_locked
        const auto r = armed.find(memory);
        if (r == armed.end()) return;
        if (const auto c = cand.find(memory); c != cand.end() && c->second.armed && !c->second.let_go) return;
        armed.erase(r);
    }
    enum Why { kEdge, kDead, kWalk, kCpuMapped };
    void drop(u64 memory, Why why, u64 gen = 0) {  // twin_drop_locked
        if (auto it = live.find(memory); it != live.end()) live.erase(it);
        if (auto c = cand.find(memory); c != cand.end()) {
            if (why == kDead) {
                if (c->second.gen == gen) cand.erase(c);
            } else {
                c->second.let_go = true;
            }
        }
        armed_forget(memory);
    }
    void drain() {  // twins_drain_gone_locked
        if (gone_n == gone_seen) return;
        std::vector<std::pair<u64, u64>> q;
        q.swap(gone_q);
        gone_seen = gone_n;
        for (const auto& [memory, gen] : q) {
            if (const auto it = live.find(memory); it != live.end() && it->second.gen == gen) drop(memory, kDead, gen);
            armed_forget(memory);
        }
    }
    void drop_cpu_mapped() {  // twins_drop_cpu_mapped_locked
        if (cpu_n == cpu_seen || live.empty()) return;
        cpu_seen = cpu_n;
        std::vector<u64> gone;
        for (const auto& [memory, t] : live) {
            const auto c = cand.find(memory);
            if (c != cand.end() && c->second.gen == t.gen && c->second.let_go && cpu_mapped.count(t.gen)) gone.push_back(memory);
        }
        for (const u64 m : gone) drop(m, kCpuMapped);
    }
    bool pending() const { return gone_n != gone_seen || (cpu_n != cpu_seen && !live.empty()); }  // twins_drop_pending_locked

    // The new way's "meets nothing" (twin_filter_misses_locked).
    bool filter_misses(u64 va, u64 bytes, Way way) {
        if (way == Way::Old) return false;
        if (way == Way::New) {
            filter.refresh(live, armed);
            return !filter.meets(va, bytes);
        }
        mutant.refresh(live, armed, way);
        return !mutant.meets(va, bytes, way);
    }
    bool skip_pending(Way way) const { return way == Way::MutIgnorePending ? false : pending(); }

    // twin_locate_locked at level 3: served, and where.
    std::pair<bool, u64> locate(u64 va, u64 bytes, Way way) {
        if (!bytes || cand.empty()) return {false, 0};
        const bool cov = way == Way::New;
        if (cov) {
            g_cov.pending += pending();
            g_cov.live_seen += !live.empty();
            g_cov.armed_seen += !armed.empty();
        }
        if (way != Way::Old && !skip_pending(way) && filter_misses(va, bytes, way)) {
            g_cov.fast += cov;
            return {false, 0};
        }
        drain();
        drop_cpu_mapped();
        if (const auto lt = twins::live_holding(live, va, bytes); lt != live.end()) {
            ++served;
            g_cov.served += cov;
            return {true, lt->first * 4 + (va - lt->first)};
        }
        if (!twins::ranges_meet(armed, va, bytes)) return {false, 0};
        g_cov.slow += cov;
        // The candidate holding va, and any starting inside the range after it.
        u64 memory = 0;
        bool inside = false;
        std::vector<u64> edge;
        auto it = cand.upper_bound(va);
        if (it != cand.begin()) {
            const auto prev = std::prev(it);
            if (va < prev->first + prev->second.bytes && !prev->second.let_go && prev->second.armed) {
                if (va + bytes <= prev->first + prev->second.bytes) {
                    inside = true;
                    memory = prev->first;
                } else {
                    edge.push_back(prev->first);
                }
            }
        }
        for (; it != cand.end() && it->first < va + bytes && edge.size() < 2; ++it) {
            if (!it->second.let_go && it->second.armed) edge.push_back(it->first);
        }
        if (!edge.empty()) {  // across an edge (no render pass in the model): copied back and let go
            g_cov.edges += cov;
            for (const u64 e : edge) {
                if (live.count(e)) {
                    drop(e, kEdge);
                } else if (auto c = cand.find(e); c != cand.end()) {
                    c->second.let_go = true;
                }
            }
            return {false, 0};
        }
        if (!inside) return {false, 0};
        const Cand& c = cand.at(memory);
        auto t = live.find(memory);
        if (t != live.end() && t->second.gen != c.gen) {
            drop(memory, kDead, t->second.gen);
            t = live.end();
        }
        if (cand.find(memory) == cand.end()) return {false, 0};
        if (t == live.end()) {
            t = live.emplace(memory, Live{cand.at(memory).bytes, cand.at(memory).gen}).first;
            g_cov.made += cov;
        }
        ++served;
        return {true, t->first * 4 + (va - memory)};
    }
    // twin_covers_locked.
    bool covers(u64 va, u64 bytes, Way way) {
        if (live.empty() || !bytes) return false;
        if (way != Way::Old && !skip_pending(way) && filter_misses(va, bytes, way)) return false;
        drain();
        drop_cpu_mapped();
        const bool any = twins::live_meets(live, va, bytes);
        if (any) ++refused;
        return any;
    }
    // twin_holds_live_locked.
    bool holds_live(u64 va, u64 bytes, Way way) {
        if (live.empty() || !bytes) return false;
        if (way != Way::Old && filter_misses(va, bytes, way)) return false;
        return twins::live_holding(live, va, bytes) != live.end();
    }
    // twin_candidate_straddles at level 3.
    bool straddles(u64 va, u64 bytes, Way way) {
        if (!bytes || cand.empty()) return false;
        if ((way != Way::Old && filter_misses(va, bytes, way)) || !twins::ranges_meet(armed, va, bytes)) return false;
        auto it = cand.upper_bound(va);
        if (it != cand.begin()) {
            const auto prev = std::prev(it);
            if (!prev->second.let_go && prev->second.armed && va < prev->first + prev->second.bytes && va + bytes > prev->first + prev->second.bytes) {
                return true;
            }
        }
        for (; it != cand.end() && it->first < va + bytes; ++it) {
            if (!it->second.let_go && it->second.armed) return true;
        }
        return false;
    }
    // twin_candidate_holds at level 3.
    bool cand_holds(u64 va, u64 bytes, Way way) {
        if (!bytes || cand.empty()) return false;
        if ((way != Way::Old && filter_misses(va, bytes, way)) || !twins::ranges_meet(armed, va, bytes)) return false;
        auto it = cand.upper_bound(va);
        if (it == cand.begin()) return false;
        --it;
        return !it->second.let_go && it->second.armed && va + bytes <= it->first + it->second.bytes;
    }
};

// The random driver: slots for candidates in a few clusters, so ranges abut,
// nest under a newer resource and leave gaps; lookups around them.
struct Driver {
    std::mt19937_64 rng;
    std::vector<u64> slots;
    u64 next_gen = 1;
    explicit Driver(u64 seed) : rng(seed) {
        for (u64 cluster = 0; cluster < 4; ++cluster) {
            u64 at = 0x200000000ull + cluster * 0x10000000ull;
            for (int k = 0; k < 6; ++k) {
                slots.push_back(at);
                at += 0x1000 * (1 + rng() % 16);
            }
        }
    }
    u64 pick(u64 n) { return rng() % n; }
    u64 slot() { return slots[pick(slots.size())]; }
    u64 size() {
        static const u64 sizes[] = {0x100, 0x1000, 0x2000, 0x3000, 0x10000, 0x40};
        return sizes[pick(6)];
    }
    // A lookup range near the candidates: inside, over an edge, between, far away.
    std::pair<u64, u64> range() {
        const u64 s = slot();
        switch (pick(8)) {
            case 0: return {s + pick(0x100) * 16, 16 * (1 + pick(64))};
            case 1: return {s - 16 * (1 + pick(8)), 16 * (1 + pick(64))};  // across a start
            case 2: return {s + 0x1000 - 16 * pick(4), 16 * (1 + pick(16))};  // across an end
            case 3: return {s + pick(0x20000), 1 + pick(0x400)};
            case 4: return {0x100000000ull + pick(0x200000000ull), 4 + pick(0x10000)};  // anywhere
            case 5: return {s, 1};
            case 6: return {s + size() - 1, 1 + pick(4)};  // from a candidate's last byte
            default: return {s - 1, 2};
        }
    }
};

// One random sequence: each step mutates both copies the same way, or asks
// a lookup of both and compares. Returns the mismatches the way caused.
u64 run_sequence(u64 seed, Way way, int steps, u64* lookups = nullptr) {
    Driver d(seed);
    State a, b;  // a: the old way, b: `way`
    u64 mismatches = 0;
    const auto both = [&](auto&& f) {
        f(a);
        f(b);
    };
    for (int step = 0; step < steps; ++step) {
        const u64 op = d.pick(100);
        if (op < 10) {  // a candidate made (a counting dispatch's buffer, or a resource made over one's memory)
            const u64 m = d.slot(), bytes = d.size(), gen = d.next_gen++;
            both([&](State& s) {
                if (auto c = s.cand.find(m); c != s.cand.end()) {  // the old resource is gone: queued for the drain
                    s.gone_q.push_back({m, c->second.gen});
                    ++s.gone_n;
                }
                s.cand[m] = Cand{bytes, gen, false, false};
            });
        } else if (op < 22) {  // wanted by a counting dispatch, armed when the next recording begins
            const u64 m = d.slot();
            both([&](State& s) {
                if (auto c = s.cand.find(m); c != s.cand.end() && !c->second.armed && !c->second.let_go) {
                    c->second.armed = true;
                    s.armed.assign(m, m + c->second.bytes);
                }
            });
        } else if (op < 26) {  // the resource destroyed: the registry forgets the candidate, the drain drops its twin
            const u64 m = d.slot();
            both([&](State& s) {
                if (auto c = s.cand.find(m); c != s.cand.end()) {
                    const u64 gen = c->second.gen;
                    s.cand.erase(c);
                    s.gone_q.push_back({m, gen});
                    ++s.gone_n;
                }
            });
        } else if (op < 29) {  // a Map handed a buffer its own memory
            const u64 m = d.slot();
            both([&](State& s) {
                if (auto c = s.cand.find(m); c != s.cand.end()) {
                    s.cpu_mapped.insert(c->second.gen);
                    ++s.cpu_n;
                }
            });
        } else if (op < 32) {  // a page-table walk over a range keeps it in guest memory (twins_keep_in_guest_locked)
            const auto [va, bytes] = d.range();
            both([&](State& s) {
                std::vector<u64> hit;
                auto it = s.live.upper_bound(va);
                if (it != s.live.begin() && va < std::prev(it)->first + std::prev(it)->second.bytes) hit.push_back(std::prev(it)->first);
                for (; it != s.live.end() && it->first < va + bytes && hit.size() < 8; ++it) hit.push_back(it->first);
                for (const u64 m : hit) s.drop(m, State::kWalk);
                auto c = s.cand.upper_bound(va);
                if (c != s.cand.begin() && va < std::prev(c)->first + std::prev(c)->second.bytes) --c;
                for (; c != s.cand.end() && c->first < va + bytes; ++c) c->second.let_go = true;
            });
        } else {  // lookups, each function in turn
            const auto [va, bytes] = d.range();
            const int fn = static_cast<int>(d.pick(5));
            bool same_answer = true;
            if (fn == 0) {
                same_answer = a.locate(va, bytes, Way::Old) == b.locate(va, bytes, way);
            } else if (fn == 1) {
                same_answer = a.covers(va, bytes, Way::Old) == b.covers(va, bytes, way);
            } else if (fn == 2) {
                same_answer = a.holds_live(va, bytes, Way::Old) == b.holds_live(va, bytes, way);
            } else if (fn == 3) {
                same_answer = a.straddles(va, bytes, Way::Old) == b.straddles(va, bytes, way);
            } else {
                same_answer = a.cand_holds(va, bytes, Way::Old) == b.cand_holds(va, bytes, way);
            }
            if (lookups) ++*lookups;
            if (!same_answer || !a.same(b)) {
                ++mismatches;
                if (way == Way::New && mismatches <= 5) {
                    std::printf("  seed %llu step %d fn %d: 0x%llx + 0x%llx answered differently\n", static_cast<unsigned long long>(seed), step, fn,
                                static_cast<unsigned long long>(va), static_cast<unsigned long long>(bytes));
                }
                a = State();  // start over from nothing, both
                b = State();
            }
        }
    }
    return mismatches;
}

// ---- the miss path's cost --------------------------------------------------------
int bench() {
    std::mt19937_64 rng(7);
    twins::ShapedMap<Live> live;
    twins::ShapedMap<u64> armed;
    // A world's particle twins: two dozen buffers of 64 KiB to 4 MiB in the GX
    // heap, all armed and live.
    for (int k = 0; k < 24; ++k) {
        const u64 m = 0x210000000ull + static_cast<u64>(k) * 0x800000ull + (rng() % 0x100) * 0x1000;
        const u64 bytes = 0x10000ull << (rng() % 7);
        live.emplace(m, Live{bytes, static_cast<u64>(k + 1)});
        armed.assign(m, m + bytes);
    }
    twins::MissFilter filter;
    filter.refresh(live, armed);
    // A frame's other bindings: constants, vertices and tables across 8 GiB of
    // heap - nearly every one a miss.
    constexpr int kN = 1 << 20;
    std::vector<std::pair<u64, u64>> q(kN), near(kN);
    for (auto& r : q) r = {0x100000000ull + (rng() % 0x200000000ull) / 16 * 16, 16 + rng() % 0x4000};
    // ... and the harder misses: inside the twins' span, between them.
    for (auto& r : near) {
        do {
            r = {0x210000000ull + (rng() % 0xc000000ull) / 16 * 16, 16 + rng() % 0x400};
        } while (twins::live_meets(live, r.first, r.second) || twins::ranges_meet(armed, r.first, r.second));
    }
    // Between lookups the command processor touches megabytes of its own
    // state; the cold rounds walk 32 MiB between lookups to evict the maps.
    std::vector<std::uint8_t> evict(32u << 20, 1);
    std::atomic<u64> gone_n{0}, cpu_n{0};
    const u64 gone_seen = 0, cpu_seen = 0;
    volatile u64 sink = 0;
    // Each figure the fastest of seven passes, its loop and range reads taken
    // off (mode 0); a compiler barrier a lookup keeps the loops from merging.
    const auto pass = [&](int mode, bool cold, const std::vector<std::pair<u64, u64>>& qs) {
        const int n = cold ? kN / 32 : kN;
        u64 hits = 0, e = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) {
            if (cold) {
                for (int k = 0; k < 16; ++k) {  // 16 lines a lookup, a different place each time
                    e = (e + 64 * 4099) % evict.size();
                    hits += evict[e];
                }
            }
            const auto [va, bytes] = qs[static_cast<std::size_t>(i)];
            if (mode == 0) {
                hits += (va ^ bytes) & 1;
            } else {
                const bool pending = gone_n.load(std::memory_order_acquire) != gone_seen || (cpu_n.load(std::memory_order_relaxed) != cpu_seen && !live.empty());
                if (mode == 1) {  // the maps: the counter checks, the live walk, the armed walk
                    hits += pending || twins::live_holding(live, va, bytes) != live.end() || twins::ranges_meet(armed, va, bytes);
                } else {  // the filter
                    filter.refresh(live, armed);
                    hits += pending || filter.meets(va, bytes);
                }
            }
            asm volatile("" : "+r"(hits));
        }
        sink = sink + hits;
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / n;
    };
    const auto best = [&](int mode, bool cold, const std::vector<std::pair<u64, u64>>& qs) {
        double b = 1e30;
        for (int r = 0; r < 7; ++r) b = std::min(b, pass(mode, cold, qs));
        return b;
    };
    for (int set = 0; set < 2; ++set) {
        const auto& qs = set ? near : q;
        double res[2][3];
        for (int cold = 0; cold < 2; ++cold) {
            for (int mode = 0; mode < 3; ++mode) res[cold][mode] = best(mode, cold != 0, qs);
        }
        std::printf("%s: hot - the maps %.1f ns a lookup, the filter %.1f; cold (32 MiB walked between) - the maps %.1f, the filter %.1f [%zu spans]\n",
                    set ? "misses between the twins" : "misses across 8 GiB    ", res[0][1] - res[0][0], res[0][2] - res[0][0], res[1][1] - res[1][0],
                    res[1][2] - res[1][0], filter.spans());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return bench();

    // The filter's own test against the map tests it stands in for: random
    // live and armed sets, random ranges - "meets nothing" only where both maps say so.
    {
        std::mt19937_64 rng(1);
        u64 checked = 0, nos = 0;
        for (int round = 0; round < 200; ++round) {
            twins::ShapedMap<Live> live;
            twins::ShapedMap<u64> armed;
            twins::MissFilter f;
            const int n = static_cast<int>(rng() % 12);
            for (int k = 0; k < n; ++k) {
                const u64 m = 0x1000 * (rng() % 64);
                if (rng() % 2) {
                    live.emplace(m, Live{(rng() % 4) * 0x800, 1});  // empty twins too
                } else {
                    armed.assign(m, m + (rng() % 4) * 0x800);  // and empty ranges
                }
            }
            f.refresh(live, armed);
            for (int k = 0; k < 2000; ++k) {
                const u64 va = rng() % 0x48000, bytes = 1 + rng() % 0x3000;
                const bool maps = twins::live_meets(live, va, bytes) || twins::ranges_meet(armed, va, bytes) ||
                                  twins::live_holding(live, va, bytes) != live.end();
                const bool filt = f.meets(va, bytes);
                CHECK(filt || !maps);  // never "meets nothing" where a map meets
                ++checked;
                nos += !filt;
            }
        }
        CHECK(nos > checked / 4);  // and it does say "nothing" for most of them
        std::printf("filter against the maps: %llu ranges, %llu answered 'meets nothing'\n", static_cast<unsigned long long>(checked),
                    static_cast<unsigned long long>(nos));
    }

    // The shape counts every insert and erase, and only those.
    {
        twins::ShapedMap<u64> m;
        CHECK(m.shape() == 0);
        m.assign(1, 2);
        m.assign(1, 3);
        CHECK(m.shape() == 2);
        const auto it = m.find(1);
        CHECK(it != m.end() && it->second == 3);
        CHECK(m.shape() == 2);  // a find moves nothing
        m.erase(it);
        CHECK(m.shape() == 3 && m.empty());
        twins::ShapedMap<Live> l;
        l.emplace(5, Live{1, 1});
        CHECK(l.shape() == 1);
    }

    // The random sequences: no mismatch the new way, every mutant caught.
    u64 lookups = 0, mismatches = 0;
    for (u64 seed = 1; seed <= 400; ++seed) mismatches += run_sequence(seed, Way::New, 3000, &lookups);
    CHECK(mismatches == 0);
    std::printf("random sequences: %llu lookups, %llu answered differently by the filter\n", static_cast<unsigned long long>(lookups),
                static_cast<unsigned long long>(mismatches));
    std::printf("  locate's lookups: %llu answered by the filter, %llu with a drop pending, %llu served by a live twin, %llu on to the candidates "
                "(%llu twins made, %llu edges); %llu with live twins, %llu with armed ranges\n",
                g_cov.fast, g_cov.pending, g_cov.served, g_cov.slow, g_cov.made, g_cov.edges, g_cov.live_seen, g_cov.armed_seen);
    CHECK(g_cov.fast > 0 && g_cov.pending > 0 && g_cov.served > 0 && g_cov.made > 0 && g_cov.edges > 0);
    const std::pair<Way, const char*> mutants[] = {{Way::MutStale, "a stale filter (built once)"},
                                                   {Way::MutOffByOne, "an off-by-one in its test"},
                                                   {Way::MutIgnorePending, "pending drops ignored"},
                                                   {Way::MutNoArmed, "the armed ranges left out"}};
    for (const auto& [way, name] : mutants) {
        u64 caught = 0;
        for (u64 seed = 1; seed <= 400 && !caught; ++seed) caught += run_sequence(seed, way, 3000);
        CHECK(caught > 0);
        std::printf("mutant, %s: %s\n", name, caught ? "caught" : "NOT CAUGHT");
    }

    if (g_failed) {
        std::printf("twin_filter_test: %d failed\n", g_failed);
        return 1;
    }
    std::printf("twin_filter_test: ok\n");
    return 0;
}
