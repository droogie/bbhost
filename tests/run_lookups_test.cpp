// The run's two lookup memos (host/run_lookups.h), without the game.
//
// The GX buffer's extent: a registry of buffers and textures - made, destroyed,
// their memory moved by a Map, released by an unmap, their holders reused -
// changed only through the header's helpers that move the shape's generation,
// with lookups at the edges of every entry (its start, its end, the next one's
// start, the gaps) and in runs of the same buffer, as a draw's records ask. A
// memo's answer must be the answer gx_buffer_end worked out before it (copied
// below as it was), at every lookup.
//
// The run's publication: the slots of a few sets of two ways each (so runs
// share sets), the command processor's run refilled with names and lookups
// from small pools (so a run comes back, with the same lookups or other ones),
// other runs publishing over its way, targets let go (the erase generation)
// and flips going by. Whenever the memo says the call returns, the call as
// publish_run_targets_locked made it before (copied below) must return too.
//
// Then fourteen deliberate mistakes, each of which must be caught.
#define RUN_LOOKUPS_MUTANTS
#include "host/run_lookups.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <unordered_map>
#include <vector>

int g_run_lookups_mutant = 0;
const int run_lookups::g_mode = 1;

namespace {

int g_failed = 0;
void check(bool ok, const char* what) {
    if (ok) return;
    ++g_failed;
    if (!g_run_lookups_mutant && g_failed <= 24) std::printf("FAIL: %s\n", what);
}

std::uint64_t g_rng = 0x2545f4914f6cdd1dull;
std::uint64_t next() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}
std::uint64_t pick(std::uint64_t n) { return next() % n; }

// ---- The GX buffer's extent ----

struct Info {
    bool alive = true;
    std::uint8_t type = 1;
    std::uint64_t memory = 0;
    std::uint32_t bytes = 0;
};
struct Registry {
    std::unordered_map<std::uint64_t, Info> by_holder;
    std::map<std::uint64_t, std::uint64_t> by_memory;
    std::atomic<std::uint64_t> gen{1};
};

// gx_buffer_end as it was (engine/gx_resources.cpp), over the model.
std::uint64_t end_before(const Registry& r, std::uint64_t va) {
    auto it = r.by_memory.upper_bound(va);
    if (it == r.by_memory.begin()) return 0;
    --it;
    const auto h = r.by_holder.find(it->second);
    if (h == r.by_holder.end() || !h->second.alive || h->second.type != 1) return 0;
    const Info& i = h->second;
    return va < i.memory + i.bytes ? i.memory + i.bytes : 0;
}

// The registry's changes, as engine/gx_resources.cpp makes them.
void make(Registry& r, std::uint64_t key, std::uint64_t memory, std::uint32_t bytes, std::uint8_t type) {
    using namespace run_lookups;
    if (auto old = r.by_holder.find(key); old != r.by_holder.end()) {  // the holder's address reused before its destructor
        if (auto m = r.by_memory.find(old->second.memory); m != r.by_memory.end() && m->second == key) erase_moved(r.by_memory, m, r.gen);
        erase_moved(r.by_holder, old, r.gen);
    }
    if (auto m = r.by_memory.find(memory); m != r.by_memory.end() && m->second != key) {  // the memory reused
        if (auto h = r.by_holder.find(m->second); h != r.by_holder.end()) erase_moved(r.by_holder, h, r.gen);
    }
    set_moved(r.by_memory, memory, key, r.gen);
    Info i;
    i.memory = memory;
    i.bytes = bytes;
    i.type = type;
    emplace_moved(r.by_holder, key, i, r.gen);
}
void destroy(Registry& r, std::uint64_t key) {
    using namespace run_lookups;
    auto it = r.by_holder.find(key);
    if (it == r.by_holder.end()) return;
    if (auto m = r.by_memory.find(it->second.memory); m != r.by_memory.end() && m->second == key) erase_moved(r.by_memory, m, r.gen);
    erase_moved(r.by_holder, it, r.gen);
}
void move_memory(Registry& r, std::uint64_t key, std::uint64_t memory) {  // a Map that renamed the memory
    using namespace run_lookups;
    auto it = r.by_holder.find(key);
    if (it == r.by_holder.end() || it->second.memory == memory) return;
    if (auto m = r.by_memory.find(it->second.memory); m != r.by_memory.end() && m->second == key) erase_moved(r.by_memory, m, r.gen);
    it->second.memory = memory;
    shape_moved(r.gen);
    set_moved(r.by_memory, memory, key, r.gen);
}
void release(Registry& r, std::uint64_t va, std::uint64_t len) {  // an unmap: dead, out of the index
    using namespace run_lookups;
    for (auto it = r.by_memory.lower_bound(va); it != r.by_memory.end() && it->first < va + len;) {
        if (auto h = r.by_holder.find(it->second); h != r.by_holder.end()) {
            h->second.alive = false;
            shape_moved(r.gen);
        }
        const auto here = it++;
        erase_moved(r.by_memory, here, r.gen);
    }
}

// A thread's lookup with the memo, as gx_buffer_end makes it.
struct Asker {
    run_lookups::ExtentMemo memo;
    std::uint64_t hits = 0, asks = 0;
    std::uint64_t end(const Registry& r, std::uint64_t va) {
        ++asks;
        if (run_lookups::extent_hit(memo, r.gen.load(std::memory_order_acquire), va)) {
            ++hits;
            return memo.end;
        }
        run_lookups::ExtentMemo m;
        const std::uint64_t e = run_lookups::extent_lookup(
            r.by_memory,
            [&](std::uint64_t key) -> const Info* {
                const auto h = r.by_holder.find(key);
                return h == r.by_holder.end() ? nullptr : &h->second;
            },
            va, &m);
        m.gen = r.gen.load(std::memory_order_relaxed);
        memo = m;
        return e;
    }
};

// The addresses worth asking about around the registry's entries.
std::vector<std::uint64_t> edges(const Registry& r) {
    std::vector<std::uint64_t> v = {0, 1, ~0ull, ~0ull - 1};
    for (const auto& [memory, key] : r.by_memory) {
        v.push_back(memory);
        if (memory) v.push_back(memory - 1);
        v.push_back(memory + 1);
        if (const auto h = r.by_holder.find(key); h != r.by_holder.end()) {
            const std::uint64_t e = h->second.memory + h->second.bytes;
            v.push_back(e - 1);
            v.push_back(e);
            v.push_back(e + 1);
            v.push_back(h->second.memory + h->second.bytes / 2);
        }
    }
    return v;
}

void extent_cases() {
    Registry r;
    Asker a, b;  // two threads' memos
    // Buffers 4 KiB apart with gaps, textures among them, one buffer over another's memory.
    std::uint64_t key = 0x1000;
    for (int k = 0; k < 24; ++k) make(r, key++, 0x10000 + 0x2000ull * k, 0x1000 + 0x100 * (k % 5), k % 4 == 3 ? 3 : 1);
    std::uint64_t asked = 0, hits_in_runs = 0, run_asks = 0;
    for (int round = 0; round < 3000; ++round) {
        // A draw's records: runs of lookups in one buffer (the memo's case), then the edges.
        if (!r.by_memory.empty()) {
            auto it = r.by_memory.begin();
            std::advance(it, static_cast<long>(pick(r.by_memory.size())));
            const std::uint64_t base = it->first;
            const std::uint64_t h0 = a.hits;
            for (int k = 0; k < 6; ++k) {
                const std::uint64_t va = base + pick(0x1800);
                check(a.end(r, va) == end_before(r, va), "a run's lookup in one buffer answered otherwise than before");
                ++run_asks;
            }
            hits_in_runs += a.hits - h0;
        }
        const std::vector<std::uint64_t> e = edges(r);
        for (int k = 0; k < 12; ++k) {
            const std::uint64_t va = e[pick(e.size())];
            check(a.end(r, va) == end_before(r, va), "a lookup at an entry's edge answered otherwise than before");
            check(b.end(r, va) == end_before(r, va), "the other thread's lookup at an edge answered otherwise than before");
            ++asked;
        }
        // A change of the shape, then the same addresses again (a memo of the old shape must not answer).
        const std::uint64_t va0 = e[pick(e.size())], before = a.end(r, va0);
        (void)before;
        switch (pick(7)) {
            case 0: make(r, key++, 0x10000 + 0x800 * pick(96), 0x400 + 0x200 * static_cast<std::uint32_t>(pick(8)), pick(3) ? 1 : 3); break;
            case 1:
                if (!r.by_holder.empty()) {
                    auto it = r.by_holder.begin();
                    std::advance(it, static_cast<long>(pick(r.by_holder.size())));
                    destroy(r, it->first);
                }
                break;
            case 2:
                if (!r.by_holder.empty()) {
                    auto it = r.by_holder.begin();
                    std::advance(it, static_cast<long>(pick(r.by_holder.size())));
                    move_memory(r, it->first, 0x10000 + 0x800 * pick(96));
                }
                break;
            case 3: release(r, 0x10000 + 0x800 * pick(96), 0x800 * (1 + pick(4))); break;
            case 4:
                if (!r.by_holder.empty()) {  // a holder's address reused: the same key made again elsewhere
                    auto it = r.by_holder.begin();
                    std::advance(it, static_cast<long>(pick(r.by_holder.size())));
                    make(r, it->first, 0x10000 + 0x800 * pick(96), 0x600, 1);
                }
                break;
            default: break;  // nothing changes: the memos keep answering
        }
        check(a.end(r, va0) == end_before(r, va0), "after a change, a lookup at the address asked before it answered otherwise than before");
        check(b.end(r, va0) == end_before(r, va0), "after a change, the other thread's lookup answered otherwise than before");
    }
    // Direct cases: each kind of change, the memo made just before it.
    {
        Registry q;
        Asker t;
        make(q, 1, 0x1000, 0x100, 1);
        make(q, 2, 0x2000, 0x100, 1);
        check(t.end(q, 0x1010) == 0x1100, "a buffer's end");
        check(t.end(q, 0x1020) == 0x1100 && t.hits == 1, "the same buffer again answered by the memo");
        check(t.end(q, 0x1100) == 0, "just past the buffer's end: 0");
        check(t.end(q, 0x1fff) == 0, "in the gap before the next buffer: 0");
        check(t.end(q, 0x1080) == 0x1100, "back inside the buffer after the gap");
        check(t.end(q, 0x0fff) == 0, "below the first buffer: 0");
        check(t.end(q, 0x2000) == 0x2100, "the next buffer's start");
        check(t.end(q, 0x1010) == 0x1100, "the first buffer again");
        destroy(q, 1);
        check(t.end(q, 0x1010) == 0, "a destroyed buffer: 0");
        make(q, 3, 0x1000, 0x80, 1);
        check(t.end(q, 0x1010) == 0x1080, "a new buffer over the memory: its end");
        move_memory(q, 3, 0x3000);
        check(t.end(q, 0x1010) == 0, "a buffer whose memory a Map moved: 0 at the old memory");
        check(t.end(q, 0x3010) == 0x3080, "and its end at the new");
        release(q, 0x3000, 0x1000);
        check(t.end(q, 0x3010) == 0, "a buffer whose memory was unmapped: 0");
        make(q, 4, 0x4000, 0x100, 3);
        check(t.end(q, 0x4010) == 0, "a texture: 0");
        make(q, 4, 0x4000, 0x100, 1);
        check(t.end(q, 0x4010) == 0x4100, "the holder reused for a buffer: its end");
    }
    // Each change moves the generation by itself, whatever comes after it in the same change of the registry: the
    // thread that synchronised with it has synchronised with its move.
    {
        Registry q;
        std::uint64_t g0 = q.gen.load();
        run_lookups::set_moved(q.by_memory, std::uint64_t{0x1000}, std::uint64_t{7}, q.gen);
        check(q.gen.load() > g0, "an index set moves the generation");
        g0 = q.gen.load();
        run_lookups::emplace_moved(q.by_holder, std::uint64_t{7}, Info{}, q.gen);
        check(q.gen.load() > g0, "a holder's insert moves the generation");
        g0 = q.gen.load();
        run_lookups::erase_moved(q.by_memory, q.by_memory.begin(), q.gen);
        check(q.gen.load() > g0 && q.by_memory.empty(), "an index erase moves the generation");
        g0 = q.gen.load();
        run_lookups::erase_moved(q.by_holder, q.by_holder.begin(), q.gen);
        check(q.gen.load() > g0 && q.by_holder.empty(), "a holder's erase moves the generation");
        g0 = q.gen.load();
        run_lookups::shape_moved(q.gen);
        check(q.gen.load() > g0, "a field's change moves the generation");
    }
    check(hits_in_runs * 2 > run_asks, "the memo answers most of a run's lookups in one buffer");
    check(asked > 0, "lookups made");
}

// ---- The run's publication ----

constexpr int kNames = 4, kSets = 4, kWays = 2;
constexpr std::uint64_t kAge = 8;  // kRtPubAge
struct Run {
    bool valid = false;
    std::uint32_t names[kNames] = {};
    std::uint32_t looked[3] = {};
    std::uint32_t reached = 0;
    std::uint64_t version = 0;
};
struct Slot {
    std::uint64_t seq = 0;
    std::uint32_t names[kNames] = {};
    Run m;
    std::uint64_t erase_gen = 0, flip = 0;
};
struct Slots {
    Slot s[kSets * kWays];
    std::uint64_t erase_gen = 1, flip = 100, writes = 0;
};
std::uint32_t set_of(const std::uint32_t* n) {
    std::uint32_t h = 2166136261u;
    for (int k = 0; k < kNames; ++k) h = (h ^ n[k]) * 16777619u;
    return (h % kSets) * kWays;
}
// publish_run_targets_locked as it was (render.cpp), over the model: whether it returns without writing; writing
// when `apply`. The way it took in `*took`.
bool returns_before(Slots& g, const Run& m, bool apply, Slot** took) {
    Slot* const set = &g.s[set_of(m.names)];
    Slot* way = nullptr;
    for (int k = 0; k < kWays && !way; ++k) {
        if (set[k].seq && std::memcmp(set[k].names, m.names, sizeof(m.names)) == 0) way = &set[k];
    }
    if (!way) {
        way = &set[0];
        for (int k = 1; k < kWays; ++k) {
            const bool empty = !set[k].seq;
            if (empty || (way->seq && set[k].flip < way->flip)) way = &set[k];
            if (empty) break;
        }
    }
    Slot& p = *way;
    *took = way;
    if (p.seq && p.erase_gen == g.erase_gen && p.flip + kAge / 2 >= g.flip && std::memcmp(p.names, m.names, sizeof(m.names)) == 0 &&
        std::memcmp(p.m.looked, m.looked, sizeof(m.looked)) == 0 && p.m.reached == m.reached)
        return true;
    if (!apply) return false;
    p.seq += 2;
    std::memcpy(p.names, m.names, sizeof(m.names));
    p.m = m;
    p.erase_gen = g.erase_gen;
    p.flip = g.flip;
    ++g.writes;
    return false;
}
// The call with the memo, as publish_run_targets_locked makes it now.
struct Publisher {
    run_lookups::PubMemo memo;
    std::uint64_t hits = 0, calls = 0;
    void publish(Slots& g, const Run& m) {
        if (!m.valid) return;
        ++calls;
        const std::uint64_t seq_now = memo.slot ? static_cast<const Slot*>(memo.slot)->seq : 0;
        if (run_lookups::pub_hit(memo, &m, m.version, seq_now, g.erase_gen, g.flip, kAge)) {
            ++hits;
            Slot* took = nullptr;
            Slots copy = g;  // asked without writing
            check(returns_before(copy, m, false, &took), "the memo returned where the call before would have published");
            return;
        }
        Slot* took = nullptr;
        returns_before(g, m, true, &took);
        memo = run_lookups::PubMemo{&m, m.version, took, took->seq, took->erase_gen, took->flip};
    }
};

void publication_cases() {
    Slots g;
    Run m, other;
    Publisher cp;
    std::uint64_t version = 0;
    const std::uint32_t name_pool[6][kNames] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 9, 9, 9}, {1, 2, 3, 5}, {7, 7, 1, 2}, {4, 3, 2, 1}};
    const std::uint32_t looked_pool[3][3] = {{11, 12, 0}, {11, 13, 0}, {21, 0, 0}};
    const auto refill = [&](Run& r) {
        r.valid = true;
        std::memcpy(r.names, name_pool[pick(6)], sizeof(r.names));
        std::memcpy(r.looked, looked_pool[pick(3)], sizeof(r.looked));
        r.reached = static_cast<std::uint32_t>(pick(2) ? 3 : 7);
        r.version = ++version;
    };
    refill(m);
    for (int step = 0; step < 20000; ++step) {
        switch (pick(12)) {
            case 0:
            case 1: refill(m); break;                // the run's first draw: its lookups (run_targets)
            case 2: m.valid = false; m.version = ++version; break;  // a draw that does not share
            case 3: {                                // another run publishing (the command processor's next run, another thread's)
                Slot* took = nullptr;
                refill(other);
                returns_before(g, other, true, &took);
                break;
            }
            case 4: ++g.erase_gen; break;           // a target let go
            case 5: g.flip += pick(4); break;        // flips go by
            default: break;                          // the run's next draw: the same answers (run_targets shared them)
        }
        if (!m.valid && pick(3) == 0) refill(m);
        cp.publish(g, m);
    }
    check(cp.hits * 4 > cp.calls, "the memo answers a good share of the run's calls");
    // Direct cases: each condition alone sends the call to the full path.
    {
        Slots q;
        Publisher p;
        Run r;
        r.valid = true;
        std::memcpy(r.names, name_pool[0], sizeof(r.names));
        std::memcpy(r.looked, looked_pool[0], sizeof(r.looked));
        r.reached = 3;
        r.version = 1;
        p.publish(q, r);
        check(q.writes == 1 && p.hits == 0, "the first call publishes");
        p.publish(q, r);
        check(p.hits == 1 && q.writes == 1, "the same run again: the memo returns");
        std::memcpy(r.looked, looked_pool[1], sizeof(r.looked));
        r.version = 2;
        p.publish(q, r);
        check(p.hits == 1 && q.writes == 2, "the run's lookups changed (its version moved): published");
        ++q.erase_gen;
        p.publish(q, r);
        check(p.hits == 1 && q.writes == 3, "a target let go: published");
        q.flip += kAge / 2;
        p.publish(q, r);
        check(p.hits == 2 && q.writes == 3, "kRtPubAge / 2 flips on: still the memo's");
        q.flip += 1;
        p.publish(q, r);
        check(p.hits == 2 && q.writes == 4, "a flip more: published");
        Run o = r;  // another run whose names share the set, published into the way
        static_cast<Slot*>(const_cast<void*>(p.memo.slot))->seq += 2;
        p.publish(q, r);
        check(p.hits == 2 && q.writes == 4, "the way written since (its sequence moved): asked in full");
        (void)o;
    }
}

}  // namespace

int main() {
    extent_cases();
    publication_cases();
    int failed = g_failed;
    static const char* const kMistakes[14] = {
        "the run's version not checked",
        "the way's sequence not checked",
        "the erase generation not checked",
        "the publication's age not checked",
        "the publication's whole age allowed (kRtPubAge, not half)",
        "the shape's generation not checked",
        "the extent's start not checked",
        "the extent's end taken in (<=)",
        "a buffer's extent not stopped at its end",
        "past a buffer's end, the extent started at its start",
        "the shape's generation never moved",
        "an erase not moving the generation",
        "an index set not moving the generation",
        "a holder's insert not moving the generation",
    };
    for (int m = 1; m <= 14; ++m) {
        g_failed = 0;
        g_run_lookups_mutant = m;
        g_rng = 0x2545f4914f6cdd1dull;
        extent_cases();
        publication_cases();
        g_run_lookups_mutant = 0;
        if (g_failed == 0) {
            std::printf("FAIL: mistake %d (%s) not caught\n", m, kMistakes[m - 1]);
            ++failed;
        } else {
            std::printf("mistake %d (%s) caught: %d checks failed\n", m, kMistakes[m - 1], g_failed);
        }
    }
    std::printf(failed ? "run_lookups_test: %d failed\n" : "run_lookups_test: ok\n", failed);
    return failed ? 1 : 0;
}
