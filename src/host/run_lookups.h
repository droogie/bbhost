// Two lookups the command processor's in-order draws repeat within a run of
// draws, answered from the last one while what that answer was made of is
// unchanged (BBHOST_RUN_LOOKUPS):
//   - the run's targets published for the pool's draws (render.cpp
//     publish_run_targets_locked): a draw of the run asks again what its run's
//     first draw published, and the slot already holds it - names_of, the
//     names' hash and the slot's checks every time, for the same answer;
//   - the extent of the GX buffer a vertex record starts in
//     (engine/gx_resources.cpp gx_buffer_end): the registry's lock, shared with
//     every GX Map, and its map walk for each record whose extent is asked,
//     73-92% of them in the buffer the lookup before found.
// BBHOST_RUN_LOOKUPS: 1 on (unset), 0 off (both looked up every time, as
// before), 2 (`compare`) every answer worked out as before and returned, and
// each one the memo would have given checked against it - the differences
// counted and logged (the exit report's run-lookups lines). Pure below;
// tests/run_lookups_test.cpp drives both with their mistakes.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>

#ifdef RUN_LOOKUPS_MUTANTS
extern int g_run_lookups_mutant;
#define RUN_LOOKUPS_MUTANT(n) (g_run_lookups_mutant == (n))
#else
#define RUN_LOOKUPS_MUTANT(n) false
#endif

namespace run_lookups {

// BBHOST_RUN_LOOKUPS (host/list_resolve.cpp reads it at the start).
extern const int g_mode;
inline bool on() { return g_mode != 0; }
inline bool compare() { return g_mode == 2; }

// ---- The run's publication -------------------------------------------------
//
// publish_run_targets_locked(m) returns without writing when the way of the
// names' set that holds m's names was published at the erase generation now,
// at most kRtPubAge / 2 flips ago, with m's lookups (looked, reached). After
// any call the slot it took holds m's names and lookups at that generation
// (published then, or found so). So while m is unchanged - its version, which
// run_targets moves whenever it refills or drops the answers - and the slot's
// sequence is the one it had after that call (only the command processor
// writes a slot, every write moving its sequence; one way of a set holds a
// set of names at most, so the search finds this slot again), the call would
// find the same slot holding the same names and lookups: it returns at once
// when the generation is the remembered one and the publication still that
// young - the condition the full call checks, on the same values.
struct PubMemo {
    const void* run = nullptr;   // the RunTargets it was made for (the thread's own)
    std::uint64_t version = 0;   // its version then
    const void* slot = nullptr;  // the way that holds its names
    std::uint64_t seq = 0;       // that way's sequence after the call
    std::uint64_t erase_gen = 0, flip = 0;  // the way's erase generation and flip after the call
};
// Whether the call for `run` (at `version`) returns as the full one would:
// `seq_now` is the remembered way's sequence now.
inline bool pub_hit(const PubMemo& m, const void* run, std::uint64_t version, std::uint64_t seq_now, std::uint64_t erase_gen_now,
                    std::uint64_t flip_now, std::uint64_t age) {
    if (!m.slot || m.run != run) return false;
    if (m.version != version && !RUN_LOOKUPS_MUTANT(1)) return false;                // the run's answers changed
    if (m.seq != seq_now && !RUN_LOOKUPS_MUTANT(2)) return false;                    // the way written since
    if (m.erase_gen != erase_gen_now && !RUN_LOOKUPS_MUTANT(3)) return false;        // a target let go since
    if (RUN_LOOKUPS_MUTANT(4)) return true;                                          // (the age not checked)
    if (RUN_LOOKUPS_MUTANT(5)) return m.flip + age >= flip_now;                      // (the publication's whole age)
    return m.flip + age / 2 >= flip_now;
}

// ---- The GX buffer's extent ------------------------------------------------
//
// gx_buffer_end(va) finds the registry's last memory key at or below va, its
// resource, and answers that resource's end when it is a live buffer whose
// memory va lies before the end of, else 0. That answer is a function of the
// registry's shape alone - its memory index, its set of holders, and each
// entry's memory, size, type and liveness - and stays the same for every
// address of an interval around va: the lookup gives it with the answer, and
// it holds while the shape's generation, which the registry moves under its
// lock after each change of the shape, is the one read with it.
struct ExtentMemo {
    std::uint64_t gen = 0;  // the shape's generation it holds for (0: none; the generation starts at 1)
    std::uint64_t lo = 0, hi = 0;  // the addresses [lo, hi) that answer `end`
    std::uint64_t end = 0;
};
inline bool extent_hit(const ExtentMemo& m, std::uint64_t gen, std::uint64_t va) {
    if (m.gen != gen && !RUN_LOOKUPS_MUTANT(6)) return false;
    if (RUN_LOOKUPS_MUTANT(7)) return va < m.hi;
    if (RUN_LOOKUPS_MUTANT(8)) return va >= m.lo && va <= m.hi;
    return va >= m.lo && va < m.hi;
}
// The lookup over `by_memory` (memory base -> key, ordered) and `find` (key ->
// a pointer to the resource's fields `alive`, `type`, `memory`, `bytes`, or
// null): the answer, and in `*m` the interval that answers the same (its
// generation is the caller's). The interval stops short of the next memory
// key (another entry from there) and, for a buffer's end, of the end (0 past
// it); an address past the last key never has ~0 in it.
template <class ByMemory, class Find>
std::uint64_t extent_lookup(const ByMemory& by_memory, Find&& find, std::uint64_t va, ExtentMemo* m) {
    auto it = by_memory.upper_bound(va);
    const std::uint64_t next = it == by_memory.end() ? ~0ull : static_cast<std::uint64_t>(it->first);
    if (it == by_memory.begin()) {
        m->lo = 0;
        m->hi = next;
        m->end = 0;
        return 0;
    }
    --it;
    const std::uint64_t lo = it->first;
    const auto* r = find(it->second);
    if (!r || !r->alive || r->type != 1) {
        m->lo = lo;
        m->hi = next;
        m->end = 0;
        return 0;
    }
    const std::uint64_t end = r->memory + r->bytes;
    if (va < end) {
        m->lo = lo;
        m->hi = RUN_LOOKUPS_MUTANT(9) ? next : std::min(next, end);
        m->end = end;
        return end;
    }
    m->lo = RUN_LOOKUPS_MUTANT(10) ? lo : std::max(lo, end);
    m->hi = next;
    m->end = 0;
    return 0;
}

// The registry's changes of its shape, each followed at once by the
// generation's move (release) - before anything that could release a lock or
// free memory, so no thread can have synchronised with a change and not with
// its move (an erased node is freed after the move: extract, then the node
// handle's end). Under the registry's lock.
inline void shape_moved(std::atomic<std::uint64_t>& gen) {
    if (!RUN_LOOKUPS_MUTANT(11)) gen.fetch_add(1, std::memory_order_release);
}
template <class Map, class It>
void erase_moved(Map& map, It it, std::atomic<std::uint64_t>& gen) {
    auto node = map.extract(it);
    if (!RUN_LOOKUPS_MUTANT(12)) shape_moved(gen);
}
template <class Map, class Key, class Value>
void set_moved(Map& map, const Key& key, const Value& value, std::atomic<std::uint64_t>& gen) {
    map[key] = value;
    if (!RUN_LOOKUPS_MUTANT(13)) shape_moved(gen);
}
template <class Map, class Key, class Value>
void emplace_moved(Map& map, const Key& key, const Value& value, std::atomic<std::uint64_t>& gen) {
    map.emplace(key, value);
    if (!RUN_LOOKUPS_MUTANT(14)) shape_moved(gen);
}

}  // namespace run_lookups
