// The effects' node emission (decomp/sfx/node_emit.cpp) against the game's own,
// sub_2c4e6b0 (tests/decomp/eboot_kit.h), on generated staged nodes: records of
// every kind (waiting for their end, children, CPU particles, neither) with
// their flags drawn, the controller holding or not and on the record or not,
// the instance ending or not, the node stopping or not and its root's or not,
// emission counts below, at and past their limit, times before, at and past
// the next and the end, the record count wide enough to tell a 64-bit compare
// from a 32-bit one, special values. Everything the function reaches further
// is a stub that records its arguments and the node's bytes at the call; the
// controller's two queries answer from the controller (sub_2c82190's own +4,
// and a byte of the case's), so asking again answers the same.
//   - Game: ours is the game's - the same answer, calls in order (the node as
//     each saw it) and memory.
//   - Edges, then Event at the node's own time when Edges answers 1: the game's
//     effects and memory - the controller's queries asked twice aside.
//   - Event at another time: where the game run at that time makes an event
//     and takes no edge, the same event (calls and memory, the node's time
//     aside); elsewhere nothing done and 0 - children only on a whole step's
//     frame.
// Each of the source's mutants must be caught. Skips without the 1.09 eboot.
#define DECOMP_SFX_NODE_EMIT_MUTANTS
int g_sfx_emit_mutant = 0;
#include "decomp/sfx/node_emit.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x2c4e6b0u);
bool g_quiet = false;

u32 bits(float f) {
    u32 u;
    std::memcpy(&u, &f, 4);
    return u;
}

// --- the stubs ------------------------------------------------------------------------

std::mt19937_64 g_answers;  // reset to the case's seed before each run
u64 g_node = 0;             // the case's node, whose bytes each call records
bool g_skip_time = false;   // leave the node's time out of the recorded bytes (an event at another time)

u64 fnv(u64 p, std::size_t n) {
    u64 h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) {
        if (g_skip_time && i >= 0x7c && i < 0x80) continue;
        h = (h ^ reinterpret_cast<const u8*>(p)[i]) * 1099511628211ull;
    }
    return h;
}
struct Call {
    int what;
    u64 a, b, c, d, e, f;
    u64 node_bytes;
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_calls;
void rec(int what, u64 a = 0, u64 b = 0, u64 c = 0, u64 d = 0, u64 e = 0, u64 f = 0) {
    g_calls.push_back({what, a, b, c, d, e, f, g_node ? fnv(g_node, 0x100) : 0});
}
u64 draw() { return g_answers(); }

enum { kCallStateOf = 1, kCallStateHolds, kCallSetUp, kCallEndModules, kCallLinger, kCallAdd, kCallBind, kCallChildren, kCallChildrenAt, kCallCount };

GUEST_ABI u32 stub_state_of(u64 controller) {
    rec(kCallStateOf, controller);
    return load<u32>(controller + 4);  // sub_2c82190's own read
}
GUEST_ABI u8 stub_state_holds(u64 controller, u32 state) {
    rec(kCallStateHolds, controller, state);
    // the case's answer for this state: mostly 0 or 1, now and then another byte
    return load<u8>(controller + 0x20 + (state & 7));
}
GUEST_ABI u8 stub_set_up(u64 node, u64 instance_definition, u64 index, u64 record) {
    rec(kCallSetUp, node, instance_definition, index, record);
    const u64 r = draw();
    return (r & 15) == 0 ? static_cast<u8>(r >> 8) : static_cast<u8>((r >> 4) & 1);
}
GUEST_ABI void stub_end_modules(u64 container) { rec(kCallEndModules, container); }
GUEST_ABI u32 stub_linger(u64 node, u64 instance_definition, u64 context) {
    rec(kCallLinger, node, instance_definition, context);
    return static_cast<u32>(draw() % 10);
}
GUEST_ABI void stub_add(u64 container, u32 count) { rec(kCallAdd, container, count); }
GUEST_ABI void stub_bind(u64 instance_definition, u64 node, u32 id, u64 data) { rec(kCallBind, instance_definition, node, id, data); }
GUEST_ABI void stub_children(u32 id, u64 instance_definition, u64 node, u64 data, u64 d18) { rec(kCallChildren, id, instance_definition, node, data, d18); }
GUEST_ABI void stub_children_at(u32 id, u64 instance_definition, u64 node, u64 data, u64 d18, u64 extra) {
    rec(kCallChildrenAt, id, instance_definition, node, data, d18, extra);
}
GUEST_ABI s32 stub_count(u64 emitter, u64 node, float* time) {
    rec(kCallCount, emitter, node, bits(*time));
    const u64 r = draw();
    return (r & 7) == 0 ? -static_cast<s32>((r >> 8) & 3) : static_cast<s32>((r >> 8) % 5);  // now and then none, or below none
}

// --- memory ---------------------------------------------------------------------------

std::vector<std::vector<u8>> g_blocks;
u64 block(std::size_t n) {
    g_blocks.emplace_back(n + 64, 0);
    const u64 p = (reinterpret_cast<u64>(g_blocks.back().data()) + 63) & ~63ull;
    for (std::size_t k = 0; k + 4 <= n; k += 4) store<u32>(p + k, g_rng.u32());
    return p;
}
float value(float lo, float hi) { return g_rng.value(lo, hi); }

struct Case {
    u64 node, inst, def, ctx;
    u64 seed;
    u32 no_time;
    std::vector<std::pair<u64, std::size_t>> watched;
};

Case make_case() {
    g_blocks.clear();
    Case c{};
    c.seed = g_rng.next();
    c.inst = block(0x98);
    c.def = block(0x40);
    c.ctx = g_rng.chance(0.2) ? 0 : block(0x40);
    // The definition: +0 the record count (now and then above 2^32 with a small low word), +8 the index table (8),
    // +0x10 the records, +0x18 what the children are handed.
    const int records = g_rng.range(1, 6);
    const u64 d = block(0x40);
    const u64 index = block(4 * 8), recs = block(0xa0 * (records + 2));
    for (int i = 0; i < 8; ++i) store<u32>(index + 4 * i, static_cast<u32>(g_rng.range(0, records + 1)));
    const u64 emitter_vt = block(0x80);
    store<u64>(emitter_vt + 0x58, reinterpret_cast<u64>(&stub_count));
    for (int i = 0; i < records + 2; ++i) {
        const u64 r = recs + 0xa0 * i;
        u32 rf = g_rng.u32() & ~0x337u;
        const int kind = g_rng.range(0, 7);
        if (kind == 0) rf |= 1;                        // waits for its end
        else if (kind <= 2) rf |= 2;                   // children
        else if (kind <= 6) rf |= 4;                   // CPU particles
        if (g_rng.chance(0.15)) rf |= 1;               // more than one kind: the game's order picks
        if (g_rng.chance(0.15)) rf |= 2;
        if (g_rng.chance(0.3)) rf |= 0x10;             // parks the next at FLT_MAX past its limit
        if (g_rng.chance(0.5)) rf |= 0x20;             // children with the record's +0x70
        if (g_rng.chance(0.3)) rf |= 0x100;
        if (g_rng.chance(0.3)) rf |= 0x200;
        store<u32>(r + 4, rf);
        u64 emitter = 0;
        if (g_rng.chance(0.85)) {
            emitter = block(0x20);
            store<u64>(emitter, emitter_vt);
        }
        store<u64>(r + 0x28, emitter);
        store<u64>(r + 0x98, g_rng.chance(0.4) ? block(0x20) : 0);
    }
    u64 count = static_cast<u64>(records);
    if (g_rng.chance(0.2)) count |= 1ull << 32;
    if (g_rng.chance(0.15)) count = static_cast<u64>(g_rng.range(0, 7)) << 32;
    store<u64>(d, count);
    store<u64>(d + 8, index);
    store<u64>(d + 0x10, recs);
    // The node.
    c.node = block(0x100);
    u32 flags = (g_rng.u32() & ~0xfu) | 8u | 1u;  // staged and active
    if (g_rng.chance(0.3)) flags |= 2;             // stopping
    store<u32>(c.node + 0x78, flags);
    store<u64>(c.node + 0x58, c.inst);
    const bool root = g_rng.chance(0.5);
    store<u64>(c.inst + 0x78, block(0x10));
    store<u64>(c.node + 0x60, root ? load<u64>(c.inst + 0x78) : block(0x10));
    u8 iflags = static_cast<u8>(g_rng.u32());
    if (g_rng.chance(0.6)) iflags |= 2;  // live instances mostly have it
    store<u8>(c.inst + 0x88, iflags);
    // The controller: its state (+4, what sub_2c82190 answers) and whether each state holds (the case's bytes at +0x20).
    const u64 ctl = block(0x30);
    const u32 at = static_cast<u32>(g_rng.range(0, 7));
    store<u32>(ctl + 4, g_rng.chance(0.6) ? at : static_cast<u32>(g_rng.range(0, 7)));
    for (int i = 0; i < 8; ++i) {
        const u32 r = g_rng.u32();
        store<u8>(ctl + 0x20 + i, (r & 15) == 0 ? static_cast<u8>(r >> 8) : static_cast<u8>(g_rng.chance(0.8) ? 1 : 0));
    }
    store<u64>(c.node + 0xc8, ctl);
    store<u64>(c.node + 0xd0, d);
    store<u32>(c.node + 0xd8, at);
    store<u32>(c.node + 0xdc, 1);
    const float t = value(0, 8);
    store<float>(c.node + 0x7c, t);
    const u32 limit = static_cast<u32>(g_rng.range(0, 6)) | (g_rng.chance(0.1) ? 0x80000000u : 0u);
    store<u32>(c.node + 0xe8, limit);
    store<u32>(c.node + 0xe0, g_rng.chance(0.15) ? limit : static_cast<u32>(g_rng.range(0, 7)) | (g_rng.chance(0.05) ? 0x80000000u : 0u));
    store<float>(c.node + 0xe4, g_rng.chance(0.2) ? t : value(t - 2, t + 2));
    store<float>(c.node + 0xec, g_rng.chance(0.15) ? t : value(t - 4, t + 8));
    store<float>(c.node + 0xf0, value(0, 0.5f));
    const float nt = value(-2, 2);
    std::memcpy(&c.no_time, &nt, 4);
    c.watched = {{c.node, 0x100}, {c.inst, 0x98}, {d, 0x20}, {recs, static_cast<std::size_t>(0xa0 * (records + 2))}, {ctl, 0x30}};
    return c;
}

using Snapshot = std::vector<std::vector<u8>>;
Snapshot snap(const Case& c) {
    Snapshot s;
    for (const auto& [p, n] : c.watched) {
        s.emplace_back(n);
        std::memcpy(s.back().data(), reinterpret_cast<void*>(p), n);
    }
    return s;
}
void restore(const Case& c, const Snapshot& s) {
    for (std::size_t i = 0; i < c.watched.size(); ++i) std::memcpy(reinterpret_cast<void*>(c.watched[i].first), s[i].data(), c.watched[i].second);
}

using EmitFn = GUEST_ABI u32 (*)(u64 node, u64 definition, u64 context);

struct Run {
    u32 answer;
    std::vector<Call> calls;
    Snapshot after;
};
template <class F>
Run run(const Case& c, F&& f) {
    store<u32>(reinterpret_cast<u64>(eboot_kit::at(kNoTime)), c.no_time);
    g_answers.seed(c.seed);
    g_calls.clear();
    g_node = c.node;
    Run r;
    r.answer = f();
    r.calls = g_calls;
    r.after = snap(c);
    g_node = 0;
    return r;
}
std::vector<Call> effects(const std::vector<Call>& calls) {
    std::vector<Call> out;
    for (const Call& x : calls)
        if (x.what != kCallStateOf && x.what != kCallStateHolds) out.push_back(x);
    return out;
}
bool has(const std::vector<Call>& calls, int what) {
    for (const Call& x : calls)
        if (x.what == what) return true;
    return false;
}
// Whether the node's bytes differ outside its time.
bool node_differs_but_time(const Snapshot& a, const Snapshot& b) {
    for (std::size_t i = 0; i < a[0].size(); ++i)
        if ((i < 0x7c || i >= 0x80) && a[0][i] != b[0][i]) return true;
    for (std::size_t k = 1; k < a.size(); ++k)
        if (a[k] != b[k]) return true;
    return false;
}

long g_reached[kCallCount + 1] = {};
long g_events_game = 0, g_events_other = 0, g_edges_cases = 0;

const char* one(const Case& c, int which) {
    const auto game = eboot_kit::fn<EmitFn>(0x2c4e6b0);
    const Snapshot before = snap(c);
    const char* why = nullptr;
    if (which == 0) {
        // Game.
        const Run g = run(c, [&] { return game(c.node, c.def, c.ctx); });
        restore(c, before);
        if (!g_quiet)
            for (const Call& x : g.calls) ++g_reached[x.what];
        const Run o = run(c, [&] { return sfx_node_emit(c.node, c.def, c.ctx); });
        restore(c, before);
        why = g.answer != o.answer ? "Game: the answer" : !(g.calls == o.calls) ? "Game: the calls" : g.after != o.after ? "Game: memory" : nullptr;
    } else if (which == 1) {
        // Edges, then Event at the node's own time when it answers 1: the game's effects.
        const Run g = run(c, [&] { return game(c.node, c.def, c.ctx); });
        restore(c, before);
        const float t = load<float>(c.node + 0x7c);
        const Run o = run(c, [&] {
            const u32 a = sfx_node::emit(c.node, c.def, c.ctx, sfx_node::Emit::Edges, 0.0f, true);
            if (a == 1) sfx_node::emit(c.node, c.def, c.ctx, sfx_node::Emit::Event, t, true);
            return a;
        });
        restore(c, before);
        if (!g_quiet) ++g_edges_cases;
        why = g.answer != o.answer ? "Edges: the answer" : !(effects(g.calls) == effects(o.calls)) ? "Edges+Event: the effects"
              : g.after != o.after                                                                   ? "Edges+Event: memory"
                                                                                                     : nullptr;
    } else if (which == 3) {
        // Edges alone: the game's run without its event - where the game made one, no count, add, children or binder
        // and the count and next as they were; elsewhere the game's run itself.
        const Run g = run(c, [&] { return game(c.node, c.def, c.ctx); });
        restore(c, before);
        const Run o = run(c, [&] { return sfx_node::emit(c.node, c.def, c.ctx, sfx_node::Emit::Edges, 0.0f, true); });
        restore(c, before);
        // The game made an event: a count, an add or children - or, with no emitter, its count moved with no linger after.
        const bool counted = load<u32>(g.after[0].data(), 0xe0) != load<u32>(before[0].data(), 0xe0);
        const bool event = has(g.calls, kCallCount) || has(g.calls, kCallAdd) || has(g.calls, kCallChildren) || has(g.calls, kCallChildrenAt) ||
                           (counted && g.answer == 1 && !has(g.calls, kCallLinger) && !has(g.calls, kCallEndModules));
        if (!event) {
            why = g.answer != o.answer ? "Edges: the answer" : !(g.calls == o.calls) ? "Edges: the calls" : g.after != o.after ? "Edges: memory" : nullptr;
        } else {
            std::vector<Call> want;
            for (const Call& x : g.calls)
                if (x.what != kCallCount && x.what != kCallAdd && x.what != kCallChildren && x.what != kCallChildrenAt && x.what != kCallBind) want.push_back(x);
            Snapshot want_after = g.after;
            std::memcpy(want_after[0].data() + 0xe0, before[0].data() + 0xe0, 8);  // the count and the next as they were
            if (!(want == o.calls) && !g_quiet && std::getenv("EMIT_DEBUG")) {
                std::printf("  want:");
                for (const Call& x : want) std::printf(" %d", x.what);
                std::printf("  ours:");
                for (const Call& x : o.calls) std::printf(" %d", x.what);
                std::printf("  game:");
                for (const Call& x : g.calls) std::printf(" %d", x.what);
                std::printf("\n");
            }
            why = o.answer != 1 ? "Edges: the answer where the game made an event" : !(want == o.calls) ? "Edges: the calls where the game made an event"
                  : o.after != want_after                                                               ? "Edges: memory where the game made an event"
                                                                                                         : nullptr;
        }
    } else {
        // Event at another time.
        const float t = g_rng.chance(0.3) ? load<float>(c.node + 0xe4) : value(load<float>(c.node + 0x7c) - 1, load<float>(c.node + 0x7c) + 1);
        const bool children = g_rng.chance(0.5);
        const u32 t_was = load<u32>(c.node + 0x7c);
        g_skip_time = true;
        store<float>(c.node + 0x7c, t);
        const Snapshot at_t = snap(c);
        const Run g = run(c, [&] { return game(c.node, c.def, c.ctx); });
        restore(c, before);
        const Run o = run(c, [&] { return sfx_node::emit(c.node, c.def, c.ctx, sfx_node::Emit::Event, t, children); });
        restore(c, before);
        g_skip_time = false;
        (void)t_was;
        // The game's run at t made an event and took no edge: an answer of 1, no linger, set-up or reset, the record
        // index kept, and its count moved.
        const bool edge = g.answer != 1 || has(g.calls, kCallLinger) || has(g.calls, kCallSetUp) || has(g.calls, kCallEndModules) ||
                          load<u32>(g.after[0].data(), 0xd8) != load<u32>(at_t[0].data(), 0xd8);
        const bool counted = load<u32>(g.after[0].data(), 0xe0) != load<u32>(at_t[0].data(), 0xe0);
        const bool child = has(g.calls, kCallChildren) || has(g.calls, kCallChildrenAt);
        const bool want_event = !edge && counted && (children || !child);
        if (!g_quiet) (want_event ? g_events_other : g_events_game) += 1;
        if (want_event) {
            why = o.answer != 1                                   ? "Event: no event where the game made one"
                  : !(effects(g.calls) == effects(o.calls))       ? "Event: the event's calls"
                  : node_differs_but_time(g.after, o.after)        ? "Event: memory after the event"
                  : load<u32>(o.after[0].data(), 0x7c) != t_was ? "Event: the node's time moved"
                                                                   : nullptr;
        } else {
            why = o.answer != 0 ? "Event: an answer where the game took an edge or made none" : !effects(o.calls).empty() ? "Event: a call where none is due"
                  : o.after != before                                                                                     ? "Event: memory changed where none is due"
                                                                                                                          : nullptr;
        }
    }
    return why;
}

int check(int cases, int which) {
    int bad = 0;
    for (int k = 0; k < cases; ++k) {
        const Case c = make_case();
        const u32 f = 0;
        (void)f;
        const char* why = one(c, which);
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("case %d (node flags 0x%x, record index %u, made %u of %u): %s differs\n", k, load<u32>(c.node + 0x78), load<u32>(c.node + 0xd8),
                        load<u32>(c.node + 0xe0), load<u32>(c.node + 0xe8), why);
    }
    return bad;
}

}  // namespace

int main() {
    eboot_kit::load("decomp_sfx_node_emit");
    if (std::memcmp(eboot_kit::at(0x2c4e6b0), kEmitEntry, sizeof kEmitEntry) != 0) {
        std::printf("decomp_sfx_node_emit: sub_2c4e6b0's entry is not what the leaf expects\n");
        return 1;
    }
    eboot_kit::stub(kStateOf, reinterpret_cast<const void*>(&stub_state_of));
    eboot_kit::stub(kStateHolds, reinterpret_cast<const void*>(&stub_state_holds));
    eboot_kit::stub(kSetUp, reinterpret_cast<const void*>(&stub_set_up));
    eboot_kit::stub(kEndModules, reinterpret_cast<const void*>(&stub_end_modules));
    eboot_kit::stub(kLinger, reinterpret_cast<const void*>(&stub_linger));
    eboot_kit::stub(kContainerAdd, reinterpret_cast<const void*>(&stub_add));
    eboot_kit::stub(kBind, reinterpret_cast<const void*>(&stub_bind));
    eboot_kit::stub(kChildren, reinterpret_cast<const void*>(&stub_children));
    eboot_kit::stub(kChildrenAt, reinterpret_cast<const void*>(&stub_children_at));

    constexpr int kCases = 40000;
    int bad = 0;
    for (int which = 0; which < 4; ++which) {
        g_rng.specials = false;
        bad += check(kCases, which);
        g_rng.specials = true;
        bad += check(kCases / 2, which);
    }
    eboot_kit::host_fp();
    bad += check(kCases / 4, 0);
    eboot_kit::guest_fp();
    // Every path reached.
    {
        static const char* names[kCallCount + 1] = {"", "state of", "state holds", "set-up", "end modules", "linger", "container add", "bind",
                                                    "children", "children at", "emitter count"};
        int unreached = 0;
        for (int w = 1; w <= kCallCount; ++w)
            if (!g_reached[w]) {
                std::printf("decomp_sfx_node_emit: no case reached %s\n", names[w]);
                ++unreached;
            }
        if (!g_events_other || !g_events_game) {
            std::printf("decomp_sfx_node_emit: events at another time %ld, none due %ld\n", g_events_other, g_events_game);
            ++unreached;
        }
        if (unreached) return 1;
    }
    // Each mutant must make some case differ: Game (4, 5), Edges (2, 8), Event (1, 3, 6, 7).
    int missed = 0;
    g_quiet = true;
    for (int m = 1; m <= 8; ++m) {
        g_sfx_emit_mutant = m;
        g_rng.specials = false;
        const int found = check(20000, 0) + check(20000, 1) + check(20000, 2) + check(20000, 3);
        if (!found) {
            std::printf("decomp_sfx_node_emit: mutant %d not caught\n", m);
            ++missed;
        }
    }
    g_sfx_emit_mutant = 0;
    if (bad || missed) {
        std::printf("decomp_sfx_node_emit: %d cases differ, %d mutants missed\n", bad, missed);
        return 1;
    }
    std::printf("decomp_sfx_node_emit: %d staged nodes as the game's (Game %d, Edges then Event %d, an Event at another time %d, Edges "
                "alone %d, with special values in a third and %d under the host's MXCSR); %ld events at another time made as the game's, "
                "%ld left; 8 mutants caught\n",
                4 * (kCases + kCases / 2) + kCases / 4, kCases + kCases / 2 + kCases / 4, kCases + kCases / 2, kCases + kCases / 2,
                kCases + kCases / 2, kCases / 4, g_events_other, g_events_game);
    return 0;
}
