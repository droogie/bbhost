// The effects' node update (decomp/sfx/node_update.cpp) against the game's own,
// sub_2c4df80 (tests/decomp/eboot_kit.h), on generated nodes: plain and staged,
// in every state and above, dead, inactive, stopping, asking for their render
// items, the root of an instance that is ending or not, with and without a
// render context, their times, the step and the "no time" value drawn with the
// special values (zeros of both signs, denormals, infinities, NaNs) and the
// record count wide enough to tell a 64-bit compare from a 32-bit one.
// Everything the update reaches further is a stub that records its arguments
// and the node's bytes as they are at the call, and answers from a stream both
// runs draw the same way: the same answer, calls (in order, with the node as
// each one saw it) and memory, or the case is reported. Each of the source's
// mutants must be caught. Skips without the 1.09 eboot.
#define DECOMP_SFX_NODE_UPDATE_MUTANTS
int g_sfx_node_mutant = 0;
#include "decomp/sfx/node_update.cpp"

#include "eboot_kit.h"
#include "engine/sfx_schedule.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x2c4df80u);
bool g_quiet = false;

u32 bits(float f) {
    u32 u;
    std::memcpy(&u, &f, 4);
    return u;
}

// --- the stubs' answers and what they saw -------------------------------------------

std::mt19937_64 g_answers;  // reset to the case's seed before each run
u64 g_node = 0;             // the case's node, whose bytes each call records

u64 fnv(u64 p, std::size_t n) {
    u64 h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ reinterpret_cast<const u8*>(p)[i]) * 1099511628211ull;
    return h;
}
struct Call {
    int what;
    u64 a, b, c, d, e, f, g;
    u64 node_bytes;
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_calls;
void rec(int what, u64 a = 0, u64 b = 0, u64 c = 0, u64 d = 0, u64 e = 0, u64 f = 0, u64 g = 0) {
    g_calls.push_back({what, a, b, c, d, e, f, g, g_node ? fnv(g_node, 0x100) : 0});
}
u64 draw() { return g_answers(); }
// A byte answer: mostly 0 or 1, now and then another value (a bool the game reads as a byte).
u8 byte_answer() {
    const u64 r = draw();
    if ((r & 15) == 0) return static_cast<u8>(r >> 8);
    return static_cast<u8>((r >> 4) & 1);
}

std::vector<u64> g_pointers;  // blocks the stubs may answer with
u64 pointer_answer() {
    const u64 r = draw();
    if (g_pointers.empty() || (r & 3) == 0) return 0;
    return g_pointers[(r >> 2) % g_pointers.size()];
}

// The smooth steps' split: the container stub records the step and the node's times each call saw, the
// container call published for its modules (sfx_node::smooth_call), and neither records a call nor draws, so
// the counted parts' calls and answers stay a whole step's.
bool g_split = false;
struct Seen {
    u32 step, t, e;
    u64 context;
    u64 pub_node;  // the published call: its node, the node's time before the offset, a whole step or not
    u32 pub_time;
    bool pub_tick;
    bool operator==(const Seen&) const = default;
};
std::vector<Seen> g_seen;

GUEST_ABI void stub_container(u64 container, float* step, u64 definition, u64 context) {
    if (g_split) {
        const sfx_node::SmoothCall pub = sfx_node::smooth_call();
        g_seen.push_back({bits(*step), load<u32>(container - 0x80 + 0x7c), load<u32>(container - 0x80 + 0xe0), context, pub.node,
                          pub.node ? pub.time : 0u, pub.node ? pub.tick : false});
        return;
    }
    rec(1, container, bits(*step), definition, context);
    // A container update can mark its node dead (state 3's check): the node update must not look again.
    if ((draw() & 7) == 0) store<u32>(container - 0x80 + 0x78, load<u32>(container - 0x80 + 0x78) | 4u);
}
GUEST_ABI u32 stub_state_of(u64 controller) {
    rec(2, controller);
    return static_cast<u32>(draw() % 8);  // a staged node's state 3 reads the index table at it (8 entries)
}
GUEST_ABI u8 stub_state_holds(u64 controller, u32 state) {
    rec(3, controller, state);
    return byte_answer();
}
GUEST_ABI u64 stub_pick(u64 definition, u64 instance_definition, u64 node) {
    rec(4, definition, instance_definition, node);
    return pointer_answer();
}
GUEST_ABI void stub_apply(u64 definition, u64 instance_definition, u64 node) { rec(5, definition, instance_definition, node); }
GUEST_ABI void stub_bind(u64 instance_definition, u64 node, u32 id, u64 data) { rec(6, instance_definition, node, id, data); }
GUEST_ABI u8 stub_set_up(u64 node, u64 instance_definition, u64, u64 record) {
    rec(7, node, instance_definition, record);  // the third is not read (its register holds the records, or what the binder left)
    return byte_answer();
}
GUEST_ABI u32 stub_emit(u64 node, u64 instance_definition, u64 context) {
    rec(8, node, instance_definition, context);
    const u64 r = draw();
    return (r & 15) == 0 ? static_cast<u32>(r >> 32) : static_cast<u32>((r >> 4) % 10);
}
GUEST_ABI u32 stub_linger(u64 node, u64 instance_definition, u64 context) {
    rec(9, node, instance_definition, context);
    return static_cast<u32>(draw() % 10);
}
GUEST_ABI void stub_end_modules(u64 container) { rec(10, container); }
GUEST_ABI u8 stub_modules_done(u64 container) {
    rec(11, container);
    return byte_answer();
}
GUEST_ABI u8 stub_children_gone(u64 container) {
    rec(12, container);
    return byte_answer();
}
GUEST_ABI u64 stub_item_data(u64 data) {
    rec(13, data);
    return pointer_answer();
}
GUEST_ABI u8 stub_one_item(u64 container, u64 context, u64 definition, u32 flag, u64 allocator) {
    rec(14, container, context, definition, flag, allocator);
    return byte_answer();
}
GUEST_ABI u8 stub_three_items(u64 container, u64 context, u64 a, u64 b, u64 c, u32 kind, u32 flag20, u32 flag40, float* time, u64 allocator) {
    rec(15, container, context, a, b, c, (static_cast<u64>(kind) << 32) | (flag20 & 0xffu) << 8 | (flag40 & 0xffu), (static_cast<u64>(bits(*time)) << 32) ^ allocator);
    return byte_answer();
}
GUEST_ABI u64 stub_allocator(u64 object, u64 context, u64 index, u64 vtable) {
    rec(16, object, context, index, vtable);
    return pointer_answer();
}

// --- memory -------------------------------------------------------------------------

std::vector<std::vector<u8>> g_blocks;
u64 block(std::size_t n) {
    g_blocks.emplace_back(n + 64, 0);
    const u64 p = (reinterpret_cast<u64>(g_blocks.back().data()) + 63) & ~63ull;
    for (std::size_t k = 0; k + 4 <= n; k += 4) store<u32>(p + k, g_rng.u32());
    return p;
}
float value(float lo, float hi) { return g_rng.value(lo, hi); }
float step_value() {
    switch (g_rng.range(0, 7)) {
    case 0: return 0.0f;
    case 1: return 1.0f / 30.0f;
    case 2: return 1.0f / 60.0f;
    case 3: return 1.0f / 120.0f;
    default: return value(0.0f, 0.1f);
    }
}

struct Case {
    u64 node, inst, def, ctx, records_def, alloc_object;
    float step;
    u32 no_time;
    u64 seed;
    std::vector<std::pair<u64, std::size_t>> watched;  // memory compared after both runs
};

Case make_case() {
    g_blocks.clear();
    g_pointers.clear();
    Case c{};
    c.seed = g_rng.next();
    for (int i = 0; i < 4; ++i) g_pointers.push_back(block(0x40));
    // The instance: its root node and its "asked to end" bit (+0x88 bit 3).
    c.inst = block(0x98);
    // The instance's definition: +8 the object whose vtable +0x70 gives the items' allocator.
    c.alloc_object = block(0x20);
    const u64 vt = block(0x80);
    store<u64>(vt + 0x70, reinterpret_cast<u64>(&stub_allocator));
    store<u64>(c.alloc_object, vt);
    c.def = block(0x40);
    store<u64>(c.def + 8, c.alloc_object);
    // A staged node's records: +0 the count (now and then above 2^32 with a small low word), +8 the index table, +0x10 the records.
    const int records = g_rng.range(1, 6);
    c.records_def = block(0x40);
    const u64 index = block(4 * 8), recs = block(0xa0 * (records + 2));  // the index table may point one past the count
    for (int i = 0; i < 8; ++i) store<u32>(index + 4 * i, static_cast<u32>(g_rng.range(0, records + 1)));  // now and then past the count
    for (int i = 0; i < records + 2; ++i) {
        const u64 r = recs + 0xa0 * i;
        u32 rf = g_rng.u32() & ~(1u | 2u | 4u | 0x100u);
        if (g_rng.chance(0.5)) rf |= 1;
        if (g_rng.chance(0.5)) rf |= 2;
        if (g_rng.chance(0.5)) rf |= 4;
        if (g_rng.chance(0.5)) rf |= 0x100;
        store<u32>(r + 4, rf);
        store<float>(r + 8, value(0, 4));
        store<u64>(r + 0x38, g_rng.chance(0.5) ? g_pointers[0] : 0);
        store<u64>(r + 0x58, g_rng.chance(0.8) ? g_pointers[1] : 0);
        store<u64>(r + 0x78, g_pointers[2]);
        store<u64>(r + 0x88, g_pointers[3]);
    }
    u64 count = static_cast<u64>(records);
    if (g_rng.chance(0.2)) count |= 1ull << 32;  // a 32-bit compare would see the low word alone
    if (g_rng.chance(0.2)) count = static_cast<u64>(g_rng.range(0, 7)) << 32;
    store<u64>(c.records_def, count);
    store<u64>(c.records_def + 8, index);
    store<u64>(c.records_def + 0x10, recs);
    // The node.
    c.node = block(0x100);
    const bool staged = g_rng.chance(0.6);
    u32 flags = g_rng.u32() & ~0x7fu;  // high bits as they come
    if (g_rng.chance(0.9)) flags |= 1;
    if (g_rng.chance(0.3)) flags |= 2;
    if (staged) flags |= 8;
    if (g_rng.chance(0.1)) flags |= 4;
    if (g_rng.chance(0.5)) flags |= 0x10;
    if (g_rng.chance(0.5)) flags |= 0x20;
    if (g_rng.chance(0.5)) flags |= 0x40;
    store<u32>(c.node + 0x78, flags);
    store<u64>(c.node + 0x58, c.inst);
    const bool root = g_rng.chance(0.5);
    store<u64>(c.node + 0x60, root ? g_pointers[0] : g_pointers[1]);
    store<u64>(c.inst + 0x78, g_pointers[0]);
    store<u8>(c.inst + 0x88, static_cast<u8>(g_rng.u32()));
    store<u64>(c.node + 0x70, g_rng.chance(0.5) ? g_pointers[2] : 0);
    store<float>(c.node + 0x7c, value(0, 8));
    store<u64>(c.node + 0xc8, block(0x28));
    if (staged) {
        store<u64>(c.node + 0xd0, c.records_def);
        store<u32>(c.node + 0xd8, static_cast<u32>(g_rng.range(0, 7)));
        store<u32>(c.node + 0xdc, g_rng.chance(0.08) ? static_cast<u32>(g_rng.range(8, 12)) : static_cast<u32>(g_rng.range(0, 7)));
    } else {
        store<u64>(c.node + 0xd0, g_pointers[3]);
        // a plain node's +0xd8: its switch object, whose +0x11 says whether it lets an ending root go
        if (g_rng.chance(0.25)) {
            store<u64>(c.node + 0xd8, 0);
        } else {
            const u64 sw = block(0x20);
            store<u8>(sw + 0x11, g_rng.chance(0.5) ? 1 : 0);
            store<u64>(c.node + 0xd8, sw);
        }
    }
    store<float>(c.node + 0xe0, value(0, 4));
    store<float>(c.node + 0xf0, value(0, 8));
    c.ctx = g_rng.chance(0.2) ? 0 : block(0x40);
    c.step = g_rng.specials && g_rng.chance(0.2) ? value(0, 0.1f) : step_value();
    // Now and then the node's time and the step both NaNs of different payloads: the add's operand order shows.
    if (g_rng.chance(0.03)) {
        store<u32>(c.node + 0x7c, 0x7fc00001u);
        u32 nan = 0xffc00002u;
        std::memcpy(&c.step, &nan, 4);
    }
    const float nt = value(-2, 2);
    std::memcpy(&c.no_time, &nt, 4);
    c.watched = {{c.node, 0x100}, {c.inst, 0x98}, {c.def, 0x40}, {c.records_def, 0x18}, {recs, static_cast<std::size_t>(0xa0 * (records + 2))}};
    if (c.ctx) c.watched.push_back({c.ctx, 0x40});
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

using NodeUpdate = GUEST_ABI u8 (*)(u64 node, float* step, u64 definition, u64 context);

struct Run {
    u8 answer;
    std::vector<Call> calls;
    Snapshot after;
    u32 step_after;
};
Run run(NodeUpdate f, const Case& c) {
    store<u32>(reinterpret_cast<u64>(eboot_kit::at(kNoTime)), c.no_time);
    g_answers.seed(c.seed);
    g_calls.clear();
    g_node = c.node;
    float step = c.step;
    Run r;
    r.answer = f(c.node, &step, c.def, c.ctx);
    r.calls = g_calls;
    r.after = snap(c);
    r.step_after = bits(step);
    g_node = 0;
    return r;
}

// How often the game's runs reached each stub and each state, over the unmutated checks: every path the
// update has must be in the corpus.
long g_reached[17] = {}, g_states[13] = {}, g_dead_answers = 0;

int check(int cases) {
    const auto game = eboot_kit::fn<NodeUpdate>(0x2c4df80);
    int bad = 0;
    for (int k = 0; k < cases; ++k) {
        const Case c = make_case();
        const Snapshot before = snap(c);
        const u32 flags = load<u32>(c.node + 0x78), state = load<u32>(c.node + 0xdc);
        const Run g = run(game, c);
        restore(c, before);
        if (!g_quiet) {
            for (const Call& x : g.calls) ++g_reached[x.what];
            if (flags & 8) ++g_states[state > 12 ? 12 : state];
            g_dead_answers += g.answer == 0;
        }
        const Run o = run(&sfx_node_update, c);
        restore(c, before);
        const char* why = g.answer != o.answer         ? "answer"
                          : !(g.calls == o.calls)      ? "calls"
                          : g.after != o.after         ? "memory"
                          : g.step_after != o.step_after ? "the step"
                                                       : nullptr;
        if (why && ++bad <= 6 && !g_quiet) {
            std::printf("node case %d (flags 0x%x, state %u, step 0x%08x, context %s): %s differs", k, flags, state, bits(c.step),
                        c.ctx ? "yes" : "no", why);
            if (!(g.calls == o.calls)) {
                std::size_t i = 0;
                while (i < g.calls.size() && i < o.calls.size() && g.calls[i] == o.calls[i]) ++i;
                std::printf(" - at call %zu: the game's %d, ours %d (of %zu and %zu)", i, i < g.calls.size() ? g.calls[i].what : -1,
                            i < o.calls.size() ? o.calls[i].what : -1, g.calls.size(), o.calls.size());
            }
            std::printf("\n");
        }
    }
    return bad;
}


// --- the smooth steps' split ------------------------------------------------------

constexpr float kThirtieth = 1.0f / 30.0f;  // the game's 1/30, 0x3d088889

float addss(float x, float y) {
    __asm__("addss %1, %0" : "+x"(x) : "x"(y));
    return x;
}
float as_float(u32 u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// A time a container call sees: moved on by the offset, unless the offset is zero or the time is the "no time" value.
u32 shifted(u32 base, float offset, u32 none) { return offset == 0.0f || base == none ? base : bits(addss(as_float(base), offset)); }

// One node, two ways: the game's function once a whole 1/30 s step for `ticks` steps (a 30 fps world), and
// the split update every frame of a world at `fps` (0: frames from 1/240 s to 1/20 s at random) through the
// effects' schedule until as many whole steps have passed. On each whole step: the same answer, the same
// counted calls in order with the node as each saw it, the same memory - the counted parts are the game's.
// On each frame between: no counted call, nothing changed, the node kept. On every frame: each container
// call given the frame's step, the node's time put the part of a step past its whole step the frame began
// (a plain node's definition time too, and by the part it ends for its second call), nothing moved for a
// zero offset or the "no time" value. At 30 every frame is a whole step and the calls are the game's own.
int check_split(int cases, double fps) {
    const auto game = eboot_kit::fn<NodeUpdate>(0x2c4df80);
    int bad = 0;
    g_split = true;
    for (int k = 0; k < cases; ++k) {
        const Case c = make_case();
        const u32 none = c.no_time;
        if (g_rng.chance(0.1)) store<u32>(c.node + 0x7c, none);  // a node whose time is "no time"
        const float scale = g_rng.chance(0.5) ? 1.0f : g_rng.uniform(0.25f, 2.0f);
        const int ticks = g_rng.range(1, 5);
        const Snapshot before = snap(c);
        store<u32>(reinterpret_cast<u64>(eboot_kit::at(kNoTime)), none);
        g_node = c.node;
        // The 30 fps world.
        std::vector<u8> a_answer;
        std::vector<std::vector<Call>> a_calls;
        std::vector<std::vector<Seen>> a_seen;
        std::vector<Snapshot> a_after;
        for (int t = 0; t < ticks; ++t) {
            g_answers.seed(c.seed + 0x9e3779b97f4a7c15ull * static_cast<u64>(t + 1));
            g_calls.clear();
            g_seen.clear();
            float step = kThirtieth * scale;
            a_answer.push_back(game(c.node, &step, c.def, c.ctx));
            a_calls.push_back(g_calls);
            a_seen.push_back(g_seen);
            a_after.push_back(snap(c));
            if (!a_answer.back()) break;
        }
        restore(c, before);
        // The split world.
        std::mt19937_64 jitter(c.seed);
        double carry = 0.0;
        std::size_t t = 0;
        const char* why = nullptr;
        for (int frame = 0; t < a_answer.size() && frame < 4000 && !why; ++frame) {
            const double seconds = fps > 0.0 ? 1.0 / fps : std::uniform_real_distribution<double>(1.0 / 240, 1.0 / 20)(jitter);
            const sfx_schedule::Frame f = sfx_schedule::next(carry, seconds, 30);
            sfx_node::Steps st{f.whole ? f.seconds * scale : 0.0f, f.frame * scale, f.from * scale, f.into * scale, f.whole > 0};
            st.smooth = fps != 30.0 && (jitter() & 3) != 0;  // most calls published, some as a split without it; at 30 the game's own
            const u32 t_entry = load<u32>(c.node + 0x7c), e_entry = load<u32>(c.node + 0xe0);
            const u32 flags = load<u32>(c.node + 0x78), state = load<u32>(c.node + 0xdc);
            const Snapshot pre = snap(c);
            if (st.tick) g_answers.seed(c.seed + 0x9e3779b97f4a7c15ull * static_cast<u64>(t + 1));
            g_calls.clear();
            g_seen.clear();
            const u8 r = sfx_node::update(c.node, st, c.def, c.ctx);
            // The container calls this frame should have made, and what each should have seen.
            std::vector<Seen> want;
            const u64 pn = st.smooth ? c.node : 0;
            if (!(flags & 4) && (flags & 1)) {
                const u32 frame_step = bits(st.frame);
                if (!(flags & 8)) {
                    bool picked = false;
                    for (const Call& x : g_calls) picked |= x.what == 5;
                    const u32 e1 = picked ? none : e_entry;
                    const u32 e2 = st.tick ? bits(addss(as_float(e1), st.counted)) : e1;
                    want.push_back({frame_step, shifted(t_entry, st.from, none), shifted(e1, st.from, none), c.ctx, pn, pn ? t_entry : 0u,
                                    pn ? st.tick : false});
                    want.push_back({frame_step, shifted(t_entry, st.from, none), shifted(e2, st.into, none), r ? c.ctx : 0, pn, pn ? t_entry : 0u,
                                    pn ? st.tick : false});
                } else {
                    const u32 base = st.tick && state == 2 ? none : t_entry;
                    // a staged node's +0xe0 is its emission count: whatever it holds at the call, never moved
                    want.push_back({frame_step, shifted(base, st.from, none), g_seen.empty() ? 0u : g_seen.back().e, r ? c.ctx : 0, pn,
                                    pn ? base : 0u, pn ? st.tick : false});
                }
            }
            if (!(g_seen == want)) why = "the container calls (their step, the times they saw, their context, the call published)";
            else if (sfx_node::smooth_call().node) why = "a container call still published after the update";
            else if (st.tick) {
                if (r != a_answer[t]) why = "a whole step's answer";
                else if (!(g_calls == a_calls[t])) why = "a whole step's counted calls";
                else if (snap(c) != a_after[t]) why = "the node after a whole step";
                else if (fps == 30.0 && !(g_seen == a_seen[t])) why = "at 30, the container calls against the game's";
                if (!r) break;
                ++t;
            } else {
                if (r != 1) why = "a frame between whole steps' answer";
                else if (!g_calls.empty()) why = "a counted call between whole steps";
                else if (snap(c) != pre) why = "the node changed between whole steps";
            }
        }
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("split case %d (%s, flags 0x%x, state %u, scale %g, at %g fps): %s differs\n", k,
                        (load<u32>(before[0].data(), 0x78) & 8) ? "staged" : "plain", load<u32>(before[0].data(), 0x78), load<u32>(before[0].data(), 0xdc),
                        scale, fps, why);
    }
    g_split = false;
    g_node = 0;
    return bad;
}
}  // namespace

int main() {
    eboot_kit::load("decomp_sfx_node_update");
    if (std::memcmp(eboot_kit::at(0x2c4df80), kNodeEntry, sizeof kNodeEntry) != 0) {
        std::printf("decomp_sfx_node_update: sub_2c4df80's entry is not what the leaf expects\n");
        return 1;
    }
    eboot_kit::stub(kContainerUpdate, reinterpret_cast<const void*>(&stub_container));
    eboot_kit::stub(kStateOf, reinterpret_cast<const void*>(&stub_state_of));
    eboot_kit::stub(kStateHolds, reinterpret_cast<const void*>(&stub_state_holds));
    eboot_kit::stub(kPick, reinterpret_cast<const void*>(&stub_pick));
    eboot_kit::stub(kApply, reinterpret_cast<const void*>(&stub_apply));
    eboot_kit::stub(kBind, reinterpret_cast<const void*>(&stub_bind));
    eboot_kit::stub(kSetUp, reinterpret_cast<const void*>(&stub_set_up));
    eboot_kit::stub(kEmit, reinterpret_cast<const void*>(&stub_emit));
    eboot_kit::stub(kLinger, reinterpret_cast<const void*>(&stub_linger));
    eboot_kit::stub(kEndModules, reinterpret_cast<const void*>(&stub_end_modules));
    eboot_kit::stub(kModulesDone, reinterpret_cast<const void*>(&stub_modules_done));
    eboot_kit::stub(kChildrenGone, reinterpret_cast<const void*>(&stub_children_gone));
    eboot_kit::stub(kItemData, reinterpret_cast<const void*>(&stub_item_data));
    eboot_kit::stub(kOneItem, reinterpret_cast<const void*>(&stub_one_item));
    eboot_kit::stub(kThreeItems, reinterpret_cast<const void*>(&stub_three_items));

    constexpr int kCases = 40000;
    int bad = 0;
    g_rng.specials = false;
    bad += check(kCases);
    g_rng.specials = true;
    bad += check(kCases);
    // The host's floating-point environment too (MXCSR without flush-to-zero):
    // the game's code and ours must agree under either.
    eboot_kit::host_fp();
    bad += check(kCases / 4);
    eboot_kit::guest_fp();
    // Every path reached: each stub, each state 0-7 and above, and both answers.
    {
        static const char* names[17] = {"",        "container", "state of", "state holds", "pick",  "apply",        "bind",     "set-up", "emit",
                                        "linger", "end modules", "modules done", "children", "item data", "one item", "three items", "allocator"};
        int unreached = 0;
        for (int w = 1; w <= 16; ++w)
            if (!g_reached[w]) {
                std::printf("decomp_sfx_node_update: no case reached %s\n", names[w]);
                ++unreached;
            }
        for (int st = 0; st <= 8; ++st)
            if (!g_states[st]) {
                std::printf("decomp_sfx_node_update: no staged node in state %d%s\n", st, st == 8 ? " or above" : "");
                ++unreached;
            }
        if (!g_dead_answers) {
            std::printf("decomp_sfx_node_update: no node answered dead\n");
            ++unreached;
        }
        if (unreached) return 1;
    }
    // The smooth steps' split at the rates a player picks, a jittered rate and 30 (the game's own calls).
    constexpr int kSplitCases = 6000;
    int split_bad = 0;
    g_rng.specials = false;
    for (const double fps : {30.0, 60.0, 120.0, 144.0, 0.0}) split_bad += check_split(kSplitCases, fps);
    g_rng.specials = true;
    split_bad += check_split(kSplitCases, 0.0);
    bad += split_bad;
    // Each mutant must make some case differ: the game's sequence (1-10) and the split (11-19).
    int missed = 0;
    g_quiet = true;
    for (int m = 1; m <= 19; ++m) {
        g_sfx_node_mutant = m;
        g_rng.specials = true;
        const int found = m <= 10 ? check(20000) : check_split(3000, 120.0) + check_split(3000, 0.0);
        if (!found) {
            std::printf("decomp_sfx_node_update: mutant %d not caught\n", m);
            ++missed;
        }
    }
    g_sfx_node_mutant = 0;
    if (bad || missed) {
        std::printf("decomp_sfx_node_update: %d of %d cases differ, %d mutants missed\n", bad, 2 * kCases + kCases / 4, missed);
        return 1;
    }
    std::printf("decomp_sfx_node_update: %d nodes as the game's (%d with special values, %d under the host's MXCSR); %d split at 30, 60, 120, "
                "144 and jittered as the game's whole steps, their container calls published for the light; 19 mutants caught\n",
                2 * kCases + kCases / 4, kCases, kCases / 4, 6 * kSplitCases);
    return 0;
}
