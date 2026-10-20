// The effects' step gate (decomp/sfx/step_gate.cpp) against the game's own
// code (tests/decomp/eboot_kit.h), on synthetic effects:
//   - a node on a held frame (node_held) against the game's node update,
//     0x2c4df80: the same container calls, step adds and answer for a node
//     whose update changes nothing else (a staged node staying in state 3, a
//     plain one whose conditions pick no new definition, an inactive one),
//     for a dead one the game's answer of 0 (free it) against ours of 1 (kept
//     for the next whole step), neither touching it, and for a node in a
//     state whose update would move it (2, 6, 7, 5, 0) the same container
//     call with its state, emission and definition fields left as they were;
//   - the instance update, 0x2c4ff30: ours on a frame not held against the
//     game's (the trampoline's place); on a held frame against the game's
//     with its node update node_held and its controllers recorded - the same
//     nodes, frees and memory, and none of ours calling a controller; the
//     game's still frame (SprjSfx+0x8c) left to the game;
//   - the FFX light module's update, 0x131e660: ours on a held frame against
//     the game's run with the rate +0x80 at 0 and the mode +0x38 at 1, both
//     put back - every byte of the module and of its GX light, every call -
//     the phase +0x30 and +0x110 unchanged; not held, or a pre-roll's call,
//     the game's as it is;
//   - the wobble motions' updates, 0x2e89d10, 0x2e8e430 and 0x2e896f0: not
//     held (or on the game's own still frame) ours against the game's - the
//     same calls and every byte of the motion, its node, the parent and the
//     FX RNG; held, one call of the base motion with the same arguments and
//     nothing else touched - no count, no re-roll, no draw, no lerp;
//   - a GPU particle system's record taking an item, 0x2cbcf50, with the game's
//     own GPU particle type 0x2713 update (0x12e81b0) under test curves: not a
//     frame between whole steps (whatever the gate), or another type, or an
//     item with a step, or a record whose elapsed time is already 0, or the
//     game's own still frame, ours against the game's - every byte of the
//     record and its type; between whole steps, the gate on or off, the game's
//     call with the record's previous time +0x24 and the type's sizes +0x1c /
//     +0x20 and colours +0x50..+0x7f put back - every other byte the game's,
//     and the elapsed time the type's draw then reads the last whole step's;
// and each of the source's mutants caught. Skips without the 1.09 eboot.
#define DECOMP_SFX_STEP_GATE_MUTANTS
int g_sfx_gate_mutant = 0;
#include "decomp/sfx/step_gate.cpp"

#include "eboot_kit.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <xmmintrin.h>

namespace {

eboot_kit::Rng g_rng(0x57e9ca7e);
bool g_quiet = false;  // the mutant runs: only whether they differ

// --- what the stubs saw ------------------------------------------------------

struct Call {
    int what;  // 1 container, 2 controller, 3 node release, 4 node clear, 5 pool free, 6 light enable, 7 light params
    u64 a, b, c, d;
    u32 step;
};
std::vector<Call> g_calls;
u32 bits(float f) {
    u32 u;
    std::memcpy(&u, &f, 4);
    return u;
}
void rec(int what, u64 a, u64 b, u64 c, u64 d, float step = 0) { g_calls.push_back({what, a, b, c, d, bits(step)}); }
bool same_calls(const std::vector<Call>& x, const std::vector<Call>& y) {
    if (x.size() != y.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (x[i].what != y[i].what || x[i].a != y[i].a || x[i].b != y[i].b || x[i].c != y[i].c || x[i].d != y[i].d || x[i].step != y[i].step)
            return false;
    return true;
}
std::size_t count(const std::vector<Call>& v, int what) {
    std::size_t n = 0;
    for (const Call& c : v) n += c.what == what;
    return n;
}

GUEST_ABI void stub_container(u64 container, float* step, u64 definition, u64 context) { rec(1, container, definition, context, 0, *step); }
GUEST_ABI u64 stub_controller(u64 controller, u64 instance, float step) {
    rec(2, controller, instance, 0, 0, step);
    store<float>(controller + 8, load<float>(controller + 8) + step);
    return 0;
}
GUEST_ABI void stub_release(u64 node) { rec(3, node, 0, 0, 0); }
GUEST_ABI void stub_clear(u64 node) { rec(4, node, 0, 0, 0); }
GUEST_ABI void stub_pool_free(u64 pool, u64 p, u64 size, u64 alignment) { rec(5, pool, p, size, alignment); }
GUEST_ABI void stub_light_enable(u64 a, u64 light, u64 on) { rec(6, a, light, on & 0xff, 0); }
GUEST_ABI void stub_light_params(u64 light, u64 block) { rec(7, light, block, 0, 0); }
// The node update's further reach, for the states it moves: each answers as
// a plain value and is seen by its call.
GUEST_ABI u64 stub_state1(u64 node, u64, u64) { rec(8, node, 1, 0, 0); return 3; }
GUEST_ABI u64 stub_state4(u64 node, u64, u64) { rec(8, node, 4, 0, 0); return 5; }
GUEST_ABI u8 stub_state0_init(u64 node, u64, u64, u64) { rec(8, node, 0, 0, 0); return 1; }
GUEST_ABI void stub_quiet(u64 a) { rec(9, a, 0, 0, 0); }
GUEST_ABI u8 stub_finished(u64 container) { rec(9, container, 1, 0, 0); return static_cast<u8>(load<u64>(container + 0x40) & 1); }
GUEST_ABI u64 stub_transitions(u64, u64, u64) { return 0; }

// Curve objects the light module reads at the node's age: +0x58 of each one's
// vtable, (curve, node, &age) -> a vector made from the curve's own numbers.
GUEST_ABI __m128 curve_at(u64 curve, u64, const float* age) {
    const float k = load<float>(curve + 8), c = load<float>(curve + 12);
    return _mm_set_ps(c, k, c - *age, *age * k + c);
}

// --- memory -------------------------------------------------------------------

std::vector<std::vector<u8>> g_blocks;
u64 block(std::size_t n) {
    g_blocks.emplace_back(n + 64, 0);
    return (reinterpret_cast<u64>(g_blocks.back().data()) + 63) & ~63ull;
}
float value(float lo, float hi) { return g_rng.value(lo, hi); }

// A definition 0x2c4df80 reads: +0 the record count, +8 the index table,
// +0x10 the 0xa0-byte records (+4 flags, +8 the start time).
u64 definition(int records) {
    const u64 d = block(0x40);
    const u64 index = block(4 * 8), recs = block(0xa0 * records);
    for (int i = 0; i < 8; ++i) store<u32>(index + 4 * i, static_cast<u32>(i % records));
    for (int i = 0; i < records; ++i) {
        store<u32>(recs + 0xa0 * i + 4, static_cast<u32>(g_rng.range(0, 3)));
        store<float>(recs + 0xa0 * i + 8, value(0, 4));
    }
    store<u64>(d, static_cast<u64>(records));
    store<u64>(d + 8, index);
    store<u64>(d + 0x10, recs);
    return d;
}
// A controller: +4 its state, +0x10 its definition (+8 the state count).
u64 controller(u32 state, u32 states) {
    const u64 c = block(0x28), d = block(0x20);
    store<u64>(d + 8, states);
    store<u32>(c + 4, state);
    store<float>(c + 8, value(0, 3));
    store<u64>(c + 0x10, d);
    return c;
}
u64 node(u64 instance, u32 flags, u32 state) {
    const u64 n = block(0x100);
    for (int k = 0; k < 0x100; k += 4) store<u32>(n + k, g_rng.u32());
    store<u64>(n + 0x58, instance);
    store<u64>(n + 0x60, 0);
    store<u64>(n + 0x68, 0);
    store<u64>(n + 0x70, 0);
    store<u32>(n + 0x78, flags);
    store<float>(n + 0x7c, value(0, 8));
    store<u64>(n + 0xa0, 0);  // the container's module +0x20: none (finished)
    store<u64>(n + 0xc0, g_rng.u32() & 1);
    store<u32>(n + 0xd8, 0);
    store<u32>(n + 0xdc, state);
    store<float>(n + 0xe0, value(0, 4));
    store<float>(n + 0xf0, value(0, 8));
    return n;
}

// --- the node -------------------------------------------------------------

using NodeUpdate = GUEST_ABI u8 (*)(u64 node, float* step, u64 definition, u64 context);

int check_nodes(int cases) {
    const auto game = eboot_kit::fn<NodeUpdate>(0x2c4df80);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const u64 inst = block(0x98);
        store<u64>(inst + 0x78, block(8));  // not this node: no root-node stop
        const u64 def = definition(4);
        const int kind = g_rng.range(0, 4);  // 0 staged in state 3, 1 plain, 2 dead, 3 inactive, 4 staged in a moving state
        u32 flags = 1 | (g_rng.u32() & 0x22);  // active, maybe stopping (2), maybe 0x20
        u32 state = 3;
        if (kind == 0 || kind == 4) flags |= 8;
        if (kind == 2) flags |= 4;
        if (kind == 3) flags &= ~1u;
        if (kind == 4) {
            static const u32 moving[] = {0, 2, 5, 6, 7};
            state = moving[g_rng.range(0, 4)];
        }
        const u64 n = node(inst, flags, state);
        // its controller says state 0, the node's own index: state 3 stays
        store<u64>(n + 0xc8, controller(0, 4));
        store<u64>(n + 0xd0, def);
        if (kind == 0) store<u32>(n + 0x78, flags & ~2u);  // not stopping: state 3 stays
        const u64 ctx = block(0x40);
        const float step = g_rng.chance(0.7) ? 0.0f : value(0, 0.1f);
        std::vector<u8> before(0x100), after_game(0x100), after_ours(0x100);
        std::memcpy(before.data(), reinterpret_cast<void*>(n), 0x100);
        float s1 = step, s2 = step;
        g_calls.clear();
        const u8 r_game = game(n, &s1, def, ctx);
        const std::vector<Call> calls_game = g_calls;
        std::memcpy(after_game.data(), reinterpret_cast<void*>(n), 0x100);
        std::memcpy(reinterpret_cast<void*>(n), before.data(), 0x100);
        g_calls.clear();
        const u8 r_ours = node_held(n, &s2, def, ctx);
        const std::vector<Call> calls_ours = g_calls;
        std::memcpy(after_ours.data(), reinterpret_cast<void*>(n), 0x100);
        bool ok;
        std::string why;
        if (kind == 2) {
            // dead: the game's says free it (0); ours keeps it for the next whole step (1); neither calls or writes
            ok = r_game == 0 && calls_game.empty() && after_game == before && r_ours == 1 && calls_ours.empty() && after_ours == before;
            if (!ok) why = r_ours != 1 ? "a dead node's answer (freed on a held frame)" : "a dead node";
        } else if (kind != 4) {
            ok = r_game == r_ours && same_calls(calls_game, calls_ours) && after_game == after_ours;
            if (!ok) why = r_game != r_ours ? "answer" : !same_calls(calls_game, calls_ours) ? "calls" : "memory";
        } else {
            // the game's moves the node on; ours runs its container once and leaves the rest
            std::vector<u8> want = before;
            const float t = load<float>(n + 0x7c);
            (void)t;
            float t7c;
            std::memcpy(&t7c, before.data() + 0x7c, 4);
            t7c += step;
            std::memcpy(want.data() + 0x7c, &t7c, 4);
            ok = r_ours == 1 && calls_ours.size() == 1 && calls_ours[0].what == 1 && calls_ours[0].a == n + 0x80 && calls_ours[0].c == ctx &&
                 calls_ours[0].step == bits(step) && after_ours == want;
            if (!ok) why = "a moving state's node";
        }
        if (!ok && ++bad <= 6 && !g_quiet)
            std::printf("node case %d (kind %d, flags 0x%x, state %u, step %g): %s differs\n", c, kind, flags, state, step, why.c_str());
    }
    return bad;
}

// --- the instance ------------------------------------------------------------

GUEST_ABI u8 node_held_as_game(u64 node, float* step, u64 definition, u64 context) { return node_held(node, step, definition, context); }

struct Instance {
    u64 inst, ctx, sfx;
    std::vector<u64> nodes, ctls;
};
Instance make_instance() {
    Instance s;
    s.inst = block(0x98);
    for (int k = 0; k < 0x98; k += 4) store<u32>(s.inst + k, g_rng.u32());
    u32 flags = g_rng.u32() & 0xff;
    if (g_rng.chance(0.8)) flags = (flags | 1) & ~4u;
    store<u32>(s.inst + 0x88, flags);
    store<float>(s.inst + 8, g_rng.chance(0.5) ? 1.0f : value(0, 2));
    store<u64>(s.inst + 0x38, g_rng.chance(0.9) ? definition(2) : 0);
    // controllers
    u64 prev = 0;
    const int nc = g_rng.range(0, 3);
    for (int i = 0; i < nc; ++i) {
        const u64 c = controller(0, 2);
        store<u64>(c + 0x20, 0);
        if (prev) store<u64>(prev + 0x20, c);
        else store<u64>(s.inst + 0x80, c);
        prev = c;
        s.ctls.push_back(c);
    }
    if (!nc) store<u64>(s.inst + 0x80, 0);
    // a node tree: a root, its children, a grandchild
    const int nn = g_rng.range(0, 5);
    for (int i = 0; i < nn; ++i) {
        u32 nf = 1 | (g_rng.chance(0.5) ? 8 : 0);
        if (g_rng.chance(0.2)) nf |= 4;
        if (g_rng.chance(0.1)) nf &= ~1u;
        const u64 n = node(s.inst, nf, 3);
        s.nodes.push_back(n);
    }
    // links: node 0 the root; 1.. its children in a sibling chain; the last a grandchild of 1
    for (std::size_t i = 0; i < s.nodes.size(); ++i) {
        store<u64>(s.nodes[i] + 0x60, 0);
        store<u64>(s.nodes[i] + 0x68, 0);
        store<u64>(s.nodes[i] + 0x70, 0);
    }
    if (!s.nodes.empty()) {
        store<u64>(s.inst + 0x78, s.nodes[0]);
        for (std::size_t i = 1; i < s.nodes.size(); ++i) {
            const bool grand = i == s.nodes.size() - 1 && i >= 3;
            const u64 parent = grand ? s.nodes[1] : s.nodes[0];
            store<u64>(s.nodes[i] + 0x60, parent);
            if (!load<u64>(parent + 0x70)) {
                store<u64>(parent + 0x70, s.nodes[i]);
            } else {
                u64 sib = load<u64>(parent + 0x70);
                while (load<u64>(sib + 0x68)) sib = load<u64>(sib + 0x68);
                store<u64>(sib + 0x68, s.nodes[i]);
            }
        }
    } else {
        store<u64>(s.inst + 0x78, 0);
    }
    s.ctx = block(0x40);
    return s;
}

struct Snapshot {
    std::vector<std::vector<u8>> mem;
};
Snapshot snap(const Instance& s) {
    Snapshot out;
    auto add = [&](u64 p, std::size_t n) {
        out.mem.emplace_back(n);
        std::memcpy(out.mem.back().data(), reinterpret_cast<void*>(p), n);
    };
    add(s.inst, 0x98);
    for (u64 c : s.ctls) add(c, 0x28);
    for (u64 n : s.nodes) add(n, 0x100);
    return out;
}
void restore(const Instance& s, const Snapshot& v) {
    std::size_t i = 0;
    auto put = [&](u64 p, std::size_t n) { std::memcpy(reinterpret_cast<void*>(p), v.mem[i++].data(), n); };
    put(s.inst, 0x98);
    for (u64 c : s.ctls) put(c, 0x28);
    for (u64 n : s.nodes) put(n, 0x100);
}
bool same(const Snapshot& a, const Snapshot& b) { return a.mem == b.mem; }

int check_instances(int cases, u64 sfx) {
    const auto game = eboot_kit::fn<InstanceUpdate>(0x2c4ff30);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const Instance s = make_instance();
        const float step = g_rng.chance(0.6) ? 0.0f : value(0, 0.1f);
        const bool still = g_rng.chance(0.15);
        store<u32>(sfx + 0x8c, still ? 1u : 0u);
        const Snapshot before = snap(s);
        // the frame not held: ours is the game's
        decomp_sfx_gate_frame(false);
        g_calls.clear();
        const u8 r1 = game(s.inst, s.ctx, step);
        const std::vector<Call> c1 = g_calls;
        const Snapshot a1 = snap(s);
        restore(s, before);
        g_calls.clear();
        const u8 r2 = sfx_instance_update(s.inst, s.ctx, step);
        const std::vector<Call> c2 = g_calls;
        const Snapshot a2 = snap(s);
        restore(s, before);
        bool ok = r1 == r2 && same_calls(c1, c2) && same(a1, a2);
        const char* why = ok ? "" : "the frame not held";
        // held: against the game's with its controllers left out of the record
        if (ok) {
            decomp_sfx_gate_frame(true);
            g_calls.clear();
            const u8 r3 = game(s.inst, s.ctx, step);
            std::vector<Call> c3;
            for (const Call& k : g_calls)
                if (k.what != 2 || still) c3.push_back(k);
            Snapshot a3 = snap(s);
            restore(s, before);
            if (!still) {  // the game's controllers moved their own time: put it back for the compare
                for (std::size_t i = 0; i < s.ctls.size(); ++i) std::memcpy(a3.mem[1 + i].data(), before.mem[1 + i].data(), 0x28);
            }
            g_calls.clear();
            const u8 r4 = sfx_instance_update(s.inst, s.ctx, step);
            const std::vector<Call> c4 = g_calls;
            const Snapshot a4 = snap(s);
            restore(s, before);
            decomp_sfx_gate_frame(false);
            ok = r3 == r4 && same_calls(c3, c4) && same(a3, a4) && (still || count(c4, 2) == 0);
            if (!ok) why = still ? "the game's still frame" : "the held frame";
        }
        if (!ok && ++bad <= 6 && !g_quiet)
            std::printf("instance case %d (flags 0x%x, %zu nodes, %zu controllers, step %g%s): %s differs\n", c, load<u32>(s.inst + 0x88),
                        s.nodes.size(), s.ctls.size(), step, still ? ", still frame" : "", why);
    }
    return bad;
}

// --- the light ---------------------------------------------------------------

struct Light {
    u64 module, node, inst, light, block;
    std::size_t module_size = 0x120, light_size = 0x140;
};
u64 curve() {
    const u64 c = block(0x20), vt = block(0x60);
    store<u64>(vt + 0x58, reinterpret_cast<u64>(&curve_at));
    store<u64>(c, vt);
    store<float>(c + 8, value(-2, 2));
    store<float>(c + 12, value(-2, 2));
    return c;
}
Light make_light() {
    Light l;
    l.module = block(l.module_size);
    for (std::size_t k = 0; k < l.module_size; k += 4) store<float>(l.module + k, value(-3, 3));
    l.inst = block(0x98);
    l.node = block(0x100);
    for (int k = 0; k < 0x100; k += 4) store<float>(l.node + k, value(-50, 50));
    store<u64>(l.node + 0x58, l.inst);
    store<float>(l.node + 0x7c, value(0, 10));
    l.block = 0;
    if (g_rng.chance(0.5)) {
        l.block = block(0x220);
        for (int k = 0; k < 0x220; k += 4) store<float>(l.block + k, value(-2, 2));
        store<u64>(l.block + 0x1f0, 0);  // no visibility test
        store<u8>(l.block + 0x204, static_cast<u8>(g_rng.u32() & 1));
        store<u32>(l.block + 0x208, g_rng.u32());
        store<u8>(l.block + 0x20c, static_cast<u8>(g_rng.u32() & 1));
        store<u8>(l.block + 0x20d, static_cast<u8>(g_rng.u32() & 1));
    }
    store<u64>(l.inst + 0x40, l.block);
    l.light = block(l.light_size);
    for (std::size_t k = 0; k < l.light_size; k += 4) store<float>(l.light + k, value(-1, 1));
    store<u8>(l.light + 0xc4, g_rng.chance(0.8) ? 0 : 1);
    store<u8>(l.light + 0xc5, g_rng.chance(0.8) ? 0 : 1);
    store<u64>(l.module + 0x8, l.node);
    store<float>(l.module + 0x10, value(0, 5));
    store<u8>(l.module + 0x18, static_cast<u8>(g_rng.u32() & 1));
    store<u64>(l.module + 0x20, block(0x10));
    store<u64>(l.module + 0x28, l.light);
    store<float>(l.module + 0x30, value(-100, 100));
    store<u8>(l.module + 0x34, static_cast<u8>(g_rng.u32() & 1));
    store<u32>(l.module + 0x38, static_cast<u32>(g_rng.range(0, 2)));
    store<u32>(l.module + 0x3c, static_cast<u32>(g_rng.range(0, 2)));
    for (const u64 off : {0x40, 0x48, 0x50, 0x58, 0x60}) store<u64>(l.module + off, curve());
    store<u64>(l.module + 0x68, g_rng.chance(0.7) ? curve() : 0);
    store<u64>(l.module + 0x70, g_rng.chance(0.7) ? curve() : 0);
    store<s32>(l.module + 0x7c, g_rng.range(-1, 2));
    store<float>(l.module + 0x80, value(-2, 2));
    for (const u64 off : {0x88, 0x8c, 0x90}) store<float>(l.module + off, g_rng.chance(0.3) ? 0.0f : value(-1, 1));
    store<s32>(l.module + 0xa4, g_rng.range(-1, 1));
    store<s32>(l.module + 0xb4, g_rng.range(-1, 1));
    return l;
}
struct LightState {
    std::vector<u8> module, light;
};
LightState light_state(const Light& l) {
    LightState s;
    s.module.resize(l.module_size);
    s.light.resize(l.light_size);
    std::memcpy(s.module.data(), reinterpret_cast<void*>(l.module), l.module_size);
    std::memcpy(s.light.data(), reinterpret_cast<void*>(l.light), l.light_size);
    return s;
}
void light_restore(const Light& l, const LightState& s) {
    std::memcpy(reinterpret_cast<void*>(l.module), s.module.data(), l.module_size);
    std::memcpy(reinterpret_cast<void*>(l.light), s.light.data(), l.light_size);
}
bool same_bytes(const std::vector<u8>& a, const std::vector<u8>& b) {
    for (std::size_t k = 0; k + 4 <= a.size(); k += 4) {
        float x, y;
        std::memcpy(&x, a.data() + k, 4);
        std::memcpy(&y, b.data() + k, 4);
        if (!eboot_kit::same_float(x, y)) return false;
    }
    return true;
}

int check_lights(int cases, u64 sfx, u64 preroll_ctx) {
    const auto game = eboot_kit::fn<LightUpdate>(0x131e660);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const Light l = make_light();
        const u64 params = block(0x80);
        const bool pre = g_rng.chance(0.1);
        const u64 ctx = pre ? preroll_ctx : block(0x40);
        const bool still = g_rng.chance(0.1);
        store<u32>(sfx + 0x8c, still ? 1u : 0u);
        const LightState before = light_state(l);
        // the game's, as it is
        g_calls.clear();
        game(l.module, params, ctx);
        const std::vector<Call> c_plain = g_calls;
        const LightState plain = light_state(l);
        light_restore(l, before);
        // the game's with the rate at 0 and the mode at 1, both put back
        store<float>(l.module + 0x80, 0.0f);
        store<u32>(l.module + 0x38, 1u);
        g_calls.clear();
        game(l.module, params, ctx);
        const std::vector<Call> c_held = g_calls;
        LightState held_want = light_state(l);
        std::memcpy(held_want.module.data() + 0x80, before.module.data() + 0x80, 4);
        std::memcpy(held_want.module.data() + 0x38, before.module.data() + 0x38, 4);
        std::memcpy(held_want.module.data() + 0x30, before.module.data() + 0x30, 4);
        light_restore(l, before);
        for (const bool between : {false, true}) {
            decomp_sfx_gate_frame(between);
            g_calls.clear();
            sfx_light_update(l.module, params, ctx);
            const std::vector<Call> c_ours = g_calls;
            const LightState ours = light_state(l);
            light_restore(l, before);
            const bool held = between && !still && !pre;
            const LightState& want = held ? held_want : plain;
            const std::vector<Call>& want_calls = held ? c_held : c_plain;
            bool ok = same_calls(want_calls, c_ours) && same_bytes(want.module, ours.module) && same_bytes(want.light, ours.light);
            if (ok && held) {  // the phase and the blend as they were
                ok = std::memcmp(ours.module.data() + 0x30, before.module.data() + 0x30, 4) == 0 &&
                     std::memcmp(ours.module.data() + 0x110, before.module.data() + 0x110, 4) == 0;
            }
            if (!ok && ++bad <= 6 && !g_quiet) {
                int mo = -1, lo = -1;
                for (std::size_t k = 0; k + 4 <= want.module.size(); k += 4) {
                    float x, y;
                    std::memcpy(&x, want.module.data() + k, 4);
                    std::memcpy(&y, ours.module.data() + k, 4);
                    if (!eboot_kit::same_float(x, y)) { mo = static_cast<int>(k); break; }
                }
                for (std::size_t k = 0; k + 4 <= want.light.size(); k += 4) {
                    float x, y;
                    std::memcpy(&x, want.light.data() + k, 4);
                    std::memcpy(&y, ours.light.data() + k, 4);
                    if (!eboot_kit::same_float(x, y)) { lo = static_cast<int>(k); break; }
                }
                std::printf("light case %d (mode %u, %s%s%s): ours differs - calls %s (%zu/%zu), module +0x%x, light +0x%x\n", c,
                            load<u32>(l.module + 0x38), between ? "between" : "whole", still ? ", still frame" : "", pre ? ", pre-roll" : "",
                            same_calls(want_calls, c_ours) ? "same" : "differ", want_calls.size(), c_ours.size(), mo, lo);
            }
        }
        decomp_sfx_gate_frame(false);
    }
    return bad;
}

// --- the wobble motions --------------------------------------------------------

// The base motions (0x2e866f0, 0x2e8bfd0) seen by their call, and the FX RNG
// (0x2c6d580) as a state of the test's: what the motions draw comes from it.
GUEST_ABI void stub_base_a(u64 motion, u64 node, const float* step, u64 local) { rec(10, motion, node, local, 0, *step); }
GUEST_ABI void stub_base_b(u64 motion, u64 node, const float* step, u64 local) { rec(11, motion, node, local, 0, *step); }
alignas(16) u8 g_rng_state[0x20];
GUEST_ABI u64 stub_fx_rng() { return reinterpret_cast<u64>(g_rng_state); }

using MotionUpdate = GUEST_ABI void (*)(u64 motion, u64 node, const float* step, u64 local);

// kind 0: 0x2e89d10 (count +0x6c, period +0x68, curve +0x60); 1: 0x2e8e430
// (+0x7c, +0x78, +0x70); 2: 0x2e896f0 (the follow, then 0x2e89d10's work).
int check_wobbles(int cases, u64 sfx) {
    static const u64 kAddr[] = {kWobbleA, kWobbleB, kWobbleFollow};
    static const MotionFn kOurs[] = {&sfx_wobble_a, &sfx_wobble_b, &sfx_wobble_follow};
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const int kind = g_rng.range(0, 2);
        const u64 motion = block(0x100), n = block(0x100), parent = block(0x100), local = block(0x40);
        for (int k = 0; k < 0x100; k += 4) {
            store<float>(motion + k, value(-2, 2));
            store<float>(n + k, value(-50, 50));
            store<float>(parent + k, value(-50, 50));
        }
        for (int k = 0; k < 0x40; k += 4) store<float>(local + k, value(-2, 2));
        store<float>(n + 0x7c, value(0, 10));  // the node's time, which the curves read
        const u64 count = kind == 1 ? 0x7c : 0x6c, period = count - 4, curve_at_ = kind == 1 ? 0x70 : 0x60;
        const u32 per = static_cast<u32>(g_rng.range(1, 6));
        store<u32>(motion + period, per);
        store<u32>(motion + count, static_cast<u32>(g_rng.range(0, static_cast<int>(per))));  // often at the period's end
        store<u64>(motion + curve_at_, g_rng.chance(0.85) ? curve() : 0);
        if (kind == 2) {
            // the follow: the parent node (node+0x60) or none, the parent seen last (+0xa0), its rows kept at +0x70..+0x9f,
            // the factor's curve (+0xa8) or the default factor (+0xb0)
            store<u64>(n + 0x60, g_rng.chance(0.85) ? parent : 0);
            store<u64>(motion + 0xa0, g_rng.chance(0.7) ? parent : 0);
            if (g_rng.chance(0.3)) {
                for (int k = 0; k < 0x30; k += 4) store<u32>(motion + 0x70 + k, load<u32>(parent + 0x20 + k));  // the parent did not move
            }
            store<u64>(motion + 0xa8, g_rng.chance(0.6) ? curve() : 0);
            store<float>(motion + 0xb0, value(0, 1));
        }
        for (int k = 0; k < 0x20; k += 4) store<u32>(reinterpret_cast<u64>(g_rng_state) + k, g_rng.u32() | 1);
        // held, the game's own still frame, or not held
        const int frame = g_rng.range(0, 3);  // 0, 1 held; 2 the game's still frame; 3 not held
        decomp_sfx_gate_frame(frame != 3);
        store<u32>(sfx + 0x8c, frame == 2 ? 1u : 0u);
        const bool held_frame = frame <= 1;
        const float step = held_frame ? 0.0f : value(0, 0.1f);

        auto snap = [&]() {
            std::vector<u8> s(0x100 * 3 + 0x40 + 0x20);
            std::memcpy(s.data(), reinterpret_cast<void*>(motion), 0x100);
            std::memcpy(s.data() + 0x100, reinterpret_cast<void*>(n), 0x100);
            std::memcpy(s.data() + 0x200, reinterpret_cast<void*>(parent), 0x100);
            std::memcpy(s.data() + 0x300, reinterpret_cast<void*>(local), 0x40);
            std::memcpy(s.data() + 0x340, g_rng_state, 0x20);
            return s;
        };
        auto put = [&](const std::vector<u8>& s) {
            std::memcpy(reinterpret_cast<void*>(motion), s.data(), 0x100);
            std::memcpy(reinterpret_cast<void*>(n), s.data() + 0x100, 0x100);
            std::memcpy(reinterpret_cast<void*>(parent), s.data() + 0x200, 0x100);
            std::memcpy(reinterpret_cast<void*>(local), s.data() + 0x300, 0x40);
            std::memcpy(g_rng_state, s.data() + 0x340, 0x20);
        };
        const std::vector<u8> before = snap();
        float s1 = step, s2 = step;
        g_calls.clear();
        eboot_kit::fn<MotionUpdate>(kAddr[kind])(motion, n, &s1, local);
        const std::vector<Call> calls_game = g_calls;
        const std::vector<u8> after_game = snap();
        put(before);
        g_calls.clear();
        kOurs[kind](motion, n, &s2, local);
        const std::vector<Call> calls_ours = g_calls;
        const std::vector<u8> after_ours = snap();
        bool ok;
        if (!held_frame) {
            ok = same_calls(calls_game, calls_ours) && after_game == after_ours;
        } else {
            // the base motion once, with the same arguments, and nothing else: no count, re-roll, draw or lerp
            const int base = kind == 1 ? 11 : 10;
            ok = calls_ours.size() == 1 && calls_ours[0].what == base && calls_ours[0].a == motion && calls_ours[0].b == n &&
                 calls_ours[0].c == local && calls_ours[0].step == bits(step) && after_ours == before;
        }
        if (!ok && ++bad <= 6 && !g_quiet)
            std::printf("wobble case %d (kind %d, frame %d, count %u of %u): %s differs\n", c, kind, frame, load<u32>(motion + count), per,
                        held_frame ? "the held frame" : "the game's frame");
        put(before);
    }
    decomp_sfx_gate_frame(false);
    store<u32>(sfx + 0x8c, 0u);
    return bad;
}

// --- the GPU particle type 0x2713's record ---------------------------------------

// Test curves for the type's definition (vtable +0x58: (curve, node, &time) -> the value at it, a vector;
// +0x90: (curve, node) -> its largest value): a + b t in each lane, the lanes apart.
GUEST_ABI __m128 curve_2713(u64 curve, u64, const float* t) {
    const float a = load<float>(curve + 8), b = load<float>(curve + 12);
    return _mm_set_ps(a + 3 * b * *t, a + 2 * b * *t, a - b * *t, a + b * *t);
}
GUEST_ABI float curve_2713_max(u64 curve, u64) { return std::fabs(load<float>(curve + 8)) + std::fabs(load<float>(curve + 12)); }
alignas(64) u8 g_curve_2713_vt[0xa0];
GUEST_ABI u64 stub_scratch_mark() { return 0; }
GUEST_ABI void stub_scratch_release(u64) {}
// Another GPU particle type's update (vtable +0x10: (&bounds, type, record +0x10, buffer)): its bounds, nothing else.
GUEST_ABI void stub_other_type(u64 out, u64, u64, u64) { std::memset(reinterpret_cast<void*>(out), 0, 32); }
alignas(64) u8 g_other_type_vt[0x40];

struct Record {
    u64 record, item, type;
};
Record make_record(bool type_2713) {
    Record r;
    r.record = block(0x80);
    r.item = block(0x70);
    r.type = block(0xb0);
    const u64 def = block(0xe0), counts = block(0x20), node = block(0x100);
    store<u64>(r.type, type_2713 ? game_address(kGxType2713) : reinterpret_cast<u64>(g_other_type_vt));
    store<u64>(r.type + 8, def);
    for (const u64 off : {0x48ull, 0x58ull, 0x88ull, 0x98ull, 0xa8ull}) {
        const u64 c = block(0x20);
        store<u64>(c, reinterpret_cast<u64>(g_curve_2713_vt));
        store<float>(c + 8, value(0, 2));
        store<float>(c + 12, g_rng.chance(0.3) ? 0.0f : value(-20, 20));  // flat now and then: no change to see
        store<u64>(def + off, c);
    }
    for (int k = 0x1c; k < 0xa0; k += 4) store<float>(r.type + k, value(-1, 2));
    store<u64>(r.record + 8, r.type);
    store<u64>(r.record + 0x10, node);
    store<u64>(r.record + 0x18, counts);  // +8 = 0: no buffers
    const float t = value(0, 50);
    store<float>(r.record + 0x28, t);
    // the last item's elapsed time: a whole step's, or 0 (the record just made, or a zero step last)
    store<float>(r.record + 0x24, g_rng.chance(0.8) ? t - 1.0f / 30.0f : t);
    for (int k = 0; k < 0x60; k += 4) store<float>(r.item + k, value(-3, 3));
    store<float>(r.item + 0x60, g_rng.chance(0.8) ? 0.0f : 1.0f / 30.0f);
    return r;
}
std::vector<u8> record_bytes(const Record& r) {
    std::vector<u8> v(0x80 + 0xb0);
    std::memcpy(v.data(), reinterpret_cast<void*>(r.record), 0x80);
    std::memcpy(v.data() + 0x80, reinterpret_cast<void*>(r.type), 0xb0);
    return v;
}
void put_record(const Record& r, const std::vector<u8>& v) {
    std::memcpy(reinterpret_cast<void*>(r.record), v.data(), 0x80);
    std::memcpy(reinterpret_cast<void*>(r.type), v.data() + 0x80, 0xb0);
}

int check_records(int cases, u64 sfx) {
    const auto game = eboot_kit::fn<GxRecordFn>(kGxRecordUpdate);
    int bad = 0, held_cases = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const bool type_2713 = g_rng.chance(0.8), held_frame = g_rng.chance(0.7), still = g_rng.chance(0.1);
        const Record r = make_record(type_2713);
        store<u32>(sfx + 0x8c, still ? 1u : 0u);
        const std::vector<u8> before = record_bytes(r);
        // the game's
        game(r.record, r.item);
        const std::vector<u8> theirs = record_bytes(r);
        put_record(r, before);
        // ours: on any frame between whole steps, the gate on or off
        decomp_sfx_gate_frame(held_frame && g_rng.chance(0.5));
        decomp_sfx_step_frame(held_frame);
        sfx_gx_record_update(r.record, r.item);
        decomp_sfx_step_frame(false);
        decomp_sfx_gate_frame(false);
        const std::vector<u8> ours = record_bytes(r);
        const bool takes = type_2713 && held_frame && !still && load<float>(r.item + 0x60) == 0.0f &&
                           load<u32>(reinterpret_cast<u64>(before.data()) + 0x24) != load<u32>(reinterpret_cast<u64>(before.data()) + 0x28);
        const char* why = nullptr;
        if (!takes) {
            if (ours != theirs) why = "not held: the game's";
        } else {
            ++held_cases;
            // the record's previous time and the type's five values as before, every other byte the game's
            std::vector<u8> want = theirs;
            std::memcpy(want.data() + 0x24, before.data() + 0x24, 4);
            std::memcpy(want.data() + 0x80 + 0x1c, before.data() + 0x80 + 0x1c, 8);
            std::memcpy(want.data() + 0x80 + 0x50, before.data() + 0x80 + 0x50, 0x30);
            if (ours != want) why = "held: the previous time and the five values kept, the rest the game's";
            // the draw's elapsed time: the last whole step's
            float pa, ca, pb, cb;
            std::memcpy(&pa, ours.data() + 0x24, 4);
            std::memcpy(&ca, ours.data() + 0x28, 4);
            std::memcpy(&pb, before.data() + 0x24, 4);
            std::memcpy(&cb, before.data() + 0x28, 4);
            if (!why && bits(ca - pa) != bits(cb - pb)) why = "held: the elapsed time the draw reads";
        }
        put_record(r, before);
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("record case %d (type 0x2713 %d, held %d, still %d, step %g): %s differs\n", c, type_2713, held_frame, still,
                        load<float>(r.item + 0x60), why);
    }
    store<u32>(sfx + 0x8c, 0u);
    if (!g_quiet) std::printf("decomp_sfx_step_gate: %d GPU particle record cases (%d held for type 0x2713), %d differ\n", cases, held_cases, bad);
    return bad;
}

}  // namespace

int main() {
    eboot_kit::load("decomp_sfx_step_gate");
    if (std::memcmp(eboot_kit::at(0x2c4ff30), kInstanceEntry, sizeof kInstanceEntry) != 0 ||
        std::memcmp(eboot_kit::at(0x131e660), kLightEntry, sizeof kLightEntry) != 0 ||
        std::memcmp(eboot_kit::at(kWobbleA), kWobbleEntry, sizeof kWobbleEntry) != 0 ||
        std::memcmp(eboot_kit::at(kWobbleB), kWobbleEntry, sizeof kWobbleEntry) != 0 ||
        std::memcmp(eboot_kit::at(kWobbleFollow), kFollowEntry, sizeof kFollowEntry) != 0 ||
        std::memcmp(eboot_kit::at(kGxRecordUpdate), kGxRecordEntry, sizeof kGxRecordEntry) != 0) {
        std::printf("decomp_sfx_step_gate: the six entries are not where the gate expects them\n");
        return 1;
    }
    // The singletons: SprjSfx (its still-frame word +0x8c), the node pool, and
    // a pre-roll's stand-in render context - outside the cases' blocks.
    alignas(64) static u8 sfx_mem[0xa0], pool_mem[0x40], preroll_mem[0x10];
    const u64 sfx = reinterpret_cast<u64>(sfx_mem), pool = reinterpret_cast<u64>(pool_mem), preroll_ctx = reinterpret_cast<u64>(preroll_mem);
    store<u64>(reinterpret_cast<u64>(eboot_kit::at(kSprjSfx)), sfx);
    store<u64>(reinterpret_cast<u64>(eboot_kit::at(kNodePool)), pool);
    store<u64>(preroll_ctx, reinterpret_cast<u64>(eboot_kit::at(kPrerollContext)));
    // What both reach and a test cannot set up.
    eboot_kit::stub(kContainerUpdate, reinterpret_cast<const void*>(&stub_container));
    eboot_kit::stub(0x2c81730, reinterpret_cast<const void*>(&stub_transitions));
    eboot_kit::stub(0x2c81790, reinterpret_cast<const void*>(&stub_quiet));
    eboot_kit::stub(0x2c4e6b0, reinterpret_cast<const void*>(&stub_state1));
    eboot_kit::stub(0x2c4ec70, reinterpret_cast<const void*>(&stub_state4));
    eboot_kit::stub(0x2c4f660, reinterpret_cast<const void*>(&stub_state0_init));
    eboot_kit::stub(0x2c54280, reinterpret_cast<const void*>(&stub_quiet));
    eboot_kit::stub(0x2c4c6c0, reinterpret_cast<const void*>(&stub_quiet));
    eboot_kit::stub(0x2c4c6b0, reinterpret_cast<const void*>(&stub_finished));
    eboot_kit::stub(0x2c4cdd0, reinterpret_cast<const void*>(&stub_finished));
    eboot_kit::stub(0x2701fc0, reinterpret_cast<const void*>(&stub_light_enable));
    eboot_kit::stub(0x12a2500, reinterpret_cast<const void*>(&stub_light_params));
    g_game_light = eboot_kit::at(0x131e660);
    g_game_instance = eboot_kit::at(0x2c4ff30);
    g_game_wobble_a = eboot_kit::at(kWobbleA);
    g_game_wobble_b = eboot_kit::at(kWobbleB);
    g_game_wobble_follow = eboot_kit::at(kWobbleFollow);
    g_game_gx_record = eboot_kit::at(kGxRecordUpdate);

    constexpr int kCases = 20000;
    int bad = 0, total = 0;
    auto run = [&](bool specials) {
        g_rng.specials = specials;
        int b = check_nodes(kCases);
        b += check_lights(kCases, sfx, preroll_ctx);
        return b;
    };
    bad += run(false);
    bad += run(true);
    total += 4 * kCases;
    // node_held's mutants against the game's own node update, while it is still there.
    int missed = 0;
    g_quiet = true;
    for (const int m : {2, 3, 8}) {
        g_sfx_gate_mutant = m;
        g_rng.specials = false;
        if (!check_nodes(2000)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught\n", m);
            ++missed;
        }
    }
    g_sfx_gate_mutant = 0;
    // The instance update last: the game's node update is node_held's from here.
    eboot_kit::stub(kControllerUpdate, reinterpret_cast<const void*>(&stub_controller));
    eboot_kit::stub(0x2c4df80, reinterpret_cast<const void*>(&node_held_as_game));
    eboot_kit::stub(kNodeRelease, reinterpret_cast<const void*>(&stub_release));
    eboot_kit::stub(kNodeClear, reinterpret_cast<const void*>(&stub_clear));
    eboot_kit::stub(kPoolFree, reinterpret_cast<const void*>(&stub_pool_free));
    g_rng.specials = false;
    bad += check_instances(kCases, sfx);
    g_rng.specials = true;
    bad += check_instances(kCases, sfx);
    total += 2 * kCases;
    std::printf("decomp_sfx_step_gate: %d cases (nodes, lights, instances; ordinary values, then special ones), %d differ\n", total, bad);
    if (bad) return 1;

    // The rest of the source's mutants caught.
    for (const int m : {1, 4, 5, 6, 7}) {
        g_sfx_gate_mutant = m;
        g_rng.specials = false;
        int caught = 0;
        if (m >= 5) {
            caught += check_lights(2000, sfx, preroll_ctx);
        } else {
            caught += check_instances(2000, sfx);
            caught += check_lights(2000, sfx, preroll_ctx);
        }
        if (!caught) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught\n", m);
            ++missed;
        }
    }
    g_sfx_gate_mutant = 0;
    // The wobble motions last: their base motions and the FX RNG the test's from here.
    eboot_kit::stub(kWobbleABase, reinterpret_cast<const void*>(&stub_base_a));
    eboot_kit::stub(kWobbleBBase, reinterpret_cast<const void*>(&stub_base_b));
    eboot_kit::stub(0x2c6d580, reinterpret_cast<const void*>(&stub_fx_rng));
    g_quiet = false;
    g_rng.specials = false;
    int wbad = check_wobbles(kCases, sfx);
    g_rng.specials = true;
    wbad += check_wobbles(kCases, sfx);
    std::printf("decomp_sfx_step_gate: %d wobble motion cases (ordinary values, then special ones), %d differ\n", 2 * kCases, wbad);
    if (wbad) return 1;
    g_quiet = true;
    for (const int m : {1, 9, 10, 11, 12}) {
        g_sfx_gate_mutant = m;
        g_rng.specials = false;
        if (!check_wobbles(2000, sfx)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught by the wobble motions\n", m);
            if (m != 1) ++missed;  // 1 is the instances' and lights' to catch; here only a second look
        }
    }
    g_sfx_gate_mutant = 0;
    // The GPU particle type 0x2713's record last: the scratch allocator the record update brackets its work with
    // stubbed, the type's curves the test's.
    store<u64>(reinterpret_cast<u64>(g_curve_2713_vt) + 0x58, reinterpret_cast<u64>(&curve_2713));
    store<u64>(reinterpret_cast<u64>(g_curve_2713_vt) + 0x90, reinterpret_cast<u64>(&curve_2713_max));
    store<u64>(reinterpret_cast<u64>(g_other_type_vt) + 0x10, reinterpret_cast<u64>(&stub_other_type));
    eboot_kit::stub(0x247d5b0, reinterpret_cast<const void*>(&stub_scratch_mark));
    eboot_kit::stub(0x247d5f0, reinterpret_cast<const void*>(&stub_scratch_release));
    g_quiet = false;
    g_rng.specials = false;
    int rbad = check_records(kCases, sfx);
    g_rng.specials = true;
    rbad += check_records(kCases, sfx);
    if (rbad) return 1;
    g_quiet = true;
    for (const int m : {13, 14, 15, 16, 17, 18}) {
        g_sfx_gate_mutant = m;
        g_rng.specials = false;
        if (!check_records(2000, sfx)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught by the GPU particle records\n", m);
            ++missed;
        }
    }
    g_sfx_gate_mutant = 0;
    std::printf("decomp_sfx_step_gate: %d of 18 mutants caught\n", 18 - missed);
    return missed ? 1 : 0;
}
