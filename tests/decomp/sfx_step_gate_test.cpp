// The effects' step gate (decomp/sfx/step_gate.cpp) against the game's own
// code (tests/decomp/eboot_kit.h), on synthetic effects:
//   - a node on a held frame (node_held) against the game's node update,
//     0x2c4df80: the same container calls, step adds and answer for a node
//     whose update changes nothing else (a staged node staying in state 3, a
//     plain one whose conditions pick no new definition, a dead one, an
//     inactive one), and for a node in a state whose update would move it
//     (2, 6, 7, 5, 0) the same container call with its state, emission and
//     definition fields left as they were;
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
// and each of the source's mutants caught. Skips without the 1.09 eboot.
#define DECOMP_SFX_STEP_GATE_MUTANTS
int g_sfx_gate_mutant = 0;
#include "decomp/sfx/step_gate.cpp"

#include "eboot_kit.h"

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
        if (kind != 4) {
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

}  // namespace

int main() {
    eboot_kit::load("decomp_sfx_step_gate");
    if (std::memcmp(eboot_kit::at(0x2c4ff30), kInstanceEntry, sizeof kInstanceEntry) != 0 ||
        std::memcmp(eboot_kit::at(0x131e660), kLightEntry, sizeof kLightEntry) != 0) {
        std::printf("decomp_sfx_step_gate: the two entries are not where the gate expects them\n");
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
    for (const int m : {2, 3}) {
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
    std::printf("decomp_sfx_step_gate: %d of 7 mutants caught\n", 7 - missed);
    return missed ? 1 : 0;
}
