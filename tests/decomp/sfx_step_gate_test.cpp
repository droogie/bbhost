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
//   - the same light under the smooth steps, inside a node's real update
//     (sfx_node::update, its container calls published): frame sequences at
//     60, 120, 144 and jittered rates against the game's light stepped at 30 -
//     after every whole step the phase +0x30 and a blending mode's words
//     (+0x110; +0xf0..+0x10f, +0x114, +0x118) bit-equal to the game's calls at
//     the node's time of each whole step, and on every frame its GX light
//     byte-equal to the game's call at the frame's time with the rate as 0 and
//     a blending mode as keeping, on that counted state;
//   - the effect manager's light fades, 0x12e5500: the smooth steps off, or
//     the game's own still frame, the game's call as it is; on, the game's call
//     with the frame's step times the manager's time scale (0 while it is
//     still) - the list, its lights and every call;
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
//   - the same record update with the ribbon, FFX-format GX type 0x28 (its
//     update 0x2ec1e00 and set-up 0x2ec47f0 the game's), under the smooth
//     steps at 60, 120, 144 and jittered rates, against the game's update once
//     a whole step at the whole step's time and place: after every whole step
//     its insert history (times, places, lengths, scalars, the seed, the end
//     time) and the FX RNG bit-equal, the points both hold bit-equal (which
//     old ones are held may differ by what a frame's own time lets go);
//     between whole steps none of that moves and no point is smoothed; on
//     every frame the last drawn point the emitter at the frame's own time, the
//     record's times as the game's one call leaves them, the definition back;
//   - the same record update with type 0x2713 under the smooth steps: a smooth
//     instance's (marked for the frame) the game's call with the whole step's
//     elapsed time for the type's curves, the record's times then the frame's
//     and its elapsed one the type's again for the draw; an unmarked one or
//     the game's own still frame the game's call;
//   - the same record update with StandardParticle's and Spark's simulations
//     (0x13040b0, 0x12fef10, the game's) under the smooth steps, an emitter a
//     case (dense ones the cap reaches, small ones, a wind share that
//     truncates, an end time or a limit of events, new ones at their set-up's
//     interval), in 2, 3, 4, 5 or 8 frames a whole step or uneven splits,
//     against the game's update a whole step at a time: what each whole step's
//     frames emit, its events, the emitter on or off and its timer at its end;
//     frames of one update a window every byte and the FX RNG the game's; and
//     over 40 s, emitters whose interval is drawn at random emitting the same
//     events a second (within 3%) as 30's whole steps;
//   - the wobble motions 0x570e6d0 and 0x570e690 under the smooth steps, in a
//     staged node's real update at 60, 120, 144 and jittered rates against
//     the game's update once a whole step: the count and the FX RNG after
//     every whole step bit-equal, untouched between, the base motion once a
//     frame by the frame's step, and 0x570e690's follow between whole steps
//     every byte the game's update with its count held;
//   - the wind sources' registry step, 0x1314660, with the game's own wind
//     sources (0x56ec140, stepped by 0x2d552d0) and default objects: the
//     smooth steps off, the game's own still frame or a step not the effect
//     manager's, the game's call; on, the game's call with the frame's own
//     time on the manager's scale; over a second of frames the sources' ages
//     and gust phases as 30 whole steps leave them;
// and each of the source's mutants caught. Skips without the 1.09 eboot.
#define DECOMP_SFX_STEP_GATE_MUTANTS
int g_sfx_gate_mutant = 0;
#include "decomp/sfx/step_gate.cpp"

#include "eboot_kit.h"

#include "engine/sfx_schedule.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/mman.h>
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

bool g_times = false;  // the smooth steps' cases: each container call's node time (+0x7c) and definition time (+0xe0) recorded too
// The smooth light cases: a light module the container of its node runs (as the game's container update runs its modules).
u64 g_light_node = 0, g_light_module = 0, g_light_params = 0;
// The smooth wobble cases: a motion the container of its node runs (ours: sfx_wobble_a or sfx_wobble_follow).
u64 g_wobble_node = 0, g_wobble_motion = 0, g_wobble_local = 0;
GUEST_ABI void (*g_wobble_fn)(u64, u64, const float*, u64) = nullptr;
GUEST_ABI void stub_container(u64 container, float* step, u64 definition, u64 context) {
    const u64 seen = g_times ? (static_cast<u64>(load<u32>(container - 4)) << 32) | load<u32>(container + 0x60) : 0;
    rec(1, container, definition, context, seen, *step);
    if (g_light_module && container == g_light_node + 0x80) sfx_light_update(g_light_module, g_light_params, context);
    if (g_wobble_motion && container == g_wobble_node + 0x80) g_wobble_fn(g_wobble_motion, g_wobble_node, step, g_wobble_local);
}
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

// --- the smooth steps ----------------------------------------------------------
//
// With effect steps smooth an instance whose classes are all on the converted
// list (a test list here, of the game's own classes) runs the frame domain.
// Against the game's instance update with sfx_node_update - the game's node
// update as ours, which the node update's own test holds to the game's - in
// its node update's place: at 30 (every frame a whole step, the frame's time
// the whole step's, nothing carried) the same, every call, every byte, the
// times the containers see. On a frame between whole steps (a quarter of a
// step at 120, a quarter in) no controller and nothing counted: each active
// node's container calls with the frame's step times the manager's time scale
// times the instance's (none while the manager is still), its times put the
// frame's start past their whole step - a plain node's definition time its
// end for the second call - and back after, nothing of the instance's or its
// nodes' changed. A class off the list (a motion, a module, a binder's
// target, a GX binder's type, a composite's part or a composite too deep)
// keeps the instance on the held frames' update, counted under that class; a
// released or inactive instance, or the game's own still frame, is the game's.

constexpr u64 kOnMotions[] = {0x570e650, 0x570e790};  // the Euler and rotation motions
constexpr u64 kOnModules[] = {0x56eb2e0, 0x5756e50};  // the children follow, the Sprj module
constexpr u64 kCpuBinder = 0x56e3090, kGxBinder = 0x56e3100;
constexpr u64 kOnGxTypes[] = {0x56eb1d0, 0x56eb220};
constexpr u64 kComposite = 0x570e750;
constexpr u64 kOffModule = 0x5760f20, kOffWobble = 0x570e6d0, kOffGxType = 0x56eb030;
constexpr u64 kTestConverted[] = {0x570e650, 0x570e790, 0x56eb2e0, 0x5756e50, 0x56e3090, 0x56e3100, 0x56eb1d0, 0x56eb220};
constexpr u64 kNoTimeAt = 0x5ac82e8;

u64 object_of(u64 bn) {
    const u64 o = block(0x30);
    store<u64>(o, game_address(bn));
    return o;
}
template <std::size_t N>
u64 any(const u64 (&from)[N]) {
    return from[g_rng.range(0, static_cast<int>(N) - 1)];
}
// A motion on the list: one of the two, or a composite `levels` deep whose parts all are.
u64 motion_on(int levels) {
    if (levels <= 0) return object_of(any(kOnMotions));
    const u64 m = object_of(kComposite);
    store<u64>(m + 8, motion_on(levels - 1));
    store<u64>(m + 0x10, g_rng.chance(0.5) ? motion_on(levels - 1) : object_of(any(kOnMotions)));
    return m;
}
// A pending binder (the container's list +0x30: next +0x18, target +0x20) on a node.
void pend(u64 node, u64 target) {
    const u64 item = block(0x28);
    store<u64>(item + 0x18, load<u64>(node + 0xb0));
    store<u64>(item + 0x20, target);
    store<u64>(node + 0xb0, item);
}
// A GX binder whose type instance (*(+0x10) +8) is of `type`.
u64 gx_binder(u64 type) {
    const u64 b = object_of(kGxBinder), holder = block(0x20);
    store<u64>(holder + 8, object_of(type));
    store<u64>(b + 0x10, holder);
    return b;
}

// Makes the instance's nodes ones the whole node update can run - controllers
// (+0xc8), definitions (+0xd0), a plain node's switch object (+0xd8) - with
// times that are sometimes the "no time" value, and gives them classes: all on
// the list, or (`off`) one class off it somewhere. The class off it, or 0.
u64 classes(const Instance& s, bool off, u32 none) {
    const int at = s.nodes.empty() ? -1 : g_rng.range(0, static_cast<int>(s.nodes.size()) - 1);
    const int where = g_rng.range(0, 7);
    u64 kept = 0;
    for (std::size_t i = 0; i < s.nodes.size(); ++i) {
        const u64 n = s.nodes[i];
        store<u64>(n + 0xc8, controller(0, 4));
        store<u64>(n + 0xd0, definition(4));
        if (!(load<u32>(n + 0x78) & 8)) {
            u64 sw = 0;
            if (g_rng.chance(0.6)) {
                sw = block(0x20);
                store<u8>(sw + 0x11, g_rng.chance(0.5) ? 1 : 0);
            }
            store<u64>(n + 0xd8, sw);
        }
        if (g_rng.chance(0.15)) store<u32>(n + 0xe0, none);
        if (g_rng.chance(0.1)) store<u32>(n + 0x7c, none);
        const int mk = g_rng.range(0, 3);
        store<u64>(n + 0x90, mk == 0 ? 0 : motion_on(mk - 1));
        store<u64>(n + 0x98, g_rng.chance(0.5) ? object_of(any(kOnModules)) : 0);
        store<u64>(n + 0xa0, g_rng.chance(0.2) ? object_of(any(kOnModules)) : 0);
        store<u64>(n + 0xa8, g_rng.chance(0.3) ? object_of(any(kOnModules)) : 0);
        store<u64>(n + 0xb0, 0);
        const int pending = g_rng.range(0, 2);
        for (int k = 0; k < pending; ++k) pend(n, g_rng.chance(0.5) ? object_of(kCpuBinder) : gx_binder(any(kOnGxTypes)));
        if (!off || static_cast<int>(i) != at) continue;
        switch (where) {
        case 0: store<u64>(n + 0x90, object_of(kOffWobble)); kept = kOffWobble; break;
        case 1: store<u64>(n + 0x98, object_of(kOffModule)); kept = kOffModule; break;
        case 2: store<u64>(n + 0xa0, object_of(kOffModule)); kept = kOffModule; break;
        case 3: store<u64>(n + 0xa8, object_of(kOffModule)); kept = kOffModule; break;
        case 4: pend(n, object_of(kOffModule)); kept = kOffModule; break;
        case 5: pend(n, gx_binder(kOffGxType)); kept = kOffGxType; break;
        case 6: {  // a composite with a part off the list
            const u64 m = object_of(kComposite);
            store<u64>(m + 8, motion_on(g_rng.range(0, 1)));
            store<u64>(m + 0x10, object_of(kOffWobble));
            store<u64>(n + 0x90, m);
            kept = kOffWobble;
            break;
        }
        default:  // composites five deep: the fifth one is not looked into
            store<u64>(n + 0x90, motion_on(5));
            kept = kComposite;
            break;
        }
    }
    return kept;
}

GUEST_ABI u8 node_update_as_game(u64 node, float* step, u64 definition, u64 context) { return sfx_node_update(node, step, definition, context); }

u64 kept_count(u64 bn) {
    for (std::size_t i = 0; i < kKeptSlots; ++i)
        if (g_kept_class[i].load() == game_address(bn)) return g_kept_count[i].load();
    return 0;
}
u32 addss(u32 x, float y) {
    float v;
    std::memcpy(&v, &x, 4);
    __asm__("addss %1, %0" : "+x"(v) : "x"(y));
    return bits(v);
}
// A time as a container call sees it: moved on by the offset, unless that is zero or the time is "no time".
u32 shifted(u32 t, float offset, u32 none) { return offset == 0.0f || t == none ? t : addss(t, offset); }
u64 times(u32 t, u32 e) { return (static_cast<u64>(t) << 32) | e; }

struct SmoothSeen {
    long took = 0, kept = 0, game = 0, between_calls = 0, shifted_t = 0, shifted_e = 0, controllers_30 = 0;
    long by_class[4] = {};  // kept by a module, the wobble, the GX type left out, the composite too deep
} g_seen_smooth;

int check_smooth(int cases, u64 sfx, u64 manager) {
    const auto game = eboot_kit::fn<InstanceUpdate>(0x2c4ff30);
    const u32 none = load<u32>(reinterpret_cast<u64>(eboot_kit::at(kNoTimeAt)));
    eboot_kit::stub(0x2c4df80, reinterpret_cast<const void*>(&node_update_as_game));
    g_converted = kTestConverted;
    g_converted_count = sizeof kTestConverted / sizeof kTestConverted[0];
    decomp_sfx_smooth(true);
    g_times = true;
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const Instance s = make_instance();
        const u64 kept = classes(s, g_rng.chance(0.35), none);
        const bool still = g_rng.chance(0.1), manager_still = g_rng.chance(0.1);
        store<u32>(sfx + 0x8c, still ? 1u : 0u);
        store<u8>(manager + 0x5ce, manager_still ? 1 : 0);
        const float time_scale = g_rng.chance(0.6) ? 1.0f : value(0, 2);
        store<float>(manager + 0x5d8, time_scale);
        const u32 flags = load<u32>(s.inst + 0x88);
        const bool live = !still && !(flags & 4) && (flags & 1);  // the smooth steps' to decide
        const bool takes = live && !kept;
        const Snapshot before = snap(s);
        const char* why = nullptr;
        // At 30, the manager's step: 1/30 times its time scale, none while it is still.
        {
            const float step = manager_still ? 0.0f : 1.0f / 30.0f * time_scale;
            decomp_sfx_gate_frame(false);
            decomp_sfx_frame_time(1.0f / 30.0f, 0.0f, 0.0f);
            g_calls.clear();
            const u8 r1 = game(s.inst, s.ctx, step);
            const std::vector<Call> c1 = g_calls;
            const Snapshot a1 = snap(s);
            restore(s, before);
            const u64 took = g_smooth_updates.load(), held_back = g_smooth_kept.load(), by_class = kept ? kept_count(kept) : 0;
            g_calls.clear();
            const u8 r2 = sfx_instance_update(s.inst, s.ctx, step);
            const std::vector<Call> c2 = g_calls;
            const Snapshot a2 = snap(s);
            restore(s, before);
            g_seen_smooth.controllers_30 += takes ? static_cast<long>(count(c2, 2)) : 0;
            if (r1 != r2 || !same_calls(c1, c2) || !same(a1, a2)) why = "at 30, against the game's";
            else if (g_smooth_updates.load() - took != (takes ? 1u : 0u)) why = "which instances take the smooth steps";
            else if (g_smooth_kept.load() - held_back != (live && kept ? 1u : 0u)) why = "the count of those a class kept";
            else if (live && kept && kept_count(kept) != by_class + 1) why = "the class that kept it";
        }
        // A frame between whole steps at 120, a quarter of a step in.
        if (!why) {
            const float frame = 1.0f / 120.0f, from = 1.0f / 120.0f, into = 2.0f / 120.0f;
            decomp_sfx_gate_frame(true);
            decomp_sfx_frame_time(frame, from, into);
            g_calls.clear();
            const u8 r = sfx_instance_update(s.inst, s.ctx, 0.0f);
            const std::vector<Call> calls = g_calls;
            const Snapshot after = snap(s);
            restore(s, before);
            if (takes) {
                const float scale = load<float>(s.inst + 8);
                const u32 step = bits((manager_still ? 0.0f : frame * time_scale) * scale);
                const float from_s = manager_still ? 0.0f : from * time_scale * scale;
                const float into_s = manager_still ? 0.0f : into * time_scale * scale;
                const u64 def = load<u64>(s.inst + 0x38);
                // Each active node's calls, in its own order (the walk's order is the game's next function's).
                std::vector<Call> want;
                for (const u64 n : s.nodes) {
                    const u32 nf = load<u32>(n + 0x78);
                    if (!def || (nf & 4) || !(nf & 1)) continue;
                    const u32 t = shifted(load<u32>(n + 0x7c), from_s, none), e = load<u32>(n + 0xe0);
                    if (!(nf & 8)) {
                        want.push_back({1, n + 0x80, def, s.ctx, times(t, shifted(e, from_s, none)), step});
                        want.push_back({1, n + 0x80, def, s.ctx, times(t, shifted(e, into_s, none)), step});
                    } else {
                        want.push_back({1, n + 0x80, def, s.ctx, times(t, e), step});
                    }
                }
                auto by_node = [](std::vector<Call> v) {
                    std::stable_sort(v.begin(), v.end(), [](const Call& x, const Call& y) { return x.a < y.a; });
                    return v;
                };
                if (r != 1 || !same_calls(by_node(calls), by_node(want))) why = "a frame between: the calls";
                else if (!same(before, after)) why = "a frame between: the instance or a node changed";
                ++g_seen_smooth.took;
                g_seen_smooth.between_calls += static_cast<long>(want.size());
                for (const u64 n : s.nodes) {
                    const u32 nf = load<u32>(n + 0x78);
                    if (!def || (nf & 4) || !(nf & 1)) continue;
                    g_seen_smooth.shifted_t += shifted(load<u32>(n + 0x7c), from_s, none) != load<u32>(n + 0x7c);
                    g_seen_smooth.shifted_e += !(nf & 8) && shifted(load<u32>(n + 0xe0), into_s, none) != load<u32>(n + 0xe0);
                }
            } else if (live) {
                // a class off the list: the held frames' update, as with the smooth steps off
                decomp_sfx_smooth(false);
                g_calls.clear();
                const u8 r0 = sfx_instance_update(s.inst, s.ctx, 0.0f);
                const std::vector<Call> c0 = g_calls;
                const Snapshot a0 = snap(s);
                restore(s, before);
                decomp_sfx_smooth(true);
                if (r != r0 || !same_calls(calls, c0) || !same(after, a0)) why = "a frame between, a class off the list: the held frames'";
                ++g_seen_smooth.kept;
                ++g_seen_smooth.by_class[kept == kOffModule ? 0 : kept == kOffWobble ? 1 : kept == kOffGxType ? 2 : 3];
            } else {
                // released, inactive or the game's own still frame: the game's
                g_calls.clear();
                const u8 r0 = game(s.inst, s.ctx, 0.0f);
                const std::vector<Call> c0 = g_calls;
                const Snapshot a0 = snap(s);
                restore(s, before);
                if (r != r0 || !same_calls(calls, c0) || !same(after, a0)) why = "a frame between: the game's own update";
                ++g_seen_smooth.game;
            }
            decomp_sfx_gate_frame(false);
        }
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("smooth case %d (flags 0x%x, %zu nodes, %zu controllers, %s%s%s, scales %g/%g): %s differs\n", c, flags, s.nodes.size(),
                        s.ctls.size(), kept ? "a class off the list" : "all converted", still ? ", the game's still frame" : "",
                        manager_still ? ", the manager still" : "", time_scale, load<float>(s.inst + 8), why);
    }
    store<u32>(sfx + 0x8c, 0);
    g_times = false;
    decomp_sfx_smooth(false);
    g_converted = kConverted;
    g_converted_count = sizeof kConverted / sizeof kConverted[0];
    eboot_kit::stub(0x2c4df80, reinterpret_cast<const void*>(&node_held_as_game));
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
Light make_light(u64 on_node = 0, u64 on_inst = 0) {
    Light l;
    l.module = block(l.module_size);
    for (std::size_t k = 0; k < l.module_size; k += 4) store<float>(l.module + k, value(-3, 3));
    l.inst = on_inst ? on_inst : block(0x98);
    l.node = on_node;
    if (!l.node) {
        l.node = block(0x100);
        for (int k = 0; k < 0x100; k += 4) store<float>(l.node + k, value(-50, 50));
        store<u64>(l.node + 0x58, l.inst);
        store<float>(l.node + 0x7c, value(0, 10));
    }
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

// --- the light under the smooth steps -------------------------------------------

struct SmoothLightSeen {
    long cases = 0, frames = 0, ticks = 0, light_calls = 0, flicker = 0, blending = 0, plain = 0;
} g_seen_sl;

// A copy of a light's module and GX light that the game's own function runs on (the same node).
struct LightCopy {
    u64 module, light;
};
LightCopy copy_light(const Light& l, u64 module_from, u64 light_from) {
    LightCopy c{block(l.module_size), block(l.light_size)};
    std::memcpy(reinterpret_cast<void*>(c.module), reinterpret_cast<void*>(module_from), l.module_size);
    std::memcpy(reinterpret_cast<void*>(c.light), reinterpret_cast<void*>(light_from), l.light_size);
    store<u64>(c.module + 0x28, c.light);
    return c;
}
bool same_words(u64 a, u64 b, u64 off, std::size_t n) { return std::memcmp(reinterpret_cast<void*>(a + off), reinterpret_cast<void*>(b + off), n) == 0; }

int check_smooth_lights(int cases, u64 sfx) {
    const auto game = eboot_kit::fn<LightUpdate>(0x131e660);
    const u32 none = load<u32>(reinterpret_cast<u64>(eboot_kit::at(kNoTimeAt)));
    store<u32>(sfx + 0x8c, 0);
    decomp_sfx_gate_frame(false);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        // A node the node update keeps quiet - plain (two container calls an update), or staged in state 3,
        // which its controller holds (one) - and its light.
        const u64 inst = block(0x98);
        store<u64>(inst + 0x78, block(8));  // not this node: no root-node stop
        const u64 def = definition(4);
        const bool plain = g_rng.chance(0.5);
        const u64 n = node(inst, plain ? 1u : 9u, 3);
        store<u64>(n + 0xc8, controller(0, 4));
        store<u64>(n + 0xd0, def);
        store<u64>(n + 0xd8, 0);
        for (int k = 0x20; k < 0x50; k += 4) store<float>(n + k, value(-50, 50));
        store<float>(n + 0x7c, value(0, 6));
        if (load<u32>(n + 0x7c) == none) store<float>(n + 0x7c, 1.0f);
        const Light l = make_light(n, inst);
        if (g_rng.chance(0.6)) store<s32>(l.module + 0x7c, 1);  // the flicker on
        const u64 params = block(0x80), ctx = block(0x40);
        LightCopy game30 = copy_light(l, l.module, l.light);
        const u32 mode = load<u32>(l.module + 0x38), colour_mode = load<u32>(l.module + 0x3c);
        const int rate_kind = g_rng.range(0, 3);  // 60, 120, 144, jittered
        const float scale = g_rng.chance(0.5) ? 1.0f : value(0.25f, 2.0f);
        double carry = 0.0;
        const int frames = g_rng.range(4, 40);
        const char* why = nullptr;
        int at = -1;
        ++g_seen_sl.cases;
        g_seen_sl.flicker += load<s32>(l.module + 0x7c) > 0;
        g_seen_sl.blending += mode == 2 || colour_mode == 2;
        g_seen_sl.plain += plain;
        for (int f = 0; f < frames && !why; ++f) {
            const double dt = rate_kind == 0 ? 1.0 / 60 : rate_kind == 1 ? 1.0 / 120 : rate_kind == 2 ? 1.0 / 144 : g_rng.value(1.0f / 150, 1.0f / 40);
            const sfx_schedule::Frame fr = sfx_schedule::next(carry, dt, 30);
            sfx_node::Steps st;
            st.smooth = true;
            st.tick = fr.whole > 0;
            st.counted = fr.seconds * scale;
            st.frame = fr.frame * scale;
            st.from = fr.from * scale;
            st.into = fr.into * scale;
            const u32 t_step = load<u32>(n + 0x7c);
            g_calls.clear();
            g_light_node = n;
            g_light_module = l.module;
            g_light_params = params;
            sfx_node::update(n, st, def, ctx);
            g_light_module = 0;
            long calls = 0;
            for (const Call& k : g_calls) calls += k.what == 1 && k.a == n + 0x80;
            ++g_seen_sl.frames;
            g_seen_sl.light_calls += calls;
            const u32 t_after = load<u32>(n + 0x7c);
            // The game at 30: on a whole step, each of the update's container calls runs the light at the node's
            // time of the step.
            if (st.tick) {
                ++g_seen_sl.ticks;
                for (long k = 0; k < calls; ++k) {
                    store<u32>(n + 0x7c, t_step);
                    game(game30.module, params, ctx);
                }
                store<u32>(n + 0x7c, t_after);
                if (!same_words(l.module, game30.module, 0x30, 4)) why = "the phase after a whole step";
                else if (mode == 2 && !same_words(l.module, game30.module, 0x110, 4)) why = "the blended +0x110 after a whole step";
                else if (colour_mode == 2 && (!same_words(l.module, game30.module, 0xf0, 0x20) || !same_words(l.module, game30.module, 0x114, 8)))
                    why = "the blended colours after a whole step";
            }
            // The frame's: the game's call at the frame's time with the rate as 0 and a blending mode as keeping, on
            // the counted state - its GX light.
            if (!why && calls) {
                const LightCopy want = copy_light(l, game30.module, game30.light);
                store<float>(want.module + 0x80, 0.0f);
                if (mode == 2) store<u32>(want.module + 0x38, 1u);
                if (colour_mode == 2) store<u32>(want.module + 0x3c, 1u);
                store<u32>(n + 0x7c, shifted(t_step, st.from, none));
                game(want.module, params, ctx);
                store<u32>(n + 0x7c, t_after);
                std::vector<u8> a(l.light_size), b(l.light_size);
                std::memcpy(a.data(), reinterpret_cast<void*>(l.light), l.light_size);
                std::memcpy(b.data(), reinterpret_cast<void*>(want.light), l.light_size);
                if (!same_bytes(a, b)) why = "the GX light at the frame's time";
            }
            if (why) at = f;
        }
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("smooth light case %d (%s node, modes %u/%u, flicker %d, rate kind %d, scale %g): %s differs at frame %d\n", c,
                        plain ? "a plain" : "a staged", mode, colour_mode, load<s32>(l.module + 0x7c), rate_kind, scale, why, at);
    }
    return bad;
}

// --- the light fades ----------------------------------------------------------

GUEST_ABI void stub_light_release(u64 bank, u64 light) { rec(12, bank, light, 0, 0); }
GUEST_ABI void stub_light_free(u64 light) { rec(13, light, 0, 0, 0); }

int check_light_fades(int cases, u64 sfx) {
    const auto game = eboot_kit::fn<LightFadesFn>(kLightFades);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const u64 manager = block(0x600), list = manager + 0x118;
        const u64 vt = block(0x10);
        store<u64>(vt, reinterpret_cast<u64>(&stub_light_free));
        const int entries = g_rng.range(0, 5);
        const u64 begin = block(0x40 * 6);
        std::vector<u64> lights;
        for (int e = 0; e < entries; ++e) {
            const u64 en = begin + 0x40 * e;
            for (int k = 0; k < 0x20; k += 4) store<float>(en + k, value(-2, 2));
            store<float>(en + 0x20, value(0.05f, 3));
            store<float>(en + 0x24, g_rng.chance(0.2) ? value(-0.1f, 0.0f) : value(0, 3));
            for (const u64 off : {0x28, 0x30, 0x38}) {
                u64 light = 0;
                if (g_rng.chance(0.7)) {
                    light = block(0xd0);
                    for (int k = 0; k < 0xd0; k += 4) store<float>(light + k, value(-1, 1));
                    store<u64>(light, vt);
                    store<s32>(light + 8, g_rng.range(1, 3));
                    store<u8>(light + 0xc5, g_rng.chance(0.8) ? 0 : 1);
                    lights.push_back(light);
                }
                store<u64>(en + off, light);
            }
        }
        store<u64>(list + 0x10, begin);
        store<u64>(list + 0x18, begin + 0x40 * entries);
        const bool smooth = g_rng.chance(0.7), still = g_rng.chance(0.15), game_still = g_rng.chance(0.1);
        const float ts = g_rng.chance(0.6) ? 1.0f : value(0, 2);
        store<u8>(manager + 0x5ce, still ? 1 : 0);
        store<float>(manager + 0x5d8, ts);
        store<u32>(sfx + 0x8c, game_still ? 1u : 0u);
        const float frame = g_rng.chance(0.3) ? 1.0f / 30 : value(1.0f / 200, 1.0f / 30);
        const float step = still || game_still ? 0.0f : (g_rng.chance(0.5) ? 1.0f / 30 : 0.0f) * ts;
        decomp_sfx_smooth(smooth);
        decomp_sfx_frame_time(frame, 0.0f, 0.0f);
        // the memory both may change: the list, its entries and the lights
        auto snap_all = [&]() {
            std::vector<u8> v(0x20 + 0x40 * 6);
            std::memcpy(v.data(), reinterpret_cast<void*>(list), 0x20);
            std::memcpy(v.data() + 0x20, reinterpret_cast<void*>(begin), 0x40 * 6);
            for (const u64 li : lights) {
                const std::size_t o = v.size();
                v.resize(o + 0xd0);
                std::memcpy(v.data() + o, reinterpret_cast<void*>(li), 0xd0);
            }
            return v;
        };
        auto put_all = [&](const std::vector<u8>& v) {
            std::memcpy(reinterpret_cast<void*>(list), v.data(), 0x20);
            std::memcpy(reinterpret_cast<void*>(begin), v.data() + 0x20, 0x40 * 6);
            std::size_t o = 0x20 + 0x40 * 6;
            for (const u64 li : lights) {
                std::memcpy(reinterpret_cast<void*>(li), v.data() + o, 0xd0);
                o += 0xd0;
            }
        };
        const std::vector<u8> before = snap_all();
        const float want_step = smooth && !game_still ? (still ? 0.0f : frame * ts) : step;
        g_calls.clear();
        game(list, want_step);
        const std::vector<Call> c_want = g_calls;
        const std::vector<u8> want = snap_all();
        put_all(before);
        g_calls.clear();
        sfx_light_fades(list, step);
        const std::vector<Call> c_ours = g_calls;
        const std::vector<u8> ours = snap_all();
        if ((!same_calls(c_want, c_ours) || !same_bytes(want, ours)) && ++bad <= 6 && !g_quiet)
            std::printf("light fade case %d (%d entries, smooth %d, still %d, the game's still %d, step %g, frame %g, scale %g): ours differs\n", c,
                        entries, smooth, still, game_still, step, frame, ts);
    }
    decomp_sfx_smooth(false);
    store<u32>(sfx + 0x8c, 0);
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

// --- the wobble motions under the smooth steps -------------------------------------------

// 0x570e890's update (sub_2e8de10) as the patched image runs it: its follow, then a call of sub_2e8e430 with its own four
// arguments (at 0x2e8e19f), which the image sends to ours, sfx_wobble_b. The test runs the game's function with that call
// caught - a stub over sub_2e8e430's entry for the call, its bytes put back after - and then ours with the arguments caught.
constexpr u64 kFollowB = 0x2e8de10;
struct Caught {
    u64 motion = 0, node = 0, local = 0;
    const float* step = nullptr;
    int n = 0;
} g_caught;
GUEST_ABI void catch_wobble_b(u64 motion, u64 node, const float* step, u64 local) {
    g_caught.motion = motion, g_caught.node = node, g_caught.step = step, g_caught.local = local;
    ++g_caught.n;
}
u8 g_wobble_b_entry[14];
void put_code(u64 bn, const u8* bytes, std::size_t n) {
    const u64 page = bn & ~0xfffull, end = (bn + n + 0xfff) & ~0xfffull;
    mprotect(reinterpret_cast<void*>(page), end - page, PROT_READ | PROT_WRITE | PROT_EXEC);
    std::memcpy(eboot_kit::at(bn), bytes, n);
    mprotect(reinterpret_cast<void*>(page), end - page, PROT_READ | PROT_EXEC);
}
GUEST_ABI void follow_b_patched(u64 motion, u64 node, const float* step, u64 local) {
    g_caught = {};
    eboot_kit::stub(kWobbleB, reinterpret_cast<const void*>(&catch_wobble_b));
    eboot_kit::fn<MotionUpdate>(kFollowB)(motion, node, step, local);
    put_code(kWobbleB, g_wobble_b_entry, sizeof g_wobble_b_entry);
    if (g_caught.n == 1 && g_caught.motion == motion && g_caught.node == node && g_caught.step == step && g_caught.local == local)
        sfx_wobble_b(motion, node, step, local);
    else
        rec(99, motion, node, local, static_cast<u64>(g_caught.n), 0.0f);  // not the call the disassembly shows
}

// The wobble motions in a staged node's real update (sfx_node::update, its container call published), frame sequences at
// 60, 120, 144 and jittered rates, the parent moving on every frame, against the game's function once a whole step (at 30):
//   - 0x570e6d0 (sub_2e89d10, ours sfx_wobble_a): count +0x6c, period +0x68, the re-roll's curve +0x60, base 0x2e866f0;
//   - 0x570e690 (sub_2e896f0, ours sfx_wobble_follow): a share (curve +0xa8, or +0xb0) of the parent's move since its last
//     call (the parent's rows kept at +0x70, the parent seen at +0xa0), then 0x570e6d0's work;
//   - 0x570e8d0 (sub_2e8e430, ours sfx_wobble_b): count +0x7c, period +0x78, the re-roll's curve +0x70, base 0x2e8bfd0;
//   - 0x570e890 (sub_2e8de10, the game's, its call of sub_2e8e430 ours): a share (curve +0xb8) of the parent's move since its
//     last call (rows kept at +0x80, the parent seen at +0xb0), its direction +0x10 turned with it, then 0x570e8d0's work.
// After every whole step the count and the FX RNG bit-equal to the game's at 30 (the re-rolls on the same steps with the
// same draws); between whole steps neither moves; on every frame the base motion called once with the frame's step; and
// between whole steps a follow's call every byte the game's own with its count held (the follow's share of the parent's
// move since the last frame, its parent's rows kept up to the frame).
struct WobbleKind {
    const char* name;
    u64 game;                       // the game's update
    MotionUpdate ours;              // what the patched image runs for it
    u64 count, period, curve;       // the call count, its period, the re-roll's curve
    int base;                       // the base motion's call record
    u64 rows, seen, share, fixed;   // a follow: the parent's rows kept, the parent seen, the share's curve, a fixed share (0: none)
};
const WobbleKind kWobbleKinds[] = {
    {"wobble 0x570e6d0", kWobbleA, &sfx_wobble_a, 0x6c, 0x68, 0x60, 10, 0, 0, 0, 0},
    {"follow 0x570e690", kWobbleFollow, &sfx_wobble_follow, 0x6c, 0x68, 0x60, 10, 0x70, 0xa0, 0xa8, 0xb0},
    {"wobble 0x570e8d0", kWobbleB, &sfx_wobble_b, 0x7c, 0x78, 0x70, 11, 0, 0, 0, 0},
    {"follow 0x570e890", kFollowB, &follow_b_patched, 0x7c, 0x78, 0x70, 11, 0x80, 0xb0, 0xb8, 0},
};
struct SeenSW {
    long cases = 0, frames = 0, ticks = 0, rerolls = 0, follows = 0;
    long kinds[4] = {0, 0, 0, 0};
} g_seen_sw;
int check_smooth_wobbles(int cases, u64 sfx) {
    store<u32>(sfx + 0x8c, 0);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const WobbleKind& K = kWobbleKinds[g_rng.range(0, 3)];
        const bool follow = K.rows != 0;
        const auto game = eboot_kit::fn<MotionUpdate>(K.game);
        const u64 inst = block(0x98);
        store<u64>(inst + 0x78, block(8));  // not this node: no root-node stop
        const u64 def = definition(4);
        const u64 n = node(inst, 9u, 3);  // staged, in state 3: one container call an update
        store<u64>(n + 0xc8, controller(0, 4));
        store<u64>(n + 0xd0, def);
        store<u64>(n + 0xd8, 0);
        const u64 motion = block(0x100), parent = block(0x100), local = block(0x40);
        for (int k = 0; k < 0x100; k += 4) store<float>(motion + k, value(-2, 2));
        for (int k = 0; k < 0x40; k += 4) store<float>(local + k, value(-2, 2));
        for (int k = 0x20; k < 0x50; k += 4) store<float>(n + k, value(-50, 50));
        const u32 per = static_cast<u32>(g_rng.range(1, 6));
        store<u32>(motion + K.period, per);
        store<u32>(motion + K.count, static_cast<u32>(g_rng.range(0, static_cast<int>(per) - 1)));
        store<u64>(motion + K.curve, g_rng.chance(0.85) ? curve() : 0);
        // the parent: its rows a rotation and a place moving along a path
        double p0[3], v[3], w0 = value(-1, 1), wr = value(-3, 3);
        for (int i = 0; i < 3; ++i) p0[i] = value(-20, 20), v[i] = value(-8, 8);
        auto put_parent = [&](double t) {
            const double a = w0 + wr * t, c_ = std::cos(a), s_ = std::sin(a);
            const double rows[3][4] = {{c_, 0, s_, p0[0] + v[0] * t}, {0, 1, 0, p0[1] + v[1] * t}, {-s_, 0, c_, p0[2] + v[2] * t}};
            for (int r = 0; r < 3; ++r)
                for (int k = 0; k < 4; ++k) store<float>(parent + 0x20 + 0x10 * r + 4 * k, static_cast<float>(rows[r][k]));
        };
        if (follow) {
            store<u64>(n + 0x60, parent);
            store<u64>(motion + K.seen, parent);
            put_parent(0.0);
            for (int k = 0; k < 0x30; k += 4) store<u32>(motion + K.rows + k, load<u32>(parent + 0x20 + k));
            store<u64>(motion + K.share, !K.fixed || g_rng.chance(0.5) ? curve() : 0);
            if (K.fixed) store<float>(motion + K.fixed, value(0, 1));
        }
        for (int k = 0; k < 0x20; k += 4) store<u32>(reinterpret_cast<u64>(g_rng_state) + k, g_rng.u32() | 1);
        // the game at 30: a copy of the motion, the node and the RNG
        const u64 motion30 = block(0x100), n30 = block(0x100), parent30 = block(0x100);
        std::memcpy(reinterpret_cast<void*>(motion30), reinterpret_cast<void*>(motion), 0x100);
        std::memcpy(reinterpret_cast<void*>(n30), reinterpret_cast<void*>(n), 0x100);
        if (follow) {
            store<u64>(n30 + 0x60, parent30);
            store<u64>(motion30 + K.seen, parent30);
        }
        u8 rng30[0x20], rng[0x20];
        std::memcpy(rng30, g_rng_state, sizeof rng30);
        std::memcpy(rng, g_rng_state, sizeof rng);
        const int rate_kind = g_rng.range(0, 3);
        const int frames = g_rng.range(4, 48);
        double carry = 0.0, tau = 0.0;
        const char* why = nullptr;
        int at = -1;
        ++g_seen_sw.cases;
        ++g_seen_sw.kinds[&K - kWobbleKinds];
        for (int f = 0; f < frames && !why; ++f) {
            const double dt = rate_kind == 0 ? 1.0 / 60 : rate_kind == 1 ? 1.0 / 120 : rate_kind == 2 ? 1.0 / 144 : g_rng.value(1.0f / 150, 1.0f / 40);
            const sfx_schedule::Frame fr = sfx_schedule::next(carry, dt, 30);
            tau += fr.frame;
            sfx_node::Steps st;
            st.smooth = true;
            st.tick = fr.whole > 0;
            st.counted = fr.seconds;
            st.frame = fr.frame;
            st.from = fr.from;
            st.into = fr.into;
            decomp_sfx_gate_frame(!st.tick);
            if (follow) put_parent(tau);
            const u32 count_before = load<u32>(motion + K.count);
            u8 rng_before[0x20];
            std::memcpy(rng_before, rng, sizeof rng_before);
            // between whole steps, the follow's call as the game's own with its count held, on copies
            std::vector<u8> want_motion, want_node;
            if (follow && !st.tick) {
                const u64 m2 = block(0x100), n2 = block(0x100);
                std::memcpy(reinterpret_cast<void*>(m2), reinterpret_cast<void*>(motion), 0x100);
                std::memcpy(reinterpret_cast<void*>(n2), reinterpret_cast<void*>(n), 0x100);
                store<u32>(m2 + K.period, 0xffffffffu);
                // the container call's node time: the frame's start past its whole step (the curves read at it)
                if (st.from != 0.0f) store<float>(n2 + 0x7c, add_ss(load<float>(n + 0x7c), st.from));
                float s2 = st.frame;
                u8 keep[0x20];
                std::memcpy(keep, g_rng_state, sizeof keep);
                std::memcpy(g_rng_state, rng, sizeof rng);
                g_calls.clear();
                game(m2, n2, &s2, local);
                std::memcpy(g_rng_state, keep, sizeof keep);
                store<u32>(m2 + K.period, per);
                store<u32>(m2 + K.count, count_before);
                want_motion.assign(reinterpret_cast<u8*>(m2), reinterpret_cast<u8*>(m2) + 0x100);
                want_node.assign(reinterpret_cast<u8*>(n2), reinterpret_cast<u8*>(n2) + 0x100);
            }
            // the frame domain: the node's update, its container call running ours
            g_wobble_node = n;
            g_wobble_motion = motion;
            g_wobble_local = local;
            g_wobble_fn = K.ours;
            std::memcpy(g_rng_state, rng, sizeof rng);
            g_calls.clear();
            const u32 time_before = load<u32>(n + 0x7c);
            sfx_node::update(n, st, def, block(0x40));
            g_wobble_motion = 0;
            std::memcpy(rng, g_rng_state, sizeof rng);
            ++g_seen_sw.frames;
            // the base motion: once, with the frame's step
            long bases = 0;
            for (const Call& k : g_calls) {
                if (k.what == K.base) {
                    ++bases;
                    if (k.step != bits(st.frame)) why = "the base motion's step (the frame's)";
                }
                if (k.what == 99) why = "sub_2e8de10's call of sub_2e8e430 (not once, with its own arguments)";
            }
            if (!why && bases != 1) why = "the base motion called once a frame";
            if (!why && !st.tick) {
                if (load<u32>(motion + K.count) != count_before) why = "between whole steps: the count";
                else if (std::memcmp(rng, rng_before, sizeof rng) != 0) why = "between whole steps: a draw of the FX RNG";
                else if (follow) {
                    ++g_seen_sw.follows;
                    // the node's time in the call was the frame's (put back after): compare the node but its time
                    std::vector<u8> got(reinterpret_cast<u8*>(n), reinterpret_cast<u8*>(n) + 0x100);
                    std::memcpy(got.data() + 0x7c, want_node.data() + 0x7c, 4);
                    if (std::memcmp(reinterpret_cast<void*>(motion), want_motion.data(), 0x100) != 0) why = "between whole steps: the follow's motion";
                    else if (got != want_node) why = "between whole steps: the follow's node";
                    else if (std::memcmp(reinterpret_cast<void*>(motion + K.rows), reinterpret_cast<void*>(parent + 0x20), 0x30) != 0)
                        why = "between whole steps: the parent's rows kept up to the frame";
                }
            }
            if (!why && st.tick) {
                // the game at 30, once this whole step: its parent at the step's time
                ++g_seen_sw.ticks;
                if (follow) std::memcpy(reinterpret_cast<void*>(parent30 + 0x20), reinterpret_cast<void*>(parent + 0x20), 0x30);
                store<u32>(n30 + 0x7c, time_before);
                float h = fr.seconds;
                std::memcpy(g_rng_state, rng30, sizeof rng30);
                const u32 c30 = load<u32>(motion30 + K.count);
                game(motion30, n30, &h, local);
                std::memcpy(rng30, g_rng_state, sizeof rng30);
                g_seen_sw.rerolls += load<u32>(motion30 + K.count) != c30 + 1;
                if (load<u32>(motion + K.count) != load<u32>(motion30 + K.count)) why = "after a whole step: the count";
                else if (std::memcmp(rng, rng30, sizeof rng) != 0) why = "after a whole step: the FX RNG";
            }
            if (why) at = f;
        }
        decomp_sfx_gate_frame(false);
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("smooth wobble case %d (%s, period %u, rate kind %d): %s differs at frame %d\n", c, K.name, per, rate_kind, why, at);
    }
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

// --- the classifications kept between whole steps ---------------------------------------------

// classify against first_unconverted's own answer, frame by frame, under the real converted list: frames at 120 (a whole
// step, then three between), the instance's nodes and classes changed the ways the game changes them - on a whole
// step a node set up, a binder bound, a module made or let go (any class, off the list too); between them only a
// pending binder taken by the GX side or a GX binder's type made (a ribbon among them). With a whole step the answer
// is the walk's; between them the decision must be the walk's, or held where the walk now finds the instance all on the
// list (its unconverted binder taken since), and a smooth instance's marks hold every type the walk would mark. Then the
// cases the slot's checks are for: an instance not updated on the whole step (the serial), another instance at a
// released or reused address (the shape, the release), a GX binder bound between whole steps (its type to mark).
void cache_classes(u64 n, bool off) {
    const int mk = g_rng.range(0, 2);
    store<u64>(n + 0x90, mk == 0 ? 0 : motion_on(mk - 1));
    store<u64>(n + 0x98, g_rng.chance(0.5) ? object_of(any(kOnModules)) : 0);
    store<u64>(n + 0xa0, 0);
    store<u64>(n + 0xa8, off ? object_of(kOffModule) : g_rng.chance(0.3) ? object_of(any(kOnModules)) : 0);
    store<u64>(n + 0xb0, 0);
    const int pending = g_rng.range(0, 2);
    for (int k = 0; k < pending; ++k) pend(n, g_rng.chance(0.5) ? object_of(kCpuBinder) : gx_binder(any(kOnGxTypes)));
}
// A GX binder whose record has no type yet (the GX side makes it).
u64 unbound_binder() {
    const u64 b = object_of(kGxBinder), holder = block(0x20);
    store<u64>(holder + 8, 0);
    store<u64>(b + 0x10, holder);
    return b;
}
void add_child(u64 inst, u64 n) {
    const u64 root = load<u64>(inst + 0x78);
    store<u64>(n + 0x60, root);
    store<u64>(n + 0x68, 0);
    store<u64>(n + 0x70, 0);
    if (!load<u64>(root + 0x70)) {
        store<u64>(root + 0x70, n);
        return;
    }
    u64 sib = load<u64>(root + 0x70);
    while (load<u64>(sib + 0x68)) sib = load<u64>(sib + 0x68);
    store<u64>(sib + 0x68, n);
}
std::vector<u64> tree_nodes(u64 inst) {
    std::vector<u64> v;
    for (u64 n = load<u64>(inst + 0x78); n; n = node_next(n)) v.push_back(n);
    return v;
}
struct CacheSeen {
    long frames = 0, reused = 0, held_kept = 0;
} g_seen_cache;

// One frame's classify against the walk; `between`: the cached answer may hold an instance the walk finds all on the
// list, and its marks may hold more types. Returns a reason it differs, or nullptr.
const char* cache_frame(u64 inst, bool between) {
    decomp_sfx_gate_frame(between);
    const u64 reused_before = g_class_reused.load();
    u64 m1[kMarks], m2[kMarks];
    int n1 = 0, n2 = 0;
    const u64 k1 = classify(inst, m1, &n1);
    const u64 k2 = first_unconverted(inst, m2, &n2);
    ++g_seen_cache.frames;
    g_seen_cache.reused += static_cast<long>(g_class_reused.load() - reused_before);
    if (!between) {
        if (k1 != k2) return "with a whole step: another class kept";
        if (!k1 && (n1 != n2 || !std::equal(m1, m1 + std::min(n1, kMarks), m2))) return "with a whole step: other marks";
        return nullptr;
    }
    if (!k1 && k2) return "between whole steps: smooth where the walk keeps it";
    if (k1 && !k2) {
        ++g_seen_cache.held_kept;
        return nullptr;
    }
    if (!k1) {
        if (n2 > kMarks) return n1 > kMarks ? nullptr : "between whole steps: too many marks not seen";
        for (int i = 0; i < n2; ++i)
            if (std::find(m1, m1 + std::min(n1, kMarks), m2[i]) == m1 + std::min(n1, kMarks)) return "between whole steps: a type to mark missing";
    }
    return nullptr;
}

int check_class_cache(int cases, u64 sfx) {
    g_converted = kConverted;
    g_converted_count = sizeof kConverted / sizeof kConverted[0];
    decomp_sfx_smooth(true);
    int bad = 0;
    auto fail = [&](int c, const char* what, const char* why) {
        if (++bad <= 6 && !g_quiet) std::printf("class cache %s %d: %s\n", what, c, why);
    };
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        Instance s = make_instance();
        while (s.nodes.empty()) s = make_instance();
        store<u32>(s.inst + 0x88, 1);
        for (const u64 n : s.nodes) cache_classes(n, g_rng.chance(0.15));
        const char* why = nullptr;
        for (int f = 0; f < 12 && !why; ++f) {
            const bool between = f % 4 != 0;
            std::vector<u64> nodes = tree_nodes(s.inst);
            if (!between) {
                // a whole step: what the game sets up or lets go then
                const int what = g_rng.range(0, 5);
                if (what == 0) {
                    const u64 n = node(s.inst, 1, 3);
                    cache_classes(n, g_rng.chance(0.2));
                    add_child(s.inst, n);
                } else if (what == 1) {
                    const u64 n = nodes[g_rng.range(0, static_cast<int>(nodes.size()) - 1)];
                    pend(n, g_rng.chance(0.15) ? object_of(kOffModule) : g_rng.chance(0.4) ? unbound_binder() : gx_binder(any(kOnGxTypes)));
                } else if (what == 2) {
                    const u64 n = nodes[g_rng.range(0, static_cast<int>(nodes.size()) - 1)];
                    store<u64>(n + 0xa8, g_rng.chance(0.3) ? object_of(kOffModule) : 0);
                }
            } else {
                // between whole steps: a pending binder taken, or a GX binder's type made
                const u64 n = nodes[g_rng.range(0, static_cast<int>(nodes.size()) - 1)];
                if (g_rng.chance(0.3) && load<u64>(n + 0xb0)) store<u64>(n + 0xb0, load<u64>(load<u64>(n + 0xb0) + 0x18));
                for (u64 item = load<u64>(n + 0xb0); item; item = load<u64>(item + 0x18)) {
                    const u64 t = load<u64>(item + 0x20);
                    if (!t || load<u64>(t) != game_address(kGxBinder)) continue;
                    const u64 holder = load<u64>(t + 0x10);
                    if (holder && !load<u64>(holder + 8) && g_rng.chance(0.5)) {
                        const int kind = g_rng.range(0, 3);
                        store<u64>(holder + 8, object_of(kind == 0 ? kGxTypeRibbon : kind == 1 ? kOffGxType : any(kOnGxTypes)));
                    }
                }
            }
            why = cache_frame(s.inst, between);
        }
        if (why) fail(c, "case", why);
    }
    // the serial: classified between whole steps, not updated on the next whole step, given a class off the list then
    for (int c = 0; c < 200; ++c) {
        g_blocks.clear();
        Instance s = make_instance();
        while (s.nodes.empty()) s = make_instance();
        store<u32>(s.inst + 0x88, 1);
        for (const u64 n : s.nodes) cache_classes(n, false);
        if (cache_frame(s.inst, false) || cache_frame(s.inst, true)) continue;
        decomp_sfx_gate_frame(false);  // the next whole step, without this instance
        store<u64>(s.nodes[0] + 0xa8, object_of(kOffModule));
        u64 m[kMarks];
        int n = 0;
        decomp_sfx_gate_frame(true);
        if (!classify(s.inst, m, &n)) {
            fail(c, "serial", "a classification kept from the last whole step");
            break;
        }
    }
    // the shape: another instance made at the address between whole steps (a new definition and node tree)
    for (int c = 0; c < 200; ++c) {
        g_blocks.clear();
        Instance s = make_instance();
        while (s.nodes.empty()) s = make_instance();
        store<u32>(s.inst + 0x88, 1);
        for (const u64 n : s.nodes) cache_classes(n, false);
        if (cache_frame(s.inst, false) || cache_frame(s.inst, true)) continue;
        store<u64>(s.inst + 0x38, definition(2));
        const u64 root = node(s.inst, 1, 3);
        cache_classes(root, true);
        store<u64>(s.inst + 0x78, root);
        u64 m[kMarks];
        int n = 0;
        decomp_sfx_gate_frame(true);
        if (!classify(s.inst, m, &n)) {
            fail(c, "shape", "another instance at the address took the old one's classification");
            break;
        }
    }
    // the release: the instance released and another made at its address with the same definition and nodes, one of
    // its classes off the list
    for (int c = 0; c < 200; ++c) {
        g_blocks.clear();
        Instance s = make_instance();
        while (s.nodes.empty()) s = make_instance();
        store<u32>(s.inst + 0x88, 1);
        for (const u64 n : s.nodes) cache_classes(n, false);
        if (cache_frame(s.inst, false) || cache_frame(s.inst, true)) continue;
        const u64 def = load<u64>(s.inst + 0x38), root = load<u64>(s.inst + 0x78), child = load<u64>(root + 0x70);
        store<u32>(s.inst + 0x88, 1 | 4);
        sfx_instance_update(s.inst, s.ctx, 0.0f);  // its last update: the game's lets its tree go
        store<u32>(s.inst + 0x88, 1);
        store<u64>(s.inst + 0x38, def);
        store<u64>(s.inst + 0x78, root);
        store<u64>(root + 0x60, 0);
        store<u64>(root + 0x68, 0);
        store<u64>(root + 0x70, child);
        if (child) {
            store<u64>(child + 0x60, root);
            store<u64>(child + 0x68, 0);
            store<u64>(child + 0x70, 0);
        }
        store<u32>(root + 0x78, 1);
        cache_classes(root, true);
        u64 m[kMarks];
        int n = 0;
        decomp_sfx_gate_frame(true);
        if (!classify(s.inst, m, &n)) {
            fail(c, "release", "an instance made at a released one's address took its classification");
            break;
        }
    }
    // a GX binder bound between whole steps: its ribbon marked on that frame
    for (int c = 0; c < 200; ++c) {
        g_blocks.clear();
        Instance s = make_instance();
        while (s.nodes.empty()) s = make_instance();
        store<u32>(s.inst + 0x88, 1);
        for (const u64 n : s.nodes) cache_classes(n, false);
        const u64 b = unbound_binder();
        pend(s.nodes[0], b);
        if (cache_frame(s.inst, false) || cache_frame(s.inst, true)) continue;
        const u64 ribbon = object_of(kGxTypeRibbon);
        store<u64>(load<u64>(b + 0x10) + 8, ribbon);
        u64 m[kMarks];
        int n = 0;
        decomp_sfx_gate_frame(true);
        const u64 k = classify(s.inst, m, &n);
        if (k || std::find(m, m + std::min(n, kMarks), ribbon) == m + std::min(n, kMarks)) {
            fail(c, "binding", "a GX binder bound between whole steps: its ribbon not marked");
            break;
        }
    }
    decomp_sfx_gate_frame(false);
    decomp_sfx_smooth(false);
    if (!g_quiet)
        std::printf("decomp_sfx_step_gate: %d class cache cases (%ld frames, %ld classifications taken from the whole step, %ld held a "
                    "frame longer than the walk would), %d differ\n",
                    cases, g_seen_cache.frames, g_seen_cache.reused, g_seen_cache.held_kept, bad);
    return bad;
}

// --- GPU particle type 0x2713 under the smooth steps -------------------------------------

// A smooth instance's 0x2713 record (marked for the frame): its type reads its curves at the whole step's elapsed time
// (1/30 on the manager's and the instance's scales) - the game's call with that step - and the record's times end as the
// frame's step leaves them, the previous one where the draw reads the same elapsed time; an unmarked one (a held
// instance's, a pre-roll's) or the game's own still frame, the game's call as it is.
struct Seen2713 {
    long cases = 0, whole = 0, game = 0, prev_nudged = 0;
} g_seen_2713;
int check_records_smooth(int cases, u64 sfx, u64 manager) {
    const auto game = eboot_kit::fn<GxRecordFn>(kGxRecordUpdate);
    int bad = 0;
    decomp_sfx_smooth(true);
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const Record r = make_record(true);
        const u64 node = load<u64>(r.record + 0x10), inst = block(0x98);
        store<u64>(node + 0x58, inst);
        const float scale = g_rng.chance(0.6) ? 1.0f : value(0.5f, 2.0f), time_scale = g_rng.chance(0.7) ? 1.0f : value(0.5f, 1.5f);
        store<float>(inst + 8, scale);
        store<float>(manager + 0x5d8, time_scale);
        const bool still = g_rng.chance(0.1), marked = g_rng.chance(0.8), between = g_rng.chance(0.6);
        store<u32>(sfx + 0x8c, still ? 1u : 0u);
        const float frame = value(1.0f / 240, 1.0f / 31);
        decomp_sfx_step_frame(between);
        decomp_sfx_frame_time(frame, 0.0f, 0.0f);
        if (marked) ribbon_mark(r.type);
        const float step = frame * time_scale * scale;
        store<float>(r.item + 0x60, step);
        const std::vector<u8> before = record_bytes(r);
        const bool takes = marked && !still;
        const float whole = 1.0f / 30.0f * time_scale * scale;
        // the reference: the game's call with the whole step (when it takes), else with the item as it is
        if (takes) store<float>(r.item + 0x60, whole);
        game(r.record, r.item);
        store<float>(r.item + 0x60, step);
        const std::vector<u8> theirs = record_bytes(r);
        put_record(r, before);
        sfx_gx_record_update(r.record, r.item);
        const std::vector<u8> ours = record_bytes(r);
        ++g_seen_2713.cases;
        const char* why = nullptr;
        if (!takes) {
            ++g_seen_2713.game;
            if (ours != theirs) why = "not marked, or the still frame: the game's call";
        } else {
            ++g_seen_2713.whole;
            // the type and every record byte but the two times as the game's call with the whole step
            std::vector<u8> want = theirs;
            std::memcpy(want.data() + 0x24, ours.data() + 0x24, 8);
            const float c0 = load<float>(reinterpret_cast<u64>(before.data()) + 0x28);
            const float cur = load<float>(reinterpret_cast<u64>(ours.data()) + 0x28), prev = load<float>(reinterpret_cast<u64>(ours.data()) + 0x24);
            const float elapsed = sub_ss(load<float>(reinterpret_cast<u64>(theirs.data()) + 0x28), c0);
            if (ours != want) why = "the type's values (the curves at the whole step's elapsed time)";
            else if (bits(cur) != bits(add_ss(c0, step))) why = "the record's time (the frame's step on it)";
            else if (bits(sub_ss(cur, prev)) != bits(elapsed)) why = "the elapsed time the draw reads";
            g_seen_2713.prev_nudged += bits(prev) != bits(sub_ss(cur, elapsed));
        }
        put_record(r, before);
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("smooth 0x2713 record case %d (marked %d, still %d, scale %g x %g): %s differs\n", c, marked, still, scale, time_scale, why);
    }
    decomp_sfx_smooth(false);
    decomp_sfx_step_frame(false);
    store<u32>(sfx + 0x8c, 0u);
    return bad;
}

// --- GPU particle emission under the smooth steps ------------------------------------------
//
// StandardParticle's and Spark's simulations (the game's sub_13040b0 and sub_12fef10) under a type update of the
// test's (their vtables' +0x10: the record's elapsed time handed to the simulation at the type +0x6b0, as sub_12f18f0
// and sub_12ed0b0 hand it), through the game's record update and ours. An emitter a case: a fixed count and interval
// (no draw changes them), sometimes a cap within reach, +0x3940 not 1, an end time or a limit of events. The game's
// record update a whole step at a time (1/30 on the manager's and the instance's scales) against ours over the frame
// domain's frames of each whole step (2, 3, 4, 5 or 8 of them, or an uneven split): what the frames emit (+0x3878, taken
// and zeroed after each update as the dispatch takes it) the whole step's emission, and at its end the same events, the
// emitter on or off as the whole step leaves it and its timer within the frames' sums' rounding (0 after a cap). And
// frames just under a whole step, one update a window: every byte of the emitter and the FX RNG as the game's call with
// the same step.
using SimFn = GUEST_ABI void (*)(u64 sim, float dt);
GUEST_ABI void stub_standard_type(u64 out, u64 type, u64 rec, u64) {
    eboot_kit::fn<SimFn>(0x13040b0)(type + kTypeSimulation, sub_ss(load<float>(rec + 0x18), load<float>(rec + 0x14)));
    std::memset(reinterpret_cast<void*>(out), 0, 32);
}
GUEST_ABI void stub_spark_type(u64 out, u64 type, u64 rec, u64) {
    eboot_kit::fn<SimFn>(0x12fef10)(type + kTypeSimulation, sub_ss(load<float>(rec + 0x18), load<float>(rec + 0x14)));
    std::memset(reinterpret_cast<void*>(out), 0, 32);
}
alignas(64) u8 g_emit_rng[0x80];
struct ParticleCase {
    u64 record, item, type, sim;
};
constexpr std::size_t kSimBytes = 0x3b40;
ParticleCase make_particles(bool spark, float scale, float whole, int kind) {
    ParticleCase e;
    e.record = block(0x80);
    e.item = block(0x70);
    e.type = block(kTypeSimulation + kSimBytes);
    const u64 node = block(0x100), counts = block(0x20), inst = block(0x98);
    store<u64>(e.type, game_address(spark ? kGxSpark : kGxStandard));
    store<u64>(e.type + 8, block(0x100));
    store<u64>(e.record + 8, e.type);
    store<u64>(e.record + 0x10, node);
    store<u64>(e.record + 0x18, counts);  // +8 = 0: no buffers
    store<u64>(node + 0x58, inst);
    store<float>(inst + 8, scale);
    // the record's time small: its rounding moves the elapsed time a type reads off the item's step, and the timer sums
    // with it (the windows count the steps themselves)
    const float t = value(0, 0.5f);
    store<float>(e.record + 0x24, t);
    store<float>(e.record + 0x28, t);
    for (int k = 0; k < 0x60; k += 4) store<float>(e.item + k, value(-3, 3));
    const u64 sim = e.sim = e.type + kTypeSimulation;
    store<u8>(sim + 0x3860, 0);
    store<u32>(sim + 0x2a1c, 0);
    store<u32>(sim + 0x283c, 0xfffffff);
    store<u8>(sim + 0x3872, 1);
    // the interval: just over the re-roll's floor (the dense ones; at 0.01 itself 3 x 1/30 = 10 x 0.01 makes a timer a
    // cap set to 0 meet an interval exactly, where rounding - 30's or the frames' - decides the event's step) or wider; the
    // re-roll gives it again (+0x2604 = +0x2608 = 0)
    const float interval = kind == 0 || g_rng.chance(0.4) ? value(0.0101f, 0.0104f) : value(0.0105f, 0.12f);
    // a new emitter: its set-up's interval (0.01, sub_12fda09), its events at that for 30's first whole step, then the
    // definition's from the first re-roll on
    const bool fresh = g_rng.chance(0.4);
    store<float>(sim + 0x3888, fresh ? 0.01f : interval);
    store<float>(sim + 0x2600, interval);
    store<float>(sim + 0x2604, 0.0f);
    store<float>(sim + 0x2608, 0.0f);
    const float sigma = g_rng.chance(0.7) ? 1.0f : value(0.3f, 1.5f);
    store<float>(sim + 0x393c, sigma);
    // a count an event: within reach of the cap (16 or more, at the floor), or small
    const int count = kind == 0 ? g_rng.range(16, 120) : g_rng.range(0, 12);
    store<u32>(sim + 0x25e0, static_cast<u32>(count));
    store<u32>(sim + 0x25e4, static_cast<u32>(count));
    store<u32>(sim + 0x25dc, g_rng.chance(0.7) ? 0u : static_cast<u32>(g_rng.range(0, 30)));
    store<u32>(sim + 0x3944, g_rng.chance(0.8) ? 0u : static_cast<u32>(g_rng.range(0, 6)));
    store<u32>(sim + 0x29c4, g_rng.chance(0.6) ? 0u : static_cast<u32>(g_rng.range(0, 40)));
    store<float>(sim + 0x29a8, g_rng.chance(0.6) ? 1.0f : value(0.0f, 3.0f));
    store<float>(sim + 0x29ac, g_rng.chance(0.6) ? 1.0f : value(0.0f, 2.0f));
    store<u32>(sim + 0x2890, static_cast<u32>(g_rng.chance(0.5) ? 2048 : g_rng.range(0, 64)));
    // what the wind leaves of it: 1, or a share that truncates
    store<float>(sim + 0x3940, kind == 2 || g_rng.chance(0.3) ? value(0.15f, 2.5f) : 1.0f);
    // the limits: none, or an end time, or a limit of events
    const bool limits = kind == 3 || g_rng.chance(0.25);
    store<u32>(sim + 0x253c, limits ? 0u : 1u);
    // an end time inside a whole step, away from its ends (where the frames' sums and 30's could round apart)
    const float own = fresh || g_rng.chance(0.5) ? 0.0f : static_cast<float>(g_rng.range(1, 6)) * whole;
    store<float>(sim + 0x2540, limits && g_rng.chance(0.6) ? own + (static_cast<float>(g_rng.range(0, 14)) + value(0.1f, 0.9f)) * whole : 1e9f);
    store<u32>(sim + 0x261c, limits && g_rng.chance(0.6) ? 1u : 0u);
    store<u32>(sim + 0x2620, static_cast<u32>(g_rng.range(0, 16)));
    store<u32>(sim + 0x3884, static_cast<u32>(fresh || g_rng.chance(0.7) ? 0 : g_rng.range(0, 6)));
    store<float>(sim + 0x3880, fresh ? 0.0f : value(0.0f, interval * sigma));  // as an update leaves it
    // its own time: new, or some whole steps on (as 30's updates leave it)
    store<float>(sim + 0x387c, own);
    store<float>(sim + 0x3a2c, value(0, 100));
    return e;
}
std::vector<u8> sim_bytes(const ParticleCase& e) {
    std::vector<u8> v(kSimBytes + 0x80);
    std::memcpy(v.data(), reinterpret_cast<void*>(e.sim), kSimBytes);
    std::memcpy(v.data() + kSimBytes, reinterpret_cast<void*>(e.record), 0x80);
    return v;
}
void put_sim(const ParticleCase& e, const std::vector<u8>& v) {
    std::memcpy(reinterpret_cast<void*>(e.sim), v.data(), kSimBytes);
    std::memcpy(reinterpret_cast<void*>(e.record), v.data() + kSimBytes, 0x80);
}
void emit_rng_seed(u32 seed) {
    std::memset(g_emit_rng, 0, sizeof g_emit_rng);
    for (int k = 0x30; k < 0x40; k += 4) store<u32>(reinterpret_cast<u64>(g_emit_rng) + k, seed * 2654435761u + static_cast<u32>(k) * 40503u + 1u);
}
struct EmitSeen {
    long cases = 0, steps = 0, frames = 0, capped = 0, truncated = 0, ended = 0, limited = 0, single = 0;
} g_seen_emit;
int check_emission(int cases, u64 sfx, u64 manager) {
    const auto game = eboot_kit::fn<GxRecordFn>(kGxRecordUpdate);
    int bad = 0;
    decomp_sfx_smooth(true);
    store<u32>(sfx + 0x8c, 0u);
    store<u8>(manager + 0x5ce, 0);
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        // a new emitter (the last case's may have had its address, with its own time behind this one's)
        for (EmitSlot& x : g_emit_slots) {
            x.sim.store(0);
            x.open = 0;
        }
        const bool spark = g_rng.chance(0.5);
        const float scale = g_rng.chance(0.7) ? 1.0f : value(0.5f, 2.0f), time_scale = g_rng.chance(0.8) ? 1.0f : value(0.5f, 1.5f);
        store<float>(manager + 0x5d8, time_scale);
        const float whole = 1.0f / 30.0f * time_scale * scale;
        const int kind = g_rng.range(0, 3);  // 0 a cap within reach, 1 small counts, 2 a share that truncates, 3 limits
        const ParticleCase e = make_particles(spark, scale, whole, kind);
        const std::vector<u8> start = sim_bytes(e);
        const u32 seed = g_rng.u32();
        const char* why = nullptr;
        int at = -1;
        ++g_seen_emit.cases;
        if (g_rng.chance(0.15)) {
            // one update a window: frames just under a whole step, ours against the game's call with the same step (no
            // limits: a window's end time stops an emitter at its whole step's end, not at the update's)
            ++g_seen_emit.single;
            store<u32>(e.sim + 0x253c, 1u);
            const std::vector<u8> start1 = sim_bytes(e);
            const float frame = 1.0f / 30.0f * 0.999999f;
            const float step = frame * time_scale * scale;
            std::vector<u8> theirs[12];
            u8 rng_theirs[12][16];
            emit_rng_seed(seed);
            for (int k = 0; k < 12; ++k) {
                store<float>(e.item + 0x60, step);
                game(e.record, e.item);
                theirs[k] = sim_bytes(e);
                std::memcpy(rng_theirs[k], g_emit_rng + 0x30, 16);
                store<u32>(e.sim + 0x3878, 0);
            }
            put_sim(e, start1);
            emit_rng_seed(seed);
            for (int k = 0; k < 12 && !why; ++k) {
                decomp_sfx_frame_time(frame, 0.0f, 0.0f);
                store<float>(e.item + 0x60, step);
                sfx_gx_record_update(e.record, e.item);
                if (sim_bytes(e) != theirs[k]) why = "one update a window: the emitter";
                else if (std::memcmp(rng_theirs[k], g_emit_rng + 0x30, 16) != 0) why = "one update a window: the FX RNG";
                if (why) at = k;
                store<u32>(e.sim + 0x3878, 0);
            }
        } else {
            // the reference: the game's update a whole step at a time
            constexpr int kSteps = 16;
            s32 emitted[kSteps];
            u32 events[kSteps], on[kSteps];
            float timer[kSteps];
            emit_rng_seed(seed);
            for (int k = 0; k < kSteps; ++k) {
                store<float>(e.item + 0x60, whole);
                game(e.record, e.item);
                emitted[k] = load<s32>(e.sim + 0x3878);
                store<u32>(e.sim + 0x3878, 0);
                events[k] = load<u32>(e.sim + 0x3884);
                on[k] = load<u8>(e.sim + 0x3872);
                timer[k] = load<float>(e.sim + 0x3880);
            }
            // ours: the frames of each whole step, as the frame domain gives them
            put_sim(e, start);
            emit_rng_seed(seed ^ 0x9e3779b9u);
            const int split = g_rng.range(0, 5);
            const int ns[5] = {2, 3, 4, 5, 8};
            for (int k = 0; k < kSteps && !why; ++k) {
                float parts[8];
                int n = 0;
                if (split < 5 && g_rng.chance(0.75)) {
                    n = ns[split];
                    for (int j = 0; j < n; ++j) parts[j] = whole / static_cast<float>(n);
                } else {
                    n = g_rng.range(2, 6);
                    float left = whole;
                    for (int j = 0; j < n - 1; ++j) {
                        parts[j] = left * value(0.15f, 0.6f);
                        left -= parts[j];
                    }
                    parts[n - 1] = left;
                }
                s32 sum = 0;
                for (int j = 0; j < n; ++j) {
                    // the frame's own time on the instance's scales: (frame x time scale) x scale is the part
                    const float frame = parts[j] / scale / time_scale;
                    decomp_sfx_frame_time(frame, 0.0f, 0.0f);
                    const float step = frame * time_scale * scale;
                    store<float>(e.item + 0x60, step);
                    sfx_gx_record_update(e.record, e.item);
                    sum += load<s32>(e.sim + 0x3878);
                    store<u32>(e.sim + 0x3878, 0);
                    ++g_seen_emit.frames;
                }
                ++g_seen_emit.steps;
                if (sum != emitted[k]) why = "what a whole step's frames emit";
                else if (load<u32>(e.sim + 0x3884) != events[k]) why = "the events at a whole step's end";
                else if (load<u8>(e.sim + 0x3872) != on[k]) why = "the emitter on or off at a whole step's end";
                else {
                    const float t = load<float>(e.sim + 0x3880);
                    if (timer[k] == 0.0f ? t != 0.0f : !(std::fabs(t - timer[k]) <= 2e-5f)) why = "the timer at a whole step's end";
                }
                if (why) at = k;
                if (timer[k] == 0.0f && emitted[k] > 0) ++g_seen_emit.capped;
            }
            if (load<float>(e.sim + 0x3940) != 1.0f) ++g_seen_emit.truncated;
            if (load<u32>(e.sim + 0x253c) == 0) {
                ++g_seen_emit.limited;
                if (!on[kSteps - 1]) ++g_seen_emit.ended;
            }
        }
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("emission case %d (%s, kind %d, count %u, interval %g, cap fields %u/%u, x%g, limits %u end %g max %u/%u): %s differs at step %d\n", c,
                        spark ? "Spark" : "StandardParticle", kind, load<u32>(reinterpret_cast<u64>(start.data()) + 0x25e0),
                        load<float>(reinterpret_cast<u64>(start.data()) + 0x3888), load<u32>(reinterpret_cast<u64>(start.data()) + 0x29c4),
                        load<u32>(reinterpret_cast<u64>(start.data()) + 0x2890), load<float>(reinterpret_cast<u64>(start.data()) + 0x3940),
                        load<u32>(reinterpret_cast<u64>(start.data()) + 0x253c), load<float>(reinterpret_cast<u64>(start.data()) + 0x2540),
                        load<u32>(reinterpret_cast<u64>(start.data()) + 0x261c), load<u32>(reinterpret_cast<u64>(start.data()) + 0x2620), why, at);
    }
    decomp_sfx_smooth(false);
    return bad;
}

// Emitters whose interval the re-roll draws at random (+0x2604..+0x2608): 30's update takes every event of its step at
// one interval, frames re-roll after each of theirs, so the game's own update a frame emits less where the interval is
// short (an interval of 0..0.1 s: about 27 events a second at 30, 20 at 120). Over 40 s, the window's frames against the
// game's whole steps: the same events a second within 3% (the draws differ), and the game's own frames for comparison.
struct RateSeen {
    double game30 = 0, ours = 0, theirs = 0;
    int emitters = 0;
} g_seen_rates;
int check_emission_rates(u64 sfx, u64 manager) {
    const auto game = eboot_kit::fn<GxRecordFn>(kGxRecordUpdate);
    int bad = 0;
    decomp_sfx_smooth(true);
    store<u32>(sfx + 0x8c, 0u);
    store<u8>(manager + 0x5ce, 0);
    store<float>(manager + 0x5d8, 1.0f);
    const struct {
        float base, hi;
    } kinds[] = {{0.0f, 0.1f}, {0.0f, 0.03f}, {0.02f, 0.06f}, {0.01f, 0.05f}};
    for (const auto& k : kinds)
        for (const int fps : {60, 120, 144}) {
            long n30 = 0, nours = 0, ntheirs = 0;
            for (int rep = 0; rep < 8; ++rep) {
                g_blocks.clear();
                for (EmitSlot& x : g_emit_slots) {
                    x.sim.store(0);
                    x.open = 0;
                }
                ParticleCase e = make_particles(g_rng.chance(0.5), 1.0f, 1.0f / 30.0f, 1);
                store<u32>(e.sim + 0x25e0, 1u);
                store<u32>(e.sim + 0x25e4, 1u);
                for (const u64 off : {0x25dcull, 0x3944ull, 0x29c4ull}) store<u32>(e.sim + off, 0u);
                store<u32>(e.sim + 0x253c, 1u);
                store<float>(e.sim + 0x3940, 1.0f);
                store<float>(e.sim + 0x393c, 1.0f);
                store<u32>(e.sim + 0x2890, 4096u);
                store<float>(e.sim + 0x2600, k.base);
                store<float>(e.sim + 0x2604, 0.0f);
                store<float>(e.sim + 0x2608, k.hi);
                const std::vector<u8> start = sim_bytes(e);
                const float frame = 1.0f / static_cast<float>(fps);
                emit_rng_seed(g_rng.u32());
                for (int f = 0; f < 40 * 30; ++f) {
                    store<float>(e.item + 0x60, 1.0f / 30.0f);
                    game(e.record, e.item);
                    n30 += load<s32>(e.sim + 0x3878);
                    store<u32>(e.sim + 0x3878, 0);
                }
                for (int arm = 0; arm < 2; ++arm) {
                    put_sim(e, start);
                    emit_rng_seed(g_rng.u32());
                    for (int f = 0; f < 40 * fps; ++f) {
                        decomp_sfx_frame_time(frame, 0.0f, 0.0f);
                        store<float>(e.item + 0x60, frame);
                        if (arm == 0) sfx_gx_record_update(e.record, e.item);
                        else game(e.record, e.item);
                        (arm == 0 ? nours : ntheirs) += load<s32>(e.sim + 0x3878);
                        store<u32>(e.sim + 0x3878, 0);
                    }
                }
            }
            g_seen_rates.game30 += n30;
            g_seen_rates.ours += nours;
            g_seen_rates.theirs += ntheirs;
            ++g_seen_rates.emitters;
            const double r = static_cast<double>(nours) / static_cast<double>(n30 ? n30 : 1);
            if (!(r > 0.97 && r < 1.03) && ++bad <= 6 && !g_quiet)
                std::printf("emission rate (interval %g + 0..%g, %d fps): %.2f a second against 30's %.2f (the game's frames %.2f)\n", k.base, k.hi, fps,
                            nours / 320.0, n30 / 320.0, ntheirs / 320.0);
        }
    decomp_sfx_smooth(false);
    return bad;
}

// --- the wind sources' registry step ---------------------------------------------------

// A registry (*0x58b0bb0): +0x28 a list's sentinel, each node's +0 the next and +0x10 its entry, an entry's +0x48 its
// object - a wind source (0x56ec140, stepped by sub_2d552d0), a default object (0x56ebf70, steps nothing) or none.
struct WindRegistry {
    u64 registry;
    std::vector<u64> sources;
};
WindRegistry make_wind_registry() {
    WindRegistry w;
    w.registry = block(0x110);
    const u64 head = block(0x20);
    store<u64>(w.registry + 0x28, head);
    u64 last = head;
    for (int i = g_rng.range(0, 8); i > 0; --i) {
        const u64 node = block(0x20), entry = block(0x60);
        store<u64>(last, node);
        last = node;
        store<u64>(node + 0x10, entry);
        const double kind = g_rng.value(0, 1);
        if (kind < 0.6) {
            const u64 src = block(0x240);
            for (int k = 8; k < 0x240; k += 4) store<float>(src + k, value(-50, 50));
            store<u64>(src, game_address(0x56ec140));
            store<float>(src + 0x1a8, value(0, 3));
            store<float>(src + 0x1f4, value(0, 100));
            store<u64>(entry + 0x48, src);
            w.sources.push_back(src);
        } else if (kind < 0.8) {
            const u64 obj = block(0xd0);
            store<u64>(obj, game_address(0x56ebf70));
            store<u64>(entry + 0x48, obj);
        } else {
            store<u64>(entry + 0x48, 0);
        }
    }
    store<u64>(last, head);  // circular: the last node's next the sentinel
    return w;
}
std::vector<u8> wind_bytes(const WindRegistry& w) {
    std::vector<u8> v;
    for (const u64 src : w.sources) {
        const std::size_t at = v.size();
        v.resize(at + 0x240);
        std::memcpy(v.data() + at, reinterpret_cast<void*>(src), 0x240);
    }
    return v;
}
void put_wind(const WindRegistry& w, const std::vector<u8>& v) {
    std::size_t at = 0;
    for (const u64 src : w.sources) {
        std::memcpy(reinterpret_cast<void*>(src), v.data() + at, 0x240);
        at += 0x240;
    }
}
struct WindSeen {
    long cases = 0, smooth = 0, other = 0, seconds = 0;
} g_seen_wind;

// The registry step (sub_1314660 as ours) against the game's: the smooth steps off, the game's own still frame, or a
// step that is not the manager's (its whole step on its time scale, or none), the game's call as it is; on, the game's
// call with the frame's own time on the manager's scale - every byte of every source. And over a second, the sources'
// ages and gust phases the same at 60, 120, 144 and jittered rates as at 30, within the adds' rounding.
int check_wind_sources(int cases, u64 sfx, u64 manager) {
    const auto game = eboot_kit::fn<WindStepFn>(kWindSourcesStep);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const WindRegistry w = make_wind_registry();
        const bool smooth = g_rng.chance(0.8), still = g_rng.chance(0.1), between = g_rng.chance(0.6), other = g_rng.chance(0.1);
        const float time_scale = g_rng.chance(0.6) ? 1.0f : value(0.25f, 2.0f), frame = value(1.0f / 240, 1.0f / 30);
        store<float>(manager + 0x5d8, time_scale);
        store<u32>(sfx + 0x8c, still ? 1u : 0u);
        decomp_sfx_smooth(smooth);
        decomp_sfx_step_frame(between);
        decomp_sfx_frame_time(frame, 0.0f, 0.0f);
        const float manager_step = (between ? 0.0f : 1.0f / 30.0f) * time_scale;
        const float step = other ? value(-1, 1) : manager_step;
        const bool takes = smooth && !still && bits(step) == bits(manager_step);
        const float want = takes ? frame * time_scale : step;
        const std::vector<u8> before = wind_bytes(w);
        game(w.registry, want);
        const std::vector<u8> theirs = wind_bytes(w);
        put_wind(w, before);
        sfx_wind_sources_step(w.registry, step);
        const std::vector<u8> ours = wind_bytes(w);
        ++g_seen_wind.cases;
        g_seen_wind.smooth += takes;
        g_seen_wind.other += smooth && !still && !takes;
        if (ours != theirs && ++bad <= 6 && !g_quiet)
            std::printf("wind sources case %d (%zu sources, smooth %d, still %d, between %d, another's step %d, scale %g): the step differs\n", c,
                        w.sources.size(), smooth, still, between, other, time_scale);
    }
    // a second of frames against 30 whole steps
    store<u32>(sfx + 0x8c, 0);
    for (int c = 0; c < cases / 10; ++c) {
        g_blocks.clear();
        const WindRegistry w = make_wind_registry();
        if (w.sources.empty()) continue;
        const float time_scale = g_rng.chance(0.6) ? 1.0f : value(0.25f, 2.0f);
        store<float>(manager + 0x5d8, time_scale);
        const std::vector<u8> before = wind_bytes(w);
        decomp_sfx_smooth(false);
        for (int k = 0; k < 30; ++k) game(w.registry, 1.0f / 30.0f * time_scale);
        const std::vector<u8> at30 = wind_bytes(w);
        put_wind(w, before);
        decomp_sfx_smooth(true);
        const int rate_kind = g_rng.range(0, 3);
        double carry = 0.0, total = 0.0;
        int frames = 0;
        while (total < 1.0 - 1e-9 && frames < 1000) {
            double dt = rate_kind == 0 ? 1.0 / 60 : rate_kind == 1 ? 1.0 / 120 : rate_kind == 2 ? 1.0 / 144 : g_rng.value(1.0f / 150, 1.0f / 40);
            if (total + dt > 1.0) dt = 1.0 - total;
            const sfx_schedule::Frame fr = sfx_schedule::next(carry, dt, 30);
            total += dt;
            ++frames;
            decomp_sfx_step_frame(fr.whole == 0);
            decomp_sfx_frame_time(fr.frame, fr.from, fr.into);
            sfx_wind_sources_step(w.registry, static_cast<float>(fr.whole) / 30.0f * time_scale);
        }
        decomp_sfx_smooth(false);
        decomp_sfx_step_frame(false);
        ++g_seen_wind.seconds;
        for (std::size_t i = 0; i < w.sources.size(); ++i) {
            const u64 at = 0x240 * i;
            for (const u64 off : {0x1f4ull, 0x1b4ull}) {
                const double a = load<float>(reinterpret_cast<u64>(at30.data()) + at + off), b = load<float>(w.sources[i] + off);
                const double scale = std::fabs(a) + 1.0;
                if (std::fabs(a - b) > 2e-6 * frames * scale && ++bad <= 6 && !g_quiet)
                    std::printf("wind sources second %d (rate kind %d, %d frames): +0x%llx %.9g against 30's %.9g\n", c, rate_kind, frames,
                                static_cast<unsigned long long>(off), b, a);
            }
        }
    }
    return bad;
}

// --- the wind system's warm-up ---------------------------------------------------

// The wind system (*0x5940e00) as its receivers' update (sub_27174e0, the game's own here) reads it, with no receivers:
// its list +0x18 empty, its counts +0x20 and +0x38 0, the reset's one-shot +0x4c, the warm-up +0x4d and its count +0x50;
// the thread's runtime heap (rel_dlruntime_obj_mb) and gettimeofday the test's. Frame sequences at 60, 120, 144 and
// jittered rates through ours, the gate on or off, against the game's once a whole step (at 30): after every whole step
// the system's bytes the game's at 30 after as many whole steps (the warm-up ends on the same whole step: 60 of them,
// 2 s); between whole steps +0x4d and +0x50 untouched and every other byte the game's own call's (+0x4c cleared by the
// first call, as the game clears it).
u8* runtime_caps(u8* out, void*, int) {
    std::memset(out, 0, 8);
    out[0] = 0x20;  // a heap containers may use
    return out;
}
void* runtime_alloc(void*, std::size_t size, std::size_t) { return std::malloc(size ? size : 1); }
void runtime_free(void*, void* p) { std::free(p); }
void* g_runtime_vt[16];
alignas(16) u8 g_runtime[0x100];
GUEST_ABI int stub_gettimeofday(void* tv, void*) {
    std::memset(tv, 0, 16);
    return 0;
}
struct SeenWarmup {
    long cases = 0, frames = 0, held = 0, ended = 0;
} g_seen_warmup;
int check_wind_warmup(int cases) {
    const auto game = eboot_kit::fn<WindReceiversFn>(kWindReceivers);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        u64 sys[2];
        for (u64& w : sys) {
            w = block(0x80);
            const u64 head = block(0x20);
            store<u64>(head, head);
            store<u64>(head + 8, head);
            store<u64>(w + 0x18, head);
        }
        // a reset just made, a warm-up partway, or none
        const int kind = g_rng.range(0, 2);
        const u8 reset = kind == 0 ? 1 : g_rng.chance(0.2) ? 1 : 0, on = kind == 2 ? 0 : 1;
        const u32 left = kind == 0 ? 60u : static_cast<u32>(g_rng.range(0, 70));
        for (const u64 w : sys) {
            store<u8>(w + 0x4c, reset);
            store<u8>(w + 0x4d, on);
            store<u32>(w + 0x50, left);
        }
        const int rate_kind = g_rng.range(0, 3);
        const int frames = g_rng.range(4, 400);
        const bool gate = g_rng.chance(0.7);
        double carry = 0.0;
        const char* why = nullptr;
        int at = -1;
        ++g_seen_warmup.cases;
        auto bytes = [](u64 w) {
            std::vector<u8> v(0x80);
            std::memcpy(v.data(), reinterpret_cast<const void*>(w), v.size());
            std::memset(v.data() + 0x18, 0, 8);  // the list heads are each one's own
            return v;
        };
        for (int f = 0; f < frames && !why; ++f) {
            const double dt = rate_kind == 0 ? 1.0 / 60 : rate_kind == 1 ? 1.0 / 120 : rate_kind == 2 ? 1.0 / 144 : g_rng.value(1.0f / 150, 1.0f / 40);
            const sfx_schedule::Frame fr = sfx_schedule::next(carry, dt, 30);
            const bool tick = fr.whole > 0;
            const float step = tick ? 1.0f / 30.0f : 0.0f;
            // the game's own call on a copy: what this frame's call would be as the game's
            const u64 copy = block(0x80);
            std::memcpy(reinterpret_cast<void*>(copy), reinterpret_cast<void*>(sys[0]), 0x80);
            game(copy, step);
            decomp_sfx_gate_frame(!tick && gate);
            decomp_sfx_step_frame(!tick);
            const u8 on0 = load<u8>(sys[0] + 0x4d);
            const u32 left0 = load<u32>(sys[0] + 0x50);
            sfx_wind_receivers(sys[0], step);
            decomp_sfx_step_frame(false);
            decomp_sfx_gate_frame(false);
            ++g_seen_warmup.frames;
            if (!tick) {
                ++g_seen_warmup.held;
                std::vector<u8> want = bytes(copy);
                want[0x4d] = on0;
                std::memcpy(want.data() + 0x50, &left0, 4);
                if (bytes(sys[0]) != want) why = "between whole steps: the count held, the rest the game's own call's";
            } else {
                // the game at 30, once this whole step
                game(sys[1], step);
                if (bytes(sys[0]) != bytes(sys[1])) why = "after a whole step: the game's at 30";
                g_seen_warmup.ended += on && !load<u8>(sys[1] + 0x4d) && load<u8>(sys[1] + 0x4c) == 0 && left == 0;
            }
            if (why) at = f;
        }
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("wind warm-up case %d (reset %u, on %u, left %u, rate kind %d, gate %d): %s differs at frame %d\n", c, reset, on, left, rate_kind,
                        gate, why, at);
    }
    return bad;
}

// --- the ribbon (FFX-format GX type 0x28) under the smooth steps --------------------------

// The ribbon's set-up allocates its point arrays from an allocator (vtable +0x58: (allocator, size, alignment)).
GUEST_ABI u64 stub_ribbon_alloc(u64, u64 size, u64) { return block(size); }
alignas(64) u8 g_ribbon_alloc_vt[0x80];
alignas(64) u8 g_ribbon_alloc[0x10];
// A property record of an FFX-format definition: kind 0 (the value of the last key at or before the time; for the
// bounds' table, the largest value), its keys {count n, n times, n values}.
void ribbon_property(u64 at, int keys, float lo, float hi) {
    const u64 k = block(4 + 8 * static_cast<std::size_t>(keys));
    store<s32>(k, keys);
    float t = 0;
    for (int i = 0; i < keys; ++i) {
        store<float>(k + 4 + 4 * static_cast<u64>(i), t);
        store<float>(k + 4 + 4 * static_cast<u64>(keys + i), value(lo, hi));
        t += value(0.05f, 1.5f);
    }
    store<u64>(at, 0);
    store<u64>(at + 8, k);
}

struct Ribbon {
    u64 record, item, type, def, node, inst;
    u8 rng[0x20];
};
// The type's word fields its update keeps (the end time +0x18 to the history's +0x74, room +0x78, +0x84), and the
// point arrays: times, places, lengths, the scalar (+0xb0, unless +0x68's kind is 0x1c), the vectors (mode +0x3c 1).
// (With the definition's +0x128 bit 1 every call writes the lengths' array afresh from the points' count and index -
// a texture coordinate, not history: left out.)
std::vector<u64> ribbon_arrays(const Ribbon& r) {
    std::vector<u64> a = {0x88, 0x90, 0x98, 0xa0};
    if (!(load<u32>(r.def + 0x128) & 2)) a.push_back(0xa8);
    if (load<u32>(r.def + 0x68) != 0x1c) a.push_back(0xb0);
    if (load<u32>(r.def + 0x3c) == 1)
        for (const u64 off : {0xb8ull, 0xc0ull, 0xc8ull}) a.push_back(off);
    return a;
}
bool ribbon_fields_same(const Ribbon& a, const Ribbon& b) {
    return std::memcmp(reinterpret_cast<void*>(a.type + 0x18), reinterpret_cast<void*>(b.type + 0x18), 0x78 - 0x18 + 4) == 0 &&
           load<u32>(a.type + 0x84) == load<u32>(b.type + 0x84);
}
// The kept points of a against those of b from index d on (b's first d gone early from a).
bool ribbon_points_same(const Ribbon& a, const Ribbon& b, u32 d) {
    const u32 n = load<u32>(a.type + 0x7c);
    for (const u64 off : ribbon_arrays(a))
        if (std::memcmp(reinterpret_cast<void*>(load<u64>(a.type + off)), reinterpret_cast<void*>(load<u64>(b.type + off) + 4 * static_cast<u64>(d)),
                        4 * static_cast<std::size_t>(n)) != 0)
            return false;
    return true;
}
std::vector<u8> bytes_of_block(u64 p, std::size_t n) {
    std::vector<u8> v(n);
    std::memcpy(v.data(), reinterpret_cast<void*>(p), n);
    return v;
}
std::vector<u8> ribbon_snapshot(const Ribbon& r) {
    const u32 n = load<u32>(r.type + 0x7c);
    std::vector<u8> v(0xf0);
    std::memcpy(v.data(), reinterpret_cast<void*>(r.type), 0xf0);
    for (const u64 off : ribbon_arrays(r)) {
        const std::size_t at = v.size();
        v.resize(at + 4 * static_cast<std::size_t>(n));
        std::memcpy(v.data() + at, reinterpret_cast<void*>(load<u64>(r.type + off)), 4 * static_cast<std::size_t>(n));
    }
    return v;
}
// A snapshot's kept points against the ribbon's now, the first d gone since.
bool ribbon_kept_since(const Ribbon& r, const std::vector<u8>& snap, u32 d) {
    const u32 n = load<u32>(r.type + 0x7c), m = load<u32>(reinterpret_cast<u64>(snap.data()) + 0x7c);
    if (n + d != m) return false;
    std::size_t at = 0xf0;
    for (const u64 off : ribbon_arrays(r)) {
        if (std::memcmp(reinterpret_cast<void*>(load<u64>(r.type + off)), snap.data() + at + 4 * static_cast<std::size_t>(d), 4 * static_cast<std::size_t>(n)) != 0)
            return false;
        at += 4 * static_cast<std::size_t>(m);
    }
    return true;
}

// After a whole step, the frame domain's kept points (a) against the game's at 30 (b): both a run of the ribbon's
// inserted points ending at its newest, the part both hold bit-equal. Which old points are still held differs: the
// game drops the oldest a block of +0x94 + 1 at a time when the next block's first point has expired (or one by one
// with few left) and makes room for an update's new points, once an update - the frame domain's frames between whole
// steps do the same at their own, later, times, so a block or a room's worth goes a frame early, and the blocks then
// fall at other points. So a point only one of them holds is past its life by the frame's time, give or take an
// insert's spacing (`spacing`: the whole step past the definition's interval), or went for room.
bool ribbon_tails_agree(const Ribbon& a, const Ribbon& b, float now, double spacing) {
    const u32 na = load<u32>(a.type + 0x7c), nb = load<u32>(b.type + 0x7c);
    if (!na || !nb) return na == nb || (na + nb) <= load<u32>(a.def + 0x94) + 2;
    const u64 ta = load<u64>(a.type + 0x88), tb = load<u64>(b.type + 0x88);
    // the newest the same, then where each one's oldest is in the other
    if (load<u32>(ta + 4 * static_cast<u64>(na - 1)) != load<u32>(tb + 4 * static_cast<u64>(nb - 1))) return false;
    u32 skip_a = 0, skip_b = 0;
    if (na >= nb) skip_a = na - nb;
    else skip_b = nb - na;
    if (load<u32>(ta + 4 * static_cast<u64>(skip_a)) != load<u32>(tb + 4 * static_cast<u64>(skip_b))) return false;
    const u32 common = na - skip_a;
    for (const u64 off : ribbon_arrays(a))
        if (std::memcmp(reinterpret_cast<void*>(load<u64>(a.type + off) + 4 * static_cast<u64>(skip_a)),
                        reinterpret_cast<void*>(load<u64>(b.type + off) + 4 * static_cast<u64>(skip_b)), 4 * static_cast<std::size_t>(common)) != 0)
            return false;
    // the points only one of them holds
    const float life = load<float>(a.def + 0x8c);
    const u32 cap = load<u32>(b.type + 0x78), add = std::min(cap, 2 * load<u32>(b.def + 0x94) + 3);
    const u32 room = nb + add > cap ? nb + add - cap : 0;
    u32 young_b = 0;  // b's that a let go before their life ended: room's
    for (u32 i = 0; i < skip_b; ++i)
        if (!(static_cast<double>(load<float>(tb + 4 * static_cast<u64>(i))) + life <= static_cast<double>(now) + 1e-6)) ++young_b;
    if (young_b > room) return false;
    for (u32 i = 0; i < skip_a; ++i)
        if (!(static_cast<double>(load<float>(ta + 4 * static_cast<u64>(i))) + life <= static_cast<double>(now) + spacing + 1e-6)) return false;
    return true;
}

// The emitter: a place along a path (a line, a curve and a swing), rows the record's matrix (the place in the w lanes).
struct Emitter {
    float p0[3], v[3], a[3], amp[3], w;
    float rows[12];
    void at(double t, u64 out) const {
        for (int i = 0; i < 3; ++i) {
            const double x = p0[i] + v[i] * t + 0.5 * a[i] * t * t + amp[i] * std::sin(w * t);
            store<float>(out + 16 * static_cast<u64>(i), rows[4 * i]);
            store<float>(out + 16 * static_cast<u64>(i) + 4, rows[4 * i + 1]);
            store<float>(out + 16 * static_cast<u64>(i) + 8, rows[4 * i + 2]);
            store<float>(out + 16 * static_cast<u64>(i) + 12, static_cast<float>(x));
        }
    }
};

Ribbon make_ribbon(const Emitter& e, u64 def, float t0, float scale, const u8* rng) {
    Ribbon r;
    r.def = def;
    r.record = block(0x80);
    r.item = block(0x70);
    r.type = block(0xf0);
    r.node = block(0x100);
    r.inst = block(0x98);
    const u64 counts = block(0x20);  // +8 = 0: no buffers
    store<u64>(r.record + 8, r.type);
    store<u64>(r.record + 0x10, r.node);
    store<u64>(r.record + 0x18, counts);
    store<float>(r.record + 0x24, t0);
    store<float>(r.record + 0x28, t0);
    e.at(0.0, r.record + 0x50);
    store<u32>(r.node + 0x78, 0);  // its time scale 1
    store<u64>(r.node + 0x58, r.inst);
    store<u32>(r.node + 0xdc, 3);
    store<float>(r.inst + 8, scale);
    // as sub_2cf80c0 makes one: the base, the vtable, then the set-up (the arrays, the first point, two draws)
    eboot_kit::fn<GUEST_ABI void (*)(u64, u64)>(0x2ec4230)(r.type, def);
    store<u64>(r.type, game_address(kGxTypeRibbon));
    std::memcpy(g_rng_state, rng, sizeof g_rng_state);
    eboot_kit::fn<GUEST_ABI u8 (*)(u64, u64, u64)>(0x2ec47f0)(r.type, r.record + 0x10, reinterpret_cast<u64>(g_ribbon_alloc));
    std::memcpy(r.rng, g_rng_state, sizeof r.rng);
    return r;
}

struct RibbonSeen {
    long cases = 0, frames = 0, ticks = 0, between = 0, adjusted = 0, inserts = 0, smoothed = 0, gone_early = 0, ending = 0, degenerate = 0, too_small = 0;
} g_seen_rb;

// The ribbon's smooth steps (sfx_gx_record_update) against the game's own record update and ribbon update
// (0x2cbcf50, 0x2ec1e00), the game's set-up (0x2ec47f0) making both: frame sequences at 60, 120, 144 and jittered rates,
// the record's item each frame the frame's step and the emitter's place at the frame's end; against them, the game's
// update once a whole step, as at 30, with the same place and the whole step's time on the frame domain's clock (the
// record's time less the frame's carry past the step). After every whole step: the history (inserts' times and places,
// lengths, scalars, the per-ribbon seed, the end time) and the FX RNG bit-equal, the kept points the same but for those
// the frame's own time has already let go; between whole steps none of that moves; on every frame the last drawn point
// the emitter's place at the frame's own time, the record's times as the game's one call leaves them and the
// definition back in the type.
int check_ribbons(int cases, u64 sfx, u64 manager) {
    const auto game = eboot_kit::fn<GxRecordFn>(kGxRecordUpdate);
    store<u32>(sfx + 0x8c, 0);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        // the definition
        const u64 def = block(0x140);
        for (int k = 0; k < 0x130; k += 4) store<u32>(def + k, g_rng.u32());
        const u32 mode = g_rng.chance(0.5) ? 1u : 0u;
        store<u32>(def + 0x3c, mode);
        ribbon_property(def + 0x58, g_rng.range(1, 3), 0.01f, 2.0f);
        if (g_rng.chance(0.2)) {
            store<u64>(def + 0x68, 0x1c);  // no scalar array
            store<u64>(def + 0x70, 0);
        } else {
            ribbon_property(def + 0x68, g_rng.range(1, 3), -1.0f, 1.0f);
        }
        ribbon_property(def + 0x78, g_rng.range(1, 3), 0.0f, 1.0f);
        const float interval = g_rng.chance(0.6) ? 0.0f : g_rng.chance(0.5) ? value(0.0f, 0.12f) : value(0.025f, 0.04f);
        store<float>(def + 0x88, interval);
        store<float>(def + 0x8c, value(0.05f, 1.2f));
        store<s32>(def + 0x90, g_rng.chance(0.5) ? -1 : g_rng.range(4, 40));
        store<u32>(def + 0x94, static_cast<u32>(g_rng.range(0, 3)));
        store<float>(def + 0x98, g_rng.chance(0.5) ? 0.0f : value(0.05f, 0.6f));
        store<u32>(def + 0x128, g_rng.u32() & 3);
        // the emitter, the scales, the frame rate
        Emitter e;
        for (int i = 0; i < 3; ++i) {
            e.p0[i] = value(-20, 20);
            e.v[i] = g_rng.chance(0.15) ? 0.0f : value(-15, 15);
            e.a[i] = g_rng.chance(0.5) ? 0.0f : value(-10, 10);
            e.amp[i] = g_rng.chance(0.5) ? 0.0f : value(0, 2);
        }
        e.w = value(1, 20);
        {
            double q[4], qn = 0;
            for (double& x : q) {
                x = value(-1, 1);
                qn += x * x;
            }
            qn = std::sqrt(qn) + 1e-9;
            const double x = q[0] / qn, y = q[1] / qn, z = q[2] / qn, w = q[3] / qn;
            const double m[12] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0, 2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                                  2 * (y * z - x * w), 0, 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0};
            for (int i = 0; i < 12; ++i) e.rows[i] = static_cast<float>(m[i]);
        }
        const float scale = g_rng.chance(0.6) ? 1.0f : value(0.5f, 2.0f), time_scale = g_rng.chance(0.7) ? 1.0f : value(0.5f, 1.5f);
        store<float>(manager + 0x5d8, time_scale);
        store<u8>(manager + 0x5ce, 0);
        const float t0 = g_rng.chance(0.3) ? 0.0f : value(0, 30);
        const double tick_time = 1.0 / 30.0 * time_scale * scale, interval_bound = interval + 2.0 * tick_time;
        u8 rng[0x20];
        for (int k = 0; k < 0x20; k += 4) store<u32>(reinterpret_cast<u64>(rng) + k, g_rng.u32() | 1);
        Ribbon F = make_ribbon(e, def, t0, scale, rng);
        // A set-up whose room (+0x78, from the interval, the life and the FX data rate) is less than one update's new
        // points (2 x +0x94 + 3) has the game's update write past its arrays: not a ribbon to compare.
        if (load<u32>(F.type + 0x78) < 2 * load<u32>(def + 0x94) + 3) {
            ++g_seen_rb.too_small;
            --c;
            continue;
        }
        Ribbon R = make_ribbon(e, def, t0, scale, rng);
        // an unmarked twin (a pre-roll's or a held instance's ribbon) through ours, and the game's own beside it
        Ribbon U = make_ribbon(e, def, t0, scale, rng), G = make_ribbon(e, def, t0, scale, rng);
        const int rate_kind = g_rng.range(0, 3);  // 60, 120, 144, jittered
        const int frames = g_rng.range(8, 160);
        const int end_at = g_rng.chance(0.5) ? g_rng.range(4, frames) : frames + 1;  // the node ends (state 4) at the first whole step from here
        double carry = 0.0, tau = 0.0;
        const char* why = nullptr;
        int at = -1;
        std::vector<u8> snap = ribbon_snapshot(F);
        u8 snap_rng[0x20];
        std::memcpy(snap_rng, F.rng, sizeof snap_rng);
        ++g_seen_rb.cases;
        g_seen_rb.smoothed += load<float>(def + 0x98) > 0.0f;
        decomp_sfx_smooth(true);
        for (int f = 0; f < frames && !why; ++f) {
            const double dt = rate_kind == 0 ? 1.0 / 60 : rate_kind == 1 ? 1.0 / 120 : rate_kind == 2 ? 1.0 / 144 : g_rng.value(1.0f / 150, 1.0f / 40);
            const sfx_schedule::Frame fr = sfx_schedule::next(carry, dt, 30);
            tau += fr.frame;
            const bool tick = fr.whole > 0;
            decomp_sfx_step_frame(!tick);
            decomp_sfx_frame_time(fr.frame, fr.from, fr.into);
            if (tick && f >= end_at && load<u32>(F.node + 0xdc) == 3) {
                store<u32>(F.node + 0xdc, 4);
                store<u32>(R.node + 0xdc, 4);
                ++g_seen_rb.ending;
            }
            // the frame domain's item: the emitter now, the frame's step on the scales
            const float step = fr.frame * time_scale * scale;
            for (int k = 0; k < 0x60; k += 4) store<u32>(F.item + k, 0);
            e.at(tau, F.item);
            store<float>(F.item + 0x60, step);
            const u32 prev = load<u32>(F.record + 0x28);
            const float into = fr.into * time_scale * scale;
            const bool adjusted = tick && into > 0.0f && into < step;
            const float whole_time = adjusted ? add_ss(std::bit_cast<float>(prev), step - into) : add_ss(std::bit_cast<float>(prev), step);
            const float head_time = adjusted ? add_ss(whole_time, into) : whole_time;
            // the whole step's place: the rows that far between the last frame's (the record's) and this one's
            float whole_rows[12];
            {
                const float share = adjusted ? (step - into) / step : 1.0f;
                for (int i = 0; i < 12; ++i) {
                    const float last = load<float>(F.record + 0x50 + 4 * static_cast<u64>(i)), now = load<float>(F.item + 4 * static_cast<u64>(i));
                    whole_rows[i] = adjusted ? last + (now - last) * share : now;
                }
            }
            ribbon_mark(F.type);  // as smooth_update marks its instance's ribbons
            std::memcpy(g_rng_state, F.rng, sizeof g_rng_state);
            sfx_gx_record_update(F.record, F.item);
            std::memcpy(F.rng, g_rng_state, sizeof F.rng);
            // the unmarked twin: the game's call, every byte
            std::memcpy(reinterpret_cast<void*>(U.item), reinterpret_cast<void*>(F.item), 0x70);
            std::memcpy(reinterpret_cast<void*>(G.item), reinterpret_cast<void*>(F.item), 0x70);
            std::memcpy(g_rng_state, U.rng, sizeof g_rng_state);
            sfx_gx_record_update(U.record, U.item);
            std::memcpy(U.rng, g_rng_state, sizeof U.rng);
            std::memcpy(g_rng_state, G.rng, sizeof g_rng_state);
            game(G.record, G.item);
            std::memcpy(G.rng, g_rng_state, sizeof G.rng);
            {
                std::vector<u8> u = ribbon_snapshot(U), g = ribbon_snapshot(G);
                for (const u64 off : {0x88ull, 0x90ull, 0x98ull, 0xa0ull, 0xa8ull, 0xb0ull, 0xb8ull, 0xc0ull, 0xc8ull, 0x8ull}) {
                    // the pointers differ: the arrays' bytes compared through them (ribbon_snapshot), the words beside
                    store<u64>(reinterpret_cast<u64>(u.data()) + off, 0);
                    store<u64>(reinterpret_cast<u64>(g.data()) + off, 0);
                }
                const std::vector<u8> ur = bytes_of_block(U.record, 0x80), gr = bytes_of_block(G.record, 0x80);
                if (u != g || std::memcmp(U.rng, G.rng, sizeof U.rng) != 0 ||
                    std::memcmp(ur.data() + 0x20, gr.data() + 0x20, 0x60) != 0)
                    why = "a ribbon not marked this frame (a pre-roll's, a held instance's): the game's call";
            }
            ++g_seen_rb.frames;
            g_seen_rb.adjusted += adjusted;
            // every frame: the definition back, the record's times the game's one call's, the last drawn point the emitter at the
            // frame's time
            const u32 n = load<u32>(F.type + 0x7c), drawn = load<u32>(F.type + 0x80);
            if (load<u64>(F.type + 8) != def) {
                why = "the definition not put back";
                store<u64>(F.type + 8, def);
            } else if (load<u32>(F.record + 0x24) != prev || load<u32>(F.record + 0x28) != bits(add_ss(std::bit_cast<float>(prev), step))) {
                why = "the record's times";
            } else if (n + drawn >= 1) {
                // the last drawn point: the emitter at the frame's time - or, where the emitter moved too little since the
                // last insert for the game to draw a point to it, that insert's own, or (nothing drawn, the emitter not moved
                // off the newest kept point) that point's own
                const u64 last = 4 * static_cast<u64>(n + drawn - 1);
                const bool degenerate = (load<u32>(load<u64>(F.type + 0x88) + last) == load<u32>(F.type + 0x1c) &&
                                         load<u32>(load<u64>(F.type + 0x90) + last) == load<u32>(F.type + 0x24) &&
                                         load<u32>(load<u64>(F.type + 0x98) + last) == load<u32>(F.type + 0x28) &&
                                         load<u32>(load<u64>(F.type + 0xa0) + last) == load<u32>(F.type + 0x2c)) ||
                                        (drawn == 0 && load<u32>(load<u64>(F.type + 0x90) + last) == load<u32>(F.item + 12) &&
                                         load<u32>(load<u64>(F.type + 0x98) + last) == load<u32>(F.item + 28) &&
                                         load<u32>(load<u64>(F.type + 0xa0) + last) == load<u32>(F.item + 44));
                if (degenerate) {
                    const double dx = load<float>(load<u64>(F.type + 0x90) + last) - load<float>(F.item + 12),
                                 dy = load<float>(load<u64>(F.type + 0x98) + last) - load<float>(F.item + 28),
                                 dz = load<float>(load<u64>(F.type + 0xa0) + last) - load<float>(F.item + 44);
                    if (std::sqrt(dx * dx + dy * dy + dz * dz) > 1e-6) why = "the last drawn point is the last insert, the emitter away from it";
                    ++g_seen_rb.degenerate;
                } else if (load<u32>(load<u64>(F.type + 0x88) + last) != bits(head_time)) {
                    why = "the last drawn point's time";
                }
                else if (!degenerate && (load<u32>(load<u64>(F.type + 0x90) + last) != load<u32>(F.item + 12) ||
                                         load<u32>(load<u64>(F.type + 0x98) + last) != load<u32>(F.item + 28) ||
                                         load<u32>(load<u64>(F.type + 0xa0) + last) != load<u32>(F.item + 44)))
                    why = "the last drawn point's place (the emitter's at the frame's time)";
            }
            if (!why && !tick) {
                // between whole steps: no insert, draw or smoothing - the history, the RNG and the kept points as at the last
                // whole step, but for those the frame's time has let go
                ++g_seen_rb.between;
                const u32 m = load<u32>(reinterpret_cast<u64>(snap.data()) + 0x7c);
                if (std::memcmp(reinterpret_cast<void*>(F.type + 0x18), snap.data() + 0x18, 0x78 - 0x18 + 4) != 0 ||
                    load<u32>(F.type + 0x84) != load<u32>(reinterpret_cast<u64>(snap.data()) + 0x84))
                    why = "between whole steps: the history moved (an insert)";
                else if (std::memcmp(F.rng, snap_rng, sizeof snap_rng) != 0) why = "between whole steps: a draw of the FX RNG";
                else if (n > m || !ribbon_kept_since(F, snap, m - n)) why = "between whole steps: the kept points moved (smoothing)";
            }
            if (!why && tick) {
                // the game at 30: once a whole step, the same place, the whole step's time
                ++g_seen_rb.ticks;
                std::memcpy(reinterpret_cast<void*>(R.item), reinterpret_cast<void*>(F.item), 0x60);
                std::memcpy(reinterpret_cast<void*>(R.item), whole_rows, sizeof whole_rows);
                store<float>(R.item + 0x60, 0.0f);
                store<float>(R.record + 0x28, whole_time);
                const u32 r_last = load<u32>(R.type + 0x1c);
                std::memcpy(g_rng_state, R.rng, sizeof g_rng_state);
                game(R.record, R.item);
                std::memcpy(R.rng, g_rng_state, sizeof R.rng);
                g_seen_rb.inserts += load<u32>(R.type + 0x1c) != r_last;
                const u32 nr = load<u32>(R.type + 0x7c);
                g_seen_rb.gone_early += nr > n ? nr - n : n - nr;
                if (!ribbon_fields_same(F, R)) why = "after a whole step: the history (inserts, places, scalars, seed, end time)";
                else if (std::memcmp(F.rng, R.rng, sizeof F.rng) != 0) why = "after a whole step: the FX RNG";
                else if (!ribbon_tails_agree(F, R, head_time, interval_bound)) why = "after a whole step: the kept points";
                snap = ribbon_snapshot(F);
                std::memcpy(snap_rng, F.rng, sizeof snap_rng);
            }
            if (why) at = f;
        }
        decomp_sfx_smooth(false);
        decomp_sfx_step_frame(false);
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("ribbon case %d (rate kind %d, interval %g, life %g, subdiv %u, smoothing %g, mode %u, scale %g x %g): %s differs at frame %d\n", c,
                        rate_kind, interval, load<float>(def + 0x8c), load<u32>(def + 0x94), load<float>(def + 0x98), mode, scale, time_scale, why, at);
    }
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
        std::memcmp(eboot_kit::at(kGxRecordUpdate), kGxRecordEntry, sizeof kGxRecordEntry) != 0 ||
        std::memcmp(eboot_kit::at(kLightFades), kLightFadesEntry, sizeof kLightFadesEntry) != 0 ||
        std::memcmp(eboot_kit::at(kWindSourcesStep), kWindStepEntry, sizeof kWindStepEntry) != 0) {
        std::printf("decomp_sfx_step_gate: the eight entries are not where the gate expects them\n");
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
    // The light fades let a GX light go through the GX light bank (sub_263ede0, the bank *(0x5940dd8) +0x5870).
    eboot_kit::stub(0x263ede0, reinterpret_cast<const void*>(&stub_light_release));
    alignas(64) static u8 gx_mem[0x5900];
    store<u64>(reinterpret_cast<u64>(eboot_kit::at(0x5940dd8)), reinterpret_cast<u64>(gx_mem));
    g_game_light_fades = eboot_kit::at(kLightFades);
    g_game_light = eboot_kit::at(0x131e660);
    g_game_instance = eboot_kit::at(0x2c4ff30);
    g_game_wobble_a = eboot_kit::at(kWobbleA);
    g_game_wobble_b = eboot_kit::at(kWobbleB);
    std::memcpy(g_wobble_b_entry, eboot_kit::at(kWobbleB), sizeof g_wobble_b_entry);  // follow_b_patched puts it back
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
    // The light under the smooth steps, inside the node update, and the light fades.
    int lbad = 0;
    for (const bool specials : {false, true}) {
        g_rng.specials = specials;
        lbad += check_smooth_lights(kCases / 4, sfx);
        lbad += check_light_fades(kCases / 2, sfx);
    }
    std::printf("decomp_sfx_step_gate: %d smooth light cases (%ld frames, %ld whole steps, %ld light calls; %ld with the flicker, %ld blending, "
                "%ld on plain nodes) and %d light fade cases, %d differ\n",
                kCases / 2, g_seen_sl.frames, g_seen_sl.ticks, g_seen_sl.light_calls, g_seen_sl.flicker, g_seen_sl.blending, g_seen_sl.plain,
                kCases, lbad);
    if (lbad) return 1;
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
    for (const int m : {22, 23, 24, 25, 26, 27, 28}) {
        g_sfx_gate_mutant = m;
        g_rng.specials = false;
        if (!(m == 28 ? check_light_fades(2000, sfx) : check_smooth_lights(2000, sfx))) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught\n", m);
            ++missed;
        }
    }
    g_sfx_gate_mutant = 0;
    g_quiet = false;
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
    // The build's converted list: never the composite motion (it is its parts), nor the grid simulation. (The FFX light
    // and the wobbles 0x570e6d0 / 0x570e690 / 0x570e8d0 / 0x570e890 are on it: their leaves count once a whole step
    // themselves, checked here.)
    for (const u64 vt : kConverted)
        for (const u64 off : {0x570e750ull, 0x56eb030ull})
            if (vt == off) {
                std::printf("decomp_sfx_step_gate: 0x%llx is on the converted list\n", static_cast<unsigned long long>(vt));
                return 1;
            }
    // The smooth steps (the effect manager's time scale +0x5d8 and its own still flag +0x5ce, behind SprjSfx +0x70).
    alignas(64) static u8 manager_mem[0x600];
    const u64 manager = reinterpret_cast<u64>(manager_mem);
    store<u64>(sfx + 0x70, manager);
    g_rng.specials = false;
    int sbad = check_smooth(kCases / 2, sfx, manager);
    std::printf("decomp_sfx_step_gate: %d smooth-step instance cases, %d differ - %ld took the smooth steps (%ld container calls between whole "
                "steps, %ld node times and %ld definition times moved for them; %ld controller calls at 30), %ld kept by a class off the list "
                "(a module %ld, a wobble %ld, a GX type %ld, a composite too deep %ld), %ld the game's\n",
                kCases / 2, sbad, g_seen_smooth.took, g_seen_smooth.between_calls, g_seen_smooth.shifted_t, g_seen_smooth.shifted_e,
                g_seen_smooth.controllers_30, g_seen_smooth.kept, g_seen_smooth.by_class[0], g_seen_smooth.by_class[1], g_seen_smooth.by_class[2],
                g_seen_smooth.by_class[3], g_seen_smooth.game);
    if (sbad) return 1;
    g_quiet = true;
    for (const int m : {20, 21}) {
        g_sfx_gate_mutant = m;
        if (!check_smooth(2000, sfx, manager)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught by the smooth steps\n", m);
            ++missed;
        }
    }
    g_sfx_gate_mutant = 0;

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
    for (const int m : {1, 9, 10, 11, 12, 44, 45}) {
        g_sfx_gate_mutant = m;
        g_rng.specials = false;
        if (!check_wobbles(2000, sfx)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught by the wobble motions\n", m);
            if (m != 1) ++missed;  // 1 is the instances' and lights' to catch; here only a second look
        }
    }
    g_sfx_gate_mutant = 0;
    // The wobble motions under the smooth steps (the bases and the FX RNG still the test's).
    g_quiet = false;
    g_rng.specials = false;
    const int swbad = check_smooth_wobbles(kCases / 2, sfx);
    std::printf("decomp_sfx_step_gate: %d smooth wobble motion cases (0x570e6d0 %ld, 0x570e690 %ld, 0x570e8d0 %ld, 0x570e890 %ld; %ld frames, %ld "
                "whole steps, %ld re-rolls at 30, %ld follows between whole steps), %d differ\n",
                kCases / 2, g_seen_sw.kinds[0], g_seen_sw.kinds[1], g_seen_sw.kinds[2], g_seen_sw.kinds[3], g_seen_sw.frames, g_seen_sw.ticks,
                g_seen_sw.rerolls, g_seen_sw.follows, swbad);
    if (swbad) return 1;
    g_quiet = true;
    for (const int m : {42, 43, 44, 45}) {
        g_sfx_gate_mutant = m;
        if (!check_smooth_wobbles(2000, sfx)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught by the smooth wobble motions\n", m);
            ++missed;
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
    // The classifications kept between whole steps.
    g_quiet = false;
    g_rng.specials = false;
    const int ccbad = check_class_cache(kCases / 2, sfx);
    if (ccbad) return 1;
    g_quiet = true;
    for (const int m : {51, 52, 53, 54}) {
        g_sfx_gate_mutant = m;
        if (!check_class_cache(400, sfx)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught by the class cache\n", m);
            ++missed;
        }
    }
    g_sfx_gate_mutant = 0;
    // GPU particle type 0x2713 under the smooth steps.
    g_quiet = false;
    g_rng.specials = false;
    const int s2bad = check_records_smooth(kCases, sfx, manager);
    std::printf("decomp_sfx_step_gate: %d smooth GPU particle type 0x2713 record cases (%ld at the whole step's elapsed time, %ld the game's; %ld "
                "previous times moved by an ulp or more to keep the draw's elapsed time), %d differ\n",
                kCases, g_seen_2713.whole, g_seen_2713.game, g_seen_2713.prev_nudged, s2bad);
    if (s2bad) return 1;
    g_quiet = true;
    for (const int m : {40, 41}) {
        g_sfx_gate_mutant = m;
        if (!check_records_smooth(2000, sfx, manager)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught by the smooth 0x2713 records\n", m);
            ++missed;
        }
    }
    g_sfx_gate_mutant = 0;
    // GPU particle emission under the smooth steps: the two types' updates the test's, the FX RNG holder its own.
    store<u64>(game_address(kGxStandard) + 0x10, reinterpret_cast<u64>(&stub_standard_type));
    store<u64>(game_address(kGxSpark) + 0x10, reinterpret_cast<u64>(&stub_spark_type));
    store<u64>(reinterpret_cast<u64>(eboot_kit::at(0x5940de8)), reinterpret_cast<u64>(g_emit_rng));
    g_quiet = false;
    g_rng.specials = false;
    const int ebad = check_emission(kCases / 4, sfx, manager);
    std::printf("decomp_sfx_step_gate: %d GPU particle emission cases (%ld whole steps against %ld frames, %ld of them capped; %ld emitters "
                "truncating, %ld with limits, %ld ended; %ld one update a window), %d differ\n",
                kCases / 4, g_seen_emit.steps, g_seen_emit.frames, g_seen_emit.capped, g_seen_emit.truncated, g_seen_emit.limited,
                g_seen_emit.ended, g_seen_emit.single, ebad);
    if (ebad) return 1;
    const int rbad2 = check_emission_rates(sfx, manager);
    std::printf("decomp_sfx_step_gate: GPU particle emission at random intervals over 40 s: %.1f events a second in 30's whole steps, %.1f in "
                "frames counted a whole step at a time (the game's own update a frame: %.1f), %d of %d rates off by over 3%%\n",
                g_seen_rates.game30 / (g_seen_rates.emitters * 320.0), g_seen_rates.ours / (g_seen_rates.emitters * 320.0),
                g_seen_rates.theirs / (g_seen_rates.emitters * 320.0), rbad2, g_seen_rates.emitters);
    if (rbad2) return 1;
    g_quiet = true;
    for (const int m : {56, 57, 58, 59, 60, 61, 62}) {
        g_sfx_gate_mutant = m;
        if (!check_emission(1000, sfx, manager)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught by the GPU particle emission\n", m);
            ++missed;
        }
    }
    g_sfx_gate_mutant = 0;
    // The wind sources' registry step.
    g_game_wind_step = eboot_kit::at(kWindSourcesStep);
    g_quiet = false;
    g_rng.specials = false;
    const int wsbad = check_wind_sources(kCases, sfx, manager);
    std::printf("decomp_sfx_step_gate: %d wind source registry cases (%ld by the frame's time, %ld another caller's step) and %ld seconds of frames "
                "against 30's whole steps, %d differ\n",
                kCases, g_seen_wind.smooth, g_seen_wind.other, g_seen_wind.seconds, wsbad);
    if (wsbad) return 1;
    g_quiet = true;
    for (const int m : {38, 39}) {
        g_sfx_gate_mutant = m;
        if (!check_wind_sources(2000, sfx, manager)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught by the wind sources\n", m);
            ++missed;
        }
    }
    g_sfx_gate_mutant = 0;
    // The wind system's warm-up: the thread's runtime heap and gettimeofday the test's.
    g_runtime_vt[0x20 / 8] = reinterpret_cast<void*>(&runtime_caps);
    g_runtime_vt[0x58 / 8] = reinterpret_cast<void*>(&runtime_alloc);
    g_runtime_vt[0x70 / 8] = reinterpret_cast<void*>(&runtime_free);
    {
        void* const vt = g_runtime_vt;
        std::memcpy(g_runtime + 0x28, &vt, 8);
        u8* const runtime = g_runtime;
        std::memcpy(static_cast<u8*>(eboot_kit::tcb()) - 0x730, &runtime, 8);  // rel_dlruntime_obj_mb
    }
    if (!eboot_kit::bind("gettimeofday", reinterpret_cast<const void*>(&stub_gettimeofday))) {
        std::printf("decomp_sfx_step_gate: no gettimeofday import to bind\n");
        return 1;
    }
    g_game_wind_receivers = eboot_kit::at(kWindReceivers);
    g_quiet = false;
    g_rng.specials = false;
    const int wwbad = check_wind_warmup(kCases);
    std::printf("decomp_sfx_step_gate: %d wind system warm-up cases (%ld frames, %ld between whole steps with the count held, %ld warm-ups ended), "
                "%d differ\n",
                kCases, g_seen_warmup.frames, g_seen_warmup.held, g_seen_warmup.ended, wwbad);
    if (wwbad) return 1;
    g_quiet = true;
    for (const int m : {49, 50}) {
        g_sfx_gate_mutant = m;
        if (!check_wind_warmup(2000)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught by the wind warm-up\n", m);
            ++missed;
        }
    }
    g_sfx_gate_mutant = 0;
    // The ribbon last: its set-up's allocator the test's; the manager's time scale per case.
    store<u64>(reinterpret_cast<u64>(g_ribbon_alloc_vt) + 0x58, reinterpret_cast<u64>(&stub_ribbon_alloc));
    store<u64>(reinterpret_cast<u64>(g_ribbon_alloc), reinterpret_cast<u64>(g_ribbon_alloc_vt));
    g_quiet = false;
    g_rng.specials = false;
    const int bbad = check_ribbons(kCases / 2, sfx, manager);
    std::printf("decomp_sfx_step_gate: %d ribbon cases (%ld frames: %ld whole steps, %ld between, %ld whole steps past the frame's carry; %ld inserts; "
                "%ld with smoothing, %ld ending; %ld old points held by one and not the other after a whole step; %ld set-ups too small for an update, "
                "not compared), %d differ\n",
                kCases / 2, g_seen_rb.frames, g_seen_rb.ticks, g_seen_rb.between, g_seen_rb.adjusted, g_seen_rb.inserts, g_seen_rb.smoothed,
                g_seen_rb.ending, g_seen_rb.gone_early, g_seen_rb.too_small, bbad);
    if (bbad) return 1;
    {
        // the marks' table: a frame's marks gone at the next; a ribbon whose slots this frame are all others' finds none
        decomp_sfx_frame_time(1.0f / 120, 0.0f, 0.0f);
        const u64 t = 0x123450000ull;
        if (!ribbon_mark(t) || !ribbon_marked(t)) {
            std::printf("decomp_sfx_step_gate: a ribbon's mark not found in its frame\n");
            return 1;
        }
        decomp_sfx_frame_time(1.0f / 120, 0.0f, 0.0f);
        if (ribbon_marked(t)) {
            std::printf("decomp_sfx_step_gate: a ribbon's mark kept into the next frame\n");
            return 1;
        }
        int taken = 0;
        for (u64 k = 1; k < 1 << 20 && taken < 16; ++k)
            if (ribbon_slot(k << 4) == ribbon_slot(t) && ribbon_mark(k << 4)) ++taken;
        // the 16 slots from t's are this frame's now (probes run on from each one's own)
        bool full = true;
        for (std::size_t i = 0; i < 16; ++i)
            full &= (g_ribbon_marks[(ribbon_slot(t) + i) & (kRibbonSlots - 1)].load() & ~kMarkType) == ribbon_frame();
        if (!full || ribbon_mark(t) || ribbon_marked(t)) {
            std::printf("decomp_sfx_step_gate: a full stretch of the marks' table took a ribbon\n");
            return 1;
        }
        decomp_sfx_frame_time(1.0f / 120, 0.0f, 0.0f);
    }
    g_quiet = true;
    for (const int m : {29, 30, 31, 32, 33, 34, 35, 37}) {
        g_sfx_gate_mutant = m;
        if (!check_ribbons(2000, sfx, manager)) {
            std::printf("decomp_sfx_step_gate: mutant %d not caught by the ribbon\n", m);
            ++missed;
        }
    }
    g_sfx_gate_mutant = 0;
    std::printf("decomp_sfx_step_gate: %d of 56 mutants caught\n", 56 - missed);
    return missed ? 1 : 0;
}
