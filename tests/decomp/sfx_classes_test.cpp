// The effect classes the smooth effect steps convert (decomp/sfx/step_gate.cpp's
// list), each against the game's own code (tests/decomp/eboot_kit.h). The frame
// domain changes only the step a class's update is handed - the held frames
// already run every one each frame, with a step of 0 between whole steps - so a class
// is converted when what it does with its step is time: one whole step run
// once with 1/30 s, and run as the frame domain runs it - 2 or 4 frames of 1/60
// or 1/120 s, the node's time put as far past its whole step as each frame
// begins - end where the class's integration says they should, and a step of 0
// changes nothing it accumulates. Each case:
//   - the Euler motion 0x570e650 (sub_2e88510): velocity +0x10 changed along
//     itself by the step times its speed curve (+8, at the node's time) times
//     +0x24, then the node's position by the step times the velocity, facing
//     it with +0x28. A constant curve (or none): the same velocity, the
//     position apart by the Euler term h^2 a (n-1)/2n exactly; a sloped one:
//     the velocity apart by h^2 k b (n-1)/2n (the curve read at each frame);
//     the same facing; a step of 0 moves nothing;
//   - the gravity motion 0x570e610 (sub_2e87a10) and the wind motion 0x570e5d0
//     (sub_2e866f0, the wobbles' base): the Euler motion's update with gravity
//     (data_5ad1150 times a curve at the node's time, in the parent's frame)
//     and, for the wind motion, the effects' wind world's force at the node's
//     place and its drift added to the move: gravity alone the velocity and
//     position apart by their Euler terms exactly; every case without a
//     reversal or a stop held to the update's equations (in double), whole
//     step and frames alike; facing along the final move in both;
//   - the direction motions 0x570e850 (sub_2e8d740), 0x570e810 (sub_2e8cc60,
//     with gravity) and 0x570e7d0 (sub_2e8bfd0, with the wind world; the
//     wobble 0x570e8d0's base): the velocity +0x20 by the step times the
//     local force, the move the step times the velocity plus the speed curve
//     times the direction +0x10 (never written) plus the drift: no branch, so
//     every case held to the update's equations in double, whole step and
//     frames alike; facing along the final move;
//   - the rotation motion 0x570e790 (sub_2e8b100): the node turned by the step
//     times its rates (+0x10..+0x18, or three curves): one axis turns the
//     same, three within the Euler angles' (h w)^2; a step of 0 turns nothing;
//   - the transform motion 0x570e710 (sub_2e8a3d0): never reads its step
//     (rdx is overwritten first): every byte the same for any step;
//   - the curve motion 0x570e910 (sub_2e8ec00): the position its curve at
//     the node's time, the rotation kept; never reads its step: every byte
//     the same for any step;
//   - the composite motion 0x570e750 (sub_2e8ae30): its two parts, in order,
//     with its own arguments;
//   - the children module 0x56eb2e0 (sub_12fccf0): never reads the container's
//     parameters (rsi is overwritten first), where the step is: the same calls
//     and bytes for any step;
//   - the FFX spot light module 0x56ebc40 (sub_1320260): its GX spot light
//     from curves at the node's age, two modes that store, keep or blend from
//     stored values by a curve at the age (storing nothing): never reads its
//     params or context, where the step is - the same calls and bytes for any
//     - and the frames' last call, at the node's time, leaves the module and
//     the light as the whole step's call there does;
//   - the Sprj module 0x5756e50 (sub_23749d0): its model and its sound advance
//     by the step times its rate +0xc4 - the game's product handed to both, and
//     the frames' sum within a rounding of the whole step's - and nothing else
//     it does depends on the step;
//   - the CPU particles 0x56e3090 (sub_2cbb9c0): the system's time +0x4c moved
//     by each item's step, particles ended by it (birth + life) and the item's
//     emission count born at it: the frames' last item carrying the count (the
//     whole step's, as the container's first call on a frame with one), the
//     same particles alive and born, at the same time, with the same seeds;
//   - the GX binder 0x56e3100 (sub_2cbcf50): the record's time chain (+0x24 =
//     +0x28, +0x28 += the item's step) and its type's update once an item,
//     handed that elapsed time;
//   - the FFX-format GX types 0x6b (0x56e35d0, sub_2ec0a60: only its bounds),
//     0x14 / 0x3b (0x56e3530, sub_2eb6550: +0x10 += the elapsed time times a
//     rate at the scaled time), 0x2b (0x56e3580, sub_2ebc280: +0x20, or +0x18
//     and +0x1c, the same) and 3 / 0x3d (0x56ebe20, sub_1325cd0: +0x18 and
//     +0x1c the same): the same bounds and bytes, the accumulations within a
//     rounding a frame, nothing added by a step of 0;
//   - the wind sources 0x283c (0x56eadf0, sub_12e3c30) and 0x283d (0x56eac40,
//     sub_12e27b0, no velocity): its fade (+0xd0) by
//     the elapsed time, its strengths from a curve at the record's time and
//     that fade, its velocity the handle's move over the elapsed time: the
//     same shape, place and bounds, the fade, strengths and velocity within
//     the times' and places' rounding; a step of 0 fades nothing and moves
//     at 0;
//   - the GPU particle simulations StandardParticle (sub_13040b0) and Spark
//     (sub_12fef10): their time and emission timers moved by the elapsed time
//     and an emission each interval - over a second the same particles emitted
//     in frames as in whole steps; the per-update cap (an update past it emits
//     the cap and drops the rest of its timer) is measured, not held: frames
//     emit past what a capped emitter emits in whole steps.
// Skips without the 1.09 eboot.
#include "eboot_kit.h"

#include "decomp/guest.h"
#include "guest_abi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <xmmintrin.h>

using namespace decomp;

namespace {

eboot_kit::Rng g_rng(0xc1a55e5u);
bool g_quiet = false;
// The checks' controls: a frame domain that gets it wrong, which each check must catch - 1 the emission
// count on every frame, 2 the node's time left at its whole step, 3 each frame handed the whole step,
// 4 each simulation update the whole step's time.
int g_control = 0;

std::vector<std::vector<u8>> g_blocks;
u64 block(std::size_t n) {
    g_blocks.emplace_back(n + 64, 0);
    return (reinterpret_cast<u64>(g_blocks.back().data()) + 63) & ~63ull;
}
u32 bits(float f) {
    u32 u;
    std::memcpy(&u, &f, 4);
    return u;
}
float uni(float lo, float hi) { return g_rng.uniform(lo, hi); }
// A game function's address hidden from the optimizer: GCC 16 at -O2 carries
// a constant address into a helper's clone and takes the call through it for
// unreachable.
template <class F>
F opaque(F f) {
    __asm__ volatile("" : "+r"(f));
    return f;
}
// The game's addss with the node's value first: the node's time put past its whole step.
float addss(float x, float y) {
    __asm__("addss %1, %0" : "+x"(x) : "x"(y));
    return x;
}
std::vector<u8> bytes_of(u64 p, std::size_t n) {
    std::vector<u8> v(n);
    std::memcpy(v.data(), reinterpret_cast<void*>(p), n);
    return v;
}
void put_bytes(u64 p, const std::vector<u8>& v) { std::memcpy(reinterpret_cast<void*>(p), v.data(), v.size()); }

struct V4 {
    double x = 0, y = 0, z = 0, w = 0;
};
V4 v4(u64 at) { return {load<float>(at), load<float>(at + 4), load<float>(at + 8), load<float>(at + 12)}; }
void put4(u64 at, double x, double y, double z, double w) {
    store<float>(at, static_cast<float>(x));
    store<float>(at + 4, static_cast<float>(y));
    store<float>(at + 8, static_cast<float>(z));
    store<float>(at + 12, static_cast<float>(w));
}
double len3(const V4& a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
double dist3(const V4& a, const V4& b) { return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)); }
// The angle between two unit quaternions, from their difference (acos of a dot
// product near 1 would turn a float rounding into a milliradian).
double quat_angle(const V4& a, const V4& b) {
    auto len4 = [](double x, double y, double z, double w) { return std::sqrt(x * x + y * y + z * z + w * w); };
    const double minus = len4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w), plus = len4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    return 4.0 * std::asin(std::min(1.0, std::min(minus, plus) / 2.0));
}
void random_quat(u64 at) {
    double q[4], n = 0;
    for (double& c : q) {
        c = uni(-1, 1);
        n += c * c;
    }
    n = std::sqrt(n) + 1e-9;
    put4(at, q[0] / n, q[1] / n, q[2] / n, q[3] / n);
}

// The curves the motions read at the node's time: vtable +0x58, (curve, node,
// &time) -> a + b time.
GUEST_ABI float curve_linear(u64 curve, u64, const float* t) { return load<float>(curve + 8) + load<float>(curve + 12) * *t; }
alignas(64) u8 g_curve_vtable[0x80];
u64 curve(float a, float b) {
    const u64 c = block(0x20);
    store<u64>(c, reinterpret_cast<u64>(g_curve_vtable));
    store<float>(c + 8, a);
    store<float>(c + 12, b);
    return c;
}

// --- the motions ------------------------------------------------------------------

using MotionFn = GUEST_ABI void (*)(u64 motion, u64 node, const float* step, u64 local);
constexpr u64 kEuler = 0x2e88510, kRotation = 0x2e8b100, kTransform = 0x2e8a3d0, kCompositeUpdate = 0x2e8ae30;
constexpr u64 kEulerVt = 0x570e650, kRotationVt = 0x570e790, kCompositeVt = 0x570e750;

// The node a motion moves: position +0, rotation +0x10, time +0x7c.
u64 make_node() {
    const u64 n = block(0x100);
    const float x = uni(-10, 10), y = uni(-10, 10), z = uni(-10, 10);
    put4(n, x, y, z, 1);
    random_quat(n + 0x10);
    store<float>(n + 0x7c, uni(0, 5));
    return n;
}

// One whole step h of a motion: once (as at 30), or in n frames of h/n, the
// node's time put i h/n past its whole step for the i-th frame's call and put
// back after, as the frame domain's container calls put it.
void run(MotionFn f, u64 motion, u64 node, u64 local, float h, int n) {
    f = opaque(f);
    const float t = load<float>(node + 0x7c);
    const double frame = static_cast<double>(h) / n;
    for (int i = 0; i < n; ++i) {
        const float from = static_cast<float>(i * frame);
        float s = n == 1 || g_control == 3 ? h : static_cast<float>(frame);
        store<float>(node + 0x7c, i && g_control != 2 ? addss(t, from) : t);
        f(motion, node, &s, local);
        store<float>(node + 0x7c, t);
    }
}

// The three Euler motions, by layout (velocity +0x10, the speed curve +8):
//   - 0x570e650 (sub_2e88510, 0x40 bytes): the speed curve's scale +0x24, facing +0x28, no gravity;
//   - 0x570e610 (sub_2e87a10, 0x40 bytes): the gravity curve +0x20 (its value times the effects' gravity,
//     data_5ad1150, turned into the motion's frame by the inverse of the local transform it is handed - the parent's
//     rows), the speed curve's scale +0x30, facing +0x34;
//   - 0x570e5d0 (sub_2e866f0, 0x60 bytes; also the wobble motions' base): the effects' wind world (data_5ad1148): its
//     force at the node's world place (vtable +0x10, with the motion's sources at +0x20) plus the gravity curve +0x48's
//     gravity, turned into the motion's frame; its drift (vtable +0x28 for the instance's +4) times +0x50, turned the
//     same way, added to the velocity for the move only; the speed curve's scale +0x58, facing +0x5c.
// Each call: v += step x the local force (gravity x its curve at the node's time, and the world's); then, with a speed
// curve, v changes along itself by step x the curve x the scale (from rest along the node's facing; a change that would
// reverse it stops it); then the position += step x (v + the local drift), facing the move when asked.
struct EulerLayout {
    const char* name;
    u64 fn, vt;
    u64 k, face, gravity;  // the speed curve's scale and the facing switch; the gravity curve's slot (0: none)
    u64 drift;             // the wind world's drift scale (0: no wind world)
    u64 size;
};
constexpr u64 kGravityMotion = 0x2e87a10, kGravityVt = 0x570e610, kGravity = 0x5ad1150;
constexpr u64 kWindMotion = 0x2e866f0, kWindMotionVt = 0x570e5d0, kWindWorld = 0x5ad1148;
constexpr EulerLayout kEulerLayout{"Euler motion", kEuler, kEulerVt, 0x24, 0x28, 0, 0, 0x40};
constexpr EulerLayout kGravityLayout{"gravity motion", kGravityMotion, kGravityVt, 0x30, 0x34, 0x20, 0, 0x40};
constexpr EulerLayout kWindLayout{"wind motion", kWindMotion, kWindMotionVt, 0x58, 0x5c, 0x48, 0x50, 0x60};

// The wind world the wind motion asks (a test object at data_5ad1148): a force for a place and a drift for an id.
alignas(16) float g_world_force[4], g_world_drift[4];
GUEST_ABI __m128 stub_world_force(u64, const float*, u64) { return _mm_load_ps(g_world_force); }
GUEST_ABI const float* stub_world_drift(u64, u32) { return g_world_drift; }
alignas(64) u8 g_world_vt[0x40];
alignas(64) u8 g_world[0x40];

// A rigid transform's three rows (rotation, the translation in the w lanes), as a parent's matrix is handed to a motion.
void rigid_rows(u64 m) {
    const u64 q = block(16);
    random_quat(q);
    const V4 r = v4(q);
    const double x = r.x, y = r.y, z = r.z, w = r.w;
    put4(m, 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), uni(-5, 5));
    put4(m + 16, 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), uni(-5, 5));
    put4(m + 32, 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), uni(-5, 5));
}

// The update in double, for a case that never comes near rest or a reversal: one call of step h at the node's time t.
struct EulerModel {
    double v[3], p[3];
    double gl[3] = {0, 0, 0};       // the local gravity (R^T G); 0 without a gravity curve
    double ga = 0, gb = 0;          // the gravity curve
    bool speed = false;
    double sa = 0, sb = 0, k = 0;   // the speed curve and its scale
    double fl[3] = {0, 0, 0};       // the wind world's force and drift, local
    double dl[3] = {0, 0, 0};
    double low = 1e30;              // the least speed met before a speed change, and how near a reversal it came
    double margin = 1e30;
    void call(double h, double t) {
        const double cg = ga + gb * t;
        for (int i = 0; i < 3; ++i) v[i] += h * (gl[i] * cg + fl[i]);
        if (speed) {
            const double s = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            const double K = h * (sa + sb * t) * k;
            low = std::min(low, s);
            margin = std::min(margin, s + K);
            if (-K <= s) {
                if (K != 0 && s > 1e-5)
                    for (int i = 0; i < 3; ++i) v[i] += K * v[i] / s;
            } else {
                v[0] = v[1] = v[2] = 0;
            }
        }
        for (int i = 0; i < 3; ++i) p[i] += h * (v[i] + dl[i]);
    }
    // A whole step h at t, as run() calls the game's (the node's time i h/n past t for the i-th of n frames).
    void steps(double h, double t, int n) {
        for (int i = 0; i < n; ++i) call(n == 1 ? h : h / n, i ? t + i * (h / n) : t);
    }
};

int check_euler_layout(int cases, const EulerLayout& L) {
    const auto euler = eboot_kit::fn<MotionFn>(L.fn);
    int bad = 0, tight = 0, sloped = 0, clamped = 0, facing = 0, gravity = 0, modelled = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const u64 motion = block(L.size), node = make_node(), local = block(0x40);
        store<u64>(motion, game_address(L.vt));
        // the gravity motion: its parent's rows, the effects' gravity and its curve (none, constant or sloped);
        // a case of the gravity motion has its speed curve less often, so that gravity alone is common
        EulerModel m;
        bool has_gravity = false;
        double G[3] = {0, 0, 0};
        double fmax = 0;  // the wind world's force, its size
        if (L.drift) {
            // the wind world: a force (now and then none) and a drift for the node's instance; the node's world place
            for (int i = 0; i < 4; ++i) {
                g_world_force[i] = g_rng.chance(0.2) ? 0.0f : uni(-6, 6);
                g_world_drift[i] = g_rng.chance(0.2) ? 0.0f : uni(-3, 3);
            }
            store<float>(motion + L.drift, g_rng.chance(0.2) ? 0.0f : uni(-2, 2));
            const u64 inst = block(0x98);
            store<u32>(inst + 4, g_rng.u32());
            store<u64>(node + 0x58, inst);
            for (const u64 at : {0x2cull, 0x3cull, 0x4cull}) store<float>(node + at, uni(-50, 50));
        }
        if (L.gravity) {
            rigid_rows(local);
            if (g_rng.chance(0.5)) G[1] = -9.8;
            else
                for (double& g : G) g = uni(-12, 12);
            put4(reinterpret_cast<u64>(eboot_kit::at(kGravity)), G[0], G[1], G[2], uni(-3, 3));
            has_gravity = g_rng.chance(0.8);
            if (has_gravity) {
                m.ga = uni(-2, 2);
                m.gb = g_rng.chance(0.5) ? 0.0f : uni(-2, 2);
                store<u64>(motion + L.gravity, curve(static_cast<float>(m.ga), static_cast<float>(m.gb)));
                // R^T G, the rows' rotation
                for (int i = 0; i < 3; ++i) m.gl[i] = load<float>(local + 4 * i) * G[0] + load<float>(local + 16 + 4 * i) * G[1] + load<float>(local + 32 + 4 * i) * G[2];
            } else {
                store<u64>(motion + L.gravity, 0);
                m.ga = m.gb = 0;
            }
        }
        if (L.drift) {
            // the world's force and the scaled drift, turned into the motion's frame (R^T)
            const double ws = load<float>(motion + L.drift);
            for (int i = 0; i < 3; ++i) {
                m.fl[i] = load<float>(local + 4 * i) * g_world_force[0] + load<float>(local + 16 + 4 * i) * g_world_force[1] +
                          load<float>(local + 32 + 4 * i) * g_world_force[2];
                m.dl[i] = ws * (load<float>(local + 4 * i) * g_world_drift[0] + load<float>(local + 16 + 4 * i) * g_world_drift[1] +
                                load<float>(local + 32 + 4 * i) * g_world_drift[2]);
            }
            fmax = std::sqrt(m.fl[0] * m.fl[0] + m.fl[1] * m.fl[1] + m.fl[2] * m.fl[2]);
        }
        const bool has_curve = g_rng.chance(L.gravity ? 0.6 : 0.7), slope = has_curve && g_rng.chance(0.5);
        const float a = uni(-3, 3), b = slope ? uni(-2, 2) : 0.0f, k = uni(-2, 2);
        store<u64>(motion + 8, has_curve ? curve(a, b) : 0);
        // a velocity of 1 to 6, or now and then none
        const bool still = g_rng.chance(0.1);
        V4 d{uni(-1, 1), uni(-1, 1), uni(-1, 1), 0};
        const double dl = len3(d) + 1e-9;
        const double speed = still ? 0.0 : g_rng.chance(0.2) ? uni(0.01f, 0.3f) : uni(1, 6);
        put4(motion + 0x10, d.x / dl * speed, d.y / dl * speed, d.z / dl * speed, 0);
        store<float>(motion + L.k, k);
        store<u8>(motion + L.face, g_rng.chance(0.5) ? 1 : 0);
        const float h = g_rng.chance(0.7) ? 1.0f / 30.0f : static_cast<float>(1.0 / 30.0 * uni(0.25f, 2.0f));
        // a first call with no step (the w lanes become the x ones, as every call leaves them)
        float zero = 0.0f;
        euler(motion, node, &zero, local);
        const std::vector<u8> m0 = bytes_of(motion, L.size), n0 = bytes_of(node, 0x100);
        const char* why = nullptr;
        // a step of 0 moves nothing
        euler(motion, node, &zero, local);
        if (bytes_of(motion, L.size) != m0 || bytes_of(node, 0x100) != n0) why = "a step of 0";
        put_bytes(motion, m0);
        put_bytes(node, n0);
        // whole step
        run(euler, motion, node, local, h, 1);
        const V4 pa = v4(node), va = v4(motion + 0x10), qa = v4(node + 0x10);
        const double t0 = load<float>(node + 0x7c);
        const V4 v0 = v4(reinterpret_cast<u64>(m0.data()) + 0x10), p0 = v4(reinterpret_cast<u64>(n0.data()));
        m.speed = has_curve;
        m.sa = a, m.sb = b, m.k = k;
        auto model = [&](int n) {
            EulerModel x = m;
            x.v[0] = v0.x, x.v[1] = v0.y, x.v[2] = v0.z;
            x.p[0] = p0.x, x.p[1] = p0.y, x.p[2] = p0.z;
            x.steps(h, t0, n);
            return x;
        };
        const EulerModel ma = model(1);
        double gmax = 0;  // the largest gravity acceleration the step meets
        if (has_gravity) gmax = std::sqrt(m.gl[0] * m.gl[0] + m.gl[1] * m.gl[1] + m.gl[2] * m.gl[2]) * (std::fabs(m.ga) + std::fabs(m.gb) * (t0 + h));
        gmax += fmax;
        if (has_gravity) ++gravity;
        for (const int n : {2, 4}) {
            if (why) break;
            put_bytes(motion, m0);
            put_bytes(node, n0);
            run(euler, motion, node, local, h, n);
            const V4 pb = v4(node), vb = v4(motion + 0x10), qb = v4(node + 0x10);
            const double t = load<float>(node + 0x7c);
            const double fk_max = has_curve ? (std::fabs(a) + std::fabs(b) * (t + h)) * std::fabs(k) : 0.0;
            const double euler_term = static_cast<double>(h) * h * (n - 1) / (2.0 * n);
            const double round = 2e-5 * (1.0 + len3(pa) + speed + h * gmax);
            const double vround = 2e-5 * (1.0 + speed + h * gmax);
            // the game's code against the model, whole step and frames, where the model's branches are not in doubt
            const EulerModel mb = model(n);
            const bool clean = ma.low > 1e-3 && mb.low > 1e-3 && ma.margin > 1e-4 && mb.margin > 1e-4;
            if (clean) {
                ++modelled;
                const V4 mva{ma.v[0], ma.v[1], ma.v[2], 0}, mvb{mb.v[0], mb.v[1], mb.v[2], 0};
                const V4 mpa{ma.p[0], ma.p[1], ma.p[2], 0}, mpb{mb.p[0], mb.p[1], mb.p[2], 0};
                if (dist3(va, mva) > vround || dist3(vb, mvb) > vround) why = "the velocity against the update's equations";
                else if (dist3(pa, mpa) > round || dist3(pb, mpb) > round) why = "the position against the update's equations";
                if (why && !g_quiet)
                    std::printf("  %s model: v %.6g %.6g %.6g / %.6g %.6g %.6g, p apart %.3g / %.3g\n", L.name, va.x - mva.x, va.y - mva.y, va.z - mva.z,
                                vb.x - mvb.x, vb.y - mvb.y, vb.z - mvb.z, dist3(pa, mpa), dist3(pb, mpb));
            }
            if (!why && has_gravity && !has_curve && !L.drift) {
                // gravity alone: the velocity apart by G gb h^2 (n-1)/2n (the curve read at each frame's time), the position by
                // G c(t0) h^2 (n-1)/2n - G gb h^3 (n+1)(n-1)/6n^2 (the Euler terms)
                ++tight;
                const double dv = m.gb * euler_term, c0 = m.ga + m.gb * t0;
                const double dp = c0 * euler_term - m.gb * static_cast<double>(h) * h * h * (n + 1) * (n - 1) / (6.0 * n * n);
                const V4 want_v{va.x + m.gl[0] * dv, va.y + m.gl[1] * dv, va.z + m.gl[2] * dv, 0};
                const V4 want_p{pa.x - m.gl[0] * dp, pa.y - m.gl[1] * dp, pa.z - m.gl[2] * dp, 0};
                if (dist3(vb, want_v) > vround) why = "gravity's velocity (the curve read at each frame)";
                else if (dist3(pb, want_p) > round) why = "gravity's position (the Euler terms)";
            } else if (!why && !has_gravity && !L.drift && has_curve && speed > 0 && speed > 1.5 * h * fk_max + 1e-3) {
                // the speed curve alone, no reversal within the step: the velocity stays on its line
                if (!slope) {
                    ++tight;
                    const double acc = static_cast<double>(a) * k;  // along the direction
                    if (dist3(va, vb) > 2e-5 * (1.0 + speed)) why = "a constant curve's velocity";
                    // pb - pa = -(Euler term) a d, d the velocity's direction
                    V4 want{pa.x - euler_term * acc * va.x / len3(va), pa.y - euler_term * acc * va.y / len3(va), pa.z - euler_term * acc * va.z / len3(va),
                            0};
                    if (!why && dist3(pb, want) > round) why = "a constant curve's position (the Euler term)";
                } else {
                    ++sloped;
                    // the speed apart by h^2 k b (n-1)/2n: the curve read at each frame's time
                    const double dv = euler_term * static_cast<double>(k) * b;
                    if (std::fabs(len3(vb) - len3(va) - dv) > 2e-5 * (1.0 + speed)) why = "a sloped curve's speed (the curve read at each frame)";
                    if (!why && dist3(pa, pb) > euler_term * fk_max + h * std::fabs(dv) + round) why = "a sloped curve's position";
                }
            } else if (!why && !clean && has_curve && speed > 0) {
                ++clamped;  // a reversal within the step stops it: both stop, the frames having moved less
                if (dist3(pa, pb) > h * (speed + h * (fk_max + gmax)) + round) why = "a reversal";
            } else if (!why && !clean && dist3(pa, pb) > euler_term * (fk_max + gmax) + h * h * gmax + round) {
                why = "from rest";
            }
            const double moved_a = std::sqrt((va.x + m.dl[0]) * (va.x + m.dl[0]) + (va.y + m.dl[1]) * (va.y + m.dl[1]) + (va.z + m.dl[2]) * (va.z + m.dl[2]));
            const double moved_b = std::sqrt((vb.x + m.dl[0]) * (vb.x + m.dl[0]) + (vb.y + m.dl[1]) * (vb.y + m.dl[1]) + (vb.z + m.dl[2]) * (vb.z + m.dl[2]));
            if (!why && load<u8>(motion + L.face) && clean && moved_a > 1e-2 && moved_b > 1e-2) {
                // facing the last call's move (sub_2e871a0: a rotation whose Z axis looks along it; the rotation before is not
                // read): in each run along that run's final velocity, so the two apart as far as the velocities are (which
                // the model holds), whatever the roll does near the up axis
                ++facing;
                const auto forward = [](const V4& q) {
                    return V4{2 * (q.x * q.z + q.y * q.w), 2 * (q.y * q.z - q.x * q.w), 1 - 2 * (q.x * q.x + q.y * q.y), 0};
                };
                const auto angle = [](const V4& u, const V4& v) {
                    const double c = (u.x * v.x + u.y * v.y + u.z * v.z) / std::max(1e-30, len3(u) * len3(v));
                    return std::acos(std::max(-1.0, std::min(1.0, c)));
                };
                const V4 fa = forward(qa), fb = forward(qb);
                const V4 ma_{va.x + m.dl[0], va.y + m.dl[1], va.z + m.dl[2], 0}, mb_{vb.x + m.dl[0], vb.y + m.dl[1], vb.z + m.dl[2], 0};
                const double off_a = angle(fa, ma_), off_b = angle(fb, mb_), apart = angle(fa, fb), turn = angle(ma_, mb_);
                const double noise = 2.4e-7 * (1.0 + len3(pa)) / (h / n * std::min(len3(ma_), len3(mb_))) + 1e-4;
                if (off_a > noise || off_b > noise) why = "the facing (not along the final move)";
                else if (apart > turn + 2.0 * noise) why = "the facing (the two runs' further apart than their moves)";
                if (why && !g_quiet) std::printf("  facing: off %.3g / %.3g, apart %.3g, the moves %.3g\n", off_a, off_b, apart, turn);
            }
        }
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("%s case %d (gravity %d, curve %d slope %d, speed %g, h %g): %s differs\n", L.name, c, has_gravity, has_curve, slope, speed, h, why);
    }
    if (!g_quiet)
        std::printf("decomp_sfx_classes: %s: %d cases (%d with gravity; %d held to the update's equations, %d with one Euler term exact, %d sloped, "
                    "%d reversing, %d facing), %d differ\n",
                    L.name, cases, gravity, modelled, tight, sloped, clamped, facing, bad);
    return bad;
}
int check_euler(int cases) { return check_euler_layout(cases, kEulerLayout); }
int check_gravity_motion(int cases) { return check_euler_layout(cases, kGravityLayout); }
int check_wind_motion(int cases) { return check_euler_layout(cases, kWindLayout); }

// The direction motions, by layout: the speed curve +8 (at the node's time) times the direction +0x10 (the set-up's,
// normalised; the update never writes it) added to the velocity +0x20 for the move - no scale, no reversal, no stop:
//   - 0x570e850 (sub_2e8d740, 0x40 bytes): no force (v += step x 0), facing +0x38;
//   - 0x570e810 (sub_2e8cc60, 0x50 bytes): the gravity curve +0x30's gravity, turned into the motion's frame, facing +0x44;
//   - 0x570e7d0 (sub_2e8bfd0, 0x70 bytes; the wobble 0x570e8d0's base): the wind world's force at the node's world place
//     (its sources at +0x30) plus the gravity curve +0x58's gravity, turned into the motion's frame; the world's drift
//     (vtable +0x28 for the instance's +4) times +0x60, turned the same way, added to the move; facing +0x6c.
// Each call: v += step x the local force; p += step x (v + the curve x the direction + the local drift); facing the move
// when asked. No branch: every case is held to these equations in double, the whole step and its frames alike.
struct DirLayout {
    const char* name;
    u64 fn, vt;
    u64 face, gravity, drift;  // the facing switch; the gravity curve's slot (0: none); the drift scale (0: no wind world)
    u64 size;
};
constexpr DirLayout kDirLayout{"direction motion", 0x2e8d740, 0x570e850, 0x38, 0, 0, 0x40};
constexpr DirLayout kDirGravityLayout{"direction gravity motion", 0x2e8cc60, 0x570e810, 0x44, 0x30, 0, 0x50};
constexpr DirLayout kDirWindLayout{"direction wind motion", 0x2e8bfd0, 0x570e7d0, 0x6c, 0x58, 0x60, 0x70};

struct DirModel {
    double v[3], p[3], d[3];
    double gl[3] = {0, 0, 0}, ga = 0, gb = 0;  // the local gravity and its curve
    bool speed = false;
    double sa = 0, sb = 0;                     // the speed curve
    double fl[3] = {0, 0, 0}, dl[3] = {0, 0, 0};
    double move[3] = {0, 0, 0};  // the last call's move
    void call(double h, double t) {
        const double cg = ga + gb * t, cs = speed ? sa + sb * t : 0.0;
        for (int i = 0; i < 3; ++i) v[i] += h * (gl[i] * cg + fl[i]);
        for (int i = 0; i < 3; ++i) {
            move[i] = h * (v[i] + cs * d[i] + dl[i]);
            p[i] += move[i];
        }
    }
    void steps(double h, double t, int n) {
        for (int i = 0; i < n; ++i) call(n == 1 ? h : h / n, i ? t + i * (h / n) : t);
    }
};

int check_direction_layout(int cases, const DirLayout& L) {
    const auto motion_fn = eboot_kit::fn<MotionFn>(L.fn);
    int bad = 0, gravity = 0, sloped = 0, facing = 0, winds = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const u64 motion = block(L.size), node = make_node(), local = block(0x40);
        store<u64>(motion, game_address(L.vt));
        rigid_rows(local);
        // a world vector turned into the motion's frame: the inverse of the parent's rows (R^T)
        const auto to_local = [&](const double w[3], double out[3]) {
            for (int i = 0; i < 3; ++i)
                out[i] = load<float>(local + 4 * i) * w[0] + load<float>(local + 16 + 4 * i) * w[1] + load<float>(local + 32 + 4 * i) * w[2];
        };
        DirModel m;
        double amax = 0;  // the largest acceleration the step meets
        if (L.drift) {
            // the wind world: a force (now and then none) and a drift for the node's instance; the node's world place
            for (int i = 0; i < 4; ++i) {
                g_world_force[i] = g_rng.chance(0.2) ? 0.0f : uni(-6, 6);
                g_world_drift[i] = g_rng.chance(0.2) ? 0.0f : uni(-3, 3);
            }
            store<float>(motion + L.drift, g_rng.chance(0.2) ? 0.0f : uni(-2, 2));
            const u64 inst = block(0x98);
            store<u32>(inst + 4, g_rng.u32());
            store<u64>(node + 0x58, inst);
            for (const u64 at : {0x2cull, 0x3cull, 0x4cull}) store<float>(node + at, uni(-50, 50));
            const double ws = load<float>(motion + L.drift);
            const double wf[3] = {g_world_force[0], g_world_force[1], g_world_force[2]};
            const double wd[3] = {ws * g_world_drift[0], ws * g_world_drift[1], ws * g_world_drift[2]};
            to_local(wf, m.fl);
            to_local(wd, m.dl);
            amax += len3({m.fl[0], m.fl[1], m.fl[2], 0});
            ++winds;
        }
        bool has_gravity = false;
        if (L.gravity) {
            double G[3] = {0, -9.8, 0};
            if (g_rng.chance(0.5))
                for (double& g : G) g = uni(-12, 12);
            put4(reinterpret_cast<u64>(eboot_kit::at(kGravity)), G[0], G[1], G[2], uni(-3, 3));
            has_gravity = g_rng.chance(0.8);
            if (has_gravity) {
                m.ga = uni(-2, 2);
                m.gb = g_rng.chance(0.5) ? 0.0f : uni(-2, 2);
                store<u64>(motion + L.gravity, curve(static_cast<float>(m.ga), static_cast<float>(m.gb)));
                // G as the game reads it (floats)
                const double Gf[3] = {static_cast<float>(G[0]), static_cast<float>(G[1]), static_cast<float>(G[2])};
                to_local(Gf, m.gl);
                ++gravity;
            } else {
                store<u64>(motion + L.gravity, 0);
            }
        }
        const bool has_curve = g_rng.chance(0.75), slope = has_curve && g_rng.chance(0.5);
        const float a = uni(-3, 3), b = slope ? uni(-2, 2) : 0.0f;
        store<u64>(motion + 8, has_curve ? curve(a, b) : 0);
        if (slope) ++sloped;
        // the direction: normalised as the set-up leaves it (now and then not: the update does not care); a velocity (the
        // set-up's 0, or one the forces gave)
        V4 d{uni(-1, 1), uni(-1, 1), uni(-1, 1), 0};
        const double dn = g_rng.chance(0.8) ? len3(d) + 1e-9 : 1.0;
        put4(motion + 0x10, d.x / dn, d.y / dn, d.z / dn, 0);
        const bool rest = g_rng.chance(0.3);
        put4(motion + 0x20, rest ? 0 : uni(-4, 4), rest ? 0 : uni(-4, 4), rest ? 0 : uni(-4, 4), 0);
        store<u8>(motion + L.face, g_rng.chance(0.5) ? 1 : 0);
        const float h = g_rng.chance(0.7) ? 1.0f / 30.0f : static_cast<float>(1.0 / 30.0 * uni(0.25f, 2.0f));
        // a first call with no step (the w lanes become the x ones, as every call leaves them)
        float zero = 0.0f;
        motion_fn(motion, node, &zero, local);
        const std::vector<u8> m0 = bytes_of(motion, L.size), n0 = bytes_of(node, 0x100);
        const char* why = nullptr;
        // a step of 0 moves nothing
        motion_fn(motion, node, &zero, local);
        if (bytes_of(motion, L.size) != m0 || bytes_of(node, 0x100) != n0) why = "a step of 0";
        // the whole step, then its frames, each against the equations
        const V4 v0 = v4(reinterpret_cast<u64>(m0.data()) + 0x20), p0 = v4(reinterpret_cast<u64>(n0.data()));
        const V4 d0 = v4(reinterpret_cast<u64>(m0.data()) + 0x10);
        const double t0 = load<float>(node + 0x7c);
        m.speed = has_curve, m.sa = a, m.sb = b;
        auto model = [&](int n) {
            DirModel x = m;
            x.v[0] = v0.x, x.v[1] = v0.y, x.v[2] = v0.z;
            x.p[0] = p0.x, x.p[1] = p0.y, x.p[2] = p0.z;
            x.d[0] = d0.x, x.d[1] = d0.y, x.d[2] = d0.z;
            x.steps(h, t0, n);
            return x;
        };
        if (has_gravity) amax += len3({m.gl[0], m.gl[1], m.gl[2], 0}) * (std::fabs(m.ga) + std::fabs(m.gb) * (t0 + h));
        const double cmax = has_curve ? std::fabs(a) + std::fabs(b) * (t0 + h) : 0.0;
        for (const int n : {1, 2, 4}) {
            if (why) break;
            put_bytes(motion, m0);
            put_bytes(node, n0);
            run(motion_fn, motion, node, local, h, n);
            const V4 p = v4(node), v = v4(motion + 0x20), q = v4(node + 0x10);
            const DirModel mm = model(n);
            const double speed = len3(v0) + h * amax;
            const double vround = 2e-5 * (1.0 + speed);
            const double round = 2e-5 * (1.0 + len3(p0) + speed + cmax + len3({m.dl[0], m.dl[1], m.dl[2], 0}));
            if (dist3(v, {mm.v[0], mm.v[1], mm.v[2], 0}) > vround) why = n == 1 ? "the whole step's velocity" : "the frames' velocity";
            else if (dist3(p, {mm.p[0], mm.p[1], mm.p[2], 0}) > round) why = n == 1 ? "the whole step's position" : "the frames' position";
            else if (bytes_of(motion + 0x10, 16) != std::vector<u8>(m0.begin() + 0x10, m0.begin() + 0x20)) why = "the direction (written)";
            if (why && !g_quiet)
                std::printf("  %s n %d: v apart %.3g (%.3g), p apart %.3g (%.3g)\n", L.name, n, dist3(v, {mm.v[0], mm.v[1], mm.v[2], 0}), vround,
                            dist3(p, {mm.p[0], mm.p[1], mm.p[2], 0}), round);
            const double moved = std::sqrt(mm.move[0] * mm.move[0] + mm.move[1] * mm.move[1] + mm.move[2] * mm.move[2]);
            if (!why && load<u8>(motion + L.face) && moved > 1e-2 * h / n) {
                // facing the last call's move (sub_2e871a0: a rotation whose Z axis looks along it; the rotation before is
                // not read), in each run along that run's final move
                ++facing;
                const V4 f{2 * (q.x * q.z + q.y * q.w), 2 * (q.y * q.z - q.x * q.w), 1 - 2 * (q.x * q.x + q.y * q.y), 0};
                const double cosine = (f.x * mm.move[0] + f.y * mm.move[1] + f.z * mm.move[2]) / std::max(1e-30, len3(f) * moved);
                const double off = std::acos(std::max(-1.0, std::min(1.0, cosine)));
                const double noise = 2.4e-7 * (1.0 + len3(p)) / moved + 1e-4;
                if (off > noise) why = "the facing (not along the final move)";
                if (why && !g_quiet) std::printf("  facing: off %.3g (noise %.3g), n %d\n", off, noise, n);
            }
        }
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("%s case %d (gravity %d, curve %d slope %d, h %g): %s differs\n", L.name, c, has_gravity, has_curve, slope, h, why);
    }
    if (!g_quiet)
        std::printf("decomp_sfx_classes: %s: %d cases (%d with gravity, %d with the wind world, %d sloped speed curves, %d facings), "
                    "%d differ\n",
                    L.name, cases, gravity, winds, sloped, facing, bad);
    return bad;
}
int check_direction_motion(int cases) { return check_direction_layout(cases, kDirLayout); }
int check_direction_gravity(int cases) { return check_direction_layout(cases, kDirGravityLayout); }
int check_direction_wind(int cases) { return check_direction_layout(cases, kDirWindLayout); }

int check_rotation(int cases) {
    const auto rotation = eboot_kit::fn<MotionFn>(kRotation);
    int bad = 0, single = 0, multi = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const u64 motion = block(0x40), node = make_node(), local = block(0x40);
        store<u64>(motion, game_address(kRotationVt));
        const bool curves = g_rng.chance(0.4), one_axis = g_rng.chance(0.35);
        const int axis = g_rng.range(0, 2);
        double w[3];
        for (int i = 0; i < 3; ++i) w[i] = (one_axis && i != axis) ? 0.0 : uni(-6, 6);
        if (curves) {
            // three constant curves, their scales the rates (+0x20 with +8, +0x24 with +0x10, +0x28 with +0x18)
            store<u64>(motion + 8, curve(1.0f, 0.0f));
            store<u64>(motion + 0x10, curve(1.0f, 0.0f));
            store<u64>(motion + 0x18, curve(1.0f, 0.0f));
            store<float>(motion + 0x20, static_cast<float>(w[0]));
            store<float>(motion + 0x24, static_cast<float>(w[1]));
            store<float>(motion + 0x28, static_cast<float>(w[2]));
        } else {
            store<u64>(motion + 8, 0);
            store<float>(motion + 0x10, static_cast<float>(w[0]));
            store<float>(motion + 0x14, static_cast<float>(w[1]));
            store<float>(motion + 0x18, static_cast<float>(w[2]));
        }
        store<u8>(motion + 0x30, g_rng.chance(0.5) ? 1 : 0);
        const float h = 1.0f / 30.0f;
        const std::vector<u8> m0 = bytes_of(motion, 0x40), n0 = bytes_of(node, 0x100);
        const V4 q0 = v4(node + 0x10), p0 = v4(node);
        const char* why = nullptr;
        double worst = 0, worst_th = 0;
        float zero = 0.0f;
        rotation(motion, node, &zero, local);
        if (quat_angle(v4(node + 0x10), q0) > 1e-5 || dist3(v4(node), p0) != 0.0) why = "a step of 0";
        put_bytes(node, n0);
        run(rotation, motion, node, local, h, 1);
        const V4 qa = v4(node + 0x10), pa = v4(node);
        const double wl = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        for (const int n : {2, 4}) {
            if (why) break;
            put_bytes(motion, m0);
            put_bytes(node, n0);
            run(rotation, motion, node, local, h, n);
            const V4 qb = v4(node + 0x10), pb = v4(node);
            const double turned = quat_angle(qa, qb), th = h * wl;
            worst = std::max(worst, turned);
            worst_th = th;
            if (dist3(pa, pb) != 0.0) why = "the position (a rotation moves nothing)";
            else if (one_axis && turned > 2e-5 * (1 + th)) why = "one axis's turn";
            else if (!one_axis && turned > th * th + 2e-5) why = "three axes' turn (past the Euler angles' (h w)^2)";
        }
        (one_axis ? single : multi) += 1;
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("rotation case %d (curves %d, rates %g %g %g, order %u; turned %.3g apart, a step's turn %.3g): %s differs\n", c, curves, w[0], w[1],
                        w[2], load<u8>(motion + 0x30), worst, worst_th, why);
    }
    if (!g_quiet) std::printf("decomp_sfx_classes: rotation motion: %d cases (%d one axis, %d three), %d differ\n", cases, single, multi, bad);
    return bad;
}

// A 4x4 matrix: a rotation's rows and a translation.
void random_matrix(u64 m) {
    const u64 q = block(16);
    random_quat(q);
    const V4 r = v4(q);
    const double x = r.x, y = r.y, z = r.z, w = r.w;
    put4(m, 1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0);
    put4(m + 16, 2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0);
    put4(m + 32, 2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0);
    const float tx = uni(-5, 5), ty = uni(-5, 5), tz = uni(-5, 5);
    put4(m + 48, tx, ty, tz, 1);
}

int check_transform(int cases) {
    const auto transform = eboot_kit::fn<MotionFn>(kTransform);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const u64 motion = block(0x60), node = make_node(), local = block(0x40), parent = block(0x40), other = block(0x40);
        random_matrix(parent);
        random_matrix(other);
        random_matrix(motion + 0x10);
        random_matrix(local);
        store<u64>(motion + 8, parent);
        store<u64>(motion + 0x50, other);
        store<u8>(motion + 0x58, g_rng.chance(0.5) ? 1 : 0);
        store<u8>(motion + 0x59, g_rng.chance(0.5) ? 1 : 0);
        const std::vector<u8> m0 = bytes_of(motion, 0x60), n0 = bytes_of(node, 0x100);
        std::vector<u8> m1, n1;
        bool same = true;
        for (const float s : {1.0f / 30.0f, 0.0f, 1.0f / 120.0f, 1.0f, -1.0f}) {
            put_bytes(motion, m0);
            put_bytes(node, n0);
            float step = s;
            transform(motion, node, &step, local);
            if (m1.empty()) {
                m1 = bytes_of(motion, 0x60);
                n1 = bytes_of(node, 0x100);
            } else if (bytes_of(motion, 0x60) != m1 || bytes_of(node, 0x100) != n1) {
                same = false;
            }
        }
        if (!same && ++bad <= 6 && !g_quiet) std::printf("transform case %d: a step changed what it did\n", c);
    }
    if (!g_quiet) std::printf("decomp_sfx_classes: transform motion: %d cases, every byte the same for 5 steps, %d differ\n", cases, bad);
    return bad;
}

// The curve motion 0x570e910 (sub_2e8ec00): the node's position (+0) the value of its curve (+8, vtable +0x58: a
// vector) at the node's time (+0x7c), its rotation (+0x10) kept; it never reads its step - the frame domain's node time
// is the whole of what it is handed. Every byte the same for any step, the position the curve at the node's time.
GUEST_ABI __m128 curve_vector(u64 curve, u64, const float* t) {
    const float k = load<float>(curve + 8), c = load<float>(curve + 12);
    return _mm_set_ps(1.0f, c - *t, *t * 0.5f + c, *t * k + c);
}
alignas(64) u8 g_curve4_vtable[0x80];
int check_curve_motion(int cases) {
    const auto motion_fn = eboot_kit::fn<MotionFn>(0x2e8ec00);
    store<u64>(reinterpret_cast<u64>(g_curve4_vtable) + 0x58, reinterpret_cast<u64>(&curve_vector));
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const u64 motion = block(0x20), node = make_node(), local = block(0x40), cv = block(0x20);
        store<u64>(cv, reinterpret_cast<u64>(g_curve4_vtable));
        store<float>(cv + 8, uni(-3, 3));
        store<float>(cv + 12, uni(-20, 20));
        const bool has = g_rng.chance(0.9);
        store<u64>(motion + 8, has ? cv : 0);
        store<float>(node + 0x7c, uni(0, 30));
        const std::vector<u8> m0 = bytes_of(motion, 0x20), n0 = bytes_of(node, 0x100);
        std::vector<u8> m1, n1;
        const char* why = nullptr;
        for (const float st : {1.0f / 30.0f, 0.0f, 1.0f / 120.0f, 1.0f, -1.0f}) {
            put_bytes(motion, m0);
            put_bytes(node, n0);
            float step = g_control == 5 ? 0.0f : st;
            motion_fn(motion, node, &step, local);
            if (g_control == 5 && st != 0.0f) store<float>(node, load<float>(node) + st);  // the control: a position moved by the step
            if (m1.empty()) {
                m1 = bytes_of(motion, 0x20);
                n1 = bytes_of(node, 0x100);
            } else if (bytes_of(motion, 0x20) != m1 || bytes_of(node, 0x100) != n1) {
                why = "a step changed what it did";
            }
        }
        if (!why) {
            std::vector<u8> want = n0;
            if (has) {
                float t;
                std::memcpy(&t, n0.data() + 0x7c, 4);
                const __m128 v = curve_vector(cv, node, &t);
                std::memcpy(want.data(), &v, 16);
            }
            if (n1 != want) why = has ? "the position is not the curve at the node's time" : "no curve, yet the node changed";
        }
        if (why && ++bad <= 6 && !g_quiet) std::printf("curve motion case %d: %s\n", c, why);
    }
    if (!g_quiet) std::printf("decomp_sfx_classes: curve motion: %d cases, every byte the same for 5 steps, %d differ\n", cases, bad);
    return bad;
}

int check_composite(int cases) {
    const auto composite = eboot_kit::fn<MotionFn>(kCompositeUpdate);
    const auto euler = eboot_kit::fn<MotionFn>(kEuler);
    const auto rotation = eboot_kit::fn<MotionFn>(kRotation);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const u64 node = make_node(), local = block(0x40);
        const u64 first = block(0x40), second = block(0x40), both = block(0x20);
        store<u64>(first, game_address(kEulerVt));
        const float ca = uni(-2, 2), cb = uni(-1, 1);
        store<u64>(first + 8, g_rng.chance(0.5) ? curve(ca, cb) : 0);
        const float vx = uni(-3, 3), vy = uni(-3, 3), vz = uni(-3, 3);
        put4(first + 0x10, vx, vy, vz, 0);
        store<float>(first + 0x24, uni(-2, 2));
        store<u8>(first + 0x28, g_rng.chance(0.5) ? 1 : 0);
        store<u64>(second, game_address(kRotationVt));
        store<float>(second + 0x10, uni(-4, 4));
        store<float>(second + 0x14, uni(-4, 4));
        store<float>(second + 0x18, uni(-4, 4));
        store<u64>(both, game_address(kCompositeVt));
        store<u64>(both + 8, first);
        store<u64>(both + 0x10, second);
        const std::vector<u8> f0 = bytes_of(first, 0x40), s0 = bytes_of(second, 0x40), n0 = bytes_of(node, 0x100);
        float step = g_rng.chance(0.5) ? 1.0f / 30.0f : 1.0f / 120.0f;
        composite(both, node, &step, local);
        const std::vector<u8> f1 = bytes_of(first, 0x40), s1 = bytes_of(second, 0x40), n1 = bytes_of(node, 0x100);
        put_bytes(first, f0);
        put_bytes(second, s0);
        put_bytes(node, n0);
        euler(first, node, &step, local);
        rotation(second, node, &step, local);
        if ((bytes_of(first, 0x40) != f1 || bytes_of(second, 0x40) != s1 || bytes_of(node, 0x100) != n1) && ++bad <= 6 && !g_quiet)
            std::printf("composite case %d: not its parts in order\n", c);
    }
    if (!g_quiet) std::printf("decomp_sfx_classes: composite motion: %d cases, its parts in order, %d differ\n", cases, bad);
    return bad;
}

// --- the modules ----------------------------------------------------------------------

// What the modules' stubs saw.
struct Rec {
    int what;
    u64 a, b;
    u32 dt;
};
std::vector<Rec> g_rec;
bool g_answer = false;  // what the asking stubs answer
using ModuleFn = GUEST_ABI void (*)(u64 module, u64 params, u64 context);
using ModuleCFn = GUEST_ABI void (*)(u64 module, u64 container, u64 params, u64 context);
constexpr u64 kChildren = 0x12fccf0, kSprj = 0x23749d0;
constexpr u64 kFxGlobals = 0x58b0b30, kChildAllocator = 0x5ac8290, kChildFlag = 0x5940df8;

// The children module's reach: their manager (vtable +0x18 move, +0x28 remove, +0x30 ask), a child's set-up,
// the node's modules being done, a release and the item's free.
GUEST_ABI void stub_child_move(u64, u64 a, u64 b, const float* at) { g_rec.push_back({1, a, b, bits(at[0]) ^ bits(at[1]) ^ bits(at[2])}); }
GUEST_ABI void stub_child_remove(u64, u64 a, u64 b) { g_rec.push_back({2, a, b, 0}); }
GUEST_ABI u8 stub_child_ask(u64, u64 a, u64 b) {
    g_rec.push_back({3, a, b, 0});
    return static_cast<u8>((a ^ b) & 1);
}
GUEST_ABI u8 stub_child_set_up(u64, u64 item, u64 node, const float* at) {
    g_rec.push_back({4, item, node, bits(at[0])});
    return static_cast<u8>(item >> 6 & 1);
}
GUEST_ABI u8 stub_modules_done(u64 container) {
    g_rec.push_back({5, container, 0, 0});
    return g_answer ? 1 : 0;
}
GUEST_ABI void stub_child_release(u64 globals, u64 id) { g_rec.push_back({6, globals, id, 0}); }
GUEST_ABI void stub_child_free(u64, u64 item, u64, u64) { g_rec.push_back({7, item, 0, 0}); }
alignas(64) u8 g_child_manager_vt[0x40];

int check_children(int cases) {
    const auto children = eboot_kit::fn<ModuleFn>(kChildren);
    int bad = 0, calls = 0;
    alignas(64) static u8 globals[0xc0], manager[0x10];
    store<u64>(reinterpret_cast<u64>(manager), reinterpret_cast<u64>(g_child_manager_vt));
    store<u64>(reinterpret_cast<u64>(eboot_kit::at(kFxGlobals)), reinterpret_cast<u64>(globals));
    store<u64>(reinterpret_cast<u64>(eboot_kit::at(kChildAllocator)), reinterpret_cast<u64>(globals));
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        store<u64>(reinterpret_cast<u64>(globals) + 0xb8, g_rng.chance(0.9) ? reinterpret_cast<u64>(manager) : 0);
        store<u8>(reinterpret_cast<u64>(eboot_kit::at(kChildFlag)), g_rng.chance(0.5) ? 1 : 0);
        g_answer = g_rng.chance(0.5);
        const u64 module = block(0x30), node = make_node();
        for (const u64 row : {0x20ull, 0x30ull, 0x40ull}) {
            const float x = uni(-5, 5), y = uni(-5, 5), z = uni(-5, 5), w = uni(-5, 5);
            put4(node + row, x, y, z, w);
        }
        static const u32 states[] = {0, 1, 3, 4, 5, 6, 7, 9};
        store<u32>(node + 0xdc, states[g_rng.range(0, 7)]);
        store<u64>(module + 8, node);
        u64 prev = 0;
        std::vector<u64> items;
        for (int i = g_rng.range(0, 4); i > 0; --i) {
            const u64 item = block(0x28);
            store<u64>(item + 0x10, g_rng.u32());
            store<u64>(item + 0x18, g_rng.u32());
            for (int k = 0x20; k < 0x24; ++k) store<u8>(item + k, g_rng.chance(0.4) ? 1 : 0);
            if (prev) store<u64>(prev, item);
            else store<u64>(module + 0x20, item);
            prev = item;
            items.push_back(item);
        }
        const std::vector<u8> m0 = bytes_of(module, 0x30), n0 = bytes_of(node, 0x100);
        std::vector<std::vector<u8>> i0;
        for (const u64 it : items) i0.push_back(bytes_of(it, 0x28));
        // two parameter blocks: the whole step's and a frame's, everything else in them different too
        const u64 pa = block(0x70), pb = block(0x70);
        for (int k = 0; k < 0x70; k += 4) {
            store<u32>(pa + k, g_rng.u32());
            store<u32>(pb + k, g_rng.u32());
        }
        store<float>(pa + 0x60, 1.0f / 30.0f);
        store<float>(pb + 0x60, 1.0f / 120.0f);
        g_rec.clear();
        children(module, pa, 0);
        const std::vector<Rec> ra = g_rec;
        const std::vector<u8> ma = bytes_of(module, 0x30), na = bytes_of(node, 0x100);
        std::vector<std::vector<u8>> ia;
        for (const u64 it : items) ia.push_back(bytes_of(it, 0x28));
        put_bytes(module, m0);
        put_bytes(node, n0);
        for (std::size_t i = 0; i < items.size(); ++i) put_bytes(items[i], i0[i]);
        g_rec.clear();
        children(module, pb, 0);
        bool same = g_rec.size() == ra.size() && bytes_of(module, 0x30) == ma && bytes_of(node, 0x100) == na;
        for (std::size_t i = 0; same && i < g_rec.size(); ++i)
            same = g_rec[i].what == ra[i].what && g_rec[i].a == ra[i].a && g_rec[i].b == ra[i].b && g_rec[i].dt == ra[i].dt;
        for (std::size_t i = 0; same && i < items.size(); ++i) same = bytes_of(items[i], 0x28) == ia[i];
        calls += static_cast<int>(ra.size());
        if (!same && ++bad <= 6 && !g_quiet) std::printf("children case %d (%zu items): the step changed what it did\n", c, items.size());
    }
    if (!g_quiet) std::printf("decomp_sfx_classes: children module: %d cases (%d calls out), the same for the whole step's and a frame's, %d differ\n", cases, calls, bad);
    return bad;
}

// The Sprj module's reach: its model's advance (sub_1c3e5d0) and update (sub_1c36d50), both handed the
// step times the rate; a set-up once; a lookup and a set per call; its output's collection.
GUEST_ABI void stub_model_advance(u64 object, float dt) { g_rec.push_back({10, object, 0, bits(dt)}); }
GUEST_ABI void stub_model_update(u64 object, u64, u64, u64, float dt) { g_rec.push_back({11, object, 0, bits(dt)}); }
GUEST_ABI void stub_model_set_up(u64 object) { g_rec.push_back({12, object, 0, 0}); }
alignas(16) u64 g_lookup_answer = 0x1234;
GUEST_ABI u64 stub_lookup(u64) { return reinterpret_cast<u64>(&g_lookup_answer); }
GUEST_ABI void stub_model_set(u64 object, u64 a, u64 b) { g_rec.push_back({13, object, a, static_cast<u32>(b)}); }
GUEST_ABI void stub_reserve(u64, u32) {}
GUEST_ABI void stub_collect(u64, u64, u64) {}
alignas(64) u8 g_model_vt[0x40];

int check_sprj(int cases) {
    const auto sprj = eboot_kit::fn<ModuleCFn>(kSprj);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const u64 module = block(0xd0), model = block(0x40), anim = block(0x10), state = block(0x80), setter = block(0x10), count = block(0x10);
        store<u64>(module + 0x18, model);
        store<u64>(module + 0x20, anim);
        store<u64>(module + 0x28, count);
        const float rate = g_rng.chance(0.5) ? 1.0f : uni(0.1f, 3.0f);
        store<float>(module + 0xc4, rate);
        store<u64>(model + 0x18, setter);
        store<u64>(setter, reinterpret_cast<u64>(g_model_vt));
        store<u64>(model + 0x20, state);
        store<u8>(state + 0xc, g_rng.chance(0.7) ? 1 : 0);
        store<u64>(state + 0x78, g_rng.u32());
        const std::vector<u8> s0 = bytes_of(state, 0x80);
        const u64 params = block(0x70);
        const float h = 1.0f / 30.0f;
        const char* why = nullptr;
        // the whole step
        g_rec.clear();
        store<float>(params + 0x60, h);
        sprj(module, block(0x40), params, 0);
        const std::vector<Rec> ra = g_rec;
        float whole = 0;
        std::memcpy(&whole, &ra[0].dt, 4);
        if (ra.size() < 3 || ra[0].what != 10 || ra[1].what != 11 || ra[0].dt != bits(h * rate) || ra[1].dt != bits(h * rate)) why = "the whole step's dt";
        for (const int n : {2, 4}) {
            if (why) break;
            put_bytes(state, s0);
            g_rec.clear();
            double sum = 0;
            for (int i = 0; i < n; ++i) {
                const float s = static_cast<float>(static_cast<double>(h) / n);
                store<float>(params + 0x60, s);
                const std::size_t before = g_rec.size();
                sprj(module, block(0x40), params, 0);
                // each call: the advance and the update with the frame's step times the rate, then the per-call set
                if (g_rec.size() < before + 3 || g_rec[before].dt != bits(s * rate) || g_rec[before + 1].dt != bits(s * rate)) {
                    why = "a frame's dt";
                    break;
                }
                float d;
                std::memcpy(&d, &g_rec[before].dt, 4);
                sum += d;
            }
            if (!why && std::fabs(sum - whole) > n * 1.2e-7 * std::fabs(whole) + 1e-9) why = "the frames' dt against the whole step's";
            // the set-up once, as for the whole step
            if (!why && std::count_if(g_rec.begin(), g_rec.end(), [](const Rec& r) { return r.what == 12; }) !=
                            std::count_if(ra.begin(), ra.end(), [](const Rec& r) { return r.what == 12; }))
                why = "the set-up (once)";
        }
        // a step of 0: both handed 0
        if (!why) {
            put_bytes(state, s0);
            g_rec.clear();
            store<float>(params + 0x60, 0.0f);
            sprj(module, block(0x40), params, 0);
            if (g_rec.size() < 2 || g_rec[0].dt != bits(0.0f * rate) || g_rec[1].dt != bits(0.0f * rate)) why = "a step of 0";
        }
        if (why && ++bad <= 6 && !g_quiet) std::printf("sprj case %d (rate %g): %s differs\n", c, rate, why);
    }
    if (!g_quiet) std::printf("decomp_sfx_classes: Sprj module: %d cases, %d differ\n", cases, bad);
    return bad;
}

// --- the FFX spot light module 0x56ebc40 (sub_1320260) ----------------------------------
//
// Its update sets its GX spot light from curves at the node's age (the node's time +0x7c less the module's start
// +0x10): the transform from the node's rows (sub_264a360), two colours (+0x70, +0x80) from the vec4 curves +0x38 and
// +0x40 times the instance's colour and scale, the ranges and the cone (sub_2649ca0, by way of atanf) from the curves
// +0x48..+0x60, the visibility (sub_2702000) from what they give. Two modes, +0xa0 for the colours and +0xa4 for the
// ranges: 0 stores the curves' values (+0xb0, +0xc0, +0xe0, +0xe4; +0xd0..+0xdc), 1 keeps the stored ones, 2 blends
// from the stored ones to the curves by a curve at the age (+0x88, +0x80), storing nothing. It never reads its params
// or its context, where the step is, and it counts nothing a call: the frames' last call, at the node's time, leaves
// what the whole step's call there leaves.

constexpr u64 kSpotUpdate = 0x1320260;

// A vec4 curve: (curve, node, &time) -> a + b t in every lane, a at +0x10 and b at +0x20.
GUEST_ABI __m128 curve4_linear(u64 c, u64, const float* t) {
    const __m128 a = _mm_loadu_ps(reinterpret_cast<const float*>(static_cast<std::uintptr_t>(c + 0x10)));
    const __m128 b = _mm_loadu_ps(reinterpret_cast<const float*>(static_cast<std::uintptr_t>(c + 0x20)));
    return _mm_add_ps(a, _mm_mul_ps(b, _mm_set1_ps(*t)));
}
alignas(64) u8 g_spot_curve4_vt[0x80];
u64 curve4() {
    const u64 c = block(0x40);
    store<u64>(c, reinterpret_cast<u64>(g_spot_curve4_vt));
    for (int k = 0; k < 4; ++k) {
        store<float>(c + 0x10 + 4 * k, uni(0.0f, 2.0f));
        store<float>(c + 0x20 + 4 * k, uni(-1.0f, 1.0f));
    }
    return c;
}
// The cone's setter (light, near, far, two angles) and the visibility's (owner, light, on), recorded; the query of
// what the light is attached to answered from its arguments; atanf the host's.
GUEST_ABI void stub_cone(u64 light, float a, float b, float c, float d) {
    g_rec.push_back({20, light, static_cast<u64>(bits(a)) | static_cast<u64>(bits(b)) << 32,
                     bits(c) ^ (bits(d) << 7 | bits(d) >> 25)});
}
GUEST_ABI void stub_visibility(u64 owner, u64 light, u32 on) { g_rec.push_back({21, owner, light, on & 0xff}); }
GUEST_ABI u64 stub_pass(u64 x) { return x; }
GUEST_ABI u32 stub_attached(u64 x, u64 node, u64) { return static_cast<u32>((x ^ node) >> 6 & 1); }
GUEST_ABI float host_atanf(float x) { return std::atan(x); }

struct Spot {
    u64 module, node, light, instance;
};
Spot make_spot() {
    Spot s;
    s.module = block(0x100);
    s.node = make_node();
    s.light = block(0x240);
    s.instance = block(0x50);
    for (int k = 0; k < 0x240; k += 4) store<u32>(s.light + k, g_rng.u32() & 0x3fffffffu);
    store<u8>(s.light + 0xc4, 0);
    store<u8>(s.light + 0xc5, g_rng.chance(0.2) ? 1 : 0);
    for (const u64 row : {0x20ull, 0x30ull, 0x40ull}) put4(s.node + row, uni(-5, 5), uni(-5, 5), uni(-5, 5), uni(-5, 5));
    store<u64>(s.node + 0x58, s.instance);
    put4(s.instance + 0x10, uni(0, 1), uni(0, 1), uni(0, 1), uni(0, 1));
    for (const u64 o : {0x20ull, 0x24ull, 0x28ull}) store<float>(s.instance + o, uni(0.5f, 2.0f));
    if (g_rng.chance(0.7)) {
        const u64 r = block(0x230);
        for (int k = 0; k < 0x230; k += 4) store<float>(r + k, uni(0.1f, 2.0f));
        store<float>(r + 0x60, uni(0.5f, 2.0f));
        for (const u64 o : {0x204ull, 0x20cull, 0x20dull, 0x20eull, 0x218ull}) store<u8>(r + o, g_rng.chance(0.5) ? 1 : 0);
        store<u32>(r + 0x208, g_rng.u32());
        u64 attach = 0;
        if (g_rng.chance(0.5)) {
            attach = block(0x170);
            store<u64>(attach + 0x18, block(0x10));
            store<u32>(attach + 0x164, g_rng.u32());
        }
        store<u64>(r + 0x1f0, attach);
        store<u32>(r + 0x1f8, g_rng.u32());
        store<u64>(s.instance + 0x40, r);
    } else {
        store<u64>(s.instance + 0x40, 0);
    }
    const u64 m = s.module;
    store<u64>(m + 8, s.node);
    store<float>(m + 0x10, uni(0.0f, 2.0f));
    store<u8>(m + 0x18, g_rng.chance(0.5) ? 1 : 0);
    store<u64>(m + 0x20, block(0x10));
    store<u64>(m + 0x28, s.light);
    store<u8>(m + 0x30, g_rng.chance(0.5) ? 1 : 0);
    store<u64>(m + 0x38, curve4());
    store<u64>(m + 0x40, curve4());
    for (const u64 o : {0x48ull, 0x50ull, 0x58ull, 0x60ull, 0x80ull, 0x88ull})
        store<u64>(m + o, curve(uni(0.5f, 3.0f), uni(-0.5f, 0.5f)));
    for (const u64 o : {0x90ull, 0x98ull}) store<u64>(m + o, g_rng.chance(0.5) ? curve(uni(0.5f, 3.0f), uni(-0.5f, 0.5f)) : 0);
    store<float>(m + 0x6c, uni(0.5f, 2.0f));
    store<float>(m + 0x70, uni(0.5f, 2.0f));
    store<u32>(m + 0xa0, static_cast<u32>(g_rng.range(0, 2)));
    store<u32>(m + 0xa4, static_cast<u32>(g_rng.range(0, 2)));
    for (int k = 0xb0; k < 0xe8; k += 4) store<float>(m + k, uni(0.1f, 2.0f));
    return s;
}

int check_spot_light(int cases) {
    const auto spot = opaque(eboot_kit::fn<ModuleFn>(kSpotUpdate));
    int bad = 0, held = 0, blended = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const Spot s = make_spot();
        held += load<u32>(s.module + 0xa0) == 1 || load<u32>(s.module + 0xa4) == 1;
        blended += load<u32>(s.module + 0xa0) == 2 || load<u32>(s.module + 0xa4) == 2;
        const std::vector<u8> m0 = bytes_of(s.module, 0x100), l0 = bytes_of(s.light, 0x240), n0 = bytes_of(s.node, 0x100);
        auto reset = [&] {
            put_bytes(s.module, m0);
            put_bytes(s.light, l0);
            put_bytes(s.node, n0);
            g_rec.clear();
        };
        const char* why = nullptr;
        // (1) the step is in the params, which it never reads, nor its context: two of each, everything in them different
        const u64 pa = block(0x70), pb = block(0x70), ctx = block(0x40);
        for (int k = 0; k < 0x70; k += 4) {
            store<u32>(pa + k, g_rng.u32());
            store<u32>(pb + k, g_rng.u32());
        }
        store<float>(pa + 0x60, 1.0f / 30.0f);
        store<float>(pb + 0x60, 1.0f / 120.0f);
        reset();
        spot(s.module, pa, 0);
        const std::vector<Rec> ra = g_rec;
        const std::vector<u8> ma = bytes_of(s.module, 0x100), la = bytes_of(s.light, 0x240);
        reset();
        spot(s.module, pb, ctx);
        bool same = g_rec.size() == ra.size() && bytes_of(s.module, 0x100) == ma && bytes_of(s.light, 0x240) == la;
        for (std::size_t i = 0; same && i < g_rec.size(); ++i)
            same = g_rec[i].what == ra[i].what && g_rec[i].a == ra[i].a && g_rec[i].b == ra[i].b && g_rec[i].dt == ra[i].dt;
        if (!same) why = "the params or the context changed what it did";
        // (2) the whole step's call at the node's time T, against the frame domain's 4 calls at 120 - the node's time put
        // T - 3h, T - 2h, T - h and T, as the container calls put it - and their last call's light and module
        const float t = load<float>(s.node + 0x7c), h = 1.0f / 120.0f;
        if (!why) {
            reset();
            spot(s.module, pa, 0);
            const std::vector<u8> mw = bytes_of(s.module, 0x100), lw = bytes_of(s.light, 0x240);
            Rec cone_w{};
            for (const Rec& r : g_rec)
                if (r.what == 20) cone_w = r;
            reset();
            Rec cone_f{};
            for (int i = 3; i >= 0; --i) {
                const float at = g_control == 2 ? addss(t, -3 * h) : addss(t, -i * h);
                store<float>(s.node + 0x7c, at);
                spot(s.module, pb, 0);
            }
            store<float>(s.node + 0x7c, t);
            for (const Rec& r : g_rec)
                if (r.what == 20) cone_f = r;
            if (bytes_of(s.module, 0x100) != mw) why = "the module after the frames against the whole step";
            else if (bytes_of(s.light, 0x240) != lw) why = "the light after the frames against the whole step";
            else if (cone_f.a != cone_w.a || cone_f.dt != cone_w.dt) why = "the cone after the frames against the whole step";
        }
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("spot light case %d (modes %u, %u): %s\n", c, load<u32>(s.module + 0xa0), load<u32>(s.module + 0xa4), why);
    }
    if (!g_quiet)
        std::printf("decomp_sfx_classes: FFX spot light: %d cases (%d with a mode keeping, %d blending), the step never read and the frames' last "
                    "call the whole step's, %d differ\n",
                    cases, held, blended, bad);
    return bad;
}

// --- the CPU particles (the binder 0x56e3090's system) ----------------------------------

using SystemFn = GUEST_ABI void (*)(u64 system, u64 item);
constexpr u64 kCpuSystem = 0x2cbb9c0, kGxRecord = 0x2cbcf50;

alignas(64) u8 g_scratch[1 << 16];
alignas(64) u8 g_fx_rng[0x40];
u32 g_next_id = 0;
GUEST_ABI u64 stub_mark() { return 0; }
GUEST_ABI u64 stub_scratch(u64, u64) { return reinterpret_cast<u64>(g_scratch); }
GUEST_ABI void stub_nothing() {}
GUEST_ABI u64 stub_fx_rng() { return reinterpret_cast<u64>(g_fx_rng); }
GUEST_ABI void stub_bounds(u64 a, u64 b) {
    std::memset(reinterpret_cast<void*>(a), 0, 16);
    std::memset(reinterpret_cast<void*>(b), 0, 16);
}
// The emitter attribute's create (vtable +0x18: (attribute, system +0x30, buffer, count) -> made): names the new particles.
GUEST_ABI u32 stub_create(u64, u64 system30, u64, u32 n) {
    const u64 system = system30 - 0x30;
    const u32 live = load<u32>(system + 0xd0);
    for (u32 j = 0; j < n; ++j) store<u32>(load<u64>(system + 0xe0) + 4 * (live + j), 1000 + g_next_id++);
    return n;
}
GUEST_ABI u32 stub_init(u64, u64, u64, u32) { return 0; }
GUEST_ABI void stub_attribute_bounds(u64 out, u64, u64, u64, u32) { std::memset(reinterpret_cast<void*>(out), 0, 32); }
GUEST_ABI void stub_swap(u64, u32, u32) {}
// Each attribute's +0x20 ((attribute, system +0x30)), before an item's emission.
GUEST_ABI void stub_attribute_prepare(u64, u64) {}
alignas(64) u8 g_emitter_vt[0x40], g_init_vt[0x40], g_bounds_vt[0x40];

struct Particles {
    u64 system, item;
    u32 capacity;
};
Particles make_particles() {
    Particles p;
    p.capacity = static_cast<u32>(g_rng.range(8, 48));
    const u64 s = block(0x120), def = block(0x40), d8 = block(0x20), d10 = block(0x20), holder = block(0x60), q = block(0x60);
    p.system = s;
    store<u64>(s + 8, d8);
    store<u64>(s + 0x10, d10);
    const u64 emitter = block(0x10), init = block(0x10), bounds = block(0x10);
    store<u64>(emitter, reinterpret_cast<u64>(g_emitter_vt));
    store<u64>(init, reinterpret_cast<u64>(g_init_vt));
    store<u64>(bounds, reinterpret_cast<u64>(g_bounds_vt));
    store<u64>(s + 0x18, emitter);
    store<u64>(s + 0x20, init);
    store<u64>(s + 0x28, bounds);
    store<u64>(holder + 0x58, q);  // no node behind it: no buffers asked of it
    store<u64>(s + 0x30, holder);
    store<u64>(s + 0x38, def);
    store<u32>(s + 0x40, p.capacity);
    const float t = uni(0, 20);
    store<float>(s + 0x48, t);
    store<float>(s + 0x4c, t);
    // the definition: life L x k1 + L x k2 x a particle's draw + c; the emitter where the item says
    const float L = uni(0.05f, 1.5f);
    store<float>(def + 0x28, L);
    store<float>(def + 0x2c, uni(0.5f, 1.5f));
    store<float>(def + 0x30, uni(0.0f, 0.8f));
    store<float>(def + 0x34, uni(0.0f, 0.2f));
    store<u32>(def + 0x38, 1);
    for (int k = 0; k < 7; ++k) store<u64>(s + 0xd8 + 8 * k, block(4 * p.capacity + 16));
    store<u64>(s + 0x110, g_rng.chance(0.6) ? block(4 * p.capacity + 16) : 0);
    store<u64>(s + 0x118, g_rng.chance(0.3) ? block(4 * p.capacity + 16) : 0);
    const u32 live = static_cast<u32>(g_rng.range(0, static_cast<int>(p.capacity)));
    store<u32>(s + 0xd0, live);
    for (u32 i = 0; i < live; ++i) {
        store<float>(load<u64>(s + 0xd8) + 4 * i, t - uni(0, 2.5f));  // born up to 2.5 s ago
        store<u32>(load<u64>(s + 0xe0) + 4 * i, i);
        if (const u64 seeds = load<u64>(s + 0x110)) store<u32>(seeds + 4 * i, g_rng.u32() | 1);
    }
    p.item = block(0x70);
    put4(p.item, 1, 0, 0, 0);
    put4(p.item + 16, 0, 1, 0, 0);
    put4(p.item + 32, 0, 0, 1, 0);
    const float ex = uni(-5, 5), ey = uni(-5, 5), ez = uni(-5, 5);
    put4(p.item + 48, ex, ey, ez, 1);
    return p;
}
// A particle's end (the game's formula): birth + L k1 + c, plus L k2 r for its seed's draw r.
double particle_end(const Particles& p, u32 i) {
    const u64 s = p.system, def = load<u64>(s + 0x38);
    const float L = load<float>(def + 0x28), lk1 = L * load<float>(def + 0x2c), c = load<float>(def + 0x34);
    const float birth = load<float>(load<u64>(s + 0xd8) + 4 * i);
    float life = lk1 + birth;
    if (const u64 seeds = load<u64>(s + 0x110)) {
        const u32 v = load<u32>(seeds + 4 * i);
        const u32 x = (v << 13) ^ v, y = (x >> 17) ^ x;
        const u32 r = (((y << 5) ^ y) & 0x7fffff) | 0x3f800000;
        float rf;
        std::memcpy(&rf, &r, 4);
        life = lk1 + ((L * load<float>(def + 0x30)) * (rf - 1.0f) + birth);
    }
    return static_cast<double>(c + life);
}
struct Particle {
    u32 id;
    float birth;
    u32 seed;
    bool operator<(const Particle& o) const { return id < o.id; }
};
struct Alive {
    std::vector<Particle> by_id;
    float time = 0;
    u32 live = 0;
    int newborn() const {
        return static_cast<int>(std::count_if(by_id.begin(), by_id.end(), [](const Particle& x) { return x.id >= 1000; }));
    }
};
Alive alive(const Particles& p) {
    Alive a;
    a.time = load<float>(p.system + 0x4c);
    a.live = load<u32>(p.system + 0xd0);
    const u64 seeds = load<u64>(p.system + 0x110);
    for (u32 i = 0; i < a.live; ++i)
        a.by_id.push_back({load<u32>(load<u64>(p.system + 0xe0) + 4 * i), load<float>(load<u64>(p.system + 0xd8) + 4 * i),
                           seeds ? load<u32>(seeds + 4 * i) : 0});
    std::sort(a.by_id.begin(), a.by_id.end());
    return a;
}
bool same_particles(const Alive& a, const Alive& b) {
    if (a.by_id.size() != b.by_id.size()) return false;
    for (std::size_t i = 0; i < a.by_id.size(); ++i)
        if (a.by_id[i].id != b.by_id[i].id || bits(a.by_id[i].birth) != bits(b.by_id[i].birth) || a.by_id[i].seed != b.by_id[i].seed) return false;
    return true;
}
std::vector<std::vector<u8>> snapshot(const Particles& p) {
    std::vector<std::vector<u8>> v;
    v.push_back(bytes_of(p.system, 0x120));
    for (int k = 0; k < 7; ++k) v.push_back(bytes_of(load<u64>(p.system + 0xd8 + 8 * k), 4 * p.capacity + 16));
    for (const u64 off : {0x110ull, 0x118ull})
        if (const u64 a = load<u64>(p.system + off)) v.push_back(bytes_of(a, 4 * p.capacity + 16));
    v.push_back(bytes_of(reinterpret_cast<u64>(g_fx_rng), sizeof g_fx_rng));
    return v;
}
void restore(const Particles& p, const std::vector<std::vector<u8>>& v) {
    std::size_t i = 0;
    put_bytes(p.system, v[i++]);
    for (int k = 0; k < 7; ++k) put_bytes(load<u64>(p.system + 0xd8 + 8 * k), v[i++]);
    for (const u64 off : {0x110ull, 0x118ull})
        if (const u64 a = load<u64>(p.system + off)) put_bytes(a, v[i++]);
    put_bytes(reinterpret_cast<u64>(g_fx_rng), v[i++]);
}

int check_cpu_particles(int cases) {
    const auto system = eboot_kit::fn<SystemFn>(kCpuSystem);
    int bad = 0, ended = 0, born = 0, boundary = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const Particles p = make_particles();
        for (int k = 0; k < 0x40; k += 4) store<u32>(reinterpret_cast<u64>(g_fx_rng) + k, g_rng.u32() | 1);
        const float h = 1.0f / 30.0f;
        const u32 emission = g_rng.chance(0.7) ? static_cast<u32>(g_rng.range(0, 12)) : 0;
        const std::vector<std::vector<u8>> s0 = snapshot(p);
        const char* why = nullptr;
        // a step of 0 and no emission: nothing ends, nothing is born
        {
            const Alive before = alive(p);
            store<float>(p.item + 0x60, 0.0f);
            store<u32>(p.item + 0x64, 0);
            bool stale = false;  // a particle already past its end (made so) would end now
            for (u32 i = 0; i < before.live; ++i) stale |= before.time >= particle_end(p, i);
            system(p.system, p.item);
            const Alive after = alive(p);
            if (!stale && (!same_particles(after, before) || bits(after.time) != bits(before.time))) why = "a step of 0";
            restore(p, s0);
        }
        // the whole step: one item, the count with it
        g_next_id = 0;
        store<float>(p.item + 0x60, h);
        store<u32>(p.item + 0x64, emission);
        system(p.system, p.item);
        const Alive a = alive(p);
        for (const int n : {2, 4}) {
            if (why) break;
            restore(p, s0);
            g_next_id = 0;
            for (int i = 0; i < n; ++i) {
                store<float>(p.item + 0x60, static_cast<float>(static_cast<double>(h) / n));
                store<u32>(p.item + 0x64, i == n - 1 || g_control == 1 ? emission : 0);  // the frame with the whole step carries the count
                system(p.system, p.item);
            }
            const Alive b = alive(p);
            // a particle whose end lies between the two times (a rounding apart) may go either way
            bool near = false;
            restore(p, s0);
            const u32 live0 = load<u32>(p.system + 0xd0);
            for (u32 i = 0; i < live0; ++i) {
                const double e = particle_end(p, i);
                near |= (e > std::min(a.time, b.time) - 1e-6 && e <= std::max(a.time, b.time) + 1e-6);
            }
            if (std::fabs(static_cast<double>(a.time) - b.time) > 4e-6 * (1 + std::fabs(a.time))) why = "the system's time";
            else if (near) ++boundary;
            else if (a.by_id.size() != b.by_id.size()) why = "how many are alive";
            else {
                for (std::size_t i = 0; i < a.by_id.size() && !why; ++i) {
                    const Particle &x = a.by_id[i], &y = b.by_id[i];
                    if (x.id != y.id) why = "which are alive";
                    else if (x.id < 1000 && (bits(x.birth) != bits(y.birth) || x.seed != y.seed)) why = "an older particle's birth or seed";
                    else if (x.id >= 1000 && (x.seed != y.seed || bits(x.birth) != bits(a.time) || bits(y.birth) != bits(b.time)))
                        why = "a newborn's birth (the time its count came at) or seed";
                }
            }
            if (n == 4 && !why && !near) {
                ended += static_cast<int>(live0) - (static_cast<int>(a.live) - a.newborn());
                born += a.newborn();
            }
        }
        if (why && ++bad <= 6 && !g_quiet) std::printf("cpu particles case %d (capacity %u, emission %u): %s differs\n", c, p.capacity, emission, why);
    }
    if (!g_quiet)
        std::printf("decomp_sfx_classes: CPU particles: %d cases (%d ended, %d born over their whole steps; %d with an end within a rounding of the step's, "
                    "not compared), %d differ\n",
                    cases, ended, born, boundary, bad);
    return bad;
}

// --- the GX binder's record and its type -------------------------------------------

struct Seen {
    u32 prev, cur;
};
std::vector<Seen> g_type_seen;
// The type's update (vtable +0x10: (&bounds, type, record +0x10, buffer)): what it reads of the record's times.
GUEST_ABI void stub_type_update(u64 out, u64, u64 record10, u64) {
    g_type_seen.push_back({load<u32>(record10 + 0x14), load<u32>(record10 + 0x18)});
    std::memset(reinterpret_cast<void*>(out), 0, 32);
}
alignas(64) u8 g_type_vt[0x40];

int check_gx_record(int cases) {
    const auto record_update = eboot_kit::fn<SystemFn>(kGxRecord);
    int bad = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const u64 record = block(0x80), type = block(0x20), counts = block(0x20), item = block(0x70);
        store<u64>(type, reinterpret_cast<u64>(g_type_vt));
        store<u64>(record + 8, type);
        store<u64>(record + 0x18, counts);  // +8 = 0: no buffers
        const float t = g_rng.chance(0.2) ? uni(0, 3000) : uni(0, 30);
        store<float>(record + 0x24, t);
        store<float>(record + 0x28, t);
        const float h = 1.0f / 30.0f;
        const std::vector<u8> r0 = bytes_of(record, 0x80);
        const char* why = nullptr;
        g_type_seen.clear();
        store<float>(item + 0x60, h);
        record_update(record, item);
        const float ta = load<float>(record + 0x28);
        if (g_type_seen.size() != 1 || g_type_seen[0].prev != bits(t) || g_type_seen[0].cur != bits(addss(t, h))) why = "the whole step's time chain";
        for (const int n : {2, 4}) {
            if (why) break;
            put_bytes(record, r0);
            g_type_seen.clear();
            float cur = t;
            for (int i = 0; i < n; ++i) {
                const float s = g_control == 3 ? h : static_cast<float>(static_cast<double>(h) / n);
                store<float>(item + 0x60, s);
                record_update(record, item);
                const float next = addss(cur, s);
                if (g_type_seen.size() != static_cast<std::size_t>(i + 1) || g_type_seen[i].prev != bits(cur) || g_type_seen[i].cur != bits(next)) {
                    why = "a frame's time chain (the type handed its elapsed time each item)";
                    break;
                }
                cur = next;
            }
            if (!why && std::fabs(static_cast<double>(load<float>(record + 0x28)) - ta) > (n + 1) * 1.2e-7 * (1 + std::fabs(ta)))
                why = "the record's time after the frames";
        }
        // a step of 0: the elapsed time is 0, the time stays
        if (!why) {
            put_bytes(record, r0);
            g_type_seen.clear();
            store<float>(item + 0x60, 0.0f);
            record_update(record, item);
            if (g_type_seen.size() != 1 || g_type_seen[0].prev != g_type_seen[0].cur || load<u32>(record + 0x28) != bits(t)) why = "a step of 0";
        }
        if (why && ++bad <= 6 && !g_quiet) std::printf("gx record case %d (time %g): %s differs\n", c, t, why);
    }
    if (!g_quiet) std::printf("decomp_sfx_classes: GX binder's record: %d cases, %d differ\n", cases, bad);
    return bad;
}

// --- the FFX-format GX types ------------------------------------------------------------

// Their properties are records the type's definition holds inline - +0 a kind, the index into an
// evaluator table, +8 its keys {count n, n times, n values} - and the kind 0 of each table is the plain
// one: 0x56dc6a0's the value of the last key at or before the time, 0x56dc900's the largest value, and
// 0x570eec0's the value handed to it.
using TypeUpdate = GUEST_ABI void (*)(u64 bounds, u64 type, u64 record10, u64 buffer);
void property(u64 at, const std::vector<std::pair<float, float>>& keys) {
    const u64 k = block(4 + 8 * keys.size());
    store<s32>(k, static_cast<s32>(keys.size()));
    for (std::size_t i = 0; i < keys.size(); ++i) {
        store<float>(k + 4 + 4 * i, keys[i].first);
        store<float>(k + 4 + 4 * (keys.size() + i), keys[i].second);
    }
    store<u64>(at, 0);
    store<u64>(at + 8, k);
}
// The record an FFX-format type's update reads (record +0x10): its node (+0), birth +0x10, previous +0x14
// and current +0x18 times, the matrix rows +0x20 / +0x30, +0x70; the node a plain one (its time scale 1),
// its instance's sizes +0x20 / +0x24.
u64 ffx_record(float cur, float birth) {
    const u64 r = block(0x80), node = block(0x100), inst = block(0x98);
    store<u32>(node + 0x78, 0);
    store<u64>(node + 0x58, inst);
    store<float>(inst + 0x20, uni(0.5f, 2));
    store<float>(inst + 0x24, uni(0.5f, 2));
    store<u64>(r, node);
    store<float>(r + 0x10, birth);
    store<float>(r + 0x14, cur);
    store<float>(r + 0x18, cur);
    for (int k = 0x20; k < 0x40; k += 4) store<float>(r + k, uni(-20, 20));
    store<u32>(r + 0x70, g_rng.u32());
    return r;
}

// The bounds the type 3 / 0x3d update ends with: its own vtable +0x40 (type, out, &record +0x40, three radii), a stub
// of the test's that writes the radii as the bounds.
GUEST_ABI void stub_type_bounds(u64, u64 out, u64, float a, float b, float c) {
    store<float>(out, a);
    store<float>(out + 4, b);
    store<float>(out + 8, c);
    store<float>(out + 12, 1.0f);
    std::memset(reinterpret_cast<void*>(out + 16), 0, 16);
}
alignas(64) u8 g_type3d_vt[0x80];
// The root definition's value type 3 / 0x3d reads (sub_2712f40): a fixed one.
GUEST_ABI float stub_root_value(u64) { return 0.25f; }

// The FFX-format GX types' updates (vtable +0x10), each a step's worth of what time does:
//   - 0x6b (0x56e35d0, sub_2ec0a60): only the bounds, from its properties' largest values;
//   - 0x14 / 0x3b (0x56e3530, sub_2eb6550): +0x10 += the elapsed time x the rate +0xa8 (at the scaled time,
//     through the blend +0x180);
//   - 0x2b (0x56e3580, sub_2ebc280): +0x20 += it x +0xf0 (mode +0x40 = 0), or +0x18 += it x +0x120 and +0x1c += it
//     x +0x130 (sub_2ebc510, through the blends +0x110 / +0x140 and a parent at +0x28);
//   - 3 / 0x3d (0x56ebe20, sub_1325cd0 -> sub_2ebe1b0): +0x18 += it x +0x118, +0x1c += it x +0x128 (blend +0x158,
//     a parent at +0x30), +0x110 the root definition's value, the bounds through its own +0x40;
// and their bounds. Frames of 1/60 or 1/120 s add up to the whole step's within a rounding each (and the times'
// own rounding), a step of 0 adds nothing, and the bounds are the same.
struct FfxType {
    const char* name;
    u64 fn;
    std::vector<u64> bounds_props;  // records the bounds read (0x56dc900's kind 0: the largest value)
    std::vector<u64> rate_props;    // the rates (0x56dc6a0's kind 0)
    std::vector<u64> blend_kinds;   // their blends' kinds (0x570eec0's kind 0: the value handed on)
    u64 flag;                       // the radius switch
    u64 parent;                     // the type's parent pointer (0: none)
};
int check_ffx_types(int cases) {
    const FfxType types[] = {
        {"0x6b", 0x2ec0a60, {0x40, 0x50, 0x60, 0x80, 0x90}, {}, {}, 0x130, 0},
        {"0x14", 0x2eb6550, {0x48, 0x58, 0x68, 0x78, 0x88}, {0xa8}, {0x1c4}, 0x1e8, 0x20},
        {"0x2b", 0x2ebc280, {0x50, 0x60, 0x70, 0x90, 0xa0, 0xb0}, {0xf0, 0x120, 0x130}, {0x1a8, 0x1ac}, 0x1b8, 0x28},
        {"0x3d", 0x1325cd0, {0x48, 0x58, 0x68}, {0x118, 0x128}, {0x1cc}, 0x1c4, 0x30},
    };
    int bad = 0;
    for (const FfxType& ty : types) {
        const auto update = eboot_kit::fn<TypeUpdate>(ty.fn);
        for (int c = 0; c < cases; ++c) {
            g_blocks.clear();
            const u64 type = block(0x140), def = block(0x200), parent = block(0x40);
            store<u64>(type, reinterpret_cast<u64>(g_type3d_vt));
            store<u64>(type + 8, def);
            for (const u64 off : {0x10, 0x18, 0x1c, 0x20}) store<float>(type + off, uni(-5, 5));
            if (ty.parent) store<u64>(type + ty.parent, g_rng.chance(0.5) ? parent : 0);
            for (int k = 0x20; k < 0x40; k += 4) store<float>(parent + k, uni(-2, 2));
            for (const u64 off : ty.bounds_props) {
                std::vector<std::pair<float, float>> keys;
                const int n = g_rng.range(1, 3);
                for (int i = 0; i < n; ++i) keys.push_back({uni(0, 2), uni(-3, 3)});
                property(def + off, keys);
            }
            std::vector<float> rates;
            for (const u64 off : ty.rate_props) {
                rates.push_back(g_rng.chance(0.3) ? 0.0f : uni(-4, 4));
                property(def + off, {{0.0f, rates.back()}});
            }
            for (const u64 off : ty.blend_kinds) store<u32>(def + off, 0);
            store<u32>(def + ty.flag, g_rng.u32() & 1);
            if (ty.fn == 0x2ebc280) store<u32>(def + 0x40, g_rng.chance(0.5) ? 0u : 1u);  // 0x2b's mode (+0x20, or +0x18 and +0x1c)
            const float t = g_rng.chance(0.2) ? uni(0, 1000) : uni(0, 20);
            const u64 rec = ffx_record(t, t - uni(0, 5));
            store<float>(load<u64>(load<u64>(rec) + 0x58) + 0x28, uni(0.5f, 2));
            const u64 buf = block(0x100);
            const std::vector<u8> type0 = bytes_of(type, 0x140), rec0 = bytes_of(rec, 0x80);
            const float h = 1.0f / 30.0f;
            alignas(16) u8 b_whole[32], b_frame[32];
            store<float>(rec + 0x14, t);
            store<float>(rec + 0x18, t + h);
            update(reinterpret_cast<u64>(b_whole), type, rec, buf);
            const std::vector<u8> type_whole = bytes_of(type, 0x140);
            const char* why = nullptr;
            // the times' own rounding: each frame's elapsed time is the difference of two floats near t
            const double ulp = std::ldexp(1.0, std::ilogb(std::max(1.0, std::fabs(static_cast<double>(t)) + 1.0)) - 23);
            double rmax = 0;
            for (const float r : rates) rmax = std::max(rmax, static_cast<double>(std::fabs(r)));
            for (const int n : {2, 4}) {
                put_bytes(type, type0);
                put_bytes(rec, rec0);
                float cur = t;
                for (int i = 0; i < n; ++i) {
                    const float next = g_control == 3 ? cur + h : cur + h / n;
                    store<float>(rec + 0x14, cur);
                    store<float>(rec + 0x18, next);
                    update(reinterpret_cast<u64>(b_frame), type, rec, buf);
                    cur = next;
                }
                if (std::memcmp(b_whole, b_frame, 32) != 0) {
                    why = "the bounds";
                    break;
                }
                for (u64 off = 0; off < 0x140 && !why; off += 4) {
                    float a, b;
                    std::memcpy(&a, type_whole.data() + off, 4);
                    std::memcpy(&b, reinterpret_cast<void*>(type + off), 4);
                    if (eboot_kit::same_float(a, b)) continue;
                    const double tol = (n + 1) * 1.2e-7 * (1 + std::fabs(static_cast<double>(a))) + (n + 1) * 1.2e-7 * rmax * h + 2.0 * n * ulp * rmax;
                    if (std::fabs(static_cast<double>(a) - b) > tol) why = "an accumulation";
                }
                if (why) break;
            }
            // a step of 0: nothing added
            if (!why) {
                put_bytes(type, type0);
                put_bytes(rec, rec0);
                store<float>(rec + 0x14, t);
                store<float>(rec + 0x18, t);
                update(reinterpret_cast<u64>(b_frame), type, rec, buf);
                for (const u64 off : {0x10, 0x18, 0x1c, 0x20})
                    if (load<u32>(type + off) != load<u32>(reinterpret_cast<u64>(type0.data()) + off)) why = "a step of 0";
            }
            if (why && ++bad <= 6 && !g_quiet) std::printf("FFX type %s case %d (time %g): %s differs\n", ty.name, c, t, why);
        }
    }
    if (!g_quiet) std::printf("decomp_sfx_classes: FFX-format GX types 0x6b, 0x14/0x3b, 0x2b and 3/0x3d: %d cases each, %d differ\n", cases, bad);
    return bad;
}

// --- the wind source, GX type 0x283c ----------------------------------------------------
//
// sub_12e3c30 (0x56eadf0 +0x10): an FFX wind source through the wind system's handle +0xe0 - its shape
// (+0x20: 1 a sphere, 2 a box, 3 and 4 a segment) at the record's translation, its strengths (+0x78,
// +0x7c) times a curve (+0xb8) at the record's time and, while its node ends (state 4 to 7), times what
// is left of a fade (+0xd0 += the elapsed time, over +0x58), the transform, and (+0x44) a velocity: the
// handle's move over the elapsed time times +0x48 (0 for an elapsed time of 0). An ended node
// (sub_2c4f110) lets the handle go. The wind system's calls are the test's, each seen by its last call;
// the handle's position is the translation last set on it.
struct WindSeen {
    int releases = 0, shapes = 0, strengths = 0, velocities = 0;
    float shape[28] = {};  // the last shape call's vectors and radius
    float strength[2] = {};
    float velocity[4] = {};
    float position[4] = {};  // the handle's
};
WindSeen g_wind;
bool g_wind_ended = false;  // sub_2c4f110's answer
GUEST_ABI u8 stub_node_ended(u64) { return g_wind_ended ? 1 : 0; }
GUEST_ABI void stub_wind_release(u64 handle_at) {
    ++g_wind.releases;
    store<u64>(handle_at, ~0ull);
}
GUEST_ABI __m128 stub_wind_position(u64) { return _mm_loadu_ps(g_wind.position); }
void wind_shape(int kind, const float* a, const float* b, const float* c, float r) {
    ++g_wind.shapes;
    std::memset(g_wind.shape, 0, sizeof g_wind.shape);
    g_wind.shape[0] = static_cast<float>(kind);
    std::memcpy(g_wind.shape + 1, a, 16);
    if (b) std::memcpy(g_wind.shape + 5, b, 16);
    if (c) std::memcpy(g_wind.shape + 9, c, 64);
    g_wind.shape[25] = r;
}
GUEST_ABI void stub_wind_sphere(u64, const float* at, float r) { wind_shape(1, at, nullptr, nullptr, r); }
GUEST_ABI void stub_wind_box(u64, const float* lo, const float* hi, const float* rotation) { wind_shape(2, lo, hi, rotation, 0.0f); }
GUEST_ABI void stub_wind_segment3(u64, const float* a, const float* b, float r) { wind_shape(3, a, b, nullptr, r); }
GUEST_ABI void stub_wind_segment4(u64, const float* a, const float* b, float r) { wind_shape(4, a, b, nullptr, r); }
GUEST_ABI void stub_wind_strength(u64, float a, float b) {
    ++g_wind.strengths;
    g_wind.strength[0] = a;
    g_wind.strength[1] = b;
}
GUEST_ABI void stub_wind_transform(u64, const float* m) { std::memcpy(g_wind.position, m + 12, 16); }
GUEST_ABI void stub_wind_velocity(u64, const float* v) {
    ++g_wind.velocities;
    std::memcpy(g_wind.velocity, v, 16);
}
GUEST_ABI u64 stub_wind_object(u64) { return 0; }

// The two wind types' layouts: 0x283c (sub_12e3c30) and 0x283d (sub_12e27b0, the same without the velocity, and its
// strength and transform set only with a curve).
struct WindLayout {
    const char* name;
    u64 fn, kind, sizes, velocity_on, velocity_scale, fade_time, strengths, curve, fade, handle;
};
constexpr WindLayout kWind283c{"0x283c", 0x12e3c30, 0x20, 0x24, 0x44, 0x48, 0x58, 0x78, 0xb8, 0xd0, 0xe0};
constexpr WindLayout kWind283d{"0x283d", 0x12e27b0, 0x18, 0x1c, 0, 0, 0x50, 0x60, 0x88, 0x98, 0xa0};
int check_wind_layout(int cases, const WindLayout& w) {
    const auto update = eboot_kit::fn<TypeUpdate>(w.fn);
    int bad = 0;
    long ending = 0, released = 0, moving = 0;
    for (int c = 0; c < cases; ++c) {
        g_blocks.clear();
        const u64 type = block(0x100), rec = block(0x80), node = block(0x100), inst = block(0x98);
        for (int k = 0x10; k < 0x100; k += 4) store<float>(type + k, uni(-3, 3));
        store<u32>(type + w.kind, static_cast<u32>(g_rng.range(0, 5)));
        if (w.velocity_on) store<u32>(type + w.velocity_on, g_rng.chance(0.7) ? 1u : 0u);
        store<float>(type + w.fade_time, g_rng.chance(0.1) ? 0.0f : uni(0.05f, 2));
        store<u64>(type + w.curve, g_rng.chance(w.velocity_on ? 0.6 : 0.8) ? curve(uni(0, 2), g_rng.chance(0.5) ? 0.0f : uni(-0.5f, 0.5f)) : 0);
        store<float>(type + w.fade, g_rng.chance(0.3) ? 0.0f : uni(0, 1));
        store<u64>(type + w.handle, g_rng.chance(0.1) ? ~0ull : 0x100 + static_cast<u64>(c));
        const u32 state = static_cast<u32>(g_rng.range(0, 9));
        store<u32>(node + 0xdc, state);
        store<u64>(node + 0x58, inst);  // no overrides (+0x40)
        g_wind_ended = state >= 4 && state <= 7 && g_rng.chance(0.2);
        ending += state >= 4 && state <= 7;
        released += g_wind_ended;
        store<u64>(rec, node);
        // The record's transform: rows +0x40 / +0x50 / +0x60, the translation in their last lanes, moving by v.
        for (int k = 0x40; k < 0x70; k += 4) store<float>(rec + k, uni(-1, 1));
        const bool still = g_rng.chance(0.1);
        moving += !still;
        float p0[3], v[3];
        for (int i = 0; i < 3; ++i) {
            p0[i] = uni(-20, 20);
            v[i] = still ? 0.0f : uni(-6, 6);
        }
        auto put_at = [&](double s) {
            for (int i = 0; i < 3; ++i) store<float>(rec + 0x4c + 0x10 * static_cast<u64>(i), static_cast<float>(p0[i] + v[i] * s));
        };
        const float t = g_rng.chance(0.2) ? uni(0, 1000) : uni(0, 20);
        const float h = 1.0f / 30.0f, t1 = t + h;
        const std::vector<u8> type0 = bytes_of(type, 0x100), rec0 = bytes_of(rec, 0x80);
        auto begin = [&]() {
            put_bytes(type, type0);
            put_bytes(rec, rec0);
            g_wind = WindSeen{};
            for (int i = 0; i < 3; ++i) g_wind.position[i] = p0[i];
            g_wind.position[3] = 1.0f;
        };
        alignas(16) u8 b_whole[32], b_frame[32];
        begin();
        store<float>(rec + 0x14, t);
        store<float>(rec + 0x18, t1);
        put_at(h);
        update(reinterpret_cast<u64>(b_whole), type, rec, 0);
        const std::vector<u8> type_whole = bytes_of(type, 0x100);
        const WindSeen whole = g_wind;
        const char* why = nullptr;
        const double ulp_t = std::ldexp(1.0, std::ilogb(std::max(1.0, std::fabs(static_cast<double>(t)) + 1.0)) - 23);
        const double ulp_p = std::ldexp(1.0, std::ilogb(26.0) - 23);
        // the strength's largest size: the curve's value at the end (or 1); the fade only lowers it
        const u64 strength_curve = load<u64>(type + w.curve);
        const double s_max = strength_curve ? std::fabs(static_cast<double>(load<float>(strength_curve + 8))) +
                                                  std::fabs(static_cast<double>(load<float>(strength_curve + 12))) * (std::fabs(static_cast<double>(t)) + 1.0)
                                            : 1.0;
        for (const int n : {2, 4}) {
            begin();
            float cur = t;
            for (int i = 1; i <= n; ++i) {
                const float next = g_control == 3 ? cur + h : i == n ? t1 : t + h * static_cast<float>(i) / static_cast<float>(n);
                store<float>(rec + 0x14, cur);
                store<float>(rec + 0x18, next);
                put_at(static_cast<double>(h) * i / n);
                update(reinterpret_cast<u64>(b_frame), type, rec, 0);
                cur = next;
            }
            const WindSeen& fr = g_wind;
            if (std::memcmp(b_whole, b_frame, 32) != 0) why = "the bounds";
            else if (fr.releases != whole.releases) why = "the handle let go";
            else if ((fr.shapes > 0) != (whole.shapes > 0) || std::memcmp(fr.shape, whole.shape, sizeof fr.shape) != 0) why = "the shape";
            else if (std::memcmp(fr.position, whole.position, 16) != 0) why = "the handle's position";
            else if ((fr.strengths > 0) != (whole.strengths > 0) || (fr.velocities > 0) != (whole.velocities > 0)) why = "the calls made";
            if (why) break;
            for (u64 off = 0; off < 0x100 && !why; off += 4) {
                const float a = load<float>(reinterpret_cast<u64>(type_whole.data()) + off), b = load<float>(type + off);
                if (eboot_kit::same_float(a, b)) continue;
                if (off != w.fade || std::fabs(static_cast<double>(a) - b) > 2.0 * n * (ulp_t + 1.2e-7 * (1 + std::fabs(static_cast<double>(a)))))
                    why = off == w.fade ? "the fade" : "a word of the type";
            }
            if (why) break;
            // the strengths: the curve at the same time, the fade within its rounding
            const double fade_err = 2.0 * n * (ulp_t + 1.2e-7) / std::max(0.05, static_cast<double>(load<float>(type + w.fade_time)));
            for (int i = 0; i < 2 && !why; ++i) {
                const double a = whole.strength[i], b = fr.strength[i];
                const double k = s_max * std::fabs(static_cast<double>(load<float>(type + w.strengths + 4 * static_cast<u64>(i))));
                if (!eboot_kit::same_float(whole.strength[i], fr.strength[i]) && std::fabs(a - b) > 4e-6 * (1 + std::fabs(a)) + 4.0 * fade_err * k)
                    why = "the strengths";
            }
            // the velocity: the move over the elapsed time (times 0.09), within the positions' and the times' rounding
            double vmax = 0;
            for (int i = 0; i < 4; ++i) vmax = std::max(vmax, std::fabs(static_cast<double>(whole.velocity[i])));
            const double tol_v = 4e-6 * (1 + vmax) + vmax * 2.0 * n * ulp_t / h +
                                 3.0 * n * ulp_p / h * s_max * (w.velocity_scale ? std::fabs(static_cast<double>(load<float>(type + w.velocity_scale))) : 0.0);
            for (int i = 0; i < 4 && !why; ++i) {
                const double a = whole.velocity[i], b = fr.velocity[i];
                if (!eboot_kit::same_float(whole.velocity[i], fr.velocity[i]) && std::fabs(a - b) > tol_v) why = "the velocity";
            }
            if (why) break;
        }
        // a step of 0: no fade added, no velocity
        if (!why) {
            begin();
            store<float>(rec + 0x14, t);
            store<float>(rec + 0x18, t);
            put_at(0.0);
            update(reinterpret_cast<u64>(b_frame), type, rec, 0);
            if (load<u32>(type + w.fade) != load<u32>(reinterpret_cast<u64>(type0.data()) + w.fade)) why = "a step of 0's fade";
            for (int i = 0; i < 3 && !why; ++i)
                if (g_wind.velocities && g_wind.velocity[i] != 0.0f) why = "a step of 0's velocity";
        }
        if (why && ++bad <= 6 && !g_quiet)
            std::printf("wind source %s case %d (shape %u, state %u, time %g): %s differs\n", w.name, c,
                        load<u32>(reinterpret_cast<u64>(type0.data()) + w.kind), state, t, why);
    }
    if (!g_quiet)
        std::printf("decomp_sfx_classes: the wind source %s: %d cases (%ld ending, %ld of them let go, %ld moving), %d differ\n", w.name, cases, ending,
                    released, moving, bad);
    return bad;
}
int check_wind_source(int cases) { return check_wind_layout(cases, kWind283c) + check_wind_layout(cases, kWind283d); }

// --- the GPU particle simulations ------------------------------------------------------

using SimFn = GUEST_ABI void (*)(u64 sim, float dt);
constexpr u64 kStandardSim = 0x13040b0, kSparkSim = 0x12fef10, kFxRngHolder = 0x5940de8;
alignas(64) u8 g_rng_holder[0x80];

struct Emitter {
    float interval;  // s
    int count;       // particles an emission
    bool capped;     // whether an update at 30 passes the cap
};
// A simulation with its emission alone: the parameter blends (+0x3860), the GX
// paths (+0x2a1c) and the follow (+0x283c) off; never ending (+0x253c); a fixed
// interval and count (no draws change them).
u64 make_sim(const Emitter& e) {
    const u64 sim = block(0x3b40);
    store<u8>(sim + 0x3860, 0);
    store<u32>(sim + 0x2a1c, 0);
    store<u32>(sim + 0x283c, 0xfffffff);
    store<u32>(sim + 0x253c, 1);
    store<u8>(sim + 0x3872, 1);
    store<float>(sim + 0x3888, e.interval);
    store<float>(sim + 0x393c, 1.0f);
    store<float>(sim + 0x2600, e.interval);
    store<float>(sim + 0x2604, 0.0f);
    store<float>(sim + 0x2608, 0.0f);
    store<u32>(sim + 0x25e0, static_cast<u32>(e.count));
    store<u32>(sim + 0x25e4, static_cast<u32>(e.count));
    store<u32>(sim + 0x25dc, 0);
    store<u32>(sim + 0x3944, 0);
    store<float>(sim + 0x3940, 1.0f);
    store<u32>(sim + 0x29c4, 0);
    store<float>(sim + 0x29a8, 1.0f);
    store<float>(sim + 0x29ac, 1.0f);
    store<u32>(sim + 0x2890, 1000);
    store<float>(sim + 0x3a2c, uni(0, 100));
    return sim;
}

int check_sims(int cases) {
    int bad = 0, capped_cases = 0;
    double capped_ratio = 0;
    for (const u64 fn : {kStandardSim, kSparkSim}) {
        const auto sim_update = eboot_kit::fn<SimFn>(fn);
        for (int c = 0; c < cases; ++c) {
            g_blocks.clear();
            Emitter e;
            // An update emits at most its cap - 20, or 16 + 2 x the count from 2 on - and an interval is at
            // least 0.01 s (the re-roll's floor): an emitter of up to 7 a time never meets it at 30; of 16 or
            // more at 100 a second meets it every whole step.
            e.capped = g_rng.chance(0.25);
            e.count = e.capped ? g_rng.range(16, 40) : g_rng.range(1, 7);
            e.interval = e.capped ? 0.01f : uni(0.01f, 0.4f);
            const Emitter em = e;
            long totals[3] = {};
            float times[3] = {};
            const int ns[3] = {1, 2, 4};
            for (int k = 0; k < 3; ++k) {
                g_blocks.clear();
                std::memset(g_rng_holder, 0x5a, sizeof g_rng_holder);
                const u64 sim = make_sim(em);
                store<float>(sim + 0x3a2c, 10.0f);
                for (int tick = 0; tick < 30; ++tick)
                    for (int i = 0; i < ns[k]; ++i) {
                        sim_update(sim, static_cast<float>(1.0 / 30.0 / (g_control == 4 ? 1 : ns[k])));
                        totals[k] += load<u32>(sim + 0x3878);
                    }
                times[k] = load<float>(sim + 0x3a2c);
            }
            const char* why = nullptr;
            for (int k = 1; k < 3 && !why; ++k) {
                if (std::fabs(static_cast<double>(times[k]) - times[0]) > 30 * ns[k] * 1.2e-6 * 11.0) why = "the simulation's time";
                else if (!e.capped && std::labs(totals[k] - totals[0]) > e.count) why = "the particles emitted in a second";
                else if (e.capped && totals[k] < totals[0]) why = "a capped emitter emitting less in frames";
            }
            if (e.capped) {
                ++capped_cases;
                capped_ratio += totals[0] ? static_cast<double>(totals[2]) / totals[0] : 1.0;
            }
            if (why && ++bad <= 6 && !g_quiet)
                std::printf("%s case %d (interval %g, count %d, capped %d: %ld %ld %ld): %s differs\n", fn == kStandardSim ? "StandardParticle" : "Spark", c,
                            e.interval, e.count, e.capped, totals[0], totals[1], totals[2], why);
        }
    }
    if (!g_quiet)
        std::printf("decomp_sfx_classes: StandardParticle and Spark simulations: %d cases each, %d differ; %d capped emitters emit %.2fx their whole steps' "
                    "particles in 120's frames\n",
                    cases, bad, capped_cases, capped_cases ? capped_ratio / capped_cases : 0.0);
    return bad;
}

}  // namespace

int main() {
    eboot_kit::load("decomp_sfx_classes");
    store<u64>(reinterpret_cast<u64>(g_curve_vtable) + 0x58, reinterpret_cast<u64>(&curve_linear));
    for (u8* vt : {g_emitter_vt, g_init_vt, g_bounds_vt}) store<u64>(reinterpret_cast<u64>(vt) + 0x10, reinterpret_cast<u64>(&stub_swap));
    store<u64>(reinterpret_cast<u64>(g_emitter_vt) + 0x18, reinterpret_cast<u64>(&stub_create));
    store<u64>(reinterpret_cast<u64>(g_init_vt) + 0x18, reinterpret_cast<u64>(&stub_init));
    store<u64>(reinterpret_cast<u64>(g_bounds_vt) + 0x18, reinterpret_cast<u64>(&stub_attribute_bounds));
    store<u64>(reinterpret_cast<u64>(g_type_vt) + 0x10, reinterpret_cast<u64>(&stub_type_update));
    store<u64>(reinterpret_cast<u64>(g_type3d_vt) + 0x40, reinterpret_cast<u64>(&stub_type_bounds));
    eboot_kit::stub(0x2712f40, reinterpret_cast<const void*>(&stub_root_value));
    // The wind motion's wind world (data_5ad1148): the test's force and drift.
    store<u64>(reinterpret_cast<u64>(g_world_vt) + 0x10, reinterpret_cast<u64>(&stub_world_force));
    store<u64>(reinterpret_cast<u64>(g_world_vt) + 0x28, reinterpret_cast<u64>(&stub_world_drift));
    store<u64>(reinterpret_cast<u64>(g_world), reinterpret_cast<u64>(g_world_vt));
    store<u64>(reinterpret_cast<u64>(eboot_kit::at(kWindWorld)), reinterpret_cast<u64>(g_world));
    // The wind source's reach: the node's end (sub_2c4f110) and the wind system's calls.
    eboot_kit::stub(0x2c4f110, reinterpret_cast<const void*>(&stub_node_ended));
    eboot_kit::stub(0x2717e10, reinterpret_cast<const void*>(&stub_wind_release));
    eboot_kit::stub(0x27182b0, reinterpret_cast<const void*>(&stub_wind_position));
    eboot_kit::stub(0x2718050, reinterpret_cast<const void*>(&stub_wind_sphere));
    eboot_kit::stub(0x27181c0, reinterpret_cast<const void*>(&stub_wind_box));
    eboot_kit::stub(0x27180c0, reinterpret_cast<const void*>(&stub_wind_segment3));
    eboot_kit::stub(0x2718140, reinterpret_cast<const void*>(&stub_wind_segment4));
    eboot_kit::stub(0x2718440, reinterpret_cast<const void*>(&stub_wind_strength));
    eboot_kit::stub(0x2718370, reinterpret_cast<const void*>(&stub_wind_transform));
    eboot_kit::stub(0x2718240, reinterpret_cast<const void*>(&stub_wind_velocity));
    eboot_kit::stub(0x2717e50, reinterpret_cast<const void*>(&stub_wind_object));
    for (u8* vt : {g_emitter_vt, g_init_vt, g_bounds_vt}) store<u64>(reinterpret_cast<u64>(vt) + 0x20, reinterpret_cast<u64>(&stub_attribute_prepare));
    // What the systems reach and a test does not set up: the scratch allocator, the buffers' fills, the bounds, the FX RNG.
    eboot_kit::stub(0x247d5b0, reinterpret_cast<const void*>(&stub_mark));
    eboot_kit::stub(0x247d4b0, reinterpret_cast<const void*>(&stub_scratch));
    eboot_kit::stub(0x247d5f0, reinterpret_cast<const void*>(&stub_nothing));
    eboot_kit::stub(0x2c799a0, reinterpret_cast<const void*>(&stub_nothing));
    eboot_kit::stub(0x2cd0480, reinterpret_cast<const void*>(&stub_bounds));
    eboot_kit::stub(0x2c6d580, reinterpret_cast<const void*>(&stub_fx_rng));
    store<u64>(reinterpret_cast<u64>(eboot_kit::at(kFxRngHolder)), reinterpret_cast<u64>(g_rng_holder));
    // The modules' reach.
    store<u64>(reinterpret_cast<u64>(g_child_manager_vt) + 0x18, reinterpret_cast<u64>(&stub_child_move));
    store<u64>(reinterpret_cast<u64>(g_child_manager_vt) + 0x28, reinterpret_cast<u64>(&stub_child_remove));
    store<u64>(reinterpret_cast<u64>(g_child_manager_vt) + 0x30, reinterpret_cast<u64>(&stub_child_ask));
    eboot_kit::stub(0x12fcc00, reinterpret_cast<const void*>(&stub_child_set_up));
    eboot_kit::stub(0x2c4c6b0, reinterpret_cast<const void*>(&stub_modules_done));
    eboot_kit::stub(0x2704e00, reinterpret_cast<const void*>(&stub_child_release));
    eboot_kit::stub(0x2fa1c20, reinterpret_cast<const void*>(&stub_child_free));
    eboot_kit::stub(0x1c3e5d0, reinterpret_cast<const void*>(&stub_model_advance));
    eboot_kit::stub(0x1c36d50, reinterpret_cast<const void*>(&stub_model_update));
    eboot_kit::stub(0x213c500, reinterpret_cast<const void*>(&stub_model_set_up));
    eboot_kit::stub(0xac5970, reinterpret_cast<const void*>(&stub_lookup));
    eboot_kit::stub(0x2160500, reinterpret_cast<const void*>(&stub_reserve));
    eboot_kit::stub(0x1c37770, reinterpret_cast<const void*>(&stub_collect));
    store<u64>(reinterpret_cast<u64>(g_model_vt) + 0x30, reinterpret_cast<u64>(&stub_model_set));
    // The FFX spot light's reach: its cone's setter, its visibility, what its light is attached to; atanf.
    store<u64>(reinterpret_cast<u64>(g_spot_curve4_vt) + 0x58, reinterpret_cast<u64>(&curve4_linear));
    eboot_kit::stub(0x2649ca0, reinterpret_cast<const void*>(&stub_cone));
    eboot_kit::stub(0x2702000, reinterpret_cast<const void*>(&stub_visibility));
    eboot_kit::stub(0x131dbb0, reinterpret_cast<const void*>(&stub_pass));
    eboot_kit::stub(0x2c545f0, reinterpret_cast<const void*>(&stub_pass));
    eboot_kit::stub(0x2713030, reinterpret_cast<const void*>(&stub_attached));
    if (!eboot_kit::bind("atanf", reinterpret_cast<const void*>(&host_atanf))) {
        std::printf("decomp_sfx_classes: no atanf import to bind\n");
        return 1;
    }

    constexpr int kCases = 4000;
    int bad = 0;
    bad += check_euler(kCases);
    bad += check_gravity_motion(kCases);
    bad += check_wind_motion(kCases);
    bad += check_direction_motion(kCases);
    bad += check_direction_gravity(kCases);
    bad += check_direction_wind(kCases);
    bad += check_rotation(kCases);
    bad += check_transform(kCases / 4);
    bad += check_curve_motion(kCases / 4);
    bad += check_composite(kCases / 4);
    bad += check_children(kCases / 4);
    bad += check_sprj(kCases / 4);
    bad += check_spot_light(kCases / 4);
    bad += check_cpu_particles(kCases);
    bad += check_gx_record(kCases);
    bad += check_sims(kCases / 8);
    bad += check_ffx_types(kCases);
    bad += check_wind_source(kCases);
    // The controls: each check catches the frame domain it guards against.
    g_quiet = true;
    int missed = 0;
    const struct {
        int control;
        int (*check)(int);
        const char* what;
    } controls[] = {{1, check_cpu_particles, "the emission count on every frame (the CPU particles)"},
                    {2, check_euler, "the node's time left at its whole step (the Euler motion's curve)"},
                    {2, check_spot_light, "the node's time left at its whole step (the FFX spot light's curves)"},
                    {2, check_gravity_motion, "the node's time left at its whole step (the gravity motion's curves)"},
                    {3, check_gravity_motion, "each frame handed the whole step (the gravity motion)"},
                    {2, check_wind_motion, "the node's time left at its whole step (the wind motion's curves)"},
                    {3, check_wind_motion, "each frame handed the whole step (the wind motion)"},
                    {2, check_direction_motion, "the node's time left at its whole step (the direction motion's curve)"},
                    {3, check_direction_motion, "each frame handed the whole step (the direction motion)"},
                    {2, check_direction_gravity, "the node's time left at its whole step (the direction gravity motion's curves)"},
                    {3, check_direction_gravity, "each frame handed the whole step (the direction gravity motion)"},
                    {2, check_direction_wind, "the node's time left at its whole step (the direction wind motion's curves)"},
                    {3, check_direction_wind, "each frame handed the whole step (the direction wind motion)"},
                    {3, check_gx_record, "each frame handed the whole step (the GX record)"},
                    {3, check_ffx_types, "each frame handed the whole step (the FFX-format GX types' rates)"},
                    {3, check_wind_source, "each frame handed the whole step (the wind source's fade, strengths and velocity)"},
                    {5, check_curve_motion, "a position moved by the step (the curve motion)"},
                    {4, check_sims, "each simulation update the whole step's time"}};
    for (const auto& k : controls) {
        g_control = k.control;
        if (!k.check(400)) {
            std::printf("decomp_sfx_classes: control %d not caught: %s\n", k.control, k.what);
            ++missed;
        }
    }
    g_control = 0;
    const int ncontrols = static_cast<int>(sizeof controls / sizeof controls[0]);
    std::printf("decomp_sfx_classes: %s; %d of %d controls caught\n", bad ? "the converted classes do not all hold" : "the converted classes hold",
                ncontrols - missed, ncontrols);
    return bad || missed ? 1 : 0;
}
