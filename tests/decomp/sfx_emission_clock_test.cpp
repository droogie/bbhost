// The emission clock (decomp/sfx/node_update.cpp's clock_events, the node
// emission's Edges and Event halves, decomp/sfx/node_emit.cpp) against the
// game's emission at 60 updates a second in exact arithmetic - sub_2c4e6b0's
// rule: an event at each update whose time reaches the next, the next moved
// to that time and the interval - on one staged node emitting CPU particles
// or children, its interval on the 1/60 s grid (0, 1/120, 1/60, 1/30, 0.05,
// 1/15) or well off it, in state 1 or set up on the first whole step (state
// 0), its limit and end far off and its controller on its record, stepped by
// the node update through the effects' schedule (engine/sfx_schedule.h) at 30,
// 60, 120 and 144 fps and jittered frame times: the same events in order -
// each count given the sub-step's time within the whole steps' roundings, the
// counts added, the children - and the same count made; children made on
// whole steps' frames only. The game's own float adds would now and then let
// an interval of two sub-steps or more wait one more (its next a rounding past
// the update's time); the clock meets the next within a few roundings of the
// node's time, as the exact rule does. Each of the clock's mutants must be
// caught. Skips without the 1.09 eboot.
#define DECOMP_SFX_NODE_UPDATE_MUTANTS
int g_sfx_node_mutant = 0;
#include "decomp/sfx/node_update.cpp"

#include "eboot_kit.h"
#include "engine/sfx_schedule.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x60u);
bool g_quiet = false;

u32 bits(float f) {
    u32 u;
    std::memcpy(&u, &f, 4);
    return u;
}
float addss(float x, float y) {
    __asm__("addss %1, %0" : "+x"(x) : "x"(y));
    return x;
}

constexpr u64 kAdd = 0x2c4c7f0;         // (container, count)
constexpr u64 kKids = 0x2c6d780;        // children
constexpr u64 kKidsAt = 0x2c77050;      // children with the record's +0x70

std::mt19937_64 g_answers;
struct Event {
    int what;  // 1 a count (the time given), 2 an add (the count), 3 children
    u32 a;
    bool tick;  // made on a whole step's frame (ours)
    bool operator==(const Event& o) const { return what == o.what && a == o.a; }
};
std::vector<Event> g_events;
bool g_frame_tick = true;

GUEST_ABI void stub_container(u64, float*, u64, u64) {}
GUEST_ABI u32 stub_state_of(u64 controller) { return load<u32>(controller + 4); }
GUEST_ABI u8 stub_state_holds(u64, u32) { return 1; }
GUEST_ABI u64 stub_pick(u64, u64, u64) { return 0; }
GUEST_ABI void stub_apply(u64, u64, u64) {}
GUEST_ABI void stub_bind(u64, u64, u32, u64) {}
GUEST_ABI u8 stub_set_up(u64, u64, u64, u64) { return 1; }
GUEST_ABI u32 stub_linger(u64, u64, u64) {
    std::printf("decomp_sfx_emission_clock: a linger in a case set up to have none\n");
    return 1;
}
GUEST_ABI void stub_end_modules(u64) {}
GUEST_ABI u8 stub_modules_done(u64) { return 1; }
GUEST_ABI u8 stub_children_gone(u64) { return 0; }
GUEST_ABI u64 stub_item_data(u64) { return 0; }
GUEST_ABI u8 stub_one_item(u64, u64, u64, u32, u64) { return 1; }
GUEST_ABI u8 stub_three_items(u64, u64, u64, u64, u64, u32, u32, u32, float*, u64) { return 1; }
GUEST_ABI s32 stub_count(u64, u64, float* time) {
    g_events.push_back({1, bits(*time), g_frame_tick});
    return static_cast<s32>(g_answers() % 4);
}
GUEST_ABI void stub_add(u64, u32 count) { g_events.push_back({2, count, g_frame_tick}); }
GUEST_ABI void stub_kids(u32 id, u64, u64, u64, u64) { g_events.push_back({3, id, g_frame_tick}); }
GUEST_ABI void stub_kids_at(u32 id, u64, u64, u64, u64, u64) { g_events.push_back({3, id ^ 0x80000000u, g_frame_tick}); }

std::vector<std::vector<u8>> g_blocks;
u64 block(std::size_t n) {
    g_blocks.emplace_back(n + 64, 0);
    return (reinterpret_cast<u64>(g_blocks.back().data()) + 63) & ~63ull;
}

struct Case {
    u64 node, def, inst;
    u64 seed;
    bool children, staged0;
    float interval, scale;
};

Case make_case() {
    g_blocks.clear();
    Case c{};
    c.seed = g_rng.next();
    c.children = g_rng.chance(0.3);
    c.staged0 = g_rng.chance(0.3);
    c.scale = g_rng.chance(0.6) ? 1.0f : g_rng.uniform(0.5f, 1.5f);
    // Intervals on the 1/60 s grid (0, 1/120, 1/60, 1/30, 0.05, 1/15 - the scale on them as on the sub-steps) or well
    // off it (a whole number of sub-steps and a tenth to nine tenths of one): no next within a rounding of a sub-step
    // but on one.
    {
        const float h = (1.0f / 60.0f) * c.scale;
        static const float kSteps[] = {0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 4.0f};
        c.interval = g_rng.chance(0.75) ? kSteps[g_rng.range(0, 5)] * h : (static_cast<float>(g_rng.range(0, 15)) + g_rng.uniform(0.1f, 0.9f)) * h;
    }
    c.inst = block(0x98);
    store<u8>(c.inst + 0x88, 3);  // live: bit 1 (the controller asked), not ending
    store<u64>(c.inst + 0x78, block(0x10));
    c.def = block(0x40);
    const u64 d = block(0x40), index = block(4 * 8), recs = block(0xa0 * 2);
    for (int i = 0; i < 8; ++i) store<u32>(index + 4 * i, 0);
    const u64 vt = block(0x80);
    store<u64>(vt + 0x58, reinterpret_cast<u64>(&stub_count));
    const u64 emitter = block(0x20);
    store<u64>(emitter, vt);
    store<u32>(recs + 4, c.children ? (2u | (g_rng.chance(0.5) ? 0x20u : 0u)) : 4u);
    store<float>(recs + 8, 0.0f);  // its start: at once
    store<u64>(recs + 0x28, emitter);
    store<u32>(recs + 0x80, static_cast<u32>(g_rng.range(1, 1000)));
    store<u64>(d, 1);
    store<u64>(d + 8, index);
    store<u64>(d + 0x10, recs);
    c.node = block(0x100);
    store<u32>(c.node + 0x78, 1u | 8u);  // active, staged
    store<u64>(c.node + 0x58, c.inst);
    store<u64>(c.node + 0x60, block(0x10));  // not the root
    const u64 ctl = block(0x30);
    store<u32>(ctl + 4, 0);
    store<u64>(c.node + 0xc8, ctl);
    store<u64>(c.node + 0xd0, d);
    store<u32>(c.node + 0xd8, 0);
    store<u32>(c.node + 0xdc, c.staged0 ? 0u : 1u);
    const float t = g_rng.uniform(0.0f, 4.0f);
    store<float>(c.node + 0x7c, t);
    store<u32>(c.node + 0xe0, 0);
    {
        const float h = (1.0f / 60.0f) * c.scale;
        const int m = g_rng.range(0, 5);
        store<float>(c.node + 0xe4, g_rng.chance(0.5) ? t + static_cast<float>(m) * h : t + (static_cast<float>(m) + g_rng.uniform(0.1f, 0.9f)) * h);
    }
    store<u32>(c.node + 0xe8, 0x7fffffffu);
    store<float>(c.node + 0xec, 1.0e9f);
    store<float>(c.node + 0xf0, c.interval);
    return c;
}

using EmitFn = GUEST_ABI u32 (*)(u64 node, u64 definition, u64 context);

int check(int cases, double fps) {
    int bad = 0;
    for (int k = 0; k < cases; ++k) {
        const Case c = make_case();
        std::vector<u8> before(0x100);
        std::memcpy(before.data(), reinterpret_cast<void*>(c.node), 0x100);
        // Ours: frames through the schedule.
        std::mt19937_64 jitter(c.seed);
        double carry = 0.0;
        g_answers.seed(c.seed);
        g_events.clear();
        const float t0 = load<float>(c.node + 0x7c);
        const float counted_unscaled = 1.0f / 30.0f;
        float last_end = t0, steps_end = t0;
        int ticks = 0;
        const int frames = g_rng.range(4, 40);
        for (int frame = 0; frame < frames; ++frame) {
            const double seconds = fps > 0.0 ? 1.0 / fps : std::uniform_real_distribution<double>(1.0 / 240, 1.0 / 20)(jitter);
            const sfx_schedule::Frame f = sfx_schedule::next(carry, seconds, 30);
            sfx_node::Steps st{f.whole ? f.seconds * c.scale : 0.0f, f.frame * c.scale, f.from * c.scale, f.into * c.scale, f.whole > 0};
            st.smooth = true;
            st.sub = (1.0f / 60.0f) * c.scale;
            const float base = load<float>(c.node + 0x7c);
            g_frame_tick = st.tick;
            sfx_node::update(c.node, st, c.def, 0);
            last_end = st.tick ? addss(addss(base, st.counted), st.into) : addss(base, st.into);
            if (st.tick) {
                ++ticks;
                steps_end = addss(base, st.counted);
            }
        }
        const std::vector<Event> ours = g_events;
        const u32 ours_made = load<u32>(c.node + 0xe0), ours_next = load<u32>(c.node + 0xe4);
        std::memcpy(reinterpret_cast<void*>(c.node), before.data(), 0x100);
        // The game's emission at 60 updates a second, in exact arithmetic: an event at every 1/60 s sub-step of the node's
        // time that reaches the next, the next moved to that sub-step and the interval, one count drawn an event (a CPU
        // emitter's: its count, then its add when above none; children: the children), for as many sub-steps as ours
        // covered - two a whole step from its time, up to where the last frame's span ended (children: the whole steps').
        int sub_steps = 0;
        {
            const float limit = c.staged0 && !ticks ? t0 : c.children ? steps_end : last_end;
            float base = t0;
            for (int j = 0; j < 4000 && base < limit; ++j, base = addss(base, counted_unscaled * c.scale)) {
                float s = base;
                for (int q = 0; q < 2 && s < limit; ++q, s = addss(s, (1.0f / 60.0f) * c.scale)) ++sub_steps;
            }
        }
        g_answers.seed(c.seed);
        g_events.clear();
        const double h = static_cast<double>((1.0f / 60.0f) * c.scale);
        double next = static_cast<double>(load<float>(c.node + 0xe4));
        const u32 id = load<u32>(load<u64>(load<u64>(c.node + 0xd0) + 0x10) + 0x80);
        const bool at = (load<u32>(load<u64>(load<u64>(c.node + 0xd0) + 0x10) + 4) & 0x20) != 0;
        std::vector<double> model_times;
        u32 model_made = 0;
        for (int k = 0; k < sub_steps; ++k) {
            const double t = static_cast<double>(t0) + k * h;
            const double tol = (t > 1.0 ? t : 1.0) * 1e-7;
            if (t + tol < next) continue;
            ++model_made;
            if (c.children) {
                g_events.push_back({3, at ? id ^ 0x80000000u : id, true});
            } else {
                model_times.push_back(t);
                g_events.push_back({1, 0, true});
                const s32 n = static_cast<s32>(g_answers() % 4);
                if (n > 0) g_events.push_back({2, static_cast<u32>(n), true});
            }
            next = t + static_cast<double>(c.interval);
        }
        const std::vector<Event> game = g_events;
        const u32 game_made = model_made;
        // The same events in order - their kinds and counts, a count's time the model's within the whole steps'
        // roundings (ours lie on the node's whole steps' float times) - and the same count made.
        auto as_f = [](u32 u) { float f; std::memcpy(&f, &u, 4); return f; };
        bool same = game.size() == ours.size();
        std::size_t ti = 0;
        for (std::size_t i = 0; same && i < game.size(); ++i) {
            if (game[i].what != ours[i].what) same = false;
            else if (game[i].what == 1) {
                const double want = model_times[ti++], got = static_cast<double>(as_f(ours[i].a));
                same = std::fabs(got - want) <= (want > 1.0 ? want : 1.0) * 2e-5;
            } else same = game[i].a == ours[i].a;
        }
        (void)ours_next;
        const char* why = nullptr;
        if (!same) why = "the events";
        else if (game_made != ours_made) why = "the count made";
        else if (c.children)
            for (const Event& e : ours)
                if (!e.tick) why = "children made between whole steps";
        if (why && ++bad <= 6 && !g_quiet) {
            std::printf("clock case %d (%s, state %d, interval %g, scale %g, at %g fps, %d frames, %d whole steps): %s differ - the game's %zu, "
                        "ours %zu",
                        k, c.children ? "children" : "CPU particles", c.staged0 ? 0 : 1, c.interval, c.scale, fps, frames, ticks, why, game.size(),
                        ours.size());
            std::size_t i = 0;
            while (i < game.size() && i < ours.size() && game[i] == ours[i]) ++i;
            if (i < game.size() || i < ours.size())
                std::printf(" (first apart at %zu: the game's %d/%08x, ours %d/%08x)", i, i < game.size() ? game[i].what : -1,
                            i < game.size() ? game[i].a : 0u, i < ours.size() ? ours[i].what : -1, i < ours.size() ? ours[i].a : 0u);
            std::printf("\n");
        }
    }
    return bad;
}

}  // namespace

int main() {
    eboot_kit::load("decomp_sfx_emission_clock");
    eboot_kit::stub(kContainerUpdate, reinterpret_cast<const void*>(&stub_container));
    eboot_kit::stub(kStateOf, reinterpret_cast<const void*>(&stub_state_of));
    eboot_kit::stub(kStateHolds, reinterpret_cast<const void*>(&stub_state_holds));
    eboot_kit::stub(kPick, reinterpret_cast<const void*>(&stub_pick));
    eboot_kit::stub(kApply, reinterpret_cast<const void*>(&stub_apply));
    eboot_kit::stub(kBind, reinterpret_cast<const void*>(&stub_bind));
    eboot_kit::stub(kSetUp, reinterpret_cast<const void*>(&stub_set_up));
    eboot_kit::stub(kLinger, reinterpret_cast<const void*>(&stub_linger));
    eboot_kit::stub(kEndModules, reinterpret_cast<const void*>(&stub_end_modules));
    eboot_kit::stub(kModulesDone, reinterpret_cast<const void*>(&stub_modules_done));
    eboot_kit::stub(kChildrenGone, reinterpret_cast<const void*>(&stub_children_gone));
    eboot_kit::stub(kItemData, reinterpret_cast<const void*>(&stub_item_data));
    eboot_kit::stub(kOneItem, reinterpret_cast<const void*>(&stub_one_item));
    eboot_kit::stub(kThreeItems, reinterpret_cast<const void*>(&stub_three_items));
    eboot_kit::stub(kAdd, reinterpret_cast<const void*>(&stub_add));
    eboot_kit::stub(kKids, reinterpret_cast<const void*>(&stub_kids));
    eboot_kit::stub(kKidsAt, reinterpret_cast<const void*>(&stub_kids_at));
    sfx_node::set_emission_clock(2);

    constexpr int kCases = 6000;
    int bad = 0;
    for (const double fps : {30.0, 60.0, 120.0, 144.0, 0.0}) bad += check(kCases, fps);
    // Each of the clock's mutants (node_update.cpp 20-23) must make some case differ.
    int missed = 0;
    g_quiet = true;
    for (int m = 20; m <= 24; ++m) {
        g_sfx_node_mutant = m;
        const int found = check(3000, 60.0) + check(3000, 120.0) + check(3000, 0.0);
        if (!found) {
            std::printf("decomp_sfx_emission_clock: mutant %d not caught\n", m);
            ++missed;
        }
    }
    g_sfx_node_mutant = 0;
    if (bad || missed) {
        std::printf("decomp_sfx_emission_clock: %d of %d cases differ, %d mutants missed\n", bad, 5 * kCases, missed);
        return 1;
    }
    std::printf("decomp_sfx_emission_clock: %d emitting nodes at 30, 60, 120, 144 and jittered frame times make the events of 60 updates a "
                "second (children on whole steps' frames); 5 mutants caught\n",
                5 * kCases);
    return 0;
}
