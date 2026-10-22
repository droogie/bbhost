// Which draws lavapipe's collapse of every colour slot to slot 0's blend state
// changes (host/lvp_split.h): every combination of bound, write mask, blend and
// equation over three slots against the rule written out, the game's draws by
// name, then four mutants, each of which a check here must catch.
#include "host/lvp_split.h"

#include <cstdint>
#include <cstdio>

namespace {

int g_failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

using gpu::LvpSlot;
using Rule = bool (*)(const LvpSlot*, int);

// The collapse simulated: each slot drawn with slot 0's state (a gap at slot 0
// has none: nothing written, nothing blended), against the slot's own. A slot
// the draw writes is drawn differently when mask, blend or equation differ; a
// slot it does not write is written only if the applied state blends a channel.
bool expected(const LvpSlot* s, int count) {
    const std::uint32_t m0 = s[0].bound ? s[0].mask : 0u;
    const bool b0 = s[0].bound && s[0].blend;
    for (int t = 1; t < count; ++t) {
        if (!s[t].bound) continue;
        const bool written = s[t].mask != 0;
        if (written) {
            const bool same = s[t].mask == m0 && s[t].blend == b0 && (!b0 || s[t].eq == s[0].eq);
            if (!same) return true;
        } else if (b0 && m0 != 0) {
            return true;
        }
    }
    return false;
}

const std::uint32_t kMasks[3] = {0x0, 0x7, 0xf};
const std::uint32_t kEqs[2] = {0x0f05, 0x0104};

// One slot from a code 0-12: 12 unbound, else mask (code % 3), blend (code / 3 % 2), equation (code / 6).
LvpSlot slot_of(int code) {
    LvpSlot s;
    if (code == 12) return s;
    s.bound = true;
    s.mask = kMasks[code % 3];
    s.blend = (code / 3) % 2 != 0;
    s.eq = s.blend ? kEqs[code / 6] : 0u;
    return s;
}

int run_checks(Rule rule, bool report) {
    int wrong = 0, n = 0;
    for (int a = 0; a < 13; ++a) {
        for (int b = 0; b < 13; ++b) {
            for (int c = 0; c < 13; ++c) {
                const LvpSlot s[3] = {slot_of(a), slot_of(b), slot_of(c)};
                const int count = c != 12 ? 3 : b != 12 ? 2 : 1;
                ++n;
                const bool got = rule(s, count), want = expected(s, count);
                if (got != want) {
                    ++wrong;
                    if (report && wrong <= 6) std::fprintf(stderr, "  wrong: slots %d %d %d -> %d, want %d\n", a, b, c, got, want);
                }
            }
        }
    }
    if (report) std::printf("lvp_split_test: %d combinations, %d wrong\n", n, wrong);
    return wrong;
}

// Mutants: a slot the draw does not write counted whatever slot 0 does, the
// equations ignored, the write masks ignored, a gap at slot 0 read as bound.
bool mutant_unwritten_always(const LvpSlot* s, int count) {
    for (int t = 1; t < count; ++t) {
        if (s[t].bound && !s[t].mask) return true;
    }
    return gpu::lvp_blend_collapse_changes(s, count);
}
bool mutant_no_eq(const LvpSlot* s, int count) {
    LvpSlot c[8];
    for (int t = 0; t < count; ++t) {
        c[t] = s[t];
        c[t].eq = 0;
    }
    return gpu::lvp_blend_collapse_changes(c, count);
}
bool mutant_no_mask(const LvpSlot* s, int count) {
    LvpSlot c[8];
    for (int t = 0; t < count; ++t) {
        c[t] = s[t];
        if (c[t].mask) c[t].mask = 0xf;
    }
    return gpu::lvp_blend_collapse_changes(c, count);
}
bool mutant_gap_bound(const LvpSlot* s, int count) {
    LvpSlot c[8];
    for (int t = 0; t < count; ++t) c[t] = s[t];
    if (!c[0].bound) {
        c[0].bound = true;
        c[0].mask = 0xf;  // what a stale state would hold
    }
    return gpu::lvp_blend_collapse_changes(c, count);
}

LvpSlot bound(std::uint32_t mask, bool blend, std::uint32_t eq) {
    LvpSlot s;
    s.bound = true;
    s.mask = mask;
    s.blend = blend;
    s.eq = blend ? eq : 0u;
    return s;
}

}  // namespace

int main() {
    CHECK(run_checks(&gpu::lvp_blend_collapse_changes, true) == 0);

    // The G-buffer's puddle decals: six slots written (RGB) and blended alike, the seventh not exported.
    LvpSlot decal[7];
    for (int t = 0; t < 6; ++t) decal[t] = bound(0x7, true, kEqs[0]);
    decal[6] = bound(0x0, false, 0);
    CHECK(gpu::lvp_blend_collapse_changes(decal, 7));
    // The G-buffer's opaque geometry: six slots written alike, not blended, the seventh not exported.
    LvpSlot geometry[7];
    for (int t = 0; t < 6; ++t) geometry[t] = bound(0xf, false, 0);
    geometry[6] = bound(0x0, false, 0);
    CHECK(!gpu::lvp_blend_collapse_changes(geometry, 7));
    // The effects' soft sprites: one slot written and blended, the second not exported.
    const LvpSlot sprite[2] = {bound(0xf, true, kEqs[1]), bound(0x0, false, 0)};
    CHECK(gpu::lvp_blend_collapse_changes(sprite, 2));
    // Two slots written and blended alike: drawn the same either way.
    const LvpSlot alike[2] = {bound(0xf, true, kEqs[0]), bound(0xf, true, kEqs[0])};
    CHECK(!gpu::lvp_blend_collapse_changes(alike, 2));
    // A gap at slot 0 (Cainhurst's ground has none) and slot 1 written: nothing would be.
    const LvpSlot gap0[2] = {LvpSlot{}, bound(0xf, false, 0)};
    CHECK(gpu::lvp_blend_collapse_changes(gap0, 2));
    // One slot: nothing to collapse.
    const LvpSlot one[1] = {bound(0x7, true, kEqs[0])};
    CHECK(!gpu::lvp_blend_collapse_changes(one, 1));

    struct Mutant {
        const char* name;
        Rule rule;
    } mutants[] = {{"an unwritten slot counted whatever slot 0 does", &mutant_unwritten_always},
                   {"equations ignored", &mutant_no_eq},
                   {"write masks ignored", &mutant_no_mask},
                   {"a gap at slot 0 read as bound", &mutant_gap_bound}};
    for (const Mutant& m : mutants) {
        const int caught = run_checks(m.rule, false);
        std::printf("lvp_split_test: mutant '%s': %d combinations wrong%s\n", m.name, caught, caught ? " (caught)" : " - NOT CAUGHT");
        CHECK(caught > 0);
    }
    if (g_failures) {
        std::fprintf(stderr, "lvp_split_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("lvp_split_test: ok\n");
    return 0;
}
