// The blend state of a colour target a draw writes no channel of (host/masked_blend.h):
// zeroed when the write mask is empty, the game's own otherwise - every write
// mask, blend enable, factor and op the GCN registers can give - and the game's
// in every case with the switch off. Then three mutants of the rule, each of
// which a check here must catch.
#include "host/masked_blend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

bool same(const VkPipelineColorBlendAttachmentState& a, const VkPipelineColorBlendAttachmentState& b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

// Game blend states as fixed_state makes them: every write mask, blending on
// and off, the factors and ops a CB_BLEND*_CONTROL names (0x60010504, the puddle
// decals' alpha blend, among them).
std::vector<VkPipelineColorBlendAttachmentState> game_states() {
    std::vector<VkPipelineColorBlendAttachmentState> out;
    const VkBlendFactor factors[] = {VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                     VK_BLEND_FACTOR_DST_COLOR};
    const VkBlendOp ops[] = {VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT, VK_BLEND_OP_MAX};
    for (std::uint32_t mask = 0; mask < 16; ++mask) {
        for (int enable = 0; enable < 2; ++enable) {
            for (const VkBlendFactor src : factors) {
                for (const VkBlendFactor dst : factors) {
                    for (const VkBlendOp op : ops) {
                        VkPipelineColorBlendAttachmentState b{};
                        b.blendEnable = enable ? VK_TRUE : VK_FALSE;
                        b.srcColorBlendFactor = src;
                        b.dstColorBlendFactor = dst;
                        b.colorBlendOp = op;
                        b.srcAlphaBlendFactor = dst;  // the alpha half differs from the colour half
                        b.dstAlphaBlendFactor = src;
                        b.alphaBlendOp = op;
                        b.colorWriteMask = mask;
                        out.push_back(b);
                    }
                }
            }
        }
    }
    return out;
}

using Rule = VkPipelineColorBlendAttachmentState (*)(const VkPipelineColorBlendAttachmentState&, bool);

// The checks, against a rule: the number of states it got wrong.
int run_checks(Rule rule, bool report) {
    int wrong = 0;
    const VkPipelineColorBlendAttachmentState zero{};
    for (const VkPipelineColorBlendAttachmentState& game : game_states()) {
        const VkPipelineColorBlendAttachmentState on = rule(game, true);
        const VkPipelineColorBlendAttachmentState off = rule(game, false);
        bool ok = true;
        if ((game.colorWriteMask & 0xf) == 0) {
            // Nothing written: blending off, the whole state zeroed (one key for all of them), mask still empty.
            ok = ok && same(on, zero) && on.blendEnable == VK_FALSE && on.colorWriteMask == 0;
        } else {
            // Something written: the game's state to the bit - its blend, factors, ops and mask.
            ok = ok && same(on, game);
        }
        ok = ok && same(off, game);  // the switch off: the game's state whatever its mask
        if (!ok) {
            ++wrong;
            if (report && wrong <= 4) {
                std::fprintf(stderr, "  wrong: mask %x enable %d src %d dst %d op %d -> on: enable %d mask %x, off: enable %d mask %x\n",
                             game.colorWriteMask, game.blendEnable, game.srcColorBlendFactor, game.dstColorBlendFactor, game.colorBlendOp,
                             on.blendEnable, on.colorWriteMask, off.blendEnable, off.colorWriteMask);
            }
        }
    }
    return wrong;
}

// Mutants: a rule that zeroes the blend of a target whose mask is not empty (an
// alpha-only one), one that turns blending off but keeps the dead factors
// (equivalent states keep separate pipelines), and one that ignores the switch.
VkPipelineColorBlendAttachmentState mutant_any_mask(const VkPipelineColorBlendAttachmentState& game, bool on) {
    if (on && (game.colorWriteMask & 0x7) == 0) return VkPipelineColorBlendAttachmentState{};  // alpha-only writes zeroed too
    return game;
}
VkPipelineColorBlendAttachmentState mutant_enable_only(const VkPipelineColorBlendAttachmentState& game, bool on) {
    VkPipelineColorBlendAttachmentState b = game;
    if (on && (game.colorWriteMask & 0xf) == 0) b.blendEnable = VK_FALSE;
    return b;
}
VkPipelineColorBlendAttachmentState mutant_always(const VkPipelineColorBlendAttachmentState& game, bool) {
    return gpu::masked_blend_state(game, true);
}

}  // namespace

int main() {
    const std::size_t states = game_states().size();
    const int wrong = run_checks(&gpu::masked_blend_state, true);
    CHECK(wrong == 0);
    std::printf("masked_blend_test: %zu game blend states, %d wrong\n", states, wrong);

    // The predicate on its own.
    VkPipelineColorBlendAttachmentState b{};
    b.blendEnable = VK_TRUE;
    b.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    b.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    CHECK(gpu::masked_blend_dead(b));
    b.colorWriteMask = VK_COLOR_COMPONENT_A_BIT;
    CHECK(!gpu::masked_blend_dead(b));
    b.colorWriteMask = 0x10;  // no channel among the four
    CHECK(gpu::masked_blend_dead(b));

    struct Mutant {
        const char* name;
        Rule rule;
    } mutants[] = {{"zeroes a target with a non-zero mask (alpha only)", &mutant_any_mask},
                   {"leaves the dead factors", &mutant_enable_only},
                   {"ignores the switch", &mutant_always}};
    for (const Mutant& m : mutants) {
        const int caught = run_checks(m.rule, false);
        std::printf("masked_blend_test: mutant '%s': %d states wrong%s\n", m.name, caught, caught ? " (caught)" : " - NOT CAUGHT");
        CHECK(caught > 0);
    }
    if (g_failures) {
        std::fprintf(stderr, "masked_blend_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("masked_blend_test: ok\n");
    return 0;
}
