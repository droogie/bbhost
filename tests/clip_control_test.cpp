// PA_CL_CLIP_CNTL as Vulkan pipeline state (host/clip_control.h): depth clamp
// where z is not clipped - CLIP_DISABLE (16), ZCLIP_NEAR_DISABLE (26) or
// ZCLIP_FAR_DISABLE (27) - and rasterizer discard for DX_RASTERIZATION_KILL
// (22), whatever the register's other fields hold; the old reading (depth
// clamp from bit 16 alone, nothing discarded) with the switch off. Every
// combination of the four bits under several values of the others, GX's own
// encodings among them, then four mutants, each of which a check here must
// catch.
#include "host/clip_control.h"

#include <cstdint>
#include <cstdio>
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

using Rule = gpu::ClipControlState (*)(std::uint32_t, bool);

// The register's other fields, none of which the rule reads: user clip planes
// (0-5), PS_UCP_Y_SCALE_NEG and PS_UCP_MODE (13-15), UCP_CULL_ONLY_ENA (17),
// BOUNDARY_EDGE_FLAG_ENA (18), DX_CLIP_SPACE_DEF (19, which GX sets on every
// rasterizer state), DIS_CLIP_ERR_DETECT (20), VTX_KILL_OR (21),
// DX_LINEAR_ATTR_CLIP_ENA (24), VTE_VPORT_PROVOKE_DISABLE (25).
const std::uint32_t kOthers[] = {0u,           1u << 19,          0x3fu,          (1u << 19) | (1u << 24), (1u << 21) | (1u << 20),
                                 0x0003e000u, (1u << 25) | 0x7u, 0x0f7fe03fu & ~0x0c410000u};

// The expected state: z not clipped -> clamp; the kill bit -> discard.
gpu::ClipControlState expected(std::uint32_t c, bool decode) {
    gpu::ClipControlState e;
    if (!decode) {
        e.depth_clamp = (c >> 16) & 1;
        return e;
    }
    const bool clip_off = (c >> 16) & 1, near_off = (c >> 26) & 1, far_off = (c >> 27) & 1;
    e.depth_clamp = clip_off || near_off || far_off;
    e.rasterizer_discard = (c >> 22) & 1;
    return e;
}

int run_checks(Rule rule, bool report) {
    int wrong = 0;
    for (std::uint32_t bits = 0; bits < 16; ++bits) {
        const std::uint32_t fields = ((bits & 1) ? gpu::kClipDisable : 0u) | ((bits & 2) ? gpu::kDxRasterizationKill : 0u) |
                                     ((bits & 4) ? gpu::kZclipNearDisable : 0u) | ((bits & 8) ? gpu::kZclipFarDisable : 0u);
        for (const std::uint32_t other : kOthers) {
            const std::uint32_t c = fields | other;
            for (int decode = 0; decode < 2; ++decode) {
                const gpu::ClipControlState got = rule(c, decode != 0);
                const gpu::ClipControlState want = expected(c, decode != 0);
                if (got.depth_clamp != want.depth_clamp || got.rasterizer_discard != want.rasterizer_discard) {
                    ++wrong;
                    if (report && wrong <= 6) {
                        std::fprintf(stderr, "  wrong: clip_cntl %08x decode %d -> clamp %d discard %d, want %d %d\n", c, decode, got.depth_clamp,
                                     got.rasterizer_discard, want.depth_clamp, want.rasterizer_discard);
                    }
                }
            }
        }
    }
    return wrong;
}

// Mutants: bit 16 read as ZCLIP_NEAR_DISABLE and the z bits ignored (the
// reading before the decode), the kill bit ignored, clamp only when both z
// planes are unclipped, and the switch ignored.
gpu::ClipControlState mutant_bit16_near(std::uint32_t c, bool decode) {
    gpu::ClipControlState s = gpu::clip_control_state(c, decode);
    if (decode) s.depth_clamp = (c & gpu::kClipDisable) != 0;
    return s;
}
gpu::ClipControlState mutant_kill_ignored(std::uint32_t c, bool decode) {
    gpu::ClipControlState s = gpu::clip_control_state(c, decode);
    s.rasterizer_discard = false;
    return s;
}
gpu::ClipControlState mutant_both_planes(std::uint32_t c, bool decode) {
    gpu::ClipControlState s = gpu::clip_control_state(c, decode);
    if (decode) {
        s.depth_clamp = (c & gpu::kClipDisable) || ((c & gpu::kZclipNearDisable) && (c & gpu::kZclipFarDisable));
    }
    return s;
}
gpu::ClipControlState mutant_switch_ignored(std::uint32_t c, bool) { return gpu::clip_control_state(c, true); }

}  // namespace

int main() {
    const int wrong = run_checks(&gpu::clip_control_state, true);
    CHECK(wrong == 0);
    std::printf("clip_control_test: %zu register values, both switch settings, %d wrong\n", 16 * (sizeof(kOthers) / sizeof(kOthers[0])), wrong);

    // GX's rasterizer builder (0x2578550) writes DX_CLIP_SPACE_DEF and, from
    // D3D11's DepthClipEnable, both z bits or neither: its depth clamp is the
    // GX description's !depth_clip either way.
    const std::uint32_t gx_clip = 1u << 19, gx_noclip = (1u << 19) | gpu::kZclipNearDisable | gpu::kZclipFarDisable;
    CHECK(!gpu::clip_control_state(gx_clip, true).depth_clamp);
    CHECK(gpu::clip_control_state(gx_noclip, true).depth_clamp);
    CHECK(!gpu::clip_control_state(gx_noclip, true).rasterizer_discard);
    // The old reading missed the GX encoding of DepthClipEnable = FALSE.
    CHECK(!gpu::clip_control_state(gx_noclip, false).depth_clamp);

    struct Mutant {
        const char* name;
        Rule rule;
    } mutants[] = {{"bit 16 read as ZCLIP_NEAR_DISABLE, z bits ignored", &mutant_bit16_near},
                   {"DX_RASTERIZATION_KILL ignored", &mutant_kill_ignored},
                   {"clamp only with both z planes unclipped", &mutant_both_planes},
                   {"switch ignored", &mutant_switch_ignored}};
    for (const Mutant& m : mutants) {
        const int caught = run_checks(m.rule, false);
        std::printf("clip_control_test: mutant '%s': %d values wrong%s\n", m.name, caught, caught ? " (caught)" : " - NOT CAUGHT");
        CHECK(caught > 0);
    }
    if (g_failures) {
        std::fprintf(stderr, "clip_control_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("clip_control_test: ok\n");
    return 0;
}
