// Which draws start a command buffer of their own on lavapipe.
//
// Lavapipe (Mesa's llvmpipe Vulkan driver, 26.1 at least) starts every command
// buffer with independent blending, and loses it for the rest of the command
// buffer once a pipeline without colour attachments - a depth-only one - has
// been bound: from then on every colour attachment of a draw gets attachment
// 0's blend state - blend enable, equations and write mask. An attachment the
// draw writes with a state of its own is drawn with attachment 0's; one the
// pixel shader does not export (write mask 0) is written whenever attachment 0
// blends, with an undefined value blended in - the G-buffer's puddle decals
// export six of the seven targets bound, blend into them, and left NaN in the
// seventh. A new command buffer starts right again; a new rendering instance
// or binding everything again does not.
//
// So on lavapipe a draw that a depth-only pipeline precedes in the same command
// buffer, and whose attachments' states differ where that collapse would change
// what it draws, starts a command buffer of its own (render.cpp). Every other
// driver records as before. BBHOST_LVP_BLEND_SPLIT=0 records such draws like
// any other.
#pragma once

#include <cstdint>

namespace gpu {

// One colour slot of a draw as the collapse sees it: `bound`, a target is bound
// there (a gap below the last bound slot is an attachment without one); `mask`,
// the draw's write mask for it (CB_TARGET_MASK cut to the pixel shader's
// exports, 0 where the shader does not write it); `blend`, blending on (and the
// format blends); `eq`, the blend factors and operations packed, 0 where
// blending is off.
struct LvpSlot {
    bool bound = false;
    std::uint32_t mask = 0;
    bool blend = false;
    std::uint32_t eq = 0;
};

// Whether drawing every slot with slot 0's state changes what the draw writes:
// a slot it writes whose state differs from slot 0's, or a slot it does not
// write while slot 0 blends a channel in. `count`: the slots up to the last
// bound one.
inline bool lvp_blend_collapse_changes(const LvpSlot* s, int count) {
    if (count <= 1) return false;
    // Slot 0 as the collapse applies it everywhere; a gap there has no state at all.
    const std::uint32_t mask0 = s[0].bound ? (s[0].mask & 0xfu) : 0u;
    const bool blend0 = s[0].bound && s[0].blend;
    const std::uint32_t eq0 = blend0 ? s[0].eq : 0u;
    for (int t = 1; t < count; ++t) {
        if (!s[t].bound) continue;
        const std::uint32_t mask = s[t].mask & 0xfu;
        if (mask) {
            const bool blend = s[t].blend;
            if (mask != mask0 || blend != blend0 || (blend ? s[t].eq : 0u) != eq0) return true;
        } else if (blend0 && mask0) {
            return true;  // not exported, but written with slot 0's blending
        }
    }
    return false;
}

}  // namespace gpu
