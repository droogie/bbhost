// The blend state of a colour attachment a draw writes no channel of.
//
// A draw's write mask for a target is the game's CB_TARGET_MASK cut to what the
// pixel shader exports (CB_SHADER_MASK), so a target the shader does not export
// has an empty mask - while the game's blend registers for it may still say
// "blend": the G-buffer's puddle decals export six of the seven targets bound
// and leave blending on for the seventh, the effects' soft sprites export one
// of two. The shader's output for such a target is undefined, and Vulkan writes
// nothing through an empty mask, so its blend state is dead on a driver that
// honours the mask. A driver may blend the undefined output into the target
// anyway; a target with an empty mask therefore carries no blend state at all
// (blending off, every factor and op 0), which is the same on a driver that
// honours the mask, and the equivalent states share one pipeline.
// BBHOST_MASKED_BLEND=0 keeps the game's state as it is.
#pragma once

#include <vulkan/vulkan_core.h>

namespace gpu {

inline bool masked_blend_dead(const VkPipelineColorBlendAttachmentState& b) { return (b.colorWriteMask & 0xf) == 0; }

// `game`, or a zeroed state (blending off, every factor and op 0) when the
// draw writes no channel of the target and `on`.
inline VkPipelineColorBlendAttachmentState masked_blend_state(const VkPipelineColorBlendAttachmentState& game, bool on) {
    if (on && masked_blend_dead(game)) return VkPipelineColorBlendAttachmentState{};
    return game;
}

}  // namespace gpu
