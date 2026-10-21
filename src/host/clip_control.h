// PA_CL_CLIP_CNTL as Vulkan pipeline state, for a draw whose rasterizer
// state comes from the registers rather than a GX description.
//
// The bits read (GFX7; Gnm's setClipControl 0x1476b30 keeps the register's
// mask 0x0f7fe03f):
//   16 CLIP_DISABLE           the clipper passes primitives through: no
//                             frustum clipping in x, y or z.
//   22 DX_RASTERIZATION_KILL  nothing is rasterized; the vertex stage still runs.
//   26 ZCLIP_NEAR_DISABLE     no clipping against the near plane.
//   27 ZCLIP_FAR_DISABLE      no clipping against the far plane.
// Wherever z is not clipped the depth block still clamps depth to the
// viewport's range, so "z not clipped" is Vulkan's depthClampEnable exactly:
// no clipping against either z plane, depth clamped to minDepth..maxDepth.
// Vulkan has no switch per plane (VK_EXT_depth_clip_enable's depthClipEnable
// is one switch for both too), so either z bit gives depth clamp: exact when
// both are set - GX's rasterizer builder 0x2578550 sets both from D3D11's
// DepthClipEnable, never one alone - and, for one alone, the other plane's
// geometry is clamped onto that plane instead of clipped (a shadow caster
// behind the near plane flattened onto it, as the bit asks, rather than lost).
// CLIP_DISABLE turns z clipping off the same way. Its x and y half Vulkan
// cannot express: a primitive is still clipped to the frustum, which differs
// only for one crossing w = 0 - inside the viewport the same pixels are drawn.
// DX_RASTERIZATION_KILL is rasterizer discard.
//
// BBHOST_CLIP_CONTROL=0 keeps the old reading: depth clamp from bit 16 alone,
// bit 22 ignored.
#pragma once

#include <cstdint>

namespace gpu {

constexpr std::uint32_t kClipDisable = 1u << 16;
constexpr std::uint32_t kDxClipSpaceDef = 1u << 19;  // D3D's clip space (z from 0 to w); clear is GL's - only counted (render.cpp)
constexpr std::uint32_t kDxRasterizationKill = 1u << 22;
constexpr std::uint32_t kZclipNearDisable = 1u << 26;
constexpr std::uint32_t kZclipFarDisable = 1u << 27;

struct ClipControlState {
    bool depth_clamp = false;         // depthClampEnable
    bool rasterizer_discard = false;  // rasterizerDiscardEnable
};

// The pipeline state a PA_CL_CLIP_CNTL value asks for; `decode` false gives
// the old reading (BBHOST_CLIP_CONTROL=0).
inline ClipControlState clip_control_state(std::uint32_t clip_cntl, bool decode) {
    ClipControlState c;
    if (!decode) {
        c.depth_clamp = (clip_cntl & kClipDisable) != 0;
        return c;
    }
    c.depth_clamp = (clip_cntl & (kClipDisable | kZclipNearDisable | kZclipFarDisable)) != 0;
    c.rasterizer_discard = (clip_cntl & kDxRasterizationKill) != 0;
    return c;
}

}  // namespace gpu
