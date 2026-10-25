// Whether a draw that binds a geometry shader is drawn with it (BBHOST_GX_GS).
//
// v0.2.19 draws the GPU particles - an export shader, a geometry shader with
// its copy shader, a pixel shader - as a pipeline of their own: the export
// shader the vertex stage, the geometry shader the geometry stage
// (render.cpp gs_draw, gcn/lift.h GsLift). Before it, such a draw went as
// GX's vertex and pixel shaders and drew nothing (the export shader writes no
// position).
//
// On an RX 580 (GCN 4) under AMD's Windows driver, the Hunter's Dream's
// particles came out as long thin streaks fanning across the whole screen.
// The same pipeline draws right on NVIDIA, on RDNA under AMD's driver (a
// Radeon 680M, an 8060S), under RADV on the Steam Deck and on lavapipe, and it
// passes the validation layer. So the path is off by default where GCN meets
// AMD's own driver, and those draws go as before:
// - GCN is the device whose one subgroup size is 64 (GCN has no wave32;
//   RDNA offers 32 and 64);
// - AMD's own driver is the proprietary one or AMDVLK.
// RADV compiles with its own compiler (ACO) and keeps the path.
// BBHOST_GX_GS=1 draws them with their geometry shader on any device, and
// =0 on none.
#pragma once

#include <vulkan/vulkan_core.h>

#include <cstdint>

namespace gpu {

// What gs_path_for sees of the device.
struct GsDevice {
    std::uint32_t vendor_id = 0;  // VkPhysicalDeviceProperties::vendorID
    VkDriverId driver_id = static_cast<VkDriverId>(0);
    // VkPhysicalDeviceSubgroupSizeControlProperties (0 where the driver gives none)
    std::uint32_t subgroup_min = 0, subgroup_max = 0;
};

enum class GsPath : std::uint8_t {
    kOn,         // the default: drawn with the geometry shader
    kOffGcnAmd,  // off by default: GCN under AMD's own driver
    kForcedOn,   // BBHOST_GX_GS=1
    kForcedOff,  // BBHOST_GX_GS=0
};

// `env`: BBHOST_GX_GS's value, or null when it is not set.
inline GsPath gs_path_for(const GsDevice& d, const char* env) {
    if (env && env[0] == '0') return GsPath::kForcedOff;
    if (env && env[0] == '1') return GsPath::kForcedOn;
    const bool amd_driver = d.driver_id == VK_DRIVER_ID_AMD_PROPRIETARY || d.driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE;
    if (amd_driver && d.subgroup_min == 64 && d.subgroup_max == 64) return GsPath::kOffGcnAmd;
    return GsPath::kOn;
}

inline bool gs_path_on(GsPath p) { return p == GsPath::kOn || p == GsPath::kForcedOn; }

}  // namespace gpu
