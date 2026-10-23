#pragma once
// Which GX buffers' memory the import thread (bb-import) imports as GX makes
// them, with window imports (BBHOST_IMPORT_WINDOWS=1): what the GPU reads
// where it lies, so its first draw finds the window imported rather than
// waiting for the driver to pin it (gpu.cpp, host_gpu_import_prefetch). Kept
// apart from Vulkan so a test can drive it.
#include <cstdint>

namespace import_prefetch {

// D3D11's usage and bind flags as GX keeps them in a buffer's holder (+0x39, +0x3b).
constexpr std::uint8_t kUsageDefault = 0, kUsageImmutable = 1, kUsageDynamic = 2, kUsageStaging = 3;
constexpr std::uint8_t kBindVertex = 0x1, kBindIndex = 0x2, kBindUnordered = 0x80;

enum class Ask {
    kNone,     // nothing ahead: no import needed, or the first binding makes it
    kInPlace,  // read in place whatever the buffer shadow does: imported now
    kMeshes,   // vertex or index data the buffer shadow reads in place when GX tells its writes
               // (BBHOST_SHADOW_IN_PLACE): imported now if the shadow does (asked again on the thread)
};

// `shadow_on`: the buffer shadow is on (its mirrors serve meshes; without it
// every mesh is read in place). `twins_level`: BBHOST_GX_TWINS. `meshes_on`:
// BBHOST_IMPORT_PREFETCH_MESHES (on unless 0).
inline Ask ask(std::uint8_t usage, std::uint8_t bind, bool shadow_on, int twins_level, bool meshes_on) {
    if (usage == kUsageStaging) return Ask::kNone;
    const bool uav = (bind & kBindUnordered) != 0, dynamic = usage == kUsageDynamic;
    const bool mesh = (bind & (kBindVertex | kBindIndex)) != 0;
    // What the GPU reads in place: an unordered-access buffer (what compute
    // writes, particles), a dynamic one (rings the game rewrites, bound from
    // host memory), a vertex or index buffer without the buffer shadow.
    if (uav || dynamic || (mesh && !shadow_on)) {
        // One the GPU writes, kept as a device-local twin: its twin is filled
        // through staging at its first use. With level 2 a dynamic one too:
        // the draws read its renamed copies.
        if ((twins_level == 1 || twins_level == 2) && usage == kUsageDefault && uav) return Ask::kNone;
        if (twins_level == 2 && dynamic) return Ask::kNone;
        return Ask::kInPlace;
    }
    // A default or immutable mesh with the shadow on: its mirror copies it
    // through staging (no import), or - its pages told - it is read in place.
    if (mesh && meshes_on && (usage == kUsageDefault || usage == kUsageImmutable)) return Ask::kMeshes;
    return Ask::kNone;
}

}  // namespace import_prefetch
