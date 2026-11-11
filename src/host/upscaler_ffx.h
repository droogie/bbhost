#pragma once
// AMD FSR 3 on Linux: bbhost's build of AMD's FidelityFX library
// (src/ffx_vk, libbbhost_fidelityfx_vk.so beside the executable) behind the
// upscaler bridge's C API (third_party/upscaler_bridge/upscaler_bridge.h), in
// bbhost itself rather than a bridge library: on Windows the bridge DLL calls
// AMD's signed FidelityFX DLLs, here host/upscaler.cpp calls this table for
// FSR and the NGX bridge, where there is one, for DLSS. The library is MIT
// (AMD's source with bbhost's portability files), loaded at run time like the
// Windows DLLs, so a build or a package without it simply has no FSR.
//
// The table follows the bridge's rules and Numity1504's Windows bridge's FSR
// path, so FSR 3 gets what it gets on Windows: a context per render size,
// output size and depth direction (HDR colour with FSR's own auto exposure,
// no reactive or transparency mask, FSR's own sharpening off - Sharpness is
// bbhost's after any upscaler), the old one freed frames_in_flight + 2
// evaluates later; the inputs read-only and the output GENERAL for the
// dispatch, put back after; the jitter, motion scale, frame time, near, far
// and field of view as the host gives them; the render sizes, jitter phases
// and LOD bias by the same formulas as the Windows bridge. It adds nothing to
// the instance or the device itself: host/gpu.cpp turns on compute shaders'
// derivative groups for FSR 3's two pyramid passes (host/upscaler.h
// upscaler_wants_compute_derivatives).
//
// Only FSR (UB_METHOD_FSR) is here; FSR 4 and frame generation are AMD's
// Windows libraries' alone.

#include "upscaler_bridge.h"

#include <cstdint>

// The table, or nullptr where there is none (off Linux).
const ub_api* upscaler_ffx_api();
// The library's file name, beside the executable.
constexpr const char* kFfxLibrary = "libbbhost_fidelityfx_vk.so";

namespace upscaler_ffx {
// The render size of an output size at a quality (UB_QUALITY_*): the Windows
// bridge's FSR ratios 1, 1.5, 1.7 and 2, each way, cut down (at least 1).
void render_size(std::uint32_t quality, std::uint32_t output_w, std::uint32_t output_h, std::uint32_t* w, std::uint32_t* h);
// The jitter sequence's length for a render and an output width: 8 x (output /
// render) squared, rounded up, at least 8.
std::uint32_t jitter_phases(std::uint32_t render_w, std::uint32_t output_w);
// A frame's jitter: Halton (2, 3) at frame % phases + 1, centred.
void jitter(std::uint32_t frame, std::uint32_t phases, float* x, float* y);
}  // namespace upscaler_ffx
