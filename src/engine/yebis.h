#pragma once

// YEBIS's post-processing switches, which Bloodborne already has.
//
// Bloodborne's post-processing is YEBIS, and the eboot carries YEBIS's own
// debug parameter names - "Yebis-Enable MotionBlur-Dof", "LensDistortion",
// "Gaussian", "LightShaft", "ColorGrading", "Yebis-DepthOfFieldQuality" and
// about 180 more. A From debug node is registered *against the field it
// edits*, so sub_25fcf20 (the YEBIS parameter builder) hands over the offsets.
//
// So the three post-effects DS3's CSGraphicsConfig exposes are not settings
// the port has to invent - they are flags the game already keeps, and turning
// one off is writing a 0 where the game wrote a 1.

#include <cstdint>

struct ElfImage;

void yebis_install(ElfImage* image);

// The settings object, or null if the path could not be resolved and checked.
// Resolved once and cached; safe to call before the game has built it.
void* yebis_settings();

// Field offsets in the **debug** block. The object holds the set twice - the
// constructor writes one at +0x68 and then copies it to +0xd0 - so a write
// goes to both, `off` and `off - 0x68`, and the port does not have to know
// which one the renderer reads.
constexpr std::uint32_t kYebisMotionBlurDof = 0xe0;   // and depth of field: one pass, one flag
// The two qualities in sub_25fcf20's order: "GlareQuality" (0..7), then
// "DepthOfFieldQuality" (0..12) - the one YEBIS picks its depth-of-field
// preset by when the context is made, and with it the edge qualities
// CPostEffect::SetDepthOfFieldEdgeQuality accepts.
constexpr std::uint32_t kYebisGlareQuality = 0x110;   // 0..7
constexpr std::uint32_t kYebisDofQuality = 0x114;     // 0..12
constexpr std::uint32_t kYebisLightShaft = 0xf8;
constexpr std::uint32_t kYebisColorGrading = 0x108;
constexpr std::uint32_t kYebisLensDistortion = 0xd4;

bool yebis_get(std::uint32_t off, std::int32_t* out);
bool yebis_set(std::uint32_t off, std::int32_t value);

// Re-asserts whatever the port has been told to hold, every poll. The game
// writes these fields itself, so a setting applied once is a setting that
// lasts until the next time the game touches it.
//
// BBHOST_YEBIS_SET=0xe0=0,0x114=0 forces raw offsets, which is how the path
// from "write a 0" to "the picture changes" was measured.
void yebis_poll();
