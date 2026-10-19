// YEBIS's post-processing parameters as source (docs/decomp.md): the PPFX
// entry points the flip pass (0x25d5f80) hands each frame's parameters to
// YEBIS's single post effect through - the scene's projections and the
// luminance scale, auto exposure and tone mapping, chromatic aberration, the
// colour matrix, the glare, the light shafts, depth of field and auto focus,
// motion blur, the feedback (afterimage) and the antialiasing: 69 entries,
// in the flip pass's order below (some it calls only in some states - auto
// focus on or off, an effect enabled). Each is ppfx.cpp's: the effect
// (s_pSinglePostEffect) and the arguments, handed to the CPostEffect method of
// the same parameters - five write their one field in place - so every value
// the game sets for a frame passes through here, where a hook can read it or
// change it (docs/modding.md). The methods stay the game's.
//
// Names: the method's, as its messages and asserts give it
// (CPostEffect::SetGlareParameters -> ppfx_set_glare_parameters); sub_<the
// entry's address> where the binary gives none; the fields the method writes,
// as offsets in the effect, say what each is. Parameter names likewise: the
// asserts' (fGlareRemapFactor, fExposure, eMatrixType, iEffectViewIndex ...)
// where there are any. A byte the entry zero-extends before the method reads
// its 32 bits is a u8 here, passed on as a u32 (as movzx does). The first
// argument of each is the PPFX context the flip pass passes (+0x2670 of the
// object it works on), which the entries leave unread: there is one post
// effect. Leaves: one call or one store each.
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <cstdint>

using namespace decomp;

#ifdef DECOMP_YEBIS_PPFX_SETTERS_MUTANTS
extern int g_ppfx_setters_mutant;
#define MUTANT(n) (g_ppfx_setters_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kSinglePostEffect = 0x5865ed0;  // s_pSinglePostEffect

u64 effect() { return load<u64>(game_address(kSinglePostEffect)); }

}  // namespace


// ---- The scene: the projections and view matrices the effects work from, and the luminance scale ----

// CPostEffect::SetRenderSceneLuminanceScale: the first three as sub_fb2720's, the fourth to +0x6ac, the scale (> 0) to +0x6b0.
DECOMP_LEAF u64 ppfx_set_render_scene_luminance_scale(u64 /*context*/, float near_z, float far_z, float fov, float value, float scene_luminance_scale) {
    return game_function<GUEST_ABI u64 (*)(u64, float, float, float, float, float)>(0xfd48d0)(effect(), near_z, far_z, fov, value, scene_luminance_scale);
}

// The view's projection: near (1 when not positive) to +0x57c, far (1000) to +0x580, from the field of view tan and atan to +0x588/+0x584.
DECOMP_LEAF u64 ppfx_sub_fb2720(u64 /*context*/, float near_z, float far_z, float fov) {
    return game_function<GUEST_ABI u64 (*)(u64, float, float, float)>(0xfd85d0)(effect(), near_z, far_z, fov);
}

// The same three for the second projection at +0x594..+0x5a4.
DECOMP_LEAF u64 ppfx_sub_fb2740(u64 /*context*/, float near_z, float far_z, float fov) {
    return game_function<GUEST_ABI u64 (*)(u64, float, float, float)>(0xfd4a40)(effect(), near_z, far_z, fov);
}

// A 4x4 matrix (a pointer to it) to +0x5ac - its z axis mirrored (0xfb8890) unless the flag - and what is made from it to +0x5ec.
DECOMP_LEAF u64 ppfx_sub_fb27d0(u64 /*context*/, u64 matrix, u8 flag) {
    return game_function<GUEST_ABI u64 (*)(u64, u64, u32)>(0xfd8cb0)(effect(), matrix, static_cast<u32>(flag));
}

// The same for the matrix at +0x62c.
DECOMP_LEAF u64 ppfx_sub_fb27f0(u64 /*context*/, u64 matrix, u8 flag) {
    return game_function<GUEST_ABI u64 (*)(u64, u64, u32)>(0xfd90f0)(effect(), matrix, static_cast<u32>(flag));
}

// A byte at +0x150 of the object at the effect's +0x128, when the effect's +0x140 is set (else 1 back).
DECOMP_LEAF u64 ppfx_sub_fb3fb0(u64 /*context*/, u8 enable) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xff90a0)(effect(), static_cast<u32>(enable));
}


// ---- Auto exposure (per view: -1 the current one) ----

// CPostEffect::SetAutoExposureEnable (view index, -1 the current one).
DECOMP_LEAF u64 ppfx_set_auto_exposure_enable(u64 /*context*/, u32 view, u8 enable) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32)>(0xfd4fc0)(effect(), view, MUTANT(4) ? static_cast<u32>(static_cast<s32>(static_cast<s8>(enable))) : static_cast<u32>(enable));
}

// CPostEffect::SetAutoExposureMiddleGray.
DECOMP_LEAF u64 ppfx_set_auto_exposure_middle_gray(u64 /*context*/, u32 view, float a, float b) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float)>(0xfd5550)(effect(), view, a, b);
}

// CPostEffect::SetAutoExposureRange.
DECOMP_LEAF u64 ppfx_set_auto_exposure_range(u64 /*context*/, u32 view, float low, float high) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float)>(0xfd53c0)(effect(), view, low, high);
}

// CPostEffect::SetAutoExposureMeteringArea (fOffsetX, fOffsetY, fWidth, fHeight).
DECOMP_LEAF u64 ppfx_set_auto_exposure_metering_area(u64 /*context*/, u32 view, float offset_x, float offset_y, float width, float height) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float, float, float)>(0xfd5870)(effect(), view, offset_x, offset_y, width, height);
}

// CPostEffect::SetAutoExposureDelay.
DECOMP_LEAF u64 ppfx_set_auto_exposure_delay(u64 /*context*/, u32 view, u32 mode, float delay) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32, float)>(0xfd56f0)(effect(), view, mode, delay);
}

// CPostEffect::SetAutoExposureAdjustment.
DECOMP_LEAF u64 ppfx_set_auto_exposure_adjustment(u64 /*context*/, u32 view, float a, float b, float c, float d, float e, float f, float g) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float, float, float, float, float, float)>(0xfd5130)(effect(), view, a, b, c, d, e, f, g);
}


// ---- Tone mapping ----

// CPostEffect::SetTonemapParameters: fExposure (> 0) to +0x700, fGamma (not below 0) to +0x704, the type and the third through 0xfde4b0, the fourth to +0x76c.
DECOMP_LEAF u64 ppfx_set_tonemap_parameters(u64 /*context*/, u32 type, float exposure, float gamma, float a, float b) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float, float, float)>(0xfd4d40)(effect(), type, exposure, gamma, a, b);
}

// A value clamped to [0, a constant] at +0x724.
DECOMP_LEAF u64 ppfx_sub_fb3560(u64 /*context*/, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, float)>(0xfd4e60)(effect(), value);
}


// ---- Chromatic aberration ----

// CPostEffect::SetChromaticAberrationEnable(enable, false).
DECOMP_LEAF u64 ppfx_set_chromatic_aberration_enable(u64 /*context*/, u8 enable) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32)>(0xfd3320)(effect(), static_cast<u32>(enable), MUTANT(2) ? 1u : 0u);
}

// The chromatic aberration's parameters: a to +0x9a4, false to +0x9a8, two 8-byte values from p and q to +0x9ac/+0x9b4.
DECOMP_LEAF u64 ppfx_sub_fb37d0(u64 /*context*/, u32 a, u64 p, u64 q) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32, u64, u64)>(0xfd33a0)(effect(), a, 0u, MUTANT(5) ? q : p, MUTANT(5) ? p : q);
}


// ---- The colour matrix (colour grading) ----

// CPostEffect::InitializeMatrix (eMatrixType).
DECOMP_LEAF u64 ppfx_initialize_matrix(u64 /*context*/, u32 matrix_type) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xfd6cb0)(effect(), matrix_type);
}

// CPostEffect::MultiplyMatrix, the variant that takes a flag.
DECOMP_LEAF u64 ppfx_multiply_matrix_fb4360(u64 /*context*/, u32 matrix_type, u8 flag) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32)>(0xff6700)(effect(), matrix_type, static_cast<u32>(flag));
}

// CPostEffect::MultiplyMatrix by a saturation (GPUMath_ColorMatrixSaturation).
DECOMP_LEAF u64 ppfx_multiply_matrix_fb4390(u64 /*context*/, u32 matrix_type, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float)>(0xff6be0)(effect(), matrix_type, value);
}

// CPostEffect::MultiplyMatrix, a third variant.
DECOMP_LEAF u64 ppfx_multiply_matrix_fb43b0(u64 /*context*/, u32 matrix_type, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float)>(0xff7290)(effect(), matrix_type, value);
}

// CPostEffect::MultiplyMatrix, a fourth variant.
DECOMP_LEAF u64 ppfx_multiply_matrix_fb43e0(u64 /*context*/, u32 matrix_type, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float)>(0xff6aa0)(effect(), matrix_type, value);
}

// Off when a is 0 (+0x73c), else on with a and b at +0x740/+0x744; the byte at +0x380 of the object at +0x128 cleared.
DECOMP_LEAF u64 ppfx_sub_fb3670(u64 /*context*/, float a, float b) {
    return game_function<GUEST_ABI u64 (*)(u64, float, float)>(0xfd4f70)(effect(), a, b);
}


// ---- The glare (bloom); the glare object is the effect's m_pRenderGlare, at +0x120 ----

// CPostEffect::SetGlareParameters: type to +0xbd4, a to +0x6bc, b to +0x6b8, fGlareRemapFactor (> 0) to +0xbd8.
DECOMP_LEAF u64 ppfx_set_glare_parameters(u64 /*context*/, u32 type, float a, float b, float remap_factor) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float, float)>(0xfd4b60)(effect(), type, MUTANT(1) ? b : a, MUTANT(1) ? a : b, remap_factor);
}

// Eighteen dwords from a pointer to +0xbe0 (and -1 to +0xbdc).
DECOMP_LEAF u64 ppfx_sub_fb2860(u64 /*context*/, u64 shape) {
    return game_function<GUEST_ABI u64 (*)(u64, u64)>(0xfd94d0)(effect(), shape);
}

// A dword at +0xbdc.
DECOMP_LEAF u64 ppfx_sub_fb2850(u64 /*context*/, u32 value) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xfd4c10)(effect(), value);
}

// CPostEffect::SetGlareBloomNumLevels: 0 or less, or over the glare object's maximum (+0xc4), is the maximum.
DECOMP_LEAF u64 ppfx_set_glare_bloom_num_levels(u64 /*context*/, u32 levels) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xff9030)(effect(), levels);
}

// A float at +0xc3c.
DECOMP_LEAF u64 ppfx_sub_fb3c10(u64 /*context*/, float value) {
    store<float>(effect() + (MUTANT(3) ? 0xc40 : 0xc3c), value);
    return 0;
}

// The glare object's +0x2d8, clamped to two constants.
DECOMP_LEAF u64 ppfx_sub_fb4130(u64 /*context*/, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, float)>(0xff91f0)(effect(), value);
}

// The glare object's +0x1a4.
DECOMP_LEAF u64 ppfx_sub_fb4100(u64 /*context*/, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, float)>(0xfd63d0)(effect(), value);
}

// The glare object's +0x2dc: the value when it and the glare object's +0xdc are positive, else +0xdc's.
DECOMP_LEAF u64 ppfx_sub_fb40f0(u64 /*context*/, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, float)>(0xff9130)(effect(), value);
}

// The glare object's +0x2d0.
DECOMP_LEAF u64 ppfx_sub_fb4120(u64 /*context*/, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, float)>(0xff91c0)(effect(), value);
}

// The glare object's +0x2d4.
DECOMP_LEAF u64 ppfx_sub_fb4110(u64 /*context*/, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, float)>(0xff9190)(effect(), value);
}

// A float at +0xc40.
DECOMP_LEAF u64 ppfx_sub_fb3c30(u64 /*context*/, float value) {
    store<float>(effect() + 0xc40, value);
    return 0;
}

// The glare object's +0x198.
DECOMP_LEAF u64 ppfx_sub_fb41a0(u64 /*context*/, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, float)>(0xfd99b0)(effect(), value);
}

// The glare object's byte at +0x16c.
DECOMP_LEAF u64 ppfx_sub_fb4160(u64 /*context*/, u8 enable) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xfd4c20)(effect(), static_cast<u32>(enable));
}

// The glare object's +0x170 and +0x184.
DECOMP_LEAF u64 ppfx_sub_fb4180(u64 /*context*/, float a, float b) {
    return game_function<GUEST_ABI u64 (*)(u64, float, float)>(0xfd4c40)(effect(), a, b);
}

// Sixteen bytes each from p and q to the glare object's +0x174 and +0x188.
DECOMP_LEAF u64 ppfx_sub_fb4190(u64 /*context*/, u64 p, u64 q) {
    return game_function<GUEST_ABI u64 (*)(u64, u64, u64)>(0xfd4c50)(effect(), p, q);
}


// ---- Light shafts (per light) ----

// CPostEffect::SetActiveLightShaft (iEffectLightIndex).
DECOMP_LEAF u64 ppfx_set_active_light_shaft(u64 /*context*/, u32 light) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xfd33d0)(effect(), light);
}

// CPostEffect::SetLightShaftSourceEnable.
DECOMP_LEAF u64 ppfx_set_light_shaft_source_enable(u64 /*context*/, u32 light, u8 enable) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32)>(0xfd9ac0)(effect(), light, static_cast<u32>(enable));
}

// A light shaft's two values.
DECOMP_LEAF u64 ppfx_sub_fb2940(u64 /*context*/, u32 light, float a, float b) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float)>(0xfd9f00)(effect(), light, a, b);
}

// A light shaft's source: a pointer, a dword and six floats.
DECOMP_LEAF u64 ppfx_sub_fb2900(u64 /*context*/, u32 light, u64 p, u32 c, float a, float b, float d, float e, float g, float h) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u64, u32, float, float, float, float, float, float)>(0xfd9c50)(effect(), light, p, c, a, b, d, e, g, MUTANT(6) ? g : h);
}

// A light shaft's colour: a pointer and four floats.
DECOMP_LEAF u64 ppfx_sub_fb2920(u64 /*context*/, u32 light, u64 p, float a, float b, float c, float d) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u64, float, float, float, float)>(0xfd9dc0)(effect(), light, p, a, b, c, d);
}

// A byte at +0x940 (0 unless the effect's +0x140 is set).
DECOMP_LEAF u64 ppfx_sub_fb2960(u64 /*context*/, u8 enable) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xfd2a90)(effect(), static_cast<u32>(enable));
}


// ---- Depth of field and auto focus ----

// CPostEffect::SetAutoFocusEnable.
DECOMP_LEAF u64 ppfx_set_auto_focus_enable(u64 /*context*/, u32 view, u8 enable) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32)>(0xfd5be0)(effect(), view, static_cast<u32>(enable));
}

// CPostEffect::SetAutoFocusAdjustment.
DECOMP_LEAF u64 ppfx_set_auto_focus_adjustment(u64 /*context*/, u32 view, u8 flag, float a, float b) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32, float, float)>(0xfd5d50)(effect(), view, static_cast<u32>(flag), a, b);
}

// CPostEffect::SetAutoFocusDelay.
DECOMP_LEAF u64 ppfx_set_auto_focus_delay(u64 /*context*/, u32 view, u32 mode, float delay) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32, float)>(0xfd6250)(effect(), view, mode, delay);
}

// CPostEffect::SetAutoFocusDistanceRange.
DECOMP_LEAF u64 ppfx_set_auto_focus_distance_range(u64 /*context*/, u32 view, float near_z, float far_z) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float)>(0xfd5f10)(effect(), view, near_z, far_z);
}

// CPostEffect::SetAutoFocusPosition.
DECOMP_LEAF u64 ppfx_set_auto_focus_position(u64 /*context*/, u32 view, float x, float y) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float)>(0xfd60a0)(effect(), view, x, y);
}

// CPostEffect::SetAutoFocusAdjustmentReset.
DECOMP_LEAF u64 ppfx_set_auto_focus_adjustment_reset(u64 /*context*/, u32 view) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xfdee90)(effect(), view);
}

// +0x154 of the object at the effect's +0x128, when the effect's +0x140 is set (else 1 back).
DECOMP_LEAF u64 ppfx_sub_fb3f80(u64 /*context*/, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, float)>(0xff9080)(effect(), value);
}

// A dword and two floats at +0xc28..+0xc30.
DECOMP_LEAF u64 ppfx_sub_fb34a0(u64 /*context*/, u32 mode, float a, float b) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float)>(0xfd4d20)(effect(), mode, a, b);
}

// +0x1c4 and +0x1c8 of the object at the effect's +0x128, when the effect's +0x140 is set (else 1 back).
DECOMP_LEAF u64 ppfx_sub_fb3fd0(u64 /*context*/, u32 a, u32 b) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32)>(0xff2590)(effect(), a, b);
}

// The depth of field's focus: +0x6d8.. +0x6ec (e only when not negative).
DECOMP_LEAF u64 ppfx_sub_fb2970(u64 /*context*/, u32 mode, float a, float b, float c, float d, float e) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float, float, float, float)>(0xfdc470)(effect(), mode, a, b, c, d, e);
}

// CPostEffect::SetDepthOfFieldEdgeQuality (eEdgeQuality).
DECOMP_LEAF u64 ppfx_set_depth_of_field_edge_quality(u64 /*context*/, u32 edge_quality) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xff8a40)(effect(), edge_quality);
}

// A byte at +0x70, beside the depth remap's parameters.
DECOMP_LEAF u64 ppfx_sub_fb3600(u64 /*context*/, u8 enable) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xfd4650)(effect(), static_cast<u32>(enable));
}

// CPostEffect::SetDepthOfFieldDepthRemapLutParameters (eRemapLutSourceChannel must be R).
DECOMP_LEAF u64 ppfx_set_depth_of_field_depth_remap_lut_parameters(u64 /*context*/, u32 a, u32 b, u32 source_channel, float value) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32, u32, float)>(0xfd4670)(effect(), a, b, source_channel, value);
}


// ---- Motion blur ----

// CPostEffect::SetMotionBlurEnable.
DECOMP_LEAF u64 ppfx_set_motion_blur_enable(u64 /*context*/, u8 enable) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xfd3b00)(effect(), static_cast<u32>(enable));
}

// A byte at +0x9e8.
DECOMP_LEAF u64 ppfx_sub_fb38b0(u64 /*context*/, u8 value) {
    store<u8>(effect() + 0x9e8, value);
    return 0;
}

// A dword at +0x9e4.
DECOMP_LEAF u64 ppfx_sub_fb3890(u64 /*context*/, u32 value) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xfd3b80)(effect(), value);
}

// Two floats at +0xa0c/+0xa10.
DECOMP_LEAF u64 ppfx_sub_fb38e0(u64 /*context*/, float a, float b) {
    return game_function<GUEST_ABI u64 (*)(u64, float, float)>(0xfd3c10)(effect(), a, b);
}

// A 4x4 matrix (a pointer to it) for +0x5ac and what is made from it, with the flag: sub_fb27d0's other form.
DECOMP_LEAF u64 ppfx_sub_fb27c0(u64 /*context*/, u64 matrix, u8 flag) {
    return game_function<GUEST_ABI u64 (*)(u64, u64, u32)>(0xfd8a70)(effect(), matrix, static_cast<u32>(flag));
}

// The same for the matrix at +0x62c.
DECOMP_LEAF u64 ppfx_sub_fb27e0(u64 /*context*/, u64 matrix, u8 flag) {
    return game_function<GUEST_ABI u64 (*)(u64, u64, u32)>(0xfd8eb0)(effect(), matrix, static_cast<u32>(flag));
}

// A float at +0x6ac (sub_fb2710's fourth).
DECOMP_LEAF u64 ppfx_sub_fb2800(u64 /*context*/, float value) {
    store<float>(effect() + 0x6ac, value);
    return 0;
}

// The motion blur's parameters at +0x9ec..+0x9fc.
DECOMP_LEAF u64 ppfx_sub_fb38f0(u64 /*context*/, u32 samples, u32 b, float a, float c, float d) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32, float, float, float)>(0xfd3b90)(effect(), samples, b, a, c, d);
}

// Two floats (not negative) and a byte at +0xa00..+0xa08.
DECOMP_LEAF u64 ppfx_sub_fb3900(u64 /*context*/, u8 flag, float a, float b) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, float, float)>(0xfd3be0)(effect(), static_cast<u32>(flag), a, b);
}


// ---- Feedback (the afterimage) ----

// CPostEffect::SetFeedbackEnable.
DECOMP_LEAF u64 ppfx_set_feedback_enable(u64 /*context*/, u8 enable) {
    return game_function<GUEST_ABI u64 (*)(u64, u32)>(0xfd40e0)(effect(), static_cast<u32>(enable));
}

// The feedback's three floats at +0x944..+0x94c.
DECOMP_LEAF u64 ppfx_sub_fb34d0(u64 /*context*/, float a, float b, float c) {
    return game_function<GUEST_ABI u64 (*)(u64, float, float, float)>(0xfd4150)(effect(), a, b, c);
}

// A float at +0x950.
DECOMP_LEAF u64 ppfx_sub_fb34e0(u64 /*context*/, float value) {
    store<float>(effect() + 0x950, value);
    return 0;
}


// ---- Post-process antialiasing ----

// CPostEffect::SetAntialiasEnable(enable, false).
DECOMP_LEAF u64 ppfx_set_antialias_enable(u64 /*context*/, u8 enable) {
    return game_function<GUEST_ABI u64 (*)(u64, u32, u32)>(0xfd31f0)(effect(), static_cast<u32>(enable), 0u);
}


namespace {
constexpr u8 kEntry_fb2710[] = {0x48, 0x8b, 0x3d, 0xb9, 0x37, 0x8b, 0x04};
constexpr u8 kEntry_fb2720[] = {0x48, 0x8b, 0x3d, 0xa9, 0x37, 0x8b, 0x04};
constexpr u8 kEntry_fb2740[] = {0x48, 0x8b, 0x3d, 0x89, 0x37, 0x8b, 0x04};
constexpr u8 kEntry_fb27d0[] = {0x48, 0x8b, 0x3d, 0xf9, 0x36, 0x8b, 0x04};
constexpr u8 kEntry_fb27f0[] = {0x48, 0x8b, 0x3d, 0xd9, 0x36, 0x8b, 0x04};
constexpr u8 kEntry_fb3fb0[] = {0x48, 0x8b, 0x3d, 0x19, 0x1f, 0x8b, 0x04};
constexpr u8 kEntry_fb31c0[] = {0x48, 0x8b, 0x3d, 0x09, 0x2d, 0x8b, 0x04};
constexpr u8 kEntry_fb32f0[] = {0x48, 0x8b, 0x3d, 0xd9, 0x2b, 0x8b, 0x04};
constexpr u8 kEntry_fb32e0[] = {0x48, 0x8b, 0x3d, 0xe9, 0x2b, 0x8b, 0x04};
constexpr u8 kEntry_fb3300[] = {0x48, 0x8b, 0x3d, 0xc9, 0x2b, 0x8b, 0x04};
constexpr u8 kEntry_fb3310[] = {0x48, 0x8b, 0x3d, 0xb9, 0x2b, 0x8b, 0x04};
constexpr u8 kEntry_fb3320[] = {0x48, 0x8b, 0x3d, 0xa9, 0x2b, 0x8b, 0x04};
constexpr u8 kEntry_fb3550[] = {0x48, 0x8b, 0x3d, 0x79, 0x29, 0x8b, 0x04};
constexpr u8 kEntry_fb3560[] = {0x48, 0x8b, 0x3d, 0x69, 0x29, 0x8b, 0x04};
constexpr u8 kEntry_fb37a0[] = {0x48, 0x8b, 0x3d, 0x29, 0x27, 0x8b, 0x04};
constexpr u8 kEntry_fb37d0[] = {0x48, 0x89, 0xc8, 0x48, 0x89, 0xd1};
constexpr u8 kEntry_fb4300[] = {0x48, 0x8b, 0x3d, 0xc9, 0x1b, 0x8b, 0x04};
constexpr u8 kEntry_fb4360[] = {0x48, 0x8b, 0x3d, 0x69, 0x1b, 0x8b, 0x04};
constexpr u8 kEntry_fb4390[] = {0x48, 0x8b, 0x3d, 0x39, 0x1b, 0x8b, 0x04};
constexpr u8 kEntry_fb43b0[] = {0x48, 0x8b, 0x3d, 0x19, 0x1b, 0x8b, 0x04};
constexpr u8 kEntry_fb43e0[] = {0x48, 0x8b, 0x3d, 0xe9, 0x1a, 0x8b, 0x04};
constexpr u8 kEntry_fb3670[] = {0x48, 0x8b, 0x3d, 0x59, 0x28, 0x8b, 0x04};
constexpr u8 kEntry_fb2840[] = {0x48, 0x8b, 0x3d, 0x89, 0x36, 0x8b, 0x04};
constexpr u8 kEntry_fb2860[] = {0x48, 0x8b, 0x3d, 0x69, 0x36, 0x8b, 0x04};
constexpr u8 kEntry_fb2850[] = {0x48, 0x8b, 0x3d, 0x79, 0x36, 0x8b, 0x04};
constexpr u8 kEntry_fb40e0[] = {0x48, 0x8b, 0x3d, 0xe9, 0x1d, 0x8b, 0x04};
constexpr u8 kEntry_fb3c10[] = {0x48, 0x8b, 0x05, 0xb9, 0x22, 0x8b, 0x04};
constexpr u8 kEntry_fb4130[] = {0x48, 0x8b, 0x3d, 0x99, 0x1d, 0x8b, 0x04};
constexpr u8 kEntry_fb4100[] = {0x48, 0x8b, 0x3d, 0xc9, 0x1d, 0x8b, 0x04};
constexpr u8 kEntry_fb40f0[] = {0x48, 0x8b, 0x3d, 0xd9, 0x1d, 0x8b, 0x04};
constexpr u8 kEntry_fb4120[] = {0x48, 0x8b, 0x3d, 0xa9, 0x1d, 0x8b, 0x04};
constexpr u8 kEntry_fb4110[] = {0x48, 0x8b, 0x3d, 0xb9, 0x1d, 0x8b, 0x04};
constexpr u8 kEntry_fb3c30[] = {0x48, 0x8b, 0x05, 0x99, 0x22, 0x8b, 0x04};
constexpr u8 kEntry_fb41a0[] = {0x48, 0x8b, 0x3d, 0x29, 0x1d, 0x8b, 0x04};
constexpr u8 kEntry_fb4160[] = {0x48, 0x8b, 0x3d, 0x69, 0x1d, 0x8b, 0x04};
constexpr u8 kEntry_fb4180[] = {0x48, 0x8b, 0x3d, 0x49, 0x1d, 0x8b, 0x04};
constexpr u8 kEntry_fb4190[] = {0x48, 0x8b, 0x3d, 0x39, 0x1d, 0x8b, 0x04};
constexpr u8 kEntry_fb28a0[] = {0x48, 0x8b, 0x3d, 0x29, 0x36, 0x8b, 0x04};
constexpr u8 kEntry_fb28d0[] = {0x48, 0x8b, 0x3d, 0xf9, 0x35, 0x8b, 0x04};
constexpr u8 kEntry_fb2940[] = {0x48, 0x8b, 0x3d, 0x89, 0x35, 0x8b, 0x04};
constexpr u8 kEntry_fb2900[] = {0x48, 0x8b, 0x3d, 0xc9, 0x35, 0x8b, 0x04};
constexpr u8 kEntry_fb2920[] = {0x48, 0x8b, 0x3d, 0xa9, 0x35, 0x8b, 0x04};
constexpr u8 kEntry_fb2960[] = {0x48, 0x8b, 0x3d, 0x69, 0x35, 0x8b, 0x04};
constexpr u8 kEntry_fb3350[] = {0x48, 0x8b, 0x3d, 0x79, 0x2b, 0x8b, 0x04};
constexpr u8 kEntry_fb34b0[] = {0x48, 0x8b, 0x3d, 0x19, 0x2a, 0x8b, 0x04};
constexpr u8 kEntry_fb3490[] = {0x48, 0x8b, 0x3d, 0x39, 0x2a, 0x8b, 0x04};
constexpr u8 kEntry_fb3470[] = {0x48, 0x8b, 0x3d, 0x59, 0x2a, 0x8b, 0x04};
constexpr u8 kEntry_fb3480[] = {0x48, 0x8b, 0x3d, 0x49, 0x2a, 0x8b, 0x04};
constexpr u8 kEntry_fb3340[] = {0x48, 0x8b, 0x3d, 0x89, 0x2b, 0x8b, 0x04};
constexpr u8 kEntry_fb3f80[] = {0x48, 0x8b, 0x3d, 0x49, 0x1f, 0x8b, 0x04};
constexpr u8 kEntry_fb34a0[] = {0x48, 0x8b, 0x3d, 0x29, 0x2a, 0x8b, 0x04};
constexpr u8 kEntry_fb3fd0[] = {0x48, 0x8b, 0x3d, 0xf9, 0x1e, 0x8b, 0x04};
constexpr u8 kEntry_fb2970[] = {0x48, 0x8b, 0x3d, 0x59, 0x35, 0x8b, 0x04};
constexpr u8 kEntry_fb45c0[] = {0x48, 0x8b, 0x3d, 0x09, 0x19, 0x8b, 0x04};
constexpr u8 kEntry_fb3600[] = {0x48, 0x8b, 0x3d, 0xc9, 0x28, 0x8b, 0x04};
constexpr u8 kEntry_fb3660[] = {0x48, 0x8b, 0x3d, 0x69, 0x28, 0x8b, 0x04};
constexpr u8 kEntry_fb3870[] = {0x48, 0x8b, 0x3d, 0x59, 0x26, 0x8b, 0x04};
constexpr u8 kEntry_fb38b0[] = {0x48, 0x8b, 0x05, 0x19, 0x26, 0x8b, 0x04};
constexpr u8 kEntry_fb3890[] = {0x48, 0x8b, 0x3d, 0x39, 0x26, 0x8b, 0x04};
constexpr u8 kEntry_fb38e0[] = {0x48, 0x8b, 0x3d, 0xe9, 0x25, 0x8b, 0x04};
constexpr u8 kEntry_fb27c0[] = {0x48, 0x8b, 0x3d, 0x09, 0x37, 0x8b, 0x04};
constexpr u8 kEntry_fb27e0[] = {0x48, 0x8b, 0x3d, 0xe9, 0x36, 0x8b, 0x04};
constexpr u8 kEntry_fb2800[] = {0x48, 0x8b, 0x05, 0xc9, 0x36, 0x8b, 0x04};
constexpr u8 kEntry_fb38f0[] = {0x48, 0x8b, 0x3d, 0xd9, 0x25, 0x8b, 0x04};
constexpr u8 kEntry_fb3900[] = {0x48, 0x8b, 0x3d, 0xc9, 0x25, 0x8b, 0x04};
constexpr u8 kEntry_fb34c0[] = {0x48, 0x8b, 0x3d, 0x09, 0x2a, 0x8b, 0x04};
constexpr u8 kEntry_fb34d0[] = {0x48, 0x8b, 0x3d, 0xf9, 0x29, 0x8b, 0x04};
constexpr u8 kEntry_fb34e0[] = {0x48, 0x8b, 0x05, 0xe9, 0x29, 0x8b, 0x04};
constexpr u8 kEntry_fb37f0[] = {0x48, 0x8b, 0x3d, 0xd9, 0x26, 0x8b, 0x04};
}  // namespace

void decomp_yebis_ppfx_setters_add() {
    const DecompFunction fns[] = {
        {"ppfx_set_render_scene_luminance_scale", "yebis", 0xfb2710, kEntry_fb2710, sizeof kEntry_fb2710, reinterpret_cast<void*>(&ppfx_set_render_scene_luminance_scale), DecompKind::Leaf},
        {"ppfx_sub_fb2720", "yebis", 0xfb2720, kEntry_fb2720, sizeof kEntry_fb2720, reinterpret_cast<void*>(&ppfx_sub_fb2720), DecompKind::Leaf},
        {"ppfx_sub_fb2740", "yebis", 0xfb2740, kEntry_fb2740, sizeof kEntry_fb2740, reinterpret_cast<void*>(&ppfx_sub_fb2740), DecompKind::Leaf},
        {"ppfx_sub_fb27d0", "yebis", 0xfb27d0, kEntry_fb27d0, sizeof kEntry_fb27d0, reinterpret_cast<void*>(&ppfx_sub_fb27d0), DecompKind::Leaf},
        {"ppfx_sub_fb27f0", "yebis", 0xfb27f0, kEntry_fb27f0, sizeof kEntry_fb27f0, reinterpret_cast<void*>(&ppfx_sub_fb27f0), DecompKind::Leaf},
        {"ppfx_sub_fb3fb0", "yebis", 0xfb3fb0, kEntry_fb3fb0, sizeof kEntry_fb3fb0, reinterpret_cast<void*>(&ppfx_sub_fb3fb0), DecompKind::Leaf},
        {"ppfx_set_auto_exposure_enable", "yebis", 0xfb31c0, kEntry_fb31c0, sizeof kEntry_fb31c0, reinterpret_cast<void*>(&ppfx_set_auto_exposure_enable), DecompKind::Leaf},
        {"ppfx_set_auto_exposure_middle_gray", "yebis", 0xfb32f0, kEntry_fb32f0, sizeof kEntry_fb32f0, reinterpret_cast<void*>(&ppfx_set_auto_exposure_middle_gray), DecompKind::Leaf},
        {"ppfx_set_auto_exposure_range", "yebis", 0xfb32e0, kEntry_fb32e0, sizeof kEntry_fb32e0, reinterpret_cast<void*>(&ppfx_set_auto_exposure_range), DecompKind::Leaf},
        {"ppfx_set_auto_exposure_metering_area", "yebis", 0xfb3300, kEntry_fb3300, sizeof kEntry_fb3300, reinterpret_cast<void*>(&ppfx_set_auto_exposure_metering_area), DecompKind::Leaf},
        {"ppfx_set_auto_exposure_delay", "yebis", 0xfb3310, kEntry_fb3310, sizeof kEntry_fb3310, reinterpret_cast<void*>(&ppfx_set_auto_exposure_delay), DecompKind::Leaf},
        {"ppfx_set_auto_exposure_adjustment", "yebis", 0xfb3320, kEntry_fb3320, sizeof kEntry_fb3320, reinterpret_cast<void*>(&ppfx_set_auto_exposure_adjustment), DecompKind::Leaf},
        {"ppfx_set_tonemap_parameters", "yebis", 0xfb3550, kEntry_fb3550, sizeof kEntry_fb3550, reinterpret_cast<void*>(&ppfx_set_tonemap_parameters), DecompKind::Leaf},
        {"ppfx_sub_fb3560", "yebis", 0xfb3560, kEntry_fb3560, sizeof kEntry_fb3560, reinterpret_cast<void*>(&ppfx_sub_fb3560), DecompKind::Leaf},
        {"ppfx_set_chromatic_aberration_enable", "yebis", 0xfb37a0, kEntry_fb37a0, sizeof kEntry_fb37a0, reinterpret_cast<void*>(&ppfx_set_chromatic_aberration_enable), DecompKind::Leaf},
        {"ppfx_sub_fb37d0", "yebis", 0xfb37d0, kEntry_fb37d0, sizeof kEntry_fb37d0, reinterpret_cast<void*>(&ppfx_sub_fb37d0), DecompKind::Leaf},
        {"ppfx_initialize_matrix", "yebis", 0xfb4300, kEntry_fb4300, sizeof kEntry_fb4300, reinterpret_cast<void*>(&ppfx_initialize_matrix), DecompKind::Leaf},
        {"ppfx_multiply_matrix_fb4360", "yebis", 0xfb4360, kEntry_fb4360, sizeof kEntry_fb4360, reinterpret_cast<void*>(&ppfx_multiply_matrix_fb4360), DecompKind::Leaf},
        {"ppfx_multiply_matrix_fb4390", "yebis", 0xfb4390, kEntry_fb4390, sizeof kEntry_fb4390, reinterpret_cast<void*>(&ppfx_multiply_matrix_fb4390), DecompKind::Leaf},
        {"ppfx_multiply_matrix_fb43b0", "yebis", 0xfb43b0, kEntry_fb43b0, sizeof kEntry_fb43b0, reinterpret_cast<void*>(&ppfx_multiply_matrix_fb43b0), DecompKind::Leaf},
        {"ppfx_multiply_matrix_fb43e0", "yebis", 0xfb43e0, kEntry_fb43e0, sizeof kEntry_fb43e0, reinterpret_cast<void*>(&ppfx_multiply_matrix_fb43e0), DecompKind::Leaf},
        {"ppfx_sub_fb3670", "yebis", 0xfb3670, kEntry_fb3670, sizeof kEntry_fb3670, reinterpret_cast<void*>(&ppfx_sub_fb3670), DecompKind::Leaf},
        {"ppfx_set_glare_parameters", "yebis", 0xfb2840, kEntry_fb2840, sizeof kEntry_fb2840, reinterpret_cast<void*>(&ppfx_set_glare_parameters), DecompKind::Leaf},
        {"ppfx_sub_fb2860", "yebis", 0xfb2860, kEntry_fb2860, sizeof kEntry_fb2860, reinterpret_cast<void*>(&ppfx_sub_fb2860), DecompKind::Leaf},
        {"ppfx_sub_fb2850", "yebis", 0xfb2850, kEntry_fb2850, sizeof kEntry_fb2850, reinterpret_cast<void*>(&ppfx_sub_fb2850), DecompKind::Leaf},
        {"ppfx_set_glare_bloom_num_levels", "yebis", 0xfb40e0, kEntry_fb40e0, sizeof kEntry_fb40e0, reinterpret_cast<void*>(&ppfx_set_glare_bloom_num_levels), DecompKind::Leaf},
        {"ppfx_sub_fb3c10", "yebis", 0xfb3c10, kEntry_fb3c10, sizeof kEntry_fb3c10, reinterpret_cast<void*>(&ppfx_sub_fb3c10), DecompKind::Leaf},
        {"ppfx_sub_fb4130", "yebis", 0xfb4130, kEntry_fb4130, sizeof kEntry_fb4130, reinterpret_cast<void*>(&ppfx_sub_fb4130), DecompKind::Leaf},
        {"ppfx_sub_fb4100", "yebis", 0xfb4100, kEntry_fb4100, sizeof kEntry_fb4100, reinterpret_cast<void*>(&ppfx_sub_fb4100), DecompKind::Leaf},
        {"ppfx_sub_fb40f0", "yebis", 0xfb40f0, kEntry_fb40f0, sizeof kEntry_fb40f0, reinterpret_cast<void*>(&ppfx_sub_fb40f0), DecompKind::Leaf},
        {"ppfx_sub_fb4120", "yebis", 0xfb4120, kEntry_fb4120, sizeof kEntry_fb4120, reinterpret_cast<void*>(&ppfx_sub_fb4120), DecompKind::Leaf},
        {"ppfx_sub_fb4110", "yebis", 0xfb4110, kEntry_fb4110, sizeof kEntry_fb4110, reinterpret_cast<void*>(&ppfx_sub_fb4110), DecompKind::Leaf},
        {"ppfx_sub_fb3c30", "yebis", 0xfb3c30, kEntry_fb3c30, sizeof kEntry_fb3c30, reinterpret_cast<void*>(&ppfx_sub_fb3c30), DecompKind::Leaf},
        {"ppfx_sub_fb41a0", "yebis", 0xfb41a0, kEntry_fb41a0, sizeof kEntry_fb41a0, reinterpret_cast<void*>(&ppfx_sub_fb41a0), DecompKind::Leaf},
        {"ppfx_sub_fb4160", "yebis", 0xfb4160, kEntry_fb4160, sizeof kEntry_fb4160, reinterpret_cast<void*>(&ppfx_sub_fb4160), DecompKind::Leaf},
        {"ppfx_sub_fb4180", "yebis", 0xfb4180, kEntry_fb4180, sizeof kEntry_fb4180, reinterpret_cast<void*>(&ppfx_sub_fb4180), DecompKind::Leaf},
        {"ppfx_sub_fb4190", "yebis", 0xfb4190, kEntry_fb4190, sizeof kEntry_fb4190, reinterpret_cast<void*>(&ppfx_sub_fb4190), DecompKind::Leaf},
        {"ppfx_set_active_light_shaft", "yebis", 0xfb28a0, kEntry_fb28a0, sizeof kEntry_fb28a0, reinterpret_cast<void*>(&ppfx_set_active_light_shaft), DecompKind::Leaf},
        {"ppfx_set_light_shaft_source_enable", "yebis", 0xfb28d0, kEntry_fb28d0, sizeof kEntry_fb28d0, reinterpret_cast<void*>(&ppfx_set_light_shaft_source_enable), DecompKind::Leaf},
        {"ppfx_sub_fb2940", "yebis", 0xfb2940, kEntry_fb2940, sizeof kEntry_fb2940, reinterpret_cast<void*>(&ppfx_sub_fb2940), DecompKind::Leaf},
        {"ppfx_sub_fb2900", "yebis", 0xfb2900, kEntry_fb2900, sizeof kEntry_fb2900, reinterpret_cast<void*>(&ppfx_sub_fb2900), DecompKind::Leaf},
        {"ppfx_sub_fb2920", "yebis", 0xfb2920, kEntry_fb2920, sizeof kEntry_fb2920, reinterpret_cast<void*>(&ppfx_sub_fb2920), DecompKind::Leaf},
        {"ppfx_sub_fb2960", "yebis", 0xfb2960, kEntry_fb2960, sizeof kEntry_fb2960, reinterpret_cast<void*>(&ppfx_sub_fb2960), DecompKind::Leaf},
        {"ppfx_set_auto_focus_enable", "yebis", 0xfb3350, kEntry_fb3350, sizeof kEntry_fb3350, reinterpret_cast<void*>(&ppfx_set_auto_focus_enable), DecompKind::Leaf},
        {"ppfx_set_auto_focus_adjustment", "yebis", 0xfb34b0, kEntry_fb34b0, sizeof kEntry_fb34b0, reinterpret_cast<void*>(&ppfx_set_auto_focus_adjustment), DecompKind::Leaf},
        {"ppfx_set_auto_focus_delay", "yebis", 0xfb3490, kEntry_fb3490, sizeof kEntry_fb3490, reinterpret_cast<void*>(&ppfx_set_auto_focus_delay), DecompKind::Leaf},
        {"ppfx_set_auto_focus_distance_range", "yebis", 0xfb3470, kEntry_fb3470, sizeof kEntry_fb3470, reinterpret_cast<void*>(&ppfx_set_auto_focus_distance_range), DecompKind::Leaf},
        {"ppfx_set_auto_focus_position", "yebis", 0xfb3480, kEntry_fb3480, sizeof kEntry_fb3480, reinterpret_cast<void*>(&ppfx_set_auto_focus_position), DecompKind::Leaf},
        {"ppfx_set_auto_focus_adjustment_reset", "yebis", 0xfb3340, kEntry_fb3340, sizeof kEntry_fb3340, reinterpret_cast<void*>(&ppfx_set_auto_focus_adjustment_reset), DecompKind::Leaf},
        {"ppfx_sub_fb3f80", "yebis", 0xfb3f80, kEntry_fb3f80, sizeof kEntry_fb3f80, reinterpret_cast<void*>(&ppfx_sub_fb3f80), DecompKind::Leaf},
        {"ppfx_sub_fb34a0", "yebis", 0xfb34a0, kEntry_fb34a0, sizeof kEntry_fb34a0, reinterpret_cast<void*>(&ppfx_sub_fb34a0), DecompKind::Leaf},
        {"ppfx_sub_fb3fd0", "yebis", 0xfb3fd0, kEntry_fb3fd0, sizeof kEntry_fb3fd0, reinterpret_cast<void*>(&ppfx_sub_fb3fd0), DecompKind::Leaf},
        {"ppfx_sub_fb2970", "yebis", 0xfb2970, kEntry_fb2970, sizeof kEntry_fb2970, reinterpret_cast<void*>(&ppfx_sub_fb2970), DecompKind::Leaf},
        {"ppfx_set_depth_of_field_edge_quality", "yebis", 0xfb45c0, kEntry_fb45c0, sizeof kEntry_fb45c0, reinterpret_cast<void*>(&ppfx_set_depth_of_field_edge_quality), DecompKind::Leaf},
        {"ppfx_sub_fb3600", "yebis", 0xfb3600, kEntry_fb3600, sizeof kEntry_fb3600, reinterpret_cast<void*>(&ppfx_sub_fb3600), DecompKind::Leaf},
        {"ppfx_set_depth_of_field_depth_remap_lut_parameters", "yebis", 0xfb3660, kEntry_fb3660, sizeof kEntry_fb3660, reinterpret_cast<void*>(&ppfx_set_depth_of_field_depth_remap_lut_parameters), DecompKind::Leaf},
        {"ppfx_set_motion_blur_enable", "yebis", 0xfb3870, kEntry_fb3870, sizeof kEntry_fb3870, reinterpret_cast<void*>(&ppfx_set_motion_blur_enable), DecompKind::Leaf},
        {"ppfx_sub_fb38b0", "yebis", 0xfb38b0, kEntry_fb38b0, sizeof kEntry_fb38b0, reinterpret_cast<void*>(&ppfx_sub_fb38b0), DecompKind::Leaf},
        {"ppfx_sub_fb3890", "yebis", 0xfb3890, kEntry_fb3890, sizeof kEntry_fb3890, reinterpret_cast<void*>(&ppfx_sub_fb3890), DecompKind::Leaf},
        {"ppfx_sub_fb38e0", "yebis", 0xfb38e0, kEntry_fb38e0, sizeof kEntry_fb38e0, reinterpret_cast<void*>(&ppfx_sub_fb38e0), DecompKind::Leaf},
        {"ppfx_sub_fb27c0", "yebis", 0xfb27c0, kEntry_fb27c0, sizeof kEntry_fb27c0, reinterpret_cast<void*>(&ppfx_sub_fb27c0), DecompKind::Leaf},
        {"ppfx_sub_fb27e0", "yebis", 0xfb27e0, kEntry_fb27e0, sizeof kEntry_fb27e0, reinterpret_cast<void*>(&ppfx_sub_fb27e0), DecompKind::Leaf},
        {"ppfx_sub_fb2800", "yebis", 0xfb2800, kEntry_fb2800, sizeof kEntry_fb2800, reinterpret_cast<void*>(&ppfx_sub_fb2800), DecompKind::Leaf},
        {"ppfx_sub_fb38f0", "yebis", 0xfb38f0, kEntry_fb38f0, sizeof kEntry_fb38f0, reinterpret_cast<void*>(&ppfx_sub_fb38f0), DecompKind::Leaf},
        {"ppfx_sub_fb3900", "yebis", 0xfb3900, kEntry_fb3900, sizeof kEntry_fb3900, reinterpret_cast<void*>(&ppfx_sub_fb3900), DecompKind::Leaf},
        {"ppfx_set_feedback_enable", "yebis", 0xfb34c0, kEntry_fb34c0, sizeof kEntry_fb34c0, reinterpret_cast<void*>(&ppfx_set_feedback_enable), DecompKind::Leaf},
        {"ppfx_sub_fb34d0", "yebis", 0xfb34d0, kEntry_fb34d0, sizeof kEntry_fb34d0, reinterpret_cast<void*>(&ppfx_sub_fb34d0), DecompKind::Leaf},
        {"ppfx_sub_fb34e0", "yebis", 0xfb34e0, kEntry_fb34e0, sizeof kEntry_fb34e0, reinterpret_cast<void*>(&ppfx_sub_fb34e0), DecompKind::Leaf},
        {"ppfx_set_antialias_enable", "yebis", 0xfb37f0, kEntry_fb37f0, sizeof kEntry_fb37f0, reinterpret_cast<void*>(&ppfx_set_antialias_enable), DecompKind::Leaf},
    };
    for (const DecompFunction& fn : fns) decomp_add(fn);
}
