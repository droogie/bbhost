// Exact decoding of 16-bit half floats in SPIR-V, shared by the translator and
// the lifter.
//
// GCN's packed exports (v_cvt_pkrtz_f16_f32, then exp ... compr) and
// v_cvt_f32_f16 carry half floats. Decoded with GLSL.std.450
// UnpackHalf2x16, a driver may fold UnpackHalf2x16(PackHalf2x16(x)) back to x
// and so drop the half-precision rounding, depending on what lies between the
// two: on the RTX 4070 driver most translated shaders were folded and
// 3eaa5555+dbeccd9c was not, so the same export differed by one target quantum
// between translation and lift.
// Decoding with exact integer and ldexp operations cannot be folded, so every
// pipeline keeps the rounding and the two builders agree on any driver.
#pragma once

#include "gcn/spirv.h"

namespace gcn {

struct SpvScalarTypes {
    spv::Id t_bool = 0, t_u32 = 0, t_i32 = 0, t_f32 = 0;
};

// The half in `h` (a u32 below 0x10000) as f32: what UnpackHalf2x16 gives for
// it, with infinities and a quiet NaN for exponent 31.
spv::Id emit_unpack_half16(spv::Module& m, const SpvScalarTypes& t, spv::Id h);

// The f32 `x` rounded toward zero to half precision, as v_cvt_pkrtz_f16_f32 and
// then the decode above leave it, without the pack: for a compressed export of
// floats the program packed itself, where decoding the pack was ~18 operations
// a component. Integer operations, which a driver cannot fold. The same for
// magnitudes from half's smallest normal (2^-14) up, zeros, infinities and
// NaNs; a finite magnitude past half's largest becomes it. Below 2^-14, where
// the half is denormal, the result keeps ten bits of mantissa instead: it
// differs by less than 2^-24, and the target's own conversion follows.
spv::Id emit_rtz_half(spv::Module& m, const SpvScalarTypes& t, spv::Id x);
// Whether compressed exports of a pair the program packed round it with
// emit_rtz_half (BBHOST_EXPORT_RTZ=0: they decode the pack, as before).
bool export_rtz_on();
// Half rounding by the device itself (set by the host, off until it is): the
// device converts f32 to f16 rounding toward zero under the module's
// RoundingModeRTZ 16 and keeps f16 denormals (DenormPreserve 16), so
// v_cvt_pkrtz_f16_f32 and the compressed exports that read it are two native
// conversions a pair instead of emit_rtz_half / emit_unpack_half16's integer
// operations - exactly GCN's round toward zero, denormals included.
void set_native_half_rtz(bool on);
bool native_half_rtz();  // and export_rtz_on()
// Whether a program may run under RoundingModeRTZ 16: its only conversion to
// f16 is v_cvt_pkrtz_f16_f32 (no v_cvt_f16_f32, which rounds to nearest, and
// no 16-bit float buffer stores).
bool program_allows_native_half_rtz(const struct Program& prog);
// Whether the legacy multiply's zero rule is written as DXVK writes D3D9's,
// min(|a|, |b|) == 0 - Mesa's compiler makes it one v_mul_legacy_f32 - or as
// a == 0 || b == 0. AMD's own compiler matches neither to its native
// instruction, and there the select is cheaper: an operand's compare serves
// every multiply it is in (a normalize's three share the length's), where each
// min is its own - the main light shader's 741 vector instructions were 711.
// The host sets the device's form (set_legacy_mul_select: true on AMD's
// drivers) before any translation; BBHOST_LEGACY_MUL=min or =select forces one.
bool legacy_mul_min_form();
void set_legacy_mul_select(bool on);
bool legacy_mul_plain();  // BBHOST_LEGACY_MUL=plain: a plain multiply (a measurement)

}  // namespace gcn
