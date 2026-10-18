// HLSL from a lifted shader (gcn/lift.h).
//
// The lifter's SPIR-V is the shader's IR of record: typed SSA in one function,
// structured selections and loops, a fixed set of globals. emit_hlsl prints
// that module as HLSL (Shader Model 6, -HV 2021) that DXC compiles back to
// SPIR-V for Vulkan with the same interface - descriptor sets and bindings,
// locations, built-ins, specialization constants - and the same computation,
// instruction for instruction where the result's bits could depend on it:
//
//   * every float value is `precise`, so DXC marks its arithmetic
//     NoContraction as the lift does (no fusing, no reassociation, no
//     folding by spirv-opt);
//   * an operation whose HLSL spelling DXC lowers to another instruction is
//     written as inline SPIR-V (vk::ext_instruction) with a readable name:
//     FSign (HLSL sign() returns an int), Ldexp (ldexp() is x * exp2(e)),
//     PackHalf2x16, FMin/FMax (min()/max() are NMin/NMax, which the lift
//     uses for v_min/v_max), bitfield extracts, and Fma: mad() is Fma
//     decorated NoContraction, which a driver must fuse, where the lift's Fma
//     leaves fusing to the driver (lavapipe splits it: the rounding differs);
//   * discard stays OpKill: the helper-lane query is the HelperInvocation
//     built-in (vk::ext_builtin_input), not IsHelperLane(), which would turn
//     every discard into a demote;
//   * quad operations are QuadReadLaneAt / QuadReadAcross*, which DXC lowers
//     to the lift's OpGroupNonUniformQuadBroadcast / QuadSwap;
//   * an Invariant output (the position, with BBHOST_VS_INVARIANT) keeps the
//     decoration as [[vk::ext_decorate(18)]] on its member;
//   * a lift for a device that rounds halves itself (gcn/half.h
//     native_half_rtz: RADV) declares DenormPreserve 16 and RoundingModeRTZ
//     16, which HLSL cannot spell: the entry point declares them as
//     vk::ext_execution_mode with their [[vk::ext_capability]], and the lift's
//     conversions between f32 and f16 and bitcasts between half pairs and
//     words are its own instructions (inline OpFConvert, OpBitcast): DXC
//     would evaluate float16_t(constant) itself, rounding to nearest. Halves
//     are only converted and moved; arithmetic on them is not printed.
//
// Tessellation: the LS is a vertex shader whose control point is an array
// output; the domain shader takes its control points as an OutputPatch and
// its patch constants beside the factors HLSL requires, with the spacing and
// winding the lift declares as inline execution modes. The game's hull - one
// function an output control point runs, then a barrier, then the patch's
// constants and factors written by the invocation that holds the patch - is
// HLSL's two phases from that one body: the control-point function runs it
// for its invocation and returns at the barrier with the control point it
// wrote; the patch-constant function runs it again as invocation 0, reading
// every control point's output from the OutputPatch after the barrier, and
// returns the constants and factors it wrote. That is the lift's result as
// long as the patch's values are written by invocation 0 only, which the
// game's hulls do (their stores are under a test of the control point index
// against 1); the replays check it.
//
// Constant buffers are ByteAddressBuffers read a dword at a time, as the lift
// reads them; the params block is a ConstantBuffer with the lift's offsets.
// Values keep the module's ids (v123), so the HLSL can be read beside the
// SPIR-V; resources keep the names the module gives them (the engine's,
// after gcn::name_lift).
//
// Provenance: the source is derived from the game's shaders. It is generated
// on the player's machine and never shipped or committed.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gcn {

struct ShaderNames;

struct HlslOptions {
    std::string title;  // the program's name, for the header comment
    // The engine's names (gcn::name_lift): each constant-buffer load at the
    // buffer's base plus a constant gets the member it reads as a comment
    // (`// cbSceneParam.FC_MatrixProj[2].w`). Null: no comments.
    const ShaderNames* names = nullptr;
    // A hull shader (the control stage): the domain, partitioning and winding
    // HLSL declares on it, as VGT_TF_PARAM gives them to the evaluation stage
    // (TranslateOptions tess_quads, tess_spacing, tess_cw). The lift's control
    // stage declares none of them; DXC's does, and they must agree.
    bool tess_quads = false;
    int tess_spacing = 0;  // 0 integer, 1 fractional-odd, 2 fractional-even
    bool tess_cw = true;
};

struct HlslResult {
    std::string source;
    std::string profile;                  // ps_6_6, vs_6_6, hs_6_6, ds_6_6
    std::string entry = "main";
    std::string language = "2021";        // -HV: 202x where vk::BufferPointer is used (a compute lift's atomics)
    std::vector<std::string> arguments;   // DXC arguments this module needs (-enable-16bit-types)
    std::vector<std::string> errors;      // what the printer does not take: empty on success
    // A sample takes its offset at run time: the Offset image operand, which
    // VK_KHR_maintenance8 allows on samples (the lift writes it only where the
    // device has the feature, gcn::runtime_sample_offsets) and DXC's own
    // validator refuses (VUID-RuntimeSpirv-Offset-10213), with no option to
    // allow it. `arguments` then turn DXC's validation off (-Vd), and the
    // caller validates the module itself with the offset allowed
    // (spvtools::ValidatorOptions::SetAllowOffsetTextureOperand) - or, with no
    // validator at hand, keeps the lift.
    bool validate_offsets = false;
    bool ok() const { return errors.empty() && !source.empty(); }
};

// Prints a lifted module as HLSL. A module outside the subset the lifter
// writes comes back with `errors` (its first unsupported construct) and the
// caller keeps the lifted SPIR-V.
HlslResult emit_hlsl(const std::vector<std::uint32_t>& spirv, const HlslOptions& options = {});

// Whether two modules declare the same interface: execution model and modes
// (with their literals: RoundingModeRTZ 16), descriptor bindings (set,
// binding, kind, image shape and texel type, buffer layout), stage inputs and
// outputs (location or built-in, type, interpolation), and specialization
// constant ids. Empty when they do, else the first difference.
std::string hlsl_interface_diff(const std::vector<std::uint32_t>& lifted, const std::vector<std::uint32_t>& compiled);

// Whether DXC kept the lift's operations where their results' bits depend on
// which instruction computes them: no float, image, quad, derivative or
// discard operation (by opcode, and by GLSL.std.450 number) that the lift
// does not use, none more often than the lift uses it (DXC may merge and drop
// them, not add them; GetDimensions' size and level queries and
// CalculateLevelOfDetail excepted), no constant of 16-bit floats the lift does
// not have (a conversion to half DXC evaluated itself), and every float add,
// subtract, multiply, divide and negate (of something other than a literal)
// NoContraction. Empty when it did, else the first difference.
std::string hlsl_operation_diff(const std::vector<std::uint32_t>& lifted, const std::vector<std::uint32_t>& compiled);

}  // namespace gcn
