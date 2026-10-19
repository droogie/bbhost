// Typed per-pixel lifting of GCN pixel shaders.
//
// The translator (translate.h) emulates the SIMD machine: registers are u32
// variables, EXEC and VCC are lane masks rebuilt with subgroup ballots, and
// every masked write selects between reloaded words. The lifter emits the
// same computation for one pixel as typed SSA values: floats stay floats, a
// lane mask is the pixel's own boolean, and the EXEC save/branch/restore idiom
// becomes selects over values computed in uniform control flow.
//
// It lifts only what it can show is the same computation and rejects anything
// else with a reason; the caller then keeps the translated shader. The facts
// each lift relied on are returned in `proof`.
#pragma once

#include "gcn/isa.h"
#include "gcn/names.h"
#include "gcn/translate.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gcn {

// The GCN program's wave and quad operations and what the lift made of them.
// Speed-only: shortcuts a wave takes that change no pixel's result - the lift
// runs per pixel without them, and an HLSL emitter drops them too. Semantic:
// operations whose results depend on the other pixels of a quad or on which
// pixels survive - kept, as Shader Model 6 quad operations, derivatives,
// helper-lane queries or discard.
struct WaveOps {
    // speed-only (dropped)
    std::uint32_t execz_skips = 0;   // s_cbranch_execz over a region: the region runs, its writes masked per pixel
    std::uint32_t kill_skips = 0;    // s_cbranch_scc0 over a kill region: the pixel's own guard bit masks it
    std::uint32_t wave_tests = 0;    // s_cbranch_vccz/vccnz on a VCC shown the same in every lane: a per-pixel condition
    std::uint32_t lane_exits = 0;    // a divergent loop's per-lane exit idiom (s_andn2_b64 + s_cbranch_scc0): a per-pixel break
    std::uint32_t whole_quad = 0;    // s_wqm_b64: helper lanes run whole quads anyway
    std::uint32_t spill_lanes = 0;   // v_writelane_b32 / v_readlane_b32 of a spilled scalar: a register
    // semantic (kept)
    std::uint32_t quad_swizzles = 0;  // ds_swizzle_b32 within a quad: QuadReadLaneAt / QuadReadAcross*
    std::uint32_t implicit_lod = 0;   // samples whose LOD comes from the quad's derivatives
    std::uint32_t lod_queries = 0;    // image_get_lod: CalculateLevelOfDetailUnclamped
    std::uint32_t helper_queries = 0; // whether a quad holds a live pixel (HelperInvocation + quad reads)
    std::uint32_t discards = 0;       // pixels discarded: EXEC clear at s_endpgm, or a clip distance below 0
    // compute: a wave's ds_append / ds_consume step in GDS made each
    // invocation's own (HLSL's IncrementCounter / DecrementCounter, the
    // lane's rank (v_mbcnt) folded into its slot); and kept, a lane's rank
    // among a mask's lanes outside that idiom (WavePrefixCountBits)
    std::uint32_t gds_steps = 0;
    std::uint32_t lane_ranks = 0;
};

struct LiftResult {
    std::vector<std::uint32_t> spirv;
    std::vector<std::string> rejections;  // why the program was not lifted; empty on success
    std::vector<std::string> proof;       // what the lift relied on
    std::string listing;                  // the program with the typed value each write produced
    WaveOps wave;                         // on success
    // The engine's identities for the module's bindings and attributes, from
    // the shader container's reflection (name_lift); empty without one.
    ShaderNames names;
    bool ok() const { return rejections.empty() && !spirv.empty(); }
};

// Names a successful lift's bindings by the container's reflection: fills
// `lift.names` and puts the engine's names on the module's variables (OpName
// only - the module computes exactly what it did). `reference` is the
// translation the lift declares the bindings of.
void name_lift(LiftResult& lift, const TranslateResult& reference, const ShaderReflection& reflection);

// Lifts a pixel shader. `reference` is the translator's result for the same
// program and options, with cb_no_fallback set: the lifted module declares the
// reference's descriptor bindings, inputs and outputs, so it runs with the
// descriptor sets written for the reference pipeline.
LiftResult lift_pixel_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference);

// Lifts a vertex shader on Vulkan vertex input (TranslateOptions::vertex_input,
// no inlined fetch shader) under the same rules: inputs at s_swappc_b64, each
// element converted as the translator converts it - by its formats as
// constants, or with vertex_formats_from_params by its descriptor in the params
// block or the relink's specialization constant -, the position and params
// exported with EXEC known set, samples at level 0.
LiftResult lift_vertex_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference);

// Lifts a compute shader of a dispatch (TranslateOptions::stage Compute, no
// tessellation role) under the same rules, against the translation with the
// same options - the dispatch's, which binds no buffer: every buffer is read
// and written through the page table, as the lift then reads and writes it,
// so the lifted module runs with the descriptor set written for the
// reference. One invocation is a lane of GCN's wave: its workgroup id and
// local id as the reference places them in SGPRs and VGPRs, memory accesses
// as the reference makes them (stores, atomics and image writes where the
// invocation's EXEC bit is set), loops on EXEC (s_and_b64 exec + an
// s_cbranch_execz exit) as loops each invocation leaves when its EXEC bit
// clears. What needs the wave's lanes together - a GDS counter's value, a
// lane's rank (v_mbcnt) - stays a subgroup operation, as in the reference,
// and only where the lift's control flow is uniform. Workgroup memory (LDS)
// is not lifted. Trailing words after a final s_endpgm that no branch reaches
// (the eboot's own programs carry some) are not part of the program.
LiftResult lift_compute_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference);

// The stages of a tessellated draw on the host's tessellator, with the LDS the
// GCN stages pass control points through as stage inputs and outputs.
//
// Attributes (TranslateOptions::tess_lds_attributes): one control point a
// patch. Each 16 bytes of it are a uvec4 slot of `control_point` at location
// 0 - an LDS byte address `a` is slot (a / 16) % attr_vec4s, component
// (a / 4) % 4, as the translator maps it - written by the LS as the vertex
// stage's output and read by the domain shader as the evaluation stage's
// input (`control_point[patch vertex][slot]`).
//
// The reference is the translation with the same options and cb_no_fallback:
// the lifted module declares its descriptor bindings and stage interface, so
// it runs where the reference ran. Every LDS address is computed as the
// program computes it; one the lift knows is a slot of its own (constant), any
// other is indexed at run time by the same arithmetic.
struct TessLift {
    std::uint32_t attr_vec4s = 16;  // TranslateOptions::tess_attr_vec4s
    // Window (TranslateOptions::tess_window): the game's own hull, a patch of
    // `control_points` whose LDS the host lays out as one window - the input
    // control points (ls_stride bytes each), the output ones (cp_stride bytes
    // each), the patch constants (patch_constants x 16 bytes) - and gives the
    // stages tessellation constants that say so (`constants`). On stage I/O:
    //   the LS writes its control point, `control_point` uvec4[ls_stride / 16]
    //   at location 0 (lift_ls_shader with attr_vec4s = ls_stride / 16);
    //   the hull (lift_hull_shader) reads the patch's as control_point[n][..],
    //   writes its own output control point to control_point_out[n][cp_stride
    //   / 16] at location 0 and the patch constants to patch_constant[npc] at
    //   location cp_stride / 16, and its factors to the tessellation levels;
    //   the domain shader reads those as its inputs.
    // An LDS address the lift knows (the constants folded, the control point
    // the invocation's own) is a slot of its own; a read it does not know is
    // decoded at run time, region by region, as the window lays it out (0 past
    // the patch constants); a write it does not know is rejected.
    bool window = false;
    std::uint32_t control_points = 1, ls_stride = 0, cp_stride = 0, patch_constants = 0;
    // The tessellation constants the reference reads and the binding it reads
    // them through (BufferBinding::path.str(); empty: none). Loads from that
    // binding fold to these words; the host makes the binding hold them.
    std::string constants_path;
    std::uint32_t constants[12] = {};
    // The stage's own params block (StageParams, its user data in member 2)
    // at `params_binding` of its descriptor set (TranslateOptions::
    // descriptor_set), when it does not share the evaluation stage's - a
    // window's pipeline gives the LS and the hull a set of their own. False:
    // kBindingParams, with the user data where the reference reads it.
    bool own_params = false;
    std::uint32_t params_binding = 0;
};
// The LS as the pipeline's vertex stage (TessRole::LsVertex) on Vulkan vertex
// input: its LDS writes are the control point's slots.
LiftResult lift_ls_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference, const TessLift& tess);
// The domain shader as the evaluation stage (TessRole::DomainTes): v0/v1 the
// tessellation coordinate, v2 and v3 the patch (0: each invocation reads only
// its own patch's control point), its LDS reads the control point's slots,
// its exports a vertex shader's.
LiftResult lift_domain_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference, const TessLift& tess);
// The hull shader of a quad patch of one control point whose factors are
// constants (the game's particles): straight-line code that stores four outer
// and two inner factors, each a VGPR last written with an inline constant or
// a literal, and otherwise only computes and writes LDS - its patch
// constants, which the evaluation stage of this model never reads (it reads
// the control point the LS wrote, passed on unchanged). Its lift is the
// control stage those factors make (gcn::make_tess_constant_tcs, which the
// caller builds with VGT_HOS_MIN/MAX's clamp): the factors and the proof.
struct HullLift {
    std::vector<float> factors;  // as stored, in order: four outer, two inner (before VGT_HOS_MIN/MAX clamp them)
    std::vector<std::string> rejections, proof;
    bool ok() const { return rejections.empty() && factors.size() == 6; }
};
HullLift lift_constant_hull(const Program& program);
// The game's own hull shader as the control stage (TessRole::HullTcs with
// tess_window; TessLift::window): one invocation an output control point -
// v1 its index << 8, the patch alone in its window - its LDS traffic the
// window's stage I/O, its first factor stores the tessellation levels in the
// order the hardware reads them (four outer and two inner for a quad patch,
// three and one for a triangle), s_barrier a control barrier in uniform
// control flow, and its writes under EXEC made only where the invocation's
// EXEC bit is set.
LiftResult lift_hull_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference, const TessLift& tess);

// The stages of a draw that binds a geometry shader (the effects' GPU
// particles), as Vulkan's vertex and geometry stages. On GCN GX's VS is an
// export shader (ES) that writes each vertex to the ESGS ring, the GS reads a
// primitive's vertices there and writes the vertices it emits to the GSVS
// ring, and a copy shader the GS carries reads those back and exports them
// (position and params). On the host the ES is the vertex stage, writing its
// ring dwords as outputs, and the GS the geometry stage, reading them as
// inputs and writing what the copy shader would export:
//   ES (lift_es_shader, GsRole::EsVertex): buffer_store_dword* through the
//     global table's ESGS write ring (the V# s_load_dwordx4 reads at dword 8
//     of the table the user data's s[0:1] points at) at soffset + offset =
//     4 * k writes dword k of the vertex - `es_gs` uvec4[(es_dwords + 3) / 4]
//     at location 0, component k % 4 of slot k / 4. It exports nothing.
//   GS (lift_gs_shader, GsRole::Geometry, point primitives): buffer_load_dword
//     through the ESGS read ring (table dword 12) at the vertex's offset
//     (v0 << 2) and soffset + offset = 256 * k reads dword k of it
//     (`es_gs[0][k / 4]`); buffer_store_dword through the GSVS write ring
//     (table dword 16) at soffset + offset = 4 * (max_vertices * d + v) writes
//     dword d of the vertex in slot v. s_sendmsg GS emit (0x22) emits the
//     invocation's next vertex - slot n for its nth, as GCN's emit counter
//     has it - with the outputs `copy` makes of that slot's dwords; GS cut
//     (0x12) ends the strip; GS done (0x03) nothing. Both only where the
//     invocation's EXEC bit is set. One invocation is a lane of GCN's wave:
//     one primitive. Where every GSVS store is at a constant offset and the
//     lift shows each emit's slot (each earlier emit made by all of its
//     invocations or by none), the vertex is the stores' values themselves;
//     otherwise (a store at a VGPR offset: the program's own vertex counter)
//     the ring is an array of max_vertices * vertex_dwords words in the
//     invocation, written at the address each store computes and read at the
//     emit counter's slot, which the lift counts as GCN does. Either way the
//     emit takes its slot as it is then: GCN's copy shader reads it after the
//     GS, the same for a program that stores into no slot it has emitted.
// Both lift against the translation of the same program as a vertex stage,
// for their other bindings (the particles' buffers, the GS's constants);
// ring V#s and the vertex offset are never data (as_u and as_f refuse them).
struct GsCopyOut {
    std::uint8_t target = 12;  // 12 the position, 32 + n param n
    std::uint8_t comp = 0;     // 0-3
    bool constant = false;     // `value` the component's bits; otherwise GSVS dword `value` of the vertex
    std::uint32_t value = 0;
};
struct GsLift {
    std::uint32_t es_dwords = 0;     // the ES's ring vertex size in dwords (its Gnm structure's word 7)
    std::uint32_t max_vertices = 0;  // the GS's most vertices a primitive (VGT_GS_MAX_VERT_OUT, word 13)
    std::uint32_t vertex_dwords = 0;  // a GSVS vertex's dwords (the stream's export size, word 9)
    std::uint32_t out_prim = 2;      // VGT_GS_OUT_PRIM_TYPE (word 7): 1 line strip, 2 triangle strip
    std::vector<GsCopyOut> copy;     // from parse_gs_copy: every component it exports, in export order
};
// The copy shader's exports as GsCopyOut: buffer_load_dword through the GSVS
// read ring (table dword 32) at the vertex's offset (v0 << 2) and soffset +
// offset = 64 * max_vertices * d loads dword d, and each export's components
// name the registers loaded, or a constant moved into one (stream 0: the
// stream test's fall-through). False, with `why`, for any other shape.
bool parse_gs_copy(const Program& copy, std::uint32_t max_vertices, std::vector<GsCopyOut>& out, std::string& why);
LiftResult lift_es_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference, const GsLift& gs);
LiftResult lift_gs_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference, const GsLift& gs);

}  // namespace gcn
