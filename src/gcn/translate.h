// GCN (Sea Islands) → SPIR-V translation.
//
// Execution model: one SPIR-V invocation per GCN lane. Scalar state (SGPRs,
// SCC, M0) is replicated per invocation; EXEC and VCC are real 64-bit masks
// rebuilt with subgroup ballots, so scalar branches stay uniform. Control
// flow is a dispatcher loop over a switch on the block offset, which keeps
// every GCN control-flow graph structured without a structurizer. Memory is
// reached through physical storage buffer addresses (the whole guest dmem is
// one host-visible buffer); images and samplers resolve through descriptor
// bindings the host fills from the same T#/S# the shader would have loaded.
#pragma once

#include "gcn/isa.h"
#include "gcn/wave.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gcn {

enum class Stage { Vertex, Pixel, Compute, TessControl, TessEval };

// How a shader reaches a resource: a user-data SGPR pair, optionally followed
// by loads (dword offsets) through memory. The host walks the same chain.
struct ResourceStep {
    std::uint32_t offset_dw = 0;  // dword offset added to the pointer
    bool vsharp = false;          // the pointer is the base address of a V# (s_buffer_load) rather than a raw pointer
    bool operator==(const ResourceStep& o) const { return offset_dw == o.offset_dw && vsharp == o.vsharp; }
};
struct ResourcePath {
    int user_sgpr = -1;               // first user SGPR of the pointer/resource
    std::vector<ResourceStep> loads;  // successive dereferences
    int final_offset_dw = 0;          // dword offset of the resource words in the last block
    bool final_vsharp = false;        // last dereference goes through a V# base
    bool immediate = false;           // resource words live directly in user SGPRs
    std::string str() const;
};

struct ImageBinding {
    ResourcePath path;
    int sgpr = 0;            // T# base SGPR at the use site
    std::uint32_t binding = 0;
    bool storage = false;
    bool r128 = false;       // MIMG r128: 4-dword V# in a T# slot
    bool da = false;         // MIMG DA: array instruction (shadPS4 ImageViewInfo is_array)
    std::uint32_t dim = 1;   // spv::Dim
    bool arrayed = false;
    bool depth = false;      // used with a comparison
    // A GCN cube T#: declared and bound as a 2D array of six faces per cube
    // (dim 2D, arrayed). Samples address it by (s, t, face) with s and t in
    // [1, 2] and 8 added to the face per cube; the translator converts.
    bool cube = false;
    // What its texels read as: 0 float, 1 unsigned, 2 signed integer (from
    // TranslateOptions::image_dims). The image is declared with that sampled
    // type; an integer load returns the texel's integer in the VGPR, as GCN's
    // does, and an integer store writes the VGPR's bits.
    std::uint32_t kind = 0;
};
struct SamplerBinding {
    ResourcePath path;
    int sgpr = 0;
    std::uint32_t binding = 0;
    bool compare = false;  // used with depth-compare sampling (image_sample_c*)
};

// A constant buffer read through a storage-buffer binding: `path` gives the
// V#'s first two dwords (user data, or loaded from a table the user data
// points at); the shader reads dwords [0, max_dw) of the V#'s memory.
struct BufferBinding {
    ResourcePath path;
    std::uint32_t binding = 0;
    std::uint32_t max_dw = 0;
    bool pointer = false;  // path holds a 64-bit address (s_load_*), not a V# (s_buffer_load_*)
    // A load indexes it (tbuffer_load_format_* with idxen): the host binds the
    // V#'s records (count x stride) and passes cb_stride / cb_w3.
    bool indexed = false;
};

// One vertex element a vertex shader takes from Vulkan vertex input instead of
// its fetch shader. The fetch shader would have loaded it with
// buffer_load_format_* from a V# record; here its raw components arrive at
// `location` and the shader converts them the same way.
struct VertexElement {
    std::uint32_t location = 0;
    std::uint32_t vdata = 0;  // first VGPR written
    std::uint32_t count = 0;  // VGPRs written, 1-4
    std::uint32_t w3 = 0;     // the record's word 3: data format [18:15], number format [14:12], DST_SEL [11:0]
};

// TranslateOptions::vertex_formats_from_params: the loader's descriptor of an
// element's word 3 (StageParams::vertex_formats), or 0 when the loader does not
// serve it and the element keeps constant formats. Bits 0-2: the kind, as
// load_vertex_input converts it (1 the raw bits of a float, or of a 32-bit
// unorm or snorm; 2 the raw bits of an integer: uint, or a 32-bit sint; 3 unorm8;
// 4 sint16). Bits 4-6: the data format's component count. Bits 8-11: the
// components past it that read one. DST_SEL must select each of the format's
// components in place and 0 or 1 past them, as every element of a cold world
// load did; the loader stays small enough not to slow the driver's compile.
inline std::uint32_t vertex_format_descriptor(std::uint32_t w3) {
    const std::uint32_t dfmt = (w3 >> 15) & 0xf, nfmt = (w3 >> 12) & 7;
    std::uint32_t count = 0, bits = 0;
    switch (dfmt) {
    case 1: count = 1; bits = 8; break;
    case 2: count = 1; bits = 16; break;
    case 3: count = 2; bits = 8; break;
    case 4: count = 1; bits = 32; break;
    case 5: count = 2; bits = 16; break;
    case 10: count = 4; bits = 8; break;
    case 11: count = 2; bits = 32; break;
    case 12: count = 4; bits = 16; break;
    case 13: count = 3; bits = 32; break;
    case 14: count = 4; bits = 32; break;
    default: return 0;  // packed or unknown
    }
    std::uint32_t kind = 0;
    if (bits == 32 && (nfmt == 7 || nfmt == 0 || nfmt == 1)) kind = 1;
    else if (nfmt == 4 || (nfmt == 5 && bits == 32)) kind = 2;
    else if (nfmt == 0 && bits == 8) kind = 3;
    else if (nfmt == 5 && bits == 16) kind = 4;
    else return 0;
    std::uint32_t ones = 0;
    for (std::uint32_t k = 0; k < 4; ++k) {
        const std::uint32_t s3 = (w3 >> (3 * k)) & 7;
        if (k < count) {
            if (s3 != 4 + k) return 0;
        } else if (s3 == 1) {
            ones |= 1u << k;
        } else if (s3 != 0) {
            return 0;
        }
    }
    return kind | count << 4 | ones << 8;
}
// Location n's descriptor is also specialization constant kVertexFormatSpecId
// + n: 0, the default, reads StageParams::vertex_formats; a pipeline's key holds
// every element's word 3, so its optimized relink can give its own descriptors
// and the driver folds the conversions away.
constexpr std::uint32_t kVertexFormatSpecId = 64;
// In a no-fallback stage, the V# word 3 of buffers[n] read by index
// (StageParams::cb_w3[n]) is also specialization constant kCbW3SpecId + n: 0,
// the default, reads the params; an optimized relink gives the words its first
// draw had (render.cpp binds the fast-linked pipeline for a draw whose differ)
// and the driver folds the DST_SEL selects away.
constexpr std::uint32_t kCbW3SpecId = 80;
// The same buffers' strides in bytes (StageParams::cb_stride[n]), as
// specialization constant kCbStrideSpecId + n, 0 reading the params: the
// element index's multiply by a run-time stride (v_mul_lo_u32, a quarter-rate
// instruction on AMD) becomes one by a constant.
constexpr std::uint32_t kCbStrideSpecId = 96;
// Whether TranslateOptions::vertex_formats_from_params can serve these elements.
inline bool vertex_formats_from_params_supported(const std::vector<VertexElement>& elements) {
    for (const VertexElement& el : elements) {
        if (el.location >= 16 || !vertex_format_descriptor(el.w3)) return false;
    }
    return true;
}

struct TranslateOptions {
    Stage stage = Stage::Pixel;
    // Constant-buffer and table loads (s_buffer_load_dword* at a constant
    // offset from a V#, s_load_dword* at a constant offset from a pointer pair,
    // either traced to user data directly or through a table, whose registers
    // the program never defines as anything else) read a storage buffer the
    // host binds over that memory. The page-table walk stays as the fallback for
    // when the host could not bind it (StageParams::cb_valid).
    bool cb_ssbo = false;
    // Images and samplers from the renderer's global descriptor
    // arrays in set `bindless_set` (kBindlessImages, kBindlessStorageImages,
    // kBindlessSamplers), each binding's slot read from the params block
    // (StageParams::image_index / sampler_index), instead of a binding of
    // their own in the stage's set. Every image type is an alias of the same
    // array; the renderer puts a view of that type at the slot.
    bool bindless = false;
    std::uint32_t bindless_set = 3;
    std::vector<std::string> cb_ssbo_exclude;  // ResourcePath::str() of V#s that stay on the page-table path
    // The host binds every one of this shader's constant buffers: emit the
    // storage-buffer reads alone, without the page-table fallback (the walks
    // and the V# loads that only fed them become dead code).
    bool cb_no_fallback = false;
    // Track where the running lane's EXEC bit is provably set (it is at entry;
    // mov, and/or, whole-quad mode and save/restore can keep it) and emit plain
    // register writes, unguarded memory writes and no final kill there.
    bool exec_known = true;
    bool object_motion = false;
    std::uint32_t motion_location = 0;
    std::uint32_t rsrc1 = 0;         // SPI_SHADER_PGM_RSRC1 / COMPUTE_PGM_RSRC1
    std::uint32_t rsrc2 = 0;          // SPI_SHADER_PGM_RSRC2 / COMPUTE_PGM_RSRC2
    std::uint32_t ps_input_ena = 0;   // SPI_PS_INPUT_ENA
    std::uint32_t ps_flat_mask = 0;   // bit n: parameter n is flat shaded
    std::uint32_t vs_out_cntl = 0;    // PA_CL_VS_OUT_CNTL: which pos exports carry misc/clip vectors
    // Clip distances without clipping. The driver clips a primitive against
    // them into new triangles whose depth differs from the uncut one's by an
    // ulp, and a later pass of the same surface without them then fails its
    // LEQUAL test in bands (the water's second pass). With this host bit in
    // vs_out_cntl the vertex stage writes clip distances 0-3 to output
    // location kClipVaryingLocation (a disabled one reads 1), and a pixel
    // stage with ps_clip_discard = n discards where any of the n it reads
    // there is negative: the same pixels go, the triangle stays whole.
    // The location is the first one past every param the pair uses (a fixed
    // 31 took the fragment stage past maxFragmentInputComponents with its
    // built-ins: the validation layer's Location-06272): bits 26-30 of
    // vs_out_cntl (reserved on GFX7), and ps_clip_discard = n | location << 8.
    static constexpr std::uint32_t kVsOutCntlClipVarying = 1u << 31;
    static constexpr int kVsOutCntlClipLocationShift = 26;
    static std::uint32_t vs_clip_location(std::uint32_t cntl) { return (cntl >> kVsOutCntlClipLocationShift) & 31; }
    std::uint32_t ps_clip_discard = 0;
    std::uint32_t ps_clip_count() const { return ps_clip_discard & 0xff; }
    std::uint32_t ps_clip_location() const { return (ps_clip_discard >> 8) & 31; }
    std::vector<std::uint8_t> ps_input_map;  // PS attr n -> VS param location (SPI_PS_INPUT_CNTL_n.OFFSET); empty = identity
    // Vertex stage with link_outputs: output location k carries param register
    // output_links[k] (kNoLink: nothing), so a pixel shader reads its input k at
    // location k whatever vertex shader it runs with, and the pairing
    // (SPI_PS_INPUT_CNTL) belongs to the vertex shader. Without it, param n is
    // written at location n.
    static constexpr std::uint8_t kNoLink = 0xff;
    bool link_outputs = false;
    std::vector<std::uint8_t> output_links;
    int debug_ps = 0;                        // 1: PS exports write magenta; 2: exports write the last sampled texel
    std::uint32_t cs_threads[3] = {64, 1, 1};
    // Workgroup memory the device allows. GCN gives each workgroup up to 64 KiB
    // of LDS, more than Vulkan devices have to offer, so the array we declare is
    // the size COMPUTE_PGM_RSRC2 asks for, capped here.
    std::uint32_t max_lds_bytes = 32 * 1024;
    const Program* fetch = nullptr;   // VS fetch shader, inlined at s_swappc_b64
    // Vertex stage without `fetch`: s_swappc_b64 loads these elements from
    // vertex input where it would have called the fetch shader.
    std::vector<VertexElement> vertex_input;
    // With vertex_input: each element's conversion comes from
    // StageParams::vertex_formats[location] (vertex_format_descriptor) at run
    // time instead of VertexElement::w3, so one translation serves every vertex
    // format with the same elements (per-shader compiles, step 6c). The elements
    // must be vertex_formats_from_params_supported.
    bool vertex_formats_from_params = false;
    // Diagnostics: emit the program-counter dispatcher even where straight-line
    // or forward-only code would do (the shape a shader with an inlined fetch
    // shader has).
    bool force_dispatcher = false;
    // Vertex stage: declare the position Invariant, so every pipeline computes
    // it identically from the same inputs (multipass depth tests rely on it).
    bool invariant_position = false;
    std::uint32_t descriptor_set = 0;
    // Where constant-buffer bindings go (cb_ssbo): their descriptor set (-1:
    // descriptor_set) and the first binding. The host puts them in a
    // push-descriptor set of their own, the vertex stage's from binding 112
    // and the pixel stage's from 128.
    int cb_descriptor_set = -1;
    std::uint32_t cb_binding_base = 112;  // kBindingStorageBuffer0
    // Image dimensions per T# use site, filled by the host from the resolved
    // T#s (spv::Dim, arrayed). Missing entries default to 2D. Bits 8-9 of the
    // dim are the texel kind (ImageBinding::kind), 0 when not given.
    std::vector<std::pair<std::uint32_t, bool>> image_dims;
    // S# force_unnormalized, in SamplerBinding order; part of pipeline identity.
    std::vector<bool> sampler_force_unnormalized;
    // Pixel stage: declare EarlyFragmentTests (the pipeline writes neither
    // depth nor stencil). Ignored when the shader exports depth.
    bool early_fragment_tests = false;
    // Tessellation, emulated. The
    // game's particles are patches of one control point that the tessellator
    // turns into quads. The LS runs as a compute stage over the draw's
    // control points and the domain shader as the vertex stage of a triangle
    // draw; the LDS the two share is a buffer at StageParams::lds_address, one
    // flat region for the whole draw, with the relative vertex and patch
    // indices made absolute so every patch has a place of its own.
    //   LsCompute: v0 the control point's vertex id (the index buffer's, or
    //   base vertex + index), v1 its index in the draw (StageParams::tess);
    //   invocations past the count repeat the last control point into slack
    //   slots past it.
    //   Domain: vertex k of patch p's grid of domain_level x domain_level
    //   quads (6 vertices each): v0/v1 the domain coordinates u, v, v2 and v3
    //   the patch.
    //
    // The host's own tessellator takes the same patches instead, and
    // then only the topology changes hands: the LS still runs as the compute
    // stage into the same flat buffer, the hull shader becomes the
    // tessellation control stage (HullTcs) and the domain shader the
    // evaluation stage (DomainTes), which reads that buffer by gl_PrimitiveID
    // exactly as Domain read it by an index it had to divide out.
    //   HullTcs: one output control point, the six buffer stores that would
    //   have gone to the tessellation-factor ring written to gl_TessLevel*
    //   instead, in the order the hardware reads them (four outer, two inner).
    //   DomainTes: v0/v1 the domain coordinates from gl_TessCoord, v2 and v3
    //   the patch from gl_PrimitiveID.
    //   LsVertex: the same LS, run as a vertex stage with the rasterizer
    //   discarded, so it writes the same buffer from inside the render pass -
    //   a compute dispatch cannot, and ending the pass for one is what a
    //   tessellated draw was paying for. One vertex a control
    //   point, so gl_VertexIndex is what the workgroup arithmetic gave.
    enum class TessRole : std::uint8_t { None, LsCompute, Domain, HullTcs, DomainTes, LsVertex };
    TessRole tess_role = TessRole::None;
    std::uint32_t domain_level = 1;
    // VGT_LS_HS_CONFIG and VGT_TF_PARAM, for the host stages.
    std::uint32_t tess_patch_control_points = 1;
    // The control points travel through the pipeline as stage
    // attributes instead of a buffer standing in for LDS, so the LS is the
    // pipeline's own vertex stage and there is no compute pass to end the
    // render pass for. One uvec4 a location, ceil(ls_stride / 16) of them; the
    // LS's stores and the domain shader's loads land on the same slot because
    // both address a patch at a flat `patch * stride + offset` and the patch
    // term is zero once each stage handles only its own.
    bool tess_lds_attributes = false;
    std::uint32_t tess_attr_vec4s = 8;
    // A hull shader that is the game's own (HullTcs; the Forbidden Woods'
    // meshes: three control points, factors it works out) addresses LDS
    // through a threadgroup's layout - inputs, then outputs, then patch
    // constants, each `num_patches` long - with a relative patch index of
    // eight bits. With tess_window set every patch gets a window of that many
    // bytes to itself instead: the LDS base moves to `patch * tess_window`,
    // the relative patch is 0, and the tessellation constants the host gives
    // these shaders describe one patch (num_patches 1). LsCompute then writes
    // control point i into window i / tess_patch_control_points at slot
    // i % that, HullTcs runs with v1 = the control point << 8 and reads its
    // user data from StageParams::vertex_formats (set 0 is the evaluation
    // stage's), and DomainTes reads the window of gl_PrimitiveID. No access
    // leaves its window: an offset past it takes the window's last dword.
    std::uint32_t tess_window = 0;
    // The LDS buffer is reached through a raw device address, which nothing
    // bounds: a ds_read whose address is garbage - a lane EXEC has switched
    // off (the translation runs reads for every lane), or an offset worked
    // out of a value that went wrong - reads wherever that lands. The GCN LDS
    // returns 0 for a read past the wave's allocation and drops such a
    // write; NVIDIA's driver mostly reads some other allocation there, AMD's
    // faults and loses the device (an RX 7600 XT in the Forbidden Woods'
    // hull draws). With this every access is checked against its window
    // (tess_window), or against StageParams::lds_bytes without one, and the
    // window's patch is held inside lds_bytes: past the end a read is 0 and
    // a write is dropped, as on the hardware. The host turns it off with
    // BBHOST_TESS_LDS_BOUND=0.
    bool tess_lds_bound = true;
    bool tess_quads = true;            // VGT_TF_PARAM type 2; triangles otherwise
    int tess_spacing = 0;              // 0 integer, 1 fractional-odd, 2 fractional-even
    bool tess_cw = true;               // topology 2 (cw) against 3 (ccw)
    // Drop a domain vertex that lands at or behind the near plane: a particle
    // the camera is inside reaches behind it, and the clipper stretches what
    // is left of the quad across the frame.
    float tess_near_cull = 0.0f;  // clip w below this: drop the vertex
    // Leave out a whole patch whose centre is at or behind the camera plane
    // (StageParams::patch_cull_w and the layout beside it): a camera-facing
    // quad there has no projection, and the clipper turns what is left of it
    // into a wedge over the frame. The game's own shaders fade those out
    // where they care - not all of them do.
    bool tess_patch_cull = false;
    bool tess_patch_cull_all = false;  // a check: leave every patch out
};

struct TranslateResult {
    std::vector<std::uint32_t> spirv;
    bool bindless = false;  // TranslateOptions::bindless: images and samplers take no binding in the stage's set
    std::vector<ImageBinding> images;
    std::vector<SamplerBinding> samplers;
    std::vector<BufferBinding> buffers;  // TranslateOptions::cb_ssbo
    // Instruction offset -> index into images / samplers / buffers: the binding
    // each image instruction and each storage-buffer load resolved to. The
    // lifter (gcn/lift.h) binds the same descriptors from these.
    std::map<std::uint32_t, std::uint32_t> image_at, sampler_at, buffer_at;
    std::vector<std::string> errors;     // unsupported instructions etc.
    std::uint32_t lds_bytes = 0;
    // Buffers the program writes (buffer and tbuffer stores, atomics): each
    // V#'s path, so the host can tell its caches of that memory that the
    // shader changed it. untraced_stores counts store sites whose V# does not
    // trace to user data.
    std::vector<ResourcePath> store_buffers;
    std::uint32_t untraced_stores = 0;
    std::uint32_t unnormalized_samples = 0;  // MIMG unorm=1: texel coordinates
    // Page-table walks the shader keeps, by reason (a load's instruction and
    // why it is not a storage-buffer read): sites in the program.
    std::map<std::string, std::uint32_t> walks;
    // Vertex-shader interface, for the rect-list geometry shader.
    std::vector<std::uint32_t> vs_params;  // output locations the params are written at
    std::vector<std::uint32_t> ps_inputs;  // pixel stage: the inputs the shader reads (input k at location k without ps_input_map)
    int vs_clip_count = 0;
    bool vs_point_size = false;
    // Why the program needs the 64-lane subgroup GCN wrote it for (gcn/wave.h
    // kWave64*): 0 = a pixel shader that computes the same in a 32-lane one,
    // kWave64NotPixel for every other stage (not analysed) - and for a result
    // translate() did not fill in, so nothing reaches wave32 unexamined. The
    // renderer runs the fragment stage at wave32 when it is 0 (gcn/wave.h). A
    // typed lift (gcn/lift.h) has no cross-lane
    // operation at all: the renderer clears it for one.
    std::uint32_t wave64_needs = kWave64NotPixel;
    bool ok() const { return errors.empty(); }
};

// Geometry shader that turns each input triangle into the axis-aligned
// rectangle GCN's RECTLIST primitive describes: the fourth vertex is
// v1 + v2 - v0 for position and every varying.
std::vector<std::uint32_t> make_rect_geometry_shader(const std::vector<std::uint32_t>& param_locations, int clip_count,
                                                     bool point_size);

// A T#'s dimensions predicted from the program alone (image_dims.cpp), in the
// form TranslateOptions::image_dims takes: dim as spv::Dim with 3 for a cube,
// arrayed. `conflict`: the T# is sampled both as a cube and not.
struct PredictedImage {
    std::uint32_t dim = 1;
    bool arrayed = false;
    bool conflict = false;
};
// One entry per paths.images, from the image instructions paths.image_at maps.
std::vector<PredictedImage> predict_image_dims(const Program& program, const TranslateResult& paths);

// Layout of the per-stage uniform block at binding 0 (std140). Guest virtual
// addresses translate through a two-level page table in device memory:
// l1[va >> 32] -> l2 table device address; l2[(va >> 16) & 0xffff] -> device
// address of that 64 KiB page (unmapped pages point at a sink page).
struct StageParams {
    std::uint64_t l1_table;       // device address of the 256-entry L1 table
    std::uint64_t lds_address;    // TranslateOptions::tess_role: the device address of the buffer standing in for LDS
    std::uint32_t user_sgpr[16];  // SPI_SHADER_USER_DATA_*
    std::uint32_t cb_valid;       // bit n: buffers[n] is bound; otherwise the shader walks the page table
    // TranslateOptions::tess_lds_bound: the bytes from lds_address the stage
    // may touch (0: none - every read is 0).
    std::uint32_t lds_bytes;
    std::uint32_t pad[2];
    std::uint32_t cb_bias_dw[16];  // buffers[n]: dword index of the V# base inside its bound range
    std::uint32_t cb_stride[16];   // buffers[n] read by index (BufferBinding::indexed): the V#'s stride in bytes
    std::uint32_t cb_w3[16];       // the same: the V#'s word 3 (data and number format, DST_SEL)
    // TranslateOptions::vertex_formats_from_params: vertex_format_descriptor
    // per vertex-input location. TessRole::LsCompute (no vertex input) reads
    // [0..4] as the draw's control points instead (kTess*).
    std::uint32_t vertex_formats[16];
    // TessRole::Domain: the row of the domain shader's projection that gives
    // clip w, the patch stride and base its constants hold, and the near plane
    // - what the prologue needs to leave out a patch whose centre is at or
    // behind the camera plane, which no camera-facing quad can be drawn from.
    float patch_cull_w[4];
    std::uint32_t patch_stride, patch_base;
    float patch_cull_near;
    std::uint32_t patch_pad;
    // TranslateOptions::bindless: each image and sampler binding's slot in the
    // renderer's global descriptor arrays (images[n], samplers[n]).
    std::uint32_t image_index[16];
    std::uint32_t sampler_index[16];
    // TranslateOptions::object_motion (offset 512, std140)
    std::uint64_t motion_positions;       // device address of clip positions buffer (PSB)
    std::uint32_t motion_store;           // vertex index base for current frame write (0 = skip)
    std::uint32_t motion_load;            // vertex index base for previous frame read (0 = skip)
    std::uint32_t motion_vertices;        // vertex count for this draw range
    std::uint32_t motion_first_vertex;    // min index / first vertex to subtract
    std::uint32_t motion_instances;       // instance count
    std::uint32_t motion_first_instance;  // first instance
    std::uint32_t motion_flags;           // bit 0: store (1), bit 1: load (2)
    std::uint32_t motion_pad;
    float motion_scale[2];                // viewport half-width, -half-height
    std::uint32_t motion_reserved[4];     // pad to 576 bytes
};
static_assert(offsetof(StageParams, image_index) == 384 && offsetof(StageParams, motion_positions) == 512 && sizeof(StageParams) == 576, "the params block's layout is the shaders'");
// TessRole::LsCompute: StageParams::vertex_formats as the draw's control points.
constexpr int kTessIndexLo = 0, kTessIndexHi = 1;  // index buffer guest address (0: none)
constexpr int kTessCount = 2;                      // control points
constexpr int kTessIndexType = 3;                  // 0: 16-bit, 1: 32-bit indices
constexpr int kTessBaseVertex = 4;                 // VGT_INDX_OFFSET, added to every vertex id
constexpr int kTessInstances = 5;                  // instances of the draw (0 reads as 1): kTessCount points each
constexpr int kTessFirstInstance = 6;              // the first of them (a draw split to fit the ring)
constexpr int kTessFirstPoint = 7;                 // the first point of each instance (a draw split by patches)
constexpr std::uint32_t kPageShift = 16;
// Direct memory is imported into the device in 1 GiB chunks.
constexpr std::uint32_t kDmemChunkShift = 30;
constexpr std::uint32_t kDmemChunks = 8;
constexpr std::uint32_t kL1Entries = 256;
constexpr std::uint32_t kL2Entries = 1u << 16;
constexpr std::uint32_t kBindingParams = 0;
// TranslateOptions::bindless: the bindings of the global set.
constexpr std::uint32_t kBindlessImages = 0, kBindlessStorageImages = 1, kBindlessSamplers = 2;
constexpr std::uint32_t kBindingImage0 = 16;
constexpr std::uint32_t kBindingSampler0 = 48;
constexpr std::uint32_t kBindingStorageImage0 = 80;
constexpr std::uint32_t kBindingStorageBuffer0 = 112;  // TranslateOptions::cb_binding_base's default
constexpr std::uint32_t kMaxBuffers = 16;  // indexed constant arrays take their own bindings

TranslateResult translate(const Program& program, const TranslateOptions& options);

// Whether the device takes a sample's texel offset from a register - a
// run-time Offset image operand, which Vulkan allows on sample instructions
// only with maintenance8's feature (set by the host, off until it is). Off,
// the translator and the lifter shift the coordinates by the offset instead:
// the game's offset samples are all image_sample_lz_o, at level 0, where that
// is the same texel (a gather keeps its Offset: shaderImageGatherExtended).
void set_runtime_sample_offsets(bool on);
bool runtime_sample_offsets();

// Every process-wide switch a translation reads besides its TranslateOptions -
// the device features the host sets above (half.h's too) and the BBHOST_*
// variables the translator looks at - as one string. The host's translation
// cache (host/translation_cache.cpp) keys each entry on it, so a run with
// another setting never takes a translation made under this one: a switch
// added to the translator belongs here.
std::string translation_switches();

// The two stages of a tessellated draw that carry no GCN code.
//
// The game's patches are one control point each, and the hull shader behind
// them only writes its factors - the domain shader never reads the patch
// constants back, which is why the emulated path could leave the hull stage
// out entirely and still draw them right. So the vertex stage has nothing to
// carry (the control points are already in the buffer the LS compute pass
// wrote, which the evaluation stage reads by patch), and the control stage has
// only the factors to write.
std::vector<std::uint32_t> make_tess_passthrough_vs();
// `attr_vec4s` non-zero: the control point arrives as stage attributes and
// the control stage passes it through.
std::vector<std::uint32_t> make_tess_constant_tcs(const float outer[4], const float inner[2], std::uint32_t control_points,
                                                 std::uint32_t attr_vec4s = 0);

}  // namespace gcn
