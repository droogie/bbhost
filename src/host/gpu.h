#pragma once

// Host GPU executor: runs translated GCN shaders on Vulkan. Works without a
// window (headless runs execute compute jobs and draws too). Guest direct
// memory is imported into Vulkan through host-pointer import, so shaders
// read and write the same pages the guest sees.

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <string>

struct GxStageRecords;
struct GpuDispatch {
    std::uint64_t code_va = 0;      // guest VA of the shader program (COMPUTE_PGM_LO/HI)
    std::uint32_t rsrc1 = 0;        // COMPUTE_PGM_RSRC1
    std::uint32_t rsrc2 = 0;        // COMPUTE_PGM_RSRC2
    std::uint32_t threads[3] = {1, 1, 1};  // COMPUTE_NUM_THREAD_X/Y/Z
    std::uint32_t user_data[16] = {};      // COMPUTE_USER_DATA_0..15
    std::uint32_t dim[3] = {1, 1, 1};      // DISPATCH_DIRECT workgroup counts
    std::uint64_t indirect_va = 0;         // DISPATCH_INDIRECT: GPU reads {x,y,z} here
    // DB_DEPTH_CLEAR / DB_STENCIL_CLEAR as the register file holds them at the
    // dispatch: a fill of a depth target's HTILE is the engine's fast clear,
    // and the clear values belong to that fill, not to the draw that later
    // binds the target (a draw from a token reads no register).
    std::uint32_t depth_clear = 0, stencil_clear = 0;
    // The counters its ds_append / ds_consume in GDS count in (gcn::
    // TranslateOptions::gds_counters): for GDS dwords gds_base.., the guest
    // word each one is (0: none) and what it starts the dispatch from. GX
    // keeps an append/consume buffer's counter in the word past its records
    // and moves it into GDS before the dispatch and back after (DMA_DATA,
    // gx_commit_compute_stage / gx_dispatch_post) - the initial count of a
    // D3D11 UAV, kept or set. The dispatch counts in the word itself; a set
    // count is written into it first, a count from another word copied in.
    enum GdsStart : std::uint8_t { kGdsKeep, kGdsSet, kGdsCopy };
    struct GdsCounter {
        std::uint64_t word = 0;
        std::uint64_t from = 0;    // kGdsCopy
        std::uint32_t value = 0;   // kGdsSet
        GdsStart start = kGdsKeep;
        std::uint32_t buffer_id = 0;  // the GX id of the buffer the word belongs to, from a token (0: not known)
    };
    static constexpr std::uint32_t kGdsCounters = 8;  // gcn::kGdsCounters
    std::uint32_t gds_base = 0;
    GdsCounter gds[kGdsCounters] = {};
    // The counters came with the dispatch token, from the engine's objects
    // (gx_trace.cpp take_cs_counters: each UAV's word and initial count), not
    // from the DMA_DATA moves through GDS around the dispatch packet.
    bool gds_from_token = false;
    // A dispatch token from GX's compute commit: the compute stage's records,
    // taken at the call from the context's pending state (gx_trace.cpp
    // snapshot_cs_records). The user data the descriptors name is built from
    // them when the dispatch runs (the rest stays user_data's); null for a
    // dispatch from its packet or from another caller.
    const GxStageRecords* cs_records = nullptr;
    // The dispatch-token shadow: work out which user-data dwords the program
    // reads (host_gpu_dispatch_user_reads), to compare only those.
    bool want_user_reads = false;
};

// Everything a draw is decoded from, in register encodings. The PM4 path
// copies it out of the command processor's register file; the GX backend
// (BBHOST_GX_BACKEND, src/hle/gx_trace.cpp) builds it from
// the GX objects the engine's draw call used.
struct GpuDrawInputs {
    std::uint32_t vs_pgm[4] = {};        // SPI_SHADER_PGM_LO/HI/RSRC1/RSRC2_VS
    std::uint32_t ps_pgm[4] = {};        // SPI_SHADER_PGM_LO/HI/RSRC1/RSRC2_PS
    std::uint32_t vs_user[16] = {};      // SPI_SHADER_USER_DATA_VS
    std::uint32_t ps_user[16] = {};      // SPI_SHADER_USER_DATA_PS
    // A tessellated draw, from its GX objects rather than the
    // register file. `tess` says the rest of these are set; vs_pgm/vs_user are
    // then the domain shader's, which is the hardware's vertex stage.
    bool tess = false;
    std::uint32_t ls_pgm[4] = {}, hs_pgm[4] = {};
    std::uint32_t ls_user[16] = {}, hs_user[16] = {};
    std::uint32_t stages = 0;        // VGT_SHADER_STAGES_EN
    std::uint32_t ls_hs_config = 0;  // VGT_LS_HS_CONFIG
    std::uint32_t tf_param = 0;      // VGT_TF_PARAM
    std::uint32_t hos_max = 0, hos_min = 0;
    // A draw that binds a geometry shader (gcn/lift.h GsLift),
    // from its GX objects: `gs` says the rest of these are set. vs_* and the
    // PS linkage above stay the ES's - what the draw is drawn with where the
    // GS's pipeline is not made (today's VS/PS) - and these are the GS's: its
    // stage registers, its copy shader's program and PA_CL_VS_OUT_CNTL, the
    // PS inputs linked to the copy shader's outputs (as GX links them to a
    // VS's), its output primitive, GSVS vertex size and most vertices, and
    // the ES's ring vertex size. Its records are stage[kGxRecordGs].
    bool gs = false;
    std::uint32_t gs_pgm[4] = {};       // SPI_SHADER_PGM_LO/HI/RSRC1/RSRC2_GS
    std::uint32_t gs_copy_pgm[2] = {};  // the copy shader's SPI_SHADER_PGM_LO/HI_VS
    std::uint32_t gs_vs_out_cntl = 0;
    std::uint32_t gs_ps_input_cntl[32] = {};
    std::uint32_t gs_out_prim = 0, gs_vertex_dwords = 0, gs_max_vertices = 0, gs_es_dwords = 0;
    std::uint32_t prim = 0;              // VGT_PRIMITIVE_TYPE
    std::uint32_t target_mask = 0;       // CB_TARGET_MASK
    std::uint32_t cb_shader_mask = 0;    // CB_SHADER_MASK
    std::uint32_t ps_col_format = 0;     // SPI_SHADER_COL_FORMAT
    std::uint32_t color[8][5] = {};      // CB_COLORn BASE, PITCH, SLICE, VIEW, INFO
    // Each target's real extent from its view's texture
    // (engine/gx_resources.h gx_view_extent: (width << 16) | height, 0 when
    // not known), which the registers give only padded; and the depth view's.
    std::uint32_t color_extent[8] = {};
    std::uint32_t depth_extent = 0;
    // The generation of the resource each target's view was made of
    // (gx_view_gen; 0: not known - the view not seen made, or no view: a
    // packet draw, YEBIS's, Scaleform's): the image a draw finds at the
    // target's base is that resource's from the draw on (render.cpp
    // target_image). A session makes far fewer than 2^32 resources.
    std::uint32_t color_gen[8] = {};
    std::uint32_t depth_gen = 0;
    std::uint32_t blend[8] = {};         // CB_BLENDn_CONTROL
    float blend_const[4] = {};           // CB_BLEND_RED..ALPHA
    std::uint32_t depth_control = 0, stencil_control = 0, stencil_ref = 0, stencil_ref_bf = 0;
    float depth_bounds[2] = {0.0f, 1.0f};  // DB_DEPTH_BOUNDS_MIN/MAX
    std::uint32_t render_control = 0, depth_clear = 0, stencil_clear = 0;  // DB_RENDER_CONTROL, DB_DEPTH_CLEAR, DB_STENCIL_CLEAR
    std::uint32_t z_info = 0, stencil_info = 0, z_read_base = 0, depth_size = 0, htile_base = 0;
    std::uint32_t depth_view = 0;  // DB_DEPTH_VIEW: SLICE_START [10:0], SLICE_MAX [23:13]
    std::uint32_t su_sc_mode = 0, clip_cntl = 0, vte_cntl = 0;
    std::uint32_t ps_input_ena = 0, ps_in_control = 0, vs_out_cntl = 0;
    std::uint32_t ps_input_cntl[32] = {};  // SPI_PS_INPUT_CNTL_0..31
    std::uint32_t ps_input_count = 32;     // entries of ps_input_cntl that are set (GX: the PS's inputs)
    float vport[6] = {};                 // PA_CL_VPORT_XSCALE, XOFFSET, YSCALE, YOFFSET, ZSCALE, ZOFFSET
    std::uint32_t screen_scissor[2] = {};   // PA_SC_SCREEN_SCISSOR_TL/BR
    std::uint32_t generic_scissor[2] = {};  // PA_SC_GENERIC_SCISSOR_TL/BR
    std::uint32_t vport_scissor[2] = {};    // PA_SC_VPORT_SCISSOR_0_TL/BR
    std::uint32_t sc_mode_cntl_0 = 0;       // PA_SC_MODE_CNTL_0 (bit 1: VPORT_SCISSOR_ENABLE)
    std::int32_t base_vertex = 0;        // VGT_INDX_OFFSET
    // Every field above is set; the draw reads nothing from the
    // register file (a YEBIS token completed from the wrapper's state).
    bool complete = false;
    // The fields only the draw's programs decide - VTE control,
    // the VS output control, the PS input table and enables, the export mask -
    // hashed once on the recording thread (gx_trace.cpp); 0 when not. The
    // pipeline key takes it instead of hashing them again at every draw.
    std::uint64_t program_fp = 0;
};

// One draw as the command processor sees it: the register file plus the
// draw packet's own fields. Register arrays are indexed by (reg - base).
struct GpuDraw {
    const std::uint32_t* sh;        // 0x2C00-based SH registers (0x400 entries)
    const std::uint32_t* ctx;       // 0xA000-based context registers (0x1000 entries)
    const std::uint32_t* uconfig;   // 0xC000-based user-config registers (0x400 entries)
    std::uint32_t index_count = 0;
    std::uint32_t instance_count = 1;
    std::uint64_t index_va = 0;     // 0: non-indexed (DRAW_INDEX_AUTO)
    std::uint32_t index_type = 0;   // 0: 16-bit, 1: 32-bit
    std::uint64_t indirect_va = 0;  // DRAW(_INDEX)_INDIRECT: GPU reads args here
    // The same draw's inputs built from GX state, when the GX backend has them:
    // `gx_render` draws from them, `gx_compare` checks them against the
    // register file (both on with BBHOST_GX_BACKEND=1, compare only with =2).
    const GpuDrawInputs* gx = nullptr;
    bool gx_render = false;
    bool gx_compare = false;
    // The GX objects the draw was made with, when it came through a GX method.
    const struct GxDrawObjects* gx_objects = nullptr;
    // Drawn from a host-draw token (BBHOST_GX_NATIVE=2) rather than a draw packet.
    bool gx_token = false;
    const char* gx_token_kind = nullptr;  // which: "native", "scaleform", "yebis" (diagnostics)
    // `gx` supplies only the programs and their user data (YEBIS, whose commit
    // the host replaced); everything else still comes from the register file,
    // which its other packets wrote.
    bool gx_partial = false;
    // The command processor maintains the context register file (off by
    // default once every draw is a token); checks against it only then.
    bool registers = true;
};

// The GX objects behind one GX draw: context, pending-state block, the call
// site of the GX method (Binary Ninja address) and the six stage shader
// objects in D3D11 order (VS HS DS GS PS CS).
struct GxDrawObjects {
    std::uint64_t ctx = 0, state = 0, caller = 0;
    std::uint64_t call_flip = 0;  // the flip counter when the GX method was called
    std::uint64_t shader[6] = {};
    const struct GxDrawRecords* records = nullptr;  // sampled draws only (BBHOST_GX_BIND_EVERY)
    // The draw's blend, depth-stencil and rasterizer state as
    // engine/gx_state.h ids, taken at the call while the objects were alive;
    // 0 where the registry did not know the object.
    std::uint32_t blend_id = 0, depth_stencil_id = 0, raster_id = 0;
    // The vertex-stage program's object (the domain shader's in a
    // tessellated draw), the pixel shader's and the input layout's, as
    // engine/gx_resources.h ids (0: not known); the renderer finds programs
    // and fetch shaders by these instead of by code address.
    std::uint32_t vs_prog_id = 0, ps_prog_id = 0, il_id = 0;
    // Geometry from the call, for the immediate draws: count,
    // instances and, for indexed draws, the address of the first index
    // (INDEX_BASE + start index * index size, as the input-assembly flush and
    // DRAW_INDEX_OFFSET_2 give it, renamed buffers included). fetch_va is the
    // input layout's fetch shader.
    bool geometry = false, indexed = false, index_known = false;
    std::uint32_t count = 0, instances = 1, index_type = 0;
    std::uint64_t index_va = 0, fetch_va = 0;
    // An indirect draw: the arguments' address (the args
    // buffer's memory plus the call's offset), count and instances then 0.
    std::uint64_t indirect_va = 0;
    // The fetch shader's vertex table as the commit builds it (0x2acfcf0):
    // the VS user-data slot of its address (0xff: none) and the 16-byte V#
    // record per table slot (vtx_valid marks the slots built).
    std::uint8_t vtx_ud = 0xff;
    std::uint16_t vtx_valid = 0;
    std::uint32_t vtx_rec[16][4] = {};
    std::uint64_t vtx_obj[16] = {};  // the buffer object behind each table slot (its home address at +0x20)
    // The index buffer's object (its holder, as the pending state names it;
    // 0: none), and the GX upload heap's hand-outs so far at the call
    // (host/gxvk.h handout_seq: the census's stamp, 0 without it).
    std::uint64_t ib_obj = 0;
    std::uint64_t call_seq = 0;
    // GX on Vulkan (host/gxvk.h): the handle of each vertex slot's buffer and
    // of the index buffer, as their holders carried it at the call (0: none).
    std::uint32_t vtx_handle[16] = {};
    std::uint32_t ib_handle = 0;
};

// One stage's resource records as the GX commit wrote them for a draw, copied
// at its draw packet's emitter: the
// shader's user-data descriptors ({type, user-data slot, resource slot,
// extra} per dword) and the context's per-slot record arrays.
struct alignas(64) GxStageRecords {
    // What the constructor clears - the counts and the per-slot bits - shares
    // the first cache line. A record set is made for every GX draw on a
    // recording thread and read on the command processor's, so each line the
    // constructor touched had to come back from the CP's core first; spread
    // over the struct they cost ~0.6 us a draw (gx-cost "allocation").
    std::uint32_t ndesc = 0;
    // BBHOST_GX_COST: which constant buffers' contents were hashed at the call
    // (cb_hash).
    std::uint16_t cb_hashed = 0;
    // BBHOST_CB_RECORD_PROOF: which were copied at the call (cb_proof).
    std::uint16_t cb_proofed = 0;
    // The shader object's resource masks (+0x0, +0x8, +0x10), which give the
    // commit's tables their slot ranges: textures 0-63 and 64-127; sampler
    // caches in bits 0-15, constant buffers 16-35, samplers 36-51.
    std::uint64_t mask[3] = {};
    std::uint64_t obj_tex_set = 0;  // a bit per slot with an object
    std::uint32_t obj_smp_set = 0, obj_cb_set = 0;
    std::uint32_t obj_smp_default = 0;  // empty sampler slots: obj_smp holds the 16-byte cache default
    // The context's 0xc0-byte block (ctx+0xb2f0) a descriptor of
    // type 7 (a user-data pointer) or 0x14 (a pointer in the extended block)
    // hands the shader, copied at the call when the stage has one.
    bool has_ctx_block = false;
    // A compute stage's records (a dispatch token's, gx_trace.cpp
    // snapshot_cs_records): the 32-byte "sampler" records (types 2/3, the
    // tables of type 9) are its UAVs' (obj_smp, slots 0-7), the sampler
    // caches (type 4, tables of type 0xa) its samplers' 16-byte records in
    // smp_cache, the constant buffers its 16 slots - the shader's embedded
    // buffer at slot 15 - in cb (obj_cb_set's bits), and a type 6 descriptor
    // is the GDS counter range 0x00200020.
    bool compute = false;
    std::uint16_t cs_smp_set = 0;  // compute: the sampler-cache slots smp_cache holds
    // desc[0..ndesc) hashed at the copy, which keys the
    // renderer's binding plans (how each binding reads these records, worked
    // out once per shader rather than at every draw); 0 for no descriptors.
    std::uint64_t desc_fp = 0;
    // The arrays are not zero-initialized; the counts and bits above say what
    // is valid: desc up to ndesc, cb_hash by cb_hashed, the obj_ arrays by the
    // obj_ bits. The commit's arrays are filled in compare mode only.
    std::uint32_t desc[64];
    std::uint8_t tex[64 * 32];        // texture records, 32 bytes per resource slot
    std::uint8_t smp[16 * 32];        // sampler records, 32 bytes
    std::uint8_t smp_cache[16 * 16];  // 16-byte sampler cache
    std::uint8_t cb[20 * 16];         // constant-buffer records, 16 bytes
    std::uint8_t ext[64 * 4];         // the extended block (user-data slots 16+)
    std::uint8_t ctx_block[0xc0];     // see has_ctx_block
    std::uint64_t cb_hash[14];
    // The same records built from the pending-state slot objects, as a native
    // draw would without the commit: texture views' +0x10 records, sampler
    // objects' +0x0, constant-buffer records from the buffer object (address
    // through the dynamic-buffer lookup 0x2aae840, +0x2c size, 0x1470960's
    // layout adjusted as sub_14710c0(record, 0x6e, 0, 0) does).
    std::uint8_t obj_tex[64 * 32];
    std::uint8_t obj_smp[16 * 32];
    std::uint8_t obj_cb[14 * 16];
    // The id of the shader-resource view behind each obj_tex
    // record (engine/gx_resources.h; 0 when not registered). A view's T# never
    // changes after it is made, so the id stands for the record.
    std::uint32_t obj_tex_id[64];
    // BBHOST_CB_RECORD_PROOF: each obj_cb buffer's copy taken at the call
    // (host/cb_proof.h handles), valid by cb_proofed.
    std::uint64_t cb_proof[14];
};
static_assert(offsetof(GxStageRecords, desc) <= 64, "what the constructor clears is one line");
struct GxDrawRecords {
    // [0] the stage that binds set 0, [1] the pixel stage, [2] the LS of a
    // tessellated draw. GX's stages are VS/HS/DS/GS/PS, which map onto the
    // pipeline as vertex/control/evaluation/-/fragment: for a tessellated draw
    // the *domain* shader is what binds set 0, so it goes in [0] and GX's VS -
    // the LS - goes in [2], where only its user data is wanted. It binds no
    // images, samplers or buffers in any run, and the control stage is
    // generated, so two sets still cover the pipeline.
    // [3] the hull shader of a tessellated draw: the Forbidden Woods' meshes
    // run the game's own hull (3 control points, computed factors), which
    // needs its user data - its tessellation constants - built like the LS's.
    // In a draw that binds a geometry shader (GpuDrawInputs::gs), which has
    // no hull, [3] is the GS's (kGxRecordGs) and [0] stays GX's VS's - the ES.
    GxStageRecords stage[4];
};
constexpr int kGxRecords = 4;
constexpr int kGxRecordSet0 = 0, kGxRecordPixel = 1, kGxRecordLs = 2, kGxRecordHs = 3, kGxRecordGs = 3;

// Creates the Vulkan instance/device. `instance_exts` are extra instance
// extensions (the window's surface extensions); `want_present` adds the
// swapchain extension. Safe to call more than once; later calls return the
// existing state. BBHOST_GPU=0 disables the executor.
bool host_gpu_init(const char* const* instance_exts, std::uint32_t n_exts, bool want_present);
// True when a Vulkan device with the required features is available;
// initialises it (headless) on first use.
bool host_gpu_available();
// The device was lost (VK_ERROR_DEVICE_LOST: the GPU hung and the system reset
// it). Nothing will be drawn again this run; the window's loop tells the
// player and closes.
bool host_gpu_device_lost();
// Whether the GPU has little memory for all the host keeps in it (gpu.cpp,
// memory_tight; a Steam Deck does). Callable before host_gpu_init: the guest
// patches size the render targets' heap by it.
bool host_gpu_memory_tight();
// Whether the GPU run on is a discrete one with little video memory free -
// under 7.5 GiB at the start (BBHOST_SMALL_VRAM=0/1 says otherwise) - so the
// small-video-memory settings apply. Before the device exists, as
// host_gpu_memory_tight.
bool host_gpu_small_vram();
// Whether the GPU run on is an integrated one (its video memory is the
// system's RAM). Before the device exists, as host_gpu_memory_tight.
bool host_gpu_integrated();
// And, integrated, whether it has 8 GiB of video memory or more (a Strix
// Halo's carve-out). Before the device exists, as host_gpu_memory_tight.
bool host_gpu_integrated_with_room();
// Queues one compute dispatch. Returns false when the shader could not be
// translated or queued (the caller logs and continues). Queued work runs at
// the next host_gpu_flush().
bool host_gpu_dispatch(const GpuDispatch& d);
// The user-data dwords (bit n: user SGPR n) the program of this thread's last
// host_gpu_dispatch touches, when it asked (GpuDispatch::want_user_reads) and
// its pipeline was found; 0xffff otherwise.
std::uint32_t host_gpu_dispatch_user_reads();
// Shadow (BBHOST_GX_NATIVE_DISPATCH=1): a dispatch token's compute records
// (GpuDispatch::cs_records) built into user data, against the register
// file's at the dispatch packet, over the registers its program reads
// (read_mask, render.cpp). Plain dwords compare directly; a table, the context
// block or the extended block by the contents its pointer reaches.
struct CsUserCompare {
    int result = 0;     // the builder's result (0: built)
    bool built = false;
    std::uint64_t dwords = 0, dwords_differ = 0, units = 0, units_differ = 0, unreadable = 0;
    int first_bad = -1;  // the first differing plain dword
    std::uint32_t bad_reg = 0, bad_built = 0;
    int first_unit_ud = -1;  // the first differing table slot or block dword: its pointer's slot
    std::uint32_t first_unit_index = 0;
    std::uint32_t unit_reg[2] = {}, unit_built[2] = {};
};
void host_gpu_compare_cs_user_data(const GxStageRecords& st, const std::uint32_t* reg, std::uint32_t read_mask, CsUserCompare& out);
// What the last host_gpu_dispatch on this thread was (BBHOST_CP_CENSUS): a
// fill or a render-target copy done without a shader, a shader the copy and
// fill detectors recognised but which still ran (an image or buffer-to-image
// copy, a fill or copy whose range was not a target's), any other shader, or
// a dispatch that failed. `program`, when given, gets the program's name.
enum GpuDispatchClass : std::uint8_t {
    kDcNone, kDcFill, kDcCopyImage, kDcImageBlit, kDcBufferToImage, kDcFillShader, kDcCopyShader, kDcShader, kDcFailed, kDcClasses
};
GpuDispatchClass host_gpu_last_dispatch_class(const char** program = nullptr);
// Queues one draw.
bool host_gpu_draw(const GpuDraw& d);
// BBHOST_RUN_COMMIT=on (host/list_resolve.h): a run's draws after its first,
// which the command processor has just drawn, committed in parts by the run
// workers and the command processor and spliced into the stream in their
// order (render.cpp run_commit). False when the run is not forked - its draws
// are then drawn as the walk reaches them; true when every one was committed,
// `failed` counting those host_gpu_draw would have called failed.
// `ids`: each draw's token, whose draw is prepared from its slot as the walk's
// take of it would prepare it (hle_gx_native_copy) where it is committed.
// 1: committed; 0: not, its draws drawn as the walk reaches them; -1: not,
// because the first draw took no answer a part could share (a tessellated or
// geometry-shader draw ahead of plain ones): the next draw may start a run.
int host_gpu_run_commit(const GpuDraw* draws, const std::uint64_t* ids, std::uint32_t n, std::uint32_t& failed);
// Runs queued work and waits for it; call before the command processor
// reads or writes memory the shaders may touch.
void host_gpu_flush();
// Exit-time statistics.
void host_gpu_report();
// Submits the recorded work without waiting (the CP does this at the end of
// every submission and before it polls memory).
void host_gpu_submit();
std::string host_gpu_profile_report();  // BBHOST_GPU_PROFILE=1: top pipelines by GPU time since the last call
std::string host_gpu_fill_stats();      // fill-pending / fill-applied counters
// Copies out of render targets: into images, and refused (copied as memory,
// the destination's image left stale), by reason. Empty without a GPU.
std::string host_gpu_target_copy_report();
// Writes the Vulkan pipeline cache under the data root when pipelines were
// created since the last save (the 300-flip report calls it); the write runs on
// a background thread. BBHOST_PIPELINE_CACHE=0 turns loading and saving off.
void host_gpu_save_pipeline_cache();  // periodic, in the background (gpu.cpp)
void host_gpu_save_pipeline_cache_at_exit();  // synchronous: the window closing, the game exiting
// Shader compiles since the last call, for the 300-flip report: compute
// pipelines optimized in the background and swapped in, translations taken
// from the translation cache or made (gpu.cpp); empty when there were none.
std::string host_gpu_compile_report();
// Guest-memory writes ordered after the queued shader work, executed by the
// GPU: label writes and CP DMA. Return false when the memory is not
// imported; the caller then flushes and writes on the host.
bool host_gpu_mem_write(std::uint64_t va, const void* data, std::size_t bytes);  // bytes % 4 == 0, <= 64 KiB
// host_gpu_mem_write that also reports the submission serial the write is recorded
// into: it is in memory once host_gpu_completed_submits() is past that serial.
bool host_gpu_mem_write_serial(std::uint64_t va, const void* data, std::size_t bytes, std::uint64_t* serial);
// The command processor's thread, for the writes it makes from here on: the
// chunk round count (host_lists.h round_seq_now) when the game submitted what
// it walks, and where it keeps the opcode of the packet it is in - a write
// that lands on memory the game has written again since is told apart
// (label_landing.h Hooks::reused). 0 / null: untagged.
void host_gpu_set_write_tag(std::uint64_t epoch, const std::uint32_t* op);
// The command processor's worker threads' jobs so far: table read masks done
// and queued (bb-tmask0-3), no-fallback pipeline variants built and queued
// (bb-lean0-1) - for frame statistics' detail (BBHOST_FRAME_DETAIL=1).
void host_render_worker_jobs(std::uint64_t out[4]);
// A CP write that must land after GPU work already recorded (what the CP writes
// itself after a wait it passed on a label that work is still to write): on the
// GPU's timeline in stream order whatever its size, not held back to the end of
// the command buffer as a small host_gpu_mem_write is, and seen by the reads
// made while recording. False - the caller waits for the GPU and writes on the
// CPU - when the range is not imported, is over 64 KiB, or may hold a texture or
// a render target (their caches follow the CPU's writes).
bool host_gpu_mem_write_ordered(std::uint64_t va, const void* data, std::size_t bytes, std::uint64_t* serial);
// Guest memory as the work being recorded now will see it (the CP writes still
// queued applied on top), and whether a GPU write recorded into a submission
// that has not run yet covers any of it.
void host_gpu_read_guest(std::uint64_t va, std::size_t bytes, void* out);
bool host_gpu_writes_pending(std::uint64_t va, std::size_t bytes);
// Submissions whose fence has retired, in order (trails the GPU: a write may be
// in memory before this passes its serial).
std::uint64_t host_gpu_completed_submits();
// The same without the GPU lock, as last stored (it may trail by a retirement):
// for a recording thread that must not wait on the renderer.
std::uint64_t host_gpu_completed_submits_seen();
// BBHOST_ARENA_WATCH: logs where va resolves and which recent deferred writes carried value.
void host_gpu_explain_value(std::uint64_t va, std::uint64_t value);
bool host_gpu_mem_fill(std::uint64_t va, std::uint32_t value, std::size_t bytes);
bool host_gpu_mem_copy(std::uint64_t dst, std::uint64_t src, std::size_t bytes);
// GDS counters (gcn::TranslateOptions::gds_counters; BBHOST_GDS_COUNTERS=0
// turns them off, and GDS is dropped as before).
bool host_gpu_gds_counters();
bool host_gpu_gds_subgroup();  // BBHOST_GDS_SUBGROUP: the lift's GDS counters a subgroup at a time (TranslateOptions::gds_subgroup)
// A copy out of or a fill of a counter word, on the GPU's timeline: the word
// is the GPU's, as GX's GDS was, and its value is the one the dispatches
// before this point in the stream left (D3D11's CopyStructureCount).
bool host_gpu_counter_copy(std::uint64_t dst, std::uint64_t src, std::size_t bytes);
bool host_gpu_counter_fill(std::uint64_t dst, std::uint32_t value, std::size_t bytes);
// A GX buffer was made over [va, va + bytes) with this D3D11 usage and bind
// flags: with window imports (BBHOST_IMPORT_WINDOWS=1), when the GPU will read
// it in place, its windows are imported ahead of their first use on a thread
// of their own; nothing otherwise. Any thread, any lock.
void host_gpu_import_prefetch(std::uint64_t va, std::uint64_t bytes, std::uint8_t usage, std::uint8_t bind);
// The memory counter's numbers (host setting memory_counter, host/window.cpp):
// `vram` the video memory the driver counts against bbhost's process (its
// budget heap's usage: our allocations and the driver's own), `cache` the part
// of ours the last video memory census found to be cache - given back when
// memory runs short (`cache_known` false before a census) - and `shared` the
// host memory our allocations hold that the GPU reads (the game's memory
// imported, upload staging), which Windows counts as shared GPU memory. On an
// integrated GPU Windows counts the video memory as shared too. False before
// the device is made.
struct GpuMemoryGlance {
    std::uint64_t vram = 0, cache = 0, shared = 0;
    bool cache_known = false, integrated = false;
};
bool host_gpu_memory_glance(GpuMemoryGlance& out);
// The video memory census run now, not logged (for the counter's cache).
void host_gpu_memory_census_refresh();
// The guest memory imports the command processor waited for so far - made at
// a binding, a device address or a walk, not by the import thread - and the
// driver's time they took (the frames and stall lines' "imports waited for").
void host_gpu_import_waits(std::uint64_t* n, std::uint64_t* us);
// The recorder's thread (bb-record) so far: nanoseconds replaying blocks into
// Vulkan, and spinning for the next block before it sleeps. False without the
// thread (recording in place or inline): the bottleneck meter's recorder load.
bool host_gpu_recorder_ns(std::uint64_t* replay_ns, std::uint64_t* spin_ns);
// A GX buffer was made over [memory, memory + bytes) with this usage, these
// bind flags and the registry's generation: with BBHOST_GX_TWINS=1 one the
// GPU writes (usage default, unordered-access bind) gets a device-local twin
// at its first GPU use (gpu.cpp, gx twins). Any thread, any lock.
void host_gpu_gx_buffer_made(std::uint64_t memory, std::uint64_t bytes, std::uint8_t usage, std::uint8_t bind, std::uint64_t gen);
// A Map of that buffer handed out its own memory: the CPU writes it in place
// (gx twins keep no device-local copy of it).
void host_gpu_gx_buffer_mapped_own(std::uint64_t memory, std::uint64_t bytes, std::uint64_t gen);
// That buffer was destroyed (gx twins drop its device-local copy).
void host_gpu_gx_buffer_gone(std::uint64_t memory, std::uint64_t gen);
// A GX buffer copy reached as a copy token, handled as the compute copy
// path handles a recognised copy shader: marks the destination's surfaces
// dirty; when the source is a render target, copies its image into the
// destination's; otherwise records buffer copies in command order, split where
// either range crosses imports. False when a piece cannot be located or source
// and destination share bytes.
// src_gen / dst_gen: the generations of the GX textures the copy moves whole
// (a CopyResource), 0 when it is not one: the source is then copied as its
// resource's image when the host holds one (textures_copy_resource_locked,
// the render-target path for a target), and the memory copy stays the
// fallback.
bool host_gpu_copy_guest(std::uint64_t dst, std::uint64_t src, std::size_t bytes, std::uint64_t src_gen = 0, std::uint64_t dst_gen = 0);
// A GX dynamic buffer's renamed bytes copied back to its own address (a copy
// token's batched copies): inside a render pass, the draws after it read a
// copy of the bytes taken now and the copy into place waits for the pass's
// end (BBHOST_COPY_VERSIONS=0: as host_gpu_copy_guest, which it falls back to).
bool host_gpu_copy_back(std::uint64_t dst, std::uint64_t src, std::size_t bytes);
// Copy-backs [0, n) (dst[i] <- src[i], bytes[i]), in order, under one hold of
// the renderer's lock, as long as each becomes a copy version (a zero-length
// one is passed over); how many it took. The caller copies the one it stops at
// on the GPU (host_gpu_copy_guest, as host_gpu_copy_back would). One lock for
// a token's copies: each release waited for the copy just made to leave the
// write-combining buffers of the staging memory.
std::size_t host_gpu_copy_backs(const std::uint64_t* dst, const std::uint64_t* src, const std::uint64_t* bytes, std::size_t n);
// A constant window a YEBIS draw reads, given at its token on the command
// processor's thread (engine/yebis_bind.cpp): written into guest memory by
// the CPU, and the next draw binds a copy of it taken as it records, so no
// draw still in flight sees the new bytes - instead of a GPU-side write,
// which ended the render pass and drained the pipeline around each one.
// False when the window must take the GPU-side write instead
// (BBHOST_YEBIS_WINDOWS=gpu, or a window that is not in mapped memory).
bool host_gpu_draw_window(std::uint64_t dst, const void* data, std::uint32_t bytes);
std::uint64_t host_gpu_guest_copy_targets();  // copy tokens handled as render-target image copies
// A DMA fill that covers a render target's memory: clears the image (in
// command order) and returns true; false when no target lives there.
bool host_gpu_clear_target(std::uint64_t va, std::uint32_t value, std::size_t bytes);
// A texture GX just created with initial data, as the T# GX
// built for it (base filled in): the image is created and its untile queued
// now, on the creating thread, so the first bind finds it. False when the
// cache already has it or could not resolve it.
bool host_gpu_texture_create_ahead(const std::uint32_t tsharp[8], std::uint64_t gen = 0);
// A fill of [va, va+bytes) with a 16-byte value, from a fill
// token at its place in the stream - what a recognised fill dispatch did.
bool host_gpu_fill(std::uint64_t va, std::size_t bytes, const float rgba[4], std::uint32_t depth_clear, std::uint32_t stencil_clear);
// A texture's subresource region updated with tightly packed
// rows (`row_bytes` each, one per block row), from an upload token: copied
// into the surface's image, which then holds what guest memory does not.
// `tsharp` (the resource's, base filled in) creates the surface when the
// cache has none yet; null when the registry has no T# for it.
bool host_gpu_upload_region(std::uint64_t base, const std::uint32_t* tsharp, std::uint32_t mip, std::uint32_t layer, std::uint32_t x,
                            std::uint32_t y, std::uint32_t w, std::uint32_t h, const void* data, std::size_t bytes, std::uint32_t row_bytes);
// A texture-to-texture copy from a copy-image token. Each side is
// a T# (the resource's, base filled in) with a subresource and an offset; the
// source may be a render target, which its T# then aliases.
struct GpuImageCopy {
    std::uint32_t src_tsharp[8], dst_tsharp[8];
    std::uint32_t src_mip, src_layer, src_x, src_y;
    std::uint32_t dst_mip, dst_layer, dst_x, dst_y;
    std::uint32_t w, h;
    // The two resources' generations at the call (0: not known): a side whose
    // image at its memory is another resource's is not used (textures.cpp).
    std::uint64_t src_gen = 0, dst_gen = 0;
};
bool host_gpu_copy_image_region(const GpuImageCopy& c);
// The memory [base, base+bytes) of a resource GX destroyed or the kernel
// unmapped: its surfaces are retired once the command processor has retired
// everything submitted up to `after_submitted` (hle_gnm_submitted_total()).
void host_gpu_texture_retire(std::uint64_t base, std::size_t bytes, std::uint64_t after_submitted, std::uint64_t gen = 0);
// GX mapped the texture whose memory is [base, base + bytes) for the CPU to
// write: its surfaces upload it again at their next use.
void host_gpu_texture_written(std::uint64_t base, std::size_t bytes);
// A CPU write GX makes over a buffer's memory - a creation's initial data -
// or memory that stops being a buffer's: the buffer shadow invalidates the
// mirror pages over it (they are not write-watched when GX buffers cover
// them). Any thread, no lock held.
void host_gpu_shadow_gx_write(std::uint64_t va, std::size_t bytes);
// A GX resource was made over [va, va + bytes): a write as above, and memory
// GX no longer hands out to be rewritten. Any thread, no lock held.
void host_gpu_shadow_gx_created(std::uint64_t va, std::size_t bytes);
// Memory GX handed the CPU to write without a call to say when it is done:
// the renamed copy a buffer Map hands out (kind 0, written until GX writes it
// back or frees it) or, under BBHOST_RESOURCE_CENSUS, an upload pool's block
// (1), a ring allocation (2) and the memory a YEBIS discard Map renames a
// resource into (3). Any thread, no lock held.
void host_gpu_shadow_gx_handed(std::uint64_t va, std::size_t bytes, int kind);
// Whether the buffer shadow acts on kind-0 hand-outs (BBHOST_SHADOW_GX and
// BBHOST_SHADOW_HANDED on): the registry then tells it of each allocation the
// ring allocator makes in place of the ring's whole renamed copy.
bool host_gpu_shadow_hands_out();
// GX writes a renamed copy in [va, va + bytes) back to its buffer (a map
// resolve's copy, gx_trace's 0x2ab2810 / 0x2ab2670 hooks) or frees it (a
// later discard Map of its buffer in the same context): the CPU's writes to
// it are over. Any thread, no lock held.
void host_gpu_shadow_gx_settled(std::uint64_t va, std::size_t bytes);
// Copies `bytes` into the host-owned table ring the GPU page
// table reaches and returns their guest address (0 when no ring could be
// mapped); a draw token's user data points at tables placed here.
// `align`: a power of two, at least 8 (a constant buffer's base wants 16).
std::uint64_t host_gpu_ring_place(const void* data, std::size_t bytes, std::size_t align = 8);
// The depth-stencil clear pass on a target without HTILE
// (its quad wrote the depth and stencil): the target whose depth surface
// begins at `base` clears at its next draw to the HTILE word's depth and the
// stencil, as an HTILE fill does for one with HTILE.
void host_gpu_depth_clear(std::uint64_t base, std::uint32_t word, std::uint32_t stencil);
// GX's ClearRenderTargetView or ClearDepthStencilView, in the view's own
// terms (a clear token, hle/gx_trace.cpp): the view's registry id and the
// target words a draw binding that view carries (GpuDrawInputs::color /
// z_info..htile_base), the view's extent, and the value. The command
// processor clears the image a draw binding the view would draw into, in
// stream order - D3D11's Clear*View, which Dark Souls III PC's GX calls.
struct GpuClearView {
    bool depth = false;           // a depth-stencil view
    std::uint32_t view_id = 0;    // gx_object_id of the view (0: not seen made)
    std::uint32_t extent = 0;     // gx_view_extent: (width << 16) | height, 0 when not known
    std::uint32_t gen = 0;        // gx_view_gen: the view's resource's generation, 0 when not known
    // Colour: CB_COLOR BASE, PITCH, SLICE, VIEW, INFO; the view's own mip,
    // first slice and slice count; whether CMASK fast clears it (CB_COLOR_INFO
    // FAST_CLEAR, where GX fills CMASK instead of the surface).
    std::uint32_t color[5] = {};
    std::uint32_t mip = 0, first_slice = 0, slices = 1;
    bool fast_clear = false;
    float rgba[4] = {};
    // Depth: DB_Z_INFO, DB_STENCIL_INFO, DB_Z_READ_BASE, DB_DEPTH_SIZE,
    // DB_DEPTH_VIEW, DB_HTILE_DATA_BASE; D3D11's flags (1 depth, 2 stencil);
    // the depth and stencil arguments; whether the view has HTILE
    // (DB_Z_INFO.TILE_SURFACE_ENABLE) and whether its HTILE words leave the
    // stencil state out (DB_STENCIL_INFO.TILE_STENCIL_DISABLE); the output
    // merger's depth and stencil clears (ctx+0xb650, +0xb654: what its commit
    // writes to DB_DEPTH_CLEAR / DB_STENCIL_CLEAR), from which GX's own pass
    // builds its HTILE word. render.cpp depth_clear_planes says what GX's
    // clear leaves from these.
    std::uint32_t z_info = 0, stencil_info = 0, z_read_base = 0, depth_size = 0, depth_view = 0, htile_base = 0;
    std::uint32_t flags = 0;
    float depth_value = 0.0f, om_depth = 0.0f;
    std::uint32_t stencil = 0, om_stencil = 0;
    bool htile = false, tile_stencil_off = false;
};
// What a clear (or, while BBHOST_GX_CLEAR_VIEWS=1 checks, the fill path's
// clear that follows it) did. `apply` false resolves and reports only.
struct GpuClearViewResult {
    const void* image = nullptr;  // the target image, null when none was made yet (apply false)
    std::uint32_t first_layer = 0, layers = 0;
    bool created = false;
};
bool host_gpu_clear_view(const GpuClearView& c, bool apply, GpuClearViewResult* out = nullptr);
// Whether the command processor can clear this view (its format has a host
// form, its words name memory); `why` says why not. Pure: callable at the call.
bool host_gpu_clear_view_resolvable(const GpuClearView& c, const char** why);
// BBHOST_GX_CLEAR_VIEWS=1: the fill tokens that follow a clear token (the
// method's own work) report what they cleared against what the token
// resolved, and the exit report counts agreements by kind.
void host_gpu_clear_view_check_begin(const GpuClearView& c, const GpuClearViewResult& expect);
void host_gpu_clear_view_check_end();
std::string host_gpu_clear_view_report();
// Cumulative counters for per-frame deltas in the flip log.
struct GpuStats {
    std::uint64_t draws, dispatches, flushes, gpu_us, pipelines, transfers, transfer_batches;
    // CP-thread wall time by phase, nanoseconds (see kPhaseNames).
    std::uint64_t phase_ns[8];
    // host_gpu_draw calls, failures, and calls that had nothing to draw.
    std::uint64_t draw_calls, draw_failures, draws_empty;
    // The draws by how their pixel shader ran (DrawPs), and the failures by
    // reason (host_gpu_draw_fail_name).
    std::uint64_t draws_ps[4];
    std::uint64_t draw_fail_why[16];
    // The same draws by how their vertex shader ran (DrawVs), and those the
    // lifter does not take (kDrawVsOther) by why (DrawVsWhy).
    std::uint64_t draws_vs[4];
    std::uint64_t draws_vs_why[16];
    // The draws on the host's tessellator (DrawTess): all of them, and those
    // whose evaluation, vertex and control stages ran lifted.
    std::uint64_t draws_tess[6];
    // Dispatches that ran a shader, and of them those that ran its typed lift.
    std::uint64_t shader_dispatches, lifted_dispatches;
    // Draws whose pixel shader discards by the vertex stage's clip distances at
    // entry (TranslateOptions::kVsOutCntlClipVarying).
    std::uint64_t draws_clip_discard;
    // BBHOST_OOB_COUNT=1 (gcn::set_oob_count): the counters' totals by kind
    // (gcn::kOobKinds, in its order) - typed loads by index past num_records,
    // past their binding, walks by index past num_records; image loads outside
    // the bound view's size, past its levels, past its layers; image stores
    // outside its size, past its layers, at a level other than 0; the view
    // smaller than the T#, larger; oob_on whether they count.
    std::uint64_t oob[11];
    bool oob_on;
};
// A draw on the host's tessellator: counted, and by which of its stages ran as
// typed lifts (BBHOST_LIFT_TESS) - the domain shader as the evaluation stage,
// the LS as the vertex stage, the hull shader as the control stage, all three;
// and the game's own hull draws drawn as native pipelines (no LS pass).
enum DrawTess { kDrawTess, kDrawTessDomainLifted, kDrawTessLsLifted, kDrawTessHullLifted, kDrawTessAllLifted, kDrawTessNative, kDrawTessCount };
// How a draw's pixel shader ran: its typed lift (BBHOST_DECOMP), a
// translation in the no-fallback pipeline (not lifted, or not asked to be),
// the page-table fallback pipeline's translation, or no pixel shader at all.
enum DrawPs { kDrawPsLifted, kDrawPsTranslated, kDrawPsFallback, kDrawPsNone, kDrawPsCount };
// How a draw's vertex shader ran: its typed lift (BBHOST_DECOMP), a
// translation on vertex input in the no-fallback pipeline (not lifted), a
// translation the lifter does not take (the fetch shader inlined, or a
// tessellation stage), or the page-table fallback pipeline's translation.
enum DrawVs { kDrawVsLifted, kDrawVsTranslated, kDrawVsOther, kDrawVsFallback, kDrawVsCount };
// Why a draw's vertex shader is one the lifter does not take (kDrawVsOther): a
// tessellation stage - the host tessellator's evaluation stage, or the
// emulated domain shader - or a vertex shader whose fetch shader stays inlined
// because the draw has no vertex input: it calls none, the draw carries no GX
// geometry (packets), vertex input is off, the game binds the vertex table
// itself (no slot in the draw's user data), the fetch shader has another shape
// than the GX input layouts', or an element has no record, no Vulkan vertex
// format, a stride shorter than itself, or data outside one imported buffer.
enum DrawVsWhy {
    kVsWhyTessHw, kVsWhyDomain, kVsWhyNoFetch, kVsWhyNoGx, kVsWhyInputOff, kVsWhyOwnTable,
    kVsWhyShape, kVsWhyRecord, kVsWhyFormat, kVsWhyStride, kVsWhyPlacement, kVsWhyCount
};
const char* host_gpu_vs_why_name(int why);
// A draw failure reason's name ("pixel shader"), or null past the last.
const char* host_gpu_draw_fail_name(int reason);
enum GpuPhase { kPhaseDraw, kPhaseDispatch, kPhasePrefetch, kPhasePipeline, kPhaseBind, kPhaseExec, kPhaseWait, kPhaseLock, kPhaseCount };
static const char* const kPhaseNames[8] = {"draw", "dispatch", "prefetch", "pipeline", "bind", "exec", "label-wait", "lock-wait"};
void host_gpu_phase_add(int phase, std::uint64_t ns);
// Scoped wall-clock accumulator for a phase. `weight` books each reading that
// many times over, so a caller can time one call in n; 0 reads no clock.
struct GpuPhaseTimer {
    int phase;
    std::uint32_t weight;
    std::chrono::steady_clock::time_point t0;
    explicit GpuPhaseTimer(int p, std::uint32_t w = 1)
        : phase(p), weight(w), t0(w ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    // Books the time so far to the current phase and starts `p`.
    void next(int p) {
        if (weight) {
            const auto t = std::chrono::steady_clock::now();
            host_gpu_phase_add(phase, weight * static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0).count()));
            t0 = t;
        }
        phase = p;
    }
    ~GpuPhaseTimer() { next(phase); }
};
// The draw path times one draw in this many (render.cpp): reading the clock at
// every draw cost ~250 ns of each - eight clock reads and five shared counters.
constexpr std::uint32_t kPhaseSample = 16;
GpuStats host_gpu_stats();

// Device handles for the presenter (valid after host_gpu_init succeeded).
struct GpuHandles {
    void* instance = nullptr;        // VkInstance
    void* physical = nullptr;        // VkPhysicalDevice
    void* device = nullptr;          // VkDevice
    void* queue = nullptr;           // VkQueue
    void* present_queue = nullptr;   // VkQueue for presents only (the same family), or null
    std::uint32_t family = 0;
    bool fifo_latest_ready = false;  // the device may present in VK_PRESENT_MODE_FIFO_LATEST_READY_KHR
};
GpuHandles host_gpu_handles();
// The command processor wrote guest memory on the CPU (DMA, WRITE_DATA):
// device-local copies of vertex and index data over it are stale.
void host_gpu_shadow_cp_write(std::uint64_t va, std::size_t bytes);
std::string host_gpu_shadow_report();
// The draw recorder thread's counts (recorder.cpp), for the same report.
std::string host_gpu_recorder_report();
// Occlusion (occlusion.cpp). A ZPASS_DONE dump (EVENT_WRITE PIXEL_PIPE_STAT_DUMP,
// index 1) at va: true when the real sample counters take it - written on the
// GPU's timeline - false for the caller's fake (BBHOST_OCCLUSION=0, memory
// not imported).
bool host_gpu_zpass_dump(std::uint64_t va);
// SET_PREDICATION: op 0 clears, 1 predicates the draws that follow on the
// ZPASS block at `block` (draw if visible, or if not), `cont` ORs it with the
// block before.
void host_gpu_set_predication(std::uint64_t block, unsigned op, bool draw_if_visible, bool cont);
std::string host_gpu_occlusion_report();  // the 300-flip line; empty when off
// The texture image heap's blocks and use (gpu.cpp).
std::string host_gpu_image_heap_report();
// The 5-second report's resource lines (host/textures.cpp
// textures_census_window_locked): the write watch's faults, arms and releases
// by owner, the texture surfaces by registration, the census of new ones.
void host_gpu_resources_report();
// Pixel shaders at wave32 (render.cpp): programs by subgroup size, when any
// were added since the last call; empty otherwise or with the feature off.
std::string host_gpu_ps_wave_report();
// A loading screen is up (engine/loading.cpp): the memory keeper makes the
// coming area's image-heap blocks and upload staging ready, and holds less
// once play resumes (gpu.cpp, start_memory_reserve).
void host_gpu_set_loading(bool loading);
// The first in-game frame (engine/loading.cpp): the shader compiles at the
// start go from half the hardware threads to a quarter (render.cpp, precompiler).
void host_gpu_world_reached();
// On a start that builds every shader again - the first one after an update
// that changed the shader translator, or the first ever - the compiles still
// queued or running on the precompile workers; 0 otherwise (the window's
// notice, host/window.cpp).
std::size_t host_gpu_shader_backlog();
// The setup window's "Compile shaders now" (host/shader_compile.h): a run
// with no game that starts the device and compiles every shader the game's
// bundles hold and the manifests name - translations, lifts, DXC's modules
// and the driver's pipelines - into the caches, on the same workers the game
// uses. Set before host_gpu_init.
void host_gpu_set_compile_session(bool on);
struct HostShaderCompileProgress {
    std::uint64_t total = 0, done = 0, failed = 0;
    std::uint64_t recipes_total = 0, recipes_done = 0;  // pipelines replayed from the manifest and the shipped list
    bool listed = false, finished = false;
    std::string reasons;  // one "why xN" a line
};
HostShaderCompileProgress host_gpu_compile_progress();
void host_gpu_compile_cancel();
// The device in use: "vendor:device:driver:pipelineCacheUUID name".
std::string host_gpu_device_identity();
std::string host_gpu_memory_budget_report();
// Where this process's video memory goes, by owner and kind, held against
// cached (gpu.cpp video_memory_report_locked): every 1,800 flips
// (BBHOST_VRAM_REPORT_FLIPS, gnm.cpp), and at exit with each owner's highest.
// Empty with BBHOST_VRAM_REPORT=0 or no GPU.
std::string host_gpu_video_memory_report(bool at_exit);
void host_gpu_log_video_memory(bool at_exit);  // the same, logged a line a call (indented but the exit's first)
// Serialises queue use between the presenter and the executor.
void host_gpu_lock();
void host_gpu_unlock();
// The renderer's queue is shared with its submission thread: the presenter
// (and anything else outside gpu.cpp) takes this around a direct submit,
// present or wait on it. It waits for the thread's pending submissions first,
// so what is submitted goes behind the game's work.
void host_gpu_queue_lock();
void host_gpu_queue_unlock();
// The presenter's blit through the submission thread, in order behind the
// game's work (VkCommandBuffer, VkSemaphore to wait on at `wait_stage`,
// VkSemaphore to signal, VkFence): its ticket, or 0 when the thread is off
// and the caller submits itself. The renderer's lock can go once it is
// queued; host_gpu_wait_submitted(ticket) before presenting what it renders.
std::uint64_t host_gpu_submit_presenter(void* cmd, void* wait, std::uint32_t wait_stage, void* signal, void* fence);
void host_gpu_queue_lock_only();  // the queue's mutex without waiting for the thread (after host_gpu_wait_submitted)
void host_gpu_wait_submitted(std::uint64_t ticket);
// The presenter's own queue (host/window.cpp, BBHOST_PRESENT_OWN_QUEUE): its
// copy of a frame goes on the second queue and waits there for the game's
// submissions up to that frame on `host_gpu_timeline()` (the VkSemaphore the
// game's submissions signal, serial + 1 each; null without one) - not behind
// everything the game queued since. It signals `host_gpu_present_timeline()`
// (null without one) with the value it registered, under the renderer's lock,
// with host_gpu_present_reads_locked: the slot being recorded then does not
// retire - free the images and descriptor sets the copy may read - before
// that value is reached.
void* host_gpu_timeline();
void* host_gpu_present_timeline();
void host_gpu_present_reads_locked(std::uint64_t target);
// How busy the GPU is with our work (gpu_busy.cpp): the union of the time
// our command buffers - the game's and the presenter's - spent on the GPU,
// from a timestamp at each one's start and end. Windows' per-process GPU
// engine counters (Task Manager's GPU column) see none of it on AMD's driver
// with hardware scheduling, for any program; this is the number to read
// instead. Cumulative since the start; callers difference two readings over
// their own wall-clock window. `on` false: off (BBHOST_GPU_BUSY=0) or the
// device has no timestamps on its queue.
struct GpuBusy {
    bool on = false;
    std::uint64_t busy_ns = 0;     // the union over both
    std::uint64_t game_ns = 0;     // the game's command buffers alone (their own union)
    std::uint64_t present_ns = 0;  // the presenter's alone
    std::uint64_t upload_ns = 0;   // the buffer shadow's uploads ahead of the game's command buffers
    std::uint64_t cmdbufs = 0, presents = 0;
};
GpuBusy host_gpu_busy();
// The presenter's command buffer (a VkCommandBuffer): its first command and
// its last (before vkEndCommandBuffer), and once its fence has signalled - or
// before the next one is recorded, which waits for it. Caller holds the lock
// for the first two.
void host_gpu_busy_present_begin(void* cmd);
void host_gpu_busy_present_end(void* cmd);
void host_gpu_busy_present_done();
// The 300-flip report's line: busy over the window since the last call.
std::string host_gpu_busy_report();
// The presenter's, before its blit: submits the renderer's recording only when
// the flip's own work is still in it - `need` is host_gpu_work_needs() taken
// when the flip arrived; ~0ull (not known) always submits. True when it did.
// Submitting whatever the command processor had recorded of the *next* frame
// broke its render pass and cost a submission on every presented frame.
bool host_gpu_submit_for_flip(std::uint64_t need);
// Until the game's submissions a flip needs (host_gpu_work_needs() as it
// arrived) have been handed to the driver's queue - not run: the presenter's
// copy on its own queue then waits on the GPU for a value already on its way.
void host_gpu_wait_work_submitted(std::uint64_t need);
// The render target the game displays (its VA from sceVideoOutRegisterBuffers)
// as an image to sample: VkImage, VkFormat, its size; false when the VA is not
// a drawn render target. Caller holds the renderer's lock (host_gpu_lock) and
// records the read on the presenter's command buffer before letting it go.
bool host_gpu_display_image(std::uint64_t display_va, void** image, std::uint32_t* format, std::uint32_t* width,
                            std::uint32_t* height);
// Records a blit of the render target the game displays (its VA from
// sceVideoOutRegisterBuffers) - src_width x src_height of it from (src_x,
// src_y) - into the rectangle (dst_x, dst_y, dst_w x dst_h) of `dst_image`
// (a VkImage) on `cmd` (a VkCommandBuffer). Caller holds the lock. Returns
// false when the VA is not a known render target. `dst_storage_view` (a
// VkImageView of `dst_image` that takes storage writes, or null): an upscale
// writes it directly instead of blitting a copy.
bool host_gpu_blit_display(void* cmd, std::uint64_t display_va, void* dst_image, std::int32_t dst_x, std::int32_t dst_y,
                           std::uint32_t dst_w, std::uint32_t dst_h, std::uint32_t src_width, std::uint32_t src_height,
                           std::uint32_t src_x = 0, std::uint32_t src_y = 0, void* dst_storage_view = nullptr);
// Writes the displayed render target as an image: PNG when the path ends in
// ".png", binary PPM otherwise (headless inspection). `all_targets`: also
// every colour RT (F12) as <stem>-rt-<base> in the same format and the draw
// list as <stem>-draws.txt. Same as BBHOST_DUMP_ALL=1 for a timed dump.
bool host_gpu_dump_display(std::uint64_t display_va, const char* path, bool all_targets = false);
// F12: dump on the next flip, into a new folder of its own beside the logs
// (logs/ or build/, BBHOST_CAPTURE_DIR), which the request makes and logs.
// Window thread (or BBHOST_F12_AT) sets, the flip takes it with its folder.
void host_gpu_request_dump();
bool host_gpu_take_dump_request(std::string* dir);
// The folder of the capture asked for last ("" before the first).
std::string host_gpu_capture_dir();
// BBHOST_CAPTURE_ARMED=1 (draw_capture.h): BBHOST_CAPTURE_DRAW may take up to
// `draws` more draws from now, into `dir` when it is not empty; 0 stops it.
void host_gpu_capture_arm(int draws, const char* dir);
// The capture's image format: ".ppm", or ".png" with BBHOST_F12_PNG=1.
const char* host_gpu_capture_ext();
// BBHOST_RT_REFILL_TEST=1: a re-created target starts with the fill its old image took (render.cpp).
void host_gpu_refill_selftest();
// --stream-selftest[=N] / BBHOST_STREAM_SELFTEST=N (host/stream_selftest.cpp):
// the command stream run N rounds four ways against a CPU model, headless,
// before anything of the game loads; 0 when every way matched the model.
int host_gpu_stream_selftest(int rounds);
void host_gpu_stall_test(unsigned ms);
void host_gpu_watch_display(std::uint64_t display_va);  // BBHOST_DUMP_ON_BRIGHT: dump when a frame washes out
void host_gpu_glitch_watch(std::uint64_t display_va, std::uint64_t mark);  // BBHOST_GLITCH=1: the glitch hunt (host/glitch.cpp)
// The draw count now, taken when a flip is queued: the frame's last draw (the
// glitch hunt's frame boundaries).
std::uint64_t host_gpu_draw_mark();
// For a flip to wait on its frame (BBHOST_FLIP_AFTER_GPU): how many
// submissions must have finished for everything recorded so far to have run,
// and how many have (~0 with no GPU to wait for).
std::uint64_t host_gpu_work_needs();
std::uint64_t host_gpu_submissions_completed();
// The GPU's PCI vendor (0x10de NVIDIA, 0x1002 AMD, 0x8086 Intel); 0 before the device is made.
std::uint32_t host_gpu_vendor_id();
// Sleeps until `need` submissions have finished (the timeline semaphore) or
// `timeout_ns` passed: true when they have. False at once without a timeline
// (the caller polls host_gpu_submissions_completed as before).
bool host_gpu_wait_completed(std::uint64_t need, std::uint64_t timeout_ns);
const char* host_present_step(double& seconds);  // window.cpp: what the presenting thread is doing  // BBHOST_STALL_TEST: hold the renderer, as a frame dump does
// Last few recorded draws, read without the renderer lock: for the watchdog,
// which runs precisely when some thread is stuck holding it.
void host_gpu_hang_report();
// Total texture uploads so far: the delta per flip window says whether a
// streamed surface (a movie frame) is reaching the GPU at the rate it changes.
std::uint64_t host_gpu_texture_uploads();
// Content-hash work (the streamed-texture re-check), cumulative.
std::uint64_t host_gpu_texture_hashes();
std::uint64_t host_gpu_texture_hash_us();
// Texture work on the command processor so far, microseconds: [0] new images and their memory, [1] upload
// staging buffers, [2] untiling, [3] recording the copies, [4] image views.
void host_gpu_texture_time_us(std::uint64_t out[5]);
// Graphics pipeline builds on the command processor so far: [0] translation us, [1] shader modules us,
// [2] vkCreateGraphicsPipelines us, [3] stage translations, [4] stages served by the stage cache.
void host_gpu_pipeline_time_us(std::uint64_t out[5]);
// What the command processor's pipeline builds spent in the driver so far, microseconds in [0..5], counts in
// [6..11]: shader libraries, interface libraries (vertex input, fragment output), links, whole pipelines,
// descriptor set and pipeline layouts, and waits for a library another thread was building; then compute
// pipelines a dispatch built itself, microseconds [12] and count [13].
void host_gpu_pipeline_work(std::uint64_t out[14]);
// Surfaces and render targets thrown away because their memory changed shape;
// each costs a full device idle.
std::uint64_t host_gpu_surface_replacements();

// Per-shader compiles, step 4: the game created a
// vertex (stage 0) or pixel (stage 1) shader from this Sony shader container
// (gx_trace.cpp hooks the GX shader creators).
void host_gpu_note_shader_created(int stage, const std::uint8_t* container, std::size_t size, std::uint64_t flip);
