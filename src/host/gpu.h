#pragma once

// Host GPU executor: runs translated GCN shaders on Vulkan. Works without a
// window (headless runs execute compute jobs and draws too). Guest direct
// memory is imported into Vulkan through host-pointer import, so shaders
// read and write the same pages the guest sees.

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <string>

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
    GxStageRecords stage[4];
};
constexpr int kGxRecords = 4;
constexpr int kGxRecordSet0 = 0, kGxRecordPixel = 1, kGxRecordLs = 2, kGxRecordHs = 3;

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
// Queues one compute dispatch. Returns false when the shader could not be
// translated or queued (the caller logs and continues). Queued work runs at
// the next host_gpu_flush().
bool host_gpu_dispatch(const GpuDispatch& d);
// Queues one draw.
bool host_gpu_draw(const GpuDraw& d);
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
// Submissions whose fence has retired, in order (trails the GPU: a write may be
// in memory before this passes its serial).
std::uint64_t host_gpu_completed_submits();
// BBHOST_ARENA_WATCH: logs where va resolves and which recent deferred writes carried value.
void host_gpu_explain_value(std::uint64_t va, std::uint64_t value);
bool host_gpu_mem_fill(std::uint64_t va, std::uint32_t value, std::size_t bytes);
bool host_gpu_mem_copy(std::uint64_t dst, std::uint64_t src, std::size_t bytes);
// A GX buffer copy reached as a copy token, handled as the compute copy
// path handles a recognised copy shader: marks the destination's surfaces
// dirty; when the source is a render target, copies its image into the
// destination's; otherwise records buffer copies in command order, split where
// either range crosses imports. False when a piece cannot be located or source
// and destination share bytes.
bool host_gpu_copy_guest(std::uint64_t dst, std::uint64_t src, std::size_t bytes);
// A GX dynamic buffer's renamed bytes copied back to its own address (a copy
// token's batched copies): inside a render pass, the draws after it read a
// copy of the bytes taken now and the copy into place waits for the pass's
// end (BBHOST_COPY_VERSIONS=0: as host_gpu_copy_guest, which it falls back to).
bool host_gpu_copy_back(std::uint64_t dst, std::uint64_t src, std::size_t bytes);
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
bool host_gpu_texture_create_ahead(const std::uint32_t tsharp[8]);
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
};
bool host_gpu_copy_image_region(const GpuImageCopy& c);
// The memory [base, base+bytes) of a resource GX destroyed or the kernel
// unmapped: its surfaces are retired once the command processor has retired
// everything submitted up to `after_submitted` (hle_gnm_submitted_total()).
void host_gpu_texture_retire(std::uint64_t base, std::size_t bytes, std::uint64_t after_submitted);
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
// Cumulative counters for per-frame deltas in the flip log.
struct GpuStats {
    std::uint64_t draws, dispatches, flushes, gpu_us, pipelines, transfers, transfer_batches;
    // CP-thread wall time by phase, nanoseconds (see kPhaseNames).
    std::uint64_t phase_ns[8];
    // host_gpu_draw calls, failures, and calls that had nothing to draw.
    std::uint64_t draw_calls, draw_failures, draws_empty;
};
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
std::string host_gpu_memory_budget_report();
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
// Surfaces and render targets thrown away because their memory changed shape;
// each costs a full device idle.
std::uint64_t host_gpu_surface_replacements();

// Per-shader compiles, step 4: the game created a
// vertex (stage 0) or pixel (stage 1) shader from this Sony shader container
// (gx_trace.cpp hooks the GX shader creators).
void host_gpu_note_shader_created(int stage, const std::uint8_t* container, std::size_t size, std::uint64_t flip);
