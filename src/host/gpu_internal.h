#pragma once

#include "host/gpu_mutex.h"
#include <cstdio>

// Shared state of the host GPU executor (gpu.cpp: device, guest memory,
// compute; render.cpp: graphics). Everything is guarded by Gpu::mu.

#include "gcn/translate.h"
#include "hle/modules.h"
#include "host/gpu.h"
#include "host/list_resolve.h"
#include "host/stream_ops.h"

#include <vulkan/vulkan.h>

// Every vkAllocateMemory/vkFreeMemory in the files that include this header
// is tagged with its site (gpu_memtrack.cpp), so a VRAM report can say which
// allocation holds what - the question when several instances share a GPU -
// and with its owner where a MemOwner names one (host/gpu_memtrack.h).
#include <chrono>
#include <string>

#include "host/gpu_memtrack.h"
VkResult bb_alloc_memory(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* cb,
                         VkDeviceMemory* out, const char* file, int line);
void bb_free_memory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* cb);
std::string bb_memory_sites_report();
// BBHOST_TEST_VRAM_MB: allocations in the memory types of `types` (a bit each)
// fail with VK_ERROR_OUT_OF_DEVICE_MEMORY once they would hold more than
// `bytes` together - a smaller card's video memory acted out on a bigger one.
void bb_memory_test_limit(std::uint32_t types, std::uint64_t bytes);
// The memory types of the device-local heap: what our allocations hold in
// them is counted (bb_memory_local_bytes), and the report's sites outside it
// are marked "@bar" (another device-local heap: bb_memory_set_heap_classes)
// or "@host".
void bb_memory_set_local_types(std::uint32_t types);
std::uint64_t bb_memory_local_bytes();
// Whether BBHOST_TEST_VRAM_MB acts out a smaller card: its limit, and what
// our allocations hold against it.
bool bb_memory_test_limited(std::uint64_t* limit, std::uint64_t* used);
#define vkAllocateMemory(d, i, a, m) bb_alloc_memory((d), (i), (a), (m), __FILE__, __LINE__)
#define vkFreeMemory(d, m, a) bb_free_memory((d), (m), (a))

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "host/heap_compact.h"  // the compaction's blocks and items (global namespace)

namespace gpu {

// BBHOST_RUN_COMMIT (render.cpp run_commit): a run worker's counters, kept
// apart while it commits its part and added to the shared ones by the command
// processor at the join - the shared counters' lines would otherwise pass
// between the cores at every draw, and their adds race. Open addressing by
// the counter's address; a full table adds the rest with a locked add.
struct BumpSink {
    static constexpr std::size_t kSlots = 512;
    std::atomic<std::uint64_t>* key[kSlots] = {};
    std::uint64_t n[kSlots] = {};
    std::uint16_t used[kSlots] = {};
    std::size_t nused = 0;
    void add(std::atomic<std::uint64_t>* c, std::uint64_t v) {
        std::size_t i = static_cast<std::size_t>((static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(c)) >> 3) * 0x9E3779B97F4A7C15ull >> 55);
        for (std::size_t probe = 0; probe < 8; ++probe, i = (i + 1) & (kSlots - 1)) {
            if (key[i] == c) {
                n[i] += v;
                return;
            }
            if (!key[i]) {
                key[i] = c;
                n[i] = v;
                used[nused++] = static_cast<std::uint16_t>(i);
                return;
            }
        }
        c->fetch_add(v, std::memory_order_relaxed);
    }
    // At the join, under g.mu: the counts into the counters, and the table emptied.
    void merge() {
        for (std::size_t k = 0; k < nused; ++k) {
            const std::uint16_t i = used[k];
            key[i]->store(key[i]->load(std::memory_order_relaxed) + n[i], std::memory_order_relaxed);
            key[i] = nullptr;
        }
        nused = 0;
    }
};
extern thread_local constinit BumpSink* t_bump_sink;

// A statistics counter only ever written under g.mu: incremented without a
// locked instruction. A locked one waits for every earlier store to drain -
// on the draw path, the params block just written - and the draw path bumps
// dozens of counters. On a run worker, into its sink (BumpSink) - looked for
// only with BBHOST_RUN_COMMIT on: a thread-local read is several loads where
// the bump is one add, and the draw path bumps at a hundred places.
inline void bump(std::atomic<std::uint64_t>& c, std::uint64_t n = 1) {
    if (list_resolve::committing()) [[unlikely]] {
        if (BumpSink* s = t_bump_sink) {
            s->add(&c, n);
            return;
        }
    }
    c.store(c.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

// A texture's memory: a range of an image-heap block, or (block < 0) an
// allocation of its own.
struct ImageMemory {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0, size = 0;
    std::int32_t block = -1;
    std::uint32_t type = UINT32_MAX;  // the memory type it is in (image_memory_in_vram); UINT32_MAX: not known
};
// Whether an image's memory is in video memory (the device-local heap; true
// when that is not known), and not in host memory for want of it.
bool image_memory_in_vram(const ImageMemory& m);

constexpr std::uint64_t kChunkBytes = 1ull << gcn::kDmemChunkShift;
// How far a whole-span import runs on into the next chunk (gpu.cpp map setup).
constexpr std::uint64_t kChunkOverlap = 64ull << 20;
constexpr std::uint32_t kMaxImages = 32;
// A frame with more draws than this takes a mid-frame submit (render.cpp).
// Measured 2026-09-26: 354 flips of ~11,000 in a combat soak, and 45 of 60 in
// the worst seconds - but those submits wait on nothing (`gpu-wait` 0-1 ms a
// window), so the ring is a symptom of a heavy frame, not its cost. Raising it
// doubles the descriptor pools as well; do not without a measurement.
constexpr std::uint32_t kMaxQueued = 2048;    // dispatches + draws per flush
constexpr std::uint32_t kParamsStride = 512;  // bytes per StageParams slot in the ring (352 bytes, a multiple of 256)
constexpr std::uint32_t kStageSlots = 2;      // VS + PS descriptor sets per draw
// Command buffers in flight, at most (BBHOST_GPU_SLOTS picks fewer, Gpu::slot_count).
// The command processor submits after every game command buffer - about 11 a
// frame in the world - so 8 let it run less than a frame ahead of the GPU: on
// the laptop it waited 1.1-1.9 s of every 5 s for a slot in heavy views, and
// the frame rate fell to 53-56 there. 16: 6-10 ms, 60 fps (2026-10-04).
constexpr int kSlots = 16;
// Upload staging's chunk size: 64 MiB, BBHOST_STAGING_CHUNK_MB=<4..64> smaller
// (each chunk given back in play is one vkFreeMemory, which on an RTX 4070
// took 22-29 ms for 64 MiB and stalled the frame behind it; gpu.cpp).
std::uint64_t staging_chunk_bytes();
constexpr std::uint64_t kStagingKeepBytes = 1024ull << 20;  // free upload staging chunks kept for reuse
// BBHOST_GPU_PROFILE timestamp queries per slot: a pair per draw/dispatch,
// the last pair spans the whole command buffer.
constexpr std::uint32_t kQueriesPerSlot = kMaxQueued * kStageSlots * 2;

// An import of host memory into Vulkan: [lo, lo + size) of a direct-memory
// chunk (offsets in its 1 GiB), or a whole flexible mapping (lo 0).
struct Chunk {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;  // of byte `lo` of the chunk
    std::uint64_t size = 0;
    std::uint64_t lo = 0;
    std::vector<VkDeviceMemory> pieces;  // dma-buf imports bound into a sparse `buffer` (gpu.cpp, dmabuf_span)
    // The idle census (BBHOST_IMPORT_IDLE_CENSUS=1, gpu.cpp): when it was made
    // and last found by dmem_import (steady ms, flip).
    std::uint64_t made_ms = 0, used_ms = 0, used_flip = 0;
};
// SCE_KERNEL_PROT_GPU_READ | SCE_KERNEL_PROT_GPU_WRITE: a mapping the GPU can reach.
constexpr int kGuestProtGpu = 0x30;
// Whether [in, in + bytes) of a direct-memory chunk is in this import.
inline bool chunk_covers(const Chunk& c, std::uint64_t in, std::uint64_t bytes) {
    return c.buffer && in >= c.lo && in - c.lo + bytes <= c.size;
}
// Direct memory is imported a 16 MiB window at a time, when the GPU side
// first resolves a range in it (a binding, a copy, an untile source, a mirror
// page, what a dispatch reads), and only inside the span the chunk's
// GPU-visible mappings cover (gpu_span): the game's CPU-only heaps, the
// never-allocated tail and the GPU heaps' untouched parts stay out of the
// driver's pinned memory - what Windows counts as the process's shared GPU
// memory. A range across windows gets one import of them all, widened to the
// whole of every import it overlaps, which it replaces: those retire (their
// windows and page-table entries move to the new one, and they are freed once
// the work recorded with them has run), so no page of a chunk is imported
// twice. An import may run on past its chunk's end into the next chunk's span
// (kChunkOverlap), so a buffer across the seam lies in one import, as the
// whole spans' overlap gives. Only with BBHOST_IMPORT_WINDOWS=1 (the counted
// fallback of the native path, gpu.cpp); otherwise each span is imported whole
// when the mappings are made.
constexpr std::uint32_t kImportWindowShift = 24;
constexpr std::uint32_t kImportWindows = 1u << (gcn::kDmemChunkShift - kImportWindowShift);  // per chunk
struct DmemChunk {
    std::uint64_t span_lo = 0, span_hi = 0;      // what may be imported (gpu_span); empty: nothing
    std::vector<std::unique_ptr<Chunk>> imports;  // the live imports (retired ones go to a slot's dead_imports)
    Chunk* window[kImportWindows] = {};           // the import that covers each window
    // When an import over each window last failed (steady ms; 0: never): the
    // window is not asked of the driver again for a while (import_backoff_ms),
    // its draws read through the page table and the sink meanwhile.
    std::uint64_t failed_ms[kImportWindows] = {};
};
// A plain device-addressable buffer (page tables, sink page).
struct DevBuffer {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    void* map = nullptr;
    std::uint64_t size = 0;
    // A device-local buffer that found video memory full and went to host
    // memory (create_dev_buffer): read across the bus until it is made again.
    bool in_host = false;
};

// A compute shader translated with its constant buffers, tables and typed
// loads read through storage buffers the dispatch pushes (set 1 of
// Gpu::cs_buffers_layout) and no page-table fallback for them
// (BBHOST_CS_BUFFERS, gpu.cpp): a dispatch every one of whose buffers binds
// runs it, any other the pipeline beside it.
struct ComputeLean {
    gcn::TranslateResult meta;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool standin = false;  // unoptimized, the optimized one queued (ComputePipeline::standin)
    std::uint64_t opt_seen = 0;
    bool lifted = false;  // a lifted program's: the lift of the lean translation (gpu.cpp build_compute_lean)
};

struct ComputePipeline {
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    gcn::TranslateResult meta;
    std::shared_ptr<ComputeLean> lean;  // null: none (off, nothing to bind, or it failed)
    std::string name;
    bool failed = false;
    // Its pipeline is still in the driver's compiler on a worker, and was not
    // waited for (gpu.cpp, take_precompiled_compute): failed until it is done.
    bool pending = false;
    // The unoptimized pipeline, standing in while the optimized one is built
    // in the background (BBHOST_CS_OPTIMIZE, gpu.cpp): the dispatch takes the
    // optimized one once it is there (adopt_optimized_compute).
    bool standin = false;
    std::uint64_t opt_seen = 0;
    // A "fill" shader: stores one constant (loaded from a constant buffer)
    // to every element of a V# in the user SGPRs. When that V# is a render
    // target the dispatch becomes an image clear (the image is the target's
    // storage here, and per-lane page-table stores are slow).
    bool fill = false;
    int fill_dst_sgpr = -1;   // user SGPR of the destination V#
    int fill_cb_sgpr = -1;    // user SGPR of the constant buffer V#
    std::uint32_t fill_cb_dw = 0;  // dword offset of the value in it
    bool fill_format = false;      // it stores with buffer_store_format_xyzw (through the V#'s format and select)
    // A "copy" shader: buffer_load_format_x from one user-data V# and
    // buffer_store_format_x to another, nothing else. A copy out of a
    // render target becomes an image copy into a snapshot image.
    bool copy = false;
    int copy_src_sgpr = -1, copy_dst_sgpr = -1;
    // The two copy shapes that move *images*, which the buffer copy
    // above does not cover. Both read their offsets and extent from a constant
    // buffer, so recognising the shape is only half of it - the dispatch still
    // has to read those to know what to copy.
    //   image -> image, a 2D sub-rectangle (a040875f, 8,100 dispatches a run)
    //   buffer -> 1D array image at an offset (306ce628, 9,560)
    bool image_blit = false;     // MIMG load + MIMG store, nothing else
    bool buffer_to_image = false;  // MUBUF load + MIMG store
    int img_src_sgpr = -1;       // the source V#'s user SGPR, for buffer_to_image
    bool untraced_stores_logged = false;
    bool gds_overlap_checked = false;     // GDS counters: whether a V# of the program covers a counter word, looked at once
    // Its module is the program's typed lift (gcn::lift_compute_shader,
    // BBHOST_LIFT_COMPUTE), not its translation; otherwise why not (empty:
    // not asked for), and the dispatches that ran it (under g.mu).
    bool lifted = false;
    std::string lift_why;
    std::uint64_t dispatches = 0;
    // The user-data dwords the translation touches at all (table_slots_read's
    // `any`), worked out the first time a dispatch asks for it
    // (GpuDispatch::want_user_reads: the dispatch-token shadow).
    std::uint32_t user_reads = 0xffff;
    bool user_reads_known = false;
    bool oob_named = false;  // BBHOST_OOB_COUNT: its counter slot named (note_oob_compute_slot)
};

// Guest VA -> (VkBuffer, offset) for index buffers.
struct UntileCheck;  // below, with the GPU untile

struct Located {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize avail = 0;
};

struct Gpu {
    std::uint32_t max_lds_bytes = 32 * 1024;  // maxComputeSharedMemorySize
    float max_anisotropy = 16.0f;             // maxSamplerAnisotropy
    float max_lod_bias = 15.0f;               // maxSamplerLodBias
    bool mirror_clamp = false;                // samplerMirrorClampToEdge, enabled
    bool in_transfer = false;                 // a coalesced run of guest-memory transfers is open
    // Fence writes held back to the end of the command buffer (see
    // host_gpu_mem_write): recording them inline broke the render pass and
    // drained the pipeline hundreds of times a frame.
    struct SmallWrite {
        std::uint64_t va;
        std::uint32_t bytes;
        std::uint8_t data[8];
        std::uint64_t epoch = 0;  // host_gpu_set_write_tag: the command processor's
        std::uint32_t op = 0;
    };
    std::vector<SmallWrite> deferred_writes;
    // CP writes recorded into submissions the GPU has not finished. A draw
    // resolves its T#s and S#s from guest memory while it is recorded, and the
    // resource table it reads was dumped by the constant engine a packet
    // earlier - into the command stream, not yet into memory. Reads made while
    // recording (read_guest_locked) see these on top of guest memory. Disjoint
    // spans by address, each tagged with the submission it was recorded into.
    struct PendingSpan {
        std::vector<std::uint8_t> data;
        std::uint64_t serial = 0;  // value of `flushes` while it was recorded
    };
    std::map<std::uint64_t, PendingSpan> pending_writes;
    std::uint64_t completed_submits = 0;  // submissions whose fence has retired, in order
    bool has_depth_bounds = false;        // VkPhysicalDeviceFeatures::depthBounds
    bool has_depth_bias_clamp = false;    // VkPhysicalDeviceFeatures::depthBiasClamp
    bool has_tessellation = false;        // VkPhysicalDeviceFeatures::tessellationShader
    std::uint32_t subgroup_stages = 0;    // VkPhysicalDeviceSubgroupProperties::supportedStages
    // Pixel shaders at wave32 (gcn/wave.h):
    // decided at device creation (gpu.cpp), given to each fragment stage by
    // render.cpp's fragment_stage_wave. 0: the device's own subgroup size for
    // every stage, as before (BBHOST_PS_WAVE32=0, or a device whose subgroups
    // are 32 wide already or that cannot be asked); 32: a required size of 32
    // for a program that computes the same in a 32-lane subgroup
    // (TranslateResult::wave64_needs 0), the device's size required of the
    // others; 1: the driver chooses between 32 and its default
    // (BBHOST_PS_WAVE32=driver, ALLOW_VARYING_SUBGROUP_SIZE).
    std::uint32_t ps_wave = 0;
    std::uint32_t subgroup_size = 0;                 // VkPhysicalDeviceSubgroupProperties::subgroupSize: the default
    std::uint32_t subgroup_min = 0, subgroup_max = 0;  // VkPhysicalDeviceSubgroupSizeControlProperties
    std::uint32_t subgroup_required_stages = 0;      // ... ::requiredSubgroupSizeStages
    bool has_gpl = false;                 // VK_EXT_graphics_pipeline_library enabled, with fast linking
    bool has_sparse = false;              // sparseBinding + sparseResidencyBuffer on a queue with SPARSE_BINDING
    // Direct memory imported as dma-bufs of the memfd (/dev/udmabuf) rather
    // than as host pointers, which AMD's Linux driver refuses for it (gpu.cpp,
    // choose_dmem_import). can_dmabuf: VK_KHR_external_memory_fd and
    // VK_EXT_external_memory_dma_buf enabled; dmabuf_sparse: sparseBinding too,
    // for spans longer than one dma-buf.
    bool can_dmabuf = false, dmabuf_sparse = false, dmem_dmabuf = false;
    int udmabuf = -1;
    PFN_vkGetMemoryFdPropertiesKHR get_memory_fd_props = nullptr;
    bool has_memory_budget = false;  // VK_EXT_memory_budget enabled
    // Video memory: the device-local heap images, targets and the buffer
    // shadow's mirrors are made in, and what this process may hold in it - the
    // driver's budget when the device was made (the card's memory less what
    // other programs held), else the heap's size; BBHOST_TEST_VRAM_MB lowers
    // it. vram_short: an image found no room there and went to host memory
    // (image_memory_alloc); the next submission takes it and has the buffer
    // shadow give its mirrors back (shadow_give_back_locked).
    std::uint32_t local_heap = 0;
    std::uint64_t local_budget = 0;
    std::atomic<bool> vram_short{false};
    // VK_KHR_pipeline_executable_properties enabled, for BBHOST_PIPELINE_STATS (render.cpp)
    bool pipeline_stats = false;
    // Depth targets in layouts of their own (render.cpp, rt_set_layout_locked):
    // on AMD's own drivers without VK_KHR_unified_image_layouts, where GENERAL
    // costs a depth target its compression; BBHOST_DEPTH_LAYOUTS=0/1 decides
    // instead on any device.
    bool depth_layouts = false;
    // shaderFloat16 and shaderInt16 enabled: the host's own shaders may do f16
    // math (the half-precision FSR 1, fsr.cpp).
    bool has_f16_math = false;
    // The game's submissions' timeline (Vulkan 1.2 timeline semaphore): each
    // signals its serial + 1 (Gpu::flushes), so how far the GPU got is one
    // counter read without the renderer's lock, and a wait for a frame is a
    // sleep in the driver, not a poll (host_gpu_wait_completed). Null without
    // timelineSemaphore, or with BBHOST_TIMELINE=0: the slots' fences, polled.
    VkSemaphore timeline = VK_NULL_HANDLE;
    // The presenter's own-queue copies' timeline (host_gpu_present_reads_locked):
    // each signals the value it registered. Made with `timeline`.
    VkSemaphore present_timeline = VK_NULL_HANDLE;
    bool has_device_fault = false;            // VK_EXT_device_fault
    bool has_checkpoints = false;             // VK_NV_device_diagnostic_checkpoints, or AMD's buffer markers
    PFN_vkCmdSetCheckpointNV cmd_checkpoint = nullptr;
    // VK_AMD_buffer_marker: the checkpoint index written as the GPU starts a
    // draw or dispatch (top of pipe, marker 0) and once everything before it
    // has finished (bottom of pipe, marker 1), into host memory that is still
    // readable after a device loss.
    PFN_vkCmdWriteBufferMarkerAMD cmd_buffer_marker = nullptr;
    DevBuffer marker_buf;
    GpuMutex mu;  // host/gpu_mutex.h: BBHOST_LOCK_PROFILE names who holds it
    // The queue's own mutex: the submission thread (gpu.cpp, submit_thread)
    // submits without g.mu, so every use of `queue` takes this - through
    // QueueGuard, which first waits for the thread's pending submissions.
    std::mutex queue_mu;
    bool tried = false;
    bool ok = false;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;  // the family's second queue, the presenter's own; null with one
    std::uint32_t family_queues = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    // The command buffer being recorded and its fence/pool: aliases of
    // slots[slot] so the recording code does not care about slots. Recording
    // code takes it through g_cmd(), which first lets the draw recorder
    // (recorder.cpp) catch up; only slot changes touch cmd_ directly.
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    // Submissions in flight. A slot is reused once its fence signalled;
    // staging buffers recorded into it are freed then.
    struct Slot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        bool in_flight = false;
        std::uint64_t serial = 0;  // Gpu::flushes when this slot was submitted
        // The presenter's own-queue copy recorded while this slot was the one
        // being recorded: its value on Gpu::present_timeline, waited for
        // before the slot retires (0: none).
        std::uint64_t present_target = 0;
        // What the submission holds, for the line a wait for it that runs long
        // writes (fence_wait_locked): the draws, dispatches and tessellated
        // draws recorded since the submission before, and the flip.
        std::uint64_t held_draws = 0, held_dispatches = 0, held_tess = 0, held_flip = 0;
        bool said_stuck = false;  // that line was written for this submission
        std::vector<DevBuffer> garbage;
        std::vector<Chunk> dead_imports;  // window imports a merged one replaced (gpu.cpp retire_import_locked)
        std::vector<DevBuffer> staging_chunks;  // the upload staging chunks this slot's uploads are in, each once (Gpu::staging_users)
        std::vector<DevBuffer> scratch_chunks;  // the same for the device-local untile scratch (Gpu::scratch_users)
        // Images (with their memory and views) whose guest memory was
        // re-purposed. Destroying them takes only the completion of this
        // slot's submission, which is ordered after every earlier one, so
        // nothing in flight can still be reading them.
        std::vector<VkImage> dead_images;
        std::vector<VkDeviceMemory> dead_memory;
        std::vector<ImageMemory> dead_ranges;  // image-heap ranges, released once this slot retired
        std::vector<std::pair<VkDescriptorPool, VkDescriptorSet>> dead_sets;  // cached sets let go (render.cpp set cache)
        std::vector<VkImageView> dead_views;
        std::vector<UntileCheck> untile_checks;  // BBHOST_GPU_UNTILE=2 pairs, checked once this slot retired
        // The command processor's writes this submission fills on the GPU
        // (BBHOST_LABEL_LANDING=0), tagged: checked once it retired for one
        // the GPU wrote over memory the game had written again since
        // (note_fill_reused, fills_retired_locked).
        std::vector<SmallWrite> fill_checks;
        // BBHOST_GLITCH=1 (glitch.cpp): an occlusion query around each draw,
        // and the draw record each one belongs to, read once the slot's fence
        // has signalled.
        VkQueryPool cov_pool = VK_NULL_HANDLE;
        std::vector<std::uint64_t> cov_recs;
        bool cov_reset = false;  // this recording reset the pool before its first query
        // BBHOST_GPU_PROFILE=1: timestamp pairs around each draw/dispatch.
        VkQueryPool qpool = VK_NULL_HANDLE;
        std::vector<const std::string*> qnames;  // pipeline name per pair
        // BBHOST_GPU_PROFILE=2: per pair, the recording sites (g_cmd() callers)
        // between it and the pair before - what the gap between them ran -
        // and those after the last pair.
        std::vector<std::array<void*, 4>> qgaps;
        std::array<void*, 4> tail_gap{};
        bool qreset = false;
        bool span = false;  // the command-buffer span pair was written
        VkQueryPool stat_pool = VK_NULL_HANDLE;  // Gpu::profile_stats: one query per pair
        bool stat_open = false;                  // the last pair's statistics query is still active
        // The GPU busy meter (gpu_busy.cpp, on unless BBHOST_GPU_BUSY=0): a
        // timestamp at the command buffer's start and one at its end, read
        // once the slot retired. `busy_written`: this recording wrote both.
        VkQueryPool busy_pool = VK_NULL_HANDLE;
        bool busy_written = false;
        // The buffer shadow's upload command buffer, which runs just ahead of
        // this one in the same submission, wrote its start (query 2).
        bool busy_upload = false;
        // Descriptor sets allocated ahead from this slot's pool, by layout
        // (alloc_set_locked); dropped with the pool's reset.
        std::vector<std::pair<VkDescriptorSetLayout, std::vector<VkDescriptorSet>>> spare_sets;
    };
    Slot slots[kSlots];
    int slot = 0;
    int slot_count = kSlots;  // slots in use (BBHOST_GPU_SLOTS); submission n is in slots[n % slot_count]
    // Upload staging chunks free for reuse (acquire_staging_locked).
    std::vector<DevBuffer> staging_free;
    std::uint64_t staging_free_bytes = 0;
    // The chunk uploads are placed in now, whichever slot records them, and
    // how many slots list each chunk: one is free once no slot does.
    DevBuffer staging_cur;
    std::uint64_t staging_cur_used = 0;
    std::map<VkBuffer, std::uint32_t> staging_users;
    // Device-local scratch the GPU untile works in (acquire_scratch_locked):
    // the same scheme, without the memory keeper.
    std::vector<DevBuffer> scratch_free;
    std::uint64_t scratch_free_bytes = 0;
    DevBuffer scratch_cur;
    std::uint64_t scratch_cur_used = 0;
    std::map<VkBuffer, std::uint32_t> scratch_users;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;         // compute: one set
    // Compute with its buffers pushed (ComputeLean): set 0 as pipe_layout's,
    // set 1 cb_push_layout. Null without push descriptors.
    VkPipelineLayout cs_buffers_layout = VK_NULL_HANDLE;
    VkPipelineLayout gfx_pipe_layout = VK_NULL_HANDLE;     // graphics: set 0 VS, set 1 PS
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkPipelineCache cache = VK_NULL_HANDLE;
    VkPipelineCache side_cache = VK_NULL_HANDLE;  // PipelineCacheUse: creations while a save serializes `cache`
    // The bb-cs-optimize threads' builds (BBHOST_CS_OPTIMIZE): a cache of their
    // own, merged into `cache` where the side cache is, so that a build taking
    // minutes never holds a save waiting for `cache`'s users (gpu.cpp).
    VkPipelineCache opt_cache = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    bool has_maint8 = false;
    // Vulkan 1.3's pipelineCreationCacheControl: a pipeline can be asked for
    // only if the pipeline cache already has it (VK_PIPELINE_CREATE_FAIL_ON_
    // PIPELINE_COMPILE_REQUIRED_BIT), which answers without compiling.
    bool cache_control = false;
    std::uint32_t vendor_id = 0;  // VkPhysicalDeviceProperties::vendorID (0x10de NVIDIA, 0x1002 AMD)
    VkDriverId driver_id = {};    // VkPhysicalDeviceDriverProperties::driverID (VK_DRIVER_ID_MESA_LLVMPIPE: lavapipe)
    bool integrated = false;      // VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: its device-local memory is system RAM
    bool tight = false;           // little memory in all its heaps (gpu.cpp, memory_tight)
    // A discrete GPU with under 7.5 GiB of video memory free at the start
    // (gpu.cpp, small_vram): fewer spares, less scratch, a tessellation ring
    // that grows as it is needed, display buffers no larger than the start.
    bool small_vram = false;
    bool has_push_descriptor = false;       // VK_KHR_push_descriptor enabled
    PFN_vkCmdPushDescriptorSetKHR cmd_push_descriptor_set = nullptr;
    // VK_EXT_extended_dynamic_state3's depth clamp (BBHOST_DYNAMIC_DEPTH_CLAMP=0
    // turns it off): pre-rasterization libraries leave it to the draw.
    bool dynamic_depth_clamp = false;
    PFN_vkCmdSetDepthClampEnableEXT cmd_set_depth_clamp_enable = nullptr;
    // Graphics constant buffers, pushed per draw into set 2 (cb_push_on()):
    // the vertex stage's at bindings 112-127, the pixel stage's at 128-143.
    VkDescriptorSetLayout cb_push_layout = VK_NULL_HANDLE;
    // BBHOST_BINDLESS=1 (bindless.cpp): graphics stages read
    // images and samplers from one global set (set 3); their own sets hold the
    // params block alone. gfx_set_layout is the graphics stage sets' layout:
    // set_layout's, or the params block alone when bindless.
    bool bindless = false;
    VkDescriptorSetLayout gfx_set_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout bindless_layout = VK_NULL_HANDLE;
    VkDescriptorPool bindless_pool = VK_NULL_HANDLE;
    VkDescriptorSet bindless_set = VK_NULL_HANDLE;
    bool present_capable = false;
    // VK_KHR_present_mode_fifo_latest_ready (or its EXT original) with its
    // feature on: the presenter may ask for FIFO_LATEST_READY (window.cpp).
    // The extension's name, or null.
    const char* fifo_latest_ready_ext = nullptr;
    // Stage parameter ring: one block per queued dispatch/draw stage.
    VkBuffer ubo = VK_NULL_HANDLE;
    VkDeviceMemory ubo_mem = VK_NULL_HANDLE;
    void* ubo_map = nullptr;
    std::uint32_t queued = 0;   // params/descriptor slots used in the current command buffer
    bool recording = false;
    std::uint64_t record_serial = 0;  // bumped whenever g.cmd_ starts recording (render.cpp's state cache)
    std::uint64_t flushes = 0;  // submissions
    // Dummy resources for untraceable/unsupported images.
    VkImage dummy_image = VK_NULL_HANDLE;
    VkDeviceMemory dummy_mem = VK_NULL_HANDLE;
    VkImageView dummy_view = VK_NULL_HANDLE;
    // Invalid T# (type&8==0, no r128): GCN returns 0. Opaque black (A=1) on
    // that path scaled YEBIS blur offsets to full strength (smear after the
    // bars went away).
    VkImage dummy_zero_image = VK_NULL_HANDLE;
    VkDeviceMemory dummy_zero_mem = VK_NULL_HANDLE;
    VkImageView dummy_zero_view = VK_NULL_HANDLE;
    // One 1x1 image + view per (spv::Dim, arrayed) pair, so every image
    // binding a shader declares can be written even when the T# resolves to
    // nothing: an unwritten descriptor is undefined, and reading one is a
    // plausible source of a GPU page fault.
    std::map<std::uint32_t, VkImage> dummy_dim_images;
    std::map<std::uint32_t, VkDeviceMemory> dummy_dim_mem;
    std::map<std::uint32_t, VkImageView> dummy_dim_views;
    // 1x1 D32_SFLOAT images at the far plane, by (dim << 1) | arrayed (2D, 2D
    // array, cube, cube array): what a depth-compare binding samples where its
    // colour view cannot compare (depth_dummy_view_for).
    std::map<std::uint32_t, VkImage> dummy_depth_images;
    std::map<std::uint32_t, VkDeviceMemory> dummy_depth_mem;
    std::map<std::uint32_t, VkImageView> dummy_depth_views;
    VkSampler dummy_sampler = VK_NULL_HANDLE;
    // Constant buffers as storage buffers (TranslateOptions::cb_ssbo, BBHOST_CB_SSBO=0 to turn off):
    // the buffer bound where a V# could not be, and the device limits for binding one.
    bool cb_ssbo = true;
    bool cb_lean = true;  // pipelines without the fallback for draws whose constant buffers all bind
    bool exec_known = true;  // translate with static EXEC knowledge (BBHOST_EXEC_KNOWN=0 turns it off)
    VkBuffer dummy_ssbo = VK_NULL_HANDLE;
    VkDeviceMemory dummy_ssbo_mem = VK_NULL_HANDLE;
    VkDeviceSize ssbo_align = 1;
    VkDeviceSize ssbo_max_range = 0;
    // Guest memory.
    void* mirror = nullptr;
    std::uint64_t mirror_size = 0;
    DmemChunk dmem[gcn::kDmemChunks];
    DevBuffer l1, sink, sink_l2;
    DevBuffer oob;  // BBHOST_OOB_COUNT=1: gcn::set_oob_count's counters (gcn::kOobSlots x kOobKinds dwords)
    std::map<std::uint32_t, DevBuffer> l2;          // by va >> 32
    std::map<std::uint64_t, Chunk> anon_imports;    // by va: imported flexible mappings
    std::vector<GuestMapInfo> maps;                 // last snapshot
    std::uint64_t maps_hash = 0;
    VkPhysicalDeviceMemoryProperties mem_props{};
    std::uint32_t host_pointer_align = 4096;
    PFN_vkGetMemoryHostPointerPropertiesEXT get_host_pointer_props = nullptr;
    std::map<std::uint64_t, ComputePipeline> pipelines;
    // Stats
    std::atomic<std::uint64_t> dispatches{0}, failures{0}, translated{0}, dummy_images{0};
    // Dispatches that ran a shader (not a recognised fill or copy), and of
    // them those whose module was the program's typed lift.
    std::atomic<std::uint64_t> shader_dispatches{0}, lifted_dispatches{0};
    std::atomic<std::uint64_t> transfer_batches{0};  // barrier pairs the transfers above needed
    std::atomic<std::uint64_t> rt_replacements{0};   // render targets re-created (each idles the device)
    std::atomic<std::uint64_t> draws{0}, draw_failures{0}, gfx_pipelines{0};
    // The draws by how their pixel shader ran (DrawPs, gpu.h), beside `draws`.
    std::atomic<std::uint64_t> draws_ps[4] = {};
    std::atomic<std::uint64_t> draws_clip_discard{0};  // GpuStats::draws_clip_discard
    std::atomic<std::uint64_t> draws_vs[4] = {};  // ... and by how their vertex shader ran (DrawVs)
    std::atomic<std::uint64_t> draws_vs_why[16] = {};  // kDrawVsOther by why (DrawVsWhy)
    std::atomic<std::uint64_t> draws_tess[6] = {};     // the host tessellator's draws by lifted stages (DrawTess)
    // Why each failed draw failed (DrawFail), reported at exit beside the total.
    std::atomic<std::uint64_t> draw_fail_why[16] = {};
    // host_gpu_draw calls, and those that returned without drawing (no target, or no colour and no depth write).
    std::atomic<std::uint64_t> draw_calls{0}, draws_empty{0};
    std::atomic<std::uint64_t> gpu_us{0};
    // What a submission is waiting for, for the hang watchdog (host_gpu_watchdog):
    // the slot whose fence a thread is inside, and when it started.
    std::atomic<int> waiting_slot{-1};
    std::atomic<std::uint64_t> wait_started_ns{0};
    std::atomic<std::uint64_t> transfers{0};  // GPU-side CP memory writes
    // GPU profile (BBHOST_GPU_PROFILE=1): nanoseconds and count by pipeline.
    // =2 also charges the time between pairs to what was recorded in it
    // (profile_gap_ns, by recording sites).
    bool profile = false;
    bool profile_gaps = false;
    bool profile_passes = false;  // BBHOST_GPU_PROFILE=3: a pair a render pass, not a draw (render.cpp)
    // BBHOST_GPU_PROFILE=3 (or =1 with BBHOST_GPU_PROFILE_STATS=1) where the
    // device counts them: a pipeline-statistics query beside each pair (vertices and primitives in, vertex-shader
    // invocations, primitives into and out of clipping, fragment-shader
    // invocations), summed by name like profile_ns.
    bool profile_stats = false;
    std::map<std::string, std::array<std::uint64_t, 6>> profile_stats_sum;
    std::map<std::array<void*, 4>, std::pair<std::uint64_t, std::uint64_t>> profile_gap_ns;
    float timestamp_period_ns = 1.0f;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> profile_ns;
    // Whole command buffers, first command to last: includes the transfers,
    // barriers and render-pass changes the draw/dispatch pairs miss.
    std::uint64_t cmdbuf_ns = 0, cmdbuf_count = 0;
    // The GPU busy meter (gpu_busy.cpp) is on: the slots and the presenter
    // have their timestamp pairs.
    bool busy = false;
    std::atomic<std::uint64_t> phase_ns[8] = {};
};
extern Gpu g;

// Why host_gpu_draw returned without drawing (Gpu::draw_fail_why).
enum DrawFail : int {
    kFailTessOff,           // BBHOST_TESS=0
    kFailTessPlan,          // a tessellated draw's plan
    kFailTessLs,            // its LS pass
    kFailPrimitive,         // a primitive type with no Vulkan topology
    kFailNoVs,              // no vertex shader address
    kFailVsProgram,         // the vertex shader would not translate
    kFailPsProgram,         // the pixel shader would not translate
    kFailFetch,             // the fetch shader is unreadable
    kFailPipeline,          // the pipeline failed to build
    kFailSet,               // a descriptor set from the cache's pool
    kFailSetAlloc,          // a descriptor set allocation
    kFailFallbackBinding,   // the fallback translation binds differently (one draw)
    kFailPipelineCreate,    // creating the (lean-first) pipeline object
    kFailIndexBuffer,       // the index buffer is not in imported memory
    kFailIndirectArgs,      // the indirect arguments are not in imported memory
    kFailGpuGone,           // no GPU to draw with: the device is lost (or never came up)
    kFailCount
};
static_assert(kFailCount <= 16, "Gpu::draw_fail_why has 16 slots");
inline void draw_failed(DrawFail why) {
    g.draw_failures.fetch_add(1, std::memory_order_relaxed);
    g.draw_fail_why[why].fetch_add(1, std::memory_order_relaxed);
}

// The cache a pipeline creation passes. Serializing the main cache to save it
// (~1 s at 300 MB, every three minutes once pipelines were created) holds it
// against creation in the driver: the command processor's next pipeline
// waited out the save, a 0.4-1 s stall. Meanwhile creations go to a side
// cache instead, which the save merges into the main one when it is done,
// and merging needs the main cache to itself: `users` counts the creations
// holding it.
extern std::atomic<bool> g_cache_divert;
extern std::atomic<int> g_cache_users;
extern std::atomic<std::uint64_t> g_cache_diverted;  // creations that took the side cache
struct PipelineCacheUse {
    VkPipelineCache cache = VK_NULL_HANDLE;
    bool main = false;
    PipelineCacheUse() {
        g_cache_users.fetch_add(1);
        main = !g_cache_divert.load() || g.side_cache == VK_NULL_HANDLE;
        if (!main) {
            g_cache_users.fetch_sub(1);
            g_cache_diverted.fetch_add(1, std::memory_order_relaxed);
        }
        cache = main ? g.cache : g.side_cache;
    }
    ~PipelineCacheUse() {
        if (main) g_cache_users.fetch_sub(1);
    }
    PipelineCacheUse(const PipelineCacheUse&) = delete;
    PipelineCacheUse& operator=(const PipelineCacheUse&) = delete;
};

// render.cpp: a program's resource paths, translated off the command processor
// when it was created, for paths_for to take.
void offer_paths(const std::vector<std::uint32_t>& words, const gcn::TranslateResult& paths);

// render.cpp: the stage manifest (stages the draws translated, compiled at the
// next start), kept beside the pipeline cache.
void stage_manifest_load(const std::string& path);
void stage_manifest_save(const std::string& path);
void stage_manifest_save_async(const std::string& path);
std::string stage_manifest_path();
// The pipeline manifest (render.cpp PipelineRecipe): <data root>/bbhost/pipeline-manifest.bin.
void pipeline_manifest_load(const std::string& path, const std::string& list_path);
void pipeline_manifest_save(const std::string& path);
void pipeline_manifest_save_async(const std::string& path);
std::string pipeline_manifest_path();
// The shader sweep (render.cpp): every shader in the game's bundles compiled
// into the caches at the first start for this translator and device, the
// stamp at `stamp_path` (<data root>/bbhost/shader-sweep.txt) saying it ran.
void shader_sweep_start(const std::string& stamp_path, const std::string& eboot_path);
// The compile session (host_gpu_set_compile_session): set before the device
// starts; the sweep then takes the bundles' shaders at once, in full.
inline std::atomic<bool> g_compile_session{false};
void compile_session_progress(std::uint64_t& total, std::uint64_t& done, std::uint64_t& failed, bool& listed, bool& finished,
                              std::string& reasons, std::uint64_t& recipes_total, std::uint64_t& recipes_done);
void compile_session_cancel();
// What it has left while the title or a loading screen is up (else 0).
std::size_t shader_sweep_left();
// render.cpp: the precompile workers run on half the hardware threads until
// the first in-game frame and while a loading screen is up, a quarter of them
// otherwise.
void precompile_set_loading(bool loading);
void precompile_set_world_reached();
// The jobs queued and compiling on the precompile workers.
std::size_t precompile_backlog();

// gpu.cpp
bool init_locked();
// Closes an open run of guest-memory transfers (see transfer_begin_locked).
// Must run before any work that reads what those transfers wrote.
void transfer_flush_locked();
// Records the fence writes host_gpu_mem_write held back, at the end of the
// command buffer they belong to.
void flush_deferred_writes_locked();
void flush_locked();       // submits and waits for every submission in flight
void submit_locked();      // submits the current command buffer; waits only for the slot it moves to
// Waits for submission `serial` and every one before it (those still in
// flight retire, oldest first); nothing for one not submitted yet.
void retire_through_locked(std::uint64_t serial);
// Waits until submission `serial` has run, retiring nothing: for a recording
// in progress, which slot cleanup in its middle could disturb. A GPU taken
// for stuck meanwhile (fence_wait_locked) ends the wait; nothing recorded
// after that goes to the GPU (submit_locked).
void wait_submission_locked(std::uint64_t serial);
std::uint64_t render_tess_draws();  // the tessellated draws so far (render.cpp)
// Copies guest memory as the work being recorded now will see it: the CP writes
// still queued in unfinished submissions applied on top (Gpu::pending_writes).
void read_guest_locked(std::uint64_t va, std::size_t bytes, void* out);
// A dispatch token's compute-stage user data from its records (render.cpp):
// the dwords its descriptors write, tables in the host ring; `set` says
// which. False when a descriptor is one the builder does not take.
bool render_cs_user_data_locked(const GxStageRecords& st, std::uint32_t user[16], std::uint16_t* set);
void defer_destroy(const DevBuffer& b);  // freed once the current command buffer retired
// Upload staging: `size` bytes placed in a mapped host-visible chunk the current command buffer shares with the ones
// before and after it; the chunk is free for reuse once all of them retired. `b` is the chunk with `map` advanced to
// the span; `offset` is the span's offset in it.
bool acquire_staging_locked(DevBuffer& b, std::uint64_t size, VkDeviceSize& offset);
// A span of device-local scratch for this command buffer, 256-byte aligned
// (b.buffer, b.address at the chunk's start, `offset` into it); free for
// reuse once the slot retires.
bool acquire_scratch_locked(DevBuffer& b, std::uint64_t size, VkDeviceSize& offset);
std::string staging_pool_stats();  // "chunks created N, uploads placed M"
// The same for an image and its views: avoids a vkDeviceWaitIdle per surface
// whose memory the game re-used, which cost most of a world load's GPU time.
void defer_destroy_image(VkImage image, VkDeviceMemory memory);
// Device-local memory for sampled textures (gpu.cpp image heap): carved out
// of 256 MiB blocks instead of one vkAllocateMemory - a kernel call - per
// texture, which was ~10% of the command processor in frames where new
// areas streamed in. Larger images get memory of their own.
// texture: a sampled image or a copy made for one, which the video memory
// reserve keeps out of the last of video memory (vram_reserve_bytes);
// otherwise a render target or snapshot, which may take the reserve.
// cls: the blocks it goes in (HeapClass; -1: kHeapTextures for a texture,
// kHeapTargets otherwise) - render targets, snapshots and what the GPU
// writes live as long as an area does, textures come and go as the game
// streams them, and apart the textures' blocks empty (BBHOST_HEAP_CLASSES).
enum HeapClass : int { kHeapTargets = 0, kHeapTextures = 1 };
bool image_memory_alloc(const VkMemoryRequirements& req, ImageMemory& out, bool texture = false, int cls = -1);
// The class of the block `m` was placed in (-1: none, -2: memory of its own).
int image_memory_class_locked(const ImageMemory& m);
bool image_heap_classes_on();  // BBHOST_HEAP_CLASSES
// The textures' block in video memory with the least in use, not one in
// `skip` (-1: none, or only one texture block), what it has in use, and the
// free room in all the textures' blocks together.
int image_heap_drain_candidate_locked(std::uint64_t* class_free, std::uint64_t* used, const std::vector<int>& skip);
// The textures' blocks drain (textures.cpp, BBHOST_HEAP_DRAIN): see there.
void drain_texture_blocks_locked();
// The targets' blocks drain (render.cpp drain_target_blocks_locked): the
// targets' block that holds least when the class has two or more - its index,
// the bytes it holds and the free room of the others' (-1: none); a block no
// image goes in while it is emptied (-1: none); a block's bytes in use; and an
// image's memory from room the heap's video memory blocks already have -
// never a new block, never host memory (false when there is none).
int image_heap_target_drain_candidate_locked(std::uint64_t* used, std::uint64_t* room_elsewhere);
void image_heap_avoid_block_locked(int block);
int image_heap_avoided_block_locked();
std::uint64_t image_heap_block_used_locked(int block);
bool image_memory_place_existing_locked(const VkMemoryRequirements& req, ImageMemory& out, int cls);
// textures.cpp: whether the block holds a surface that cannot be made again
// (a shader or a copy wrote it), and letting go of what it holds that can -
// textures, copies of targets' regions, level aliases (made again at their
// next bind); returns the bytes let go.
bool textures_block_pinned_locked(int block);
std::uint64_t textures_drop_block_locked(int block);
void drain_target_blocks_locked();  // render.cpp, once a second from the submission
std::uint64_t image_heap_spare_in_play();  // the memory keeper's spare blocks in play (gpu.cpp)
std::string target_drain_report();  // render.cpp: what the targets' blocks drain did
// The image heap's compaction (render.cpp compact_image_heap_locked,
// host/heap_compact.h; BBHOST_HEAP_COMPACT, on where the image heap is lean,
// in place of the textures' drain and after the targets' drain): its switch
// (gpu.cpp); the heap's blocks as it
// sees them, the memory type of its images and the placement's switches
// (gpu.cpp); the surfaces of each block it would copy and whether a block
// holds one that can neither move nor go (textures.cpp); a turn's textures -
// those that need no copy let go, then copies while `credit` lasts, false
// when one found no room (textures.cpp); the turn, once a second from the
// submission, and what it has done for the census (render.cpp).
bool heap_compact_on();
bool world_reached();  // render.cpp: the first in-game frame has come (a block made before it is the boot's, not the play's)
void image_heap_blocks_locked(std::vector<heap_compact::Block>& out, std::uint32_t* type, bool* classes, bool* overflow, bool* fullest);
void textures_compact_items_locked(std::vector<std::vector<heap_compact::Item>>& items, std::vector<bool>& pinned, std::uint64_t hot_ms);
struct CompactTextureTurn {
    std::uint64_t let_go = 0, let_go_bytes = 0, copied = 0, copied_bytes = 0, busy = 0;
};
bool textures_compact_turn_locked(int block, std::uint64_t hot_ms, std::int64_t& credit, std::size_t& copied_this_turn, CompactTextureTurn& out);
void compact_image_heap_locked();
std::string heap_compact_report();
// The drain so far: blocks emptied, textures and MiB taken out, blocks left
// because a surface the GPU wrote sits in them.
void textures_drain_counts(std::uint64_t out[4]);
// The textures' blocks' pinned surfaces (one a shader or a copy wrote) moved
// into the targets' room, for the block the drain named (textures.cpp): in the
// flush, while its recording is open. And what it has done, for the census.
void textures_relocate_pinned_locked();
std::string textures_relocate_report();
// The targets' drain (render.cpp) asking for block `block`'s pinned surfaces
// to move at the next flush, as the textures' drain does: true when they will
// (BBHOST_HEAP_RELOCATE on, none failed, 32 MiB at most, no other block named).
bool textures_relocate_block_locked(int block);
// The video memory this process may still take, in bytes (negative: over
// budget): BBHOST_TEST_VRAM_MB's limit less what our allocations hold against
// it, else the driver's budget for the heap less its usage
// (VK_EXT_memory_budget, read at most every 100 ms), else the budget at the
// start less what our allocations hold there. Any thread.
std::int64_t vram_headroom();
// The video memory kept for render targets, snapshots and device buffers
// (BBHOST_VRAM_RESERVE_MB; 0: the budget is not followed, as before):
// textures take no new video memory out of it, the memory keeper makes no
// spare in it, and the untile scratch takes no new chunk there.
std::uint64_t vram_reserve_bytes();
// Whether video memory is down to the reserve (headroom under reserve +
// `more` bytes); false when the reserve is off.
bool vram_within_reserve(std::uint64_t more = 0);
// The caches' budget and the lean image heap (gpu.cpp, host/apu_memory.h),
// set with the reserve once the device is chosen: the budget in bytes (0:
// none); whether the video memory we hold, with `more` about to be made, is
// over it; how far over it is (0 under it); whether the image heap is lean -
// no spare block in play, the textures' blocks drained at the lean
// thresholds. Any thread.
void cache_budget_init();
std::uint64_t cache_budget_bytes();
bool cache_over_budget(std::uint64_t more = 0);
std::uint64_t cache_budget_excess();
bool heap_lean_on();
// The world settled: no loading screen, and BBHOST_CACHE_SETTLE_S (30) since
// the last one ended. The budget and the lean heap press only then
// (cache_budget_pressing: settled and over the budget), so an area being
// entered keeps what it makes ahead.
bool world_settled();
bool cache_budget_pressing(std::uint64_t more = 0);
// Textures kept out of the reserve so far (placed in host memory instead).
std::uint64_t vram_reserve_refusals();
// Whether what a shortage put in host memory comes back once video memory has
// room (BBHOST_VRAM_RETURN, on with the reserve): the mirrors, textures,
// render targets and snapshots, device buffers.
bool vram_return_on();
// Whether video memory has `bytes` free above the reserve and `margin`.
bool vram_room_for(std::uint64_t bytes, std::uint64_t margin = 256ull << 20);
// The render targets and snapshots in host memory, one a call at most, moved
// back into video memory (render.cpp; while recording - it records the copy).
void render_return_targets_locked();
// Textures in host memory dropped, at most 64 MiB a call, to be made again in
// video memory at their next bind (textures.cpp).
void textures_return_from_host_locked();
// Textures not bound for a while evicted while video memory is down to the
// reserve and the image heap's blocks are short of room (textures.cpp,
// BBHOST_VRAM_EVICT_S).
void evict_cold_surfaces_locked();
// Free room in the image heap's blocks of video memory (gpu.cpp).
// cls (HeapClass): only the room that class may take - its blocks' and the
// empty ones'; -1 every block's.
std::uint64_t image_heap_vram_free_locked(int cls = -1);
std::string vram_return_report();  // the 5-second report's counts
// What the image heap's images hold by kind, for the video memory report
// (video_memory_report, gpu.cpp). Under g.mu. Bytes in video memory, and
// what a shortage put in host memory apart (rt_host, tex_host).
struct VramKind {
    std::uint64_t bytes = 0, n = 0;
    void add(std::uint64_t b) {
        bytes += b;
        ++n;
    }
};
struct VramCensus {
    // render.cpp: colour and depth targets (g_rts), the display buffers' size
    // apart; of all three, those no draw, fill or copy has written for 10 s
    // (rt_idle); copy snapshots (g_snapshots), likewise.
    VramKind rt_colour, rt_depth, rt_display, rt_idle, snapshots, snap_idle, rt_host;
    // textures.cpp: surfaces a shader or a copy wrote (and those not written
    // for 10 s), textures made ahead and never bound, textures no bind looked
    // up for 20 s (what the cold eviction takes when video memory is down to
    // the reserve), the rest; copies of targets' regions and level aliases.
    VramKind gpu_written, gpu_written_idle, tex_ahead, tex_cold, tex_bound, regions, tex_host;
    // The image heap's classes (BBHOST_HEAP_CLASSES): surfaces that turned
    // out GPU-written but sit in the textures' blocks, and uploaded ones in
    // the targets' - the guess at creation missed both.
    VramKind gpu_written_in_tex, tex_in_targets;
    // depth_copy.cpp and other images with memory of their own in video
    // memory: counted by gpu_memtrack's owners instead.
    // By image heap block (ImageMemory::block), bytes of each kind: colour
    // targets, depth targets, snapshots, GPU-written surfaces, textures,
    // copies of targets' regions and levels - what keeps a block from emptying.
    enum : int { kBlockColour, kBlockDepth, kBlockSnapshot, kBlockGpuWritten, kBlockTexture, kBlockRegion, kBlockKinds };
    std::vector<std::array<std::uint64_t, kBlockKinds>> by_block;
    void note_block(std::int32_t block, int kind, std::uint64_t bytes) {
        if (block < 0 || block > 4096) return;
        if (by_block.size() <= static_cast<std::size_t>(block)) by_block.resize(static_cast<std::size_t>(block) + 1, {});
        by_block[static_cast<std::size_t>(block)][static_cast<std::size_t>(kind)] += bytes;
    }
};
// display_w x display_h: the display buffers' size (0 x 0: not known).
void render_vram_census_locked(VramCensus& c, std::uint32_t display_w, std::uint32_t display_h);  // render.cpp
void textures_vram_census_locked(VramCensus& c);  // textures.cpp
// The video memory report's lines (gpu.cpp): the driver's count and ours, what
// is held by kind and owner, what caches hold; at exit also each owner's
// highest. BBHOST_VRAM_REPORT=0: vram_report_on() is false and nothing is made.
bool vram_report_on();
std::string video_memory_report_locked(bool at_exit);
// Frees `m` with `image` once the recording in progress has retired.
void defer_destroy_image(VkImage image, const ImageMemory& m);
std::string image_heap_report();
void defer_destroy_view(VkImageView view);
void defer_destroy_private_view(VkImageView view);  // a pass's own view, never in a memo: no epoch bump
// Any direct use of g.queue (a submit, a sparse bind, a wait for idle) holds
// one of these: it waits until the submission thread has submitted all it was
// given - so the work goes behind the game's - then holds g.queue_mu.
void queue_drain();
std::string submit_thread_report();  // for the hang report
std::string submit_merge_report();   // the submission thread's jobs and queue submissions so far (recorder report)
// Writes into write-combined memory the GPU reads (params blocks, constant-
// buffer snapshots) run to the end of their last 64-byte line (gpu.cpp,
// BBHOST_WC_LINES).
extern const bool g_wc_whole_lines;
struct QueueGuard {
    QueueGuard();
    ~QueueGuard();
    QueueGuard(const QueueGuard&) = delete;
    QueueGuard& operator=(const QueueGuard&) = delete;
};
// bindless.cpp: the global descriptor arrays. Under g.mu.
bool bindless_wanted();  // BBHOST_BINDLESS=1
bool bindless_create_locked(std::uint32_t max_images, std::uint32_t max_samplers);
void bindless_dummies_locked(VkImageView image, VkImageView storage, VkSampler sampler);  // slot 0 of each array
std::uint32_t bindless_view_slot_locked(VkImageView view, bool storage);
std::uint32_t bindless_sampler_slot_locked(VkSampler sampler);
void bindless_view_retired_locked(VkImageView view);  // defer_destroy_view: its slot comes free with the view
void bindless_slot_done_locked(int slot);              // a submission slot's retired views destroyed: their slots free
std::string bindless_report();
void profile_begin_locked(const std::string* name);  // timestamp before a draw/dispatch (profiling only)
void profile_end_locked();
void begin_recording_locked();
std::uint32_t alloc_params_slot_locked(const gcn::StageParams& params, VkDescriptorBufferInfo& out);
VkDescriptorSet alloc_set_locked(VkDescriptorSetLayout layout = VK_NULL_HANDLE);  // the shared layout when null
bool rebuild_page_tables();
bool create_dev_buffer(DevBuffer& b, std::uint64_t size, bool host_visible, bool cached = false);  // cached: host-cached memory, for readbacks
// BBHOST_OOB_COUNT (gcn::set_oob_count): the counters' totals by slot and kind
// (kOobSlots x kOobKinds) in 64 bits - each 32-bit counter's change since the
// last call added, so one that wraps between calls still adds up; called once
// a second (host_gpu_stats) and at exit. Empty when not counting.
std::vector<std::uint64_t> oob_totals();
// The same: a compute program's slot named for the exit report, as render.cpp
// names a draw's stages (note_oob_slots).
void note_oob_compute_slot(std::uint32_t slot, const std::string& name);
// A host-visible buffer the CPU writes and shaders read every draw: in video
// memory the CPU can write on a discrete GPU (rings_in_vram), else as
// create_dev_buffer(b, size, true). `what` names it in the log.
bool create_upload_buffer(DevBuffer& b, std::uint64_t size, const char* what);
bool rings_in_vram();
// Whether a ring of `bytes` may go in memory type `type` (video memory the CPU
// writes) by its heap's budget (host/ring_budget.h); logs and counts it as left
// in host memory when not. `note` gets the heap's numbers for the caller's log.
bool ring_fits_budget(std::uint32_t type, VkDeviceSize bytes, const char* what, std::string* note);
// Counts a ring placed in video memory the CPU writes (true) or left in host
// memory by the budget (false), for rings_placement_report.
void note_ring_placed(bool vram, VkDeviceSize bytes);
std::string rings_placement_report();
std::uint32_t find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags want);
bool extract_program(std::uint64_t code_va, std::vector<std::uint32_t>& words, std::string& name);
// A GX shader creator ran (render.cpp; host_gpu_note_shader_created).
void note_shader_created(int stage, const std::uint8_t* container, std::size_t size, std::uint64_t flip);
// Compute shaders compiled when GX creates them (gpu.cpp; on unless
// BBHOST_PRECOMPILE=0): render.cpp registers a creation (true the first time a
// name is seen) and its precompile workers run the job.
bool compute_precompile_enabled();
bool note_compute_created(const std::string& name, std::uint64_t flip);
void precompile_compute(const std::string& name, const std::vector<std::uint8_t>& container, std::uint64_t created_flip);
void compute_precompile_report();
// The pipeline manifest's compute records (render.cpp): a compute pipeline a
// dispatch had to build itself (built_here: a shader GX never created through
// the hook, or one whose bound images differ from the creation's guess) with
// what its key was made from, so a start builds it before the dispatch; a
// dispatch that took one (built_here false) only counts the record's use - the
// shaders GX creates are compiled at their creation already.
void recipe_note_compute(const std::string& name, const std::vector<std::uint32_t>& words, std::uint64_t key, std::uint32_t rsrc1,
                         std::uint32_t rsrc2, const std::uint32_t threads[3], const std::vector<std::pair<std::uint32_t, bool>>& dims,
                         const std::vector<bool>& modes, bool built_here);
// On a precompile worker: a compute record's pipeline, as precompile_compute
// builds one from a container but with the record's images and samplers.
// False when it failed (or another build had its key).
bool precompile_compute_recorded(const std::string& name, const std::vector<std::uint32_t>& words, std::uint32_t rsrc1, std::uint32_t rsrc2,
                                 const std::uint32_t threads[3], const std::vector<std::pair<std::uint32_t, bool>>& dims,
                                 const std::vector<bool>& modes);
// Where guest bytes [va, va + bytes) are, in imported memory: one buffer
// holds them all (imported now when they are direct memory the GPU may reach),
// avail bytes of it from the offset. A null buffer otherwise.
Located locate(std::uint64_t va, std::uint64_t bytes = 1);
// The import of direct-memory chunk `chunk` holding [in, in + bytes) (offsets
// in the chunk), made now when that lies in the chunk's GPU-visible span;
// nullptr when it does not, or the import failed.
// With past_end (and window imports) the range may run on into the next
// chunk's span by up to kChunkOverlap, as one import of this chunk's.
const Chunk* dmem_import(std::uint32_t chunk, std::uint64_t in, std::uint64_t bytes, bool past_end = false);
// Whether direct memory is imported whole (the default on a discrete GPU, and
// under Linux) rather than by window on demand (BBHOST_IMPORT_WINDOWS=1,
// gpu.import_windows in the config, or the default of an integrated GPU under
// Windows; host/import_mode.h). Decided once - at the GPU's
// start, or at an earlier first ask when the environment or config says, else
// once the device is chosen - and read
// inline after that: the command processor asks dozens of times a draw (a
// call to a function-local static was 1.2-1.5% of bb-cp0 at 60 fps).
extern std::atomic<signed char> g_imports_whole;  // -1 until decided, then 1 or 0
bool imports_whole_decide();
inline bool imports_whole() {
    const signed char v = g_imports_whole.load(std::memory_order_relaxed);
    return v >= 0 ? v != 0 : imports_whole_decide();
}
// The windows guest bytes [va, va + bytes) touch, each imported on its own if
// it is not: for what the shaders reach through the page table, which needs
// the pages mapped but no one buffer over them.
void import_windows(std::uint64_t va, std::uint64_t bytes);
// The path whose locate() or import_windows() is under way (a vertex buffer,
// a label's write, a copy token...), for the exit report's census of what
// still takes window imports: set for the call's length, nullptr outside one.
extern thread_local constinit const char* t_import_site;
struct ImportSite {
    const char* prev;
    explicit ImportSite(const char* site) : prev(t_import_site) { t_import_site = site; }
    ~ImportSite() { t_import_site = prev; }
    ImportSite(const ImportSite&) = delete;
    ImportSite& operator=(const ImportSite&) = delete;
};
// BBHOST_IMPORT_AUDIT=1 (gpu.cpp): counts the GPU side's resolutions per guest
// mapping - kind 0 locate, 1 a device address, 2 a vertex/index mirror source.
extern const bool g_import_audit;
void import_audit(const GuestMapInfo& mi, int kind);
// BBHOST_IMPORT_AUDIT=1: a guest range the GPU side reads or writes in place,
// by how (the 64 KiB direct-memory pages, summed per kind and per window).
enum ImportUse { kUseBound, kUsePageTable, kUseUntile, kUseMirror, kUseTransfer, kUseIndex, kImportUses };
void audit_range(std::uint64_t va, std::uint64_t bytes, ImportUse use);
// The device address of guest bytes when they lie in one imported mapping,
// else 0 (gpu.cpp).
VkDeviceAddress guest_device_address(std::uint64_t va, std::size_t bytes);
// locate() without the GX buffers' device-local twins: guest memory's import.
Located locate_guest(std::uint64_t va, std::uint64_t bytes);
// GX buffers the GPU writes, as device resources (BBHOST_GX_TWINS=1; gpu.cpp,
// gx twins). twin_locate_locked: [va, va + bytes) inside one - `out` the twin
// (made and filled from guest memory at its first use), `address` its device
// address when asked; false: guest memory serves it. twin_covers_locked: the
// range overlaps a live twin, whose guest bytes are stale - no CPU copy of
// them. twins_release_locked: a reader of guest memory (`why` 0 a page-table
// walk, 1 a texture) reaches them - copied back and let go.
// GX on Vulkan, stage 1 (host/gxvk.h, gxvk_vk.cpp): GX buffers bound by
// handle. gxvk_resolve_locked: where a vertex record's element or an index
// range [va, va + bytes) of the GX buffer whose holder carried `handle` at
// the call lies in that buffer's device-local copy (placed and filled in this
// recording's uploads at its first binding) - gxvk::kFalls with `out` the
// copy's range, else why not (gxvk::Fallback). A vertex record may count on
// past its buffer's end (its V# counts from the buffer's start, its base an
// offset into it): it is bound all the same, out.avail what the copy holds
// from va - the draw reads only its vertices, inside the buffer (the compare
// mode checks that). gxvk_count_locked counts a
// binding's outcome (kFalls: by handle). gxvk_buffers_bind_mode: whether
// draws bind the copies (BBHOST_GXVK_BUFFERS=1) rather than only check them
// beside today's path (2): gxvk_disagree_locked counts what such a check
// found, and retires the entry.
bool gxvk_buffers_bind_mode();
int gxvk_resolve_locked(int use, std::uint32_t handle, std::uint64_t holder, std::uint64_t va, std::uint64_t bytes, Located& out);
// Whether the last resolve bound a buffer after its end (its holder released,
// its memory unmapped or reused): the copy holds what the draw recorded
// before the end reads, and guest memory may not any more.
bool gxvk_last_resolve_ended();
void gxvk_count_locked(int use, int result);
enum GxvkDisagree : int {
    gxvk_kCopyVersion,   // today's path finds a copy version over the bytes
    gxvk_kWindow,        // ... the bytes in one of the draw's CPU-written windows
    gxvk_kSnapshot,      // ... the bytes on a page the game rewrites (a snapshot)
    gxvk_kLayout,        // the vertex input planned from the handles is not today's
    gxvk_kGuestChanged,  // the buffer's bytes no longer hash as they did when copied
    gxvk_kDeviceCopy,    // the device copy, read back, is not those bytes
    gxvk_kPastEnd,       // a draw reads a record past its buffer's end (its V# counted from the buffer's start)
    gxvk_kDisagrees,
};
void gxvk_disagree_locked(int kind, std::uint32_t handle);
// At a recording's start: retired entries taken back (their ranges free once
// the work recorded with them has run). At a slot's retirement: the compare
// run's read-backs of the copies its uploads made, checked.
void gxvk_begin_recording_locked();
void gxvk_slot_retired_locked(int slot);
std::string gxvk_report();
bool twins_on();
// BBHOST_RUN_COMMIT: at a run's fork, true when no twin is owed a drop (a
// buffer gone or mapped by the CPU since the last lookup) - the twins' miss
// filter then made current, so the parts' lookups only read. False: the fork
// is refused, and the next draw's lookup drops it in order.
bool twins_settled_locked();
bool import_census_on();  // BBHOST_IMPORT_CENSUS=1: the exit report's census of guest memory in place
// BBHOST_GX_TWINS: 0 off, 1 the buffers shaders write, 2 also those GPU copies
// write; 3 the GPU particles' buffers alone (BBHOST_PARTICLE_TWINS: on a
// discrete GPU unless set); -1 until the device is chosen.
int twins_level();
// The device is chosen: level 3 on a discrete GPU (or BBHOST_PARTICLE_TWINS=1),
// else 0 - unless BBHOST_GX_TWINS set it.
void twins_decide(bool discrete);
// Level 3: a dispatch that counts in GDS binds [va...]: the candidate holding
// va is armed when the next recording begins (twins_arm_wanted_locked, from
// begin_recording_locked), its twin made at that recording's first use.
void twins_want_locked(std::uint64_t va);
void twins_arm_wanted_locked();
bool twin_locate_locked(std::uint64_t va, std::uint64_t bytes, Located& out, VkDeviceAddress* address = nullptr);
bool twin_covers_locked(std::uint64_t va, std::uint64_t bytes);
bool twin_candidate_holds(std::uint64_t va, std::uint64_t bytes);  // a twin holds it, or will at its first binding
bool twin_holds_live_locked(std::uint64_t va, std::uint64_t bytes);  // inside a live twin: its stores leave guest memory as it was
bool twin_candidate_straddles(std::uint64_t va, std::uint64_t bytes);  // runs across a candidate's edge
void twin_note_import_near_locked(std::uint64_t va, std::uint64_t bytes, const char* site);  // logs a few imports over a candidate's kind
// A GPU copy (a copy token, a copy version) about to write [dst, dst + bytes):
// with BBHOST_GX_TWINS=2 the GX buffer it lies in becomes a candidate.
void twin_note_copy_target_locked(std::uint64_t dst, std::uint64_t bytes);
void twins_release_locked(std::uint64_t va, std::uint64_t bytes, int why);
// [va, va + bytes) is read by the CPU (an occlusion block): twins over it are
// copied back and let go, and candidates over it let go before they are made.
void twins_keep_in_guest_locked(std::uint64_t va, std::uint64_t bytes);
void twins_buffer_made(std::uint64_t memory, std::uint64_t bytes, std::uint8_t usage, std::uint8_t bind, std::uint64_t gen);
void twins_buffer_mapped_own(std::uint64_t memory, std::uint64_t gen);
void twins_buffer_gone(std::uint64_t memory, std::uint64_t gen);
std::string twins_report();
// The buffer shadow's command buffer for this recording's uploads, run before
// the recording's own work (buffer_shadow.cpp upload_cmd_locked).
VkCommandBuffer shadow_upload_cmd_locked();
// A PS4 tiled texture level untiled on the GPU into linear elements
// (gpu_untile.cpp, shaders/untile.comp), recorded into the current command
// buffer outside a render pass. `mode` 0 linear, 1 1D tiled thin, 2 2D tiled
// thin; esize 4, 8 or 16.
struct UntileGpuPass {
    VkDeviceAddress src = 0, dst = 0;
    std::uint64_t src_slice_bytes = 0, dst_slice_bytes = 0;
    std::uint32_t width_e = 0, height_e = 0, pitch_e = 0, esize = 0, mode = 0, slices = 1;
    std::uint32_t bank_height = 1, aspect = 1, banks = 8;
};
// A depth target's Z plane into an R32_SFLOAT colour image (STORAGE usage,
// GENERAL layout, both) as one compute pass (depth_copy.cpp); false when it
// could not be recorded and the caller copies through a buffer instead.
bool depth_copy_available_locked();
// `depth_layout`: the layout the depth image is in (rt_set_layout_locked), which its sampled descriptor names.
bool depth_copy_record_locked(VkImage depth, VkFormat depth_format, VkImage dst, std::uint32_t width, std::uint32_t height,
                              VkImageLayout depth_layout = VK_IMAGE_LAYOUT_GENERAL);
bool untile_gpu_available_locked();
void untile_gpu_record_locked(const UntileGpuPass& pass);
// A compare run's pair (BBHOST_GPU_UNTILE=2): the GPU's untile and the CPU's
// of the same upload, checked when the slot that wrote them has retired.
struct UntileCheck {
    const std::uint8_t* gpu = nullptr;
    const std::uint8_t* cpu = nullptr;
    std::size_t bytes = 0;
    std::uint64_t base = 0;
};
void untile_checks_run_locked(std::vector<UntileCheck>& checks);
std::string untile_gpu_report();
// buffer_shadow.cpp: a device-local copy of [va, va+bytes) for reading as
// vertex or index data, into `loc`, when one is valid or can be made so now
// (the copy runs in shadow_end_uploads_locked's buffer, ahead of this
// recording); `loc` keeps the import otherwise. A write over a range makes the
// next read copy it again, or - written by the GPU - bind it from the import.
bool buffer_shadow_on();  // BBHOST_BUFFER_SHADOW, else on except on an integrated GPU or with tight memory
// The lite path instead, on an integrated GPU with room (BBHOST_SHADOW_LITE):
// no mirrors, and vertex data a Map handed out lately copied as its draws record.
bool shadow_lite_on();
// BBHOST_RUN_BATCH: the draw draw_impl is resolving (its number on this
// thread, 0 outside one): the lite path drains GX's notices once for it, at
// its first vertex record, instead of at each.
extern thread_local constinit std::uint64_t t_lite_draw;
// BBHOST_RUN_COMMIT: GX's Map notices drained once for a run's parts at its
// fork (the lite shadow's drain, which its draws then skip).
void shadow_lite_drain_locked();
// And GX's texture notices (textures.cpp), which refresh_surface_locked drains
// in order and a run part's draw does not.
void textures_drain_gx_writes_locked();
// BBHOST_RUN_COMMIT=compare: the ring vertex records are copied into as their
// draws record (its places differ between two commits of one draw).
VkBuffer shadow_snap_buffer();

// BBHOST_DRAW_CONTEXT=chunks (docs/list-resolve.md, committing per run, step
// 1): a draw takes its rings' space from chunks a context owns, the way a
// worker committing part of a run will - the GX ring's tables (gx_ring_place),
// the constant-buffer copies' ring with its snapshot memo
// (resolve_stage_buffers) and the vertex copies' ring (buffer_shadow.cpp
// snap_copy_locked) - each chunk taken from its ring as that ring's own
// allocation would take it, and dropped when the recording it was taken for is
// submitted. Off (the default) nothing goes through one.
struct DrawCtx {
    std::uint64_t flushes = ~0ull;             // the recording (g.flushes) the chunks were taken for
    std::uint64_t gx_next = 0, gx_end = 0;     // the GX ring's chunk: addresses
    VkDeviceSize cb_next = 0, cb_end = 0;      // the constant-buffer copies' ring's chunk: offsets
    VkDeviceSize snap_next = 0, snap_end = 0;  // the vertex copies' ring's chunk
    // The constant-buffer snapshots this context made in the recording: a
    // buffer snapshotted again binds its copy (resolve_stage_buffers' memo).
    struct CbMemo {
        std::uint64_t base = 0, bytes = 0, at = 0, serial = ~0ull, lap = 0;
    };
    static constexpr std::size_t kCbMemo = 256;
    CbMemo cb_memo[kCbMemo];
    CbMemo& cb_memo_for(std::uint64_t base) { return cb_memo[((base >> 4) ^ (base >> 14)) & (kCbMemo - 1)]; }
    // A new recording: the chunks are its predecessor's.
    void take_recording(std::uint64_t now) {
        if (flushes == now) return;
        flushes = now;
        gx_next = gx_end = 0;
        cb_next = cb_end = 0;
        snap_next = snap_end = 0;
    }
};
// The context draw_impl allocates through (null outside one, and with the
// switch off).
extern thread_local constinit DrawCtx* t_draw_ctx;
inline DrawCtx* draw_ctx() { return list_resolve::g_draw_ctx || list_resolve::commit_draws() ? t_draw_ctx : nullptr; }
// BBHOST_RUN_COMMIT=compare (render.cpp run_commit): what a draw resolved -
// what it binds and draws - as it would be the same committed in a part or in
// order: what lives in the rings it copies into (whose places differ) by the
// guest memory it came from, and its params blocks with their ring addresses
// left out. Filled as draw_impl records the draw.
struct RunDigest {
    bool done = false;  // it reached its draw call
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet sets[2] = {};
    VkViewport vp{};
    VkRect2D sc{};
    float bounds[2] = {};
    std::uint32_t stencil[6] = {};
    float blend[4] = {};
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VkDeviceSize index_offset = 0;
    std::uint32_t index_type = 0;
    std::uint32_t call_kind = 0, count = 0, instances = 0;
    std::int32_t vertex_offset = 0;
    std::uint64_t vertex = 0;      // its vertex input: each binding's guest address and stride
    std::uint64_t params[2] = {};  // each stage's params block
    std::uint64_t buffers = 0;     // its constant buffers: each binding's guest base, size and outcome
    std::uint64_t first[6][3] = {};  // the first bindings' (base, size, outcome), for the log
    std::uint32_t nbuffers = 0;
    std::uint32_t user[2][16] = {};  // each stage's user data as its params block had it, for the log
};
extern thread_local constinit RunDigest t_run_digest;  // the drawing thread's last draw's

// BBHOST_RUN_COMMIT=on (render.cpp run_commit): a draw of a forked run - one
// a run worker (or the command processor, in its own part) commits off the
// order, or one the command processor commits in order at the join after a
// part stopped before it - and what the fork reserved for it. Null outside one.
// A draw in a part (`part`) takes its params slots from its own reservation and
// fills its own packet, and stops - before any shared side effect - wherever
// draw_impl would change what other draws share; the command processor then
// commits it and the rest of its part itself, in order, at the join.
struct RunDraw {
    std::uint32_t params_next = 0, params_end = 0;  // the params slots reserved for it (alloc_params_slot_locked)
    std::uint64_t packet = 0;                        // its packet's number (DrawCmds::start_reserved)
    std::uint64_t record = 0;                        // its draw record's number (g_draw_recs), with records on
    bool part = false;                               // committed in a part, off the order
    bool started = false;                            // its packet was started (a draw that ends before has none)
    int stop = -1;                                   // why a part stopped at it (render.cpp RunStop), -1: it did not
    int part_index = 0;                              // its part (the join's per-part notes)
    std::uint32_t effects = 0;                       // committed in order: the stops a part would have met in it (bit RunStop)
    bool fresh_packet = false;                       // compare mode's commit in order: a packet of its own, not the reserved one
    RunDigest digest;                                // compare mode: what it resolved in its part
    // What its commit leaves for the join to add where the draws share it: the
    // pipeline it bound (its draw counts) and the vertex-shader census.
    void* drew = nullptr;
    bool drew_gx = false;
    const std::string* vs_other = nullptr;
    int vs_other_why = 0;
};
extern thread_local constinit RunDraw* t_run_draw;
// The most parts a run is split into: the command processor's and a run
// worker's each (BBHOST_RUN_COMMIT_THREADS, at most 16).
constexpr int kRunMostParts = 17;
// Why a run part stopped at a draw (RunDraw::stop): where draw_impl would
// have changed what other draws share, or done something in its order.
enum RunStop : int {
    kRunStopParams,     // more params slots than reserved for it
    kRunStopTargets,    // its targets' inputs differ from the run's first draw's
    kRunStopTables,     // the page tables to rebuild, particle arguments to make
    kRunStopProgram,    // a program, fetch shader, translation or stage plan not made yet, or a lazy field unset
    kRunStopPipeline,   // its pipeline not made, still building, or a variant to queue or create
    kRunStopImages,     // a texture view or sampler not made yet, a region copy
    kRunStopBuffers,    // a constant buffer read through the page table, a vertex record in no import
    kRunStopRefresh,    // a texture not kept fresh in this recording yet (its refresh, maybe an upload)
    kRunStopStore,      // a binding the shader stores to
    kRunStopRing,       // a copy ring's chunk past its lap's segments (a wrap, or a segment whose recording is running)
    kRunStopSets,       // a descriptor set the cache does not hold, or a bindless slot to take
    kRunStopOrder,      // copies to put in place, owed clears, a hazard's barrier, a query, a pass to change
    kRunStopIndex,      // its indices not where the mirrors or the imports already hold them
    kRunStopMirror,     // a mirror's page to copy, its window or the mirror to make, a GX buffer's copy to place, a record to stage
    kRunStopRare,       // anything else off the common path (a table rebuild, a census, a first bind)
    kRunStops
};
// In a run part's draw: stops it for `why` and says so. In a forked run's
// draw committed in order: notes `why` among its effects (what a part would
// have stopped at) and goes on. Outside both: false.
inline bool run_part_stop(int why) {
    if (!list_resolve::commit_draws()) return false;  // (no thread-local read with it off)
    RunDraw* const r = t_run_draw;
    if (!r) return false;
    if (!r->part) {
        r->effects |= 1u << why;
        return false;
    }
    if (r->stop < 0) r->stop = why;
    return true;
}
// The stops whose work, done in order, changes what no other draw is answered:
// a cache filled (programs, translations, plans, input layouts), a texture
// refreshed (only draws that sample it, which stop there too, read what it
// uploads), a copy ring's segment waited for, a params slot past the
// reservation, a copy into device memory made before the recording's draws
// (a mirror's page, a GX buffer's copy: a draw that reads it stops there too).
// After one of those the run's later parts stand.
constexpr std::uint32_t kRunLocalStops =
    1u << kRunStopParams | 1u << kRunStopProgram | 1u << kRunStopRefresh | 1u << kRunStopRing | 1u << kRunStopMirror;
inline bool run_part() {
    if (!list_resolve::commit_draws()) return false;
    const RunDraw* const r = t_run_draw;
    return r && r->part;
}
// A run part's draw that has stopped (its first stop kept).
inline bool run_part_stopped() {
    if (!list_resolve::commit_draws()) return false;
    const RunDraw* const r = t_run_draw;
    return r && r->part && r->stop >= 0;
}
// The copy rings' chunks a run part's draws take (the constant buffers' in
// render.cpp resolve_stage_buffers, the vertex records' in buffer_shadow.cpp
// snap_ring_take_locked) are taken under this lock, and only inside the
// segments the ring's lap already holds for the open recording: a wrap round
// the ring or a new segment, which may wait for the GPU, stops the draw.
extern HostLock g_part_ring_mu;
// Whether the shadow reads told pages in place (BBHOST_SHADOW_IN_PLACE and,
// with window imports, BBHOST_SHADOW_IN_PLACE_WINDOWS): only once the device
// is made, which it is decided by.
bool shadow_in_place_on();
void shadow_locate_locked(std::uint64_t va, std::uint64_t bytes, Located& loc);
// Why the last shadow_locate_locked on this thread left its range where it
// was, or that a mirror serves it (BBHOST_VB_RACE's census and lines).
enum : std::uint8_t {
    kLocNotAsked = 0,  // the shadow off, or not asked
    kLocTwin,          // a GX buffer's device-local twin holds it
    kLocNoMirror,      // in no chunk with a mirror
    kLocInPlace,       // read in place on purpose: told pages (BBHOST_SHADOW_IN_PLACE)
    kLocCrossing,      // across two mirror windows
    kLocServed,        // a mirror serves it
    kLocHost,          // a page in host memory for good, or one the GPU writes in this recording
    kLocRoom,          // no room in the mirrors' share for now
    kLocWindow,        // its mirror window is not made
    kLocNoImport,      // no import to copy it from (whole imports)
    kLocGpuPending,    // a GPU write still to land on a page to copy (window imports)
    kLocNoCopy,        // no upload commands or staging for the copy
    kLocWhys
};
std::uint8_t shadow_locate_why();
// Counts a range bound in place that shadow_locate_locked was not asked
// about again (plan_vertex_input: its mirror question answered kLocInPlace
// for the same range in the same draw).
void shadow_count_placed_bind();
// GX's hand-outs over [va]: bit k when a hand-out of kind k was ever over
// its 4 KiB page - 0 Maps' renamed copies, 1 the upload pool's blocks, 2 ring
// allocations, 3 YEBIS's renames (1-3 with BBHOST_RESOURCE_CENSUS=1) - and
// bit 4 while a renamed copy over its 64 KiB page is outstanding.
unsigned shadow_hand_kinds_locked(std::uint64_t va);
// The same without a host location first: true, `out` the mirror, when the
// mirrors serve [va, va+bytes) whole (copying what they must); false when it
// is read from host memory (an import). Window imports bind meshes this way,
// so their memory needs no import (its copies come through staging).
bool shadow_mirror_locked(std::uint64_t va, std::uint64_t bytes, Located& out);
// A copy of [va, va+bytes) - `loc`, its import, which the mirrors keep in host
// memory - made for one draw in device memory, in the same uploads; `loc`
// becomes the copy. False, `loc` unchanged, for a range past 1 MiB, one the
// GPU writes earlier in this recording, or no room left in the staging chunks.
bool shadow_stage_locked(std::uint64_t va, std::uint64_t bytes, Located& loc);
// The bytes [va, va+bytes) of a page the game keeps rewriting, copied on the
// CPU as the draw records into a ring the GPU reads; `loc` becomes the copy.
// False, `loc` unchanged, for a range on no dynamic page, past 1 MiB, or one a
// GPU write is still to land on (BBHOST_VERTEX_SNAPSHOT).
bool shadow_snapshot_locked(std::uint64_t va, std::uint64_t bytes, Located& loc);
// Whether shadow_snapshot_locked may copy any of [va, va+bytes) - a page of it
// the game rewrites, or a range this cannot tell cheaply - before the caller
// works out how much of it a draw reads.
bool shadow_snapshot_maybe_locked(std::uint64_t va, std::uint64_t bytes);
// A vertex record's or an index range's bytes in no guest-memory import (its
// import failed, or none holds it whole): `out` a copy of them - the mirrors'
// pages copied through staging, else the `read_bytes` the draw reads copied as
// it records into the snapshot ring - so a failed import costs speed, not the
// picture (host/import_fallback.h; BBHOST_IMPORT_COPY=0 leaves such a range).
// Returns the import_fallback::Choice made; import_fallback::served says
// whether `out` holds the range. And its counts, for the imports' exit line.
int shadow_import_fallback_locked(std::uint64_t va, std::uint64_t bytes, std::uint64_t read_bytes, Located& out);
std::string shadow_import_fallback_report();
// What the pages of [va, va+bytes) are to the buffer shadow now, the least
// placeable page deciding (BBHOST_VB_RACE=2's census): read in place, told
// and never handed out (what in place reads; mirrored), under renamed copies
// GX hands out, write-watched, rewritten (host memory for good), not tracked.
enum : int { kPlaceNone = 0, kPlacePlaced, kPlaceTold, kPlaceHanded, kPlaceWatched, kPlaceDynamic, kPlaceUntracked, kPlaceClasses };
int shadow_place_class_locked(std::uint64_t va, std::uint64_t bytes);
void shadow_written_locked(std::uint64_t va, std::uint64_t bytes, bool by_gpu);
// BBHOST_INDIRECT_STATS=1: an indirect dispatch (kind 0) or draw (1, 2
// indexed) recorded now, whose arguments are read once its submission is
// done - the counts GPU work wrote there (the particles' counter copies). Once
// a second the log says what they held. `counting`: a dispatch whose program
// counts in GDS.
void note_indirect_locked(std::uint64_t args_va, int kind, bool counting);
// indirect_clamp.cpp: an indirect dispatch's arguments (src at src_offset,
// guest `va`) copied into a ring slot and checked on the GPU in stream order,
// a count past 65,535 groups a dimension zeroed; the dispatch then reads
// buffer/offset. False when the check is off (BBHOST_INDIRECT_GUARD=0) or
// unavailable: read the arguments in place. `name`: the program, logged once
// if it is ever clamped (when its submission is done).
bool indirect_clamp_locked(VkBuffer src, VkDeviceSize src_offset, std::uint64_t va, const std::string& name, VkBuffer& buffer,
                           VkDeviceSize& offset);
void indirect_clamp_retired_locked(std::uint64_t serial);
std::string indirect_clamp_report();
// BBHOST_GX_PARTICLES (gpu.cpp): how a draw that binds a geometry shader - the
// GPU particles - is drawn: as one vertex stage that runs the export and
// geometry shaders for each vertex (gcn/gs_vertex.h), on every device; =gs, a
// debug switch, with the geometry shader.
enum class ParticlePath : std::uint8_t { kGs, kVs };
ParticlePath particle_path();
// render.cpp: the submission `serial` is done: the particles' argument lists
// its passes read may be written again.
void particle_args_retired_locked(std::uint64_t serial);
// render.cpp: after a dispatch (and its barrier) or a fill, the particles'
// draws' arguments are stale: the next draw writes them from the game's for
// their live counts (particle_args_before_draw_locked; with
// BBHOST_GX_PARTICLE_ARGS=each, written here).
void particle_args_after_dispatch_locked();
void particle_args_before_draw_locked();
// Diagnostic: describes what the mirror holds for a range and whether a page it
// is still serving has been written since this recording's check of it.
void shadow_probe_locked(std::uint64_t va, std::uint64_t bytes, char* out, std::size_t n);
VkCommandBuffer shadow_end_uploads_locked();
// The sparse page binds this recording's copies need, issued on the queue
// (vkQueueBindSparse); returns the semaphore the submission must wait on, or
// VK_NULL_HANDLE when there were none. Call after shadow_end_uploads_locked.
VkSemaphore shadow_bind_sparse_locked();
// The video memory the mirrors may hold, from g.local_budget, said once the
// device is made: what is left above the room the images and targets need.
void shadow_budget_locked();
// Video memory ran short (g.vram_short): every mirror goes into `slot` - the
// submission just made, the last whose work can read them - to be freed when
// it retires, and draws read vertex and index data from host memory from
// then on.
void shadow_give_back_locked(Gpu::Slot& slot);
// Mirrors given back come back once video memory has had room for them a
// while (vram_return_on). Called at each submission that gave nothing back.
void shadow_return_locked();
// The mirrors start over (into `slot`) when a load comes after a shortage of
// room (the live share). Called at each submission.
void shadow_flush_at_load_locked(Gpu::Slot& slot);
// Whether a loading screen is up (host_gpu_set_loading).
bool loading_screen_now();
std::string shadow_report();
// The memory keeper (gpu.cpp start_memory_reserve) runs: the mirrors' memory
// is made on its thread from now on (shadow_keeper_work, each of its turns),
// unless BBHOST_SHADOW_ASYNC=0.
void shadow_keeper_started();
void shadow_keeper_work();
// Wakes the memory keeper for a turn now (gpu.cpp).
void keeper_wake();
// What the mirrors hold, for the video memory report: whole mirrors (a guest
// chunk each, where there is no sparse binding) and sparse 64 MiB blocks, and
// how much of either holds a page copy; the share (cap, 0 with the live
// share, which follows the budget), and whether they were given back.
struct ShadowCensus {
    bool live = false, sparse = false, given_back = false;
    std::uint64_t cap = 0;
    std::uint64_t whole_bytes = 0, block_bytes = 0, data_bytes = 0;
    std::uint32_t whole = 0, blocks = 0;
    std::uint32_t windows = 0;  // 64 MiB windows of the chunks that hold a page copy or a bound page
};
void shadow_vram_census_locked(ShadowCensus& c);
// Draws the mirrors' live share refused room so far (refused copies and the
// draws read from host memory while refused): a sign that video memory the
// image heap's blocks hold idle would be better used by the mirrors.
std::uint64_t shadow_room_refusals_locked();
std::uint64_t fnv1a(const void* data, std::size_t n, std::uint64_t h = 1469598103934665603ull);
void bind_dummy_images(VkDescriptorSet set, const gcn::TranslateResult& meta, std::vector<VkWriteDescriptorSet>& writes,
                       std::vector<VkDescriptorImageInfo>& infos, VkDescriptorImageInfo& smp_info);
// A 1x1 view matching a binding's (spv::Dim, arrayed), for when the T# gives
// nothing usable. Never null for a dimension the translator can emit.
VkImageView dummy_view_for(std::uint32_t dim, bool arrayed, bool zero = false);
// BBHOST_COMPARE_GUARD (on unless 0): a depth-compare binding that resolves to a
// colour view whose format the device cannot compare in (VkFormatProperties3's
// SAMPLED_IMAGE_DEPTH_COMPARISON, which lavapipe's R32F lacks) samples
// a far-plane depth dummy of its shape instead; `view` where the device compares
// in its format, has no such dummy, or the guard is off. tsharp: the binding's T#.
VkImageView compare_guard_view(VkImageView view, const std::uint32_t* tsharp, std::uint32_t dim, bool arrayed);
extern std::atomic<std::uint64_t> g_compare_guarded;

// Guest memory as the translated shaders reach it (draw capture): the host
// bytes and guest address behind a storage-buffer binding over imported
// memory, and whether a 64 KiB guest page is in the GPU page table (pages that
// are not read the sink). Null / false outside imported memory.
const std::uint8_t* imported_bytes_locked(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize bytes, std::uint64_t* guest_va);
bool page_in_gpu_table_locked(std::uint64_t va);

// Fixed-function state of a graphics pipeline, derived from the draw's
// registers (render.cpp): what the pipeline is created with, and what a draw
// capture records.
struct GfxFixedState {
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cull_mode = 0;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depth_clamp = false;
    // PA_CL_CLIP_CNTL's DX_RASTERIZATION_KILL (host/clip_control.h): the pipeline
    // takes rasterizer discard as a dynamic state and the draw sets it.
    bool rasterizer_discard = false;
    bool depth_test = false, depth_write = false, depth_bounds_test = false, stencil_test = false;
    VkCompareOp depth_compare = VK_COMPARE_OP_NEVER;
    VkStencilOpState front{}, back{};
    // The rasterizer state's depth bias: D3D11's and Vulkan's
    // meaning alike - the constant in the depth format's resolvable units, the
    // slope scale, the clamp (0 none).
    bool depth_bias = false;
    float bias_constant = 0.0f, bias_clamp = 0.0f, bias_slope = 0.0f;
    std::vector<VkPipelineColorBlendAttachmentState> blends;  // one per bound colour target, slot order
    std::vector<VkFormat> color_formats;
    std::vector<int> color_slots;                             // CB slot of each attachment
    VkFormat depth_format = VK_FORMAT_UNDEFINED, stencil_format = VK_FORMAT_UNDEFINED;
};

// What a sampled view is (textures.cpp keeps these for draw capture): the
// image it views, how the image was created and the view's own create info.
struct ViewRecord {
    VkImage image = VK_NULL_HANDLE;
    VkImageCreateInfo image_info{};
    VkImageViewCreateInfo view_info{};
    std::uint64_t guest_base = 0;
    bool render_target = false;  // a view of a render target or snapshot image
};
bool describe_view_locked(VkImageView view, ViewRecord& out);
bool describe_sampler_locked(VkSampler sampler, VkSamplerCreateInfo& out);
void forget_view_locked(VkImageView view);  // the view is queued for destruction

// Render target / depth image keyed by guest base address (render.cpp).
struct RtImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    ImageMemory memory;  // from the image heap (rt_image_memory, render.cpp)
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0, height = 0;
    std::uint64_t base = 0;
    bool depth = false;
    bool storage = false;      // created with STORAGE usage (a depth snapshot: depth_copy_record_locked writes it)
    bool initialised = false;  // layout moved to GENERAL
    // Depth targets: the HTILE metadata buffer (DB_HTILE_DATA_BASE). The
    // engine clears depth by filling HTILE; the next draw that binds the
    // target then clears the image to DB_DEPTH_CLEAR / DB_STENCIL_CLEAR.
    std::uint64_t htile = 0;
    bool htile_clear_pending = false;
    // The clear values captured at that fill (DB_DEPTH_CLEAR / DB_STENCIL_CLEAR
    // when the fill dispatch ran); a draw from a token clears to these.
    std::uint32_t htile_clear_depth = 0, htile_clear_stencil = 0;
    // Colour targets drawn through CB_COLOR*_VIEW slices (a cube's six faces,
    // array targets): SLICE_MAX + 1 layers, each `slice_bytes` of guest memory
    // from `base`. `view` is layer 0; layer_views are the other attachments.
    std::uint32_t layers = 1;
    std::uint64_t slice_bytes = 0;
    std::map<std::uint32_t, VkImageView> layer_views;
    // Created VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT: a T# that reads it through an
    // integer format gets a view in that format (render_target_view).
    bool mutable_format = false;
    // Its views limited to rt_view_formats(format) (a VkImageFormatListCreateInfo):
    // the formats RADV keeps delta colour compression for.
    bool format_listed = false;
    // Colour targets: the last write was a fill of guest memory [fill_va,
    // fill_va + fill_bytes) with fill_rgba, and no pass or copy has written
    // the image since. A target re-created over that memory in another shape
    // starts with the fill (rt_image): the game clears, then draws in the new
    // shape, and the clear went to the old image.
    bool fill_last = false;
    float fill_rgba[4] = {};
    std::uint64_t fill_va = 0;
    std::uint64_t pending_seen = 0;  // g_pending_gen when apply_pending_clear last scanned for it (render.cpp)
    std::size_t fill_bytes = 0;
    // A clear token's clear not made yet (render.cpp, BBHOST_GX_CLEAR_LAZY):
    // the next pass that binds the image whole takes it as its load op;
    // anything else that ends a pass (a copy, a dispatch, a readback, a draw
    // that samples it, the submission) makes it first.
    bool view_clear = false;
    bool view_clear_load = false;  // taken by the pass the current draw begins
    VkImageAspectFlags view_clear_aspects = 0;
    VkClearValue view_clear_value{};
    std::uint32_t view_clear_first = 0, view_clear_count = 0;
    // The flip it was made at, and the flip and the time (steady clock, ms) a
    // draw, a fill or a copy last wrote it: its age and order for the reuse
    // retire (image_reuse_sweep_locked). last_write numbers the writes (one
    // counter for every target): what the clear costs compare to tell whether
    // a draw came between two clears.
    std::uint64_t created_flip = 0, written_flip = 0, written_ms = 0, last_write = 0;
    // The submission (hle_gnm_cp_job_seq) that last wrote it, and the
    // generation of the GX resource it is the image of (0: none known) - it
    // goes when that resource does (render_retire_generation_locked).
    std::uint64_t written_seq = 0, gx_gen = 0;
    // Its resource died (steady ms; 0: alive) with this many submissions
    // made, and that resource's memory: retired once idle for the reuse idle
    // time, or taken by the memory's next resource if one draws into it first
    // or is made over it without data (render_retire_reused_locked).
    std::uint64_t dead_ms = 0, dead_after = 0, dead_memory = 0;
    // How its image was made, to make it again elsewhere (a target that went
    // to host memory for want of video memory comes back: render_return_targets_locked).
    VkImageCreateFlags create_flags = 0;
    VkImageUsageFlags usage = 0;
    // A depth target in layouts of its own (Gpu::depth_layouts; render.cpp,
    // rt_set_layout_locked): drawn in DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    // sampled in DEPTH_STENCIL_READ_ONLY_OPTIMAL, cleared by transfer in
    // TRANSFER_DST_OPTIMAL, GENERAL for copies and readbacks - `layout` the one
    // its last use left it in, in recording order. Every other image stays
    // in GENERAL.
    bool managed_layout = false;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};
// The bytes a target covers in guest memory (render.cpp).
std::size_t rt_size_bytes(const RtImage& r);
// Lazy clears (render.cpp): a clear token's clear owed to `r` made now, by
// transfer, before something reads or writes the image; every owed clear
// (a dispatch, the submission, a frame dump). `why` is what made it, for the
// exit report.
enum ViewClearWhy : int {
    kVcwAttachment,  // a draw binds the image, its pass does not cover it whole
    kVcwSampled,     // a draw's vertex or pixel stage samples it
    kVcwTess,        // a tessellated draw (its other stages sample too)
    kVcwOlder,       // a newer clear of the image does not cover the owed one
    kVcwClear,       // another clear of the image (the fill path's, DB_RENDER_CONTROL)
    kVcwCopy,        // a copy reads or writes it
    kVcwTexture,     // a texture view of it is made (render_target_view, rt_copied_view)
    kVcwReadback,    // read back to memory
    kVcwReplaced,    // its image is replaced by another
    kVcwDispatch,    // a compute dispatch
    kVcwSubmit,      // the submission
    kVcwDump,        // a frame dump or capture
    kVcwOther,
    kVcwCount
};
void view_clear_apply_locked(RtImage& r, int why = kVcwOther);
void view_clears_flush_locked(int why = kVcwOther);
// Images over memory the game has given to something newer. On the console a
// render target, a copy's snapshot or a shader-written texture is only memory:
// when the game frees it and makes a newer object there, the old contents are
// gone. Here each is an image kept by its guest address, which nothing removed
// but an image of another shape at the same base - so an area's targets and
// compute textures stayed after the game had moved on, and device memory
// climbed with every new place (1080p: +1.5 GiB of image heap over ten
// places, 4K: a 12 GB card full by the fifth). Once a second
// (image_reuse_sweep_locked, from the submit) an image no draw, fill or copy
// has written for BBHOST_IMG_REUSE_IDLE seconds (10), on memory no live GX
// resource holds whole, goes once a GX resource, target or snapshot made after
// its last write holds part of its memory. BBHOST_IMG_RETIRE_REUSED=0 keeps
// them. All under g.mu.
void image_reuse_sweep_locked();
struct ReuseRetired {
    std::uint64_t n = 0, bytes = 0;
    std::uint64_t dead_n = 0;  // targets and snapshots of a dead GX resource retired once idle (counted apart from n)
};
// Each scan goes on from where the last one stopped and stops at `deadline`,
// so a sweep holds g.mu for a bounded time however many images there are.
// `address_rule`: whether the rule by a newer object over an idle image's
// memory applies (BBHOST_IMG_RETIRE_REUSED); the images of dead GX resources
// (marked by render_retire_generation_locked / retire_surfaces_locked) go once
// idle either way, unless the memory's next resource wrote them first.
ReuseRetired render_retire_reused_locked(std::uint64_t now_ms, std::uint64_t idle_ms,
                                         std::chrono::steady_clock::time_point deadline, bool address_rule);  // render.cpp: targets and snapshots
ReuseRetired textures_retire_reused_locked(std::uint64_t now_ms, std::uint64_t idle_ms,
                                           std::chrono::steady_clock::time_point deadline, bool address_rule);  // textures.cpp: shader-written surfaces
// Whether a target or snapshot made after `after_flip` (other than `self`)
// covers part of [va, va + bytes).
bool render_target_newer_overlapping_locked(std::uint64_t va, std::size_t bytes, std::uint64_t after_flip, const RtImage* self);
// The copies made of the target at `base` for regions a texture read (textures.cpp).
void textures_drop_rt_regions_locked(std::uint64_t base);
// The reuse retire's line for the 5-second report.
std::string image_reuse_report_locked();
// The draws' and clears' target lookups by their views' resource generations (render.cpp rt_take_generation).
std::string draw_gen_report_locked();
RtImage* find_render_target(std::uint64_t base);
// BBHOST_LAYOUT_FIXES (on unless 0; textures.cpp): a target's every layer and
// level leaves UNDEFINED before it is used as GENERAL - a copy into a target
// nothing has made GENERAL yet, a target carried from a texture - which
// g_layout_fixed[0] counts the first kind of.
bool render_layout_fixes();
extern std::atomic<std::uint64_t> g_layout_fixed[4];
// The view formats a colour target of `format` is created for (at most 5, the
// format first): its channel layout's other numeric kinds, which keep RADV's
// delta colour compression (render.cpp).
std::uint32_t rt_view_formats(VkFormat format, VkFormat* out);
const RtImage* rt_overlapping_locked(std::uint64_t va, std::size_t bytes);  // a drawn target over the range, or null

// Ring of recent draws (render.cpp writes one per draw, under g.mu): printed
// with a frame dump and a hang, and read by the glitch hunt (glitch.cpp).
struct DrawRec {
    char name[40];
    std::uint32_t count, inst, prim;
    float vp[4];
    std::int32_t sc[4];
    std::uint64_t rt0, depth, tex0;
    std::uint64_t rt[7];   // colour targets 1-7 (MRT), for the F12 draw file
    std::uint64_t tex[8];  // first eight PS image bases (tex0 repeated in tex[0])
    std::uint32_t tsharp[8][8];  // and their T#s, for the F12 dump's texture table
    std::uint32_t blend0, depth_ctl, mask0;
    std::uint32_t shader_mask, col_format;  // CB_SHADER_MASK, SPI_SHADER_COL_FORMAT
    bool indexed;
    // What the geometry came from: the index buffer, the vertex bindings
    // (vertex-input draws; 0 for a draw that fetches in its shader), the
    // base vertex, and an indirect draw's arguments.
    std::uint32_t index_type;  // 0: 16-bit, 1: 32-bit
    std::int32_t base_vertex;
    std::uint64_t index_va, indirect_va;
    std::uint64_t vb[2];
    std::uint32_t vb_stride[2];
    std::uint32_t dummies;  // images it read that resolved to nothing (a dummy bound instead)
};
// Dummy images bound since the draw being recorded began (render.cpp).
extern thread_local constinit std::uint32_t g_draw_dummies;  // the drawing thread's
// ~8 world frames: the glitch hunt compares a frame's draws with its
// neighbours' a few flips after they were recorded.
constexpr std::size_t kDrawRecs = 16384;
extern DrawRec g_draw_recs[kDrawRecs];
// BBHOST_VB_RACE=1 (render.cpp): each non-indexed draw's vertex range hashed
// when recorded and again when its submission retires - memory rewritten while
// the draw waited for the GPU. =2: the ranges the buffer shadow's mirrors
// serve too (`mirrored`), hashed at their ends - what reading them in place
// would have raced - counted by page class (shadow_place_class_locked). =3:
// those the mirrors serve with window imports too (shadow_mirror_locked), and
// mirrored indices. Ranges read in place are counted by why the mirrors left
// them there (`why`, shadow_locate_why); `holder` is the GX buffer the draw
// bound them by, for the lines. BBHOST_VB_RACE_LOG=all: a line for every
// change, not only the first ones. Caller holds the lock.
bool vb_race_on();
bool vb_race_mirrored_on();
void vb_race_note_locked(const char* name, std::uint64_t va, std::uint64_t bytes, bool mirrored = false, std::uint8_t why = 0,
                         std::uint64_t holder = 0);
void vb_race_retired_locked(std::uint64_t serial);
void vb_race_exit_report_locked();  // the counts at exit, by page class too (a short run's only line)
// buffer_shadow.cpp: GPU writes by page (every shadow_written_locked by the
// GPU, the shadow on or off), and whether one over [va, va+bytes) is still to
// land - recorded in a submission that has not retired. Caller holds the lock.
void gpu_writes_note_locked(std::uint64_t va, std::uint64_t bytes);
bool gpu_writes_pending_locked(std::uint64_t va, std::uint64_t bytes);
// Whether a GPU write over [va, va+bytes) was recorded in recording `serial` or later.
bool gpu_wrote_since_locked(std::uint64_t va, std::uint64_t bytes, std::uint64_t serial);
// The newest recording whose GPU writes the tables above have let go of
// (they keep only writes still to land when they fill): gpu_wrote_since_locked
// cannot answer for a serial at or before it.
std::uint64_t gpu_writes_dropped_locked();
// g.completed_submits as last stored, for readers without the lock (gpu.cpp).
extern std::atomic<std::uint64_t> g_completed_submits_seen;
extern std::uint64_t g_draw_rec_next;

// A compute shader's pipeline at its first use, with or without the driver's
// optimizer as BBHOST_CS_OPTIMIZE says (gpu.cpp). `standin`: it is the
// unoptimized one, and the optimized one is being built on a bb-cs-optimize
// thread under `opt_key`; adopt_optimized_compute swaps it in. `watch`: the
// name the watchdog lists it under while it compiles. `layout`: null for
// Gpu::pipe_layout.
VkPipeline first_compute_pipeline(VkShaderModule module, std::uint64_t opt_key, const std::string& name,
                                  const std::vector<std::uint32_t>& spirv, bool& standin, const std::string& watch,
                                  VkPipelineLayout layout = VK_NULL_HANDLE);
// A stand-in's optimized pipeline, when it is built: true when `pipeline` now
// is it (and `standin` false). Cheap when nothing new was built since `seen`.
bool adopt_optimized_compute(std::uint64_t opt_key, VkPipeline& pipeline, bool& standin, std::uint64_t& seen);
// The exit report's lines on it, and the shaders whose optimized build was
// still running at exit after a minute (kept unoptimized at the next start).
void compute_optimize_report();
std::string compute_optimize_window();    // the 300-flip report's piece; empty when nothing happened
std::uint64_t compute_optimized_built();  // optimized builds so far: the periodic cache save looks for new ones
// The watchdog's line on compute pipelines still compiling (gpu.cpp), their
// SPIR-V written to the data directory's tmp.
void compute_compiling_report();

// ---- the glitch hunt (glitch.cpp, BBHOST_GLITCH=1) ----
// Finds the frames that flash or stretch and names the draws behind them:
// every presented frame is box-filtered on the GPU and compared with both
// its neighbours, and every draw's covered samples come from an occlusion
// query and are compared with the same draw's in the frames either side.
// The GPU busy meter (gpu_busy.cpp): the union of the time our command
// buffers spent on the GPU, a timestamp at each one's start and end.
void busy_init_locked();                           // after the slots exist: the query pools, or off
void busy_begin_locked(Gpu::Slot& sl);             // a recording's first command
void busy_end_locked(Gpu::Slot& sl);               // its last, before vkEndCommandBuffer
void busy_upload_begin_locked(Gpu::Slot& sl, VkCommandBuffer cmd);  // the uploads ahead of it: their start
void busy_submitted_locked(Gpu::Slot& sl);         // it is on its way to the queue
void busy_retired_locked(Gpu::Slot& sl, bool ok);  // its fence signalled (ok), or it never will
std::string busy_exit_report();                    // the whole run's line, for host_gpu_report
struct DrawCall;
bool glitch_on();
VkQueryControlFlags glitch_query_flags();                     // precise when the device has it
void glitch_device_features(VkPhysicalDeviceFeatures& enable);  // at device creation
void glitch_slot_init_locked(Gpu::Slot& sl);                  // after the device is made
void glitch_begin_recording_locked();                         // a recording's first commands
void glitch_draw_locked(DrawCall& call);                      // a query for draw g_draw_rec_next
void glitch_slot_done_locked(Gpu::Slot& sl);                  // its fence has signalled
std::string glitch_report();

// ---- occlusion queries and predication (occlusion.cpp) ----
// The PS4's depth-block sample counters behind ZPASS_DONE dumps, from Vulkan
// occlusion queries, written into guest memory on the GPU's timeline; and
// SET_PREDICATION, on the CPU when the result is known, else by conditional
// rendering. BBHOST_OCCLUSION=0: the old fake (every query reads visible).
bool occlusion_on();
void occlusion_device_features(VkPhysicalDeviceFeatures& enable, std::vector<const char*>& exts, void** chain);  // before vkCreateDevice
void occlusion_device_ready_locked();     // after the slots are made
void occlusion_begin_recording_locked();  // a recording's first commands
bool occlusion_draw_entry();              // host_gpu_draw's start: false when SET_PREDICATION culls the draw
// BBHOST_RUN_COMMIT (render.cpp run_commit): whether a run's parts may commit
// draws - no query interval open, no predication that skips draws - and the
// draws they committed, counted as issued at the join (occlusion_draw_entry's).
bool occlusion_run_ok();
void occlusion_run_issued(std::uint64_t n);
void occlusion_draw_locked(DrawCall& call);
void occlusion_submit_locked();  // before the submission's label writes
void occlusion_record_cond(VkCommandBuffer cmd, const DrawCall& call, bool begin);
std::string occlusion_report(bool totals);
// BBHOST_GLITCH_WATCH=<pipeline prefix>[,...]: the guest ranges those
// pipelines' draws bind are hashed when recorded and again once the GPU has
// run them, and a spike's report says whether its inputs changed in between.
// BBHOST_GLITCH_SNAPSHOT=1 binds copies taken at record time instead: the
// experiment that says whether a race is what spikes.
void glitch_watch_draw_locked(const std::string& pipeline);  // before a draw's bindings are resolved
bool glitch_watching_draw();                                  // the draw being resolved is watched
bool glitch_snapshot_draw();                                  // ... and its buffers are to be copied
// kind: 0 a buffer binding (constants, storage), 1 the index buffer, 2 a tessellated LS's fetch.
void glitch_watch_read_locked(std::uint64_t base, std::uint64_t bytes, int kind = 0);
// ... and what the watched draw was given, as text for a spike's report: a
// stage's params as the shader sees them, with the tables its user data
// points at, and each buffer binding (render.cpp's resolve_stage_buffers).
void glitch_watch_params_locked(const gcn::StageParams& params, int stage);
void glitch_watch_note_locked(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
// Walks a ResourcePath with the stage's user SGPRs and guest memory;
// `touched`, when given, collects the guest ranges (address, bytes) it read.
bool resolve_resource(const gcn::ResourcePath& path, const std::uint32_t* user, int ndw, std::uint32_t* out,
                      std::vector<std::pair<std::uint64_t, std::uint64_t>>* touched = nullptr);
// BBHOST_WATCH_CHANGED_CB: the constant buffer the renderer is watching (0 when
// none), so copy dispatches into it can be logged.
inline std::atomic<std::uint64_t> g_watch_cb_lo{0}, g_watch_cb_hi{0};

// Shader liveness (shader_liveness.cpp): bit n set when the translated module
// still reads the user-data table or block at slot n after dead-code
// elimination. False, with every bit set, when that cannot be told.
// `any`, when given: bit n set when the module touches user_sgpr[n] at all,
// as an address or as data (every bit when that cannot be told).
bool table_slots_read(const gcn::TranslateResult& meta, std::uint32_t& mask, std::uint32_t* any = nullptr);
// Rewrites a params slot alloc_params_slot_locked handed out, before the
// draw's sets are bound.
void rewrite_params_slot_locked(const gcn::StageParams& params, const VkDescriptorBufferInfo& slot);

// `gx`: a GX draw's records for the stage; bindings they cover come from
// them instead of user data.
std::vector<bool> sampler_modes_for(const gcn::TranslateResult& meta, const std::uint32_t* user, const GxStageRecords* gx = nullptr);
// A stage's images and samplers, resolved once per draw or dispatch.
struct StageImages {
    struct Image {
        std::uint32_t w[8]{};  // the T# (or r128 V#) words
        VkImageView view = VK_NULL_HANDLE;
        std::uint32_t dim = 0;
        bool arrayed = false;
        bool resolved = false;  // the resource path was readable
        std::uint32_t slot = 0;  // stage 2, bindless: the view's slot from its entry (0: looked up when bound)
        std::uint64_t base = 0;  // stage 2: the T#'s base from the view's entry (0: tsharp_base(w) when wanted)
    };
    std::vector<Image> images;        // parallel to TranslateResult::images
    std::vector<VkSampler> samplers;  // parallel to TranslateResult::samplers; null when unresolved
};
// Resolves and uploads/creates every texture and sampler the stage needs (may
// flush), so that the later bind step never invalidates freshly allocated
// sets. `stage_user` are the stage's 16 user SGPRs.
// One stage's bindings as the pipeline key resolves them, over paths_for's
// lists: the image dimensions and sampler modes a translation depends on, and
// the words each binding resolved to. The prefetch takes the words when the
// pipeline's translation lists the same bindings (key_stage_matches), so a draw
// resolves each binding once.
struct KeyStage {
    struct Words {
        std::uint32_t w[8]{};
        bool resolved = false;
    };
    std::vector<std::pair<std::uint32_t, bool>> dims;  // parallel to paths_for's images
    std::vector<bool> modes;                           // parallel to its samplers
    std::vector<Words> images, samplers;
};
// Fills `out` for a stage; `paths` is null for a stage without a program. The
// vectors keep their storage from draw to draw.
void resolve_key_stage(const gcn::TranslateResult* paths, const std::uint32_t* stage_user, const GxStageRecords* gx, KeyStage& out);
// True when `meta`, a pipeline's translation, lists the image and sampler
// bindings of `paths` in the same order.
bool key_stage_matches(const gcn::TranslateResult& meta, const gcn::TranslateResult& paths);
// `resolved`: the pipeline key's words for these bindings, used instead of
// resolving them again.
void prefetch_stage_images(const gcn::TranslateResult& meta, const std::uint32_t* stage_user, StageImages& out,
                           const GxStageRecords* gx = nullptr, const KeyStage* resolved = nullptr);
// Writes what prefetch_stage_images resolved into the set: real views where
// there are any, a dummy of the right shape otherwise.
void bind_stage_images(VkDescriptorSet set, const gcn::TranslateResult& meta, const StageImages& pre,
                       std::vector<VkWriteDescriptorSet>& writes, std::vector<VkDescriptorImageInfo>& infos,
                       const char* pipeline_name);
// Default translation of a program (resource paths only), cached by code hash.
const gcn::TranslateResult& paths_for(std::uint64_t hash, const std::vector<std::uint32_t>& words, gcn::Stage stage,
                                      std::uint32_t rsrc1, std::uint32_t rsrc2);
// Image dimensions (spv::Dim, arrayed) per binding, from the resolved T#s.
std::vector<std::pair<std::uint32_t, bool>> image_dims_for(const gcn::TranslateResult& meta, const std::uint32_t* stage_user,
                                                           const GxStageRecords* gx = nullptr);

// textures.cpp
VkImageView texture_view(const std::uint32_t* tsharp, bool storage, bool& dims_ok, std::uint32_t& dim, bool& arrayed,
                         bool shader_arrayed = false, bool r128 = false);
// The same for a T# a shader-resource view holds, by the view's
// id (GxStageRecords::obj_tex_id): its T# never changes, so the id stands for
// the words and nothing of them is hashed or compared.
// `slot` and `base`, when given: the view's bindless slot and its T#'s base from its entry (stage 2; 0 when it has
// none).
VkImageView texture_view_by_id(std::uint32_t id, const std::uint32_t* tsharp, bool storage, bool& dims_ok, std::uint32_t& dim,
                               bool& arrayed, bool shader_arrayed = false, bool r128 = false, std::uint32_t* slot = nullptr,
                               std::uint64_t* base = nullptr);
std::string texture_view_by_id_report();
// The binding variant texture_view_by_id keys its entries by, as it works it out from its flags.
inline std::uint8_t texture_view_flags(bool storage, bool shader_arrayed, bool r128) {
    return static_cast<std::uint8_t>((storage ? 1 : 0) | (shader_arrayed ? 2 : 0) | (r128 ? 4 : 0));
}
// Stage 2 (BBHOST_GXVK_IMAGES >= 1): the pipeline key's dims (image_dim_of) for a binding of a GX view, from the view's
// entry when it is live and has them; false: decode the T#, and give a non-depth binding's dims to
// texture_key_dims_note, which keeps them in the entry for the view's later bindings.
bool texture_key_dims_by_id(std::uint32_t id, std::uint8_t flags, bool depth, std::uint32_t& dim, bool& arrayed);
void texture_key_dims_note(std::uint32_t id, std::uint8_t flags, std::uint32_t dim, bool arrayed);
// BBHOST_GXVK_IMAGES as a number: 1 (the default) stage 2's handles, 0 (or census) today's path, 2 both, compared.
int textures_images_mode();
// Asks for the entry texture_view_by_id will read for `id` (the table is
// larger than the caches; a draw asks for its views' entries together).
void texture_view_by_id_prefetch(std::uint32_t id);
// Stage 2a's census (gxvk::census()): an image binding prefetch_stage_images resolved - source 0 by its
// shader-resource view's id, 1 from its T# in a GX draw, 2 from its T# in another draw - and the view it got (null:
// none), classed by the surface the lookup found; counted, and reported with textures_report.
void textures_census_binding(int source, VkImageView view);
void textures_census_report();
// Stage 2's census (BBHOST_GXVK_IMAGES=census): a target a draw binds, colour or depth, and whether a GX view the
// registry saw made names it (its resource's generation known), and whether target_image's slot memo answered.
void textures_census_target(int depth, bool gx_view, bool memo_hit);
VkSampler sampler_for(const std::uint32_t* ssharp, bool compare);
// BBHOST_SAMPLER_FIXES (on unless 0; textures.cpp): a sampler's state within
// what the device allows - MIRROR_CLAMP_TO_EDGE with its feature (else
// MIRRORED_REPEAT), maxLod at or above minLod, the LOD bias within
// maxSamplerLodBias.
bool sampler_fixes_on();
// spv::Dim of a T# type as the translator is told it: DimCube for a cube, which
// the translator then declares, and texture_view binds, as a 2D array of faces.
std::uint32_t tsharp_dim(std::uint32_t type, bool& arrayed);
// What a T#'s texels read as: 0 float, 1 unsigned integer, 2 signed integer -
// the class of the Vulkan format it is viewed in. The translator types the
// image to match (TranslateOptions::image_dims, bits 8-9 of the dim).
std::uint32_t tsharp_kind(const std::uint32_t* w);
std::size_t format_bytes_per_pixel(VkFormat format);
void textures_report();
// What is resolving textures on this thread (a draw token's kind, "dispatch",
// "upload token", ...), for the census of surfaces no GX resource covers;
// set for a scope by ResolveKind (textures.cpp defines it).
extern thread_local constinit const char* t_resolve_kind;
// The host image of one GX resource (engine/gx_resources.h: its memory and
// generation), a render target's, a copy's snapshot's or a texture surface's:
// what a copy between resources can copy image to image. False when the host
// has none of that generation there (a surface of another generation, or none).
struct HostImageRef {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0, height = 0, layers = 1, levels = 1;
    bool target = false;         // a render target or snapshot (RtImage), else a texture surface
    bool depth = false;          // a depth target
    bool gpu_written = false;    // its image holds what the guest memory does not
};
bool host_image_by_resource_locked(std::uint64_t memory, std::uint64_t gen, HostImageRef* out);
// A GX texture copied whole as memory whose source surface a shader or copy
// wrote: copied image to image into the destination resource's target or
// surface (textures.cpp). 1 copied, 0 not that case (counted by reason).
int textures_copy_resource_locked(std::uint64_t src, std::uint64_t src_gen, std::uint64_t dst, std::uint64_t dst_gen);
std::string textures_resource_copy_report();
// BBHOST_COPY_BY_RESOURCE (on unless 0, and with BBHOST_RES_IDENTITY): copies
// and texture binds take a target by its resource's generation (gpu.cpp).
bool copy_by_resource_on();
// Images belong to their GX resource's generation (BBHOST_RES_IDENTITY, on
// unless 0): retired with it, never handed to a later resource at the same
// memory (textures.cpp).
bool res_identity_on();
// The colour and depth targets and the snapshots in [base, base + bytes)
// that are generation `gen`'s images: marked dead, and retired by the reuse
// sweep once idle (render.cpp).
// A target something wrote in a submission made after `after_submitted` (the
// count when the resource died) has a newer owner: it stays, and takes the
// generation of the resource that now holds its memory.
void render_retire_generation_locked(std::uint64_t base, std::size_t bytes, std::uint64_t gen, std::uint64_t after_submitted);
struct ResolveKind {
    const char* prev;
    explicit ResolveKind(const char* kind) : prev(t_resolve_kind) { t_resolve_kind = kind; }
    ~ResolveKind() { t_resolve_kind = prev; }
    ResolveKind(const ResolveKind&) = delete;
    ResolveKind& operator=(const ResolveKind&) = delete;
};
// The 5-second report's resource line(s): the write watch's faults, arms and
// releases by owner, the hashes, the surfaces by registration and the census
// of new ones; with BBHOST_RESOURCE_CENSUS=1 also the images whose memory no
// live GX resource holds (textures.cpp, render.cpp). Under g.mu.
std::string textures_census_window_locked();
struct OrphanImages {
    std::uint64_t n = 0, bytes = 0;            // no live GX resource over any of its memory
    std::uint64_t partial = 0, partial_bytes = 0;  // one over part of it, none over all
    std::uint64_t gone = 0, gone_bytes = 0;    // the image of a GX resource (generation) that is no longer live
    std::uint64_t yebis = 0, yebis_bytes = 0;  // of `n`, on memory YEBIS's own device made (BBHOST_RESOURCE_CENSUS)
    std::uint64_t scaleform = 0, scaleform_bytes = 0;  // of `n`, on a Scaleform texture's (the same)
};
// Colour and depth targets [0], snapshots [1] whose memory no live GX
// resource covers (render.cpp).
void render_orphans_locked(OrphanImages out[2]);
// Drops the descriptor-keyed view memo in textures.cpp: call it wherever an
// image view, a surface or a render target is created or destroyed, so a
// remembered descriptor cannot resolve to a view that is gone or stale.
void bump_view_epoch();
std::string view_epoch_sites_report();  // BBHOST_EPOCH_SITES=1
std::uint64_t view_epoch();  // moves whenever a view or image is let go (textures.cpp)
std::uint64_t textures_upload_count();
void textures_time_us(std::uint64_t out[5]);  // image+memory, staging, untile, record, views
std::uint64_t textures_hash_count();
std::uint64_t textures_hash_us();
std::uint64_t textures_replacements();
// A T#'s base address in bytes (word0 plus word1[7:0], << 8). BASE_ADDRESS_HI
// is documented as address bits but this title writes 0x40 there on some
// descriptors whose real base is in word0; those fall back to word0 << 8.
// A mapped unshifted word0 is a V#, not a T# — do not use it as an image base.
std::uint64_t tsharp_base(const std::uint32_t* w);
// A prefetched image's T# base: its view entry's (stage 2), else decoded from its words.
inline std::uint64_t image_base(const StageImages::Image& im) { return im.base ? im.base : tsharp_base(im.w); }
// shadPS4 Image::GetBaseType: Color2DArray / Color2DMsaa / Color2DMsaaArray
// sample as Color2D. A single-layer type-13 T# at a colour-target address is
// the YEBIS ping-pong; treating it as a 2D array made the SPIR-V shape miss
// the aliased 2D view.
bool tsharp_sample_as_2d(std::uint32_t type, std::uint32_t base_array, std::uint32_t last_array,
                         bool shader_arrayed = false);
// GPU work overwrote this range: cached surfaces covering it must be re-read
// from guest memory at their next use (the periodic content hash is too slow
// for a texture the game rewrites every frame). Caller holds the GPU lock.
// Returns how many cached surfaces the range overlaps.
// `memory_written`: false for a fill, which reaches images (a target, HTILE,
// the textures it clears) and never the guest memory, and for a copy made
// image to image: the surfaces under it are not marked, so nothing hashes
// their memory for a change that is not there.
// `source` names the writer for the census of what marks GX textures.
int mark_surfaces_dirty_locked(std::uint64_t va, std::size_t bytes, bool memory_written = true, const char* source = "other");
// Whether a cached surface overlaps the range. Caller holds the GPU lock.
bool surfaces_may_cover_locked(std::uint64_t va, std::size_t bytes);
// GX's dynamic-buffer copy-backs as copy versions (gpu.cpp): the bytes
// go into a staging copy at the token, the draws after it bind that copy, and
// the copies into place wait for the next transfer batch, dispatch or
// submission - no pass end or barrier at the token. False when this copy must
// go to the GPU as it stands.
bool copy_version_make_locked(std::uint64_t dst, std::uint64_t src, std::uint64_t bytes);
// The pending versions' copies into place, recorded once no pass is open.
void copy_versions_flush_locked();
// A render pass ended: the flush, when they wait for that (BBHOST_COPY_VERSIONS=1).
void copy_versions_pass_end_locked();
// The copies into place now, the pass ended for them if one is open: a
// binding across a version's edge, a page-table read over one, a pipeline
// that walks the page table.
enum { kCvStraddle, kCvPageTable, kCvWalks, kCvRetired, kCvPutTwin, kCvPutReasons };
void copy_versions_put_in_place_locked(int why);
// Whether a pending version's copy into place writes `buffer` (a window import
// about to retire: its copies go into place first).
bool copy_versions_name_buffer_locked(VkBuffer buffer);
// The same for [lo, hi) of `buffer` (a GX twin's slice about to be let go).
bool copy_versions_into_locked(VkBuffer buffer, VkDeviceSize lo, VkDeviceSize hi);
// Whether a pending version's destination overlaps guest [lo, hi).
bool copy_versions_overlap_locked(std::uint64_t lo, std::uint64_t hi);
// A binding of [base, base + bytes) against the pending versions:
// kCopyVersionInside (`out` is the copy to bind), kCopyVersionStraddles (it
// covers part of one: put them in place first), or kCopyVersionNone.
enum { kCopyVersionNone, kCopyVersionInside, kCopyVersionStraddles };
int copy_version_lookup_locked(std::uint64_t base, std::uint64_t bytes, Located* out);
// Draw captures take a binding's bytes from the guest memory it imports. A
// binding resolved to a copy of guest memory taken as the draw records (a
// YEBIS window's, the aligned scratch) is noted with the guest address its
// first byte stands for (gpu.cpp keeps the last few); 0 when none is noted.
void note_copy_binding_locked(const VkDescriptorBufferInfo& info, std::uint64_t guest);
std::uint64_t copy_binding_guest_locked(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range);
// The bytes of [base, base + bytes) as this point in the stream has them, over
// `to` (which holds guest memory's): for a snapshot taken as a draw records.
void copy_versions_overlay_locked(std::uint64_t base, std::uint64_t bytes, std::uint8_t* to);
bool copy_versions_pending_locked();
std::string copy_versions_report();  // "" when none were made or refused
bool render_pass_open_locked();
std::string render_barriers_report();  // "" unless BBHOST_PASS_BARRIERS=lazy
// Clears the sampled textures a fill covers whole, where no render target took
// it (textures.cpp); how many.
int textures_fill_surfaces_locked(std::uint64_t va, std::size_t bytes, const float rgba[4]);
extern std::atomic<std::uint64_t> g_fill_surfaces, g_fill_surfaces_skipped;
// Forget the cached sampled views of a render target before its image dies.
void invalidate_rt_views(std::uint64_t base);
// Depth layouts (render.cpp; RtImage::managed_layout): `r` taken to `want`
// for its next use - a barrier from the layout it is in, every aspect and
// layer, recorded with no pass open (one open is ended); nothing for an image
// not managed or already there. `discard`: its contents are not wanted (a
// clear of all of it follows).
void rt_set_layout_locked(RtImage& r, VkImageLayout want, bool discard = false);
// The same for a VkImage when it is a managed target's (diagnostic readbacks
// that know the image alone).
void rt_image_layout_general_locked(VkImage image);
// A sampled view of a managed depth target (render_target_view) and the
// layout descriptors of it name; GENERAL for every other view.
void note_managed_view(VkImageView view, std::uint64_t base);
void forget_managed_view(VkImageView view);
VkImageLayout sampled_view_layout(VkImageView view);
// The managed depth targets a draw's or dispatch's stages sample, taken to
// their read-only layout before its pass begins (a pass open is ended).
// `attachment`: the draw's own depth target, which a sampled image must not
// be (it leaves the managed layouts then).
void depth_layouts_for_images_locked(const StageImages* stages, int nstages, RtImage* attachment);
std::string depth_layouts_report();  // "" when off
// Marks where the GPU is: on a device loss the driver reports the last
// checkpoint each queue stage reached, which names the draw that faulted.
// kind 1 = draw, 2 = dispatch, kGpuMarkSubmission = a submission's edges:
// index = serial * 4 + 0 its buffer-shadow uploads (they run after the wait
// for their sparse binds), 1 its recording's start, 2 its recording's end -
// so a loss between one frame's last draw and the next one's first says
// whether the GPU was in a submission's tail, its wait or its uploads.
// AMD's markers go through the stream (an op, or a draw's packet:
// DrawCmds::marker), NVIDIA's checkpoints in place.
constexpr unsigned kGpuMarkSubmission = 3;
void gpu_checkpoint(unsigned kind, std::uint64_t index);
void gpu_checkpoint_cmd(VkCommandBuffer cmd, unsigned kind, std::uint64_t index);
// AMD's buffer markers: the 32-bit value for a kind and index, and the two
// writes (the top of the pipe, then the bottom) into g.marker_buf.
std::uint32_t gpu_marker_value(unsigned kind, std::uint64_t index);
void gpu_write_markers(VkCommandBuffer cmd, std::uint32_t value);
// Logs one recorded draw by index, if it is still in the ring.
void describe_draw_record(std::uint64_t index);
// A device fault's address against the tessellation LDS ring (render.cpp):
// inside it (which region) or how far past or before it, and the recent
// tessellated draw whose LDS holds it or ends just before it. "" before the
// ring exists.
std::string tess_lds_describe(std::uint64_t address, std::uint64_t precision);
void surface_queue_writeback(std::uint64_t base);  // after a storage-image dispatch; guest memory is not written
bool texture_dump_locked(std::uint64_t base, const char* path, std::uint32_t level = 0, std::uint32_t layer = 0);  // the surface at `base` (blitted to RGBA8) as a PNG or PPM, by the path
// One line on the sampled surface at `base` - its image, its source layout and
// how it has been kept up to date - for the F12 dump's texture table; empty
// when there is none.
std::string texture_describe_locked(std::uint64_t base);
// A ring of the events that decide what a texture holds - storage dispatches,
// copies and fills, render targets, surface replacements and uploads - each
// with the guest range it touched. The F12 dump lists, under every texture it
// describes, the events that touched that texture's memory (tex_events_for),
// then the most recent ones (tex_events_write). printf-style, but only the
// arguments are kept: the text is made when a dump asks for it (making it at
// every fill and copy was 0.4% of the command processor). The format is
// checked as printf's; the kept kinds follow from its conversions.
struct TexEventArgs {
    static constexpr unsigned kArgs = 16, kStrings = 96;
    std::uint64_t args[kArgs];
    char strings[kStrings];  // the %s arguments, each ending in a 0; args hold their offsets
    unsigned n = 0, used = 0;
    TexEventArgs() { strings[kStrings - 1] = 0; }
    template <class T> void put(T v) {
        if constexpr (std::is_same_v<T, const char*> || std::is_same_v<T, char*>) {
            put_string(v);
        } else if constexpr (std::is_floating_point_v<T>) {
            const double d = static_cast<double>(v);
            std::uint64_t bits;
            std::memcpy(&bits, &d, sizeof(bits));
            put_bits(bits);
        } else if constexpr (std::is_enum_v<T>) {
            put(static_cast<std::underlying_type_t<T>>(v));
        } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
            put_bits(static_cast<std::uint64_t>(static_cast<std::int64_t>(v)));
        } else if constexpr (std::is_integral_v<T>) {
            put_bits(static_cast<std::uint64_t>(v));
        } else {
            static_assert(std::is_pointer_v<T>, "tex_event: an argument printf would not take");
            put_bits(reinterpret_cast<std::uintptr_t>(v));
        }
    }
    void put_string(const char* s) {
        if (used >= kStrings) {
            put_bits(kStrings - 1);  // always 0
            return;
        }
        put_bits(used);
        while (s && *s && used < kStrings - 1) strings[used++] = *s++;
        strings[used++] = 0;
    }
    void put_bits(std::uint64_t b) {
        if (n < kArgs) args[n++] = b;
    }
};
void tex_event_keep(std::uint64_t lo, std::uint64_t bytes, const char* fmt, const TexEventArgs& args);
template <class... A> void tex_event_args(std::uint64_t lo, std::uint64_t bytes, const char* fmt, A... a) {
    TexEventArgs k;
    (k.put(a), ...);
    tex_event_keep(lo, bytes, fmt, k);
}
inline void tex_event_format(const char*, ...) __attribute__((format(printf, 1, 2)));
inline void tex_event_format(const char*, ...) {}
#define tex_event(lo, bytes, ...)                              \
    do {                                                       \
        if (false) ::gpu::tex_event_format(__VA_ARGS__);       \
        ::gpu::tex_event_args((lo), (bytes), __VA_ARGS__);     \
    } while (0)
void tex_events_for(std::FILE* f, std::uint64_t lo, std::uint64_t bytes);
void tex_events_write(std::FILE* f);
// The guest bytes a cached surface covers (0 when there is none).
std::size_t texture_src_bytes_locked(std::uint64_t base);
void textures_before_submit();                      // the uploads this submission copies are untiled (their prep pool jobs done)
void textures_prewait_for_submit();                 // without g.mu: the same wait for the jobs queued so far, so the one under it is short
// Creation-time upload and destruction (see gpu.h's entry points).
bool textures_create_ahead_locked(const std::uint32_t* tsharp, std::uint64_t gen = 0);
bool textures_upload_region_locked(std::uint64_t base, const std::uint32_t* tsharp, std::uint32_t mip, std::uint32_t layer, std::uint32_t x,
                                   std::uint32_t y, std::uint32_t w, std::uint32_t h, const void* data, std::size_t bytes, std::uint32_t row_bytes);
bool textures_copy_image_region_locked(const GpuImageCopy& c);
// A colour target created where a shader stored a texture of its size and
// texel size: that texture's level 0 is copied into the target's layer 0
// (rt_image). False when there is no such texture.
bool textures_carry_into_target_locked(std::uint64_t base, std::uint32_t width, std::uint32_t height, std::size_t bytes_per_pixel,
                                       VkImage target);
void textures_retire(std::uint64_t base, std::size_t bytes, std::uint64_t after_submitted, std::uint64_t gen = 0);
// The generation of the live GX texture that takes a dead resource's image
// (0: none, and the image goes): one made without data at the dead
// resource's memory, or one covering the image [image_base, +image_bytes)
// whole - the image the address rule kept. A texture made with data has
// texels of its own (made ahead, or at its first bind).
std::uint64_t gx_successor_gen(std::uint64_t dead_memory, std::uint64_t image_base, std::size_t image_bytes);
void textures_gx_written(std::uint64_t base, std::size_t bytes);  // a texture GX mapped (host_gpu_texture_written)
void textures_submitted(int slot);                  // the recorded write-backs belong to this submission
void textures_retired(int slot);                    // that submission completed: copy them to guest memory

// render.cpp
void render_end_pass_locked();   // ends dynamic rendering if active (before dispatch/flush)
void render_report();
bool render_blit_display_locked(VkCommandBuffer cmd, std::uint64_t display_va, VkImage dst, VkRect2D area, VkExtent2D src,
                                std::uint32_t src_x = 0, std::uint32_t src_y = 0, VkImageView dst_view = VK_NULL_HANDLE);
// host/fsr.cpp: FSR 1 from [src_x, src_y, sw x sh] of `src` into `area` of
// `dst`; false when it cannot (the caller blits). `dst_view`, when `dst`
// takes storage writes: the last pass writes it directly.
bool fsr_upscale_locked(VkCommandBuffer cmd, VkImage src, VkFormat src_format, std::uint32_t src_w, std::uint32_t src_h,
                        std::uint32_t src_x, std::uint32_t src_y, std::uint32_t sw, std::uint32_t sh, VkImage dst, VkRect2D area,
                        VkImageView dst_view = VK_NULL_HANDLE);
// A CP DMA fill that covers a render target: clear the image instead.
bool render_clear_by_fill_locked(std::uint64_t va, std::size_t bytes, std::uint32_t value);
// A render target whose memory a shader fills with one value: clear the image.
bool render_clear_target_locked(std::uint64_t va, std::size_t bytes, const float rgba[4]);
// A fill of a depth target's HTILE buffer: the clear happens at the next draw,
// to the depth the fill word encodes and the stencil clear value the register
// file held at the fill (`depth_clear` is DB_DEPTH_CLEAR then, for the check).
bool render_htile_fill_locked(std::uint64_t va, std::size_t bytes, std::uint32_t word, std::uint32_t depth_clear, std::uint32_t stencil_clear);
// Detected fill shader: image-clear if the target exists, otherwise remember
// the value and skip the translated compute (per-lane page-table stores of an
// 8 MiB UI clear were ~3 ms each, and they ran before the image existed).
bool render_handle_fill_locked(std::uint64_t va, std::size_t bytes, const float rgba[4], std::uint32_t depth_clear,
                               std::uint32_t stencil_clear);
std::string render_fill_stats();  // " fill-pending=N fill-applied=M"
// Command-processor pipeline builds so far: [0] translate us, [1] modules us, [2] vkCreateGraphicsPipelines us,
// [3] stage translations, [4] stages the cache served (with BBHOST_STAGE_CACHE=0: translations whose key was seen before).
void render_pipeline_time_us(std::uint64_t out[5]);
// host_gpu_pipeline_work's graphics part, [0..11].
void render_pipeline_work(std::uint64_t out[12]);
// Compute pipelines a dispatch built on the command processor so far: [0] microseconds, [1] count.
void compute_pipeline_work(std::uint64_t out[2]);
// A compute pipeline the command processor built for a draw (the host
// tessellator's LS pass), for the same count.
void note_cp_compute_build(std::uint64_t us);
// A shader copies a render target's memory elsewhere: copy the image into a snapshot at dst
// (or into the target there of its shape); a copy one smaller destination long gets the
// source's top-left overlap (copy_overlap_locked). `via` names the copy in the log ("copy
// token", "copy dispatch"); a copy from a target it refuses is counted and logged once per
// pair (render_target_copy_report).
bool render_copy_target_locked(std::uint64_t src_base, std::uint64_t dst_base, std::size_t bytes, const char* via);
// The 5-second report's line: copies into images, and the ones refused by reason.
std::string render_target_copy_report();

// ---- recorder.cpp: the command stream (BBHOST_RECORDER, BBHOST_STREAM_SUBMIT) ----
// The command buffer being recorded, once the recorder has replayed every
// block handed to it: whatever records next lands after them. Under g.mu.
VkCommandBuffer g_cmd();
// BBHOST_GPU_PROFILE=2: a recording site (g_cmd()'s caller) noted for the gap
// the next profiled draw or dispatch closes.
void profile_note_site_locked(void* site);
void recorder_drain();
bool recorder_enabled();
std::string recorder_report();
// The stream carries the command buffers too: the recorder begins and ends
// them and hands their submissions to bb-submit, and rec() appends ops
// (BBHOST_STREAM_SUBMIT, on unless 0, and the recorder on).
bool stream_on();
// Until submission `serial` has gone to the queue (or failed): a fence the
// command processor waits for must have been submitted. Under g.mu.
void stream_wait_submitted(std::uint64_t serial);
// Until the recorder has replayed every block published so far (queue_drain;
// any thread but the recorder).
void stream_drain_published();
std::string stream_last_op();  // for the device-loss and hang reports
// gpu.cpp: a submission handed to bb-submit, or submitted here under the
// queue's lock without it; `serial` (a game submission's Gpu::flushes, ~0
// otherwise) and `foreign_id` (kForeignSubmit's) are published once it went.
void stream_submit_job(const VkCommandBuffer* cmds, std::uint32_t ncmds, VkSemaphore wait, VkPipelineStageFlags wait_stage, VkSemaphore signal,
                       VkFence fence, std::uint32_t items, std::uint64_t serial, std::uint64_t foreign_id);
std::uint64_t stream_submitted_serial();  // the game submissions gone to the queue: the last one's serial + 1
// What a submission costs on its way (recorder.cpp reports it): the command
// buffers' begin and end, wherever they run; slot retirement and its fence
// wait.
struct StreamStats {
    std::atomic<std::uint64_t> endbegin_ns{0}, endbegin_n{0}, retire_ns{0}, retires{0}, fence_ns{0}, submits{0};
};
extern StreamStats g_stream_stats;
// The self-test (stream_selftest.cpp): the recorder's mode between rounds -
// 0 none (everything in place), 1 inline, 2 its thread - and whether the
// stream carries the submissions. False while anything is open or recording.
bool stream_test_set_mode(int mode, bool submit);
struct StreamTestCounts {
    std::uint64_t blocks = 0, draws = 0, ops = 0, submits = 0, drains = 0, drain_waits = 0;
};
StreamTestCounts stream_test_counts();
extern std::atomic<std::uint64_t> g_validation_messages;  // gpu.cpp: what BBHOST_VK_VALIDATE=1's layer reported
// BBHOST_VK_VALIDATE=1's census (gpu.cpp): an object named for the layer's
// messages - a draw's pipeline by its programs - and, for the messages that
// name none (a pipeline's creation), what the thread is building while a
// ValidationScope lives. Both cost a branch with the layer off.
bool validation_on();
void vk_debug_name(VkObjectType type, std::uint64_t handle, const char* name);
extern thread_local constinit const char* t_validation_scope;
struct ValidationScope {
    const char* prev;
    explicit ValidationScope(const char* what) : prev(t_validation_scope) { t_validation_scope = what; }
    ~ValidationScope() { t_validation_scope = prev; }
    ValidationScope(const ValidationScope&) = delete;
    ValidationScope& operator=(const ValidationScope&) = delete;
};

// Flushes this core's write-combining buffers. AMD's Windows driver records
// into write-combined memory, and a release store does not order those
// buffers against another core (recorder.cpp, "Write combining"): every
// hand-over of a command buffer, or of host-visible memory the GPU reads,
// fences first.
inline void write_combine_fence() {
#if defined(__x86_64__)
    __builtin_ia32_sfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// The command processor's vkCmd* calls without the command buffer: with the
// stream on, each appends an op the recorder replays in order (its arrays
// copied, so the caller's may go at once); otherwise each records in place on
// g_cmd()'s command buffer. Under g.mu.
class Rec {
public:
    // Stream only (stream_on()): the slot's command buffer begun; ended and
    // its submission handed over (published at once); another thread's
    // command buffer submitted in stream order, an id host_gpu_wait_submitted
    // takes.
    void begin(VkCommandBuffer cmd);
    void end_submit(const stream::EndSubmit& s);
    std::uint64_t foreign_submit(VkCommandBuffer cmd, VkSemaphore wait, VkPipelineStageFlags wait_stage, VkSemaphore signal, VkFence fence);
    void pipeline_barrier(VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags dep, std::uint32_t nmem,
                          const VkMemoryBarrier* mem, std::uint32_t nbuf, const VkBufferMemoryBarrier* buf, std::uint32_t nimg,
                          const VkImageMemoryBarrier* img);
    void transfer_barrier(bool begin);  // record_transfer_barrier
    void copy_order_barrier();          // record_copy_order_barrier
    void pass_barrier();                // record_pass_barrier
    void end_rendering(bool barrier);   // record_end_rendering
    void begin_rendering(const VkRenderingInfo& ri);
    void fill_buffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, std::uint32_t data);
    void copy_buffer(VkBuffer src, VkBuffer dst, std::uint32_t n, const VkBufferCopy* regions);
    void copy_buffer_to_image(VkBuffer buffer, VkImage image, VkImageLayout layout, std::uint32_t n, const VkBufferImageCopy* regions);
    void copy_image_to_buffer(VkImage image, VkImageLayout layout, VkBuffer buffer, std::uint32_t n, const VkBufferImageCopy* regions);
    void clear_color_image(VkImage image, VkImageLayout layout, const VkClearColorValue& color, std::uint32_t n,
                           const VkImageSubresourceRange* ranges);
    void reset_query_pool(VkQueryPool pool, std::uint32_t first, std::uint32_t count);
    void begin_query(VkQueryPool pool, std::uint32_t query, VkQueryControlFlags flags);
    void end_query(VkQueryPool pool, std::uint32_t query);
    void copy_query_results(VkQueryPool pool, std::uint32_t first, std::uint32_t count, VkBuffer dst, VkDeviceSize offset, VkDeviceSize stride,
                            VkQueryResultFlags flags);
    void write_timestamp(VkPipelineStageFlagBits stage, VkQueryPool pool, std::uint32_t query);
    void bind_pipeline(VkPipelineBindPoint bind_point, VkPipeline pipeline);
    void bind_sets(VkPipelineBindPoint bind_point, VkPipelineLayout layout, std::uint32_t first, std::uint32_t n, const VkDescriptorSet* sets,
                   std::uint32_t ndynamic = 0, const std::uint32_t* dynamic = nullptr);
    void push_constants(VkPipelineLayout layout, VkShaderStageFlags stages, std::uint32_t offset, std::uint32_t size, const void* data);
    // vkCmdPushDescriptorSetKHR of n storage buffers into set `set` (a
    // dispatch's, ComputeLean).
    void push_buffers(VkPipelineBindPoint bind_point, VkPipelineLayout layout, std::uint32_t set, std::uint32_t n, const std::uint32_t* bindings,
                      const VkDescriptorBufferInfo* infos);
    // vkUpdateDescriptorSets at this point of the stream (a set the op
    // stream's earlier commands may still bind in this command buffer).
    void update_sets(std::uint32_t n, const VkWriteDescriptorSet* writes);
    void dispatch(std::uint32_t x, std::uint32_t y, std::uint32_t z);
    void dispatch_indirect(VkBuffer buffer, VkDeviceSize offset);
    void marker(std::uint32_t value);  // gpu_write_markers
    void clear_depth_stencil_image(VkImage image, VkImageLayout layout, const VkClearDepthStencilValue& value, std::uint32_t n,
                                   const VkImageSubresourceRange* ranges);
    void copy_image(VkImage src, VkImageLayout src_layout, VkImage dst, VkImageLayout dst_layout, std::uint32_t n, const VkImageCopy* regions);
};
Rec& rec();

// State a pipeline linked from libraries takes per draw.
struct DrawLibraryState {
    VkCullModeFlags cull = 0;
    VkFrontFace front = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkBool32 depth_test = 0, depth_write = 0, bounds_test = 0, stencil_test = 0;
    VkCompareOp depth_compare = VK_COMPARE_OP_NEVER;
    VkStencilOpState front_ops{}, back_ops{};
    // Depth bias: the pre-rasterization libraries enable it with dynamic
    // values, which are 0 for a state without one.
    float bias_constant = 0.0f, bias_clamp = 0.0f, bias_slope = 0.0f;
    VkBool32 depth_clamp = 0;  // under g.dynamic_depth_clamp
};
struct DrawCall {
    enum Kind { kDirect, kIndexed, kIndirect, kIndexedIndirect } kind = kDirect;
    std::uint32_t count = 0, instances = 0;
    std::int32_t vertex_offset = 0;
    VkBuffer buffer = VK_NULL_HANDLE;  // indirect arguments
    VkDeviceSize offset = 0;
    // An occlusion query around the draw: its coverage (BBHOST_GLITCH=1) or
    // the PS4's sample counters (occlusion.cpp).
    VkQueryPool query_pool = VK_NULL_HANDLE;
    std::uint32_t query = 0;
    VkQueryControlFlags query_flags = 0;
    // SET_PREDICATION resolved on the GPU (occlusion.cpp): the draw goes ahead
    // only if this word is nonzero (VK_EXT_conditional_rendering).
    VkBuffer cond_buffer = VK_NULL_HANDLE;
    VkDeviceSize cond_offset = 0;
};
constexpr std::uint32_t kMaxVertexBindings = 16;
void record_library_state(VkCommandBuffer cmd, const DrawLibraryState& s, bool has_depth_bounds);
void record_stencil_words(VkCommandBuffer cmd, const std::uint32_t words[6]);
void record_draw_call(VkCommandBuffer cmd, const DrawCall& call);

// One draw's descriptor writes and commands. Deferred (and the recorder on),
// they go into a packet the recorder thread replays, handed over when this
// is destroyed or publish() is called; otherwise each is recorded at once.
// Under g.mu, one at a time.
class DrawCmds {
public:
    DrawCmds() = default;  // no packet until start()
    explicit DrawCmds(bool deferred);
    ~DrawCmds();
    // The packet, when `deferred` and the recorder is on. One started earlier
    // is kept, or handed to the recorder when `deferred` is false.
    void start(bool deferred);
    // BBHOST_RUN_COMMIT: the packet numbered `number`, reserved at a run's fork
    // (recorder_reserve_packets). In a part (`part`) it stays out of the
    // stream when published - the command processor appends it in its order
    // at the join (recorder_append_packet) - and never becomes the open draw.
    void start_reserved(std::uint64_t number, bool part);
    DrawCmds(const DrawCmds&) = delete;
    DrawCmds& operator=(const DrawCmds&) = delete;
    bool deferred() const { return packet_ != nullptr; }
    void publish();
    void update_sets(const VkWriteDescriptorSet* writes, std::size_t n);
    // The draw's own writes, taken rather than copied: `writes` and `images`
    // are swapped with the packet's spent vectors (their buffers, which the
    // writes point into, move with them); `buffers` and the `n_extra` infos
    // at `extra` (the params blocks, on the stack) are copied. Afterwards the
    // caller's `writes` and `images` hold nothing it wrote.
    void take_sets(std::vector<VkWriteDescriptorSet>& writes, std::vector<VkDescriptorImageInfo>& images,
                   const std::vector<VkDescriptorBufferInfo>& buffers, const VkDescriptorBufferInfo* extra, std::size_t n_extra);
    // `discard`: the pipeline takes rasterizer discard as a dynamic state, set
    // on after it is bound (a draw PA_CL_CLIP_CNTL's DX_RASTERIZATION_KILL kills).
    void bind_pipeline(VkPipeline pipeline, bool discard = false);
    void library_state(const DrawLibraryState& s, bool has_depth_bounds);
    // `dynamic`: the params blocks' offsets (set_cache_on()), else null.
    // `bindless`: the pipeline's stages read the global set, bound as set 3.
    void bind_sets(VkPipelineLayout layout, const VkDescriptorSet sets[2], const std::uint32_t* dynamic = nullptr, bool bindless = false);
    // Constant buffers into set 2 (cb_push_on()): `n` bindings and their ranges.
    void push_buffers(VkPipelineLayout layout, const std::uint32_t* bindings, const VkDescriptorBufferInfo* infos, std::uint32_t n);
    // One more set at `index`, after bind_sets' (a pipeline's own, past the
    // shared layout's sets), with up to two dynamic offsets.
    void bind_stage_set(VkPipelineLayout layout, std::uint32_t index, VkDescriptorSet set, std::uint32_t dynamic_count = 0,
                        const std::uint32_t* dynamic = nullptr);
    void viewport(const VkViewport& vp);
    void scissor(const VkRect2D& sc);
    void depth_bounds(float lo, float hi);
    void stencil(const std::uint32_t words[6]);
    void blend_constants(const float c[4]);
    void index_buffer(VkBuffer buffer, VkDeviceSize offset, VkIndexType type);
    void vertex_buffers(std::uint32_t n, const VkBuffer* buffers, const VkDeviceSize* offsets);
    void marker(std::uint32_t value);  // AMD's buffer markers before the draw (gpu_write_markers)
    void draw(const DrawCall& call);
    // Ending the current pass (vkCmdEndRendering and, unless `barrier` is
    // false, the barrier after it) and beginning one; a pass end's barrier
    // owed from earlier (lazy pass barriers). False when the packet cannot
    // take them in order (commands after them are already in it): the caller
    // records in place.
    bool end_rendering(bool barrier = true);
    bool pass_barrier();
    bool begin_rendering(const VkRenderingInfo& ri);
    // A transfer batch's barriers and its buffer copies, replayed
    // after a pass end and before a pass begin in the same packet: a copy
    // token or a CP write recorded here does not wait for the recorder to
    // finish the draws before it, as recording in place does (g_cmd()).
    // False when the packet cannot take them in order, or is full.
    bool transfer_begin();
    bool copy_buffer(VkBuffer src, VkBuffer dst, const VkBufferCopy& region);
    // The next copy waits for the batch's copies before it (record_copy_order_barrier).
    bool copy_order();
    bool transfer_end();
    // A render target's top-left width x height into an image of its own
    // (record_region_copy), after the transfer batch and before the pass
    // begin. False when the packet cannot take it in order, or is full.
    bool copy_region(VkImage src, VkImage dst, std::uint32_t width, std::uint32_t height, bool dst_initialised);
    // The draw whose packet is open, if any. g_cmd() hands it to the
    // recorder first, so nothing recorded in place overtakes it.
    static DrawCmds* open();
    bool holds_commands() const;

private:
    void* packet_ = nullptr;
    bool part_ = false;  // a run part's packet (start_reserved)
};
// BBHOST_RUN_COMMIT (recorder.cpp): whether draws are recorded on the
// recorder's thread (the forks need it); `n` packet numbers reserved for a
// run, waiting for the recorder while its ring has no room for them (under
// g.mu, nothing open), the first returned; a reserved packet's draw appended
// to the stream in its order at the join; and the appended ops handed over.
bool recorder_threaded();
std::uint64_t recorder_reserve_packets(std::uint32_t n);
void recorder_append_packet(std::uint64_t number);
void recorder_publish_appended();
// The draw draw_impl is resolving, its packet started now if it has none
// (render.cpp): a region copy made while its images are prefetched goes in it
// with the pass end before it. Null outside a draw, or when it records in place.
DrawCmds* draw_packet_early_locked();
// The barrier render_end_pass_locked records after vkCmdEndRendering.
void record_end_rendering(VkCommandBuffer cmd, bool barrier = true);
void record_pass_barrier(VkCommandBuffer cmd);  // what makes a pass's attachment writes visible to everything after
// BBHOST_BARRIER_MASKS (on unless 0; gpu.cpp): the barriers after a pass, a
// dispatch and the tessellation LS pass name every access that can follow -
// shader stores before a pass ends, vertex fetch and attachments after a
// dispatch, the tessellation stages after the LS pass.
bool barrier_masks_fixed();
// A transfer batch's barrier: into the transfer stage (`begin`), or out of it.
void record_transfer_barrier(VkCommandBuffer cmd, bool begin);
// Inside a batch: the transfers recorded before it finish before the ones after.
void record_copy_order_barrier(VkCommandBuffer cmd);
// A render target's top-left width x height copied into `dst` (textures.cpp
// rt_copied_view), with the barriers around it; `dst` starts undefined until
// its first copy.
void record_region_copy(VkCommandBuffer cmd, VkImage src, VkImage dst, std::uint32_t width, std::uint32_t height, bool dst_initialised);

// ---- descriptor sets cached by content (render.cpp, BBHOST_SET_CACHE) ----
// On (the default), a stage's params block is a dynamic uniform buffer - the
// whole ring at offset 0, one block long, its offset given when the set is
// bound - and constant buffers are bound whole, their offsets in the params.
// A set then holds only its layout's views, samplers and buffer handles, and
// ~99% of a frame's sets repeat one already made.
bool set_cache_on();
// Constant buffers go in a push-descriptor set of their own (set 2) rather
// than in the stage's set, which the cache then keys without them: with the
// cache, whenever the device has push descriptors.
bool cb_push_on();
// Pushed constant buffers bound at their V# bases, shaders translated with
// TranslateOptions::cb_bias_zero (render.cpp, BBHOST_CB_EXACT).
bool cb_exact_bindings();
// A dispatch's storage buffers (ComputeLean::meta's), resolved as a draw's
// are - bound where they are, or copied as recorded for memory the game
// writes on the CPU - into params (cb_valid, biases, strides) and `infos`,
// one a buffer. The dispatch's own stores must be noted as GPU writes first
// (gpu_writes_note_locked), so nothing it writes is read from a copy.
// render.cpp; under g.mu, with the dispatch's recording begun.
void resolve_dispatch_buffers(const gcn::TranslateResult& meta, const char* name, const std::uint32_t* user, gcn::StageParams& params,
                              std::vector<VkDescriptorBufferInfo>& infos);
// BBHOST_PIPELINE_STATS=<name>: whether a compute shader is named, and its
// pipeline's statistics and disassembly to build/pipeline-stats (render.cpp).
bool compute_pipeline_stats_wanted(const std::string& name);
void write_compute_pipeline_stats(VkPipeline p, const std::string& name);
VkDescriptorType params_descriptor_type();
// The params descriptor with the cache on (set up with the ring).
extern VkDescriptorBufferInfo g_params_desc;
// A view is being let go: the cached sets that hold it are dropped. Under g.mu.
void set_cache_forget_view_locked(VkImageView view);
// Every cached set is dropped (a buffer they may name is being let go: a
// retired window import, without push descriptors). Under g.mu.
void set_cache_drop_all_locked();
std::string set_cache_report();

}  // namespace gpu
