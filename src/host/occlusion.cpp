// Occlusion queries and predication, the way the PS4's depth blocks do them.
//
// The hardware has no begin and end: each depth block (DB, eight on a base
// PS4) counts the samples that pass, and an EVENT_WRITE of ZPASS_DONE /
// PIXEL_PIPE_STAT_DUMP makes every DB store its count, bit 63 set, at the
// dump's address + 16 * db. Gnm's occlusion query is a block of eight
// {begin, end} pairs the game zeroes, dumps into before (+0) and after (+8)
// the draw it measures, and reads back: the sum of end - begin is how many
// samples passed. Bloodborne measures its culling proxies that way (GX
// GetData 0x256a2f0) - each window one draw, about 180 of them a frame in a
// dense view - and YEBIS its luminance. Until now every dump wrote a rising
// fake count (gnm_exec.cpp), so every object the game tested was visible and
// it culled nothing: on the same APU, Kyo measured the game at ~51% GPU with
// real counts against ~89% without, and 90 -> 130 fps uncapped in a heavy
// spot, with 86-88% of the tested objects occluded.
//
// Here (the design is Kyo's KyoPS4x counter, #217-#220, done natively):
// - The counted draws. After a dump the next draws get a Vulkan occlusion
//   query each, begun and ended around the draw on the recorder's thread
//   (DrawCall::query_pool), so a query never spans a render pass. 64 of them
//   after a dump to the first half of a 16-byte cell (a Gnm block's begin;
//   BBHOST_OCCLUSION_CAP), one after any other dump (an end, or a block that
//   is not 16-aligned). A draw past those, or one that reached no query (a
//   draw bbhost did not record, a full pool), is "unmeasured": it adds a
//   large count, so a difference spanning it reads visible, never occluded.
// - The dumps. The command processor records the stream as items - query,
//   unmeasured, dump - and the GPU walks them (shaders/occlusion.comp) once
//   the draws have run: query results are copied with their availability,
//   a running total T is kept the way the DBs keep theirs, and each dump
//   writes T over the eight DBs with the valid bit into guest memory by
//   device address. That happens before the submission's label writes, so
//   the counts are there when the game's end-of-pipe fence says the GPU is
//   done - the PS4's order, with no CPU round trip and nothing predicted.
//   A query not available when the copy runs reads plainly visible, and its
//   dump is written all the same. AMD's driver resolves none while the
//   window is unfocused or there is none (Kyo #220; a copy that waits for
//   them loses the device, headless), and a dump left unwritten kept its
//   block not ready for good: Bloodborne's GX culling entries stay pending
//   on such a block and it starts no other test, so one alt-tab, or one
//   query late, ended the game's culling until the next start - with it on,
//   slower than off (its occluders still drawn, nothing culled).
// - Precise counts when the device has occlusionQueryPrecise (the game sets
//   PERFECT_ZPASS_COUNTS); without it a nonzero result counts as plainly
//   visible (NVIDIA's unprecise results are 0/1, and one sample read as
//   "culled" was the best explanation of a black world on KyoPS4x, #219).
// - SET_PREDICATION (PM4 0x20) with a ZPASS operation: when every dump into
//   the block has retired, the CPU reads the block and the draws up to the
//   predicate's clear are skipped before any of their work (host_gpu_draw)
//   or drawn; when a dump is still in flight, the items so far are resolved
//   at once (the pass ends) and the shader also writes the predicate, which
//   the draws read through VK_EXT_conditional_rendering. Without the
//   extension, or for an unknown block, they are drawn: a wrong "occluded"
//   makes geometry vanish.
//
// DB_COUNT_CONTROL is not honoured: the context registers are not kept for
// token draws, and counting a draw the game left uncounted only makes a
// result more visible. PIXEL_PIPE_STAT_RESET is ignored, as by KyoPS4x.
//
// BBHOST_OCCLUSION=0: the old fake. BBHOST_PREDICATION=0: SET_PREDICATION
// ignored (every predicated draw drawn). Counts in the 300-flip report and at
// exit ("occlusion:").

#include "host/gpu_internal.h"
#include "host/shaders/occlusion.spv.h"
#include "log.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace gpu {
namespace {

constexpr std::uint32_t kQueries = 4096;  // a recording's queries (one slot's pool)
constexpr std::uint32_t kItems = 16384;   // a recording's items
constexpr std::uint32_t kPreds = 1024;    // GPU predicates, a ring of two words each
constexpr std::uint32_t kPipes = 8;       // DBs on a base PS4 (16 on a Neo, which this is not)
constexpr std::uint64_t kBlockExtent = 16 * (kPipes - 1) + 8;  // the words one dump writes

// An item of the stream the shader walks (shaders/occlusion.comp).
enum : std::uint32_t { kItemQuery = 0, kItemUnmeasured = 1, kItemDump = 2, kItemPredicate = 3 };
struct Item {
    std::uint32_t x, y, z, w;
};
// The shader's counters (State::counters).
enum { kGpuWritten, kGpuNotReady, kGpuUnavailable, kGpuPairsVisible, kGpuPairsOccluded, kGpuPredDraw, kGpuPredSkip, kGpuPredUnknown,
       kGpuCounters = 12 };
struct State {
    std::uint64_t total, acc;
    std::uint32_t bad, pad;
    std::uint32_t counters[kGpuCounters];
};
struct Push {
    std::uint64_t items, results, state, preds;
    std::uint32_t count, flags;
};
static_assert(sizeof(Push) == 40, "the shader's push-constant block");

const bool g_wanted = [] {
    const char* e = std::getenv("BBHOST_OCCLUSION");
    return !(e && e[0] == '0');
}();
const bool g_pred_wanted = [] {
    const char* e = std::getenv("BBHOST_PREDICATION");
    return !(e && e[0] == '0');
}();
// Queries after a block's begin dump: Bloodborne's windows are one draw.
const std::uint32_t g_cap = [] {
    const char* e = std::getenv("BBHOST_OCCLUSION_CAP");
    const long v = e && *e ? std::strtol(e, nullptr, 10) : 64;
    return static_cast<std::uint32_t>(v < 1 ? 1 : v > 1024 ? 1024 : v);
}();

// BBHOST_OCCLUSION_WAIT=1 (a diagnostic): the copy of the results waits on
// the GPU for each query. Off, a query not yet available when the copy runs
// reads visible; on, a query a driver never resolves hangs the GPU (AMD's,
// headless: the device was lost within 20 dumps, 2026-10-14).
const bool g_wait = [] {
    const char* e = std::getenv("BBHOST_OCCLUSION_WAIT");
    return e && e[0] == '1';
}();

bool g_on = false;  // decided at device creation
bool g_precise = false;
bool g_cond = false;  // VK_EXT_conditional_rendering enabled
VkPhysicalDeviceConditionalRenderingFeaturesEXT g_cond_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
PFN_vkCmdBeginConditionalRenderingEXT g_begin_cond = nullptr;
PFN_vkCmdEndConditionalRenderingEXT g_end_cond = nullptr;
VkPipelineLayout g_layout = VK_NULL_HANDLE;
VkPipeline g_pipeline = VK_NULL_HANDLE;
DevBuffer g_state;  // State, host-visible: the report reads its counters
DevBuffer g_preds;  // kPreds x {draw, seen}
std::uint32_t g_pred_next = 0;

// Per command-buffer slot: its queries, its items and where their results go.
struct SlotOcc {
    VkQueryPool pool = VK_NULL_HANDLE;
    DevBuffer items, results;
    std::uint32_t queries = 0, nitems = 0;            // used by this recording
    std::uint32_t queries_done = 0, items_done = 0;   // resolved so far
    std::uint32_t to_reset = kQueries;                 // reset at the next recording's start (all, the first time)
    bool ready = false;                                // this recording reset its queries
};
SlotOcc g_occ[kSlots];

// The interval since the last dump (under g.mu).
struct Interval {
    bool open = false;
    bool unmeasured = false;  // a counted draw in it got no query
    std::uint32_t cap = 0, queries = 0;
    std::uint64_t issued_at = 0;   // g_issued at the dump
    std::uint64_t measured = 0;    // draws that got a query
    std::uint64_t last_draw = 0;
};
Interval g_iv;
// Draws entering host_gpu_draw, and the one being drawn: a draw that never
// reached a query (failed, skipped, turned into a clear) is told apart from
// one that did by these.
std::atomic<std::uint64_t> g_issued{0};
thread_local std::uint64_t t_draw = 0;

// SET_PREDICATION: what the draws up to its clear do.
enum PredMode : int { kPredNone, kPredSkip, kPredGpu, kPredDraw };
std::atomic<int> g_pred_mode{kPredNone};
std::uint32_t g_pred_index = 0;  // kPredGpu: the predicate word pair
std::uint32_t g_pred_seen = 0;   // CPU-resolved so far under CONTINUE: visible | unknown << 1
// The submission each block's last dump is recorded into: a block whose
// dumps have all retired can be read on the CPU.
std::unordered_map<std::uint64_t, std::uint64_t> g_block_serial;

struct Counts {
    std::uint64_t dumps = 0, fake = 0, lost = 0, queries = 0, unmeasured = 0, unmeasured_intervals = 0, resolves = 0, mid_resolves = 0;
    std::uint64_t pred_sets = 0, pred_known = 0, pred_gpu_sets = 0, pred_gpu_draws = 0, pred_drawn = 0;
    std::uint32_t gpu[kGpuCounters] = {};
};
Counts g_counts;
std::atomic<std::uint64_t> g_pred_skipped{0};

void push_item(SlotOcc& so, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t w) {
    static_cast<Item*>(so.items.map)[so.nitems++] = Item{x, y, z, w};
}

bool make_pipeline_locked() {
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(k_occlusion_spv);
    smi.pCode = k_occlusion_spv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.device, &smi, nullptr, &module) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    bool ok = vkCreatePipelineLayout(g.device, &pli, nullptr, &g_layout) == VK_SUCCESS;
    if (ok) {
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = module;
        ci.stage.pName = "main";
        ci.layout = g_layout;
        ok = vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &ci, nullptr, &g_pipeline) == VK_SUCCESS;
    }
    vkDestroyShaderModule(g.device, module, nullptr);
    return ok;
}

// The predicate words: host-visible (they start as "draw"), readable by
// conditional rendering and by the shader through its address.
bool make_preds_locked() {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 8ull * kPreds;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (g_cond) bci.usage |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
    if (vkCreateBuffer(g.device, &bci, nullptr, &g_preds.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.device, g_preds.buffer, &req);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &flags;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex =
        find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    MemOwner own_mem("occlusion queries");
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &g_preds.memory) != VK_SUCCESS) {
        vkDestroyBuffer(g.device, g_preds.buffer, nullptr);
        g_preds.buffer = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(g.device, g_preds.buffer, g_preds.memory, 0);
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = g_preds.buffer;
    g_preds.address = vkGetBufferDeviceAddress(g.device, &bai);
    g_preds.size = bci.size;
    if (vkMapMemory(g.device, g_preds.memory, 0, VK_WHOLE_SIZE, 0, &g_preds.map) != VK_SUCCESS) return false;
    auto* w = static_cast<std::uint32_t*>(g_preds.map);
    for (std::uint32_t k = 0; k < kPreds; ++k) {
        w[2 * k] = 1;      // draw
        w[2 * k + 1] = 2;  // unknown
    }
    return true;
}

// The items recorded since the last resolve, walked on the GPU now: the
// render pass ends, the queries' results are copied out, the shader runs.
// At a submission's end, and mid-recording for a predicate.
void resolve_locked(bool mid) {
    SlotOcc& so = g_occ[g.slot];
    if (!so.ready || so.items_done == so.nitems) return;
    transfer_flush_locked();
    render_end_pass_locked();
    // Ops of the command stream (recorder.cpp): the recorder replays them
    // after the draws whose queries they read; without it, in place.
    // The draws (their queries end with them) and every earlier write of
    // guest memory, before the copy and the shader.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                           nullptr, 0, nullptr);
    if (so.queries > so.queries_done) {
        // With availability and without waiting: a query AMD never resolves
        // must not hang the GPU; the shader leaves its dump unwritten.
        rec().copy_query_results(so.pool, so.queries_done, so.queries - so.queries_done, so.results.buffer, 16ull * so.queries_done, 16,
                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT | (g_wait ? VK_QUERY_RESULT_WAIT_BIT : 0));
        VkMemoryBarrier cb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        cb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        cb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        rec().pipeline_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &cb, 0, nullptr, 0, nullptr);
    }
    const Push push{so.items.address + 16ull * so.items_done, so.results.address, g_state.address, g_preds.address,
                    so.nitems - so.items_done, g_precise ? 0u : 1u};
    rec().bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, g_pipeline);
    rec().push_constants(g_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    rec().dispatch(1, 1, 1);
    // The counts in guest memory before anything after them - the label
    // fills, the game's own reads, conditional rendering's predicate reads.
    VkMemoryBarrier ab{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    ab.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ab.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &ab, 0,
                           nullptr, 0, nullptr);
    so.items_done = so.nitems;
    so.queries_done = so.queries;
    ++g_counts.resolves;
    if (mid) ++g_counts.mid_resolves;
}

// A block on the CPU: visible (1), occluded (0), or not ready (2).
std::uint32_t block_on_cpu(std::uint64_t block) {
    if (!hle_kernel_va_mapped(block, 16 * kPipes)) return 2;
    const auto* w = reinterpret_cast<const volatile std::uint64_t*>(static_cast<std::uintptr_t>(block));
    constexpr std::uint64_t kValid = 1ull << 63;
    std::uint64_t sum = 0;
    for (std::uint32_t k = 0; k < kPipes; ++k) {
        const std::uint64_t b = w[2 * k], e = w[2 * k + 1];
        if (!(b & kValid) || !(e & kValid)) return 2;
        sum += (e & ~kValid) - (b & ~kValid);
    }
    return sum ? 1 : 0;
}

}  // namespace

bool occlusion_on() { return g_on; }

void occlusion_device_features(VkPhysicalDeviceFeatures& enable, std::vector<const char*>& exts, void** chain) {
    if (!g_wanted) {
        host_log("occlusion: off (BBHOST_OCCLUSION=0): every query the game makes reads visible, and nothing is culled by it");
        return;
    }
    if (glitch_on()) {
        host_log("occlusion: off while the glitch hunt has a query around every draw (BBHOST_GLITCH)");
        return;
    }
    VkPhysicalDeviceFeatures sup{};
    vkGetPhysicalDeviceFeatures(g.phys, &sup);
    g_precise = sup.occlusionQueryPrecise == VK_TRUE;
    if (g_precise) enable.occlusionQueryPrecise = VK_TRUE;
    if (g_pred_wanted) {
        std::uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> props(n);
        vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &n, props.data());
        bool has = false;
        for (const VkExtensionProperties& p : props) has = has || std::strcmp(p.extensionName, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME) == 0;
        if (has) {
            VkPhysicalDeviceConditionalRenderingFeaturesEXT q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
            VkPhysicalDeviceFeatures2 qf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            qf.pNext = &q;
            vkGetPhysicalDeviceFeatures2(g.phys, &qf);
            if (q.conditionalRendering) {
                exts.push_back(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
                g_cond_features.conditionalRendering = VK_TRUE;
                g_cond_features.pNext = *chain;
                *chain = &g_cond_features;
                g_cond = true;
            }
        }
    }
    g_on = true;
}

void occlusion_device_ready_locked() {
    if (!g_on) return;
    g_on = false;
    if (g_cond) {
        g_begin_cond = reinterpret_cast<PFN_vkCmdBeginConditionalRenderingEXT>(vkGetDeviceProcAddr(g.device, "vkCmdBeginConditionalRenderingEXT"));
        g_end_cond = reinterpret_cast<PFN_vkCmdEndConditionalRenderingEXT>(vkGetDeviceProcAddr(g.device, "vkCmdEndConditionalRenderingEXT"));
        g_cond = g_begin_cond && g_end_cond;
    }
    const char* failed = nullptr;
    if (!make_pipeline_locked()) failed = "its compute pipeline";
    MemOwner own_mem("occlusion queries");
    if (!failed && !create_dev_buffer(g_state, sizeof(State), true, true)) failed = "its state buffer";
    if (!failed) std::memset(g_state.map, 0, sizeof(State));
    if (!failed && !make_preds_locked()) failed = "its predicate buffer";
    for (int k = 0; !failed && k < g.slot_count; ++k) {
        SlotOcc& so = g_occ[k];
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_OCCLUSION;
        qci.queryCount = kQueries;
        if (vkCreateQueryPool(g.device, &qci, nullptr, &so.pool) != VK_SUCCESS) failed = "a query pool";
        else if (!create_dev_buffer(so.items, 16ull * kItems, true)) failed = "an item buffer";
        else if (!create_dev_buffer(so.results, 16ull * kQueries, false)) failed = "a result buffer";
    }
    if (failed) {
        host_log("occlusion: off - %s could not be made; every query reads visible, as before", failed);
        return;
    }
    g_on = true;
    host_log("occlusion: real sample counts - a query on each of the %u draws after a block's first dump (1 after any other), "
             "written into guest memory on the GPU's timeline before the labels%s; %s; SET_PREDICATION %s "
             "(BBHOST_OCCLUSION=0: the old always-visible fake)",
             g_cap, g_wait ? " (the copy waits for every query: BBHOST_OCCLUSION_WAIT=1)" : "", g_precise ? "precise" : "not precise (the device has no occlusionQueryPrecise: a nonzero count reads plainly visible)",
             !g_pred_wanted ? "ignored (BBHOST_PREDICATION=0)"
             : g_cond       ? "skips on the CPU when the result is known, else conditional rendering"
                            : "skips on the CPU when the result is known, else draws (no VK_EXT_conditional_rendering)");
}

void occlusion_begin_recording_locked() {
    if (!g_on) return;
    SlotOcc& so = g_occ[g.slot];
    // The queries the slot's last recording used; its submission has retired.
    if (so.to_reset) rec().reset_query_pool(so.pool, 0, so.to_reset);
    so.to_reset = 0;
    so.queries = so.nitems = so.queries_done = so.items_done = 0;
    so.ready = true;
}

// A run's parts commit no draw that takes a query: the interval since the
// last dump has given out its cap of them (or has no room), so every draw left
// in it is unmeasured - which the join notes for them all.
bool occlusion_run_ok() {
    if (!g_on) return true;
    if (g_pred_mode.load(std::memory_order_relaxed) == kPredSkip) return false;
    if (!g_iv.open) return true;
    const SlotOcc& so = g_occ[g.slot];
    return !so.ready || g_iv.queries >= g_iv.cap || so.queries >= kQueries || so.nitems + 64 > kItems;
}

void occlusion_run_issued(std::uint64_t n) {
    if (!g_on || !n) return;
    g_issued.fetch_add(n, std::memory_order_relaxed);
    if (g_iv.open) {
        g_iv.unmeasured = true;
        g_counts.unmeasured += n;
    }
}

bool occlusion_draw_entry() {
    if (!g_on) return true;
    if (g_pred_mode.load(std::memory_order_relaxed) == kPredSkip) {
        g_pred_skipped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    t_draw = g_issued.fetch_add(1, std::memory_order_relaxed) + 1;
    return true;
}

void occlusion_draw_locked(DrawCall& call) {
    if (!g_on) return;
    switch (g_pred_mode.load(std::memory_order_relaxed)) {
        case kPredGpu:
            call.cond_buffer = g_preds.buffer;
            call.cond_offset = 8ull * g_pred_index;
            if (!run_part()) ++g_counts.pred_gpu_draws;  // a run part's (BBHOST_RUN_COMMIT): not counted
            break;
        case kPredDraw:
            if (!run_part()) ++g_counts.pred_drawn;
            break;
        default: break;
    }
    // A run part's draw (BBHOST_RUN_COMMIT): forked only where no draw takes a
    // query (occlusion_run_ok), and noted unmeasured at the join.
    if (!g_iv.open || run_part()) return;
    SlotOcc& so = g_occ[g.slot];
    // Room is left for the dumps: a full item buffer must not lose one.
    if (!so.ready || g_iv.queries >= g_iv.cap || so.queries >= kQueries || so.nitems + 64 > kItems) {
        g_iv.unmeasured = true;
        ++g_counts.unmeasured;
        return;
    }
    call.query_pool = so.pool;
    call.query = so.queries++;
    call.query_flags = g_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0;
    so.to_reset = so.queries;
    push_item(so, kItemQuery, call.query, 0, 0);
    ++g_iv.queries;
    ++g_counts.queries;
    if (t_draw != g_iv.last_draw) {
        g_iv.last_draw = t_draw;
        ++g_iv.measured;
    }
}

void occlusion_submit_locked() {
    if (!g_on) return;
    resolve_locked(false);
    g_occ[g.slot].ready = false;  // until the slot's next recording resets its queries
}

void occlusion_record_cond(VkCommandBuffer cmd, const DrawCall& call, bool begin) {
    if (!g_cond) return;
    if (begin) {
        VkConditionalRenderingBeginInfoEXT ci{VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT};
        ci.buffer = call.cond_buffer;
        ci.offset = call.cond_offset;
        g_begin_cond(cmd, &ci);
    } else {
        g_end_cond(cmd);
    }
}

std::string occlusion_report(bool totals) {
    static Counts last;
    Counts now = g_counts;
    if (g_state.map) {
        State st;
        std::memcpy(&st, g_state.map, sizeof(st));  // the GPU's, as far as it has got
        std::memcpy(now.gpu, st.counters, sizeof(now.gpu));
    }
    const Counts& base = totals ? Counts{} : last;
    const auto d = [&](std::uint64_t Counts::*m) { return static_cast<unsigned long long>(now.*m - base.*m); };
    const auto dg = [&](int k) { return static_cast<unsigned long long>(now.gpu[k] - base.gpu[k]); };
    const unsigned long long visible = dg(kGpuPairsVisible), occluded = dg(kGpuPairsOccluded);
    static std::uint64_t last_skipped = 0;
    const std::uint64_t skipped = g_pred_skipped.load(std::memory_order_relaxed);
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "occlusion%s: %llu dumps (%llu fake: not imported; %llu lost), %llu queries, %llu unmeasured draws (%llu intervals read visible), %llu resolves "
                  "(%llu mid-recording); on the GPU %llu dumps written, %llu of them read visible for a query not available (%llu queries), pairs %llu "
                  "visible / %llu occluded (%.1f%% occluded); predication: %llu sets (%llu known on the CPU, %llu on the GPU), "
                  "draws %llu skipped on the CPU, %llu GPU-predicated (%llu drawn, %llu skipped, %llu unknown), %llu drawn",
                  totals ? " totals" : "", d(&Counts::dumps), d(&Counts::fake), d(&Counts::lost), d(&Counts::queries),
                  d(&Counts::unmeasured), d(&Counts::unmeasured_intervals), d(&Counts::resolves), d(&Counts::mid_resolves), dg(kGpuWritten), dg(kGpuNotReady),
                  dg(kGpuUnavailable), visible, occluded, visible + occluded ? 100.0 * static_cast<double>(occluded) / static_cast<double>(visible + occluded) : 0.0,
                  d(&Counts::pred_sets), d(&Counts::pred_known), d(&Counts::pred_gpu_sets),
                  static_cast<unsigned long long>(totals ? skipped : skipped - last_skipped), d(&Counts::pred_gpu_draws), dg(kGpuPredDraw),
                  dg(kGpuPredSkip), dg(kGpuPredUnknown), d(&Counts::pred_drawn));
    if (!totals) {
        last = now;
        last_skipped = skipped;
    }
    return buf;
}

}  // namespace gpu

using namespace gpu;

bool host_gpu_zpass_dump(std::uint64_t va) {
    if (!g_on) return false;
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok || (va & 7)) return false;
    const ImportSite site("an occlusion dump");
    // The game reads the block on the CPU (GetData): never a GX twin's copy
    // (BBHOST_GX_TWINS) - one over it is copied back and let go first.
    if (twins_on()) twins_keep_in_guest_locked(va, kBlockExtent);
    const VkDeviceAddress at = guest_device_address(va, kBlockExtent);
    if (!at) {
        // Not imported memory: the shader cannot reach it; the caller's fake.
        if (g_counts.fake++ < 4) {
            host_log("occlusion: a dump to 0x%llx is not in imported memory; it gets the old fake count",
                     static_cast<unsigned long long>(va));
        }
        return false;
    }
    begin_recording_locked();
    SlotOcc& so = g_occ[g.slot];
    if (!so.ready) return false;
    const std::uint64_t issued = g_issued.load(std::memory_order_relaxed);
    // Every draw since the last dump measured? Otherwise the interval reads visible.
    const bool unmeasured = !g_iv.open || g_iv.unmeasured || issued - g_iv.issued_at != g_iv.measured;
    if (so.nitems + 2 > kItems) {
        ++g_counts.lost;  // not written: the block stays not ready
    } else {
        if (unmeasured) {
            push_item(so, kItemUnmeasured, 0, 0, 0);
            ++g_counts.unmeasured_intervals;
        }
        // A block's second dump: the shader also reads the pair back, for the
        // report - only where the whole block is one import, so the read
        // cannot leave it.
        const bool pair = (va & 15) == 8 && guest_device_address(va - 8, 16 * kPipes) == at - 8;
        push_item(so, kItemDump | kPipes << 8 | (pair ? 1u << 16 : 0u), static_cast<std::uint32_t>(at), static_cast<std::uint32_t>(at >> 32),
                  0);
        shadow_written_locked(va, kBlockExtent, true);  // lands on the GPU's timeline
        ++g_counts.dumps;
    }
    if (g_block_serial.size() > 65536) g_block_serial.clear();
    g_block_serial[va & ~15ull] = g.flushes;
    g_iv.open = true;
    g_iv.unmeasured = false;
    g_iv.cap = (va & 15) == 0 ? g_cap : 1;
    g_iv.queries = 0;
    g_iv.issued_at = issued;
    g_iv.measured = 0;
    g_iv.last_draw = t_draw;
    return true;
}

void host_gpu_set_predication(std::uint64_t block, unsigned op, bool draw_if_visible, bool cont) {
    if (!g_on || !g_pred_wanted) return;
    std::lock_guard<GpuMutex> lock(g.mu);
    ++g_counts.pred_sets;
    if (op == 0) {
        g_pred_mode.store(kPredNone, std::memory_order_relaxed);
        g_pred_seen = 0;
        return;
    }
    if (op != 1) {
        // Primitive-count predication (streamout): not emulated, drawn.
        static bool logged = false;
        if (!logged) {
            logged = true;
            host_log("occlusion: SET_PREDICATION operation %u (not a ZPASS one) - its draws are drawn", op);
        }
        g_pred_mode.store(kPredDraw, std::memory_order_relaxed);
        return;
    }
    const int mode = g_pred_mode.load(std::memory_order_relaxed);
    const auto it = g_block_serial.find(block);
    const bool pending = it != g_block_serial.end() && it->second >= g.completed_submits;
    // CONTINUE on a CPU result that already reads "visible": visible it stays.
    const bool cpu_chain = cont && (mode == kPredSkip || mode == kPredDraw || mode == kPredNone);
    if (!pending && !(cont && mode == kPredGpu)) {
        ++g_counts.pred_known;
        const std::uint32_t r = block_on_cpu(block);
        const std::uint32_t seen = (cpu_chain ? g_pred_seen : 0) | (r == 2 ? 2u : r);
        g_pred_seen = seen;
        const bool draw = (seen & 2) || ((seen & 1) != 0) == draw_if_visible;
        g_pred_mode.store(draw ? kPredDraw : kPredSkip, std::memory_order_relaxed);
        return;
    }
    if (!g_cond) {
        g_pred_mode.store(kPredDraw, std::memory_order_relaxed);
        return;
    }
    if (cpu_chain && ((g_pred_seen & 2) || ((g_pred_seen & 1) && draw_if_visible))) {
        // Unknown, or already visible for a draw-if-visible, on the CPU:
        // OR-ing in another block cannot cull. (Otherwise the GPU takes the
        // new block alone, which can only draw more.)
        g_pred_mode.store(kPredDraw, std::memory_order_relaxed);
        return;
    }
    const ImportSite site("an occlusion predicate");
    if (twins_on()) twins_keep_in_guest_locked(block, 16 * kPipes);  // as for a dump
    const VkDeviceAddress at = guest_device_address(block, 16 * kPipes);
    begin_recording_locked();
    SlotOcc& so = g_occ[g.slot];
    if (!at || !so.ready || so.nitems + 1 > kItems) {
        g_pred_mode.store(kPredDraw, std::memory_order_relaxed);
        return;
    }
    const bool gpu_cont = cont && mode == kPredGpu;
    const std::uint32_t index = gpu_cont ? g_pred_index : g_pred_next++ % kPreds;
    push_item(so, kItemPredicate | kPipes << 8 | (draw_if_visible ? 1u << 16 : 0u) | (gpu_cont ? 1u << 17 : 0u), static_cast<std::uint32_t>(at),
              static_cast<std::uint32_t>(at >> 32), index);
    resolve_locked(true);
    ++g_counts.pred_gpu_sets;
    g_pred_index = index;
    g_pred_mode.store(kPredGpu, std::memory_order_relaxed);
}

std::string host_gpu_occlusion_report() {
    if (!g_on) return {};
    std::lock_guard<GpuMutex> lock(g.mu);
    return occlusion_report(false);
}
